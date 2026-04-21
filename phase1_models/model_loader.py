"""Phase 1 detector training and evaluation support for DERGuardian.

This module implements model loader logic for residual-window model training,
inference, packaging, metrics, or reporting. It supports the frozen benchmark
path and related audits while keeping benchmark selection separate from replay,
heldout synthetic zero-day-like, and extension contexts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import json
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from common.io_utils import read_json, write_json
from phase1_models.context.phase1_fusion import build_fusion_inputs
from phase1_models.feature_builder import transform_features, transform_to_tokens
from phase1_models.model_utils import READY_MODEL_ROOT
from phase1_models.neural_models import GRUClassifier, LSTMClassifier, MLPAutoencoder, TinyTransformerClassifier, TokenBaselineClassifier
from phase1_models.neural_training import predict_autoencoder_errors, predict_classifier_scores
from phase1_models.ready_package_utils import apply_saved_calibration
from phase1_models.run_full_evaluation import build_sequence_dataset_segments, prepare_full_run_data


@dataclass(slots=True)
class LoadedPhase1Package:
    """In-memory representation of a saved Phase 1 detector package."""

    package_dir: Path
    manifest: dict[str, Any]
    config: dict[str, Any]
    thresholds: dict[str, Any]
    calibration: dict[str, Any]
    feature_columns: list[str]
    preprocessing: dict[str, Any]
    checkpoint: Any
    model: Any


def load_phase1_package(package_dir: str | Path) -> LoadedPhase1Package:
    """Load a frozen detector package from disk without retraining it."""

    package_path = Path(package_dir)
    manifest = read_json(package_path / "model_manifest.json")
    config = read_json(package_path / manifest["config_filename"])
    thresholds = read_json(package_path / manifest["thresholds_filename"])
    calibration = read_json(package_path / manifest["calibration_filename"])
    feature_columns = list(read_json(package_path / manifest["feature_columns_filename"]))
    with (package_path / manifest["preprocessing_filename"]).open("rb") as handle:
        preprocessing = pickle.load(handle)
    checkpoint_path = package_path / manifest["checkpoint_filename"]
    checkpoint = _load_checkpoint(checkpoint_path)
    model = _materialize_model(manifest, config, checkpoint)
    return LoadedPhase1Package(
        package_dir=package_path,
        manifest=manifest,
        config=config,
        thresholds=thresholds,
        calibration=calibration,
        feature_columns=feature_columns,
        preprocessing=preprocessing,
        checkpoint=checkpoint,
        model=model,
    )


def run_phase1_inference(package: LoadedPhase1Package, windows_df: pd.DataFrame) -> dict[str, Any]:
    """Run detector inference using the package's saved preprocessing and threshold.

    This function is used by replay, heldout synthetic, and deployment-style
    audits. It does not alter canonical benchmark selection; it only reuses the
    frozen package contract.
    """

    package_name = str(package.manifest.get("model_name"))
    family = str(package.manifest.get("model_family"))
    threshold = float(
        package.thresholds.get("primary_threshold")
        or package.thresholds.get("threshold")
        or package.thresholds.get("decision_threshold")
        or 0.5
    )
    frame = windows_df.copy()
    standardizer = package.preprocessing.get("standardizer")
    discretizer = package.preprocessing.get("discretizer")

    if family == "fusion_calibrator":
        meta = _prediction_metadata_frame(frame)
        feature_values = frame[package.feature_columns].astype(float).fillna(0.0).to_numpy()
        raw_scores = _linear_response(feature_values, package.calibration)
        calibrated_scores = apply_saved_calibration(feature_values, package.calibration)
    elif package_name == "threshold_baseline":
        meta = _prediction_metadata_frame(frame)
        transformed = transform_features(frame, package.feature_columns, standardizer)
        raw_scores = np.mean(np.abs(transformed), axis=1).astype(float)
        calibrated_scores = apply_saved_calibration(raw_scores, package.calibration)
    elif package_name == "isolation_forest":
        meta = _prediction_metadata_frame(frame)
        transformed = transform_features(frame, package.feature_columns, standardizer)
        raw_scores = (-package.model.score_samples(transformed)).astype(float)
        calibrated_scores = apply_saved_calibration(raw_scores, package.calibration)
    elif package_name == "autoencoder":
        meta = _prediction_metadata_frame(frame)
        transformed = transform_features(frame, package.feature_columns, standardizer).astype(np.float32)
        raw_scores = predict_autoencoder_errors(package.model, transformed)
        calibrated_scores = apply_saved_calibration(raw_scores, package.calibration)
    elif package_name in {"gru", "lstm", "transformer", "llm_baseline"}:
        seq_len = int(package.config.get("sequence_length") or package.manifest.get("sequence_length") or 1)
        token_input = bool(package.config.get("token_input", False))
        # Sequence detectors need the same segmentation contract they used
        # during Phase 1 training so saved thresholds remain meaningful.
        x_values, _, meta = build_sequence_dataset_segments(
            frame,
            package.feature_columns,
            seq_len,
            standardizer,
            discretizer if token_input else None,
        )
        raw_scores = predict_classifier_scores(package.model, x_values, token_input=token_input)
        calibrated_scores = apply_saved_calibration(raw_scores, package.calibration)
    else:
        raise ValueError(f"Unsupported Phase 1 package: {package_name}")

    predictions = meta.copy()
    predictions["raw_score"] = np.asarray(raw_scores, dtype=float)
    predictions["score"] = np.asarray(calibrated_scores, dtype=float)
    predictions["predicted"] = (predictions["score"] >= threshold).astype(int)
    return {
        "raw_scores": np.asarray(raw_scores, dtype=float),
        "calibrated_scores": np.asarray(calibrated_scores, dtype=float),
        "predicted_labels": predictions["predicted"].to_numpy(dtype=int),
        "predictions": predictions,
        "metadata": {
            "model_name": package_name,
            "model_family": family,
            "threshold": threshold,
            "feature_columns": package.feature_columns,
            "package_dir": str(package.package_dir),
        },
    }


def verify_all_ready_packages(project_root: str | Path | None = None) -> dict[str, Any]:
    """Recompute saved package predictions and report any score drift."""

    root = Path(project_root) if project_root is not None else ROOT
    ready_root = root / "outputs" / READY_MODEL_ROOT
    package_dirs = sorted(path for path in ready_root.iterdir() if path.is_dir())
    full_data = prepare_full_run_data(root)
    test_frame = full_data.split_df[full_data.split_df["split_name"] == "test"].copy().reset_index(drop=True)
    reasoning_table = pd.read_parquet(root / "outputs" / "reports" / "model_full_run_artifacts" / "phase1_context_reasoning_outputs.parquet")

    detector_cache: dict[str, pd.DataFrame] = {}
    verification_rows: list[dict[str, Any]] = []
    for package_dir in package_dirs:
        package = load_phase1_package(package_dir)
        package_name = str(package.manifest.get("model_name"))
        saved_predictions_path = package_dir / package.manifest["predictions_filename"]
        saved_predictions = pd.read_parquet(saved_predictions_path)
        if package_name.startswith("fusion_"):
            input_frame = _build_fusion_verification_input(root, test_frame, reasoning_table, detector_cache, package_name)
        else:
            input_frame = test_frame
        result = run_phase1_inference(package, input_frame)
        merged = _merge_prediction_frames(result["predictions"], saved_predictions)
        max_abs_diff = float((merged["score_recomputed"] - merged["score_saved"]).abs().max()) if not merged.empty else 0.0
        predicted_match = bool((merged["predicted_recomputed"] == merged["predicted_saved"]).all()) if not merged.empty else True
        score_match = bool(np.allclose(merged["score_recomputed"], merged["score_saved"], atol=1e-9)) if not merged.empty else True
        verification_rows.append(
            {
                "package_name": package_dir.name,
                "model_name": package_name,
                "rows_compared": int(len(merged)),
                "max_abs_diff": max_abs_diff,
                "score_match": score_match,
                "predicted_match": predicted_match,
            }
        )
        if not package_name.startswith("fusion_"):
            detector_cache[package_name] = result["predictions"].copy()

    payload = {
        "ready_root": str(ready_root),
        "packages": verification_rows,
    }
    output_path = root / "outputs" / "reports" / "model_full_run_artifacts" / "phase1_ready_package_verification.json"
    write_json(payload, output_path)
    return payload


def _build_fusion_verification_input(
    root: Path,
    test_frame: pd.DataFrame,
    reasoning_table: pd.DataFrame,
    detector_cache: dict[str, pd.DataFrame],
    package_name: str,
) -> pd.DataFrame:
    detector_package = load_phase1_package(root / "outputs" / READY_MODEL_ROOT / "threshold_baseline")
    token_package = load_phase1_package(root / "outputs" / READY_MODEL_ROOT / "llm_baseline")
    detector_predictions = detector_cache.get("threshold_baseline")
    if detector_predictions is None:
        detector_predictions = run_phase1_inference(detector_package, test_frame)["predictions"]
        detector_cache["threshold_baseline"] = detector_predictions.copy()
    token_predictions = detector_cache.get("llm_baseline")
    if token_predictions is None:
        token_predictions = run_phase1_inference(token_package, test_frame)["predictions"]
        detector_cache["llm_baseline"] = token_predictions.copy()
    token_val = token_predictions.copy()
    detector_val = detector_predictions.copy()
    val_frame, test_result = build_fusion_inputs(
        detector_val=detector_val,
        detector_test=detector_predictions.copy(),
        reasoning_table=reasoning_table,
        token_val=token_val,
        token_test=token_predictions.copy(),
    )
    _ = val_frame
    return test_result


def _merge_prediction_frames(recomputed: pd.DataFrame, saved: pd.DataFrame) -> pd.DataFrame:
    left = recomputed.copy()
    right = saved.copy()
    for frame in (left, right):
        for column in ["window_start_utc", "window_end_utc"]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], utc=True)
    join_keys = [column for column in ["window_start_utc", "window_end_utc", "scenario_id"] if column in left.columns and column in right.columns]
    merged = left.merge(
        right,
        on=join_keys,
        how="inner",
        suffixes=("_recomputed", "_saved"),
        validate="one_to_one",
    )
    return merged


def _prediction_metadata_frame(frame: pd.DataFrame) -> pd.DataFrame:
    selected_columns = [
        column
        for column in [
            "window_start_utc",
            "window_end_utc",
            "scenario_id",
            "scenario_window_id",
            "attack_present",
            "attack_family",
            "attack_severity",
            "attack_affected_assets",
            "split_name",
        ]
        if column in frame.columns
    ]
    meta = frame[selected_columns].copy()
    if "scenario_window_id" in meta.columns:
        meta["scenario_id"] = meta["scenario_window_id"].astype(str)
        meta = meta.drop(columns=["scenario_window_id"])
    elif "scenario_id" in meta.columns:
        meta["scenario_id"] = meta["scenario_id"].astype(str)
    return meta


def _load_checkpoint(checkpoint_path: Path) -> Any:
    if checkpoint_path.suffix == ".pt":
        return torch.load(checkpoint_path, map_location="cpu")
    with checkpoint_path.open("rb") as handle:
        return pickle.load(handle)


def _materialize_model(manifest: dict[str, Any], config: dict[str, Any], checkpoint: Any) -> Any:
    model_name = str(manifest.get("model_name"))
    architecture = config.get("architecture_parameters", {})
    if model_name == "threshold_baseline":
        return checkpoint
    if model_name == "isolation_forest":
        return checkpoint["model"]
    if model_name == "autoencoder":
        model = MLPAutoencoder(
            input_dim=int(architecture["input_dim"]),
            hidden_dim=int(architecture["hidden_dim"]),
            bottleneck_dim=int(architecture["bottleneck_dim"]),
        )
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    if model_name == "gru":
        model = GRUClassifier(
            input_dim=int(architecture["input_dim"]),
            hidden_dim=int(architecture["hidden_dim"]),
        )
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    if model_name == "lstm":
        model = LSTMClassifier(
            input_dim=int(architecture["input_dim"]),
            hidden_dim=int(architecture["hidden_dim"]),
        )
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    if model_name == "transformer":
        model = TinyTransformerClassifier(
            input_dim=int(architecture["input_dim"]),
            d_model=int(architecture["d_model"]),
            nhead=int(architecture["nhead"]),
            num_layers=int(architecture["num_layers"]),
        )
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    if model_name == "llm_baseline":
        model = TokenBaselineClassifier(
            vocab_size=int(architecture["vocab_size"]),
            embed_dim=int(architecture["embed_dim"]),
        )
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    if model_name.startswith("fusion_"):
        return checkpoint
    raise ValueError(f"Unsupported package model: {model_name}")


def _linear_response(feature_values: np.ndarray, calibration_payload: dict[str, Any]) -> np.ndarray:
    values = np.asarray(feature_values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    calibration_type = str(calibration_payload.get("calibration_type", "identity"))
    if calibration_type == "identity":
        return values[:, 0].astype(float)
    coefficients = np.asarray(calibration_payload.get("coefficients", []), dtype=float).reshape(-1)
    intercept = np.asarray(calibration_payload.get("intercept", []), dtype=float).reshape(-1)
    intercept_value = float(intercept[0]) if intercept.size else 0.0
    return (values @ coefficients.reshape(-1, 1)).reshape(-1) + intercept_value


def main() -> None:
    """Run the command-line entrypoint for the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Load and verify standardized Phase 1 ready-model packages.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--package-dir", default="")
    parser.add_argument("--verify-all", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root)
    if args.verify_all:
        payload = verify_all_ready_packages(root)
        print(json.dumps(payload, indent=2))
        return
    if not args.package_dir:
        raise SystemExit("--package-dir is required unless --verify-all is used.")
    package = load_phase1_package(args.package_dir)
    print(json.dumps(package.manifest, indent=2))


if __name__ == "__main__":
    main()
