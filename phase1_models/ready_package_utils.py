from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import math
import pickle

import numpy as np
import pandas as pd
import torch

from common.io_utils import ensure_dir, write_dataframe, write_json
from phase1_models.model_utils import READY_MODEL_ROOT


PACKAGE_VERSION = "1.0.0"
DEFAULT_SEED = 1729


def build_split_summary(split_df: pd.DataFrame) -> dict[str, Any]:
    frame = split_df.copy()
    summary = (
        frame.groupby(["split_name", "attack_present"], observed=False)
        .size()
        .reset_index(name="window_count")
        .sort_values(["split_name", "attack_present"])
    )
    payload = {
        "rows": summary.to_dict(orient="records"),
        "total_windows": int(len(frame)),
        "attack_windows": int(frame["attack_present"].sum()) if "attack_present" in frame.columns else 0,
    }
    if "scenario_window_id" in frame.columns:
        scenario_counts = (
            frame[frame["attack_present"] == 1]
            .groupby("scenario_window_id", observed=False)
            .size()
            .reset_index(name="attack_window_count")
            .rename(columns={"scenario_window_id": "scenario_id"})
        )
        payload["scenario_attack_counts"] = scenario_counts.to_dict(orient="records")
    return payload


def serialize_calibration(
    calibrator: Any,
    *,
    input_names: list[str] | None = None,
) -> dict[str, Any]:
    names = list(input_names or ["raw_score"])
    if calibrator is None:
        return {
            "calibration_type": "identity",
            "input_names": names,
            "coefficients": [],
            "intercept": [],
        }
    coefficients = np.asarray(getattr(calibrator, "coef_", []), dtype=float).reshape(-1).tolist()
    intercept = np.asarray(getattr(calibrator, "intercept_", []), dtype=float).reshape(-1).tolist()
    return {
        "calibration_type": "logistic_regression",
        "input_names": names,
        "coefficients": coefficients,
        "intercept": intercept,
    }


def apply_saved_calibration(feature_values: np.ndarray, calibration_payload: dict[str, Any]) -> np.ndarray:
    values = np.asarray(feature_values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    calibration_type = str(calibration_payload.get("calibration_type", "identity"))
    if calibration_type == "identity":
        return values[:, 0].astype(float)
    coefficients = np.asarray(calibration_payload.get("coefficients", []), dtype=float).reshape(-1)
    intercept = np.asarray(calibration_payload.get("intercept", []), dtype=float).reshape(-1)
    intercept_value = float(intercept[0]) if intercept.size else 0.0
    logits = values @ coefficients.reshape(-1, 1)
    logits = logits.reshape(-1) + intercept_value
    return (1.0 / (1.0 + np.exp(-logits))).astype(float)


def export_ready_model_package(
    *,
    root: Path,
    package_name: str,
    model_name: str,
    model_family: str,
    checkpoint_kind: str,
    checkpoint_payload: Any,
    preprocessing_payload: dict[str, Any],
    feature_columns: list[str],
    thresholds_payload: dict[str, Any],
    calibration_payload: dict[str, Any],
    config_payload: dict[str, Any],
    training_history: dict[str, list[float]],
    metrics_payload: dict[str, Any],
    predictions: pd.DataFrame,
    split_summary: dict[str, Any],
    notes: str,
    training_supervision: str,
    input_schema_summary: dict[str, Any] | None = None,
    package_version: str = PACKAGE_VERSION,
    seed: int = DEFAULT_SEED,
) -> Path:
    package_dir = ensure_dir(root / "outputs" / READY_MODEL_ROOT / package_name)
    checkpoint_filename = "checkpoint.pt" if checkpoint_kind == "torch" else "checkpoint.pkl"
    preprocessing_filename = "preprocessing.pkl"
    feature_columns_filename = "feature_columns.json"
    thresholds_filename = "thresholds.json"
    calibration_filename = "calibration.json"
    config_filename = "config.json"
    training_history_filename = "training_history.json"
    metrics_filename = "metrics.json"
    predictions_filename = "predictions.parquet"
    split_summary_filename = "split_summary.json"
    manifest_filename = "model_manifest.json"

    checkpoint_path = package_dir / checkpoint_filename
    if checkpoint_kind == "torch":
        if isinstance(checkpoint_payload, torch.nn.Module):
            torch.save(checkpoint_payload.state_dict(), checkpoint_path)
        else:
            torch.save(checkpoint_payload, checkpoint_path)
    elif checkpoint_kind == "pickle":
        with checkpoint_path.open("wb") as handle:
            pickle.dump(checkpoint_payload, handle)
    else:
        raise ValueError(f"Unsupported checkpoint kind: {checkpoint_kind}")

    with (package_dir / preprocessing_filename).open("wb") as handle:
        pickle.dump(preprocessing_payload, handle)
    write_json(feature_columns, package_dir / feature_columns_filename)
    write_json(thresholds_payload, package_dir / thresholds_filename)
    write_json(calibration_payload, package_dir / calibration_filename)
    write_json(config_payload, package_dir / config_filename)
    write_json(training_history, package_dir / training_history_filename)
    write_json(metrics_payload, package_dir / metrics_filename)
    write_json(split_summary, package_dir / split_summary_filename)
    write_dataframe(predictions, package_dir / predictions_filename, fmt="parquet")

    manifest = {
        "model_name": model_name,
        "model_family": model_family,
        "package_version": package_version,
        "checkpoint_filename": checkpoint_filename,
        "preprocessing_filename": preprocessing_filename,
        "feature_columns_filename": feature_columns_filename,
        "thresholds_filename": thresholds_filename,
        "calibration_filename": calibration_filename,
        "config_filename": config_filename,
        "training_history_filename": training_history_filename,
        "metrics_filename": metrics_filename,
        "predictions_filename": predictions_filename,
        "split_summary_filename": split_summary_filename,
        "seed": int(seed),
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split_metadata_summary": split_summary,
        "input_schema_summary": input_schema_summary or build_input_schema_summary(feature_columns, config_payload),
        "sequence_length": config_payload.get("sequence_length"),
        "architecture_parameters": config_payload.get("architecture_parameters", {}),
        "training_supervision": training_supervision,
        "notes": notes,
    }
    write_json(manifest, package_dir / manifest_filename)
    return package_dir


def export_ready_fusion_package(
    *,
    root: Path,
    mode_name: str,
    feature_names: list[str],
    thresholds_payload: dict[str, Any],
    calibration_payload: dict[str, Any],
    config_payload: dict[str, Any],
    metrics_payload: dict[str, Any],
    training_history: dict[str, list[float]],
    predictions: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    split_summary: dict[str, Any],
) -> Path:
    package_name = f"fusion_{mode_name}"
    package_dir = export_ready_model_package(
        root=root,
        package_name=package_name,
        model_name=package_name,
        model_family="fusion_calibrator",
        checkpoint_kind="pickle",
        checkpoint_payload={"fusion_mode": mode_name},
        preprocessing_payload={"standardizer": None, "discretizer": None},
        feature_columns=feature_names,
        thresholds_payload=thresholds_payload,
        calibration_payload=calibration_payload,
        config_payload=config_payload,
        training_history=training_history,
        metrics_payload=metrics_payload,
        predictions=predictions,
        split_summary=split_summary,
        notes="Validation-fitted fusion package combining saved detector/context/token scores.",
        training_supervision="validation_calibrated_fusion",
        input_schema_summary={
            "required_columns": feature_names,
            "feature_mode": "fusion_score_frame",
            "time_columns": ["window_start_utc", "window_end_utc"],
        },
    )
    write_json({"fusion_mode": mode_name, **config_payload}, package_dir / "fusion_config.json")
    write_json(thresholds_payload, package_dir / "fusion_thresholds.json")
    write_json(calibration_payload, package_dir / "fusion_calibration.json")
    write_json(
        {
            "fusion_mode": mode_name,
            "metrics": metrics_payload,
        },
        package_dir / f"fusion_metrics_{mode_name}.json",
    )
    write_json(
        {
            "fusion_mode": mode_name,
            "predictions_file": f"fused_predictions_{mode_name}.parquet",
            "validation_predictions_file": "validation_predictions.parquet",
        },
        package_dir / "fusion_manifest.json",
    )
    write_dataframe(predictions, package_dir / f"fused_predictions_{mode_name}.parquet", fmt="parquet")
    write_dataframe(validation_predictions, package_dir / "validation_predictions.parquet", fmt="parquet")
    return package_dir


def build_input_schema_summary(feature_columns: list[str], config_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_count": int(len(feature_columns)),
        "feature_mode": config_payload.get("feature_mode", "aligned_residual"),
        "required_time_columns": ["window_start_utc", "window_end_utc"],
        "required_identity_columns": ["scenario_id"],
        "feature_preview": feature_columns[:10],
    }


def architecture_from_model(model: Any, model_name: str, *, input_dim: int, seq_len: int | None = None, token_bins: int | None = None) -> dict[str, Any]:
    if model_name == "autoencoder":
        return {
            "input_dim": int(input_dim),
            "hidden_dim": int(model.encoder[0].out_features),
            "bottleneck_dim": int(model.encoder[2].out_features),
        }
    if model_name == "gru":
        return {
            "input_dim": int(input_dim),
            "hidden_dim": int(model.gru.hidden_size),
            "num_layers": int(model.gru.num_layers),
            "sequence_length": int(seq_len or 0),
        }
    if model_name == "lstm":
        return {
            "input_dim": int(input_dim),
            "hidden_dim": int(model.lstm.hidden_size),
            "num_layers": int(model.lstm.num_layers),
            "sequence_length": int(seq_len or 0),
        }
    if model_name == "transformer":
        layer = model.encoder.layers[0]
        return {
            "input_dim": int(input_dim),
            "d_model": int(model.proj.out_features),
            "nhead": int(layer.self_attn.num_heads),
            "num_layers": int(len(model.encoder.layers)),
            "dim_feedforward": int(layer.linear1.out_features),
            "sequence_length": int(seq_len or 0),
        }
    if model_name == "llm_baseline":
        return {
            "vocab_size": int(model.embedding.num_embeddings),
            "embed_dim": int(model.embedding.embedding_dim),
            "sequence_length": int(seq_len or 0),
            "discretization_bins": int(token_bins or 0),
        }
    return {}


def discretizer_bin_count(discretizer: Any) -> int:
    if discretizer is None or not getattr(discretizer, "edges", None):
        return 0
    return max(max(int(len(edges) - 1), 0) for edges in discretizer.edges.values())


def json_safe_metrics(metrics_payload: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in metrics_payload.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned
