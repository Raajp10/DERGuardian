"""Phase 1 context and fusion helper for DERGuardian.

This module builds or consumes structured context artifacts around normal-system
behavior, feature evidence, and detector outputs. It supports explanation and
fusion workflows but is not the canonical detector-selection mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common.io_utils import write_dataframe
from phase1_models.metrics import compute_binary_metrics, compute_curve_payload, detection_latency_table, per_scenario_metrics
from phase1_models.model_utils import write_json
from phase1_models.ready_package_utils import export_ready_fusion_package, serialize_calibration


FUSION_MODES = (
    "detector_only",
    "detector_plus_context",
    "detector_plus_context_plus_token",
)


@dataclass(slots=True)
class FusionModeArtifacts:
    """Structured object used by the Phase 1 context and fusion support workflow."""

    mode_name: str
    val_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    metrics: dict[str, Any]
    curves: dict[str, list[float]]
    per_scenario: pd.DataFrame
    latency: pd.DataFrame
    metadata: dict[str, Any]


def build_fusion_inputs(
    detector_val: pd.DataFrame,
    detector_test: pd.DataFrame,
    reasoning_table: pd.DataFrame,
    token_val: pd.DataFrame | None = None,
    token_test: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build fusion inputs for the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    reasoning_core = reasoning_table.copy()
    reasoning_core["window_start_utc"] = pd.to_datetime(reasoning_core["window_start_utc"], utc=True)
    reasoning_core["window_end_utc"] = pd.to_datetime(reasoning_core["window_end_utc"], utc=True)
    join_keys = ["window_start_utc", "window_end_utc", "scenario_id"]
    reasoning_core = reasoning_core[join_keys + ["anomaly_reasoning_score", "confidence_score", "likely_anomaly_family"]]

    val_frame = detector_val.merge(reasoning_core, on=join_keys, how="left")
    test_frame = detector_test.merge(reasoning_core, on=join_keys, how="left")
    for frame in (val_frame, test_frame):
        frame["anomaly_reasoning_score"] = frame["anomaly_reasoning_score"].fillna(0.0).astype(float)
        frame["confidence_score"] = frame["confidence_score"].fillna(0.0).astype(float)
        frame["likely_anomaly_family"] = frame["likely_anomaly_family"].fillna("unknown_anomaly").astype(str)

    if token_val is not None:
        val_frame = _attach_token_scores(val_frame, token_val)
    else:
        val_frame["token_score"] = 0.0
    if token_test is not None:
        test_frame = _attach_token_scores(test_frame, token_test)
    else:
        test_frame["token_score"] = 0.0
    return val_frame, test_frame


def run_fusion_suite(
    *,
    detector_model_name: str,
    labels_df: pd.DataFrame,
    detector_val: pd.DataFrame,
    detector_test: pd.DataFrame,
    reasoning_table: pd.DataFrame,
    output_model_dir: Path,
    output_report_dir: Path,
    token_val: pd.DataFrame | None = None,
    token_test: pd.DataFrame | None = None,
    split_summary: dict[str, Any] | None = None,
) -> tuple[dict[str, FusionModeArtifacts], pd.DataFrame]:
    """Run fusion suite for the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    output_model_dir.mkdir(parents=True, exist_ok=True)
    output_report_dir.mkdir(parents=True, exist_ok=True)
    val_frame, test_frame = build_fusion_inputs(
        detector_val=detector_val,
        detector_test=detector_test,
        reasoning_table=reasoning_table,
        token_val=token_val,
        token_test=token_test,
    )
    results: dict[str, FusionModeArtifacts] = {}
    summary_rows: list[dict[str, Any]] = []
    for mode_name in FUSION_MODES:
        feature_names = fusion_feature_names(mode_name)
        artifacts = fit_fusion_mode(
            mode_name=mode_name,
            feature_names=feature_names,
            detector_model_name=detector_model_name,
            labels_df=labels_df,
            val_frame=val_frame,
            test_frame=test_frame,
        )
        results[mode_name] = artifacts
        mode_model_dir = output_model_dir / mode_name
        mode_report_dir = output_report_dir / mode_name
        mode_model_dir.mkdir(parents=True, exist_ok=True)
        mode_report_dir.mkdir(parents=True, exist_ok=True)
        write_dataframe(artifacts.val_predictions, mode_model_dir / "validation_predictions.parquet", fmt="parquet")
        write_dataframe(artifacts.test_predictions, mode_model_dir / "predictions.parquet", fmt="parquet")
        write_dataframe(artifacts.per_scenario, mode_report_dir / "per_scenario_metrics.csv", fmt="csv")
        write_dataframe(artifacts.latency, mode_report_dir / "detection_latency_analysis.csv", fmt="csv")
        write_json(
            {
                "metrics": artifacts.metrics,
                "curves": artifacts.curves,
                "metadata": artifacts.metadata,
            },
            mode_model_dir / "results.json",
        )
        export_ready_fusion_package(
            root=output_model_dir.parents[2],
            mode_name=mode_name,
            feature_names=feature_names,
            thresholds_payload={
                "primary_threshold": float(artifacts.metadata.get("threshold", 0.5) or 0.5),
                "selection_method": "validation_threshold_scan",
            },
            calibration_payload=serialize_calibration(
                _fusion_metadata_to_calibrator_like(artifacts.metadata),
                input_names=feature_names,
            ),
            config_payload={
                "feature_mode": "fusion_score_frame",
                "sequence_length": None,
                "token_input": False,
                "architecture_parameters": {
                    "feature_names": feature_names,
                    "base_detector": detector_model_name,
                },
                "base_detector": detector_model_name,
                "fusion_mode": mode_name,
            },
            metrics_payload=artifacts.metrics,
            training_history={"train_loss": [], "val_loss": []},
            predictions=artifacts.test_predictions,
            validation_predictions=artifacts.val_predictions,
            split_summary=split_summary or {},
        )
        summary_rows.append(
            {
                "fusion_mode": mode_name,
                "base_detector": detector_model_name,
                "precision": artifacts.metrics.get("precision"),
                "recall": artifacts.metrics.get("recall"),
                "f1": artifacts.metrics.get("f1"),
                "roc_auc": artifacts.metrics.get("roc_auc"),
                "average_precision": artifacts.metrics.get("average_precision"),
                "threshold": artifacts.metrics.get("threshold"),
                "features_used": ",".join(feature_names),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    write_dataframe(summary_df, output_report_dir / "fusion_ablation_table.csv", fmt="csv")
    return results, summary_df


def fusion_feature_names(mode_name: str) -> list[str]:
    """Handle fusion feature names within the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if mode_name == "detector_only":
        return ["score"]
    if mode_name == "detector_plus_context":
        return ["score", "anomaly_reasoning_score"]
    if mode_name == "detector_plus_context_plus_token":
        return ["score", "anomaly_reasoning_score", "token_score"]
    raise ValueError(f"Unsupported fusion mode: {mode_name}")


def fit_fusion_mode(
    *,
    mode_name: str,
    feature_names: list[str],
    detector_model_name: str,
    labels_df: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> FusionModeArtifacts:
    """Fit fusion mode for the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    x_val = val_frame[feature_names].astype(float).fillna(0.0).to_numpy()
    y_val = val_frame["attack_present"].astype(int).to_numpy()
    calibrator: LogisticRegression | None
    if len(np.unique(y_val)) < 2:
        calibrator = None
        val_scores = val_frame[feature_names[0]].astype(float).fillna(0.0).to_numpy()
    else:
        calibrator = LogisticRegression(random_state=1729, solver="lbfgs")
        calibrator.fit(x_val, y_val)
        val_scores = calibrator.predict_proba(x_val)[:, 1].astype(float)
    threshold, _ = choose_best_threshold(val_scores, y_val)
    val_predictions = val_frame.copy()
    val_predictions["score"] = val_scores
    val_predictions["predicted"] = (val_predictions["score"] >= threshold).astype(int)

    x_test = test_frame[feature_names].astype(float).fillna(0.0).to_numpy()
    test_scores = calibrator.predict_proba(x_test)[:, 1].astype(float) if calibrator is not None else test_frame[feature_names[0]].astype(float).fillna(0.0).to_numpy()
    test_predictions = test_frame.copy()
    test_predictions["score"] = test_scores
    test_predictions["predicted"] = (test_predictions["score"] >= threshold).astype(int)

    metrics = compute_binary_metrics(test_predictions["attack_present"].to_numpy(), test_predictions["score"].to_numpy(), threshold)
    curves = compute_curve_payload(test_predictions["attack_present"].to_numpy(), test_predictions["score"].to_numpy())
    latency = detection_latency_table(test_predictions, labels_df, mode_name)
    per_scenario = per_scenario_metrics(test_predictions, labels_df, mode_name)
    metadata = {
        "fusion_mode": mode_name,
        "base_detector": detector_model_name,
        "feature_names": feature_names,
        "threshold": float(threshold),
        "coefficients": calibrator.coef_.astype(float).tolist() if calibrator is not None else [],
        "intercept": calibrator.intercept_.astype(float).tolist() if calibrator is not None else [],
        "calibration_mode": "logistic_regression" if calibrator is not None else "passthrough_detector_score",
    }
    return FusionModeArtifacts(
        mode_name=mode_name,
        val_predictions=val_predictions,
        test_predictions=test_predictions,
        metrics=metrics,
        curves=curves,
        per_scenario=per_scenario,
        latency=latency,
        metadata=metadata,
    )


def choose_best_threshold(scores: np.ndarray, y_true: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Handle choose best threshold within the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    candidate_thresholds = np.unique(np.quantile(scores, np.linspace(0.05, 0.995, 60)))
    best_threshold = float(candidate_thresholds[0]) if len(candidate_thresholds) else 0.5
    best_metrics: dict[str, Any] | None = None
    for threshold in candidate_thresholds:
        metrics = compute_binary_metrics(y_true, scores, float(threshold))
        candidate = (
            float(metrics.get("recall") or 0.0),
            float(metrics.get("precision") or 0.0),
            float(metrics.get("f1") or 0.0),
            float(metrics.get("average_precision") or 0.0),
        )
        incumbent = (
            float(best_metrics.get("recall") or 0.0),
            float(best_metrics.get("precision") or 0.0),
            float(best_metrics.get("f1") or 0.0),
            float(best_metrics.get("average_precision") or 0.0),
        ) if best_metrics is not None else None
        if incumbent is None or candidate > incumbent:
            best_threshold = float(threshold)
            best_metrics = metrics
    assert best_metrics is not None
    return best_threshold, best_metrics


def _fusion_metadata_to_calibrator_like(metadata: dict[str, Any]) -> Any:
    class _Calibrator:
        def __init__(self, coefficients: list[float], intercept: list[float]) -> None:
            self.coef_ = np.asarray([coefficients], dtype=float)
            self.intercept_ = np.asarray(intercept, dtype=float)

    return _Calibrator(
        coefficients=list(metadata.get("coefficients", [])),
        intercept=list(metadata.get("intercept", [])),
    ) if metadata.get("calibration_mode") == "logistic_regression" else None


def _attach_token_scores(base_frame: pd.DataFrame, token_frame: pd.DataFrame) -> pd.DataFrame:
    token_core = token_frame.copy()
    token_core["window_start_utc"] = pd.to_datetime(token_core["window_start_utc"], utc=True)
    token_core["window_end_utc"] = pd.to_datetime(token_core["window_end_utc"], utc=True)
    token_core = token_core[["window_start_utc", "window_end_utc", "scenario_id", "score"]].rename(columns={"score": "token_score"})
    merged = base_frame.merge(token_core, on=["window_start_utc", "window_end_utc", "scenario_id"], how="left")
    merged["token_score"] = merged["token_score"].fillna(0.0).astype(float)
    return merged


def main() -> None:
    """Run the command-line entrypoint for the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Phase 1 score fusion helper. Typically invoked by run_full_evaluation.py.")
    parser.add_argument("--project-root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.project_root)
    print(
        "phase1_fusion.py is designed to be called from the canonical full-run pipeline after detector, "
        "context-reasoner, and token-model predictions have been generated. "
        f"Project root resolved to {root}."
    )


if __name__ == "__main__":
    main()
