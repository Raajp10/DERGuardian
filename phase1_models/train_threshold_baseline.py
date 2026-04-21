"""Phase 1 detector training and evaluation support for DERGuardian.

This module implements train threshold baseline logic for residual-window model training,
inference, packaging, metrics, or reporting. It supports the frozen benchmark
path and related audits while keeping benchmark selection separate from replay,
heldout synthetic zero-day-like, and extension contexts.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from phase1_models.dataset_loader import load_window_dataset_bundle
from phase1_models.feature_builder import fit_standardizer, select_numeric_feature_columns, transform_features
from phase1_models.metrics import compute_binary_metrics, compute_curve_payload
from phase1_models.model_utils import ensure_model_paths, write_json, write_pickle, write_predictions
from phase1_models.run_full_evaluation import prepare_full_run_data, run_threshold_baseline
from phase1_models.thresholds import quantile_thresholds


def _train_threshold_model_legacy(project_root: str | Path | None = None, max_features: int = 128, threshold_name: str = "p99") -> dict[str, object]:
    bundle = load_window_dataset_bundle(project_root)
    feature_columns = select_numeric_feature_columns(bundle.clean_windows, max_features=max_features)
    standardizer = fit_standardizer(bundle.clean_windows, feature_columns)
    x_clean = transform_features(bundle.clean_windows, feature_columns, standardizer)
    x_attacked = transform_features(bundle.attacked_windows, feature_columns, standardizer)
    clean_scores = np.mean(np.abs(x_clean), axis=1)
    attacked_scores = np.mean(np.abs(x_attacked), axis=1)
    thresholds = quantile_thresholds(clean_scores)
    threshold = thresholds[threshold_name]
    metrics = compute_binary_metrics(bundle.attacked_windows["attack_present"].to_numpy(), attacked_scores, threshold)
    curves = compute_curve_payload(bundle.attacked_windows["attack_present"].to_numpy(), attacked_scores)
    predictions = bundle.attacked_windows[["window_start_utc", "window_end_utc", "scenario_id", "attack_present"]].copy()
    predictions["score"] = attacked_scores
    predictions["predicted"] = (attacked_scores >= threshold).astype(int)

    paths = ensure_model_paths("threshold_baseline", project_root)
    model_payload = {
        "model_name": "threshold_baseline",
        "feature_columns": feature_columns,
        "thresholds": thresholds,
        "selected_threshold": threshold_name,
        "standardizer_mean": standardizer.mean.tolist(),
        "standardizer_std": standardizer.std.tolist(),
        "training_time_seconds": 0.0,
        "inference_time_seconds": float(len(x_attacked) / max(len(x_attacked), 1)),
        "parameter_count": len(feature_columns) * 2,
    }
    write_pickle(model_payload, paths.model_dir / "model.pkl")
    write_json({"metrics": metrics, "curves": curves, "model_info": model_payload, "history": {"train_loss": [], "val_loss": []}}, paths.model_dir / "results.json")
    write_predictions(predictions, paths.model_dir / "predictions.parquet")
    return {"model_name": "threshold_baseline", "metrics": metrics, "info": model_payload}


def train_threshold_model(
    project_root: str | Path | None = None,
    max_features: int = 128,
    threshold_name: str = "p99",
    run_mode: str = "full",
    feature_counts: list[int] | None = None,
    buffer_windows: int = 2,
) -> dict[str, object]:
    """Handle train threshold model within the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    root = Path(project_root) if project_root is not None else ROOT
    if run_mode == "legacy-smoke":
        return _train_threshold_model_legacy(project_root=root, max_features=max_features, threshold_name=threshold_name)
    selected_feature_counts = feature_counts or [max_features]
    full_data = prepare_full_run_data(root, buffer_windows=buffer_windows)
    result = run_threshold_baseline(
        root=root,
        full_data=full_data,
        feature_counts=selected_feature_counts,
        report_root=root / "outputs" / "reports" / "model_full_run_artifacts",
        model_root_name="models_full_run",
        artifact_root_name="model_full_run_artifacts",
    )
    return {"model_name": "threshold_baseline", "metrics": result["summary"], "info": result["summary"]}


def main() -> None:
    """Run the command-line entrypoint for the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Train the repaired quantile-threshold anomaly baseline.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--max-features", type=int, default=128)
    parser.add_argument("--threshold", default="p99", choices=["p95", "p99", "p99.5"])
    parser.add_argument("--run-mode", default="full", choices=["full", "legacy-smoke"])
    parser.add_argument("--feature-counts", default="")
    parser.add_argument("--buffer-windows", type=int, default=2)
    args = parser.parse_args()

    feature_counts = [int(item) for item in args.feature_counts.split(",") if item.strip()] if args.feature_counts else None
    result = train_threshold_model(
        project_root=args.project_root,
        max_features=args.max_features,
        threshold_name=args.threshold,
        run_mode=args.run_mode,
        feature_counts=feature_counts,
        buffer_windows=args.buffer_windows,
    )
    print(f"Saved threshold baseline artifacts for {result['model_name']}")


if __name__ == "__main__":
    main()
