"""Phase 1 detector training and evaluation support for DERGuardian.

This module implements train autoencoder logic for residual-window model training,
inference, packaging, metrics, or reporting. It supports the frozen benchmark
path and related audits while keeping benchmark selection separate from replay,
heldout synthetic zero-day-like, and extension contexts.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from phase1_models.dataset_loader import load_window_dataset_bundle
from phase1_models.feature_builder import chronological_split, fit_standardizer, select_numeric_feature_columns, transform_features
from phase1_models.metrics import compute_binary_metrics, compute_curve_payload
from phase1_models.model_utils import ensure_model_paths, parameter_count, write_json, write_pickle, write_predictions, write_torch_state
from phase1_models.neural_models import MLPAutoencoder
from phase1_models.neural_training import predict_autoencoder_errors, train_autoencoder
from phase1_models.run_full_evaluation import prepare_full_run_data, run_autoencoder


def _train_autoencoder_model_legacy(project_root: str | Path | None = None, max_features: int = 128, epochs: int = 6) -> dict[str, object]:
    bundle = load_window_dataset_bundle(project_root)
    feature_columns = select_numeric_feature_columns(bundle.clean_windows, max_features=max_features)
    train_df, val_df, _ = chronological_split(bundle.clean_windows)
    standardizer = fit_standardizer(train_df, feature_columns)
    x_train = transform_features(train_df, feature_columns, standardizer).astype(np.float32)
    x_val = transform_features(val_df, feature_columns, standardizer).astype(np.float32)
    x_attacked = transform_features(bundle.attacked_windows, feature_columns, standardizer).astype(np.float32)

    model = MLPAutoencoder(input_dim=len(feature_columns), hidden_dim=min(128, max(32, len(feature_columns))), bottleneck_dim=max(16, len(feature_columns) // 8))
    model, history, training_time = train_autoencoder(model, x_train, x_val, epochs=epochs)
    clean_scores = predict_autoencoder_errors(model, x_val)
    inference_start = time.perf_counter()
    attacked_scores = predict_autoencoder_errors(model, x_attacked)
    inference_time = time.perf_counter() - inference_start
    threshold = float(np.quantile(clean_scores, 0.99))
    metrics = compute_binary_metrics(bundle.attacked_windows["attack_present"].to_numpy(), attacked_scores, threshold)
    curves = compute_curve_payload(bundle.attacked_windows["attack_present"].to_numpy(), attacked_scores)

    predictions = bundle.attacked_windows[["window_start_utc", "window_end_utc", "scenario_id", "attack_present"]].copy()
    predictions["score"] = attacked_scores
    predictions["predicted"] = (attacked_scores >= threshold).astype(int)

    paths = ensure_model_paths("autoencoder", project_root)
    model_info = {
        "model_name": "autoencoder",
        "feature_columns": feature_columns,
        "threshold": threshold,
        "training_time_seconds": training_time,
        "inference_time_seconds": inference_time,
        "parameter_count": parameter_count(model),
    }
    write_pickle({"standardizer": standardizer, "model_info": model_info}, paths.model_dir / "metadata.pkl")
    write_torch_state(model, paths.model_dir / "model.pt")
    write_json({"metrics": metrics, "curves": curves, "model_info": model_info, "history": history}, paths.model_dir / "results.json")
    write_predictions(predictions, paths.model_dir / "predictions.parquet")
    return {"model_name": "autoencoder", "metrics": metrics, "info": model_info}


def train_autoencoder_model(
    project_root: str | Path | None = None,
    max_features: int = 128,
    epochs: int = 6,
    run_mode: str = "full",
    feature_counts: list[int] | None = None,
    buffer_windows: int = 2,
    patience: int = 5,
) -> dict[str, object]:
    """Handle train autoencoder model within the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    root = Path(project_root) if project_root is not None else ROOT
    if run_mode == "legacy-smoke":
        return _train_autoencoder_model_legacy(project_root=root, max_features=max_features, epochs=epochs)
    selected_feature_counts = feature_counts or [max_features]
    full_data = prepare_full_run_data(root, buffer_windows=buffer_windows)
    result = run_autoencoder(
        root=root,
        full_data=full_data,
        feature_counts=selected_feature_counts,
        epochs=epochs,
        patience=patience,
        report_root=root / "outputs" / "reports" / "model_full_run_artifacts",
        model_root_name="models_full_run",
        artifact_root_name="model_full_run_artifacts",
    )
    return {"model_name": "autoencoder", "metrics": result["summary"], "info": result["summary"]}


def main() -> None:
    """Run the command-line entrypoint for the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Train the repaired MLP autoencoder reconstruction baseline.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--max-features", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--run-mode", default="full", choices=["full", "legacy-smoke"])
    parser.add_argument("--feature-counts", default="")
    parser.add_argument("--buffer-windows", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    feature_counts = [int(item) for item in args.feature_counts.split(",") if item.strip()] if args.feature_counts else None
    result = train_autoencoder_model(
        project_root=args.project_root,
        max_features=args.max_features,
        epochs=args.epochs,
        run_mode=args.run_mode,
        feature_counts=feature_counts,
        buffer_windows=args.buffer_windows,
        patience=args.patience,
    )
    print(f"Saved autoencoder artifacts for {result['model_name']}")


if __name__ == "__main__":
    main()
