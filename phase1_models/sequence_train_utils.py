"""Phase 1 detector training and evaluation support for DERGuardian.

This module implements sequence train utils logic for residual-window model training,
inference, packaging, metrics, or reporting. It supports the frozen benchmark
path and related audits while keeping benchmark selection separate from replay,
heldout synthetic zero-day-like, and extension contexts.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from phase1_models.dataset_loader import load_window_dataset_bundle
from phase1_models.feature_builder import build_sequence_dataset, chronological_split, fit_standardizer, select_numeric_feature_columns
from phase1_models.metrics import compute_binary_metrics, compute_curve_payload
from phase1_models.model_utils import ensure_model_paths, parameter_count, write_json, write_pickle, write_predictions, write_torch_state
from phase1_models.neural_training import predict_classifier_scores, train_classifier


def train_sequence_classifier_model(
    model_name: str,
    model_ctor,
    project_root: str | Path | None = None,
    max_features: int = 96,
    seq_len: int = 8,
    epochs: int = 6,
) -> dict[str, object]:
    """Handle train sequence classifier model within the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    bundle = load_window_dataset_bundle(project_root)
    feature_columns = select_numeric_feature_columns(bundle.clean_windows, max_features=max_features)
    standardizer = fit_standardizer(bundle.clean_windows, feature_columns)
    train_df, val_df, test_df = chronological_split(bundle.attacked_windows)
    x_train, y_train, train_meta = build_sequence_dataset(train_df, feature_columns, seq_len, standardizer)
    x_val, y_val, val_meta = build_sequence_dataset(val_df, feature_columns, seq_len, standardizer)
    x_test, y_test, test_meta = build_sequence_dataset(test_df, feature_columns, seq_len, standardizer)

    model = model_ctor(len(feature_columns))
    model, history, training_time = train_classifier(model, x_train, y_train, x_val, y_val, epochs=epochs)
    inference_start = time.perf_counter()
    test_scores = predict_classifier_scores(model, x_test)
    inference_time = time.perf_counter() - inference_start
    threshold = 0.5
    metrics = compute_binary_metrics(y_test, test_scores, threshold)
    curves = compute_curve_payload(y_test, test_scores)
    predictions = test_meta.copy()
    predictions["score"] = test_scores
    predictions["predicted"] = (test_scores >= threshold).astype(int)

    paths = ensure_model_paths(model_name, project_root)
    model_info = {
        "model_name": model_name,
        "feature_columns": feature_columns,
        "seq_len": seq_len,
        "threshold": threshold,
        "training_time_seconds": training_time,
        "inference_time_seconds": inference_time,
        "parameter_count": parameter_count(model),
    }
    write_pickle({"standardizer": standardizer, "model_info": model_info}, paths.model_dir / "metadata.pkl")
    write_torch_state(model, paths.model_dir / "model.pt")
    write_json({"metrics": metrics, "curves": curves, "model_info": model_info, "history": history}, paths.model_dir / "results.json")
    write_predictions(predictions, paths.model_dir / "predictions.parquet")
    return {"model_name": model_name, "metrics": metrics, "info": model_info}
