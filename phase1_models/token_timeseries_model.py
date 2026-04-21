from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from phase1_models.dataset_loader import load_window_dataset_bundle
from phase1_models.feature_builder import chronological_split, fit_discretizer, select_numeric_feature_columns, transform_to_tokens
from phase1_models.metrics import compute_binary_metrics, compute_curve_payload
from phase1_models.model_utils import ensure_model_paths, parameter_count, write_json, write_pickle, write_predictions, write_torch_state
from phase1_models.neural_models import TokenBaselineClassifier
from phase1_models.neural_training import predict_classifier_scores, train_classifier
from phase1_models.run_full_evaluation import prepare_full_run_data, run_sequence_model


TOKEN_BASELINE_INTERNAL_NAME = "llm_baseline"
TOKEN_BASELINE_DISPLAY_NAME = "Tokenized LLM-Style Baseline"
TOKEN_BASELINE_NOTES = (
    "Tokenized time-series language-style baseline that quantizes numeric residual features into discrete bins "
    "and learns token-sequence patterns. It is an experimental local sequence model, not a foundation LLM."
)


def build_token_model(vocab_size: int = 16, embed_dim: int = 32) -> TokenBaselineClassifier:
    return TokenBaselineClassifier(vocab_size=vocab_size, embed_dim=embed_dim)


def train_token_timeseries_language_style_baseline(
    project_root: str | Path | None = None,
    max_features: int = 48,
    seq_len: int = 8,
    epochs: int = 6,
    run_mode: str = "full",
    feature_counts: list[int] | None = None,
    seq_lens: list[int] | None = None,
    buffer_windows: int = 2,
    patience: int = 5,
) -> dict[str, object]:
    root = Path(project_root) if project_root is not None else ROOT
    if run_mode == "legacy-smoke":
        return _train_token_baseline_legacy(project_root=root, max_features=max_features, seq_len=seq_len, epochs=epochs)

    selected_feature_counts = feature_counts or [max_features]
    selected_seq_lens = seq_lens or [seq_len]
    full_data = prepare_full_run_data(root, buffer_windows=buffer_windows)
    result = run_sequence_model(
        root=root,
        full_data=full_data,
        model_name=TOKEN_BASELINE_INTERNAL_NAME,
        model_ctor=lambda _: build_token_model(vocab_size=16, embed_dim=32),
        feature_counts=selected_feature_counts,
        seq_lens=selected_seq_lens,
        epochs=epochs,
        patience=patience,
        token_input=True,
        report_root=root / "outputs" / "reports" / "model_full_run_artifacts",
        model_root_name="models_full_run",
        artifact_root_name="model_full_run_artifacts",
    )
    return {"model_name": TOKEN_BASELINE_INTERNAL_NAME, "metrics": result["summary"], "info": result["summary"]}


def _train_token_baseline_legacy(
    project_root: str | Path | None = None,
    max_features: int = 48,
    seq_len: int = 8,
    epochs: int = 6,
) -> dict[str, object]:
    bundle = load_window_dataset_bundle(project_root)
    feature_columns = select_numeric_feature_columns(bundle.clean_windows, max_features=max_features)
    discretizer = fit_discretizer(bundle.clean_windows, feature_columns, bins=16)
    train_df, val_df, test_df = chronological_split(bundle.attacked_windows)

    x_train_tokens = transform_to_tokens(train_df, feature_columns, discretizer)
    x_val_tokens = transform_to_tokens(val_df, feature_columns, discretizer)
    x_test_tokens = transform_to_tokens(test_df, feature_columns, discretizer)
    y_train = train_df["attack_present"].astype(float).to_numpy()
    y_val = val_df["attack_present"].astype(float).to_numpy()
    y_test = test_df["attack_present"].astype(float).to_numpy()

    x_train, y_train_seq, _ = _build_token_sequences(x_train_tokens, y_train, train_df.reset_index(drop=True), seq_len)
    x_val, y_val_seq, _ = _build_token_sequences(x_val_tokens, y_val, val_df.reset_index(drop=True), seq_len)
    x_test, y_test_seq, rows = _build_token_sequences(x_test_tokens, y_test, test_df.reset_index(drop=True), seq_len)

    model = build_token_model(vocab_size=16, embed_dim=32)
    model, history, training_time = train_classifier(
        model,
        x_train,
        y_train_seq,
        x_val,
        y_val_seq,
        epochs=epochs,
        token_input=True,
    )
    inference_start = time.perf_counter()
    test_scores = predict_classifier_scores(model, x_test, token_input=True)
    inference_time = time.perf_counter() - inference_start
    threshold = 0.5
    metrics = compute_binary_metrics(y_test_seq, test_scores, threshold)
    curves = compute_curve_payload(y_test_seq, test_scores)

    predictions = pd.DataFrame(rows)
    predictions["score"] = test_scores
    predictions["predicted"] = (test_scores >= threshold).astype(int)

    paths = ensure_model_paths(TOKEN_BASELINE_INTERNAL_NAME, project_root)
    model_info = {
        "model_name": TOKEN_BASELINE_INTERNAL_NAME,
        "display_name": TOKEN_BASELINE_DISPLAY_NAME,
        "feature_columns": feature_columns,
        "seq_len": seq_len,
        "threshold": threshold,
        "training_time_seconds": training_time,
        "inference_time_seconds": inference_time,
        "parameter_count": parameter_count(model),
        "notes": TOKEN_BASELINE_NOTES,
    }
    write_pickle({"discretizer": discretizer, "model_info": model_info}, paths.model_dir / "metadata.pkl")
    write_torch_state(model, paths.model_dir / "model.pt")
    write_json({"metrics": metrics, "curves": curves, "model_info": model_info, "history": history}, paths.model_dir / "results.json")
    write_predictions(predictions, paths.model_dir / "predictions.parquet")
    return {"model_name": TOKEN_BASELINE_INTERNAL_NAME, "metrics": metrics, "info": model_info}


def _build_token_sequences(tokens: np.ndarray, labels: np.ndarray, frame: pd.DataFrame, seq_len: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    sequences: list[np.ndarray] = []
    sequence_labels: list[float] = []
    rows: list[dict[str, object]] = []
    for end_idx in range(seq_len - 1, len(tokens)):
        start_idx = end_idx - seq_len + 1
        sequences.append(tokens[start_idx : end_idx + 1])
        sequence_labels.append(float(labels[end_idx]))
        rows.append(
            {
                "window_start_utc": frame.iloc[end_idx]["window_start_utc"],
                "window_end_utc": frame.iloc[end_idx]["window_end_utc"],
                "scenario_id": frame.iloc[end_idx]["scenario_id"],
                "attack_present": int(labels[end_idx]),
            }
        )
    return np.asarray(sequences, dtype=np.int64), np.asarray(sequence_labels, dtype=np.float32), rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the tokenized time-series language-style baseline.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--max-features", type=int, default=48)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--run-mode", default="full", choices=["full", "legacy-smoke"])
    parser.add_argument("--feature-counts", default="")
    parser.add_argument("--seq-lens", default="")
    parser.add_argument("--buffer-windows", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    feature_counts = [int(item) for item in args.feature_counts.split(",") if item.strip()] if args.feature_counts else None
    seq_lens = [int(item) for item in args.seq_lens.split(",") if item.strip()] if args.seq_lens else None
    result = train_token_timeseries_language_style_baseline(
        project_root=args.project_root,
        max_features=args.max_features,
        seq_len=args.seq_len,
        epochs=args.epochs,
        run_mode=args.run_mode,
        feature_counts=feature_counts,
        seq_lens=seq_lens,
        buffer_windows=args.buffer_windows,
        patience=args.patience,
    )
    print(f"Saved token baseline artifacts for {result['model_name']}")


if __name__ == "__main__":
    main()
