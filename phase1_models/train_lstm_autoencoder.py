"""Train and evaluate an extension-only LSTM autoencoder detector.

This module adds a real sequence reconstruction branch for DERGuardian without
altering the frozen canonical benchmark outputs. It uses the same residual
window artifacts as the Phase 1 detector comparison where feasible, trains only
on benign canonical windows, calibrates an anomaly threshold on validation
windows, and reports both canonical-test-split and heldout-synthetic
zero-day-like evidence as extension results.

Outputs are written under ``artifacts/extensions`` and ``docs/reports`` so the
existing benchmark-selected ``transformer @ 60s`` path remains the source of
truth.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml


NON_FEATURE_COLUMNS = {
    "window_start_utc",
    "window_end_utc",
    "window_seconds",
    "scenario_id",
    "run_id",
    "split",
    "split_id",
    "split_name",
    "dataset_name",
    "attack_present",
    "attack_family",
    "attack_severity",
    "attack_affected_assets",
    "scenario_window_id",
}


DEFAULT_CONFIG: dict[str, Any] = {
    "context": "extension_only_lstm_autoencoder",
    "canonical_benchmark_preserved": "transformer @ 60s",
    "input_contract": "canonical residual_windows.parquet feature columns; replay features missing in heldout artifacts are filled with neutral zero values before standardization",
    "window_labels": ["5s", "10s", "60s", "300s"],
    "feature_count": 64,
    "seq_len_by_window": {"5s": 12, "10s": 12, "60s": 12, "300s": 4},
    "epochs": 4,
    "batch_size": 128,
    "learning_rate": 0.001,
    "hidden_dim": 48,
    "latent_dim": 16,
    "seed": 42,
    "max_sequences": {"train": 5000, "val": 2500, "test": 3500, "heldout": 3500},
    "threshold_selection": "validation_f1_sweep_if_labels_exist_else_train_normal_99th_percentile",
    "output_note": "Extension evidence only; not used to replace the frozen canonical benchmark winner.",
}


class SequenceStandardizer:
    """Mean/std standardizer fit on benign canonical training residuals."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    @classmethod
    def fit(cls, frame: pd.DataFrame, feature_columns: list[str]) -> "SequenceStandardizer":
        values = _feature_matrix(frame, feature_columns, None)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < 1e-8] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
        values = _feature_matrix(frame, feature_columns, None)
        return (values - self.mean) / self.std


class LSTMSequenceAutoencoder(nn.Module):
    """Compact LSTM encoder/decoder for residual-window reconstruction."""

    def __init__(self, input_dim: int, hidden_dim: int = 48, latent_dim: int = 16) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.ReLU())
        self.from_latent = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        latent = self.to_latent(hidden[-1])
        decoder_seed = self.from_latent(latent).unsqueeze(1).repeat(1, x.shape[1], 1)
        decoded, _ = self.decoder(decoder_seed)
        return self.output(decoded)


def _feature_matrix(frame: pd.DataFrame, feature_columns: list[str], standardizer: SequenceStandardizer | None) -> np.ndarray:
    working = pd.DataFrame(index=frame.index)
    for column in feature_columns:
        working[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else 0.0
    values = working.astype(float).fillna(0.0).to_numpy(dtype=np.float32)
    if standardizer is None:
        return values
    return (values - standardizer.mean) / standardizer.std


def _load_config(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        return {**DEFAULT_CONFIG, **loaded}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(DEFAULT_CONFIG, handle, sort_keys=False)
    return DEFAULT_CONFIG.copy()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def _split_frame(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    split_column = None
    for candidate in ("split", "split_name", "split_id"):
        if candidate in frame.columns:
            split_column = candidate
            break
    if split_column is not None:
        return frame.loc[frame[split_column].astype(str).str.lower() == split_name].copy()
    # Fallback for older artifacts: preserve chronological train/val/test order.
    ordered = frame.sort_values("window_start_utc").reset_index(drop=True)
    n_rows = len(ordered)
    train_end = int(n_rows * 0.6)
    val_end = int(n_rows * 0.8)
    if split_name == "train":
        return ordered.iloc[:train_end].copy()
    if split_name == "val":
        return ordered.iloc[train_end:val_end].copy()
    return ordered.iloc[val_end:].copy()


def _select_features(frame: pd.DataFrame, max_features: int) -> list[str]:
    numeric = [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not numeric:
        return []
    variance = frame[numeric].astype(float).fillna(0.0).var(ddof=0).sort_values(ascending=False)
    return [column for column in variance.index if math.isfinite(float(variance[column]))][:max_features]


def _build_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    standardizer: SequenceStandardizer,
    seq_len: int,
    max_sequences: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    frame = frame.sort_values(["scenario_id", "window_start_utc"]).reset_index(drop=True)
    if frame.empty or len(frame) < seq_len:
        return np.empty((0, seq_len, len(feature_columns)), dtype=np.float32), np.empty((0,), dtype=np.int64), pd.DataFrame()

    transformed = standardizer.transform(frame, feature_columns).astype(np.float32)
    labels = frame["attack_present"].astype(int).to_numpy() if "attack_present" in frame.columns else np.zeros(len(frame), dtype=np.int64)
    sequences: list[np.ndarray] = []
    sequence_labels: list[int] = []
    rows: list[dict[str, Any]] = []

    for _, group in frame.groupby("scenario_id", sort=False):
        indices = group.index.to_numpy()
        if len(indices) < seq_len:
            continue
        for offset in range(seq_len - 1, len(indices)):
            window_indices = indices[offset - seq_len + 1 : offset + 1]
            end_idx = int(window_indices[-1])
            sequences.append(transformed[window_indices])
            sequence_labels.append(int(labels[end_idx]))
            rows.append(
                {
                    "window_start_utc": frame.loc[end_idx, "window_start_utc"],
                    "window_end_utc": frame.loc[end_idx, "window_end_utc"],
                    "scenario_id": frame.loc[end_idx, "scenario_id"],
                    "attack_present": int(labels[end_idx]),
                    "attack_family": frame.loc[end_idx, "attack_family"] if "attack_family" in frame.columns else "unknown",
                    "attack_severity": frame.loc[end_idx, "attack_severity"] if "attack_severity" in frame.columns else "unknown",
                    "attack_affected_assets": frame.loc[end_idx, "attack_affected_assets"] if "attack_affected_assets" in frame.columns else "",
                }
            )

    if not sequences:
        return np.empty((0, seq_len, len(feature_columns)), dtype=np.float32), np.empty((0,), dtype=np.int64), pd.DataFrame()

    x = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(sequence_labels, dtype=np.int64)
    meta = pd.DataFrame(rows)
    if max_sequences is not None and len(y) > max_sequences:
        # Deterministic thinning keeps broad temporal coverage without adding randomness to the reported run.
        keep = np.linspace(0, len(y) - 1, max_sequences, dtype=int)
        x = x[keep]
        y = y[keep]
        meta = meta.iloc[keep].reset_index(drop=True)
    return x, y, meta.reset_index(drop=True)


def _train_model(
    model: nn.Module,
    train_x: np.ndarray,
    val_x: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[nn.Module, dict[str, list[float]], float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loader = DataLoader(TensorDataset(torch.tensor(train_x, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_x, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_val = float("inf")
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        train_losses: list[float] = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                val_losses.append(float(criterion(model(batch), batch).item()))
        mean_train = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_val = float(np.mean(val_losses)) if val_losses else float("nan")
        history["train_loss"].append(mean_train)
        history["val_loss"].append(mean_val)
        if mean_val < best_val:
            best_val = mean_val
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    training_time = time.perf_counter() - start
    model.load_state_dict(best_state)
    return model.cpu(), history, training_time


def _score_model(model: nn.Module, x: np.ndarray, batch_size: int) -> tuple[np.ndarray, float]:
    if len(x) == 0:
        return np.array([], dtype=float), 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()
    scores: list[np.ndarray] = []
    start = time.perf_counter()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            score = torch.mean((recon - batch) ** 2, dim=(1, 2)).cpu().numpy()
            scores.append(score)
    elapsed = time.perf_counter() - start
    return np.concatenate(scores, axis=0), elapsed


def _calibrate_threshold(train_scores: np.ndarray, val_scores: np.ndarray, val_y: np.ndarray) -> tuple[float, str]:
    if len(val_scores) and len(np.unique(val_y)) > 1:
        candidates = np.unique(np.quantile(val_scores, np.linspace(0.05, 0.995, 180)))
        best_threshold = float(candidates[0])
        best_f1 = -1.0
        for threshold in candidates:
            score = f1_score(val_y, (val_scores >= threshold).astype(int), zero_division=0)
            if score > best_f1:
                best_f1 = float(score)
                best_threshold = float(threshold)
        return best_threshold, "validation_f1_sweep"
    return float(np.quantile(train_scores, 0.99)) if len(train_scores) else 0.0, "train_normal_99th_percentile"


def _metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else np.nan,
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else np.nan,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else np.nan,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def _latency_seconds(predictions: pd.DataFrame) -> tuple[float, float]:
    if predictions.empty or "window_start_utc" not in predictions.columns:
        return np.nan, np.nan
    frame = predictions.copy()
    frame["window_start_utc"] = pd.to_datetime(frame["window_start_utc"], utc=True)
    latencies: list[float] = []
    for scenario_id, group in frame.groupby("scenario_id"):
        if str(scenario_id).lower() == "benign":
            continue
        positives = group.loc[group["attack_present"].astype(int) == 1].sort_values("window_start_utc")
        if positives.empty:
            continue
        attack_start = positives["window_start_utc"].iloc[0]
        detections = group.loc[(group["predicted"].astype(int) == 1) & (group["window_start_utc"] >= attack_start)].sort_values("window_start_utc")
        if not detections.empty:
            latencies.append(float((detections["window_start_utc"].iloc[0] - attack_start).total_seconds()))
    if not latencies:
        return np.nan, np.nan
    return float(np.mean(latencies)), float(np.median(latencies))


def _write_predictions(meta: pd.DataFrame, scores: np.ndarray, threshold: float, path: Path, split_name: str) -> pd.DataFrame:
    predictions = meta.copy()
    predictions["split_name"] = split_name
    predictions["score"] = scores
    predictions["predicted"] = (scores >= threshold).astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(path, index=False)
    return predictions


def _evaluate_predictions(
    *,
    context: str,
    window_label: str,
    window_seconds: int,
    step_seconds: int,
    predictions: pd.DataFrame,
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    threshold_method: str,
    inference_time: float,
    training_time: float,
    feature_columns: list[str],
    seq_len: int,
    parameter_count: int,
    model_size_bytes: int,
    sequence_counts: dict[str, int],
    predictions_path: Path,
    notes: str,
) -> dict[str, Any]:
    metric_values = _metrics(y_true, scores, threshold) if len(y_true) else {}
    mean_latency, median_latency = _latency_seconds(predictions)
    row = {
        "evaluation_context": context,
        "canonical_or_extension": "extension",
        "model_name": "lstm_autoencoder",
        "window_label": window_label,
        "window_seconds": window_seconds,
        "step_seconds": step_seconds,
        "status": "completed" if len(y_true) else "blocked",
        "threshold": threshold,
        "threshold_method": threshold_method,
        "feature_count": len(feature_columns),
        "seq_len": seq_len,
        "training_time_seconds": training_time,
        "inference_time_seconds": inference_time,
        "inference_time_ms_per_prediction": float((inference_time / len(y_true)) * 1000.0) if len(y_true) else np.nan,
        "parameter_count": parameter_count,
        "model_size_bytes": model_size_bytes,
        "mean_detection_latency_seconds": mean_latency,
        "median_detection_latency_seconds": median_latency,
        "evaluated_sequences": int(len(y_true)),
        "attack_sequences": int(np.asarray(y_true).sum()) if len(y_true) else 0,
        "prediction_path": _repo_rel(predictions_path),
        "notes": notes,
    }
    row.update(metric_values)
    row.update(sequence_counts)
    return row


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _window_seconds(label: str) -> int:
    return int(label.rstrip("s"))


def _step_seconds(label: str) -> int:
    return max(1, _window_seconds(label) // 5)


def _evaluate_heldout_inputs(
    *,
    root: Path,
    window_label: str,
    feature_columns: list[str],
    standardizer: SequenceStandardizer,
    model: nn.Module,
    threshold: float,
    threshold_method: str,
    config: dict[str, Any],
    training_time: float,
    parameter_count: int,
    model_size_bytes: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    heldout_root = root / "outputs" / "window_size_study" / "improved_phase3" / "replay_inputs"
    if not heldout_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    paths = sorted(heldout_root.glob(f"*/*{window_label}/residual_windows.parquet"))
    if not paths:
        paths = sorted(heldout_root.glob(f"*/{window_label}/residual_windows.parquet"))
    for path in paths[:6]:
        frame = pd.read_parquet(path)
        seq_len = int(config["seq_len_by_window"][window_label])
        x, y, meta = _build_sequences(
            frame,
            feature_columns,
            standardizer,
            seq_len,
            max_sequences=int(config["max_sequences"].get("heldout", 3500)),
        )
        scores, inference_time = _score_model(model, x, int(config["batch_size"]))
        bundle_id = path.parent.parent.name if path.parent.name == window_label else path.parent.name
        prediction_path = output_dir / "predictions" / f"lstm_autoencoder_{window_label}_{_safe_name(bundle_id)}.parquet"
        predictions = _write_predictions(meta, scores, threshold, prediction_path, "heldout_synthetic")
        rows.append(
            _evaluate_predictions(
                context="heldout_synthetic_zero_day_like_extension",
                window_label=window_label,
                window_seconds=_window_seconds(window_label),
                step_seconds=_step_seconds(window_label),
                predictions=predictions,
                scores=scores,
                y_true=y,
                threshold=threshold,
                threshold_method=threshold_method,
                inference_time=inference_time,
                training_time=training_time,
                feature_columns=feature_columns,
                seq_len=seq_len,
                parameter_count=parameter_count,
                model_size_bytes=model_size_bytes,
                sequence_counts={"heldout_sequences": int(len(y)), "train_sequences": np.nan, "val_sequences": np.nan, "test_sequences": np.nan},
                predictions_path=prediction_path,
                notes=f"Heldout synthetic replay residuals from {bundle_id}; extension evaluation only.",
            )
        )
    return rows


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)[:120]


def run_lstm_autoencoder_extension(project_root: Path, config_path: Path) -> pd.DataFrame:
    """Run the extension experiment and return all result rows."""

    config = _load_config(config_path)
    _seed_everything(int(config["seed"]))
    extension_dir = project_root / "artifacts" / "extensions"
    docs_dir = project_root / "docs" / "reports"
    figure_dir = project_root / "docs" / "figures"
    output_dir = project_root / "outputs" / "phase1_lstm_autoencoder_extension"
    for directory in (extension_dir, docs_dir, figure_dir, output_dir):
        directory.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    window_models: dict[str, tuple[nn.Module, SequenceStandardizer, list[str], float, str, int, int, int, float]] = {}

    for window_label in config["window_labels"]:
        residual_path = project_root / "outputs" / "window_size_study" / window_label / "residual_windows.parquet"
        if not residual_path.exists():
            all_rows.append(
                {
                    "evaluation_context": "canonical_benchmark_test_split_extension",
                    "canonical_or_extension": "extension",
                    "model_name": "lstm_autoencoder",
                    "window_label": window_label,
                    "window_seconds": _window_seconds(window_label),
                    "status": "blocked",
                    "notes": f"Missing residual window artifact: {_repo_rel(residual_path)}",
                }
            )
            continue

        frame = pd.read_parquet(residual_path)
        frame["window_start_utc"] = pd.to_datetime(frame["window_start_utc"], utc=True)
        train_frame = _split_frame(frame, "train")
        val_frame = _split_frame(frame, "val")
        test_frame = _split_frame(frame, "test")
        train_clean = train_frame.loc[train_frame["attack_present"].astype(int) == 0].copy()
        if train_clean.empty:
            train_clean = train_frame.copy()
        feature_columns = _select_features(train_clean, int(config["feature_count"]))
        seq_len = int(config["seq_len_by_window"][window_label])
        standardizer = SequenceStandardizer.fit(train_clean, feature_columns)

        train_x, train_y, _ = _build_sequences(
            train_clean,
            feature_columns,
            standardizer,
            seq_len,
            max_sequences=int(config["max_sequences"]["train"]),
        )
        val_x, val_y, val_meta = _build_sequences(
            val_frame,
            feature_columns,
            standardizer,
            seq_len,
            max_sequences=int(config["max_sequences"]["val"]),
        )
        test_x, test_y, test_meta = _build_sequences(
            test_frame,
            feature_columns,
            standardizer,
            seq_len,
            max_sequences=int(config["max_sequences"]["test"]),
        )
        if len(train_x) == 0 or len(val_x) == 0 or len(test_x) == 0:
            all_rows.append(
                {
                    "evaluation_context": "canonical_benchmark_test_split_extension",
                    "canonical_or_extension": "extension",
                    "model_name": "lstm_autoencoder",
                    "window_label": window_label,
                    "window_seconds": _window_seconds(window_label),
                    "status": "blocked",
                    "notes": "Insufficient residual windows to build train/validation/test sequences.",
                }
            )
            continue

        model = LSTMSequenceAutoencoder(
            input_dim=len(feature_columns),
            hidden_dim=int(config["hidden_dim"]),
            latent_dim=int(config["latent_dim"]),
        )
        model, history, training_time = _train_model(
            model,
            train_x,
            val_x,
            epochs=int(config["epochs"]),
            batch_size=int(config["batch_size"]),
            learning_rate=float(config["learning_rate"]),
        )
        train_scores, _ = _score_model(model, train_x, int(config["batch_size"]))
        val_scores, _ = _score_model(model, val_x, int(config["batch_size"]))
        threshold, threshold_method = _calibrate_threshold(train_scores, val_scores, val_y)
        test_scores, inference_time = _score_model(model, test_x, int(config["batch_size"]))
        prediction_path = output_dir / "predictions" / f"lstm_autoencoder_{window_label}_canonical_test.parquet"
        predictions = _write_predictions(test_meta, test_scores, threshold, prediction_path, "test")
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        model_size_bytes = parameter_count * 4
        checkpoint_path = output_dir / "checkpoints" / f"lstm_autoencoder_{window_label}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_columns": feature_columns,
                "standardizer_mean": standardizer.mean,
                "standardizer_std": standardizer.std,
                "threshold": threshold,
                "config": config,
                "history": history,
            },
            checkpoint_path,
        )
        sequence_counts = {
            "train_sequences": int(len(train_x)),
            "val_sequences": int(len(val_y)),
            "test_sequences": int(len(test_y)),
        }
        all_rows.append(
            _evaluate_predictions(
                context="canonical_benchmark_test_split_extension",
                window_label=window_label,
                window_seconds=_window_seconds(window_label),
                step_seconds=_step_seconds(window_label),
                predictions=predictions,
                scores=test_scores,
                y_true=test_y,
                threshold=threshold,
                threshold_method=threshold_method,
                inference_time=inference_time,
                training_time=training_time,
                feature_columns=feature_columns,
                seq_len=seq_len,
                parameter_count=parameter_count,
                model_size_bytes=model_size_bytes,
                sequence_counts=sequence_counts,
                predictions_path=prediction_path,
                notes=(
                    "LSTM autoencoder extension trained on benign canonical residual windows. "
                    "It is distinct from the existing MLP autoencoder and does not alter canonical model selection."
                ),
            )
        )
        window_models[window_label] = (
            model,
            standardizer,
            feature_columns,
            threshold,
            threshold_method,
            parameter_count,
            model_size_bytes,
            seq_len,
            training_time,
        )

        all_rows.extend(
            _evaluate_heldout_inputs(
                root=project_root,
                window_label=window_label,
                feature_columns=feature_columns,
                standardizer=standardizer,
                model=model,
                threshold=threshold,
                threshold_method=threshold_method,
                config=config,
                training_time=training_time,
                parameter_count=parameter_count,
                model_size_bytes=model_size_bytes,
                output_dir=output_dir,
            )
        )

    results = pd.DataFrame(all_rows)
    results_path = extension_dir / "phase1_lstm_autoencoder_results.csv"
    results.to_csv(results_path, index=False)
    _write_vs_all(project_root, results, extension_dir / "phase1_lstm_autoencoder_vs_all_models.csv")
    _write_report(results, config, docs_dir / "phase1_lstm_autoencoder_eval_report.md", results_path)
    _write_figure(results, figure_dir / "phase1_lstm_autoencoder_comparison.png")
    return results


def _write_vs_all(project_root: Path, results: pd.DataFrame, output_path: Path) -> None:
    benchmark_path = project_root / "artifacts" / "benchmark" / "phase1_window_model_comparison_full.csv"
    if benchmark_path.exists():
        benchmark = pd.read_csv(benchmark_path)
    else:
        benchmark = pd.DataFrame()
    canonical_rows = results.loc[results["evaluation_context"].eq("canonical_benchmark_test_split_extension")].copy()
    common_columns = list(dict.fromkeys(list(benchmark.columns) + list(canonical_rows.columns)))
    combined = pd.concat(
        [benchmark.reindex(columns=common_columns), canonical_rows.reindex(columns=common_columns)],
        ignore_index=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)


def _write_report(results: pd.DataFrame, config: dict[str, Any], output_path: Path, results_path: Path) -> None:
    canonical = results.loc[results["evaluation_context"].eq("canonical_benchmark_test_split_extension")].copy()
    heldout = results.loc[results["evaluation_context"].eq("heldout_synthetic_zero_day_like_extension")].copy()
    best = (
        canonical.loc[canonical.get("f1", pd.Series(dtype=float)).notna()].sort_values("f1", ascending=False).head(1)
        if "f1" in canonical.columns
        else pd.DataFrame()
    )
    best_text = "No completed LSTM autoencoder window was produced."
    if not best.empty:
        row = best.iloc[0]
        best_text = f"Best extension row: {row['window_label']} with F1={row['f1']:.4f}, precision={row['precision']:.4f}, recall={row['recall']:.4f}."
    lines = [
        "# Phase 1 LSTM Autoencoder Extension Evaluation",
        "",
        "This report documents a newly added LSTM autoencoder detector branch. It is an extension experiment only.",
        "The frozen canonical benchmark winner remains **transformer @ 60s**; this branch does not replace or relabel that result.",
        "",
        "## Input Contract",
        "",
        f"- Residual window inputs: `outputs/window_size_study/<window>/residual_windows.parquet`.",
        f"- Feature count: {config['feature_count']} high-variance residual features selected from benign canonical training windows.",
        "- Training data: benign canonical training split only.",
        "- Thresholding: validation-label F1 sweep when labels are available; otherwise 99th percentile of train-normal reconstruction error.",
        "- Heldout synthetic evaluation: available replay residual inputs are scored with the trained extension model and reported separately.",
        "",
        "## Canonical-Test-Split Extension Rows",
        "",
        canonical.reindex(
            columns=[
                "window_label",
                "status",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "false_positive_rate",
                "evaluated_sequences",
                "notes",
            ]
        ).to_markdown(index=False) if not canonical.empty else "No canonical-test-split rows were produced.",
        "",
        "## Heldout Synthetic Rows",
        "",
        heldout.reindex(
            columns=[
                "window_label",
                "status",
                "precision",
                "recall",
                "f1",
                "evaluated_sequences",
                "notes",
            ]
        ).head(20).to_markdown(index=False) if not heldout.empty else "No heldout synthetic rows were produced.",
        "",
        "## Decision",
        "",
        best_text,
        "These results are useful evidence for a real LSTM autoencoder branch, but they are not canonical benchmark selection evidence.",
        f"Source CSV: `{_repo_rel(results_path)}`.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figure(results: pd.DataFrame, output_path: Path) -> None:
    canonical = results.loc[results["evaluation_context"].eq("canonical_benchmark_test_split_extension")].copy()
    canonical = canonical.loc[canonical["status"].eq("completed")]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    if canonical.empty:
        plt.text(0.5, 0.5, "No completed LSTM autoencoder rows", ha="center", va="center")
        plt.axis("off")
    else:
        x = np.arange(len(canonical))
        plt.bar(x - 0.2, canonical["f1"].astype(float), width=0.4, label="F1")
        plt.bar(x + 0.2, canonical["recall"].astype(float), width=0.4, label="Recall")
        plt.xticks(x, canonical["window_label"].astype(str))
        plt.ylim(0, 1.05)
        plt.ylabel("Score")
        plt.title("LSTM Autoencoder Extension by Window")
        plt.legend()
        plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    """Run the LSTM autoencoder extension from the command line."""

    parser = argparse.ArgumentParser(description="Train/evaluate extension-only LSTM autoencoder detector.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument(
        "--config",
        default=str(ROOT / "artifacts" / "extensions" / "phase1_lstm_autoencoder_config.yaml"),
        help="Extension config path. Created with safe defaults when missing.",
    )
    args = parser.parse_args()
    results = run_lstm_autoencoder_extension(Path(args.project_root), Path(args.config))
    print(json.dumps({"rows": int(len(results)), "completed": int(results["status"].eq("completed").sum())}, indent=2))


if __name__ == "__main__":
    main()
