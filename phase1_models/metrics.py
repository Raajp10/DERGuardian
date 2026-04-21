"""Phase 1 detector training and evaluation support for DERGuardian.

This module implements metrics logic for residual-window model training,
inference, packaging, metrics, or reporting. It supports the frozen benchmark
path and related audits while keeping benchmark selection separate from replay,
heldout synthetic zero-day-like, and extension contexts.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve, f1_score

LATENCY_REFERENCE_CANDIDATES = (
    "detector_emission_utc",
    "emission_timestamp_utc",
    "detection_timestamp_utc",
    "window_end_utc",
    "window_start_utc",
)


def compute_binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | list[list[int]] | None]:
    """Compute binary metrics for the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    y_pred = (scores >= float(threshold)).astype(int)
    metrics: dict[str, float | list[list[int]] | None] = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_pred.mean()),
        "average_precision": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else None,
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "macro_precision": _macro_precision(y_true, y_pred),
        "macro_recall": _macro_recall(y_true, y_pred),
        "weighted_precision": _weighted_precision(y_true, y_pred),
        "weighted_recall": _weighted_recall(y_true, y_pred),
    }
    return metrics


def compute_curve_payload(y_true: np.ndarray, scores: np.ndarray) -> dict[str, list[float]]:
    """Compute curve payload for the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    payload: dict[str, list[float]] = {"roc_fpr": [], "roc_tpr": [], "pr_precision": [], "pr_recall": []}
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, scores)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        payload = {
            "roc_fpr": fpr.astype(float).tolist(),
            "roc_tpr": tpr.astype(float).tolist(),
            "pr_precision": precision.astype(float).tolist(),
            "pr_recall": recall.astype(float).tolist(),
        }
    return payload


def detection_latency_table(predictions: pd.DataFrame, labels: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Handle detection latency table within the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if predictions.empty or labels.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "scenario_id",
                "attack_start_utc",
                "first_detection_utc",
                "latency_seconds",
                "latency_reference_column",
            ]
        )
    frame = predictions.copy()
    if "window_start_utc" in frame.columns:
        frame["window_start_utc"] = pd.to_datetime(frame["window_start_utc"], utc=True)
    if "window_end_utc" in frame.columns:
        frame["window_end_utc"] = pd.to_datetime(frame["window_end_utc"], utc=True)
    latency_reference_column = _resolve_latency_reference_column(frame)
    frame["_latency_reference_utc"] = pd.to_datetime(frame[latency_reference_column], utc=True)
    labels_frame = labels.copy()
    labels_frame["start_time_utc"] = pd.to_datetime(labels_frame["start_time_utc"], utc=True)
    labels_frame["end_time_utc"] = pd.to_datetime(labels_frame["end_time_utc"], utc=True)
    rows: list[dict[str, object]] = []
    for _, label in labels_frame.iterrows():
        detections = frame[
            (frame["_latency_reference_utc"] >= label["start_time_utc"])
            & (frame["_latency_reference_utc"] <= label["end_time_utc"] + pd.Timedelta(minutes=15))
            & (frame["predicted"] == 1)
        ].sort_values("_latency_reference_utc")
        first_detection = detections["_latency_reference_utc"].iloc[0] if not detections.empty else pd.NaT
        latency = float((first_detection - label["start_time_utc"]).total_seconds()) if pd.notna(first_detection) else np.nan
        rows.append(
            {
                "model_name": model_name,
                "scenario_id": label["scenario_id"],
                "attack_start_utc": label["start_time_utc"],
                "first_detection_utc": first_detection,
                "latency_seconds": latency,
                "latency_reference_column": latency_reference_column,
            }
        )
    return pd.DataFrame(rows)


def per_scenario_metrics(predictions: pd.DataFrame, labels: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Handle per scenario metrics within the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if predictions.empty:
        return pd.DataFrame(columns=["model_name", "scenario_id", "precision", "recall", "f1", "support"])
    frame = predictions.copy()
    frame["window_start_utc"] = pd.to_datetime(frame["window_start_utc"], utc=True)
    if "window_end_utc" in frame.columns:
        frame["window_end_utc"] = pd.to_datetime(frame["window_end_utc"], utc=True)
    labels_frame = labels.copy()
    labels_frame["start_time_utc"] = pd.to_datetime(labels_frame["start_time_utc"], utc=True)
    labels_frame["end_time_utc"] = pd.to_datetime(labels_frame["end_time_utc"], utc=True)
    rows: list[dict[str, object]] = []
    for _, label in labels_frame.iterrows():
        scenario_id = str(label["scenario_id"])
        mask = (
            (frame["window_start_utc"] < label["end_time_utc"] + pd.Timedelta(minutes=15))
            & (frame.get("window_end_utc", frame["window_start_utc"]) > label["start_time_utc"] - pd.Timedelta(minutes=15))
        )
        scenario_frame = frame.loc[mask].copy()
        if scenario_frame.empty:
            rows.append({"model_name": model_name, "scenario_id": scenario_id, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})
            continue
        y_true = scenario_frame["attack_present"].astype(int).to_numpy()
        y_pred = scenario_frame["predicted"].astype(int).to_numpy()
        rows.append(
            {
                "model_name": model_name,
                "scenario_id": scenario_id,
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "support": int(y_true.sum()),
            }
        )
    return pd.DataFrame(rows)


def _macro_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((precision_score(y_true, y_pred, pos_label=0, zero_division=0) + precision_score(y_true, y_pred, pos_label=1, zero_division=0)) / 2.0)


def _macro_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((recall_score(y_true, y_pred, pos_label=0, zero_division=0) + recall_score(y_true, y_pred, pos_label=1, zero_division=0)) / 2.0)


def _weighted_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    support_0 = max(int((y_true == 0).sum()), 1)
    support_1 = max(int((y_true == 1).sum()), 1)
    total = support_0 + support_1
    p0 = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    p1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    return float((support_0 * p0 + support_1 * p1) / total)


def _weighted_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    support_0 = max(int((y_true == 0).sum()), 1)
    support_1 = max(int((y_true == 1).sum()), 1)
    total = support_0 + support_1
    r0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    r1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    return float((support_0 * r0 + support_1 * r1) / total)


def _resolve_latency_reference_column(frame: pd.DataFrame) -> str:
    for column in LATENCY_REFERENCE_CANDIDATES:
        if column in frame.columns:
            return column
    raise KeyError(
        "Predictions frame does not contain any supported latency reference column. "
        f"Expected one of: {LATENCY_REFERENCE_CANDIDATES}"
    )
