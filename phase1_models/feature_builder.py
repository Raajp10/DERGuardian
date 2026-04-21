"""Phase 1 detector training and evaluation support for DERGuardian.

This module implements feature builder logic for residual-window model training,
inference, packaging, metrics, or reporting. It supports the frozen benchmark
path and related audits while keeping benchmark selection separate from replay,
heldout synthetic zero-day-like, and extension contexts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


NON_FEATURE_COLUMNS = {
    "window_start_utc",
    "window_end_utc",
    "window_seconds",
    "scenario_id",
    "run_id",
    "split_id",
    "dataset_name",
    "attack_present",
    "attack_family",
    "attack_severity",
    "attack_affected_assets",
}


@dataclass(slots=True)
class Standardizer:
    """Structured object used by the Phase 1 detector modeling workflow."""

    mean: np.ndarray
    std: np.ndarray


@dataclass(slots=True)
class Discretizer:
    """Structured object used by the Phase 1 detector modeling workflow."""

    edges: dict[str, np.ndarray]


def select_numeric_feature_columns(df: pd.DataFrame, max_features: int = 128) -> list[str]:
    """Select numeric feature columns for the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    numeric = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not numeric:
        return []
    variance = df[numeric].var(ddof=0).sort_values(ascending=False)
    selected = [column for column in variance.index.tolist() if not np.isnan(variance[column])]
    return selected[:max_features]


def fit_standardizer(df: pd.DataFrame, feature_columns: list[str]) -> Standardizer:
    """Fit standardizer for the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    values = df[feature_columns].astype(float).fillna(0.0).to_numpy()
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-8] = 1.0
    return Standardizer(mean=mean, std=std)


def transform_features(df: pd.DataFrame, feature_columns: list[str], standardizer: Standardizer | None = None) -> np.ndarray:
    """Transform features for the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    values = df[feature_columns].astype(float).fillna(0.0).to_numpy()
    if standardizer is None:
        return values
    return (values - standardizer.mean) / standardizer.std


def build_sequence_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    seq_len: int,
    standardizer: Standardizer | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build sequence dataset for the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    frame = df.sort_values("window_start_utc").reset_index(drop=True)
    features = transform_features(frame, feature_columns, standardizer)
    labels = frame["attack_present"].astype(int).to_numpy() if "attack_present" in frame.columns else np.zeros(len(frame), dtype=int)
    sequences: list[np.ndarray] = []
    sequence_labels: list[int] = []
    rows: list[dict[str, object]] = []
    for end_idx in range(seq_len - 1, len(frame)):
        start_idx = end_idx - seq_len + 1
        sequences.append(features[start_idx : end_idx + 1])
        sequence_labels.append(int(labels[end_idx]))
        rows.append(
            {
                "window_start_utc": frame.loc[end_idx, "window_start_utc"],
                "window_end_utc": frame.loc[end_idx, "window_end_utc"],
                "scenario_id": frame.loc[end_idx, "scenario_id"],
                "attack_present": int(labels[end_idx]),
            }
        )
    return np.asarray(sequences, dtype=np.float32), np.asarray(sequence_labels, dtype=np.float32), pd.DataFrame(rows)


def fit_discretizer(df: pd.DataFrame, feature_columns: list[str], bins: int = 16) -> Discretizer:
    """Fit discretizer for the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    edges: dict[str, np.ndarray] = {}
    for column in feature_columns:
        series = df[column].astype(float).fillna(0.0)
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        boundaries = np.unique(np.quantile(series.to_numpy(), quantiles))
        if len(boundaries) < 3:
            boundaries = np.array([series.min() - 1e-6, series.max() + 1e-6], dtype=float)
        edges[column] = boundaries
    return Discretizer(edges=edges)


def transform_to_tokens(df: pd.DataFrame, feature_columns: list[str], discretizer: Discretizer) -> np.ndarray:
    """Transform to tokens for the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    tokens = []
    for column in feature_columns:
        series = df[column].astype(float).fillna(0.0).to_numpy()
        boundaries = discretizer.edges[column]
        token_ids = np.digitize(series, boundaries[1:-1], right=False)
        tokens.append(token_ids)
    return np.stack(tokens, axis=1).astype(np.int64)


def chronological_split(df: pd.DataFrame, train_fraction: float = 0.6, val_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Handle chronological split within the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    frame = df.sort_values("window_start_utc").reset_index(drop=True)
    n_rows = len(frame)
    train_end = max(int(n_rows * train_fraction), 1)
    val_end = max(train_end + int(n_rows * val_fraction), train_end + 1)
    val_end = min(val_end, n_rows)
    train = frame.iloc[:train_end].reset_index(drop=True)
    val = frame.iloc[train_end:val_end].reset_index(drop=True)
    test = frame.iloc[val_end:].reset_index(drop=True)
    if test.empty:
        test = val.copy()
    return train, val, test
