"""Phase 1 detector training and evaluation support for DERGuardian.

This module implements dataset loader logic for residual-window model training,
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

import pandas as pd

from common.io_utils import read_dataframe


@dataclass(slots=True)
class WindowDatasetBundle:
    """Structured object used by the Phase 1 detector modeling workflow."""

    clean_windows: pd.DataFrame
    attacked_windows: pd.DataFrame
    attack_labels: pd.DataFrame


def load_window_dataset_bundle(project_root: str | Path | None = None) -> WindowDatasetBundle:
    """Load window dataset bundle for the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    root = Path(project_root) if project_root is not None else ROOT
    clean_path = root / "outputs" / "windows" / "merged_windows_clean.parquet"
    attacked_candidates = [
        root / "outputs" / "attacked" / "merged_windows.parquet",
        root / "outputs" / "windows" / "merged_windows_attacked.parquet",
    ]
    labels_candidates = [
        root / "outputs" / "attacked" / "attack_labels.parquet",
        root / "outputs" / "labeled" / "attack_labels.csv",
    ]
    attacked_path = next((path for path in attacked_candidates if path.exists()), attacked_candidates[-1])
    labels_path = next((path for path in labels_candidates if path.exists()), labels_candidates[-1])

    clean_windows = _prepare_windows(read_dataframe(clean_path), "clean")
    attacked_windows = _prepare_windows(read_dataframe(attacked_path), "attacked")
    attack_labels = read_dataframe(labels_path)
    if not attack_labels.empty:
        for column in ["start_time_utc", "end_time_utc"]:
            if column in attack_labels.columns:
                attack_labels[column] = pd.to_datetime(attack_labels[column], utc=True)
    return WindowDatasetBundle(clean_windows=clean_windows, attacked_windows=attacked_windows, attack_labels=attack_labels)


def _prepare_windows(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    frame = df.copy()
    for column in ["window_start_utc", "window_end_utc"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    frame["dataset_name"] = dataset_name
    if "attack_present" not in frame.columns:
        frame["attack_present"] = 0
    return frame
