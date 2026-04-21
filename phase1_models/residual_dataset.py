from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.io_utils import read_dataframe, write_dataframe
from phase1_models.model_utils import CANONICAL_ARTIFACT_ROOT


RESIDUAL_ARTIFACT_FILENAME = "residual_windows_full_run.parquet"

WINDOW_META_COLUMNS = {
    "window_start_utc",
    "window_end_utc",
    "window_seconds",
    "scenario_id",
    "run_id",
    "split_id",
    "attack_present",
    "attack_family",
    "attack_severity",
    "attack_affected_assets",
}

REQUIRED_RESIDUAL_COLUMNS = {
    "window_start_utc",
    "window_end_utc",
    "window_seconds",
    "scenario_id",
    "run_id",
    "split_id",
    "attack_present",
    "attack_family",
    "attack_severity",
    "attack_affected_assets",
    "scenario_window_id",
    "split_name",
}


def canonical_residual_artifact_path(root: Path, artifact_root_name: str = CANONICAL_ARTIFACT_ROOT) -> Path:
    return root / "outputs" / "reports" / artifact_root_name / RESIDUAL_ARTIFACT_FILENAME


def canonical_window_source_paths(root: Path) -> tuple[Path, Path, Path]:
    return (
        root / "outputs" / "windows" / "merged_windows_clean.parquet",
        root / "outputs" / "attacked" / "merged_windows.parquet",
        root / "outputs" / "attacked" / "attack_labels.parquet",
    )


def read_window_sources(
    root: Path,
    clean_windows_path: Path | None = None,
    attacked_windows_path: Path | None = None,
    labels_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean_path, attacked_path, resolved_labels_path = canonical_window_source_paths(root)
    clean = read_dataframe(clean_windows_path or clean_path)
    attacked = read_dataframe(attacked_windows_path or attacked_path)
    labels = read_dataframe(labels_path or resolved_labels_path)
    return normalize_window_sources(clean, attacked, labels)


def normalize_window_sources(
    clean_windows: pd.DataFrame,
    attacked_windows: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean = clean_windows.copy()
    attacked = attacked_windows.copy()
    labels = labels_df.copy()
    for frame in [clean, attacked]:
        if "window_start_utc" in frame.columns:
            frame["window_start_utc"] = pd.to_datetime(frame["window_start_utc"], utc=True)
        if "window_end_utc" in frame.columns:
            frame["window_end_utc"] = pd.to_datetime(frame["window_end_utc"], utc=True)
    if not labels.empty:
        if "start_time_utc" in labels.columns:
            labels["start_time_utc"] = pd.to_datetime(labels["start_time_utc"], utc=True)
        if "end_time_utc" in labels.columns:
            labels["end_time_utc"] = pd.to_datetime(labels["end_time_utc"], utc=True)
    return clean, attacked, labels


def annotate_scenario_windows(frame: pd.DataFrame, labels_df: pd.DataFrame, default_scenario: str = "benign") -> pd.DataFrame:
    annotated = frame.copy()
    if "scenario_window_id" not in annotated.columns:
        annotated["scenario_window_id"] = default_scenario
    else:
        annotated["scenario_window_id"] = annotated["scenario_window_id"].fillna(default_scenario).astype(str)
    if labels_df.empty:
        return annotated
    for _, label in labels_df.iterrows():
        mask = (annotated["window_start_utc"] < label["end_time_utc"]) & (annotated["window_end_utc"] > label["start_time_utc"])
        annotated.loc[mask, "scenario_window_id"] = str(label["scenario_id"])
    return annotated


def build_aligned_residual_dataframe(
    clean_windows: pd.DataFrame,
    attacked_windows: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    clean, attacked, labels = normalize_window_sources(clean_windows, attacked_windows, labels_df)
    merged = attacked.merge(clean, on="window_start_utc", how="inner", suffixes=("_attacked", "_clean"))
    metadata = pd.DataFrame(
        {
            "window_start_utc": merged["window_start_utc"],
            "window_end_utc": _prefer_column(merged, "window_end_utc_attacked", "window_end_utc_clean"),
            "window_seconds": _prefer_column(merged, "window_seconds_attacked", "window_seconds_clean").astype(int),
            "scenario_id": _prefer_column(merged, "scenario_id_attacked", "scenario_id_clean").astype(str),
            "run_id": _prefer_column(merged, "run_id_attacked", "run_id_clean").astype(str),
            "split_id": _prefer_column(merged, "split_id_attacked", "split_id_clean").astype(str),
            "attack_present": merged["attack_present_attacked"].astype(int),
            "attack_family": merged["attack_family_attacked"].astype(str),
            "attack_severity": merged["attack_severity_attacked"].astype(str),
            "attack_affected_assets": merged["attack_affected_assets_attacked"].astype(str),
        }
    )
    residual_columns: dict[str, pd.Series] = {
        "window_start_utc": metadata["window_start_utc"],
        "window_end_utc": metadata["window_end_utc"],
        "window_seconds": metadata["window_seconds"],
        "scenario_id": metadata["scenario_id"],
        "run_id": metadata["run_id"],
        "split_id": metadata["split_id"],
        "attack_present": metadata["attack_present"],
        "attack_family": metadata["attack_family"],
        "attack_severity": metadata["attack_severity"],
        "attack_affected_assets": metadata["attack_affected_assets"],
        "scenario_window_id": pd.Series("benign", index=metadata.index, dtype="object"),
    }
    common_numeric = [
        column
        for column in attacked.columns
        if column in clean.columns
        and column not in WINDOW_META_COLUMNS
        and pd.api.types.is_numeric_dtype(attacked[column])
        and pd.api.types.is_numeric_dtype(clean[column])
    ]
    for column in common_numeric:
        residual_columns[f"delta__{column}"] = (
            merged[f"{column}_attacked"].astype(float) - merged[f"{column}_clean"].astype(float)
        ).fillna(0.0)
    residual_df = pd.DataFrame(residual_columns).sort_values("window_start_utc").reset_index(drop=True)
    return annotate_scenario_windows(residual_df, labels)


def load_persisted_residual_dataframe(path: Path) -> pd.DataFrame:
    frame = read_dataframe(path)
    if "window_start_utc" in frame.columns:
        frame["window_start_utc"] = pd.to_datetime(frame["window_start_utc"], utc=True)
    if "window_end_utc" in frame.columns:
        frame["window_end_utc"] = pd.to_datetime(frame["window_end_utc"], utc=True)
    return frame.sort_values("window_start_utc").reset_index(drop=True)


def persist_residual_dataframe(frame: pd.DataFrame, path: Path) -> Path:
    ordered = frame.sort_values("window_start_utc").reset_index(drop=True)
    return write_dataframe(ordered, path, fmt="parquet")


def residual_artifact_is_usable(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        frame = load_persisted_residual_dataframe(path)
    except Exception:
        return False
    missing = REQUIRED_RESIDUAL_COLUMNS.difference(frame.columns)
    delta_columns = [column for column in frame.columns if column.startswith("delta__")]
    return not missing and bool(delta_columns)


def residual_artifact_is_fresh(path: Path, source_paths: tuple[Path, Path, Path]) -> bool:
    if not path.exists():
        return False
    residual_mtime = path.stat().st_mtime
    return all(source.exists() and residual_mtime >= source.stat().st_mtime for source in source_paths)


def _prefer_column(frame: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
    if primary in frame.columns:
        return frame[primary]
    if fallback in frame.columns:
        return frame[fallback]
    return pd.Series(pd.NA, index=frame.index)
