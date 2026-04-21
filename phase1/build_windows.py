from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from common.config import WindowConfig
from common.io_utils import read_dataframe, write_dataframe
from common.time_alignment import reconstruct_nominal_timestamps


def build_merged_windows(
    measured_df: pd.DataFrame,
    cyber_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    windows: WindowConfig,
) -> pd.DataFrame:
    measured = measured_df.copy()
    measured["timestamp_utc"] = pd.to_datetime(measured["timestamp_utc"], utc=True)
    measured["analysis_timestamp_utc"] = reconstruct_nominal_timestamps(measured)
    cyber = cyber_df.copy()
    if not cyber.empty:
        cyber["timestamp_utc"] = pd.to_datetime(cyber["timestamp_utc"], utc=True)
    labels = labels_df.copy()
    if not labels.empty:
        labels["start_time_utc"] = pd.to_datetime(labels["start_time_utc"], utc=True)
        labels["end_time_utc"] = pd.to_datetime(labels["end_time_utc"], utc=True)

    start = measured["analysis_timestamp_utc"].min()
    end = measured["analysis_timestamp_utc"].max()
    step = pd.Timedelta(seconds=windows.step_seconds)
    width = pd.Timedelta(seconds=windows.window_seconds)
    numeric_columns = [
        column
        for column in measured.columns
        if column not in {"timestamp_utc", "scenario_id", "run_id", "split_id", "source_layer"}
        and pd.api.types.is_numeric_dtype(measured[column])
    ]
    records: list[dict[str, object]] = []
    cursor = start
    while cursor + width <= end:
        window_end = cursor + width
        frame = measured[(measured["analysis_timestamp_utc"] >= cursor) & (measured["analysis_timestamp_utc"] < window_end)]
        if frame.empty:
            cursor += step
            continue
        record: dict[str, object] = {
            "window_start_utc": cursor,
            "window_end_utc": window_end,
            "window_seconds": windows.window_seconds,
            "scenario_id": frame["scenario_id"].iloc[0],
            "run_id": frame["run_id"].iloc[0],
            "split_id": frame["split_id"].iloc[0],
        }
        for column in numeric_columns:
            mean_value, std_value, min_value, max_value, last_value = _robust_numeric_window_stats(frame[column])
            record[f"{column}__mean"] = mean_value
            record[f"{column}__std"] = std_value
            record[f"{column}__min"] = min_value
            record[f"{column}__max"] = max_value
            record[f"{column}__last"] = last_value

        if not cyber.empty:
            cyber_slice = cyber[(cyber["timestamp_utc"] >= cursor) & (cyber["timestamp_utc"] < window_end)]
            record["cyber_event_count_total"] = int(len(cyber_slice))
            record["cyber_auth_count"] = int((cyber_slice["event_type"] == "authentication").sum()) if "event_type" in cyber_slice.columns else 0
            record["cyber_command_count"] = int((cyber_slice["event_type"] == "command").sum()) if "event_type" in cyber_slice.columns else 0
            record["cyber_config_count"] = int((cyber_slice["event_type"] == "configuration").sum()) if "event_type" in cyber_slice.columns else 0
            record["cyber_telemetry_count"] = int((cyber_slice["event_type"] == "telemetry").sum()) if "event_type" in cyber_slice.columns else 0
            record["cyber_attack_count"] = int((cyber_slice["event_type"] == "attack").sum()) if "event_type" in cyber_slice.columns else 0
            record["cyber_auth_failures"] = int((cyber_slice.get("auth_result", pd.Series(dtype=str)) == "fail").sum())
            protocols = cyber_slice.get("protocol", pd.Series(dtype=str)).fillna("unknown")
            for protocol in ["modbus", "dnp3", "mqtt"]:
                record[f"cyber_protocol_{protocol}_count"] = int((protocols.str.lower() == protocol).sum())
        else:
            record["cyber_event_count_total"] = 0
            record["cyber_attack_count"] = 0

        if not labels.empty:
            overlapping = labels[(labels["start_time_utc"] < window_end) & (labels["end_time_utc"] > cursor)].copy()
            qualifying = _qualifying_labels(overlapping, cursor, window_end, windows.min_attack_overlap_fraction)
            record["attack_present"] = int(not qualifying.empty)
            record["attack_family"] = "|".join(sorted(set(qualifying.get("attack_family", pd.Series(dtype=str)).astype(str)))) if not qualifying.empty else "benign"
            record["attack_severity"] = "|".join(sorted(set(qualifying.get("severity", pd.Series(dtype=str)).astype(str)))) if not qualifying.empty else "none"
            record["attack_affected_assets"] = "|".join(sorted(set(_flatten(qualifying.get("affected_assets", pd.Series(dtype=object))))))
        else:
            record["attack_present"] = 0
            record["attack_family"] = "benign"
            record["attack_severity"] = "none"
            record["attack_affected_assets"] = ""
        records.append(record)
        cursor += step
    return pd.DataFrame(records)


def _robust_numeric_window_stats(series: pd.Series) -> tuple[float, float, float, float, float]:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = numeric.dropna()
    if finite.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    mean_value = float(finite.mean())
    std_value = float(finite.std(ddof=0))
    min_value = float(finite.min())
    max_value = float(finite.max())
    last_value = float(finite.iloc[-1])
    return mean_value, std_value, min_value, max_value, last_value


def _flatten(series: pd.Series) -> list[str]:
    values: list[str] = []
    for item in series.tolist():
        if isinstance(item, (list, tuple, np.ndarray)):
            values.extend(str(value) for value in item)
        elif isinstance(item, str):
            values.append(item)
        elif pd.notna(item):
            values.append(str(item))
    return values


def _qualifying_labels(
    labels: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    min_overlap_fraction: float,
) -> pd.DataFrame:
    if labels.empty:
        return labels
    qualifying_mask = labels.apply(
        lambda row: _overlap_fraction(
            pd.Timestamp(row["start_time_utc"]),
            pd.Timestamp(row["end_time_utc"]),
            window_start,
            window_end,
        )
        >= float(min_overlap_fraction),
        axis=1,
    )
    return labels.loc[qualifying_mask].copy()


def _overlap_fraction(
    attack_start: pd.Timestamp,
    attack_end: pd.Timestamp,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> float:
    overlap_start = max(attack_start, window_start)
    overlap_end = min(attack_end, window_end)
    overlap_seconds = max((overlap_end - overlap_start).total_seconds(), 0.0)
    window_seconds = max((window_end - window_start).total_seconds(), 1.0)
    return float(overlap_seconds / window_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model-ready merged windows from measured physical and cyber data.")
    parser.add_argument("--measured", required=True)
    parser.add_argument("--cyber", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-seconds", type=int, default=300)
    parser.add_argument("--step-seconds", type=int, default=60)
    parser.add_argument("--min-attack-overlap-fraction", type=float, default=0.2)
    args = parser.parse_args()

    windows = WindowConfig(
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        min_attack_overlap_fraction=args.min_attack_overlap_fraction,
    )
    merged = build_merged_windows(
        measured_df=read_dataframe(args.measured),
        cyber_df=read_dataframe(args.cyber),
        labels_df=read_dataframe(args.labels),
        windows=windows,
    )
    suffix = Path(args.output).suffix.lower().lstrip(".") or "parquet"
    write_dataframe(merged, args.output, fmt=suffix)


if __name__ == "__main__":
    main()
