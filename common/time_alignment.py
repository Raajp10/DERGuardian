from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def apply_latency(series: pd.Series, base_latency_steps: int, jitter_steps: int, rng: np.random.Generator) -> pd.Series:
    if base_latency_steps <= 0 and jitter_steps <= 0:
        return series.copy()
    jitters = rng.integers(-jitter_steps, jitter_steps + 1, size=len(series))
    delays = np.maximum(base_latency_steps + jitters, 0)
    values = series.to_numpy(copy=True)
    delayed = []
    for idx, shift in enumerate(delays):
        src = max(idx - int(shift), 0)
        delayed.append(values[src])
    return pd.Series(delayed, index=series.index, name=series.name)


def apply_clock_offset(index: pd.DatetimeIndex, offset_seconds: float = 0.0, drift_ppm: float = 0.0) -> pd.DatetimeIndex:
    seconds = np.arange(len(index), dtype=float)
    drift_seconds = seconds * drift_ppm * 1e-6
    offset = pd.to_timedelta(offset_seconds + drift_seconds, unit="s")
    return index + offset


def apply_missing_bursts(series: pd.Series, burst_probability: float, burst_length_range: tuple[int, int], rng: np.random.Generator) -> pd.Series:
    values = series.copy()
    length = len(values)
    idx = 0
    while idx < length:
        if rng.random() < burst_probability:
            burst = int(rng.integers(burst_length_range[0], burst_length_range[1] + 1))
            values.iloc[idx : idx + burst] = np.nan
            idx += burst
        idx += 1
    return values


def align_events_to_index(
    events: pd.DataFrame,
    index: pd.DatetimeIndex,
    timestamp_column: str = "timestamp_utc",
    tolerance: str = "2s",
) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    frame = events.copy()
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True)
    target = pd.DataFrame({"timestamp_utc": index})
    aligned = pd.merge_asof(
        frame.sort_values(timestamp_column),
        target.sort_values("timestamp_utc"),
        left_on=timestamp_column,
        right_on="timestamp_utc",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )
    return aligned


def infer_sample_rate_seconds(index: Iterable[pd.Timestamp]) -> int:
    series = pd.Index(index)
    if len(series) < 2:
        return 0
    delta = (series[1] - series[0]).total_seconds()
    return int(delta)


def reconstruct_nominal_timestamps(
    frame: pd.DataFrame,
    timestamp_column: str = "timestamp_utc",
    simulation_index_column: str = "simulation_index",
    sample_rate_column: str = "sample_rate_seconds",
) -> pd.Series:
    if timestamp_column not in frame.columns:
        raise KeyError(f"Missing timestamp column `{timestamp_column}`.")
    timestamps = pd.to_datetime(frame[timestamp_column], utc=True)
    if simulation_index_column not in frame.columns:
        return pd.Series(timestamps, index=frame.index, name="analysis_timestamp_utc")

    simulation_index = frame[simulation_index_column].astype(int)
    if sample_rate_column in frame.columns:
        nominal_sample_seconds = float(pd.to_numeric(frame[sample_rate_column], errors="coerce").dropna().median())
        if not np.isfinite(nominal_sample_seconds) or nominal_sample_seconds <= 0.0:
            nominal_sample_seconds = 1.0
    else:
        nominal_sample_seconds = 1.0

    anchors = timestamps - pd.to_timedelta(simulation_index * nominal_sample_seconds, unit="s")
    anchor_ns = pd.Series(anchors).astype("int64").median()
    base_timestamp = pd.to_datetime(int(round(anchor_ns)), utc=True).round("s")
    nominal = base_timestamp + pd.to_timedelta(simulation_index * nominal_sample_seconds, unit="s")
    return pd.Series(nominal, index=frame.index, name="analysis_timestamp_utc")
