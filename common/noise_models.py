"""Shared utility support for DERGuardian.

This module provides noise models helpers used across the Phase 1 data
pipeline, Phase 2 scenario pipeline, and Phase 3 evaluation/reporting layers.
The functions here are infrastructure code: they prepare paths, metadata,
profiles, graphs, units, or time alignment without changing canonical detector
outputs or benchmark decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from common.config import PipelineConfig
from common.time_alignment import apply_clock_offset, apply_latency, apply_missing_bursts


@dataclass(slots=True)
class ImpairmentSummary:
    """Structured object used by the shared DERGuardian utility workflow."""

    column: str
    noise_std: float
    missing_fraction: float
    latency_seconds: float
    observed: bool


def build_measured_layer(
    truth_df: pd.DataFrame,
    config: PipelineConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[ImpairmentSummary]]:
    """Build measured layer for the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    base_columns = [
        "timestamp_utc",
        "simulation_index",
        "scenario_id",
        "run_id",
        "split_id",
        "sample_rate_seconds",
    ]
    measured_columns = _select_measured_columns(truth_df.columns.tolist(), config)
    measured_df = truth_df[base_columns + measured_columns].copy()
    measured_df["source_layer"] = "measured"

    measured_index = pd.to_datetime(measured_df["timestamp_utc"], utc=True)
    observed_clock = apply_clock_offset(
        measured_index,
        offset_seconds=float(rng.normal(0.0, 0.25)),
        drift_ppm=config.measurement.default_clock_drift_ppm,
    )
    measured_df["timestamp_utc"] = observed_clock.astype("datetime64[ns, UTC]")

    summaries: list[ImpairmentSummary] = []
    for column in measured_columns:
        series = measured_df[column]
        if pd.api.types.is_numeric_dtype(series):
            noisy, summary = _impair_numeric_series(series, column, config, rng)
            measured_df[column] = noisy
            summaries.append(summary)
        else:
            impaired = series.copy()
            mask = rng.random(len(series)) < min(config.measurement.default_missing_probability * 0.25, 0.01)
            impaired.loc[mask] = None
            measured_df[column] = impaired
            summaries.append(
                ImpairmentSummary(
                    column=column,
                    noise_std=0.0,
                    missing_fraction=float(mask.mean()),
                    latency_seconds=float(config.measurement.default_latency_seconds),
                    observed=True,
                )
            )
    return measured_df, summaries


def impairment_summaries_to_frame(summaries: list[ImpairmentSummary]) -> pd.DataFrame:
    """Handle impairment summaries to frame within the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return pd.DataFrame(
        [
            {
                "column": item.column,
                "noise_std": item.noise_std,
                "missing_fraction": item.missing_fraction,
                "latency_seconds": item.latency_seconds,
                "observed": item.observed,
            }
            for item in summaries
        ]
    )


def _select_measured_columns(columns: list[str], config: PipelineConfig) -> list[str]:
    keep = []
    observed_buses = set(config.measured_bus_observability)
    for column in columns:
        if column in {"source_layer"}:
            continue
        if column.startswith(("env_", "feeder_", "pv_", "bess_", "regulator_", "capacitor_", "switch_", "breaker_", "relay_", "derived_", "load_aggregate")):
            keep.append(column)
            continue
        if column.startswith("line_"):
            keep.append(column)
            continue
        if column.startswith("bus_"):
            bus_name = column.split("_", 2)[1]
            if bus_name in observed_buses:
                keep.append(column)
    return keep


def _impair_numeric_series(
    series: pd.Series,
    column: str,
    config: PipelineConfig,
    rng: np.random.Generator,
) -> tuple[pd.Series, ImpairmentSummary]:
    profile = _column_profile(column, config)
    values = series.astype(float).copy()
    scale = np.maximum(np.abs(values.to_numpy()), 1.0)
    additive_noise = rng.normal(0.0, profile["additive_std"], size=len(values))
    multiplicative_noise = rng.normal(0.0, profile["relative_std"], size=len(values))
    noisy = values + additive_noise + multiplicative_noise * scale
    noisy = pd.Series(noisy, index=values.index, name=values.name)
    if profile["latency_steps"] > 0 or profile["latency_jitter_steps"] > 0:
        noisy = apply_latency(noisy, profile["latency_steps"], profile["latency_jitter_steps"], rng)
    if profile["stale_probability"] > 0.0:
        noisy = _apply_stale_values(noisy, profile["stale_probability"], rng)
    if profile["missing_probability"] > 0.0:
        noisy = apply_missing_bursts(noisy, profile["missing_probability"], (1, 6), rng)
    if column.endswith("_soc"):
        noisy = noisy.clip(0.0, 1.0)
    if column.endswith("_state"):
        noisy = noisy.round()
    summary = ImpairmentSummary(
        column=column,
        noise_std=float(np.nanstd((noisy - values).to_numpy())),
        missing_fraction=float(noisy.isna().mean()),
        latency_seconds=float(profile["latency_steps"] * config.simulation_resolution_seconds),
        observed=True,
    )
    return noisy, summary


def _apply_stale_values(series: pd.Series, stale_probability: float, rng: np.random.Generator) -> pd.Series:
    values = series.copy()
    for idx in range(1, len(values)):
        if rng.random() < stale_probability:
            values.iloc[idx] = values.iloc[idx - 1]
    return values


def _column_profile(column: str, config: PipelineConfig) -> dict[str, float | int]:
    measurement = config.measurement
    additive = measurement.default_numeric_noise_std
    relative = measurement.default_relative_noise_std
    missing = measurement.default_missing_probability
    stale = measurement.default_stale_probability
    latency = measurement.default_latency_seconds
    jitter = measurement.default_latency_jitter_seconds

    if column.startswith("feeder_") or column.startswith("line_"):
        additive *= 2.5
        relative *= 1.5
        latency = max(latency, 3)
    elif column.startswith("bus_"):
        additive *= 0.8
        relative *= 0.8
    elif column.startswith(("pv_", "bess_")):
        additive *= 1.2
        relative *= 1.0
    elif column.startswith("env_"):
        additive *= 0.3
        relative *= 0.4
        missing *= 0.3
        stale *= 0.2
        latency = 0
        jitter = 0
    elif column.startswith(("regulator_", "capacitor_", "switch_")):
        additive = 0.0
        relative = 0.0
        missing *= 0.5
        stale *= 0.5
        latency = 1
        jitter = 0

    return {
        "additive_std": additive,
        "relative_std": relative,
        "missing_probability": missing,
        "stale_probability": stale,
        "latency_steps": max(int(math.ceil(latency / config.simulation_resolution_seconds)), 0),
        "latency_jitter_steps": max(int(math.ceil(jitter / config.simulation_resolution_seconds)), 0),
    }
