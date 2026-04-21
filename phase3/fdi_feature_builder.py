"""Phase 3 evaluation and analysis support for DERGuardian.

This module implements fdi feature builder logic for detector evaluation, ablations,
zero-day-like heldout synthetic analysis, latency sweeps, or final reporting.
It keeps benchmark, replay, heldout synthetic, and extension results separated.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def augment_fdi_features(
    residual_df: pd.DataFrame,
    attacked_windows: pd.DataFrame,
    clean_windows: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Handle augment fdi features within the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    augmented = residual_df.copy()
    attacked = attacked_windows.copy().sort_values("window_start_utc").reset_index(drop=True)
    clean = clean_windows.copy().sort_values("window_start_utc").reset_index(drop=True)
    attacked["window_start_utc"] = pd.to_datetime(attacked["window_start_utc"], utc=True)
    clean["window_start_utc"] = pd.to_datetime(clean["window_start_utc"], utc=True)
    merged = attacked.merge(clean, on="window_start_utc", how="inner", suffixes=("_attacked", "_clean"))
    aligned = augmented.merge(merged, on="window_start_utc", how="inner", suffixes=("", "__window"))

    new_features: dict[str, np.ndarray] = {}

    def residual_gap(feature_a: str, feature_b: str) -> np.ndarray:
        attacked_gap = _column(aligned, f"{feature_a}_attacked") - _column(aligned, f"{feature_b}_attacked")
        clean_gap = _column(aligned, f"{feature_a}_clean") - _column(aligned, f"{feature_b}_clean")
        return attacked_gap - clean_gap

    new_features["fdi__cyber_command_pressure"] = (
        np.abs(_residual_column(aligned, "delta__cyber_auth_count"))
        + np.abs(_residual_column(aligned, "delta__cyber_attack_count"))
        + np.abs(_residual_column(aligned, "delta__cyber_command_count"))
    )
    new_features["fdi__cyber_telemetry_pressure"] = (
        np.abs(_residual_column(aligned, "delta__cyber_telemetry_count"))
        + np.abs(_residual_column(aligned, "delta__cyber_config_count"))
        + np.abs(_residual_column(aligned, "delta__cyber_protocol_modbus_count"))
    )
    new_features["fdi__pv60_dispatch_gap_mean"] = _residual_column(aligned, "delta__derived_pv_pv60_availability_residual_kw__mean")
    new_features["fdi__pv60_dispatch_gap_last"] = _residual_column(aligned, "delta__derived_pv_pv60_availability_residual_kw__last")
    new_features["fdi__pv60_curtailment_delta_mean"] = _residual_column(aligned, "delta__pv_pv60_curtailment_frac__mean")
    new_features["fdi__pv60_curtailment_delta_last"] = _residual_column(aligned, "delta__pv_pv60_curtailment_frac__last")
    new_features["fdi__pv60_ramp_bias_mean"] = _residual_column(aligned, "delta__derived_pv_pv60_ramp_kw_per_s__mean")
    new_features["fdi__pv60_ramp_bias_std"] = _residual_column(aligned, "delta__derived_pv_pv60_ramp_kw_per_s__std")
    new_features["fdi__pv60_voltage_bus60_gap_mean"] = residual_gap("pv_pv60_terminal_v_pu__mean", "bus_60_v_pu_phase_a__mean")
    new_features["fdi__pv60_voltage_bus60_gap_last"] = residual_gap("pv_pv60_terminal_v_pu__last", "bus_60_v_pu_phase_a__last")
    new_features["fdi__pv60_voltage_feeder_gap_mean"] = residual_gap("pv_pv60_terminal_v_pu__mean", "feeder_v_pu_phase_a__mean")
    new_features["fdi__pv60_voltage_feeder_gap_last"] = residual_gap("pv_pv60_terminal_v_pu__last", "feeder_v_pu_phase_a__last")
    new_features["fdi__pv60_bus_feeder_gap_mean"] = residual_gap("bus_60_v_pu_phase_a__mean", "feeder_v_pu_phase_a__mean")
    new_features["fdi__pv60_voltage_drift_proxy"] = (
        (_column(aligned, "pv_pv60_terminal_v_pu__last_attacked") - _column(aligned, "pv_pv60_terminal_v_pu__mean_attacked"))
        - (_column(aligned, "pv_pv60_terminal_v_pu__last_clean") - _column(aligned, "pv_pv60_terminal_v_pu__mean_clean"))
    )
    new_features["fdi__pv60_voltage_range_delta"] = (
        (_column(aligned, "pv_pv60_terminal_v_pu__max_attacked") - _column(aligned, "pv_pv60_terminal_v_pu__min_attacked"))
        - (_column(aligned, "pv_pv60_terminal_v_pu__max_clean") - _column(aligned, "pv_pv60_terminal_v_pu__min_clean"))
    )
    attacked_dispatch_ratio = _safe_divide(
        _column(aligned, "pv_pv60_p_kw__mean_attacked"),
        np.maximum(_column(aligned, "pv_pv60_available_kw__mean_attacked"), 1e-6),
    )
    clean_dispatch_ratio = _safe_divide(
        _column(aligned, "pv_pv60_p_kw__mean_clean"),
        np.maximum(_column(aligned, "pv_pv60_available_kw__mean_clean"), 1e-6),
    )
    new_features["fdi__pv60_dispatch_ratio_delta"] = attacked_dispatch_ratio - clean_dispatch_ratio
    new_features["fdi__pv60_voltage_power_decoupling"] = (
        np.abs(new_features["fdi__pv60_voltage_bus60_gap_mean"]) + np.abs(new_features["fdi__pv60_dispatch_ratio_delta"])
    )
    new_features["fdi__pv60_command_telemetry_mismatch"] = (
        np.abs(new_features["fdi__cyber_command_pressure"]) * np.abs(new_features["fdi__pv60_voltage_bus60_gap_mean"])
    )
    new_features["fdi__pv60_stealth_bias_index"] = (
        np.abs(new_features["fdi__pv60_voltage_bus60_gap_mean"])
        + np.abs(new_features["fdi__pv60_voltage_feeder_gap_mean"])
        + np.abs(new_features["fdi__pv60_dispatch_gap_mean"]) / 100.0
    )

    for name, values in new_features.items():
        augmented[name] = np.asarray(values, dtype=float)
    return augmented, sorted(new_features.keys())


def fdi_feature_candidates(frame: pd.DataFrame) -> list[str]:
    """Handle fdi feature candidates within the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    priority_columns = [
        column
        for column in frame.columns
        if column.startswith("fdi__")
        or column in {
            "delta__cyber_auth_count",
            "delta__cyber_attack_count",
            "delta__cyber_command_count",
            "delta__cyber_telemetry_count",
            "delta__cyber_auth_failures",
            "delta__pv_pv60_terminal_v_pu__mean",
            "delta__pv_pv60_terminal_v_pu__last",
            "delta__derived_pv_pv60_availability_residual_kw__mean",
            "delta__derived_pv_pv60_availability_residual_kw__last",
            "delta__bus_60_v_pu_phase_a__mean",
            "delta__feeder_v_pu_phase_a__mean",
        }
    ]
    return sorted(set(priority_columns))


def _column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        return np.zeros(len(frame), dtype=float)
    return frame[name].astype(float).fillna(0.0).to_numpy()


def _residual_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        return np.zeros(len(frame), dtype=float)
    return frame[name].astype(float).fillna(0.0).to_numpy()


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    denominator = np.where(np.abs(denominator) < 1e-6, 1.0, denominator)
    return numerator / denominator

