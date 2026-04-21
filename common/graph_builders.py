"""Shared utility support for DERGuardian.

This module provides graph builders helpers used across the Phase 1 data
pipeline, Phase 2 scenario pipeline, and Phase 3 evaluation/reporting layers.
The functions here are infrastructure code: they prepare paths, metadata,
profiles, graphs, units, or time alignment without changing canonical detector
outputs or benchmark decisions.
"""

from __future__ import annotations

from pathlib import Path
import ast

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_validation_graphs(
    truth_df: pd.DataFrame,
    measured_df: pd.DataFrame,
    cyber_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    output_dir: str | Path,
    baseline_truth_df: pd.DataFrame | None = None,
    baseline_measured_df: pd.DataFrame | None = None,
    impairment_df: pd.DataFrame | None = None,
    reference_profiles_df: pd.DataFrame | None = None,
) -> list[str]:
    """Build validation graphs for the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = _coerce_timeframe(truth_df)
    measured = _coerce_timeframe(measured_df)
    cyber = _coerce_timeframe(cyber_df)
    labels = _coerce_labels(labels_df)
    baseline_truth = _coerce_timeframe(baseline_truth_df) if baseline_truth_df is not None else None
    baseline_measured = _coerce_timeframe(baseline_measured_df) if baseline_measured_df is not None else None
    reference_profiles = _coerce_timeframe(reference_profiles_df) if reference_profiles_df is not None else None

    graph_paths = [
        _plot_feeder_power(truth, out_dir / "feeder_power.png"),
        _plot_selected_bus_voltages(truth, out_dir / "selected_bus_voltages.png"),
        _plot_pv_and_environment(truth, out_dir / "pv_vs_environment.png"),
        _plot_pv_seasonal_profiles(reference_profiles, out_dir / "pv_seasonal_profiles.png"),
        _plot_bess_soc(truth, out_dir / "bess_soc.png"),
        _plot_tap_activity(truth, out_dir / "regulator_taps.png"),
        _plot_weekday_weekend_load(reference_profiles, out_dir / "weekday_vs_weekend_load.png"),
        _plot_seasonal_load(reference_profiles, out_dir / "seasonal_load_comparison.png"),
        _plot_missingness(measured, out_dir / "measured_missingness.png"),
        _plot_measurement_noise_statistics(impairment_df, out_dir / "measurement_noise_statistics.png"),
        _plot_voltage_violations(truth, out_dir / "voltage_violation_statistics.png"),
        _plot_correlation_heatmap(truth, out_dir / "signal_correlation_heatmap.png"),
        _plot_cyber_distribution(cyber, out_dir / "cyber_event_distribution.png"),
        _plot_attack_distribution(labels, out_dir / "attack_distribution.png"),
        _plot_scenario_coverage(labels, out_dir / "scenario_coverage.png"),
    ]
    if baseline_truth is not None and baseline_measured is not None:
        graph_paths.extend(
            [
                _plot_clean_vs_attacked_overlay(truth, baseline_truth, labels, out_dir / "clean_vs_attacked_overlay.png"),
                _plot_attack_impact(measured, baseline_measured, labels, out_dir / "attack_impact_signals.png"),
                _plot_attack_effect_timing(truth, baseline_truth, labels, out_dir / "attack_effect_timing.png"),
            ]
        )
    return [path for path in graph_paths if path]


def _coerce_timeframe(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    frame = df.copy()
    if "timestamp_utc" in frame.columns:
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame


def _coerce_labels(df: pd.DataFrame) -> pd.DataFrame:
    labels = df.copy()
    if labels.empty:
        return labels
    for column in ["start_time_utc", "end_time_utc"]:
        if column in labels.columns:
            labels[column] = pd.to_datetime(labels[column], utc=True)
    for column in ["affected_assets", "affected_signals", "causal_metadata"]:
        if column in labels.columns:
            labels[column] = labels[column].apply(_parse_object)
    return labels


def _parse_object(value: object) -> object:
    if isinstance(value, (list, dict)):
        return value
    if pd.isna(value):
        return []
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return value


def _annotated_empty_plot(path: Path, title: str, message: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _plot_feeder_power(df: pd.DataFrame, path: Path) -> str:
    if df.empty:
        return _annotated_empty_plot(path, "Feeder Active and Reactive Power", "No truth data available.")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["timestamp_utc"], df["feeder_p_kw_total"], label="P (kW)")
    ax.plot(df["timestamp_utc"], df["feeder_q_kvar_total"], label="Q (kvar)")
    ax.set_title("Feeder Active and Reactive Power")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _plot_selected_bus_voltages(df: pd.DataFrame, path: Path) -> str:
    columns = [column for column in df.columns if column.startswith(("bus_13_v_pu", "bus_60_v_pu", "bus_83_v_pu", "bus_114_v_pu"))][:8]
    if df.empty or not columns:
        return _annotated_empty_plot(path, "Selected Bus Voltage Magnitudes", "No selected bus-voltage series are available.")
    fig, ax = plt.subplots(figsize=(12, 4))
    for column in columns:
        ax.plot(df["timestamp_utc"], df[column], label=column)
    ax.set_title("Selected Bus Voltage Magnitudes")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _plot_pv_and_environment(df: pd.DataFrame, path: Path) -> str:
    pv_columns = [item for item in df.columns if item.startswith("pv_") and item.endswith("_p_kw")]
    if df.empty or not pv_columns:
        return _annotated_empty_plot(path, "PV Output and Environmental Drivers", "No PV production series are available.")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for column in pv_columns:
        axes[0].plot(df["timestamp_utc"], df[column], label=column)
    axes[0].set_title("PV Output")
    axes[0].legend(loc="upper right", fontsize=8)
    if {"env_irradiance_wm2", "env_temperature_c"}.issubset(df.columns):
        axes[1].plot(df["timestamp_utc"], df["env_irradiance_wm2"], label="Irradiance")
        axes[1].plot(df["timestamp_utc"], df["env_temperature_c"], label="Temperature")
    axes[1].set_title("Environmental Drivers")
    axes[1].legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _plot_pv_seasonal_profiles(reference_df: pd.DataFrame | None, path: Path) -> str:
    if reference_df is None or reference_df.empty or "pv_aggregate_potential_kw" not in reference_df.columns:
        return _annotated_empty_plot(path, "PV Seasonal Profiles", "Representative seasonal PV profiles are unavailable.")
    fig, ax = plt.subplots(figsize=(12, 4))
    grouped = reference_df.groupby(["season", "hour_of_day"], observed=False)["pv_aggregate_potential_kw"].mean().reset_index()
    for season in ["winter", "spring", "summer", "fall"]:
        season_frame = grouped[grouped["season"] == season]
        if not season_frame.empty:
            ax.plot(season_frame["hour_of_day"], season_frame["pv_aggregate_potential_kw"], label=season.title())
    ax.set_title("Representative PV Output Across Seasons")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("kW")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _plot_bess_soc(df: pd.DataFrame, path: Path) -> str:
    columns = [column for column in df.columns if column.startswith("bess_") and column.endswith("_soc")]
    if df.empty or not columns:
        return _annotated_empty_plot(path, "BESS SOC", "No BESS SOC traces are available.")
    fig, ax = plt.subplots(figsize=(12, 4))
    for column in columns:
        ax.plot(df["timestamp_utc"], df[column], label=column)
    ax.set_title("BESS SOC")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _plot_tap_activity(df: pd.DataFrame, path: Path) -> str:
    columns = [column for column in df.columns if column.startswith("regulator_") and column.endswith("_tap_pos")]
    if df.empty or not columns:
        return _annotated_empty_plot(path, "Regulator Tap Activity", "No regulator tap telemetry is available.")
    fig, ax = plt.subplots(figsize=(12, 4))
    for column in columns:
        ax.step(df["timestamp_utc"], df[column], where="post", label=column)
    ax.set_title("Regulator Tap Activity")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _plot_weekday_weekend_load(reference_df: pd.DataFrame | None, path: Path) -> str:
    if reference_df is None or reference_df.empty:
        return _annotated_empty_plot(path, "Weekday vs Weekend Load Comparison", "Representative load profiles are unavailable.")
    fig, ax = plt.subplots(figsize=(12, 4))
    grouped = reference_df.groupby(["day_type", "hour_of_day"], observed=False)["load_aggregate_p_kw"].mean().reset_index()
    for day_type in ["weekday", "weekend"]:
        frame = grouped[grouped["day_type"] == day_type]
        if not frame.empty:
            ax.plot(frame["hour_of_day"], frame["load_aggregate_p_kw"], label=day_type.title())
    ax.set_title("Representative Weekday vs Weekend Aggregate Load")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("kW")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _plot_seasonal_load(reference_df: pd.DataFrame | None, path: Path) -> str:
    if reference_df is None or reference_df.empty:
        return _annotated_empty_plot(path, "Seasonal Load Comparison", "Representative seasonal load profiles are unavailable.")
    fig, ax = plt.subplots(figsize=(12, 4))
    grouped = reference_df.groupby(["season", "hour_of_day"], observed=False)["load_aggregate_p_kw"].mean().reset_index()
    for season in ["winter", "spring", "summer", "fall"]:
        frame = grouped[grouped["season"] == season]
        if not frame.empty:
            ax.plot(frame["hour_of_day"], frame["load_aggregate_p_kw"], label=season.title())
    ax.set_title("Representative Seasonal Aggregate Load")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("kW")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _plot_missingness(df: pd.DataFrame, path: Path) -> str:
    if df.empty:
        return _annotated_empty_plot(path, "Measured-Layer Missingness", "No measured-layer data is available.")
    missing = df.isna().mean().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(missing.index, missing.to_numpy())
    ax.set_title("Measured-Layer Missingness")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _plot_measurement_noise_statistics(impairment_df: pd.DataFrame | None, path: Path) -> str:
    if impairment_df is None or impairment_df.empty:
        return _annotated_empty_plot(path, "Measurement Noise Statistics", "Measurement impairment summary is unavailable.")
    frame = impairment_df.sort_values("noise_std", ascending=False).head(15)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(frame["column"], frame["noise_std"], color="#35618f", label="Noise std")
    ax1.set_ylabel("Noise std")
    ax1.tick_params(axis="x", rotation=90)
    ax2 = ax1.twinx()
    ax2.plot(frame["column"], frame["missing_fraction"], color="#bf5b17", marker="o", label="Missing fraction")
    ax2.set_ylabel("Missing fraction")
    ax1.set_title("Measurement Noise and Missingness Statistics")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _plot_voltage_violations(df: pd.DataFrame, path: Path) -> str:
    if df.empty or "derived_voltage_violation_count" not in df.columns:
        return _annotated_empty_plot(path, "Voltage Violation Statistics", "Voltage-violation metrics are unavailable.")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(df["timestamp_utc"], df["derived_voltage_violation_count"])
    axes[0].set_title("Voltage-Violation Count Over Time")
    axes[1].hist(df["derived_voltage_violation_count"], bins=min(20, max(len(df) // 50, 5)))
    axes[1].set_title("Voltage-Violation Count Distribution")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _plot_correlation_heatmap(df: pd.DataFrame, path: Path) -> str:
    if df.empty:
        return _annotated_empty_plot(path, "Signal Correlation Heatmap", "No truth-layer data is available.")
    numeric = df.select_dtypes(include=[np.number])
    preferred = [
        "feeder_p_kw_total",
        "feeder_q_kvar_total",
        "load_aggregate_p_kw",
        "env_irradiance_wm2",
        "env_temperature_c",
        "derived_voltage_violation_count",
        "bess_bess48_soc",
        "bess_bess108_soc",
        "pv_pv35_p_kw",
        "pv_pv60_p_kw",
        "pv_pv83_p_kw",
    ]
    columns = [column for column in preferred if column in numeric.columns]
    if len(columns) < 4:
        columns = numeric.columns[:12].tolist()
    corr = numeric[columns].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns)
    ax.set_title("Signal Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _plot_cyber_distribution(df: pd.DataFrame, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    if df.empty or "event_type" not in df.columns:
        ax.text(0.5, 0.5, "No cyber events available", ha="center", va="center")
    else:
        counts = df["event_type"].value_counts()
        ax.bar(counts.index.astype(str), counts.to_numpy())
        ax.tick_params(axis="x", rotation=30)
    ax.set_title("Cyber Event Distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _plot_attack_distribution(df: pd.DataFrame, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    if df.empty or "attack_family" not in df.columns:
        ax.text(0.5, 0.5, "No attacks in current split", ha="center", va="center")
    else:
        counts = df["attack_family"].astype(str).value_counts()
        ax.bar(counts.index.astype(str), counts.to_numpy())
        ax.tick_params(axis="x", rotation=30)
    ax.set_title("Attack Family Distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _plot_scenario_coverage(df: pd.DataFrame, path: Path) -> str:
    if df.empty or not {"scenario_id", "start_time_utc", "end_time_utc"}.issubset(df.columns):
        return _annotated_empty_plot(path, "Scenario Coverage", "No attack scenarios are available for coverage plotting.")
    frame = df.sort_values("start_time_utc").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, max(3, 0.6 * len(frame))))
    for idx, (_, row) in enumerate(frame.iterrows()):
        start = mdates.date2num(pd.Timestamp(row["start_time_utc"]).to_pydatetime())
        end = mdates.date2num(pd.Timestamp(row["end_time_utc"]).to_pydatetime())
        ax.barh(idx, end - start, left=start, height=0.5, align="center")
    ax.set_yticks(range(len(frame)))
    ax.set_yticklabels(frame["scenario_id"].astype(str))
    ax.xaxis_date()
    ax.set_title("Scenario Coverage Timeline")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _plot_clean_vs_attacked_overlay(attacked_df: pd.DataFrame, baseline_df: pd.DataFrame, labels_df: pd.DataFrame, path: Path) -> str:
    if attacked_df.empty or baseline_df.empty:
        return _annotated_empty_plot(path, "Clean vs Attacked Overlay", "Baseline or attacked truth data is unavailable.")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    signals = ["feeder_p_kw_total", "feeder_q_kvar_total", _default_overlay_signal(attacked_df, labels_df)]
    for ax, signal in zip(axes, signals):
        if signal not in attacked_df.columns or signal not in baseline_df.columns:
            ax.text(0.5, 0.5, f"Signal unavailable: {signal}", ha="center", va="center")
            continue
        merged = attacked_df[["timestamp_utc", signal]].merge(
            baseline_df[["timestamp_utc", signal]],
            on="timestamp_utc",
            suffixes=("_attacked", "_baseline"),
        )
        ax.plot(merged["timestamp_utc"], merged[f"{signal}_baseline"], label="Clean baseline")
        ax.plot(merged["timestamp_utc"], merged[f"{signal}_attacked"], label="Attacked", alpha=0.8)
        for _, label in labels_df.iterrows():
            ax.axvspan(label["start_time_utc"], label["end_time_utc"], color="red", alpha=0.08)
        ax.set_title(signal)
        ax.legend(loc="upper right")
    fig.suptitle("Clean vs Attacked Overlay")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _plot_attack_impact(attacked_df: pd.DataFrame, baseline_df: pd.DataFrame, labels_df: pd.DataFrame, path: Path) -> str:
    if attacked_df.empty or baseline_df.empty or labels_df.empty:
        return _annotated_empty_plot(path, "Attack Impact Signals", "Attack labels or comparison data are unavailable.")
    scenarios = labels_df.head(4)
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, max(4, 3 * len(scenarios))), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, (_, label) in zip(axes, scenarios.iterrows()):
        signal = _first_label_signal(label, attacked_df, baseline_df)
        if signal is None:
            ax.text(0.5, 0.5, "No observable attacked signal available", ha="center", va="center")
            ax.set_title(str(label["scenario_id"]))
            continue
        duration = max((label["end_time_utc"] - label["start_time_utc"]).total_seconds(), 1.0)
        padding = pd.Timedelta(seconds=max(60.0, duration * 0.5))
        start = label["start_time_utc"] - padding
        end = label["end_time_utc"] + padding
        attack_slice = attacked_df[(attacked_df["timestamp_utc"] >= start) & (attacked_df["timestamp_utc"] <= end)][["timestamp_utc", signal]]
        baseline_slice = baseline_df[(baseline_df["timestamp_utc"] >= start) & (baseline_df["timestamp_utc"] <= end)][["timestamp_utc", signal]]
        merged = attack_slice.merge(baseline_slice, on="timestamp_utc", suffixes=("_attacked", "_baseline"))
        ax.plot(merged["timestamp_utc"], merged[f"{signal}_baseline"], label="Clean baseline")
        ax.plot(merged["timestamp_utc"], merged[f"{signal}_attacked"], label="Attacked", alpha=0.85)
        ax.axvspan(label["start_time_utc"], label["end_time_utc"], color="red", alpha=0.08)
        ax.set_title(f"{label['scenario_id']} :: {signal}")
        ax.legend(loc="upper right")
    fig.suptitle("Anomaly Injection Impact Plots")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _plot_attack_effect_timing(attacked_df: pd.DataFrame, baseline_df: pd.DataFrame, labels_df: pd.DataFrame, path: Path) -> str:
    if attacked_df.empty or baseline_df.empty or labels_df.empty:
        return _annotated_empty_plot(path, "Attack Onset to Physical Effect Timing", "Attack labels or comparison data are unavailable.")
    timing_rows: list[tuple[str, float]] = []
    for _, label in labels_df.iterrows():
        signal = _first_label_signal(label, attacked_df, baseline_df)
        if signal is None:
            continue
        start = label["start_time_utc"]
        end = label["end_time_utc"]
        attacked_slice = attacked_df[(attacked_df["timestamp_utc"] >= start) & (attacked_df["timestamp_utc"] <= end)]
        baseline_slice = baseline_df[(baseline_df["timestamp_utc"] >= start) & (baseline_df["timestamp_utc"] <= end)]
        delay = _effect_delay_seconds(attacked_slice, baseline_slice, signal)
        if delay is not None:
            timing_rows.append((str(label["scenario_id"]), delay))
    if not timing_rows:
        return _annotated_empty_plot(path, "Attack Onset to Physical Effect Timing", "No observable effect delays were detected.")
    fig, ax = plt.subplots(figsize=(10, 4))
    ids = [item[0] for item in timing_rows]
    delays = [item[1] for item in timing_rows]
    ax.bar(ids, delays)
    ax.set_title("Attack Onset to Physical Effect Timing")
    ax.set_ylabel("Seconds")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _default_overlay_signal(attacked_df: pd.DataFrame, labels_df: pd.DataFrame) -> str:
    for _, label in labels_df.iterrows():
        signal = _first_label_signal(label, attacked_df, attacked_df)
        if signal is not None:
            return signal
    for fallback in ["bess_bess48_soc", "pv_pv83_p_kw", "bus_83_v_pu_phase_a"]:
        if fallback in attacked_df.columns:
            return fallback
    return attacked_df.select_dtypes(include=[np.number]).columns[0]


def _first_label_signal(label: pd.Series, attacked_df: pd.DataFrame, baseline_df: pd.DataFrame) -> str | None:
    signals = []
    metadata = label.get("causal_metadata", {})
    if isinstance(metadata, dict):
        observable = metadata.get("observable_signals", [])
        if isinstance(observable, list):
            signals.extend(str(item) for item in observable)
    affected = label.get("affected_signals", [])
    if isinstance(affected, list):
        signals.extend(str(item) for item in affected)
    for signal in signals:
        if signal in attacked_df.columns and signal in baseline_df.columns and pd.api.types.is_numeric_dtype(attacked_df[signal]):
            return signal
    return None


def _effect_delay_seconds(attacked_df: pd.DataFrame, baseline_df: pd.DataFrame, signal: str) -> float | None:
    if signal not in attacked_df.columns or signal not in baseline_df.columns:
        return None
    merged = attacked_df[["timestamp_utc", signal]].merge(
        baseline_df[["timestamp_utc", signal]],
        on="timestamp_utc",
        suffixes=("_attacked", "_baseline"),
    )
    if merged.empty:
        return None
    delta = (merged[f"{signal}_attacked"] - merged[f"{signal}_baseline"]).abs()
    threshold = max(float(delta.std(ddof=0)) * 2.0, 0.01)
    crossed = merged.loc[delta > threshold, "timestamp_utc"]
    if crossed.empty:
        return None
    return float((crossed.iloc[0] - merged["timestamp_utc"].iloc[0]).total_seconds())
