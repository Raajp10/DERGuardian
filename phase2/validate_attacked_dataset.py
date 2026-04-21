"""Validate Phase 2 attacked data against labels and the clean baseline.

The checks confirm that scenario labels, cyber events, observable physical
effects, measured-layer effects, and merged windows are internally consistent.
Validation results support scientific auditability; they do not relabel heldout
synthetic evidence as canonical benchmark evidence.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import ast
from dataclasses import asdict
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from common.metadata_schema import ValidationCheck
from common.io_utils import read_dataframe


def validate_attacked_layers(
    attacked_truth_df: pd.DataFrame,
    attacked_measured_df: pd.DataFrame,
    cyber_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    baseline_truth_df: pd.DataFrame,
    baseline_measured_df: pd.DataFrame,
    windows_df: pd.DataFrame | None = None,
) -> list[ValidationCheck]:
    """Return validation checks for attacked truth, measured, cyber, and labels."""

    checks: list[ValidationCheck] = []
    labels = _coerce_label_frame(labels_df)
    attacked_truth = attacked_truth_df.copy()
    attacked_measured = attacked_measured_df.copy()
    baseline_truth = baseline_truth_df.copy()
    baseline_measured = baseline_measured_df.copy()
    for frame in [attacked_truth, attacked_measured, baseline_truth, baseline_measured]:
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    cyber = cyber_df.copy()
    if not cyber.empty:
        cyber["timestamp_utc"] = pd.to_datetime(cyber["timestamp_utc"], utc=True)

    checks.append(ValidationCheck("attack_label_count", "pass" if len(labels) > 0 else "fail", "Attack labels were emitted for attacked generation.", float(len(labels))))
    required_label_fields = [
        "scenario_id",
        "scenario_name",
        "attack_family",
        "severity",
        "start_time_utc",
        "end_time_utc",
        "affected_assets",
        "affected_signals",
        "target_component",
        "causal_metadata",
    ]
    completeness = float(np.mean([labels[column].notna().mean() for column in required_label_fields if column in labels.columns])) if not labels.empty else 0.0
    checks.append(
        ValidationCheck(
            "label_metadata_completeness",
            "pass" if completeness >= 0.99 else "warn",
            "Average non-null completeness across required scenario-label metadata fields.",
            completeness,
            0.99,
        )
    )
    scenarios = set(labels["scenario_id"].astype(str)) if not labels.empty else set()
    cyber_scenarios = {note.split("campaign-")[-1] for note in cyber.get("campaign_id", pd.Series(dtype=str)).dropna().astype(str)}
    coverage = len(scenarios & cyber_scenarios) / max(len(scenarios), 1)
    checks.append(ValidationCheck("cyber_attack_event_coverage", "pass" if coverage >= 1.0 else "warn", "Fraction of labeled scenarios with explicit cyber attack events.", float(coverage), 1.0))

    effect_delays: list[float] = []
    physical_effect_scores: list[float] = []
    measured_effect_scores: list[float] = []
    for _, label in labels.iterrows():
        start = pd.Timestamp(label["start_time_utc"]).tz_convert("UTC")
        end = pd.Timestamp(label["end_time_utc"]).tz_convert("UTC")
        attack_cyber = cyber[(cyber["timestamp_utc"] >= start) & (cyber["timestamp_utc"] <= end) & (cyber.get("attack_flag", 0) == 1)]
        if attack_cyber.empty:
            continue
        observable_signals = _extract_observable_signals(label)
        if not observable_signals:
            observable_signals = [signal for signal in label["affected_signals"] if signal in attacked_truth.columns or signal in attacked_measured.columns]
        # Compare attacked data against the clean baseline over the same time
        # span so plausibility is based on observed deltas, not labels alone.
        window_truth = attacked_truth[(attacked_truth["timestamp_utc"] >= start) & (attacked_truth["timestamp_utc"] < end)]
        base_truth_window = baseline_truth[(baseline_truth["timestamp_utc"] >= start) & (baseline_truth["timestamp_utc"] < end)]
        window_measured = attacked_measured[(attacked_measured["timestamp_utc"] >= start) & (attacked_measured["timestamp_utc"] < end)]
        base_measured_window = baseline_measured[(baseline_measured["timestamp_utc"] >= start) & (baseline_measured["timestamp_utc"] < end)]
        score = _window_delta_score(window_truth, base_truth_window, observable_signals)
        measured_score = _window_delta_score(window_measured, base_measured_window, observable_signals)
        if score > 0:
            physical_effect_scores.append(score)
        if measured_score > 0:
            measured_effect_scores.append(measured_score)
        delay = _effect_delay_seconds(window_truth, base_truth_window, observable_signals)
        if delay is not None:
            effect_delays.append(delay)

    checks.append(
        ValidationCheck(
            "physical_effect_presence",
            "pass" if physical_effect_scores and float(np.mean(physical_effect_scores)) > 0.01 else "warn",
            "Mean attacked-vs-clean truth delta across labeled windows.",
            float(np.mean(physical_effect_scores)) if physical_effect_scores else 0.0,
            0.01,
        )
    )
    checks.append(
        ValidationCheck(
            "measured_effect_presence",
            "pass" if measured_effect_scores and float(np.mean(measured_effect_scores)) > 0.01 else "warn",
            "Mean attacked-vs-clean measured delta across labeled windows.",
            float(np.mean(measured_effect_scores)) if measured_effect_scores else 0.0,
            0.01,
        )
    )
    checks.append(
        ValidationCheck(
            "attack_onset_delay_seconds",
            "pass" if effect_delays else "warn",
            "Average delay from attack start to first observable physical effect.",
            float(np.mean(effect_delays)) if effect_delays else None,
        )
    )

    protocols = set(cyber.get("protocol", pd.Series(dtype=str)).dropna().astype(str).str.lower())
    checks.append(ValidationCheck("protocol_diversity", "pass" if {"modbus", "dnp3", "mqtt"}.issubset(protocols) else "warn", f"Protocols observed: {sorted(protocols)}"))
    family_diversity = int(labels["attack_family"].nunique()) if "attack_family" in labels.columns and not labels.empty else 0
    checks.append(
        ValidationCheck(
            "scenario_family_diversity",
            "pass" if family_diversity >= 2 else "warn",
            "Count of distinct scenario families represented in the attacked label set.",
            float(family_diversity),
            2.0,
        )
    )
    timing_complete = (
        cyber.get("latency_seconds", pd.Series(dtype=float)).notna().mean() > 0.99
        and cyber.get("jitter_seconds", pd.Series(dtype=float)).notna().mean() > 0.99
        and cyber.get("clock_offset_seconds", pd.Series(dtype=float)).notna().mean() > 0.99
    )
    checks.append(ValidationCheck("cyber_timing_fields", "pass" if timing_complete else "warn", "Latency and jitter fields are populated for nearly all cyber events."))

    if windows_df is not None and not windows_df.empty and "attack_present" in windows_df.columns:
        attack_window_fraction = float(windows_df["attack_present"].mean())
        checks.append(ValidationCheck("attacked_window_fraction", "pass" if attack_window_fraction > 0 else "warn", "Fraction of merged windows labeled as attacked.", attack_window_fraction))
    return checks


def _coerce_label_frame(labels_df: pd.DataFrame) -> pd.DataFrame:
    labels = labels_df.copy()
    if labels.empty:
        return labels
    for column in ["affected_assets", "affected_signals", "causal_metadata"]:
        if column in labels.columns:
            labels[column] = labels[column].apply(_parse_object)
    return labels


def _parse_object(value: object) -> object:
    if isinstance(value, (list, dict)):
        return value
    if pd.isna(value):
        return [] if value != value else value
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return value


def _extract_observable_signals(label: pd.Series) -> list[str]:
    metadata = label.get("causal_metadata", {})
    if isinstance(metadata, dict):
        signals = metadata.get("observable_signals", [])
        if isinstance(signals, list):
            return [str(item) for item in signals]
    return []


def _window_delta_score(attacked: pd.DataFrame, baseline: pd.DataFrame, signals: list[str]) -> float:
    """Compute the average absolute attacked-vs-baseline signal delta."""

    deltas: list[float] = []
    for signal in signals:
        if signal in attacked.columns and signal in baseline.columns:
            merged = _align_signal_frames(attacked, baseline, signal)
            if merged.empty:
                continue
            attacked_values = pd.to_numeric(merged[f"{signal}_attacked"], errors="coerce")
            baseline_values = pd.to_numeric(merged[f"{signal}_baseline"], errors="coerce")
            valid = attacked_values.notna() & baseline_values.notna()
            if not valid.any():
                continue
            diff = (attacked_values.loc[valid] - baseline_values.loc[valid]).abs().mean()
            if pd.notna(diff):
                deltas.append(float(diff))
    return float(np.mean(deltas)) if deltas else 0.0


def _effect_delay_seconds(attacked: pd.DataFrame, baseline: pd.DataFrame, signals: list[str]) -> float | None:
    for signal in signals:
        if signal in attacked.columns and signal in baseline.columns:
            merged = _align_signal_frames(attacked, baseline, signal)
            if merged.empty:
                continue
            attacked_values = pd.to_numeric(merged[f"{signal}_attacked"], errors="coerce")
            baseline_values = pd.to_numeric(merged[f"{signal}_baseline"], errors="coerce")
            valid = attacked_values.notna() & baseline_values.notna()
            if not valid.any():
                continue
            delta = (attacked_values.loc[valid] - baseline_values.loc[valid]).abs()
            threshold = max(float(delta.std(ddof=0)) * 2.0, 0.01)
            crossed = merged.loc[valid, "timestamp_utc"].loc[delta > threshold]
            if not crossed.empty:
                return float((crossed.iloc[0] - merged["timestamp_utc"].iloc[0]).total_seconds())
    return None


def _align_signal_frames(attacked: pd.DataFrame, baseline: pd.DataFrame, signal: str) -> pd.DataFrame:
    tolerance = pd.Timedelta(seconds=max(_infer_sample_seconds(attacked), _infer_sample_seconds(baseline), 1) * 2)
    attack_frame = attacked[["timestamp_utc", signal]].sort_values("timestamp_utc").rename(columns={signal: f"{signal}_attacked"})
    baseline_frame = baseline[["timestamp_utc", signal]].sort_values("timestamp_utc").rename(columns={signal: f"{signal}_baseline"})
    merged = pd.merge_asof(
        attack_frame,
        baseline_frame,
        on="timestamp_utc",
        direction="nearest",
        tolerance=tolerance,
    )
    return merged.dropna(subset=[f"{signal}_attacked", f"{signal}_baseline"])


def _infer_sample_seconds(df: pd.DataFrame) -> int:
    if "timestamp_utc" not in df.columns or len(df) < 2:
        return 1
    stamps = pd.to_datetime(df["timestamp_utc"], utc=True).sort_values()
    return max(int((stamps.iloc[1] - stamps.iloc[0]).total_seconds()), 1)


def main() -> None:
    """CLI entrypoint for writing attacked-layer validation JSON and Markdown."""

    parser = argparse.ArgumentParser(description="Validate attacked layers against labels and clean baseline.")
    parser.add_argument("--attacked-truth", required=True)
    parser.add_argument("--attacked-measured", required=True)
    parser.add_argument("--cyber", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--baseline-truth", required=True)
    parser.add_argument("--baseline-measured", required=True)
    parser.add_argument("--windows", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    windows_df = read_dataframe(args.windows) if args.windows else None
    checks = validate_attacked_layers(
        attacked_truth_df=read_dataframe(args.attacked_truth),
        attacked_measured_df=read_dataframe(args.attacked_measured),
        cyber_df=read_dataframe(args.cyber),
        labels_df=read_dataframe(args.labels),
        baseline_truth_df=read_dataframe(args.baseline_truth),
        baseline_measured_df=read_dataframe(args.baseline_measured),
        windows_df=windows_df,
    )
    payload = [asdict(check) for check in checks]
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Attacked Dataset Validation", ""]
    for item in checks:
        metric = f" metric={item.metric:.4f}" if item.metric is not None else ""
        threshold = f" threshold={item.threshold}" if item.threshold is not None else ""
        lines.append(f"- {item.name}: {item.status}. {item.detail}{metric}{threshold}")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
