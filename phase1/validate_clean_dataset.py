from __future__ import annotations

from pathlib import Path
import argparse
from dataclasses import asdict
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from common.metadata_schema import ValidationCheck
from common.io_utils import read_dataframe


def validate_clean_layers(
    truth_df: pd.DataFrame,
    measured_df: pd.DataFrame,
    cyber_df: pd.DataFrame,
) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    timestamps = pd.to_datetime(truth_df["timestamp_utc"], utc=True)
    sample_rate = (timestamps.iloc[1] - timestamps.iloc[0]).total_seconds() if len(timestamps) > 1 else 0.0
    continuity_ok = timestamps.diff().dropna().dt.total_seconds().eq(sample_rate).all()
    checks.append(ValidationCheck("time_continuity", "pass" if continuity_ok else "warn", "Truth-layer timestamps use a consistent cadence.", sample_rate))

    residual = truth_df["derived_feeder_power_balance_residual_kw"].abs().quantile(0.95)
    checks.append(ValidationCheck("power_balance", "pass" if residual < 400.0 else "warn", "95th percentile feeder power-balance residual.", float(residual), 400.0))

    voltage_violations = truth_df["derived_voltage_violation_count"].mean()
    checks.append(ValidationCheck("voltage_profile", "pass", "Average count of out-of-band bus-voltage observations per sample.", float(voltage_violations)))

    soc_columns = [column for column in truth_df.columns if column.startswith("bess_") and column.endswith("_soc")]
    soc_ok = all(truth_df[column].between(0.0, 1.0).all() for column in soc_columns)
    checks.append(ValidationCheck("soc_bounds", "pass" if soc_ok else "fail", "All BESS SOC traces remain within [0, 1]."))

    soc_residual_columns = [column for column in truth_df.columns if column.startswith("derived_bess_") and column.endswith("_soc_consistency_residual")]
    soc_residual = float(truth_df[soc_residual_columns].abs().quantile(0.99).max()) if soc_residual_columns else 0.0
    checks.append(
        ValidationCheck(
            "soc_consistency",
            "pass" if soc_residual < 0.02 else "warn",
            "99th percentile absolute BESS SOC step residual.",
            soc_residual,
            0.02,
        )
    )

    pv_residual_columns = [column for column in truth_df.columns if column.startswith("derived_pv_") and column.endswith("_availability_residual_kw")]
    pv_negative_fraction = float((truth_df[pv_residual_columns] < -0.05).mean().mean()) if pv_residual_columns else 0.0
    checks.append(
        ValidationCheck(
            "pv_availability_consistency",
            "pass" if pv_negative_fraction == 0.0 else "warn",
            "Fraction of PV samples where dispatched active power exceeds available active power.",
            pv_negative_fraction,
            0.0,
        )
    )

    feeder_ramp = float(truth_df["derived_feeder_ramp_kw_per_s"].abs().quantile(0.99)) if "derived_feeder_ramp_kw_per_s" in truth_df.columns else 0.0
    checks.append(
        ValidationCheck(
            "feeder_ramp_rate",
            "pass" if feeder_ramp < 600.0 else "warn",
            "99th percentile feeder active-power ramp rate.",
            feeder_ramp,
            600.0,
        )
    )

    missing_fraction = measured_df.isna().mean().mean()
    checks.append(ValidationCheck("measured_missingness", "pass" if 0.0001 < missing_fraction < 0.15 else "warn", "Average measured-layer missingness fraction.", float(missing_fraction), 0.15))

    measured_timestamps = pd.to_datetime(measured_df["timestamp_utc"], utc=True)
    truth_timestamps = pd.to_datetime(truth_df["timestamp_utc"], utc=True)
    mean_offset = float((measured_timestamps - truth_timestamps).dt.total_seconds().mean()) if len(measured_timestamps) == len(truth_timestamps) else 0.0
    checks.append(
        ValidationCheck(
            "measurement_time_offset",
            "pass" if abs(mean_offset) > 0.0 else "warn",
            "Average measured-layer timestamp offset relative to truth.",
            mean_offset,
        )
    )

    truth_bus_columns = [column for column in truth_df.columns if column.startswith("bus_") and "_v_pu_" in column]
    measured_bus_columns = [column for column in measured_df.columns if column.startswith("bus_") and "_v_pu_" in column]
    bus_observability_ratio = float(len(measured_bus_columns) / max(len(truth_bus_columns), 1))
    checks.append(
        ValidationCheck(
            "sparse_observability",
            "pass" if bus_observability_ratio < 1.0 else "warn",
            "Measured-layer bus-voltage observability ratio relative to truth.",
            bus_observability_ratio,
            1.0,
        )
    )

    event_types = set(cyber_df["event_type"].astype(str).tolist()) if not cyber_df.empty and "event_type" in cyber_df.columns else set()
    coverage_ok = {"authentication", "command", "configuration", "telemetry"}.issubset(event_types)
    checks.append(ValidationCheck("cyber_coverage", "pass" if coverage_ok else "warn", f"Cyber layer contains categories: {sorted(event_types)}"))

    timing_complete = (
        not cyber_df.empty
        and cyber_df.get("latency_seconds", pd.Series(dtype=float)).notna().mean() > 0.99
        and cyber_df.get("jitter_seconds", pd.Series(dtype=float)).notna().mean() > 0.99
        and cyber_df.get("clock_offset_seconds", pd.Series(dtype=float)).notna().mean() > 0.99
    )
    checks.append(
        ValidationCheck(
            "cyber_timing_quality",
            "pass" if timing_complete else "warn",
            "Cyber events include populated latency, jitter, and clock-offset fields.",
        )
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the clean cyber-physical dataset.")
    parser.add_argument("--truth", required=True)
    parser.add_argument("--measured", required=True)
    parser.add_argument("--cyber", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    checks = validate_clean_layers(
        truth_df=read_dataframe(args.truth),
        measured_df=read_dataframe(args.measured),
        cyber_df=read_dataframe(args.cyber),
    )
    payload = [asdict(check) for check in checks]
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Clean Dataset Validation", ""]
    for item in checks:
        metric = f" metric={item.metric:.4f}" if item.metric is not None else ""
        threshold = f" threshold={item.threshold}" if item.threshold is not None else ""
        lines.append(f"- {item.name}: {item.status}. {item.detail}{metric}{threshold}")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
