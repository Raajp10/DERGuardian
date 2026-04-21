"""Phase 1 context and fusion helper for DERGuardian.

This module builds or consumes structured context artifacts around normal-system
behavior, feature evidence, and detector outputs. It supports explanation and
fusion workflows but is not the canonical detector-selection mechanism.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from common.io_utils import read_dataframe, write_dataframe, write_jsonl
from phase1_models.model_utils import CANONICAL_ARTIFACT_ROOT
from phase1_models.residual_dataset import canonical_residual_artifact_path
from phase3_explanations.shared import humanize_feature_name


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
    "scenario_window_id",
    "split_name",
}

PV_ASSETS = ("pv35", "pv60", "pv83")
BESS_ASSETS = ("bess48", "bess108")


def canonical_context_summary_path(root: Path, artifact_root_name: str = CANONICAL_ARTIFACT_ROOT) -> Path:
    """Handle canonical context summary path within the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return root / "outputs" / "reports" / artifact_root_name / "phase1_context_summaries.jsonl"


def canonical_context_table_path(root: Path, artifact_root_name: str = CANONICAL_ARTIFACT_ROOT) -> Path:
    """Handle canonical context table path within the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return root / "outputs" / "reports" / artifact_root_name / "phase1_context_summaries.parquet"


def load_context_sources(
    root: Path,
    residual_path: Path | None = None,
    attacked_windows_path: Path | None = None,
    clean_windows_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load context sources for the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    residual = read_dataframe(residual_path or canonical_residual_artifact_path(root)).copy()
    attacked = read_dataframe(attacked_windows_path or (root / "outputs" / "attacked" / "merged_windows.parquet")).copy()
    clean = read_dataframe(clean_windows_path or (root / "outputs" / "windows" / "merged_windows_clean.parquet")).copy()
    for frame in (residual, attacked, clean):
        if "window_start_utc" in frame.columns:
            frame["window_start_utc"] = pd.to_datetime(frame["window_start_utc"], utc=True)
        if "window_end_utc" in frame.columns:
            frame["window_end_utc"] = pd.to_datetime(frame["window_end_utc"], utc=True)
    return residual, attacked, clean


def build_context_summaries(
    *,
    residual_df: pd.DataFrame,
    attacked_windows: pd.DataFrame,
    clean_windows: pd.DataFrame,
    top_k: int = 8,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Build context summaries for the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    attacked_lookup = attacked_windows.set_index("window_start_utc", drop=False)
    clean_lookup = clean_windows.set_index("window_start_utc", drop=False)
    summaries: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []

    for _, residual_row in residual_df.sort_values("window_start_utc").iterrows():
        start = pd.Timestamp(residual_row["window_start_utc"]).tz_convert("UTC")
        attacked_row = attacked_lookup.loc[start]
        clean_row = clean_lookup.loc[start]
        if isinstance(attacked_row, pd.DataFrame):
            attacked_row = attacked_row.iloc[0]
        if isinstance(clean_row, pd.DataFrame):
            clean_row = clean_row.iloc[0]

        top_deviations = compute_top_deviating_signals(attacked_row, clean_row, top_k=top_k)
        mapped_assets = sorted({item["asset_id"] for item in top_deviations if item["asset_id"] not in {"feeder", "environment", "derived", "unknown"}})
        mapped_components = sorted({item["component_type"] for item in top_deviations if item["component_type"] != "unknown"})
        environment_context = build_environment_context(attacked_row, clean_row)
        feeder_context = build_feeder_context(attacked_row, clean_row)
        voltage_context = build_voltage_context(attacked_row, clean_row)
        der_dispatch_context = build_der_dispatch_consistency(attacked_row, clean_row)
        soc_context = build_soc_consistency(attacked_row, clean_row)
        summary_stats = build_summary_statistics(top_deviations, voltage_context, der_dispatch_context, soc_context)
        compact_summary = compose_compact_summary(
            top_deviations=top_deviations,
            mapped_assets=mapped_assets,
            voltage_context=voltage_context,
            der_dispatch_context=der_dispatch_context,
            soc_context=soc_context,
        )

        payload = {
            "window_start_utc": start.isoformat().replace("+00:00", "Z"),
            "window_end_utc": pd.Timestamp(residual_row["window_end_utc"]).tz_convert("UTC").isoformat().replace("+00:00", "Z"),
            "window_seconds": int(residual_row.get("window_seconds", 0) or 0),
            "split_name": str(residual_row.get("split_name", "")),
            "scenario_id": str(residual_row.get("scenario_window_id", residual_row.get("scenario_id", "benign"))),
            "source_scenario_id": str(residual_row.get("scenario_id", "unknown")),
            "mapped_assets": mapped_assets,
            "mapped_components": mapped_components,
            "environment_context": environment_context,
            "feeder_context": feeder_context,
            "voltage_violations": voltage_context,
            "der_dispatch_consistency": der_dispatch_context,
            "soc_consistency": soc_context,
            "top_deviating_signals": top_deviations,
            "summary_statistics": summary_stats,
            "compact_summary": compact_summary,
        }
        summaries.append(payload)
        flat_rows.append(
            {
                "window_start_utc": payload["window_start_utc"],
                "window_end_utc": payload["window_end_utc"],
                "window_seconds": payload["window_seconds"],
                "split_name": payload["split_name"],
                "scenario_id": payload["scenario_id"],
                "source_scenario_id": payload["source_scenario_id"],
                "mapped_assets_json": json.dumps(mapped_assets),
                "mapped_components_json": json.dumps(mapped_components),
                "changed_channel_count": int(summary_stats["changed_channel_count"]),
                "changed_voltage_channel_count": int(summary_stats["changed_voltage_channel_count"]),
                "max_abs_relative_change": float(summary_stats["max_abs_relative_change"]),
                "mean_abs_relative_change": float(summary_stats["mean_abs_relative_change"]),
                "voltage_violation_flag": int(voltage_context["violation_flag"]),
                "voltage_violation_count": int(voltage_context["violation_count_last"]),
                "pv_dispatch_issue_count": int(der_dispatch_context["pv_dispatch_issue_count"]),
                "bess_dispatch_issue_count": int(der_dispatch_context["bess_dispatch_issue_count"]),
                "soc_issue_count": int(soc_context["soc_issue_count"]),
                "compact_summary": compact_summary,
            }
        )

    flat_frame = pd.DataFrame(flat_rows)
    return summaries, flat_frame


def persist_context_summaries(
    root: Path,
    summaries: list[dict[str, Any]],
    flat_frame: pd.DataFrame,
    artifact_root_name: str = CANONICAL_ARTIFACT_ROOT,
) -> tuple[Path, Path]:
    """Handle persist context summaries within the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    jsonl_path = canonical_context_summary_path(root, artifact_root_name=artifact_root_name)
    table_path = canonical_context_table_path(root, artifact_root_name=artifact_root_name)
    write_jsonl(summaries, jsonl_path)
    write_dataframe(flat_frame, table_path, fmt="parquet")
    return jsonl_path, table_path


def build_and_persist_context_summaries(
    root: Path,
    top_k: int = 8,
    artifact_root_name: str = CANONICAL_ARTIFACT_ROOT,
) -> tuple[list[dict[str, Any]], pd.DataFrame, tuple[Path, Path]]:
    """Build and persist context summaries for the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    residual, attacked, clean = load_context_sources(root)
    summaries, flat_frame = build_context_summaries(
        residual_df=residual,
        attacked_windows=attacked,
        clean_windows=clean,
        top_k=top_k,
    )
    paths = persist_context_summaries(root, summaries, flat_frame, artifact_root_name=artifact_root_name)
    return summaries, flat_frame, paths


def compute_top_deviating_signals(attacked_row: pd.Series, clean_row: pd.Series, top_k: int = 8) -> list[dict[str, Any]]:
    """Compute top deviating signals for the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    records: list[dict[str, Any]] = []
    numeric_columns = [
        column
        for column in attacked_row.index
        if column not in WINDOW_META_COLUMNS
        and column in clean_row.index
        and pd.api.types.is_numeric_dtype(type(attacked_row[column]))
        and pd.api.types.is_numeric_dtype(type(clean_row[column]))
    ]
    for column in numeric_columns:
        attacked_value = _safe_float(attacked_row[column], default=0.0)
        clean_value = _safe_float(clean_row[column], default=0.0)
        delta = attacked_value - clean_value
        scale = max(abs(attacked_value), abs(clean_value), 1.0)
        relative_change = delta / scale
        ranking_score = abs(relative_change)
        if ranking_score < 0.005 and abs(delta) < 0.25:
            continue
        signal, aggregation = split_window_feature_name(column)
        mapping = map_signal_to_asset_component(signal)
        records.append(
            {
                "feature": f"delta__{column}",
                "signal": signal,
                "aggregation": aggregation,
                "asset_id": mapping["asset_id"],
                "component_type": mapping["component_type"],
                "signal_group": mapping["signal_group"],
                "attacked_value": round(attacked_value, 6),
                "baseline_value": round(clean_value, 6),
                "delta": round(delta, 6),
                "relative_change": round(relative_change, 6),
                "ranking_score": round(ranking_score, 6),
                "direction": "increase" if delta >= 0 else "decrease",
                "description": humanize_feature_name(f"delta__{column}"),
            }
        )
    records.sort(key=lambda item: item["ranking_score"], reverse=True)
    return records[:top_k]


def split_window_feature_name(column: str) -> tuple[str, str]:
    """Handle split window feature name within the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if "__" not in column:
        return column, "value"
    signal, aggregation = column.rsplit("__", 1)
    return signal, aggregation


def map_signal_to_asset_component(signal: str) -> dict[str, str]:
    """Handle map signal to asset component within the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parts = signal.split("_")
    if signal.startswith("pv_") and len(parts) >= 2:
        return {"asset_id": parts[1], "component_type": "pv", "signal_group": parts[0]}
    if signal.startswith("bess_") and len(parts) >= 2:
        return {"asset_id": parts[1], "component_type": "bess", "signal_group": parts[0]}
    if signal.startswith("regulator_") and len(parts) >= 2:
        return {"asset_id": parts[1], "component_type": "regulator", "signal_group": parts[0]}
    if signal.startswith("capacitor_") and len(parts) >= 2:
        return {"asset_id": parts[1], "component_type": "capacitor", "signal_group": parts[0]}
    if signal.startswith("switch_") and len(parts) >= 2:
        return {"asset_id": parts[1], "component_type": "switch", "signal_group": parts[0]}
    if signal.startswith("bus_") and len(parts) >= 2:
        return {"asset_id": f"bus{parts[1]}", "component_type": "bus", "signal_group": parts[0]}
    if signal.startswith("feeder_"):
        return {"asset_id": "feeder", "component_type": "feeder", "signal_group": "feeder"}
    if signal.startswith("env_"):
        return {"asset_id": "environment", "component_type": "environment", "signal_group": "env"}
    if signal.startswith("derived_"):
        return {"asset_id": "feeder", "component_type": "derived", "signal_group": "derived"}
    return {"asset_id": "unknown", "component_type": "unknown", "signal_group": "unknown"}


def build_environment_context(attacked_row: pd.Series, clean_row: pd.Series) -> dict[str, Any]:
    """Build environment context for the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    fields = {
        "irradiance_wm2": "env_irradiance_wm2__last",
        "temperature_c": "env_temperature_c__last",
        "wind_speed_mps": "env_wind_speed_mps__last",
        "humidity_pct": "env_humidity_pct__last",
        "cloud_index": "env_cloud_index__last",
    }
    payload: dict[str, Any] = {}
    for alias, column in fields.items():
        if column in attacked_row.index:
            payload[alias] = round(_safe_float(attacked_row[column], default=0.0), 6)
            payload[f"{alias}_delta"] = round(_safe_float(attacked_row[column], default=0.0) - _safe_float(clean_row.get(column, attacked_row[column]), default=0.0), 6)
    return payload


def build_feeder_context(attacked_row: pd.Series, clean_row: pd.Series) -> dict[str, Any]:
    """Build feeder context for the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    payload: dict[str, Any] = {}
    for column in [
        "feeder_p_kw_total__last",
        "feeder_q_kvar_total__last",
        "feeder_v_pu_phase_a__last",
        "feeder_v_pu_phase_b__last",
        "feeder_v_pu_phase_c__last",
    ]:
        if column not in attacked_row.index:
            continue
        alias = column.replace("__last", "")
        payload[alias] = round(_safe_float(attacked_row[column], default=0.0), 6)
        payload[f"{alias}_delta"] = round(_safe_float(attacked_row[column], default=0.0) - _safe_float(clean_row.get(column, attacked_row[column]), default=0.0), 6)
    payload["feeder_operating_band"] = infer_feeder_operating_band(_safe_float(attacked_row.get("feeder_p_kw_total__last", 0.0), default=0.0))
    return payload


def build_voltage_context(attacked_row: pd.Series, clean_row: pd.Series) -> dict[str, Any]:
    """Build voltage context for the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    count_column = "derived_voltage_violation_count__last"
    flag_column = "derived_voltage_violation_flag__last"
    voltage_candidates = []
    for column in attacked_row.index:
        if "_v_pu_" not in column or not column.endswith("__last"):
            continue
        attacked_value = _safe_float(attacked_row[column], default=0.0)
        clean_value = _safe_float(clean_row.get(column, attacked_value), default=0.0)
        delta = attacked_value - clean_value
        if abs(delta) < 0.005:
            continue
        voltage_candidates.append(
            {
                "signal": column.replace("__last", ""),
                "attacked_value": round(attacked_value, 6),
                "baseline_value": round(clean_value, 6),
                "delta": round(delta, 6),
            }
        )
    voltage_candidates.sort(key=lambda item: abs(float(item["delta"])), reverse=True)
    return {
        "violation_count_last": int(round(_safe_float(attacked_row.get(count_column, 0.0), default=0.0))),
        "violation_flag": int(round(_safe_float(attacked_row.get(flag_column, 0.0), default=0.0))),
        "top_voltage_deviations": voltage_candidates[:5],
    }


def build_der_dispatch_consistency(attacked_row: pd.Series, clean_row: pd.Series) -> dict[str, Any]:
    """Build der dispatch consistency for the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    pv_records: list[dict[str, Any]] = []
    bess_records: list[dict[str, Any]] = []
    for asset in PV_ASSETS:
        available_col = f"pv_{asset}_available_kw__last"
        actual_col = f"pv_{asset}_p_kw__last"
        curtail_col = f"pv_{asset}_curtailment_frac__last"
        if available_col not in attacked_row.index or actual_col not in attacked_row.index:
            continue
        available = _safe_float(attacked_row[available_col], default=0.0)
        actual = _safe_float(attacked_row[actual_col], default=0.0)
        clean_available = _safe_float(clean_row.get(available_col, available), default=0.0)
        clean_actual = _safe_float(clean_row.get(actual_col, actual), default=0.0)
        dispatch_gap = available - actual
        baseline_gap = clean_available - clean_actual
        pv_records.append(
            {
                "asset_id": asset,
                "available_kw": round(available, 6),
                "dispatch_kw": round(actual, 6),
                "dispatch_gap_kw": round(dispatch_gap, 6),
                "dispatch_gap_delta_kw": round(dispatch_gap - baseline_gap, 6),
                "curtailment_frac": round(_safe_float(attacked_row.get(curtail_col, 0.0), default=0.0), 6),
                "baseline_curtailment_frac": round(_safe_float(clean_row.get(curtail_col, 0.0), default=0.0), 6),
                "issue_flag": int(abs(dispatch_gap - baseline_gap) >= max(5.0, 0.08 * max(abs(available), 1.0))),
            }
        )
    for asset in BESS_ASSETS:
        p_col = f"bess_{asset}_p_kw__last"
        charge_col = f"bess_{asset}_available_charge_kw__last"
        discharge_col = f"bess_{asset}_available_discharge_kw__last"
        status_col = f"bess_{asset}_status__last"
        if p_col not in attacked_row.index:
            continue
        p_kw = _safe_float(attacked_row[p_col], default=0.0)
        available_charge = _safe_float(attacked_row.get(charge_col, 0.0), default=0.0)
        available_discharge = _safe_float(attacked_row.get(discharge_col, 0.0), default=0.0)
        dispatch_violation = max(0.0, p_kw - available_discharge) if p_kw >= 0 else max(0.0, abs(p_kw) - available_charge)
        clean_dispatch_violation = 0.0
        if p_col in clean_row.index:
            clean_p = _safe_float(clean_row[p_col], default=0.0)
            clean_available_charge = _safe_float(clean_row.get(charge_col, 0.0), default=0.0)
            clean_available_discharge = _safe_float(clean_row.get(discharge_col, 0.0), default=0.0)
            clean_dispatch_violation = max(0.0, clean_p - clean_available_discharge) if clean_p >= 0 else max(0.0, abs(clean_p) - clean_available_charge)
        bess_records.append(
            {
                "asset_id": asset,
                "dispatch_kw": round(p_kw, 6),
                "available_charge_kw": round(available_charge, 6),
                "available_discharge_kw": round(available_discharge, 6),
                "status": round(_safe_float(attacked_row.get(status_col, 0.0), default=0.0), 6),
                "dispatch_violation_kw": round(dispatch_violation, 6),
                "dispatch_violation_delta_kw": round(dispatch_violation - clean_dispatch_violation, 6),
                "issue_flag": int(dispatch_violation > 5.0 or abs(dispatch_violation - clean_dispatch_violation) > 5.0),
            }
        )
    return {
        "pv_dispatch_checks": pv_records,
        "bess_dispatch_checks": bess_records,
        "pv_dispatch_issue_count": int(sum(int(item["issue_flag"]) for item in pv_records)),
        "bess_dispatch_issue_count": int(sum(int(item["issue_flag"]) for item in bess_records)),
    }


def build_soc_consistency(attacked_row: pd.Series, clean_row: pd.Series) -> dict[str, Any]:
    """Build soc consistency for the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    records: list[dict[str, Any]] = []
    for asset in BESS_ASSETS:
        soc_col = f"bess_{asset}_soc__last"
        soc_min_col = f"bess_{asset}_soc_min__last"
        soc_max_col = f"bess_{asset}_soc_max__last"
        if soc_col not in attacked_row.index:
            continue
        soc = _safe_float(attacked_row[soc_col], default=0.0)
        soc_min = _safe_float(attacked_row.get(soc_min_col, 0.0), default=0.0)
        soc_max = _safe_float(attacked_row.get(soc_max_col, 1.0), default=1.0)
        clean_soc = _safe_float(clean_row.get(soc_col, soc), default=0.0)
        lower_margin = soc - soc_min
        upper_margin = soc_max - soc
        out_of_bounds = soc < soc_min - 1e-6 or soc > soc_max + 1e-6
        records.append(
            {
                "asset_id": asset,
                "soc": round(soc, 6),
                "baseline_soc": round(clean_soc, 6),
                "soc_delta": round(soc - clean_soc, 6),
                "soc_min": round(soc_min, 6),
                "soc_max": round(soc_max, 6),
                "lower_margin": round(lower_margin, 6),
                "upper_margin": round(upper_margin, 6),
                "issue_flag": int(out_of_bounds or abs(soc - clean_soc) > 0.05),
            }
        )
    return {
        "bess_soc_checks": records,
        "soc_issue_count": int(sum(int(item["issue_flag"]) for item in records)),
    }


def build_summary_statistics(
    top_deviations: list[dict[str, Any]],
    voltage_context: dict[str, Any],
    der_dispatch_context: dict[str, Any],
    soc_context: dict[str, Any],
) -> dict[str, Any]:
    """Build summary statistics for the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    relative_changes = [abs(float(item.get("relative_change", 0.0))) for item in top_deviations]
    changed_channel_count = int(sum(value >= 0.02 for value in relative_changes))
    changed_voltage_channel_count = int(sum(1 for item in top_deviations if "_v_pu_" in str(item.get("feature", ""))))
    return {
        "max_abs_relative_change": round(float(max(relative_changes) if relative_changes else 0.0), 6),
        "mean_abs_relative_change": round(float(np.mean(relative_changes) if relative_changes else 0.0), 6),
        "changed_channel_count": changed_channel_count,
        "changed_voltage_channel_count": changed_voltage_channel_count,
        "voltage_violation_flag": int(voltage_context["violation_flag"]),
        "voltage_violation_count": int(voltage_context["violation_count_last"]),
        "pv_dispatch_issue_count": int(der_dispatch_context["pv_dispatch_issue_count"]),
        "bess_dispatch_issue_count": int(der_dispatch_context["bess_dispatch_issue_count"]),
        "soc_issue_count": int(soc_context["soc_issue_count"]),
    }


def compose_compact_summary(
    *,
    top_deviations: list[dict[str, Any]],
    mapped_assets: list[str],
    voltage_context: dict[str, Any],
    der_dispatch_context: dict[str, Any],
    soc_context: dict[str, Any],
) -> str:
    """Handle compose compact summary within the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    asset_text = ", ".join(mapped_assets[:3]) if mapped_assets else "feeder context"
    top_signals = ", ".join(item["signal"] for item in top_deviations[:3]) if top_deviations else "no strong deltas"
    notes: list[str] = [f"Primary deviations concentrate around {asset_text}.", f"Top changed signals: {top_signals}."]
    if voltage_context["violation_flag"] or voltage_context["violation_count_last"]:
        notes.append(
            f"Voltage violations are present with count {voltage_context['violation_count_last']}."
        )
    if der_dispatch_context["pv_dispatch_issue_count"] or der_dispatch_context["bess_dispatch_issue_count"]:
        notes.append(
            "DER dispatch consistency checks show mismatches between availability and realized dispatch."
        )
    if soc_context["soc_issue_count"]:
        notes.append("BESS SOC consistency checks are materially shifted.")
    return " ".join(notes)


def infer_feeder_operating_band(feeder_p_kw_total: float) -> str:
    """Handle infer feeder operating band within the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    abs_power = abs(feeder_p_kw_total)
    if abs_power >= 6000.0:
        return "high_load"
    if abs_power >= 3500.0:
        return "moderate_load"
    return "light_load"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        converted = float(value)
    except Exception:
        return default
    if np.isnan(converted):
        return default
    return converted


def main() -> None:
    """Run the command-line entrypoint for the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Build structured Phase 1 context summaries from aligned residual windows.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--artifact-root", default=CANONICAL_ARTIFACT_ROOT)
    args = parser.parse_args()

    root = Path(args.project_root)
    summaries, flat_frame, paths = build_and_persist_context_summaries(
        root=root,
        top_k=args.top_k,
        artifact_root_name=args.artifact_root,
    )
    print(f"Wrote {len(summaries)} context summaries to {paths[0]}")
    print(f"Wrote flattened context table to {paths[1]}")


if __name__ == "__main__":
    main()
