"""Shared utility support for DERGuardian.

This module provides units and channel dictionary helpers used across the Phase 1 data
pipeline, Phase 2 scenario pipeline, and Phase 3 evaluation/reporting layers.
The functions here are infrastructure code: they prepare paths, metadata,
profiles, graphs, units, or time alignment without changing canonical detector
outputs or benchmark decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd


@dataclass(slots=True)
class ChannelEntry:
    """Structured object used by the shared DERGuardian utility workflow."""

    channel: str
    dtype: str
    units: str
    layer: str
    description: str


PHASE_SUFFIX_MAP = {"phase_a": "phase A", "phase_b": "phase B", "phase_c": "phase C"}


def infer_channel_entry(column: str, dtype: str, layer: str) -> ChannelEntry:
    """Handle infer channel entry within the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if column == "timestamp_utc":
        return ChannelEntry(column, dtype, "UTC ISO 8601", layer, "Timestamp in coordinated universal time.")
    if column in {"window_start_utc", "window_end_utc", "start_time_utc", "end_time_utc", "ingest_timestamp_utc"}:
        return ChannelEntry(column, dtype, "UTC ISO 8601", layer, f"Timestamp field `{column}`.")
    if column == "simulation_index":
        return ChannelEntry(column, dtype, "index", layer, "Sequential simulation sample index.")
    if column in {"scenario_id", "run_id", "split_id", "source_layer"}:
        return ChannelEntry(column, dtype, "categorical", layer, f"Metadata field `{column}`.")
    if column in {"scenario_name", "target_component"}:
        return ChannelEntry(column, dtype, "categorical", layer, f"Scenario metadata field `{column}`.")
    if column in {"affected_assets", "affected_signals", "attack_affected_assets"}:
        return ChannelEntry(column, dtype, "list", layer, f"List-valued attack metadata field `{column}`.")
    if column == "causal_metadata":
        return ChannelEntry(column, dtype, "json", layer, "Structured causal metadata for the attack label.")
    if column == "sample_rate_seconds":
        return ChannelEntry(column, dtype, "seconds", layer, "Nominal sample period for the row.")
    if column == "window_seconds":
        return ChannelEntry(column, dtype, "seconds", layer, "Window length for aggregated model-ready data.")
    if column.startswith("event_") or column in {
        "actor",
        "user_id",
        "service_id",
        "role",
        "source_host",
        "source_ip",
        "source_port",
        "destination_host",
        "destination_ip",
        "destination_port",
        "auth_method",
        "auth_result",
        "lockout_status",
        "mfa_used",
        "protocol",
        "protocol_resource_type",
        "operation",
        "resource",
        "command_type",
        "requested_value",
        "command_result",
        "observable_signal",
        "observable_value",
        "response_code",
        "latency_seconds",
        "clock_offset_seconds",
        "jitter_seconds",
        "bytes_in",
        "bytes_out",
        "packet_count",
        "flow_id",
        "result",
        "notes",
        "campaign_id",
        "attack_flag",
        "attack_family",
        "severity",
        "error_code",
    }:
        return _cyber_entry(column, dtype, layer)
    if column.startswith("env_"):
        return _environment_entry(column, dtype, layer)
    if column.startswith("feeder_"):
        return _feeder_entry(column, dtype, layer)
    if column.startswith("bus_"):
        return _bus_entry(column, dtype, layer)
    if column.startswith("line_"):
        return _line_entry(column, dtype, layer)
    if column.startswith("pv_"):
        return _pv_entry(column, dtype, layer)
    if column.startswith("bess_"):
        return _bess_entry(column, dtype, layer)
    if column.startswith("load_"):
        return _load_entry(column, dtype, layer)
    if column.startswith("regulator_"):
        return ChannelEntry(column, dtype, "tap", layer, "Regulator tap position or derived control signal.")
    if column.startswith("capacitor_"):
        return ChannelEntry(column, dtype, "state", layer, "Capacitor bank discrete state.")
    if column.startswith("switch_") or column.startswith("breaker_"):
        return ChannelEntry(column, dtype, "state", layer, "Switch or breaker open/closed state.")
    if column.startswith("relay_"):
        return ChannelEntry(column, dtype, "state", layer, "Protection or alarm state proxy.")
    if column.startswith("derived_"):
        return ChannelEntry(column, dtype, "derived", layer, "Derived validation or anomaly-analysis feature.")
    if column.startswith("cyber_"):
        return ChannelEntry(column, dtype, "cyber", layer, "Windowed cyber feature derived from event stream.")
    if column.startswith("attack_"):
        return ChannelEntry(column, dtype, "label", layer, "Attack label or scenario metadata.")
    return ChannelEntry(column, dtype, "unknown", layer, "Channel not yet described by the automatic dictionary.")


def _environment_entry(column: str, dtype: str, layer: str) -> ChannelEntry:
    units = {
        "env_irradiance_wm2": "W/m^2",
        "env_temperature_c": "degC",
        "env_wind_speed_mps": "m/s",
        "env_humidity_pct": "%",
        "env_cloud_index": "0-1",
        "env_day_of_week": "categorical",
        "env_month": "month",
        "env_is_weekend": "flag",
        "env_is_holiday": "flag",
        "env_season": "categorical",
    }.get(column, "environment")
    return ChannelEntry(column, dtype, units, layer, f"Exogenous environmental or calendar driver `{column}`.")


def _feeder_entry(column: str, dtype: str, layer: str) -> ChannelEntry:
    if "_angle_deg" in column:
        units = "degrees"
    elif "_p_kw" in column:
        units = "kW"
    elif "_q_kvar" in column:
        units = "kvar"
    elif "losses_kvar" in column:
        units = "kvar"
    elif "losses_kw" in column:
        units = "kW"
    elif "_v_pu" in column:
        units = "pu"
    elif "_i_a" in column:
        units = "A"
    else:
        units = "feeder"
    return ChannelEntry(column, dtype, units, layer, "Feeder-head or substation observable.")


def _bus_entry(column: str, dtype: str, layer: str) -> ChannelEntry:
    match = re.match(r"bus_(?P<bus>[^_]+)_(?P<signal>.+)", column)
    if match is None:
        return ChannelEntry(column, dtype, "pu", layer, "Bus-level observable.")
    bus = match.group("bus")
    signal = match.group("signal")
    units = "pu" if "v_pu" in signal else "degrees" if "angle_deg" in signal else "bus"
    return ChannelEntry(column, dtype, units, layer, f"Bus `{bus}` electrical state for `{signal}`.")


def _line_entry(column: str, dtype: str, layer: str) -> ChannelEntry:
    units = "degrees" if "_angle_deg" in column else "A" if "_current_a" in column else "kW" if "_p_kw" in column else "kvar" if "_q_kvar" in column else "line"
    return ChannelEntry(column, dtype, units, layer, "Selected branch power-flow or current observable.")


def _pv_entry(column: str, dtype: str, layer: str) -> ChannelEntry:
    if "_p_kw" in column or "_available_kw" in column:
        units = "kW"
    elif "_q_kvar" in column:
        units = "kvar"
    elif "_terminal_v_pu" in column:
        units = "pu"
    elif "_terminal_i_a" in column:
        units = "A"
    elif "_curtailment_frac" in column:
        units = "per unit"
    elif column.endswith(("_status", "_status_cmd")):
        units = "flag"
    elif column.endswith(("_mode", "_mode_cmd")):
        units = "categorical"
    else:
        units = "state"
    return ChannelEntry(column, dtype, units, layer, "Photovoltaic asset state, capability, or control mode.")


def _bess_entry(column: str, dtype: str, layer: str) -> ChannelEntry:
    if "_soc" in column:
        units = "per unit"
    elif column.endswith(("_soc_min", "_soc_max")):
        units = "per unit"
    elif "_energy_kwh" in column:
        units = "kWh"
    elif "_p_kw" in column or "_target_kw" in column or "_actual_kw" in column or "_available_charge_kw" in column or "_available_discharge_kw" in column:
        units = "kW"
    elif "_q_kvar" in column:
        units = "kvar"
    elif "_terminal_v_pu" in column:
        units = "pu"
    elif "_terminal_i_a" in column:
        units = "A"
    elif column.endswith(("_status", "_status_cmd")):
        units = "flag"
    elif column.endswith(("_mode", "_mode_cmd", "_state")):
        units = "categorical"
    else:
        units = "state"
    return ChannelEntry(column, dtype, units, layer, "Battery energy storage asset state, dispatch, or status.")


def _load_entry(column: str, dtype: str, layer: str) -> ChannelEntry:
    units = "kW" if "_p_kw" in column else "kvar" if "_q_kvar" in column else "load"
    return ChannelEntry(column, dtype, units, layer, "Aggregate or class-specific feeder load metric.")


def _cyber_entry(column: str, dtype: str, layer: str) -> ChannelEntry:
    unit_map = {
        "timestamp_utc": "UTC ISO 8601",
        "ingest_timestamp_utc": "UTC ISO 8601",
        "latency_seconds": "seconds",
        "clock_offset_seconds": "seconds",
        "jitter_seconds": "seconds",
        "bytes_in": "bytes",
        "bytes_out": "bytes",
        "packet_count": "packets",
        "source_port": "port",
        "destination_port": "port",
        "observable_value": "varies",
        "attack_flag": "flag",
    }
    units = unit_map.get(column, "cyber")
    return ChannelEntry(column, dtype, units, layer, f"Cyber or protocol event field `{column}`.")


def build_channel_dictionary(df: pd.DataFrame, layer: str) -> list[ChannelEntry]:
    """Build channel dictionary for the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return [infer_channel_entry(column, str(df[column].dtype), layer) for column in df.columns]


def write_data_dictionary(dictionary_entries: list[ChannelEntry], output_path: str) -> str:
    """Write data dictionary for the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    rows = [
        "| channel | dtype | units | layer | description |",
        "|---|---|---|---|---|",
    ]
    for item in dictionary_entries:
        rows.append(
            f"| {item.channel} | {item.dtype} | {item.units} | {item.layer} | {item.description} |"
        )
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(rows))
        handle.write("\n")
    return output_path
