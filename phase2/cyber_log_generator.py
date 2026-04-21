"""Phase 2 scenario and attacked-dataset support for DERGuardian.

This module implements cyber log generator logic for schema-bound synthetic attack
scenarios, injection compilation, cyber logs, labels, validation, or reporting.
Generated scenarios are heldout synthetic evidence and are not claimed as
real-world zero-day proof.
"""

from __future__ import annotations

from itertools import count
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from common.config import PipelineConfig


def generate_baseline_cyber_events(
    truth_df: pd.DataFrame,
    control_df: pd.DataFrame,
    config: PipelineConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate baseline cyber events for the Phase 2 scenario and attacked-dataset workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    truth = truth_df.copy()
    truth["timestamp_utc"] = pd.to_datetime(truth["timestamp_utc"], utc=True)
    control = control_df.copy()
    control["timestamp_utc"] = pd.to_datetime(control["timestamp_utc"], utc=True)
    event_ids = count(1)
    events: list[dict[str, object]] = []
    telemetry_step = max(int(round(config.cyber_telemetry_interval_seconds / config.simulation_resolution_seconds)), 1)
    auth_step = max(int(round(config.cyber_auth_interval_minutes * 60 / config.simulation_resolution_seconds)), 1)
    reference_ts = truth["timestamp_utc"].min()
    hosts = {
        "scada": ("scada-core", "10.10.0.10"),
        "derms": ("derms-app", "10.10.0.20"),
        "hmi": ("operator-hmi", "10.10.0.30"),
        "broker": ("mqtt-broker", "10.10.0.40"),
    }
    protocol_ports = {"tls": 443, "https": 443, "modbus": 502, "dnp3": 20000, "mqtt": 1883}

    for idx in range(0, len(truth), auth_step):
        ts = truth["timestamp_utc"].iloc[idx]
        events.append(
            _event(
                event_id=next(event_ids),
                timestamp=ts,
                event_type="authentication",
                actor="derms_service",
                user_id="svc_derms",
                role="service",
                source_host=hosts["derms"][0],
                source_ip=hosts["derms"][1],
                destination_host=hosts["scada"][0],
                destination_ip=hosts["scada"][1],
                source_port=protocol_ports["tls"],
                destination_port=protocol_ports["tls"],
                auth_method="mTLS",
                auth_result="success",
                lockout_status=False,
                mfa_used=False,
                protocol="tls",
                latency_seconds=float(rng.integers(8, 20)) / 1000.0,
                clock_offset_seconds=_clock_offset_seconds(hosts["derms"][0], ts, reference_ts),
                jitter_seconds=float(rng.integers(1, 4)) / 1000.0,
                bytes_in=int(rng.integers(240, 520)),
                bytes_out=int(rng.integers(240, 520)),
                flow_id=f"flow-auth-{idx}",
                packet_count=int(rng.integers(2, 6)),
                result="success",
            )
        )

    for asset_column in [column for column in control.columns if column.endswith("_mode")]:
        ts = control["timestamp_utc"].iloc[0]
        events.append(
            _event(
                event_id=next(event_ids),
                timestamp=ts,
                event_type="configuration",
                actor="derms_operator",
                user_id="op_derms",
                role="engineer",
                source_host=hosts["hmi"][0],
                source_ip=hosts["hmi"][1],
                destination_host=hosts["derms"][0],
                destination_ip=hosts["derms"][1],
                source_port=protocol_ports["https"],
                destination_port=protocol_ports["https"],
                protocol="https",
                operation="PATCH",
                resource=asset_column,
                protocol_resource_type="parameter",
                command_type="mode_init",
                requested_value=str(control[asset_column].iloc[0]),
                command_result="success",
                latency_seconds=float(rng.integers(40, 120)) / 1000.0,
                clock_offset_seconds=_clock_offset_seconds(hosts["hmi"][0], ts, reference_ts),
                jitter_seconds=float(rng.integers(5, 12)) / 1000.0,
                bytes_in=int(rng.integers(320, 880)),
                bytes_out=int(rng.integers(320, 880)),
                flow_id=f"flow-config-{asset_column}",
                packet_count=int(rng.integers(4, 10)),
                result="success",
            )
        )

    for idx in range(0, len(truth), telemetry_step):
        ts = truth["timestamp_utc"].iloc[idx]
        row = truth.iloc[idx]
        resources = [
            ("feeder", "dnp3", "feeder_p_kw_total", row["feeder_p_kw_total"]),
            ("pv60", "modbus", "pv_pv60_p_kw", row.get("pv_pv60_p_kw")),
            ("pv83", "modbus", "pv_pv83_p_kw", row.get("pv_pv83_p_kw")),
            ("bess48", "mqtt", "bess_bess48_soc", row.get("bess_bess48_soc")),
            ("bess108", "mqtt", "bess_bess108_soc", row.get("bess_bess108_soc")),
        ]
        for asset, protocol, signal, value in resources:
            source = hosts["scada"] if protocol == "dnp3" else hosts["derms"] if protocol == "modbus" else hosts["broker"]
            dest = _asset_endpoint(asset)
            success = bool(rng.random() > 0.002)
            response_code = "success" if success else "timeout"
            events.append(
                _event(
                    event_id=next(event_ids),
                    timestamp=ts,
                    event_type="telemetry",
                    actor="telemetry_service",
                    service_id="svc_telemetry",
                    role="service",
                    source_host=source[0],
                    source_ip=source[1],
                    source_port=protocol_ports[protocol],
                    destination_host=dest[0],
                    destination_ip=dest[1],
                    destination_port=protocol_ports[protocol],
                    protocol=protocol,
                    operation="read" if protocol != "mqtt" else "publish",
                    resource=signal,
                    protocol_resource_type="register" if protocol == "modbus" else "object" if protocol == "dnp3" else "topic",
                    observable_signal=signal,
                    observable_value=None if pd.isna(value) else float(value),
                    response_code=response_code,
                    latency_seconds=float(rng.integers(10, 80)) / 1000.0,
                    clock_offset_seconds=_clock_offset_seconds(source[0], ts, reference_ts),
                    jitter_seconds=float(rng.integers(1, 8)) / 1000.0,
                    bytes_in=int(rng.integers(64, 256)),
                    bytes_out=int(rng.integers(64, 256)),
                    packet_count=int(rng.integers(2, 7)),
                    flow_id=f"flow-{protocol}-{asset}-{idx}",
                    result=response_code,
                    error_code=None if success else "timeout",
                )
            )

    change_columns = [column for column in control.columns if column.endswith(("_target_kw", "_curtailment_frac", "_mode", "_status"))]
    for column in change_columns:
        shifted = control[column].shift(1)
        change_mask = shifted.ne(control[column]) | shifted.isna()
        for idx in control.index[change_mask]:
            ts = control.loc[idx, "timestamp_utc"]
            events.append(
                _event(
                    event_id=next(event_ids),
                    timestamp=ts,
                    event_type="command",
                    actor="derms_operator",
                    user_id="op_derms",
                    role="operator",
                    source_host=hosts["hmi"][0],
                    source_ip=hosts["hmi"][1],
                    source_port=protocol_ports["modbus" if column.startswith(("pv_", "bess_")) else "dnp3"],
                    destination_host=hosts["derms"][0],
                    destination_ip=hosts["derms"][1],
                    destination_port=protocol_ports["modbus" if column.startswith(("pv_", "bess_")) else "dnp3"],
                    protocol="modbus" if column.startswith(("pv_", "bess_")) else "dnp3",
                    command_type=column,
                    protocol_resource_type="register" if column.startswith(("pv_", "bess_")) else "object",
                    requested_value=str(control.loc[idx, column]),
                    command_result="success",
                    latency_seconds=float(rng.integers(30, 150)) / 1000.0,
                    clock_offset_seconds=_clock_offset_seconds(hosts["hmi"][0], ts, reference_ts),
                    jitter_seconds=float(rng.integers(3, 18)) / 1000.0,
                    bytes_in=int(rng.integers(220, 760)),
                    bytes_out=int(rng.integers(220, 760)),
                    flow_id=f"flow-command-{column}-{idx}",
                    packet_count=int(rng.integers(3, 8)),
                    result="success",
                )
            )
    return pd.DataFrame(events).sort_values("timestamp_utc").reset_index(drop=True)


def inject_attack_events(
    cyber_df: pd.DataFrame,
    scenarios: list[dict[str, object]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Handle inject attack events within the Phase 2 scenario and attacked-dataset workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    events = cyber_df.copy()
    protocol_ports = {"tls": 443, "https": 443, "modbus": 502, "dnp3": 20000, "mqtt": 1883}
    reference_ts = pd.to_datetime(events["timestamp_utc"], utc=True).min() if not events.empty else pd.Timestamp("2025-01-01T00:00:00Z")
    next_id = int(events["event_id"].max()) + 1 if not events.empty else 1
    injected: list[dict[str, object]] = []
    for scenario in scenarios:
        start = pd.Timestamp(scenario["start_time_utc"]).tz_convert("UTC")
        destination = _asset_endpoint(str(scenario.get("target_asset")))
        injected.append(
            _event(
                event_id=next_id,
                timestamp=start - pd.Timedelta(seconds=15),
                event_type="authentication",
                attack_flag=1,
                actor="intruder",
                user_id="unknown_user",
                role="unknown",
                source_host="compromised-vpn",
                source_ip="172.16.99.10",
                source_port=54432,
                destination_host="operator-hmi",
                destination_ip="10.10.0.30",
                destination_port=443,
                auth_method="password",
                auth_result="fail",
                lockout_status=False,
                protocol="tls",
                latency_seconds=float(rng.integers(20, 80)) / 1000.0,
                clock_offset_seconds=_clock_offset_seconds("compromised-vpn", start, reference_ts),
                jitter_seconds=float(rng.integers(4, 20)) / 1000.0,
                bytes_in=int(rng.integers(180, 600)),
                bytes_out=int(rng.integers(180, 600)),
                packet_count=int(rng.integers(2, 8)),
                flow_id=f"flow-attack-auth-{scenario['scenario_id']}",
                result="fail",
                notes=f"Precursor to {scenario['scenario_id']}",
            )
        )
        next_id += 1
        injected.append(
            _event(
                event_id=next_id,
                timestamp=start,
                event_type="attack",
                attack_flag=1,
                actor="intruder",
                source_host="compromised-vpn",
                source_ip="172.16.99.10",
                source_port=54432,
                destination_host=destination[0],
                destination_ip=destination[1],
                destination_port=protocol_ports.get(str(scenario.get("protocol", "unknown")), None),
                protocol=str(scenario.get("protocol", "unknown")),
                operation="write" if str(scenario.get("protocol")) != "mqtt" else "publish",
                command_type=str(scenario.get("injection_type")),
                resource=str(scenario.get("target_signal")),
                protocol_resource_type="topic" if str(scenario.get("protocol")) == "mqtt" else "object" if str(scenario.get("protocol")) == "dnp3" else "register",
                attack_family=str(scenario.get("category")),
                severity=str(scenario.get("severity")),
                latency_seconds=float(rng.integers(20, 120)) / 1000.0,
                clock_offset_seconds=_clock_offset_seconds("compromised-vpn", start, reference_ts),
                flow_id=f"flow-attack-{scenario['scenario_id']}",
                packet_count=int(rng.integers(2, 10)),
                jitter_seconds=float(rng.integers(4, 16)) / 1000.0,
                bytes_in=int(rng.integers(160, 720)),
                bytes_out=int(rng.integers(160, 720)),
                result="attack",
                campaign_id=f"campaign-{scenario['scenario_id']}",
                notes=str(scenario.get("expected_effect")),
            )
        )
        next_id += 1
    if injected:
        events = pd.concat([events, pd.DataFrame(injected)], ignore_index=True, sort=False)
    return events.sort_values("timestamp_utc").reset_index(drop=True)


def _clock_offset_seconds(host: str | None, timestamp: pd.Timestamp, reference_ts: pd.Timestamp) -> float:
    identity = host or "unknown-host"
    base = (sum(ord(char) for char in identity) % 17) - 8
    elapsed_seconds = max((pd.Timestamp(timestamp).tz_convert("UTC") - pd.Timestamp(reference_ts).tz_convert("UTC")).total_seconds(), 0.0)
    drift = elapsed_seconds * (8.0 + (len(identity) % 7)) * 1e-3
    return float(base + drift) / 1000.0


def _asset_endpoint(asset: str) -> tuple[str, str]:
    host = f"field-{asset}"
    tail = 50 + (sum(ord(char) for char in asset) % 150)
    return host, f"10.10.1.{tail}"


def _event(
    event_id: int,
    timestamp: pd.Timestamp,
    event_type: str,
    actor: str | None = None,
    attack_flag: int = 0,
    user_id: str | None = None,
    service_id: str | None = None,
    role: str | None = None,
    source_host: str | None = None,
    source_ip: str | None = None,
    destination_host: str | None = None,
    destination_ip: str | None = None,
    source_port: int | None = None,
    destination_port: int | None = None,
    auth_method: str | None = None,
    auth_result: str | None = None,
    lockout_status: bool | None = None,
    mfa_used: bool | None = None,
    protocol: str | None = None,
    operation: str | None = None,
    resource: str | None = None,
    protocol_resource_type: str | None = None,
    command_type: str | None = None,
    requested_value: str | None = None,
    command_result: str | None = None,
    observable_signal: str | None = None,
    observable_value: float | None = None,
    response_code: str | None = None,
    latency_seconds: float | None = None,
    clock_offset_seconds: float | None = None,
    jitter_seconds: float | None = None,
    bytes_in: int | None = None,
    bytes_out: int | None = None,
    packet_count: int | None = None,
    flow_id: str | None = None,
    result: str | None = None,
    attack_family: str | None = None,
    severity: str | None = None,
    error_code: str | None = None,
    campaign_id: str | None = None,
    notes: str | None = None,
) -> dict[str, object]:
    ts = pd.Timestamp(timestamp).tz_convert("UTC")
    ingest = ts + pd.to_timedelta(latency_seconds or 0.0, unit="s")
    return {
        "event_id": event_id,
        "timestamp_utc": ts,
        "ingest_timestamp_utc": ingest,
        "event_type": event_type,
        "attack_flag": attack_flag,
        "attack_family": attack_family,
        "severity": severity,
        "actor": actor,
        "user_id": user_id,
        "service_id": service_id,
        "role": role,
        "source_host": source_host,
        "source_ip": source_ip,
        "source_port": source_port,
        "destination_host": destination_host,
        "destination_ip": destination_ip,
        "destination_port": destination_port,
        "auth_method": auth_method,
        "auth_result": auth_result,
        "lockout_status": lockout_status,
        "mfa_used": mfa_used,
        "protocol": protocol,
        "operation": operation,
        "resource": resource,
        "protocol_resource_type": protocol_resource_type,
        "command_type": command_type,
        "requested_value": requested_value,
        "command_result": command_result,
        "observable_signal": observable_signal,
        "observable_value": observable_value,
        "response_code": response_code,
        "latency_seconds": latency_seconds,
        "clock_offset_seconds": clock_offset_seconds,
        "jitter_seconds": jitter_seconds,
        "bytes_in": bytes_in,
        "bytes_out": bytes_out,
        "packet_count": packet_count,
        "flow_id": flow_id,
        "result": result,
        "error_code": error_code,
        "campaign_id": campaign_id,
        "notes": notes,
    }
