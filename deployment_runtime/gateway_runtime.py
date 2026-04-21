"""Offline deployment-runtime support for DERGuardian.

This module implements gateway runtime logic used by the workstation deployment
benchmark and runtime demonstration code. It describes lightweight offline
behavior only; it is not evidence of field edge deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import json
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment_runtime.runtime_common import (
    ensure_dir,
    load_runtime_config,
    max_severity,
    normalize_timestamp,
    runtime_output_paths,
    timestamp_to_z,
    write_json,
)


@dataclass(slots=True)
class GatewaySummary:
    """Structured object used by the offline deployment-runtime workflow."""

    gateway_id: str
    forwarded_packet_count: int
    forwarded_packet_paths: list[str]
    gateway_audit_log_path: str


class GatewayRuntime:
    """Structured object used by the offline deployment-runtime workflow."""

    def __init__(self, config_path: str | Path | None = None, gateway_id: str | None = None) -> None:
        self.config = load_runtime_config(config_path)
        self.gateway_id = gateway_id or str(self.config.get("gateway_id", "site-gateway-demo"))
        self.runtime_paths = runtime_output_paths(self.config)
        self.gateway_root = ensure_dir(self.runtime_paths["gateway_root"] / self.gateway_id)
        self.forwarded_dir = ensure_dir(self.gateway_root / "forwarded_packets")
        self.site_metadata = dict(self.config.get("gateway", {}).get("site_metadata", {}))
        self.fusion_window_seconds = int(self.config.get("gateway", {}).get("fusion_window_seconds", 180))

    def ingest_packet_files(self, packet_paths: list[str | Path]) -> GatewaySummary:
        packets = [json.loads(Path(path).read_text(encoding="utf-8")) for path in packet_paths]
        return self.ingest_packets(packets)

    def ingest_packets(self, packets: list[dict[str, Any]]) -> GatewaySummary:
        ordered_packets = sorted(packets, key=lambda item: normalize_timestamp(item["window_start_utc"]))
        audit_rows: list[dict[str, Any]] = []
        forwarded_packets: list[dict[str, Any]] = []
        for packet in ordered_packets:
            started = time.perf_counter()
            action = "forwarded"
            target = self._find_matching_forward_packet(forwarded_packets, packet)
            if target is None:
                forwarded_packet = self._build_forwarded_packet(packet)
                forwarded_packets.append(forwarded_packet)
            else:
                if packet["model_name"] in target["supporting_models"]:
                    action = "deduplicated"
                    target["repeat_count"] = int(target.get("repeat_count", 1)) + 1
                    target["related_alert_ids"].append(packet["alert_id"])
                    target["gateway_notes"].append(f"Suppressed repeated {packet['model_name']} alert at {packet['window_start_utc']}.")
                else:
                    action = "fused"
                    target["supporting_models"].append(packet["model_name"])
                    target["supporting_profiles"].append(packet["model_profile"])
                    target["related_alert_ids"].append(packet["alert_id"])
                    target["repeat_count"] = int(target.get("repeat_count", 1)) + 1
                    target["severity"] = max_severity([target["severity"], packet["severity"]])
                    target["gateway_notes"].append(f"Fused {packet['model_name']} support into existing site incident.")
                    score_by_model = dict(target.get("score_by_model", {}))
                    score_by_model[packet["model_name"]] = round(float(packet["score"]), 6)
                    target["score_by_model"] = score_by_model
            process_ms = (time.perf_counter() - started) * 1000.0
            audit_rows.append(
                {
                    "gateway_id": self.gateway_id,
                    "action": action,
                    "alert_id": packet["alert_id"],
                    "site_id": packet["site_id"],
                    "model_name": packet["model_name"],
                    "model_profile": packet["model_profile"],
                    "severity": packet["severity"],
                    "window_start_utc": packet["window_start_utc"],
                    "window_end_utc": packet["window_end_utc"],
                    "processing_ms": round(process_ms, 6),
                }
            )
        forwarded_packet_paths = []
        for packet in forwarded_packets:
            output_path = self.forwarded_dir / f"{packet['gateway_packet_id']}.json"
            write_json(packet, output_path)
            forwarded_packet_paths.append(str(output_path))
        audit_path = self.gateway_root / "gateway_audit_log.csv"
        pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
        return GatewaySummary(
            gateway_id=self.gateway_id,
            forwarded_packet_count=len(forwarded_packets),
            forwarded_packet_paths=forwarded_packet_paths,
            gateway_audit_log_path=str(audit_path),
        )

    def _build_forwarded_packet(self, packet: dict[str, Any]) -> dict[str, Any]:
        return {
            **packet,
            "gateway_packet_id": f"{self.gateway_id}-{packet['alert_id']}",
            "gateway_id": self.gateway_id,
            "forwarded_at_utc": timestamp_to_z(packet["timestamp_utc"]),
            "site_metadata": self.site_metadata,
            "supporting_models": [packet["model_name"]],
            "supporting_profiles": [packet["model_profile"]],
            "score_by_model": {packet["model_name"]: round(float(packet["score"]), 6)},
            "related_alert_ids": [packet["alert_id"]],
            "repeat_count": 1,
            "gateway_notes": ["Forwarded from edge runtime."],
        }

    def _find_matching_forward_packet(
        self,
        forwarded_packets: list[dict[str, Any]],
        packet: dict[str, Any],
    ) -> dict[str, Any] | None:
        packet_start = normalize_timestamp(packet["window_start_utc"])
        packet_assets = set(packet.get("affected_assets", []))
        for candidate in forwarded_packets:
            candidate_start = normalize_timestamp(candidate["window_start_utc"])
            if candidate.get("site_id") != packet.get("site_id"):
                continue
            delta_seconds = abs((packet_start - candidate_start).total_seconds())
            if delta_seconds > self.fusion_window_seconds:
                continue
            candidate_assets = set(candidate.get("affected_assets", []))
            if packet_assets and candidate_assets and packet_assets.isdisjoint(candidate_assets):
                continue
            return candidate
        return None


def build_parser() -> argparse.ArgumentParser:
    """Build parser for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Fuse and forward edge alert packets through the lightweight gateway runtime.")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config_runtime.json"))
    parser.add_argument("--edge-alert-dir", required=True)
    parser.add_argument("--gateway-id")
    return parser


def main() -> None:
    """Run the command-line entrypoint for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = build_parser()
    args = parser.parse_args()
    packet_paths = sorted(str(path) for path in Path(args.edge_alert_dir).glob("*.json"))
    runtime = GatewayRuntime(config_path=args.config, gateway_id=args.gateway_id)
    summary = runtime.ingest_packet_files(packet_paths)
    print(
        pd.Series(
            {
                "gateway_id": summary.gateway_id,
                "forwarded_packet_count": summary.forwarded_packet_count,
                "gateway_audit_log_path": summary.gateway_audit_log_path,
            }
        ).to_json()
    )


if __name__ == "__main__":
    main()
