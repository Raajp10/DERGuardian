"""Offline deployment-runtime support for DERGuardian.

This module implements control center runtime logic used by the workstation deployment
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
    normalize_timestamp,
    runtime_output_paths,
    select_model_profiles,
    write_json,
)


@dataclass(slots=True)
class ControlCenterSummary:
    """Structured object used by the offline deployment-runtime workflow."""

    control_center_id: str
    forwarded_packet_count: int
    central_alert_table_path: str
    control_center_alert_summary_path: str
    control_ingest_log_path: str
    explanation_count: int


class ControlCenterRuntime:
    """Structured object used by the offline deployment-runtime workflow."""

    def __init__(self, config_path: str | Path | None = None, control_center_id: str | None = None) -> None:
        self.config = load_runtime_config(config_path)
        self.control_center_id = control_center_id or str(self.config.get("control_center_id", "control-center-demo"))
        self.runtime_paths = runtime_output_paths(self.config)
        self.control_root = ensure_dir(self.runtime_paths["control_center_root"] / self.control_center_id)
        self.explanations_dir = ensure_dir(self.control_root / "explanations")
        self.enable_explanation_adapter = bool(self.config.get("control_center", {}).get("enable_explanation_adapter", True))
        self.profile_lookup = {
            profile.runtime_name: profile
            for profile in select_model_profiles(self.config, model_mode="threshold_plus_autoencoder")
        }

    def ingest_packet_files(self, packet_paths: list[str | Path]) -> ControlCenterSummary:
        packets = [json.loads(Path(path).read_text(encoding="utf-8")) for path in packet_paths]
        return self.ingest_packets(packets)

    def ingest_packets(self, packets: list[dict[str, Any]]) -> ControlCenterSummary:
        ordered_packets = sorted(packets, key=lambda item: normalize_timestamp(item["window_start_utc"]))
        central_rows: list[dict[str, Any]] = []
        ingest_rows: list[dict[str, Any]] = []
        explanation_count = 0
        for packet in ordered_packets:
            started = time.perf_counter()
            explanation_path = self._maybe_generate_explanation(packet)
            if explanation_path is not None:
                explanation_count += 1
                packet["explanation_packet_path"] = str(explanation_path)
            ingest_ms = (time.perf_counter() - started) * 1000.0
            central_rows.append(
                {
                    "control_center_id": self.control_center_id,
                    "site_id": packet["site_id"],
                    "gateway_id": packet.get("gateway_id"),
                    "gateway_packet_id": packet.get("gateway_packet_id"),
                    "asset_id": packet.get("asset_id"),
                    "affected_assets": "|".join(packet.get("affected_assets", [])),
                    "severity": packet["severity"],
                    "model_name": packet["model_name"],
                    "model_profile": packet["model_profile"],
                    "supporting_models": "|".join(packet.get("supporting_models", [])),
                    "score": packet["score"],
                    "threshold": packet["threshold"],
                    "window_start_utc": packet["window_start_utc"],
                    "window_end_utc": packet["window_end_utc"],
                    "packet_source": packet["packet_source"],
                    "local_context_path": packet.get("local_context_path"),
                    "explanation_packet_path": packet.get("explanation_packet_path"),
                }
            )
            ingest_rows.append(
                {
                    "control_center_id": self.control_center_id,
                    "gateway_packet_id": packet.get("gateway_packet_id"),
                    "site_id": packet["site_id"],
                    "model_profile": packet["model_profile"],
                    "severity": packet["severity"],
                    "ingest_ms": round(ingest_ms, 6),
                    "explanation_generated": int(explanation_path is not None),
                }
            )
        central_alerts = pd.DataFrame(central_rows)
        summary = (
            central_alerts.groupby(["site_id", "severity", "model_profile"], observed=False)
            .size()
            .reset_index(name="alert_count")
            .sort_values(["site_id", "severity", "model_profile"])
        ) if not central_alerts.empty else pd.DataFrame(columns=["site_id", "severity", "model_profile", "alert_count"])
        central_alert_table_path = self.control_root / "central_alerts.csv"
        summary_path = self.control_root / "control_center_alert_summary.csv"
        ingest_log_path = self.control_root / "control_ingest_log.csv"
        central_alerts.to_csv(central_alert_table_path, index=False)
        summary.to_csv(summary_path, index=False)
        pd.DataFrame(ingest_rows).to_csv(ingest_log_path, index=False)
        write_json(
            {
                "control_center_id": self.control_center_id,
                "forwarded_packet_count": int(len(central_alerts)),
                "explanation_count": int(explanation_count),
                "central_alert_table_path": str(central_alert_table_path),
                "control_center_alert_summary_path": str(summary_path),
                "control_ingest_log_path": str(ingest_log_path),
            },
            self.control_root / "control_center_manifest.json",
        )
        return ControlCenterSummary(
            control_center_id=self.control_center_id,
            forwarded_packet_count=len(central_alerts),
            central_alert_table_path=str(central_alert_table_path),
            control_center_alert_summary_path=str(summary_path),
            control_ingest_log_path=str(ingest_log_path),
            explanation_count=explanation_count,
        )

    def _maybe_generate_explanation(self, packet: dict[str, Any]) -> Path | None:
        if not self.enable_explanation_adapter:
            return None
        profile = self.profile_lookup.get(str(packet.get("model_profile")))
        try:
            from phase3_explanations.build_explanation_packet import build_packet
            from phase3_explanations.classify_attack_family import classify_from_packet
            from phase3_explanations.generate_explanations_llm import grounded_draft_explanation, render_incident_summary_markdown
        except Exception:
            return None
        payload = None
        if profile is not None and profile.predictions_path and profile.results_path and profile.attacked_windows_path:
            try:
                payload = build_packet(
                    predictions_path=profile.predictions_path,
                    results_path=profile.results_path,
                    attacked_windows_path=profile.attacked_windows_path,
                    clean_windows_path=profile.reference_windows_path,
                    model_name=profile.model_name,
                    window_start=packet["window_start_utc"],
                )
            except Exception:
                payload = None
        if payload is None:
            payload = self._build_bridge_explanation_packet(packet)
            payload["candidate_families"] = classify_from_packet(payload)["candidate_families"]
        explanation = grounded_draft_explanation(payload)
        output_path = self.explanations_dir / f"{packet['alert_id']}.json"
        write_json(
            {
                "packet": payload,
                "explanation": explanation,
            },
            output_path,
        )
        markdown_path = self.explanations_dir / f"{packet['alert_id']}.md"
        markdown_path.write_text(render_incident_summary_markdown(payload, explanation), encoding="utf-8")
        return output_path

    def _build_bridge_explanation_packet(self, packet: dict[str, Any]) -> dict[str, Any]:
        physical_evidence = []
        for item in packet.get("top_features", []):
            delta = float(item.get("residual") or 0.0)
            observed = float(item.get("window_value") or 0.0)
            baseline = float(item.get("reference_value") or 0.0)
            signal = str(item.get("signal") or "")
            physical_evidence.append(
                {
                    "asset": packet.get("asset_id") or (packet.get("affected_assets") or ["feeder"])[0],
                    "signal": signal,
                    "category": self._category_for_signal(signal),
                    "aggregation": item.get("aggregation", "value"),
                    "observed_value": round(observed, 6),
                    "baseline_value": round(baseline, 6),
                    "delta": round(delta, 6),
                    "relative_change": round(delta / max(abs(observed), abs(baseline), 1.0), 6),
                    "direction": "increase" if delta >= 0 else "decrease",
                    "source_feature": item.get("feature"),
                }
            )
        return {
            "incident_id": packet["alert_id"],
            "model_name": packet["model_name"],
            "anomaly_score": float(packet["score"]),
            "threshold": float(packet["threshold"]),
            "score_margin": float(packet["score"]) - float(packet["threshold"]),
            "affected_assets": packet.get("affected_assets", []),
            "top_features": [
                {
                    "feature": item.get("feature"),
                    "description": item.get("feature"),
                }
                for item in packet.get("top_features", [])
            ],
            "physical_evidence": physical_evidence,
            "cyber_evidence": [],
            "detector_evidence": {
                "prediction_source": packet.get("packet_source"),
                "candidate_family_hints": [],
            },
            "offline_evaluation": {
                "scenario_label": None,
                "scenario_metadata": None,
            },
        }

    @staticmethod
    def _category_for_signal(signal: str) -> str:
        if signal.startswith("bess_") and "soc" in signal:
            return "soc_inconsistency"
        if signal.startswith("bess_"):
            return "bess_power_jump"
        if signal.startswith("pv_") and ("curtailment" in signal or "available_kw" in signal):
            return "curtailment_inconsistency"
        if signal.startswith("pv_"):
            return "pv_dispatch_drop"
        if signal.startswith("bus_") and "angle_deg" in signal:
            return "voltage_angle_shift"
        if signal.startswith("bus_"):
            return "voltage_profile_shift"
        if signal.startswith("feeder_"):
            return "feeder_power_shift"
        return "physical_signal_shift"


def build_parser() -> argparse.ArgumentParser:
    """Build parser for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Collect gateway packets into the lightweight control-center runtime.")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config_runtime.json"))
    parser.add_argument("--gateway-forwarded-dir", required=True)
    parser.add_argument("--control-center-id")
    return parser


def main() -> None:
    """Run the command-line entrypoint for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = build_parser()
    args = parser.parse_args()
    packet_paths = sorted(str(path) for path in Path(args.gateway_forwarded_dir).glob("*.json"))
    runtime = ControlCenterRuntime(config_path=args.config, control_center_id=args.control_center_id)
    summary = runtime.ingest_packet_files(packet_paths)
    print(
        pd.Series(
            {
                "control_center_id": summary.control_center_id,
                "forwarded_packet_count": summary.forwarded_packet_count,
                "control_center_alert_summary_path": summary.control_center_alert_summary_path,
                "explanation_count": summary.explanation_count,
            }
        ).to_json()
    )


if __name__ == "__main__":
    main()
