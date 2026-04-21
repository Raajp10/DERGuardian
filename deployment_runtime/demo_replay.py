"""Offline deployment-runtime support for DERGuardian.

This module implements demo replay logic used by the workstation deployment
benchmark and runtime demonstration code. It describes lightweight offline
behavior only; it is not evidence of field edge deployment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment_runtime.control_center_runtime import ControlCenterRuntime
from deployment_runtime.edge_runtime import EdgeRuntime
from deployment_runtime.export_deployment_report import export_deployment_report
from deployment_runtime.gateway_runtime import GatewayRuntime
from deployment_runtime.runtime_common import load_runtime_config, normalize_timestamp, runtime_output_paths, timestamp_to_z, write_json


DEFAULT_ATTACK_SCENARIO = "scn_unauthorized_bess48_discharge"


def resolve_model_mode(args: argparse.Namespace) -> str:
    """Handle resolve model mode within the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if args.autoencoder_only:
        return "autoencoder_only"
    if args.threshold_plus_autoencoder:
        return "threshold_plus_autoencoder"
    return "threshold_only"


def resolve_dataset_kind(args: argparse.Namespace) -> str:
    """Handle resolve dataset kind within the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if args.clean:
        return "clean"
    return "attacked"


def select_demo_window(
    config: dict[str, Any],
    dataset_kind: str,
    scenario_id: str | None,
    start_time: str | None,
    end_time: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Select demo window for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if start_time and end_time:
        return start_time, end_time, scenario_id
    demo_cfg = dict(config.get("demo", {}))
    pre_padding_seconds = int(demo_cfg.get("pre_attack_padding_seconds", 900))
    post_padding_seconds = int(demo_cfg.get("post_attack_padding_seconds", 900))
    if dataset_kind == "attacked":
        selected_scenario = scenario_id or str(demo_cfg.get("default_attacked_scenario_id", DEFAULT_ATTACK_SCENARIO))
        labels = pd.read_parquet(Path("outputs") / "attacked" / "attack_labels.parquet")
        labels["start_time_utc"] = pd.to_datetime(labels["start_time_utc"], utc=True)
        labels["end_time_utc"] = pd.to_datetime(labels["end_time_utc"], utc=True)
        matched = labels[labels["scenario_id"] == selected_scenario]
        if matched.empty:
            raise ValueError(f"Unable to find attack label for scenario `{selected_scenario}`.")
        row = matched.iloc[0]
        start = normalize_timestamp(row["start_time_utc"]) - pd.Timedelta(seconds=pre_padding_seconds)
        end = normalize_timestamp(row["end_time_utc"]) + pd.Timedelta(seconds=post_padding_seconds)
        return timestamp_to_z(start), timestamp_to_z(end), selected_scenario
    clean_default = demo_cfg.get("default_clean_window", {})
    clean_start = start_time or clean_default.get("start_time_utc") or "2025-06-01T01:45:00Z"
    clean_end = end_time or clean_default.get("end_time_utc") or "2025-06-01T02:20:00Z"
    return clean_start, clean_end, scenario_id


def dataset_paths(dataset_kind: str) -> tuple[str, str]:
    """Handle dataset paths within the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if dataset_kind == "clean":
        return (
            str(Path("outputs") / "clean" / "measured_physical_timeseries.parquet"),
            str(Path("outputs") / "clean" / "cyber_events.parquet"),
        )
    return (
        str(Path("outputs") / "attacked" / "measured_physical_timeseries.parquet"),
        str(Path("outputs") / "attacked" / "cyber_events.parquet"),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build parser for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Run an end-to-end lightweight deployment replay demo.")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config_runtime.json"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--clean", action="store_true")
    mode.add_argument("--attacked", action="store_true")
    parser.add_argument("--scenario-id")
    parser.add_argument("--start-time")
    parser.add_argument("--end-time")
    parser.add_argument("--threshold-plus-autoencoder", action="store_true")
    parser.add_argument("--autoencoder-only", action="store_true")
    return parser


def main() -> None:
    """Run the command-line entrypoint for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = build_parser()
    args = parser.parse_args()
    config = load_runtime_config(args.config)
    dataset_kind = resolve_dataset_kind(args)
    model_mode = resolve_model_mode(args)
    measured_path, cyber_path = dataset_paths(dataset_kind)
    start_time, end_time, selected_scenario = select_demo_window(
        config=config,
        dataset_kind=dataset_kind,
        scenario_id=args.scenario_id,
        start_time=args.start_time,
        end_time=args.end_time,
    )

    edge = EdgeRuntime(config_path=args.config, model_mode=model_mode)
    edge_summary = edge.run_replay(
        measured_path=measured_path,
        cyber_path=cyber_path,
        dataset_kind=dataset_kind,
        start_time_utc=start_time,
        end_time_utc=end_time,
        dataset_label=f"{dataset_kind}_demo",
    )
    if dataset_kind == "attacked" and edge_summary.edge_alert_count < 1:
        raise RuntimeError("Attacked replay did not produce any edge alerts.")

    gateway = GatewayRuntime(config_path=args.config)
    gateway_summary = gateway.ingest_packet_files(edge_summary.alert_packet_paths)
    control = ControlCenterRuntime(config_path=args.config)
    control_summary = control.ingest_packet_files(gateway_summary.forwarded_packet_paths)
    runtime_paths = runtime_output_paths(config)
    demo_summary = {
        "dataset_kind": dataset_kind,
        "dataset_label": f"{dataset_kind}_demo",
        "model_mode": model_mode,
        "selected_scenario_id": selected_scenario,
        "start_time_utc": start_time,
        "end_time_utc": end_time,
        "edge_alert_count": edge_summary.edge_alert_count,
        "gateway_forwarded_count": gateway_summary.forwarded_packet_count,
        "control_center_forwarded_count": control_summary.forwarded_packet_count,
        "control_center_explanation_count": control_summary.explanation_count,
        "edge_alert_log_path": edge_summary.edge_alert_log_path,
        "inference_trace_path": edge_summary.inference_trace_path,
        "gateway_audit_log_path": gateway_summary.gateway_audit_log_path,
        "control_center_alert_summary_path": control_summary.control_center_alert_summary_path,
    }
    demo_summary_path = write_json(demo_summary, runtime_paths["output_root"] / "demo_summary.json")
    exported = export_deployment_report(
        config_path=args.config,
        edge_alert_log_path=edge_summary.edge_alert_log_path,
        inference_trace_path=edge_summary.inference_trace_path,
        gateway_audit_log_path=gateway_summary.gateway_audit_log_path,
        control_center_alert_summary_path=control_summary.control_center_alert_summary_path,
        central_alert_table_path=control_summary.central_alert_table_path,
        control_ingest_log_path=control_summary.control_ingest_log_path,
        demo_summary_path=demo_summary_path,
    )
    demo_summary = {**demo_summary, **exported}
    demo_summary_path = write_json(demo_summary, runtime_paths["output_root"] / "demo_summary.json")
    print(json.dumps({**demo_summary, "demo_summary_path": str(demo_summary_path)}, indent=2))


if __name__ == "__main__":
    main()
