"""Offline deployment-runtime support for DERGuardian.

This module implements export deployment report logic used by the workstation deployment
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

from deployment_runtime.latency_budget import build_latency_budget_table
from deployment_runtime.runtime_common import load_runtime_config, resolve_repo_path, runtime_output_paths


def _to_markdown(frame: pd.DataFrame, limit: int | None = None) -> str:
    if frame.empty:
        return "| empty |\n|---|\n"
    preview = frame.head(limit) if limit is not None else frame
    header = "| " + " | ".join(str(column) for column in preview.columns) + " |"
    divider = "|" + "|".join(["---"] * len(preview.columns)) + "|"
    rows = [header, divider]
    for _, row in preview.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in preview.columns) + " |")
    return "\n".join(rows)


def export_deployment_report(
    *,
    config_path: str | Path | None = None,
    edge_alert_log_path: str | Path,
    inference_trace_path: str | Path,
    gateway_audit_log_path: str | Path,
    control_center_alert_summary_path: str | Path,
    central_alert_table_path: str | Path,
    control_ingest_log_path: str | Path,
    demo_summary_path: str | Path | None = None,
) -> dict[str, str]:
    """Handle export deployment report within the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    config = load_runtime_config(config_path)
    runtime_paths = runtime_output_paths(config)
    report_root = runtime_paths["reports_root"]
    report_root.mkdir(parents=True, exist_ok=True)

    edge_alerts = pd.read_csv(resolve_repo_path(edge_alert_log_path)) if Path(resolve_repo_path(edge_alert_log_path)).exists() else pd.DataFrame()
    inference_trace = pd.read_csv(resolve_repo_path(inference_trace_path)) if Path(resolve_repo_path(inference_trace_path)).exists() else pd.DataFrame()
    gateway_log = pd.read_csv(resolve_repo_path(gateway_audit_log_path)) if Path(resolve_repo_path(gateway_audit_log_path)).exists() else pd.DataFrame()
    control_summary = pd.read_csv(resolve_repo_path(control_center_alert_summary_path)) if Path(resolve_repo_path(control_center_alert_summary_path)).exists() else pd.DataFrame()
    central_alerts = pd.read_csv(resolve_repo_path(central_alert_table_path)) if Path(resolve_repo_path(central_alert_table_path)).exists() else pd.DataFrame()
    control_ingest = pd.read_csv(resolve_repo_path(control_ingest_log_path)) if Path(resolve_repo_path(control_ingest_log_path)).exists() else pd.DataFrame()
    demo_summary = json.loads(resolve_repo_path(demo_summary_path).read_text(encoding="utf-8")) if demo_summary_path and Path(resolve_repo_path(demo_summary_path)).exists() else {}

    edge_export = report_root / "edge_alerts.csv"
    gateway_export = report_root / "gateway_alert_log.csv"
    control_export = report_root / "control_center_alert_summary.csv"
    latency_export = report_root / "latency_budget_table.csv"
    report_path = report_root / "deployment_runtime_report.md"

    edge_alerts.to_csv(edge_export, index=False)
    gateway_log.to_csv(gateway_export, index=False)
    control_summary.to_csv(control_export, index=False)
    latency_budget = build_latency_budget_table(
        edge_trace_path=inference_trace_path,
        gateway_audit_path=gateway_audit_log_path,
        control_ingest_path=control_ingest_log_path,
    )
    latency_budget.to_csv(latency_export, index=False)

    lines = ["# Deployment Runtime Report", ""]
    lines.append("## Implemented Prototype Scope")
    lines.append("- Implemented: edge replay ingestion, rolling window emission, threshold-baseline inference, optional zero-day autoencoder inference, local alert packet generation, local context buffering, gateway fusion/deduplication, control-center alert collection, and optional explanation attachment.")
    lines.append("- Still conceptual: production cloud orchestration, live OT authentication boundaries, cryptographic transport hardening, and automatic control actions.")
    lines.append("")
    lines.append("## Deployment Policy")
    lines.append("- Edge default model: `threshold_baseline` because it is the strongest operational same-scenario detector and is lightweight to score.")
    lines.append("- Edge optional secondary model: `autoencoder` because Phase 3 zero-day results identify the autoencoder family as the strongest current zero-day detector.")
    lines.append("- Gateway and control center are aggregation/reporting layers only; explanation remains upstream and operator-facing.")
    lines.append("")
    lines.append("## Demo Run Summary")
    if demo_summary:
        for key in ["dataset_kind", "dataset_label", "model_mode", "selected_scenario_id", "start_time_utc", "end_time_utc"]:
            if key in demo_summary:
                lines.append(f"- {key}: `{demo_summary[key]}`")
        if "edge_alert_count" in demo_summary:
            lines.append(f"- edge_alert_count: `{demo_summary['edge_alert_count']}`")
        if "gateway_forwarded_count" in demo_summary:
            lines.append(f"- gateway_forwarded_count: `{demo_summary['gateway_forwarded_count']}`")
        if "control_center_forwarded_count" in demo_summary:
            lines.append(f"- control_center_forwarded_count: `{demo_summary['control_center_forwarded_count']}`")
    else:
        lines.append("- No demo summary manifest was provided.")
    lines.append("")
    lines.append("## Edge Alerts")
    lines.append(_to_markdown(edge_alerts, limit=10))
    lines.append("")
    lines.append("## Gateway Log")
    lines.append(_to_markdown(gateway_log, limit=10))
    lines.append("")
    lines.append("## Control Center Summary")
    lines.append(_to_markdown(control_summary))
    lines.append("")
    lines.append("## Central Alert Table")
    lines.append(_to_markdown(central_alerts, limit=10))
    lines.append("")
    lines.append("## Latency Budget")
    lines.append(_to_markdown(latency_budget))
    lines.append("")
    lines.append("## Current Limitations")
    lines.append("- Clean-reference windows are currently replayed from local project artifacts rather than a live historian profile service.")
    lines.append("- The explanation hook is replay-grounded and best-effort: it prefers matching upstream prediction artifacts and otherwise falls back to a lighter bridge packet into the existing explanation logic.")
    lines.append("- Gateway fusion is intentionally lightweight and intended for thesis/paper support rather than production SOC workflow enforcement.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "deployment_runtime_report": str(report_path),
        "edge_alerts_csv": str(edge_export),
        "gateway_alert_log_csv": str(gateway_export),
        "control_center_alert_summary_csv": str(control_export),
        "latency_budget_table_csv": str(latency_export),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build parser for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Export deployment runtime report artifacts.")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config_runtime.json"))
    parser.add_argument("--edge-alert-log", required=True)
    parser.add_argument("--inference-trace", required=True)
    parser.add_argument("--gateway-audit-log", required=True)
    parser.add_argument("--control-center-summary", required=True)
    parser.add_argument("--central-alert-table", required=True)
    parser.add_argument("--control-ingest-log", required=True)
    parser.add_argument("--demo-summary")
    return parser


def main() -> None:
    """Run the command-line entrypoint for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = build_parser()
    args = parser.parse_args()
    exported = export_deployment_report(
        config_path=args.config,
        edge_alert_log_path=args.edge_alert_log,
        inference_trace_path=args.inference_trace,
        gateway_audit_log_path=args.gateway_audit_log,
        control_center_alert_summary_path=args.control_center_summary,
        central_alert_table_path=args.central_alert_table,
        control_ingest_log_path=args.control_ingest_log,
        demo_summary_path=args.demo_summary,
    )
    print(json.dumps(exported, indent=2))


if __name__ == "__main__":
    main()
