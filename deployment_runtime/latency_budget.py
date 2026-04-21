"""Offline deployment-runtime support for DERGuardian.

This module implements latency budget logic used by the workstation deployment
benchmark and runtime demonstration code. It describes lightweight offline
behavior only; it is not evidence of field edge deployment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment_runtime.runtime_common import resolve_repo_path


def summarize_latency_series(stage: str, values: pd.Series, note: str) -> dict[str, Any]:
    """Handle summarize latency series within the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    cleaned = pd.to_numeric(values, errors="coerce").dropna()
    if cleaned.empty:
        return {
            "stage": stage,
            "sample_count": 0,
            "mean_ms": np.nan,
            "p50_ms": np.nan,
            "p95_ms": np.nan,
            "max_ms": np.nan,
            "note": note,
        }
    return {
        "stage": stage,
        "sample_count": int(len(cleaned)),
        "mean_ms": round(float(cleaned.mean()), 6),
        "p50_ms": round(float(cleaned.quantile(0.50)), 6),
        "p95_ms": round(float(cleaned.quantile(0.95)), 6),
        "max_ms": round(float(cleaned.max()), 6),
        "note": note,
    }


def build_latency_budget_table(
    edge_trace_path: str | Path,
    gateway_audit_path: str | Path,
    control_ingest_path: str | Path,
) -> pd.DataFrame:
    """Build latency budget table for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    edge_trace_resolved = resolve_repo_path(edge_trace_path)
    gateway_audit_resolved = resolve_repo_path(gateway_audit_path)
    control_ingest_resolved = resolve_repo_path(control_ingest_path)
    edge_trace = pd.read_csv(edge_trace_resolved) if edge_trace_resolved.exists() else pd.DataFrame()
    gateway_log = pd.read_csv(gateway_audit_resolved) if gateway_audit_resolved.exists() else pd.DataFrame()
    control_log = pd.read_csv(control_ingest_resolved) if control_ingest_resolved.exists() else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    if not edge_trace.empty:
        rows.append(summarize_latency_series("edge_window_build", edge_trace["window_build_ms"], "Rolling aggregation and window emission."))
        for model_profile, group in edge_trace.groupby("model_profile", observed=False):
            rows.append(
                summarize_latency_series(
                    f"edge_inference_{model_profile}",
                    group["inference_ms"],
                    f"Model inference latency for profile `{model_profile}`.",
                )
            )
        alert_edge = edge_trace[edge_trace["alert_emitted"] == 1]
        rows.append(summarize_latency_series("edge_packet_build", alert_edge["packet_build_ms"], "Alert packet assembly and context capture."))
        rows.append(summarize_latency_series("edge_total_alert_path", alert_edge["edge_total_ms"], "End-to-end edge path for alerting windows."))
    if not gateway_log.empty:
        rows.append(summarize_latency_series("gateway_forward", gateway_log["processing_ms"], "Gateway fusion, deduplication, and forwarding."))
    if not control_log.empty:
        rows.append(summarize_latency_series("control_center_ingest", control_log["ingest_ms"], "Control-center ingestion and optional explanation attachment."))
    budget = pd.DataFrame(rows)
    if not budget.empty:
        end_to_end_components = budget[budget["stage"].isin(["edge_total_alert_path", "gateway_forward", "control_center_ingest"])]["mean_ms"].dropna()
        if not end_to_end_components.empty:
            budget = pd.concat(
                [
                    budget,
                    pd.DataFrame(
                        [
                            {
                                "stage": "end_to_end_mean_path",
                                "sample_count": int(end_to_end_components.count()),
                                "mean_ms": round(float(end_to_end_components.sum()), 6),
                                "p50_ms": np.nan,
                                "p95_ms": np.nan,
                                "max_ms": np.nan,
                                "note": "Mean edge + gateway + control-center alert path.",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    return budget


def build_parser() -> argparse.ArgumentParser:
    """Build parser for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Build a lightweight latency budget table from runtime traces.")
    parser.add_argument("--edge-trace", required=True)
    parser.add_argument("--gateway-audit", required=True)
    parser.add_argument("--control-ingest", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    """Run the command-line entrypoint for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = build_parser()
    args = parser.parse_args()
    budget = build_latency_budget_table(args.edge_trace, args.gateway_audit, args.control_ingest)
    output_path = resolve_repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    budget.to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
