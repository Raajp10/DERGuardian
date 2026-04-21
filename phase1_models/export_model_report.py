"""Phase 1 detector training and evaluation support for DERGuardian.

This module implements export model report logic for residual-window model training,
inference, packaging, metrics, or reporting. It supports the frozen benchmark
path and related audits while keeping benchmark selection separate from replay,
heldout synthetic zero-day-like, and extension contexts.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from phase1_models.model_utils import CANONICAL_ARTIFACT_ROOT, LEGACY_ARTIFACT_ROOT, apply_model_display_names


def main() -> None:
    """Run the command-line entrypoint for the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Export a concise markdown report for phase-1 model artifacts.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--run-mode", default="full", choices=["full", "smoke", "legacy-smoke"])
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    root = Path(args.project_root)
    run_mode = "legacy-smoke" if args.run_mode == "smoke" else args.run_mode
    if run_mode == "full":
        artifact_root = root / "outputs" / "reports" / CANONICAL_ARTIFACT_ROOT
        output_path = Path(args.output) if args.output else artifact_root / "model_report_export.md"
        sections = [
            ("Model Summary", artifact_root / "model_summary_table_full.csv"),
            ("Per-Scenario Metrics", artifact_root / "per_scenario_metrics.csv"),
            ("Threshold Sweep", artifact_root / "threshold_sweep_table.csv"),
            ("Feature Separability", artifact_root / "feature_separability_table.csv"),
            ("Detection Latency", artifact_root / "detection_latency_analysis_full.csv"),
            ("Class Balance", artifact_root / "class_balance_table.csv"),
            ("Window Sweep", artifact_root / "window_sweep_table.csv"),
            ("Fusion Ablation", artifact_root / "fusion_ablation_table.csv"),
        ]
        title = "# Phase 1 Exported Full-Run Summary"
    else:
        artifact_root = root / "outputs" / "reports" / LEGACY_ARTIFACT_ROOT
        output_path = Path(args.output) if args.output else artifact_root / "model_report.md"
        sections = [
            ("Model Summary", artifact_root / "model_summary_table.csv"),
            ("Inference Cost", artifact_root / "inference_cost_table.csv"),
            ("Ablation Configuration", artifact_root / "ablation_table.csv"),
            ("Detection Latency", artifact_root / "detection_latency_analysis.csv"),
        ]
        title = "# Phase 1 Legacy Smoke Model Report"

    lines = [title, ""]
    if run_mode == "full":
        lines.append("This exported summary is a companion report.")
        lines.append("The canonical `model_report_full.md` path is owned by `phase1_models/run_full_evaluation.py`.")
        lines.append("")
    for section_title, path in sections:
        lines.append(f"## {section_title}")
        if section_title == "Detection Latency":
            lines.append("Latency is anchored to detector emission time when present; otherwise offline windowed detectors use `window_end_utc`.")
            lines.append("")
        if section_title == "Fusion Ablation":
            lines.append("Fusion is optional and combines detector scores with context reasoning confidence and token-model scores.")
            lines.append("")
        if path.exists():
            frame = pd.read_csv(path)
            frame = apply_model_display_names(frame)
            lines.append(_to_markdown(frame))
        else:
            lines.append("Artifact not available.")
        lines.append("")

    if run_mode == "full":
        context_jsonl = artifact_root / "phase1_context_summaries.jsonl"
        reasoning_jsonl = artifact_root / "phase1_context_reasoning_outputs.jsonl"
        lines.append("## Context Builder")
        if context_jsonl.exists():
            lines.append(f"- Context summaries: `{context_jsonl}`")
            lines.append(f"- Context table: `{artifact_root / 'phase1_context_summaries.parquet'}`")
        else:
            lines.append("Artifact not available.")
        lines.append("")
        lines.append("## Context-Aware Reasoning")
        if reasoning_jsonl.exists():
            lines.append(f"- Reasoning outputs: `{reasoning_jsonl}`")
            lines.append(f"- Reasoning table: `{artifact_root / 'phase1_context_reasoning_outputs.parquet'}`")
        else:
            lines.append("Artifact not available.")
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "| empty |\n|---|\n"
    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    divider = "|" + "|".join(["---"] * len(frame.columns)) + "|"
    rows = [header, divider]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in frame.columns) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
