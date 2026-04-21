from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from phase3.experiment_utils import to_markdown, write_phase3_report


def build_zero_day_report(artifact_root: Path) -> Path:
    summary_csv = artifact_root / "zero_day_model_summary.csv"
    per_scenario_csv = artifact_root / "zero_day_per_scenario_metrics.csv"
    latency_csv = artifact_root / "zero_day_latency_analysis.csv"
    manifest_path = artifact_root / "zero_day_run_manifest.json"

    summary_df = pd.read_csv(summary_csv) if summary_csv.exists() else pd.DataFrame()
    per_scenario_df = pd.read_csv(per_scenario_csv) if per_scenario_csv.exists() else pd.DataFrame()
    latency_df = pd.read_csv(latency_csv) if latency_csv.exists() else pd.DataFrame()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    sections: list[tuple[str, str]] = []
    sections.append(
        (
            "Protocol",
            "\n".join(
                [
                    f"- execution_mode: `{manifest.get('execution_mode', 'unknown')}`",
                    f"- holdout_scenarios: `{manifest.get('holdout_scenarios', [])}`",
                    f"- models: `{manifest.get('models', [])}`",
                    f"- notes: `{manifest.get('notes', 'not provided')}`",
                ]
            ),
        )
    )
    sections.append(("Aggregate Model Summary", to_markdown(summary_df)))
    sections.append(("Per-Held-Out Scenario Metrics", to_markdown(per_scenario_df)))
    sections.append(("Latency Analysis", to_markdown(latency_df)))
    return write_phase3_report(
        artifact_root / "zero_day_report.md",
        "# Phase 3 Zero-Day Report",
        sections,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Phase 3 zero-day markdown report from generated artifacts.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--artifact-root", default="")
    args = parser.parse_args()

    root = Path(args.project_root)
    artifact_root = Path(args.artifact_root) if args.artifact_root else root / "outputs" / "reports" / "phase3_zero_day_artifacts"
    report_path = build_zero_day_report(artifact_root)
    print(f"Zero-day report written to {report_path}")


if __name__ == "__main__":
    main()

