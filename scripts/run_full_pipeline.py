"""Release-facing orchestration entrypoint for DERGuardian.

This script prints or runs the main Phase 1, Phase 2, and Phase 3 commands in a
single documented order. It is intentionally conservative: existing completion
markers are respected by default so frozen benchmark artifacts are not rewritten
during a normal release smoke run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PipelineStage:
    """One reproducible pipeline command plus the files that prove completion."""

    name: str
    description: str
    command: list[str]
    completion_markers: tuple[str, ...]


def project_path(path: str) -> Path:
    """Resolve a repository-relative path without baking in machine paths."""

    return ROOT / path


STAGES: tuple[PipelineStage, ...] = (
    PipelineStage(
        name="phase1_clean_data",
        description="Generate clean OpenDSS DER truth, measured, cyber, schedule, and environment artifacts.",
        command=[sys.executable, "-m", "phase1.build_clean_dataset", "--config", "configs/pipeline_config.yaml"],
        completion_markers=(
            "outputs/clean/truth_physical_timeseries.parquet",
            "outputs/clean/measured_physical_timeseries.parquet",
            "outputs/clean/cyber_events.parquet",
        ),
    ),
    PipelineStage(
        name="phase1_validate_clean",
        description="Validate clean physical and cyber layers.",
        command=[
            sys.executable,
            "-m",
            "phase1.validate_clean_dataset",
            "--truth",
            "outputs/clean/truth_physical_timeseries.parquet",
            "--measured",
            "outputs/clean/measured_physical_timeseries.parquet",
            "--cyber",
            "outputs/clean/cyber_events.parquet",
            "--output-json",
            "outputs/clean/clean_validation.json",
            "--output-md",
            "outputs/clean/clean_validation.md",
        ],
        completion_markers=("outputs/clean/clean_validation.json", "outputs/clean/clean_validation.md"),
    ),
    PipelineStage(
        name="phase2_validate_scenarios",
        description="Validate canonical structured attack scenarios against schema and clean-data bounds.",
        command=[
            sys.executable,
            "-m",
            "phase2.validate_scenarios",
            "--scenarios",
            "phase2/research_attack_scenarios.json",
            "--output",
            "outputs/attacked/scenario_validation.json",
        ],
        completion_markers=("outputs/attacked/scenario_validation.json",),
    ),
    PipelineStage(
        name="phase2_generate_attacked",
        description="Generate attacked physical, measured, cyber, labels, manifests, and attacked windows.",
        command=[
            sys.executable,
            "-m",
            "phase2.generate_attacked_dataset",
            "--scenarios",
            "phase2/research_attack_scenarios.json",
        ],
        completion_markers=(
            "outputs/attacked/measured_physical_timeseries.parquet",
            "outputs/attacked/cyber_events.parquet",
            "outputs/attacked/attack_labels.parquet",
        ),
    ),
    PipelineStage(
        name="phase2_validate_attacked",
        description="Validate attacked layers against baseline and labels.",
        command=[
            sys.executable,
            "-m",
            "phase2.validate_attacked_dataset",
            "--attacked-truth",
            "outputs/attacked/truth_physical_timeseries.parquet",
            "--attacked-measured",
            "outputs/attacked/measured_physical_timeseries.parquet",
            "--cyber",
            "outputs/attacked/cyber_events.parquet",
            "--labels",
            "outputs/attacked/attack_labels.parquet",
            "--baseline-truth",
            "outputs/clean/truth_physical_timeseries.parquet",
            "--baseline-measured",
            "outputs/clean/measured_physical_timeseries.parquet",
            "--windows",
            "outputs/windows/merged_windows_attacked.parquet",
            "--output-json",
            "outputs/attacked/attacked_validation.json",
            "--output-md",
            "outputs/attacked/attacked_validation.md",
        ],
        completion_markers=("outputs/attacked/attacked_validation.json", "outputs/attacked/attacked_validation.md"),
    ),
    PipelineStage(
        name="phase1_window_model_benchmark",
        description="Build residual windows and run the frozen-style window/model benchmark without changing published conclusions.",
        command=[sys.executable, "scripts/run_window_size_study.py", "--window-sizes", "5,10,60,300", "--skip-existing"],
        completion_markers=("outputs/window_size_study/final_window_comparison.csv",),
    ),
    PipelineStage(
        name="phase3_detector_contexts",
        description="Run detector-side coverage, heldout synthetic zero-day-like, and context-comparison reporting.",
        command=[sys.executable, "scripts/run_detector_window_zero_day_audit.py"],
        completion_markers=("artifacts/zero_day_like/zero_day_model_window_results_full.csv",),
    ),
    PipelineStage(
        name="phase3_xai_reports",
        description="Rebuild XAI validation and conservative final report layer.",
        command=[sys.executable, "scripts/build_xai_and_final_reports.py"],
        completion_markers=("docs/reports/xai_final_validation_report.md",),
    ),
    PipelineStage(
        name="phase3_deployment_benchmark",
        description="Run offline lightweight deployment benchmark on available workstation hardware.",
        command=[sys.executable, "scripts/run_deployment_benchmark.py"],
        completion_markers=("artifacts/deployment/deployment_benchmark_results.csv",),
    ),
    PipelineStage(
        name="publication_reports",
        description="Verify the clean publication layout and final conservative release checklist.",
        command=[sys.executable, "scripts/final_perfection_verify.py"],
        completion_markers=("FINAL_PUBLISHABLE_REPO_DECISION.md", "docs/reports/internal_audits/FINAL_PERFECTION_CHECKLIST.md"),
    ),
)


def markers_exist(stage: PipelineStage) -> bool:
    """Return true when all completion markers exist and are non-empty."""

    return all(project_path(marker).exists() and project_path(marker).stat().st_size > 0 for marker in stage.completion_markers)


def run_stage(stage: PipelineStage, *, dry_run: bool, force: bool) -> None:
    """Print and optionally execute one stage while preserving existing outputs."""

    status = "ready"
    if markers_exist(stage) and not force:
        status = "skip-existing"
    print(f"[{status}] {stage.name}: {stage.description}")
    print("  command:", " ".join(stage.command))
    if status == "skip-existing" or dry_run:
        return
    subprocess.run(stage.command, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    """Parse release-runner options for dry runs, forced reruns, or one stage."""

    parser = argparse.ArgumentParser(description="Run the DERGuardian pipeline in phase order.")
    parser.add_argument("--stage", choices=[stage.name for stage in STAGES], default=None, help="Run only one stage.")
    parser.add_argument("--force", action="store_true", help="Run stages even if completion markers already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing them.")
    return parser.parse_args()


def main() -> None:
    """Run the selected pipeline stages in the documented phase order."""

    args = parse_args()
    selected = [stage for stage in STAGES if args.stage in {None, stage.name}]
    print("DERGuardian full pipeline orchestrator")
    print("Canonical benchmark outputs are not overwritten unless --force is used with scripts that regenerate them.")
    for stage in selected:
        run_stage(stage, dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
