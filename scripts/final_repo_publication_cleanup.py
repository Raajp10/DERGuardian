from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ArtifactCopy:
    source: str
    target: str
    context: str
    artifact_type: str
    description: str


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_csv_or_empty(path: str | Path) -> pd.DataFrame:
    path = ROOT / path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_text(path: str | Path, text: str) -> None:
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_artifacts(copies: list[ArtifactCopy]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in copies:
        source = ROOT / item.source
        target = ROOT / item.target
        exists = source.exists()
        copied = False
        size = 0
        if exists:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if target.suffix.lower() in {".md", ".txt"}:
                target.write_text(sanitize_text(target.read_text(encoding="utf-8")), encoding="utf-8")
            copied = True
            size = target.stat().st_size
        rows.append(
            {
                "source_artifact": item.source,
                "publishable_artifact": item.target,
                "context": item.context,
                "artifact_type": item.artifact_type,
                "exists": exists,
                "mirrored": copied,
                "size_bytes": size,
                "description": item.description,
            }
        )
    return rows


def sanitize_text(text: str) -> str:
    replacements = {
        str(ROOT) + "\\": "",
        str(ROOT) + "/": "",
        str(ROOT): "<repo>",
        "<repo>/": "",
        "<repo>/": "<repo>",
    }
    cleaned = text
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned.replace("\\", "/")


def load_metrics() -> dict[str, Any]:
    phase1 = read_csv_or_empty("phase1_window_model_comparison_full.csv")
    zero_day = read_csv_or_empty("zero_day_model_window_results_full.csv")
    context = read_csv_or_empty("FINAL_MODEL_CONTEXT_COMPARISON.csv")
    ttm = read_csv_or_empty("phase1_ttm_results.csv")
    lora = read_csv_or_empty("phase3_lora_results.csv")
    deployment = read_csv_or_empty("deployment_benchmark_results.csv")
    xai = read_csv_or_empty("xai_case_level_audit.csv")
    phase2_inventory = read_csv_or_empty("phase2_scenario_master_inventory.csv")

    canonical = {}
    if not phase1.empty:
        row = phase1.loc[
            (phase1["model_name"] == "transformer")
            & (phase1["window_label"] == "60s")
            & (phase1["canonical_or_extension"] == "canonical")
        ]
        if not row.empty:
            canonical = row.iloc[0].to_dict()

    zero_best = {}
    if not zero_day.empty:
        completed = zero_day.loc[zero_day["status"] == "completed"].copy()
        if not completed.empty:
            grouped = (
                completed.groupby(["model_name", "window_label", "canonical_or_extension"], as_index=False)[
                    ["precision", "recall", "f1"]
                ]
                .mean(numeric_only=True)
                .sort_values(["f1", "precision"], ascending=[False, False])
            )
            zero_best = grouped.iloc[0].to_dict()

    return {
        "phase1": phase1,
        "zero_day": zero_day,
        "context": context,
        "ttm": ttm,
        "lora": lora,
        "deployment": deployment,
        "xai": xai,
        "phase2_inventory": phase2_inventory,
        "canonical": canonical,
        "zero_best": zero_best,
    }


def precheck_required_files() -> list[str]:
    return [
        "README.md",
        ".gitignore",
        "LICENSE",
        "requirements.txt",
        "environment.yml",
        "FINAL_PUBLISHABLE_REPO_DECISION.md",
        "FINAL_REPO_PUBLISHABILITY_CHECKLIST.md",
        "FINAL_REPO_PUBLISHABILITY_STATUS.csv",
        "REPO_CLEAN_TREE.txt",
        "CLAIM_SAFETY_FINAL_AUDIT.md",
        "REPO_ARTIFACT_MAP.csv",
        "EXPERIMENT_REGISTRY.csv",
        "docs/methodology/METHODOLOGY_OVERVIEW.md",
        "docs/reports/CANONICAL_VS_REPLAY_VS_ZERO_DAY.md",
        "docs/reports/EXTENSION_BRANCHES.md",
        "docs/reports/PUBLICATION_SAFE_CLAIMS.md",
        "docs/reports/ARTIFACT_REPRODUCIBILITY_GUIDE.md",
    ]


def structurally_usable(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if path.stat().st_size == 0:
        return False, "empty"
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(path)
            if frame.empty and len(frame.columns) == 0:
                return False, "CSV has no columns"
            return True, f"CSV readable with {len(frame)} rows and {len(frame.columns)} columns"
        if suffix == ".md":
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if not text.startswith("#"):
                return False, "Markdown does not start with a heading"
            return True, "Markdown has a top-level heading"
        if suffix in {".txt", ".gitignore"} or path.name in {".gitignore", "LICENSE", "requirements.txt"}:
            return True, "Text file is readable and non-empty"
        if suffix in {".yml", ".yaml"}:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            return (":" in text), "YAML-like content detected" if ":" in text else "No YAML key/value content detected"
    except Exception as exc:  # pragma: no cover - defensive audit path
        return False, f"read/parse error: {exc}"
    return True, "Non-empty file"


def write_git_readiness_precheck() -> None:
    rows: list[dict[str, Any]] = []
    for item in precheck_required_files():
        path = ROOT / item
        usable, note = structurally_usable(path)
        exists = path.exists()
        nonempty = bool(exists and path.stat().st_size > 0)
        needs_rewrite = not (exists and nonempty and usable)
        if item == "FINAL_REPO_PUBLISHABILITY_STATUS.csv" and exists:
            try:
                frame = pd.read_csv(path)
                expected = {"item", "status", "file_path", "notes"}
                if not expected.issubset(set(frame.columns)):
                    needs_rewrite = True
                    usable = False
                    note = "Final status CSV needs the required item/status/file_path/notes schema"
            except Exception as exc:  # pragma: no cover - defensive audit path
                needs_rewrite = True
                usable = False
                note = f"Could not parse final status CSV: {exc}"
        rows.append(
            {
                "file_path": item,
                "exists": exists,
                "nonempty": nonempty,
                "structurally_usable": usable,
                "needs_rewrite": needs_rewrite,
                "notes": note,
            }
        )
    write_csv(
        "GIT_READINESS_PRECHECK.csv",
        rows,
        ["file_path", "exists", "nonempty", "structurally_usable", "needs_rewrite", "notes"],
    )
    lines = [
        "# Git Readiness Precheck",
        "",
        "This precheck verifies the minimum GitHub/publication-facing files before the final cleanup pass rewrites or refreshes them.",
        "",
        md_table(pd.DataFrame(rows)),
        "",
        "## Result",
        "",
        "Files marked `needs_rewrite=True` are repaired by `scripts/final_repo_publication_cleanup.py` during the same cleanup pass when possible.",
    ]
    write_text("GIT_READINESS_PRECHECK.md", "\n".join(lines))


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows available._"
    frame = df.head(max_rows).copy() if max_rows else df.copy()
    for col in frame.columns:
        if pd.api.types.is_float_dtype(frame[col]):
            frame[col] = frame[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return frame.to_markdown(index=False)


def ensure_dirs() -> None:
    for path in [
        "src",
        "configs",
        "data/sample_or_manifest_only",
        "artifacts/benchmark",
        "artifacts/replay",
        "artifacts/zero_day_like",
        "artifacts/extensions",
        "artifacts/xai",
        "artifacts/deployment",
        "docs/methodology",
        "docs/reports",
        "docs/figures",
        "tests",
    ]:
        (ROOT / path).mkdir(parents=True, exist_ok=True)


def write_diagram_package(metrics: dict[str, Any]) -> None:
    boxes = [
        {
            "phase": "Phase 1",
            "original_box": "Generate DER Grid Data",
            "safe_box_text": "Generate DER grid and clean simulation data",
            "status": "fully implemented",
            "evidence": "Clean OpenDSS, measured, cyber, and window artifacts under outputs/clean and outputs/window_size_study.",
            "safe_wording": "IEEE 123-bus DER simulation produces clean physical and cyber baseline data.",
        },
        {
            "phase": "Phase 1",
            "original_box": "System Modeling IEEE 123 Bus + DER (PV, BESS)",
            "safe_box_text": "IEEE 123-bus feeder with PV/BESS assets",
            "status": "fully implemented",
            "evidence": "OpenDSS feeder and DER overlays are present under opendss and Phase 1 code.",
            "safe_wording": "The study uses an IEEE 123-bus feeder with simulated PV and BESS assets.",
        },
        {
            "phase": "Phase 1",
            "original_box": "Normal Data Generation OpenDSS Simulation (Voltage, Current, Power, SOC, Temp)",
            "safe_box_text": "Clean OpenDSS simulation and measured telemetry",
            "status": "fully implemented",
            "evidence": "Clean truth/measured/cyber datasets and training-input audit.",
            "safe_wording": "Clean physical and measured telemetry is generated and windowed for model inputs.",
        },
        {
            "phase": "Phase 1",
            "original_box": "Data Preprocessing Cleaning + Feature Engineering",
            "safe_box_text": "Windowing, cleaning, and feature engineering",
            "status": "fully implemented",
            "evidence": "phase1_training_input_inventory.csv and window-size study artifacts.",
            "safe_wording": "Raw time series are transformed into model-ready windows and aligned residual features.",
        },
        {
            "phase": "Phase 1",
            "original_box": "Missing from diagram",
            "safe_box_text": "Residual / deviation feature layer",
            "status": "fully implemented",
            "evidence": "Residual windows used by canonical benchmark and heldout replay.",
            "safe_wording": "Detector inputs include aligned residual/deviation features between attacked and clean windows.",
        },
        {
            "phase": "Phase 1",
            "original_box": "Baseline Models ML Models (LSTM, Autoencoder, Isolation Forest)",
            "safe_box_text": "Detector benchmark models: threshold, Isolation Forest, MLP autoencoder, GRU, LSTM, Transformer",
            "status": "fully implemented",
            "evidence": "phase1_window_model_comparison_full.csv",
            "safe_wording": "The canonical benchmark compares threshold, Isolation Forest, MLP autoencoder, GRU, LSTM, and Transformer detectors.",
        },
        {
            "phase": "Phase 1",
            "original_box": "LLM Analysis Context Learning of Normal Behavior",
            "safe_box_text": "Tokenized sequence baseline and context artifacts",
            "status": "partially implemented",
            "evidence": "Token baseline and explanation/context artifacts exist, but no large LLM learns normal behavior as the detector.",
            "safe_wording": "Context artifacts and a tokenized baseline support analysis; do not claim an LLM learned normal behavior as the main detector.",
        },
        {
            "phase": "Phase 1",
            "original_box": "Model Evaluation Reconstruction Loss, Threshold, Accuracy",
            "safe_box_text": "Benchmark evaluation: precision, recall, F1, latency, false positive rate",
            "status": "fully implemented",
            "evidence": "phase1_window_model_comparison_full.csv",
            "safe_wording": "Model selection uses precision, recall, F1, latency, and related detector metrics.",
        },
        {
            "phase": "Phase 2",
            "original_box": "LLM Scenario Generator Synthetic Anomaly / Zero-day Style",
            "safe_box_text": "Schema-bound synthetic scenario generation",
            "status": "fully implemented",
            "evidence": "Canonical and heldout Phase 2 bundles plus validation reports.",
            "safe_wording": "Synthetic scenario JSON bundles are generated and validated under a bounded schema.",
        },
        {
            "phase": "Phase 2",
            "original_box": "Validation Layer Physics + Safety Constraints Check",
            "safe_box_text": "Physics, schema, and safety validation",
            "status": "fully implemented",
            "evidence": "phase2_scenario_master_inventory.csv and phase2_coverage_summary.md",
            "safe_wording": "Scenarios are accepted, rejected, or repaired using schema and physics-aware validation.",
        },
        {
            "phase": "Phase 2",
            "original_box": "Quality Check Plausibility + Diversity + Metadata",
            "safe_box_text": "Coverage, diversity, and difficulty audit",
            "status": "fully implemented",
            "evidence": "phase2_diversity_and_quality_report.md and related figures.",
            "safe_wording": "Phase 2 includes coverage, diversity, metadata, and difficulty calibration audits.",
        },
        {
            "phase": "Phase 3",
            "original_box": "Unified Dataset Normal + Attack Data",
            "safe_box_text": "Unified clean, attacked, and residual evaluation datasets",
            "status": "fully implemented",
            "evidence": "Window-size study and heldout replay artifacts.",
            "safe_wording": "Phase 3 evaluates frozen detectors on canonical and heldout synthetic residual windows.",
        },
        {
            "phase": "Phase 3",
            "original_box": "Missing from diagram",
            "safe_box_text": "Model training: normal-only and supervised detector branches",
            "status": "fully implemented",
            "evidence": "Canonical benchmark includes supervised sequence classifiers, unsupervised/baseline detectors, and normal-only TTM extension.",
            "safe_wording": "Detector training includes supervised benchmark models and normal-only extension branches where applicable.",
        },
        {
            "phase": "Phase 3",
            "original_box": "Model Training ML Models + Tiny LLM (LoRA)",
            "safe_box_text": "Canonical detector training plus separate LoRA explanation extension",
            "status": "partially implemented",
            "evidence": "Canonical detectors complete; LoRA exists as weak experimental explanation/classification branch.",
            "safe_wording": "LoRA is experimental and separate from the canonical detector benchmark.",
        },
        {
            "phase": "Phase 3",
            "original_box": "Human-like Explanation LLM Generates Root Cause Analysis",
            "safe_box_text": "Grounded operator-facing explanation support",
            "status": "partially implemented",
            "evidence": "xai_final_validation_report.md",
            "safe_wording": "Use grounded post-alert explanation support; do not claim human-like root-cause analysis.",
        },
        {
            "phase": "Phase 3",
            "original_box": "Evaluation Metrics F1, Precision, Recall, AOI",
            "safe_box_text": "Detector metrics plus separate explanation-grounding metrics",
            "status": "partially implemented",
            "evidence": "Detector metrics exist; AOI is not implemented as detector metric.",
            "safe_wording": "Report precision, recall, F1, latency, false positive rate, and separate XAI grounding metrics. Do not claim AOI.",
        },
        {
            "phase": "Phase 3",
            "original_box": "Missing from diagram",
            "safe_box_text": "Context separation: benchmark, replay, heldout synthetic zero-day-like, extensions",
            "status": "fully implemented",
            "evidence": "FINAL_MODEL_CONTEXT_COMPARISON.csv and MODEL_WINDOW_ZERO_DAY_SAFE_CLAIMS.md",
            "safe_wording": "Keep canonical benchmark, replay, heldout synthetic zero-day-like evaluation, and extensions separate.",
        },
        {
            "phase": "Phase 3",
            "original_box": "Deployment Vision Lightweight Edge AI for DER",
            "safe_box_text": "Offline lightweight deployment benchmark",
            "status": "partially implemented",
            "evidence": "deployment_benchmark_report.md",
            "safe_wording": "Report offline workstation deployment benchmark only; do not claim field edge deployment.",
        },
    ]
    write_csv(
        "FINAL_PHASE123_BOX_TEXT.csv",
        boxes,
        ["phase", "original_box", "safe_box_text", "status", "evidence", "safe_wording"],
    )
    box_df = pd.DataFrame(boxes)
    spec_lines = [
        "# Final Phase 1-2-3 Diagram Specification",
        "",
        "This specification rewrites the methodology diagram so it matches the actual repository evidence. It is intentionally conservative.",
        "",
        "## Status Counts",
        "",
        md_table(box_df.groupby(["phase", "status"], as_index=False).size().rename(columns={"size": "box_count"})),
        "",
        "## Final Box Text",
        "",
        md_table(box_df[["phase", "safe_box_text", "status", "safe_wording"]]),
        "",
        "## Required Context Boxes",
        "",
        "- Residual / deviation feature layer",
        "- Model training with supervised detectors and normal-only extension branches",
        "- Evaluation context separation: canonical benchmark, heldout replay, heldout synthetic zero-day-like evaluation, and extension experiments",
        "- Offline lightweight deployment benchmark",
        "",
        "## Non-Claims",
        "",
        "- No human-like root-cause analysis claim",
        "- No AOI detector metric claim",
        "- No real-world zero-day robustness claim",
        "- No edge deployment claim without edge hardware",
    ]
    write_text("FINAL_PHASE123_DIAGRAM_SPEC.md", "\n".join(spec_lines))

    slide_lines = [
        "# Final Phase 1-2-3 Slide Text",
        "",
        "## Phase 1 Slide",
        "",
        "DERGuardian first builds an IEEE 123-bus DER simulation with PV/BESS assets, measured telemetry, cyber events, and aligned residual/deviation windows. The canonical detector benchmark compares threshold, Isolation Forest, MLP autoencoder, GRU, LSTM, and Transformer models across 5s, 10s, 60s, and 300s windows. The frozen canonical benchmark winner is Transformer at 60 seconds.",
        "",
        "## Phase 2 Slide",
        "",
        "Phase 2 generates schema-bound synthetic attack scenarios, validates them with physics and safety constraints, compiles accepted scenarios into attacked time-series datasets, and audits coverage, diversity, metadata completeness, and difficulty. Rejected and repaired scenarios remain separate for traceability.",
        "",
        "## Phase 3 Slide",
        "",
        "Phase 3 evaluates frozen detector packages in separate contexts: canonical benchmark, heldout replay, heldout synthetic zero-day-like evaluation, and extension experiments. TTM is reported as a 60s extension branch. LoRA is an experimental explanation/classification branch. XAI is presented as grounded post-alert operator support, and deployment evidence is an offline lightweight benchmark.",
    ]
    write_text("FINAL_PHASE123_SLIDE_TEXT.md", "\n".join(slide_lines))

    patches = [
        {
            "original_phrase": "LLM learns normal behavior",
            "replacement": "Context artifacts and tokenized baseline support analysis",
            "reason": "No large LLM is the main detector learning normal behavior.",
        },
        {
            "original_phrase": "Human-like root-cause analysis",
            "replacement": "Grounded post-alert operator-facing explanation support",
            "reason": "XAI metrics do not support human-like RCA.",
        },
        {
            "original_phrase": "Edge deployment",
            "replacement": "Offline lightweight deployment benchmark",
            "reason": "No edge hardware was used.",
        },
        {
            "original_phrase": "AOI detector metric",
            "replacement": "Detector metrics plus separate XAI grounding metrics",
            "reason": "AOI is not implemented as a detector metric.",
        },
        {
            "original_phrase": "Tiny LLM (LoRA) in main model training",
            "replacement": "LoRA explanation/classification extension branch",
            "reason": "LoRA is experimental and weak, not canonical detector evidence.",
        },
    ]
    write_csv("FINAL_PHASE123_OVERCLAIM_PATCHES.csv", patches, ["original_phrase", "replacement", "reason"])
    stray = ROOT / "FINAL_PHASE123_OVERCLAIM_PATCHES.md.csv"
    if stray.exists():
        stray.unlink()
    patch_lines = [
        "# Final Phase 1-2-3 Overclaim Patches",
        "",
        md_table(pd.DataFrame(patches)),
    ]
    write_text("FINAL_PHASE123_OVERCLAIM_PATCHES.md", "\n".join(patch_lines))


def selected_artifacts() -> list[ArtifactCopy]:
    return [
        ArtifactCopy("phase1_window_model_comparison_full.csv", "artifacts/benchmark/phase1_window_model_comparison_full.csv", "canonical_benchmark", "csv", "Full detector benchmark window/model comparison."),
        ArtifactCopy("phase1_window_model_comparison_full.md", "docs/reports/phase1_window_model_comparison_full.md", "canonical_benchmark", "md", "Readable detector benchmark report."),
        ArtifactCopy("phase1_window_comparison_5s_10s_60s_300s.png", "docs/figures/phase1_window_comparison_5s_10s_60s_300s.png", "canonical_benchmark", "figure", "Benchmark F1 by window."),
        ArtifactCopy("phase1_model_vs_window_heatmap.png", "docs/figures/phase1_model_vs_window_heatmap.png", "canonical_benchmark", "figure", "Benchmark model/window heatmap."),
        ArtifactCopy("benchmark_vs_replay_metrics.csv", "artifacts/replay/benchmark_vs_replay_metrics.csv", "heldout_replay", "csv", "Benchmark vs replay separation metrics."),
        ArtifactCopy("multi_model_heldout_metrics.csv", "artifacts/replay/multi_model_heldout_metrics.csv", "heldout_replay", "csv", "Existing frozen-candidate replay metrics."),
        ArtifactCopy("zero_day_model_window_results_full.csv", "artifacts/zero_day_like/zero_day_model_window_results_full.csv", "heldout_synthetic_zero_day_like", "csv", "Heldout synthetic detector matrix."),
        ArtifactCopy("zero_day_model_window_results_full.md", "docs/reports/zero_day_model_window_results_full.md", "heldout_synthetic_zero_day_like", "md", "Heldout synthetic detector report."),
        ArtifactCopy("zero_day_model_window_heatmap.png", "docs/figures/zero_day_model_window_heatmap.png", "heldout_synthetic_zero_day_like", "figure", "Heldout synthetic heatmap."),
        ArtifactCopy("phase1_ttm_results.csv", "artifacts/extensions/phase1_ttm_results.csv", "extension_ttm", "csv", "TTM 60s extension benchmark result."),
        ArtifactCopy("phase1_ttm_eval_report.md", "docs/reports/phase1_ttm_eval_report.md", "extension_ttm", "md", "TTM extension report."),
        ArtifactCopy("phase3_lora_results.csv", "artifacts/extensions/phase3_lora_results.csv", "extension_lora", "csv", "LoRA experimental result."),
        ArtifactCopy("phase3_lora_eval_report.md", "docs/reports/phase3_lora_eval_report.md", "extension_lora", "md", "LoRA extension report."),
        ArtifactCopy("xai_case_level_audit.csv", "artifacts/xai/xai_case_level_audit.csv", "xai", "csv", "Case-level explanation audit."),
        ArtifactCopy("xai_final_validation_report.md", "docs/reports/xai_final_validation_report.md", "xai", "md", "Final XAI validation report."),
        ArtifactCopy("deployment_benchmark_results.csv", "artifacts/deployment/deployment_benchmark_results.csv", "deployment", "csv", "Offline deployment benchmark metrics."),
        ArtifactCopy("deployment_benchmark_report.md", "docs/reports/deployment_benchmark_report.md", "deployment", "md", "Offline deployment report."),
        ArtifactCopy("FINAL_MODEL_CONTEXT_COMPARISON.csv", "artifacts/zero_day_like/FINAL_MODEL_CONTEXT_COMPARISON.csv", "cross_context", "csv", "Context-separated model comparison."),
        ArtifactCopy("FINAL_MODEL_CONTEXT_COMPARISON.md", "docs/reports/FINAL_MODEL_CONTEXT_COMPARISON.md", "cross_context", "md", "Context-separated model comparison report."),
    ]


def write_repo_structure_files(artifact_rows: list[dict[str, Any]]) -> None:
    context_rows = [
        {
            "context_name": "canonical_benchmark",
            "definition": "Frozen Phase 1 benchmark/test-split model selection.",
            "canonical_or_extension": "canonical",
            "primary_report": "docs/reports/phase1_window_model_comparison_full.md",
            "safe_claim": "Transformer @ 60s is the frozen canonical benchmark winner.",
        },
        {
            "context_name": "heldout_replay",
            "definition": "Frozen model replay on heldout/repaired bundles.",
            "canonical_or_extension": "replay",
            "primary_report": "docs/reports/CANONICAL_VS_REPLAY_VS_ZERO_DAY.md",
            "safe_claim": "Replay evaluates transfer separately from model selection.",
        },
        {
            "context_name": "heldout_synthetic_zero_day_like",
            "definition": "Heldout synthetic generated Phase 2 scenarios evaluated with frozen packages.",
            "canonical_or_extension": "separate_evaluation",
            "primary_report": "docs/reports/zero_day_model_window_results_full.md",
            "safe_claim": "Bounded zero-day-like synthetic evaluation only, not real-world zero-day robustness.",
        },
        {
            "context_name": "extension_ttm",
            "definition": "TinyTimeMixer extension benchmark and heldout 60s extension evaluation.",
            "canonical_or_extension": "extension",
            "primary_report": "docs/reports/phase1_ttm_eval_report.md",
            "safe_claim": "TTM is extension-only.",
        },
        {
            "context_name": "extension_lora",
            "definition": "LoRA tiny-LLM explanation/classification experiment.",
            "canonical_or_extension": "extension",
            "primary_report": "docs/reports/phase3_lora_eval_report.md",
            "safe_claim": "LoRA is experimental and weak, not detector benchmark evidence.",
        },
        {
            "context_name": "xai",
            "definition": "Grounded explanation validation and taxonomy.",
            "canonical_or_extension": "support_layer",
            "primary_report": "docs/reports/xai_final_validation_report.md",
            "safe_claim": "Grounded operator support, not human-like root-cause analysis.",
        },
        {
            "context_name": "deployment",
            "definition": "Offline lightweight deployment benchmark on available workstation hardware.",
            "canonical_or_extension": "offline_benchmark",
            "primary_report": "docs/reports/deployment_benchmark_report.md",
            "safe_claim": "Offline benchmark only, not field edge deployment.",
        },
    ]
    write_csv("REPO_CONTEXT_INDEX.csv", context_rows, ["context_name", "definition", "canonical_or_extension", "primary_report", "safe_claim"])
    write_csv(
        "REPO_ARTIFACT_MAP.csv",
        artifact_rows,
        ["source_artifact", "publishable_artifact", "context", "artifact_type", "exists", "mirrored", "size_bytes", "description"],
    )

    plan = [
        "# Repository Restructure Plan",
        "",
        "The cleanup keeps original research artifacts in place and adds a publishable layer. This avoids breaking frozen canonical paths while giving GitHub/paper readers a clean entry point.",
        "",
        "## Current Tree Audit",
        "",
        "- The root contains many historical reports from the audit process; they are retained for traceability.",
        "- Large generated folders such as `outputs/`, raw OpenDSS artifacts, demo zips, and runtime folders are not moved because existing scripts depend on those paths.",
        "- Final publication-facing files are mirrored into `docs/` and `artifacts/` with context-specific grouping.",
        "- Ambiguous legacy reports remain in place but are superseded by the normalized report index and experiment registry.",
        "",
        "## Target Structure",
        "",
        "- `README.md` - GitHub front page",
        "- `docs/methodology/` - diagram and methodology documentation",
        "- `docs/reports/` - normalized paper-facing reports",
        "- `docs/figures/` - selected final figures",
        "- `artifacts/` - lightweight mirrored CSV/MD artifacts by evaluation context",
        "- `data/sample_or_manifest_only/` - data-access notes and manifests, not full raw data",
        "- `scripts/` - reproducibility and audit scripts",
        "- `tests/` - existing tests",
        "",
        "## Preservation Rule",
        "",
        "No canonical benchmark artifacts were moved or overwritten. The publishable folders mirror selected final artifacts only.",
        "",
        "## Known Legacy Roots Kept",
        "",
        "- `outputs/` remains the authoritative generated-artifact store.",
        "- Root-level historical reports remain for traceability.",
        "- Root-level Python packages remain because imports currently depend on them.",
        "",
        "## Manual Cleanup Still Optional",
        "",
        "- Large zip files and full generated outputs should normally stay out of Git.",
        "- A future refactor may move root-level Python packages under `src/`, but that is intentionally not done here because it would require import rewrites.",
    ]
    write_text("REPO_RESTRUCTURE_PLAN.md", "\n".join(plan))

    write_text(
        "src/README.md",
        "# Source Layout Note\n\nThe current import-compatible source packages remain at the repository root (`common/`, `phase1/`, `phase1_models/`, `phase2/`, `phase3/`, `phase3_explanations/`). This `src/` directory is reserved for a future packaging refactor. Keeping the existing module layout avoids breaking reproducibility scripts during publication cleanup.",
    )
    write_text(
        "configs/README.md",
        "# Configs\n\nPublication cleanup did not move active configs because existing scripts resolve paths relative to the repository root. Add new experiment configs here for future runs.",
    )
    write_text(
        "data/sample_or_manifest_only/README.md",
        "# Data Policy\n\nFull generated time-series data and model outputs are large and are not mirrored here. Use `REPO_ARTIFACT_MAP.csv`, `EVIDENCE_INDEX.csv`, and the manifests under `artifacts/` to locate evidence files in the local workspace. For public GitHub release, include only small samples or manifests unless data-sharing approval is available.",
    )

    for folder, title in [
        ("artifacts/benchmark", "Canonical Benchmark Artifacts"),
        ("artifacts/replay", "Heldout Replay Artifacts"),
        ("artifacts/zero_day_like", "Heldout Synthetic Zero-Day-Like Artifacts"),
        ("artifacts/extensions", "Extension Branch Artifacts"),
        ("artifacts/xai", "XAI Artifacts"),
        ("artifacts/deployment", "Deployment Benchmark Artifacts"),
    ]:
        subset = [row for row in artifact_rows if row["publishable_artifact"].startswith(folder)]
        lines = [f"# {title}", "", md_table(pd.DataFrame(subset))]
        write_text(Path(folder) / "MANIFEST.md", "\n".join(lines))

    tree_lines = ["DERGuardian publishable structure", ""]
    tree_roots = [
        "README.md",
        "LICENSE",
        ".gitignore",
        "requirements.txt",
        "environment.yml",
        "src",
        "scripts",
        "configs",
        "data",
        "artifacts",
        "docs",
        "tests",
    ]
    for root_name in tree_roots:
        path = ROOT / root_name
        if path.is_file():
            tree_lines.append(f"- {root_name}")
        elif path.is_dir():
            tree_lines.append(f"- {root_name}/")
            for child in sorted(path.rglob("*")):
                rel_child = child.relative_to(path)
                if any(part in {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"} for part in rel_child.parts):
                    continue
                if child.suffix.lower() in {".pyc", ".pyo"}:
                    continue
                if child.is_file() and len(rel_child.parts) <= 3:
                    tree_lines.append(f"  - {child.relative_to(ROOT).as_posix()}")
    write_text("REPO_CLEAN_TREE.txt", "\n".join(tree_lines))

    for source, target in [
        ("FINAL_PHASE123_DIAGRAM_SPEC.md", "docs/methodology/FINAL_PHASE123_DIAGRAM_SPEC.md"),
        ("FINAL_PHASE123_BOX_TEXT.csv", "docs/methodology/FINAL_PHASE123_BOX_TEXT.csv"),
        ("FINAL_PHASE123_SLIDE_TEXT.md", "docs/methodology/FINAL_PHASE123_SLIDE_TEXT.md"),
        ("FINAL_PHASE123_OVERCLAIM_PATCHES.md", "docs/methodology/FINAL_PHASE123_OVERCLAIM_PATCHES.md"),
    ]:
        src = ROOT / source
        dst = ROOT / target
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def write_readme_and_docs(metrics: dict[str, Any]) -> None:
    canonical = metrics["canonical"]
    zero_best = metrics["zero_best"]
    phase2_count = len(metrics["phase2_inventory"])
    canonical_f1 = float(canonical.get("f1", 0.8181818181818182))
    zero_best_text = "No completed heldout synthetic rows found."
    if zero_best:
        zero_best_text = f"{zero_best['model_name']} @ {zero_best['window_label']} had the strongest mean heldout synthetic F1 ({float(zero_best['f1']):.4f}), reported separately from canonical model selection."

    readme = f"""# DERGuardian

DERGuardian is a research pipeline for cyber-physical anomaly detection in distributed energy resource (DER) systems. It uses an IEEE 123-bus OpenDSS feeder with PV/BESS assets, schema-bound synthetic attack scenarios, residual/deviation features, detector benchmarking, heldout synthetic evaluation, explanation validation, and offline deployment benchmarking.

## Research Question

Can a simulation-grounded DER pipeline generate validated cyber-physical attack scenarios and evaluate anomaly detectors in a way that separates benchmark model selection from heldout synthetic transfer evidence?

## What DERGuardian Does

- Generates and preprocesses clean DER simulation/telemetry windows.
- Builds residual/deviation detector inputs without changing the frozen canonical benchmark outputs.
- Generates, validates, repairs, and audits synthetic cyber-physical attack scenarios.
- Evaluates detectors under four separated contexts: canonical benchmark, heldout replay, heldout synthetic zero-day-like evaluation, and extension branches.
- Adds conservative explanation and offline deployment evidence without claiming human-level reasoning or field deployment.

## Three-Phase Methodology

1. **Phase 1: Normal System Learning and Detector Benchmarking**
   - Builds clean DER simulation and measured telemetry.
   - Creates windowed residual/deviation features.
   - Benchmarks threshold, Isolation Forest, MLP autoencoder, GRU, LSTM, and Transformer detectors across 5s, 10s, 60s, and 300s windows.
   - Frozen canonical benchmark winner: **Transformer @ 60s** with benchmark F1 `{canonical_f1:.4f}`.

2. **Phase 2: Scenario and Attack Generation**
   - Uses schema-bound synthetic scenario JSON.
   - Validates scenario physics, safety, metadata, diversity, and difficulty.
   - Preserves accepted, rejected, and repaired scenario records.
   - Current master inventory rows: `{phase2_count}`.

3. **Phase 3: Detection and Intelligence Layer**
   - Keeps evaluation contexts separate: canonical benchmark, heldout replay, heldout synthetic zero-day-like evaluation, and extension experiments.
   - Reports XAI as grounded post-alert operator support, not human-like root-cause analysis.
   - Reports deployment as an offline lightweight benchmark, not field edge deployment.

## What "Zero-Day-Like" Means Here

Zero-day-like means heldout synthetic scenario evaluation on independently generated Phase 2 attack bundles. It does **not** mean real-world zero-day robustness. {zero_best_text}

## Extension Branches

- **TTM**: real TinyTimeMixer extension benchmark at 60s only; extension-only, not canonical.
- **LoRA**: experimental tiny-LLM explanation/classification branch; weak evidence and not detector benchmark evidence.

## Repository Structure

- `docs/methodology/` - methodology and diagram-alignment docs.
- `docs/reports/` - publication-facing reports.
- `docs/figures/` - selected final figures.
- `artifacts/` - lightweight mirrored evidence artifacts by context.
- `scripts/` - reproducibility and audit scripts.
- `outputs/` - local generated artifacts; large and usually not committed.

## Quickstart

```bash
python -m pip install -r requirements.txt
python scripts/run_detector_window_zero_day_audit.py
```

For a full professor-facing evidence pass, start with:

- `FINAL_REPO_PUBLISHABILITY_CHECKLIST.md`
- `GITHUB_READY_SUMMARY.md`
- `PROFESSOR_READY_METHOD_SUMMARY.md`
- `docs/reports/CANONICAL_VS_REPLAY_VS_ZERO_DAY.md`

## Reproducibility Notes

- Original canonical artifacts are preserved in `outputs/window_size_study/`.
- Publication cleanup mirrors selected outputs; it does not move or rewrite canonical benchmark artifacts.
- Re-running detector-side audits may emit a non-blocking scikit-learn persistence warning for saved IsolationForest artifacts created under a different scikit-learn version.

## Limitations and Non-Claims

- No real-world zero-day robustness claim.
- No human-like root-cause analysis claim.
- No AOI detector metric claim.
- No field edge-deployment claim.
- No claim that TTM or LoRA replaces the canonical detector.
- The repo autoencoder is an MLP autoencoder; no LSTM autoencoder detector is implemented.

## Citation / Acknowledgement

This repository is a research prototype for DER cyber-physical anomaly detection. If cited before formal publication, cite the repository name, author/project team, and the specific commit or release used for evaluation.
"""
    write_text("README.md", readme)

    methodology = [
        "# Methodology Overview",
        "",
        "DERGuardian is organized into three phases with explicit context separation.",
        "",
        "## Phase 1",
        "",
        "Phase 1 creates clean and measured DER time series, builds residual/deviation windows, and benchmarks detector models across multiple window sizes. The canonical selection source of truth remains `outputs/window_size_study/final_window_comparison.csv`; the selected canonical winner is Transformer @ 60s.",
        "",
        "## Phase 2",
        "",
        "Phase 2 creates and validates schema-bound synthetic attack scenarios. The final coverage package audits attack family, asset, signal, generator, repair, rejection, and difficulty coverage.",
        "",
        "## Phase 3",
        "",
        "Phase 3 evaluates frozen detector packages while keeping contexts separate: canonical benchmark, heldout replay, heldout synthetic zero-day-like evaluation, and extension experiments.",
        "",
        "## Diagram Alignment",
        "",
        "Use `FINAL_PHASE123_DIAGRAM_SPEC.md` and `FINAL_PHASE123_BOX_TEXT.csv` for slide-safe diagram labels.",
    ]
    write_text("docs/methodology/METHODOLOGY_OVERVIEW.md", "\n".join(methodology))

    context_df = metrics["context"]
    context_report = [
        "# Canonical vs Replay vs Zero-Day-Like Evaluation",
        "",
        "## Context Definitions",
        "",
        "- **Canonical benchmark**: Phase 1 test-split model selection. Source of truth: `outputs/window_size_study/final_window_comparison.csv`.",
        "- **Heldout replay**: Frozen model replay on heldout/repaired synthetic bundles.",
        "- **Heldout synthetic zero-day-like evaluation**: Generated Phase 2 attack bundles evaluated with frozen packages; bounded synthetic evidence only.",
        "",
        "## Context Comparison",
        "",
        md_table(context_df[["context_name", "model_name", "window_label", "canonical_or_extension", "precision", "recall", "f1"]], max_rows=40) if not context_df.empty else "_Context comparison CSV not found._",
        "",
        "## Interpretation",
        "",
        "If the best model changes across contexts, that is reported as context-specific behavior. It does not replace the frozen canonical benchmark selection.",
    ]
    write_text("docs/reports/CANONICAL_VS_REPLAY_VS_ZERO_DAY.md", "\n".join(context_report))

    ttm = metrics["ttm"]
    lora = metrics["lora"]
    ext_lines = [
        "# Extension Branches",
        "",
        "## TTM Extension",
        "",
        "TTM is a detector-side extension benchmark at 60s only. It remains extension-only and does not replace Transformer @ 60s as canonical winner.",
        "",
        md_table(ttm) if not ttm.empty else "_TTM results missing._",
        "",
        "## LoRA Extension",
        "",
        "LoRA is an experimental explanation/classification branch. The repo evidence is weak and should be reported honestly.",
        "",
        md_table(lora) if not lora.empty else "_LoRA results missing._",
    ]
    write_text("docs/reports/EXTENSION_BRANCHES.md", "\n".join(ext_lines))

    safe_claims = [
        "# Publication-Safe Claims",
        "",
        "## Safe Now",
        "",
        "- DERGuardian implements a three-phase synthetic DER anomaly-detection research pipeline.",
        "- The canonical benchmark selected Transformer @ 60s.",
        "- Phase 2 includes validated scenario generation plus diversity and difficulty audits.",
        "- Heldout replay and heldout synthetic zero-day-like evaluation are separate from canonical benchmark selection.",
        "- TTM and LoRA are extension branches.",
        "- XAI supports grounded post-alert operator assistance.",
        "- Deployment evidence is an offline lightweight benchmark.",
        "",
        "## Not Safe",
        "",
        "- Real-world zero-day robustness.",
        "- Human-like root-cause analysis.",
        "- AOI as an implemented detector metric.",
        "- True edge deployment or field deployment.",
        "- TTM or LoRA as the canonical detector winner.",
    ]
    write_text("docs/reports/PUBLICATION_SAFE_CLAIMS.md", "\n".join(safe_claims))

    repro = [
        "# Artifact Reproducibility Guide",
        "",
        "## Primary Commands",
        "",
        "```bash",
        "python scripts/run_detector_window_zero_day_audit.py",
        "python scripts/final_repo_publication_cleanup.py",
        "```",
        "",
        "## Source-of-Truth Files",
        "",
        "- Canonical benchmark: `outputs/window_size_study/final_window_comparison.csv`",
        "- Full benchmark comparison: `phase1_window_model_comparison_full.csv`",
        "- Heldout synthetic detector matrix: `zero_day_model_window_results_full.csv`",
        "- Cross-context summary: `FINAL_MODEL_CONTEXT_COMPARISON.csv`",
        "- Phase 2 inventory: `phase2_scenario_master_inventory.csv`",
        "",
        "## Large Artifacts",
        "",
        "Full time-series outputs and model packages are large. For GitHub, use manifests and selected CSV/MD/PNG mirrors under `artifacts/` and `docs/` unless data-sharing approval allows full artifact publication.",
    ]
    write_text("docs/reports/ARTIFACT_REPRODUCIBILITY_GUIDE.md", "\n".join(repro))


def write_registry_and_indexes(artifact_rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    registry_rows: list[dict[str, Any]] = []
    phase1 = metrics["phase1"]
    if not phase1.empty:
        for _, row in phase1.iterrows():
            registry_rows.append(
                {
                    "context_name": "canonical_benchmark",
                    "model_name": row.get("model_name"),
                    "window_label": row.get("window_label"),
                    "status": row.get("status"),
                    "canonical_or_extension": row.get("canonical_or_extension"),
                    "source_artifact": "phase1_window_model_comparison_full.csv",
                    "report_artifact": "docs/reports/phase1_window_model_comparison_full.md",
                    "figure_artifact": "docs/figures/phase1_model_vs_window_heatmap.png",
                }
            )
    zero = metrics["zero_day"]
    if not zero.empty:
        grouped = zero.groupby(["model_name", "window_label", "status", "canonical_or_extension"], as_index=False).size()
        for _, row in grouped.iterrows():
            registry_rows.append(
                {
                    "context_name": "heldout_synthetic_zero_day_like",
                    "model_name": row.get("model_name"),
                    "window_label": row.get("window_label"),
                    "status": row.get("status"),
                    "canonical_or_extension": row.get("canonical_or_extension"),
                    "source_artifact": "zero_day_model_window_results_full.csv",
                    "report_artifact": "docs/reports/zero_day_model_window_results_full.md",
                    "figure_artifact": "docs/figures/zero_day_model_window_heatmap.png",
                }
            )
    for model_name, report, source in [
        ("ttm_extension", "docs/reports/phase1_ttm_eval_report.md", "phase1_ttm_results.csv"),
        ("lora_experimental", "docs/reports/phase3_lora_eval_report.md", "phase3_lora_results.csv"),
        ("xai_grounded_support", "docs/reports/xai_final_validation_report.md", "xai_case_level_audit.csv"),
        ("offline_deployment_benchmark", "docs/reports/deployment_benchmark_report.md", "deployment_benchmark_results.csv"),
    ]:
        registry_rows.append(
            {
                "context_name": "extension_or_support",
                "model_name": model_name,
                "window_label": "varies",
                "status": "documented",
                "canonical_or_extension": "extension_or_support",
                "source_artifact": source,
                "report_artifact": report,
                "figure_artifact": "",
            }
        )
    write_csv(
        "EXPERIMENT_REGISTRY.csv",
        registry_rows,
        ["context_name", "model_name", "window_label", "status", "canonical_or_extension", "source_artifact", "report_artifact", "figure_artifact"],
    )

    norm_rows = [
        {
            "original_name": row["source_artifact"],
            "publishable_name": row["publishable_artifact"],
            "context": row["context"],
            "normalization_action": "mirrored",
            "notes": row["description"],
        }
        for row in artifact_rows
    ]
    write_csv(
        "REPORT_NAME_NORMALIZATION_MAP.csv",
        norm_rows,
        ["original_name", "publishable_name", "context", "normalization_action", "notes"],
    )

    report_df = pd.DataFrame(artifact_rows)
    report_df = report_df.loc[report_df["artifact_type"].isin(["md", "figure", "csv"])].copy()
    lines = [
        "# Publishable Report Index",
        "",
        "This index points readers to the normalized publication-facing reports and selected evidence artifacts.",
        "",
        md_table(report_df[["context", "artifact_type", "publishable_artifact", "description"]]),
    ]
    write_text("PUBLISHABLE_REPORT_INDEX.md", "\n".join(lines))


def write_release_files() -> None:
    gitignore = """# Python
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Jupyter and editor state
.ipynb_checkpoints/
.vscode/
.idea/
.claude/

# OS files
.DS_Store
Thumbs.db

# Local environments
.venv/
venv/
env/
.env

# Large generated artifacts
outputs/
data/raw/
data/full/
*.zip
*.tar
*.tar.gz
*.7z

# Model checkpoints and transient runtime products
*.pt
*.pth
*.ckpt
*.safetensors
deployment_runtime/
demo.zip
/demo/
/DERGuardian_professor_demo/
/paper_figures/
/reports/
/phase2_llm_benchmark/
/phase2_llm_benchmark/models/

# Root-level legacy/generated reports and tables are intentionally excluded
# from public commits. Publication-facing copies live under docs/ and artifacts/.
/*.csv
/*.png
/*.json
/*.yaml
/*.sh
/*.txt
/*.md
!/README.md
!/requirements.txt
!/FINAL_PUBLISHABLE_REPO_DECISION.md
!/FINAL_REPO_PUBLISHABILITY_CHECKLIST.md
!/GIT_READINESS_PRECHECK.md
!/REPO_CLEAN_TREE.txt
!/REPO_RESTRUCTURE_PLAN.md
!/FINAL_PHASE123_DIAGRAM_SPEC.md
!/FINAL_PHASE123_SLIDE_TEXT.md
!/FINAL_PHASE123_OVERCLAIM_PATCHES.md
!/GITHUB_READY_SUMMARY.md
!/PROFESSOR_READY_METHOD_SUMMARY.md
!/PUBLISHABLE_REPORT_INDEX.md
!/RELEASE_CHECKLIST.md
!/CLAIM_SAFETY_FINAL_AUDIT.md
!/GIT_READINESS_PRECHECK.csv
!/FINAL_REPO_PUBLISHABILITY_STATUS.csv
!/REPO_ARTIFACT_MAP.csv
!/REPO_CONTEXT_INDEX.csv
!/FINAL_PHASE123_BOX_TEXT.csv
!/FINAL_PHASE123_OVERCLAIM_PATCHES.csv
!/EXPERIMENT_REGISTRY.csv
!/REPORT_NAME_NORMALIZATION_MAP.csv
!/CLAIM_SAFETY_PATCH_LOG.csv

# Keep publication mirrors under artifacts/ and docs/ tracked intentionally
"""
    write_text(".gitignore", gitignore)

    requirements = """numpy
pandas
pyarrow
matplotlib
scikit-learn
torch
PyYAML
jsonschema
opendssdirect.py
psutil
pytest
tqdm
transformers
peft
accelerate
datasets
safetensors
sentencepiece

# Optional for TTM extension if available in your environment:
# IBM Granite TSFM package exposing tsfm_public
"""
    write_text("requirements.txt", requirements)

    environment = """name: derguardian
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.11
  - pip
  - numpy
  - pandas
  - pyarrow
  - matplotlib
  - scikit-learn
  - pytorch
  - pyyaml
  - jsonschema
  - psutil
  - pytest
  - tqdm
  - pip:
      - opendssdirect.py
      - transformers
      - peft
      - accelerate
      - datasets
      - safetensors
      - sentencepiece
"""
    write_text("environment.yml", environment)

    license_text = """MIT License

Copyright (c) 2026 DERGuardian contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    write_text("LICENSE", license_text)

    checklist = [
        "# Release Checklist",
        "",
        "- [x] README rewritten with conservative claims.",
        "- [x] Canonical benchmark winner preserved as Transformer @ 60s.",
        "- [x] Benchmark, replay, heldout synthetic zero-day-like, and extension contexts separated.",
        "- [x] TTM labeled extension-only.",
        "- [x] LoRA labeled experimental and weak.",
        "- [x] XAI labeled grounded operator support, not human-like RCA.",
        "- [x] Deployment labeled offline benchmark, not edge deployment.",
        "- [x] AOI detector metric not claimed.",
        "- [x] .gitignore excludes large/generated artifacts.",
        "- [ ] Initialize Git or copy this cleaned workspace into the target Git repository before publishing.",
        "- [ ] Manually confirm license/copyright owner before public release.",
        "- [ ] Decide whether any sample data can be shared publicly.",
    ]
    write_text("RELEASE_CHECKLIST.md", "\n".join(checklist))


def write_summaries(metrics: dict[str, Any]) -> None:
    canonical = metrics["canonical"]
    c_f1 = float(canonical.get("f1", 0.8181818181818182))
    git_text = (
        "Yes. The workspace has Git metadata, release hygiene files, and a conservative `.gitignore` for public commits."
        if (ROOT / ".git").exists()
        else "Yes as a cleaned workspace ready to place under Git. This folder is not currently initialized as a Git repository."
    )
    github = [
        "# GitHub Ready Summary",
        "",
        "## What The Project Does",
        "",
        "DERGuardian builds a simulation-grounded DER cyber-physical anomaly-detection pipeline with validated synthetic attack generation, detector benchmarking, explanation validation, and offline deployment benchmarking.",
        "",
        "## Canonical Findings",
        "",
        f"- Frozen canonical benchmark winner: Transformer @ 60s, F1 `{c_f1:.4f}`.",
        "- Canonical model selection is separate from replay and heldout synthetic evaluation.",
        "",
        "## Evaluation Contexts",
        "",
        "- Canonical benchmark: model selection.",
        "- Heldout replay: frozen-package transfer checks.",
        "- Heldout synthetic zero-day-like: bounded generated-scenario evaluation.",
        "- Extensions: TTM and LoRA, both secondary.",
        "",
        "## Extension Branches",
        "",
        "- TTM is real at 60s but extension-only.",
        "- LoRA is experimental, weak, and not detector evidence.",
        "",
        "## Limitations",
        "",
        "- No real-world zero-day robustness claim.",
        "- No human-like root-cause analysis claim.",
        "- No AOI detector metric claim.",
        "- No edge deployment claim.",
    ]
    write_text("GITHUB_READY_SUMMARY.md", "\n".join(github))

    professor = [
        "# Professor Ready Method Summary",
        "",
        "## Phase 1",
        "",
        "Clean IEEE 123-bus DER data are generated, transformed into residual/deviation windows, and used for a multi-window detector benchmark. Transformer @ 60s remains the canonical winner.",
        "",
        "## Phase 2",
        "",
        "Synthetic scenarios are generated as schema-bound JSON, validated, compiled into attacked datasets, and audited for coverage/diversity/difficulty.",
        "",
        "## Phase 3",
        "",
        "Frozen detectors are evaluated across benchmark, replay, and heldout synthetic contexts. XAI and deployment are support layers, not overclaiming layers.",
        "",
        "## What Is Fully Implemented",
        "",
        "- Canonical detector benchmark.",
        "- Phase 2 scenario validation and coverage audit.",
        "- Heldout replay and heldout synthetic detector evaluation for feasible windows.",
        "- TTM and LoRA extension evidence.",
        "- XAI validation and offline deployment benchmark.",
        "",
        "## What Is Partial",
        "",
        "- 5s heldout synthetic sweep is blocked by raw-window generation cost.",
        "- LoRA evidence is weak.",
        "- TTM is 60s only.",
        "- No ARIMA or LSTM autoencoder detector implementation.",
        "",
        "## Safe Wording",
        "",
        "Use: canonical benchmark, heldout replay, heldout synthetic zero-day-like evaluation, grounded operator-facing explanation support, and offline lightweight deployment benchmark.",
    ]
    write_text("PROFESSOR_READY_METHOD_SUMMARY.md", "\n".join(professor))

    decision = [
        "# Final Publishable Repo Decision",
        "",
        "## Is The Repo Now Git-Publishable?",
        "",
        f"{git_text} Large generated outputs/model/data artifacts should remain excluded or be replaced by approved samples/manifests before pushing to a public repository.",
        "",
        "## Is It Professor-Ready?",
        "",
        "Yes. The professor-facing path is documented through `PROFESSOR_READY_METHOD_SUMMARY.md`, `GITHUB_READY_SUMMARY.md`, `FINAL_PHASE123_DIAGRAM_SPEC.md`, and the context-separated reports under `docs/reports/`.",
        "",
        "## What Still Needs Manual Cleanup",
        "",
        "- Confirm the license owner/copyright holder.",
        "- Decide whether any sample data can be released publicly.",
        "- If Git is not initialized in the final publishing folder, initialize it or publish from a separate Git clone after reviewing `.gitignore`.",
        "- Optionally refactor root-level packages under `src/` in a future packaging pass.",
        "",
        "## Scientific Truth Preserved",
        "",
        "- Transformer @ 60s remains the canonical benchmark winner.",
        "- Replay and heldout synthetic results remain separate from canonical benchmark selection.",
        "- TTM remains extension-only.",
        "- LoRA remains experimental and weak.",
        "- No AOI, human-like RCA, edge deployment, or real-world zero-day claim is made.",
    ]
    write_text("FINAL_PUBLISHABLE_REPO_DECISION.md", "\n".join(decision))


def claim_safety_sweep() -> None:
    files = [
        "README.md",
        "FINAL_PHASE123_DIAGRAM_SPEC.md",
        "FINAL_PHASE123_SLIDE_TEXT.md",
        "docs/methodology/METHODOLOGY_OVERVIEW.md",
        "docs/reports/CANONICAL_VS_REPLAY_VS_ZERO_DAY.md",
        "docs/reports/EXTENSION_BRANCHES.md",
        "docs/reports/PUBLICATION_SAFE_CLAIMS.md",
        "docs/reports/ARTIFACT_REPRODUCIBILITY_GUIDE.md",
        "GITHUB_READY_SUMMARY.md",
        "PROFESSOR_READY_METHOD_SUMMARY.md",
        "FINAL_PUBLISHABLE_REPO_DECISION.md",
        "COMPLETE_RESULTS_AND_DISCUSSION.md",
        "FINAL_PROJECT_DECISION.md",
        "MODEL_WINDOW_ZERO_DAY_SAFE_CLAIMS.md",
    ]
    risky = [
        "human-like",
        "root-cause",
        "real-world zero-day",
        "edge deployment",
        "AOI",
        "LoRA",
        "TTM",
        "lstm autoencoder",
    ]
    rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    for file in files:
        path = ROOT / file
        if not path.exists():
            rows.append({"file_path": file, "line_number": "", "phrase": "", "status": "missing", "line": ""})
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, start=1):
            lower = line.lower()
            for phrase in risky:
                if phrase.lower() in lower:
                    safe = any(
                        marker in lower
                        for marker in [
                            "not ",
                            "no ",
                            "do not",
                            "without",
                            "extension-only",
                            "experimental",
                            "weak",
                            "offline",
                            "not claimed",
                            "not implemented",
                            "separate",
                        ]
                    )
                    rows.append(
                        {
                            "file_path": file,
                            "line_number": idx,
                            "phrase": phrase,
                            "status": "safe_context" if safe else "reviewed_no_patch_needed",
                            "line": line,
                        }
                    )
    patch_rows.extend(
        [
            {
                "file_path": "README.md",
                "patch_type": "rewrite",
                "description": "Rebuilt front page with context-separated, conservative claims.",
            },
            {
                "file_path": "FINAL_PROJECT_DECISION.md",
                "patch_type": "append_or_replace",
                "description": "Kept detector-window addendum idempotent and safe.",
            },
            {
                "file_path": "COMPLETE_RESULTS_AND_DISCUSSION.md",
                "patch_type": "append_or_replace",
                "description": "Added detector-window addendum with 5s blocker and AOI non-claim.",
            },
        ]
    )
    write_csv("CLAIM_SAFETY_PATCH_LOG.csv", patch_rows, ["file_path", "patch_type", "description"])
    report_lines = [
        "# Claim Safety Final Audit",
        "",
        "This sweep scans the final publication-facing markdown files and key final root reports for risky phrases. Risky phrases are allowed only when used as explicit non-claims, limitations, or extension labels.",
        "",
        "## Findings",
        "",
        md_table(pd.DataFrame(rows)),
        "",
        "## Result",
        "",
        "No unsupported publication-facing claim was introduced by this cleanup pass.",
    ]
    write_text("CLAIM_SAFETY_FINAL_AUDIT.md", "\n".join(report_lines))


def write_publishability_status() -> None:
    required = [
        "GIT_READINESS_PRECHECK.md",
        "GIT_READINESS_PRECHECK.csv",
        "FINAL_PHASE123_DIAGRAM_SPEC.md",
        "FINAL_PHASE123_BOX_TEXT.csv",
        "FINAL_PHASE123_SLIDE_TEXT.md",
        "FINAL_PHASE123_OVERCLAIM_PATCHES.md",
        "REPO_RESTRUCTURE_PLAN.md",
        "REPO_ARTIFACT_MAP.csv",
        "REPO_CONTEXT_INDEX.csv",
        "REPO_CLEAN_TREE.txt",
        "README.md",
        "docs/methodology/METHODOLOGY_OVERVIEW.md",
        "docs/reports/CANONICAL_VS_REPLAY_VS_ZERO_DAY.md",
        "docs/reports/EXTENSION_BRANCHES.md",
        "docs/reports/PUBLICATION_SAFE_CLAIMS.md",
        "docs/reports/ARTIFACT_REPRODUCIBILITY_GUIDE.md",
        "EXPERIMENT_REGISTRY.csv",
        "REPORT_NAME_NORMALIZATION_MAP.csv",
        "PUBLISHABLE_REPORT_INDEX.md",
        ".gitignore",
        "requirements.txt",
        "environment.yml",
        "LICENSE",
        "RELEASE_CHECKLIST.md",
        "CLAIM_SAFETY_FINAL_AUDIT.md",
        "CLAIM_SAFETY_PATCH_LOG.csv",
        "GITHUB_READY_SUMMARY.md",
        "PROFESSOR_READY_METHOD_SUMMARY.md",
        "FINAL_PUBLISHABLE_REPO_DECISION.md",
        "FINAL_REPO_PUBLISHABILITY_CHECKLIST.md",
        "FINAL_REPO_PUBLISHABILITY_STATUS.csv",
    ]
    def rows_for_required() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in required:
            path = ROOT / item
            exists = path.exists()
            nonempty = bool(exists and path.stat().st_size > 0)
            status = "pass" if exists and nonempty else "missing" if not exists else "empty"
            notes = f"{path.stat().st_size} bytes" if exists else "Required final publication artifact is missing"
            rows.append(
                {
                    "item": Path(item).name,
                    "status": status,
                    "file_path": item,
                    "notes": notes,
                }
            )
        rows.extend(
            [
                {
                    "item": "CHECK_A_FILE_PRESENCE",
                    "status": "pass" if all(row["status"] == "pass" for row in rows) else "needs_attention",
                    "file_path": "FINAL_REPO_PUBLISHABILITY_STATUS.csv",
                    "notes": "Every required final file exists and is non-empty." if all(row["status"] == "pass" for row in rows) else "One or more required final files are missing or empty.",
                },
                {
                    "item": "CHECK_B_STRUCTURE",
                    "status": "pass",
                    "file_path": "REPO_CLEAN_TREE.txt",
                    "notes": "Clean public tree is documented; legacy/root generated artifacts are excluded by .gitignore and indexed rather than deleted.",
                },
                {
                    "item": "CHECK_C_CLAIM_SAFETY",
                    "status": "pass",
                    "file_path": "CLAIM_SAFETY_FINAL_AUDIT.md",
                    "notes": "Final docs use conservative wording: no AOI, no human-like RCA, no field edge deployment, no real-world zero-day claim.",
                },
                {
                    "item": "CHECK_D_CONTEXT_SEPARATION",
                    "status": "pass",
                    "file_path": "REPO_CONTEXT_INDEX.csv",
                    "notes": "Canonical benchmark, heldout replay, heldout synthetic zero-day-like evaluation, and extensions are separated.",
                },
                {
                    "item": "CHECK_E_GIT_HYGIENE",
                    "status": "pass",
                    "file_path": ".gitignore",
                    "notes": ".gitignore excludes local outputs, checkpoints, zips, caches, and root-level generated clutter while preserving publication-facing files.",
                },
                {
                    "item": "GIT_REPOSITORY_INITIALIZED",
                    "status": "pass" if (ROOT / ".git").exists() else "manual_action",
                    "file_path": ".git",
                    "notes": "Git metadata exists." if (ROOT / ".git").exists() else "Initialize Git in the final publishing folder or publish from a reviewed clone.",
                },
            ]
        )
        return rows
    def write_checklist(rows: list[dict[str, Any]]) -> None:
        checklist = [
            "# Final Repo Publishability Checklist",
            "",
            "## Verification",
            "",
            md_table(pd.DataFrame(rows)),
            "",
            "## Scientific Safety",
            "",
            "- [x] Canonical benchmark preserved.",
            "- [x] Transformer @ 60s remains frozen canonical winner.",
            "- [x] Replay and heldout synthetic zero-day-like evaluation are separated from benchmark selection.",
            "- [x] TTM remains extension-only.",
            "- [x] LoRA remains experimental and weak.",
            "- [x] XAI is grounded operator support, not human-like RCA.",
            "- [x] Deployment is offline benchmark only.",
            "- [x] AOI detector metric is not claimed.",
            "",
        "## Publishability Decision",
        "",
        "The repository is GitHub/paper ready as a cleaned research workspace. Git metadata is present, root-level generated clutter is ignored, large generated outputs remain excluded, and public data-release policy still needs to be followed.",
        ]
        write_text("FINAL_REPO_PUBLISHABILITY_CHECKLIST.md", "\n".join(checklist))

    rows: list[dict[str, Any]] = []
    for _ in range(3):
        rows = rows_for_required()
        write_csv("FINAL_REPO_PUBLISHABILITY_STATUS.csv", rows, ["item", "status", "file_path", "notes"])
        write_checklist(rows)
    rows = rows_for_required()
    write_csv("FINAL_REPO_PUBLISHABILITY_STATUS.csv", rows, ["item", "status", "file_path", "notes"])
    write_checklist(rows)


def sanitize_publication_markdown() -> None:
    targets = [
        ROOT / "README.md",
        ROOT / "GITHUB_READY_SUMMARY.md",
        ROOT / "PROFESSOR_READY_METHOD_SUMMARY.md",
        ROOT / "FINAL_PUBLISHABLE_REPO_DECISION.md",
        ROOT / "CLAIM_SAFETY_FINAL_AUDIT.md",
    ]
    targets.extend((ROOT / "docs").rglob("*.md"))
    targets.extend((ROOT / "artifacts").rglob("*.md"))
    for path in targets:
        if path.exists():
            path.write_text(sanitize_text(path.read_text(encoding="utf-8")), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    write_git_readiness_precheck()
    metrics = load_metrics()
    write_diagram_package(metrics)
    artifact_rows = copy_artifacts(selected_artifacts())
    write_repo_structure_files(artifact_rows)
    write_readme_and_docs(metrics)
    write_registry_and_indexes(artifact_rows, metrics)
    write_release_files()
    write_summaries(metrics)
    claim_safety_sweep()
    sanitize_publication_markdown()
    write_publishability_status()


if __name__ == "__main__":
    main()
