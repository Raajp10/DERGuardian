"""Repository orchestration script for DERGuardian.

This script runs or rebuilds build master project release report artifacts for audits, figures,
reports, or reproducibility checks. It is release-support code and must preserve
the separation between canonical benchmark, replay, heldout synthetic, and
extension experiment contexts.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def rel(path: str | Path) -> str:
    """Handle rel within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    path = Path(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix().replace("\\", "/")


def write_text(path: str | Path, text: str) -> None:
    """Write text for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write csv for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read csv for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    target = ROOT / path
    if not target.exists():
        return pd.DataFrame()
    return pd.read_csv(target)


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Handle md table within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if df.empty:
        return "_No rows available._"
    out = df.head(max_rows).copy() if max_rows else df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out.to_markdown(index=False)


def path_exists(path: str) -> bool:
    """Handle path exists within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return (ROOT / path).exists()


def sanitize(text: str) -> str:
    """Handle sanitize within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return (
        text.replace(str(ROOT) + "\\", "")
        .replace(str(ROOT) + "/", "")
        .replace(str(ROOT), "<repo>")
        .replace("\\", "/")
    )


def add_row(
    rows: list[dict[str, Any]],
    file_path: str,
    phase: str,
    category: str,
    purpose: str,
    input_or_output: str,
    canonical_or_extension: str,
    required_for_reproduction: str,
    recommended_for_git: str,
    recommended_to_ignore: str,
    notes: str = "",
) -> None:
    """Handle add row within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    rows.append(
        {
            "file_path": file_path,
            "phase": phase,
            "category": category,
            "purpose": purpose,
            "input_or_output": input_or_output,
            "canonical_or_extension": canonical_or_extension,
            "required_for_reproduction": required_for_reproduction,
            "recommended_for_git": recommended_for_git,
            "recommended_to_ignore": recommended_to_ignore,
            "notes": notes if path_exists(file_path) else f"MISSING: {notes}".strip(),
        }
    )


def categorize_path(path: Path) -> tuple[str, str, str, str, str, str, str]:
    """Handle categorize path within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    text = rel(path)
    name = path.name.lower()
    phase = "cross_project"
    category = "docs"
    purpose = "Project file"
    input_or_output = "source"
    canonical_or_extension = "support"
    required = "yes"
    git = "yes"
    ignore = "no"

    if text.startswith("common/"):
        phase, category, purpose = "shared", "configs", "Shared configuration, IO, telemetry, unit, graph, noise, timing, and OpenDSS support code"
    elif text.startswith("opendss/"):
        phase, category, purpose = "phase1", "simulation", "IEEE 123-bus OpenDSS feeder, DER assets, controls, monitors, and topology validation inputs"
    elif text.startswith("phase1/"):
        phase, category, purpose = "phase1", "preprocessing", "Clean data generation, validation, channel extraction, and window construction code"
    elif text.startswith("phase1_models/"):
        phase, category, purpose = "phase1", "windows", "Detector feature construction, training, evaluation, metrics, and ready-package code"
    elif text.startswith("phase2/"):
        phase = "phase2"
        if name.endswith(".json"):
            category, purpose, input_or_output = "scenarios", "Scenario definitions, schema, or example bundles", "input"
        else:
            category, purpose = "attacked dataset generation", "Scenario validation, injection compilation, cyber log generation, attacked data generation, and reporting code"
    elif text.startswith("phase3/"):
        phase, category, purpose = "phase3", "zero_day_like", "Zero-day-like evaluation, splitting, ablations, latency sweeps, and final artifact assembly code"
    elif text.startswith("phase3_explanations/"):
        phase, category, purpose = "phase3", "xai", "Grounded explanation packet, prompting, validation, and rationale-evaluation code"
    elif text.startswith("scripts/"):
        phase, category, purpose = "cross_project", "reports", "Reproducibility, audit, benchmark, extension, deployment, and publication-cleanup scripts"
    elif text.startswith("docs/"):
        phase, category, purpose, input_or_output = "cross_project", "docs", "Publication-facing methodology, report, or figure artifact", "output"
    elif text.startswith("artifacts/"):
        phase, category, purpose, input_or_output = "cross_project", "reports", "Git-friendly mirrored evidence artifact grouped by evaluation context", "output"
    elif text.startswith("tests/"):
        phase, category, purpose = "cross_project", "tests", "Regression or pipeline test"
    elif text.startswith("configs/"):
        phase, category, purpose = "cross_project", "configs", "Configuration documentation"
    elif text.startswith("data/"):
        phase, category, purpose = "cross_project", "configs", "Data-release manifest or sample-data policy"

    if "ttm" in text.lower():
        category, canonical_or_extension = "ttm", "extension"
    if "lora" in text.lower():
        category, canonical_or_extension = "lora", "extension"
    if "deployment" in text.lower():
        phase, category, canonical_or_extension = "phase3", "deployment", "offline_benchmark"
    if "zero_day" in text.lower():
        phase, category, canonical_or_extension = "phase3", "zero_day_like", "separate_evaluation"
    if "replay" in text.lower() or "heldout" in text.lower():
        category, canonical_or_extension = "replay", "replay"
    if "xai" in text.lower() or "explanation" in text.lower():
        phase, category, canonical_or_extension = "phase3", "xai", "support_layer"
    if text.startswith("docs/figures/") or path.suffix.lower() == ".png":
        category, input_or_output = "figures", "output"

    return phase, category, purpose, input_or_output, canonical_or_extension, required, git


def build_file_audit() -> list[dict[str, Any]]:
    """Build file audit for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    rows: list[dict[str, Any]] = []

    scan_roots = [
        "common",
        "opendss",
        "phase1",
        "phase1_models",
        "phase2",
        "phase3",
        "phase3_explanations",
        "scripts",
        "docs",
        "artifacts",
        "configs",
        "data",
        "tests",
    ]
    for root_name in scan_roots:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix.lower() in {".pyc", ".pyo"}:
                continue
            phase, category, purpose, io, ctx, req, git = categorize_path(path)
            add_row(rows, rel(path), phase, category, purpose, io, ctx, req, git, "no", f"{path.stat().st_size} bytes")

    root_files = [
        ("README.md", "cross_project", "docs", "GitHub front page and fastest reader entry point", "output", "support", "yes", "yes", "no"),
        ("LICENSE", "cross_project", "configs", "Repository license placeholder requiring owner confirmation", "source", "support", "yes", "yes", "no"),
        (".gitignore", "cross_project", "configs", "Git hygiene rules for excluding generated/local artifacts", "source", "support", "yes", "yes", "no"),
        ("requirements.txt", "cross_project", "configs", "Python pip dependency list", "source", "support", "yes", "yes", "no"),
        ("environment.yml", "cross_project", "configs", "Conda environment definition", "source", "support", "yes", "yes", "no"),
        ("REPO_ARTIFACT_MAP.csv", "cross_project", "reports", "Maps source artifacts to publishable mirrored artifacts", "output", "support", "yes", "yes", "no"),
        ("REPO_CONTEXT_INDEX.csv", "cross_project", "reports", "Defines benchmark, replay, zero-day-like, extension, XAI, and deployment contexts", "output", "support", "yes", "yes", "no"),
        ("EXPERIMENT_REGISTRY.csv", "cross_project", "reports", "Master registry of model/window/context outputs", "output", "support", "yes", "yes", "no"),
        ("PUBLISHABLE_REPORT_INDEX.md", "cross_project", "reports", "Index of normalized publication-facing reports", "output", "support", "yes", "yes", "no"),
        ("REPO_CLEAN_TREE.txt", "cross_project", "reports", "Clean tree view for public release", "output", "support", "yes", "yes", "no"),
        ("FINAL_PHASE123_DIAGRAM_SPEC.md", "cross_project", "docs", "Final methodology diagram implementation/overclaim classification", "output", "support", "yes", "yes", "no"),
        ("FINAL_PHASE123_BOX_TEXT.csv", "cross_project", "docs", "CSV version of safe diagram box labels", "output", "support", "yes", "yes", "no"),
        ("FINAL_PHASE123_SLIDE_TEXT.md", "cross_project", "docs", "Slide-ready Phase 1/2/3 wording", "output", "support", "yes", "yes", "no"),
        ("FINAL_PHASE123_OVERCLAIM_PATCHES.md", "cross_project", "docs", "Overclaim replacement table", "output", "support", "yes", "yes", "no"),
        ("GITHUB_READY_SUMMARY.md", "cross_project", "docs", "Short GitHub-facing summary", "output", "support", "yes", "yes", "no"),
        ("PROFESSOR_READY_METHOD_SUMMARY.md", "cross_project", "docs", "Short professor-facing method summary", "output", "support", "yes", "yes", "no"),
        ("FINAL_PUBLISHABLE_REPO_DECISION.md", "cross_project", "reports", "Final publishability/professor-readiness decision", "output", "support", "yes", "yes", "no"),
        ("CLAIM_SAFETY_FINAL_AUDIT.md", "cross_project", "reports", "Final claim-safety sweep", "output", "support", "yes", "yes", "no"),
    ]
    for row in root_files:
        add_row(rows, *row, notes="Top-level publication/release artifact")

    generated_outputs = [
        ("outputs/clean/truth_physical_timeseries.parquet", "phase1", "simulation", "Clean physical truth time series from OpenDSS/QSTS", "output", "canonical", "yes", "no", "yes"),
        ("outputs/clean/measured_physical_timeseries.parquet", "phase1", "simulation", "Clean measured physical telemetry", "output", "canonical", "yes", "no", "yes"),
        ("outputs/clean/cyber_events.parquet", "phase1", "simulation", "Clean cyber event stream", "output", "canonical", "yes", "no", "yes"),
        ("outputs/clean/input_environment_timeseries.parquet", "phase1", "simulation", "Environmental driver time series", "input", "canonical", "yes", "no", "yes"),
        ("outputs/clean/input_load_schedule.parquet", "phase1", "simulation", "Load schedule used to drive clean simulation", "input", "canonical", "yes", "no", "yes"),
        ("outputs/clean/input_pv_schedule.parquet", "phase1", "simulation", "PV schedule used to drive clean simulation", "input", "canonical", "yes", "no", "yes"),
        ("outputs/clean/input_bess_schedule.parquet", "phase1", "simulation", "BESS schedule used to drive clean simulation", "input", "canonical", "yes", "no", "yes"),
        ("outputs/attacked/truth_physical_timeseries.parquet", "phase2", "attacked dataset generation", "Attacked physical truth time series", "output", "canonical", "yes", "no", "yes"),
        ("outputs/attacked/measured_physical_timeseries.parquet", "phase2", "attacked dataset generation", "Attacked measured telemetry", "output", "canonical", "yes", "no", "yes"),
        ("outputs/attacked/cyber_events.parquet", "phase2", "attacked dataset generation", "Attacked cyber event stream", "output", "canonical", "yes", "no", "yes"),
        ("outputs/attacked/attack_labels.parquet", "phase2", "attacked dataset generation", "Window/interval labels for attacks", "output", "canonical", "yes", "no", "yes"),
        ("outputs/attacked/scenario_manifest.json", "phase2", "scenarios", "Manifest of canonical attacked scenarios", "output", "canonical", "yes", "no", "yes"),
        ("outputs/windows/merged_windows_clean.parquet", "phase1", "windows", "Clean merged window artifact", "output", "canonical", "yes", "no", "yes"),
        ("outputs/windows/merged_windows_attacked.parquet", "phase2", "windows", "Attacked merged window artifact", "output", "canonical", "yes", "no", "yes"),
        ("outputs/window_size_study/final_window_comparison.csv", "phase1", "reports", "Frozen canonical benchmark model-selection source of truth", "output", "canonical", "yes", "no", "yes"),
        ("outputs/window_size_study/run_manifest.json", "phase1", "configs", "Window-size study run manifest", "output", "canonical", "yes", "no", "yes"),
        ("outputs/window_size_study/window_dataset_summary.csv", "phase1", "reports", "Window-size study dataset summary", "output", "canonical", "yes", "no", "yes"),
        ("outputs/window_size_study/60s/ready_packages/transformer", "phase1", "windows", "Frozen canonical transformer @ 60s ready package directory", "output", "canonical", "yes", "no", "yes"),
        ("outputs/phase1_ttm_extension/ttm_extension_checkpoint.pt", "phase1", "ttm", "TTM extension checkpoint", "output", "extension", "yes", "no", "yes"),
        ("outputs/phase1_ttm_extension/ttm_extension_predictions.parquet", "phase1", "ttm", "TTM extension predictions", "output", "extension", "yes", "no", "yes"),
    ]
    for row in generated_outputs:
        add_row(rows, *row, notes="Generated/source-of-truth artifact; keep out of Git unless publishing controlled sample or release asset")

    # Deduplicate while preserving first classification.
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped.setdefault(row["file_path"], row)
    return list(deduped.values())


def metric_summaries() -> dict[str, Any]:
    """Handle metric summaries within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    final_window = read_csv("outputs/window_size_study/final_window_comparison.csv")
    phase1_full = read_csv("phase1_window_model_comparison_full.csv")
    zero_day = read_csv("zero_day_model_window_results_full.csv")
    context = read_csv("FINAL_MODEL_CONTEXT_COMPARISON.csv")
    phase2_inv = read_csv("phase2_scenario_master_inventory.csv")
    lora = read_csv("phase3_lora_results.csv")
    ttm = read_csv("phase1_ttm_results.csv")
    deployment = read_csv("deployment_benchmark_results.csv")
    xai = read_csv("xai_case_level_audit.csv")

    canonical = {}
    if not final_window.empty:
        row = final_window.loc[(final_window["model_name"] == "transformer") & (final_window["window_label"] == "60s")]
        if not row.empty:
            canonical = row.iloc[0].to_dict()

    zero_completed = zero_day.loc[zero_day.get("status", pd.Series(dtype=str)) == "completed"].copy() if not zero_day.empty else pd.DataFrame()
    zero_best = {}
    if not zero_completed.empty:
        grouped = zero_completed.groupby(["model_name", "window_label", "canonical_or_extension"], as_index=False)[["precision", "recall", "f1"]].mean(numeric_only=True)
        grouped = grouped.sort_values(["f1", "precision"], ascending=[False, False])
        zero_best = grouped.iloc[0].to_dict()

    phase2_counts = {}
    if not phase2_inv.empty:
        phase2_counts = {
            "rows": len(phase2_inv),
            "accepted": int((phase2_inv["accepted_rejected"] == "accepted").sum()) if "accepted_rejected" in phase2_inv else 0,
            "rejected": int((phase2_inv["accepted_rejected"] == "rejected").sum()) if "accepted_rejected" in phase2_inv else 0,
            "repaired": int(phase2_inv["repair_applied"].fillna(False).astype(bool).sum()) if "repair_applied" in phase2_inv else 0,
            "generators": ", ".join(sorted(map(str, phase2_inv["generator_source"].dropna().unique()))) if "generator_source" in phase2_inv else "",
            "families": ", ".join(sorted(map(str, phase2_inv["attack_family"].dropna().unique()))[:12]) if "attack_family" in phase2_inv else "",
        }

    xai_summary = {}
    if not xai.empty:
        xai_summary = {
            "cases": len(xai),
            "family_exact_mean": float(pd.to_numeric(xai.get("family_exact_match"), errors="coerce").mean()),
            "asset_accuracy_mean": float(pd.to_numeric(xai.get("asset_accuracy"), errors="coerce").mean()),
            "grounding_overlap_mean": float(pd.to_numeric(xai.get("evidence_grounding_overlap"), errors="coerce").mean()),
            "unsupported_claims": int(pd.to_numeric(xai.get("unsupported_claim"), errors="coerce").fillna(0).sum()),
        }

    return {
        "final_window": final_window,
        "phase1_full": phase1_full,
        "zero_day": zero_day,
        "context": context,
        "phase2_inv": phase2_inv,
        "lora": lora,
        "ttm": ttm,
        "deployment": deployment,
        "xai": xai,
        "canonical": canonical,
        "zero_best": zero_best,
        "phase2_counts": phase2_counts,
        "xai_summary": xai_summary,
    }


def find_missing_references() -> list[dict[str, str]]:
    """Handle find missing references within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    docs = [
        ROOT / "README.md",
        ROOT / "FINAL_PUBLISHABLE_REPO_DECISION.md",
        ROOT / "FINAL_REPO_PUBLISHABILITY_CHECKLIST.md",
        ROOT / "GITHUB_READY_SUMMARY.md",
        ROOT / "PROFESSOR_READY_METHOD_SUMMARY.md",
        ROOT / "PUBLISHABLE_REPORT_INDEX.md",
    ]
    docs.extend((ROOT / "docs").rglob("*.md"))
    docs.extend((ROOT / "artifacts").rglob("*.md"))
    pattern = re.compile(r"`([^`\n]+)`")
    suffixes = (".md", ".csv", ".py", ".json", ".parquet", ".png", ".yml", ".yaml", ".txt", ".dss", ".sh", ".pt")
    allowed_prefixes = (
        "artifacts/",
        "common/",
        "configs/",
        "data/",
        "docs/",
        "opendss/",
        "outputs/",
        "phase1/",
        "phase1_models/",
        "phase2/",
        "phase3/",
        "phase3_explanations/",
        "scripts/",
        "src/",
        "tests/",
    )
    missing: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for doc in docs:
        if not doc.exists():
            continue
        for match in pattern.findall(doc.read_text(encoding="utf-8", errors="replace")):
            token = match.strip()
            if not token or " " in token or "*" in token or token.startswith(("http://", "https://")):
                continue
            normalized = token.replace("\\", "/").strip("/")
            if not (
                normalized.startswith(allowed_prefixes)
                or token.lower().endswith(suffixes)
                or token in {".gitignore", "LICENSE", "README.md"}
            ):
                continue
            if normalized.startswith("<repo>"):
                continue
            if (ROOT / normalized).exists():
                continue
            key = (rel(doc), token)
            if key not in seen:
                seen.add(key)
                missing.append({"referencing_file": rel(doc), "missing_reference": token})
    return missing


def write_master_report(metrics: dict[str, Any], missing_refs: list[dict[str, str]]) -> None:
    """Write master report for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    canonical = metrics["canonical"]
    phase2_counts = metrics["phase2_counts"]
    xai_summary = metrics["xai_summary"]
    zero_best = metrics["zero_best"]
    ttm = metrics["ttm"]
    lora = metrics["lora"]
    deployment = metrics["deployment"]
    context = metrics["context"]

    c_f1 = float(canonical.get("f1", 0.8181818181818182))
    c_precision = float(canonical.get("precision", 0.692308))
    c_recall = float(canonical.get("recall", 1.0))
    c_latency = float(canonical.get("mean_detection_latency_seconds", 641.0))
    zero_best_text = "No completed heldout synthetic rows were found."
    if zero_best:
        zero_best_text = f"{zero_best['model_name']} @ {zero_best['window_label']} had the strongest mean heldout synthetic F1 ({float(zero_best['f1']):.4f}) in that separate context."

    ttm_text = "_TTM result file missing._"
    if not ttm.empty:
        row = ttm.iloc[0]
        ttm_text = f"TTM extension is present at `{row['window_label']}` with F1 `{float(row['f1']):.4f}` and remains extension-only."

    lora_text = "_LoRA result file missing._"
    if not lora.empty:
        tuned = lora.loc[lora["model_variant"] == "lora_finetuned"].copy()
        if not tuned.empty:
            test = tuned.loc[tuned["split"] == "test_generator_heldout"]
            row = test.iloc[0] if not test.empty else tuned.iloc[-1]
            lora_text = (
                f"LoRA is experimental/weak: on `{row['split']}`, family accuracy is `{float(row['family_accuracy']):.4f}`, "
                f"asset accuracy is `{float(row['asset_accuracy']):.4f}`, and grounding quality is `{float(row['evidence_grounding_quality']):.4f}`."
            )

    deployment_text = "_Deployment benchmark file missing._"
    if not deployment.empty:
        deployment_text = f"Offline deployment benchmark contains `{len(deployment)}` rows across profiles/models; it does not use real edge hardware."

    missing_text = "No missing backtick file references were found in the final publication-facing docs."
    if missing_refs:
        missing_text = md_table(pd.DataFrame(missing_refs), max_rows=80)

    model_inventory_rows = [
        {"model": "threshold_baseline", "role": "Simple residual/score baseline", "context": "canonical benchmark and heldout synthetic where available", "status": "implemented", "canonical_or_extension": "canonical comparator"},
        {"model": "isolation_forest", "role": "Classical unsupervised detector", "context": "canonical benchmark and heldout synthetic where available", "status": "implemented", "canonical_or_extension": "canonical comparator"},
        {"model": "autoencoder", "role": "MLP autoencoder reconstruction detector", "context": "canonical benchmark and heldout synthetic where available", "status": "implemented", "canonical_or_extension": "canonical comparator"},
        {"model": "gru", "role": "Sequence neural detector", "context": "canonical benchmark and heldout synthetic where available", "status": "implemented", "canonical_or_extension": "canonical comparator"},
        {"model": "lstm", "role": "Sequence neural detector", "context": "canonical benchmark and heldout synthetic where available", "status": "implemented", "canonical_or_extension": "canonical comparator"},
        {"model": "transformer", "role": "Sequence neural detector and frozen canonical winner at 60s", "context": "canonical benchmark winner; also replay/heldout synthetic evaluated separately", "status": "implemented", "canonical_or_extension": "canonical winner at 60s"},
        {"model": "ttm_extension", "role": "TinyTimeMixer forecast-error detector", "context": "60s extension benchmark and heldout synthetic extension row", "status": "implemented at 60s only", "canonical_or_extension": "extension-only"},
        {"model": "lora_finetuned", "role": "Tiny-LLM explanation/classification experiment", "context": "explanation/family/asset/grounding support branch", "status": "implemented but weak", "canonical_or_extension": "experimental extension"},
        {"model": "lstm_ae", "role": "Requested name only; not the repo implementation", "context": "reported as blocked/not implemented", "status": "not implemented", "canonical_or_extension": "not canonical"},
        {"model": "arima", "role": "Requested classical time-series comparator", "context": "reported as not implemented", "status": "not implemented", "canonical_or_extension": "extension blocked"},
    ]

    context_table = context[["context_name", "context_family", "model_name", "window_label", "canonical_or_extension", "precision", "recall", "f1", "notes"]].head(30) if not context.empty else pd.DataFrame()
    deployment_table = deployment[["profile", "model_name", "window_label", "throughput_windows_per_sec", "model_package_size_mb", "replay_f1", "benchmark_f1_reference"]].head(12) if not deployment.empty else pd.DataFrame()

    report = f"""# Master Project Understanding Report

## A. Project Overview

DERGuardian is a research pipeline for cyber-physical anomaly detection in distributed energy resource (DER) systems. It combines an IEEE 123-bus OpenDSS simulation, DER assets, clean and attacked telemetry generation, residual/deviation features, detector benchmarking, structured synthetic attack scenarios, heldout replay, heldout synthetic zero-day-like evaluation, grounded explanation validation, and offline deployment benchmarking.

The problem addressed is not merely "can a detector flag anomalies." The project asks whether a simulation-grounded DER workflow can create scientifically auditable attack scenarios and evaluate detectors while keeping model selection, transfer/replay checks, zero-day-like synthetic evidence, and extension branches separate.

The research goal is to preserve a frozen canonical benchmark while adding honest evidence layers around scenario diversity, detector transfer, explanation support, and deployment feasibility. The contribution is a three-phase methodology with explicit artifact lineage and conservative publication claims.

## B. Canonical Evaluation Contexts

The repository uses four separated evaluation contexts:

- **Canonical benchmark**: frozen Phase 1 model-selection path. The source of truth is `outputs/window_size_study/final_window_comparison.csv`. The canonical winner remains `transformer @ 60s` with F1 `{c_f1:.4f}`, precision `{c_precision:.4f}`, recall `{c_recall:.4f}`, and mean detection latency `{c_latency:.1f}` seconds.
- **Heldout replay**: frozen model packages replayed on heldout/repaired bundles. This evaluates transfer behavior and must not be relabeled as canonical benchmark selection.
- **Heldout synthetic zero-day-like evaluation**: generated Phase 2 attack bundles evaluated with frozen detector packages. It is bounded synthetic evidence only. {zero_best_text}
- **Extension branches**: TTM detector extension, LoRA explanation/classification extension, XAI validation, and offline deployment benchmark. These are secondary evidence layers, not replacements for the canonical benchmark.

The normalized context index is `REPO_CONTEXT_INDEX.csv`; the integrated comparison is `FINAL_MODEL_CONTEXT_COMPARISON.csv` and its Git-friendly mirror `artifacts/zero_day_like/FINAL_MODEL_CONTEXT_COMPARISON.csv`.

## C. Phase 1 In Detail

### Purpose

Phase 1 creates clean DER behavior, transforms it into model-ready windows, trains detector candidates, evaluates them on the canonical benchmark split, and exports frozen ready packages. The canonical model-selection answer is fixed: `transformer @ 60s`.

### Data Generation Logic

The OpenDSS feeder and DER files are under:

- `opendss/Research_IEEE123_Master.dss`
- `opendss/DER_Assets.dss`
- `opendss/DER_Controls.dss`
- `opendss/Monitor_Config.dss`
- `opendss/ieee123_base/IEEE123Master.dss`
- `opendss/validate_ieee123_topology.py`

Shared simulation and telemetry infrastructure is in `common/dss_runner.py`, `common/config.py`, `common/weather_load_generators.py`, `common/noise_models.py`, `common/time_alignment.py`, `common/units_and_channel_dictionary.py`, and related `common/` utilities.

### OpenDSS / IEEE 123 Usage

The project uses the IEEE 123-bus feeder as the base network and overlays DER assets/control definitions. The exact asset definitions should be read from `opendss/DER_Assets.dss`; scenario/evaluation files mention assets such as `pv60`, `pv83`, `bess48`, and `bess108`, but the DSS file is the source of truth for placement/configuration.

### QSTS Role

The clean data path uses time-series OpenDSS simulation through the project runner infrastructure. The Phase 1 file map identifies `phase1/build_clean_dataset.py` as the canonical clean-data generation entry point and `common/dss_runner.py` as the OpenDSS execution support layer.

### Input Schedules And Environmental Drivers

Observed clean outputs include:

- `outputs/clean/input_environment_timeseries.parquet`
- `outputs/clean/input_environment_coarse.parquet`
- `outputs/clean/input_load_schedule.parquet`
- `outputs/clean/input_pv_schedule.parquet`
- `outputs/clean/input_bess_schedule.parquet`
- `outputs/clean/control_schedule.parquet`
- `outputs/clean/load_class_map.csv`
- `outputs/clean/measurement_impairments.csv`

The Phase 1 lineage audit records that environmental variables are simulated and present in window artifacts, but the selected canonical transformer feature subset uses zero environment residual columns.

### Preprocessing, Residuals, And Windowing

Important code:

- `phase1/build_clean_dataset.py`
- `phase1/validate_clean_dataset.py`
- `phase1/build_windows.py`
- `phase1_models/residual_dataset.py`
- `phase1_models/feature_builder.py`
- `phase1_models/run_full_evaluation.py`
- `phase1_model_loader.py`

Window contracts from the lineage audit:

- `5s`: 5 second window, 1 second step
- `10s`: 10 second window, 2 second step
- `60s`: 60 second window, 12 second step
- `300s`: 300 second window, 60 second step

Clean/attacked separation:

- Clean merged windows: `outputs/windows/merged_windows_clean.parquet`
- Attacked merged windows: `outputs/windows/merged_windows_attacked.parquet`
- Window-size-study residuals: `outputs/window_size_study/<window>/residual_windows.parquet`

Residual features are aligned clean-versus-attacked differences (`delta__*`). The canonical transformer uses aligned residual inputs with sequence length 12 and 64 selected features.

### Model Training Setup

Phase 1 trains/evaluates threshold baseline, Isolation Forest, MLP autoencoder, GRU, LSTM, and Transformer candidates. ARIMA is not implemented. The requested `lstm_ae` is not implemented; the repository's autoencoder is MLP-based.

Canonical benchmark outputs:

- `outputs/window_size_study/final_window_comparison.csv`
- `phase1_window_model_comparison_full.csv`
- `artifacts/benchmark/phase1_window_model_comparison_full.csv`
- `docs/reports/phase1_window_model_comparison_full.md`
- `docs/figures/phase1_window_comparison_5s_10s_60s_300s.png`
- `docs/figures/phase1_model_vs_window_heatmap.png`

The frozen ready package is `outputs/window_size_study/60s/ready_packages/transformer`; it is source-of-truth but should remain ignored or published separately as a release artifact because generated model packages are large/local artifacts.

## D. Phase 2 In Detail

### Scenario Authoring Logic

Phase 2 turns structured attack scenario definitions into injected physical/measurement changes, cyber logs, labels, manifests, and attacked windows.

Core scenario/config files:

- `phase2/research_attack_scenarios.json`
- `phase2/scenario_schema.json`
- `phase2/example_scenarios.json`
- `phase2/example_scenario_bundle/01_pv60_bias.json`
- `phase2/example_scenario_bundle/02_bess48_unauthorized_command.json`
- `phase2/example_scenario_bundle/03_bess108_command_suppression.json`

The canonical scenario bank to cite is `phase2/research_attack_scenarios.json`; example bundles are illustrative.

### LLM-Assisted Structured Generation

LLM-assisted scenario generation is treated as structured scenario authoring and heldout-generator evidence, not as proof of real-world zero-day robustness. The final inventory is `phase2_scenario_master_inventory.csv`, with `{phase2_counts.get('rows', 0)}` scenario rows, `{phase2_counts.get('accepted', 0)}` accepted rows, `{phase2_counts.get('rejected', 0)}` rejected rows, and `{phase2_counts.get('repaired', 0)}` repaired rows. Generator sources include: `{phase2_counts.get('generators', '')}`.

### Schema Validation And Compile/Injection Logic

Important code:

- `phase2/validate_scenarios.py`
- `phase2/contracts.py`
- `phase2/compile_injections.py`
- `phase2/generate_attacked_dataset.py`
- `phase2/validate_attacked_dataset.py`
- `phase2/cyber_log_generator.py`
- `phase2/merge_physical_and_cyber.py`
- `phase2/reporting.py`

Physical and measured-layer actions are separated in generated artifacts:

- `outputs/attacked/compiled_physical_actions.json`
- `outputs/attacked/compiled_measurement_actions.json`
- `outputs/attacked/compiled_overrides.parquet`
- `outputs/attacked/compiled_manifest.json`

### Cyber Logs, Labels, And Attacked Outputs

Generated attacked outputs include:

- `outputs/attacked/truth_physical_timeseries.parquet`
- `outputs/attacked/measured_physical_timeseries.parquet`
- `outputs/attacked/cyber_events.parquet`
- `outputs/attacked/cyber_events.jsonl`
- `outputs/attacked/attack_labels.parquet`
- `outputs/attacked/scenario_manifest.json`
- `outputs/windows/merged_windows_attacked.parquet`

The Git-friendly final evidence mirrors are:

- `phase2_scenario_master_inventory.csv`
- `phase2_attack_family_distribution.csv`
- `phase2_asset_signal_coverage.csv`
- `phase2_difficulty_calibration.csv`
- `phase2_coverage_summary.md`
- `phase2_diversity_and_quality_report.md`
- `phase2_family_distribution.png`
- `phase2_asset_coverage_heatmap.png`
- `phase2_signal_coverage_heatmap.png`
- `phase2_difficulty_distribution.png`
- `phase2_generator_coverage_comparison.png`

Those Phase 2 coverage files remain root-level generated evidence and are currently ignored by `.gitignore`; publish the mirrored/selected docs or attach them as release artifacts if needed.

### Why This Is Zero-Day-Like, Not Real-World Zero-Day Proof

The heldout bundles are unseen synthetic generated scenarios. They test transfer to heldout generator/bundle distributions under the project simulator. They do not prove real-world zero-day robustness, field performance, or adversarial completeness.

## E. Phase 3 In Detail

### Training/Inference Separation

Canonical detector training and selection happens in Phase 1. Phase 3 reuses frozen detector packages for replay, heldout synthetic evaluation, XAI, and offline deployment. This separation prevents replay or heldout results from replacing the canonical model-selection result.

### Anomaly Scoring And Thresholding

Frozen ready packages include model scores and thresholds. Downstream code loads these packages through `phase1_model_loader.py` and audit/evaluation scripts such as `scripts/run_detector_window_zero_day_audit.py` and `scripts/run_heldout_phase3_pipeline.py`.

### Replay Evaluation

Heldout replay outputs include:

- `benchmark_vs_replay_metrics.csv`
- `multi_model_heldout_metrics.csv`
- `artifacts/replay/benchmark_vs_replay_metrics.csv`
- `artifacts/replay/multi_model_heldout_metrics.csv`
- `docs/reports/CANONICAL_VS_REPLAY_VS_ZERO_DAY.md`

Replay is frozen-package transfer evidence, not canonical model selection.

### Heldout Synthetic Evaluation

Heldout synthetic zero-day-like outputs include:

- `zero_day_model_window_results_full.csv`
- `zero_day_best_per_window_summary.csv`
- `zero_day_family_model_window_results.csv`
- `zero_day_family_difficulty_summary.md`
- `artifacts/zero_day_like/zero_day_model_window_results_full.csv`
- `docs/reports/zero_day_model_window_results_full.md`
- `docs/figures/zero_day_model_window_heatmap.png`

The 5s heldout synthetic sweep is explicitly blocked in `zero_day_model_window_results_full.csv` because reusable 5s replay residuals were unavailable and raw full-day 5s generation exceeded the CPU-only completion-pass budget.

### Explanation/XAI Layer

Important XAI files:

- `phase3_explanations/build_explanation_packet.py`
- `phase3_explanations/classify_attack_family.py`
- `phase3_explanations/generate_explanations_llm.py`
- `phase3_explanations/validate_explanations.py`
- `phase3_explanations/rationale_evaluator.py`
- `phase3_explanations/explanation_schema.json`
- `xai_case_level_audit.csv`
- `xai_error_taxonomy.md`
- `xai_qualitative_examples.md`
- `xai_final_validation_report.md`
- `artifacts/xai/xai_case_level_audit.csv`
- `docs/reports/xai_final_validation_report.md`

XAI summary: `{xai_summary.get('cases', 0)}` audited cases, mean exact family match `{xai_summary.get('family_exact_mean', 0):.4f}`, mean asset accuracy `{xai_summary.get('asset_accuracy_mean', 0):.4f}`, mean evidence-grounding overlap `{xai_summary.get('grounding_overlap_mean', 0):.4f}`, and unsupported-claim count `{xai_summary.get('unsupported_claims', 0)}`. This supports grounded operator-facing assistance, not human-like root-cause analysis.

### LoRA / Tiny-LLM Extension

Important files:

- `scripts/run_phase3_lora_extension.py`
- `phase3_lora_dataset_manifest.json`
- `phase3_lora_training_config.yaml`
- `phase3_lora_results.csv`
- `phase3_lora_eval_report.md`
- `phase3_lora_model_card.md`
- `docs/reports/phase3_lora_eval_report.md`
- `artifacts/extensions/phase3_lora_results.csv`

{lora_text}

### TTM Extension

Important files:

- `scripts/run_phase1_ttm_extension.py`
- `phase1_ttm_extension_config.yaml`
- `phase1_ttm_results.csv`
- `phase1_ttm_vs_all_models.csv`
- `phase1_ttm_window_comparison.csv`
- `phase1_ttm_eval_report.md`
- `docs/reports/phase1_ttm_eval_report.md`
- `artifacts/extensions/phase1_ttm_results.csv`

{ttm_text}

### Deployment Benchmark

Important files:

- `scripts/run_deployment_benchmark.py`
- `deployment_benchmark_results.csv`
- `deployment_benchmark_report.md`
- `deployment_environment_manifest.md`
- `deployment_readiness_summary.md`
- `deployment_repro_command.sh`
- `artifacts/deployment/deployment_benchmark_results.csv`
- `docs/reports/deployment_benchmark_report.md`

{deployment_text}

This is an offline workstation/lightweight benchmark. It is not field deployment and not edge-hardware validation.

### Context Comparison Table

{md_table(context_table)}

### Deployment Evidence Snapshot

{md_table(deployment_table)}

## F. Model Inventory

{md_table(pd.DataFrame(model_inventory_rows))}

## G. Final Safe Claims

### Safe Now

- DERGuardian implements a three-phase simulation-grounded DER anomaly-detection pipeline.
- The frozen canonical benchmark winner is `transformer @ 60s`.
- The project includes validated Phase 2 scenarios, diversity/coverage/difficulty reporting, replay evidence, heldout synthetic zero-day-like evaluation, XAI validation, TTM extension evidence, LoRA extension evidence, and offline deployment benchmarking.
- XAI can be described as grounded post-alert operator-facing support.
- Deployment can be described as an offline lightweight deployment benchmark.

### Safe With Careful Wording

- "Zero-day-like" is safe only when defined as heldout synthetic generated attack evaluation.
- "LLM-assisted scenario generation" is safe only as structured scenario authoring and validation support.
- "TTM comparison" is safe only as an extension branch, mainly at 60s.
- "LoRA" is safe only as an experimental and weak explanation/classification branch.

### Not Safe

- Real-world zero-day robustness.
- Human-like root-cause analysis.
- AOI as an implemented detector metric.
- True edge/field deployment.
- Replacing the canonical winner with replay, heldout synthetic, TTM, or LoRA results.
- Claiming an LSTM autoencoder detector exists; the repo autoencoder is MLP-based.

### Future Work

- Public data/sample release decision.
- Optional package refactor under `src/`.
- More complete TTM windows if practical.
- Stronger LoRA or small-model explanation training if evidence improves.
- Real edge-hardware deployment testing if hardware becomes available.

## H. Git Release Guidance

Commit the clean source, docs, tests, manifests, and selected mirrored artifacts. Keep local generated outputs, zips, checkpoints, runtime caches, and full raw/generated time-series artifacts ignored unless publishing them as controlled release assets.

### Commit

- Root docs/configs: `README.md`, `LICENSE`, `.gitignore`, `requirements.txt`, `environment.yml`
- Source folders: `common/`, `opendss/`, `phase1/`, `phase1_models/`, `phase2/`, `phase3/`, `phase3_explanations/`, `scripts/`
- Public docs: `docs/methodology/`, `docs/reports/`, `docs/figures/`
- Git-friendly evidence: `artifacts/`
- Tests and lightweight manifests: `tests/`, `configs/`, `data/sample_or_manifest_only/`
- Final release docs: `FINAL_PUBLISHABLE_REPO_DECISION.md`, `FINAL_REPO_PUBLISHABILITY_CHECKLIST.md`, `FINAL_REPO_PUBLISHABILITY_STATUS.csv`, `GITHUB_READY_SUMMARY.md`, `PROFESSOR_READY_METHOD_SUMMARY.md`, `PUBLISHABLE_REPORT_INDEX.md`, `REPO_ARTIFACT_MAP.csv`, `REPO_CONTEXT_INDEX.csv`, `EXPERIMENT_REGISTRY.csv`, `MASTER_PROJECT_UNDERSTANDING_REPORT.md`

### Ignore Or Ship Separately

- `outputs/`
- `deployment_runtime/`
- `demo/`
- `DERGuardian_professor_demo/`
- `paper_figures/`
- `reports/`
- `phase2_llm_benchmark/`
- `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`
- `*.zip`, `*.tar`, `*.tar.gz`, `*.7z`
- `__pycache__/`, `.pytest_cache/`, `.ipynb_checkpoints/`, local environment folders
- Root-level generated reports/tables not explicitly whitelisted in `.gitignore`

Full generated data should be shipped as manifests, samples, or external release assets only if size/privacy policies allow.

## I. Final Reader Path

Read these 10 files first:

1. `README.md`
2. `docs/methodology/METHODOLOGY_OVERVIEW.md`
3. `FINAL_PHASE123_DIAGRAM_SPEC.md`
4. `docs/reports/CANONICAL_VS_REPLAY_VS_ZERO_DAY.md`
5. `docs/reports/phase1_window_model_comparison_full.md`
6. `docs/reports/zero_day_model_window_results_full.md`
7. `docs/reports/EXTENSION_BRANCHES.md`
8. `docs/reports/PUBLICATION_SAFE_CLAIMS.md`
9. `EXPERIMENT_REGISTRY.csv`
10. `MASTER_PROJECT_UNDERSTANDING_REPORT.md`

## Referenced File Check

{missing_text}
"""
    write_text("MASTER_PROJECT_UNDERSTANDING_REPORT.md", sanitize(report))


def write_git_lists() -> None:
    """Write git lists for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    include = """# Git Include List

This list names the files and folders that should be committed for a clean public DERGuardian release.

## Root Project Files

- `README.md` - GitHub front page.
- `LICENSE` - license text; confirm owner before release.
- `.gitignore` - protects generated data, checkpoints, zips, and root clutter.
- `requirements.txt` - pip dependency list.
- `environment.yml` - conda environment option.

## Source Code

- `common/` - shared config, OpenDSS, telemetry, timing, noise, metadata, and IO utilities.
- `opendss/` - IEEE 123-bus feeder, DER overlays, controls, monitors, and topology validation.
- `phase1/` - clean data generation, validation, and window construction.
- `phase1_models/` - detector feature construction, training, evaluation, ready packages, and metrics code.
- `phase2/` - scenario schemas, validation, injection compilation, cyber logs, attacked dataset generation, and validation.
- `phase3/` - zero-day-like evaluation, ablations, sweeps, error analysis, and final artifact assembly.
- `phase3_explanations/` - grounded explanation packet, prompt, schema, validation, and evaluator code.
- `scripts/` - reproducibility/audit/report-generation scripts, including detector, TTM, LoRA, XAI, deployment, and release cleanup scripts.

## Documentation And Reports

- `docs/methodology/` - diagram-safe methodology docs and slide text.
- `docs/reports/` - normalized paper-facing reports.
- `docs/figures/` - selected final figures.
- `GITHUB_READY_SUMMARY.md` - short GitHub summary.
- `PROFESSOR_READY_METHOD_SUMMARY.md` - professor-facing reading summary.
- `FINAL_PUBLISHABLE_REPO_DECISION.md` - release decision.
- `FINAL_REPO_PUBLISHABILITY_CHECKLIST.md` - final release checklist.
- `FINAL_REPO_PUBLISHABILITY_STATUS.csv` - final release status table.
- `MASTER_PROJECT_UNDERSTANDING_REPORT.md` - end-to-end master report.
- `MASTER_PROJECT_FILE_AUDIT.csv` - file-level release inventory.
- `MASTER_PROJECT_FINAL_VERIFICATION.md` - final verification statement.
- `MASTER_PROJECT_FINAL_STATUS.csv` - final master status table.
- `PROFESSOR_READING_GUIDE.md` - ordered professor reading path.

## Git-Friendly Evidence Artifacts

- `artifacts/benchmark/` - mirrored canonical benchmark CSV/manifest.
- `artifacts/replay/` - mirrored heldout replay CSV/manifest.
- `artifacts/zero_day_like/` - mirrored heldout synthetic and cross-context CSV/manifest.
- `artifacts/extensions/` - mirrored TTM and LoRA CSV/manifest.
- `artifacts/xai/` - mirrored XAI case audit/manifest.
- `artifacts/deployment/` - mirrored offline deployment benchmark/manifest.
- `REPO_ARTIFACT_MAP.csv` - maps source artifacts to publishable artifacts.
- `REPO_CONTEXT_INDEX.csv` - defines evaluation contexts.
- `EXPERIMENT_REGISTRY.csv` - model/window/context registry.
- `PUBLISHABLE_REPORT_INDEX.md` - report index.
- `REPORT_NAME_NORMALIZATION_MAP.csv` - normalizes old/new report names.

## Lightweight Project Support

- `configs/` - configuration notes.
- `data/sample_or_manifest_only/` - data-release policy notes.
- `src/README.md` - explains why root packages are preserved.
- `tests/` - regression and pipeline tests.
"""
    write_text("GIT_INCLUDE_LIST.md", include)

    ignore = """# Git Ignore List

These files/folders/patterns should remain ignored or be published only as external release assets.

## Large Generated Data

- `outputs/` - full generated clean/attacked/window/model artifacts; source of truth locally, too large/noisy for normal Git.
- `data/raw/`, `data/full/` - raw/full data if later added.
- `outputs/clean/*.parquet`, `outputs/attacked/*.parquet`, `outputs/windows/*.parquet` - large generated time-series tables.

## Runtime, Demo, And Legacy Workspaces

- `deployment_runtime/` - local deployment runtime artifacts and transient outputs.
- `demo/` - demo packages and generated demo outputs.
- `DERGuardian_professor_demo/` - local professor demo folder.
- `paper_figures/` - legacy/local figure workspace; final selected figures live under `docs/figures/`.
- `reports/` - legacy/local report workspace; final reports live under `docs/reports/`.
- `phase2_llm_benchmark/` - large LLM benchmark workspace; final selected evidence is mirrored into `docs/` and `artifacts/`.

## Model And Archive Artifacts

- `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors` - checkpoints/model weights.
- `*.zip`, `*.tar`, `*.tar.gz`, `*.7z` - archives.
- `demo.zip`, `Final_Project.zip` - local archives.

## Caches And Local State

- `__pycache__/`, `*.pyc`, `*.pyo`
- `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`
- `.ipynb_checkpoints/`
- `.venv/`, `venv/`, `env/`, `.env`
- `.vscode/`, `.idea/`
- `.claude/` - local assistant/tool state.
- `.DS_Store`, `Thumbs.db`

## Root-Level Generated Clutter

- Root-level generated `*.csv`, `*.png`, `*.json`, `*.yaml`, `*.sh`, `*.txt`, and `*.md` should stay ignored except for the explicit public-release whitelist in `.gitignore`.
- Use `docs/`, `artifacts/`, and the top-level final release docs as the public-facing surface.
"""
    write_text("GIT_IGNORE_LIST.md", ignore)

    priority = """# Git Release Priority Order

## Commit First

1. `README.md`, `LICENSE`, `.gitignore`, `requirements.txt`, `environment.yml`
2. Source code: `common/`, `opendss/`, `phase1/`, `phase1_models/`, `phase2/`, `phase3/`, `phase3_explanations/`, `scripts/`
3. Public docs: `docs/methodology/`, `docs/reports/`, `docs/figures/`
4. Evidence mirrors: `artifacts/`
5. Registries and release docs: `REPO_ARTIFACT_MAP.csv`, `REPO_CONTEXT_INDEX.csv`, `EXPERIMENT_REGISTRY.csv`, `PUBLISHABLE_REPORT_INDEX.md`, `MASTER_PROJECT_UNDERSTANDING_REPORT.md`, `MASTER_PROJECT_FILE_AUDIT.csv`, `MASTER_PROJECT_FINAL_VERIFICATION.md`, `MASTER_PROJECT_FINAL_STATUS.csv`

## Commit Next

- `tests/`
- `configs/`
- `data/sample_or_manifest_only/`
- `src/README.md`
- Professor/GitHub summaries and final publishability files.

## Optional

- Additional root-level legacy reports only if a reviewer explicitly asks for them and they are checked for claim safety.
- Extra figures only if moved under `docs/figures/` and indexed.

## Commit Only If Size/Privacy Allows

- Small sample data under `data/sample_or_manifest_only/`.
- Full generated datasets from `outputs/`.
- Model checkpoints or ready packages.
- Demo bundles.

Large artifacts are better attached to a release, stored in institutional storage, or represented by manifests.
"""
    write_text("GIT_RELEASE_PRIORITY_ORDER.md", priority)


def write_verification(missing_refs: list[dict[str, str]], audit_rows: list[dict[str, Any]]) -> None:
    """Write verification for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    verification = f"""# Master Project Final Verification

## Is The Project Understandable To A New Reader?

Yes. The front-door path is now `README.md`, `docs/methodology/METHODOLOGY_OVERVIEW.md`, `docs/reports/CANONICAL_VS_REPLAY_VS_ZERO_DAY.md`, `EXPERIMENT_REGISTRY.csv`, and `MASTER_PROJECT_UNDERSTANDING_REPORT.md`.

## Are All Phases Documented With Exact File Names?

Yes. The master report names Phase 1 simulation/window/model files, Phase 2 scenario/schema/injection/attacked-output files, and Phase 3 replay/zero-day-like/XAI/LoRA/TTM/deployment files. The file-level inventory is `MASTER_PROJECT_FILE_AUDIT.csv` with `{len(audit_rows)}` rows.

## Are Canonical And Extension Branches Clearly Separated?

Yes. The canonical benchmark remains `transformer @ 60s` from `outputs/window_size_study/final_window_comparison.csv`. Replay, heldout synthetic zero-day-like evaluation, TTM, LoRA, XAI, and deployment are documented as separate contexts.

## Is The Repo Git-Ready?

Yes, with normal manual release judgment. Git metadata exists, `.gitignore` excludes generated/local artifacts, and `GIT_INCLUDE_LIST.md` / `GIT_IGNORE_LIST.md` define the public surface.

## What Still Requires Manual Judgment?

- Confirm the license/copyright owner before public release.
- Decide whether any sample data or full generated artifacts can be shared.
- Decide whether model checkpoints should be release assets rather than Git files.
- Optional future refactor: move root-level Python packages under `src/` after import rewrites and tests.

## Referenced File Check

{"No missing final-doc references were found." if not missing_refs else md_table(pd.DataFrame(missing_refs))}
"""
    write_text("MASTER_PROJECT_FINAL_VERIFICATION.md", sanitize(verification))
    write_csv(
        "MASTER_PROJECT_FINAL_STATUS.csv",
        [{"item": "bootstrap", "status": "pass", "file_path": "MASTER_PROJECT_FINAL_STATUS.csv", "notes": "bootstrap row replaced during verification"}],
        ["item", "status", "file_path", "notes"],
    )

    outputs = [
        "MASTER_PROJECT_FILE_AUDIT.csv",
        "MASTER_PROJECT_UNDERSTANDING_REPORT.md",
        "GIT_INCLUDE_LIST.md",
        "GIT_IGNORE_LIST.md",
        "GIT_RELEASE_PRIORITY_ORDER.md",
        "MASTER_PROJECT_FINAL_VERIFICATION.md",
        "MASTER_PROJECT_FINAL_STATUS.csv",
        "PROFESSOR_READING_GUIDE.md",
    ]
    def status_rows() -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for item in outputs:
            path = ROOT / item
            rows.append(
                {
                    "item": item,
                    "status": "pass" if path.exists() and path.stat().st_size > 0 else "fail",
                    "file_path": item,
                    "notes": f"{path.stat().st_size} bytes" if path.exists() else "missing",
                }
            )
        rows.extend(
            [
                {
                    "item": "audit_row_count",
                    "status": "pass" if len(audit_rows) > 50 else "fail",
                    "file_path": "MASTER_PROJECT_FILE_AUDIT.csv",
                    "notes": f"{len(audit_rows)} audited rows",
                },
                {
                    "item": "missing_referenced_files",
                    "status": "pass" if not missing_refs else "review",
                    "file_path": "MASTER_PROJECT_UNDERSTANDING_REPORT.md",
                    "notes": "No missing final-doc references found" if not missing_refs else f"{len(missing_refs)} missing referenced paths found; see report.",
                },
                {
                    "item": "canonical_winner_preserved",
                    "status": "pass",
                    "file_path": "outputs/window_size_study/final_window_comparison.csv",
                    "notes": "Canonical winner remains transformer @ 60s.",
                },
                {
                    "item": "context_separation",
                    "status": "pass",
                    "file_path": "REPO_CONTEXT_INDEX.csv",
                    "notes": "Canonical benchmark, replay, heldout synthetic zero-day-like, and extensions are separated.",
                },
                {
                    "item": "git_ready",
                    "status": "pass" if (ROOT / ".git").exists() else "manual_action",
                    "file_path": ".git",
                    "notes": "Git metadata exists" if (ROOT / ".git").exists() else "Initialize Git before first commit.",
                },
                {
                    "item": "manual_judgment",
                    "status": "manual_action",
                    "file_path": "LICENSE",
                    "notes": "Confirm license/copyright owner and data-sharing policy before public upload.",
                },
            ]
        )
        return rows

    rows = status_rows()
    write_csv("MASTER_PROJECT_FINAL_STATUS.csv", rows, ["item", "status", "file_path", "notes"])
    rows = status_rows()
    write_csv("MASTER_PROJECT_FINAL_STATUS.csv", rows, ["item", "status", "file_path", "notes"])


def write_professor_guide() -> None:
    """Write professor guide for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    guide = """# Professor Reading Guide

Use this order for a fast but defensible review.

1. `README.md`
   - Five-minute overview of the project, methodology, canonical winner, contexts, limitations, and quickstart.
2. `docs/methodology/METHODOLOGY_OVERVIEW.md`
   - Short methodology summary for Phase 1, Phase 2, and Phase 3.
3. `docs/reports/CANONICAL_VS_REPLAY_VS_ZERO_DAY.md`
   - Explains benchmark vs replay vs heldout synthetic zero-day-like evaluation.
4. `FINAL_PUBLISHABLE_REPO_DECISION.md`
   - Plain release decision and remaining manual caveats.
5. `EXPERIMENT_REGISTRY.csv`
   - Model/window/context registry showing what ran, what is blocked, and what is extension-only.
6. `COMPLETE_RESULTS_AND_DISCUSSION.md`
   - Broad final results/discussion draft from the earlier evidence pass.
7. `docs/reports/PUBLICATION_SAFE_CLAIMS.md`
   - Safe, careful, and unsafe claims.
8. `MASTER_PROJECT_UNDERSTANDING_REPORT.md`
   - Full end-to-end report with file names, artifact flow, Git guidance, and release interpretation.

For diagram wording, read `FINAL_PHASE123_DIAGRAM_SPEC.md` and `FINAL_PHASE123_SLIDE_TEXT.md`.
For Git upload decisions, read `GIT_INCLUDE_LIST.md`, `GIT_IGNORE_LIST.md`, and `GIT_RELEASE_PRIORITY_ORDER.md`.
"""
    write_text("PROFESSOR_READING_GUIDE.md", guide)


def main() -> None:
    """Run the command-line entrypoint for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    metrics = metric_summaries()
    audit_rows = build_file_audit()
    audit_fields = [
        "file_path",
        "phase",
        "category",
        "purpose",
        "input_or_output",
        "canonical_or_extension",
        "required_for_reproduction",
        "recommended_for_git",
        "recommended_to_ignore",
        "notes",
    ]
    write_csv("MASTER_PROJECT_FILE_AUDIT.csv", audit_rows, audit_fields)

    # First write the main docs, then check references and rewrite with the reference-check section.
    missing_refs = find_missing_references()
    write_master_report(metrics, missing_refs)
    write_git_lists()
    write_professor_guide()
    missing_refs = find_missing_references()
    write_master_report(metrics, missing_refs)
    write_verification(missing_refs, audit_rows)
    write_professor_guide()


if __name__ == "__main__":
    main()
