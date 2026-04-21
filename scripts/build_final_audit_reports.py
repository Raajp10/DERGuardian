from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from methodology_alignment_common import parse_sequence, write_markdown


FINAL_ALIGNMENT_PATH = ROOT / "FINAL_DIAGRAM_ALIGNMENT.md"
SAFE_LABELS_PATH = ROOT / "DIAGRAM_SAFE_LABELS.md"
DIAGRAM_AUDIT_CSV = ROOT / "diagram_box_audit.csv"
STATUS_PATH = ROOT / "COMPLETE_PROJECT_STATUS.md"
DISCUSSION_PATH = ROOT / "COMPLETE_RESULTS_AND_DISCUSSION.md"
CLAIMS_PATH = ROOT / "COMPLETE_PUBLICATION_SAFE_CLAIMS.md"
DECISION_PATH = ROOT / "FINAL_PROJECT_DECISION.md"
XAI_SAFE_PATCH_PATH = ROOT / "xai_safe_claims_patch.md"
EXEC_SUMMARY_PATH = ROOT / "FINAL_EXECUTIVE_VERIFICATION_SUMMARY.md"
MISSING_PARTIAL_PATH = ROOT / "FINAL_MISSING_OR_PARTIAL_ITEMS.md"
RERUN_COMMANDS_PATH = ROOT / "FINAL_RERUN_COMMANDS.sh"


def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path.resolve())


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_none_"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return "```text\n" + frame.to_string(index=False) + "\n```"


def _load_all_model_summary_rows() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted((ROOT / "outputs" / "window_size_study").glob("*/reports/model_summary.csv")):
        frame = pd.read_csv(path)
        frame["source_file"] = _safe_rel(path)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _truthy(series: pd.Series) -> pd.Series:
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y"})
    )


def _box_rows() -> list[dict[str, str]]:
    return [
        {
            "phase": "Phase 1",
            "diagram_box": "Generate DER Grid Data",
            "implementation_status": "fully implemented",
            "label_safety": "safe",
            "safe_label": "Generate DER grid data",
            "evidence_files": "PHASE1_DATA_LINEAGE_AUDIT.md; outputs/clean/; outputs/attacked/",
            "notes": "Raw canonical clean and attacked DER time-series artifacts are present.",
        },
        {
            "phase": "Phase 1",
            "diagram_box": "System Modeling IEEE 123 Bus + DER (PV, BESS)",
            "implementation_status": "fully implemented",
            "label_safety": "safe",
            "safe_label": "IEEE 123-bus DER system modeling",
            "evidence_files": "PHASE1_DATA_LINEAGE_AUDIT.md; outputs/clean/",
            "notes": "Canonical pipeline uses the IEEE 123-bus DER simulation stack already present in the repository.",
        },
        {
            "phase": "Phase 1",
            "diagram_box": "Normal Data Generation OpenDSS Simulation (Voltage, Current, Power, SOC, Temp)",
            "implementation_status": "fully implemented",
            "label_safety": "incomplete label",
            "safe_label": "OpenDSS normal-operation simulation with electrical, DER, and control observables",
            "evidence_files": "PHASE1_DATA_LINEAGE_AUDIT.md; outputs/window_size_study/window_dataset_summary.csv",
            "notes": "Implemented, but the original label omits feeder losses, control states, cyber counts, and derived residuals.",
        },
        {
            "phase": "Phase 1",
            "diagram_box": "Data Preprocessing Cleaning + Feature Engineering",
            "implementation_status": "fully implemented",
            "label_safety": "safe",
            "safe_label": "Data preprocessing, residualization, and feature engineering",
            "evidence_files": "PHASE1_DATA_LINEAGE_AUDIT.md; phase1_training_input_inventory.csv",
            "notes": "Windowing, residualization, metadata retention, feature selection, and split logic were verified.",
        },
        {
            "phase": "Phase 1",
            "diagram_box": "Baseline Models ML Models (LSTM, Autoencoder, Isolation Forest)",
            "implementation_status": "fully implemented",
            "label_safety": "unsafe/incomplete label",
            "safe_label": "Benchmark model suite (Transformer, LSTM, GRU, Autoencoder, Isolation Forest, threshold, token baseline)",
            "evidence_files": "phase1_canonical_input_summary.md; outputs/window_size_study/*/reports/model_summary.csv",
            "notes": "The model suite exists, but the original label omits Transformer and GRU and can misread the autoencoder as an LSTM autoencoder.",
        },
        {
            "phase": "Phase 1",
            "diagram_box": "LLM Analysis Context Learning of Normal Behavior",
            "implementation_status": "partially implemented",
            "label_safety": "overclaiming",
            "safe_label": "Optional context summaries for operator-facing interpretation",
            "evidence_files": "PHASE1_DATA_LINEAGE_AUDIT.md; outputs/reports/model_full_run_artifacts/",
            "notes": "Legacy context artifacts exist, but this is not part of the frozen canonical winner path.",
        },
        {
            "phase": "Phase 1",
            "diagram_box": "Model Evaluation Reconstruction Loss, Threshold (p99), Accuracy",
            "implementation_status": "partially implemented",
            "label_safety": "unsafe/incomplete label",
            "safe_label": "Benchmark evaluation with threshold calibration and detection metrics",
            "evidence_files": "outputs/window_size_study/final_window_comparison.csv; outputs/window_size_study/*/reports/",
            "notes": "Evaluation exists, but it is not universally reconstruction-loss-based and the benchmark emphasizes F1/AP/ROC rather than a single p99 accuracy label.",
        },
        {
            "phase": "Phase 2",
            "diagram_box": "System Card Creation (Grid + Constraints + Physics Rules)",
            "implementation_status": "partially implemented",
            "label_safety": "incomplete evidence",
            "safe_label": "Scenario design with grid-aware constraints and validator rules",
            "evidence_files": "phase2_scenario_master_inventory.csv; phase2_coverage_summary.md",
            "notes": "Constraint-aware scenario construction is evidenced indirectly through bundle structure and validator outputs, but no standalone explicit system-card artifact is present.",
        },
        {
            "phase": "Phase 2",
            "diagram_box": "LLM Scenario Generator Synthetic Anomaly / Zero-day Style",
            "implementation_status": "fully implemented",
            "label_safety": "overclaiming",
            "safe_label": "LLM-generated synthetic heldout attack scenarios",
            "evidence_files": "phase2_scenario_master_inventory.csv; heldout_bundle_metrics.csv",
            "notes": "LLM-generated heldout bundles exist, but 'zero-day style' should not be inflated into real-world zero-day robustness.",
        },
        {
            "phase": "Phase 2",
            "diagram_box": "Scenario JSON Library Structured Attack Definitions",
            "implementation_status": "fully implemented",
            "label_safety": "safe",
            "safe_label": "Structured scenario bundle library",
            "evidence_files": "phase2_scenario_master_inventory.csv; outputs/attacked/scenario_manifest.json",
            "notes": "Structured scenario definitions are present across canonical, heldout, repaired, and human-authored bundles.",
        },
        {
            "phase": "Phase 2",
            "diagram_box": "Validation Layer Physics + Safety Constraints Check",
            "implementation_status": "fully implemented",
            "label_safety": "safe",
            "safe_label": "Validator-guided plausibility and safety screening",
            "evidence_files": "phase2_scenario_master_inventory.csv; phase2_diversity_and_quality_report.md",
            "notes": "Accepted, rejected, and repaired statuses are explicitly tracked.",
        },
        {
            "phase": "Phase 2",
            "diagram_box": "Injection Engine Measurement + Control Manipulation",
            "implementation_status": "fully implemented",
            "label_safety": "safe",
            "safe_label": "Measurement and control attack injection",
            "evidence_files": "outputs/attacked/scenario_manifest.json; outputs/attacked/attack_labels.parquet",
            "notes": "Canonical attacked bundle and heldout bundles include measurement/control attack scenarios.",
        },
        {
            "phase": "Phase 2",
            "diagram_box": "OpenDSS Execution Physical + Cyber Simulation",
            "implementation_status": "fully implemented",
            "label_safety": "safe",
            "safe_label": "OpenDSS physical and cyber replay simulation",
            "evidence_files": "PHASE1_DATA_LINEAGE_AUDIT.md; outputs/attacked/; outputs/window_size_study/improved_phase3/replay_inputs/",
            "notes": "The repo includes both canonical simulation outputs and replay inputs built from heldout/repaired bundles.",
        },
        {
            "phase": "Phase 2",
            "diagram_box": "Attack Dataset Generation Time-Series with Labels",
            "implementation_status": "fully implemented",
            "label_safety": "safe",
            "safe_label": "Attacked time-series dataset generation with labels",
            "evidence_files": "outputs/attacked/attack_labels.parquet; phase2_scenario_master_inventory.csv",
            "notes": "Canonical and heldout bundles are linked to labeled attacked time-series artifacts.",
        },
        {
            "phase": "Phase 2",
            "diagram_box": "Quality Check Plausibility + Diversity + Metadata",
            "implementation_status": "fully implemented",
            "label_safety": "safe with updated wording",
            "safe_label": "Validated scenario quality, diversity, and coverage audit",
            "evidence_files": "phase2_coverage_summary.md; phase2_diversity_and_quality_report.md; phase2_difficulty_calibration.csv",
            "notes": "Coverage, diversity, balance, plausibility, and difficulty are now backed by persisted tables and figures.",
        },
        {
            "phase": "Phase 3",
            "diagram_box": "Unified Dataset Normal + Attack Data",
            "implementation_status": "fully implemented",
            "label_safety": "safe",
            "safe_label": "Unified clean/attacked residual benchmark dataset",
            "evidence_files": "PHASE1_DATA_LINEAGE_AUDIT.md; outputs/window_size_study/*/residual_windows.parquet",
            "notes": "Residual benchmark datasets explicitly align clean and attacked windows.",
        },
        {
            "phase": "Phase 3",
            "diagram_box": "Model Training ML Models + Tiny LLM (LoRA)",
            "implementation_status": "partially implemented",
            "label_safety": "overclaiming",
            "safe_label": "Canonical ML benchmark plus experimental LoRA explanation branch",
            "evidence_files": "outputs/window_size_study/final_window_comparison.csv; phase3_lora_results.csv; phase3_lora_run_log.txt",
            "notes": "Canonical ML benchmark is real and frozen; LoRA exists but remains an explanation-side extension, not the main detector.",
        },
        {
            "phase": "Phase 3",
            "diagram_box": "Anomaly Detection Prediction Reconstruction Loss",
            "implementation_status": "partially implemented",
            "label_safety": "unsafe label",
            "safe_label": "Anomaly detection scoring across supervised and normal-oriented models",
            "evidence_files": "outputs/window_size_study/*/reports/model_summary.csv; phase1_ttm_results.csv",
            "notes": "Detection is implemented, but not all models are reconstruction-loss-based; the canonical winner is a supervised sequence classifier.",
        },
        {
            "phase": "Phase 3",
            "diagram_box": "Decision Layer Threshold + Context Reasoning",
            "implementation_status": "partially implemented",
            "label_safety": "safe with narrowed wording",
            "safe_label": "Thresholded detection plus optional explanation/context support",
            "evidence_files": "deployment_benchmark_results.csv; xai_final_validation_report.md",
            "notes": "Thresholding is central; context reasoning exists only as an auxiliary explanation-side layer.",
        },
        {
            "phase": "Phase 3",
            "diagram_box": "Human-like Explanation LLM Generates Root Cause Analysis",
            "implementation_status": "partially implemented",
            "label_safety": "unsafe/overclaiming",
            "safe_label": "Grounded explanation layer for post-alert operator support",
            "evidence_files": "xai_case_level_audit.csv; xai_final_validation_report.md",
            "notes": "Family attribution is materially stronger than asset localization and grounding is partial, so human-like root-cause analysis is not supported.",
        },
        {
            "phase": "Phase 3",
            "diagram_box": "Evaluation Metrics F1, Precision, Recall, AOI",
            "implementation_status": "partially implemented",
            "label_safety": "unsafe/incomplete label",
            "safe_label": "Detection metrics (F1, precision, recall, AP/ROC) plus explanation grounding metrics",
            "evidence_files": "outputs/window_size_study/final_window_comparison.csv; xai_case_level_audit.csv",
            "notes": "F1/precision/recall are implemented; AOI is not a separately established repository metric and should not be claimed as such.",
        },
        {
            "phase": "Phase 3",
            "diagram_box": "Deployment Vision Lightweight Edge AI for DER",
            "implementation_status": "partially implemented",
            "label_safety": "unsafe/overclaiming",
            "safe_label": "Offline lightweight deployment benchmark for frozen detector packages",
            "evidence_files": "deployment_benchmark_results.csv; deployment_environment_manifest.md",
            "notes": "Offline replay timing and footprint are measured on a workstation CPU only; no edge hardware or field deployment evidence exists.",
        },
    ]


def main() -> None:
    benchmark = _read_csv(ROOT / "outputs" / "window_size_study" / "final_window_comparison.csv")
    all_model_rows = _load_all_model_summary_rows()
    phase2_inventory = _read_csv(ROOT / "phase2_scenario_master_inventory.csv")
    phase2_family = _read_csv(ROOT / "phase2_attack_family_distribution.csv")
    phase2_asset_signal = _read_csv(ROOT / "phase2_asset_signal_coverage.csv")
    phase2_difficulty = _read_csv(ROOT / "phase2_difficulty_calibration.csv")
    heldout_metrics = _read_csv(ROOT / "heldout_bundle_metrics.csv")
    balanced_metrics = _read_csv(ROOT / "balanced_heldout_metrics.csv")
    multi_model_metrics = _read_csv(ROOT / "multi_model_heldout_metrics.csv")
    ttm_results = _read_csv(ROOT / "phase1_ttm_results.csv")
    lora_results = _read_csv(ROOT / "phase3_lora_results.csv")
    xai_audit = _read_csv(ROOT / "xai_case_level_audit.csv")
    deployment = _read_csv(ROOT / "deployment_benchmark_results.csv")

    canonical_winner = benchmark.loc[
        (benchmark["model_name"] == "transformer") & (benchmark["window_label"] == "60s")
    ].iloc[0]
    validated_phase2 = phase2_inventory[phase2_inventory["accepted_rejected"] == "accepted"].copy()
    repaired_phase2 = phase2_inventory[_truthy(phase2_inventory["repair_applied"])].copy()
    rejected_phase2 = phase2_inventory[phase2_inventory["accepted_rejected"] == "rejected"].copy()
    difficulty_counts = (
        phase2_difficulty["difficulty_bucket"].value_counts().reindex(["easy", "moderate", "hard", "very hard"], fill_value=0)
    )
    validated_asset_count = int(
        phase2_asset_signal.loc[
            (phase2_asset_signal["dimension"] == "asset") & (phase2_asset_signal["validated_scenarios"] > 0),
            "name",
        ].nunique()
    )
    validated_signal_count = int(
        phase2_asset_signal.loc[
            (phase2_asset_signal["dimension"] == "signal") & (phase2_asset_signal["validated_scenarios"] > 0),
            "name",
        ].nunique()
    )
    undercovered_families = phase2_family[phase2_family["validated_scenarios"] <= 1]["attack_family"].tolist()
    undercovered_assets = phase2_asset_signal[
        (phase2_asset_signal["dimension"] == "asset") & (phase2_asset_signal["validated_scenarios"] <= 1)
    ]["name"].tolist()
    undercovered_signals = phase2_asset_signal[
        (phase2_asset_signal["dimension"] == "signal") & (phase2_asset_signal["validated_scenarios"] <= 1)
    ]["name"].tolist()

    family_best = (
        all_model_rows.sort_values(["f1", "average_precision", "precision"], ascending=False)
        .groupby("model_name", as_index=False)
        .first()
    )
    family_best_short = family_best[
        ["model_name", "window_label", "precision", "recall", "f1", "average_precision", "inference_time_ms_per_prediction"]
    ].sort_values("f1", ascending=False)

    ttm_row = ttm_results.iloc[0]
    lora_test = lora_results[
        (lora_results["model_variant"] == "lora_finetuned") & (lora_results["split"] == "test_generator_heldout")
    ].iloc[0]
    xai_family_accuracy = float(xai_audit["family_exact_match"].mean())
    xai_asset_accuracy = float(xai_audit["asset_accuracy"].mean())
    xai_grounding = float(xai_audit["evidence_grounding_overlap"].mean())
    xai_action = float(xai_audit["action_relevance_score"].mean())
    xai_unsupported = float(xai_audit["unsupported_claim"].mean())

    original_heldout_only = heldout_metrics[heldout_metrics["generator_source"].isin(["chatgpt", "claude", "gemini", "grok"])].copy()
    original_heldout_mean = original_heldout_only[["precision", "recall", "f1"]].mean(numeric_only=True)
    balanced_mean = balanced_metrics[["precision", "recall", "f1"]].mean(numeric_only=True)

    replay_selected = multi_model_metrics[
        ((multi_model_metrics["model_name"] == "transformer") & (multi_model_metrics["window_label"] == "60s"))
        | ((multi_model_metrics["model_name"] == "lstm") & (multi_model_metrics["window_label"] == "300s"))
        | ((multi_model_metrics["model_name"] == "threshold_baseline") & (multi_model_metrics["window_label"] == "10s"))
    ].copy()
    replay_selected = replay_selected[replay_selected["generator_source"].isin(["chatgpt", "claude", "gemini", "grok"])]
    replay_selected_summary = (
        replay_selected.groupby(["model_name", "window_label"], as_index=False)[["precision", "recall", "f1", "mean_latency_seconds"]]
        .mean(numeric_only=True)
        .sort_values("f1", ascending=False)
    )

    deployment_actual = deployment[deployment["profile"] == "workstation_cpu"].copy()
    deployment_actual = deployment_actual[
        ["model_name", "window_label", "mean_cpu_inference_ms_per_window", "throughput_windows_per_sec", "rss_peak_mb", "replay_f1"]
    ].sort_values("mean_cpu_inference_ms_per_window")
    deployment_actual_display = deployment_actual.copy()
    deployment_actual_display["mean_cpu_inference_ms_per_window"] = deployment_actual_display["mean_cpu_inference_ms_per_window"].map(lambda value: f"{float(value):.6f}")
    deployment_actual_display["throughput_windows_per_sec"] = deployment_actual_display["throughput_windows_per_sec"].map(lambda value: f"{float(value):.3f}")
    deployment_actual_display["rss_peak_mb"] = deployment_actual_display["rss_peak_mb"].map(lambda value: f"{float(value):.3f}")
    deployment_actual_display["replay_f1"] = deployment_actual_display["replay_f1"].map(lambda value: f"{float(value):.6f}")

    box_df = pd.DataFrame(_box_rows())
    box_df.to_csv(DIAGRAM_AUDIT_CSV, index=False)

    alignment_lines = [
        "# Final Diagram Alignment",
        "",
        "The table below audits each diagram box against real repository evidence. Status reflects implementation reality; label safety reflects whether the original wording can still be used without overclaiming.",
        "",
        _md_table(
            box_df[
                ["phase", "diagram_box", "implementation_status", "label_safety", "safe_label", "notes"]
            ].rename(
                columns={
                    "phase": "Phase",
                    "diagram_box": "Diagram box",
                    "implementation_status": "Implementation status",
                    "label_safety": "Label safety",
                    "safe_label": "Recommended safe label",
                    "notes": "Notes",
                }
            )
        ),
        "",
        "## Bottom line",
        "",
        "- Fully implemented boxes now cover the canonical Phase 1 pipeline, the validated Phase 2 scenario pipeline, the new Phase 2 quality audit, and the replay-evaluated detector path.",
        "- Partially implemented boxes remain where the repo has real functionality but the original label overclaims the evidence or collapses optional extension work into the canonical path.",
        "- Future-work pressure now concentrates on explicit system-card artifacts, stronger explanation grounding/localization, true edge hardware evidence, and any claim resembling real-world zero-day robustness.",
    ]
    write_markdown(FINAL_ALIGNMENT_PATH, "\n".join(alignment_lines))

    overclaim_rows = box_df[box_df["label_safety"] != "safe"][
        ["diagram_box", "safe_label", "notes"]
    ].rename(columns={"diagram_box": "Original box", "safe_label": "Safer current label", "notes": "Why"})
    safe_label_lines = [
        "# Diagram Safe Labels",
        "",
        "Use these replacements whenever the original diagram text would overclaim the current evidence.",
        "",
        _md_table(overclaim_rows),
    ]
    write_markdown(SAFE_LABELS_PATH, "\n".join(safe_label_lines))

    status_lines = [
        "# Complete Project Status",
        "",
        "## Fully complete",
        "",
        "- The canonical Phase 1 window-size benchmark is frozen and still selects `transformer @ 60s` as the source-of-truth winner.",
        "- Phase 2 canonical generation, heldout bundle validation, and replay separation are all present and verified.",
        f"- The Phase 2 scientific audit package is now backed by persisted inventory and coverage tables (`{len(phase2_inventory)}` rows total, `{len(validated_phase2)}` validated, `{len(rejected_phase2)}` rejected).",
        "- Phase 3 grounded explanation auditing is present with case-level evidence, taxonomy, and qualitative examples.",
        "- The offline lightweight deployment benchmark is present for frozen detector packages on the current workstation CPU.",
        "",
        "## Partially complete",
        "",
        f"- TinyTimeMixer now exists as a real extension benchmark (`F1={float(ttm_row['f1']):.3f}` on the canonical 60 s residual test split), but it does not replace the canonical winner.",
        f"- LoRA now exists as a real experimental explanation branch, but heldout test family accuracy is only `{float(lora_test['family_accuracy']):.3f}` and asset grounding remains `{float(lora_test['asset_accuracy']):.3f}`.",
        f"- XAI family attribution is materially stronger than asset localization (`family={xai_family_accuracy:.3f}`, `asset={xai_asset_accuracy:.3f}`, `grounding={xai_grounding:.3f}`), so explanation claims must stay conservative.",
        "- The diagram is now much closer to reality, but several original labels still need safer wording to avoid overclaiming.",
        "",
        "## Still future work",
        "",
        "- Real edge-hardware deployment or field trials.",
        "- Real-world zero-day robustness claims.",
        "- Human-like root-cause analysis claims.",
        "- A standalone ARIMA baseline implementation if that comparison is still required.",
        "- Stronger explicit system-card artifacts if the methodology diagram must be satisfied literally.",
    ]
    write_markdown(STATUS_PATH, "\n".join(status_lines))

    discussion_lines = [
        "# Complete Results And Discussion",
        "",
        "## Canonical benchmark path",
        "",
        f"- The canonical benchmark winner remains `transformer @ 60s` with benchmark F1 `{float(canonical_winner['f1']):.6f}`.",
        "- This is still the only source-of-truth model-selection result for the paper-safe benchmark path.",
        "- Family-best benchmark rows across model families are shown below for context, but they do not overwrite the canonical winner.",
        "",
        _md_table(family_best_short),
        "",
        "## Heldout replay path",
        "",
        f"- Mean original heldout replay precision / recall / F1 across ChatGPT, Claude, Gemini, and Grok: `{original_heldout_mean['precision']:.3f}` / `{original_heldout_mean['recall']:.3f}` / `{original_heldout_mean['f1']:.3f}`.",
        f"- Mean repaired replay precision / recall / F1 across repaired bundles: `{balanced_mean['precision']:.3f}` / `{balanced_mean['recall']:.3f}` / `{balanced_mean['f1']:.3f}`.",
        "- Replay remains a separate evaluation context from benchmark model selection.",
        "",
        _md_table(replay_selected_summary),
        "",
        "## Phase 2 scientific coverage and diversity",
        "",
        f"- Inventory rows audited: `{len(phase2_inventory)}`",
        f"- Validated rows used for final usable coverage: `{len(validated_phase2)}`",
        f"- Rejected rows retained for auditability: `{len(rejected_phase2)}`",
        f"- Repair-tagged rows: `{len(repaired_phase2)}`",
        f"- Distinct validated families / assets / signals: `{validated_phase2['attack_family'].nunique()}` / `{validated_asset_count}` / `{validated_signal_count}`",
        f"- Difficulty bucket counts: easy=`{int(difficulty_counts['easy'])}`, moderate=`{int(difficulty_counts['moderate'])}`, hard=`{int(difficulty_counts['hard'])}`, very hard=`{int(difficulty_counts['very hard'])}`",
        f"- Under-covered families: `{undercovered_families}`",
        f"- Under-covered assets (<=1 validated scenario): `{undercovered_assets[:10]}`",
        f"- Under-covered signals (<=1 validated scenario): `{undercovered_signals[:10]}`",
        "",
        "## Extension experiments",
        "",
        f"- TinyTimeMixer extension: `F1={float(ttm_row['f1']):.3f}`, `precision={float(ttm_row['precision']):.3f}`, `recall={float(ttm_row['recall']):.3f}`, `latency={float(ttm_row['inference_time_ms_per_prediction']):.3f} ms/window`, `params={int(ttm_row['parameter_count'])}`.",
        "- TTM used the canonical 60 s residual feature contract but required local random initialization because no compatible public pre-trained 12->1 checkpoint matched the repo's short-sequence setup.",
        f"- LoRA explanation branch heldout test: `family_accuracy={float(lora_test['family_accuracy']):.3f}`, `asset_accuracy={float(lora_test['asset_accuracy']):.3f}`, `grounding={float(lora_test['evidence_grounding_quality']):.3f}`, `latency={float(lora_test['mean_latency_ms']):.1f} ms/example`.",
        "- Both extension branches are real evidence, but neither branch replaces the canonical detector path.",
        "",
        "## XAI validation",
        "",
        f"- Heldout family accuracy: `{xai_family_accuracy:.3f}`",
        f"- Heldout asset accuracy: `{xai_asset_accuracy:.3f}`",
        f"- Heldout grounding rate: `{xai_grounding:.3f}`",
        f"- Heldout action relevance: `{xai_action:.3f}`",
        f"- Unsupported claim rate: `{xai_unsupported:.3f}`",
        "- The XAI layer is therefore useful for grounded operator support, but not strong enough for human-like root-cause-analysis claims.",
        "",
        "## Deployment benchmark",
        "",
        _md_table(deployment_actual_display),
        "",
        f"- `threshold_baseline @ 10s replay_f1={float(deployment_actual.loc[deployment_actual['model_name'] == 'threshold_baseline', 'replay_f1'].iloc[0]):.6f}`",
        f"- `lstm @ 300s replay_f1={float(deployment_actual.loc[deployment_actual['model_name'] == 'lstm', 'replay_f1'].iloc[0]):.6f}`",
        f"- `transformer @ 60s replay_f1={float(deployment_actual.loc[deployment_actual['model_name'] == 'transformer', 'replay_f1'].iloc[0]):.6f}`",
        "",
        "## Discussion",
        "",
        "- The repository now has a stronger evidence trail across benchmarking, heldout replay, Phase 2 diversity, extension experiments, XAI auditing, and offline deployment measurement.",
        "- The biggest remaining claim-risk areas are exactly the ones that should stay conservative tomorrow: human-like root-cause analysis, real-world zero-day robustness, true edge deployment, and AOI as an implemented detection metric.",
    ]
    write_markdown(DISCUSSION_PATH, "\n".join(discussion_lines))

    claims_lines = [
        "# Complete Publication Safe Claims",
        "",
        "## Safe now",
        "",
        f"- The canonical benchmark winner is still `transformer @ 60s` with benchmark F1 `{float(canonical_winner['f1']):.3f}`.",
        f"- The project now includes a verified Phase 2 coverage/diversity/difficulty audit over `{len(phase2_inventory)}` audited scenario rows with `{len(validated_phase2)}` validated usable rows.",
        f"- The heldout XAI layer supports evidence-grounded family attribution for operator support (`family={xai_family_accuracy:.3f}`) but not strong asset localization (`asset={xai_asset_accuracy:.3f}`).",
        "- The repo now includes an offline lightweight deployment benchmark on workstation CPU hardware for frozen detector packages.",
        "- TinyTimeMixer and LoRA both exist as real extension experiments with persisted outputs and rerun logs.",
        "",
        "## Safe with wording changes",
        "",
        "- `grounded explanation layer for post-alert operator support`",
        "- `canonical ML benchmark plus experimental LoRA explanation branch`",
        "- `offline lightweight deployment benchmark`",
        "- `LLM-generated synthetic heldout scenarios` instead of `zero-day style` as a robustness claim",
        "- `detection metrics plus explanation grounding metrics` instead of `F1, Precision, Recall, AOI` as a single implemented metric stack",
        "",
        "## Not safe",
        "",
        "- Human-like root-cause analysis.",
        "- Real-world zero-day robustness.",
        "- Edge deployment or field deployment.",
        "- ARIMA comparison results, because ARIMA is not implemented in the current repo.",
        "- AOI as a claimed implemented detector metric unless separately added and validated.",
        "",
        "## Missing evidence",
        "",
        "- Edge-device runtime measurements.",
        "- Human evaluation showing explanation quality at a human-analyst level.",
        "- Real zero-day or field attack trials.",
        "- Stronger explanation asset localization and signal grounding.",
    ]
    write_markdown(CLAIMS_PATH, "\n".join(claims_lines))

    decision_lines = [
        "# Final Project Decision",
        "",
        "## Is the project now aligned with the diagram?",
        "",
        "Mostly, but not literally box-for-box under the original wording. The repo now satisfies most of the intended functionality, while several original labels still overclaim the evidence and must be rewritten.",
        "",
        "## Boxes that are truly satisfied",
        "",
        "- Core Phase 1 data generation, preprocessing, and benchmark execution.",
        "- Core Phase 2 scenario bundle execution, validation, attacked dataset generation, and quality auditing.",
        "- Unified residual dataset generation, replay evaluation, and detector benchmarking.",
        "",
        "## Boxes that remain partial",
        "",
        "- LLM normal-behavior context learning.",
        "- Tiny LLM (LoRA) as part of the main detector path.",
        "- Human-like explanation / root-cause analysis.",
        "- Evaluation box as written with AOI.",
        "- Deployment box as written with edge-AI wording.",
        "- Explicit system-card artifact if interpreted literally.",
        "",
        "## What can be said tomorrow",
        "",
        "- The canonical benchmark winner remains transformer @ 60 s.",
        "- Heldout replay exists and is separate from benchmark selection.",
        "- Phase 2 now has a formal coverage/diversity/difficulty audit.",
        "- TTM and LoRA are real extension branches, but secondary.",
        "- XAI supports grounded operator-facing explanations, not human-like RCA.",
        "- Deployment evidence is offline workstation replay benchmarking, not edge deployment.",
        "",
        "## What is safe in the paper",
        "",
        "- Use the safe-now and safe-with-wording-changes lists from `COMPLETE_PUBLICATION_SAFE_CLAIMS.md`.",
        "- Keep benchmark, heldout replay, and extension experiments explicitly separated.",
        "- Do not imply that replay gains overwrite the canonical benchmark winner.",
    ]
    write_markdown(DECISION_PATH, "\n".join(decision_lines))

    xai_patch_lines = [
        "# XAI Safe Claims Patch",
        "",
        "Use the wording below anywhere the repo, talk, or paper might drift into explanation overclaiming.",
        "",
        "## Replace unsafe wording",
        "",
        "- Replace `Human-like Explanation LLM Generates Root Cause Analysis` with `Grounded explanation layer for post-alert operator support`.",
        "- Replace `root-cause analysis` with `evidence-grounded family attribution and operator support` unless a narrower context truly demands otherwise.",
        "- Avoid implying reliable asset localization; current heldout asset accuracy is limited.",
        "",
        "## Safe phrasing",
        "",
        f"- `Heldout XAI family attribution accuracy was {xai_family_accuracy:.3f}, while asset localization remained lower at {xai_asset_accuracy:.3f}.`",
        f"- `Grounding overlap averaged {xai_grounding:.3f}, so the explanation layer is useful but partial.`",
        "- `The explanation layer is post-alert and operator-facing; it does not replace the detector.`",
        "",
        "## Unsafe phrasing",
        "",
        "- `human-like root-cause analysis`",
        "- `operator-equivalent diagnosis`",
        "- `reliable automatic localization of the exact compromised asset`",
    ]
    write_markdown(XAI_SAFE_PATCH_PATH, "\n".join(xai_patch_lines))

    missing_lines = [
        "# Final Missing Or Partial Items",
        "",
        "- Real edge-hardware deployment benchmark.",
        "- Real-world zero-day or field validation.",
        "- Human-level explanation evidence.",
        "- Stronger heldout asset localization and grounding in both XAI and LoRA branches.",
        "- Explicit ARIMA implementation if that baseline is still a requirement.",
        "- Explicit standalone system-card artifact if the methodology must match the original box literally.",
    ]
    write_markdown(MISSING_PARTIAL_PATH, "\n".join(missing_lines))

    rerun_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "python scripts/run_repo_audit_pass01.py",
        "python scripts/build_phase1_lineage_audit.py",
        "python scripts/run_phase1_ttm_extension.py",
        "python scripts/build_phase2_extensions.py",
        "python scripts/run_phase3_lora_extension.py --force-retrain",
        "python scripts/run_deployment_benchmark.py",
        "python scripts/build_final_audit_reports.py",
        "python scripts/run_final_triple_verification.py",
    ]
    RERUN_COMMANDS_PATH.write_text("\n".join(rerun_lines) + "\n", encoding="utf-8")

    exec_summary_lines = [
        "# Final Executive Verification Summary",
        "",
        "## 1. Verified complete",
        "",
        "- The canonical benchmark path is real, frozen, and still selects `transformer @ 60s`.",
        "- Phase 2 now has persisted inventory, coverage, diversity, balance, and difficulty outputs.",
        "- The LoRA branch is real and now has a fresh retrain log plus evidence checklist.",
        "- The TTM branch is real and benchmarked as an extension on the canonical 60 s residual contract.",
        "- The XAI audit is case-level and quantitatively backed.",
        "- The deployment benchmark is real for offline workstation CPU replay.",
        "",
        "## 2. Fixed during this run",
        "",
        "- Repaired the Phase 2 inventory serialization so saved asset/signal lists round-trip cleanly from the persisted CSV.",
        "- Added a real TinyTimeMixer extension benchmark with saved checkpoint, results table, report, and figures.",
        "- Re-ran the LoRA branch with forced retraining and added `phase3_lora_run_log.txt` plus `phase3_lora_evidence_checklist.md`.",
        "- Added deployment environment and repro artifacts.",
        "- Rebuilt the final alignment and publication-safe reporting layer from verified evidence instead of optimistic summaries.",
        "",
        "## 3. Still partial / future work",
        "",
        "- Human-like explanation claims remain unsupported.",
        "- Edge deployment remains unsupported.",
        "- Real-world zero-day robustness remains unsupported.",
        "- AOI remains unimplemented as a standalone reported metric.",
        "- ARIMA remains unimplemented.",
        "",
        "## 4. Exact safe claims",
        "",
        f"- `Transformer @ 60s remains the canonical benchmark winner with F1 {float(canonical_winner['f1']):.3f}.`",
        f"- `Phase 2 now includes a validated scenario audit covering {len(validated_phase2)} usable scenarios across {validated_phase2['attack_family'].nunique()} attack families.`",
        f"- `The XAI layer supports grounded family attribution for operator support (family accuracy {xai_family_accuracy:.3f}), with weaker asset localization (asset accuracy {xai_asset_accuracy:.3f}).`",
        f"- `TinyTimeMixer was added as an extension benchmark and reached F1 {float(ttm_row['f1']):.3f} on the canonical 60 s residual test split without displacing the canonical winner.`",
        f"- `The LoRA branch is experimental explanation-side evidence only; heldout test family accuracy is {float(lora_test['family_accuracy']):.3f}.`",
        "- `The deployment package is an offline lightweight deployment benchmark on workstation CPU hardware.`",
        "",
        "## 5. Exact unsafe claims",
        "",
        "- `human-like root-cause analysis`",
        "- `real-world zero-day robustness`",
        "- `edge deployment`",
        "- `AOI is implemented as a detector metric`",
        "",
        "## 6. Whether the project now truly satisfies the diagram, box by box",
        "",
        "See `diagram_box_audit.csv` and `FINAL_DIAGRAM_ALIGNMENT.md`. The project now satisfies most functional boxes, but several original labels still need safer wording and remain partial under a strict literal reading.",
    ]
    write_markdown(EXEC_SUMMARY_PATH, "\n".join(exec_summary_lines))


if __name__ == "__main__":
    main()
