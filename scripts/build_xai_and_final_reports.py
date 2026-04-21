from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from methodology_alignment_common import (
    GENERATOR_SOURCES,
    ROOT,
    action_relevance_score,
    asset_match,
    evidence_overlap,
    family_match,
    load_generator_truth_lookup,
    normalize_signal,
    partial_alignment_score,
    pretty_label,
    read_json,
    write_markdown,
)


XAI_CASE_AUDIT_PATH = ROOT / "xai_case_level_audit.csv"
XAI_TAXONOMY_PATH = ROOT / "xai_error_taxonomy.md"
XAI_EXAMPLES_PATH = ROOT / "xai_qualitative_examples.md"
XAI_REPORT_PATH = ROOT / "xai_final_validation_report.md"
XAI_FAMILY_ASSET_FIG = ROOT / "xai_family_vs_asset_accuracy.png"
XAI_TAXONOMY_FIG = ROOT / "xai_error_taxonomy_breakdown.png"
XAI_GROUNDING_FIG = ROOT / "xai_grounding_vs_family_accuracy.png"

FINAL_ALIGNMENT_PATH = ROOT / "FINAL_DIAGRAM_ALIGNMENT.md"
SAFE_LABELS_PATH = ROOT / "DIAGRAM_SAFE_LABELS.md"
STATUS_PATH = ROOT / "COMPLETE_PROJECT_STATUS.md"
DISCUSSION_PATH = ROOT / "COMPLETE_RESULTS_AND_DISCUSSION.md"
CLAIMS_PATH = ROOT / "COMPLETE_PUBLICATION_SAFE_CLAIMS.md"
DECISION_PATH = ROOT / "FINAL_PROJECT_DECISION.md"


def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path.resolve())


def _string_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, list):
        return [str(item).strip() for item in values if str(item).strip()]
    if isinstance(values, str):
        text = values.strip()
        if not text:
            return []
        return [piece.strip() for piece in text.split("|") if piece.strip()]
    return [str(values).strip()]


def _compute_xai_audit() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for generator_source in GENERATOR_SOURCES:
        _, truth_lookup = load_generator_truth_lookup(generator_source)
        case_dir = ROOT / "phase2_llm_benchmark" / "new_respnse_result" / "models" / generator_source / "xai_v4" / "cases"
        for case_path in sorted(case_dir.glob("*.json")):
            payload = read_json(case_path)
            packet = payload["packet"]
            explanation = payload["explanation"]
            comparison = payload["comparison"]
            scenario_id = str(comparison["scenario_id"])
            truth = truth_lookup.get(scenario_id)
            if truth is None:
                continue

            truth_family = str(truth["attack_family"])
            predicted_family = str(explanation.get("predicted_family", ""))
            truth_assets = {str(truth["target_asset"])} | {
                str(item["target_asset"]) for item in (truth.get("additional_targets", []) or [])
            }
            predicted_assets = set(_string_list(explanation.get("asset_attribution")))
            truth_signals = {normalize_signal(str(item)) for item in truth.get("observable_signals", [])}
            predicted_signals = {
                normalize_signal(str(signal))
                for item in packet.get("grounded_evidence_items", [])
                for signal in item.get("signal_refs", [])
                if str(signal).strip()
            }
            asset_label = asset_match(truth_assets, predicted_assets)
            family_label = family_match(truth_family, predicted_family)
            grounding_overlap, grounding_label = evidence_overlap(truth_signals, predicted_signals, truth_assets, predicted_assets)
            truth_effects = " ".join(str(item) for item in truth.get("expected_physical_effects", []))
            action_score, action_label = action_relevance_score(
                truth_family,
                truth_assets,
                truth_effects,
                _string_list(explanation.get("recommended_actions")),
            )
            partial_score = partial_alignment_score(truth_family, predicted_family, grounding_overlap)
            evidence_ids = {str(item.get("evidence_id", "")).strip() for item in packet.get("grounded_evidence_items", []) if str(item.get("evidence_id", "")).strip()}
            referenced_ids = {
                str(item).strip()
                for values in (explanation.get("claim_support_map", {}) or {}).values()
                for item in values
                if str(item).strip()
            }
            unsupported_claim = bool(referenced_ids - evidence_ids) or (
                predicted_family != truth_family and grounding_overlap == 0.0
            )

            if predicted_family == truth_family and asset_label == "none":
                error_category = "correct family / wrong asset"
            elif predicted_family == truth_family and grounding_overlap < 0.5:
                error_category = "correct family / partial evidence"
            elif predicted_family != truth_family and grounding_overlap >= 0.25:
                error_category = "wrong family / grounded evidence"
            elif unsupported_claim:
                error_category = "unsupported claim"
            elif predicted_family == truth_family and asset_label == "exact" and grounding_overlap >= 0.5:
                error_category = "fully aligned"
            else:
                error_category = "mixed partial"

            top3 = [str(item.get("family", "")) for item in explanation.get("family_candidates", [])[:3]]
            rows.append(
                {
                    "generator_source": generator_source,
                    "detector": str(packet.get("detector", "")),
                    "scenario_id": scenario_id,
                    "case_id": str(packet.get("case_id", case_path.stem)),
                    "target_family": truth_family,
                    "predicted_family": predicted_family,
                    "family_exact_match": int(predicted_family == truth_family),
                    "family_top3_match": int(truth_family in top3),
                    "family_match_label": family_label,
                    "target_assets": "|".join(sorted(truth_assets)),
                    "predicted_assets": "|".join(sorted(predicted_assets)),
                    "asset_match": asset_label,
                    "asset_accuracy": 1.0 if asset_label == "exact" else 0.5 if asset_label == "partial" else 0.0,
                    "truth_signal_count": len(truth_signals),
                    "predicted_grounded_signal_count": len(predicted_signals),
                    "evidence_grounding_overlap": grounding_overlap,
                    "evidence_grounding_label": grounding_label,
                    "action_relevance_score": action_score,
                    "action_relevance_label": action_label,
                    "partial_alignment_score": partial_score,
                    "unsupported_claim": int(unsupported_claim),
                    "primary_error_category": error_category,
                    "operator_summary": str(explanation.get("operator_summary", "")),
                    "recommended_actions_excerpt": " | ".join(_string_list(explanation.get("recommended_actions"))[:2]),
                    "uncertainty_notes": str(explanation.get("uncertainty_notes", "")),
                    "case_path": _safe_rel(case_path),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["generator_source", "detector", "scenario_id"]).reset_index(drop=True)
    frame.to_csv(XAI_CASE_AUDIT_PATH, index=False)
    return frame


def _write_xai_reports(frame: pd.DataFrame) -> None:
    generator_summary = (
        frame.groupby("generator_source")
        .agg(
            case_count=("case_id", "count"),
            family_accuracy=("family_exact_match", "mean"),
            family_top3_accuracy=("family_top3_match", "mean"),
            asset_accuracy=("asset_accuracy", "mean"),
            evidence_grounding_rate=("evidence_grounding_overlap", "mean"),
            action_relevance=("action_relevance_score", "mean"),
            partial_alignment=("partial_alignment_score", "mean"),
            unsupported_claim_rate=("unsupported_claim", "mean"),
        )
        .reset_index()
    )
    overall_summary = pd.DataFrame(
        [
            {
                "generator_source": "overall",
                "case_count": int(len(frame)),
                "family_accuracy": frame["family_exact_match"].mean(),
                "family_top3_accuracy": frame["family_top3_match"].mean(),
                "asset_accuracy": frame["asset_accuracy"].mean(),
                "evidence_grounding_rate": frame["evidence_grounding_overlap"].mean(),
                "action_relevance": frame["action_relevance_score"].mean(),
                "partial_alignment": frame["partial_alignment_score"].mean(),
                "unsupported_claim_rate": frame["unsupported_claim"].mean(),
            }
        ]
    )
    taxonomy_counts = (
        frame.groupby(["generator_source", "primary_error_category"]).size().reset_index(name="count")
    )
    taxonomy_totals = frame["primary_error_category"].value_counts().rename_axis("primary_error_category").reset_index(name="count")

    taxonomy_lines = [
        "# XAI Error Taxonomy",
        "",
        "The taxonomy below is computed from the heldout v4 case packets, not from free-form narration.",
        "",
        "## Overall counts",
        "",
        "```text",
        taxonomy_totals.to_string(index=False),
        "```",
        "",
        "## By generator",
        "",
        "```text",
        taxonomy_counts.to_string(index=False),
        "```",
        "",
        "## Reading guide",
        "",
        "- `correct family / wrong asset`: family attribution landed, but localization missed the affected asset.",
        "- `correct family / partial evidence`: family is right, but cited grounded signals only partially cover the true observable set.",
        "- `wrong family / grounded evidence`: evidence overlaps the true signals, but the family call still drifts.",
        "- `unsupported claim`: unsupported support-map references or a wrong family with no grounding overlap.",
        "- `mixed partial`: partial cases that do not fit the tighter headline bins above.",
        "- `fully aligned`: correct family, correct asset set, and strong signal grounding.",
    ]
    write_markdown(XAI_TAXONOMY_PATH, "\n".join(taxonomy_lines))

    qualitative_sections = ["# XAI Qualitative Examples", ""]
    for category in ["fully aligned", "correct family / wrong asset", "correct family / partial evidence", "wrong family / grounded evidence", "unsupported claim"]:
        subset = frame[frame["primary_error_category"] == category].copy()
        if subset.empty:
            continue
        if category == "fully aligned":
            selected = subset.sort_values(["partial_alignment_score", "evidence_grounding_overlap"], ascending=False).iloc[0]
        else:
            selected = subset.sort_values(["unsupported_claim", "evidence_grounding_overlap", "asset_accuracy"], ascending=False).iloc[0]
        qualitative_sections.extend(
            [
                f"## {category}",
                "",
                f"- generator: `{selected['generator_source']}`",
                f"- detector: `{selected['detector']}`",
                f"- scenario_id: `{selected['scenario_id']}`",
                f"- target family -> predicted family: `{selected['target_family']}` -> `{selected['predicted_family']}`",
                f"- target assets -> predicted assets: `{selected['target_assets']}` -> `{selected['predicted_assets']}`",
                f"- grounding overlap: {selected['evidence_grounding_overlap']:.3f}",
                f"- operator summary: {selected['operator_summary']}",
                f"- actions excerpt: {selected['recommended_actions_excerpt']}",
                f"- uncertainty note: {selected['uncertainty_notes']}",
                "",
            ]
        )
    write_markdown(XAI_EXAMPLES_PATH, "\n".join(qualitative_sections))

    report_lines = [
        "# XAI Final Validation Report",
        "",
        "The heldout XAI layer is now audited at case level. The safe interpretation remains conservative: grounded explanation layer, post-alert operator-facing support, and evidence-grounded family attribution.",
        "",
        "## Generator summary",
        "",
        "```text",
        pd.concat([generator_summary, overall_summary], ignore_index=True).to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
        f"- Overall family accuracy: {overall_summary['family_accuracy'].iloc[0]:.3f}",
        f"- Overall asset accuracy: {overall_summary['asset_accuracy'].iloc[0]:.3f}",
        f"- Overall evidence grounding rate: {overall_summary['evidence_grounding_rate'].iloc[0]:.3f}",
        f"- Overall action relevance: {overall_summary['action_relevance'].iloc[0]:.3f}",
        f"- Unsupported claim rate: {overall_summary['unsupported_claim_rate'].iloc[0]:.3f}",
        "",
        "Family attribution is materially stronger than asset attribution. Grounding is partial rather than exhaustive, so the correct claim remains a grounded explanation layer for operator support rather than human-like root-cause analysis.",
    ]
    write_markdown(XAI_REPORT_PATH, "\n".join(report_lines))

    plot_summary = generator_summary.copy()
    plot_summary["generator_label"] = plot_summary["generator_source"].map(str.title)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(plot_summary))
    width = 0.35
    ax.bar(x - width / 2, plot_summary["family_accuracy"], width=width, label="family accuracy", color="#1b9e77")
    ax.bar(x + width / 2, plot_summary["asset_accuracy"], width=width, label="asset accuracy", color="#7570b3")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_summary["generator_label"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Heldout XAI family vs asset accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(XAI_FAMILY_ASSET_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    taxonomy_plot = taxonomy_counts.pivot(index="generator_source", columns="primary_error_category", values="count").fillna(0)
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    bottom = np.zeros(len(taxonomy_plot))
    for column in taxonomy_plot.columns:
        values = taxonomy_plot[column].to_numpy()
        ax.bar(taxonomy_plot.index, values, bottom=bottom, label=column)
        bottom += values
    ax.set_ylabel("Case count")
    ax.set_title("Heldout XAI error taxonomy breakdown")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(XAI_TAXONOMY_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(
        plot_summary["family_accuracy"],
        plot_summary["evidence_grounding_rate"],
        s=plot_summary["case_count"] * 4,
        color="#e66101",
        alpha=0.85,
    )
    for _, row in plot_summary.iterrows():
        ax.text(row["family_accuracy"] + 0.01, row["evidence_grounding_rate"] + 0.005, row["generator_label"], fontsize=8)
    ax.set_xlabel("Family accuracy")
    ax.set_ylabel("Evidence grounding rate")
    ax.set_title("Heldout grounding vs family accuracy")
    fig.tight_layout()
    fig.savefig(XAI_GROUNDING_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _read_phase2_inventory() -> pd.DataFrame:
    return pd.read_csv(ROOT / "phase2_scenario_master_inventory.csv")


def _truthy(series: pd.Series) -> pd.Series:
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y"})
    )


def _write_final_reports(xai_frame: pd.DataFrame) -> None:
    phase2_inventory = _read_phase2_inventory()
    family_distribution = _read_optional_csv(ROOT / "phase2_attack_family_distribution.csv")
    difficulty_df = _read_optional_csv(ROOT / "phase2_difficulty_calibration.csv")
    lora_results = _read_optional_csv(ROOT / "phase3_lora_results.csv")
    deployment_results = _read_optional_csv(ROOT / "deployment_benchmark_results.csv")
    benchmark_table = pd.read_csv(ROOT / "outputs" / "window_size_study" / "final_window_comparison.csv")
    canonical_winner = benchmark_table.loc[
        (benchmark_table["model_name"] == "transformer") & (benchmark_table["window_label"] == "60s")
    ].iloc[0]
    heldout_metrics = pd.read_csv(ROOT / "heldout_bundle_metrics.csv")
    balanced_metrics = pd.read_csv(ROOT / "balanced_heldout_metrics.csv")
    multi_model_metrics = pd.read_csv(ROOT / "multi_model_heldout_metrics.csv")

    validated_inventory = phase2_inventory[phase2_inventory["accepted_rejected"] == "accepted"].copy()
    repaired_rows = phase2_inventory[_truthy(phase2_inventory["repair_applied"])].copy()
    xai_overall = {
        "family_accuracy": xai_frame["family_exact_match"].mean(),
        "asset_accuracy": xai_frame["asset_accuracy"].mean(),
        "grounding_rate": xai_frame["evidence_grounding_overlap"].mean(),
        "action_relevance": xai_frame["action_relevance_score"].mean(),
    }
    lora_test_row = None
    if lora_results is not None and not lora_results.empty:
        match = lora_results[
            (lora_results["model_variant"] == "lora_finetuned") & (lora_results["split"] == "test_generator_heldout")
        ]
        if not match.empty:
            lora_test_row = match.iloc[0]
    deployment_actual = None
    if deployment_results is not None and not deployment_results.empty:
        actual = deployment_results[deployment_results["profile"] == "workstation_cpu"]
        if not actual.empty:
            deployment_actual = actual.copy()

    alignment_lines = [
        "# Final Diagram Alignment",
        "",
        "| Diagram box | Status | Evidence | Safe paper/talk wording |",
        "|---|---|---|---|",
        "| Generate DER Grid Data / IEEE 123 bus + DER modeling | fully implemented | clean OpenDSS artifacts plus canonical reports | keep current wording |",
        "| Normal data generation / preprocessing / baseline ML evaluation | fully implemented | frozen Phase 1 benchmark and ready packages | keep current wording |",
        "| System card / scenario generator / scenario JSON library / validation / injection / attacked dataset generation | fully implemented | canonical Phase 2 bundle plus heldout validator outputs | keep current wording |",
        "| Quality check plausibility + diversity + metadata | fully implemented | new Phase 2 coverage, diversity, and difficulty package | `validated scenario quality, diversity, and coverage audit` |",
        "| Unified dataset + anomaly detection + detector evaluation | fully implemented | frozen benchmark, heldout replay, balanced replay | keep benchmark/replay separated |",
        "| Model training ML models + Tiny LLM (LoRA) | partially implemented | canonical ML path is frozen; LoRA branch exists as a secondary experiment | `canonical ML benchmark + experimental LoRA explanation branch` |",
        "| Human-like explanation / root cause analysis | partially implemented | grounded explanation audit now exists, but asset grounding remains limited | `grounded explanation layer for post-alert operator support` |",
        "| Deployment vision lightweight edge AI for DER | partially implemented | offline lightweight deployment benchmark now exists; no edge hardware used | `offline lightweight deployment benchmark` |",
    ]
    write_markdown(FINAL_ALIGNMENT_PATH, "\n".join(alignment_lines))

    safe_label_lines = [
        "# Diagram Safe Labels",
        "",
        "| Original box | Safer current label | Why |",
        "|---|---|---|",
        "| Model Training ML Models + Tiny LLM (LoRA) | Model Training (canonical ML benchmark) + experimental LoRA explanation branch | LoRA exists, but only as a secondary extension experiment. |",
        "| Human-like Explanation LLM Generates Root Cause Analysis | Grounded explanation layer for post-alert operator support | XAI audit supports grounded family attribution and operator support, not human-like root-cause analysis. |",
        "| Deployment Vision Lightweight Edge AI for DER | Offline lightweight deployment benchmark for frozen detector packages | The repo now measures offline replay timing and footprint, but not real edge deployment. |",
    ]
    write_markdown(SAFE_LABELS_PATH, "\n".join(safe_label_lines))

    status_lines = [
        "# Complete Project Status",
        "",
        "## Fully complete",
        "",
        "- Phase 1 canonical benchmark path remains complete and frozen, with transformer @ 60s still the benchmark-selected winner.",
        "- Phase 2 canonical attacked bundle, heldout validation, and replay separation remain intact.",
        "- Phase 2 coverage/diversity/difficulty package is now present.",
        "- Phase 3 grounded XAI audit package is now present.",
        "",
        "## Partially complete",
        "",
        "- Tiny-LLM / LoRA is implemented as an experimental explanation-side branch, not as the canonical detector path.",
        "- Deployment is now benchmarked offline, but no real edge hardware or field deployment evidence exists.",
        "",
        "## Future work",
        "",
        "- Real edge-hardware deployment measurements.",
        "- Stronger asset-level explanation grounding.",
        "- Any claim about real-world zero-day robustness or human-level root-cause analysis.",
    ]
    write_markdown(STATUS_PATH, "\n".join(status_lines))

    heldout_means = heldout_metrics[heldout_metrics["generator_source"] != "canonical_bundle"].agg(
        {"precision": "mean", "recall": "mean", "f1": "mean"}
    )
    balanced_means = balanced_metrics.agg({"precision": "mean", "recall": "mean", "f1": "mean"})
    discussion_lines = [
        "# Complete Results And Discussion",
        "",
        "## Canonical benchmark",
        "",
        f"- The canonical benchmark winner remains `transformer @ 60s` with benchmark F1={canonical_winner['f1']:.3f}.",
        "- This remains the source-of-truth model-selection result.",
        "",
        "## Heldout replay",
        "",
        f"- Mean original heldout replay precision / recall / F1 across LLM generators: {heldout_means['precision']:.3f} / {heldout_means['recall']:.3f} / {heldout_means['f1']:.3f}.",
        "- These are replay metrics from the frozen canonical winner, not benchmark-test metrics.",
        "",
        "## Balanced repaired replay",
        "",
        f"- Mean repaired/balanced replay precision / recall / F1: {balanced_means['precision']:.3f} / {balanced_means['recall']:.3f} / {balanced_means['f1']:.3f}.",
        f"- Repair-tagged rows in the Phase 2 master inventory: {len(repaired_rows)}.",
        "",
        "## Phase 2 scientific coverage",
        "",
        f"- Total inventory rows: {len(phase2_inventory)}.",
        f"- Validated rows used for final usable coverage: {len(validated_inventory)}.",
        f"- Distinct validated source bundles: {validated_inventory['source_bundle'].nunique()}.",
        f"- Distinct validated attack families: {validated_inventory['attack_family'].nunique()}.",
        f"- Difficulty buckets available in `{_safe_rel(ROOT / 'phase2_difficulty_calibration.csv')}`.",
        "",
        "## LoRA extension",
        "",
        (
            f"- Experimental LoRA test-generator family / asset / grounding scores: "
            f"{lora_test_row['family_accuracy']:.3f} / {lora_test_row['asset_accuracy']:.3f} / {lora_test_row['evidence_grounding_quality']:.3f}."
            if lora_test_row is not None
            else "- LoRA results were not available when this report was generated."
        ),
        "- This branch should be cited as explanation-side evidence only.",
        "",
        "## XAI validation",
        "",
        f"- Heldout family accuracy: {xai_overall['family_accuracy']:.3f}.",
        f"- Heldout asset accuracy: {xai_overall['asset_accuracy']:.3f}.",
        f"- Heldout grounding rate: {xai_overall['grounding_rate']:.3f}.",
        f"- Heldout action relevance: {xai_overall['action_relevance']:.3f}.",
        "",
        "## Deployment benchmark",
        "",
        (
            "```text\n" + deployment_actual[["model_name", "window_label", "mean_cpu_inference_ms_per_window", "throughput_windows_per_sec", "rss_peak_mb", "replay_f1"]].to_string(index=False) + "\n```"
            if deployment_actual is not None
            else "Deployment benchmark results were not available when this report was generated."
        ),
        "",
        "## Discussion",
        "",
        "- The project is now materially closer to the full methodology diagram because the missing evidence layers are present beside the canonical path instead of replacing it.",
        "- The main remaining gaps are exactly the high-risk claims that still require stronger evidence: human-like root-cause analysis, real-world zero-day robustness, and true edge deployment.",
    ]
    write_markdown(DISCUSSION_PATH, "\n".join(discussion_lines))

    claims_lines = [
        "# Complete Publication Safe Claims",
        "",
        "## Safe now",
        "",
        "- The canonical benchmark winner is transformer @ 60s.",
        "- Benchmark and replay contexts are separated.",
        "- The project now includes a validated Phase 2 coverage/diversity/difficulty audit.",
        "- The project includes a grounded explanation layer with heldout case-level validation.",
        "- The project includes an offline lightweight deployment benchmark on the current machine.",
        "",
        "## Safe with wording changes",
        "",
        "- `Tiny LLM (LoRA)` is safe only as an experimental extension branch.",
        "- `Explanation` is safe only as grounded operator-facing support and evidence-grounded family attribution.",
        "- `Deployment` is safe only as offline replay-oriented benchmarking.",
        "",
        "## Not safe",
        "",
        "- Human-like root-cause analysis.",
        "- Real-world zero-day robustness.",
        "- Real edge deployment or field validation.",
        "",
        "## Missing evidence",
        "",
        "- Edge-hardware measurements.",
        "- Human study or expert-operator study for explanation usefulness.",
        "- Real operational or truly out-of-distribution zero-day evidence.",
    ]
    write_markdown(CLAIMS_PATH, "\n".join(claims_lines))

    decision_lines = [
        "# Final Project Decision",
        "",
        "## Is the project now aligned with the diagram?",
        "",
        "Substantially yes, but only after correcting the overclaiming labels. The canonical path is preserved, and the missing evidence layers now exist beside it.",
        "",
        "## Which boxes are truly satisfied?",
        "",
        "- Phase 1 data generation, preprocessing, benchmark training, and evaluation.",
        "- Phase 2 scenario generation, validation, injection, attacked dataset creation, and now coverage/diversity auditing.",
        "- Phase 3 detector replay and evaluation layers.",
        "",
        "## Which boxes are still partial?",
        "",
        "- Tiny-LLM / LoRA: implemented only as an experimental explanation branch.",
        "- Explanation box: supported only under grounded operator-support wording.",
        "- Deployment box: supported only under offline lightweight deployment benchmarking wording.",
        "",
        "## Which claims can be made tomorrow?",
        "",
        "- In the paper: canonical benchmark winner, heldout replay separation, grounded explanation validation, Phase 2 coverage audit, and offline lightweight deployment benchmark.",
        "- In the presentation: the same, with the safe labels from `DIAGRAM_SAFE_LABELS.md`.",
        "- Do not say human-level root-cause analysis, real-world zero-day robustness, or edge deployment.",
    ]
    write_markdown(DECISION_PATH, "\n".join(decision_lines))


def main() -> None:
    xai_frame = _compute_xai_audit()
    _write_xai_reports(xai_frame)
    _write_final_reports(xai_frame)
    print(
        json.dumps(
            {
                "xai_case_level_audit": _safe_rel(XAI_CASE_AUDIT_PATH),
                "xai_error_taxonomy": _safe_rel(XAI_TAXONOMY_PATH),
                "xai_qualitative_examples": _safe_rel(XAI_EXAMPLES_PATH),
                "xai_final_validation_report": _safe_rel(XAI_REPORT_PATH),
                "final_alignment": _safe_rel(FINAL_ALIGNMENT_PATH),
                "project_decision": _safe_rel(DECISION_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
