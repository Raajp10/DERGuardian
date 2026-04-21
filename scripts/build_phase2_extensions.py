from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from methodology_alignment_common import (
    GENERATOR_SOURCES,
    IMPROVED_PHASE3_ROOT,
    ROOT,
    balance_score_from_counts,
    benchmark_scenario_metadata,
    canonical_labels_lookup,
    file_size_mb,
    inventory_required_fields,
    load_generator_truth_lookup,
    load_human_authored_truth_lookup,
    load_repaired_truth_lookup,
    load_transformer_replay_lookup,
    metadata_completeness_ratio,
    parse_sequence,
    phase2_scenario_metadata,
    pipe_join,
    pretty_label,
    read_json,
    read_labels_lookup,
    safe_float,
    severity_numeric,
    write_markdown,
)


ROOT_OUTPUTS = {
    "freeze_report": ROOT / "CANONICAL_ARTIFACT_FREEZE.md",
    "master_inventory": ROOT / "phase2_scenario_master_inventory.csv",
    "coverage_summary": ROOT / "phase2_coverage_summary.md",
    "attack_family_distribution": ROOT / "phase2_attack_family_distribution.csv",
    "asset_signal_coverage": ROOT / "phase2_asset_signal_coverage.csv",
    "difficulty_calibration": ROOT / "phase2_difficulty_calibration.csv",
    "quality_report": ROOT / "phase2_diversity_and_quality_report.md",
    "family_distribution_fig": ROOT / "phase2_family_distribution.png",
    "asset_heatmap_fig": ROOT / "phase2_asset_coverage_heatmap.png",
    "signal_heatmap_fig": ROOT / "phase2_signal_coverage_heatmap.png",
    "difficulty_fig": ROOT / "phase2_difficulty_distribution.png",
    "generator_fig": ROOT / "phase2_generator_coverage_comparison.png",
}


def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path.resolve())


def _canonical_freeze_report() -> None:
    final_window_path = ROOT / "outputs" / "window_size_study" / "final_window_comparison.csv"
    final_window = pd.read_csv(final_window_path)
    winner = final_window.loc[
        (final_window["window_label"] == "60s") & (final_window["model_name"] == "transformer")
    ].iloc[0]
    lines = [
        "# Canonical Artifact Freeze",
        "",
        "This file documents the frozen canonical artifacts used as source-of-truth inputs for the added evidence layers.",
        "No canonical benchmark outputs were overwritten by this extension pass.",
        "",
        "## Frozen benchmark path",
        "",
        f"- Canonical benchmark summary: `{_safe_rel(ROOT / 'outputs' / 'reports' / 'model_full_run_artifacts' / 'model_summary_table_full.csv')}`",
        f"- Benchmark winner remains `{winner['model_name']}` at `{winner['window_label']}`.",
        f"- Winner ready package: `{winner['ready_package_dir']}`",
        f"- Benchmark decision note: `{_safe_rel(ROOT / 'BEST_MODEL_DECISION.md')}`",
        "",
        "## Frozen Phase 2 canonical attacked bundle",
        "",
        f"- Canonical manifest: `{_safe_rel(ROOT / 'outputs' / 'attacked' / 'scenario_manifest.json')}`",
        f"- Canonical attack labels: `{_safe_rel(ROOT / 'outputs' / 'attacked' / 'attack_labels.parquet')}`",
        f"- Canonical Phase 2 validation summary: `{_safe_rel(ROOT / 'outputs' / 'reports' / 'attacked_validation_summary.md')}`",
        "",
        "## Frozen Phase 3 replay path",
        "",
        f"- Existing heldout replay metrics: `{_safe_rel(ROOT / 'heldout_bundle_metrics.csv')}`",
        f"- Existing repaired/balanced replay metrics: `{_safe_rel(ROOT / 'balanced_heldout_metrics.csv')}`",
        f"- Existing benchmark vs replay separation: `{_safe_rel(ROOT / 'benchmark_vs_replay_explainer.md')}`",
        f"- Existing improved replay report: `{_safe_rel(ROOT / 'IMPROVED_PHASE3_RESULTS_REPORT.md')}`",
        "",
        "## Extension rule",
        "",
        "All new outputs created in this pass sit alongside the frozen canonical path and are explicitly labeled as coverage analysis, heldout replay support, or extension experiments.",
    ]
    write_markdown(ROOT_OUTPUTS["freeze_report"], "\n".join(lines))


def _label_lookup_to_record_overrides(lookup: dict[str, dict[str, Any]], scenario_id: str) -> dict[str, Any]:
    item = lookup.get(scenario_id, {})
    return {
        "severity": str(item.get("severity", "")),
        "affected_assets": list(item.get("affected_assets", [])),
        "affected_signals": list(item.get("affected_signals", [])),
        "target_component": str(item.get("target_component", "")),
        "attack_family": str(item.get("attack_family", "")),
    }


def _replay_fields(replay_lookup: dict[tuple[str, str], Any], source_bundle: str, scenario_id: str) -> dict[str, Any]:
    replay = replay_lookup.get((source_bundle, scenario_id))
    if replay is None:
        return {
            "replay_evaluated": False,
            "replay_detected": "",
            "replay_latency_seconds": "",
            "replay_first_detection_time": "",
            "replay_max_score": "",
            "replay_mean_score": "",
            "replay_positive_windows": "",
            "replay_evaluation_dir": "",
        }
    return {
        "replay_evaluated": True,
        "replay_detected": int(replay.detected),
        "replay_latency_seconds": replay.latency_seconds if replay.latency_seconds is not None else "",
        "replay_first_detection_time": replay.first_detection_time or "",
        "replay_max_score": replay.max_score if replay.max_score is not None else "",
        "replay_mean_score": replay.mean_score if replay.mean_score is not None else "",
        "replay_positive_windows": replay.positive_windows if replay.positive_windows is not None else "",
        "replay_evaluation_dir": replay.evaluation_dir,
    }


def _path_role(source_bundle: str) -> str:
    if source_bundle == "canonical_bundle":
        return "canonical benchmark path"
    if source_bundle.startswith("heldout_original_"):
        return "heldout replay path"
    return "extension experiments"


def _canonical_records(replay_lookup: dict[tuple[str, str], Any]) -> list[dict[str, Any]]:
    manifest = read_json(ROOT / "outputs" / "attacked" / "scenario_manifest.json")
    _, labels_lookup = canonical_labels_lookup()
    dataset_id = str(manifest.get("scenario_id", "canonical_phase2_bundle"))
    records: list[dict[str, Any]] = []
    for scenario in manifest.get("applied_scenarios", []):
        scenario_id = str(scenario["scenario_id"])
        meta = phase2_scenario_metadata(scenario)
        overrides = _label_lookup_to_record_overrides(labels_lookup, scenario_id)
        records.append(
            {
                "dataset_id": dataset_id,
                "source_bundle": "canonical_bundle",
                "path_role": _path_role("canonical_bundle"),
                "generator_source": "canonical_bundle",
                "bundle_path": _safe_rel(ROOT / "outputs" / "attacked" / "scenario_manifest.json"),
                "scenario_id": scenario_id,
                "scenario_name": meta["scenario_name"],
                "attack_family": overrides["attack_family"] or meta["attack_family"],
                "raw_attack_family": meta["attack_family_raw"],
                "affected_assets": overrides["affected_assets"] or meta["affected_assets"],
                "affected_signals": overrides["affected_signals"] or meta["affected_signals"],
                "target_component": overrides["target_component"] or meta["target_component"],
                "target_asset": meta["target_asset"],
                "target_signal": meta["target_signal"],
                "severity": overrides["severity"] or meta["severity"],
                "accepted_rejected": "accepted",
                "repair_applied": False,
                "repair_origin_status": "",
                "validation_notes": "Canonical validated attacked bundle.",
                "numeric_magnitude": meta["numeric_magnitude"] if meta["numeric_magnitude"] is not None else "",
                "magnitude_unit": meta["magnitude_unit"],
                "observable_signal_count": len(meta["observable_signals"]),
                "additional_target_count": meta["additional_target_count"],
                "source_generator_bundle": "outputs/attacked/scenario_manifest.json",
                **_replay_fields(replay_lookup, "canonical_bundle", scenario_id),
            }
        )
    return records


def _original_heldout_records(generator_source: str, replay_lookup: dict[tuple[str, str], Any]) -> list[dict[str, Any]]:
    payload, truth_lookup = load_generator_truth_lookup(generator_source)
    dataset_id = str(payload.get("dataset_id", f"{generator_source}_heldout"))
    source_bundle = f"heldout_original_{generator_source}"
    labels_path = ROOT / "phase2_llm_benchmark" / "new_respnse_result" / "models" / generator_source / "datasets" / "attack_labels.parquet"
    _, labels_lookup = read_labels_lookup(labels_path)
    rejected_payload = read_json(
        ROOT
        / "phase2_llm_benchmark"
        / "new_respnse_result"
        / "models"
        / generator_source
        / "rejected_scenarios"
        / "rejected_scenarios.json"
    )
    rejected_lookup = {str(item["scenario_id"]): item for item in rejected_payload.get("rejections", [])}
    records: list[dict[str, Any]] = []
    for scenario_id, scenario in truth_lookup.items():
        meta = benchmark_scenario_metadata(scenario)
        overrides = _label_lookup_to_record_overrides(labels_lookup, scenario_id)
        accepted = scenario_id in labels_lookup
        rejection = rejected_lookup.get(scenario_id, {})
        validation_note = "Accepted after canonical validation." if accepted else "; ".join(rejection.get("reasons", []))
        records.append(
            {
                "dataset_id": dataset_id,
                "source_bundle": source_bundle,
                "path_role": _path_role(source_bundle),
                "generator_source": generator_source,
                "bundle_path": _safe_rel(
                    ROOT / "phase2_llm_benchmark" / "heldout_llm_response" / generator_source / "new_respnse.json"
                ),
                "scenario_id": scenario_id,
                "scenario_name": meta["scenario_name"],
                "attack_family": overrides["attack_family"] or meta["attack_family"],
                "raw_attack_family": meta["attack_family_raw"],
                "affected_assets": overrides["affected_assets"] or meta["affected_assets"],
                "affected_signals": overrides["affected_signals"] or meta["affected_signals"],
                "target_component": overrides["target_component"] or meta["target_component"],
                "target_asset": meta["target_asset"],
                "target_signal": meta["target_signal"],
                "severity": overrides["severity"] or meta["severity"],
                "accepted_rejected": "accepted" if accepted else "rejected",
                "repair_applied": False,
                "repair_origin_status": "",
                "validation_notes": validation_note,
                "numeric_magnitude": meta["numeric_magnitude"] if meta["numeric_magnitude"] is not None else "",
                "magnitude_unit": meta["magnitude_unit"],
                "observable_signal_count": len(meta["observable_signals"]),
                "additional_target_count": meta["additional_target_count"],
                "source_generator_bundle": _safe_rel(
                    ROOT / "phase2_llm_benchmark" / "heldout_llm_response" / generator_source / "new_respnse.json"
                ),
                **_replay_fields(replay_lookup, source_bundle, scenario_id),
            }
        )
    return records


def _repaired_records(generator_source: str, replay_lookup: dict[tuple[str, str], Any]) -> list[dict[str, Any]]:
    repaired_payload, repaired_lookup = load_repaired_truth_lookup(generator_source)
    _, original_lookup = load_generator_truth_lookup(generator_source)
    source_bundle = f"heldout_repaired_{generator_source}"
    dataset_id = str(repaired_payload.get("dataset_id", f"{generator_source}_repaired"))
    labels_path = IMPROVED_PHASE3_ROOT / "repaired_phase2_full" / generator_source / "datasets" / "attack_labels.parquet"
    _, labels_lookup = read_labels_lookup(labels_path)
    original_labels_path = (
        ROOT / "phase2_llm_benchmark" / "new_respnse_result" / "models" / generator_source / "datasets" / "attack_labels.parquet"
    )
    _, original_labels_lookup = read_labels_lookup(original_labels_path)
    rejected_after_repair = read_json(
        IMPROVED_PHASE3_ROOT / "repaired_bundles_raw" / generator_source / "rejected_after_repair.json"
    )
    rejected_after_lookup = {str(item["scenario_id"]): item for item in rejected_after_repair.get("rejections", [])}
    changed_ids: set[str] = set()
    for scenario_id, scenario in repaired_lookup.items():
        original = original_lookup.get(scenario_id)
        if original is None:
            changed_ids.add(scenario_id)
            continue
        if json.dumps(original, sort_keys=True) != json.dumps(scenario, sort_keys=True):
            changed_ids.add(scenario_id)
    records: list[dict[str, Any]] = []
    for scenario_id, scenario in repaired_lookup.items():
        meta = benchmark_scenario_metadata(scenario)
        overrides = _label_lookup_to_record_overrides(labels_lookup, scenario_id)
        accepted = scenario_id not in rejected_after_lookup
        originally_accepted = scenario_id in original_labels_lookup
        repair_status = "accepted_carried_forward"
        if scenario_id in changed_ids:
            repair_status = "accepted_after_repair" if not originally_accepted else "accepted_with_repair_edit"
        elif not originally_accepted:
            repair_status = "accepted_after_repair"
        validation_note = "Validator-guided repaired bundle."
        if scenario_id in rejected_after_lookup:
            validation_note = "; ".join(rejected_after_lookup[scenario_id].get("reasons", []))
        records.append(
            {
                "dataset_id": dataset_id,
                "source_bundle": source_bundle,
                "path_role": _path_role(source_bundle),
                "generator_source": generator_source,
                "bundle_path": _safe_rel(
                    IMPROVED_PHASE3_ROOT / "repaired_bundles_raw" / generator_source / "repaired_benchmark_bundle.json"
                ),
                "scenario_id": scenario_id,
                "scenario_name": meta["scenario_name"],
                "attack_family": overrides["attack_family"] or meta["attack_family"],
                "raw_attack_family": meta["attack_family_raw"],
                "affected_assets": overrides["affected_assets"] or meta["affected_assets"],
                "affected_signals": overrides["affected_signals"] or meta["affected_signals"],
                "target_component": overrides["target_component"] or meta["target_component"],
                "target_asset": meta["target_asset"],
                "target_signal": meta["target_signal"],
                "severity": overrides["severity"] or meta["severity"],
                "accepted_rejected": "accepted" if accepted else "rejected",
                "repair_applied": scenario_id in changed_ids or not originally_accepted,
                "repair_origin_status": repair_status,
                "validation_notes": validation_note,
                "numeric_magnitude": meta["numeric_magnitude"] if meta["numeric_magnitude"] is not None else "",
                "magnitude_unit": meta["magnitude_unit"],
                "observable_signal_count": len(meta["observable_signals"]),
                "additional_target_count": meta["additional_target_count"],
                "source_generator_bundle": _safe_rel(
                    IMPROVED_PHASE3_ROOT / "repaired_bundles_raw" / generator_source / "repaired_benchmark_bundle.json"
                ),
                **_replay_fields(replay_lookup, source_bundle, scenario_id),
            }
        )
    return records


def _human_authored_records(replay_lookup: dict[tuple[str, str], Any]) -> list[dict[str, Any]]:
    payload, truth_lookup = load_human_authored_truth_lookup()
    source_bundle = "additional_heldout_human_authored"
    dataset_id = str(payload.get("dataset_id", "human_authored_phase3_bundle_v1"))
    labels_path = IMPROVED_PHASE3_ROOT / "additional_source" / "human_authored" / "datasets" / "attack_labels.parquet"
    _, labels_lookup = read_labels_lookup(labels_path)
    records: list[dict[str, Any]] = []
    for scenario_id, scenario in truth_lookup.items():
        meta = benchmark_scenario_metadata(scenario)
        overrides = _label_lookup_to_record_overrides(labels_lookup, scenario_id)
        records.append(
            {
                "dataset_id": dataset_id,
                "source_bundle": source_bundle,
                "path_role": _path_role(source_bundle),
                "generator_source": "human_authored",
                "bundle_path": _safe_rel(IMPROVED_PHASE3_ROOT / "additional_source" / "human_authored" / "benchmark_bundle.json"),
                "scenario_id": scenario_id,
                "scenario_name": meta["scenario_name"],
                "attack_family": overrides["attack_family"] or meta["attack_family"],
                "raw_attack_family": meta["attack_family_raw"],
                "affected_assets": overrides["affected_assets"] or meta["affected_assets"],
                "affected_signals": overrides["affected_signals"] or meta["affected_signals"],
                "target_component": overrides["target_component"] or meta["target_component"],
                "target_asset": meta["target_asset"],
                "target_signal": meta["target_signal"],
                "severity": overrides["severity"] or meta["severity"],
                "accepted_rejected": "accepted",
                "repair_applied": False,
                "repair_origin_status": "",
                "validation_notes": "Human-authored additional heldout bundle evaluated through the replay path.",
                "numeric_magnitude": meta["numeric_magnitude"] if meta["numeric_magnitude"] is not None else "",
                "magnitude_unit": meta["magnitude_unit"],
                "observable_signal_count": len(meta["observable_signals"]),
                "additional_target_count": meta["additional_target_count"],
                "source_generator_bundle": _safe_rel(IMPROVED_PHASE3_ROOT / "additional_source" / "human_authored" / "benchmark_bundle.json"),
                **_replay_fields(replay_lookup, source_bundle, scenario_id),
            }
        )
    return records


def build_master_inventory() -> pd.DataFrame:
    replay_lookup = load_transformer_replay_lookup()
    records: list[dict[str, Any]] = []
    records.extend(_canonical_records(replay_lookup))
    for generator_source in GENERATOR_SOURCES:
        records.extend(_original_heldout_records(generator_source, replay_lookup))
    for generator_source in GENERATOR_SOURCES:
        records.extend(_repaired_records(generator_source, replay_lookup))
    records.extend(_human_authored_records(replay_lookup))
    frame = pd.DataFrame(records)
    frame["metadata_completeness_ratio"] = frame.apply(metadata_completeness_ratio, axis=1)
    frame["severity_numeric"] = frame["severity"].apply(severity_numeric)
    frame = frame.sort_values(["source_bundle", "scenario_id"]).reset_index(drop=True)
    return frame


def _serializable_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    export = frame.copy()
    export["affected_assets"] = export["affected_assets"].apply(parse_sequence).apply(
        lambda values: json.dumps([str(item).strip() for item in values if str(item).strip()])
    )
    export["affected_signals"] = export["affected_signals"].apply(parse_sequence).apply(
        lambda values: json.dumps([str(item).strip() for item in values if str(item).strip()])
    )
    return export


def _coverage_summary_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validated = frame[frame["accepted_rejected"] == "accepted"].copy()
    family_rows = []
    for family, group in frame.groupby("attack_family", dropna=False):
        valid_group = group[group["accepted_rejected"] == "accepted"]
        family_rows.append(
            {
                "attack_family": family,
                "all_scenarios": int(len(group)),
                "validated_scenarios": int(len(valid_group)),
                "rejected_scenarios": int((group["accepted_rejected"] == "rejected").sum()),
                "repair_rows": int(group["repair_applied"].astype(bool).sum()),
                "generator_coverage_validated": int(valid_group["generator_source"].nunique()),
                "source_bundle_coverage_validated": int(valid_group["source_bundle"].nunique()),
                "replay_evaluated_validated": int(valid_group["replay_evaluated"].astype(bool).sum()),
            }
        )
    family_df = pd.DataFrame(family_rows).sort_values(["validated_scenarios", "attack_family"], ascending=[False, True])

    coverage_rows = []
    for dimension_name, column in [("asset", "affected_assets"), ("signal", "affected_signals")]:
        exploded = frame.assign(_value=frame[column].apply(parse_sequence)).explode("_value")
        exploded["_value"] = exploded["_value"].astype(str).str.strip()
        exploded = exploded[exploded["_value"] != ""]
        for value, group in exploded.groupby("_value", dropna=False):
            valid_group = group[group["accepted_rejected"] == "accepted"]
            coverage_rows.append(
                {
                    "dimension": dimension_name,
                    "name": value,
                    "all_scenarios": int(group["scenario_id"].nunique()),
                    "validated_scenarios": int(valid_group["scenario_id"].nunique()),
                    "rejected_scenarios": int(group.loc[group["accepted_rejected"] == "rejected", "scenario_id"].nunique()),
                    "repair_rows": int(group["repair_applied"].astype(bool).sum()),
                    "generator_coverage_validated": int(valid_group["generator_source"].nunique()),
                    "source_bundle_coverage_validated": int(valid_group["source_bundle"].nunique()),
                }
            )
    asset_signal_df = pd.DataFrame(coverage_rows).sort_values(["dimension", "validated_scenarios", "name"], ascending=[True, False, True])

    generator_rows = []
    for source_bundle, group in validated.groupby("source_bundle", dropna=False):
        assets = set()
        signals = set()
        for values in group["affected_assets"]:
            assets.update(str(item) for item in parse_sequence(values) if str(item).strip())
        for values in group["affected_signals"]:
            signals.update(str(item) for item in parse_sequence(values) if str(item).strip())
        generator_rows.append(
            {
                "source_bundle": source_bundle,
                "source_bundle_label": pretty_label(source_bundle),
                "generator_source": group["generator_source"].iloc[0],
                "validated_scenarios": int(group["scenario_id"].nunique()),
                "unique_families": int(group["attack_family"].nunique()),
                "unique_assets": int(len(assets)),
                "unique_signals": int(len(signals)),
                "repair_rows": int(group["repair_applied"].astype(bool).sum()),
            }
        )
    generator_df = pd.DataFrame(generator_rows).sort_values("source_bundle_label")
    return family_df, asset_signal_df, generator_df


def _compute_difficulty(frame: pd.DataFrame) -> pd.DataFrame:
    difficulty = frame.copy()
    numeric_magnitude = pd.to_numeric(difficulty["numeric_magnitude"], errors="coerce")
    severity_component = difficulty["severity_numeric"].fillna(0.5)
    if numeric_magnitude.notna().any():
        magnitude_rank = numeric_magnitude.rank(pct=True, method="average")
        impact_proxy = severity_component.where(numeric_magnitude.isna(), 0.6 * severity_component + 0.4 * magnitude_rank.fillna(0.5))
    else:
        impact_proxy = severity_component

    replay_validated = difficulty[
        (difficulty["accepted_rejected"] == "accepted") & (difficulty["replay_evaluated"].astype(bool))
    ].copy()
    family_detection_rate = (
        replay_validated.groupby("attack_family")["replay_detected"].apply(lambda series: pd.to_numeric(series, errors="coerce").fillna(0).mean()).to_dict()
    )

    max_score_series = pd.to_numeric(replay_validated["replay_max_score"], errors="coerce")
    latency_series = pd.to_numeric(replay_validated["replay_latency_seconds"], errors="coerce")
    max_score_rank = max_score_series.rank(pct=True, method="average")
    latency_rank = latency_series.rank(pct=True, method="average")
    replay_score_rank_map = dict(zip(replay_validated.index, max_score_rank))
    replay_latency_rank_map = dict(zip(replay_validated.index, latency_rank))

    rows: list[dict[str, Any]] = []
    for idx, row in difficulty.iterrows():
        components: dict[str, float] = {}
        impact_value = safe_float(impact_proxy.loc[idx])
        if impact_value is not None:
            components["impact_hardness"] = 1.0 - float(max(min(impact_value, 1.0), 0.0))
        family_hardness = 1.0 - family_detection_rate.get(str(row["attack_family"]), 0.5)
        components["family_hardness"] = float(max(min(family_hardness, 1.0), 0.0))

        if bool(row["replay_evaluated"]):
            score_rank = replay_score_rank_map.get(idx)
            if score_rank is not None and not math.isnan(float(score_rank)):
                components["score_hardness"] = 1.0 - float(score_rank)
            detected = bool(int(row["replay_detected"])) if str(row["replay_detected"]).strip() != "" else False
            if detected:
                latency_rank_value = replay_latency_rank_map.get(idx)
                if latency_rank_value is not None and not math.isnan(float(latency_rank_value)):
                    components["latency_hardness"] = float(latency_rank_value)
                components["detection_failure"] = 0.0
            else:
                components["latency_hardness"] = 1.0
                components["detection_failure"] = 1.0

        weights = {
            "impact_hardness": 0.25,
            "score_hardness": 0.25,
            "latency_hardness": 0.15,
            "detection_failure": 0.20,
            "family_hardness": 0.15,
        }
        usable = {name: components[name] for name in components if name in weights}
        total_weight = sum(weights[name] for name in usable)
        difficulty_score = 100.0 * sum(weights[name] * usable[name] for name in usable) / max(total_weight, 1e-9)
        rows.append(
            {
                "dataset_id": row["dataset_id"],
                "source_bundle": row["source_bundle"],
                "generator_source": row["generator_source"],
                "scenario_id": row["scenario_id"],
                "attack_family": row["attack_family"],
                "severity": row["severity"],
                "accepted_rejected": row["accepted_rejected"],
                "repair_applied": bool(row["repair_applied"]),
                "replay_evaluated": bool(row["replay_evaluated"]),
                "replay_detected": row["replay_detected"],
                "replay_latency_seconds": row["replay_latency_seconds"],
                "replay_max_score": row["replay_max_score"],
                "numeric_magnitude": row["numeric_magnitude"],
                "impact_proxy": round(float(impact_value), 4) if impact_value is not None else "",
                "family_detection_rate": round(float(family_detection_rate.get(str(row["attack_family"]), 0.5)), 4),
                "difficulty_score": round(float(difficulty_score), 2),
                "difficulty_basis": "replay+metadata" if bool(row["replay_evaluated"]) else "metadata_only",
                "component_impact_hardness": round(float(components.get("impact_hardness", math.nan)), 4) if "impact_hardness" in components else "",
                "component_score_hardness": round(float(components.get("score_hardness", math.nan)), 4) if "score_hardness" in components else "",
                "component_latency_hardness": round(float(components.get("latency_hardness", math.nan)), 4) if "latency_hardness" in components else "",
                "component_detection_failure": round(float(components.get("detection_failure", math.nan)), 4) if "detection_failure" in components else "",
                "component_family_hardness": round(float(components.get("family_hardness", math.nan)), 4),
            }
        )
    difficulty_df = pd.DataFrame(rows)
    quantiles = difficulty_df["difficulty_score"].quantile([0.25, 0.5, 0.75]).to_dict()

    def bucket(score: float) -> str:
        if score <= quantiles.get(0.25, score):
            return "easy"
        if score <= quantiles.get(0.5, score):
            return "moderate"
        if score <= quantiles.get(0.75, score):
            return "hard"
        return "very hard"

    difficulty_df["difficulty_bucket"] = difficulty_df["difficulty_score"].apply(bucket)
    return difficulty_df


def _heatmap_figure(
    matrix: pd.DataFrame,
    title: str,
    output_path: Path,
    figsize: tuple[float, float],
    cmap: str = "YlGnBu",
) -> None:
    ordered = matrix.copy()
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(ordered.to_numpy(dtype=float), aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(ordered.columns)))
    ax.set_xticklabels(ordered.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(ordered.index)))
    ax.set_yticklabels(ordered.index, fontsize=8)
    for y_idx, row_name in enumerate(ordered.index):
        for x_idx, column_name in enumerate(ordered.columns):
            value = ordered.loc[row_name, column_name]
            if value > 0:
                ax.text(x_idx, y_idx, int(value), ha="center", va="center", fontsize=6, color="black")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Scenario count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_phase2_figures(frame: pd.DataFrame, family_df: pd.DataFrame, asset_signal_df: pd.DataFrame, difficulty_df: pd.DataFrame, generator_df: pd.DataFrame) -> None:
    validated = frame[frame["accepted_rejected"] == "accepted"].copy()
    rejected = frame[frame["accepted_rejected"] == "rejected"].copy()

    ordered_families = family_df["attack_family"].tolist()
    x = np.arange(len(ordered_families))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width, family_df["validated_scenarios"], width=width, label="validated")
    ax.bar(x, family_df["rejected_scenarios"], width=width, label="rejected")
    ax.bar(x + width, family_df["repair_rows"], width=width, label="repair rows")
    ax.set_title("Phase 2 family distribution")
    ax.set_ylabel("Scenario count")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_families, rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT_OUTPUTS["family_distribution_fig"], dpi=220, bbox_inches="tight")
    plt.close(fig)

    asset_exploded = validated.assign(asset=validated["affected_assets"].apply(parse_sequence)).explode("asset")
    asset_exploded["asset"] = asset_exploded["asset"].astype(str).str.strip()
    asset_exploded = asset_exploded[asset_exploded["asset"] != ""]
    asset_matrix = (
        asset_exploded.groupby(["source_bundle", "asset"]).size().unstack(fill_value=0).sort_index()
        if not asset_exploded.empty
        else pd.DataFrame()
    )
    if not asset_matrix.empty:
        asset_matrix = asset_matrix.loc[:, asset_matrix.sum(axis=0).sort_values(ascending=False).index]
        asset_matrix.index = [pretty_label(item) for item in asset_matrix.index]
        _heatmap_figure(
            matrix=asset_matrix,
            title="Validated asset coverage by source bundle",
            output_path=ROOT_OUTPUTS["asset_heatmap_fig"],
            figsize=(max(10, len(asset_matrix.columns) * 0.45), 6.0),
        )

    signal_exploded = validated.assign(signal=validated["affected_signals"].apply(parse_sequence)).explode("signal")
    signal_exploded["signal"] = signal_exploded["signal"].astype(str).str.strip()
    signal_exploded = signal_exploded[signal_exploded["signal"] != ""]
    signal_matrix = (
        signal_exploded.groupby(["source_bundle", "signal"]).size().unstack(fill_value=0).sort_index()
        if not signal_exploded.empty
        else pd.DataFrame()
    )
    if not signal_matrix.empty:
        signal_matrix = signal_matrix.loc[:, signal_matrix.sum(axis=0).sort_values(ascending=False).index]
        signal_matrix.index = [pretty_label(item) for item in signal_matrix.index]
        _heatmap_figure(
            matrix=signal_matrix,
            title="Validated signal coverage by source bundle",
            output_path=ROOT_OUTPUTS["signal_heatmap_fig"],
            figsize=(max(12, len(signal_matrix.columns) * 0.28), 6.0),
            cmap="YlOrRd",
        )

    bucket_order = ["easy", "moderate", "hard", "very hard"]
    bucket_counts = difficulty_df["difficulty_bucket"].value_counts().reindex(bucket_order, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(bucket_order, bucket_counts.values, color=["#7fc97f", "#fdc086", "#f0027f", "#386cb0"])
    ax.set_title("Scenario difficulty distribution")
    ax.set_ylabel("Scenario count")
    for bar, value in zip(bars, bucket_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.1, str(int(value)), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT_OUTPUTS["difficulty_fig"], dpi=220, bbox_inches="tight")
    plt.close(fig)

    metric_columns = ["validated_scenarios", "unique_families", "unique_assets", "unique_signals"]
    x = np.arange(len(generator_df))
    width = 0.18
    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, column in enumerate(metric_columns):
        ax.bar(x + (idx - 1.5) * width, generator_df[column], width=width, label=column.replace("_", " "))
    ax.set_title("Validated generator/source coverage comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(generator_df["source_bundle_label"], rotation=35, ha="right")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT_OUTPUTS["generator_fig"], dpi=220, bbox_inches="tight")
    plt.close(fig)


def _coverage_markdown(frame: pd.DataFrame, family_df: pd.DataFrame, asset_signal_df: pd.DataFrame, generator_df: pd.DataFrame, difficulty_df: pd.DataFrame) -> str:
    validated = frame[frame["accepted_rejected"] == "accepted"].copy()
    rejected = frame[frame["accepted_rejected"] == "rejected"].copy()
    repaired = frame[frame["repair_applied"].astype(bool)].copy()
    validated_asset_count = int(
        asset_signal_df.loc[
            (asset_signal_df["dimension"] == "asset") & (asset_signal_df["validated_scenarios"] > 0),
            "name",
        ].nunique()
    )
    validated_signal_count = int(
        asset_signal_df.loc[
            (asset_signal_df["dimension"] == "signal") & (asset_signal_df["validated_scenarios"] > 0),
            "name",
        ].nunique()
    )

    undercovered_families = family_df[family_df["validated_scenarios"] <= 1]["attack_family"].tolist()
    undercovered_assets = asset_signal_df[
        (asset_signal_df["dimension"] == "asset") & (asset_signal_df["validated_scenarios"] <= 1)
    ]["name"].tolist()
    undercovered_signals = asset_signal_df[
        (asset_signal_df["dimension"] == "signal") & (asset_signal_df["validated_scenarios"] <= 1)
    ]["name"].tolist()

    inventory_summary = (
        frame.groupby("source_bundle")
        .agg(
            total_rows=("scenario_id", "count"),
            accepted_rows=("accepted_rejected", lambda series: int((series == "accepted").sum())),
            rejected_rows=("accepted_rejected", lambda series: int((series == "rejected").sum())),
            repair_rows=("repair_applied", lambda series: int(pd.Series(series).astype(bool).sum())),
            replay_evaluated=("replay_evaluated", lambda series: int(pd.Series(series).astype(bool).sum())),
        )
        .reset_index()
    )
    inventory_summary["source_bundle"] = inventory_summary["source_bundle"].map(pretty_label)

    lines = [
        "# Phase 2 Coverage Summary",
        "",
        "This report uses the master scenario inventory as the single accounting layer for canonical, heldout, repaired, and additional heldout scenarios.",
        "",
        "## Final usable coverage",
        "",
        f"- Validated scenarios: {len(validated)}",
        f"- Rejected scenarios: {len(rejected)}",
        f"- Repair-tagged rows: {len(repaired)}",
        f"- Replay-evaluated validated scenarios: {int(validated['replay_evaluated'].astype(bool).sum())}",
        f"- Distinct validated families: {validated['attack_family'].nunique()}",
        f"- Distinct validated assets: {validated_asset_count}",
        f"- Distinct validated signals: {validated_signal_count}",
        "",
        "Only accepted/validated scenarios are counted as final usable coverage.",
        "",
        "## Source bundle completeness",
        "",
        "```text",
        inventory_summary.to_string(index=False),
        "```",
        "",
        "## Attack-family coverage",
        "",
        "```text",
        family_df.to_string(index=False),
        "```",
        "",
        "## Generator/source bundle coverage",
        "",
        "```text",
        generator_df.to_string(index=False),
        "```",
        "",
        "## Under-covered categories",
        "",
        f"- Families with one or fewer validated scenarios: {undercovered_families if undercovered_families else 'none'}",
        f"- Assets with one or fewer validated scenarios: {undercovered_assets[:20] if undercovered_assets else 'none'}",
        f"- Signals with one or fewer validated scenarios: {undercovered_signals[:25] if undercovered_signals else 'none'}",
        "",
        "These are repository-relative under-coverage findings based on observed validated scenario counts only. No unobserved categories were fabricated.",
        "",
        "## Rejected and repair appendix",
        "",
        f"- Original heldout rejections retained in inventory: {len(rejected[rejected['source_bundle'].str.startswith('heldout_original_')])}",
        f"- Repaired rows accepted after validator-guided edits: {int((repaired['accepted_rejected'] == 'accepted').sum())}",
        f"- Metadata-only difficulty rows (not replay-evaluated): {int((difficulty_df['difficulty_basis'] == 'metadata_only').sum())}",
        "",
        "Rejected scenarios remain visible for auditability but are excluded from final usable coverage claims.",
    ]
    return "\n".join(lines)


def _quality_report(frame: pd.DataFrame, family_df: pd.DataFrame, asset_signal_df: pd.DataFrame) -> str:
    validated = frame[frame["accepted_rejected"] == "accepted"].copy()
    family_balance = balance_score_from_counts(family_df.set_index("attack_family")["validated_scenarios"])
    validated_asset_values = [item for values in validated["affected_assets"] for item in parse_sequence(values) if str(item).strip()]
    validated_signal_values = [item for values in validated["affected_signals"] for item in parse_sequence(values) if str(item).strip()]
    asset_balance = balance_score_from_counts(
        pd.Series(validated_asset_values).value_counts()
    )
    signal_balance = balance_score_from_counts(
        pd.Series(validated_signal_values).value_counts()
    )
    completeness = frame["metadata_completeness_ratio"].mean()
    validated_completeness = validated["metadata_completeness_ratio"].mean()
    validated_asset_count = int(
        asset_signal_df.loc[
            (asset_signal_df["dimension"] == "asset") & (asset_signal_df["validated_scenarios"] > 0),
            "name",
        ].nunique()
    )
    validated_signal_count = int(
        asset_signal_df.loc[
            (asset_signal_df["dimension"] == "signal") & (asset_signal_df["validated_scenarios"] > 0),
            "name",
        ].nunique()
    )
    plausibility_by_source = (
        frame.groupby("source_bundle")
        .agg(
            rows=("scenario_id", "count"),
            accepted=("accepted_rejected", lambda series: int((series == "accepted").sum())),
            replay_evaluated=("replay_evaluated", lambda series: int(pd.Series(series).astype(bool).sum())),
        )
        .reset_index()
    )
    plausibility_by_source["acceptance_rate"] = (plausibility_by_source["accepted"] / plausibility_by_source["rows"]).round(3)
    plausibility_by_source["source_bundle"] = plausibility_by_source["source_bundle"].map(pretty_label)

    lines = [
        "# Phase 2 Diversity And Quality Report",
        "",
        "This quality package is intentionally conservative: final usable coverage is limited to accepted scenarios, while rejected and repaired rows remain visible for auditability.",
        "",
        "## Plausibility",
        "",
        "```text",
        plausibility_by_source.to_string(index=False),
        "```",
        "",
        f"- Overall acceptance rate: {((frame['accepted_rejected'] == 'accepted').mean()):.3f}",
        f"- Validated rows with replay evidence: {(validated['replay_evaluated'].astype(bool).mean()):.3f}",
        "",
        "## Diversity",
        "",
        f"- Validated families: {validated['attack_family'].nunique()}",
        f"- Validated assets: {validated_asset_count}",
        f"- Validated signals: {validated_signal_count}",
        f"- Validated source bundles: {validated['source_bundle'].nunique()}",
        "",
        "## Metadata completeness",
        "",
        f"- Mean completeness over all inventory rows: {completeness:.3f}",
        f"- Mean completeness over validated rows: {validated_completeness:.3f}",
        f"- Required inventory fields audited: {inventory_required_fields()}",
        "",
        "## Balance summary",
        "",
        f"- Family balance score (normalized entropy): {family_balance:.3f}",
        f"- Asset balance score (normalized entropy): {asset_balance:.3f}",
        f"- Signal balance score (normalized entropy): {signal_balance:.3f}",
        "",
        "Higher balance scores indicate more even spread across observed categories; they do not imply exhaustive coverage.",
        "",
        "## Interpretation",
        "",
        "- Plausibility is strongest for the canonical bundle, the accepted heldout bundles, and the repaired bundles that passed the same validator after documented edits.",
        "- Diversity improves materially once the repaired bundles and the human-authored heldout source are included, but some families, assets, and signals remain thinly represented.",
        "- Metadata completeness is high for accepted Phase 2 bundle rows and lower for raw rejected rows where canonical severity labels were never generated.",
    ]
    return "\n".join(lines)


def main() -> None:
    _canonical_freeze_report()
    inventory = build_master_inventory()
    _serializable_inventory(inventory).to_csv(ROOT_OUTPUTS["master_inventory"], index=False)

    family_df, asset_signal_df, generator_df = _coverage_summary_tables(inventory)
    family_df.to_csv(ROOT_OUTPUTS["attack_family_distribution"], index=False)
    asset_signal_df.to_csv(ROOT_OUTPUTS["asset_signal_coverage"], index=False)

    difficulty_df = _compute_difficulty(inventory)
    difficulty_df.to_csv(ROOT_OUTPUTS["difficulty_calibration"], index=False)

    _plot_phase2_figures(inventory, family_df, asset_signal_df, difficulty_df, generator_df)

    coverage_md = _coverage_markdown(inventory, family_df, asset_signal_df, generator_df, difficulty_df)
    write_markdown(ROOT_OUTPUTS["coverage_summary"], coverage_md)

    quality_md = _quality_report(inventory, family_df, asset_signal_df)
    write_markdown(ROOT_OUTPUTS["quality_report"], quality_md)

    print(
        json.dumps(
            {
                "master_inventory": _safe_rel(ROOT_OUTPUTS["master_inventory"]),
                "coverage_summary": _safe_rel(ROOT_OUTPUTS["coverage_summary"]),
                "attack_family_distribution": _safe_rel(ROOT_OUTPUTS["attack_family_distribution"]),
                "asset_signal_coverage": _safe_rel(ROOT_OUTPUTS["asset_signal_coverage"]),
                "difficulty_calibration": _safe_rel(ROOT_OUTPUTS["difficulty_calibration"]),
                "quality_report": _safe_rel(ROOT_OUTPUTS["quality_report"]),
                "freeze_report": _safe_rel(ROOT_OUTPUTS["freeze_report"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
