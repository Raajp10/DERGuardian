"""Phase 3 evaluation and analysis support for DERGuardian.

This module implements error analysis logic for detector evaluation, ablations,
zero-day-like heldout synthetic analysis, latency sweeps, or final reporting.
It keeps benchmark, replay, heldout synthetic, and extension results separated.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from common.io_utils import ensure_dir
from phase1_models.model_utils import display_model_name


SCENARIO_ID_PATTERN = re.compile(r"(scn_[A-Za-z0-9_]+)")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _safe_float(value: Any) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    return int(round(_safe_float(value)))


def _safe_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _visibility_signature(entry: dict[str, Any]) -> str:
    target_component = str(entry.get("target_component", "unknown"))
    additional_target_count = _safe_int(entry.get("additional_target_count", 0))
    if target_component == "measured_layer":
        return "measured_primary"
    if target_component in {"pv", "bess", "regulator", "capacitor", "switch"} and additional_target_count > 0:
        return "coordinated_physical_primary"
    if target_component in {"pv", "bess", "regulator", "capacitor", "switch"}:
        return "physical_primary"
    return "unknown"


def _match_explanation_scenario(payload: dict[str, Any]) -> str:
    for candidate in [
        payload.get("scenario_id"),
        payload.get("holdout_scenario"),
        payload.get("offline_evaluation", {}).get("scenario_label"),
        payload.get("suspected_attack_family"),
    ]:
        if isinstance(candidate, str) and candidate.startswith("scn_"):
            return candidate
    text_fields: list[str] = []
    for key in ("summary", "incident_id"):
        value = payload.get(key)
        if isinstance(value, str):
            text_fields.append(value)
    for key in ("cyber_evidence_used", "why_flagged"):
        for item in _safe_list(payload.get(key)):
            if isinstance(item, dict):
                for sub_value in item.values():
                    if isinstance(sub_value, str):
                        text_fields.append(sub_value)
            elif isinstance(item, str):
                text_fields.append(item)
    for text in text_fields:
        match = SCENARIO_ID_PATTERN.search(text)
        if match:
            return match.group(1)
    return ""


def _load_explanation_lookup(report_root: Path) -> dict[str, dict[str, str]]:
    explanation_root = report_root / "explanation_artifacts" / "explanations"
    lookup: dict[str, dict[str, str]] = {}
    if not explanation_root.exists():
        return lookup
    for path in sorted(explanation_root.glob("*.json")):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        scenario_id = _match_explanation_scenario(payload)
        summary = str(payload.get("summary", "")).strip()
        if scenario_id and summary and scenario_id not in lookup:
            lookup[scenario_id] = {
                "summary": summary,
                "path": str(path),
                "incident_id": str(payload.get("incident_id", "")),
            }
    return lookup


def _load_scenario_metadata(report_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    effect_path = report_root / "phase2_effect_summary.json"
    difficulty_path = report_root / "phase2_scenario_difficulty.json"
    effect_summary = pd.DataFrame(_load_json(effect_path)) if effect_path.exists() else pd.DataFrame()
    difficulty_summary = pd.DataFrame(_load_json(difficulty_path)) if difficulty_path.exists() else pd.DataFrame()
    if not effect_summary.empty:
        effect_summary["affected_artifact_layers"] = effect_summary["affected_artifact_layers"].apply(_safe_list)
        effect_summary["primary_observed_changed_signals"] = effect_summary["primary_observed_changed_signals"].apply(_safe_list)
        effect_summary["visibility_signature"] = effect_summary.apply(lambda row: _visibility_signature(row.to_dict()), axis=1)
    if not difficulty_summary.empty and "component_scores" in difficulty_summary.columns:
        difficulty_summary["component_scores"] = difficulty_summary["component_scores"].apply(lambda value: value if isinstance(value, dict) else {})
    return effect_summary, difficulty_summary


def _prediction_rows(model_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for holdout_dir in sorted(model_root.iterdir()):
        if not holdout_dir.is_dir():
            continue
        holdout_scenario = holdout_dir.name
        for model_dir in sorted(holdout_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            prediction_path = model_dir / "predictions.parquet"
            if not prediction_path.exists():
                continue
            frame = pd.read_parquet(prediction_path)
            if frame.empty:
                continue
            attack_mask = pd.to_numeric(frame["attack_present"], errors="coerce").fillna(0).astype(int) == 1
            predicted_mask = pd.to_numeric(frame["predicted"], errors="coerce").fillna(0).astype(int) == 1
            benign_mask = ~attack_mask
            support = int(attack_mask.sum())
            benign_support = int(benign_mask.sum())
            tp = int((attack_mask & predicted_mask).sum())
            fn = int((attack_mask & ~predicted_mask).sum())
            fp = int((benign_mask & predicted_mask).sum())
            tn = int((benign_mask & ~predicted_mask).sum())
            recall = tp / support if support else 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            first_detection = None
            if "window_end_utc" in frame.columns:
                attack_detections = frame.loc[attack_mask & predicted_mask, "window_end_utc"]
                if not attack_detections.empty:
                    first_detection = pd.to_datetime(attack_detections, utc=True).min()
            rows.append(
                {
                    "holdout_scenario": holdout_scenario,
                    "scenario_id": holdout_scenario,
                    "model_name": model_dir.name,
                    "model_display_name": display_model_name(model_dir.name),
                    "prediction_path": str(prediction_path),
                    "window_count": int(len(frame)),
                    "attack_window_count": support,
                    "benign_window_count": benign_support,
                    "true_positive_count": tp,
                    "false_negative_count": fn,
                    "false_positive_count": fp,
                    "true_negative_count": tn,
                    "precision": precision,
                    "recall": recall,
                    "f1": _f1(precision, recall),
                    "false_negative_rate": fn / support if support else 0.0,
                    "false_positive_rate": fp / benign_support if benign_support else 0.0,
                    "detected_scenario": bool(tp > 0),
                    "first_attack_detection_utc": first_detection,
                }
            )
    return pd.DataFrame(rows)


def _spearman_correlation(frame: pd.DataFrame, x_column: str, y_column: str) -> float | None:
    subset = frame[[x_column, y_column]].dropna()
    if len(subset) < 2 or subset[x_column].nunique() < 2 or subset[y_column].nunique() < 2:
        return None
    value = subset[x_column].corr(subset[y_column], method="spearman")
    if pd.isna(value):
        return None
    return float(value)


def _format_correlation(value: float | None) -> str:
    return "not enough variation" if value is None else f"{value:.3f}"


def _top_signals(entry: dict[str, Any]) -> list[str]:
    return [str(item) for item in _safe_list(entry.get("primary_observed_changed_signals"))[:5]]


def build_error_analysis(project_root: Path) -> dict[str, Path]:
    """Build error analysis for the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    report_root = ensure_dir(project_root / "outputs" / "reports")
    zero_day_root = project_root / "outputs" / "phase3_zero_day" / "zero_day_models"
    zero_day_artifact_root = project_root / "outputs" / "reports" / "phase3_zero_day_artifacts"

    prediction_df = _prediction_rows(zero_day_root)
    if prediction_df.empty:
        raise FileNotFoundError(f"No Phase 3 prediction artifacts found under {zero_day_root}")

    effect_df, difficulty_df = _load_scenario_metadata(report_root)
    scenario_meta = effect_df.merge(
        difficulty_df[
            [
                "scenario_id",
                "difficulty_score",
                "difficulty_band",
                "component_scores",
                "effect_strength_scalar",
                "changed_channel_total",
                "observability_signal_count",
                "affected_layer_count",
                "explanation",
            ]
        ]
        if not difficulty_df.empty
        else pd.DataFrame(columns=["scenario_id"]),
        on="scenario_id",
        how="left",
    )

    enriched_df = prediction_df.merge(
        scenario_meta[
            [
                "scenario_id",
                "scenario_name",
                "attack_family",
                "target_component",
                "target_asset",
                "target_signal",
                "duration_seconds",
                "severity",
                "affected_artifact_layers",
                "visibility_signature",
                "effect_presence_pass",
                "primary_observed_changed_signals",
                "physical_truth_changed",
                "measured_layer_changed",
                "cyber_layer_changed",
                "additional_target_count",
                "difficulty_score",
                "difficulty_band",
                "effect_strength_scalar",
                "changed_channel_total",
                "observability_signal_count",
                "affected_layer_count",
            ]
        ]
        if not scenario_meta.empty
        else pd.DataFrame(columns=["scenario_id"]),
        on="scenario_id",
        how="left",
    )
    if "affected_artifact_layers" in enriched_df.columns:
        enriched_df["affected_artifact_layers_str"] = enriched_df["affected_artifact_layers"].apply(
            lambda value: "|".join(str(item) for item in _safe_list(value))
        )
    else:
        enriched_df["affected_artifact_layers_str"] = ""
    if "primary_observed_changed_signals" in enriched_df.columns:
        enriched_df["primary_observed_changed_signals_str"] = enriched_df["primary_observed_changed_signals"].apply(
            lambda value: "|".join(str(item) for item in _safe_list(value))
        )
    else:
        enriched_df["primary_observed_changed_signals_str"] = ""

    fp_df = enriched_df[
        [
            "model_name",
            "model_display_name",
            "holdout_scenario",
            "scenario_name",
            "attack_family",
            "target_component",
            "benign_window_count",
            "false_positive_count",
            "true_negative_count",
            "false_positive_rate",
            "prediction_path",
        ]
    ].sort_values(["false_positive_count", "model_name", "holdout_scenario"], ascending=[False, True, True]).reset_index(drop=True)

    fn_df = enriched_df[
        [
            "model_name",
            "model_display_name",
            "scenario_id",
            "scenario_name",
            "attack_family",
            "target_component",
            "target_asset",
            "duration_seconds",
            "attack_window_count",
            "true_positive_count",
            "false_negative_count",
            "false_negative_rate",
            "recall",
            "precision",
            "f1",
            "detected_scenario",
            "difficulty_score",
            "difficulty_band",
            "effect_strength_scalar",
            "visibility_signature",
            "changed_channel_total",
            "affected_layer_count",
            "observability_signal_count",
            "prediction_path",
        ]
    ].sort_values(["false_negative_rate", "scenario_id", "model_name"], ascending=[False, True, True]).reset_index(drop=True)

    scenario_patterns_df = (
        enriched_df.groupby(
            [
                "scenario_id",
                "scenario_name",
                "attack_family",
                "target_component",
                "target_asset",
                "duration_seconds",
                "visibility_signature",
                "difficulty_score",
                "difficulty_band",
                "effect_strength_scalar",
                "changed_channel_total",
                "observability_signal_count",
                "affected_layer_count",
                "effect_presence_pass",
                "affected_artifact_layers_str",
                "primary_observed_changed_signals_str",
            ],
            observed=False,
        )
        .agg(
            model_count=("model_name", "nunique"),
            detected_model_count=("detected_scenario", "sum"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_false_negative_rate=("false_negative_rate", "mean"),
            mean_false_positive_rate=("false_positive_rate", "mean"),
            total_false_negative_count=("false_negative_count", "sum"),
            total_false_positive_count=("false_positive_count", "sum"),
        )
        .reset_index()
    )
    scenario_patterns_df["missed_model_count"] = scenario_patterns_df["model_count"] - scenario_patterns_df["detected_model_count"]
    scenario_patterns_df = scenario_patterns_df.sort_values(["mean_recall", "mean_f1", "scenario_id"], ascending=[True, True, True]).reset_index(drop=True)
    scenario_patterns_df["affected_artifact_layers"] = scenario_patterns_df["affected_artifact_layers_str"].apply(
        lambda value: [item for item in str(value).split("|") if item]
    )
    scenario_patterns_df["primary_observed_changed_signals"] = scenario_patterns_df["primary_observed_changed_signals_str"].apply(
        lambda value: [item for item in str(value).split("|") if item]
    )

    model_totals_df = (
        enriched_df.groupby(["model_name", "model_display_name"], observed=False)
        .agg(
            holdout_scenario_count=("scenario_id", "nunique"),
            false_positive_count=("false_positive_count", "sum"),
            false_negative_count=("false_negative_count", "sum"),
            mean_false_positive_rate=("false_positive_rate", "mean"),
            mean_false_negative_rate=("false_negative_rate", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            detected_scenario_count=("detected_scenario", "sum"),
        )
        .reset_index()
        .sort_values(["mean_f1", "mean_recall", "false_negative_count"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    family_patterns_df = (
        enriched_df.groupby(["attack_family", "model_name"], observed=False)
        .agg(
            scenario_count=("scenario_id", "nunique"),
            mean_recall=("recall", "mean"),
            mean_false_negative_rate=("false_negative_rate", "mean"),
            mean_false_positive_rate=("false_positive_rate", "mean"),
        )
        .reset_index()
        .sort_values(["attack_family", "mean_recall", "model_name"], ascending=[True, True, True])
    )
    component_patterns_df = (
        enriched_df.groupby(["target_component", "model_name"], observed=False)
        .agg(
            scenario_count=("scenario_id", "nunique"),
            mean_recall=("recall", "mean"),
            mean_false_negative_rate=("false_negative_rate", "mean"),
        )
        .reset_index()
        .sort_values(["target_component", "mean_recall", "model_name"], ascending=[True, True, True])
    )
    visibility_patterns_df = (
        scenario_patterns_df.groupby("visibility_signature", observed=False)
        .agg(
            scenario_count=("scenario_id", "nunique"),
            mean_recall=("mean_recall", "mean"),
            mean_false_negative_rate=("mean_false_negative_rate", "mean"),
            mean_false_positive_rate=("mean_false_positive_rate", "mean"),
            mean_difficulty_score=("difficulty_score", "mean"),
        )
        .reset_index()
        .sort_values(["mean_recall", "mean_difficulty_score"], ascending=[True, False])
    )

    difficulty_correlation = _spearman_correlation(scenario_patterns_df, "difficulty_score", "mean_recall")
    effect_correlation = _spearman_correlation(scenario_patterns_df, "effect_strength_scalar", "mean_recall")

    best_model_row = model_totals_df.iloc[0].to_dict() if not model_totals_df.empty else {}
    hardest_scenarios = scenario_patterns_df.head(3).to_dict(orient="records")
    easiest_scenarios = scenario_patterns_df.sort_values(["mean_recall", "mean_f1"], ascending=[False, False]).head(3).to_dict(orient="records")

    key_findings = [
        (
            f"{display_model_name(best_model_row.get('model_name', 'unknown'))} had the strongest mean unseen-scenario F1 "
            f"at {best_model_row.get('mean_f1', 0.0):.3f}, with {int(best_model_row.get('detected_scenario_count', 0))} "
            f"of {int(best_model_row.get('holdout_scenario_count', 0))} held-out scenarios detected."
        )
        if best_model_row
        else "No Phase 3 model totals were available."
    ]
    if hardest_scenarios:
        hardest = hardest_scenarios[0]
        key_findings.append(
            f"The hardest held-out scenario in the expanded bundle was {hardest['scenario_id']} "
            f"({hardest['attack_family']} / {hardest['target_component']}), with mean recall {hardest['mean_recall']:.3f} "
            f"and difficulty band {hardest.get('difficulty_band', 'unknown')}."
        )
    if easiest_scenarios:
        easiest = easiest_scenarios[0]
        key_findings.append(
            f"The easiest held-out scenario was {easiest['scenario_id']} with mean recall {easiest['mean_recall']:.3f} "
            f"across the detector suite."
        )
    key_findings.append(
        f"Scenario difficulty versus mean recall (Spearman) = {_format_correlation(difficulty_correlation)}; "
        f"effect-strength versus mean recall (Spearman) = {_format_correlation(effect_correlation)}."
    )
    if not visibility_patterns_df.empty:
        weakest_visibility = visibility_patterns_df.iloc[0].to_dict()
        key_findings.append(
            f"The weakest visibility regime in the current bundle was {weakest_visibility['visibility_signature']}, "
            f"with mean recall {weakest_visibility['mean_recall']:.3f} across scenarios in that regime."
        )

    summary_payload = {
        "scenario_count": int(scenario_patterns_df["scenario_id"].nunique()) if not scenario_patterns_df.empty else 0,
        "model_count": int(model_totals_df["model_name"].nunique()) if not model_totals_df.empty else 0,
        "best_model_overall": best_model_row,
        "false_positive_totals_by_model": model_totals_df[
            ["model_name", "model_display_name", "false_positive_count", "mean_false_positive_rate"]
        ].to_dict(orient="records"),
        "false_negative_totals_by_model": model_totals_df[
            ["model_name", "model_display_name", "false_negative_count", "mean_false_negative_rate"]
        ].to_dict(orient="records"),
        "hardest_scenarios_by_mean_recall": hardest_scenarios,
        "easiest_scenarios_by_mean_recall": easiest_scenarios,
        "difficulty_vs_mean_recall_spearman": difficulty_correlation,
        "effect_strength_vs_mean_recall_spearman": effect_correlation,
        "visibility_patterns": visibility_patterns_df.to_dict(orient="records"),
        "family_patterns": family_patterns_df.to_dict(orient="records"),
        "component_patterns": component_patterns_df.to_dict(orient="records"),
        "key_findings": key_findings,
        "notes": [
            "These error-analysis artifacts are reporting layers built on the canonical Phase 3 unseen-scenario evaluation outputs.",
            "Difficulty is a bounded repository-specific heuristic imported from the Phase 2 analysis layer, not a universal stealth metric.",
            "Correlations are descriptive and should not be over-interpreted as causal claims.",
        ],
    }

    explanation_lookup = _load_explanation_lookup(report_root)

    success_candidates = scenario_patterns_df.sort_values(["mean_recall", "detected_model_count"], ascending=[False, False]).to_dict(orient="records")
    success_case = next((entry for entry in success_candidates if entry["scenario_id"] in explanation_lookup), success_candidates[0] if success_candidates else {})
    difficult_case = scenario_patterns_df.sort_values(["mean_recall", "missed_model_count"], ascending=[True, False]).to_dict(orient="records")
    difficult_case_entry = difficult_case[0] if difficult_case else {}

    case_studies_path = report_root / "phase3_case_studies.md"
    case_lines = [
        "# Phase 3 Case Studies",
        "",
        "These compact case studies summarize one representative successful unseen-scenario detection and one representative difficult case.",
        "They are grounded in the expanded holdout benchmark, the Phase 2 effect/difficulty artifacts, and canonical explanation outputs when available.",
        "",
    ]
    for label, entry in [("Representative successful unseen-scenario case", success_case), ("Representative difficult or missed scenario case", difficult_case_entry)]:
        if not entry:
            continue
        scenario_id = str(entry["scenario_id"])
        explanation = explanation_lookup.get(scenario_id)
        scenario_rows = enriched_df[enriched_df["scenario_id"] == scenario_id].sort_values(["recall", "f1"], ascending=[False, False])
        best_model = scenario_rows.iloc[0].to_dict() if not scenario_rows.empty else {}
        weakest_model = scenario_rows.iloc[-1].to_dict() if not scenario_rows.empty else {}
        case_lines.extend(
            [
                f"## {label}",
                "",
                f"- Scenario: `{scenario_id}` ({entry.get('attack_family', 'unknown')} / {entry.get('target_component', 'unknown')} / {entry.get('target_asset', 'unknown')})",
                f"- Mean detector recall across models: `{entry.get('mean_recall', 0.0):.3f}`",
                f"- Difficulty: `{entry.get('difficulty_band', 'unknown')}` at score `{entry.get('difficulty_score', 0.0):.2f}`",
                f"- Visibility signature: `{entry.get('visibility_signature', 'unknown')}`",
                f"- Signals with strongest observed change: `{_top_signals(entry)}`",
                (
                    f"- Best model on this holdout: `{best_model.get('model_display_name', 'unknown')}` "
                    f"with recall `{best_model.get('recall', 0.0):.3f}` and F1 `{best_model.get('f1', 0.0):.3f}`"
                    if best_model
                    else "- Best-model detail unavailable."
                ),
                (
                    f"- Weakest model on this holdout: `{weakest_model.get('model_display_name', 'unknown')}` "
                    f"with recall `{weakest_model.get('recall', 0.0):.3f}` and F1 `{weakest_model.get('f1', 0.0):.3f}`"
                    if weakest_model
                    else "- Weakest-model detail unavailable."
                ),
                (
                    f"- Explanation layer summary: {explanation['summary']}"
                    if explanation
                    else "- Explanation layer summary: no canonical explanation artifact was linked to this scenario in the current repository outputs."
                ),
                "",
            ]
        )
    case_studies_path.write_text("\n".join(case_lines) + "\n", encoding="utf-8")

    summary_json = report_root / "phase3_error_analysis_summary.json"
    summary_md = report_root / "phase3_error_analysis_summary.md"
    fp_csv = report_root / "phase3_false_positive_analysis.csv"
    fn_csv = report_root / "phase3_false_negative_analysis.csv"
    patterns_csv = report_root / "phase3_per_scenario_error_patterns.csv"

    _write_json(summary_json, summary_payload)
    fp_df.to_csv(fp_csv, index=False)
    fn_df.to_csv(fn_csv, index=False)
    scenario_patterns_df.to_csv(patterns_csv, index=False)

    lines = [
        "# Phase 3 Error Analysis Summary",
        "",
        f"- Scenario count: `{summary_payload['scenario_count']}`",
        f"- Model count: `{summary_payload['model_count']}`",
        f"- Best aggregate unseen-scenario model: `{display_model_name(best_model_row.get('model_name', 'unknown'))}`",
        f"- Difficulty vs mean recall (Spearman): `{_format_correlation(difficulty_correlation)}`",
        f"- Effect strength vs mean recall (Spearman): `{_format_correlation(effect_correlation)}`",
        "",
        "## Key Findings",
        "",
    ]
    lines.extend(f"- {item}" for item in key_findings)
    lines.extend(
        [
            "",
            "## Hardest Scenarios",
            "",
        ]
    )
    for row in hardest_scenarios:
        lines.append(
            f"- `{row['scenario_id']}`: mean_recall `{row['mean_recall']:.3f}`, "
            f"difficulty `{row.get('difficulty_band', 'unknown')}`, "
            f"visibility `{row.get('visibility_signature', 'unknown')}`"
        )
    lines.extend(
        [
            "",
            "## Easiest Scenarios",
            "",
        ]
    )
    for row in easiest_scenarios:
        lines.append(
            f"- `{row['scenario_id']}`: mean_recall `{row['mean_recall']:.3f}`, "
            f"difficulty `{row.get('difficulty_band', 'unknown')}`, "
            f"visibility `{row.get('visibility_signature', 'unknown')}`"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary_payload["notes"])
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "summary_json": summary_json,
        "summary_md": summary_md,
        "false_positive_csv": fp_csv,
        "false_negative_csv": fn_csv,
        "scenario_patterns_csv": patterns_csv,
        "case_studies_md": case_studies_path,
        "zero_day_artifact_root": zero_day_artifact_root,
    }


def main() -> None:
    """Run the command-line entrypoint for the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(
        description=(
            "Build publication-grade Phase 3 error-analysis artifacts from the canonical unseen-scenario evaluation outputs. "
            "These are analysis reports layered on top of existing prediction artifacts and do not alter the canonical executable path."
        )
    )
    parser.add_argument("--project-root", default=str(ROOT))
    args = parser.parse_args()

    outputs = build_error_analysis(Path(args.project_root))
    print("Generated Phase 3 error-analysis artifacts:")
    for key, value in sorted(outputs.items()):
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
