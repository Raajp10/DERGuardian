"""Phase 1 context and fusion helper for DERGuardian.

This module builds or consumes structured context artifacts around normal-system
behavior, feature evidence, and detector outputs. It supports explanation and
fusion workflows but is not the canonical detector-selection mechanism.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from common.io_utils import read_json, write_dataframe, write_json, write_jsonl
from phase1_models.context.phase1_context_builder import canonical_context_summary_path
from phase1_models.model_utils import CANONICAL_ARTIFACT_ROOT


PROMPT_TEMPLATE = """You are reviewing a bounded DER anomaly context summary.

Use only the provided structured fields:
- top_deviating_signals
- mapped_assets and mapped_components
- environmental_context
- feeder_context
- voltage_violations
- der_dispatch_consistency
- soc_consistency

Return:
1. a likely anomaly family from the repository family set
2. a concise operator-readable explanation
3. a confidence score between 0 and 1
4. likely affected assets

This prompt is for defensive anomaly triage only. Do not invent commands, payloads, or remediation actions outside safe operator checks.
"""

SUPPORTED_FAMILIES = (
    "false_data_injection",
    "replay",
    "unauthorized_command",
    "command_delay",
    "command_suppression",
    "degradation",
    "telemetry_corruption",
    "coordinated_campaign",
    "unknown_anomaly",
)


def canonical_reasoning_jsonl_path(root: Path, artifact_root_name: str = CANONICAL_ARTIFACT_ROOT) -> Path:
    """Handle canonical reasoning jsonl path within the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return root / "outputs" / "reports" / artifact_root_name / "phase1_context_reasoning_outputs.jsonl"


def canonical_reasoning_table_path(root: Path, artifact_root_name: str = CANONICAL_ARTIFACT_ROOT) -> Path:
    """Handle canonical reasoning table path within the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return root / "outputs" / "reports" / artifact_root_name / "phase1_context_reasoning_outputs.parquet"


def canonical_reasoner_model_dir(root: Path, model_root_name: str = "models_full_run") -> Path:
    """Handle canonical reasoner model dir within the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return root / "outputs" / model_root_name / "context_reasoner"


def load_context_summaries(path: Path) -> list[dict[str, Any]]:
    """Load context summaries for the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def run_reasoner_on_contexts(context_summaries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Run reasoner on contexts for the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    records: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    for summary in context_summaries:
        reasoning = reason_from_context(summary)
        payload = {
            "window_start_utc": summary["window_start_utc"],
            "window_end_utc": summary["window_end_utc"],
            "split_name": summary.get("split_name", ""),
            "scenario_id": summary["scenario_id"],
            "likely_anomaly_family": reasoning["likely_anomaly_family"],
            "human_readable_explanation": reasoning["human_readable_explanation"],
            "confidence_score": reasoning["confidence_score"],
            "likely_affected_assets": reasoning["likely_affected_assets"],
            "anomaly_reasoning_score": reasoning["anomaly_reasoning_score"],
            "prompt_template_version": "phase1_context_v1",
            "reasoning_mode": "local_prompt_style_baseline",
            "reasoning_evidence": reasoning["reasoning_evidence"],
        }
        records.append(payload)
        table_rows.append(
            {
                "window_start_utc": payload["window_start_utc"],
                "window_end_utc": payload["window_end_utc"],
                "split_name": payload["split_name"],
                "scenario_id": payload["scenario_id"],
                "likely_anomaly_family": payload["likely_anomaly_family"],
                "confidence_score": float(payload["confidence_score"]),
                "anomaly_reasoning_score": float(payload["anomaly_reasoning_score"]),
                "likely_affected_assets_json": json.dumps(payload["likely_affected_assets"]),
                "human_readable_explanation": payload["human_readable_explanation"],
            }
        )
    return records, pd.DataFrame(table_rows)


def persist_reasoning_outputs(
    root: Path,
    records: list[dict[str, Any]],
    table: pd.DataFrame,
    artifact_root_name: str = CANONICAL_ARTIFACT_ROOT,
    model_root_name: str = "models_full_run",
) -> tuple[Path, Path, Path]:
    """Handle persist reasoning outputs within the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    jsonl_path = canonical_reasoning_jsonl_path(root, artifact_root_name=artifact_root_name)
    table_path = canonical_reasoning_table_path(root, artifact_root_name=artifact_root_name)
    model_dir = canonical_reasoner_model_dir(root, model_root_name=model_root_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = model_dir / "prompt_template.txt"
    metadata_path = model_dir / "reasoner_metadata.json"
    prompt_path.write_text(PROMPT_TEMPLATE, encoding="utf-8")
    write_json(
        {
            "reasoning_mode": "local_prompt_style_baseline",
            "supported_families": list(SUPPORTED_FAMILIES),
            "notes": "This is an optional context-aware reasoning layer. It is not the primary detector and uses deterministic heuristics instead of an external provider-backed LLM.",
        },
        metadata_path,
    )
    write_jsonl(records, jsonl_path)
    write_dataframe(table, table_path, fmt="parquet")
    return jsonl_path, table_path, metadata_path


def build_and_persist_reasoning_outputs(
    root: Path,
    context_summaries: list[dict[str, Any]] | None = None,
    artifact_root_name: str = CANONICAL_ARTIFACT_ROOT,
    model_root_name: str = "models_full_run",
) -> tuple[list[dict[str, Any]], pd.DataFrame, tuple[Path, Path, Path]]:
    """Build and persist reasoning outputs for the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if context_summaries is None:
        context_summaries = load_context_summaries(canonical_context_summary_path(root, artifact_root_name=artifact_root_name))
    records, table = run_reasoner_on_contexts(context_summaries)
    paths = persist_reasoning_outputs(
        root=root,
        records=records,
        table=table,
        artifact_root_name=artifact_root_name,
        model_root_name=model_root_name,
    )
    return records, table, paths


def reason_from_context(summary: dict[str, Any]) -> dict[str, Any]:
    """Handle reason from context within the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    top_signals = [str(item.get("signal", "")) for item in summary.get("top_deviating_signals", [])]
    top_features = summary.get("top_deviating_signals", [])
    assets = list(summary.get("mapped_assets", []))
    components = list(summary.get("mapped_components", []))
    stats = summary.get("summary_statistics", {})
    voltage = summary.get("voltage_violations", {})
    der_dispatch = summary.get("der_dispatch_consistency", {})
    soc_context = summary.get("soc_consistency", {})

    max_rel = float(stats.get("max_abs_relative_change", 0.0) or 0.0)
    mean_rel = float(stats.get("mean_abs_relative_change", 0.0) or 0.0)
    changed_channels = int(stats.get("changed_channel_count", 0) or 0)
    voltage_flag = int(voltage.get("violation_flag", 0) or 0)
    pv_issues = int(der_dispatch.get("pv_dispatch_issue_count", 0) or 0)
    bess_issues = int(der_dispatch.get("bess_dispatch_issue_count", 0) or 0)
    soc_issues = int(soc_context.get("soc_issue_count", 0) or 0)

    has_cyber_indicator = any(signal.startswith("cyber_") for signal in top_signals)
    has_control_state_signal = any(
        ("tap_pos" in signal) or signal.endswith("_status") or signal.endswith("_state") or ("curtailment_frac" in signal)
        for signal in top_signals
    )
    has_voltage_signal = any(("terminal_v_pu" in signal) or ("_v_pu_" in signal) for signal in top_signals)
    has_soc_signal = any("soc" in signal for signal in top_signals)
    has_dispatch_signal = any(signal.endswith("_p_kw") or ("feeder_p_kw_total" in signal) for signal in top_signals)

    score = 0.03 + min(0.12, max_rel * 0.12) + min(0.08, changed_channels * 0.008)
    family = "unknown_anomaly"
    reasons: list[str] = []

    if has_cyber_indicator:
        score += 0.08
        reasons.append("Cyber-layer aggregate counts are elevated inside the window.")

    if len(set(assets)) >= 2 and (pv_issues or bess_issues or soc_issues or voltage_flag or has_control_state_signal):
        family = "coordinated_campaign"
        score += 0.16
        reasons.append("More than one asset or component family appears among the dominant deviations.")
    if any("curtailment_frac" in signal or "available_kw" in signal for signal in top_signals):
        if pv_issues or voltage_flag:
            family = "unauthorized_command"
            score += 0.12
            reasons.append("PV availability and curtailment features changed together with feeder-visible impact.")
        elif has_cyber_indicator:
            family = "telemetry_corruption"
            score += 0.08
            reasons.append("PV availability and curtailment features shifted without equally strong feeder-wide corroboration.")
    if has_soc_signal:
        if soc_issues and not voltage_flag and bess_issues == 0:
            family = "false_data_injection"
            score += 0.15
            reasons.append("State-of-charge evidence moved materially while broader power-flow changes stayed limited.")
        elif bess_issues or has_control_state_signal:
            family = "unauthorized_command"
            score += 0.1
            reasons.append("Storage SOC and dispatch moved together, consistent with a control-side disturbance.")
    if has_control_state_signal:
        if max_rel >= 0.15:
            family = "unauthorized_command"
            score += 0.14
            reasons.append("Discrete control-state channels changed abruptly inside the window.")
        else:
            family = "command_delay"
            score += 0.08
            reasons.append("Discrete control-state movement is visible but comparatively muted or delayed.")
    if has_voltage_signal and has_cyber_indicator and voltage_flag == 0 and max_rel < 0.1:
        family = "replay"
        score += 0.05
        reasons.append("Voltage telemetry deviates in a comparatively isolated way without strong voltage-limit violations.")
    if has_dispatch_signal and max_rel < 0.08 and changed_channels <= 2:
        family = "degradation"
        score += 0.03
        reasons.append("A small number of feeder-wide channels shifted without a strong multi-asset signature.")
    if not reasons:
        reasons.append("The context shows residual deviations, but the family evidence remains weak and should be treated cautiously.")

    score = max(0.05, min(score, 0.97))
    anomaly_score = max(0.02, min(score, 0.99))
    if (
        not voltage_flag
        and not pv_issues
        and not bess_issues
        and not soc_issues
        and not has_control_state_signal
        and not has_cyber_indicator
    ):
        family = "unknown_anomaly"
        anomaly_score = min(0.22, 0.08 + max_rel * 0.08)
        reasons = ["Residual changes are present, but the window lacks explicit voltage, dispatch, SOC, or cyber indicators strong enough for anomaly-family commitment."]
    elif max_rel < 0.025 and changed_channels <= 1 and not voltage_flag and not pv_issues and not bess_issues and not soc_issues:
        family = "unknown_anomaly"
        anomaly_score = min(anomaly_score, 0.18)
        reasons = ["The context is close to nominal and does not support a strong anomaly-family commitment."]

    explanation = build_human_explanation(summary, family, reasons, anomaly_score)
    likely_assets = assets or infer_assets_from_features(top_features)
    return {
        "likely_anomaly_family": family if family in SUPPORTED_FAMILIES else "unknown_anomaly",
        "human_readable_explanation": explanation,
        "confidence_score": round(anomaly_score if family == "unknown_anomaly" else score, 6),
        "anomaly_reasoning_score": round(anomaly_score, 6),
        "likely_affected_assets": likely_assets,
        "reasoning_evidence": reasons[:4],
    }


def build_human_explanation(summary: dict[str, Any], family: str, reasons: list[str], anomaly_score: float) -> str:
    """Build human explanation for the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    assets = summary.get("mapped_assets", [])
    feeder_band = summary.get("feeder_context", {}).get("feeder_operating_band", "unknown load")
    top_text = ", ".join(item.get("description", item.get("signal", "")) for item in summary.get("top_deviating_signals", [])[:3])
    asset_text = ", ".join(assets[:3]) if assets else "the feeder"
    confidence_band = "low" if anomaly_score < 0.35 else "moderate" if anomaly_score < 0.7 else "high"
    first_reason = reasons[0] if reasons else "Residual evidence is present."
    return (
        f"This window is most consistent with `{family}` around {asset_text}. "
        f"Top changed signals were {top_text or 'limited residual features'}, observed during {feeder_band}. "
        f"{first_reason} Overall context confidence is {confidence_band}."
    )


def infer_assets_from_features(top_features: list[dict[str, Any]]) -> list[str]:
    """Handle infer assets from features within the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    assets = []
    for item in top_features:
        asset = str(item.get("asset_id", ""))
        if asset and asset not in {"feeder", "environment", "unknown", "derived"} and asset not in assets:
            assets.append(asset)
    return assets


def main() -> None:
    """Run the command-line entrypoint for the Phase 1 context and fusion support workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Run the optional local prompt-style Phase 1 reasoning layer.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--artifact-root", default=CANONICAL_ARTIFACT_ROOT)
    parser.add_argument("--context-jsonl", default="")
    args = parser.parse_args()

    root = Path(args.project_root)
    context_path = Path(args.context_jsonl) if args.context_jsonl else canonical_context_summary_path(root, artifact_root_name=args.artifact_root)
    contexts = load_context_summaries(context_path)
    _, table, paths = build_and_persist_reasoning_outputs(
        root=root,
        context_summaries=contexts,
        artifact_root_name=args.artifact_root,
    )
    print(f"Wrote {len(table)} reasoning outputs to {paths[0]}")
    print(f"Wrote flattened reasoning table to {paths[1]}")
    print(f"Wrote reasoner metadata to {paths[2]}")


if __name__ == "__main__":
    main()
