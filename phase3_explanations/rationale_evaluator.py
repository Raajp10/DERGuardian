from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from phase3_explanations.shared import OUTPUT_ROOT, asset_mentions, read_json, write_text
from phase3_explanations.validate_explanations import validate_explanation


REQUIRED_FIELDS = [
    "incident_id",
    "summary",
    "suspected_attack_family",
    "family_confidence",
    "why_flagged",
    "physical_evidence_used",
    "cyber_evidence_used",
    "operator_actions",
    "confidence_note",
    "limitations",
]


def explanation_assets(explanation: dict[str, Any]) -> set[str]:
    assets = set()
    for section in ("physical_evidence_used", "cyber_evidence_used"):
        for item in explanation.get(section, []):
            asset = item.get("asset")
            if asset:
                assets.add(str(asset).lower())
    assets.update(asset_mentions(explanation.get("summary", "")))
    return assets


def completeness_score(explanation: dict[str, Any]) -> float:
    filled = 0
    for field in REQUIRED_FIELDS:
        value = explanation.get(field)
        if value not in (None, "", [], {}):
            filled += 1
    return round(filled / len(REQUIRED_FIELDS), 4)


def manual_rubric_template() -> dict[str, str]:
    return {
        "manual_review_summary_clarity": "",
        "manual_review_evidence_alignment": "",
        "manual_review_operator_usefulness": "",
    }


def to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def evaluate_directory(packet_dir: Path, explanation_dir: Path, schema_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    schema = read_json(schema_path)
    rows: list[dict[str, Any]] = []
    explanation_files = sorted(explanation_dir.glob("*.json"))
    for explanation_file in explanation_files:
        explanation = read_json(explanation_file)
        incident_id = explanation.get("incident_id")
        packet_file = packet_dir / f"{incident_id}.json"
        if not packet_file.exists():
            continue
        packet = read_json(packet_file)
        validation = validate_explanation(packet, explanation, schema)
        truth = packet.get("offline_evaluation", {}).get("scenario_label") or {}
        truth_family = truth.get("attack_family")
        truth_assets = {str(item).lower() for item in truth.get("affected_assets", [])} if isinstance(truth.get("affected_assets"), list) else set()
        predicted_assets = explanation_assets(explanation)
        overlap = len(truth_assets & predicted_assets)
        asset_accuracy = round(overlap / max(len(truth_assets), 1), 4) if truth_assets else None
        row = {
            "incident_id": incident_id,
            "ground_truth_family": truth_family or "",
            "predicted_family": explanation.get("suspected_attack_family"),
            "family_match": bool(truth_family and truth_family == explanation.get("suspected_attack_family")),
            "asset_attribution_accuracy": asset_accuracy if asset_accuracy is not None else "",
            "evidence_grounding_rate": validation["grounding_rate"],
            "explanation_completeness": completeness_score(explanation),
            "used_unknown": explanation.get("suspected_attack_family") == "unknown_anomaly",
            **manual_rubric_template(),
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        summary = {
            "family_classification_accuracy": 0.0,
            "asset_attribution_accuracy": 0.0,
            "evidence_grounding_rate": 0.0,
            "unknown_usage_rate": 0.0,
            "explanation_completeness": 0.0,
            "incident_count": 0,
        }
        return frame, summary

    family_known = frame[frame["ground_truth_family"] != ""]
    summary = {
        "family_classification_accuracy": round(float(family_known["family_match"].mean()), 4) if not family_known.empty else 0.0,
        "asset_attribution_accuracy": round(
            float(pd.to_numeric(frame["asset_attribution_accuracy"], errors="coerce").dropna().mean()),
            4,
        )
        if not frame.empty
        else 0.0,
        "evidence_grounding_rate": round(float(frame["evidence_grounding_rate"].mean()), 4),
        "unknown_usage_rate": round(float(frame["used_unknown"].mean()), 4),
        "explanation_completeness": round(float(frame["explanation_completeness"].mean()), 4),
        "incident_count": int(len(frame)),
    }
    return frame, summary


def build_report(frame: pd.DataFrame, summary: dict[str, Any]) -> str:
    lines = [
        "# Explanation Evaluation Report",
        "",
        f"- Incidents evaluated: `{summary['incident_count']}`",
        f"- Family classification accuracy: `{summary['family_classification_accuracy']}`",
        f"- Asset attribution accuracy: `{summary['asset_attribution_accuracy']}`",
        f"- Evidence grounding rate: `{summary['evidence_grounding_rate']}`",
        f"- Unknown usage rate: `{summary['unknown_usage_rate']}`",
        f"- Explanation completeness: `{summary['explanation_completeness']}`",
        "",
        "## Per-Incident Table",
        "",
    ]
    if frame.empty:
        lines.append("No packet/explanation pairs were available for evaluation.")
    else:
        lines.append(to_markdown(frame))
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluator for explanation quality against known scenario labels.")
    parser.add_argument("--packet-dir", default=str(OUTPUT_ROOT / "packets"))
    parser.add_argument("--explanation-dir", default=str(OUTPUT_ROOT / "explanations"))
    parser.add_argument("--schema", default=str(ROOT / "phase3_explanations" / "explanation_schema.json"))
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, summary = evaluate_directory(Path(args.packet_dir), Path(args.explanation_dir), Path(args.schema))
    csv_path = output_dir / "explanation_eval_table.csv"
    frame.to_csv(csv_path, index=False)
    report_text = build_report(frame, summary)
    report_path = write_text(report_text, output_dir / "explanation_eval_report.md")
    (output_dir / "explanation_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Explanation evaluation table written to {csv_path}")
    print(f"Explanation evaluation report written to {report_path}")


if __name__ == "__main__":
    main()
