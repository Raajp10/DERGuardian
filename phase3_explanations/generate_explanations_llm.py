"""Phase 3 grounded explanation support for DERGuardian.

This module implements generate explanations llm logic for post-alert explanation packets,
family attribution, evidence grounding, or validation. It supports operator-facing
explanation evidence and does not claim human-like root-cause analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3_explanations.shared import (
    PACKAGE_ROOT,
    build_safe_operator_actions,
    confidence_band,
    read_json,
    write_json,
    write_text,
)
from phase3_explanations.validate_explanations import validate_explanation


def extract_json_from_text(text: str) -> Any:
    """Handle extract json from text within the Phase 3 grounded explanation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    code_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    for block in code_blocks:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    raise ValueError("Unable to extract JSON from the provided LLM response.")


def render_prompt_user(packet: dict[str, Any], schema: dict[str, Any], template_text: str) -> str:
    """Handle render prompt user within the Phase 3 grounded explanation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return (
        template_text.replace("__PACKET_JSON__", json.dumps(packet, indent=2))
        .replace("__SCHEMA_JSON__", json.dumps(schema, indent=2))
    )


def build_confidence_note(packet: dict[str, Any], family: dict[str, Any]) -> str:
    """Build confidence note for the Phase 3 grounded explanation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    score = float(family.get("score", 0.0))
    has_cyber = bool(packet.get("cyber_evidence"))
    has_physical = bool(packet.get("physical_evidence"))
    if score < 0.45:
        return "Evidence is weak relative to the closed taxonomy, so the family assignment should be treated as tentative."
    if has_cyber and has_physical:
        return "Confidence is higher because both cyber and physical evidence point in the same direction."
    return "Confidence is moderate because the explanation is grounded, but one modality provides limited corroboration."


def build_limitations(packet: dict[str, Any], family: dict[str, Any]) -> list[str]:
    """Build limitations for the Phase 3 grounded explanation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    limitations = []
    if not packet.get("cyber_evidence"):
        limitations.append("Cyber evidence is limited in the alert window and its configured lookback period.")
    if not packet.get("physical_evidence"):
        limitations.append("Physical evidence is limited to aggregated window features rather than raw sub-second traces.")
    if float(family.get("score", 0.0)) < 0.55:
        limitations.append("The suspected family remains low confidence because the evidence pattern does not cleanly separate among multiple taxonomy labels.")
    limitations.append("Scenario metadata, when present, is for offline evaluation only and should not be treated as live operational truth.")
    return limitations


def _alert_status_language(packet: dict[str, Any]) -> tuple[str, str]:
    score = float(packet.get("anomaly_score", 0.0))
    threshold = float(packet.get("threshold", 0.0))
    margin = float(packet.get("score_margin", score - threshold))
    predicted_alert = bool(packet.get("predicted_alert", False))
    if predicted_alert and margin >= 0.0:
        summary = (
            f"{packet['model_name']} raised alert {packet['incident_id']} because the anomaly score "
            f"{score:.4f} exceeded the threshold {threshold:.4f}."
        )
        why = f"The detector score exceeded threshold by {margin:.4f}."
    else:
        summary = (
            f"{packet['model_name']} reviewed window {packet['incident_id']} and did not raise an alert because the anomaly score "
            f"{score:.4f} remained below the threshold {threshold:.4f}."
        )
        why = f"The detector score remained {abs(margin):.4f} below threshold."
    return summary, why


def grounded_draft_explanation(packet: dict[str, Any]) -> dict[str, Any]:
    """Handle grounded draft explanation within the Phase 3 grounded explanation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    candidates = packet.get("candidate_families", [])
    family = candidates[0] if candidates else {"label": "unknown_anomaly", "score": 0.3, "confidence": "low", "reasoning": []}
    top_physical = packet.get("physical_evidence", [])[:3]
    top_cyber = packet.get("cyber_evidence", [])[:3]
    actions = build_safe_operator_actions(packet)

    assets = packet.get("affected_assets", [])
    asset_text = ", ".join(assets) if assets else "the monitored feeder assets"
    top_feature_names = ", ".join(item.get("description", item.get("feature", "")) for item in packet.get("top_features", [])[:3])
    alert_summary, threshold_line = _alert_status_language(packet)
    summary = (
        f"{alert_summary} The strongest structured evidence points to {asset_text}, and the current best closed-set classification is "
        f"`{family['label']}`."
    )

    why_flagged = [
        threshold_line,
        f"Top discriminative window features were {top_feature_names}.",
    ]
    if top_physical:
        first = top_physical[0]
        why_flagged.append(
            f"Physical evidence showed {first['signal']} moving {first['direction']} versus the clean matched window."
        )
    if top_cyber:
        first_cyber = top_cyber[0]
        why_flagged.append(
            f"Cyber evidence included {first_cyber['kind']} activity on {first_cyber.get('resource') or first_cyber.get('detail')}."
        )

    return {
        "incident_id": packet["incident_id"],
        "summary": summary,
        "suspected_attack_family": family["label"],
        "family_confidence": {
            "level": family.get("confidence", confidence_band(float(family.get("score", 0.0)))),
            "score": round(float(family.get("score", 0.0)), 3),
        },
        "why_flagged": why_flagged,
        "physical_evidence_used": top_physical,
        "cyber_evidence_used": top_cyber,
        "operator_actions": actions,
        "confidence_note": build_confidence_note(packet, family),
        "limitations": build_limitations(packet, family),
    }


def render_incident_summary_markdown(packet: dict[str, Any], explanation: dict[str, Any]) -> str:
    """Handle render incident summary markdown within the Phase 3 grounded explanation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    heading = "Incident Summary" if bool(packet.get("predicted_alert", False)) else "Window Review"
    reason_heading = "Why Flagged" if bool(packet.get("predicted_alert", False)) else "Why Reviewed"
    lines = [
        f"# {heading}: {packet['incident_id']}",
        "",
        explanation["summary"],
        "",
        "## Suspected Family",
        "",
        f"- Label: `{explanation['suspected_attack_family']}`",
        f"- Confidence: `{explanation['family_confidence']['level']}` ({explanation['family_confidence']['score']})",
        "",
        f"## {reason_heading}",
        "",
    ]
    for item in explanation["why_flagged"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Physical Evidence Used", ""])
    if explanation["physical_evidence_used"]:
        for item in explanation["physical_evidence_used"]:
            lines.append(
                f"- `{item['signal']}` ({item['aggregation']}): observed `{item['observed_value']}`, baseline `{item['baseline_value']}`, delta `{item['delta']}`"
            )
    else:
        lines.append("- No strong physical evidence was available in the packet.")
    lines.extend(["", "## Cyber Evidence Used", ""])
    if explanation["cyber_evidence_used"]:
        for item in explanation["cyber_evidence_used"]:
            lines.append(
                f"- `{item['kind']}` via `{item.get('protocol', 'unknown')}` on `{item.get('resource') or item.get('detail')}` at `{item['timestamp_utc']}`"
            )
    else:
        lines.append("- No strong cyber evidence was available in the packet.")
    lines.extend(["", "## Defensive Next Steps", ""])
    for action in explanation["operator_actions"]:
        lines.append(f"- {action}")
    lines.extend(["", "## Confidence Note", "", f"- {explanation['confidence_note']}", "", "## Limitations", ""])
    for item in explanation["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Run the command-line entrypoint for the Phase 3 grounded explanation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Export prompt packages, ingest LLM output, or generate grounded draft explanations.")
    parser.add_argument("--packet", required=True)
    parser.add_argument("--schema", default=str(PACKAGE_ROOT / "explanation_schema.json"))
    parser.add_argument("--system-prompt", default=str(PACKAGE_ROOT / "explanation_prompt_system.txt"))
    parser.add_argument("--user-template", default=str(PACKAGE_ROOT / "explanation_prompt_user_template.txt"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "reports" / "explanation_artifacts"))
    parser.add_argument("--export-prompt-package", action="store_true")
    parser.add_argument("--ingest-response", help="Path to an external LLM response containing JSON.")
    parser.add_argument("--simulate-grounded-output", action="store_true", help="Generate a deterministic grounded draft explanation for self-checks.")
    args = parser.parse_args()

    packet = read_json(args.packet)
    schema = read_json(args.schema)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.export_prompt_package:
        system_text = Path(args.system_prompt).read_text(encoding="utf-8")
        template_text = Path(args.user_template).read_text(encoding="utf-8")
        rendered_user = render_prompt_user(packet, schema, template_text)
        write_text(system_text, output_dir / "prompt_system.txt")
        write_text(template_text, output_dir / "prompt_user_template.txt")
        write_text(rendered_user, output_dir / "prompt_user_rendered.txt")
        write_json(packet, output_dir / "explanation_packet.json")
        write_json(schema, output_dir / "explanation_schema.json")
        print(f"Prompt package written to {output_dir}")

    explanation: dict[str, Any] | None = None
    if args.simulate_grounded_output:
        explanation = grounded_draft_explanation(packet)
    elif args.ingest_response:
        raw_text = Path(args.ingest_response).read_text(encoding="utf-8")
        explanation = extract_json_from_text(raw_text)

    if explanation is not None:
        validation = validate_explanation(packet, explanation, schema)
        if not validation["valid"]:
            raise ValueError(f"Explanation failed validation: {validation['errors']}")
        explanation_path = write_json(explanation, output_dir / "explanation_output.json")
        markdown_path = write_text(render_incident_summary_markdown(packet, explanation), output_dir / "incident_summary.md")
        write_json(validation, output_dir / "validation_report.json")
        print(f"Explanation JSON written to {explanation_path}")
        print(f"Incident summary written to {markdown_path}")


if __name__ == "__main__":
    main()
