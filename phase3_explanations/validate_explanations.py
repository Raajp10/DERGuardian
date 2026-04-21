"""Phase 3 grounded explanation support for DERGuardian.

This module implements validate explanations logic for post-alert explanation packets,
family attribution, evidence grounding, or validation. It supports operator-facing
explanation evidence and does not claim human-like root-cause analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jsonschema

from phase3_explanations.shared import (
    ATTACK_FAMILY_TAXONOMY,
    PACKAGE_ROOT,
    allowed_operator_action,
    asset_mentions,
    extract_allowed_assets_and_signals,
    read_json,
    write_json,
)


ABSOLUTE_CERTAINTY_TERMS = (
    "definitely",
    "certainly",
    "confirmed root cause",
    "proven",
    "no doubt",
    "guaranteed",
)


def validate_explanation(packet: dict[str, Any], explanation: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Validate explanation for the Phase 3 grounded explanation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    errors: list[str] = []
    warnings: list[str] = []

    try:
        jsonschema.validate(explanation, schema)
    except Exception as exc:
        errors.append(f"JSON schema validation failed: {exc}")

    family = explanation.get("suspected_attack_family")
    if family not in ATTACK_FAMILY_TAXONOMY:
        errors.append(f"Unsupported family label: {family}")

    allowed_assets, allowed_signals = extract_allowed_assets_and_signals(packet)

    mentioned_assets = []
    for field in ("summary", "why_flagged", "confidence_note", "limitations"):
        value = explanation.get(field)
        if isinstance(value, list):
            for item in value:
                mentioned_assets.extend(asset_mentions(str(item)))
        else:
            mentioned_assets.extend(asset_mentions(str(value or "")))
    unknown_assets = sorted({asset for asset in mentioned_assets if asset not in allowed_assets})
    if unknown_assets:
        errors.append(f"Explanation mentions assets not grounded in the packet: {unknown_assets}")

    grounding_checks = {
        "physical_evidence_used": explanation.get("physical_evidence_used", []),
        "cyber_evidence_used": explanation.get("cyber_evidence_used", []),
    }
    grounded_references = 0
    total_references = 0
    for section, items in grounding_checks.items():
        for item in items:
            total_references += 1
            if not isinstance(item, dict):
                errors.append(f"{section} must contain objects only.")
                continue
            signal = item.get("signal") or item.get("resource") or item.get("source_feature")
            asset = item.get("asset")
            signal_ok = signal is None or str(signal) in allowed_signals or any(str(signal).startswith(prefix) for prefix in allowed_signals)
            asset_ok = asset is None or str(asset).lower() in allowed_assets
            if signal_ok and asset_ok:
                grounded_references += 1
            else:
                errors.append(f"Ungrounded reference in {section}: {item}")

    for action in explanation.get("operator_actions", []):
        if not allowed_operator_action(str(action)):
            errors.append(f"Unsafe or non-defensive operator action: {action}")

    for field in ("summary", "confidence_note"):
        text = str(explanation.get(field, "")).lower()
        if any(term in text for term in ABSOLUTE_CERTAINTY_TERMS):
            errors.append(f"Unsupported certainty language found in {field}.")

    limitations = explanation.get("limitations", [])
    if not limitations:
        warnings.append("No limitations were provided; weak-evidence cases should describe uncertainty.")

    confidence = explanation.get("family_confidence", {})
    score = confidence.get("score") if isinstance(confidence, dict) else None
    if isinstance(score, (int, float)) and float(score) > 0.9 and family == "unknown_anomaly":
        warnings.append("Unknown label paired with very high confidence is internally inconsistent.")

    predicted_alert = bool(packet.get("predicted_alert", False))
    score_margin = float(packet.get("score_margin", 0.0))
    summary_text = str(explanation.get("summary", "")).lower()
    why_text = " ".join(str(item) for item in explanation.get("why_flagged", [])).lower()
    combined_text = summary_text + " " + why_text
    if predicted_alert:
        if score_margin < 0.0:
            errors.append("Packet indicates predicted_alert=true but score_margin is negative.")
        if "did not raise an alert" in combined_text or "remained below the threshold" in combined_text:
            errors.append("Alert explanation incorrectly claims the score stayed below threshold.")
    else:
        if "raised incident" in combined_text or "raised alert" in combined_text or "exceeded the threshold" in combined_text:
            errors.append("Non-alert explanation incorrectly claims an alert was raised or threshold was exceeded.")
        if score_margin >= 0.0:
            warnings.append("Packet indicates predicted_alert=false but score_margin is non-negative; check upstream scoring semantics.")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "grounding_rate": round(grounded_references / max(total_references, 1), 4),
        "grounded_references": grounded_references,
        "total_references": total_references,
    }


def main() -> None:
    """Run the command-line entrypoint for the Phase 3 grounded explanation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Validate a structured explanation against the explanation contract.")
    parser.add_argument("--packet", required=True)
    parser.add_argument("--explanation", required=True)
    parser.add_argument("--schema", default=str(PACKAGE_ROOT / "explanation_schema.json"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "reports" / "explanation_artifacts" / "validation_report.json"))
    args = parser.parse_args()

    packet = read_json(args.packet)
    explanation = read_json(args.explanation)
    schema = read_json(args.schema)
    report = validate_explanation(packet, explanation, schema)
    output_path = write_json(report, args.output)
    print(json.dumps(report, indent=2))
    print(f"Validation report written to {output_path}")


if __name__ == "__main__":
    main()
