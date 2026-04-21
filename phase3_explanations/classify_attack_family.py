"""Phase 3 grounded explanation support for DERGuardian.

This module implements classify attack family logic for post-alert explanation packets,
family attribution, evidence grounding, or validation. It supports operator-facing
explanation evidence and does not claim human-like root-cause analysis.
"""

from __future__ import annotations

from typing import Any
import argparse

from phase3_explanations.shared import ATTACK_FAMILY_TAXONOMY, confidence_band, read_json, write_json


def classify_from_packet(packet: dict[str, Any], max_candidates: int = 4) -> dict[str, Any]:
    """Handle classify from packet within the Phase 3 grounded explanation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    scores = {label: 0.0 for label in ATTACK_FAMILY_TAXONOMY}
    reasons: dict[str, list[str]] = {label: [] for label in ATTACK_FAMILY_TAXONOMY}

    physical = packet.get("physical_evidence", [])
    cyber = packet.get("cyber_evidence", [])
    affected_assets = packet.get("affected_assets", [])
    detector = packet.get("detector_evidence", {})

    suspicious_command = sum(1 for item in cyber if item.get("kind") == "suspicious_command")
    auth_failures = sum(1 for item in cyber if item.get("kind") == "auth_failure")
    attack_events = sum(1 for item in cyber if item.get("kind") == "attack_event")
    telemetry_flags = sum(1 for item in cyber if item.get("kind") in {"telemetry_mismatch", "telemetry_corruption_flag"})
    config_changes = sum(1 for item in cyber if item.get("kind") == "config_change")
    campaigns = {item.get("campaign_id") for item in cyber if item.get("campaign_id")}
    family_tags = {str(item.get("family")) for item in cyber if item.get("family") in ATTACK_FAMILY_TAXONOMY}

    pv_drop = any(item.get("category") == "pv_dispatch_drop" for item in physical)
    bess_jump = any(item.get("category") == "bess_power_jump" for item in physical)
    soc_inconsistency = any(item.get("category") == "soc_inconsistency" for item in physical)
    regulator_deviation = any(item.get("category") == "regulator_tap_deviation" for item in physical)
    feeder_shift = any(item.get("category") == "feeder_power_shift" for item in physical)
    voltage_shift = any(item.get("category") == "voltage_angle_shift" for item in physical)
    minimal_physical_change = not physical or all(abs(float(item.get("relative_change", 0.0))) < 0.08 for item in physical)
    control_assets = {
        str(asset)
        for asset in affected_assets
        if not str(asset).lower().startswith(("bus", "feeder", "substation"))
    }

    candidate_hints = detector.get("candidate_family_hints", [])
    for hint in candidate_hints:
        label = hint.get("label")
        if label in scores:
            scores[label] += float(hint.get("weight", 0.0))
            reasons[label].append(str(hint.get("reason", "detector hint")))

    if len(family_tags) == 1:
        tagged_family = next(iter(family_tags))
        scores[tagged_family] += 0.58
        reasons[tagged_family].append("Attack-tagged cyber evidence is consistent with this family.")

    if suspicious_command and (bess_jump or pv_drop or feeder_shift):
        scores["unauthorized_command"] += 0.92
        reasons["unauthorized_command"].append("Suspicious command activity aligns with a physical DER response.")

    if "command_suppression" in family_tags:
        scores["command_suppression"] += 0.7
        reasons["command_suppression"].append("Attack-tagged cyber evidence explicitly indicates command suppression.")

    if suspicious_command == 0 and (soc_inconsistency or feeder_shift) and any("target_kw" in str(item.get("signal", "")) for item in physical):
        scores["command_suppression"] += 0.78
        reasons["command_suppression"].append("Physical dispatch changed without a matching command trace in the lookback window.")

    if "false_data_injection" in family_tags:
        scores["false_data_injection"] += 0.62
        reasons["false_data_injection"].append("Attack-tagged cyber evidence indicates false-data behavior.")

    if telemetry_flags and minimal_physical_change:
        scores["false_data_injection"] += 0.82
        reasons["false_data_injection"].append("Telemetry mismatch is present while physical behavior remains comparatively stable.")

    if "telemetry_corruption" in family_tags:
        scores["telemetry_corruption"] += 0.48
        reasons["telemetry_corruption"].append("Attack-tagged cyber evidence indicates telemetry corruption.")

    if telemetry_flags and not minimal_physical_change:
        scores["telemetry_corruption"] += 0.67
        reasons["telemetry_corruption"].append("Telemetry anomalies are visible and physical behavior is also disturbed.")

    if pv_drop and any("available_kw" in str(item.get("signal", "")) for item in physical):
        scores["curtailment_inconsistency"] += 0.86
        reasons["curtailment_inconsistency"].append("PV dispatch fell below inferred available output inside the same window.")

    if len(control_assets) >= 2 or (regulator_deviation and (pv_drop or bess_jump)):
        scores["coordinated_campaign"] += 0.9
        reasons["coordinated_campaign"].append("Multiple assets or campaign-linked cyber evidence indicate coordinated behavior.")
    elif len(control_assets) >= 2 and len(campaigns) >= 1:
        scores["coordinated_campaign"] += 0.22
        reasons["coordinated_campaign"].append("Multiple control assets appear with a common campaign tag.")

    if (feeder_shift or voltage_shift or bess_jump or pv_drop) and suspicious_command == 0 and telemetry_flags == 0:
        scores["physical_dispatch_anomaly"] += 0.58
        reasons["physical_dispatch_anomaly"].append("Physical dispatch shifted materially without strong cyber corroboration.")

    if telemetry_flags and (feeder_shift or pv_drop or regulator_deviation):
        scores["cyber_physical_mismatch"] += 0.74
        reasons["cyber_physical_mismatch"].append("Cyber and physical observations diverge in a way that suggests cross-layer inconsistency.")

    if auth_failures and not suspicious_command:
        scores["unauthorized_command"] += 0.18
        reasons["unauthorized_command"].append("Authentication precursor activity raises the likelihood of unauthorized access.")

    if config_changes and regulator_deviation:
        scores["coordinated_campaign"] += 0.12
        reasons["coordinated_campaign"].append("Control-plane configuration changes coincide with regulator deviation.")

    if attack_events and not any(scores[label] > 0.5 for label in ATTACK_FAMILY_TAXONOMY if label != "unknown_anomaly"):
        family = next((item.get("family") for item in cyber if item.get("family") in ATTACK_FAMILY_TAXONOMY), None)
        if family:
            scores[family] += 0.7
            reasons[family].append("Attack-tagged cyber event supports this family.")

    ranked = [
        {
            "label": label,
            "score": round(min(score, 1.0), 3),
            "confidence": confidence_band(min(score, 1.0)),
            "reasoning": reasons[label][:3],
        }
        for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if score > 0
    ]
    if not ranked:
        ranked = [
            {
                "label": "unknown_anomaly",
                "score": 0.3,
                "confidence": "low",
                "reasoning": ["Evidence did not cleanly match any closed-set family."],
            }
        ]
    if ranked[0]["score"] < 0.45:
        ranked.insert(
            0,
            {
                "label": "unknown_anomaly",
                "score": round(max(0.35, 1.0 - ranked[0]["score"]), 3),
                "confidence": "low",
                "reasoning": ["Best available evidence is weak, so the anomaly should remain uncommitted."],
            },
        )
    ranked = ranked[:max_candidates]
    return {
        "primary_family": ranked[0]["label"],
        "candidate_families": ranked,
    }


def main() -> None:
    """Run the command-line entrypoint for the Phase 3 grounded explanation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Rule-based closed-set attack-family preclassifier.")
    parser.add_argument("--packet", required=True, help="Explanation packet JSON.")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    packet = read_json(args.packet)
    result = classify_from_packet(packet)
    if args.output:
        write_json(result, args.output)
    else:
        print(result)


if __name__ == "__main__":
    main()
