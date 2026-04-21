# Explanation Contract

## Packet Contract

The packet built by `build_explanation_packet.py` contains:

- `incident_id`
- `window`
- `model_name`
- `anomaly_score`
- `threshold`
- `score_margin`
- `predicted_alert`
- `severity`
- `affected_assets`
- `top_features`
- `physical_evidence`
- `cyber_evidence`
- `detector_evidence`
- `candidate_families`
- `offline_evaluation`

## Explanation Output Contract

The explanation JSON must match `explanation_schema.json` and include:

- `incident_id`
- `summary`
- `suspected_attack_family`
- `family_confidence`
- `why_flagged`
- `physical_evidence_used`
- `cyber_evidence_used`
- `operator_actions`
- `confidence_note`
- `limitations`

## Safe Response Rules

Allowed operator actions:

- inspect command logs
- verify setpoints versus schedules
- compare telemetry against local device logs
- inspect protocol source and actor identity
- verify DER dispatch policy
- escalate coordinated anomalies

Disallowed content:

- offensive or procedural attack instructions
- exploit chains
- steps to manipulate DERs
- commands that would control the feeder or grid

## Grounding Rules

- Every asset or signal named in the explanation must already exist in the packet.
- Evidence lists must reference entries already present in the packet.
- Strong certainty language should be avoided unless the packet evidence is unusually strong.
