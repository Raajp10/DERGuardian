# Phase 3 Explanations

`phase3_explanations/` adds a grounded explanation layer on top of the existing anomaly-detection pipeline. It converts detector alerts plus structured physical and cyber evidence into a constrained explanation packet, a closed-set family preclassification, a JSON explanation contract, and a human-readable incident summary.

## Functional Role

This layer sits after anomaly detection. It does not change control settings, it does not issue DER commands, and it does not let the LLM invent raw evidence. The only material sent into the LLM step is the compact explanation packet built from structured artifacts already produced by the repository:

- anomaly predictions and thresholds from the trained models
- attacked and clean matched windows
- cyber-event logs
- offline scenario labels and manifest metadata for evaluation only

## Closed Family Taxonomy

- `false_data_injection`
- `unauthorized_command`
- `command_suppression`
- `telemetry_corruption`
- `curtailment_inconsistency`
- `coordinated_campaign`
- `physical_dispatch_anomaly`
- `cyber_physical_mismatch`
- `unknown_anomaly`

The explanation output must always choose one of the labels above or fall back to `unknown_anomaly`.

## What Ground Truth Is Used

Ground truth is split into two roles:

- Live explanation grounding:
  uses detector outputs, attacked-vs-clean window deltas, and cyber-event evidence only
- Offline evaluation only:
  uses `outputs/attacked/scenario_manifest.json` and `outputs/attacked/attack_labels.parquet`

Scenario metadata is never required for the explanation itself. It is attached only under `offline_evaluation` so we can score explanation quality later.

## What Gets Passed To The LLM

The LLM never receives raw full-length time series. The packet contains:

- incident ID and time window
- model name, anomaly score, threshold, and severity
- affected assets
- top discriminative window features
- structured physical evidence entries
- structured cyber evidence entries
- candidate families from the rule-based preclassifier

## What The LLM May Infer

Allowed:

- choose the most plausible family from the closed taxonomy
- explain why the alert fired using only packet evidence
- describe uncertainty and limitations
- recommend safe defensive follow-up checks

Not allowed:

- inventing unobserved root causes
- naming assets or signals not present in the packet
- producing offensive or procedural attack instructions
- suggesting grid-control actions

## Workflow

1. Build the explanation packet:

```bash
python phase3_explanations/build_explanation_packet.py --scenario-id scn_coordinated_pv83_reg4a_disturbance --output outputs/reports/explanation_artifacts/explanation_packet.json
```

2. Export a prompt package for an external LLM:

```bash
python phase3_explanations/generate_explanations_llm.py --packet outputs/reports/explanation_artifacts/explanation_packet.json --export-prompt-package --output-dir outputs/reports/explanation_artifacts/prompt_package
```

3. Or generate a deterministic grounded draft explanation for self-checks:

```bash
python phase3_explanations/generate_explanations_llm.py --packet outputs/reports/explanation_artifacts/explanation_packet.json --simulate-grounded-output --output-dir outputs/reports/explanation_artifacts
```

4. Validate the structured explanation:

```bash
python phase3_explanations/validate_explanations.py --packet outputs/reports/explanation_artifacts/explanation_packet.json --explanation outputs/reports/explanation_artifacts/explanation_output.json
```

5. Evaluate explanations offline against scenario labels:

```bash
python phase3_explanations/rationale_evaluator.py --packet-dir outputs/reports/explanation_artifacts/packets --explanation-dir outputs/reports/explanation_artifacts/explanations
```

## Validation Checks

The validator checks:

- JSON schema validity
- family label inside the closed taxonomy
- explanation references grounded in the packet
- asset mentions constrained to packet assets
- operator actions remain defensive only
- unsupported certainty language is not used

## Offline Evaluation

The evaluator writes:

- `outputs/reports/explanation_artifacts/explanation_eval_table.csv`
- `outputs/reports/explanation_artifacts/explanation_eval_report.md`

It scores:

- family classification accuracy
- asset attribution accuracy
- evidence grounding rate
- unknown usage rate
- explanation completeness

See `docs/explanation_usage.md` and `docs/explanation_contract.md` for the integration contract and field-level details.
