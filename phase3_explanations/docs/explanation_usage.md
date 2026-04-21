# Explanation Usage

## Overview

The explanation layer takes a detector alert and compacts it into a structured packet that is safe to hand to an LLM or to a deterministic fallback renderer. The packet is intentionally small and evidence-focused.

## Inputs

- predictions parquet from a detector run
- results JSON with threshold metadata
- attacked and clean matched windows
- attacked cyber-event log
- attack labels and scenario manifest for offline scoring only

## Typical Commands

Build a packet for the highest-scoring alert in a specific scenario:

```bash
python phase3_explanations/build_explanation_packet.py --scenario-id scn_unauthorized_bess48_discharge --output outputs/reports/explanation_artifacts/packets/scn_unauthorized_bess48_discharge.json
```

Create a prompt package for an external LLM:

```bash
python phase3_explanations/generate_explanations_llm.py --packet outputs/reports/explanation_artifacts/packets/scn_unauthorized_bess48_discharge.json --export-prompt-package --output-dir outputs/reports/explanation_artifacts/prompt_package
```

Generate a deterministic draft explanation for smoke tests:

```bash
python phase3_explanations/generate_explanations_llm.py --packet outputs/reports/explanation_artifacts/packets/scn_unauthorized_bess48_discharge.json --simulate-grounded-output --output-dir outputs/reports/explanation_artifacts/explanations_tmp
```

Validate an LLM response:

```bash
python phase3_explanations/validate_explanations.py --packet outputs/reports/explanation_artifacts/packets/scn_unauthorized_bess48_discharge.json --explanation outputs/reports/explanation_artifacts/explanations_tmp/explanation_output.json
```

## Evidence Scope

The packet includes only:

- compact window-level detector evidence
- attacked-vs-clean physical deltas
- summarized cyber-event evidence
- candidate family hints

The LLM is not given raw time-series traces and should not reconstruct them.
