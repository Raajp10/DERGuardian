# Real-World Zero-Day Claim Audit

## Decision

DERGuardian still does **not** support a real-world zero-day robustness claim.

The strongest safe wording remains:

> DERGuardian evaluates detector transfer on heldout synthetic, zero-day-like Phase 2 attack scenarios generated outside the canonical benchmark path.

## Evidence That Exists

- Heldout synthetic scenario bundles exist and are evaluated separately from the canonical benchmark.
- Detector-side heldout synthetic results are recorded in `artifacts/zero_day_like/zero_day_model_window_results_full.csv`.
- Additional extension evidence exists for TTM and the new LSTM autoencoder branch, but these are still synthetic/replay contexts.
- Scenario diversity, XAI, AOI, and deployment evidence now add supporting audit layers, not real-world external validation.

## Evidence That Does Not Exist

- No field DER deployment data.
- No independently collected real cyberattack traces.
- No real-world incident labels.
- No hardware-in-the-loop validation against a physical DER testbed.
- No external benchmark demonstrating zero-day robustness beyond the generated scenario families.

## Safe Wording

- "heldout synthetic zero-day-like evaluation"
- "unseen generated attack scenario evaluation"
- "synthetic transfer evaluation across heldout Phase 2 bundles"

## Unsafe Wording

- "real-world zero-day robustness"
- "proven zero-day detection"
- "field-validated unseen attack robustness"
- "deployment-ready zero-day DER defense"

## Bottom Line

This pass strengthens in-repository evidence but does not add external evidence. The real-world zero-day non-claim remains scientifically correct.
