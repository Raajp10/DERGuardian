# Professor Ready Method Summary

## Phase 1

Clean IEEE 123-bus DER data are generated, transformed into residual/deviation windows, and used for a multi-window detector benchmark. The frozen canonical benchmark winner remains `transformer @ 60s`.

## Phase 2

Synthetic attack scenarios are authored as schema-bound JSON, validated, compiled into attacked datasets, and audited for coverage, diversity, difficulty, repair status, and usable heldout coverage.

## Phase 3

Frozen detectors are evaluated across clearly separated contexts: canonical benchmark, heldout replay, heldout synthetic zero-day-like evaluation, and extension experiments. XAI and deployment are support layers; they are not used to claim human-like root-cause analysis or field edge deployment.

## What Is Fully Implemented

- Canonical detector benchmark with preserved benchmark model selection.
- Phase 2 scenario validation plus coverage, diversity, and difficulty audit.
- Heldout replay and heldout synthetic detector evaluation for feasible windows.
- TTM and LoRA extension evidence, reported separately from the canonical detector.
- XAI validation and offline lightweight deployment benchmark.

## What Is Partial

- The 5s heldout synthetic sweep is blocked by raw-window generation cost.
- LoRA evidence is experimental and weak.
- TTM is an extension-only comparison and is not the canonical detector.
- No ARIMA detector implementation is available in the release-facing model set.
- The implemented autoencoder is an MLP autoencoder, not an LSTM autoencoder.

## Safe Wording

Use: canonical benchmark, heldout replay, heldout synthetic zero-day-like evaluation, grounded operator-facing explanation support, TTM extension comparison, experimental LoRA explanation branch, and offline lightweight deployment benchmark.

