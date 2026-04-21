# Professor Ready Method Summary

## Phase 1

Clean IEEE 123-bus DER data are generated, transformed into residual/deviation windows, and used for a multi-window detector benchmark. Transformer @ 60s remains the canonical winner.

## Phase 2

Synthetic scenarios are generated as schema-bound JSON, validated, compiled into attacked datasets, and audited for coverage/diversity/difficulty.

## Phase 3

Frozen detectors are evaluated across benchmark, replay, and heldout synthetic contexts. XAI and deployment are support layers, not overclaiming layers.

## What Is Fully Implemented

- Canonical detector benchmark.
- Phase 2 scenario validation and coverage audit.
- Heldout replay and heldout synthetic detector evaluation for feasible windows.
- TTM and LoRA extension evidence.
- XAI validation and offline deployment benchmark.

## What Is Partial

- 5s heldout synthetic sweep is blocked by raw-window generation cost.
- LoRA evidence is weak.
- TTM is 60s only.
- No ARIMA or LSTM autoencoder detector implementation.

## Safe Wording

Use: canonical benchmark, heldout replay, heldout synthetic zero-day-like evaluation, grounded operator-facing explanation support, and offline lightweight deployment benchmark.
