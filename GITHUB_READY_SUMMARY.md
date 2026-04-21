# GitHub Ready Summary

## What The Project Does

DERGuardian builds a simulation-grounded DER cyber-physical anomaly-detection pipeline with validated synthetic attack generation, detector benchmarking, explanation validation, and offline deployment benchmarking.

## Canonical Findings

- Frozen canonical benchmark winner: Transformer @ 60s, F1 `0.8182`.
- Canonical model selection is separate from replay and heldout synthetic evaluation.

## Evaluation Contexts

- Canonical benchmark: model selection.
- Heldout replay: frozen-package transfer checks.
- Heldout synthetic zero-day-like: bounded generated-scenario evaluation.
- Extensions: TTM and LoRA, both secondary.

## Extension Branches

- TTM is real at 60s but extension-only.
- LoRA is experimental, weak, and not detector evidence.

## Limitations

- No real-world zero-day robustness claim.
- No human-like root-cause analysis claim.
- No AOI detector metric claim.
- No edge deployment claim.
