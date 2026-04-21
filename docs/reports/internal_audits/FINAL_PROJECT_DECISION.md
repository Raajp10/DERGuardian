# Final Project Decision

## Is the project now aligned with the diagram?

Mostly, but not literally box-for-box under the original wording. The repo now satisfies most of the intended functionality, while several original labels still overclaim the evidence and must be rewritten.

## Boxes that are truly satisfied

- Core Phase 1 data generation, preprocessing, and benchmark execution.
- Core Phase 2 scenario bundle execution, validation, attacked dataset generation, and quality auditing.
- Unified residual dataset generation, replay evaluation, and detector benchmarking.

## Boxes that remain partial

- LLM normal-behavior context learning.
- Tiny LLM (LoRA) as part of the main detector path.
- Human-like explanation / root-cause analysis.
- Evaluation box as written with AOI.
- Deployment box as written with edge-AI wording.
- Explicit system-card artifact if interpreted literally.

## What can be said tomorrow

- The canonical benchmark winner remains transformer @ 60 s.
- Heldout replay exists and is separate from benchmark selection.
- Phase 2 now has a formal coverage/diversity/difficulty audit.
- TTM and LoRA are real extension branches, but secondary.
- XAI supports grounded operator-facing explanations, not human-like RCA.
- Deployment evidence is offline workstation replay benchmarking, not an edge-hardware deployment result.

## What is safe in the paper

- Use the safe-now and safe-with-wording-changes lists from `COMPLETE_PUBLICATION_SAFE_CLAIMS.md`.
- Keep benchmark, heldout replay, and extension experiments explicitly separated.
- Do not imply that replay gains overwrite the canonical benchmark winner.

## Detector Window / Zero-Day-Like Audit Addendum

- Canonical benchmark winner unchanged: yes, `transformer @ 60s` remains frozen.
- Full detector window comparison completed: yes for implemented detector families across `5s`, `10s`, `60s`, and `300s`.
- Heldout synthetic detector sweep completed: yes for implemented ready-package detector families at `10s`, `60s`, and `300s`; `5s` is documented as blocked by CPU-only raw window-generation cost.
- TTM included at `60s` only as an extension; ARIMA remains unimplemented.
- AOI should not be claimed for this detector-side pass.
