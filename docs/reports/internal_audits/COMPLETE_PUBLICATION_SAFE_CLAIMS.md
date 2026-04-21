# Complete Publication Safe Claims

## Safe now

- The canonical benchmark winner is still `transformer @ 60s` with benchmark F1 `0.818`.
- The project now includes a verified Phase 2 coverage/diversity/difficulty audit over `99` audited scenario rows with `88` validated usable rows.
- The heldout XAI layer supports evidence-grounded family attribution for operator support (`family=0.774`) but not strong asset localization (`asset=0.268`).
- The repo now includes an offline lightweight deployment benchmark on workstation CPU hardware for frozen detector packages.
- TinyTimeMixer and LoRA both exist as real extension experiments with persisted outputs and rerun logs.

## Safe with wording changes

- `grounded explanation layer for post-alert operator support`
- `canonical ML benchmark plus experimental LoRA explanation branch`
- `offline lightweight deployment benchmark`
- `LLM-generated synthetic heldout scenarios` instead of `zero-day style` as a robustness claim
- `detection metrics plus explanation grounding metrics` instead of `F1, Precision, Recall, AOI` as a single implemented metric stack

## Not safe

- Human-like root-cause analysis.
- Real-world zero-day robustness.
- Edge deployment or field deployment.
- ARIMA comparison results, because ARIMA is not implemented in the current repo.
- AOI as a claimed implemented detector metric unless separately added and validated.

## Missing evidence

- Edge-device runtime measurements.
- Human evaluation showing explanation quality at a human-analyst level.
- Real zero-day or field attack trials.
- Stronger explanation asset localization and signal grounding.
