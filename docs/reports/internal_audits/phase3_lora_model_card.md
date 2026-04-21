# Phase 3 LoRA Model Card

## Status

Experimental extension branch for bounded family/asset/explanation JSON generation.

## Base model

- `google/flan-t5-small`
- LoRA adapter only; canonical detector remains unchanged.

## Training data

- Training generators: ['chatgpt', 'claude']
- Validation generator: gemini
- Test generator: grok
- Supervision uses structured runtime packets plus ground-truth scenario family/assets/signals derived from heldout benchmark bundles.

## Intended use

- Post-alert operator-facing family attribution
- Likely affected asset suggestion
- Grounded signal citation in a compact JSON response

## Not intended use

- Primary anomaly detection
- Human-level root-cause analysis claims
- Real-world deployment claims

## Limitations

- Trained only on bounded synthetic heldout explanation packets.
- Human-authored heldout bundle was excluded because aligned xai_v4 supervision packets were not present.
- Results should be cited as explanation-side evidence, not benchmark-winning detector evidence.
