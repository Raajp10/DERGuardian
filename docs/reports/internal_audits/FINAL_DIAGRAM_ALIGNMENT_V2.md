# Final Diagram Alignment V2

This file gives the final safe diagram wording after the repository cleanup.

## Phase 1

Safe label:

> Normal-system simulation, preprocessing, residual/deviation window construction, and detector training/benchmarking

Implementation status: fully implemented for the canonical benchmark path.

Notes:

- Phase 1 is the training and canonical model-selection path.
- The canonical benchmark winner remains `transformer @ 60s`.
- The actual benchmark includes threshold baseline, Isolation Forest, MLP autoencoder, GRU, LSTM, and Transformer models.
- Do not label the autoencoder as an LSTM autoencoder.

## Phase 2

Safe label:

> Schema-bound synthetic attack scenario generation, validation, injection, and attacked dataset construction

Implementation status: implemented with accepted/rejected/repaired scenario inventory and coverage/difficulty audit.

Notes:

- LLM assistance belongs here only as structured scenario authoring support.
- Heldout synthetic scenarios are zero-day-like only in the bounded synthetic sense.
- Do not claim real-world zero-day robustness.

## Phase 3

Safe label:

> Frozen-detector inference, heldout replay, heldout synthetic zero-day-like evaluation, XAI support, and offline deployment benchmarking

Implementation status: implemented as separate evidence contexts.

Notes:

- Phase 3 is inference/evaluation with frozen detector packages, not canonical model selection.
- Replay is replay.
- Heldout synthetic zero-day-like evaluation is separate from canonical benchmark selection.
- TTM is extension-only.
- LoRA is experimental/weak.
- XAI is grounded operator support, not human-like root-cause analysis.
- Deployment is offline workstation/lightweight benchmarking, not field edge deployment.
- AOI is not implemented and must not appear as a detector metric.

## Replacement Labels

| Original/Overclaiming Label | Final Safe Label |
|---|---|
| LLM learns normal behavior | Context summaries and bounded reasoning artifacts support interpretation |
| Model Training ML Models + Tiny LLM (LoRA) | Canonical detector training plus separate LoRA explanation/classification extension |
| Human-like Explanation LLM Generates Root Cause Analysis | Grounded post-alert operator-facing explanation support |
| Evaluation Metrics F1, Precision, Recall, AOI | Detector metrics plus separate XAI grounding metrics; AOI not implemented |
| Deployment Vision Lightweight Edge AI for DER | Offline lightweight deployment benchmark |
