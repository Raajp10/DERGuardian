# Improved Publication-safe Claims

## Safe To Claim Now

- The repository now separates benchmark test-split metrics from frozen-model heldout replay metrics in a paper-safe way.
- The heldout replay pass is scientifically cleaner because the previous finite-value clipping is no longer mislabeled as non-finite residual corruption.
- The saved 60s transformer can be compared fairly against other frozen saved winners (10s threshold baseline and 300s LSTM) without retraining Phase 1 from scratch.
- Cross-generator heldout replay remains variable across generator sources, so generalization claims should stay cautious.

## Safe With Minor Wording Changes

- 'Zero-day-like heldout cross-generator replay' is acceptable if it is explicitly described as bounded synthetic replay across independently produced scenario bundles.
- 'Grounded explanation layer' is acceptable if the text also notes that family attribution is stronger than asset attribution and evidence grounding is partial.

## Not Safe To Claim

- Real-world zero-day robustness.
- Human-like root-cause analysis.
- Universal transfer across all generators or all unseen attack bundles.

## Missing Evidence Still Required

- External utility or lab data outside this synthetic scenario family.
- More than one independent non-LLM heldout source.
- Stability checks across alternate calibration thresholds and repeated seeds for heldout replay.
