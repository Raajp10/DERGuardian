# Benchmark vs Replay Explainer

## Why The Numbers Differ

- Benchmark metrics come from the canonical Phase 1 attacked test split. They are the saved model-selection numbers.
- Replay metrics come from frozen packages applied to full attacked bundles generated after model selection. These bundles have different scenario mixes, different acceptance filters, and different distribution shift.
- A replay score lower than the benchmark score is not automatically a coding error. It often reflects harder or shifted inputs rather than a mismatch in thresholding.

## Which Table To Cite

- Model selection: `final_window_comparison.csv` and the `benchmark_test_split` rows in `benchmark_vs_replay_metrics.csv`.
- Heldout generalization: `multi_model_heldout_metrics.csv` and the `heldout_replay_mean` rows in `benchmark_vs_replay_metrics.csv`.
- Latency discussion: use replay latency for bundle-level response claims and benchmark latency for within-benchmark model tradeoffs. Do not mix them in the same sentence without naming the context.

## Metrics Snapshot

```text
        model_name window_label      evaluation_context  precision   recall       f1  average_precision  roc_auc  mean_latency_seconds                                                                                                                                                                                    data_scope_note
threshold_baseline          10s    benchmark_test_split   0.966912 0.568649 0.716133           0.674127 0.661367            416.142857                                                                                          Canonical Phase 1 attacked test split from the saved benchmark run. Use this context for model selection.
threshold_baseline          10s canonical_bundle_replay   0.483989 0.555986 0.517495           0.562826 0.760188              9.000000                                                                   Full canonical attacked bundle replay with a frozen saved package. Use this context for in-domain replay, not test-split claims.
threshold_baseline          10s     heldout_replay_mean   0.414570 0.272966 0.251209           0.311335 0.649676            114.466667 Mean across frozen-package replays on accepted heldout bundles (chatgpt, claude, gemini, grok, plus the added independent heldout source). Use this context for heldout generalization discussion.
       transformer          60s    benchmark_test_split   0.692308 1.000000 0.818182           0.906840 0.997416            641.000000                                                                                          Canonical Phase 1 attacked test split from the saved benchmark run. Use this context for model selection.
       transformer          60s canonical_bundle_replay   0.827839 0.823315 0.825571           0.840915 0.907456              4.692308                                                                   Full canonical attacked bundle replay with a frozen saved package. Use this context for in-domain replay, not test-split claims.
       transformer          60s     heldout_replay_mean   0.825609 0.562208 0.661286           0.610495 0.822981              3.700000 Mean across frozen-package replays on accepted heldout bundles (chatgpt, claude, gemini, grok, plus the added independent heldout source). Use this context for heldout generalization discussion.
              lstm         300s    benchmark_test_split   0.333333 1.000000 0.500000           0.526515 0.983456            871.000000                                                                                          Canonical Phase 1 attacked test split from the saved benchmark run. Use this context for model selection.
              lstm         300s canonical_bundle_replay   0.936170 0.936170 0.936170           0.942743 0.971604             61.000000                                                                   Full canonical attacked bundle replay with a frozen saved package. Use this context for in-domain replay, not test-split claims.
              lstm         300s     heldout_replay_mean   0.946255 0.718485 0.809885           0.824964 0.918301             65.666667 Mean across frozen-package replays on accepted heldout bundles (chatgpt, claude, gemini, grok, plus the added independent heldout source). Use this context for heldout generalization discussion.
```
