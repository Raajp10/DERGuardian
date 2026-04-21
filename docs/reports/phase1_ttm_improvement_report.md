# TTM Improvement Pass

This pass did not rerun the expensive TTM training job. The existing run is real and already includes a saved checkpoint, predictions, latency rows, and training history.
The improvement is an evidence join: AOI is added where computable and existing heldout synthetic rows are kept extension-only.

## Key Result

- Best TTM F1 in available rows: `0.722222` in context `canonical_benchmark_test_split_extension`.
- TTM remains extension-only and does not replace transformer @ 60s as the canonical benchmark winner.
- Updated CSV: `artifacts/extensions/phase1_ttm_updated_results.csv`
