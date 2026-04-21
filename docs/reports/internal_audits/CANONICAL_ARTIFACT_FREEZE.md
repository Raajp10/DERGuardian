# Canonical Artifact Freeze

This file documents the frozen canonical artifacts used as source-of-truth inputs for the added evidence layers.
No canonical benchmark outputs were overwritten by this extension pass.

## Frozen benchmark path

- Canonical benchmark summary: `outputs\reports\model_full_run_artifacts\model_summary_table_full.csv`
- Benchmark winner remains `transformer` at `60s`.
- Winner ready package: `<repo>/outputs\window_size_study\60s\ready_packages\transformer`
- Benchmark decision note: `BEST_MODEL_DECISION.md`

## Frozen Phase 2 canonical attacked bundle

- Canonical manifest: `outputs\attacked\scenario_manifest.json`
- Canonical attack labels: `outputs\attacked\attack_labels.parquet`
- Canonical Phase 2 validation summary: `outputs\reports\attacked_validation_summary.md`

## Frozen Phase 3 replay path

- Existing heldout replay metrics: `heldout_bundle_metrics.csv`
- Existing repaired/balanced replay metrics: `balanced_heldout_metrics.csv`
- Existing benchmark vs replay separation: `benchmark_vs_replay_explainer.md`
- Existing improved replay report: `IMPROVED_PHASE3_RESULTS_REPORT.md`

## Extension rule

All new outputs created in this pass sit alongside the frozen canonical path and are explicitly labeled as coverage analysis, heldout replay support, or extension experiments.
