# Publishable Report Index

This index points readers to the normalized publication-facing reports and selected evidence artifacts.

| context                         | artifact_type   | publishable_artifact                                           | description                                      |
|:--------------------------------|:----------------|:---------------------------------------------------------------|:-------------------------------------------------|
| canonical_benchmark             | csv             | artifacts/benchmark/phase1_window_model_comparison_full.csv    | Full detector benchmark window/model comparison. |
| canonical_benchmark             | md              | docs/reports/phase1_window_model_comparison_full.md            | Readable detector benchmark report.              |
| canonical_benchmark             | figure          | docs/figures/phase1_window_comparison_5s_10s_60s_300s.png      | Benchmark F1 by window.                          |
| canonical_benchmark             | figure          | docs/figures/phase1_model_vs_window_heatmap.png                | Benchmark model/window heatmap.                  |
| heldout_replay                  | csv             | artifacts/replay/benchmark_vs_replay_metrics.csv               | Benchmark vs replay separation metrics.          |
| heldout_replay                  | csv             | artifacts/replay/multi_model_heldout_metrics.csv               | Existing frozen-candidate replay metrics.        |
| heldout_synthetic_zero_day_like | csv             | artifacts/zero_day_like/zero_day_model_window_results_full.csv | Heldout synthetic detector matrix.               |
| heldout_synthetic_zero_day_like | md              | docs/reports/zero_day_model_window_results_full.md             | Heldout synthetic detector report.               |
| heldout_synthetic_zero_day_like | figure          | docs/figures/zero_day_model_window_heatmap.png                 | Heldout synthetic heatmap.                       |
| extension_ttm                   | csv             | artifacts/extensions/phase1_ttm_results.csv                    | TTM 60s extension benchmark result.              |
| extension_ttm                   | md              | docs/reports/phase1_ttm_eval_report.md                         | TTM extension report.                            |
| extension_lora                  | csv             | artifacts/extensions/phase3_lora_results.csv                   | LoRA experimental result.                        |
| extension_lora                  | md              | docs/reports/phase3_lora_eval_report.md                        | LoRA extension report.                           |
| xai                             | csv             | artifacts/xai/xai_case_level_audit.csv                         | Case-level explanation audit.                    |
| xai                             | md              | docs/reports/xai_final_validation_report.md                    | Final XAI validation report.                     |
| deployment                      | csv             | artifacts/deployment/deployment_benchmark_results.csv          | Offline deployment benchmark metrics.            |
| deployment                      | md              | docs/reports/deployment_benchmark_report.md                    | Offline deployment report.                       |
| cross_context                   | csv             | artifacts/zero_day_like/FINAL_MODEL_CONTEXT_COMPARISON.csv     | Context-separated model comparison.              |
| cross_context                   | md              | docs/reports/FINAL_MODEL_CONTEXT_COMPARISON.md                 | Context-separated model comparison report.       |
