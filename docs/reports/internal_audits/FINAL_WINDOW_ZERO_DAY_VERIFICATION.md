# Final Window / Zero-Day Verification

## Output Status

| artifact_path                                                                        | exists   | nonempty   |   size_bytes | artifact_type   |
|:-------------------------------------------------------------------------------------|:---------|:-----------|-------------:|:----------------|
| <repo>/detector_window_model_coverage_audit.csv        | True     | True       |         9659 | csv             |
| <repo>/detector_window_model_coverage_summary.md       | True     | True       |         1793 | md              |
| <repo>/phase1_window_model_comparison_full.csv         | True     | True       |        12884 | csv             |
| <repo>/phase1_window_model_comparison_full.md          | True     | True       |         9347 | md              |
| <repo>/phase1_window_comparison_5s_10s_60s_300s.png    | True     | True       |        70246 | png             |
| <repo>/phase1_model_vs_window_heatmap.png              | True     | True       |        65963 | png             |
| <repo>/phase1_best_per_window_summary.csv              | True     | True       |          626 | csv             |
| <repo>/phase1_ttm_window_comparison.csv                | True     | True       |        12884 | csv             |
| <repo>/phase1_ttm_window_comparison.md                 | True     | True       |        10449 | md              |
| <repo>/phase1_ttm_vs_transformer_gru_lstm_if_arima.png | True     | True       |        68965 | png             |
| <repo>/zero_day_window_model_coverage_audit.csv        | True     | True       |        27776 | csv             |
| <repo>/zero_day_window_model_coverage_summary.md       | True     | True       |         4017 | md              |
| <repo>/zero_day_model_window_results_full.csv          | True     | True       |       104524 | csv             |
| <repo>/zero_day_model_window_results_full.md           | True     | True       |         5610 | md              |
| <repo>/zero_day_f1_by_window.png                       | True     | True       |       164944 | png             |
| <repo>/zero_day_precision_recall_by_model.png          | True     | True       |       112081 | png             |
| <repo>/zero_day_model_window_heatmap.png               | True     | True       |        68388 | png             |
| <repo>/zero_day_best_per_window_summary.csv            | True     | True       |          565 | csv             |
| <repo>/zero_day_family_model_window_results.csv        | True     | True       |        47857 | csv             |
| <repo>/zero_day_family_difficulty_summary.md           | True     | True       |         1990 | md              |
| <repo>/FINAL_MODEL_CONTEXT_COMPARISON.csv              | True     | True       |        10248 | csv             |
| <repo>/FINAL_MODEL_CONTEXT_COMPARISON.md               | True     | True       |         9789 | md              |
| <repo>/FINAL_MODEL_CONTEXT_COMPARISON.png              | True     | True       |       112025 | png             |
| <repo>/MODEL_WINDOW_ZERO_DAY_SAFE_CLAIMS.md            | True     | True       |         1991 | md              |
| <repo>/FINAL_WINDOW_ZERO_DAY_VERIFICATION.md           | True     | True       |         5460 | md              |
| <repo>/FINAL_WINDOW_ZERO_DAY_OUTPUT_STATUS.csv         | True     | True       |         2529 | csv             |

## Logical Checks

| check_name                       | passed   | notes                                                       |
|:---------------------------------|:---------|:------------------------------------------------------------|
| canonical_winner_preserved       | True     | Transformer @ 60s still present as canonical completed row. |
| ttm_extension_separated          | True     | TTM rows must stay extension-labeled.                       |
| zero_day_separate_from_benchmark | True     | Zero-day-like rows must keep their own context label.       |
| aoi_not_claimed_in_safe_claims   | True     | Safe-claims file explicitly blocks AOI detector claim.      |

## Separation Checks

- Canonical benchmark, replay, heldout synthetic zero-day-like evaluation, and extension rows are kept separate by explicit context fields.
- LoRA explanation artifacts were not used as detector benchmark evidence in this pass.

## Runtime Notes

- Re-running this script emits a non-blocking scikit-learn version warning when loading the saved IsolationForest package (`1.8.0` artifact loaded under local `1.7.2`). The warning is documented here because it affects reproducibility hygiene, but it did not prevent the saved-package evaluations from completing.
