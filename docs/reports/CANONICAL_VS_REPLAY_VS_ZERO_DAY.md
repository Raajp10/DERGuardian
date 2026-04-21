# Canonical vs Replay vs Zero-Day-Like Evaluation

## Context Definitions

- **Canonical benchmark**: Phase 1 test-split model selection. Source of truth: `outputs/window_size_study/final_window_comparison.csv`.
- **Heldout replay**: Frozen model replay on heldout/repaired synthetic bundles.
- **Heldout synthetic zero-day-like evaluation**: Generated Phase 2 attack bundles evaluated with frozen packages; bounded synthetic evidence only.

## Context Comparison

| context_name                             | model_name         | window_label   | canonical_or_extension   |   precision |   recall |     f1 |
|:-----------------------------------------|:-------------------|:---------------|:-------------------------|------------:|---------:|-------:|
| canonical_benchmark_test_split           | transformer        | 5s             | canonical                |      0.1013 |   0.9994 | 0.1839 |
| canonical_benchmark_test_split           | gru                | 5s             | canonical                |      0.1013 |   0.9994 | 0.1839 |
| canonical_benchmark_test_split           | autoencoder        | 5s             | canonical                |      0.0863 |   0.8194 | 0.1561 |
| canonical_benchmark_test_split           | isolation_forest   | 5s             | canonical                |      0.1208 |   0.9508 | 0.2144 |
| canonical_benchmark_test_split           | threshold_baseline | 5s             | canonical                |      0.0601 |   0.4938 | 0.1072 |
| canonical_benchmark_test_split           | lstm               | 5s             | canonical                |      0.1011 |   0.9972 | 0.1835 |
| canonical_benchmark_test_split           | transformer        | 10s            | canonical                |      0.0986 |   0.9921 | 0.1793 |
| canonical_benchmark_test_split           | gru                | 10s            | canonical                |      0.0959 |   0.9616 | 0.1743 |
| canonical_benchmark_test_split           | autoencoder        | 10s            | canonical                |      0.0861 |   0.8184 | 0.1559 |
| canonical_benchmark_test_split           | isolation_forest   | 10s            | canonical                |      0.2029 |   0.9514 | 0.3345 |
| canonical_benchmark_test_split           | threshold_baseline | 10s            | canonical                |      0.9669 |   0.5686 | 0.7161 |
| canonical_benchmark_test_split           | lstm               | 10s            | canonical                |      0.0936 |   0.9952 | 0.1712 |
| canonical_benchmark_test_split           | transformer        | 60s            | canonical                |      0.6923 |   1      | 0.8182 |
| canonical_benchmark_test_split           | gru                | 60s            | canonical                |      0.775  |   0.8611 | 0.8158 |
| canonical_benchmark_test_split           | autoencoder        | 60s            | canonical                |      0.0887 |   0.8228 | 0.1602 |
| canonical_benchmark_test_split           | isolation_forest   | 60s            | canonical                |      0.2227 |   0.943  | 0.3603 |
| canonical_benchmark_test_split           | threshold_baseline | 60s            | canonical                |      0.7899 |   0.5949 | 0.6787 |
| canonical_benchmark_test_split           | ttm_extension      | 60s            | extension                |      0.6341 |   0.8387 | 0.7222 |
| canonical_benchmark_test_split           | lstm               | 60s            | canonical                |      0.75   |   0.8333 | 0.7895 |
| canonical_benchmark_test_split           | transformer        | 300s           | canonical                |      0.2308 |   0.75   | 0.3529 |
| canonical_benchmark_test_split           | gru                | 300s           | canonical                |      0.2143 |   0.75   | 0.3333 |
| canonical_benchmark_test_split           | autoencoder        | 300s           | canonical                |      0.0987 |   0.8333 | 0.1765 |
| canonical_benchmark_test_split           | isolation_forest   | 300s           | canonical                |      0.1226 |   0.8889 | 0.2155 |
| canonical_benchmark_test_split           | threshold_baseline | 300s           | canonical                |      0.2872 |   0.75   | 0.4154 |
| canonical_benchmark_test_split           | lstm               | 300s           | canonical                |      0.3333 |   1      | 0.5    |
| existing_frozen_candidate_heldout_replay | threshold_baseline | 10s            | replay                   |      0.4146 |   0.273  | 0.2512 |
| existing_frozen_candidate_heldout_replay | transformer        | 60s            | replay                   |      0.8256 |   0.5622 | 0.6613 |
| existing_frozen_candidate_heldout_replay | lstm               | 300s           | replay                   |      0.9463 |   0.7185 | 0.8099 |
| heldout_synthetic_zero_day_like          | transformer        | 10s            | canonical                |      0.0395 |   0.9645 | 0.0758 |
| heldout_synthetic_zero_day_like          | gru                | 10s            | canonical                |      0.0411 |   0.9208 | 0.0784 |
| heldout_synthetic_zero_day_like          | autoencoder        | 10s            | canonical                |      0.0498 |   0.7804 | 0.0932 |
| heldout_synthetic_zero_day_like          | isolation_forest   | 10s            | canonical                |      0.0358 |   0.8968 | 0.0687 |
| heldout_synthetic_zero_day_like          | threshold_baseline | 10s            | canonical                |      0.4146 |   0.273  | 0.2512 |
| heldout_synthetic_zero_day_like          | lstm               | 10s            | canonical                |      0.0407 |   0.9224 | 0.0776 |
| heldout_synthetic_zero_day_like          | transformer        | 60s            | canonical                |      0.8256 |   0.5622 | 0.6613 |
| heldout_synthetic_zero_day_like          | gru                | 60s            | canonical                |      0.737  |   0.3835 | 0.48   |
| heldout_synthetic_zero_day_like          | autoencoder        | 60s            | canonical                |      0.0498 |   0.8227 | 0.0939 |
| heldout_synthetic_zero_day_like          | isolation_forest   | 60s            | canonical                |      0.0381 |   0.8929 | 0.0729 |
| heldout_synthetic_zero_day_like          | threshold_baseline | 60s            | canonical                |      0.5031 |   0.4091 | 0.4083 |
| heldout_synthetic_zero_day_like          | ttm_extension      | 60s            | extension                |      0.5083 |   0.2128 | 0.2832 |

## Interpretation

If the best model changes across contexts, that is reported as context-specific behavior. It does not replace the frozen canonical benchmark selection.
