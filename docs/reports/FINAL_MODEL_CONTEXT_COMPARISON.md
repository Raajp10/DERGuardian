# Final Model Context Comparison

This file keeps the major evaluation contexts separate: canonical benchmark, existing replay, heldout synthetic zero-day-like evaluation, and the TTM extension branch.

## Best Row Per Context

| context_name                             | model_name   | window_label   |     f1 | canonical_or_extension   | notes                                                                |
|:-----------------------------------------|:-------------|:---------------|-------:|:-------------------------|:---------------------------------------------------------------------|
| canonical_benchmark_test_split           | transformer  | 60s            | 0.8182 | canonical                | Frozen canonical benchmark context.                                  |
| existing_frozen_candidate_heldout_replay | lstm         | 300s           | 0.8099 | replay                   | Existing improved replay context using frozen candidate subset only. |
| heldout_synthetic_zero_day_like          | lstm         | 300s           | 0.8099 | canonical                | New full heldout synthetic detector sweep from frozen packages.      |

## Full Context Table

| context_name                             | model_name         | window_label   | canonical_or_extension   |   precision |   recall |     f1 |   mean_latency_seconds |
|:-----------------------------------------|:-------------------|:---------------|:-------------------------|------------:|---------:|-------:|-----------------------:|
| canonical_benchmark_test_split           | transformer        | 5s             | canonical                |      0.1013 |   0.9994 | 0.1839 |               336.538  |
| canonical_benchmark_test_split           | gru                | 5s             | canonical                |      0.1013 |   0.9994 | 0.1839 |               336.538  |
| canonical_benchmark_test_split           | autoencoder        | 5s             | canonical                |      0.0863 |   0.8194 | 0.1561 |               357.75   |
| canonical_benchmark_test_split           | isolation_forest   | 5s             | canonical                |      0.1208 |   0.9508 | 0.2144 |               343.5    |
| canonical_benchmark_test_split           | threshold_baseline | 5s             | canonical                |      0.0601 |   0.4938 | 0.1072 |               432.4    |
| canonical_benchmark_test_split           | lstm               | 5s             | canonical                |      0.1011 |   0.9972 | 0.1835 |               336.538  |
| canonical_benchmark_test_split           | transformer        | 10s            | canonical                |      0.0986 |   0.9921 | 0.1793 |               343.462  |
| canonical_benchmark_test_split           | gru                | 10s            | canonical                |      0.0959 |   0.9616 | 0.1743 |               343.462  |
| canonical_benchmark_test_split           | autoencoder        | 10s            | canonical                |      0.0861 |   0.8184 | 0.1559 |               358      |
| canonical_benchmark_test_split           | isolation_forest   | 10s            | canonical                |      0.2029 |   0.9514 | 0.3345 |               347.333  |
| canonical_benchmark_test_split           | threshold_baseline | 10s            | canonical                |      0.9669 |   0.5686 | 0.7161 |               416.143  |
| canonical_benchmark_test_split           | lstm               | 10s            | canonical                |      0.0936 |   0.9952 | 0.1712 |               351.462  |
| canonical_benchmark_test_split           | transformer        | 60s            | canonical                |      0.6923 |   1      | 0.8182 |               641      |
| canonical_benchmark_test_split           | gru                | 60s            | canonical                |      0.775  |   0.8611 | 0.8158 |               641      |
| canonical_benchmark_test_split           | autoencoder        | 60s            | canonical                |      0.0887 |   0.8228 | 0.1602 |               402      |
| canonical_benchmark_test_split           | isolation_forest   | 60s            | canonical                |      0.2227 |   0.943  | 0.3603 |               384      |
| canonical_benchmark_test_split           | threshold_baseline | 60s            | canonical                |      0.7899 |   0.5949 | 0.6787 |               509      |
| canonical_benchmark_test_split           | ttm_extension      | 60s            | extension                |      0.6341 |   0.8387 | 0.7222 |               706.6    |
| canonical_benchmark_test_split           | lstm               | 60s            | canonical                |      0.75   |   0.8333 | 0.7895 |               641      |
| canonical_benchmark_test_split           | transformer        | 300s           | canonical                |      0.2308 |   0.75   | 0.3529 |               871      |
| canonical_benchmark_test_split           | gru                | 300s           | canonical                |      0.2143 |   0.75   | 0.3333 |               871      |
| canonical_benchmark_test_split           | autoencoder        | 300s           | canonical                |      0.0987 |   0.8333 | 0.1765 |               571      |
| canonical_benchmark_test_split           | isolation_forest   | 300s           | canonical                |      0.1226 |   0.8889 | 0.2155 |               556      |
| canonical_benchmark_test_split           | threshold_baseline | 300s           | canonical                |      0.2872 |   0.75   | 0.4154 |               614.333  |
| canonical_benchmark_test_split           | lstm               | 300s           | canonical                |      0.3333 |   1      | 0.5    |               871      |
| existing_frozen_candidate_heldout_replay | threshold_baseline | 10s            | replay                   |      0.4146 |   0.273  | 0.2512 |               114.467  |
| existing_frozen_candidate_heldout_replay | transformer        | 60s            | replay                   |      0.8256 |   0.5622 | 0.6613 |                 3.7    |
| existing_frozen_candidate_heldout_replay | lstm               | 300s           | replay                   |      0.9463 |   0.7185 | 0.8099 |                65.6667 |
| heldout_synthetic_zero_day_like          | transformer        | 10s            | canonical                |      0.0395 |   0.9645 | 0.0758 |                 1      |
| heldout_synthetic_zero_day_like          | gru                | 10s            | canonical                |      0.0411 |   0.9208 | 0.0784 |                 1.05   |
| heldout_synthetic_zero_day_like          | autoencoder        | 10s            | canonical                |      0.0498 |   0.7804 | 0.0932 |                 1      |
| heldout_synthetic_zero_day_like          | isolation_forest   | 10s            | canonical                |      0.0358 |   0.8968 | 0.0687 |                 1      |
| heldout_synthetic_zero_day_like          | threshold_baseline | 10s            | canonical                |      0.4146 |   0.273  | 0.2512 |               114.467  |
| heldout_synthetic_zero_day_like          | lstm               | 10s            | canonical                |      0.0407 |   0.9224 | 0.0776 |                 1.05   |
| heldout_synthetic_zero_day_like          | transformer        | 60s            | canonical                |      0.8256 |   0.5622 | 0.6613 |                 3.7    |
| heldout_synthetic_zero_day_like          | gru                | 60s            | canonical                |      0.737  |   0.3835 | 0.48   |                15.1067 |
| heldout_synthetic_zero_day_like          | autoencoder        | 60s            | canonical                |      0.0498 |   0.8227 | 0.0939 |                 1      |
| heldout_synthetic_zero_day_like          | isolation_forest   | 60s            | canonical                |      0.0381 |   0.8929 | 0.0729 |                 1      |
| heldout_synthetic_zero_day_like          | threshold_baseline | 60s            | canonical                |      0.5031 |   0.4091 | 0.4083 |                46.4    |
| heldout_synthetic_zero_day_like          | ttm_extension      | 60s            | extension                |      0.5083 |   0.2128 | 0.2832 |                 1      |
| heldout_synthetic_zero_day_like          | lstm               | 60s            | canonical                |      0.9236 |   0.3693 | 0.5247 |                15.4    |
| heldout_synthetic_zero_day_like          | transformer        | 300s           | canonical                |      0.8855 |   0.6637 | 0.7518 |                49.5333 |
| heldout_synthetic_zero_day_like          | gru                | 300s           | canonical                |      0.9261 |   0.6747 | 0.7721 |                65      |
| heldout_synthetic_zero_day_like          | autoencoder        | 300s           | canonical                |      0.0602 |   0.9407 | 0.1132 |                 1      |
| heldout_synthetic_zero_day_like          | isolation_forest   | 300s           | canonical                |      0.0306 |   0.5361 | 0.0578 |               340.714  |
| heldout_synthetic_zero_day_like          | threshold_baseline | 300s           | canonical                |      0.7309 |   0.3812 | 0.4872 |                40      |
| heldout_synthetic_zero_day_like          | lstm               | 300s           | canonical                |      0.9463 |   0.7185 | 0.8099 |                65.6667 |

## Safe Reading

- `transformer @ 60s` remains the canonical benchmark winner.
- A different best row in replay or heldout synthetic evaluation would describe a different context, not a replacement of the canonical benchmark selection.
- TTM remains extension-only.
