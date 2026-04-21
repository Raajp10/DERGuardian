# Zero-Day-Like Model / Window Results

This report covers heldout synthetic scenario evaluation on generated Phase 2 attack bundles and the human-authored additional heldout source. It is kept separate from the canonical benchmark path.

## Scope

- Context: heldout synthetic scenario evaluation / zero-day-like evaluation
- Bundles included: `chatgpt`, `claude`, `gemini`, `grok`, `human_authored`
- Canonical benchmark winner remains `transformer @ 60s` and is not replaced by this report
- This is not real-world zero-day evidence
- `5s` heldout synthetic coverage is documented as blocked in this pass because reusable 5s replay residuals did not exist and raw 5s bundle-window generation exceeded the CPU-only wall-clock budget

## Mean Results Across Heldout Bundles

| window_label   | model_name         | canonical_or_extension   |   mean_precision |   mean_recall |   mean_f1 |   mean_false_positive_rate |   mean_latency_seconds |   bundle_count |
|:---------------|:-------------------|:-------------------------|-----------------:|--------------:|----------:|---------------------------:|-----------------------:|---------------:|
| 10s            | threshold_baseline | canonical                |           0.4146 |        0.273  |    0.2512 |                     0.1171 |               114.467  |              5 |
| 10s            | autoencoder        | canonical                |           0.0498 |        0.7804 |    0.0932 |                     0.5957 |                 1      |              5 |
| 10s            | gru                | canonical                |           0.0411 |        0.9208 |    0.0784 |                     0.8538 |                 1.05   |              5 |
| 10s            | lstm               | canonical                |           0.0407 |        0.9224 |    0.0776 |                     0.8622 |                 1.05   |              5 |
| 10s            | transformer        | canonical                |           0.0395 |        0.9645 |    0.0758 |                     0.9068 |                 1      |              5 |
| 10s            | isolation_forest   | canonical                |           0.0358 |        0.8968 |    0.0687 |                     0.9408 |                 1      |              5 |
| 60s            | transformer        | canonical                |           0.8256 |        0.5622 |    0.6613 |                     0.0045 |                 3.7    |              5 |
| 60s            | lstm               | canonical                |           0.9236 |        0.3693 |    0.5247 |                     0.0014 |                15.4    |              5 |
| 60s            | gru                | canonical                |           0.737  |        0.3835 |    0.48   |                     0.0177 |                15.1067 |              5 |
| 60s            | threshold_baseline | canonical                |           0.5031 |        0.4091 |    0.4083 |                     0.0414 |                46.4    |              5 |
| 60s            | ttm_extension      | extension                |           0.5083 |        0.2128 |    0.2832 |                     0.0135 |                 1      |              5 |
| 60s            | autoencoder        | canonical                |           0.0498 |        0.8227 |    0.0939 |                     0.6456 |                 1      |              5 |
| 60s            | isolation_forest   | canonical                |           0.0381 |        0.8929 |    0.0729 |                     0.939  |                 1      |              5 |
| 300s           | lstm               | canonical                |           0.9463 |        0.7185 |    0.8099 |                     0.0024 |                65.6667 |              5 |
| 300s           | gru                | canonical                |           0.9261 |        0.6747 |    0.7721 |                     0.0031 |                65      |              5 |
| 300s           | transformer        | canonical                |           0.8855 |        0.6637 |    0.7518 |                     0.0049 |                49.5333 |              5 |
| 300s           | threshold_baseline | canonical                |           0.7309 |        0.3812 |    0.4872 |                     0.0117 |                40      |              5 |
| 300s           | autoencoder        | canonical                |           0.0602 |        0.9407 |    0.1132 |                     0.7957 |                 1      |              5 |
| 300s           | isolation_forest   | canonical                |           0.0306 |        0.5361 |    0.0578 |                     0.9182 |               340.714  |              5 |

## Best Per Window

| window_label   | model_name         |   mean_precision |   mean_recall |   mean_f1 | canonical_or_extension   |
|:---------------|:-------------------|-----------------:|--------------:|----------:|:-------------------------|
| 5s             | not_completed      |                  |               |           | blocked                  |
| 10s            | threshold_baseline |           0.4146 |        0.273  |    0.2512 | canonical                |
| 60s            | transformer        |           0.8256 |        0.5622 |    0.6613 | canonical                |
| 300s           | lstm               |           0.9463 |        0.7185 |    0.8099 | canonical                |

## Notes

- TTM is included only where real extension evidence exists (`60s`).
- ARIMA remains a documented blocker only.
- The repo autoencoder is an MLP autoencoder, not an LSTM autoencoder.
