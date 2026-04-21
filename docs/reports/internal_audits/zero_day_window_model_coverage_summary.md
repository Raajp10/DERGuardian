# Zero-Day-Like Window / Model Coverage Summary

This audit reflects the pre-existing repository state before the missing heldout synthetic detector matrix was added in this pass.

## Existing Heldout Synthetic Coverage Counts

|   window_seconds | model_name         |   bundle_count_with_existing_eval |
|-----------------:|:-------------------|----------------------------------:|
|                5 | arima              |                                 0 |
|                5 | autoencoder        |                                 0 |
|                5 | gru                |                                 0 |
|                5 | isolation_forest   |                                 0 |
|                5 | lstm               |                                 0 |
|                5 | lstm_ae            |                                 0 |
|                5 | threshold_baseline |                                 0 |
|                5 | transformer        |                                 0 |
|                5 | ttm_extension      |                                 0 |
|               10 | arima              |                                 0 |
|               10 | autoencoder        |                                 0 |
|               10 | gru                |                                 0 |
|               10 | isolation_forest   |                                 0 |
|               10 | lstm               |                                 0 |
|               10 | lstm_ae            |                                 0 |
|               10 | threshold_baseline |                                 5 |
|               10 | transformer        |                                 0 |
|               10 | ttm_extension      |                                 0 |
|               60 | arima              |                                 0 |
|               60 | autoencoder        |                                 0 |
|               60 | gru                |                                 0 |
|               60 | isolation_forest   |                                 0 |
|               60 | lstm               |                                 0 |
|               60 | lstm_ae            |                                 0 |
|               60 | threshold_baseline |                                 0 |
|               60 | transformer        |                                 5 |
|               60 | ttm_extension      |                                 0 |
|              300 | arima              |                                 0 |
|              300 | autoencoder        |                                 0 |
|              300 | gru                |                                 0 |
|              300 | isolation_forest   |                                 0 |
|              300 | lstm               |                                 5 |
|              300 | lstm_ae            |                                 0 |
|              300 | threshold_baseline |                                 0 |
|              300 | transformer        |                                 0 |
|              300 | ttm_extension      |                                 0 |

## What Existed Before Completion

- `transformer @ 60s` had existing heldout synthetic replay coverage in both `phase3_heldout` and the later improved replay package.
- `threshold_baseline @ 10s` and `lstm @ 300s` had existing improved replay coverage only.
- No existing `5s` heldout synthetic detector matrix existed.
- No existing TTM heldout synthetic evaluation existed.
- No ARIMA heldout synthetic evaluation existed because ARIMA is not implemented.

## Context Separation

- Existing `phase3_heldout` rows are frozen-model replay artifacts for `transformer @ 60s`.
- Existing `improved_phase3` rows are replay artifacts for three frozen candidates only.
- These existing rows were not a full benchmark and were not a full zero-day-like window/model sweep.
