# AOI Evaluation Report

AOI was computed only from prediction files with actual `attack_present` and `predicted` columns.
The metric is experimental and repo-specific. It does not upgrade the project to an AOI-standard benchmark claim.

## Mean AOI By Context And Model

| evaluation_context                                  | model_name         |   aoi_alert_overlap_index |
|:----------------------------------------------------|:-------------------|--------------------------:|
| ttm_extension_prediction_artifact                   | ttm_extension      |                 0.565217  |
| canonical_benchmark_prediction_artifact             | threshold_baseline |                 0.347562  |
| heldout_synthetic_zero_day_like_prediction_artifact | transformer        |                 0.328395  |
| heldout_synthetic_zero_day_like_prediction_artifact | lstm               |                 0.307647  |
| canonical_benchmark_prediction_artifact             | lstm               |                 0.295032  |
| heldout_synthetic_zero_day_like_prediction_artifact | gru                |                 0.285815  |
| canonical_benchmark_prediction_artifact             | transformer        |                 0.276585  |
| canonical_benchmark_prediction_artifact             | gru                |                 0.271412  |
| heldout_synthetic_zero_day_like_prediction_artifact | threshold_baseline |                 0.250864  |
| heldout_synthetic_zero_day_like_prediction_artifact | ttm_extension      |                 0.167997  |
| canonical_benchmark_prediction_artifact             | isolation_forest   |                 0.165351  |
| canonical_benchmark_prediction_artifact             | autoencoder        |                 0.0882545 |
| heldout_synthetic_zero_day_like_prediction_artifact | autoencoder        |                 0.056501  |
| heldout_synthetic_zero_day_like_prediction_artifact | isolation_forest   |                 0.0348418 |
| lstm_autoencoder_extension_prediction_artifact      | lstm_autoencoder   |                 0.0269577 |

## Scientific Use

AOI is useful here as an additional overlap view of alert timing and over-alerting. It should be described as an experimental Alert Overlap Index, not as a standard metric unless future work establishes that external definition.
Source CSV: `artifacts/extensions/aoi_results.csv`.
