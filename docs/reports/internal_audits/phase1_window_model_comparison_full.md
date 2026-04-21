# Full Detector Window / Model Comparison

This table is derived from real saved benchmark artifacts. It does not overwrite the frozen canonical selection file.

## Important Implementation Notes

- The canonical winner remains `transformer @ 60s`.
- The repo does not contain a real ARIMA detector implementation.
- The repo does not contain a real LSTM autoencoder detector. The available `autoencoder` is an MLP autoencoder and is reported under its actual implementation name.
- TinyTimeMixer is extension-only and is kept separate from the canonical benchmark path.

## Best Per Window

| window_label   |   window_seconds | canonical_best_model   |   canonical_best_precision |   canonical_best_recall |   canonical_best_f1 | best_model_including_extensions   |   best_f1_including_extensions | best_including_extensions_type   |
|:---------------|-----------------:|:-----------------------|---------------------------:|------------------------:|--------------------:|:----------------------------------|-------------------------------:|:---------------------------------|
| 5s             |                5 | isolation_forest       |                     0.1208 |                  0.9508 |              0.2144 | isolation_forest                  |                         0.2144 | canonical                        |
| 10s            |               10 | threshold_baseline     |                     0.9669 |                  0.5686 |              0.7161 | threshold_baseline                |                         0.7161 | canonical                        |
| 60s            |               60 | transformer            |                     0.6923 |                  1      |              0.8182 | transformer                       |                         0.8182 | canonical                        |
| 300s           |              300 | lstm                   |                     0.3333 |                  1      |              0.5    | lstm                              |                         0.5    | canonical                        |

## Full Comparison Table

| window_label   | model_name         | canonical_or_extension   | status          |   precision |   recall |     f1 |   average_precision |   roc_auc |   false_positive_rate |   mean_detection_latency_seconds |
|:---------------|:-------------------|:-------------------------|:----------------|------------:|---------:|-------:|--------------------:|----------:|----------------------:|---------------------------------:|
| 5s             | lstm_ae            | blocked                  | not_implemented |             |          |        |                     |           |                       |                                  |
| 5s             | transformer        | canonical                | completed       |      0.1013 |   0.9994 | 0.1839 |              0.2814 |    0.2756 |                1      |                          336.538 |
| 5s             | gru                | canonical                | completed       |      0.1013 |   0.9994 | 0.1839 |              0.4307 |    0.4698 |                0.9998 |                          336.538 |
| 5s             | autoencoder        | canonical                | completed       |      0.0863 |   0.8194 | 0.1561 |              0.434  |    0.3907 |                0.9993 |                          357.75  |
| 5s             | isolation_forest   | canonical                | completed       |      0.1208 |   0.9508 | 0.2144 |              0.6173 |    0.8671 |                0.7966 |                          343.5   |
| 5s             | threshold_baseline | canonical                | completed       |      0.0601 |   0.4938 | 0.1072 |              0.3731 |    0.4104 |                0.8883 |                          432.4   |
| 5s             | lstm               | canonical                | completed       |      0.1011 |   0.9972 | 0.1835 |              0.3489 |    0.3964 |                0.9999 |                          336.538 |
| 5s             | ttm_extension      | extension                | not_run         |             |          |        |                     |           |                       |                                  |
| 10s            | lstm_ae            | blocked                  | not_implemented |             |          |        |                     |           |                       |                                  |
| 10s            | transformer        | canonical                | completed       |      0.0986 |   0.9921 | 0.1793 |              0.4024 |    0.4441 |                0.9999 |                          343.462 |
| 10s            | gru                | canonical                | completed       |      0.0959 |   0.9616 | 0.1743 |              0.2457 |    0.3066 |                0.9996 |                          343.462 |
| 10s            | autoencoder        | canonical                | completed       |      0.0861 |   0.8184 | 0.1559 |              0.2925 |    0.2453 |                0.9986 |                          358     |
| 10s            | isolation_forest   | canonical                | completed       |      0.2029 |   0.9514 | 0.3345 |              0.6121 |    0.8712 |                0.4299 |                          347.333 |
| 10s            | threshold_baseline | canonical                | completed       |      0.9669 |   0.5686 | 0.7161 |              0.6741 |    0.6614 |                0.0022 |                          416.143 |
| 10s            | lstm               | canonical                | completed       |      0.0936 |   0.9952 | 0.1712 |              0.4058 |    0.4397 |                1      |                          351.462 |
| 10s            | ttm_extension      | extension                | not_run         |             |          |        |                     |           |                       |                                  |
| 60s            | lstm_ae            | blocked                  | not_implemented |             |          |        |                     |           |                       |                                  |
| 60s            | transformer        | canonical                | completed       |      0.6923 |   1      | 0.8182 |              0.9068 |    0.9974 |                0.0121 |                          641     |
| 60s            | gru                | canonical                | completed       |      0.775  |   0.8611 | 0.8158 |              0.795  |    0.9955 |                0.0068 |                          641     |
| 60s            | autoencoder        | canonical                | completed       |      0.0887 |   0.8228 | 0.1602 |              0.4025 |    0.3852 |                0.9874 |                          402     |
| 60s            | isolation_forest   | canonical                | completed       |      0.2227 |   0.943  | 0.3603 |              0.5143 |    0.8593 |                0.3846 |                          384     |
| 60s            | threshold_baseline | canonical                | completed       |      0.7899 |   0.5949 | 0.6787 |              0.6067 |    0.6077 |                0.0185 |                          509     |
| 60s            | lstm               | canonical                | completed       |      0.75   |   0.8333 | 0.7895 |              0.7857 |    0.9947 |                0.0076 |                          641     |
| 60s            | ttm_extension      | extension                | completed       |      0.6341 |   0.8387 | 0.7222 |              0.6785 |    0.9939 |                       |                          706.6   |
| 300s           | lstm_ae            | blocked                  | not_implemented |             |          |        |                     |           |                       |                                  |
| 300s           | transformer        | canonical                | completed       |      0.2308 |   0.75   | 0.3529 |              0.2085 |    0.9669 |                0.0368 |                          871     |
| 300s           | gru                | canonical                | completed       |      0.2143 |   0.75   | 0.3333 |              0.4542 |    0.9779 |                0.0404 |                          871     |
| 300s           | autoencoder        | canonical                | completed       |      0.0987 |   0.8333 | 0.1765 |              0.4905 |    0.6683 |                0.9716 |                          571     |
| 300s           | isolation_forest   | canonical                | completed       |      0.1226 |   0.8889 | 0.2155 |              0.2579 |    0.6726 |                0.8121 |                          556     |
| 300s           | threshold_baseline | canonical                | completed       |      0.2872 |   0.75   | 0.4154 |              0.6836 |    0.7571 |                0.2376 |                          614.333 |
| 300s           | lstm               | canonical                | completed       |      0.3333 |   1      | 0.5    |              0.5265 |    0.9835 |                0.0294 |                          871     |
| 300s           | ttm_extension      | extension                | not_run         |             |          |        |                     |           |                       |                                  |
