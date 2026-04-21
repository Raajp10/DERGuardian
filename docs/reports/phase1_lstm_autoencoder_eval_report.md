# Phase 1 LSTM Autoencoder Extension Evaluation

This report documents a newly added LSTM autoencoder detector branch. It is an extension experiment only.
The frozen canonical benchmark winner remains **transformer @ 60s**; this branch does not replace or relabel that result.

## Input Contract

- Residual window inputs: `outputs/window_size_study/<window>/residual_windows.parquet`.
- Feature count: 64 high-variance residual features selected from benign canonical training windows.
- Training data: benign canonical training split only.
- Thresholding: validation-label F1 sweep when labels are available; otherwise 99th percentile of train-normal reconstruction error.
- Heldout synthetic evaluation: available replay residual inputs are scored with the trained extension model and reported separately.

## Canonical-Test-Split Extension Rows

| window_label   | status    |   precision |    recall |        f1 |   roc_auc |   false_positive_rate |   evaluated_sequences | notes                                                                                                                                                                   |
|:---------------|:----------|------------:|----------:|----------:|----------:|----------------------:|----------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 5s             | completed |  0.0152702  | 0.108333  | 0.0267673 | 0.0704751 |              0.800955 |                  3500 | LSTM autoencoder extension trained on benign canonical residual windows. It is distinct from the existing MLP autoencoder and does not alter canonical model selection. |
| 10s            | completed |  0.016661   | 0.136872  | 0.029706  | 0.0829828 |              0.920433 |                  3500 | LSTM autoencoder extension trained on benign canonical residual windows. It is distinct from the existing MLP autoencoder and does not alter canonical model selection. |
| 60s            | completed |  0.00851064 | 0.0533333 | 0.0146789 | 0.0722609 |              0.690882 |                  1499 | LSTM autoencoder extension trained on benign canonical residual windows. It is distinct from the existing MLP autoencoder and does not alter canonical model selection. |
| 300s           | completed |  0.08       | 0.411765  | 0.133971  | 0.398472  |              0.572954 |                   315 | LSTM autoencoder extension trained on benign canonical residual windows. It is distinct from the existing MLP autoencoder and does not alter canonical model selection. |

## Heldout Synthetic Rows

| window_label   | status    |   precision |     recall |         f1 |   evaluated_sequences | notes                                                                                                                                                   |
|:---------------|:----------|------------:|-----------:|-----------:|----------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------|
| 10s            | completed |   0.0815217 | 0.243902   | 0.1222     |                  3500 | Heldout synthetic replay residuals from chatgpt__existing_heldout_phase2_bundle__phase2_zero_day_like_bundle_2026_04_15_c; extension evaluation only.   |
| 10s            | completed |   0.285714  | 0.222222   | 0.25       |                  3500 | Heldout synthetic replay residuals from claude__existing_heldout_phase2_bundle__claude_phase2_llm_benchmark_v2; extension evaluation only.              |
| 10s            | completed |   0.0014245 | 0.00699301 | 0.00236686 |                  3500 | Heldout synthetic replay residuals from gemini__existing_heldout_phase2_bundle__phase2_llm_synthetic_benchmark_002; extension evaluation only.          |
| 10s            | completed |   0.0290698 | 0.25       | 0.0520833  |                  3500 | Heldout synthetic replay residuals from grok__existing_heldout_phase2_bundle__phase2_llm_benchmark_grok_synthetic_03; extension evaluation only.        |
| 10s            | completed |   0         | 0          | 0          |                  3500 | Heldout synthetic replay residuals from human_authored__human_authored_heldout__human_authored_phase3_bundle_v1; extension evaluation only.             |
| 60s            | completed |   0         | 0          | 0          |                  3500 | Heldout synthetic replay residuals from chatgpt__existing_heldout_phase2_bundle__phase2_zero_day_like_bundle_2026_04_15_c; extension evaluation only.   |
| 60s            | completed |   0         | 0          | 0          |                  3500 | Heldout synthetic replay residuals from chatgpt__repaired_heldout_bundle__phase2_zero_day_like_bundle_2026_04_15_c_repaired; extension evaluation only. |
| 60s            | completed |   0         | 0          | 0          |                  3500 | Heldout synthetic replay residuals from claude__existing_heldout_phase2_bundle__claude_phase2_llm_benchmark_v2; extension evaluation only.              |
| 60s            | completed |   0         | 0          | 0          |                  3500 | Heldout synthetic replay residuals from claude__repaired_heldout_bundle__claude_phase2_llm_benchmark_v2_repaired; extension evaluation only.            |
| 60s            | completed |   0         | 0          | 0          |                  3500 | Heldout synthetic replay residuals from gemini__existing_heldout_phase2_bundle__phase2_llm_synthetic_benchmark_002; extension evaluation only.          |
| 60s            | completed |   0         | 0          | 0          |                  3500 | Heldout synthetic replay residuals from gemini__repaired_heldout_bundle__phase2_llm_synthetic_benchmark_002_repaired; extension evaluation only.        |
| 300s           | completed |   0         | 0          | 0          |                  1432 | Heldout synthetic replay residuals from chatgpt__existing_heldout_phase2_bundle__phase2_zero_day_like_bundle_2026_04_15_c; extension evaluation only.   |
| 300s           | completed |   0         | 0          | 0          |                  1432 | Heldout synthetic replay residuals from claude__existing_heldout_phase2_bundle__claude_phase2_llm_benchmark_v2; extension evaluation only.              |
| 300s           | completed |   0         | 0          | 0          |                  1432 | Heldout synthetic replay residuals from gemini__existing_heldout_phase2_bundle__phase2_llm_synthetic_benchmark_002; extension evaluation only.          |
| 300s           | completed |   0.5       | 0.107143   | 0.176471   |                  1432 | Heldout synthetic replay residuals from grok__existing_heldout_phase2_bundle__phase2_llm_benchmark_grok_synthetic_03; extension evaluation only.        |
| 300s           | completed |   0.0425532 | 0.208955   | 0.0707071  |                  1432 | Heldout synthetic replay residuals from human_authored__human_authored_heldout__human_authored_phase3_bundle_v1; extension evaluation only.             |

## Decision

Best extension row: 300s with F1=0.1340, precision=0.0800, recall=0.4118.
These results are useful evidence for a real LSTM autoencoder branch, but they are not canonical benchmark selection evidence.
Source CSV: `artifacts/extensions/phase1_lstm_autoencoder_results.csv`.
