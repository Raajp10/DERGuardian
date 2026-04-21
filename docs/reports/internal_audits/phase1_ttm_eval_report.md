# Phase 1 TinyTimeMixer Extension Evaluation

## Scope
- This branch is an extension benchmark only.
- The frozen canonical benchmark winner remains `transformer @ 60s` from `outputs/window_size_study/final_window_comparison.csv`.
- The TTM result below must not be relabeled as the canonical benchmark winner.

## Implementation Reality Check
- Architecture: `TinyTimeMixerForPrediction` from IBM Granite TSFM / `tsfm_public`.
- Data contract: canonical `60s` aligned residual windows using the frozen transformer feature subset (`64` features).
- Training mode: benign-only forecasting on train split, validation-threshold calibration on heldout validation windows, test metrics on heldout test windows.
- Context / forecast horizon: `12` input windows -> `1` predicted window.
- A compatible public pre-trained TTM checkpoint for this short residual-window contract was not available, so the architecture was trained locally from random initialization.

## Split Summary
- Train benign forecasting sequences: `3830`
- Validation sequences: `1286` (`8` positive)
- Test sequences: `1349` (`31` positive)

## TTM Extension Result
- Precision: `0.634146`
- Recall: `0.838710`
- F1: `0.722222`
- PR-AUC: `0.678520`
- ROC-AUC: `0.993930`
- Threshold after validation calibration: `0.419065`
- Parameter count: `8205`
- Training time: `708.87` s
- Inference latency: `0.871536` ms / prediction
- Mean detected attack latency: `706.600` s

## Comparison Snapshot
| model_name         | window_label   |   precision |   recall |       f1 |   average_precision |
|:-------------------|:---------------|------------:|---------:|---------:|--------------------:|
| transformer        | 60s            |    0.692308 | 1        | 0.818182 |            0.90684  |
| gru                | 60s            |    0.775    | 0.861111 | 0.815789 |            0.794976 |
| lstm               | 60s            |    0.75     | 0.833333 | 0.789474 |            0.785662 |
| llm_baseline       | 60s            |    0.756757 | 0.777778 | 0.767123 |            0.811201 |
| threshold_baseline | 10s            |    0.966912 | 0.568649 | 0.716133 |            0.674127 |
| isolation_forest   | 60s            |    0.22272  | 0.943038 | 0.360339 |            0.514285 |

## Interpretation
- This is a real TTM run, not a placeholder manifest.
- It should be read as a secondary architectural comparison on the canonical residual input contract.
- Because no repo-native public TTM checkpoint matched the short `12 -> 1` residual-window setup, the run used a randomly initialized tiny TTM and trained it locally.
- If this branch underperforms the frozen canonical detector, that is evidence, not failure to report.

## Training History
|   epoch |   train_loss |   val_loss |
|--------:|-------------:|-----------:|
|       1 |     0.306946 |   0.197888 |
|       2 |     0.268531 |   0.183052 |
|       3 |     0.258783 |   0.178497 |
|       4 |     0.256977 |   0.177547 |
|       5 |     0.255425 |   0.177625 |
|       6 |     0.255896 |   0.177009 |
|       7 |     0.254001 |   0.177112 |
|       8 |     0.256367 |   0.1769   |
|       9 |     0.252157 |   0.176642 |
|      10 |     0.254561 |   0.17687  |
|      11 |     0.252008 |   0.177087 |
|      12 |     0.248907 |   0.176549 |
|      13 |     0.250455 |   0.176631 |
|      14 |     0.248278 |   0.175138 |
|      15 |     0.249159 |   0.175977 |
|      16 |     0.247803 |   0.174429 |
|      17 |     0.247069 |   0.174548 |
|      18 |     0.245299 |   0.173515 |
|      19 |     0.246228 |   0.173194 |
|      20 |     0.24427  |   0.173241 |
|      21 |     0.242503 |   0.172587 |
|      22 |     0.242636 |   0.172934 |
|      23 |     0.244907 |   0.173328 |
|      24 |     0.243668 |   0.172602 |

## Output Files
- Results: `phase1_ttm_results.csv`
- Comparison table: `phase1_ttm_vs_all_models.csv`
- Figures: `phase1_model_comparison_extended.png`, `phase1_accuracy_latency_tradeoff_extended.png`
- Raw artifacts: `<repo>/outputs\phase1_ttm_extension`