# Detector Window / Model Coverage Summary

This audit reflects the repository state before this completion pass filled the missing detector-side heldout synthetic matrix.

## Verified Existing Canonical Benchmark Coverage

|   window_seconds |   real_model_rows |
|-----------------:|------------------:|
|                5 |                 6 |
|               10 |                 6 |
|               60 |                 6 |
|              300 |                 6 |

Real canonical benchmark coverage already existed for windows `5s`, `10s`, `60s`, and `300s` across the implemented detector models:

- `threshold_baseline`
- `isolation_forest`
- `autoencoder` (repo implementation is MLP autoencoder; no LSTM autoencoder was found)
- `gru`
- `lstm`
- `transformer`

The canonical benchmark winner remains `transformer @ 60s` from `outputs/window_size_study/final_window_comparison.csv`.

## Existing Replay / Zero-Day-Like Coverage Before Completion

- Existing saved-package replay in `phase3_heldout` covered only `transformer @ 60s` across the canonical bundle plus the original heldout generator bundles. Rows found: `5`.
- Existing improved replay in `improved_phase3` covered only three frozen candidates: `threshold_baseline @ 10s`, `transformer @ 60s`, and `lstm @ 300s`. Rows found: `18`.
- Existing TTM extension evidence covered only `60s`. Verified rows: `1` / `4`.

## Explicit Gaps Found

- No ARIMA detector benchmark implementation exists in the repo.
- No real LSTM autoencoder detector exists; the available autoencoder is an MLP autoencoder.
- No pre-existing replay / zero-day-like matrix covered all required windows and all detector models.
- No pre-existing TTM detector comparison existed beyond the single `60s` extension run.
