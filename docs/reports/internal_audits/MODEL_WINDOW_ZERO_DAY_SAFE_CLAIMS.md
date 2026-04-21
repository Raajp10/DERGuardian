# Model / Window / Zero-Day Safe Claims

## Safe To Say

- The canonical benchmark still selects `transformer @ 60s`.
- Real detector benchmark coverage exists at `5s`, `10s`, `60s`, and `300s` for the implemented canonical detector families.
- The heldout synthetic evaluation now covers the implemented frozen detector packages across `10s`, `60s`, and `300s`, and it is reported separately from the canonical benchmark.
- `5s` retained benchmark coverage, but the full heldout synthetic sweep at `5s` was blocked in this pass by raw bundle-window generation cost on CPU-only hardware.
- TTM is a real extension benchmark at `60s` and an extension heldout synthetic evaluation at `60s` only.
- The autoencoder result is for the repo's MLP autoencoder implementation, not an LSTM autoencoder.

## Best Mean Heldout Synthetic Rows By Window

| window_label   | model_name         |   precision |   recall |     f1 |
|:---------------|:-------------------|------------:|---------:|-------:|
| 10s            | threshold_baseline |      0.4146 |   0.273  | 0.2512 |
| 300s           | lstm               |      0.9463 |   0.7185 | 0.8099 |
| 60s            | transformer        |      0.8256 |   0.5622 | 0.6613 |

## Not Safe To Say

- Real-world zero-day robustness.
- That the heldout synthetic evaluation replaces the canonical benchmark winner.
- That TTM is the canonical model-selection result.
- That ARIMA was benchmarked when no implementation exists.
- That an LSTM autoencoder was benchmarked when the repo only contains an MLP autoencoder.
- AOI as a detector benchmark metric here. AOI is not implemented as part of this detector-side pass.

## Precise Wording

- Use `heldout synthetic scenario evaluation` or `zero-day-like evaluation within a bounded synthetic Phase 2 scenario benchmark`.
- Use `TTM extension benchmark` and `TTM extension heldout synthetic evaluation`.
- Keep `transformer @ 60s` as `the frozen canonical benchmark winner`.
