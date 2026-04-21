# Complete Results And Discussion

## Canonical benchmark path

- The canonical benchmark winner remains `transformer @ 60s` with benchmark F1 `0.818182`.
- This is still the only source-of-truth model-selection result for the paper-safe benchmark path.
- Family-best benchmark rows across model families are shown below for context, but they do not overwrite the canonical winner.

| model_name         | window_label   |   precision |   recall |       f1 |   average_precision |   inference_time_ms_per_prediction |
|:-------------------|:---------------|------------:|---------:|---------:|--------------------:|-----------------------------------:|
| transformer        | 60s            |   0.692308  | 1        | 0.818182 |            0.90684  |                         0.0218714  |
| gru                | 60s            |   0.775     | 0.861111 | 0.815789 |            0.794976 |                         0.0303035  |
| lstm               | 60s            |   0.75      | 0.833333 | 0.789474 |            0.785662 |                         0.0178594  |
| llm_baseline       | 60s            |   0.756757  | 0.777778 | 0.767123 |            0.811201 |                         0.0213388  |
| threshold_baseline | 10s            |   0.966912  | 0.568649 | 0.716133 |            0.674127 |                         0.00154052 |
| isolation_forest   | 60s            |   0.22272   | 0.943038 | 0.360339 |            0.514285 |                         0.0238809  |
| autoencoder        | 300s           |   0.0986842 | 0.833333 | 0.176471 |            0.490481 |                         0.00696289 |

## Heldout replay path

- Mean original heldout replay precision / recall / F1 across ChatGPT, Claude, Gemini, and Grok: `0.886` / `0.506` / `0.622`.
- Mean repaired replay precision / recall / F1 across repaired bundles: `0.879` / `0.562` / `0.680`.
- Replay remains a separate evaluation context from benchmark model selection.

| model_name         | window_label   |   precision |   recall |       f1 |   mean_latency_seconds |
|:-------------------|:---------------|------------:|---------:|---------:|-----------------------:|
| lstm               | 300s           |    0.955041 | 0.745121 | 0.829321 |                64.3333 |
| transformer        | 60s            |    0.825166 | 0.570631 | 0.665354 |                 4.375  |
| threshold_baseline | 10s            |    0.27989  | 0.26577  | 0.199412 |               140.333  |

## Phase 2 scientific coverage and diversity

- Inventory rows audited: `99`
- Validated rows used for final usable coverage: `88`
- Rejected rows retained for auditability: `11`
- Repair-tagged rows: `11`
- Distinct validated families / assets / signals: `8` / `18` / `29`
- Difficulty bucket counts: easy=`25`, moderate=`26`, hard=`24`, very hard=`24`
- Under-covered families: `['replay', 'telemetry_corruption']`
- Under-covered assets (<=1 validated scenario): `['bus48', 'creg3a', 'creg3c', 'sw4']`
- Under-covered signals (<=1 validated scenario): `['bess_bess48_terminal_i_a', 'bus_48_v_pu_phase_a', 'feeder_q_kvar_total', 'pv_pv35_q_kvar', 'pv_pv35_terminal_v_pu', 'pv_pv60_terminal_v_pu', 'pv_pv83_q_kvar', 'pv_pv83_status_cmd', 'regulator_creg3a_vreg', 'regulator_creg3c_vreg']`

## Extension experiments

- TinyTimeMixer extension: `F1=0.722`, `precision=0.634`, `recall=0.839`, `latency=0.872 ms/window`, `params=8205`.
- TTM used the canonical 60 s residual feature contract but required local random initialization because no compatible public pre-trained 12->1 checkpoint matched the repo's short-sequence setup.
- LoRA explanation branch heldout test: `family_accuracy=0.234`, `asset_accuracy=0.000`, `grounding=0.000`, `latency=3507.0 ms/example`.
- Both extension branches are real evidence, but neither branch replaces the canonical detector path.

## XAI validation

- Heldout family accuracy: `0.774`
- Heldout asset accuracy: `0.268`
- Heldout grounding rate: `0.258`
- Heldout action relevance: `0.758`
- Unsupported claim rate: `0.052`
- The XAI layer is therefore useful for grounded operator support, but not strong enough for human-like root-cause-analysis claims.

## Deployment benchmark

| model_name         | window_label   |   mean_cpu_inference_ms_per_window |   throughput_windows_per_sec |   rss_peak_mb |   replay_f1 |
|:-------------------|:---------------|-----------------------------------:|-----------------------------:|--------------:|------------:|
| threshold_baseline | 10s            |                           0.005298 |                   188754     |      2882.86  |    0        |
| lstm               | 300s           |                           0.696571 |                     1435.6   |       367.809 |    0.94     |
| transformer        | 60s            |                           1.63067  |                      613.244 |       521.578 |    0.884354 |

- `threshold_baseline @ 10s replay_f1=0.000000`
- `lstm @ 300s replay_f1=0.940000`
- `transformer @ 60s replay_f1=0.884354`

## Discussion

- The repository now has a stronger evidence trail across benchmarking, heldout replay, Phase 2 diversity, extension experiments, XAI auditing, and offline deployment measurement.
- The following claims remain unsupported and should stay out of tomorrow's paper/talk wording: human-like root-cause analysis, real-world zero-day robustness, true edge deployment, and AOI as an implemented detection metric.

## Detector Window And Zero-Day-Like Addendum

The canonical benchmark winner remains `transformer @ 60s` with benchmark F1 `0.8182`.

In the new heldout synthetic zero-day-like matrix, the strongest mean row was `lstm @ 300s` with mean F1 `0.8099`, reported separately from the canonical benchmark.

`5s` heldout synthetic coverage remains blocked in this pass because raw full-day 5s bundle-window generation exceeded the CPU-only wall-clock budget and no reusable 5s replay residuals existed.

This addendum keeps the contexts separate: canonical benchmark, existing replay, heldout synthetic zero-day-like evaluation, and the TTM extension branch.
AOI is not claimed as part of this detector-side benchmark pass.
