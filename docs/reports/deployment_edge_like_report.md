# Constrained Edge-Like Deployment Benchmark Report

This report strengthens the deployment evidence as an offline, constrained profile only.
It does not claim field edge deployment or hardware-in-the-loop validation.

| model_name         | window_label   |   mean_cpu_inference_ms_per_window |   throughput_windows_per_sec |   rss_peak_mb |   model_package_size_mb |   replay_f1 | edge_like_constraint                                                                       |
|:-------------------|:---------------|-----------------------------------:|-----------------------------:|--------------:|------------------------:|------------:|:-------------------------------------------------------------------------------------------|
| threshold_baseline | 10s            |                          0.004893  |                   204392     |      2621.3   |                 0.246   |    0        | single-thread offline profile from existing deployment benchmark                           |
| transformer        | 60s            |                          1.63853   |                      610.302 |       510.18  |                 0.333   |    0.884354 | single-thread offline profile from existing deployment benchmark                           |
| lstm               | 300s           |                          0.719595  |                     1389.67  |       364.664 |                 0.122   |    0.94     | single-thread offline profile from existing deployment benchmark                           |
| lstm_autoencoder   | 300s           |                          0.0463838 |                    21559.2   |       374.785 |                 0.18361 |    0.133971 | single-thread CPU timing of extension checkpoint on canonical 300s test residual sequences |

## Safe Wording

Use: **constrained edge-like offline benchmark** or **offline lightweight deployment benchmark**.
Do not use: **field edge deployment**.

Source CSV: `artifacts/deployment/deployment_edge_like_results.csv`.
