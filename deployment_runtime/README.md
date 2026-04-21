# Deployment Runtime Prototype

This package turns the paper deployment vision into a runnable lightweight prototype for the current DER cyber-physical project. It is intentionally defensive and local-first: edge replay and scoring, gateway fusion, control-center reporting, and an optional adapter into the existing explanation layer.

## What Is Implemented

- `edge_runtime.py`: replays telemetry from existing parquet or CSV artifacts, emits rolling windows, computes residual-style features against clean reference windows, scores `threshold_baseline`, optionally scores the zero-day `autoencoder`, and writes alert packets plus local context.
- `gateway_runtime.py`: reads edge packets, attaches site metadata, suppresses repeats, and fuses nearby model alerts into a gateway-forwarded packet.
- `control_center_runtime.py`: collects forwarded packets, writes a central alert table and summary, and attaches explanation outputs by first trying the full upstream packet builder and then falling back to a bridge into the existing Phase 3 explanation logic.
- `demo_replay.py`: runs the full path end to end.
- `latency_budget.py`: summarizes measured edge, gateway, and control-center latency from runtime traces.
- `export_deployment_report.py`: exports report-ready outputs under `outputs/reports/deployment_runtime/`.

## What Remains Conceptual

- Live transport security, device authentication, and production message bus integration.
- Cloud orchestration, fleet-scale persistence, and operator ticketing workflow integration.
- Any automatic grid-control action. This prototype is reporting-only by design.

## Deployment Architecture

1. DER site / edge ingests measured physical telemetry plus cyber events and emits rolling windows.
2. The edge compares each emitted window against a clean reference window store so the deployed models can reuse the project's aligned residual feature mode.
3. The site gateway fuses or deduplicates repeated alerts and adds site metadata.
4. The control center writes the central alert table and explanation artifacts for operator-facing review.

## Supported Models

- `threshold_baseline`
  - Default edge model.
  - Uses `outputs/models_full_run/threshold_baseline`.
  - Runs on `300s / 60s` windows.
  - Chosen because it is the strongest operational same-scenario detector and is cheap to score on edge.
- `autoencoder`
  - Optional secondary edge model.
  - Uses `outputs/phase3_zero_day/latency_window_sweep/w120_s24/autoencoder`.
  - Runs on `120s / 24s` windows.
  - Chosen because the autoencoder family is the strongest current zero-day detector in the project's Phase 3 aggregate summary.
- Heavier GRU, LSTM, and transformer models are intentionally not deployed by default.

## Running The Edge Runtime

Threshold only:

```powershell
python deployment_runtime/edge_runtime.py `
  --dataset-kind attacked `
  --measured outputs/attacked/measured_physical_timeseries.parquet `
  --cyber outputs/attacked/cyber_events.parquet `
  --start-time 2025-06-01T01:45:00Z `
  --end-time 2025-06-01T02:20:00Z
```

Threshold plus autoencoder:

```powershell
python deployment_runtime/edge_runtime.py `
  --threshold-plus-autoencoder `
  --dataset-kind attacked `
  --measured outputs/attacked/measured_physical_timeseries.parquet `
  --cyber outputs/attacked/cyber_events.parquet `
  --start-time 2025-06-01T01:45:00Z `
  --end-time 2025-06-01T02:20:00Z
```

## Running The Full Demo Replay

Attacked replay with both lightweight detectors:

```powershell
python deployment_runtime/demo_replay.py --attacked --threshold-plus-autoencoder
```

Clean replay:

```powershell
python deployment_runtime/demo_replay.py --clean
```

## Alert Packet Contract

The packet is intentionally small and edge-friendly. Core fields include:

- `alert_id`
- `timestamp_utc`
- `site_id`
- `packet_source`
- `model_name`
- `model_profile`
- `score`
- `threshold`
- `severity`
- `modality`
- `asset_id`
- `affected_assets`
- `top_features`
- `window_start_utc`
- `window_end_utc`
- `local_context_path`
- `explanation_packet_path`

See `alert_packet_schema.json` and `docs/deployment_contract.md`.

## Outputs

Runtime outputs are written under `outputs/deployment_runtime/`.

Report exports are written under `outputs/reports/deployment_runtime/`:

- `deployment_runtime_report.md`
- `edge_alerts.csv`
- `gateway_alert_log.csv`
- `control_center_alert_summary.csv`
- `latency_budget_table.csv`

## Current Limitations

- The clean reference store is replayed from project artifacts rather than a live historian or baseline profile service.
- Explanation integration prefers matching upstream artifacts and otherwise uses a bridge packet into the Phase 3 explanation logic, so it remains best-effort rather than a production investigation service.
- Gateway fusion is intentionally lightweight and designed for thesis support, not production SOC workflow enforcement.
