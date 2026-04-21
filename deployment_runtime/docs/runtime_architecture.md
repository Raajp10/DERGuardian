# Runtime Architecture

## Edge To Gateway To Control Center

The prototype implements the deployment split shown in the paper figure as a runnable local package:

1. Edge runtime
   - Replays or ingests telemetry rows.
   - Builds rolling windows incrementally.
   - Computes residual-compatible features against clean reference windows.
   - Scores the operational `threshold_baseline`.
   - Optionally scores the lightweight zero-day `autoencoder`.
   - Emits compact alert packets plus local context references.
2. Site gateway
   - Reads edge packets from one or more runtimes.
   - Adds site metadata.
   - Suppresses repeated alerts from the same source window.
   - Fuses nearby multi-model alerts into a single forwarded packet.
3. Control center
   - Collects forwarded packets.
   - Writes the central alert table and fleet-ready summary tables.
   - Optionally attaches an upstream explanation packet when the replayed window matches the Phase 3 explanation artifacts.

## Why Explanation Remains Upstream

The explanation layer is kept operator-facing and upstream for two reasons:

- It is heavier than the edge detection path and does not need to sit inline with site telemetry scoring.
- The current project already has a grounded explanation packet builder, so the deployment prototype can reference that layer without embedding a new large-model dependency inside the edge path.

## Mapping To The Paper Figure

- DER site / edge in the figure maps to `edge_runtime.py`, `stream_window_builder.py`, `local_buffer.py`, and `load_deployed_models.py`.
- Site gateway maps to `gateway_runtime.py`.
- Control center maps to `control_center_runtime.py`.
- Explanation and reporting map to the optional explanation adapter plus `export_deployment_report.py`.

## Model Placement Rationale

- `threshold_baseline` stays at edge by default because it is cheap, deterministic, and the strongest same-scenario operational detector in the current project outputs.
- `autoencoder` stays optional at edge because it is still lightweight enough to run locally and is the strongest current zero-day detector family in Phase 3, but it should remain secondary to the operational baseline.
- Heavier sequence models are intentionally not the edge default because they add complexity and latency that the current prototype does not need to demonstrate the deployment vision.
