# Demo Usage

## Full Attacked Replay

Run the default attacked demo with the operational threshold baseline plus the optional zero-day autoencoder:

```powershell
python deployment_runtime/demo_replay.py --attacked --threshold-plus-autoencoder
```

This default replay uses the `scn_unauthorized_bess48_discharge` attack window with pre/post padding so the alert flow is easy to verify.

## Clean Replay

```powershell
python deployment_runtime/demo_replay.py --clean
```

## Scenario Override

Replay a different attacked scenario:

```powershell
python deployment_runtime/demo_replay.py `
  --attacked `
  --threshold-plus-autoencoder `
  --scenario-id scn_pv60_curtailment_inconsistency
```

## Manual Time Slice

```powershell
python deployment_runtime/demo_replay.py `
  --attacked `
  --threshold-plus-autoencoder `
  --start-time 2025-06-01T12:00:00Z `
  --end-time 2025-06-01T12:40:00Z
```

## Expected Outputs

Runtime artifacts:

- `outputs/deployment_runtime/edge/...`
- `outputs/deployment_runtime/gateway/...`
- `outputs/deployment_runtime/control_center/...`

Report exports:

- `outputs/reports/deployment_runtime/deployment_runtime_report.md`
- `outputs/reports/deployment_runtime/edge_alerts.csv`
- `outputs/reports/deployment_runtime/gateway_alert_log.csv`
- `outputs/reports/deployment_runtime/control_center_alert_summary.csv`
- `outputs/reports/deployment_runtime/latency_budget_table.csv`
