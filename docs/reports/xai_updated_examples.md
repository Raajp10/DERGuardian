# XAI Updated Qualitative Examples

## Strictly Aligned Operator Support

- Case: `gemini__autoencoder__scn_01_fdi_feeder_power_ramp`
  Target family: `false_data_injection`; predicted family: `false_data_injection`.
  Target assets: `feeder`; predicted assets: `feeder`.
  Grounding overlap: 1.000; operator-support score: 1.000.
  Summary excerpt: Runtime evidence is most consistent with false_data_injection. Strongest localized evidence points to feeder. Detector crossed threshold at 2025-06-01T01:01:01+00:00.

- Case: `gemini__gru__scn_01_fdi_feeder_power_ramp`
  Target family: `false_data_injection`; predicted family: `false_data_injection`.
  Target assets: `feeder`; predicted assets: `feeder`.
  Grounding overlap: 1.000; operator-support score: 1.000.
  Summary excerpt: Runtime evidence is most consistent with false_data_injection. Strongest localized evidence points to feeder. Detector crossed threshold at 2025-06-01T01:01:01+00:00.

## Correct Family / Weak Or Wrong Asset

- Case: `chatgpt__autoencoder__scn_cmd_delay_pv60_curtailment_release`
  Target family: `command_delay`; predicted family: `command_delay`.
  Target assets: `pv60`; predicted assets: `bess108|bess48`.
  Grounding overlap: 0.000; operator-support score: 0.600.
  Summary excerpt: Runtime evidence is most consistent with command_delay. Strongest localized evidence points to bess48, bess108. Detector crossed threshold at 2025-06-01T06:01:01+00:00.

- Case: `chatgpt__autoencoder__scn_der_disconnect_pv83_status_trip`
  Target family: `DER_disconnect`; predicted family: `DER_disconnect`.
  Target assets: `pv83`; predicted assets: `bess48`.
  Grounding overlap: 0.000; operator-support score: 0.600.
  Summary excerpt: Runtime evidence is most consistent with DER_disconnect. Strongest localized evidence points to bess48. Detector crossed threshold at 2025-06-01T13:31:01+00:00.

## Correct Family / Insufficient Grounded Evidence

- Case: `chatgpt__autoencoder__scn_cmd_suppress_bess48_discharge_hold`
  Target family: `command_suppression`; predicted family: `command_suppression`.
  Target assets: `bess48`; predicted assets: `bess108|bess48`.
  Grounding overlap: 0.250; operator-support score: 0.750.
  Summary excerpt: Runtime evidence is most consistent with command_suppression. Strongest localized evidence points to bess48, bess108. Detector crossed threshold at 2025-06-01T11:01:01+00:00.

- Case: `chatgpt__gru__scn_cmd_suppress_bess48_discharge_hold`
  Target family: `command_suppression`; predicted family: `command_suppression`.
  Target assets: `bess48`; predicted assets: `bess108|bess48`.
  Grounding overlap: 0.250; operator-support score: 0.750.
  Summary excerpt: Runtime evidence is most consistent with command_suppression. Strongest localized evidence points to bess48, bess108. Detector crossed threshold at 2025-06-01T11:01:01+00:00.

## Wrong Family / Grounded Evidence

- Case: `chatgpt__isolation_forest__scn_cmd_delay_pv60_curtailment_release`
  Target family: `command_delay`; predicted family: `command_suppression`.
  Target assets: `pv60`; predicted assets: `bess48`.
  Grounding overlap: 0.500; operator-support score: 0.200.
  Summary excerpt: Runtime evidence is most consistent with command_suppression. Strongest localized evidence points to bess48. Detector crossed threshold at 2025-06-01T06:05:01+00:00.

- Case: `chatgpt__isolation_forest__scn_der_disconnect_pv83_status_trip`
  Target family: `DER_disconnect`; predicted family: `command_suppression`.
  Target assets: `pv83`; predicted assets: `bess48`.
  Grounding overlap: 0.500; operator-support score: 0.175.
  Summary excerpt: Runtime evidence is most consistent with command_suppression. Strongest localized evidence points to bess48. Detector crossed threshold at 2025-06-01T13:35:01+00:00.

## Unsupported Claim

- Case: `chatgpt__gru__scn_coord_pv60_bess48_misaligned_support`
  Target family: `coordinated_multi_asset`; predicted family: `oscillatory_control`.
  Target assets: `bess48|pv60`; predicted assets: `bess108`.
  Grounding overlap: 0.000; operator-support score: 0.000.
  Summary excerpt: Runtime evidence is most consistent with oscillatory_control. Strongest localized evidence points to bess108. Detector crossed threshold at 2025-06-01T23:02:01+00:00.

- Case: `chatgpt__isolation_forest__scn_fdi_bus150_voltage_dropout`
  Target family: `false_data_injection`; predicted family: `coordinated_multi_asset`.
  Target assets: `bus150`; predicted assets: `bess108|bess48`.
  Grounding overlap: 0.000; operator-support score: 0.250.
  Summary excerpt: Runtime evidence is most consistent with coordinated_multi_asset. Strongest localized evidence points to bess48, bess108. Detector crossed threshold at 2025-06-01T03:35:01+00:00.

