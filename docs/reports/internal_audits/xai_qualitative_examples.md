# XAI Qualitative Examples

## fully aligned

- generator: `gemini`
- detector: `autoencoder`
- scenario_id: `scn_01_fdi_feeder_power_ramp`
- target family -> predicted family: `false_data_injection` -> `false_data_injection`
- target assets -> predicted assets: `feeder` -> `feeder`
- grounding overlap: 1.000
- operator summary: Runtime evidence is most consistent with false_data_injection. Strongest localized evidence points to feeder. Detector crossed threshold at 2025-06-01T01:01:01+00:00.
- actions excerpt: Cross-check feeder_p_kw_total against historian or device-side telemetry before changing control settings. | Hold operator action on feeder until the measurement path is verified.
- uncertainty note: Constraints: Only coarse feeder-level evidence was visible before v4 localization.

## correct family / wrong asset

- generator: `grok`
- detector: `autoencoder`
- scenario_id: `scn_oscillatory_control_capacitor_c83_state`
- target family -> predicted family: `oscillatory_control` -> `oscillatory_control`
- target assets -> predicted assets: `c83` -> `feeder`
- grounding overlap: 0.500
- operator summary: Runtime evidence is most consistent with oscillatory_control. Strongest localized evidence points to feeder. Detector crossed threshold at 2025-06-01T15:01:01+00:00.
- actions excerpt: Inspect repeated controller toggles and control-loop behavior around feeder. | Check whether feeder is cycling between competing setpoints.
- uncertainty note: Constraints: Only coarse feeder-level evidence was visible before v4 localization.

## correct family / partial evidence

- generator: `gemini`
- detector: `isolation_forest`
- scenario_id: `scn_03_suppress_bess48_target`
- target family -> predicted family: `command_suppression` -> `command_suppression`
- target assets -> predicted assets: `bess48` -> `bess48`
- grounding overlap: 0.333
- operator summary: Runtime evidence is most consistent with command_suppression. Strongest localized evidence points to bess48. Detector crossed threshold at 2025-06-01T06:05:01+00:00.
- actions excerpt: Verify write acknowledgements and local controller response on bess48. | Check whether recent writes were blocked, frozen, or left unapplied.
- uncertainty note: Top runtime family candidates are close (command_suppression vs oscillatory_control).

## wrong family / grounded evidence

- generator: `grok`
- detector: `isolation_forest`
- scenario_id: `scn_command_delay_bess48_target_kw`
- target family -> predicted family: `command_delay` -> `command_suppression`
- target assets -> predicted assets: `bess48` -> `bess48`
- grounding overlap: 1.000
- operator summary: Runtime evidence is most consistent with command_suppression. Strongest localized evidence points to bess48. Detector crossed threshold at 2025-06-01T02:25:01+00:00.
- actions excerpt: Verify write acknowledgements and local controller response on bess48. | Check whether recent writes were blocked, frozen, or left unapplied.
- uncertainty note: Top runtime family candidates are close (command_suppression vs oscillatory_control).

## unsupported claim

- generator: `chatgpt`
- detector: `gru`
- scenario_id: `scn_coord_pv60_bess48_misaligned_support`
- target family -> predicted family: `coordinated_multi_asset` -> `oscillatory_control`
- target assets -> predicted assets: `bess48|pv60` -> `bess108`
- grounding overlap: 0.000
- operator summary: Runtime evidence is most consistent with oscillatory_control. Strongest localized evidence points to bess108. Detector crossed threshold at 2025-06-01T23:02:01+00:00.
- actions excerpt: Inspect repeated controller toggles and control-loop behavior around bess108. | Check whether bess108 is cycling between competing setpoints.
- uncertainty note: Constraints: Only coarse feeder-level evidence was visible before v4 localization.
