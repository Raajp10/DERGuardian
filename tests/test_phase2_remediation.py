"""Test coverage for DERGuardian test phase2 remediation behavior.

These tests exercise pipeline contracts, package readiness, or phase-level
functionality without changing scientific outputs.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from common.config import ProjectPaths, load_pipeline_config
from common.dss_runner import extract_inventory
from common.io_utils import read_dataframe, read_json
from phase2.compile_injections import build_asset_bounds, compile_scenarios, load_and_validate_scenarios
from phase2.contracts import CANONICAL_PHASE2_EXECUTION_PATH, CANONICAL_TARGET_COMPONENTS
from phase2_with_llm.validate_llm_scenarios import validate_scenarios_bundle


ROOT = Path(__file__).resolve().parents[1]


class Phase2RemediationTests(unittest.TestCase):
    """Structured object used by the test workflow."""

    def _canonical_scenario_count(self) -> int:
        payload = read_json(ROOT / "phase2" / "research_attack_scenarios.json")
        return int(len(payload["scenarios"]))

    def test_canonical_target_component_contract_is_aligned(self) -> None:
        canonical_schema = read_json(ROOT / "phase2" / "scenario_schema.json")
        llm_schema = read_json(ROOT / "phase2_with_llm" / "scenario_schema_compatible.json")
        system_card = read_json(ROOT / "phase2_with_llm" / "system_card.json")

        schema_components = set(canonical_schema["$defs"]["scenario"]["properties"]["target_component"]["enum"])
        schema_target_components = set(canonical_schema["$defs"]["target"]["properties"]["target_component"]["enum"])
        llm_components = set(llm_schema["$defs"]["scenario"]["properties"]["target_component"]["enum"])
        llm_target_components = set(llm_schema["$defs"]["target"]["properties"]["target_component"]["enum"])
        system_card_components = set(system_card["allowed_components"])

        expected = set(CANONICAL_TARGET_COMPONENTS)
        self.assertEqual(schema_components, expected)
        self.assertEqual(schema_target_components, expected)
        self.assertEqual(llm_components, expected)
        self.assertEqual(llm_target_components, expected)
        self.assertEqual(system_card_components, expected)
        self.assertNotIn("cyber_layer", expected)

    def test_json_driven_research_bundle_still_validates_and_compiles(self) -> None:
        config = load_pipeline_config(ROOT / "outputs" / "clean" / "config_snapshot.json")
        paths = ProjectPaths().ensure()
        inventory = extract_inventory(config, paths)
        truth = read_dataframe(ROOT / "outputs" / "clean" / "truth_physical_timeseries.parquet")
        measured = read_dataframe(ROOT / "outputs" / "clean" / "measured_physical_timeseries.parquet")
        available_assets = {asset.name for asset in inventory.pv_assets + inventory.bess_assets} | set(inventory.regulators) | set(inventory.capacitors) | set(inventory.switches)
        timeline = pd.to_datetime(truth["timestamp_utc"], utc=True)

        payload = load_and_validate_scenarios(
            scenario_source=ROOT / "phase2" / "research_attack_scenarios.json",
            schema_file=ROOT / "phase2" / "scenario_schema.json",
            timeline=timeline,
            available_assets=available_assets,
            available_columns=set(measured.columns),
            asset_bounds=build_asset_bounds(config, inventory),
        )
        canonical_count = self._canonical_scenario_count()
        self.assertEqual(len(payload["scenarios"]), canonical_count)
        self.assertGreaterEqual(canonical_count, 10)
        self.assertLessEqual(canonical_count, 15)

        compiled = compile_scenarios(payload, timeline)
        self.assertEqual(compiled["compiled_manifest"]["scenario_count"], canonical_count)
        self.assertGreaterEqual(len(compiled["measurement_actions"]), 1)
        self.assertGreaterEqual(len(compiled["physical_actions"]), 1)

    def test_sidecar_validation_rejects_unsupported_target_component(self) -> None:
        system_card = read_json(ROOT / "phase2_with_llm" / "system_card.json")
        payload = {
            "dataset_id": "invalid_bundle",
            "scenarios": [
                {
                    "scenario_id": "scn_bad_cyber_layer",
                    "scenario_name": "Unsupported cyber layer target",
                    "category": "telemetry_corruption",
                    "description": "Invalid test payload.",
                    "target_component": "cyber_layer",
                    "target_asset": "feeder",
                    "target_signal": "protocol",
                    "start_time_utc": "2025-06-01T01:00:00Z",
                    "duration_seconds": 120,
                    "injection_type": "bias",
                    "magnitude_units": "n/a",
                    "temporal_pattern": "step",
                    "expected_effect": "Should be rejected.",
                    "observable_signals": ["feeder_p_kw_total"],
                    "severity": "low",
                }
            ],
        }
        report, _ = validate_scenarios_bundle(
            raw_payload=payload,
            system_card=system_card,
            scenario_schema_path=ROOT / "phase2_with_llm" / "scenario_schema_compatible.json",
            phase2_schema_path=ROOT / "phase2" / "scenario_schema.json",
            compare_existing=False,
        )
        self.assertEqual(report["status"], "invalid")
        errors = "\n".join(report["errors"])
        self.assertIn("target_component", errors)
        self.assertIn("cyber_layer", errors)

    def test_canonical_validator_rejects_unsupported_target_component(self) -> None:
        payload = {
            "dataset_id": "invalid_bundle",
            "scenarios": [
                {
                    "scenario_id": "scn_bad_cyber_layer",
                    "scenario_name": "Unsupported cyber layer target",
                    "category": "telemetry_corruption",
                    "description": "Invalid test payload.",
                    "target_component": "cyber_layer",
                    "target_asset": "feeder",
                    "target_signal": "protocol",
                    "start_time_utc": "2025-06-01T01:00:00Z",
                    "duration_seconds": 120,
                    "injection_type": "bias",
                    "magnitude_units": "n/a",
                    "temporal_pattern": "step",
                    "expected_effect": "Should be rejected.",
                    "observable_signals": ["feeder_p_kw_total"],
                    "severity": "low",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "invalid.json"
            scenario_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            with self.assertRaises(Exception):
                load_and_validate_scenarios(
                    scenario_source=scenario_path,
                    schema_file=ROOT / "phase2" / "scenario_schema.json",
                )

    def test_phase2_outputs_and_qa_artifacts_exist(self) -> None:
        expected_paths = [
            ROOT / "outputs" / "attacked" / "truth_physical_timeseries.parquet",
            ROOT / "outputs" / "attacked" / "measured_physical_timeseries.parquet",
            ROOT / "outputs" / "attacked" / "cyber_events.parquet",
            ROOT / "outputs" / "attacked" / "attack_labels.parquet",
            ROOT / "outputs" / "attacked" / "compiled_measurement_actions.json",
            ROOT / "outputs" / "attacked" / "compiled_physical_actions.json",
            ROOT / "outputs" / "attacked" / "compiled_manifest.json",
            ROOT / "outputs" / "reports" / "attacked_validation_summary.json",
            ROOT / "outputs" / "reports" / "phase2_scenario_coverage_summary.json",
            ROOT / "outputs" / "reports" / "phase2_label_summary.json",
            ROOT / "outputs" / "reports" / "phase2_effect_summary.json",
            ROOT / "outputs" / "reports" / "phase2_scenario_difficulty.json",
        ]
        for path in expected_paths:
            self.assertTrue(path.exists(), msg=str(path))

    def test_phase2_coverage_summary_has_expected_structure(self) -> None:
        coverage = read_json(ROOT / "outputs" / "reports" / "phase2_scenario_coverage_summary.json")
        labels = read_json(ROOT / "outputs" / "reports" / "phase2_label_summary.json")
        canonical_count = self._canonical_scenario_count()
        self.assertEqual(int(coverage["scenario_count"]), canonical_count)
        self.assertIn("target_component_coverage", coverage)
        self.assertIn("target_asset_coverage", coverage)
        self.assertIn("duration_range_seconds", coverage)
        self.assertGreaterEqual(int(coverage["family_count"]), 6)
        self.assertIn("metadata_completeness", labels)
        self.assertIn("affected_asset_frequency", labels)

    def test_phase2_effect_summary_has_one_entry_per_canonical_scenario(self) -> None:
        effect_summary = read_json(ROOT / "outputs" / "reports" / "phase2_effect_summary.json")
        labels = read_dataframe(ROOT / "outputs" / "attacked" / "attack_labels.parquet")
        canonical_count = self._canonical_scenario_count()
        self.assertEqual(len(effect_summary), len(labels))
        self.assertEqual(len(effect_summary), canonical_count)
        for entry in effect_summary:
            self.assertIn("scenario_id", entry)
            self.assertIn("effect_presence_pass", entry)
            self.assertIn("truth_change_stats", entry)
            self.assertIn("measured_change_stats", entry)
            self.assertIn("cyber_change_stats", entry)
            self.assertTrue(entry["effect_presence_pass"])
            self.assertTrue(
                entry["physical_truth_changed"] or entry["measured_layer_changed"] or entry["cyber_layer_changed"]
            )
            self.assertGreaterEqual(int(entry["total_changed_channels_above_epsilon"]), 1)

    def test_phase2_scenario_difficulty_has_expected_structure(self) -> None:
        difficulty = read_json(ROOT / "outputs" / "reports" / "phase2_scenario_difficulty.json")
        labels = read_dataframe(ROOT / "outputs" / "attacked" / "attack_labels.parquet")
        canonical_count = self._canonical_scenario_count()
        self.assertEqual(len(difficulty), len(labels))
        self.assertEqual(len(difficulty), canonical_count)
        valid_bands = {"easy", "medium", "hard"}
        for entry in difficulty:
            self.assertIn("difficulty_score", entry)
            self.assertIn("difficulty_band", entry)
            self.assertIn("component_scores", entry)
            self.assertIn("explanation", entry)
            self.assertGreaterEqual(float(entry["difficulty_score"]), 0.0)
            self.assertLessEqual(float(entry["difficulty_score"]), 100.0)
            self.assertIn(entry["difficulty_band"], valid_bands)
            component_scores = entry["component_scores"]
            for key in [
                "magnitude_factor",
                "duration_factor",
                "observability_factor",
                "channel_spread_factor",
                "multi_layer_factor",
            ]:
                self.assertIn(key, component_scores)

    def test_phase2_manifest_records_canonical_execution_path(self) -> None:
        manifest = read_json(ROOT / "outputs" / "attacked" / "scenario_manifest.json")
        self.assertEqual(tuple(manifest["canonical_phase2_execution_path"]), CANONICAL_PHASE2_EXECUTION_PATH)
        assumptions = "\n".join(manifest["assumptions"])
        self.assertIn("JSON-driven", assumptions)
        self.assertEqual(len(manifest.get("applied_scenarios", [])), self._canonical_scenario_count())

    def test_phase2_figure_and_report_wording_matches_canonical_path(self) -> None:
        config_text = (ROOT / "paper_figure_suite" / "configs" / "phase_overview_config.json").read_text(encoding="utf-8")
        architecture_config_text = (ROOT / "paper_figure_suite" / "configs" / "architecture_config.json").read_text(encoding="utf-8")
        workflow_config_text = (ROOT / "paper_figure_suite" / "configs" / "zero_day_workflow_config.json").read_text(encoding="utf-8")
        report_text = (ROOT / "reports" / "generated_report.md").read_text(encoding="utf-8")
        system_architecture_svg = (ROOT / "paper_figure_suite" / "outputs" / "system_architecture.svg").read_text(encoding="utf-8")
        phase_overview_svg = (ROOT / "paper_figure_suite" / "outputs" / "phase_overview.svg").read_text(encoding="utf-8")
        workflow_svg = (ROOT / "paper_figure_suite" / "outputs" / "zero_day_generation_workflow.svg").read_text(encoding="utf-8")

        self.assertIn("Structured Scenario Authoring and Attack Compilation", config_text)
        self.assertIn("Phase 2 Scenario Authoring + Attack Compilation", architecture_config_text)
        self.assertIn("Optional LLM Sidecar", workflow_config_text)
        self.assertIn("optional LLM-facing Phase 2 sidecar", report_text)
        self.assertIn("phase2_effect_summary", report_text)
        self.assertIn("phase2_scenario_difficulty", report_text)
        self.assertNotIn("LLM scenario generator", system_architecture_svg)
        self.assertIn("Scenario Authoring", system_architecture_svg)
        self.assertIn("Scenario Authoring", phase_overview_svg)
        self.assertIn("Optional sidecar", workflow_svg)


if __name__ == "__main__":
    unittest.main()
