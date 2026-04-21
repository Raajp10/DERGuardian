from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from common.config import WindowConfig
from common.io_utils import read_dataframe
from phase1.build_windows import build_merged_windows
from phase1_models.metrics import detection_latency_table
from phase1_models.model_utils import CANONICAL_ARTIFACT_ROOT, CANONICAL_MODEL_ROOT
from phase1_models.residual_dataset import canonical_residual_artifact_path
from phase1_models.run_full_evaluation import prepare_full_run_data, run_full_benchmark


ROOT = Path(__file__).resolve().parents[1]


class Phase1CodeCompletionTests(unittest.TestCase):
    def test_latency_defaults_to_window_end_and_prefers_emission_time(self) -> None:
        labels = pd.DataFrame(
            [
                {
                    "scenario_id": "scn_test",
                    "start_time_utc": "2025-06-01T12:00:30Z",
                    "end_time_utc": "2025-06-01T12:10:00Z",
                }
            ]
        )
        predictions = pd.DataFrame(
            [
                {
                    "window_start_utc": "2025-06-01T12:00:00Z",
                    "window_end_utc": "2025-06-01T12:05:00Z",
                    "predicted": 1,
                }
            ]
        )
        latency = detection_latency_table(predictions, labels, "threshold_baseline")
        self.assertEqual(latency.loc[0, "latency_reference_column"], "window_end_utc")
        self.assertEqual(float(latency.loc[0, "latency_seconds"]), 270.0)

        predictions["detector_emission_utc"] = "2025-06-01T12:04:00Z"
        emission_latency = detection_latency_table(predictions, labels, "threshold_baseline")
        self.assertEqual(emission_latency.loc[0, "latency_reference_column"], "detector_emission_utc")
        self.assertEqual(float(emission_latency.loc[0, "latency_seconds"]), 210.0)

    def test_min_attack_overlap_fraction_is_applied(self) -> None:
        start = pd.Timestamp("2025-06-01T00:00:00Z")
        measured = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range(start, periods=361, freq="1s", tz="UTC"),
                "scenario_id": ["attacked"] * 361,
                "run_id": ["run-test"] * 361,
                "split_id": ["train"] * 361,
                "source_layer": ["measured"] * 361,
                "feeder_p_kw_total": [100.0] * 361,
            }
        )
        labels = pd.DataFrame(
            [
                {
                    "scenario_id": "scn_overlap",
                    "scenario_name": "Overlap threshold test",
                    "attack_family": "false_data_injection",
                    "severity": "medium",
                    "start_time_utc": "2025-06-01T00:04:00Z",
                    "end_time_utc": "2025-06-01T00:05:30Z",
                    "affected_assets": "pv60",
                    "affected_signals": "feeder_p_kw_total",
                    "target_component": "measured_layer",
                }
            ]
        )
        windows = build_merged_windows(
            measured_df=measured,
            cyber_df=pd.DataFrame(),
            labels_df=labels,
            windows=WindowConfig(window_seconds=300, step_seconds=60, min_attack_overlap_fraction=0.25),
        )
        self.assertGreaterEqual(len(windows), 2)
        self.assertEqual(int(windows.iloc[0]["attack_present"]), 0)
        self.assertEqual(str(windows.iloc[0]["attack_family"]), "benign")
        self.assertEqual(int(windows.iloc[1]["attack_present"]), 1)
        self.assertEqual(str(windows.iloc[1]["attack_family"]), "false_data_injection")

    def test_clean_and_attacked_manifest_window_counts_match_artifacts(self) -> None:
        clean_manifest = json.loads((ROOT / "outputs" / "clean" / "scenario_manifest.json").read_text(encoding="utf-8"))
        attacked_manifest = json.loads((ROOT / "outputs" / "attacked" / "scenario_manifest.json").read_text(encoding="utf-8"))

        clean_windows = read_dataframe(ROOT / "outputs" / "windows" / "merged_windows_clean.parquet")
        attacked_windows = read_dataframe(ROOT / "outputs" / "windows" / "merged_windows_attacked.parquet")

        clean_entry = next(item for item in clean_manifest["artifacts"] if item["name"] == "merged_windows_clean.parquet")
        attacked_entry = next(item for item in attacked_manifest["artifacts"] if item["name"] == "merged_windows_attacked.parquet")

        self.assertEqual(clean_entry["rows"], int(len(clean_windows)))
        self.assertEqual(clean_entry["columns"], int(len(clean_windows.columns)))
        self.assertEqual(attacked_entry["rows"], int(len(attacked_windows)))
        self.assertEqual(attacked_entry["columns"], int(len(attacked_windows.columns)))

    def test_pv35_monitor_is_present(self) -> None:
        monitor_text = (ROOT / "opendss" / "Monitor_Config.dss").read_text(encoding="utf-8")
        self.assertIn("New Monitor.Mon_PV35 element=PVSystem.pv35 terminal=1 mode=1 ppolar=no", monitor_text)

    def test_canonical_full_run_roots_are_locked(self) -> None:
        self.assertEqual(CANONICAL_ARTIFACT_ROOT, "model_full_run_artifacts")
        self.assertEqual(CANONICAL_MODEL_ROOT, "models_full_run")
        defaults = run_full_benchmark.__defaults__
        self.assertIsNotNone(defaults)
        self.assertEqual(defaults[-2], CANONICAL_ARTIFACT_ROOT)
        self.assertEqual(defaults[-1], CANONICAL_MODEL_ROOT)

    def test_token_baseline_label_is_renamed_in_generated_outputs(self) -> None:
        figure_text = (ROOT / "paper_figure_suite" / "outputs" / "main_results_comparison.svg").read_text(encoding="utf-8")
        report_text = (ROOT / "outputs" / "reports" / CANONICAL_ARTIFACT_ROOT / "model_report_full.md").read_text(encoding="utf-8")
        self.assertIn("Tokenized LLM-Style Baseline", figure_text)
        self.assertNotIn("LLM Baseline", figure_text)
        self.assertIn("Tokenized LLM-Style Baseline", report_text)

    def test_canonical_model_report_has_one_default_writer(self) -> None:
        full_run_text = (ROOT / "phase1_models" / "run_full_evaluation.py").read_text(encoding="utf-8")
        export_text = (ROOT / "phase1_models" / "export_model_report.py").read_text(encoding="utf-8")
        self.assertIn('model_report_full.md', full_run_text)
        self.assertIn('model_report_export.md', export_text)
        self.assertNotIn('artifact_root / "model_report_full.md"', export_text)

    def test_phase1_facing_files_do_not_claim_aoi(self) -> None:
        targets = [
            ROOT / "README.md",
            ROOT / "paper_figure_suite" / "configs" / "phase_overview_config.json",
            ROOT / "paper_figure_suite" / "generate_phase_overview.py",
            ROOT / "paper_figure_suite" / "generate_system_architecture.py",
            ROOT / "outputs" / "reports" / CANONICAL_ARTIFACT_ROOT / "model_report_full.md",
        ]
        for path in targets:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("AOI", text)
            self.assertNotIn("Age of Information", text)

    def test_canonical_residual_artifact_exists_and_has_expected_columns(self) -> None:
        full_data = prepare_full_run_data(ROOT, buffer_windows=2)
        residual_path = canonical_residual_artifact_path(ROOT)
        self.assertEqual(full_data.residual_artifact_path, residual_path)
        self.assertTrue(residual_path.exists())

        residual = read_dataframe(residual_path)
        required_columns = {
            "window_start_utc",
            "window_end_utc",
            "window_seconds",
            "scenario_id",
            "run_id",
            "split_id",
            "attack_present",
            "attack_family",
            "attack_severity",
            "attack_affected_assets",
            "scenario_window_id",
            "split_name",
        }
        self.assertTrue(required_columns.issubset(residual.columns))
        delta_columns = [column for column in residual.columns if column.startswith("delta__")]
        self.assertTrue(delta_columns)
        self.assertEqual(len(residual), len(full_data.split_df))
        self.assertEqual(set(residual["split_name"].dropna().unique()), {"train", "val", "test"})

    def test_canonical_residual_artifact_matches_attacked_minus_clean(self) -> None:
        residual_path = canonical_residual_artifact_path(ROOT)
        if not residual_path.exists():
            prepare_full_run_data(ROOT, buffer_windows=2)
        residual = read_dataframe(residual_path)
        clean = read_dataframe(ROOT / "outputs" / "windows" / "merged_windows_clean.parquet")
        attacked = read_dataframe(ROOT / "outputs" / "attacked" / "merged_windows.parquet")

        aligned = attacked.merge(clean, on="window_start_utc", how="inner", suffixes=("_attacked", "_clean"))
        comparison = residual.merge(
            aligned,
            on="window_start_utc",
            how="inner",
            suffixes=("_residual", "_aligned"),
        ).sort_values("window_start_utc").reset_index(drop=True)
        self.assertEqual(len(residual), len(comparison))

        delta_columns = sorted(column for column in residual.columns if column.startswith("delta__"))
        sample_delta_columns = delta_columns[:3]
        sample_rows = sorted({0, len(comparison) // 2, len(comparison) - 1})
        for delta_column in sample_delta_columns:
            base_column = delta_column[7:]
            expected = (
                comparison.loc[sample_rows, f"{base_column}_attacked"].astype(float).to_numpy()
                - comparison.loc[sample_rows, f"{base_column}_clean"].astype(float).to_numpy()
            )
            actual = comparison.loc[sample_rows, delta_column].astype(float).to_numpy()
            self.assertTrue((abs(expected - actual) < 1e-9).all(), msg=delta_column)


if __name__ == "__main__":
    unittest.main()
