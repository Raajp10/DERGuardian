"""Test coverage for DERGuardian test phase1 upgrade pipeline behavior.

These tests exercise pipeline contracts, package readiness, or phase-level
functionality without changing scientific outputs.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "outputs" / "reports" / "model_full_run_artifacts"
MODEL_ROOT = ROOT / "outputs" / "models_full_run"


class Phase1UpgradePipelineTests(unittest.TestCase):
    """Structured object used by the test workflow."""

    def test_context_summary_artifacts_exist_and_have_expected_structure(self) -> None:
        jsonl_path = REPORT_ROOT / "phase1_context_summaries.jsonl"
        parquet_path = REPORT_ROOT / "phase1_context_summaries.parquet"
        self.assertTrue(jsonl_path.exists())
        self.assertTrue(parquet_path.exists())

        table = pd.read_parquet(parquet_path)
        self.assertGreater(len(table), 0)
        required_columns = {
            "window_start_utc",
            "window_end_utc",
            "split_name",
            "scenario_id",
            "changed_channel_count",
            "max_abs_relative_change",
            "voltage_violation_flag",
            "compact_summary",
        }
        self.assertTrue(required_columns.issubset(table.columns))

        first_record = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertIn("top_deviating_signals", first_record)
        self.assertIn("environment_context", first_record)
        self.assertIn("feeder_context", first_record)
        self.assertIn("voltage_violations", first_record)
        self.assertIn("der_dispatch_consistency", first_record)
        self.assertIn("soc_consistency", first_record)

    def test_reasoning_outputs_exist_and_have_expected_fields(self) -> None:
        jsonl_path = REPORT_ROOT / "phase1_context_reasoning_outputs.jsonl"
        parquet_path = REPORT_ROOT / "phase1_context_reasoning_outputs.parquet"
        metadata_path = MODEL_ROOT / "context_reasoner" / "reasoner_metadata.json"
        prompt_path = MODEL_ROOT / "context_reasoner" / "prompt_template.txt"

        self.assertTrue(jsonl_path.exists())
        self.assertTrue(parquet_path.exists())
        self.assertTrue(metadata_path.exists())
        self.assertTrue(prompt_path.exists())

        table = pd.read_parquet(parquet_path)
        self.assertGreater(len(table), 0)
        required_columns = {
            "window_start_utc",
            "window_end_utc",
            "scenario_id",
            "likely_anomaly_family",
            "confidence_score",
            "anomaly_reasoning_score",
            "human_readable_explanation",
        }
        self.assertTrue(required_columns.issubset(table.columns))
        self.assertTrue(table["anomaly_reasoning_score"].between(0.0, 1.0).all())
        self.assertTrue(table["confidence_score"].between(0.0, 1.0).all())

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["reasoning_mode"], "local_prompt_style_baseline")
        self.assertIn("supported_families", metadata)
        self.assertIn("optional", metadata["notes"])

    def test_fusion_artifacts_exist_and_cover_all_modes(self) -> None:
        ablation_path = REPORT_ROOT / "fusion_ablation_table.csv"
        plot_path = REPORT_ROOT / "fusion_ablation_plot.png"
        overview_path = REPORT_ROOT / "phase1_upgrade_overview_panel.png"
        self.assertTrue(ablation_path.exists())
        self.assertTrue(plot_path.exists())
        self.assertTrue(overview_path.exists())

        frame = pd.read_csv(ablation_path)
        self.assertEqual(set(frame["fusion_mode"]), {"detector_only", "detector_plus_context", "detector_plus_context_plus_token"})
        self.assertTrue({"precision", "recall", "f1", "average_precision", "threshold"}.issubset(frame.columns))
        self.assertGreater(len(frame), 0)

        for mode in frame["fusion_mode"]:
            model_dir = MODEL_ROOT / "fusion" / mode
            report_dir = REPORT_ROOT / "fusion" / mode
            self.assertTrue((model_dir / "results.json").exists())
            self.assertTrue((model_dir / "predictions.parquet").exists())
            self.assertTrue((model_dir / "validation_predictions.parquet").exists())
            self.assertTrue((report_dir / "per_scenario_metrics.csv").exists())
            self.assertTrue((report_dir / "detection_latency_analysis.csv").exists())
            self.assertTrue((report_dir / "roc_curve.png").exists())
            self.assertTrue((report_dir / "precision_recall_curve.png").exists())
            self.assertTrue((report_dir / "confusion_matrix.png").exists())


if __name__ == "__main__":
    unittest.main()
