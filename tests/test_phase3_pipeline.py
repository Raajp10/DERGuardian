from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from phase3.experiment_utils import prepare_window_bundle
from phase3.zero_day_splitter import available_scenarios, build_zero_day_full_run_data


ROOT = Path(__file__).resolve().parents[1]


class Phase3PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = prepare_window_bundle(ROOT)
        cls.scenarios = available_scenarios(cls.bundle.labels_df)

    def test_zero_day_split_contract_for_representative_holdout(self) -> None:
        self.assertTrue(self.scenarios, "No Phase 3 scenarios were found in attack labels.")
        holdout_scenario = self.scenarios[0]

        full_data = build_zero_day_full_run_data(
            feature_df=self.bundle.residual_df,
            labels_df=self.bundle.labels_df,
            holdout_scenario=holdout_scenario,
            buffer_windows=2,
        )
        split_df = full_data.split_df.copy()
        self.assertFalse(split_df.empty)

        split_names = set(split_df["split_name"].astype(str).unique())
        self.assertTrue({"train", "val", "test"}.issubset(split_names))

        holdout_attack_mask = (
            split_df["scenario_window_id"].astype(str).eq(holdout_scenario)
            & split_df["attack_present"].astype(int).eq(1)
        )
        self.assertGreater(int(holdout_attack_mask.sum()), 0, "Representative holdout has no attack windows in split output.")
        self.assertTrue(split_df.loc[holdout_attack_mask, "split_name"].eq("test").all())

        leakage_mask = holdout_attack_mask & split_df["split_name"].isin(["train", "val"])
        self.assertEqual(int(leakage_mask.sum()), 0, "Held-out attack windows leaked into train/val.")

        non_holdout_attack_mask = (
            split_df["attack_present"].astype(int).eq(1)
            & ~split_df["scenario_window_id"].astype(str).eq(holdout_scenario)
        )
        self.assertGreater(
            int((non_holdout_attack_mask & split_df["split_name"].isin(["train", "val"])).sum()),
            0,
            "Expected at least one non-held-out attack window in train/val for the representative fold.",
        )

    def test_canonical_phase3_artifacts_exist_and_are_non_empty(self) -> None:
        scenario_count = len(self.scenarios)
        artifacts = {
            ROOT / "outputs" / "reports" / "phase3_zero_day_artifacts" / "zero_day_model_summary.csv": {
                "required_columns": {"model_name", "precision", "recall", "f1", "holdout_scenario_count", "total_scenarios"},
            },
            ROOT / "outputs" / "reports" / "phase3_zero_day_artifacts" / "zero_day_per_scenario_metrics.csv": {
                "required_columns": {"model_name", "scenario_id", "precision", "recall", "f1"},
            },
            ROOT / "outputs" / "reports" / "phase3_zero_day_artifacts" / "zero_day_latency_analysis.csv": {
                "required_columns": {"model_name", "scenario_id", "attack_start_utc", "latency_seconds"},
            },
            ROOT / "outputs" / "reports" / "phase3_zero_day_artifacts" / "modality_ablation_table.csv": {
                "required_columns": {"model_name", "precision", "recall", "f1", "modality"},
            },
            ROOT / "outputs" / "reports" / "phase3_zero_day_artifacts" / "fdi_ablation_table.csv": {
                "required_columns": {"model_name", "precision", "recall", "f1", "feature_variant"},
            },
        }

        for path, contract in artifacts.items():
            self.assertTrue(path.exists(), f"Expected Phase 3 artifact is missing: {path}")
            frame = pd.read_csv(path)
            self.assertGreater(len(frame), 0, f"Expected non-empty artifact: {path}")
            self.assertTrue(contract["required_columns"].issubset(frame.columns), f"Missing required columns in {path}")
            self.assertTrue(frame.notna().any().any(), f"Artifact is unexpectedly all-null: {path}")

        summary_frame = pd.read_csv(ROOT / "outputs" / "reports" / "phase3_zero_day_artifacts" / "zero_day_model_summary.csv")
        self.assertTrue((summary_frame["holdout_scenario_count"] == scenario_count).all())
        self.assertTrue((summary_frame["total_scenarios"] == scenario_count).all())

        per_scenario_frame = pd.read_csv(ROOT / "outputs" / "reports" / "phase3_zero_day_artifacts" / "zero_day_per_scenario_metrics.csv")
        self.assertEqual(int(per_scenario_frame["scenario_id"].nunique()), scenario_count)

        split_frame = pd.read_csv(ROOT / "outputs" / "reports" / "phase3_zero_day_artifacts" / "zero_day_split_summary.csv")
        self.assertEqual(int(split_frame["holdout_scenario"].nunique()), scenario_count)

        manifest_path = ROOT / "outputs" / "reports" / "phase3_zero_day_artifacts" / "zero_day_run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required_manifest_fields = {
            "evaluation_timestamp_utc",
            "random_seed",
            "execution_mode",
            "command",
            "project_root",
            "evaluated_model_names",
            "evaluated_package_paths",
            "attacked_windows_path",
            "attack_labels_path",
            "clean_windows_path",
            "residual_artifact_path",
            "holdout_scenarios",
            "output_root",
            "artifact_paths",
            "metrics_summary_path",
            "per_scenario_metrics_path",
            "latency_analysis_path",
            "explanation_artifact_paths",
            "package_reuse_mode",
        }
        self.assertTrue(required_manifest_fields.issubset(manifest.keys()))
        for key in ["attacked_windows_path", "attack_labels_path", "clean_windows_path", "metrics_summary_path", "per_scenario_metrics_path", "latency_analysis_path"]:
            self.assertTrue(Path(manifest[key]).exists(), f"Manifest path does not exist: {key} -> {manifest[key]}")

    def test_explanation_artifacts_exist_and_have_expected_structure(self) -> None:
        artifact_root = ROOT / "outputs" / "reports" / "explanation_artifacts"
        summary_path = artifact_root / "explanation_eval_summary.json"
        packet_path = artifact_root / "explanation_packet.json"
        explanation_path = artifact_root / "explanation_output.json"
        packet_dir = artifact_root / "packets"
        explanation_dir = artifact_root / "explanations"

        for path in [summary_path, packet_path, explanation_path]:
            self.assertTrue(path.exists(), f"Expected explanation artifact is missing: {path}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        required_summary_fields = {
            "family_classification_accuracy",
            "asset_attribution_accuracy",
            "evidence_grounding_rate",
            "unknown_usage_rate",
            "explanation_completeness",
            "incident_count",
        }
        self.assertTrue(required_summary_fields.issubset(summary.keys()))
        self.assertGreater(int(summary["incident_count"]), 0)

        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        explanation = json.loads(explanation_path.read_text(encoding="utf-8"))

        self.assertIn("incident_id", packet)
        self.assertIn("window", packet)
        self.assertIn("candidate_families", packet)
        self.assertTrue(packet["incident_id"])
        self.assertTrue((packet.get("window") or {}).get("start_time_utc"))
        self.assertTrue(
            ((packet.get("offline_evaluation") or {}).get("scenario_label") or {}).get("scenario_id"),
            "Expected scenario linkage in canonical explanation packet.",
        )

        self.assertIn("incident_id", explanation)
        self.assertIn("summary", explanation)
        self.assertIn("suspected_attack_family", explanation)
        self.assertIn("operator_actions", explanation)
        self.assertIn("limitations", explanation)
        self.assertTrue(str(explanation["summary"]).strip())
        self.assertEqual(packet["incident_id"], explanation["incident_id"])

        self.assertTrue(packet_dir.exists() and packet_dir.is_dir())
        self.assertTrue(explanation_dir.exists() and explanation_dir.is_dir())

        packet_files = sorted(packet_dir.glob("*.json"))
        explanation_files = sorted(explanation_dir.glob("*.json"))
        self.assertGreater(len(packet_files), 0)
        self.assertGreater(len(explanation_files), 0)
        self.assertEqual(len(packet_files), len(explanation_files))
        self.assertEqual(int(summary["incident_count"]), len(packet_files))

        sample_packet = json.loads(packet_files[0].read_text(encoding="utf-8"))
        sample_explanation_path = explanation_dir / packet_files[0].name
        self.assertTrue(sample_explanation_path.exists(), f"Missing paired explanation for {packet_files[0].name}")
        sample_explanation = json.loads(sample_explanation_path.read_text(encoding="utf-8"))

        self.assertEqual(sample_packet.get("incident_id"), sample_explanation.get("incident_id"))
        self.assertTrue(str(sample_explanation.get("summary", "")).strip())

        for packet_file in packet_files:
            packet = json.loads(packet_file.read_text(encoding="utf-8"))
            explanation = json.loads((explanation_dir / packet_file.name).read_text(encoding="utf-8"))
            combined_text = (
                str(explanation.get("summary", "")).lower()
                + " "
                + " ".join(str(item) for item in explanation.get("why_flagged", [])).lower()
            )
            if bool(packet.get("predicted_alert", False)):
                self.assertNotIn("did not raise an alert", combined_text)
                self.assertNotIn("remained below the threshold", combined_text)
            else:
                self.assertNotIn("raised incident", combined_text)
                self.assertNotIn("raised alert", combined_text)
                self.assertNotIn("exceeded the threshold", combined_text)

    def test_error_analysis_artifacts_exist_and_match_expanded_holdout_set(self) -> None:
        report_root = ROOT / "outputs" / "reports"
        summary_path = report_root / "phase3_error_analysis_summary.json"
        summary_md_path = report_root / "phase3_error_analysis_summary.md"
        fp_path = report_root / "phase3_false_positive_analysis.csv"
        fn_path = report_root / "phase3_false_negative_analysis.csv"
        patterns_path = report_root / "phase3_per_scenario_error_patterns.csv"
        case_studies_path = report_root / "phase3_case_studies.md"

        for path in [summary_path, summary_md_path, fp_path, fn_path, patterns_path, case_studies_path]:
            self.assertTrue(path.exists(), f"Expected Phase 3 error-analysis artifact is missing: {path}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(int(summary["scenario_count"]), len(self.scenarios))
        self.assertGreaterEqual(int(summary["model_count"]), 3)
        self.assertIn("key_findings", summary)
        self.assertGreater(len(summary["key_findings"]), 0)

        for csv_path, required_columns in [
            (fp_path, {"model_name", "holdout_scenario", "false_positive_count", "false_positive_rate"}),
            (fn_path, {"model_name", "scenario_id", "false_negative_count", "false_negative_rate", "difficulty_score"}),
            (patterns_path, {"scenario_id", "mean_recall", "mean_false_negative_rate", "difficulty_score", "visibility_signature"}),
        ]:
            frame = pd.read_csv(csv_path)
            self.assertGreater(len(frame), 0, f"Expected non-empty error-analysis artifact: {csv_path}")
            self.assertTrue(required_columns.issubset(frame.columns), f"Missing required columns in {csv_path}")

        patterns_frame = pd.read_csv(patterns_path)
        self.assertEqual(int(patterns_frame["scenario_id"].nunique()), len(self.scenarios))
        self.assertTrue(patterns_frame["difficulty_score"].notna().all())
        self.assertTrue(patterns_frame["mean_recall"].between(0.0, 1.0).all())

    def test_saved_phase1_package_reuse_artifacts_exist(self) -> None:
        artifact_root = ROOT / "outputs" / "reports" / "phase3_package_reuse_artifacts" / "autoencoder"
        summary_csv = artifact_root / "saved_package_summary.csv"
        summary_md = artifact_root / "saved_package_summary.md"
        manifest_path = artifact_root / "saved_package_run_manifest.json"
        self.assertTrue(summary_csv.exists())
        self.assertTrue(summary_md.exists())
        self.assertTrue(manifest_path.exists())

        summary = pd.read_csv(summary_csv)
        self.assertGreater(len(summary), 0)
        self.assertTrue({"model_name", "precision", "recall", "f1", "average_precision"}.issubset(summary.columns))

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["mode"], "saved_phase1_package_reuse")
        self.assertTrue(str(manifest["package_dir"]).endswith("phase1_ready_models\\autoencoder") or str(manifest["package_dir"]).endswith("phase1_ready_models/autoencoder"))
        required_manifest_fields = {
            "evaluation_timestamp_utc",
            "random_seed",
            "execution_mode",
            "command",
            "project_root",
            "evaluated_model_names",
            "evaluated_package_paths",
            "attacked_windows_path",
            "attack_labels_path",
            "clean_windows_path",
            "residual_artifact_path",
            "holdout_scenarios",
            "output_root",
            "artifact_paths",
            "metrics_summary_path",
            "per_scenario_metrics_path",
            "latency_analysis_path",
            "explanation_artifact_paths",
            "package_reuse_mode",
        }
        self.assertTrue(required_manifest_fields.issubset(manifest.keys()))
        self.assertTrue(Path(manifest["package_dir"]).exists())
        for key in ["attacked_windows_path", "attack_labels_path", "clean_windows_path", "metrics_summary_path", "per_scenario_metrics_path", "latency_analysis_path"]:
            self.assertTrue(Path(manifest[key]).exists(), f"Saved-package manifest path does not exist: {key} -> {manifest[key]}")


if __name__ == "__main__":
    unittest.main()
