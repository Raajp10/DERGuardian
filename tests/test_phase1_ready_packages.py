"""Test coverage for DERGuardian test phase1 ready packages behavior.

These tests exercise pipeline contracts, package readiness, or phase-level
functionality without changing scientific outputs.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from phase1_models.model_loader import verify_all_ready_packages


ROOT = Path(__file__).resolve().parents[1]
READY_ROOT = ROOT / "outputs" / "phase1_ready_models"


class Phase1ReadyPackageTests(unittest.TestCase):
    """Structured object used by the test workflow."""

    def test_all_detector_packages_have_standardized_layout(self) -> None:
        required_files = {
            "preprocessing.pkl",
            "feature_columns.json",
            "thresholds.json",
            "calibration.json",
            "config.json",
            "training_history.json",
            "metrics.json",
            "predictions.parquet",
            "model_manifest.json",
            "split_summary.json",
        }
        detector_packages = [
            "threshold_baseline",
            "isolation_forest",
            "autoencoder",
            "gru",
            "lstm",
            "transformer",
            "llm_baseline",
        ]
        for package_name in detector_packages:
            package_dir = READY_ROOT / package_name
            self.assertTrue(package_dir.exists(), f"Missing standardized package: {package_dir}")
            manifest = json.loads((package_dir / "model_manifest.json").read_text(encoding="utf-8"))
            expected = required_files | {manifest["checkpoint_filename"]}
            present = {path.name for path in package_dir.iterdir() if path.is_file()}
            self.assertTrue(expected.issubset(present), f"Package layout incomplete for {package_name}: {sorted(expected - present)}")

    def test_neural_packages_store_explicit_architecture_fields(self) -> None:
        required = {
            "autoencoder": {"input_dim", "hidden_dim", "bottleneck_dim"},
            "gru": {"input_dim", "hidden_dim", "num_layers", "sequence_length"},
            "lstm": {"input_dim", "hidden_dim", "num_layers", "sequence_length"},
            "transformer": {"input_dim", "d_model", "nhead", "num_layers", "dim_feedforward", "sequence_length"},
            "llm_baseline": {"vocab_size", "embed_dim", "sequence_length", "discretization_bins"},
        }
        for package_name, fields in required.items():
            config = json.loads((READY_ROOT / package_name / "config.json").read_text(encoding="utf-8"))
            architecture = config.get("architecture_parameters", {})
            self.assertTrue(fields.issubset(architecture.keys()), f"Missing architecture fields for {package_name}")

    def test_ready_package_reload_matches_saved_predictions(self) -> None:
        verification = verify_all_ready_packages(ROOT)
        self.assertIn("packages", verification)
        self.assertGreater(len(verification["packages"]), 0)
        for row in verification["packages"]:
            self.assertTrue(row["score_match"], f"Score mismatch for {row['package_name']}")
            self.assertTrue(row["predicted_match"], f"Prediction mismatch for {row['package_name']}")


if __name__ == "__main__":
    unittest.main()
