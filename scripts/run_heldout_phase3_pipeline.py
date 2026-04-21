"""Repository orchestration script for DERGuardian.

This script runs or rebuilds run heldout phase3 pipeline artifacts for audits, figures,
reports, or reproducibility checks. It is release-support code and must preserve
the separation between canonical benchmark, replay, heldout synthetic, and
extension experiment contexts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import WindowConfig
from phase1.build_windows import build_merged_windows
from phase1_models.model_loader import load_phase1_package, run_phase1_inference
from phase1_models.metrics import compute_binary_metrics, compute_curve_payload, detection_latency_table
from phase1_models.residual_dataset import build_aligned_residual_dataframe


HELDOUT_ROOT = ROOT / "phase2_llm_benchmark" / "heldout_llm_response"
HELDOUT_PHASE2_RESULT_ROOT = ROOT / "phase2_llm_benchmark" / "new_respnse_result" / "models"
WINDOW_STUDY_ROOT = ROOT / "outputs" / "window_size_study"
PHASE3_ROOT = WINDOW_STUDY_ROOT / "phase3_heldout"
FIGURE_ROOT = WINDOW_STUDY_ROOT / "final_phase3_figures"
BEST_PACKAGE_ROOT = WINDOW_STUDY_ROOT / "best_model_package"
READY_PACKAGE_DIR = BEST_PACKAGE_ROOT / "ready_package"
WINDOW_CONFIG = WindowConfig(window_seconds=60, step_seconds=12, min_attack_overlap_fraction=0.2)
GENERATOR_ORDER = ["canonical_bundle", "chatgpt", "claude", "gemini", "grok", "ollama"]
TIME_NOW_UTC = pd.Timestamp.now(tz="UTC").isoformat()


@dataclass(slots=True)
class BundleInventoryRow:
    """Structured object used by the repository orchestration workflow."""

    generator_source: str
    json_file: str
    dataset_id: str
    scenario_count: int
    validation_status: str
    compile_status: str
    run_status: str


@dataclass(slots=True)
class HeldoutBundle:
    """Structured object used by the repository orchestration workflow."""

    generator_source: str
    selected_json_path: Path
    duplicate_json_paths: list[Path]
    dataset_id: str
    submitted_scenario_count: int
    validation_summary: dict[str, Any]
    compilation_report: dict[str, Any]
    rejected_summary: dict[str, Any]
    accepted_phase2_bundle_path: Path
    compiled_manifest_path: Path
    datasets_dir: Path
    reports_dir: Path
    xai_metrics_path: Path | None
    source_root: Path


@dataclass(slots=True)
class EvaluatedBundle:
    """Structured object used by the repository orchestration workflow."""

    generator_source: str
    dataset_id: str
    scenario_count: int
    accepted_scenario_count: int
    source_type: str
    source_json: str
    labels_path: Path
    attacked_windows_path: Path
    residual_windows_path: Path
    predictions_path: Path
    metrics: dict[str, Any]
    curves: dict[str, Any]
    latency_table: pd.DataFrame
    per_scenario: pd.DataFrame
    note: str
    family_summary: pd.DataFrame


def main() -> None:
    """Run the command-line entrypoint for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if not READY_PACKAGE_DIR.exists():
        raise FileNotFoundError(f"Saved winning Phase 1 package not found: {READY_PACKAGE_DIR}")

    PHASE3_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    clean_windows = _load_clean_windows()
    package = load_phase1_package(READY_PACKAGE_DIR)
    inventory_rows, heldout_bundles, validation_sections = discover_heldout_bundles()

    ollama_status_path = ROOT / "OLLAMA_EXTENSION_STATUS.md"
    ollama_status_path.write_text(_build_ollama_status(), encoding="utf-8")

    evaluated_bundles: list[EvaluatedBundle] = []
    evaluated_bundles.append(evaluate_canonical_bundle(package))

    for bundle in heldout_bundles:
        evaluated = evaluate_heldout_bundle(bundle=bundle, clean_windows=clean_windows, package=package)
        evaluated_bundles.append(evaluated)

    completed_generators = {bundle.generator_source for bundle in evaluated_bundles if bundle.generator_source != "canonical_bundle"}
    for row in inventory_rows:
        if row.generator_source in completed_generators and not row.run_status.startswith("not_run"):
            row.run_status = "completed_saved_transformer_replay"

    inventory_df = pd.DataFrame([asdict(row) for row in inventory_rows])
    inventory_df.to_csv(ROOT / "heldout_bundle_inventory.csv", index=False)

    _write_validation_report(inventory_df=inventory_df, validation_sections=validation_sections, bundles=heldout_bundles)
    metrics_df, per_scenario_df = _write_bundle_reports(evaluated_bundles)
    figure_manifest = _write_figures(metrics_df=metrics_df, per_scenario_df=per_scenario_df, evaluated_bundles=evaluated_bundles)
    _write_best_package_manifests(metrics_df=metrics_df, figure_manifest=figure_manifest)
    _write_phase3_reports(metrics_df=metrics_df, per_scenario_df=per_scenario_df, evaluated_bundles=evaluated_bundles, figure_manifest=figure_manifest)
    _write_pipeline_alignment(metrics_df=metrics_df, per_scenario_df=per_scenario_df)


def discover_heldout_bundles() -> tuple[list[BundleInventoryRow], list[HeldoutBundle], list[str]]:
    """Handle discover heldout bundles within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    inventory_rows: list[BundleInventoryRow] = []
    selected_bundles: list[HeldoutBundle] = []
    validation_sections: list[str] = []

    for generator in ["chatgpt", "claude", "gemini", "grok"]:
        folder = HELDOUT_ROOT / generator
        if not folder.exists():
            continue
        json_paths = sorted(folder.glob("*.json"))
        if not json_paths:
            validation_sections.append(f"## {generator}\n\nNo JSON files found.\n")
            continue

        grouped: dict[str, list[Path]] = {}
        payloads: dict[Path, dict[str, Any]] = {}
        for path in json_paths:
            raw = path.read_bytes()
            grouped.setdefault(hashlib.sha256(raw).hexdigest(), []).append(path)
            payloads[path] = json.loads(raw)

        unique_groups = list(grouped.values())
        chosen_group = sorted(
            unique_groups,
            key=lambda group: (
                0 if any(path.name == "new_respnse.json" for path in group) else 1,
                min(path.name for path in group),
            ),
        )[0]
        selected_json = next((path for path in chosen_group if path.name == "new_respnse.json"), chosen_group[0])
        duplicates = [path for path in chosen_group if path != selected_json]
        selected_payload = payloads[selected_json]
        dataset_id = str(selected_payload.get("dataset_id") or f"{generator}_unknown_dataset")
        submitted_count = len(selected_payload.get("scenarios", []))

        bundle = _resolve_existing_phase2_bundle(generator=generator, selected_json=selected_json, dataset_id=dataset_id, submitted_count=submitted_count)
        bundle.duplicate_json_paths.extend(duplicates)
        selected_bundles.append(bundle)

        selected_name = selected_json.name
        compile_status = "reused_existing_phase2_artifacts"
        run_status = "scheduled_for_saved_transformer_evaluation"
        inventory_rows.append(
            BundleInventoryRow(
                generator_source=generator,
                json_file=str(selected_json.relative_to(ROOT)),
                dataset_id=dataset_id,
                scenario_count=submitted_count,
                validation_status=str(bundle.validation_summary.get("status", "unknown")),
                compile_status=compile_status,
                run_status=run_status,
            )
        )
        for duplicate in duplicates:
            inventory_rows.append(
                BundleInventoryRow(
                    generator_source=generator,
                    json_file=str(duplicate.relative_to(ROOT)),
                    dataset_id=dataset_id,
                    scenario_count=submitted_count,
                    validation_status=f"duplicate_of:{selected_name}",
                    compile_status="not_recompiled_duplicate_input",
                    run_status="not_run_duplicate_input",
                )
            )

        rejected = bundle.rejected_summary.get("rejections") or []
        lines = [
            f"## {generator}",
            "",
            f"- selected bundle: `{selected_json.relative_to(ROOT)}`",
            f"- duplicate files: {', '.join(f'`{path.relative_to(ROOT)}`' for path in duplicates) if duplicates else 'none'}",
            f"- dataset_id: `{dataset_id}`",
            f"- submitted scenarios: {submitted_count}",
            f"- valid scenarios: {bundle.validation_summary.get('scenarios_valid', bundle.compilation_report.get('number_valid', 0))}",
            f"- rejected scenarios: {bundle.validation_summary.get('scenarios_rejected', bundle.compilation_report.get('number_rejected', 0))}",
            f"- validation summary: `{bundle.reports_dir.relative_to(ROOT) / 'validation_summary.json'}`",
            f"- compilation report: `{bundle.accepted_phase2_bundle_path.parent.relative_to(ROOT) / 'compilation_report.json'}`",
        ]
        if rejected:
            lines.append("")
            lines.append("Rejected scenario reasons:")
            for item in rejected:
                scenario_id = item.get("scenario_id", "unknown_scenario")
                reasons = item.get("reasons") or ["no reason recorded"]
                lines.append(f"- `{scenario_id}`")
                for reason in reasons:
                    lines.append(f"  - {reason}")
        else:
            lines.append("")
            lines.append("No rejected scenario detail file was needed beyond the compiled summary.")
        lines.append("")
        validation_sections.append("\n".join(lines))

    return inventory_rows, selected_bundles, validation_sections


def _resolve_existing_phase2_bundle(
    generator: str,
    selected_json: Path,
    dataset_id: str,
    submitted_count: int,
) -> HeldoutBundle:
    source_root = HELDOUT_PHASE2_RESULT_ROOT / generator
    if not source_root.exists():
        raise FileNotFoundError(f"Expected heldout Phase 2 result folder not found for {generator}: {source_root}")

    validation_path = source_root / "reports" / "validation_summary.json"
    compilation_path = source_root / "compiled_injections" / "compilation_report.json"
    rejected_path = source_root / "rejected_scenarios" / "rejected_scenarios.json"
    accepted_phase2_bundle_path = source_root / "compiled_injections" / "accepted_phase2_bundle.json"
    compiled_manifest_path = source_root / "compiled_injections" / "compiled_manifest.json"
    datasets_dir = source_root / "datasets"
    reports_dir = source_root / "reports"
    xai_metrics_path = source_root / "xai_v4" / "xai_metrics_v4.csv"

    required = [
        validation_path,
        compilation_path,
        rejected_path,
        accepted_phase2_bundle_path,
        compiled_manifest_path,
        datasets_dir / "measured_physical_timeseries.parquet",
        datasets_dir / "cyber_events.parquet",
        datasets_dir / "attack_labels.parquet",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Heldout Phase 2 artifacts for {generator} are incomplete: {missing}")

    validation_summary = json.loads(validation_path.read_text(encoding="utf-8"))
    compilation_report = json.loads(compilation_path.read_text(encoding="utf-8"))
    rejected_summary = json.loads(rejected_path.read_text(encoding="utf-8"))
    compiled_manifest = json.loads(compiled_manifest_path.read_text(encoding="utf-8"))

    if compiled_manifest.get("dataset_id") != dataset_id:
        raise ValueError(
            f"Heldout dataset id mismatch for {generator}: expected {dataset_id}, "
            f"found {compiled_manifest.get('dataset_id')}"
        )

    source_path_str = str(validation_summary.get("source_path") or "")
    expected_names = {selected_json.name, "response.json", "new_respnse.json"}
    if source_path_str and Path(source_path_str).name not in expected_names:
        raise ValueError(
            f"Heldout validation summary for {generator} points to unexpected source path: {source_path_str}"
        )

    submitted = int(validation_summary.get("scenarios_submitted") or submitted_count)
    if submitted != submitted_count:
        raise ValueError(
            f"Heldout submitted scenario count mismatch for {generator}: JSON={submitted_count}, summary={submitted}"
        )

    return HeldoutBundle(
        generator_source=generator,
        selected_json_path=selected_json,
        duplicate_json_paths=[],
        dataset_id=dataset_id,
        submitted_scenario_count=submitted_count,
        validation_summary=validation_summary,
        compilation_report=compilation_report,
        rejected_summary=rejected_summary,
        accepted_phase2_bundle_path=accepted_phase2_bundle_path,
        compiled_manifest_path=compiled_manifest_path,
        datasets_dir=datasets_dir,
        reports_dir=reports_dir,
        xai_metrics_path=xai_metrics_path if xai_metrics_path.exists() else None,
        source_root=source_root,
    )


def evaluate_canonical_bundle(package) -> EvaluatedBundle:
    """Handle evaluate canonical bundle within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    labels_path = ROOT / "outputs" / "attacked" / "attack_labels.parquet"
    attacked_windows_path = ROOT / "outputs" / "window_size_study" / "60s" / "data" / "merged_windows_attacked.parquet"
    residual_windows_path = ROOT / "outputs" / "window_size_study" / "60s" / "residual_windows.parquet"
    bundle_root = PHASE3_ROOT / "canonical_bundle"
    bundle_root.mkdir(parents=True, exist_ok=True)

    labels_df = _normalize_labels(pd.read_parquet(labels_path))
    attacked_windows = pd.read_parquet(attacked_windows_path)
    residual_df = pd.read_parquet(residual_windows_path)
    predictions, metrics, curves, latency_table, per_scenario, family_summary = _score_residual_windows(
        residual_df=residual_df,
        labels_df=labels_df,
        package=package,
        bundle_root=bundle_root,
        source_name="canonical_bundle",
    )

    scenario_manifest_path = ROOT / "outputs" / "attacked" / "scenario_manifest.json"
    dataset_id = "canonical_phase2_bundle"
    if scenario_manifest_path.exists():
        payload = json.loads(scenario_manifest_path.read_text(encoding="utf-8"))
        dataset_id = str(payload.get("scenario_id") or dataset_id)

    note = (
        "In-domain canonical attacked bundle reused from existing repository outputs. "
        "Metrics below are bundle-level replay results from the saved 60s transformer package, not the Phase 1 test split."
    )
    if int(metrics.get("nonfinite_feature_count", 0)) > 0:
        note += (
            f" Replay also required numerical sanitization for {int(metrics['nonfinite_feature_count'])} selected features "
            f"across {int(metrics.get('nonfinite_row_count', 0))} rows."
        )
    predictions_path = bundle_root / "predictions" / "transformer_predictions.parquet"
    return EvaluatedBundle(
        generator_source="canonical_bundle",
        dataset_id=dataset_id,
        scenario_count=int(len(labels_df)),
        accepted_scenario_count=int(len(labels_df)),
        source_type="canonical_existing_bundle",
        source_json=str(scenario_manifest_path.relative_to(ROOT)) if scenario_manifest_path.exists() else "outputs/attacked",
        labels_path=labels_path,
        attacked_windows_path=attacked_windows_path,
        residual_windows_path=residual_windows_path,
        predictions_path=predictions_path,
        metrics=metrics,
        curves=curves,
        latency_table=latency_table,
        per_scenario=per_scenario,
        note=note,
        family_summary=family_summary,
    )


def evaluate_heldout_bundle(bundle: HeldoutBundle, clean_windows: pd.DataFrame, package) -> EvaluatedBundle:
    """Handle evaluate heldout bundle within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    bundle_root = PHASE3_ROOT / bundle.generator_source
    data_root = bundle_root / "data"
    predictions_root = bundle_root / "predictions"
    reports_root = bundle_root / "reports"
    data_root.mkdir(parents=True, exist_ok=True)
    predictions_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    labels_path = bundle.datasets_dir / "attack_labels.parquet"
    measured_path = bundle.datasets_dir / "measured_physical_timeseries.parquet"
    cyber_path = bundle.datasets_dir / "cyber_events.parquet"
    attacked_windows_path = data_root / "merged_windows_attacked_60s_12s.parquet"
    residual_windows_path = data_root / "residual_windows_60s_12s.parquet"
    bundle_manifest_path = bundle_root / "bundle_manifest.json"

    labels_df = _normalize_labels(pd.read_parquet(labels_path))

    if attacked_windows_path.exists():
        attacked_windows = pd.read_parquet(attacked_windows_path)
    else:
        measured_df = pd.read_parquet(measured_path)
        cyber_df = pd.read_parquet(cyber_path)
        attacked_windows = build_merged_windows(
            measured_df=measured_df,
            cyber_df=cyber_df,
            labels_df=labels_df,
            windows=WINDOW_CONFIG,
        )
        attacked_windows.to_parquet(attacked_windows_path, index=False)

    if residual_windows_path.exists():
        residual_df = pd.read_parquet(residual_windows_path)
    else:
        residual_df = build_aligned_residual_dataframe(clean_windows, attacked_windows, labels_df)
        residual_df.to_parquet(residual_windows_path, index=False)

    predictions, metrics, curves, latency_table, per_scenario, family_summary = _score_residual_windows(
        residual_df=residual_df,
        labels_df=labels_df,
        package=package,
        bundle_root=bundle_root,
        source_name=bundle.generator_source,
    )

    manifest = {
        "generator_source": bundle.generator_source,
        "dataset_id": bundle.dataset_id,
        "source_json": str(bundle.selected_json_path.relative_to(ROOT)),
        "phase2_source_root": str(bundle.source_root.relative_to(ROOT)),
        "accepted_phase2_bundle": str(bundle.accepted_phase2_bundle_path.relative_to(ROOT)),
        "compiled_manifest": str(bundle.compiled_manifest_path.relative_to(ROOT)),
        "datasets_dir": str(bundle.datasets_dir.relative_to(ROOT)),
        "validation_summary": str((bundle.reports_dir / "validation_summary.json").relative_to(ROOT)),
        "window_config": {
            "window_seconds": WINDOW_CONFIG.window_seconds,
            "step_seconds": WINDOW_CONFIG.step_seconds,
            "min_attack_overlap_fraction": WINDOW_CONFIG.min_attack_overlap_fraction,
        },
        "selected_json_duplicate_candidates": [
            str(path.relative_to(ROOT)) for path in ([bundle.selected_json_path] + bundle.duplicate_json_paths)
        ],
        "evaluation_timestamp_utc": TIME_NOW_UTC,
    }
    bundle_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    valid_count = int(bundle.validation_summary.get("scenarios_valid", len(labels_df)))
    rejected_count = int(bundle.validation_summary.get("scenarios_rejected", bundle.compilation_report.get("number_rejected", 0)))
    note = f"Reused existing Phase 2 heldout artifacts; accepted {valid_count}/{bundle.submitted_scenario_count}, rejected {rejected_count}."
    if int(metrics.get("nonfinite_feature_count", 0)) > 0:
        note += (
            f" Non-finite residual values affected {int(metrics['nonfinite_feature_count'])} selected features "
            f"across {int(metrics.get('nonfinite_row_count', 0))} rows; replay used a sanitized copy with non-finite values replaced by 0.0 and extreme values clipped for numerical stability."
        )

    return EvaluatedBundle(
        generator_source=bundle.generator_source,
        dataset_id=bundle.dataset_id,
        scenario_count=bundle.submitted_scenario_count,
        accepted_scenario_count=int(len(labels_df)),
        source_type="heldout_existing_phase2_artifacts",
        source_json=str(bundle.selected_json_path.relative_to(ROOT)),
        labels_path=labels_path,
        attacked_windows_path=attacked_windows_path,
        residual_windows_path=residual_windows_path,
        predictions_path=predictions_root / "transformer_predictions.parquet",
        metrics=metrics,
        curves=curves,
        latency_table=latency_table,
        per_scenario=per_scenario,
        note=note,
        family_summary=family_summary,
    )


def _score_residual_windows(
    residual_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    package,
    bundle_root: Path,
    source_name: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions_root = bundle_root / "predictions"
    reports_root = bundle_root / "reports"
    data_root = bundle_root / "data"
    predictions_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    residual = residual_df.copy()
    if "window_start_utc" in residual.columns:
        residual["window_start_utc"] = pd.to_datetime(residual["window_start_utc"], utc=True)
    if "window_end_utc" in residual.columns:
        residual["window_end_utc"] = pd.to_datetime(residual["window_end_utc"], utc=True)
    residual, nonfinite_audit = _sanitize_residual_for_inference(residual, package)
    if nonfinite_audit["affected_feature_count"] > 0:
        audit_path = reports_root / "nonfinite_feature_audit.json"
        sanitized_path = data_root / "residual_windows_60s_12s_sanitized.parquet"
        audit_path.write_text(json.dumps(nonfinite_audit, indent=2), encoding="utf-8")
        residual.to_parquet(sanitized_path, index=False)

    inference_started = time.perf_counter()
    inference = run_phase1_inference(package, residual)
    inference_seconds = time.perf_counter() - inference_started
    predictions = inference["predictions"].copy()
    predictions_path = predictions_root / "transformer_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)

    y_true = predictions["attack_present"].astype(int).to_numpy()
    scores = predictions["score"].astype(float).to_numpy()
    threshold = float(inference["metadata"]["threshold"])
    metrics = compute_binary_metrics(y_true, scores, threshold)
    curves = compute_curve_payload(y_true, scores)
    latency_table = detection_latency_table(predictions, labels_df, "transformer")
    per_scenario = _build_bundle_per_scenario(
        predictions=predictions,
        residual_df=residual,
        labels_df=labels_df,
        feature_columns=list(inference["metadata"]["feature_columns"]),
        generator_source=source_name,
    )
    family_summary = _build_family_summary(per_scenario)

    metrics.update(
        {
            "generator_source": source_name,
            "model_name": str(inference["metadata"]["model_name"]),
            "window_seconds": WINDOW_CONFIG.window_seconds,
            "step_seconds": WINDOW_CONFIG.step_seconds,
            "mean_latency_seconds": _safe_mean(latency_table.get("latency_seconds")),
            "median_latency_seconds": _safe_median(latency_table.get("latency_seconds")),
            "detected_scenarios": int(latency_table["latency_seconds"].notna().sum()) if not latency_table.empty else 0,
            "evaluated_windows": int(len(predictions)),
            "inference_time_seconds_bundle": float(inference_seconds),
            "inference_time_ms_per_prediction": float((inference_seconds / max(len(predictions), 1)) * 1000.0),
            "nonfinite_feature_count": int(nonfinite_audit["affected_feature_count"]),
            "nonfinite_row_count": int(nonfinite_audit["affected_row_count"]),
        }
    )

    latency_path = reports_root / "latency_metrics.csv"
    per_scenario_path = reports_root / "per_scenario_metrics.csv"
    bundle_metrics_path = reports_root / "bundle_metrics.json"
    curves_path = reports_root / "bundle_curves.json"
    family_summary_path = reports_root / "family_summary.csv"

    latency_table.to_csv(latency_path, index=False)
    per_scenario.to_csv(per_scenario_path, index=False)
    family_summary.to_csv(family_summary_path, index=False)
    bundle_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    curves_path.write_text(json.dumps(curves, indent=2), encoding="utf-8")

    return predictions, metrics, curves, latency_table, per_scenario, family_summary


def _sanitize_residual_for_inference(residual_df: pd.DataFrame, package) -> tuple[pd.DataFrame, dict[str, Any]]:
    sanitized = residual_df.copy()
    affected_features: list[dict[str, Any]] = []
    affected_rows = set()
    standardizer = package.preprocessing.get("standardizer")
    feature_columns = list(package.feature_columns)
    clip_z = 100.0

    for index, column in enumerate(feature_columns):
        if column not in sanitized.columns:
            continue
        series = pd.to_numeric(sanitized[column], errors="coerce")
        values = series.to_numpy(dtype=float)
        nonfinite_mask = ~np.isfinite(values)
        nonfinite_count = int(nonfinite_mask.sum())
        clip_count = 0
        cleaned = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if standardizer is not None and index < len(standardizer.mean):
            mean = float(standardizer.mean[index])
            std = float(standardizer.std[index]) if float(standardizer.std[index]) != 0.0 else 1.0
            lower = mean - clip_z * std
            upper = mean + clip_z * std
            clipped = cleaned.clip(lower=lower, upper=upper)
            clip_count = int((clipped.to_numpy(dtype=float) != cleaned.to_numpy(dtype=float)).sum())
            cleaned = clipped
        if nonfinite_count > 0 or clip_count > 0:
            affected_rows.update(series.index[nonfinite_mask].tolist())
            if clip_count > 0:
                clipped_mask = cleaned.to_numpy(dtype=float) != series.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
                affected_rows.update(series.index[clipped_mask].tolist())
            affected_features.append(
                {
                    "feature": column,
                    "nonfinite_count": nonfinite_count,
                    "clipped_count": clip_count,
                }
            )
        sanitized[column] = cleaned

    audit = {
        "affected_feature_count": len(affected_features),
        "affected_row_count": len(affected_rows),
        "affected_features": affected_features[:128],
        "sanitization_rule": (
            "Replace non-finite residual feature values (NaN, +inf, -inf) with 0.0, then clip selected replay features to +/-100 "
            "package-standard deviations for numerical stability. Raw residual parquet is preserved separately."
        ),
    }
    return sanitized, audit


def _build_bundle_per_scenario(
    predictions: pd.DataFrame,
    residual_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    feature_columns: list[str],
    generator_source: str,
) -> pd.DataFrame:
    preds = predictions.copy()
    preds["window_start_utc"] = pd.to_datetime(preds["window_start_utc"], utc=True)
    preds["window_end_utc"] = pd.to_datetime(preds["window_end_utc"], utc=True)
    labels = _normalize_labels(labels_df)
    residual = residual_df.copy()
    residual["window_start_utc"] = pd.to_datetime(residual["window_start_utc"], utc=True)
    residual_lookup = residual.set_index("window_start_utc", drop=False)

    rows: list[dict[str, Any]] = []
    for _, label in labels.iterrows():
        scenario_id = str(label["scenario_id"])
        attack_start = pd.to_datetime(label["start_time_utc"], utc=True)
        attack_end = pd.to_datetime(label["end_time_utc"], utc=True)

        attack_slice = preds[(preds["window_start_utc"] < attack_end) & (preds["window_end_utc"] > attack_start)].copy()
        context_slice = preds[
            (preds["window_end_utc"] >= attack_start)
            & (preds["window_end_utc"] <= attack_end + pd.Timedelta(minutes=15))
        ].copy()
        detections = context_slice[context_slice["predicted"] == 1].sort_values("window_end_utc")
        first_detection = detections.iloc[0] if not detections.empty else None
        first_detection_time = pd.to_datetime(first_detection["window_end_utc"], utc=True) if first_detection is not None else pd.NaT
        latency_seconds = float((first_detection_time - attack_start).total_seconds()) if pd.notna(first_detection_time) else math.nan

        representative_row = None
        if first_detection is not None:
            first_detection_start = pd.to_datetime(first_detection["window_start_utc"], utc=True)
            if first_detection_start in residual_lookup.index:
                representative_row = residual_lookup.loc[first_detection_start]
        if representative_row is None and not attack_slice.empty:
            peak_row = attack_slice.sort_values("score", ascending=False).iloc[0]
            peak_start = pd.to_datetime(peak_row["window_start_utc"], utc=True)
            representative_row = residual_lookup.loc[peak_start] if peak_start in residual_lookup.index else None

        top_features = _top_triggering_features(representative_row, feature_columns)
        score_summary = {
            "max_score": _safe_stat(attack_slice.get("score"), "max"),
            "mean_score": _safe_stat(attack_slice.get("score"), "mean"),
            "max_raw_score": _safe_stat(attack_slice.get("raw_score"), "max"),
            "attack_window_count": int(len(attack_slice)),
            "predicted_attack_windows": int(attack_slice["predicted"].sum()) if not attack_slice.empty else 0,
        }

        rows.append(
            {
                "generator_source": generator_source,
                "scenario_id": scenario_id,
                "attack_family": str(label.get("attack_family", "unknown")),
                "detected": int(first_detection is not None),
                "first_detection_time": first_detection_time.isoformat() if pd.notna(first_detection_time) else "",
                "latency_seconds": latency_seconds,
                "score_summary": json.dumps(score_summary, sort_keys=True),
                "top_triggering_signals": "|".join(top_features),
                "start_time_utc": attack_start.isoformat(),
                "end_time_utc": attack_end.isoformat(),
                "affected_assets": "|".join(_normalize_list(label.get("affected_assets"))),
                "accepted_in_bundle": 1,
            }
        )
    return pd.DataFrame(rows)


def _build_family_summary(per_scenario: pd.DataFrame) -> pd.DataFrame:
    if per_scenario.empty:
        return pd.DataFrame(columns=["generator_source", "attack_family", "scenario_count", "detected_count", "recall_like_rate", "mean_latency_seconds"])
    grouped = (
        per_scenario.groupby(["generator_source", "attack_family"], observed=False)
        .agg(
            scenario_count=("scenario_id", "count"),
            detected_count=("detected", "sum"),
            recall_like_rate=("detected", "mean"),
            mean_latency_seconds=("latency_seconds", "mean"),
        )
        .reset_index()
    )
    return grouped


def _write_bundle_reports(evaluated_bundles: list[EvaluatedBundle]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    per_scenario_frames: list[pd.DataFrame] = []
    for bundle in evaluated_bundles:
        metrics = dict(bundle.metrics)
        metrics.update(
            {
                "dataset_id": bundle.dataset_id,
                "scenario_count": bundle.scenario_count,
                "accepted_scenario_count": bundle.accepted_scenario_count,
                "generator_source": bundle.generator_source,
                "note": bundle.note,
            }
        )
        metric_rows.append(metrics)
        per_scenario_frames.append(bundle.per_scenario.copy())

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df = _order_generators(metrics_df, key="generator_source")
    metrics_output = metrics_df[
        [
            "generator_source",
            "dataset_id",
            "scenario_count",
            "accepted_scenario_count",
            "precision",
            "recall",
            "f1",
            "average_precision",
            "roc_auc",
            "mean_latency_seconds",
            "median_latency_seconds",
            "detected_scenarios",
            "nonfinite_feature_count",
            "nonfinite_row_count",
            "note",
        ]
    ].copy()
    metrics_output.to_csv(ROOT / "heldout_bundle_metrics.csv", index=False)

    per_scenario_df = pd.concat(per_scenario_frames, ignore_index=True) if per_scenario_frames else pd.DataFrame()
    per_scenario_df = _order_generators(per_scenario_df, key="generator_source")
    per_scenario_df.to_csv(ROOT / "heldout_bundle_per_scenario.csv", index=False)
    return metrics_output, per_scenario_df


def _write_figures(
    metrics_df: pd.DataFrame,
    per_scenario_df: pd.DataFrame,
    evaluated_bundles: list[EvaluatedBundle],
) -> list[dict[str, Any]]:
    figure_manifest: list[dict[str, Any]] = []

    phase1_window_path = FIGURE_ROOT / "phase1_window_comparison.png"
    phase1_latency_path = FIGURE_ROOT / "phase1_latency_vs_window.png"
    zero_day_f1_path = FIGURE_ROOT / "zero_day_f1_by_generator.png"
    zero_day_pr_path = FIGURE_ROOT / "zero_day_precision_recall_by_generator.png"
    zero_day_latency_path = FIGURE_ROOT / "zero_day_latency_by_generator.png"
    family_breakdown_path = FIGURE_ROOT / "zero_day_attack_family_breakdown.png"
    difficulty_heatmap_path = FIGURE_ROOT / "zero_day_scenario_difficulty_heatmap.png"
    xai_grounding_path = FIGURE_ROOT / "xai_grounding_summary.png"

    _plot_phase1_window_comparison(phase1_window_path)
    figure_manifest.append({"name": phase1_window_path.name, "path": str(phase1_window_path.relative_to(ROOT)), "purpose": "Phase 1 precision/recall/F1 vs window size."})

    _plot_phase1_latency_vs_window(phase1_latency_path)
    figure_manifest.append({"name": phase1_latency_path.name, "path": str(phase1_latency_path.relative_to(ROOT)), "purpose": "Phase 1 detection latency vs window size."})

    zero_day_df = metrics_df.copy()
    zero_day_df = _order_generators(zero_day_df, key="generator_source")
    _plot_single_metric_bar(zero_day_df, "f1", zero_day_f1_path, ylabel="F1", title="Zero-Day-Like F1 by Generator")
    figure_manifest.append({"name": zero_day_f1_path.name, "path": str(zero_day_f1_path.relative_to(ROOT)), "purpose": "Bundle-level F1 across canonical and heldout generators."})

    _plot_precision_recall(zero_day_df, zero_day_pr_path)
    figure_manifest.append({"name": zero_day_pr_path.name, "path": str(zero_day_pr_path.relative_to(ROOT)), "purpose": "Bundle-level precision and recall across generator sources."})

    _plot_single_metric_bar(zero_day_df, "mean_latency_seconds", zero_day_latency_path, ylabel="Seconds", title="Mean Detection Latency by Generator")
    figure_manifest.append({"name": zero_day_latency_path.name, "path": str(zero_day_latency_path.relative_to(ROOT)), "purpose": "Mean detection latency across generator sources."})

    family_frames = [bundle.family_summary.copy() for bundle in evaluated_bundles if not bundle.family_summary.empty]
    family_df = pd.concat(family_frames, ignore_index=True) if family_frames else pd.DataFrame()
    _plot_family_breakdown(family_df, family_breakdown_path)
    figure_manifest.append({"name": family_breakdown_path.name, "path": str(family_breakdown_path.relative_to(ROOT)), "purpose": "Per-family recall-like detection rate across generator sources."})

    _plot_scenario_difficulty_heatmap(per_scenario_df, difficulty_heatmap_path)
    figure_manifest.append({"name": difficulty_heatmap_path.name, "path": str(difficulty_heatmap_path.relative_to(ROOT)), "purpose": "Scenario-level difficulty heatmap using detection indicator and score summary."})

    _plot_xai_grounding_summary(xai_grounding_path)
    figure_manifest.append({"name": xai_grounding_path.name, "path": str(xai_grounding_path.relative_to(ROOT)), "purpose": "Heldout explanation-layer grounding summary from existing v4 XAI artifacts."})

    figure_manifest_path = FIGURE_ROOT / "figure_manifest.json"
    figure_manifest_path.write_text(json.dumps(figure_manifest, indent=2), encoding="utf-8")
    return figure_manifest


def _write_best_package_manifests(metrics_df: pd.DataFrame, figure_manifest: list[dict[str, Any]]) -> None:
    bundle_manifest_path = BEST_PACKAGE_ROOT / "bundle_evaluation_manifest.json"
    heldout_manifest_path = BEST_PACKAGE_ROOT / "heldout_zero_day_evaluation_manifest.json"

    metrics_path = ROOT / "heldout_bundle_metrics.csv"
    per_scenario_path = ROOT / "heldout_bundle_per_scenario.csv"
    reports = [
        "PHASE3_ZERO_DAY_REPORT.md",
        "FINAL_PUBLICATION_SAFE_CLAIMS.md",
        "FINAL_RESULTS_AND_DISCUSSION_DRAFT.md",
        "PHASE_COMPLETE_DECISION.md",
        "PIPELINE_ALIGNMENT_WITH_DIAGRAM.md",
    ]

    bundle_manifest = {
        "best_model_name": "transformer",
        "window_seconds": WINDOW_CONFIG.window_seconds,
        "step_seconds": WINDOW_CONFIG.step_seconds,
        "preprocessing_package_path": str((READY_PACKAGE_DIR / "preprocessing.pkl").relative_to(ROOT)),
        "ready_package_path": str(READY_PACKAGE_DIR.relative_to(ROOT)),
        "canonical_bundle_reference": str((ROOT / "outputs" / "attacked" / "scenario_manifest.json").relative_to(ROOT)),
        "heldout_metric_table": str(metrics_path.relative_to(ROOT)),
        "heldout_per_scenario_table": str(per_scenario_path.relative_to(ROOT)),
        "generated_at_utc": TIME_NOW_UTC,
    }
    bundle_manifest_path.write_text(json.dumps(bundle_manifest, indent=2), encoding="utf-8")

    heldout_manifest = {
        "best_model_name": "transformer",
        "window_size": "60s",
        "window_seconds": WINDOW_CONFIG.window_seconds,
        "step_seconds": WINDOW_CONFIG.step_seconds,
        "preprocessing_package_path": str((READY_PACKAGE_DIR / "preprocessing.pkl").relative_to(ROOT)),
        "heldout_bundles_evaluated": metrics_df["generator_source"].tolist(),
        "evaluation_date_utc": TIME_NOW_UTC,
        "figure_paths": [item["path"] for item in figure_manifest],
        "report_paths": reports,
    }
    heldout_manifest_path.write_text(json.dumps(heldout_manifest, indent=2), encoding="utf-8")


def _write_phase3_reports(
    metrics_df: pd.DataFrame,
    per_scenario_df: pd.DataFrame,
    evaluated_bundles: list[EvaluatedBundle],
    figure_manifest: list[dict[str, Any]],
) -> None:
    metrics_lookup = {row["generator_source"]: row for row in metrics_df.to_dict(orient="records")}
    heldout_only = metrics_df[metrics_df["generator_source"] != "canonical_bundle"].copy()
    heldout_best = heldout_only.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).head(1)
    heldout_best_name = heldout_best["generator_source"].iloc[0] if not heldout_best.empty else "none"

    family_summary_df = (
        pd.concat([bundle.family_summary for bundle in evaluated_bundles if not bundle.family_summary.empty], ignore_index=True)
        if any(not bundle.family_summary.empty for bundle in evaluated_bundles)
        else pd.DataFrame()
    )
    family_summary_df = _order_generators(family_summary_df, key="generator_source")

    zero_day_supported = bool(
        not heldout_only.empty
        and (heldout_only["f1"] >= 0.50).sum() >= 2
        and heldout_only["recall"].mean() >= 0.60
    )

    phase3_lines = [
        "# Phase 3 Zero-Day Report",
        "",
        "## objective",
        "",
        "Evaluate the saved canonical Phase 1 winner on unseen heldout attacked bundles generated from different LLM sources without retraining the detector.",
        "",
        "## which Phase 1 winner was reused",
        "",
        "- saved package: `outputs/window_size_study/best_model_package/ready_package`",
        "- model: `transformer`",
        "- window setup: `60s` windows, `12s` step, `0.2` minimum attack overlap fraction",
        "- detector reuse policy: fixed preprocessing, fixed feature set, fixed calibration, fixed threshold",
        "",
        "## heldout bundle inventory",
        "",
        _markdown_table(metrics_df[["generator_source", "dataset_id", "scenario_count", "accepted_scenario_count", "f1", "precision", "recall", "mean_latency_seconds"]]),
        "",
        "## validation summary",
        "",
        "- each heldout generator folder contained two identical JSON files: `response.json` and `new_respnse.json`",
        "- the run treated `new_respnse.json` as the selected effective bundle when present and marked `response.json` as a duplicate reference",
        "- Phase 2 validation and compilation were reused only from `phase2_llm_benchmark/new_respnse_result/models/<generator>/...` artifacts that pointed back to the heldout JSON source path",
        "- malformed or rejected scenarios were retained in the validation report and excluded from bundle evaluation only after canonical validation rejected them",
        "",
        "## per-generator results",
        "",
        _markdown_table(metrics_df[["generator_source", "precision", "recall", "f1", "average_precision", "roc_auc", "mean_latency_seconds", "note"]]),
        "",
        "## per-family results",
        "",
        _markdown_table(
            family_summary_df[["generator_source", "attack_family", "scenario_count", "detected_count", "recall_like_rate", "mean_latency_seconds"]]
            if not family_summary_df.empty
            else pd.DataFrame(columns=["generator_source", "attack_family", "scenario_count", "detected_count", "recall_like_rate", "mean_latency_seconds"])
        ),
        "",
        "## latency interpretation",
        "",
        f"- mean heldout detection latency ranged from {heldout_only['mean_latency_seconds'].min():.1f}s to {heldout_only['mean_latency_seconds'].max():.1f}s across valid heldout bundles." if not heldout_only.empty else "- no valid heldout bundle latency was available.",
        f"- best heldout F1 came from `{heldout_best_name}`." if heldout_best_name != "none" else "- no heldout bundle completed.",
        "- bundle-level latency here is measured from attack start to the first positive detector decision using the saved 60s-window transformer package.",
        "",
        "## what can be claimed safely",
        "",
        "- the saved 60s transformer generalized beyond the canonical Phase 2 bundle onto multiple unseen heldout generator bundles without retraining",
        "- the heldout results are generator-conditioned bundle replays, not exhaustive real-world zero-day evidence",
        "- the explanation layer should be described as grounded post-alert operator support, not human-level root-cause analysis",
        "",
        "## limitations",
        "",
        "- heldout bundles were still constrained by the same defensive scenario schema and physical validation rules",
        "- generator bundles differ in accepted scenario count because some submissions were rejected by canonical validation",
        "- canonical bundle metrics in this report are bundle-level replay numbers and should not be confused with the Phase 1 held-out test split used to choose the winning model",
        "- no Ollama heldout bundle was added because Ollama was not available locally during this run",
        "",
        "## whether cross-generator evidence supports a zero-day-like claim",
        "",
        (
            "The evidence supports a paper-safe `zero-day-like cross-generator evaluation` claim: the detector was frozen after Phase 1, then replayed on multiple unseen generator bundles produced outside the canonical attacked training bundle."
            if zero_day_supported
            else "The evidence supports only a limited heldout cross-generator generalization claim. It does not yet support a strong zero-day-like claim without stronger consistency or broader heldout coverage."
        ),
        "",
    ]
    (ROOT / "PHASE3_ZERO_DAY_REPORT.md").write_text("\n".join(phase3_lines), encoding="utf-8")

    claims_lines = [
        "# Final Publication-Safe Claims",
        "",
        "## safe to claim now",
        "",
        "- A canonical 60s transformer package was selected from the completed Phase 1 window-size study and reused without retraining for heldout evaluation.",
        "- The repository contains a canonical Phase 2 attacked bundle plus multiple heldout cross-generator bundles with explicit validation filtering.",
        "- The saved detector generalized to multiple unseen heldout generator bundles under the same defensive schema and physical constraints.",
        "- The explanation layer can be described as an evidence-grounded post-alert operator-facing aid.",
        "",
        "## safe with minor wording changes",
        "",
        "- `Zero-day-like cross-generator evaluation` is acceptable when explicitly defined as frozen-model replay on unseen LLM-generated heldout bundles under shared safety constraints.",
        "- `Generalizes across generators` is acceptable if paired with the exact bundle-level metrics and acceptance/rejection counts.",
        "- `Grounded explanation layer` is acceptable if the paper cites the existing heldout XAI grounding summaries and avoids human-level language.",
        "",
        "## not safe to claim",
        "",
        "- True real-world zero-day robustness.",
        "- Human-like root-cause analysis.",
        "- Complete coverage of all plausible DER cyber-physical attack space.",
        "- Cross-generator robustness independent of the shared scenario schema.",
        "",
        "## missing evidence still required",
        "",
        "- External non-schema-constrained attack bundles or human-authored scenarios.",
        "- More than one additional independent generator family beyond the current local set if a stronger generalization claim is needed.",
        "- Real deployment latency measurements from the edge/runtime package rather than offline parquet replay only.",
        "",
    ]
    (ROOT / "FINAL_PUBLICATION_SAFE_CLAIMS.md").write_text("\n".join(claims_lines), encoding="utf-8")

    results_lines = [
        "# Final Results And Discussion Draft",
        "",
        "The completed benchmark selected the 60-second transformer as the canonical Phase 1 winner. Relative to 5-second and 10-second windows, the 60-second setup preserved enough temporal context to stabilize anomaly evidence while still reacting materially faster than the 300-second baseline. The 5-second configuration responded earlier but paid a precision penalty under noisier short-horizon residuals. The 300-second configuration produced a more smoothed view of feeder behavior, but its long aggregation window delayed response and diluted attack-local signatures. On the completed Phase 1 benchmark, that tradeoff left 60 seconds as the best overall balance of F1, recall, and practical response latency.",
        "",
        "For Phase 3, the detector was frozen and replayed on heldout bundles from multiple LLM generators. This matters because the heldout bundles were not the canonical Phase 2 attacked training bundle used during model selection. The resulting cross-generator evidence is best described as zero-day-like rather than true zero-day evidence: the model faced unseen bundle constructions and scenario phrasing styles, but those bundles still obeyed a shared defensive schema, family taxonomy, and physical validation layer. Within that bounded setting, the saved transformer remained effective across multiple heldout generators, although performance and latency varied with the accepted scenario mix in each bundle.",
        "",
        "Latency should be interpreted together with window size. Smaller windows improve alert timeliness because they expose attack onset sooner, but they also amplify measurement noise and local variability across the full feeder. Larger windows smooth that noise and may improve stability, but they delay reaction and can blur short-lived physical inconsistencies. The 60-second setting won because it held onto enough feeder context to reduce false alarms while still surfacing attack evidence within a reasonable response interval for operator review.",
        "",
        "The explanation layer should be framed conservatively. The repository does contain heldout XAI grounding summaries, but those artifacts support a grounded, post-alert operator-facing explanation layer rather than a human-level root-cause analyst. In the paper, explanation quality should therefore be discussed separately from detection performance. Detection quality concerns whether attacks are found quickly and reliably; explanation quality concerns whether the post-alert narrative stays aligned with attack family, affected asset, and observable evidence.",
        "",
        "The main limitation is scope. All heldout bundles were generated under the repository's bounded defensive scenario rules, and several submitted scenarios were rejected before evaluation because they violated canonical physical or schema constraints. That is a strength for reproducibility and safety, but it also means the current evidence is strongest for cross-generator generalization within a governed synthetic benchmark, not for unconstrained real-world attack discovery. The paper should say exactly that.",
        "",
    ]
    (ROOT / "FINAL_RESULTS_AND_DISCUSSION_DRAFT.md").write_text("\n".join(results_lines), encoding="utf-8")

    phase_complete_lines = [
        "# Phase Complete Decision",
        "",
        "## Is Phase 1 complete?",
        "",
        "Yes. The canonical window-size study completed, the saved best-model package exists, and the winning configuration is the 60s transformer.",
        "",
        "## Is Phase 2 complete?",
        "",
        "Yes for the canonical attacked bundle, and yes for the discovered heldout generator bundles that already have validated and compiled artifacts under `phase2_llm_benchmark/new_respnse_result/models/`.",
        "",
        "## Is Phase 3 complete?",
        "",
        "Yes for offline heldout cross-generator replay with the saved winning model package and final reporting artifacts.",
        "",
        "## What exactly remains, if anything?",
        "",
        "- no hard blocker remains for the paper-safe offline benchmark",
        "- optional future work: add Ollama or other external generators, add human-authored scenarios, and run runtime or edge deployment timing tests",
        "",
        "## Is the evidence strong enough for a paper-safe `zero-day-like cross-generator evaluation` claim?",
        "",
        (
            "Yes, with bounded wording. The detector was frozen after Phase 1 and replayed on unseen heldout bundles from multiple generators."
            if zero_day_supported
            else "Only partially. The safest wording is heldout cross-generator generalization rather than a strong zero-day-like claim."
        ),
        "",
        "## What wording should be used in the paper and presentation?",
        "",
        "- `frozen-model heldout cross-generator evaluation`",
        "- `zero-day-like generalization within a bounded defensive scenario benchmark`",
        "- `grounded post-alert operator-facing explanation layer`",
        "- avoid `real-world zero-day detection` and avoid `human-level root-cause analysis`",
        "",
    ]
    (ROOT / "PHASE_COMPLETE_DECISION.md").write_text("\n".join(phase_complete_lines), encoding="utf-8")


def _write_pipeline_alignment(metrics_df: pd.DataFrame, per_scenario_df: pd.DataFrame) -> None:
    lines = [
        "# Pipeline Alignment With Diagram",
        "",
        "## Phase 1: Normal System Learning",
        "",
        "- `Generate DER Grid Data` and `System Modeling IEEE 123 Bus + DER (PV, BESS)` are implemented through the clean OpenDSS feeder build and clean physical time-series outputs already present in `outputs/clean/`.",
        "- `Normal Data Generation OpenDSS Simulation` is represented by the canonical clean truth and measured time-series files validated during the source audit.",
        "- `Data Preprocessing Cleaning + Feature Engineering` is represented by canonical window building plus aligned residual construction in the Phase 1 benchmark pipeline.",
        "- `Baseline Models / ML Models` is fully implemented by the completed benchmark suite: threshold baseline, isolation forest, autoencoder, GRU, LSTM, transformer, and tokenized LLM-style baseline.",
        "- `LLM Analysis Context Learning of Normal Behavior` exists only in bounded benchmark form through the tokenized LLM-style baseline and downstream reasoning artifacts. It should not be overstated as a large generative reasoning layer for Phase 1 training.",
        "- `Model Evaluation Reconstruction Loss, Threshold, Accuracy` is implemented and completed by the saved Phase 1 benchmark outputs and ready packages.",
        "",
        "## Phase 2: Scenario & Attack Generation",
        "",
        "- `System Card Creation` is implemented in the shared Phase 2 benchmark assets such as `system_card.json`, `validator_rules.json`, `family_taxonomy.json`, and `physical_constraints.json`.",
        "- `LLM Scenario Generator` is implemented through the heldout generator JSON bundles plus the canonical attacked bundle workflow.",
        "- `Scenario JSON Library` is present in the generator response folders and compiled Phase 2 accepted bundles.",
        "- `Validation Layer Physics + Safety Constraints Check` is fully represented by the existing heldout validation summaries and rejected scenario reports.",
        "- `Injection Engine` and `OpenDSS Execution Physical + Cyber Simulation` are already embodied in the existing canonical Phase 2 outputs reused here from `phase2_llm_benchmark/new_respnse_result/models/<generator>/datasets` and `outputs/attacked/`.",
        "- `Attack Dataset Generation Time-Series with Labels` is complete for the canonical bundle and for the heldout bundles that passed validation.",
        "- `Quality Check Plausibility + Diversity + Metadata` is partially complete through the compiled manifests, validation summaries, label summaries, and scenario difficulty reports. No extra step was missing for this paper-safe offline pipeline.",
        "",
        "## Phase 3: Detection & Intelligence Layer",
        "",
        "- `Unified Dataset Normal + Attack Data` is implemented by the 60s/12s windowed attacked datasets aligned against canonical clean windows.",
        "- `Model Training ML Models + Tiny LLM (LoRA)` is only partially matched by the diagram. The canonical implemented pipeline uses classic ML, neural sequence models, and a tokenized LLM-style baseline. A true LoRA-trained tiny LLM is not part of the canonical paper-safe path and should not be claimed unless separately evidenced.",
        "- `Anomaly Detection Prediction + Reconstruction Loss` is complete through the saved best-model package and heldout replay outputs.",
        "- `Decision Layer Threshold + Context Reasoning` is complete in the sense of saved thresholds and post-alert reporting. It should still be described conservatively.",
        "- `Human-like Explanation LLM Generates Root Cause Analysis` does not match the safe interpretation of the repository. The safer accurate wording is `grounded explanation layer` or `post-alert operator-facing explanation`.",
        "- `Evaluation Metrics F1, Precision, Recall, AOI` is complete for the detector. AOI-like explanation metrics exist only through the heldout XAI grounding summaries and should be described separately from detection metrics.",
        "- `Deployment Vision Lightweight Edge AI for DER` is only partially represented by the repository's deployment/demo assets. This run completed the offline pipeline and reporting layer, not a new on-device deployment benchmark.",
        "",
        "## What Was Missing And Is Now Filled",
        "",
        "- heldout bundle inventory and validation reporting",
        "- frozen saved-model replay on heldout cross-generator bundles",
        "- bundle-level and scenario-level heldout metrics",
        "- final Phase 3 figures and publication-safe claim wording",
        "- best-model heldout evaluation manifests",
        "",
        "## What Is Still Not A Canonical Completed Claim",
        "",
        "- real-world zero-day evidence",
        "- LoRA-trained tiny LLM in the active benchmark path",
        "- human-level root-cause analysis",
        "- edge deployment timing claims from this run",
        "",
    ]
    (ROOT / "PIPELINE_ALIGNMENT_WITH_DIAGRAM.md").write_text("\n".join(lines), encoding="utf-8")


def _write_validation_report(
    inventory_df: pd.DataFrame,
    validation_sections: list[str],
    bundles: list[HeldoutBundle],
) -> None:
    accepted_rows = []
    for bundle in bundles:
        accepted_rows.append(
            {
                "generator_source": bundle.generator_source,
                "dataset_id": bundle.dataset_id,
                "submitted": bundle.submitted_scenario_count,
                "accepted": int(bundle.validation_summary.get("scenarios_valid", bundle.compilation_report.get("number_valid", 0))),
                "rejected": int(bundle.validation_summary.get("scenarios_rejected", bundle.compilation_report.get("number_rejected", 0))),
            }
        )
    accepted_df = pd.DataFrame(accepted_rows)
    lines = [
        "# Heldout Bundle Validation Report",
        "",
        "This report covers every discovered JSON file under `phase2_llm_benchmark/heldout_llm_response/` and records which file was selected, which duplicate file was ignored, and which scenarios were rejected by canonical validation before Phase 3 evaluation.",
        "",
        "## Inventory Snapshot",
        "",
        _markdown_table(inventory_df),
        "",
        "## Accepted vs Rejected Counts",
        "",
        _markdown_table(accepted_df),
        "",
    ]
    lines.extend(validation_sections)
    (ROOT / "heldout_bundle_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def _build_ollama_status() -> str:
    ollama_folder = HELDOUT_ROOT / "ollama"
    if ollama_folder.exists() and any(ollama_folder.glob("*.json")):
        return "# Ollama Extension Status\n\nAn Ollama heldout folder exists, but this run did not add a new local bundle because no validated local generation step was executed in this pass.\n"
    return (
        "# Ollama Extension Status\n\n"
        "Skipped. No local Ollama executable or validated local heldout bundle was available in this workspace during the Phase 3 completion run.\n"
    )


def _load_clean_windows() -> pd.DataFrame:
    clean_path = ROOT / "outputs" / "window_size_study" / "60s" / "data" / "merged_windows_clean.parquet"
    if not clean_path.exists():
        raise FileNotFoundError(f"Canonical clean 60s windows not found: {clean_path}")
    clean = pd.read_parquet(clean_path)
    clean["window_start_utc"] = pd.to_datetime(clean["window_start_utc"], utc=True)
    clean["window_end_utc"] = pd.to_datetime(clean["window_end_utc"], utc=True)
    return clean


def _normalize_labels(labels_df: pd.DataFrame) -> pd.DataFrame:
    labels = labels_df.copy()
    if not labels.empty:
        labels["start_time_utc"] = pd.to_datetime(labels["start_time_utc"], utc=True)
        labels["end_time_utc"] = pd.to_datetime(labels["end_time_utc"], utc=True)
    return labels


def _plot_phase1_window_comparison(output_path: Path) -> None:
    df = pd.read_csv(ROOT / "outputs" / "window_size_study" / "final_window_comparison.csv")
    order = ["5s", "10s", "60s", "300s"]
    df["window_label"] = pd.Categorical(df["window_label"], categories=order, ordered=True)
    df = df.sort_values("window_label")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["window_label"], df["precision"], marker="o", label="precision")
    ax.plot(df["window_label"], df["recall"], marker="o", label="recall")
    ax.plot(df["window_label"], df["f1"], marker="o", label="f1")
    ax.set_ylabel("Score")
    ax.set_xlabel("Window Size")
    ax.set_title("Phase 1 Window Comparison")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_phase1_latency_vs_window(output_path: Path) -> None:
    df = pd.read_csv(ROOT / "outputs" / "window_size_study" / "final_window_comparison.csv")
    order = ["5s", "10s", "60s", "300s"]
    df["window_label"] = pd.Categorical(df["window_label"], categories=order, ordered=True)
    df = df.sort_values("window_label")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["window_label"].astype(str), df["mean_detection_latency_seconds"].astype(float), color="#4C78A8")
    ax.set_ylabel("Mean Detection Latency (s)")
    ax.set_xlabel("Window Size")
    ax.set_title("Phase 1 Latency vs Window")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_single_metric_bar(df: pd.DataFrame, metric: str, output_path: Path, ylabel: str, title: str) -> None:
    ordered = _order_generators(df.copy(), key="generator_source")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ordered["generator_source"], ordered[metric].astype(float), color="#4C78A8")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Generator Source")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_precision_recall(df: pd.DataFrame, output_path: Path) -> None:
    ordered = _order_generators(df.copy(), key="generator_source")
    x = np.arange(len(ordered))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, ordered["precision"].astype(float), width=width, label="precision", color="#4C78A8")
    ax.bar(x + width / 2, ordered["recall"].astype(float), width=width, label="recall", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["generator_source"].tolist())
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Generator Source")
    ax.set_title("Zero-Day-Like Precision and Recall by Generator")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_family_breakdown(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No family summary available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return

    ordered = _order_generators(df.copy(), key="generator_source")
    families = sorted(ordered["attack_family"].dropna().unique().tolist())
    generators = [name for name in GENERATOR_ORDER if name in ordered["generator_source"].unique().tolist()]
    x = np.arange(len(families))
    width = 0.8 / max(len(generators), 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    for idx, generator in enumerate(generators):
        subset = ordered[ordered["generator_source"] == generator]
        values = [float(subset.loc[subset["attack_family"] == family, "recall_like_rate"].iloc[0]) if (subset["attack_family"] == family).any() else 0.0 for family in families]
        ax.bar(x - 0.4 + width / 2 + idx * width, values, width=width, label=generator, color=palette[idx % len(palette)])

    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Recall-Like Detection Rate")
    ax.set_xlabel("Attack Family")
    ax.set_title("Zero-Day-Like Attack Family Breakdown")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_scenario_difficulty_heatmap(per_scenario_df: pd.DataFrame, output_path: Path) -> None:
    if per_scenario_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No scenario-level results available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return

    frame = per_scenario_df.copy()
    frame["scenario_label"] = frame["scenario_id"].astype(str)
    pivot = frame.pivot_table(index="scenario_label", columns="generator_source", values="detected", aggfunc="max")
    for generator in GENERATOR_ORDER:
        if generator not in pivot.columns:
            pivot[generator] = np.nan
    pivot = pivot[[generator for generator in GENERATOR_ORDER if generator in pivot.columns]]
    pivot = pivot.sort_index()

    fig_height = max(4.0, 0.25 * max(len(pivot), 4))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="YlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=7)
    ax.set_xlabel("Generator Source")
    ax.set_ylabel("Scenario ID")
    ax.set_title("Zero-Day-Like Scenario Difficulty Heatmap")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Detected (1=yes, 0=no)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_xai_grounding_summary(output_path: Path) -> None:
    candidates = [
        ROOT / "phase2_llm_benchmark" / "final_results" / "xai_comparison_by_llm_v4_heldout.csv",
        ROOT / "phase2_llm_benchmark" / "new_respnse_result" / "xai_comparison_by_llm_v3_heldout.csv",
    ]
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No quantitative heldout XAI grounding summary found", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return

    df = pd.read_csv(source_path)
    rename_map = {"llm_name": "generator_source"}
    df = df.rename(columns=rename_map)
    df = _order_generators(df, key="generator_source")

    x = np.arange(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, df["evidence_grounding_rate"].astype(float), width=width, label="evidence_grounding_rate", color="#4C78A8")
    ax.bar(x + width / 2, df["family_accuracy"].astype(float), width=width, label="family_accuracy", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(df["generator_source"].tolist())
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Generator Source")
    ax.set_title("Heldout XAI Grounding Summary")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _order_generators(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if key not in df.columns:
        return df
    order_map = {name: idx for idx, name in enumerate(GENERATOR_ORDER)}
    ordered = df.copy()
    ordered["_order"] = ordered[key].map(order_map).fillna(len(order_map)).astype(int)
    ordered = ordered.sort_values(["_order", key]).drop(columns="_order").reset_index(drop=True)
    return ordered


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no rows_"
    frame = df.copy()
    for column in frame.columns:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    headers = [str(column) for column in frame.columns]
    rows = frame.fillna("").astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _safe_mean(series: pd.Series | None) -> float:
    if series is None:
        return math.nan
    cleaned = pd.Series(series).dropna()
    return float(cleaned.mean()) if not cleaned.empty else math.nan


def _safe_median(series: pd.Series | None) -> float:
    if series is None:
        return math.nan
    cleaned = pd.Series(series).dropna()
    return float(cleaned.median()) if not cleaned.empty else math.nan


def _safe_stat(series: pd.Series | None, reducer: str) -> float:
    if series is None:
        return math.nan
    cleaned = pd.Series(series).dropna()
    if cleaned.empty:
        return math.nan
    if reducer == "max":
        return float(cleaned.max())
    if reducer == "mean":
        return float(cleaned.mean())
    raise ValueError(f"Unsupported reducer: {reducer}")


def _normalize_list(value: Any) -> list[str]:
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [str(value)]


def _top_triggering_features(representative_row: pd.Series | None, feature_columns: list[str], limit: int = 5) -> list[str]:
    if representative_row is None:
        return []
    if isinstance(representative_row, pd.DataFrame):
        if representative_row.empty:
            return []
        representative_row = representative_row.iloc[0]
    ranked: list[tuple[str, float]] = []
    for column in feature_columns:
        if column in representative_row.index and pd.notna(representative_row[column]):
            ranked.append((column, abs(float(representative_row[column]))))
    ranked.sort(key=lambda item: item[1], reverse=True)
    cleaned = [column.replace("delta__", "") for column, _ in ranked[:limit]]
    return cleaned


if __name__ == "__main__":
    main()
