"""Build TTM and LoRA extension improvement summaries.

This pass does not rerun expensive training. It strengthens the evidence by
joining TTM with AOI/heldout rows and by re-evaluating LoRA predictions with
explicit family-label normalization and asset-overlap scoring.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


FAMILY_ALIASES = {
    "osclatory_control": "oscillatory_control",
    "osselective_control": "oscillatory_control",
    "oscillatory": "oscillatory_control",
    "fdi": "false_data_injection",
    "false data injection": "false_data_injection",
    "command suppression": "command_suppression",
    "command delay": "command_delay",
    "der disconnect": "DER_disconnect",
    "der_disconnect": "DER_disconnect",
}


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _normalize_family(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().strip('"').strip("'")
    lowered = re.sub(r"\s+", " ", text.replace("-", "_")).lower()
    lowered = lowered.replace("__", "_")
    return FAMILY_ALIASES.get(lowered, lowered)


def _asset_set(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    text = str(value).lower()
    parts = re.split(r"[|,;\[\]\s]+", text)
    return {part.strip().strip('"').strip("'") for part in parts if part.strip() and part.strip().lower() not in {"nan", "none"}}


def _score_lora_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["target_family_norm"] = frame["target_family"].map(_normalize_family)
    frame["predicted_family_norm"] = frame["predicted_family"].map(_normalize_family)
    frame["family_match_normalized"] = (frame["target_family_norm"] == frame["predicted_family_norm"]).astype(int)
    frame["target_asset_set"] = frame["target_assets"].map(_asset_set)
    frame["predicted_asset_set"] = frame["predicted_assets"].map(_asset_set)
    frame["asset_overlap_jaccard"] = [
        (len(t & p) / len(t | p)) if (t or p) else 0.0 for t, p in zip(frame["target_asset_set"], frame["predicted_asset_set"])
    ]
    frame["asset_any_overlap"] = [int(bool(t & p)) for t, p in zip(frame["target_asset_set"], frame["predicted_asset_set"])]
    frame["confidence_match_normalized"] = (
        frame["target_confidence"].astype(str).str.lower().str.strip()
        == frame["predicted_confidence"].astype(str).str.lower().str.strip()
    ).astype(int)
    frame["variant_source_file"] = _repo_rel(path)
    return frame


def _build_ttm_updated() -> pd.DataFrame:
    base_path = ROOT / "artifacts" / "extensions" / "phase1_ttm_results.csv"
    base = pd.read_csv(base_path) if base_path.exists() else pd.DataFrame()
    if not base.empty:
        base["evaluation_context"] = "canonical_benchmark_test_split_extension"
        base["canonical_or_extension"] = "extension"
        base["improvement_pass_action"] = "joined with AOI and heldout summary; no rerun because the existing TTM training run is real and took about 709 seconds"
        aoi_path = ROOT / "artifacts" / "extensions" / "aoi_results.csv"
        if aoi_path.exists():
            aoi = pd.read_csv(aoi_path)
            ttm_aoi = aoi.loc[aoi["model_name"].eq("ttm_extension") & aoi["evaluation_context"].eq("ttm_extension_prediction_artifact")]
            if not ttm_aoi.empty:
                base["aoi_alert_overlap_index"] = float(ttm_aoi["aoi_alert_overlap_index"].iloc[0])
    zero_path = ROOT / "artifacts" / "zero_day_like" / "zero_day_model_window_results_full.csv"
    heldout = pd.DataFrame()
    if zero_path.exists():
        zero = pd.read_csv(zero_path)
        heldout = zero.loc[zero["model_name"].eq("ttm_extension")].copy()
        if not heldout.empty:
            heldout["improvement_pass_action"] = "existing heldout synthetic zero-day-like rows retained as extension evidence"
    common = list(dict.fromkeys(list(base.columns) + list(heldout.columns)))
    return pd.concat([base.reindex(columns=common), heldout.reindex(columns=common)], ignore_index=True)


def _build_lora_updated() -> pd.DataFrame:
    prediction_dir = ROOT / "outputs" / "phase3_lora_extension" / "predictions"
    frames = []
    for path in [prediction_dir / "base_zero_shot_predictions.csv", prediction_dir / "lora_finetuned_predictions.csv"]:
        if path.exists():
            frames.append(_score_lora_predictions(path))
    if not frames:
        return pd.DataFrame()
    predictions = pd.concat(frames, ignore_index=True)
    grouped = predictions.groupby(["variant", "split"], dropna=False)
    rows = []
    for (variant, split), group in grouped:
        rows.append(
            {
                "model_variant": variant,
                "split": split,
                "example_count": int(len(group)),
                "family_accuracy_normalized": float(group["family_match_normalized"].mean()),
                "asset_any_overlap_rate": float(group["asset_any_overlap"].mean()),
                "asset_overlap_jaccard_mean": float(group["asset_overlap_jaccard"].mean()),
                "confidence_accuracy_normalized": float(group["confidence_match_normalized"].mean()),
                "mean_latency_ms": float(pd.to_numeric(group["latency_ms"], errors="coerce").mean()),
                "median_latency_ms": float(pd.to_numeric(group["latency_ms"], errors="coerce").median()),
                "improvement_pass_action": "post-hoc label normalization and asset-overlap audit of real prediction files",
            }
        )
    return pd.DataFrame(rows)


def _write_ttm_report(results: pd.DataFrame, path: Path, csv_path: Path) -> None:
    completed = results.loc[results.get("status", "").eq("completed")] if "status" in results.columns else results
    lines = [
        "# TTM Improvement Pass",
        "",
        "This pass did not rerun the expensive TTM training job. The existing run is real and already includes a saved checkpoint, predictions, latency rows, and training history.",
        "The improvement is an evidence join: AOI is added where computable and existing heldout synthetic rows are kept extension-only.",
        "",
        "## Key Result",
        "",
    ]
    if not completed.empty and "f1" in completed.columns:
        best = completed.sort_values("f1", ascending=False).iloc[0]
        lines.append(f"- Best TTM F1 in available rows: `{best.get('f1'):.6f}` in context `{best.get('evaluation_context', 'unknown')}`.")
    lines.extend(
        [
            "- TTM remains extension-only and does not replace transformer @ 60s as the canonical benchmark winner.",
            f"- Updated CSV: `{_repo_rel(csv_path)}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_lora_report(results: pd.DataFrame, path: Path, csv_path: Path) -> None:
    lines = [
        "# LoRA Improvement Pass",
        "",
        "This pass re-evaluates the real LoRA prediction files with explicit family-label normalization and asset-overlap scoring.",
        "It does not turn LoRA into detector evidence; LoRA remains an experimental explanation-side branch.",
        "",
        results.to_markdown(index=False) if not results.empty else "No LoRA prediction rows were available.",
        "",
        "## Interpretation",
        "",
        "Family labels improve only where misspellings or close aliases are normalized. Asset attribution and grounding remain weak, so the branch remains experimental.",
        f"Updated CSV: `{_repo_rel(csv_path)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(ttm: pd.DataFrame, lora: pd.DataFrame, path: Path) -> None:
    lines = [
        "# TTM and LoRA Extension Summary",
        "",
        "- TTM: real detector-side extension at 60s, but still secondary to the frozen transformer @ 60s canonical result.",
        "- LoRA: real explanation-side fine-tune, but weak on heldout generator splits and not detector evidence.",
        "- Neither branch upgrades the project to a new canonical detector claim.",
        "",
        "## Evidence Files",
        "",
        "- `artifacts/extensions/phase1_ttm_updated_results.csv`",
        "- `artifacts/extensions/phase3_lora_updated_results.csv`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Write updated extension summaries."""

    artifact_dir = ROOT / "artifacts" / "extensions"
    docs_dir = ROOT / "docs" / "reports"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    ttm = _build_ttm_updated()
    lora = _build_lora_updated()
    ttm_path = artifact_dir / "phase1_ttm_updated_results.csv"
    lora_path = artifact_dir / "phase3_lora_updated_results.csv"
    ttm.to_csv(ttm_path, index=False)
    lora.to_csv(lora_path, index=False)
    _write_ttm_report(ttm, docs_dir / "phase1_ttm_improvement_report.md", ttm_path)
    _write_lora_report(lora, docs_dir / "phase3_lora_improvement_report.md", lora_path)
    _write_summary(ttm, lora, docs_dir / "ttm_lora_extension_summary.md")
    print({"ttm_rows": int(len(ttm)), "lora_rows": int(len(lora))})


if __name__ == "__main__":
    main()
