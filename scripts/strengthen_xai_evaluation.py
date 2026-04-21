"""Strengthen DERGuardian XAI validation without upgrading unsupported claims.

The script reads the existing case-level explanation audit, adds stricter
grounding and partial-credit fields, writes updated metrics/examples, and keeps
the supported language at grounded operator support rather than human-like root
cause analysis.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _safe_mean(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").dropna().mean()) if len(series) else float("nan")


def _load_audit() -> pd.DataFrame:
    path = ROOT / "artifacts" / "xai" / "xai_case_level_audit.csv"
    if not path.exists():
        path = ROOT / "docs" / "reports" / "internal_audits" / "xai_case_level_audit.csv"
    return pd.read_csv(path)


def _augment(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in [
        "family_exact_match",
        "family_top3_match",
        "asset_accuracy",
        "evidence_grounding_overlap",
        "action_relevance_score",
        "partial_alignment_score",
        "unsupported_claim",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["grounding_pass_strict"] = ((out["evidence_grounding_overlap"] >= 0.5) & (out["unsupported_claim"] == 0)).astype(int)
    out["grounding_pass_partial"] = ((out["evidence_grounding_overlap"] > 0.0) & (out["unsupported_claim"] == 0)).astype(int)
    out["asset_full_match"] = (out["asset_accuracy"] >= 0.999).astype(int)
    out["asset_partial_match"] = ((out["asset_accuracy"] > 0.0) & (out["asset_accuracy"] < 0.999)).astype(int)
    out["action_relevance_pass"] = (out["action_relevance_score"] >= 0.5).astype(int)
    out["strict_operator_support_pass"] = (
        (out["family_exact_match"] == 1)
        & (out["asset_accuracy"] >= 0.5)
        & (out["evidence_grounding_overlap"] >= 0.5)
        & (out["action_relevance_score"] >= 0.5)
        & (out["unsupported_claim"] == 0)
    ).astype(int)
    out["operator_support_score"] = (
        0.30 * out["family_exact_match"]
        + 0.20 * out["family_top3_match"]
        + 0.20 * out["asset_accuracy"].clip(0, 1)
        + 0.20 * out["evidence_grounding_overlap"].clip(0, 1)
        + 0.05 * out["action_relevance_score"].clip(0, 1)
        + 0.05 * (1 - out["unsupported_claim"].clip(0, 1))
    )
    out["updated_error_category"] = np.select(
        [
            (out["family_exact_match"] == 1) & (out["asset_accuracy"] < 0.5),
            (out["family_exact_match"] == 1) & (out["evidence_grounding_overlap"] < 0.5),
            (out["family_exact_match"] == 0) & (out["evidence_grounding_overlap"] >= 0.5),
            out["unsupported_claim"] > 0,
            out["strict_operator_support_pass"] == 1,
        ],
        [
            "correct family / weak or wrong asset",
            "correct family / insufficient grounded evidence",
            "wrong family / grounded evidence",
            "unsupported claim",
            "strictly aligned operator support",
        ],
        default="partially aligned",
    )
    out["structured_asset_constraint_note"] = np.where(
        out["asset_accuracy"] >= 0.5,
        "structured asset constraints already support the predicted asset set",
        "asset attribution remains the weakest component; structured candidate limiting may help but is not proven here",
    )
    return out


def _metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("overall", pd.Series(["all"] * len(frame), index=frame.index))]
    if "detector" in frame.columns:
        groups.append(("detector", frame["detector"].astype(str)))
    if "generator_source" in frame.columns:
        groups.append(("generator_source", frame["generator_source"].astype(str)))
    for group_type, labels in groups:
        for label, group in frame.groupby(labels):
            rows.append(
                {
                    "group_type": group_type,
                    "group_name": label,
                    "case_count": int(len(group)),
                    "family_exact_accuracy": _safe_mean(group["family_exact_match"]),
                    "family_top3_accuracy": _safe_mean(group["family_top3_match"]),
                    "asset_accuracy_mean": _safe_mean(group["asset_accuracy"]),
                    "grounding_overlap_mean": _safe_mean(group["evidence_grounding_overlap"]),
                    "grounding_pass_strict_rate": _safe_mean(group["grounding_pass_strict"]),
                    "grounding_pass_partial_rate": _safe_mean(group["grounding_pass_partial"]),
                    "action_relevance_mean": _safe_mean(group["action_relevance_score"]),
                    "unsupported_claim_rate": _safe_mean(group["unsupported_claim"]),
                    "strict_operator_support_pass_rate": _safe_mean(group["strict_operator_support_pass"]),
                    "operator_support_score_mean": _safe_mean(group["operator_support_score"]),
                }
            )
    return pd.DataFrame(rows)


def _write_examples(frame: pd.DataFrame, output_path: Path) -> None:
    examples = []
    wanted = [
        ("strictly aligned operator support", frame.loc[frame["updated_error_category"].eq("strictly aligned operator support")].sort_values("operator_support_score", ascending=False).head(2)),
        ("correct family / weak or wrong asset", frame.loc[frame["updated_error_category"].eq("correct family / weak or wrong asset")].head(2)),
        ("correct family / insufficient grounded evidence", frame.loc[frame["updated_error_category"].eq("correct family / insufficient grounded evidence")].head(2)),
        ("wrong family / grounded evidence", frame.loc[frame["updated_error_category"].eq("wrong family / grounded evidence")].head(2)),
        ("unsupported claim", frame.loc[frame["updated_error_category"].eq("unsupported claim")].head(2)),
    ]
    for title, rows in wanted:
        if rows.empty:
            continue
        examples.append(f"## {title.title()}")
        examples.append("")
        for _, row in rows.iterrows():
            examples.extend(
                [
                    f"- Case: `{row.get('case_id', 'unknown')}`",
                    f"  Target family: `{row.get('target_family', 'unknown')}`; predicted family: `{row.get('predicted_family', 'unknown')}`.",
                    f"  Target assets: `{row.get('target_assets', '')}`; predicted assets: `{row.get('predicted_assets', '')}`.",
                    f"  Grounding overlap: {float(row.get('evidence_grounding_overlap', 0.0)):.3f}; operator-support score: {float(row.get('operator_support_score', 0.0)):.3f}.",
                    f"  Summary excerpt: {str(row.get('operator_summary', '')).strip()[:300]}",
                    "",
                ]
            )
    output_path.write_text("# XAI Updated Qualitative Examples\n\n" + "\n".join(examples) + "\n", encoding="utf-8")


def _write_report(metrics: pd.DataFrame, output_path: Path, audit_path: Path, metrics_path: Path) -> None:
    overall = metrics.loc[metrics["group_type"].eq("overall")].iloc[0]
    lines = [
        "# XAI Strengthening Report",
        "",
        "This pass strengthens the validation layer by adding strict grounding, partial-credit, and operator-support scoring to the existing case-level audit.",
        "It does not claim human-like root-cause analysis.",
        "",
        "## Updated Overall Metrics",
        "",
        f"- Cases audited: {int(overall['case_count'])}",
        f"- Family exact accuracy: {overall['family_exact_accuracy']:.3f}",
        f"- Asset accuracy mean: {overall['asset_accuracy_mean']:.3f}",
        f"- Strict grounding pass rate: {overall['grounding_pass_strict_rate']:.3f}",
        f"- Partial grounding pass rate: {overall['grounding_pass_partial_rate']:.3f}",
        f"- Strict operator-support pass rate: {overall['strict_operator_support_pass_rate']:.3f}",
        f"- Unsupported-claim rate: {overall['unsupported_claim_rate']:.3f}",
        "",
        "## Interpretation",
        "",
        "The strongest supported wording is **grounded operator support** or **structured post-alert explanation**.",
        "The audit still shows asset attribution is materially weaker than family attribution, so human-like root-cause analysis remains unsupported.",
        "",
        "## Artifacts",
        "",
        f"- Updated case audit: `{_repo_rel(audit_path)}`",
        f"- Updated metrics: `{_repo_rel(metrics_path)}`",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figure(metrics: pd.DataFrame, output_path: Path) -> None:
    overall = metrics.loc[metrics["group_type"].eq("overall")].iloc[0]
    labels = [
        "family exact",
        "asset mean",
        "strict grounding",
        "partial grounding",
        "strict support",
    ]
    values = [
        overall["family_exact_accuracy"],
        overall["asset_accuracy_mean"],
        overall["grounding_pass_strict_rate"],
        overall["grounding_pass_partial_rate"],
        overall["strict_operator_support_pass_rate"],
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(labels, values)
    plt.ylim(0, 1.0)
    plt.ylabel("Rate / mean score")
    plt.title("Updated XAI Validation Metrics")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    """Build updated XAI validation artifacts."""

    xai_dir = ROOT / "artifacts" / "xai"
    docs_dir = ROOT / "docs" / "reports"
    figures_dir = ROOT / "docs" / "figures"
    xai_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    updated = _augment(_load_audit())
    metrics = _metric_rows(updated)
    audit_path = xai_dir / "xai_updated_case_level_audit.csv"
    metrics_path = xai_dir / "xai_updated_metrics.csv"
    updated.to_csv(audit_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    _write_examples(updated, docs_dir / "xai_updated_examples.md")
    _write_report(metrics, docs_dir / "xai_strengthening_report.md", audit_path, metrics_path)
    _write_figure(metrics, figures_dir / "xai_updated_comparison.png")
    print({"xai_cases": int(len(updated)), "metric_rows": int(len(metrics))})


if __name__ == "__main__":
    main()
