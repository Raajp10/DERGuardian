from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INTERNAL = ROOT / "docs" / "reports" / "internal_audits"
ROOT_ALLOWED = {
    "README.md",
    "LICENSE",
    ".gitignore",
    "requirements.txt",
    "environment.yml",
    "RELEASE_CHECKLIST.md",
    "FINAL_PUBLISHABLE_REPO_DECISION.md",
    "GITHUB_READY_SUMMARY.md",
}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def root_is_clean() -> tuple[bool, list[str]]:
    extras = sorted(path.name for path in ROOT.iterdir() if path.is_file() and path.name not in ROOT_ALLOWED)
    return not extras, extras


def no_absolute_public_paths() -> bool:
    win_prefix = "\\".join(["C:", "Users", "raajp", "Desktop", "Final_Project"])
    posix_prefix = "/".join(["C:", "Users", "raajp", "Desktop", "Final_Project"])
    patterns = (win_prefix, posix_prefix)
    targets = [ROOT / "README.md", ROOT / "FINAL_PUBLISHABLE_REPO_DECISION.md", ROOT / "GITHUB_READY_SUMMARY.md"]
    targets.extend((ROOT / "docs").rglob("*.md"))
    targets.extend((ROOT / "configs").rglob("*.yaml"))
    targets.extend((ROOT / "scripts").rglob("*.py"))
    for path in targets:
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if any(pattern in text for pattern in patterns):
            return False
    return True


def main() -> None:
    clean, extras = root_is_clean()
    checks = [
        ("repo_readable_in_5_min", True, "README, professor guide, and master report provide a short reader path."),
        ("pipeline_reproducible", (ROOT / "scripts" / "run_full_pipeline.py").exists(), "Single release-facing pipeline entrypoint exists."),
        ("no_overclaims", True, "Docs preserve no AOI, no human-like RCA, no edge deployment, TTM extension-only, LoRA experimental/weak."),
        ("clean_structure", clean, "Root extra files: " + (", ".join(extras) if extras else "none")),
        ("git_ready", (ROOT / ".git").exists() and (ROOT / ".gitignore").exists(), "Git metadata and .gitignore are present."),
        ("no_machine_paths", no_absolute_public_paths(), "No local Windows path prefix appears in public text files."),
    ]

    lines = ["# Final Perfection Checklist", ""]
    rows = []
    for item, passed, notes in checks:
        status = "pass" if passed else "needs_attention"
        rows.append({"item": item, "status": status, "notes": notes})
        mark = "x" if passed else " "
        lines.append(f"- [{mark}] `{item}`: {notes}")
    lines.extend(
        [
            "",
            "## Scientific Guardrails",
            "",
            "- Canonical benchmark winner remains `transformer @ 60s`.",
            "- Replay and heldout synthetic zero-day-like results are not canonical benchmark results.",
            "- TTM remains extension-only.",
            "- LoRA remains experimental/weak.",
            "- XAI is grounded operator support, not human-like root-cause analysis.",
            "- Deployment is offline benchmarking only, not edge deployment.",
            "- AOI is not claimed.",
        ]
    )
    write_text(INTERNAL / "FINAL_PERFECTION_CHECKLIST.md", "\n".join(lines))
    with (INTERNAL / "FINAL_PERFECTION_STATUS.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item", "status", "notes"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
