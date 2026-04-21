#!/usr/bin/env bash
set -euo pipefail

python scripts/run_repo_audit_pass01.py
python scripts/build_phase1_lineage_audit.py
python scripts/run_phase1_ttm_extension.py
python scripts/build_phase2_extensions.py
python scripts/run_phase3_lora_extension.py --force-retrain
python scripts/run_deployment_benchmark.py
python scripts/build_final_audit_reports.py
python scripts/run_final_triple_verification.py
