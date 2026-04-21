# Git Ignore List

These files/folders/patterns should remain ignored or be published only as external release assets.

## Large Generated Data

- `outputs/` - full generated clean/attacked/window/model artifacts; source of truth locally, too large/noisy for normal Git.
- `data/raw/`, `data/full/` - raw/full data if later added.
- `outputs/clean/*.parquet`, `outputs/attacked/*.parquet`, `outputs/windows/*.parquet` - large generated time-series tables.

## Runtime, Demo, And Legacy Workspaces

- `deployment_runtime/` - local deployment runtime artifacts and transient outputs.
- `demo/` - demo packages and generated demo outputs.
- `DERGuardian_professor_demo/` - local professor demo folder.
- `paper_figures/` - legacy/local figure workspace; final selected figures live under `docs/figures/`.
- `reports/` - legacy/local report workspace; final reports live under `docs/reports/`.
- `phase2_llm_benchmark/` - large LLM benchmark workspace; final selected evidence is mirrored into `docs/` and `artifacts/`.

## Model And Archive Artifacts

- `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors` - checkpoints/model weights.
- `*.zip`, `*.tar`, `*.tar.gz`, `*.7z` - archives.
- `demo.zip`, `Final_Project.zip` - local archives.

## Caches And Local State

- `__pycache__/`, `*.pyc`, `*.pyo`
- `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`
- `.ipynb_checkpoints/`
- `.venv/`, `venv/`, `env/`, `.env`
- `.vscode/`, `.idea/`
- `.claude/` - local assistant/tool state.
- `.DS_Store`, `Thumbs.db`

## Root-Level Generated Clutter

- Root-level generated `*.csv`, `*.png`, `*.json`, `*.yaml`, `*.sh`, `*.txt`, and `*.md` should stay ignored except for the explicit public-release whitelist in `.gitignore`.
- Use `docs/`, `artifacts/`, and the top-level final release docs as the public-facing surface.
