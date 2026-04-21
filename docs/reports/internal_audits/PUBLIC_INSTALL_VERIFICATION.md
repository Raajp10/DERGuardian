# Public Install Verification

## Result

- Install method tested: `python -m venv .venv` followed by `python -m pip install -r requirements.txt`
- Python version: `3.11.9`
- Install location: D: drive verification workspace
- Pip cache location: D: drive verification workspace
- Install status: pass
- `pip check` status: pass

## Dependency Outcome

The public `requirements.txt` installed successfully in a fresh virtual environment. `pip check` reported no broken requirements.

Installed major dependency families included:

- NumPy / pandas / pyarrow
- matplotlib
- scikit-learn
- PyTorch
- PyYAML / jsonschema
- opendssdirect.py and DSS backend packages
- pytest
- transformers / PEFT / accelerate / datasets / safetensors / sentencepiece

## Non-Blocking Warning

After upgrading pip inside the new venv, `pip check` emitted a warning about a stale `pip-24.0.dist-info` metadata entry. The command still exited successfully and reported no broken requirements. This appears to be a pip self-upgrade artifact in the local verification environment rather than a DERGuardian dependency conflict.

## `environment.yml`

The conda environment file exists and is syntactically readable as the optional conda path. The primary public install path verified in this pass was the README `pip install -r requirements.txt` path.

