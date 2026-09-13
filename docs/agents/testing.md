# Testing and Validation

Use the repository's existing virtual environment for local Python work. CI workflows are authoritative for isolated dependencies and pinned versions because this repository has no root Python project manifest.

## Python

```bash
PYTHONPATH=. .venv-m11/bin/python -m pytest sim/tests/ -q
PYTHONPATH=. .venv-m11/bin/python -m pytest services/rl_scheduler/tests/ -q
PYTHONPATH=operator .venv-m11/bin/python -m pytest operator/tests/ -q
```

The RL scheduler tests require PyTorch. CI installs a CPU-only build; local training may use the CUDA-enabled `.venv-m11` environment.

## Lua and Helm

```bash
luajit tests/lua/rl_hook_test.lua
luajit tests/lua/submit_helper_test.lua
helm lint chart/
helm unittest chart/
```

`helm unittest` requires the `helm-unittest` plugin. See `.github/workflows/` for the pinned CI versions and complete test matrix. `scripts/verify-live.sh` is an integration check against a running GPU cluster, not a local unit test.
