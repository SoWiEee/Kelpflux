# Agent Guide

Kelpflux is a cloud-native Slurm-on-Kubernetes platform for elastic, shared GPU AI workloads using NVIDIA MPS and learning-assisted scheduling.

## Tooling

- Python dependency tooling uses `uv` in CI; dependencies are component-specific, and local simulation/DRL work uses `.venv-m11`.
- Run Python commands from the repository root with `PYTHONPATH=.` unless a test workflow specifies otherwise.
- CI lint: `uvx ruff@0.4.7 check operator/ sim/ services/ eval/`.
- There is no Node/npm build pipeline or project-wide typecheck command.

## Guidance

- [Testing and validation](docs/agents/testing.md)
- [Cluster operations](docs/agents/operations.md)
- [Scheduling research](docs/agents/research.md)

Do not deploy, tear down, or change live Kubernetes, Slurm, host, or GPU state unless the user explicitly requests it.
