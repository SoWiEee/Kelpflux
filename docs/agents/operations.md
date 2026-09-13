# Cluster Operations

The maintained deployment path targets the Linux + k3s + NVIDIA GPU environment documented in [cluster.md](../cluster.md). The Helm chart is under `chart/`; `chart/values-k3s.yaml` is the deployment overlay used by the repository scripts.

For a fresh host, follow [tutorial.md](../tutorial.md) and run the deployment scripts in order:

```bash
export KUBECONFIG=~/.kube/config
bash scripts/deploy-1.sh
bash scripts/deploy-2.sh
bash scripts/verify-live.sh
```

`deploy-2.sh` enables the RL scheduler. With `rlScheduler.enabled=true`, the Helm default enables `rlScheduler.placementController`; its `shadow` default is `false`, so it applies placements rather than only logging. It acts only on held pending jobs (submitted with `sbatch --hold`). Keep operational claims aligned with `chart/values.yaml` and the selected overlay.

Before changing chart behavior, inspect rendered output with `helm template` and update the relevant Helm tests. Live deploy, teardown, and host/GPU toggle commands require explicit user authorization.
