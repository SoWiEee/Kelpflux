# Cluster Operations

The maintained deployment path targets the Linux + k3s + NVIDIA GPU environment documented in [cluster.md](../cluster.md). `deploy-2.sh` layers `chart/values-2x1.yaml` over `chart/values-k3s.yaml` by default.

For a fresh host, follow [tutorial.md](../tutorial.md) and run the deployment scripts in order:

```bash
export KUBECONFIG=~/.kube/config
bash scripts/deploy-1.sh
bash scripts/deploy-2.sh
bash scripts/verify-live.sh
```

The 2x1 overlay enables `rlScheduler.reorderDaemon` for unheld, explicit-MPS jobs and disables held-job REST placement until Slurm 23.11 `required_nodes` is verified end to end. The daemon changes priority only; Slurm selects the node and dispatch time.

Before changing chart behavior, inspect rendered output with `helm template` and update the relevant Helm tests. Live deploy, teardown, and host/GPU toggle commands require explicit user authorization.
