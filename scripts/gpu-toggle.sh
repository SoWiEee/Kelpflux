#!/usr/bin/env bash
# Toggle node-1 (acane / RTX 4070) GPU between the k3s + DRA cluster and
# free-for-gaming.
#
# Under DRA (docs/dra-migration.md) the GPU is held by the
# slurm-worker-gpu-rtx4070 pod's ResourceClaim (gpu.nvidia.com, MPS) plus a
# persistent MPS keepalive. So "release" = cordon the node + evict the worker
# pod: with the node cordoned the pod stays Pending (the elastic operator keeps
# replicas=1 but cannot place it), which releases the DRA claim → the DRA driver
# tears down the per-pod MPS server. We then reset compute mode to DEFAULT so a
# game can create graphics/compute contexts.
#
#   scripts/gpu-toggle.sh release   # free the 4070 for gaming
#   scripts/gpu-toggle.sh restore   # give the 4070 back to the cluster (DRA)
#   scripts/gpu-toggle.sh status    # show current state
#
# node-2 (RTX 3080) is untouched — the cluster keeps running there.
#
# NOTE: the pre-DRA implementation toggled the gpu-operator
# `nvidia.com/gpu.deploy.operands` label; that no longer frees the GPU because
# under DRA the device is claimed by the worker pod, not the device-plugin.
set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
NODE="${GPU_NODE:-acane}"
NS="${SLURM_NS:-slurm}"
STS="${GPU_STS:-slurm-worker-gpu-rtx4070}"
POD="${STS}-0"
CTL="${SLURM_CTL:-slurm-controller-0}"
MODE="${1:-status}"

show_status() {
  echo "── ${NODE} / RTX 4070 (DRA) status ──"
  nvidia-smi --query-gpu=name,compute_mode,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /'
  local cord phase
  cord=$(kubectl get node "$NODE" -o jsonpath='{.spec.unschedulable}' 2>/dev/null)
  phase=$(kubectl get pod -n "$NS" "$POD" -o jsonpath='{.status.phase}' 2>/dev/null)
  echo "  cordoned: ${cord:-false}"
  echo "  worker pod: ${phase:-<none>}"
  echo "  compute apps:"
  nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/    /' || echo "    (none)"
  if [ "${cord:-false}" = "true" ] || [ -z "$phase" ] || [ "$phase" != "Running" ]; then
    echo "  → released for gaming"
  else
    echo "  → owned by the cluster (DRA claim active)"
  fi
}

case "$MODE" in
  release|game|free)
    echo "Releasing ${NODE}'s 4070 for gaming (DRA)…"
    kubectl cordon "$NODE"
    # Drain the Slurm node first so no job lands mid-release.
    kubectl exec -n "$NS" "$CTL" -- scontrol update nodename="$POD" state=drain reason=gaming 2>/dev/null || true
    # Evict the worker pod → releases the DRA claim → DRA tears down the MPS server.
    kubectl delete pod -n "$NS" "$POD" --grace-period=10 2>/dev/null || true
    echo "  waiting for the DRA claim / MPS to release…"
    for i in $(seq 1 24); do
      phase=$(kubectl get pod -n "$NS" "$POD" -o jsonpath='{.status.phase}' 2>/dev/null)
      [ "$phase" != "Running" ] && { echo "  worker pod not running (${phase:-gone})"; break; }
      sleep 5
    done
    sleep 3
    sudo nvidia-smi -c 0 >/dev/null 2>&1 || true   # DEFAULT compute mode for gaming
    echo "✅ released."
    show_status
    ;;
  restore|cluster|on)
    echo "Restoring ${NODE}'s 4070 to the cluster (DRA)…"
    kubectl uncordon "$NODE"
    echo "  waiting for the worker pod to schedule + become ready (DRA re-claims the GPU)…"
    for i in $(seq 1 36); do
      [ "$(kubectl get pod -n "$NS" "$POD" -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null)" = "true" ] && { echo "  worker ready"; break; }
      sleep 6
    done
    kubectl exec -n "$NS" "$CTL" -- bash -lc "scontrol update nodename=$POD state=resume 2>/dev/null; scontrol reconfigure 2>/dev/null" 2>/dev/null || true
    echo "✅ restored. (MPS keepalive re-warms the server; verify with 4 concurrent --gres=mps:25 jobs before heavy use.)"
    show_status
    ;;
  status|*)
    show_status
    echo "Usage: $0 {release|restore|status}"
    ;;
esac
