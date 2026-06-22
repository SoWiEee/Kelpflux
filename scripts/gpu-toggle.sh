#!/usr/bin/env bash
# Toggle node-1 (acane / RTX 4070) GPU between the k3s cluster (MPS/scheduling)
# and free-for-gaming. Uses the gpu-operator's per-node disable label
# `nvidia.com/gpu.deploy.operands` — which the operator RESPECTS (GFD won't
# revert it, unlike nvidia.com/mps.capable). Releasing also kills the MPS
# server (it holds the GPU's Exclusive_Process context) and sets DEFAULT
# compute mode so games can create graphics/compute contexts.
#
# Usage:
#   scripts/gpu-toggle.sh release   # free the 4070 for gaming
#   scripts/gpu-toggle.sh restore   # give the 4070 back to the cluster (MPS)
#   scripts/gpu-toggle.sh status    # show current state
#
# Note: node-2 (RTX 3080) is untouched — the cluster keeps running there.
set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
NODE="${GPU_NODE:-acane}"
MODE="${1:-status}"

mps_pods_on_node() {
  kubectl get pods -n gpu-operator -o wide 2>/dev/null \
    | grep " ${NODE} " | grep -iE 'mps-control|device-plugin-daemonset|dcgm-exporter' \
    | grep -ivE 'Completed|Terminating'
}

show_status() {
  echo "── ${NODE} / RTX 4070 status ──"
  nvidia-smi --query-gpu=name,compute_mode,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /'
  local lbl
  lbl=$(kubectl get node "$NODE" -o jsonpath='{.metadata.labels.nvidia\.com/gpu\.deploy\.operands}' 2>/dev/null)
  echo "  operands label: ${lbl:-<unset>}"
  echo "  compute apps:"
  nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/    /' || echo "    (none)"
  if mps_pods_on_node >/dev/null 2>&1 && [ -n "$(mps_pods_on_node)" ]; then
    echo "  cluster GPU operands: ON  → cluster owns the GPU"
  else
    echo "  cluster GPU operands: OFF → GPU released for gaming"
  fi
}

case "$MODE" in
  release|game|free)
    echo "Releasing ${NODE}'s 4070 for gaming…"
    kubectl label node "$NODE" nvidia.com/gpu.deploy.operands=false --overwrite
    echo "  waiting for GPU operands (device-plugin / mps-control / dcgm) to evict…"
    for i in $(seq 1 24); do
      [ -z "$(mps_pods_on_node)" ] && { echo "  operands evicted"; break; }
      sleep 5
    done
    sudo pkill -9 -f nvidia-cuda-mps-server 2>/dev/null || true
    sleep 2
    sudo nvidia-smi -c 0 >/dev/null 2>&1 || true   # DEFAULT compute mode
    echo "✅ released."
    show_status
    ;;
  restore|cluster|on)
    echo "Restoring ${NODE}'s 4070 to the cluster (MPS)…"
    kubectl label node "$NODE" nvidia.com/gpu.deploy.operands- 2>/dev/null || true
    echo "  waiting for mps-control-daemon to return (sets Exclusive_Process + starts MPS)…"
    for i in $(seq 1 36); do
      if kubectl get pods -n gpu-operator -o wide 2>/dev/null \
          | grep " ${NODE} " | grep mps-control | grep -q '2/2 *Running'; then
        echo "  mps-control-daemon ready"; break
      fi
      sleep 6
    done
    sleep 4
    echo "✅ restored. (verify MPS with a 4-concurrent --gres=mps:25 job before the eval)"
    show_status
    ;;
  status|*)
    show_status
    echo "Usage: $0 {release|restore|status}"
    ;;
esac
