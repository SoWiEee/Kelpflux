#!/usr/bin/env bash
# deploy-2.sh - Converge the live k3s cluster to the final Kelpflux runtime.
#
# Run after scripts/deploy-1.sh. This script intentionally performs each
# deployment action once: build/import the DSAC scheduler image, install or
# upgrade slurm-platform with live DSAC enabled, install or upgrade NVIDIA GPU
# Operator, then wait for the final workloads.
#
# Common knobs:
#   SKIP_BUILD=1          Skip DSAC scheduler docker build.
#   SKIP_IMPORT=1         Skip k3s ctr image import.
#   SKIP_PLATFORM=1       Skip slurm-platform Helm deployment.
#   SKIP_GPU_OPERATOR=1   Skip NVIDIA GPU Operator deployment.
#   SKIP_WAIT=1           Skip rollout/status waits after deployment.
#   NAMESPACE=slurm       Target namespace for platform resources.
#   KUBECONFIG=...        Defaults to ~/.kube/config.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAMESPACE="${NAMESPACE:-slurm}"
KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
export KUBECONFIG

SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_IMPORT="${SKIP_IMPORT:-0}"
SKIP_PLATFORM="${SKIP_PLATFORM:-0}"
SKIP_GPU_OPERATOR="${SKIP_GPU_OPERATOR:-0}"
SKIP_WAIT="${SKIP_WAIT:-0}"

HELM_RELEASE="${HELM_RELEASE:-slurm-platform}"
VALUES_FILE="${VALUES_FILE:-chart/values-k3s.yaml}"
HELM_TIMEOUT="${HELM_TIMEOUT:-10m}"
RL_IMAGE="${RL_IMAGE:-slurm-rl-scheduler:m11}"

GPU_OPERATOR_NAMESPACE="${GPU_OPERATOR_NAMESPACE:-gpu-operator}"
GPU_OPERATOR_RELEASE="${GPU_OPERATOR_RELEASE:-gpu-operator}"
GPU_OPERATOR_VERSION="${GPU_OPERATOR_VERSION:-v26.3.1}"
DEVICE_PLUGIN_CONFIG="${DEVICE_PLUGIN_CONFIG:-slurm-platform-device-plugin-config}"
DEFAULT_CONFIG_KEY="${DEFAULT_CONFIG_KEY:-default}"
MPS_ROOT="${MPS_ROOT:-/run/nvidia/mps}"

# NVIDIA DRA driver (docs/dra-migration.md) — replaces the device-plugin for
# GPU/MPS on pools with useDra=true. Set SKIP_DRA=1 to opt out (device-plugin path).
DRA_DRIVER_VERSION="${DRA_DRIVER_VERSION:-0.4.1}"
SKIP_DRA="${SKIP_DRA:-0}"

log() { printf '[%(%Y-%m-%dT%H:%M:%S%z)T] [deploy-2] %s\n' -1 "$*"; }
warn() { printf '[%(%Y-%m-%dT%H:%M:%S%z)T] [deploy-2][WARN] %s\n' -1 "$*" >&2; }
fail() { printf '[%(%Y-%m-%dT%H:%M:%S%z)T] [deploy-2][ERROR] %s\n' -1 "$*" >&2; exit 1; }

run() {
  log "+ $*"
  "$@"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

sudo_cmd() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

check_prereqs() {
  log "checking deployment prerequisites"
  require_cmd docker
  require_cmd kubectl
  require_cmd helm
  require_cmd k3s
  [[ -r "$KUBECONFIG" ]] || fail "KUBECONFIG is not readable: $KUBECONFIG. Run scripts/deploy-1.sh first or set KUBECONFIG."
  run kubectl get nodes -o wide
}

build_rl_image() {
  if [[ "$SKIP_BUILD" == "1" ]]; then
    warn "SKIP_BUILD=1; skipping DSAC scheduler image build"
    return
  fi

  log "building DSAC scheduler image: $RL_IMAGE"
  run docker build -t "$RL_IMAGE" -f "$ROOT_DIR/services/rl_scheduler/Dockerfile" "$ROOT_DIR"
}

import_rl_image() {
  if [[ "$SKIP_IMPORT" == "1" ]]; then
    warn "SKIP_IMPORT=1; skipping DSAC scheduler image import"
    return
  fi

  log "importing $RL_IMAGE into k3s containerd"
  docker save "$RL_IMAGE" | sudo_cmd k3s ctr images import -
}

adopt_lmod_configmaps() {
  local cm
  for cm in slurm-modulefile-gcc slurm-modulefile-openmpi slurm-modulefile-python3 slurm-modulefile-cuda; do
    if kubectl -n "$NAMESPACE" get configmap "$cm" >/dev/null 2>&1; then
      log "adopting existing Lmod ConfigMap into Helm release: $cm"
      run kubectl -n "$NAMESPACE" label configmap "$cm"         app.kubernetes.io/managed-by=Helm         --overwrite
      run kubectl -n "$NAMESPACE" annotate configmap "$cm"         meta.helm.sh/release-name="$HELM_RELEASE"         meta.helm.sh/release-namespace="$NAMESPACE"         --overwrite
      if [[ "$cm" == "slurm-modulefile-cuda" ]]; then
        kubectl -n "$NAMESPACE" patch configmap "$cm" --type=json -p='[{"op":"remove","path":"/data/stub.lua"}]' >/dev/null 2>&1 || true
      fi
    fi
  done
}

deploy_platform() {
  if [[ "$SKIP_PLATFORM" == "1" ]]; then
    warn "SKIP_PLATFORM=1; skipping slurm-platform Helm deployment"
    return
  fi

  adopt_lmod_configmaps

  log "converging slurm-platform with live DSAC scheduler enabled"
  run helm upgrade --install "$HELM_RELEASE" "$ROOT_DIR/chart"     -f "$ROOT_DIR/$VALUES_FILE"     -n "$NAMESPACE"     --create-namespace     --timeout "$HELM_TIMEOUT"     --wait     --set slurm.jobSubmit.enabled=true     --set gpu.autoLabel=false     --set rlScheduler.enabled=true     --set rlScheduler.lua.enabled=true     --set rlScheduler.shadowMode=false     --set rlScheduler.valueAbstain=-100000     --set rlScheduler.snapshotTtlSeconds=86400

  if [[ "$SKIP_BUILD" != "1" || "$SKIP_IMPORT" != "1" ]]; then
    log "restarting RL deployments to pick up mutable image tag $RL_IMAGE"
    run kubectl -n "$NAMESPACE" rollout restart deployment/rl-scheduler deployment/rl-snapshot-agent deployment/rl-placement-controller
  fi
}

install_gpu_operator() {
  if [[ "$SKIP_GPU_OPERATOR" == "1" ]]; then
    warn "SKIP_GPU_OPERATOR=1; skipping NVIDIA GPU Operator deployment"
    return
  fi

  if ! helm repo list 2>/dev/null | grep -q '^nvidia\b'; then
    log "adding nvidia helm repo"
    run helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
  fi
  log "refreshing nvidia helm repo"
  run helm repo update nvidia

  if ! kubectl get ns "$GPU_OPERATOR_NAMESPACE" >/dev/null 2>&1; then
    log "creating namespace $GPU_OPERATOR_NAMESPACE with privileged pod security"
    run kubectl create namespace "$GPU_OPERATOR_NAMESPACE"
  fi
  run kubectl label ns "$GPU_OPERATOR_NAMESPACE"     pod-security.kubernetes.io/enforce=privileged     pod-security.kubernetes.io/audit=privileged     pod-security.kubernetes.io/warn=privileged     --overwrite

  log "converging NVIDIA GPU Operator $GPU_OPERATOR_VERSION"
  run helm upgrade --install "$GPU_OPERATOR_RELEASE" nvidia/gpu-operator     -n "$GPU_OPERATOR_NAMESPACE"     --version "$GPU_OPERATOR_VERSION"     --timeout "$HELM_TIMEOUT"     --wait     --set driver.enabled=false     --set toolkit.enabled=false     --set devicePlugin.enabled=false     --set devicePlugin.config.name="$DEVICE_PLUGIN_CONFIG"     --set devicePlugin.config.default="$DEFAULT_CONFIG_KEY"     --set mps.root="$MPS_ROOT"     --set dcgmExporter.enabled=true     --set dcgmExporter.serviceMonitor.enabled=false     --set migManager.enabled=false     --set nodeStatusExporter.enabled=false     --set-string 'validator.plugin.env[0].name=WITH_WORKLOAD'     --set-string 'validator.plugin.env[0].value=true'
}

# NVIDIA DRA driver — replaces the device-plugin for GPU/MPS allocation on both
# GPU nodes (docs/dra-migration.md). GPU pools with useDra=true claim the GPU via
# gpu.nvidia.com ResourceClaims instead of nvidia.com/gpu. MPSSupport is gated
# and must be enabled explicitly; ComputeDomains (multi-node NVLink) is off.
install_dra_driver() {
  if [[ "$SKIP_DRA" == "1" ]]; then
    warn "SKIP_DRA=1; skipping NVIDIA DRA driver deployment"
    return
  fi
  log "converging NVIDIA DRA driver $DRA_DRIVER_VERSION"
  run helm upgrade --install dra-driver-nvidia-gpu \
    oci://registry.k8s.io/dra-driver-nvidia/charts/dra-driver-nvidia-gpu \
    --version "$DRA_DRIVER_VERSION" \
    --create-namespace --namespace dra-driver-nvidia-gpu \
    --timeout "$HELM_TIMEOUT" --wait \
    --set gpuResourcesEnabledOverride=true \
    --set resources.computeDomains.enabled=false \
    --set nvidiaDriverRoot=/ \
    --set featureGates.MPSSupport=true
}

wait_for_final_state() {
  if [[ "$SKIP_WAIT" == "1" ]]; then
    warn "SKIP_WAIT=1; skipping final rollout waits"
    return
  fi

  if [[ "$SKIP_PLATFORM" != "1" ]]; then
    log "waiting for slurm-platform workloads"
    run kubectl -n "$NAMESPACE" rollout status deployment/rl-scheduler --timeout=180s
    run kubectl -n "$NAMESPACE" rollout status deployment/rl-snapshot-agent --timeout=180s
    run kubectl -n "$NAMESPACE" rollout status deployment/rl-placement-controller --timeout=180s
    run kubectl -n "$NAMESPACE" rollout status statefulset/slurm-controller --timeout=180s
  fi

  if [[ "$SKIP_GPU_OPERATOR" != "1" ]]; then
    log "checking gpu-operator pods"
    run kubectl -n "$GPU_OPERATOR_NAMESPACE" get pods
  fi
}

summary() {
  log "deployment complete: slurm-platform + GPU Operator + live DSAC scheduler"
  printf '\nUseful checks:\n'
  printf '  kubectl -n %q get pods\n' "$NAMESPACE"
  printf '  kubectl -n %q get pods\n' "$GPU_OPERATOR_NAMESPACE"
  printf '  kubectl -n %q exec slurm-controller-0 -- curl -fsS http://rl-scheduler:8002/healthz\n' "$NAMESPACE"
  printf '  kubectl -n %q logs deploy/rl-snapshot-agent --tail=50\n' "$NAMESPACE"
}

main() {
  cd "$ROOT_DIR"
  check_prereqs
  build_rl_image
  import_rl_image
  deploy_platform
  install_gpu_operator
  install_dra_driver
  wait_for_final_state
  summary
}

main "$@"
