#!/usr/bin/env bash
# Periodically clear drained/down GPU nodes for the duration of a long live A/B.
#
# WHY: GPU jobs on this cluster intermittently trip Slurm's UnkillableStepTimeout
# during CUDA/MPS teardown → the node is auto-drained with Reason="Kill task
# failed". The harness only calls resume_nodes() ONCE per Slurm config, so a node
# that drains mid-config stays drained for every remaining seed — the run then
# silently proceeds on half a cluster (jobs pend or pile onto one card), which
# both inflates wall-clock and skews the arm comparison. held_watchdog.sh does
# NOT cover this: it only releases HELD jobs.
#
# Usage: node_resume_watchdog.sh <logfile> [interval_s]
set -uo pipefail
LOG="${1:-/tmp/node_resume_wd.log}"
INT="${2:-120}"
NS=slurm; CTL=slurm-controller-0
NODES="slurm-worker-gpu-rtx4070-0 slurm-worker-gpu-rtx3080-0"
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

echo "[$(date +%H:%M:%S)] node_resume_watchdog start (every ${INT}s)" >>"$LOG"
while true; do
  bad=$(timeout 30 kubectl exec -n $NS $CTL -- bash -lc \
        "sinfo -h -N -o '%N|%t' -n \"${NODES// /,}\" 2>/dev/null" 2>/dev/null \
        | awk -F'|' '$2 ~ /drain|down|drng|fail/ {print $1}' | sort -u)
  if [ -n "$bad" ]; then
    for N in $bad; do
      echo "[$(date +%H:%M:%S)] resuming $N (was drained/down)" >>"$LOG"
      timeout 30 kubectl exec -n $NS $CTL -- bash -lc \
        "scontrol update nodename=$N state=resume" >/dev/null 2>&1 || true
    done
  fi
  sleep "$INT"
done
