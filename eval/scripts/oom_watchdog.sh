#!/usr/bin/env bash
# OOM watchdog for the real-CUDA aimix step3 campaign.
#
# WHY: node-2 (rtx3080 worker) has only ~7.5 GB host RAM. Under real-CUDA aimix with
# qwen05b LLM co-residency it gets OOMKilled (see memory project-node2-hw-constraints).
# When the worker pod restarts, Slurm marks the node down/drain and every srun that was
# on it goes PENDING with reason "launch failed requeued held" — and stays held forever.
# The reorder/backfill eval arms submit jobs UNHELD, so a held job is ALWAYS an OOM
# casualty here; its is_done() then never fires (a held job never leaves squeue) and the
# whole campaign hangs (this is exactly what stalled the 04:07 run for 15 h).
#
# WHAT: poll the controller every INTERVAL s and self-heal:
#   1. resume any GPU node stuck in down*/drain*/fail*/unk* (so releases can actually land);
#   2. `scontrol release` any PENDING job whose reason matches "launch failed" (case-insens),
#      so it re-runs once node-2 is back; cap per-job releases at MAX_RELEASE, then scancel
#      it (a few dropped jobs out of 150 don't bias JCT and let is_done() finish) — better
#      than an infinite hang.
#
# It is READ-MOSTLY on healthy runs (no held jobs → nothing to do) and FAIL-SAFE: it only
# ever releases/cancels launch-failed held jobs and resumes downed nodes; it never touches
# a normally running/pending job. Stop it by `touch $STOPFILE` or kill.
#
#   INTERVAL=15 MAX_RELEASE=4 STOPFILE=/tmp/oom_watchdog.stop \
#     bash eval/scripts/oom_watchdog.sh   # runs until stopfile appears
set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
NS=slurm; CTL=slurm-controller-0
INTERVAL="${INTERVAL:-15}"
MAX_RELEASE="${MAX_RELEASE:-4}"
STOPFILE="${STOPFILE:-/tmp/oom_watchdog.stop}"
LOGF="${LOGF:-/tmp/oom_watchdog.log}"
GPU_NODES="${GPU_NODES:-slurm-worker-gpu-rtx4070-0 slurm-worker-gpu-rtx3080-0}"
declare -A RELCOUNT

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$LOGF"; }
ctl(){ kubectl exec -n "$NS" "$CTL" -- bash -lc "$1" 2>/dev/null; }

log "watchdog START interval=${INTERVAL}s max_release=${MAX_RELEASE} stopfile=$STOPFILE"
while [ ! -f "$STOPFILE" ]; do
  # (1) resume any GPU node not in a schedulable state
  states=$(ctl "sinfo -h -N -n '$(echo $GPU_NODES | tr ' ' ',')' -o '%N|%t'")
  while IFS='|' read -r node st; do
    [ -z "$node" ] && continue
    case "$st" in
      idle|mix|alloc|allocated|comp) : ;;   # healthy
      "") : ;;
      *) log "node $node state=$st → resume"
         ctl "scontrol update nodename=$node state=resume" >/dev/null || true ;;
    esac
  done <<< "$states"

  # (2) release launch-failed held jobs (OOM casualties); cap then scancel
  # reason field via %r; match 'launch failed' case-insensitively.
  held=$(ctl "squeue -h -t PENDING -o '%i|%r'")
  while IFS='|' read -r jid reason; do
    [ -z "$jid" ] && continue
    lc=$(echo "$reason" | tr '[:upper:]' '[:lower:]')
    case "$lc" in
      *"launch failed"*|*"launchfailure"*|*"joblaunchfailure"*)
        n=${RELCOUNT[$jid]:-0}
        if [ "$n" -ge "$MAX_RELEASE" ]; then
          log "job $jid launch-failed x$n ≥ $MAX_RELEASE → scancel (drop)"
          ctl "scancel $jid" >/dev/null || true
        else
          RELCOUNT[$jid]=$((n+1))
          log "job $jid held ('$reason') → release (#$((n+1)))"
          ctl "scontrol release $jid" >/dev/null || true
        fi
        ;;
    esac
  done <<< "$held"

  sleep "$INTERVAL"
done
log "watchdog STOP (stopfile present)"
