#!/usr/bin/env bash
# Collect a faithful 168-d raw --online-log for RLPD by running the aimix
# workload live under the learned (RDSAC-cvar) policy while live_daemon logs
# (obs, act, realised-JCT reward, next_obs) to JSONL.
#
#   serve :8003 (rdsac_cvar, obs 168)  = behaviour policy (POST /decide)
#   live_daemon (shadow, sacct-true JCT) = passive logger → shadow_logs/*.jsonl
#   this loop = keep an aimix job stream flowing for DURATION_S
#
# The daemon NEVER issues srun; it only observes squeue/sacct. Completed jobs
# linger in squeue for MinJobAge (300s) so transitions are logged ~300s after a
# job ends — harmless, we harvest the JSONL at the end.
#
#   bash eval/scripts/collect_aimix_onlinelog.sh          # 16h default
#   DURATION_S=3600 N_JOBS=20 bash eval/scripts/collect_aimix_onlinelog.sh
set -uo pipefail
cd /home/acane/Desktop/Kelpflux
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}" PYTHONPATH=. PYTHONUNBUFFERED=1

DURATION_S="${DURATION_S:-57600}"      # 16h
N_JOBS="${N_JOBS:-30}"
ROUNDS="${ROUNDS:-1}"
SERVE="${SERVE:-http://localhost:8003}"
NS=slurm; CTL=slurm-controller-0
LOGIN="${LOGIN_POD:-slurm-login-7f8cfbc48-c875f}"
GPU_NODES="${GPU_NODES:-slurm-worker-gpu-rtx4070-0,slurm-worker-gpu-rtx3080-0}"
MODEL="${MODEL:-/shared/models/qwen05b}"
CVAR_CK="${CVAR_CK:-runs/ckpts_aimix16/rdsac_cvar_s42.pt}"
POLICY_DIR="${POLICY_DIR:-/tmp/aimix_collect_policy}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUNDIR="runs/onlinelog_${STAMP}"; mkdir -p "$RUNDIR" shadow_logs
DLOG="$RUNDIR/daemon.log"; ABLOG="$RUNDIR/ab.log"; LOG="$RUNDIR/driver.log"
export SLURM_EXEC_PREFIX="kubectl exec -n $NS $CTL --"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ── serve up? (behaviour policy) ─────────────────────────────────────────────
ensure_serve(){
  curl -fsS "$SERVE/healthz" >/dev/null 2>&1 && return 0
  log "serve down → starting local 168 serve on :8003"
  mkdir -p "$POLICY_DIR"; cp -f "$CVAR_CK" "$POLICY_DIR/dsac.pt"
  fuser -k 8003/tcp 2>/dev/null && sleep 2
  SHADOW_MODE=true nohup .venv-m11/bin/python -m services.rl_scheduler.serve \
    --policy-dir "$POLICY_DIR" --port 8003 >"$RUNDIR/serve.log" 2>&1 &
  for _ in $(seq 1 30); do curl -fsS "$SERVE/healthz" >/dev/null 2>&1 && break; sleep 2; done
  curl -fsS "$SERVE/healthz" >/dev/null 2>&1 || { log "FATAL serve unreachable"; exit 1; }
}

# ── live_daemon (fixed: 168-d obs, sacct-true JCT) ───────────────────────────
DAEMON_PID=""
ensure_daemon(){
  [ -n "$DAEMON_PID" ] && kill -0 "$DAEMON_PID" 2>/dev/null && return 0
  log "starting live_daemon → shadow_logs/ (log $DLOG)"
  # plain `&` (NOT nohup): $! must be the real python PID so cleanup can kill it.
  # nohup forks a wrapper whose PID != the daemon, orphaning it on cleanup.
  .venv-m11/bin/python -u -m services.rl_scheduler.live_daemon \
    --node-name slurm-worker-gpu-rtx4070-0 slurm-worker-gpu-rtx3080-0 \
    --gpus-per-node 1 --poll-interval 20 --log-dir shadow_logs >>"$DLOG" 2>&1 &
  DAEMON_PID=$!; sleep 3
  kill -0 "$DAEMON_PID" 2>/dev/null || { log "FATAL daemon failed to start"; exit 1; }
  local jf; jf=$(grep -oE 'shadow_logs/transitions_[0-9-]+\.jsonl' "$DLOG" | tail -1)
  log "daemon PID $DAEMON_PID → $jf"; echo "$jf" > "$RUNDIR/jsonl_path"
}

cleanup(){ log "stopping driver; leaving daemon $DAEMON_PID up 400s for trailing completions";
           sleep 400
           if [ -n "$DAEMON_PID" ]; then
             kill "$DAEMON_PID" 2>/dev/null; sleep 2
             kill -0 "$DAEMON_PID" 2>/dev/null && kill -9 "$DAEMON_PID" 2>/dev/null
           fi
           local jf; jf=$(cat "$RUNDIR/jsonl_path" 2>/dev/null);
           log "DONE → online-log: $jf ($(wc -l < "$jf" 2>/dev/null || echo 0) transitions)"; }
trap cleanup EXIT

ensure_serve; ensure_daemon
END=$(( $(date +%s) + DURATION_S ))
log "### collecting for ${DURATION_S}s (until $(date -d @"$END" 2>/dev/null || echo +${DURATION_S}s)); n_jobs=$N_JOBS"

BATCH=0
while [ "$(date +%s)" -lt "$END" ]; do
  BATCH=$((BATCH+1)); ensure_serve; ensure_daemon
  SEED=$(( (RANDOM % 100000) + 1 ))
  log "batch $BATCH (seed $SEED)"
  .venv-m11/bin/python -m eval.scripts.run_heavytail_ab \
    --serve-url "$SERVE" --login-pod "$LOGIN" --namespace "$NS" --controller-pod "$CTL" \
    --family aimix --n-jobs "$N_JOBS" --sigmas 1.0 --rounds "$ROUNDS" --warmup 0 --interleave \
    --aimix-workload --llm-model "$MODEL" --placement --gpu-nodes "$GPU_NODES" \
    --arrival-mode poisson --mps-oversub 2.0 --target-max-s 20 \
    --mps-buckets 25,50,75,100 --partition gpu \
    --rdsac-cvar-ckpt "$CVAR_CK" --seed "$SEED" \
    --out-dir "$RUNDIR/batch_${BATCH}_s${SEED}" >>"$ABLOG" 2>&1 \
    || log "batch $BATCH exit $? (continuing)"
  jf=$(cat "$RUNDIR/jsonl_path" 2>/dev/null)
  log "batch $BATCH done; online-log so far: $(wc -l < "$jf" 2>/dev/null || echo 0) transitions"
done
log "### duration reached after $BATCH batches"
