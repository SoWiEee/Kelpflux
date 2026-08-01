#!/usr/bin/env bash
# Heavy-load (§6.3 future-work) live A/B: revalidate whether learned placement beats
# score in the REGIME WHERE §5.7's ceiling analysis says headroom opens (n_jobs≈125,
# 2-GPU cluster → ~10% headroom over score, of which only ~16% is ε-reachable).
#
# Arms (5-6, NO CrossQ / NO RLPD): fcfs, backfill (naive Slurm, no Lua) + score
# (backfill+multifactor+score-Lua baseline) + learned placement SAC / RDSAC-mean /
# RDSAC-cvar. Metrics recorded per arm match the §5.3 hybrid round: JCT / Makespan /
# P95 / P99 (all already emitted by run_heavytail_ab panels). Primary significance
# test = seed-level paired ΔJCT% vs score (one-sample t over training seeds).
#
# Everything is on ONE DRA MPS backend so all arms are comparable to the SAME score
# baseline (the device-plugin-vs-DRA confound lesson from §5.3). Three Slurm configs,
# restored on exit via trap.
#
# Env knobs (defaults target the real 23-seed campaign):
#   SEEDS="42 .. 64"  N_JOBS=125  ROUNDS=1  OVERSUB=2.0  LEARNED="sac rdsac_mean rdsac_cvar"
#   SMOKE=1 → tiny end-to-end check (N_JOBS=6, ROUNDS=1, SEEDS="42") to prove wiring.
#
# Usage:
#   SEEDS="$(seq 42 64)" bash eval/scripts/run_heavy125_5arm.sh          # full 23-seed
#   SMOKE=1 bash eval/scripts/run_heavy125_5arm.sh                       # wiring smoke
set -uo pipefail
cd /home/acane/Desktop/Kelpflux
export KUBECONFIG=$HOME/.kube/config PYTHONPATH=.
STAMP=$(date +%Y%m%d-%H%M%S)
TAG="${TAG:-heavy125}"
LOG="runs/${TAG}_${STAMP}.log"
CM=slurm-config-static; NS=slurm; CTL=slurm-controller-0
LOGIN="${LOGIN_POD:-slurm-login-7f8cfbc48-c875f}"
GPU_NODES="slurm-worker-gpu-rtx4070-0,slurm-worker-gpu-rtx3080-0"
SERVE="${SERVE:-http://localhost:8003}"; MODEL="${MODEL:-/shared/models/qwen05b}"
CK="${CK:-/tmp/lckpts}"
# learned arms (space-separated ckpt prefixes in $CK). Default = the 5-arm spec
# (fcfs/backfill/score/sac/rdsac_cvar); add rdsac_mean for the 6-arm §5.3 parity set.
read -r -a LEARNED <<< "${LEARNED:-sac rdsac_cvar}"

# NOTE: SEEDS must word-split on BOTH spaces and newlines. `read -r -a <<<` only
# reads the first line of a here-string, so a newline-separated SEEDS (e.g. the
# `$(seq 42 64)` default, or `SEEDS="$(seq 42 64)"`) would collapse to just the
# first seed. Unquoted array expansion splits on IFS (space+tab+newline) → correct.
# shellcheck disable=SC2206
if [ "${SMOKE:-0}" = "1" ]; then
  N_JOBS="${N_JOBS:-6}"; ROUNDS="${ROUNDS:-1}"; SEEDS=( ${SEEDS:-42} )
  log_prefix="[SMOKE] "
else
  N_JOBS="${N_JOBS:-125}"; ROUNDS="${ROUNDS:-1}"; SEEDS=( ${SEEDS:-$(seq 42 64)} )
  log_prefix=""
fi
OVERSUB="${OVERSUB:-2.0}"
ORIG=/tmp/slurm.conf.h125orig.$$
WD_PID=""
PATCHED=0   # only restore/restart slurmctld if we actually changed the config
mkdir -p runs
log(){ echo "[$(date +%H:%M:%S)] ${log_prefix}$*" | tee -a "$LOG"; }
patch_conf(){ kubectl patch cm -n $NS $CM --type merge -p "$(python3 -c "import json;print(json.dumps({'data':{'slurm.conf':open('$1').read()}}))")" >/dev/null; }
restart_ctl_wait(){ kubectl delete pod -n $NS $CTL >/dev/null 2>&1; for i in $(seq 1 30); do [ "$(kubectl get pod -n $NS $CTL -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null)" = "true" ] && return 0; sleep 6; done; return 1; }
resume_nodes(){ kubectl exec -n $NS $CTL -- bash -lc 'scontrol reconfigure 2>/dev/null; for N in slurm-worker-gpu-rtx4070-0 slurm-worker-gpu-rtx3080-0; do for i in 1 2 3; do scontrol update nodename=$N state=resume 2>/dev/null; sleep 3; done; done' >/dev/null 2>&1 || true; }
restore_orig(){ [ "$PATCHED" = "1" ] || { log "no config change to restore"; return 0; }; log "RESTORING original slurm.conf"; patch_conf "$ORIG"; restart_ctl_wait && log "cluster restored" || log "WARN restore not ready"; resume_nodes; }
cleanup(){ [ -n "$WD_PID" ] && kill "$WD_PID" 2>/dev/null; restore_orig; }
trap 'cleanup' EXIT

# ----- preflight: fail loud & early instead of hanging on a half-up cluster -----
preflight(){
  local bad=0
  curl -fsS --max-time 5 "$SERVE/healthz" >/dev/null 2>&1 || { log "PREFLIGHT FAIL: serve unreachable at $SERVE (start /tmp/lserve.py)"; bad=1; }
  local r4070; r4070=$(kubectl get pod -n $NS slurm-worker-gpu-rtx4070-0 -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null)
  [ "$r4070" = "true" ] || { log "PREFLIGHT FAIL: rtx4070 worker not Ready (got '${r4070:-missing}') — uncordon node 'acane' + restore MPS"; bad=1; }
  local r3080; r3080=$(kubectl get pod -n $NS slurm-worker-gpu-rtx3080-0 -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null)
  [ "$r3080" = "true" ] || { log "PREFLIGHT FAIL: rtx3080 worker not Ready (got '${r3080:-missing}')"; bad=1; }
  for SEED in "${SEEDS[@]}"; do for A in "${LEARNED[@]}"; do
    [ -f "${CK}/${A}_s${SEED}.pt" ] || { log "PREFLIGHT FAIL: missing ckpt ${CK}/${A}_s${SEED}.pt"; bad=1; }
  done; done
  return $bad
}

ck_args(){ local seed="$1" out=""
  for A in "${LEARNED[@]}"; do
    case "$A" in
      sac)         out="$out --sac-ckpt ${CK}/sac_s${seed}.pt";;
      rdsac_mean)  out="$out --rdsac-mean-ckpt ${CK}/rdsac_mean_s${seed}.pt";;
      rdsac_cvar)  out="$out --rdsac-cvar-ckpt ${CK}/rdsac_cvar_s${seed}.pt";;
    esac
  done; echo "$out"; }

kubectl get cm -n $NS $CM -o jsonpath='{.data.slurm\.conf}' > "$ORIG"; [ -s "$ORIG" ] || { echo FATAL cannot read slurm.conf; exit 1; }
CONF_fcfs=/tmp/slurm.conf.h125fcfs.$$; CONF_backfill=/tmp/slurm.conf.h125bf.$$
sed -e 's|^SchedulerType=.*|SchedulerType=sched/builtin|' -e 's|^PriorityType=.*|PriorityType=priority/basic|' -e '/^SchedulerParameters=/d' -e '/^JobSubmitPlugins=/d' "$ORIG" > "$CONF_fcfs"
sed -e 's|^PriorityType=.*|PriorityType=priority/basic|' -e '/^JobSubmitPlugins=/d' "$ORIG" > "$CONF_backfill"

log "### $TAG heavy-load A/B  n_jobs=$N_JOBS rounds=$ROUNDS oversub=$OVERSUB seeds=${SEEDS[*]}"
log "### learned arms: ${LEARNED[*]}"
if ! preflight; then log "ABORT: preflight failed (see above). Fix the cluster/ckpts then rerun."; exit 1; fi

# held-job watchdog (node-2 3080 restart → launch-failed-requeued-held → harness hangs)
bash eval/scripts/held_watchdog.sh "/tmp/${TAG}_wd_${STAMP}.log" & WD_PID=$!
log "held_watchdog pid=$WD_PID"

# NOTE: wait ONLY on the prewarm srun PIDs — a bare `wait` would also block on the
# never-exiting held_watchdog background job (WD_PID) and hang the whole run.
prewarm(){ local pids=(); for N in slurm-worker-gpu-rtx4070-0 slurm-worker-gpu-rtx3080-0; do timeout 180 kubectl -n $NS exec "$LOGIN" -- bash -lc "srun -p gpu -w $N --gres=mps:25 --time=5 /shared/py/bin/python3 /shared/scripts/llm_job.py --mode infer --n 1 --batch-size 4 --prompt-len 512 --gen-len 2 --model $MODEL 2>&1|tail -1" >/dev/null 2>&1 & pids+=($!); done; wait "${pids[@]}"; }
apply_verify(){ # $1 conf  $2 want-SchedulerType  $3 name
  patch_conf "$1"; PATCHED=1; restart_ctl_wait || { log "FATAL slurmctld not ready ($3)"; exit 1; }
  local got; got=$(kubectl exec -n $NS $CTL -- scontrol show config 2>/dev/null | grep -oP 'SchedulerType\s*=\s*\K\S+')
  [ "$got" = "$2" ] || { log "FATAL SchedulerType=$got != $2 ($3)"; exit 1; }
  resume_nodes; log "config $3 active (SchedulerType=$got)"; }

# ---- config1: learned (original config, Lua on) → score + learned arms interleaved ----
apply_verify "$ORIG" "sched/backfill" "learned"
prewarm
for SEED in "${SEEDS[@]}"; do
  log "  [learned] seed $SEED"
  # shellcheck disable=SC2046
  .venv-m11/bin/python -m eval.scripts.run_heavytail_ab \
    --serve-url "$SERVE" --login-pod "$LOGIN" --namespace $NS --controller-pod $CTL \
    $(ck_args "$SEED") \
    --family philly --n-jobs "$N_JOBS" --seed "$SEED" --sigmas 1.0 --rounds "$ROUNDS" --warmup 1 --interleave \
    --hybrid-workload --llm-model "$MODEL" --placement --gpu-nodes "$GPU_NODES" \
    --arrival-mode poisson --mps-oversub "$OVERSUB" --target-max-s 20 --mps-buckets 25,50,75,100 \
    --partition gpu --out-dir "runs/${TAG}_learned_s${SEED}_${STAMP}" >>"$LOG" 2>&1 || log "  [learned] s$SEED exit $?"
done

# ---- config2 & 3: Slurm-native (score panel only, no Lua) ----
run_slurm(){ # $1 name  $2 conf  $3 want-SchedulerType
  apply_verify "$2" "$3" "$1"; prewarm
  for SEED in "${SEEDS[@]}"; do
    log "  [$1] seed $SEED"
    .venv-m11/bin/python -m eval.scripts.run_heavytail_ab \
      --serve-url "$SERVE" --login-pod "$LOGIN" --namespace $NS --controller-pod $CTL \
      --family philly --n-jobs "$N_JOBS" --seed "$SEED" --sigmas 1.0 --rounds "$ROUNDS" --warmup 1 --interleave \
      --hybrid-workload --llm-model "$MODEL" --placement --gpu-nodes "$GPU_NODES" \
      --arrival-mode poisson --mps-oversub "$OVERSUB" --target-max-s 20 --mps-buckets 25,50,75,100 \
      --partition gpu --out-dir "runs/${TAG}_${1}_s${SEED}_${STAMP}" >>"$LOG" 2>&1 || log "  [$1] s$SEED exit $?"
  done; }
run_slurm fcfs     "$CONF_fcfs"     "sched/builtin"
run_slurm backfill "$CONF_backfill" "sched/backfill"

log "=== aggregating (JCT / Makespan / P95 / P99 + seed-paired ΔJCT%) ==="
.venv-m11/bin/python eval/scripts/aggregate_heavy125.py "$STAMP" "$TAG" "${LEARNED[*]}" "${SEEDS[@]}" >>"$LOG" 2>&1 \
  && cat "runs/${TAG}_${STAMP}_TABLES.md" 2>/dev/null | tee -a "$LOG"
log "${TAG}_DONE ${STAMP}"
