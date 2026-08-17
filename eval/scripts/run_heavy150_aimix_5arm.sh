#!/usr/bin/env bash
# Heavy-load (§6.3) live A/B on the AIMIX workload with the RLPD arm — the faithful
# real-data 6-arm: fcfs / backfill (naive Slurm, no Lua) + SAC / RDSAC-mean / RDSAC-cvar
# / RLPD learned placement, all paired vs the score baseline (score = backfill+multifactor+
# score-Lua; it is the paired REFERENCE, not a promoted arm). n_jobs≈150 sits at the
# upper edge of the §5.7 ceiling window (125–150) where ~10–14% headroom over score
# opens, of which only ~16% is ε-reachable.
#
# Differs from run_heavy125_5arm.sh (philly/hybrid, NO RLPD) only in: aimix workload,
# the rlpd_cvar arm (real-online-log fine-tuned, see train_rlpd_aimix16.sh), n_jobs=150,
# 16 aimix seeds, and CK=runs/ckpts_aimix16. Everything else — the three Slurm configs,
# trap-restore, preflight, held_watchdog, prewarm — is identical and load-bearing.
#
# Env knobs:
#   SEEDS="42 .. 57"  N_JOBS=150  ROUNDS=1  OVERSUB=2.0  LEARNED="sac rdsac_mean rdsac_cvar rlpd_cvar"
#   SMOKE=1 → tiny end-to-end wiring check (N_JOBS=6, ROUNDS=1, SEEDS="42").
#
# Usage:
#   SMOKE=1 bash eval/scripts/run_heavy150_aimix_5arm.sh                  # wiring smoke
#   bash eval/scripts/run_heavy150_aimix_5arm.sh                          # full 16-seed
set -uo pipefail
cd /home/acane/Desktop/Kelpflux
export KUBECONFIG=$HOME/.kube/config PYTHONPATH=.
# STAMP override + PHASES filter let a partial run be resumed: e.g. after a flaky
# apply_verify failure that only killed the backfill phase, re-run just that phase
# under the SAME STAMP so aggregate finds all learned/fcfs/backfill dirs together.
#   PHASES=backfill STAMP=20260815-135722 bash eval/scripts/run_heavy150_aimix_5arm.sh
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
PHASES="${PHASES:-learned fcfs backfill}"
want_phase(){ [[ " $PHASES " == *" $1 "* ]]; }
TAG="${TAG:-heavy150aimix}"
LOG="runs/${TAG}_${STAMP}.log"
CM=slurm-config-static; NS=slurm; CTL=slurm-controller-0
LOGIN="${LOGIN_POD:-slurm-login-7f8cfbc48-c875f}"
GPU_NODES="slurm-worker-gpu-rtx4070-0,slurm-worker-gpu-rtx3080-0"
SERVE="${SERVE:-http://localhost:8003}"; MODEL="${MODEL:-/shared/models/qwen05b}"
CK="${CK:-runs/ckpts_aimix16}"
# learned arms (space-separated ckpt prefixes in $CK). Default = the aimix 5-arm spec
# (fcfs/backfill via Slurm configs + score baseline + sac/rdsac_cvar/rlpd_cvar).
read -r -a LEARNED <<< "${LEARNED:-sac rdsac_mean rdsac_cvar rlpd_cvar}"

# SEEDS must word-split on BOTH spaces and newlines (see run_heavy125 note).
# shellcheck disable=SC2206
if [ "${SMOKE:-0}" = "1" ]; then
  N_JOBS="${N_JOBS:-6}"; ROUNDS="${ROUNDS:-1}"; SEEDS=( ${SEEDS:-42} )
  log_prefix="[SMOKE] "
else
  N_JOBS="${N_JOBS:-150}"; ROUNDS="${ROUNDS:-1}"; SEEDS=( ${SEEDS:-$(seq 42 57)} )
  log_prefix=""
fi
OVERSUB="${OVERSUB:-2.0}"
ORIG=/tmp/slurm.conf.h150orig.$$
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
      rlpd_cvar)   out="$out --rlpd-ckpt ${CK}/rlpd_cvar_s${seed}.pt";;
    esac
  done; echo "$out"; }

kubectl get cm -n $NS $CM -o jsonpath='{.data.slurm\.conf}' > "$ORIG"; [ -s "$ORIG" ] || { echo FATAL cannot read slurm.conf; exit 1; }
CONF_fcfs=/tmp/slurm.conf.h150fcfs.$$; CONF_backfill=/tmp/slurm.conf.h150bf.$$
sed -e 's|^SchedulerType=.*|SchedulerType=sched/builtin|' -e 's|^PriorityType=.*|PriorityType=priority/basic|' -e '/^SchedulerParameters=/d' -e '/^JobSubmitPlugins=/d' "$ORIG" > "$CONF_fcfs"
sed -e 's|^PriorityType=.*|PriorityType=priority/basic|' -e '/^JobSubmitPlugins=/d' "$ORIG" > "$CONF_backfill"

log "### $TAG heavy-load aimix A/B  n_jobs=$N_JOBS rounds=$ROUNDS oversub=$OVERSUB seeds=${SEEDS[*]}"
log "### learned arms: ${LEARNED[*]}  (CK=$CK)"
if ! preflight; then log "ABORT: preflight failed (see above). Fix the cluster/ckpts then rerun."; exit 1; fi

# held-job watchdog (node-2 3080 restart → launch-failed-requeued-held → harness hangs)
bash eval/scripts/held_watchdog.sh "/tmp/${TAG}_wd_${STAMP}.log" & WD_PID=$!
log "held_watchdog pid=$WD_PID"

# wait ONLY on the prewarm srun PIDs — a bare `wait` would also block on the
# never-exiting held_watchdog background job (WD_PID) and hang the whole run.
prewarm(){ local pids=(); for N in slurm-worker-gpu-rtx4070-0 slurm-worker-gpu-rtx3080-0; do timeout 180 kubectl -n $NS exec "$LOGIN" -- bash -lc "srun -p gpu -w $N --gres=mps:25 --time=5 /shared/py/bin/python3 /shared/scripts/llm_job.py --mode infer --n 1 --batch-size 4 --prompt-len 512 --gen-len 2 --model $MODEL 2>&1|tail -1" >/dev/null 2>&1 & pids+=($!); done; wait "${pids[@]}"; }
apply_verify(){ # $1 conf  $2 want-SchedulerType  $3 name
  patch_conf "$1"; PATCHED=1; restart_ctl_wait || { log "FATAL slurmctld not ready ($3)"; exit 1; }
  # pod-ready (containerStatuses.ready) precedes slurmctld RPC-ready: `scontrol show
  # config` can return an EMPTY SchedulerType for ~tens of seconds after restart.
  # Poll until the RPC answers with the expected value instead of failing on the race.
  local got="" i
  for i in $(seq 1 20); do
    got=$(kubectl exec -n $NS $CTL -- scontrol show config 2>/dev/null | grep -oP 'SchedulerType\s*=\s*\K\S+')
    [ "$got" = "$2" ] && break
    sleep 6
  done
  [ "$got" = "$2" ] || { log "FATAL SchedulerType='$got' != $2 ($3) after 20×6s poll"; exit 1; }
  resume_nodes; log "config $3 active (SchedulerType=$got)"; }

# ---- config1: learned (original config, Lua on) → learned arms (--no-score) ----
if want_phase learned; then
apply_verify "$ORIG" "sched/backfill" "learned"
prewarm
for SEED in "${SEEDS[@]}"; do
  log "  [learned] seed $SEED"
  # shellcheck disable=SC2046
  .venv-m11/bin/python -m eval.scripts.run_heavytail_ab \
    --serve-url "$SERVE" --login-pod "$LOGIN" --namespace $NS --controller-pod $CTL \
    $(ck_args "$SEED") --no-score \
    --family aimix --n-jobs "$N_JOBS" --seed "$SEED" --sigmas 1.0 --rounds "$ROUNDS" --warmup 1 --interleave \
    --aimix-workload --llm-model "$MODEL" --placement --gpu-nodes "$GPU_NODES" \
    --arrival-mode poisson --mps-oversub "$OVERSUB" --target-max-s 20 --mps-buckets 25,50,75,100 \
    --partition gpu --out-dir "runs/${TAG}_learned_s${SEED}_${STAMP}" >>"$LOG" 2>&1 || log "  [learned] s$SEED exit $?"
done
else log "SKIP learned phase (PHASES=$PHASES)"; fi

# ---- config2 & 3: Slurm-native (score panel only, no Lua) ----
run_slurm(){ # $1 name  $2 conf  $3 want-SchedulerType
  apply_verify "$2" "$3" "$1"; prewarm
  for SEED in "${SEEDS[@]}"; do
    log "  [$1] seed $SEED"
    .venv-m11/bin/python -m eval.scripts.run_heavytail_ab \
      --serve-url "$SERVE" --login-pod "$LOGIN" --namespace $NS --controller-pod $CTL \
      --family aimix --n-jobs "$N_JOBS" --seed "$SEED" --sigmas 1.0 --rounds "$ROUNDS" --warmup 1 --interleave \
      --aimix-workload --llm-model "$MODEL" --placement --gpu-nodes "$GPU_NODES" \
      --arrival-mode poisson --mps-oversub "$OVERSUB" --target-max-s 20 --mps-buckets 25,50,75,100 \
      --partition gpu --out-dir "runs/${TAG}_${1}_s${SEED}_${STAMP}" >>"$LOG" 2>&1 || log "  [$1] s$SEED exit $?"
  done; }
want_phase fcfs     && run_slurm fcfs     "$CONF_fcfs"     "sched/builtin"  || log "SKIP fcfs phase (PHASES=$PHASES)"
want_phase backfill && run_slurm backfill "$CONF_backfill" "sched/backfill" || log "SKIP backfill phase (PHASES=$PHASES)"

log "=== aggregating (JCT / Makespan / P95 / P99 + seed-paired ΔJCT% vs backfill) ==="
.venv-m11/bin/python eval/scripts/aggregate_heavy150_vs_backfill.py "$STAMP" "$TAG" "${LEARNED[*]}" "${SEEDS[@]}" >>"$LOG" 2>&1 \
  && cat "runs/${TAG}_${STAMP}_TABLES.md" 2>/dev/null | tee -a "$LOG"
log "${TAG}_DONE ${STAMP}"
