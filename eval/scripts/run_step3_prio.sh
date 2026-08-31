#!/usr/bin/env bash
# Step 3 (payoff), priority-based actuation — the fair fix for the hold/poll-release
# path, whose out-of-process latency made every policy look identical (all ~+35% vs
# backfill; see [[project-scontrol-actuation-latency-confound]]). Here RL's schedule
# is precomputed (in-memory drain over /act → order+node), then all jobs are submitted
# HELD and RELEASED-with-fixed-Priority in one controller script, so Slurm's OWN
# in-process backfill scheduler actuates by that priority at native speed. RL owns
# ORDER+NODE, Slurm owns TIMING → no poll-loop latency → policies can diverge → a fair
# test of whether fairness-trained ordering beats Slurm's backfill ordering.
#
# Two cluster-config PHASES (each applied ONCE, all its seeds run under it, then
# torn down) — mirrors run_heavy150_aimix_5arm.sh's fcfs/backfill config split:
#   fcfs : SchedulerType=sched/builtin + PriorityType=priority/basic (no backfill-
#          skip, no fast-aging) — pure FCFS control, submitted unheld, no RL.
#   main : SchedulerType=sched/backfill + PriorityMaxAge=5min — backfill control
#          (unheld) + sac/rdsac_mean/rdsac_cvar/rlpd_cvar (precompute → priority
#          actuate).
# Workload: sleep+MPS by default (§5.8, wait-dominated proxy); --real-workload
# switches every arm to the real AiMix GPU jobs (BERT/ResNet/Qwen/cuBLAS, matching
# table 6b) — MUCH slower (real compute, not a placeholder sleep), see script header
# comment in scontrol_ab.py `wrap_and_time`.
#
#   SEEDS="42 43" ARMS="backfill rdsac_cvar" PHASES="main" bash eval/scripts/run_step3_prio.sh
#   REAL_WORKLOAD=1 bash eval/scripts/run_step3_prio.sh       # full 16 seed × 6 arm, real CUDA
#   bash eval/scripts/run_step3_prio.sh                       # full 16 seed × 6 arm, sleep+MPS
set -uo pipefail
cd /home/acane/Desktop/Kelpflux
export KUBECONFIG=$HOME/.kube/config PYTHONPATH=.
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
OUT="runs/step3prio_${STAMP}"; mkdir -p "$OUT"
LOG="$OUT/run.log"
CM=slurm-config-static; NS=slurm; CTL=slurm-controller-0
SERVE="${SERVE:-http://localhost:8003}"
CK="${CK:-runs/ckpts_aimix16_fair}"
MODEL="${MODEL:-/shared/models/qwen05b}"
N_JOBS="${N_JOBS:-150}"; TARGET_MAX="${TARGET_MAX:-20}"; MAXAGE="${MAXAGE:-00:05:00}"
REAL_WORKLOAD="${REAL_WORKLOAD:-0}"   # 1 → real AiMix GPU jobs instead of sleep+MPS
ARRIVAL_MODE="${ARRIVAL_MODE:-poisson}"   # poisson (spread, matches run_heavy150) or burst
# Load sweep: run the whole grid at each oversub → poisson mean gap = mean(rt)/oversub,
# so bigger oversub = faster arrivals = DEEPER standing queue = more ordering leverage.
# The sweep shows the ordering headroom EMERGE with load (live §5.6 confirmation).
read -r -a OVERSUBS <<< "${OVERSUBS:-2 4 6}"
read -r -a SEEDS <<< "${SEEDS:-42 43 44 45 46 47 48 49}"
read -r -a ARMS  <<< "${ARMS:-backfill sac rdsac_mean rdsac_cvar rlpd_cvar}"
read -r -a PHASES <<< "${PHASES:-fcfs main}"   # fcfs, main, or both
ORIG="/tmp/slurm.conf.s3porig.$$"
CONF_FCFS="/tmp/slurm.conf.s3pfcfs.$$"; CONF_MAIN="/tmp/slurm.conf.s3pmain.$$"
PATCHED=0
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
patch_conf(){ kubectl patch cm -n $NS $CM --type merge -p "$(python3 -c "import json;print(json.dumps({'data':{'slurm.conf':open('$1').read()}}))")" >/dev/null; }
restart_ctl_wait(){ kubectl delete pod -n $NS $CTL >/dev/null 2>&1; for i in $(seq 1 30); do [ "$(kubectl get pod -n $NS $CTL -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null)" = "true" ] && return 0; sleep 6; done; return 1; }
resume_nodes(){ kubectl exec -n $NS $CTL -- bash -lc 'scontrol reconfigure 2>/dev/null; for N in slurm-worker-gpu-rtx4070-0 slurm-worker-gpu-rtx3080-0; do for i in 1 2 3; do scontrol update nodename=$N state=resume 2>/dev/null; sleep 3; done; done' >/dev/null 2>&1 || true; }
sweep(){ kubectl exec -n $NS $CTL -- bash -lc "squeue -h -o '%i %j' 2>/dev/null | awk '\$2 ~ /^sc/ {print \$1}' | xargs -r scancel 2>/dev/null" >/dev/null 2>&1 || true;
  for i in $(seq 1 20); do L=$(kubectl exec -n $NS $CTL -- bash -lc "squeue -h -o '%j' 2>/dev/null | grep -c '^sc'" 2>/dev/null); [ "${L:-0}" = "0" ] && return 0; sleep 3; done; }
restore(){ [ "$PATCHED" = 1 ] || return 0; log "RESTORING original slurm.conf"; patch_conf "$ORIG"; restart_ctl_wait && log "restored" || log "WARN restore"; resume_nodes; }
cleanup(){ sweep; restore; }
trap cleanup EXIT

apply_verify(){ # $1=conf  $2=want-SchedulerType  $3=name
  patch_conf "$1"; PATCHED=1; restart_ctl_wait || { log "FATAL ctl restart ($3)"; exit 1; }
  local got=""
  for i in $(seq 1 20); do
    got=$(kubectl exec -n $NS $CTL -- scontrol show config 2>/dev/null | grep -oP 'SchedulerType\s*=\s*\K\S+')
    [ "$got" = "$2" ] && break
    sleep 6
  done
  [ "$got" = "$2" ] || { log "FATAL SchedulerType='$got' != $2 ($3) after 20×6s poll"; exit 1; }
  resume_nodes; log "config $3 active (SchedulerType=$got)"
}

curl -fsS --max-time 5 "$SERVE/healthz" >/dev/null 2>&1 || { log "FATAL serve down @ $SERVE"; exit 1; }
kubectl get cm -n $NS $CM -o jsonpath='{.data.slurm\.conf}' > "$ORIG"; [ -s "$ORIG" ] || { log "FATAL no conf"; exit 1; }
# FCFS: no backfill-skip, priority/basic (submit-order), no aging needed.
sed -e 's|^SchedulerType=.*|SchedulerType=sched/builtin|' -e 's|^PriorityType=.*|PriorityType=priority/basic|' \
    -e '/^SchedulerParameters=/d' -e '/^JobSubmitPlugins=/d' "$ORIG" > "$CONF_FCFS"
# main: backfill + fast-aging (so age actually bites at this job's timescale).
cp "$ORIG" "$CONF_MAIN"; printf '\nPriorityMaxAge=%s\n' "$MAXAGE" >> "$CONF_MAIN"

REALFLAG=(); [ "$REAL_WORKLOAD" = "1" ] && REALFLAG=(--real-workload --llm-model "$MODEL")
ck_for(){ local arm="$1" seed="$2"; case "$arm" in backfill|fcfs) echo "";; *) echo "$CK/${arm}_s${seed}.pt";; esac; }

log "### Step 3 PRIORITY actuation: RL order+node via fixed Priority vs Backfill/FCFS"
log "### CK=$CK  N_JOBS=$N_JOBS  target_max=${TARGET_MAX}s  aging=$MAXAGE  real_workload=$REAL_WORKLOAD"
log "### arrival=$ARRIVAL_MODE  OVERSUBS=${OVERSUBS[*]}  arms=${ARMS[*]}  seeds=${SEEDS[*]}  phases=${PHASES[*]}  out=$OUT"
CUR_OUT="$OUT"; CUR_OVERSUB=2.0   # set per-oversub in the sweep loop below

WARMED=0
warmup_caches(){ # run a throwaway real-job burst once so NFS model reads / torch
  # imports / GPU+MPS spin-up are WARM before any measured run — otherwise whichever
  # arm runs first (fcfs) pays the cold-cache tail the others don't (drift confound).
  [ "$WARMED" = 1 ] && return 0; WARMED=1
  [ "${WARMUP:-1}" = "1" ] || return 0
  log "### cache warmup: ${WARMUP_JOBS:-24} real jobs (all classes) — discarded"
  sweep
  .venv-m11/bin/python -m eval.scripts.scontrol_ab \
    --arm backfill --n-jobs "${WARMUP_JOBS:-24}" --seed 999 --target-max "$TARGET_MAX" \
    "${REALFLAG[@]}" >>"$LOG" 2>&1 || log "  warmup exit $?"
  sweep
}

run_client(){ # $1=submit-arm(fcfs/backfill/priority)  $2=label(json stem)  $3=seed  $4=ckpt("" if none)
  local OJ="$CUR_OUT/${2}_s${3}.json"
  [ -f "$OJ" ] && { log "  SKIP $2 s$3 (done)"; return 0; }
  log "  [$2] seed $3 (oversub=$CUR_OVERSUB)"
  sweep
  local RELOAD=(); [ -n "$4" ] && RELOAD=(--reload-ckpt "$4")
  .venv-m11/bin/python -m eval.scripts.scontrol_ab \
    --arm "$1" --n-jobs "$N_JOBS" --seed "$3" --target-max "$TARGET_MAX" \
    --arrival-mode "$ARRIVAL_MODE" --oversub "$CUR_OVERSUB" \
    "${RELOAD[@]}" "${REALFLAG[@]}" --out-json "$OJ" >>"$LOG" 2>&1 || log "  $2 s$3 exit $?"
}

run_phases(){ # runs the fcfs/main phases at the current CUR_OVERSUB/CUR_OUT
  for PHASE in "${PHASES[@]}"; do
    case "$PHASE" in
      fcfs)
        apply_verify "$CONF_FCFS" "sched/builtin" "fcfs"
        warmup_caches
        for SEED in "${SEEDS[@]}"; do run_client fcfs fcfs "$SEED" ""; done
        ;;
      main)
        apply_verify "$CONF_MAIN" "sched/backfill" "main"
        got=""; for i in $(seq 1 20); do got=$(kubectl exec -n $NS $CTL -- scontrol show config 2>/dev/null | grep -oP 'PriorityMaxAge\s*=\s*\K\S+'); [ -n "$got" ] && [ "$got" != "7-00:00:00" ] && break; sleep 6; done
        log "PriorityMaxAge now = $got"
        warmup_caches
        for SEED in "${SEEDS[@]}"; do
          for ARM in "${ARMS[@]}"; do
            CKPT=$(ck_for "$ARM" "$SEED")
            if [ -n "$CKPT" ] && [ ! -f "$CKPT" ]; then log "  SKIP $ARM s$SEED (no ckpt)"; continue; fi
            # ACTUATION selects how the learned arms are actuated:
            #   priority (default) = §5.8 static ordering via fixed Slurm Priority
            #   online             = event-driven Option C (full select+place, run_online_arm)
            #   reorder            = Option B periodic re-prioritization (non-blocking, run_reorder_arm)
            WARM="${ACTUATION:-priority}"; [ "$ARM" = "backfill" ] && WARM=backfill
            # suffix the json label with the actuation so multiple actuation passes (same
            # seeds/out-dir) don't collide on ${ARM}_sSEED.json (priority keeps the bare name).
            LABEL="$ARM"; { [ "$WARM" != "priority" ] && [ "$WARM" != "backfill" ]; } && LABEL="${ARM}_${WARM}"
            run_client "$WARM" "$LABEL" "$SEED" "$CKPT"
          done
        done
        ;;
    esac
  done
}

for OV in "${OVERSUBS[@]}"; do
  CUR_OVERSUB="$OV"; CUR_OUT="$OUT/ov${OV}"; mkdir -p "$CUR_OUT"
  log "### ===== oversub=$OV  →  $CUR_OUT ====="
  run_phases
done
sweep
log "### done stamp=$STAMP  jsons: $(find "$OUT" -name '*.json' 2>/dev/null | wc -l)"
echo "$OUT"
