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
# Arms per seed (same job stream, same slurm.conf = backfill + 5min aging):
#   backfill : submit unheld; Slurm picks order+node (control).
#   sac/rdsac_mean/rdsac_cvar/rlpd_cvar : reload ckpt → precompute → priority actuate.
# Workload = sleep+MPS (wait-dominated tail; faithful without real CUDA).
#
#   SEEDS="42 43" ARMS="backfill rdsac_cvar" bash eval/scripts/run_step3_prio.sh
#   bash eval/scripts/run_step3_prio.sh              # full 16 seed × 5 arm
set -uo pipefail
cd /home/acane/Desktop/Kelpflux
export KUBECONFIG=$HOME/.kube/config PYTHONPATH=.
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="runs/step3prio_${STAMP}"; mkdir -p "$OUT"
LOG="$OUT/run.log"
CM=slurm-config-static; NS=slurm; CTL=slurm-controller-0
SERVE="${SERVE:-http://localhost:8003}"
CK="${CK:-runs/ckpts_aimix16_fair}"
N_JOBS="${N_JOBS:-150}"; TARGET_MAX="${TARGET_MAX:-20}"; MAXAGE="${MAXAGE:-00:05:00}"
read -r -a SEEDS <<< "${SEEDS:-42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57}"
read -r -a ARMS  <<< "${ARMS:-backfill sac rdsac_mean rdsac_cvar rlpd_cvar}"
ORIG="/tmp/slurm.conf.s3porig.$$"; FAST="/tmp/slurm.conf.s3pfast.$$"
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

curl -fsS --max-time 5 "$SERVE/healthz" >/dev/null 2>&1 || { log "FATAL serve down @ $SERVE"; exit 1; }
kubectl get cm -n $NS $CM -o jsonpath='{.data.slurm\.conf}' > "$ORIG"; [ -s "$ORIG" ] || { log "FATAL no conf"; exit 1; }
cp "$ORIG" "$FAST"; printf '\nPriorityMaxAge=%s\n' "$MAXAGE" >> "$FAST"

ck_for(){ local arm="$1" seed="$2"; case "$arm" in backfill) echo "";; *) echo "$CK/${arm}_s${seed}.pt";; esac; }

log "### Step 3 PRIORITY actuation: RL order+node via fixed Priority vs backfill"
log "### CK=$CK  N_JOBS=$N_JOBS  target_max=${TARGET_MAX}s  aging=$MAXAGE"
log "### arms=${ARMS[*]}  seeds=${SEEDS[*]}  out=$OUT"

patch_conf "$FAST"; PATCHED=1; restart_ctl_wait || { log "FATAL ctl restart"; exit 1; }
got=""; for i in $(seq 1 20); do got=$(kubectl exec -n $NS $CTL -- scontrol show config 2>/dev/null | grep -oP 'PriorityMaxAge\s*=\s*\K\S+'); [ -n "$got" ] && [ "$got" != "7-00:00:00" ] && break; sleep 6; done
log "PriorityMaxAge now = $got"
resume_nodes

for SEED in "${SEEDS[@]}"; do
  for ARM in "${ARMS[@]}"; do
    CKPT=$(ck_for "$ARM" "$SEED")
    if [ -n "$CKPT" ] && [ ! -f "$CKPT" ]; then log "  SKIP $ARM s$SEED (no ckpt)"; continue; fi
    OJ="$OUT/${ARM}_s${SEED}.json"
    [ -f "$OJ" ] && { log "  SKIP $ARM s$SEED (done)"; continue; }
    log "  [$ARM] seed $SEED"
    sweep
    RELOAD=(); [ -n "$CKPT" ] && RELOAD=(--reload-ckpt "$CKPT")
    WARM=priority; [ "$ARM" = "backfill" ] && WARM=backfill
    .venv-m11/bin/python -m eval.scripts.scontrol_ab \
      --arm "$WARM" --n-jobs "$N_JOBS" --seed "$SEED" --target-max "$TARGET_MAX" \
      "${RELOAD[@]}" --out-json "$OJ" >>"$LOG" 2>&1 || log "  $ARM s$SEED exit $?"
  done
done
sweep
log "### done stamp=$STAMP  jsons: $(ls "$OUT"/*.json 2>/dev/null | wc -l)"
echo "$OUT"
