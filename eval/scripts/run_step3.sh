#!/usr/bin/env bash
# Step 3 (payoff): fairness-trained RL job-SELECTION via the scontrol held-job
# path + fast-aging, vs Slurm's own backfill scheduling. This is the ONLY live
# path where the fairness reward can bite: it retrains job ORDERING, and only the
# scontrol pin+release path lets RL pick which held job runs next (the -w
# placement path leaves ordering to Slurm). Two tail mechanisms are attacked at
# once here (§ diagnosis):
#   (1) age-timescale mismatch → PriorityMaxAge 7d→5min so age actually accrues,
#   (2) binding rigidity        → scontrol ReqNodeList+release lets RL place+order
#                                 (slurmrestd v0.0.37 disables required_nodes).
# Arms (same job stream, same slurm.conf = backfill + 5min aging, per seed):
#   backfill    : submit unheld; Slurm picks order+node (the control to beat).
#   sac/rdsac_mean/rdsac_cvar/rlpd_cvar : reload that fairness ckpt → scontrol
#                 held-job loop (RL picks job+node). ΔJCT%/P99 vs backfill, paired.
# Workload = sleep+MPS (the tail is pure WAIT, so this reproduces the queueing
# faithfully without real CUDA — much cheaper; matches the scontrol_ab tool).
#
#   SEEDS="42 43" ARMS="backfill rdsac_cvar" bash eval/scripts/run_step3.sh
#   bash eval/scripts/run_step3.sh                 # full 16 seed × 5 arm
set -uo pipefail
cd /home/acane/Desktop/Kelpflux
export KUBECONFIG=$HOME/.kube/config PYTHONPATH=.
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="runs/step3_${STAMP}"; mkdir -p "$OUT"
LOG="$OUT/run.log"
CM=slurm-config-static; NS=slurm; CTL=slurm-controller-0
SERVE="${SERVE:-http://localhost:8003}"
CK="${CK:-runs/ckpts_aimix16_fair}"
N_JOBS="${N_JOBS:-150}"; TARGET_MAX="${TARGET_MAX:-20}"; MAXAGE="${MAXAGE:-00:05:00}"
DEADLINE_MIN="${DEADLINE_MIN:-30}"
read -r -a SEEDS <<< "${SEEDS:-42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57}"
read -r -a ARMS  <<< "${ARMS:-backfill sac rdsac_mean rdsac_cvar rlpd_cvar}"
ORIG="/tmp/slurm.conf.step3orig.$$"; FAST="/tmp/slurm.conf.step3fast.$$"
PATCHED=0
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
patch_conf(){ kubectl patch cm -n $NS $CM --type merge -p "$(python3 -c "import json;print(json.dumps({'data':{'slurm.conf':open('$1').read()}}))")" >/dev/null; }
restart_ctl_wait(){ kubectl delete pod -n $NS $CTL >/dev/null 2>&1; for i in $(seq 1 30); do [ "$(kubectl get pod -n $NS $CTL -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null)" = "true" ] && return 0; sleep 6; done; return 1; }
resume_nodes(){ kubectl exec -n $NS $CTL -- bash -lc 'scontrol reconfigure 2>/dev/null; for N in slurm-worker-gpu-rtx4070-0 slurm-worker-gpu-rtx3080-0; do for i in 1 2 3; do scontrol update nodename=$N state=resume 2>/dev/null; sleep 3; done; done' >/dev/null 2>&1 || true; }
sweep(){ kubectl exec -n $NS $CTL -- bash -lc "squeue -h -o '%i %j' 2>/dev/null | awk '\$2 ~ /^sc/ {print \$1}' | xargs -r scancel 2>/dev/null; echo swept" >/dev/null 2>&1 || true;
  for i in $(seq 1 20); do L=$(kubectl exec -n $NS $CTL -- bash -lc "squeue -h -o '%j' 2>/dev/null | grep -c '^sc'" 2>/dev/null); [ "${L:-0}" = "0" ] && return 0; sleep 3; done; }
restore(){ [ "$PATCHED" = 1 ] || return 0; log "RESTORING original slurm.conf"; patch_conf "$ORIG"; restart_ctl_wait && log "restored" || log "WARN restore"; resume_nodes; }
cleanup(){ sweep; restore; }
trap cleanup EXIT

curl -fsS --max-time 5 "$SERVE/healthz" >/dev/null 2>&1 || { log "FATAL serve down @ $SERVE"; exit 1; }
kubectl get cm -n $NS $CM -o jsonpath='{.data.slurm\.conf}' > "$ORIG"; [ -s "$ORIG" ] || { log "FATAL no conf"; exit 1; }
cp "$ORIG" "$FAST"; printf '\nPriorityMaxAge=%s\n' "$MAXAGE" >> "$FAST"

# ckpt for a learned arm (empty for backfill)
ck_for(){ local arm="$1" seed="$2"; case "$arm" in
  backfill) echo "";;
  *) echo "$CK/${arm}_s${seed}.pt";; esac; }

log "### Step 3 payoff: scontrol held-job + aging($MAXAGE) vs backfill"
log "### CK=$CK  N_JOBS=$N_JOBS  target_max=${TARGET_MAX}s  deadline=${DEADLINE_MIN}min"
log "### arms=${ARMS[*]}  seeds=${SEEDS[*]}  out=$OUT"

# apply fast-aging (SchedulerType stays sched/backfill); retry-poll for RPC-ready
patch_conf "$FAST"; PATCHED=1; restart_ctl_wait || { log "FATAL ctl restart"; exit 1; }
got=""; for i in $(seq 1 20); do got=$(kubectl exec -n $NS $CTL -- scontrol show config 2>/dev/null | grep -oP 'PriorityMaxAge\s*=\s*\K\S+'); [ -n "$got" ] && [ "$got" != "7-00:00:00" ] && break; sleep 6; done
log "PriorityMaxAge now = $got"
resume_nodes

for SEED in "${SEEDS[@]}"; do
  for ARM in "${ARMS[@]}"; do
    CKPT=$(ck_for "$ARM" "$SEED")
    if [ -n "$CKPT" ] && [ ! -f "$CKPT" ]; then log "  SKIP $ARM s$SEED (no ckpt $CKPT)"; continue; fi
    OJ="$OUT/${ARM}_s${SEED}.json"
    [ -f "$OJ" ] && { log "  SKIP $ARM s$SEED (done)"; continue; }
    log "  [$ARM] seed $SEED"
    sweep
    RELOAD=(); [ -n "$CKPT" ] && RELOAD=(--reload-ckpt "$CKPT")
    WARM=scontrol; [ "$ARM" = "backfill" ] && WARM=backfill
    .venv-m11/bin/python -m eval.scripts.scontrol_ab \
      --arm "$WARM" --n-jobs "$N_JOBS" --seed "$SEED" \
      --target-max "$TARGET_MAX" --deadline-min "$DEADLINE_MIN" \
      "${RELOAD[@]}" --out-json "$OJ" >>"$LOG" 2>&1 || log "  $ARM s$SEED exit $?"
  done
done
sweep
log "### done stamp=$STAMP  jsons: $(ls "$OUT"/*.json 2>/dev/null | wc -l)"
echo "$OUT"
