#!/usr/bin/env bash
# Retrain the learned arms on the NEW hetero-cluster model (host-RAM OOM gate +
# per-(card,class) speed matrix) and the aimix workload. The old /tmp/lckpts
# checkpoints are obsolete (obs semantics changed 166→168; node_speeds=[1,0.25]
# was wrong). UXP-RL is dropped (single-layer LAN ≠ its cloud/edge/fog setting).
#
# Arms: SAC (--no-iqn), RDSAC-mean, RDSAC-cvar. 8 seeds → 24 checkpoints, run
# MAX-way parallel on CPU (the local 4070 is held by the Slurm worker via DRA).
set -uo pipefail
cd /home/acane/Desktop/Kelpflux
export PYTHONPATH=.
# 16 cores; MAX×threads ≈ cores so parallel trainings don't oversubscribe.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
CK=/tmp/lckpts_aimix
STEPS="${STEPS:-100000}"
DEVICE="${DEVICE:-cpu}"
MAX="${MAX:-8}"                 # concurrent trainings
read -r -a SEEDS <<< "${SEEDS:-42 43 44 45 46 47 48 49}"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="runs/train_aimix_${STAMP}.log"
mkdir -p "$CK" runs
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

declare -A ARM_FLAGS=(
  [sac]="--no-iqn"
  [rdsac_mean]="--risk-mode mean"
  [rdsac_cvar]="--risk-mode cvar"
)
COMMON=(--n-nodes 2 --gpus-per-node 1 --hetero-cluster --trace aimix
        --n-jobs 50 --curriculum --total-steps "$STEPS" --warmup-steps 2000
        --device "$DEVICE" --fixed-alpha --init-alpha 0.05
        --reward-mode mo --mo-w-jct 1.0 --mo-w-util 0.05)

train_one(){ local ARM="$1" SEED="$2" OUT="runs/aimix_$1_s$2_${STAMP}"
  # shellcheck disable=SC2086
  if .venv-m11/bin/python -m services.rl_scheduler.sim_train "${COMMON[@]}" ${ARM_FLAGS[$ARM]} \
       --seed "$SEED" --out-dir "$OUT" >>"$LOG" 2>&1 && cp "$OUT/dsac.pt" "$CK/${ARM}_s${SEED}.pt"; then
    log "  OK  ${ARM} s$SEED"; else log "  FAIL ${ARM} s$SEED (exit $?)"; fi; }

log "### train aimix: ${!ARM_FLAGS[*]} × seeds ${SEEDS[*]}  steps=$STEPS MAX=$MAX"
for SEED in "${SEEDS[@]}"; do
  for ARM in sac rdsac_mean rdsac_cvar; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAX" ]; do wait -n 2>/dev/null || sleep 5; done
    train_one "$ARM" "$SEED" &
  done
done
wait
log "### done"; ls "$CK"/*.pt 2>/dev/null | wc -l | xargs echo "checkpoints:"
