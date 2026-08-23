#!/usr/bin/env bash
# BALANCE-ONLY retrain — the tail-fix experiment, single-variable ablation. The
# mo-reward aimix16 policies over-concentrated on the fast 4070 → heavy-load P99
# ≈2.4× Slurm-native (§5.2 table 6b). This retrains the learned arms adding ONLY
# P1 load-balance shaping, keeping the original mo reward unchanged (so the effect
# of balance shaping is isolated from any reward-shape change):
#   --balance-coef 5.0   P1 potential-based node-balance shaping (proven value,
#                        docs/eval-writeup §; concentration 89%→~71% in sim)
#   --reward-mode mo --mo-w-jct 1.0 --mo-w-util 0.05   (unchanged from the mo campaign)
# Everything else matches train_aimix_seeds.sh (hetero 2×1, aimix, curriculum, fixed-α).
# ⚠️ P1 caveat: balance shaping regressed RDSAC-mean in sim (risk-neutral interaction).
#
# Arms: SAC (--no-iqn), RDSAC-mean, RDSAC-cvar. 16 seeds → 48 checkpoints.
# GPU when the 4070 is released (DEVICE=cuda); else CPU.
set -uo pipefail
cd /home/acane/Desktop/Kelpflux
export PYTHONPATH=.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
CK="${CK:-runs/ckpts_aimix16_bal}"   # balance-only checkpoints (separate from mo/bs)
STEPS="${STEPS:-100000}"
DEVICE="${DEVICE:-cpu}"
MAX="${MAX:-8}"                      # concurrent trainings (use ~4 on a single GPU)
read -r -a SEEDS <<< "${SEEDS:-42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57}"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="runs/train_aimix_bal_${STAMP}.log"
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
        --reward-mode mo --mo-w-jct 1.0 --mo-w-util 0.05 --balance-coef 5.0)

train_one(){ local ARM="$1" SEED="$2" OUT="runs/aimix_bal_$1_s$2_${STAMP}"
  [ -f "$CK/${ARM}_s${SEED}.pt" ] && { log "  SKIP ${ARM} s${SEED} (checkpoint exists)"; return 0; }
  # shellcheck disable=SC2086
  if .venv-m11/bin/python -m services.rl_scheduler.sim_train "${COMMON[@]}" ${ARM_FLAGS[$ARM]} \
       --seed "$SEED" --out-dir "$OUT" >>"$LOG" 2>&1 && cp "$OUT/dsac.pt" "$CK/${ARM}_s${SEED}.pt"; then
    log "  OK  ${ARM} s$SEED"; else log "  FAIL ${ARM} s$SEED (exit $?)"; fi; }

log "### train aimix BALANCE-ONLY (mo reward): ${!ARM_FLAGS[*]} × seeds ${SEEDS[*]}  steps=$STEPS MAX=$MAX"
log "### COMMON: ${COMMON[*]}"
for SEED in "${SEEDS[@]}"; do
  for ARM in sac rdsac_mean rdsac_cvar; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAX" ]; do wait -n 2>/dev/null || sleep 5; done
    train_one "$ARM" "$SEED" &
  done
done
wait
log "### done"; ls "$CK"/*.pt 2>/dev/null | wc -l | xargs echo "checkpoints:"
