#!/usr/bin/env bash
# INTERFERENCE(+BALANCE) retrain — the bigger tail-fix lever. balance-only (see
# train_aimix_seeds_bal.sh) did NOT reduce the heavy-150 tail (§5.2 table 6b→bal:
# learned P99 stayed ≈665s) because the training env had interference=0 → packing
# the fast 4070 was "free", so the policy never felt the co-residence cost that
# causes real MPS tails. This adds a per-co-resident runtime slowdown to training
# so packing is actually penalised via the (mo) JCT reward:
#   --interference 0.3   realized_runtime = nominal·(1 + 0.3·k), k = co-residents
#                        (modeling choice; prior value used in this codebase, Plan B.
#                         Not a precise real-MPS calibration — could be swept later.)
#   --balance-coef 5.0   P1 node-balance shaping (kept, complements interference)
#   --mo-w-util 0.0      util term REMOVED — it rewarded high utilization (= packing),
#                        which works against spreading to avoid tails. So reward is
#                        effectively pure −JCT (mo_w_jct·completion) + balance potential.
# Combined tail-fix run (interference + balance + no-util); if this still leaves the
# heavy-150 P99 ≈2.4× Slurm-native the negative result is strong; if it fixes it, a
# later ablation isolates which knob mattered.
# Interference also gives RDSAC's return distribution genuine spread → its risk
# machinery (CVaR) is no longer idle (see [[project-sim-stochasticity-rdsac]]).
# Everything else matches train_aimix_seeds.sh (hetero 2×1, aimix, curriculum, fixed-α).
#
# Arms: SAC (--no-iqn), RDSAC-mean, RDSAC-cvar. 16 seeds → 48 checkpoints.
# GPU when the 4070 is released (DEVICE=cuda); else CPU.
set -uo pipefail
cd /home/acane/Desktop/Kelpflux
export PYTHONPATH=.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
CK="${CK:-runs/ckpts_aimix16_intf}"   # interference+balance checkpoints
STEPS="${STEPS:-100000}"
DEVICE="${DEVICE:-cpu}"
MAX="${MAX:-8}"                       # concurrent trainings (use ~4 on a single GPU)
INTERFERENCE="${INTERFERENCE:-0.3}"
BALANCE="${BALANCE:-5.0}"
read -r -a SEEDS <<< "${SEEDS:-42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57}"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="runs/train_aimix_intf_${STAMP}.log"
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
        --reward-mode mo --mo-w-jct 1.0 --mo-w-util 0.0
        --balance-coef "$BALANCE" --interference "$INTERFERENCE")

train_one(){ local ARM="$1" SEED="$2" OUT="runs/aimix_intf_$1_s$2_${STAMP}"
  [ -f "$CK/${ARM}_s${SEED}.pt" ] && { log "  SKIP ${ARM} s${SEED} (checkpoint exists)"; return 0; }
  # shellcheck disable=SC2086
  if .venv-m11/bin/python -m services.rl_scheduler.sim_train "${COMMON[@]}" ${ARM_FLAGS[$ARM]} \
       --seed "$SEED" --out-dir "$OUT" >>"$LOG" 2>&1 && cp "$OUT/dsac.pt" "$CK/${ARM}_s${SEED}.pt"; then
    log "  OK  ${ARM} s$SEED"; else log "  FAIL ${ARM} s$SEED (exit $?)"; fi; }

log "### train aimix INTERFERENCE=$INTERFERENCE +BALANCE=$BALANCE (mo): ${!ARM_FLAGS[*]} × seeds ${SEEDS[*]}  steps=$STEPS MAX=$MAX"
log "### COMMON: ${COMMON[*]}"
for SEED in "${SEEDS[@]}"; do
  for ARM in sac rdsac_mean rdsac_cvar; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAX" ]; do wait -n 2>/dev/null || sleep 5; done
    train_one "$ARM" "$SEED" &
  done
done
wait
log "### done"; ls "$CK"/*.pt 2>/dev/null | wc -l | xargs echo "checkpoints:"
