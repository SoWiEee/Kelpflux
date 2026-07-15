#!/usr/bin/env bash
# Train the 8 UXP-RL (Lin et al. 2025, faithful DQN) checkpoints for the
# cuBLAS + hybrid live A/B. The SAC / RDSAC-mean / RDSAC-cvar arms are REUSED
# from the existing /tmp/lckpts/{arm}_s{seed}.pt (trained 2026-07-07, 2×1
# 166-dim); only UXP-RL is new, so this trains just uxprl_s42..s49.pt.
#
# Env matched to the opponent arms: 2×1 heterogeneous (4070 fast + 3080 slow
# 0.25×), philly+ali traces, n_jobs=50, curriculum. DQN is cheap per step
# (no IQN/actor), so a generous 200k-step budget still finishes fast.
set -uo pipefail
cd /home/acane/Desktop/Kelpflux
export PYTHONPATH=.
CK=/tmp/lckpts
STEPS="${STEPS:-200000}"
DEVICE="${DEVICE:-cuda}"
read -r -a SEEDS <<< "${SEEDS:-42 43 44 45 46 47 48 49}"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="runs/train_uxprl_${STAMP}.log"
mkdir -p "$CK" runs
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "### train UXP-RL seeds ${SEEDS[*]}  steps=$STEPS device=$DEVICE"
for SEED in "${SEEDS[@]}"; do
  OUT="runs/uxprl_s${SEED}_${STAMP}"
  log "  [uxprl] seed $SEED → $OUT"
  .venv-m11/bin/python -m services.rl_scheduler.sim_train --uxprl \
    --n-nodes 2 --gpus-per-node 1 --node-speeds "1.0,0.25" \
    --trace philly ali --n-jobs 50 --curriculum \
    --total-steps "$STEPS" --warmup-steps 2000 \
    --seed "$SEED" --device "$DEVICE" --out-dir "$OUT" >>"$LOG" 2>&1 \
    && cp "$OUT/dsac.pt" "$CK/uxprl_s${SEED}.pt" \
    && log "  OK  uxprl s$SEED -> $CK/uxprl_s${SEED}.pt" \
    || log "  FAIL uxprl s$SEED (exit $?)"
done
log "### done. checkpoints:"; ls -la "$CK"/uxprl_s*.pt 2>&1 | tee -a "$LOG"
