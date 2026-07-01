#!/usr/bin/env bash
# P0 review experiments (docs/review.md §5): close the "RDSAC never wins" gap.
#
#   #3  CVaR multi-seed ablation — σ=1.0, fixed-α, arms {SAC, RDSAC-mean,
#       RDSAC-cvar}, 3 training seeds → mean±std (breaks the single-seed caveat
#       in eval-writeup §4.4.2).
#   #1  Scale crossover — σ=1.0, arms {SAC, RDSAC-cvar} vs score, across
#       1×1 / 2×1 / 2×2 topologies → does the learned policy's edge grow with
#       scale? (the "value requires scale" claim, currently only asserted.)
#
#   #2 (RDSAC's risk machinery earns its keep under σ) is NOT retrained here —
#       it reuses runs/stoch_fixedA_* / runs/stoch_sweep_2x1_* (eval §4.4); the
#       σ=1.0 arm of #3 re-confirms it with multi-seed.
#
# Self-backgrounds via nohup so it survives the launching shell. ~5–6 h on the
# RTX 4070. Override step budget:  STEPS=30000 bash eval/scripts/run_review_p0.sh
set -u
cd "$(dirname "$0")/../.." || exit 1

STEPS="${STEPS:-40000}"
DEVICE="${DEVICE:-cuda}"
PY="PYTHONPATH=. .venv-m11/bin/python"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="runs/review_p0_${STAMP}.log"
mkdir -p runs

run_sweep() {  # $1=out-dir ; rest = extra args
  local out="$1"; shift
  echo "=== [$(date +%H:%M:%S)] sweep → ${out} :: $* ===" | tee -a "$LOG"
  eval $PY eval/scripts/sweep_stochastic.py \
    --sigmas 1.0 --total-steps "$STEPS" --warmup-steps 2000 \
    --n-jobs 50 --seeds 42 43 44 45 46 --trace-families philly ali \
    --fixed-alpha --init-alpha 0.05 --device "$DEVICE" \
    --out-dir "$out" "$@" >>"$LOG" 2>&1
  echo "=== [$(date +%H:%M:%S)] done ${out} (exit $?) ===" | tee -a "$LOG"
}

main() {
  echo "### P0 review sweep started $(date) ; steps=${STEPS} device=${DEVICE}" | tee -a "$LOG"

  # ---- #3 CVaR multi-seed ablation (1×1, 3 train seeds) -------------------
  for SEED in 42 43 44; do
    run_sweep "runs/review_cvar_s${SEED}" \
      --risk-modes mean cvar --n-nodes 1 --gpus-per-node 1 --train-seed "$SEED"
  done

  # ---- #1 Scale crossover (single train seed, cvar arm) ------------------
  run_sweep "runs/review_scale_1x1" \
    --risk-mode cvar --n-nodes 1 --gpus-per-node 1 --train-seed 42
  run_sweep "runs/review_scale_2x1" \
    --risk-mode cvar --n-nodes 2 --gpus-per-node 1 --train-seed 42 --node-speeds 1.0,0.25
  run_sweep "runs/review_scale_2x2" \
    --risk-mode cvar --n-nodes 2 --gpus-per-node 2 --train-seed 42 --node-speeds 1.0,0.25

  # ---- aggregate → paper-ready tables ------------------------------------
  echo "=== [$(date +%H:%M:%S)] aggregating ===" | tee -a "$LOG"
  eval $PY eval/scripts/agg_review_p0.py "${STAMP}" >>"$LOG" 2>&1
  echo "### P0 review sweep finished $(date)" | tee -a "$LOG"
}

nohup bash -c "$(declare -f main run_sweep); STEPS='${STEPS}' DEVICE='${DEVICE}' PY='${PY}' STAMP='${STAMP}' LOG='${LOG}' main" >/dev/null 2>&1 &
echo "P0 review sweep launched (PID $!). Log: ${LOG}"
echo "Tail with:  tail -f ${LOG}"
