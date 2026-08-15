#!/usr/bin/env bash
# Produce the 16-seed RLPD arm for the aimix 5-arm eval by fine-tuning each
# rdsac_cvar_sXX base on the FAITHFUL live online-log (168-d, sacct-true JCT).
#
#   base  = runs/ckpts_aimix16/rdsac_cvar_sXX.pt   (warm-start the actor)
#   online= shadow_logs/transitions_20260814-203143.jsonl  (2786 real transitions)
#   offline prior = sim rollouts on the SAME hetero regime (--hetero-cluster),
#                   reward jct_aligned (matches the online-log −JCT/1000; the RLPD
#                   critic trains from scratch and needs offline↔online reward
#                   consistency, not parity with the base's mo-reward training).
#
# CPU only: the local 4070 is held by the Slurm worker via DRA (CUDA busy).
# Output collected as runs/ckpts_aimix16/rlpd_cvar_sXX.pt (serve/eval-loadable).
#
#   bash eval/scripts/train_rlpd_aimix16.sh
#   SEEDS="42 43" MAX=2 bash eval/scripts/train_rlpd_aimix16.sh   # subset
set -uo pipefail
cd /home/acane/Desktop/Kelpflux
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=""     # force CPU (4070 held by Slurm/DRA)

ONLINE_LOG="${ONLINE_LOG:-shadow_logs/transitions_20260814-203143.jsonl}"
CK="${CK:-runs/ckpts_aimix16}"                  # base + output checkpoints
BASE_ARM="${BASE_ARM:-rdsac_cvar}"              # warm-start source arm
OUT_ARM="${OUT_ARM:-rlpd_cvar}"                 # collected checkpoint prefix
OFFLINE_STEPS="${OFFLINE_STEPS:-50000}"
N_UPDATES="${N_UPDATES:-200}"
UTD="${UTD:-20}"
MAX="${MAX:-4}"                                 # concurrent fine-tunes
# Cap threads so MAX×threads ≈ cores (16). RLPD's 10-critic ensemble is heavier
# per update than plain RDSAC, so fewer concurrent procs than the train campaign.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
read -r -a SEEDS <<< "${SEEDS:-42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57}"

STAMP=$(date +%Y%m%d-%H%M%S)
LOG="runs/train_rlpd_aimix16_${STAMP}.log"
BASEDIR="/tmp/rlpd_base"
mkdir -p "$CK" runs "$BASEDIR"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

[ -f "$ONLINE_LOG" ] || { log "FATAL online-log missing: $ONLINE_LOG"; exit 1; }
log "### RLPD ${OUT_ARM} ← ${BASE_ARM}  seeds=${SEEDS[*]}  online=$(wc -l <"$ONLINE_LOG") tx"
log "### offline=$OFFLINE_STEPS updates=$N_UPDATES utd=$UTD MAX=$MAX OMP=$OMP_NUM_THREADS (CPU)"

train_one(){
  local SEED="$1"
  local BASE_CK="$CK/${BASE_ARM}_s${SEED}.pt"
  local OUT_CK="$CK/${OUT_ARM}_s${SEED}.pt"
  local BDIR="$BASEDIR/s${SEED}" ODIR="runs/rlpd_${OUT_ARM}_s${SEED}_${STAMP}"
  [ -f "$OUT_CK" ]  && { log "  SKIP s${SEED} (exists)"; return 0; }
  [ -f "$BASE_CK" ] || { log "  MISS base s${SEED}: $BASE_CK"; return 1; }
  mkdir -p "$BDIR"; cp -f "$BASE_CK" "$BDIR/dsac.pt"
  local t0=$(date +%s)
  if .venv-m11/bin/python -m services.rl_scheduler.rlpd_finetune \
       --base-policy "$BDIR" \
       --offline-steps "$OFFLINE_STEPS" --n-updates "$N_UPDATES" --utd-ratio "$UTD" \
       --online-log "$ONLINE_LOG" \
       --hetero-cluster --n-nodes 2 --gpus-per-node 1 \
       --trace-family aimix --n-jobs 50 \
       --fixed-alpha --init-alpha 0.05 \
       --out-dir "$ODIR" >>"$LOG" 2>&1 && cp "$ODIR/dsac.pt" "$OUT_CK"; then
    log "  OK  s${SEED}  ($(( $(date +%s) - t0 ))s) → $OUT_CK"
  else
    log "  FAIL s${SEED} (exit $?)"
  fi
}

for SEED in "${SEEDS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAX" ]; do wait -n 2>/dev/null || sleep 5; done
  train_one "$SEED" &
done
wait
n=$(ls "$CK/${OUT_ARM}"_s*.pt 2>/dev/null | wc -l)
log "### done — ${OUT_ARM} checkpoints: $n"
