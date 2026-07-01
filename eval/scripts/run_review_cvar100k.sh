#!/usr/bin/env bash
# Decisive #3 re-run at 100k steps (matches eval-writeup §4.4's budget) to
# disentangle: was §4.4's "RDSAC beats SAC at σ=1.0" a single-seed fluke, or did
# last night's 40k run simply under-train the sample-hungry IQN critic?
#
#   σ=1.0, fixed-α=0.05, arms {SAC, RDSAC-mean, RDSAC-cvar}, train seeds
#   42/43/44, families philly+ali → ΔJCT% vs score mean±std.
#
# ~8–10 h on the RTX 4070. Self-backgrounds via nohup. Kill with:
#   pkill -f run_review_cvar100k ; pkill -f sweep_stochastic
set -u
cd "$(dirname "$0")/../.." || exit 1

STEPS="${STEPS:-100000}"
DEVICE="${DEVICE:-cuda}"
PY="PYTHONPATH=. .venv-m11/bin/python"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="runs/review_cvar100k_${STAMP}.log"
mkdir -p runs

main() {
  echo "### cvar100k re-run started $(date) ; steps=${STEPS} device=${DEVICE}" | tee -a "$LOG"
  for SEED in 42 43 44; do
    local out="runs/review_cvar100k_s${SEED}"
    echo "=== [$(date +%H:%M:%S)] sweep → ${out} (train-seed ${SEED}) ===" | tee -a "$LOG"
    eval $PY eval/scripts/sweep_stochastic.py \
      --sigmas 1.0 --total-steps "$STEPS" --warmup-steps 2000 \
      --n-jobs 50 --seeds 42 43 44 45 46 --trace-families philly ali \
      --risk-modes mean cvar --fixed-alpha --init-alpha 0.05 \
      --n-nodes 1 --gpus-per-node 1 --train-seed "$SEED" \
      --device "$DEVICE" --out-dir "$out" >>"$LOG" 2>&1
    echo "=== [$(date +%H:%M:%S)] done ${out} (exit $?) ===" | tee -a "$LOG"
  done

  echo "=== [$(date +%H:%M:%S)] aggregating ===" | tee -a "$LOG"
  eval $PY - "$STAMP" >>"$LOG" 2>&1 <<'PYEOF'
import json, sys
from pathlib import Path
import numpy as np
stamp = sys.argv[1]
SEEDS = [42, 43, 44]; FAMS = ["philly", "ali"]
rows = {}
for s in SEEDS:
    f = Path(f"runs/review_cvar100k_s{s}/sweep.json")
    if not f.exists():
        continue
    for r in json.loads(f.read_text()):
        rows.setdefault((r["family"], r["model"]), []).append(
            (r.get("delta_pct", float("nan")), r.get("p99_h", float("nan"))))
out = [f"# #3 CVaR multi-seed @100k — {stamp}", "",
       "ΔJCT% vs score (mean±std across train seeds; + = beats score). p99 = tail JCT (h).",
       "", "| family | model | ΔJCT% (mean±std) | p99 (h) | n |",
       "|---|---|---:|---:|---:|"]
for fam in FAMS:
    for m in ["sac", "rdsac-mean", "rdsac-cvar"]:
        v = rows.get((fam, m))
        if not v:
            continue
        d = np.array([x[0] for x in v], float); d = d[np.isfinite(d)]
        p = np.array([x[1] for x in v], float)
        sd = np.std(d, ddof=1) if d.size > 1 else 0.0
        out.append(f"| {fam} | {m} | {np.mean(d):+.1f}±{sd:.1f} | {np.nanmean(p):.2f} | {d.size} |")
out += ["", "**Decisive read:** if rdsac-cvar now ≥ sac, §4.4 holds and 40k just "
        "under-trained the IQN critic. If sac still wins, §4.4 was single-seed luck "
        "and the RDSAC-superiority claim is refuted even in sim."]
dst = Path(f"runs/review_cvar100k_{stamp}_TABLES.md")
dst.write_text("\n".join(out))
print(f"[agg] wrote {dst}"); print("\n".join(out))
PYEOF
  echo "### cvar100k re-run finished $(date)" | tee -a "$LOG"
}

nohup bash -c "$(declare -f main); STEPS='${STEPS}' DEVICE='${DEVICE}' PY='${PY}' STAMP='${STAMP}' LOG='${LOG}' main" >/dev/null 2>&1 &
echo "cvar100k re-run launched (PID $!). Log: ${LOG}"
echo "Tail:  tail -f ${LOG}    Kill: pkill -f run_review_cvar100k; pkill -f sweep_stochastic"
