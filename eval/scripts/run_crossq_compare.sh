#!/usr/bin/env bash
# CrossQ vs SAC vs RDSAC(mean/cvar) vs score — sim 2×1, 3 train-seed, σ=1.0.
# fixed-α=0.05 across all arms removes the auto-α confound so the comparison
# isolates the critic family (CrossQ BN/no-target vs IQN/target vs scalar/target).
#
#   STEPS=45000 DEVICE=cuda bash eval/scripts/run_crossq_compare.sh
# Self-backgrounds via nohup. Kill: pkill -f run_crossq_compare; pkill -f sweep_stochastic
set -uo pipefail
cd /home/acane/Desktop/Kelpflux
export KUBECONFIG=$HOME/.kube/config PYTHONPATH=.
STEPS="${STEPS:-45000}"; DEVICE="${DEVICE:-cuda}"
STAMP="$(date +%Y%m%d-%H%M%S)"; LOG="runs/crossq_compare_${STAMP}.log"
mkdir -p runs

main() {
  echo "### crossq compare $(date) steps=${STEPS} device=${DEVICE}" | tee -a "$LOG"
  for SEED in 42 43 44; do
    echo "=== [$(date +%H:%M:%S)] train-seed ${SEED} ===" | tee -a "$LOG"
    PYTHONPATH=. .venv-m11/bin/python eval/scripts/sweep_stochastic.py \
      --sigmas 1.0 --total-steps "$STEPS" --warmup-steps 2000 \
      --n-jobs 50 --seeds 42 43 44 45 46 --trace-families philly ali \
      --risk-modes mean cvar --crossq --fixed-alpha --init-alpha 0.05 \
      --n-nodes 2 --gpus-per-node 1 --train-seed "$SEED" \
      --device "$DEVICE" --out-dir "runs/crossq_s${SEED}_${STAMP}" >>"$LOG" 2>&1
    echo "=== [$(date +%H:%M:%S)] seed ${SEED} done (exit $?) ===" | tee -a "$LOG"
  done

  PYTHONPATH=. .venv-m11/bin/python - "$STAMP" >>"$LOG" 2>&1 <<'PY'
import json, sys
from pathlib import Path
import numpy as np
stamp = sys.argv[1]; SEEDS = [42, 43, 44]; FAMS = ["philly", "ali"]
rows = {}
for s in SEEDS:
    f = Path(f"runs/crossq_s{s}_{stamp}/sweep.json")
    if not f.exists():
        continue
    for r in json.loads(f.read_text()):
        rows.setdefault((r["family"], r["model"]), []).append(
            (r.get("delta_pct", float("nan")), r.get("completed_frac", float("nan"))))

def ms(v):
    d = np.array([x[0] for x in v], float); d = d[np.isfinite(d)]
    c = np.array([x[1] for x in v], float)
    sd = np.std(d, ddof=1) if d.size > 1 else 0.0
    return f"{np.mean(d):+.1f}±{sd:.1f}", f"{np.mean(c):.0%}", int(d.size)

out = [f"# CrossQ vs SAC vs RDSAC vs score — {stamp} (2×1, sigma=1.0, 45k, fixed-alpha=0.05)", "",
       "ΔJCT% vs score (mean±std across 3 train seeds; + = beats score).", "",
       "| family | model | ΔJCT% | completion | n |", "|---|---|---:|--:|--:|"]
for fam in FAMS:
    for m in ["sac", "rdsac-mean", "rdsac-cvar", "crossq"]:
        v = rows.get((fam, m))
        if not v:
            continue
        d, c, n = ms(v)
        out.append(f"| {fam} | {m} | {d} | {c} | {n} |")
out += ["",
        "**Read:** does CrossQ (newer, stable, no target-net) match/beat the RDSAC "
        "family + SAC, and how close to score? All-negative = still no learned win at "
        "2x1 (flat strategy space), but CrossQ should train cleaner / lower-variance."]
Path(f"runs/crossq_compare_{stamp}_TABLES.md").write_text("\n".join(out))
print("\n".join(out)); print(f"[agg] runs/crossq_compare_{stamp}_TABLES.md")
PY
  echo "CROSSQ_COMPARE_DONE ${STAMP}" | tee -a "$LOG"
}

nohup bash -c "$(declare -f main); STEPS='${STEPS}' DEVICE='${DEVICE}' LOG='${LOG}' STAMP='${STAMP}' main" >/dev/null 2>&1 &
echo "crossq compare launched (PID $!). Log: ${LOG}"
