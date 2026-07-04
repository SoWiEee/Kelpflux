#!/usr/bin/env bash
# Fair CrossQ eval with its NATIVE settings, removing both confounds of the
# fixed-α run (runs/crossq_compare_*):
#   - auto-α  (drop --fixed-alpha; CrossQ's BN is designed for auto temperature)
#   - update-matched: UTD=1 × 180k env-steps = 180k updates = RDSAC's 45k × UTD4.
# CrossQ-only (--no-sac --no-rdsac), 3 train-seed, σ=1.0, philly+ali, vs the SAME
# score baseline so ΔJCT% is directly comparable to crossq_compare's RDSAC-cvar.
#
#   STEPS=180000 DEVICE=cuda bash eval/scripts/run_crossq_native.sh
set -uo pipefail
cd /home/acane/Desktop/Kelpflux
export KUBECONFIG=$HOME/.kube/config PYTHONPATH=.
STEPS="${STEPS:-180000}"; DEVICE="${DEVICE:-cuda}"
STAMP="$(date +%Y%m%d-%H%M%S)"; LOG="runs/crossq_native_${STAMP}.log"
mkdir -p runs

main() {
  echo "### crossq NATIVE (auto-α, UTD=1, ${STEPS} steps) $(date)" | tee -a "$LOG"
  for SEED in 42 43 44; do
    echo "=== [$(date +%H:%M:%S)] train-seed ${SEED} ===" | tee -a "$LOG"
    PYTHONPATH=. .venv-m11/bin/python eval/scripts/sweep_stochastic.py \
      --sigmas 1.0 --total-steps "$STEPS" --warmup-steps 2000 \
      --n-jobs 50 --seeds 42 43 44 45 46 --trace-families philly ali \
      --no-sac --no-rdsac --crossq \
      --n-nodes 2 --gpus-per-node 1 --train-seed "$SEED" \
      --device "$DEVICE" --out-dir "runs/crossqN_s${SEED}_${STAMP}" >>"$LOG" 2>&1
    echo "=== [$(date +%H:%M:%S)] seed ${SEED} done (exit $?) ===" | tee -a "$LOG"
  done

  PYTHONPATH=. .venv-m11/bin/python - "$STAMP" >>"$LOG" 2>&1 <<'PY'
import json, sys
from pathlib import Path
import numpy as np
stamp = sys.argv[1]; SEEDS = [42, 43, 44]; FAMS = ["philly", "ali"]
rows = {}
for s in SEEDS:
    f = Path(f"runs/crossqN_s{s}_{stamp}/sweep.json")
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

out = [f"# CrossQ NATIVE (auto-alpha, UTD=1, update-matched 180k) — {stamp}", "",
       "ΔJCT% vs score (mean±std, 3 train seeds; + = beats score). Compare to the "
       "fixed-alpha/UTD1-45k run: RDSAC-cvar was -13.6/-17.1, crossq -50.3/-78.7.", "",
       "| family | model | ΔJCT% | completion | n |", "|---|---|---:|--:|--:|"]
for fam in FAMS:
    v = rows.get((fam, "crossq"))
    if not v:
        continue
    d, c, n = ms(v)
    out.append(f"| {fam} | crossq-native | {d} | {c} | {n} |")
out += ["",
        "**Read:** if crossq-native is now near RDSAC-cvar (-13~-17) it was merely "
        "under-trained/mis-tuned before; if it stays deeply negative it genuinely "
        "does not fit this problem. Either way, beating score is not expected "
        "(flat strategy space)."]
Path(f"runs/crossq_native_{stamp}_TABLES.md").write_text("\n".join(out))
print("\n".join(out)); print(f"[agg] runs/crossq_native_{stamp}_TABLES.md")
PY
  echo "CROSSQ_NATIVE_DONE ${STAMP}" | tee -a "$LOG"
}

nohup bash -c "$(declare -f main); STEPS='${STEPS}' DEVICE='${DEVICE}' LOG='${LOG}' STAMP='${STAMP}' main" >/dev/null 2>&1 &
echo "crossq native launched (PID $!). Log: ${LOG}"
