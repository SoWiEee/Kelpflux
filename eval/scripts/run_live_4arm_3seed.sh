#!/usr/bin/env bash
# Complete 2-node placement A/B: 4 arms (score / SAC / RDSAC-mean / RDSAC-cvar)
# × 3 train seeds (42/43/44) × {σ=0.0, σ=1.0}, submit-time -w explicit placement,
# drift-robust interleave. Aggregates to mean±std ΔJCT%/Δp99%/ΔCVaR% per (σ,arm).
#
# node-2 (rtx3080) was stabilised via pod-restart + scontrol reconfigure; this
# script re-ensures it before each seed (resume if it flapped) so a mid-run
# node-2 blip only costs the current seed, not the whole sweep.
#
#   bash eval/scripts/run_live_4arm_3seed.sh
set -uo pipefail
cd /home/acane/Desktop/Kelpflux
export KUBECONFIG=$HOME/.kube/config PYTHONPATH=.
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="runs/live4_3seed_${STAMP}.log"
GPU_NODES="slurm-worker-gpu-rtx4070-0,slurm-worker-gpu-rtx3080-0"
LOGIN="${LOGIN_POD:-slurm-login-7f8cfbc48-c875f}"
mkdir -p runs

kubectl port-forward -n slurm svc/rl-scheduler 8002:8002 >/tmp/pf_4arm3seed.log 2>&1 &
PF=$!
trap 'kill $PF 2>/dev/null' EXIT
sleep 6
curl -fsS http://localhost:8002/healthz >/dev/null 2>&1 && echo "[run] serve reachable" | tee -a "$LOG" || echo "[run] WARN serve unreachable" | tee -a "$LOG"

ensure_nodes() {
  kubectl -n slurm exec slurm-controller-0 -- bash -lc '
    st=$(scontrol show node slurm-worker-gpu-rtx3080-0 | grep -o "State=[^ ]*")
    echo "  node-2 pre-check: $st"
    case "$st" in
      *DOWN*|*NOT_RESPONDING*)
        echo "  node-2 flapped → reconfigure + resume"
        scontrol reconfigure 2>/dev/null; sleep 3
        scontrol update nodename=slurm-worker-gpu-rtx3080-0 state=resume 2>/dev/null; sleep 6
        scontrol show node slurm-worker-gpu-rtx3080-0 | grep -o "State=[^ ]*" ;;
    esac' 2>&1
}

for SEED in 42 43 44; do
  echo "=== [$(date +%H:%M:%S)] seed ${SEED} ===" | tee -a "$LOG"
  ensure_nodes | tee -a "$LOG"
  OUT="runs/live4_s${SEED}_${STAMP}"
  .venv-m11/bin/python -m eval.scripts.run_heavytail_ab \
    --serve-url http://localhost:8002 --login-pod "$LOGIN" --controller-pod slurm-controller-0 \
    --placement --gpu-nodes "$GPU_NODES" \
    --family philly --partition gpu --n-jobs 24 --target-max-s 30 \
    --sigmas 0.0 1.0 --rounds 3 --warmup 1 --interleave \
    --sac-ckpt /models/htab/sac_s${SEED}.pt \
    --rdsac-mean-ckpt /models/htab/rdsac_mean_s${SEED}.pt \
    --rdsac-cvar-ckpt /models/htab/rdsac_cvar_s${SEED}.pt \
    --out-dir "$OUT" >>"$LOG" 2>&1
  echo "=== [$(date +%H:%M:%S)] seed ${SEED} done (exit $?) → $OUT ===" | tee -a "$LOG"
done

echo "=== [$(date +%H:%M:%S)] aggregating ===" | tee -a "$LOG"
.venv-m11/bin/python - "$STAMP" >>"$LOG" 2>&1 <<'PYEOF'
import json, sys
from pathlib import Path
import numpy as np
stamp = sys.argv[1]
data = {}
seeds_seen = set()
for seed in [42, 43, 44]:
    f = Path(f"runs/live4_s{seed}_{stamp}/reports.json")
    if not f.exists():
        continue
    seeds_seen.add(seed)
    for rep in json.loads(f.read_text()):
        sig = rep.get("sigma")
        for arm, d in rep.get("paired_vs_score", {}).items():
            k = (sig, arm)
            data.setdefault(k, {"djct": [], "dp99": [], "dcvar": [], "p": []})
            data[k]["djct"].append(d.get("djct_pct", float("nan")))
            data[k]["dp99"].append(d.get("dp99_pct", float("nan")))
            data[k]["dcvar"].append(d.get("dcvar_pct", float("nan")))
            if "ttest_p" in d:
                data[k]["p"].append(d["ttest_p"])

def ms(xs):
    a = np.array([x for x in xs if x == x], float)
    if a.size == 0:
        return "—"
    sd = np.std(a, ddof=1) if a.size > 1 else 0.0
    return f"{a.mean():+.1f}±{sd:.1f}"

out = [f"# 2-node 4-arm × 3-seed placement A/B — {stamp}", "",
       f"seeds present: {sorted(seeds_seen)}  (submit-time -w explicit placement, "
       "drift-robust interleave, 3 rounds, n_jobs=24, partition=gpu)", "",
       "ΔJCT%/Δp99%/ΔCVaR% vs score (mean±std across train seeds; **+ = learned "
       "arm FASTER/lower than score = better**).", "",
       "| σ | arm | ΔJCT% | Δp99% | ΔCVaR% | n |", "|---|---|---:|---:|---:|---:|"]
for (sig, arm) in sorted(data.keys(), key=lambda k: (k[0], k[1])):
    d = data[(sig, arm)]
    n = len([x for x in d["djct"] if x == x])
    out.append(f"| {sig} | {arm} | {ms(d['djct'])} | {ms(d['dp99'])} | {ms(d['dcvar'])} | {n} |")
out += ["",
        "**Read:** + = learned beats score. If ΔJCT% are ≤0 / CI-crossing across "
        "arms, learned placement stays statistically ≤ score at 2×1 — consistent "
        "with the sim multi-seed (§4.3) and prior live (§4.2) findings. A robust "
        "positive would be the first live win and warrants promotion."]
Path(f"runs/live4_3seed_{stamp}_TABLES.md").write_text("\n".join(out))
print("\n".join(out))
print(f"[agg] wrote runs/live4_3seed_{stamp}_TABLES.md")
PYEOF
echo "LIVE4_3SEED_DONE ${STAMP}" | tee -a "$LOG"
