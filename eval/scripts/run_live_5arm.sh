#!/usr/bin/env bash
# LIVE real-CUDA 2-node method comparison: score / SAC / RDSAC-mean / RDSAC-cvar
# / CrossQ, 3 train-seed, submit-time -w explicit placement, drift-robust
# interleave. Real-machine comparison (training is sim; live is the point).
#
# Uses a LOCAL serve on :8003 (the deployed pod image predates the CrossQ code,
# so its DSACAgent.load can't build the BN critic). Local serve = same serve.py
# with current code; /act is self-contained (no snapshot needed for placement).
# Start it first:  PYTHONPATH=. python /tmp/lserve.py   (policy-dir /tmp/lckpts)
#
# Node mapping 4070,3080 matches training (node_j=0 = fast card).
#   bash eval/scripts/run_live_5arm.sh
set -uo pipefail
cd /home/acane/Desktop/Kelpflux
export KUBECONFIG=$HOME/.kube/config PYTHONPATH=.
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="runs/live5arm_${STAMP}.log"
LOGIN="${LOGIN_POD:-slurm-login-7f8cfbc48-c875f}"
GPU_NODES="slurm-worker-gpu-rtx4070-0,slurm-worker-gpu-rtx3080-0"  # index0=4070=fast=node_j0 (matches training)
SERVE="http://localhost:8003"
CK=/tmp/lckpts
mkdir -p runs

echo "### live 5-arm (local serve) $(date)" | tee "$LOG"
curl -fsS "$SERVE/healthz" >/dev/null 2>&1 && echo "[run] local serve reachable" | tee -a "$LOG" \
  || { echo "[run] FATAL local serve unreachable at $SERVE — start /tmp/lserve.py first" | tee -a "$LOG"; exit 1; }

ensure_nodes() {
  kubectl -n slurm exec slurm-controller-0 -- bash -lc '
    st=$(scontrol show node slurm-worker-gpu-rtx3080-0 | grep -o "State=[^ ]*")
    echo "  node-2 pre-check: $st"
    case "$st" in *DOWN*|*NOT_RESPONDING*)
      scontrol reconfigure 2>/dev/null; sleep 3
      scontrol update nodename=slurm-worker-gpu-rtx3080-0 state=resume 2>/dev/null; sleep 6 ;;
    esac' 2>&1
}

for SEED in 42 43 44; do
  echo "=== [$(date +%H:%M:%S)] seed ${SEED} ===" | tee -a "$LOG"
  ensure_nodes | tee -a "$LOG"
  OUT="runs/live5arm_s${SEED}_${STAMP}"
  .venv-m11/bin/python -m eval.scripts.run_heavytail_ab \
    --serve-url "$SERVE" --login-pod "$LOGIN" \
    --namespace slurm --controller-pod slurm-controller-0 \
    --sac-ckpt "${CK}/sac_s${SEED}.pt" \
    --rdsac-mean-ckpt "${CK}/rdsac_mean_s${SEED}.pt" \
    --rdsac-cvar-ckpt "${CK}/rdsac_cvar_s${SEED}.pt" \
    --crossq-ckpt "${CK}/crossq_s${SEED}.pt" \
    --family philly --n-jobs 30 --seed 42 \
    --sigmas 1.0 --rounds 3 --warmup 1 --interleave \
    --cuda-workload --placement --gpu-nodes "$GPU_NODES" \
    --arrival-mode poisson --mps-oversub 1.0 --target-max-s 20 --mps-buckets 25,50,75,100 \
    --partition gpu --out-dir "$OUT" >>"$LOG" 2>&1
  echo "=== [$(date +%H:%M:%S)] seed ${SEED} done (exit $?) → $OUT ===" | tee -a "$LOG"
done

echo "=== [$(date +%H:%M:%S)] aggregating ===" | tee -a "$LOG"
.venv-m11/bin/python - "$STAMP" >>"$LOG" 2>&1 <<'PY'
import json, sys
from pathlib import Path
import numpy as np
stamp = sys.argv[1]; SEEDS = [42, 43, 44]
absd = {}; deld = {}; seen = set()
for s in SEEDS:
    f = Path(f"runs/live5arm_s{s}_{stamp}/reports.json")
    if not f.exists():
        continue
    seen.add(s)
    for rep in json.loads(f.read_text()):
        for arm, p in rep["panels"].items():
            absd.setdefault(arm, {"mean": [], "p99": [], "cvar": []})
            absd[arm]["mean"].append(p["mean"]); absd[arm]["p99"].append(p["p99"]); absd[arm]["cvar"].append(p["cvar"])
        for arm, d in rep["paired_vs_score"].items():
            deld.setdefault(arm, {"djct": [], "dp99": [], "dcvar": [], "p": []})
            deld[arm]["djct"].append(d["djct_pct"]); deld[arm]["dp99"].append(d["dp99_pct"])
            deld[arm]["dcvar"].append(d["dcvar_pct"]); deld[arm]["p"].append(d.get("ttest_p", float("nan")))
def ms(xs):
    a = np.array([x for x in xs if x == x], float)
    return "—" if a.size == 0 else f"{a.mean():.1f}±{np.std(a,ddof=1) if a.size>1 else 0:.1f}"
def msp(xs):
    a = np.array([x for x in xs if x == x], float)
    return "—" if a.size == 0 else f"{a.mean():+.1f}±{np.std(a,ddof=1) if a.size>1 else 0:.1f}"
arms = ["score", "SAC", "RDSAC-mean", "RDSAC-cvar", "CrossQ"]
out = [f"# LIVE real-CUDA 5-arm method comparison — {stamp}", "",
       f"seeds={sorted(seen)}. 2-node, submit-time -w, drift-robust interleave, "
       "cuBLAS workload, node_j0=4070(fast). JCT/p99/CVaR in s.", "",
       "| arm | JCT(s) | p99(s) | CVaR(s) | ΔJCT% | Δp99% | ΔCVaR% | p(max) |",
       "|---|--:|--:|--:|--:|--:|--:|--:|"]
for arm in arms:
    a = absd.get(arm)
    if not a:
        continue
    row = f"| {arm} | {ms(a['mean'])} | {ms(a['p99'])} | {ms(a['cvar'])} |"
    if arm == "score":
        row += " — | — | — | — |"
    else:
        d = deld[arm]; pmax = max([x for x in d['p'] if x == x], default=float('nan'))
        row += f" {msp(d['djct'])} | {msp(d['dp99'])} | {msp(d['dcvar'])} | {pmax:.1e} |"
    out.append(row)
out += ["", "**Read:** the real-machine method comparison. + = learned beats score. "
        "Does any of SAC(2019)/RDSAC(2020)/CrossQ(2024) beat the heuristic live?"]
Path(f"runs/live5arm_{stamp}_TABLES.md").write_text("\n".join(out))
print("\n".join(out)); print(f"[agg] runs/live5arm_{stamp}_TABLES.md")
PY
echo "LIVE5ARM_DONE ${STAMP}" | tee -a "$LOG"
