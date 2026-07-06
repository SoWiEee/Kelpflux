#!/usr/bin/env bash
# REAL-CUDA 2-node placement A/B: 4 arms × 3 train seeds × {σ0,σ1}, submit-time
# -w explicit placement, drift-robust interleave. Unlike the sleep variant, this
# uses cuBLAS gpu_workload (--cuda-workload) so GPU heterogeneity (4070 fast vs
# 3080 slow) + MPS co-residency actually surface in JCT.
#
# Node mapping FIXED to the serve snapshot order (node_j=0→3080, node_j=1→4070)
# so the RL node choice is applied to the node the model actually meant.
#
#   bash eval/scripts/run_live_4arm_3seed_cuda.sh
set -uo pipefail
cd /home/acane/Desktop/Kelpflux
export KUBECONFIG=$HOME/.kube/config PYTHONPATH=.
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="runs/live4cuda_3seed_${STAMP}.log"
# snapshot order: node_j=0 → 3080, node_j=1 → 4070
GPU_NODES="slurm-worker-gpu-rtx3080-0,slurm-worker-gpu-rtx4070-0"
LOGIN="${LOGIN_POD:-slurm-login-7f8cfbc48-c875f}"
mkdir -p runs

kubectl port-forward -n slurm svc/rl-scheduler 8002:8002 >/tmp/pf_4armcuda.log 2>&1 &
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
  OUT="runs/live4cuda_s${SEED}_${STAMP}"
  .venv-m11/bin/python -m eval.scripts.run_heavytail_ab \
    --serve-url http://localhost:8002 --login-pod "$LOGIN" --controller-pod slurm-controller-0 \
    --placement --gpu-nodes "$GPU_NODES" \
    --cuda-workload \
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
absd={}; deld={}; seen=set()
for s in [42,43,44]:
    f=Path(f"runs/live4cuda_s{s}_{stamp}/reports.json")
    if not f.exists(): continue
    seen.add(s)
    for rep in json.loads(f.read_text()):
        sig=rep["sigma"]
        for arm,p in rep["panels"].items():
            k=(sig,arm); absd.setdefault(k,{"mean":[],"p99":[],"cvar":[],"sd99":[],"done":[]})
            absd[k]["mean"].append(p["mean"]); absd[k]["p99"].append(p["p99"])
            absd[k]["cvar"].append(p["cvar"]); absd[k]["sd99"].append(p["slowdown_p99"])
            absd[k]["done"].append(p["completed"]/p["n"]*100)
        for arm,d in rep["paired_vs_score"].items():
            k=(sig,arm); deld.setdefault(k,{"djct":[],"dp99":[],"dcvar":[],"p":[]})
            deld[k]["djct"].append(d["djct_pct"]); deld[k]["dp99"].append(d["dp99_pct"])
            deld[k]["dcvar"].append(d["dcvar_pct"]); deld[k]["p"].append(d.get("ttest_p",float("nan")))
def ms(xs):
    a=np.array([x for x in xs if x==x],float)
    if a.size==0: return "—"
    return f"{a.mean():.1f}±{np.std(a,ddof=1) if a.size>1 else 0.0:.1f}"
def msp(xs):
    a=np.array([x for x in xs if x==x],float)
    if a.size==0: return "—"
    return f"{a.mean():+.1f}±{np.std(a,ddof=1) if a.size>1 else 0.0:.1f}"
arms=["score","SAC","RDSAC-mean","RDSAC-cvar"]
out=[f"# REAL-CUDA 2-node 4-arm × 3-seed placement A/B — {stamp}","",
     f"seeds={sorted(seen)}. gpu_workload cuBLAS (heterogeneity+MPS surface). "
     "node_j fixed to snapshot order (0→3080, 1→4070). JCT/p99/CVaR in s.",""]
for sig in [0.0,1.0]:
    out.append(f"## σ={sig}")
    out.append("| arm | JCT(s) | p99(s) | CVaR(s) | slowdn_p99 | done% | ΔJCT% | Δp99% | ΔCVaR% | p(max) |")
    out.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for arm in arms:
        a=absd.get((sig,arm))
        if not a: continue
        row=f"| {arm} | {ms(a['mean'])} | {ms(a['p99'])} | {ms(a['cvar'])} | {ms(a['sd99'])} | {ms(a['done'])} |"
        if arm=="score": row+=" — | — | — | — |"
        else:
            d=deld[(sig,arm)]; pmax=max([x for x in d['p'] if x==x],default=float('nan'))
            row+=f" {msp(d['djct'])} | {msp(d['dp99'])} | {msp(d['dcvar'])} | {pmax:.1e} |"
        out.append(row)
    out.append("")
out+=["**Read:** + = learned beats score. Real-CUDA surfaces heterogeneity (4070 "
      "fast/3080 slow) + MPS interference that the sleep variant (表4-1) could not."]
Path(f"runs/live4cuda_3seed_{stamp}_TABLES.md").write_text("\n".join(out))
print("\n".join(out)); print(f"[agg] wrote runs/live4cuda_3seed_{stamp}_TABLES.md")
PYEOF
echo "LIVE4CUDA_3SEED_DONE ${STAMP}" | tee -a "$LOG"
