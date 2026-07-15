#!/usr/bin/env bash
# cuBLAS + hybrid 2-node live A/B for the reworked arm set:
#   score (baseline) + SAC + RDSAC-mean + RDSAC-cvar + UXP-RL (Lin et al. 2025).
# Reuses existing /tmp/lckpts/{sac,rdsac_mean,rdsac_cvar}_s{seed}.pt; UXP-RL from
# train_uxprl_seeds.sh. NON-INVASIVE: no slurm.conf swap / node drain — just
# serve /reload per arm + real GPU job submission (placement mode). A local serve
# on :8003 runs the CURRENT code (the deployed pod predates UXP-RL). The
# held-job watchdog runs alongside to auto-release node-2 restart casualties.
#
#   bash eval/scripts/run_uxprl_eval.sh              # both workloads, 8 seeds
#   WORKLOADS="cublas" SEEDS="42 43" bash eval/scripts/run_uxprl_eval.sh
set -uo pipefail
cd /home/acane/Desktop/Kelpflux
export KUBECONFIG=$HOME/.kube/config PYTHONPATH=.
CK=/tmp/lckpts
SERVE="http://localhost:8003"
NS=slurm; CTL=slurm-controller-0
LOGIN="${LOGIN_POD:-slurm-login-7f8cfbc48-c875f}"
GPU_NODES="${GPU_NODES:-slurm-worker-gpu-rtx3080-0,slurm-worker-gpu-rtx4070-0}"
MODEL="${MODEL:-/shared/models/qwen05b}"
read -r -a SEEDS <<< "${SEEDS:-42 43 44 45 46 47 48 49}"
read -r -a WORKLOADS <<< "${WORKLOADS:-cublas hybrid}"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="runs/uxprl_eval_${STAMP}.log"
mkdir -p runs
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ── local serve on :8003 (current code, so it can load UXP-RL checkpoints) ──
log "starting local serve :8003 (policy-dir $CK)"
SHADOW_MODE=true .venv-m11/bin/python -m services.rl_scheduler.serve \
  --policy-dir "$CK" --port 8003 >"runs/uxprl_serve_${STAMP}.log" 2>&1 &
SERVE_PID=$!
# ── held-job watchdog (auto-releases node-2 restart casualties) ──
bash eval/scripts/held_watchdog.sh >"runs/uxprl_watchdog_${STAMP}.log" 2>&1 &
WD_PID=$!
trap 'kill $SERVE_PID $WD_PID 2>/dev/null' EXIT
for i in $(seq 1 30); do curl -fsS "$SERVE/healthz" >/dev/null 2>&1 && break; sleep 2; done
curl -fsS "$SERVE/healthz" >/dev/null 2>&1 || { log "FATAL local serve unreachable"; exit 1; }
log "serve up; watchdog PID $WD_PID"

workload_flag(){ case "$1" in cublas) echo "--cuda-workload";; hybrid) echo "--hybrid-workload --llm-model $MODEL";; esac; }

for WL in "${WORKLOADS[@]}"; do
  log "===== workload=$WL ====="
  for SEED in "${SEEDS[@]}"; do
    miss=0; for A in sac rdsac_mean rdsac_cvar uxprl; do
      [ -f "${CK}/${A}_s${SEED}.pt" ] || { log "SKIP $WL s$SEED missing ${A}"; miss=1; break; }; done
    [ $miss -eq 1 ] && continue
    OUT="runs/uxprl_${WL}_s${SEED}_${STAMP}"
    log "  [$WL] seed $SEED → $OUT"
    # shellcheck disable=SC2046
    .venv-m11/bin/python -m eval.scripts.run_heavytail_ab \
      --serve-url "$SERVE" --login-pod "$LOGIN" --namespace $NS --controller-pod $CTL \
      --sac-ckpt "${CK}/sac_s${SEED}.pt" --rdsac-mean-ckpt "${CK}/rdsac_mean_s${SEED}.pt" \
      --rdsac-cvar-ckpt "${CK}/rdsac_cvar_s${SEED}.pt" --uxprl-ckpt "${CK}/uxprl_s${SEED}.pt" \
      --family philly --n-jobs "${N_JOBS:-30}" --seed "$SEED" --sigmas 1.0 \
      --rounds "${ROUNDS:-3}" --warmup 1 --interleave \
      $(workload_flag "$WL") --placement --gpu-nodes "$GPU_NODES" \
      --arrival-mode poisson --mps-oversub "${OVERSUB:-2.0}" --target-max-s 20 \
      --mps-buckets 25,50,75,100 --partition gpu \
      --out-dir "$OUT" >>"$LOG" 2>&1 || log "  [$WL] s$SEED exit $?"
  done
done

log "=== aggregating (4 metrics: 平均周轉/Makespan/P95/P99 + Δ平均周轉%) ==="
.venv-m11/bin/python - "$STAMP" "${WORKLOADS[*]}" "${SEEDS[*]}" >>"$LOG" 2>&1 <<'PY'
import json, sys
from pathlib import Path
import numpy as np
stamp, wls, seeds = sys.argv[1], sys.argv[2].split(), [int(s) for s in sys.argv[3].split()]
ARMS = ["score", "SAC", "RDSAC-mean", "RDSAC-cvar", "UXP-RL"]
def ms(xs, sign=False):
    a = np.array([x for x in xs if x == x], float)
    if a.size == 0: return "—"
    sd = np.std(a, ddof=1) if a.size > 1 else 0.0
    return (f"{a.mean():+.1f}±{sd:.1f}" if sign else f"{a.mean():.1f}±{sd:.1f}")
out = [f"# UXP-RL 重評估 — cuBLAS + hybrid 2-node ({stamp})", "",
       f"seeds={seeds}. 指標: 平均周轉時間(=mean JCT)/Makespan/P95/P99 (s)；配對 Δ平均周轉% vs score。", ""]
for wl in wls:
    absd, deld = {}, {}
    for s in seeds:
        f = Path(f"runs/uxprl_{wl}_s{s}_{stamp}/reports.json")
        if not f.exists(): continue
        for rep in json.loads(f.read_text()):
            for arm, p in rep["panels"].items():
                d = absd.setdefault(arm, {"mean": [], "mk": [], "p95": [], "p99": [], "done": []})
                d["mean"].append(p["mean"]); d["mk"].append(p.get("makespan", float("nan")))
                d["p95"].append(p["p95"]); d["p99"].append(p["p99"])
                d["done"].append(p["completed"] / p["n"] * 100 if p["n"] else float("nan"))
            for arm, dd in rep["paired_vs_score"].items():
                deld.setdefault(arm, {"djct": [], "p": []})
                deld[arm]["djct"].append(dd["djct_pct"]); deld[arm]["p"].append(dd.get("ttest_p", float("nan")))
    out.append(f"## workload = {wl}")
    out.append("| arm | 平均周轉(s) | Makespan(s) | P95(s) | P99(s) | done% | Δ平均周轉% | p(max) |")
    out.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for arm in ARMS:
        a = absd.get(arm)
        if not a: continue
        row = f"| {arm} | {ms(a['mean'])} | {ms(a['mk'])} | {ms(a['p95'])} | {ms(a['p99'])} | {ms(a['done'])} |"
        if arm == "score":
            row += " — | — |"
        else:
            dd = deld.get(arm, {"djct": [], "p": []})
            pmax = max([x for x in dd["p"] if x == x], default=float("nan"))
            row += f" {ms(dd['djct'], sign=True)} | {pmax:.1e} |"
        out.append(row)
    out.append("")
out.append("**判讀:** Δ平均周轉% 正 = 學習式比 score 快。Makespan/P95/P99 越低越好。")
Path(f"runs/uxprl_eval_{stamp}_TABLES.md").write_text("\n".join(out))
print("\n".join(out)); print(f"[agg] wrote runs/uxprl_eval_{stamp}_TABLES.md")
PY
log "UXPRL_EVAL_DONE ${STAMP}"
