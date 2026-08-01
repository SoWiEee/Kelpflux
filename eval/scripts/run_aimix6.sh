#!/usr/bin/env bash
# aimix 6-arm unified A/B on ONE DRA backend:
#   fcfs | backfill | score | SAC | RDSAC-mean | RDSAC-cvar  (aimix workload)
# Learned arms trained with the multi-objective reward (−JCT + GPU utilization).
# 7 metrics: 平均JCT / P95 / P99 / Makespan / GPU利用率 / Slowdown / SLA違反率,
# + paired Δ平均JCT% vs score with Holm correction.
#
# Three Slurm configs, restored on exit (same shape as run_full8):
#   learned  = original (sched/backfill + priority/multifactor + JobSubmitPlugins=lua)
#              → run_heavytail_ab with all ckpts → score + SAC + RDSAC-mean/-cvar
#   fcfs     = sched/builtin + priority/basic, NO lua → score arm only (relabelled "fcfs")
#   backfill = sched/backfill + priority/basic, NO lua → score arm only (relabelled "backfill")
# The two Slurm-native configs strip the score Lua so they measure vanilla Slurm
# cons_tres placement — the "no smart scheduling" lower bound.
#
#   bash eval/scripts/run_aimix6.sh
#   SEEDS="42 43" CONFIGS="learned" bash eval/scripts/run_aimix6.sh
set -uo pipefail
cd /home/acane/Desktop/Kelpflux
export KUBECONFIG=$HOME/.kube/config PYTHONPATH=.
CK=/tmp/lckpts_aimix
SERVE="http://localhost:8003"
NS=slurm; CTL=slurm-controller-0; CM=slurm-config-static
LOGIN="${LOGIN_POD:-slurm-login-7f8cfbc48-c875f}"
# Node ORDER IS LOAD-BEARING: policies train with --node-speeds "1.0,0.25" (index 0 =
# FAST) and the gpu-type one-hot derives from that same speed, so index0⟺fast are
# perfectly correlated in training and the model follows POSITION. Fast (4070) MUST
# be first or every learned arm silently places onto the slow card.
GPU_NODES="${GPU_NODES:-slurm-worker-gpu-rtx4070-0,slurm-worker-gpu-rtx3080-0}"
MODEL="${MODEL:-/shared/models/qwen05b}"
# `read -r -a <<<` only consumes the FIRST line of a here-string, so a
# newline-separated SEEDS (e.g. SEEDS="$(seq 42 64)") would silently collapse to
# one seed and the run would look complete. Unquoted array expansion splits on
# IFS (space+tab+newline) instead.
# shellcheck disable=SC2206
SEEDS=( ${SEEDS:-42 43 44 45 46 47 48 49} )
# shellcheck disable=SC2206
CONFIGS=( ${CONFIGS:-learned fcfs backfill} )
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
TAG="${TAG:-aimix6}"
# Optional 7th arm: offline-fine-tuned RLPD (faithful RLPDAgent, serve-exported).
# CONFIGS="rlpd" RLPD_CKPT=/path/dsac.pt → run_heavytail_ab runs score + RLPD only.
RLPD_CKPT="${RLPD_CKPT:-}"
LOG="runs/${TAG}_${STAMP}.log"
ORIG=/tmp/slurm.conf.${TAG}orig.$$
mkdir -p runs
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

patch_conf(){ kubectl patch cm -n $NS $CM --type merge \
  -p "$(python3 -c "import json,sys;print(json.dumps({'data':{'slurm.conf':open('$1').read()}}))")" >/dev/null; }
restart_ctl_wait(){ kubectl delete pod -n $NS $CTL >/dev/null 2>&1
  for i in $(seq 1 40); do
    [ "$(kubectl get pod -n $NS $CTL -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null)" = "true" ] && return 0
    sleep 6; done; return 1; }
resume_nodes(){ kubectl exec -n $NS $CTL -- bash -lc \
  'scontrol reconfigure 2>/dev/null; for N in slurm-worker-gpu-rtx4070-0 slurm-worker-gpu-rtx3080-0; do
     for i in 1 2 3; do scontrol update nodename=$N state=resume 2>/dev/null; sleep 3; done; done' >/dev/null 2>&1 || true; }
restore_orig(){ log "RESTORING original slurm.conf"; patch_conf "$ORIG"
  restart_ctl_wait && log "cluster restored" || log "WARN restore not ready"; resume_nodes
  kill ${SERVE_PID:-0} ${WD_PID:-0} 2>/dev/null; }
trap 'restore_orig' EXIT

kubectl get cm -n $NS $CM -o jsonpath='{.data.slurm\.conf}' > "$ORIG"
[ -s "$ORIG" ] || { echo "FATAL cannot read slurm.conf"; exit 1; }
CONF_fcfs=/tmp/slurm.conf.${TAG}fcfs.$$; CONF_backfill=/tmp/slurm.conf.${TAG}bf.$$
sed -e 's|^SchedulerType=.*|SchedulerType=sched/builtin|' -e 's|^PriorityType=.*|PriorityType=priority/basic|' \
    -e '/^SchedulerParameters=/d' -e '/^JobSubmitPlugins=/d' "$ORIG" > "$CONF_fcfs"
sed -e 's|^PriorityType=.*|PriorityType=priority/basic|' -e '/^JobSubmitPlugins=/d' "$ORIG" > "$CONF_backfill"

# ── local serve on :8003 (current code, obs 168) ──
INIT_POLICY="${INIT_POLICY:-/tmp/aimix_serve_policy}"
mkdir -p "$INIT_POLICY"
if [ ! -f "$INIT_POLICY/dsac.pt" ]; then
  seed_ck=$(ls "$CK"/rdsac_cvar_s*.pt "$CK"/sac_s*.pt 2>/dev/null | head -1)
  [ -n "$seed_ck" ] && cp "$seed_ck" "$INIT_POLICY/dsac.pt" || { log "FATAL no checkpoint in $CK for serve init"; exit 1; }
fi
fuser -k 8003/tcp 2>/dev/null && { log "freed stale :8003"; sleep 2; }
SHADOW_MODE=true .venv-m11/bin/python -m services.rl_scheduler.serve \
  --policy-dir "$INIT_POLICY" --port 8003 >"runs/${TAG}_serve_${STAMP}.log" 2>&1 &
SERVE_PID=$!
bash eval/scripts/held_watchdog.sh >"runs/${TAG}_watchdog_${STAMP}.log" 2>&1 &
WD_PID=$!
for i in $(seq 1 30); do curl -fsS "$SERVE/healthz" >/dev/null 2>&1 && break; sleep 2; done
curl -fsS "$SERVE/healthz" >/dev/null 2>&1 || { log "FATAL serve unreachable"; exit 1; }
log "serve up (PID $SERVE_PID), watchdog $WD_PID"

# Warm python+torch+model into each node's page cache. MUST use sbatch: srun from the
# login pod cannot get an allocation here ("Socket timed out"), so an srun prewarm
# silently no-ops and leaves the cache cold → first LLM jobs TIMEOUT.
prewarm(){
  # aimix uses BERT + ResNet + Qwen; warm all three (torch import + NFS model/cache)
  # on each node so first jobs don't cold-load past their --time.
  log "prewarming BERT/ResNet/Qwen on both GPU nodes ..."
  local ids="" j
  for N in slurm-worker-gpu-rtx4070-0 slurm-worker-gpu-rtx3080-0; do
    for CMD in \
      "/shared/py/bin/python3 /shared/scripts/bert_job.py --mode infer --n 1 --batch-size 4" \
      "/shared/py/bin/python3 /shared/scripts/resnet_job.py --mode infer --n 1 --batch-size 4" \
      "/shared/py/bin/python3 /shared/scripts/llm_job.py --mode infer --n 1 --batch-size 4 --prompt-len 512 --gen-len 2 --model $MODEL"; do
      j=$(kubectl -n $NS exec "$LOGIN" -- bash -lc "printf '#!/bin/bash\n#SBATCH -p gpu -w $N --gres=mps:25 --time=8 -J prewarm\n$CMD\n' > /tmp/pw.sh; sbatch --parsable /tmp/pw.sh" 2>/dev/null | tr -d '\r')
      [ -n "$j" ] && ids="$ids $j"
    done
  done
  [ -z "$ids" ] && { log "  WARN prewarm submit failed; continuing cold"; return 0; }
  for i in $(seq 1 60); do sleep 10
    [ "$(kubectl -n $NS exec $CTL -- sacct -j "$(echo $ids|tr ' ' ',')" -X -n -o State 2>/dev/null | grep -cE 'RUNNING|PENDING')" = "0" ] && break
  done
  log "prewarm done ($(kubectl -n $NS exec $CTL -- sacct -j "$(echo $ids|tr ' ' ',')" -X -n -o State 2>/dev/null | tr '\n' ' '))"
}

apply_verify(){ # $1 conf  $2 want-SchedulerType  $3 name
  patch_conf "$1"; restart_ctl_wait || { log "FATAL slurmctld not ready ($3)"; exit 1; }
  local got; got=$(kubectl exec -n $NS $CTL -- scontrol show config 2>/dev/null | grep -oP 'SchedulerType\s*=\s*\K\S+')
  [ "$got" = "$2" ] || { log "FATAL SchedulerType=$got != $2 ($3)"; exit 1; }
  resume_nodes; log "config $3 active (SchedulerType=$got)"; }

common_args=(--serve-url "$SERVE" --login-pod "$LOGIN" --namespace $NS --controller-pod $CTL
  --family aimix --n-jobs "${N_JOBS:-30}" --sigmas 1.0 --rounds "${ROUNDS:-3}" --warmup 1 --interleave
  --aimix-workload --llm-model "$MODEL" --placement --gpu-nodes "$GPU_NODES"
  --arrival-mode poisson --mps-oversub "${OVERSUB:-2.0}" --target-max-s 20
  --mps-buckets 25,50,75,100 --partition gpu)

for CFG in "${CONFIGS[@]}"; do
  case "$CFG" in
    learned)  apply_verify "$ORIG"          "sched/backfill" learned ;;
    rlpd)     apply_verify "$ORIG"          "sched/backfill" rlpd ;;
    fcfs)     apply_verify "$CONF_fcfs"     "sched/builtin"  fcfs ;;
    backfill) apply_verify "$CONF_backfill" "sched/backfill" backfill ;;
  esac
  prewarm
  for SEED in "${SEEDS[@]}"; do
    OUT="runs/${TAG}_${CFG}_s${SEED}_${STAMP}"
    log "  [$CFG] seed $SEED → $OUT"
    if [ "$CFG" = "learned" ]; then
      miss=0; for A in sac rdsac_mean rdsac_cvar; do
        [ -f "${CK}/${A}_s${SEED}.pt" ] || { log "  SKIP s$SEED missing $A"; miss=1; break; }; done
      [ $miss -eq 1 ] && continue
      .venv-m11/bin/python -m eval.scripts.run_heavytail_ab "${common_args[@]}" --seed "$SEED" \
        --sac-ckpt "${CK}/sac_s${SEED}.pt" --rdsac-mean-ckpt "${CK}/rdsac_mean_s${SEED}.pt" \
        --rdsac-cvar-ckpt "${CK}/rdsac_cvar_s${SEED}.pt" \
        --out-dir "$OUT" >>"$LOG" 2>&1 || log "  [$CFG] s$SEED exit $?"
    elif [ "$CFG" = "rlpd" ]; then
      # 7th arm: score + RLPD only (single shared offline-fine-tuned checkpoint,
      # same convention as run_full8/run_slo8). RLPD pairs vs this run's score (CRN).
      [ -f "$RLPD_CKPT" ] || { log "  SKIP s$SEED missing RLPD_CKPT=$RLPD_CKPT"; continue; }
      .venv-m11/bin/python -m eval.scripts.run_heavytail_ab "${common_args[@]}" --seed "$SEED" \
        --rlpd-ckpt "$RLPD_CKPT" \
        --out-dir "$OUT" >>"$LOG" 2>&1 || log "  [$CFG] s$SEED exit $?"
    else
      # no ckpts → run_heavytail_ab runs the score arm only; with the Lua stripped this
      # measures vanilla Slurm cons_tres placement, relabelled $CFG at aggregation.
      .venv-m11/bin/python -m eval.scripts.run_heavytail_ab "${common_args[@]}" --seed "$SEED" \
        --out-dir "$OUT" >>"$LOG" 2>&1 || log "  [$CFG] s$SEED exit $?"
    fi
  done
done

log "=== aggregating ==="
.venv-m11/bin/python - "$STAMP" "$TAG" "${SEEDS[@]}" >>"$LOG" 2>&1 <<'PY'
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats
from eval.scripts.stage1_reanalysis import bootstrap_ci, tost, mde, holm_bonferroni, MARGIN
stamp, tag = sys.argv[1], sys.argv[2]; seeds = [int(s) for s in sys.argv[3:]]
LEARNED = ["SAC", "RDSAC-mean", "RDSAC-cvar"]
ORDER = ["score", "SAC", "RDSAC-mean", "RDSAC-cvar", "backfill", "fcfs"]

def panels(cfg, seed):
    f = Path(f"runs/{tag}_{cfg}_s{seed}_{stamp}/reports.json")
    if not f.exists(): return None
    out = {}
    for rep in json.loads(f.read_text()):
        for a, p in rep["panels"].items(): out.setdefault(a, []).append(p)
    return out

KEYS = ["mean", "p95", "p99", "makespan", "gpu_util", "slowdown_mean", "sla_viol", "ndone"]
def _seed_metric(ps, key):
    if key == "ndone":
        # COUNT of jobs that reached the metrics — join_records keeps only COMPLETED,
        # so TIMEOUT/FAILED jobs silently vanish from every mean. An arm that stalls
        # or misplaces jobs then looks FASTER (its slow jobs were dropped). Report it
        # so an imbalance vs score is visible instead of being read as a win.
        return float(np.mean([p["n"] for p in ps]))
    return float(np.nanmean([p.get(key, float("nan")) for p in ps]))

absd = {a: {k: [] for k in KEYS} for a in ORDER}
per = {a: [] for a in ORDER if a != "score"}   # per-seed ΔmeanJCT% vs that seed's score
for s in seeds:
    L = panels("learned", s)
    if L and "score" in L:
        sc = float(np.mean([p["mean"] for p in L["score"]]))
        for a, ps in L.items():
            if a in absd:
                for k in KEYS: absd[a][k].append(_seed_metric(ps, k))
        for a in LEARNED:
            if a in L: per[a].append(100.0*(sc-np.mean([p["mean"] for p in L[a]]))/sc)
        # Slurm-native configs: their "score" panel IS the native arm, paired to the
        # SAME seed's learned-config score (same job stream under CRN).
        for cfg in ("fcfs", "backfill"):
            N = panels(cfg, s)
            if N and "score" in N:
                for k in KEYS: absd[cfg][k].append(_seed_metric(N["score"], k))
                per[cfg].append(100.0*(sc-float(np.mean([p["mean"] for p in N["score"]])))/sc)

def ms(x, fmt="{:.1f}"):
    a = np.array([v for v in x if v == v], float)
    if not a.size: return "—"
    return f"{fmt.format(a.mean())}±{fmt.format(np.std(a, ddof=1) if a.size > 1 else 0.0)}"

arms = [a for a in ORDER if absd[a]["mean"]]
tested = [a for a in arms if a != "score" and len(per[a]) > 1]
raw = [float(stats.ttest_1samp(np.array(per[a]), 0.0)[1]) for a in tested]
holm = dict(zip(tested, holm_bonferroni(raw))) if tested else {}

out = [f"# aimix 六臂統一 A/B — {tag} {stamp}", "",
       f"seeds={seeds}；2×1、node order 4070→3080、reward=多目標(JCT+GPU利用率)。",
       "指標：平均JCT/P95/P99/Makespan(s)、GPU利用率、Slowdown、SLA違反率。Δ平均JCT% 正=比 score 快。", "",
       "| arm | 完成數 | 平均JCT(s) | P95(s) | P99(s) | Makespan(s) | GPU利用率 | Slowdown | SLA違反率 | Δ平均JCT% | Holm p |",
       "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
for a in arms:
    d = absd[a]
    row = (f"| {a} | {ms(d['ndone'], '{:.0f}')} | {ms(d['mean'])} | {ms(d['p95'])} | {ms(d['p99'])} | {ms(d['makespan'])} | "
           f"{ms(d['gpu_util'], '{:.2f}')} | {ms(d['slowdown_mean'], '{:.2f}')} | {ms(d['sla_viol'], '{:.2f}')} |")
    if a == "score":
        out.append(row + " （基準） | — |")
    else:
        x = np.array(per[a], float)
        dl = f"{x.mean():+.1f}±{(np.std(x, ddof=1) if x.size > 1 else 0):.1f}" if x.size else "—"
        hp = f"{holm[a]:.3f}" if a in holm else "—"
        out.append(row + f" {dl} | {hp} |")
if tested:
    sd_pool = float(np.sqrt(np.mean([np.var(per[a], ddof=1) for a in tested])))
    out += ["", f"pooled SD={sd_pool:.1f}% → MDE(n={len(seeds)}, power .8) ≈ {mde(sd_pool, len(seeds)):.1f}%",
            "判讀：Holm 校正後仍顯著才是穩健差異；TOST『是』= 90%CI 落在 ±5% 內（證實等價）；兩者皆否 = 此規模下不可區分。",
            "**先看『完成數』**：僅 COMPLETED 的 job 進入各項平均（TIMEOUT/FAILED 會被丟棄），故完成數明顯低於 score 的 arm，其較低的 JCT 是倖存者偏差而非較快；此時 Δ平均JCT% 不可採信。"]
Path(f"runs/{tag}_{stamp}_TABLES.md").write_text("\n".join(out))
print("\n".join(out)); print(f"[agg] wrote runs/{tag}_{stamp}_TABLES.md")
PY
log "AIMIX6_DONE ${STAMP}"
