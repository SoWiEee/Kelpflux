#!/usr/bin/env bash
# UNIFIED full-method live A/B (§4.2.3), all measured TODAY under DRA MPS so every
# arm is comparable to the SAME score baseline (fixes the device-plugin-vs-DRA
# confound). Three Slurm configs, restored on exit:
#   config1 "learned" = original (backfill+multifactor+score-Lua): run_heavytail_ab
#            with all ckpts → score + SAC + RDSAC-mean + RDSAC-cvar + CrossQ + RLPD.
#   config2 "fcfs"    = sched/builtin + priority/basic, no Lua: score arm only.
#   config3 "backfill"= sched/backfill + priority/basic, no Lua: score arm only.
# Hybrid Qwen workload, oversub 2.0, 8 workload seeds, --placement (paced; score/RL
# place via place_fn, Slurm-native place via cons_tres). Seed-level significance.
set -uo pipefail
cd /home/acane/Desktop/Kelpflux
export KUBECONFIG=$HOME/.kube/config PYTHONPATH=.
STAMP=$(date +%Y%m%d-%H%M%S)
TAG="${TAG:-full8}"
LOG="runs/${TAG}_${STAMP}.log"
CM=slurm-config-static; NS=slurm; CTL=slurm-controller-0
LOGIN="${LOGIN_POD:-slurm-login-7f8cfbc48-c875f}"
GPU_NODES="slurm-worker-gpu-rtx4070-0,slurm-worker-gpu-rtx3080-0"
SERVE="http://localhost:8003"; MODEL="${MODEL:-/shared/models/qwen05b}"
CK=/tmp/lckpts; RLPD_CKPT="${RLPD_CKPT:-${CK}/rlpd_v3.pt}"
read -r -a SEEDS <<< "${SEEDS:-42 43 44 45 46 47 48 49}"
ORIG=/tmp/slurm.conf.f8orig.$$
mkdir -p runs
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
patch_conf(){ kubectl patch cm -n $NS $CM --type merge -p "$(python3 -c "import json;print(json.dumps({'data':{'slurm.conf':open('$1').read()}}))")" >/dev/null; }
restart_ctl_wait(){ kubectl delete pod -n $NS $CTL >/dev/null 2>&1; for i in $(seq 1 30); do [ "$(kubectl get pod -n $NS $CTL -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null)" = "true" ] && return 0; sleep 6; done; return 1; }
resume_nodes(){ kubectl exec -n $NS $CTL -- bash -lc 'scontrol reconfigure 2>/dev/null; for N in slurm-worker-gpu-rtx4070-0 slurm-worker-gpu-rtx3080-0; do for i in 1 2 3; do scontrol update nodename=$N state=resume 2>/dev/null; sleep 3; done; done' >/dev/null 2>&1 || true; }
restore_orig(){ log "RESTORING original slurm.conf"; patch_conf "$ORIG"; restart_ctl_wait && log "cluster restored" || log "WARN restore not ready"; resume_nodes; }
trap 'restore_orig' EXIT
kubectl get cm -n $NS $CM -o jsonpath='{.data.slurm\.conf}' > "$ORIG"; [ -s "$ORIG" ] || { echo FATAL cannot read slurm.conf; exit 1; }
CONF_fcfs=/tmp/slurm.conf.f8fcfs.$$; CONF_backfill=/tmp/slurm.conf.f8bf.$$
sed -e 's|^SchedulerType=.*|SchedulerType=sched/builtin|' -e 's|^PriorityType=.*|PriorityType=priority/basic|' -e '/^SchedulerParameters=/d' -e '/^JobSubmitPlugins=/d' "$ORIG" > "$CONF_fcfs"
sed -e 's|^PriorityType=.*|PriorityType=priority/basic|' -e '/^JobSubmitPlugins=/d' "$ORIG" > "$CONF_backfill"
log "### $TAG unified full-method A/B $(date)"
curl -fsS "$SERVE/healthz" >/dev/null 2>&1 || { log "FATAL serve unreachable"; exit 1; }

prewarm(){ for N in slurm-worker-gpu-rtx4070-0 slurm-worker-gpu-rtx3080-0; do timeout 180 kubectl -n $NS exec "$LOGIN" -- bash -lc "srun -p gpu -w $N --gres=mps:25 --time=5 /shared/py/bin/python3 /shared/scripts/llm_job.py --mode infer --n 1 --batch-size 4 --prompt-len 512 --gen-len 2 --model $MODEL 2>&1|tail -1" >/dev/null 2>&1 & done; wait; }
apply_verify(){ # $1 conf  $2 want-SchedulerType  $3 name
  patch_conf "$1"; restart_ctl_wait || { log "FATAL slurmctld not ready ($3)"; exit 1; }
  local got; got=$(kubectl exec -n $NS $CTL -- scontrol show config 2>/dev/null | grep -oP 'SchedulerType\s*=\s*\K\S+')
  [ "$got" = "$2" ] || { log "FATAL SchedulerType=$got != $2 ($3)"; exit 1; }
  resume_nodes; log "config $3 active (SchedulerType=$got)"; }

# ---- config1: learned (original config, Lua on) ----
apply_verify "$ORIG" "sched/backfill" "learned"
prewarm
for SEED in "${SEEDS[@]}"; do
  miss=0; for A in sac rdsac_mean rdsac_cvar crossq; do [ -f "${CK}/${A}_s${SEED}.pt" ] || { log "SKIP s$SEED missing ${A}"; miss=1; break; }; done
  [ $miss -eq 1 ] && continue
  log "  [learned] seed $SEED"
  .venv-m11/bin/python -m eval.scripts.run_heavytail_ab \
    --serve-url "$SERVE" --login-pod "$LOGIN" --namespace $NS --controller-pod $CTL \
    --sac-ckpt "${CK}/sac_s${SEED}.pt" --rdsac-mean-ckpt "${CK}/rdsac_mean_s${SEED}.pt" \
    --rdsac-cvar-ckpt "${CK}/rdsac_cvar_s${SEED}.pt" --crossq-ckpt "${CK}/crossq_s${SEED}.pt" \
    --rlpd-ckpt "$RLPD_CKPT" \
    --family philly --n-jobs "${N_JOBS:-30}" --seed "$SEED" --sigmas 1.0 --rounds "${ROUNDS:-3}" --warmup 1 --interleave \
    --hybrid-workload --llm-model "$MODEL" --placement --gpu-nodes "$GPU_NODES" \
    --arrival-mode poisson --mps-oversub "${OVERSUB:-2.0}" --target-max-s 20 --mps-buckets 25,50,75,100 \
    --partition gpu --out-dir "runs/${TAG}_learned_s${SEED}_${STAMP}" >>"$LOG" 2>&1 || log "  [learned] s$SEED exit $?"
done

# ---- config2 & 3: Slurm-native (score arm only, no Lua) ----
run_slurm(){ # $1 name  $2 conf  $3 want
  apply_verify "$2" "$3" "$1"; prewarm
  for SEED in "${SEEDS[@]}"; do
    log "  [$1] seed $SEED"
    .venv-m11/bin/python -m eval.scripts.run_heavytail_ab \
      --serve-url "$SERVE" --login-pod "$LOGIN" --namespace $NS --controller-pod $CTL \
      --family philly --n-jobs "${N_JOBS:-30}" --seed "$SEED" --sigmas 1.0 --rounds "${ROUNDS:-3}" --warmup 1 --interleave \
      --hybrid-workload --llm-model "$MODEL" --placement --gpu-nodes "$GPU_NODES" \
      --arrival-mode poisson --mps-oversub "${OVERSUB:-2.0}" --target-max-s 20 --mps-buckets 25,50,75,100 \
      --partition gpu --out-dir "runs/${TAG}_${1}_s${SEED}_${STAMP}" >>"$LOG" 2>&1 || log "  [$1] s$SEED exit $?"
  done; }
run_slurm fcfs     "$CONF_fcfs"     "sched/builtin"
run_slurm backfill "$CONF_backfill" "sched/backfill"

log "=== aggregating ==="
.venv-m11/bin/python - "$STAMP" "$TAG" "${SEEDS[@]}" >>"$LOG" 2>&1 <<'PY'
import json,sys
from pathlib import Path
import numpy as np
from scipy import stats
stamp,tag=sys.argv[1],sys.argv[2]; SEEDS=[int(x) for x in sys.argv[3:]]
def jctpanel(p):
    r=json.loads(Path(p).read_text()); v=[x['panels']['score']['mean'] for x in r if 'score' in x['panels'] and x['panels']['score'].get('completed')]
    return float(np.mean(v)) if v else None
# learned run: per-seed score JCT + per-arm paired djct%
score={}; learned={a:{} for a in ["SAC","RDSAC-mean","RDSAC-cvar","CrossQ","RLPD"]}
for s in SEEDS:
    f=Path(f"runs/{tag}_learned_s{s}_{stamp}/reports.json")
    if not f.exists(): continue
    r=json.loads(f.read_text())
    sv=[x['panels']['score']['mean'] for x in r if x['panels'].get('score',{}).get('completed')]
    if sv: score[s]=float(np.mean(sv))
    for a in learned:
        dv=[x['paired_vs_score'][a]['djct_pct'] for x in r if a in x.get('paired_vs_score',{})]
        if dv: learned[a][s]=float(np.mean(dv))
# slurm-native runs: score-panel JCT per seed → djct vs learned-score
slurm={"fcfs":{}, "backfill":{}}
for c in slurm:
    for s in SEEDS:
        f=Path(f"runs/{tag}_{c}_s{s}_{stamp}/reports.json")
        if f.exists():
            j=jctpanel(f)
            if j: slurm[c][s]=j
def stat(d):
    a=np.array([d[s] for s in sorted(d)],float)
    if a.size<2: return "—","—"
    pos=int((a>0).sum()); p=stats.ttest_1samp(a,0.0).pvalue
    return f"{a.mean():+.1f}±{np.std(a,ddof=1):.1f}", f"{pos}/{a.size}",
def row(name,dd):
    a=np.array([dd[s] for s in sorted(dd)],float)
    if a.size<2: return f"| {name} | (n<2) | | |"
    pos=int((a>0).sum()); p=stats.ttest_1samp(a,0.0).pvalue
    return f"| {name} | {a.mean():+.1f}±{np.std(a,ddof=1):.1f} | {pos}/{a.size} | {p:.3f} |"
sj=np.array([score[s] for s in sorted(score)],float)
out=[f"# Unified full-method A/B ({tag}) — {stamp}","",
     f"seeds={sorted(score)} (n={len(score)}). All arms measured TODAY under DRA MPS, Qwen hybrid, "
     "oversub 2.0, 8 workload seeds. score=backfill+multifactor+score-Lua (baseline). SAC/RDSAC/"
     "CrossQ/RLPD=learned placement. fcfs=builtin+basic(no Lua), backfill=backfill+basic(no Lua)="
     "naive Slurm (cons_tres places). ΔJCT% vs score (+=faster). seed_t=one-sample t.","",
     f"score baseline JCT = {sj.mean():.1f}±{np.std(sj,ddof=1):.1f}s","",
     "| arm | ΔJCT% vs score | seed为正 | seed_t p |","|---|--:|:--:|--:|"]
for a in ["SAC","RDSAC-mean","RDSAC-cvar","CrossQ","RLPD"]: out.append(row(a,learned[a]))
for c in ["fcfs","backfill"]:
    dd={s:100.0*(score[s]-slurm[c][s])/score[s] for s in SEEDS if s in score and s in slurm[c] and score[s]>0}
    out.append(row(c,dd))
Path(f"runs/{tag}_{stamp}_TABLES.md").write_text("\n".join(out))
print("\n".join(out)); print(f"[agg] runs/{tag}_{stamp}_TABLES.md")
PY
log "${TAG}_DONE ${STAMP}"
