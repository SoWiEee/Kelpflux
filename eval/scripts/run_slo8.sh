#!/usr/bin/env bash
# §4.2.3 SLO-focused serving A/B under DRA. Serving-realistic workload: a high-QPS
# stream of MANY SHORT cuBLAS requests (short target-max-s), each with a per-request
# latency SLO (deadline = true_runtime × 4 → slowdown > 4 counts as a violation). The
# many short requests give the tail/SLO estimate statistical power that the earlier
# ~30-job batch runs lacked. All eight arms (score/SAC/RDSAC-mean/RDSAC-cvar/CrossQ/
# RLPD + Slurm-native fcfs/backfill) on the SAME DRA backend, 8 seeds. Reports SLO
# violation rate + p99 + slowdown_p99. Three Slurm configs, restored on exit.
set -uo pipefail
cd /home/acane/Desktop/Kelpflux
export KUBECONFIG=$HOME/.kube/config PYTHONPATH=.
STAMP=$(date +%Y%m%d-%H%M%S)
TAG="${TAG:-slo8}"
LOG="runs/${TAG}_${STAMP}.log"
CM=slurm-config-static; NS=slurm; CTL=slurm-controller-0
LOGIN="${LOGIN_POD:-slurm-login-7f8cfbc48-c875f}"
GPU_NODES="slurm-worker-gpu-rtx4070-0,slurm-worker-gpu-rtx3080-0"
SERVE="http://localhost:8003"
CK=/tmp/lckpts; RLPD_CKPT="${RLPD_CKPT:-${CK}/rlpd_v3.pt}"
NJOBS="${N_JOBS:-120}"; RND="${ROUNDS:-2}"; TMAXS="${TARGET_MAX_S:-8}"; OVSUB="${OVERSUB:-2.0}"
read -r -a SEEDS <<< "${SEEDS:-42 43 44 45 46 47 48 49}"
ORIG=/tmp/slurm.conf.slo8orig.$$
mkdir -p runs
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
patch_conf(){ kubectl patch cm -n $NS $CM --type merge -p "$(python3 -c "import json;print(json.dumps({'data':{'slurm.conf':open('$1').read()}}))")" >/dev/null; }
restart_ctl_wait(){ kubectl delete pod -n $NS $CTL >/dev/null 2>&1; for i in $(seq 1 30); do [ "$(kubectl get pod -n $NS $CTL -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null)" = "true" ] && return 0; sleep 6; done; return 1; }
resume_nodes(){ kubectl exec -n $NS $CTL -- bash -lc 'scontrol reconfigure 2>/dev/null; for N in slurm-worker-gpu-rtx4070-0 slurm-worker-gpu-rtx3080-0; do for i in 1 2 3; do scontrol update nodename=$N state=resume 2>/dev/null; sleep 3; done; done' >/dev/null 2>&1 || true; }
restore_orig(){ log "RESTORING original slurm.conf"; patch_conf "$ORIG"; restart_ctl_wait && log "cluster restored" || log "WARN restore not ready"; resume_nodes; }
trap 'restore_orig' EXIT
kubectl get cm -n $NS $CM -o jsonpath='{.data.slurm\.conf}' > "$ORIG"; [ -s "$ORIG" ] || { echo FATAL cannot read slurm.conf; exit 1; }
CONF_fcfs=/tmp/slurm.conf.slo8fcfs.$$; CONF_backfill=/tmp/slurm.conf.slo8bf.$$
sed -e 's|^SchedulerType=.*|SchedulerType=sched/builtin|' -e 's|^PriorityType=.*|PriorityType=priority/basic|' -e '/^SchedulerParameters=/d' -e '/^JobSubmitPlugins=/d' "$ORIG" > "$CONF_fcfs"
sed -e 's|^PriorityType=.*|PriorityType=priority/basic|' -e '/^JobSubmitPlugins=/d' "$ORIG" > "$CONF_backfill"
log "### $TAG SLO serving A/B (njobs=$NJOBS rounds=$RND tmax=${TMAXS}s oversub=$OVSUB) $(date)"
curl -fsS "$SERVE/healthz" >/dev/null 2>&1 || { log "FATAL serve unreachable"; exit 1; }

apply_verify(){ patch_conf "$1"; restart_ctl_wait || { log "FATAL slurmctld not ready ($3)"; exit 1; }
  local got; got=$(kubectl exec -n $NS $CTL -- scontrol show config 2>/dev/null | grep -oP 'SchedulerType\s*=\s*\K\S+')
  [ "$got" = "$2" ] || { log "FATAL SchedulerType=$got != $2 ($3)"; exit 1; }
  resume_nodes; log "config $3 active (SchedulerType=$got)"; }
ab(){ .venv-m11/bin/python -m eval.scripts.run_heavytail_ab \
    --serve-url "$SERVE" --login-pod "$LOGIN" --namespace $NS --controller-pod $CTL \
    "$@" --family philly --n-jobs "$NJOBS" --sigmas 1.0 --rounds "$RND" --warmup 1 --interleave \
    --placement --gpu-nodes "$GPU_NODES" --arrival-mode poisson --mps-oversub "$OVSUB" \
    --target-max-s "$TMAXS" --mps-buckets 25,50,75,100 --partition gpu ; }

apply_verify "$ORIG" "sched/backfill" "learned"
for SEED in "${SEEDS[@]}"; do
  miss=0; for A in sac rdsac_mean rdsac_cvar crossq; do [ -f "${CK}/${A}_s${SEED}.pt" ] || { log "SKIP s$SEED"; miss=1; break; }; done
  [ $miss -eq 1 ] && continue
  log "  [learned] seed $SEED"
  ab --seed "$SEED" --sac-ckpt "${CK}/sac_s${SEED}.pt" --rdsac-mean-ckpt "${CK}/rdsac_mean_s${SEED}.pt" \
     --rdsac-cvar-ckpt "${CK}/rdsac_cvar_s${SEED}.pt" --crossq-ckpt "${CK}/crossq_s${SEED}.pt" --rlpd-ckpt "$RLPD_CKPT" \
     --out-dir "runs/${TAG}_learned_s${SEED}_${STAMP}" >>"$LOG" 2>&1 || log "  [learned] s$SEED exit $?"
done
run_slurm(){ apply_verify "$2" "$3" "$1"
  for SEED in "${SEEDS[@]}"; do log "  [$1] seed $SEED"
    ab --seed "$SEED" --out-dir "runs/${TAG}_${1}_s${SEED}_${STAMP}" >>"$LOG" 2>&1 || log "  [$1] s$SEED exit $?"
  done; }
run_slurm fcfs     "$CONF_fcfs"     "sched/builtin"
run_slurm backfill "$CONF_backfill" "sched/backfill"

log "=== aggregating (SLO violation + tail) ==="
.venv-m11/bin/python - "$STAMP" "$TAG" "${SEEDS[@]}" >>"$LOG" 2>&1 <<'PY'
import json,sys
from pathlib import Path
import numpy as np
from scipy import stats
stamp,tag=sys.argv[1],sys.argv[2]; SEEDS=[int(x) for x in sys.argv[3:]]
ARMS=["score","SAC","RDSAC-mean","RDSAC-cvar","CrossQ","RLPD"]
def pull(run,arm,key):  # per-seed mean of a panel metric across rounds
    d={}
    for s in SEEDS:
        f=Path(f"runs/{tag}_{run}_s{s}_{stamp}/reports.json")
        if not f.exists(): continue
        reps=json.loads(f.read_text())
        v=[r['panels'][arm].get(key) for r in reps if arm in r['panels'] and r['panels'][arm].get('completed') and r['panels'][arm].get(key) is not None]
        if v: d[s]=float(np.mean(v))
    return d
# learned run: all learned/score arms; slurm runs: score panel = that config
viol={a:pull("learned",a,"slo_viol") for a in ARMS}
p99={a:pull("learned",a,"p99") for a in ARMS}
slp99={a:pull("learned",a,"slowdown_p99") for a in ARMS}
mean={a:pull("learned",a,"mean") for a in ARMS}
for c in ["fcfs","backfill"]:
    viol[c]=pull(c,"score","slo_viol"); p99[c]=pull(c,"score","p99"); slp99[c]=pull(c,"score","slowdown_p99"); mean[c]=pull(c,"score","mean")
sv=viol["score"]
def ms(d,scale=1.0,pct=False):
    a=np.array([d[s]*scale for s in sorted(d)],float)
    if not a.size: return "—"
    suf="%%" if pct else ""
    return f"{a.mean():.1f}±{np.std(a,ddof=1) if a.size>1 else 0:.1f}"
def dviol(d):  # per-seed Δ SLO violation pp vs score (score_viol - arm_viol), + = fewer violations = better
    x=[100*(sv[s]-d[s]) for s in SEEDS if s in sv and s in d]
    a=np.array(x,float)
    if a.size<2: return "—","—"
    p=stats.ttest_1samp(a,0).pvalue
    return f"{a.mean():+.1f}", f"{p:.3f}"
order=["score","RDSAC-cvar","backfill","CrossQ","SAC","fcfs","RDSAC-mean","RLPD"]
out=[f"# SLO serving A/B ({tag}) — {stamp}","",
     f"seeds={sorted(sv)} (n={len(sv)}). DRA MPS, high-QPS 短 cuBLAS 請求, 8 seed. "
     "SLO 違反率 = JCT > runtime×4 的請求比例(越低越好)。ΔSLOviol = score−arm 的百分點(+=違反更少=更好)。seed-t = 該Δ的 one-sample t。","",
     "| arm | SLO違反% | p99(s) | slowdown_p99 | 平均JCT(s) | ΔSLOviol(pp) | seed-t p |",
     "|---|--:|--:|--:|--:|--:|--:|"]
for a in order:
    if a not in viol or not viol[a]: continue
    if a=="score":
        out.append(f"| score | {ms(viol[a],100,True)}% | {ms(p99[a])} | {ms(slp99[a])} | {ms(mean[a])} | —(基準) | — |")
    else:
        dv,pp=dviol(viol[a])
        out.append(f"| {a} | {ms(viol[a],100,True)}% | {ms(p99[a])} | {ms(slp99[a])} | {ms(mean[a])} | {dv} | {pp} |")
Path(f"runs/{tag}_{stamp}_TABLES.md").write_text("\n".join(out))
print("\n".join(out)); print(f"[agg] runs/{tag}_{stamp}_TABLES.md")
PY
log "${TAG}_DONE ${STAMP}"
