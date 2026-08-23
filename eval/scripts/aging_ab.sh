#!/usr/bin/env bash
# Slurm-side tail fix A/B: the learned-config aging is INERT at our timescale —
# PriorityMaxAge defaults to 7 days, so a job waiting seconds-to-minutes accrues
# ~0 age priority → the RL-placed (-w bound) jobs starve to wait_max ≈720s while
# backfill (aging-bounded) caps ≈340s (§ diagnosis). This runs the learned arms
# under a FAST-AGING config (PriorityMaxAge=5min) so age actually bites, and lets
# us compare wait_max / P99 against the default-aging intf run (same ckpts/seeds).
#
#   SEEDS="42 43 44" MAXAGE=00:05:00 bash eval/scripts/aging_ab.sh
# Compare afterwards vs runs/heavy150aimix_learned_s{42,43,44}_20260821-043201.
set -uo pipefail
cd /home/acane/Desktop/Kelpflux
export KUBECONFIG=$HOME/.kube/config PYTHONPATH=.
STAMP=$(date +%Y%m%d-%H%M%S)
TAG="${TAG:-agingfast}"
LOG="runs/${TAG}_${STAMP}.log"
CM=slurm-config-static; NS=slurm; CTL=slurm-controller-0
LOGIN="${LOGIN_POD:-slurm-login-7f8cfbc48-c875f}"
GPU_NODES="slurm-worker-gpu-rtx4070-0,slurm-worker-gpu-rtx3080-0"
SERVE="${SERVE:-http://localhost:8003}"; MODEL="${MODEL:-/shared/models/qwen05b}"
CK="${CK:-runs/ckpts_aimix16_intf}"
N_JOBS="${N_JOBS:-150}"; OVERSUB="${OVERSUB:-2.0}"; MAXAGE="${MAXAGE:-00:05:00}"
read -r -a SEEDS <<< "${SEEDS:-42 43 44}"
read -r -a LEARNED <<< "${LEARNED:-sac rdsac_mean rdsac_cvar rlpd_cvar}"
ORIG=/tmp/slurm.conf.agingorig.$$; FAST=/tmp/slurm.conf.agingfast.$$
PATCHED=0; WD_PID=""
mkdir -p runs
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
patch_conf(){ kubectl patch cm -n $NS $CM --type merge -p "$(python3 -c "import json;print(json.dumps({'data':{'slurm.conf':open('$1').read()}}))")" >/dev/null; }
restart_ctl_wait(){ kubectl delete pod -n $NS $CTL >/dev/null 2>&1; for i in $(seq 1 30); do [ "$(kubectl get pod -n $NS $CTL -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null)" = "true" ] && return 0; sleep 6; done; return 1; }
resume_nodes(){ kubectl exec -n $NS $CTL -- bash -lc 'scontrol reconfigure 2>/dev/null; for N in slurm-worker-gpu-rtx4070-0 slurm-worker-gpu-rtx3080-0; do for i in 1 2 3; do scontrol update nodename=$N state=resume 2>/dev/null; sleep 3; done; done' >/dev/null 2>&1 || true; }
restore(){ [ "$PATCHED" = 1 ] || return 0; log "RESTORING original slurm.conf"; patch_conf "$ORIG"; restart_ctl_wait && log "restored" || log "WARN restore"; resume_nodes; }
cleanup(){ [ -n "$WD_PID" ] && kill "$WD_PID" 2>/dev/null; restore; }
trap cleanup EXIT

kubectl get cm -n $NS $CM -o jsonpath='{.data.slurm\.conf}' > "$ORIG"; [ -s "$ORIG" ] || { echo FATAL no conf; exit 1; }
# fast-aging = original + an explicit short PriorityMaxAge (append; last value wins).
cp "$ORIG" "$FAST"; printf '\nPriorityMaxAge=%s\n' "$MAXAGE" >> "$FAST"

curl -fsS --max-time 5 "$SERVE/healthz" >/dev/null 2>&1 || { log "FATAL serve down"; exit 1; }
log "### aging A/B: MaxAge=$MAXAGE  arms=${LEARNED[*]}  seeds=${SEEDS[*]}  n_jobs=$N_JOBS"

ck_args(){ local seed="$1" out=""
  for A in "${LEARNED[@]}"; do case "$A" in
    sac) out="$out --sac-ckpt ${CK}/sac_s${seed}.pt";;
    rdsac_mean) out="$out --rdsac-mean-ckpt ${CK}/rdsac_mean_s${seed}.pt";;
    rdsac_cvar) out="$out --rdsac-cvar-ckpt ${CK}/rdsac_cvar_s${seed}.pt";;
    rlpd_cvar) out="$out --rlpd-ckpt ${CK}/rlpd_cvar_s${seed}.pt";;
  esac; done; echo "$out"; }

# apply fast-aging config (retry-poll like the main harness)
patch_conf "$FAST"; PATCHED=1; restart_ctl_wait || { log FATAL ctl; exit 1; }
got=""; for i in $(seq 1 20); do got=$(kubectl exec -n $NS $CTL -- scontrol show config 2>/dev/null | grep -oP 'PriorityMaxAge\s*=\s*\K\S+'); [ "$got" != "7-00:00:00" ] && [ -n "$got" ] && break; sleep 6; done
log "PriorityMaxAge now = $got"
resume_nodes
bash eval/scripts/held_watchdog.sh "/tmp/${TAG}_wd_${STAMP}.log" & WD_PID=$!

for SEED in "${SEEDS[@]}"; do
  log "  [agingfast] seed $SEED"
  # shellcheck disable=SC2046
  .venv-m11/bin/python -m eval.scripts.run_heavytail_ab \
    --serve-url "$SERVE" --login-pod "$LOGIN" --namespace $NS --controller-pod $CTL \
    $(ck_args "$SEED") --no-score \
    --family aimix --n-jobs "$N_JOBS" --seed "$SEED" --sigmas 1.0 --rounds 1 --warmup 1 --interleave \
    --aimix-workload --llm-model "$MODEL" --placement --gpu-nodes "$GPU_NODES" \
    --arrival-mode poisson --mps-oversub "$OVERSUB" --target-max-s 20 --mps-buckets 25,50,75,100 \
    --partition gpu --out-dir "runs/${TAG}_learned_s${SEED}_${STAMP}" >>"$LOG" 2>&1 || log "  s$SEED exit $?"
done
log "### done stamp=$STAMP"
