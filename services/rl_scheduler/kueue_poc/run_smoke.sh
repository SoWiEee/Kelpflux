#!/usr/bin/env bash
# Kueue PoC smoke: prove the RDSAC priority controller changes the ORDER in which
# Kueue admits a queue of Jobs. Two rounds on the SAME job set:
#   baseline : no controller -> Kueue admits by (priority=0, timestamp) = submission order (FIFO)
#   rdsac    : controller on  -> Kueue admits by policy score (sjf stub: shortest first)
#
# To get a CLEAN total order we gate on quota: set the ClusterQueue cpu quota to 0
# so nothing admits, submit all Jobs (they queue as pending Workloads), let the
# controller assign priorities to ALL of them, THEN reopen quota to 1 so Kueue
# admits one-at-a-time in priority order (removing the t=0 race where the first
# Job would admit before the controller can act). Admission order is read off the
# pods' creationTimestamp — Kueue creates the pod at admission.
set -uo pipefail
cd "$(dirname "$0")"
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
NS=kueue-poc; CQ=poc-cq; CTRL=controller.py
PY="${PY:-python3}"
RUNTIMES=(50 10 40 20 30 15)   # declared runtimes (s) in submission order
SLEEP_ACTUAL=3                 # real container runtime — short, keeps smoke fast
QPATH=/spec/resourceGroups/0/flavors/0/resources/0/nominalQuota

set_quota() { kubectl patch clusterqueue "$CQ" --type=json \
  -p "[{\"op\":\"replace\",\"path\":\"${QPATH}\",\"value\":\"$1\"}]" >/dev/null; }

cleanup() { kubectl delete jobs,workloads --all -n "$NS" --ignore-not-found >/dev/null 2>&1; sleep 3; }

submit_jobs() {
  local mode=$1 i=0 rt
  for rt in "${RUNTIMES[@]}"; do
    i=$((i+1))
    cat <<EOF | kubectl apply -f - >/dev/null
apiVersion: batch/v1
kind: Job
metadata:
  name: ${mode}-r${rt}-${i}
  namespace: ${NS}
  labels: { kueue.x-k8s.io/queue-name: poc-lq }
  annotations: { poc.kelpflux/runtime-s: "${rt}" }
spec:
  backoffLimit: 0
  template:
    metadata: { labels: { poc-mode: "${mode}" } }
    spec:
      restartPolicy: Never
      containers:
        - name: c
          image: busybox:1.36
          command: ["sh","-c","sleep ${SLEEP_ACTUAL}"]
          resources: { requests: { cpu: "1", memory: "64Mi" }, limits: { cpu: "1", memory: "128Mi" } }
EOF
    sleep 0.3
  done
}

wait_done() {
  local mode=$1 t done
  for t in $(seq 1 120); do
    done=$(kubectl get jobs -n "$NS" -o json 2>/dev/null | \
      $PY -c "import json,sys; d=json.load(sys.stdin); print(sum(1 for j in d['items'] if j['metadata']['name'].startswith('${mode}-') and j.get('status',{}).get('succeeded',0)>=1))")
    [ "${done:-0}" -ge "${#RUNTIMES[@]}" ] && return 0
    sleep 2
  done
  return 1
}

admission_order() {  # pods in creation order (=admission order), runtime pulled from pod name
  kubectl get pods -n "$NS" -l "poc-mode=$1" --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null \
    | sed -E 's/.*-r([0-9]+)-[0-9]+-[a-z0-9]+$/\1/' | tr '\n' ' '
}

round() {  # $1=mode  $2=run-controller(0/1)
  local mode=$1 use_ctrl=$2
  echo "── round: ${mode} (controller=${use_ctrl}) ──"
  set_quota 0            # close the gate: nothing admits yet
  submit_jobs "$mode"
  sleep 2                # let Kueue create pending Workloads
  if [ "$use_ctrl" = "1" ]; then
    $PY "$CTRL" --namespace "$NS" --scorer sjf --once >/tmp/kueue_poc_ctrl.log 2>&1
    echo "  controller 設定的優先權："; sed -n 's/^\[ctrl\] /    /p' /tmp/kueue_poc_ctrl.log
  fi
  set_quota 1            # open the gate: admit one-at-a-time in (priority,timestamp) order
  wait_done "$mode" || echo "  WARN: not all jobs completed in time"
  echo "  提交順序 (FIFO 基準)：${RUNTIMES[*]}"
  echo "  實際准入順序 (runtime s)：$(admission_order "$mode")"
}

echo "=== Kueue PoC smoke @ $(date +%H:%M:%S) ==="
cleanup; round baseline 0
cleanup; round rdsac 1
echo
echo "=== 判讀 ==="
echo "baseline 應 ≈ 提交順序 (50 10 40 20 30 15)；rdsac 應為最短優先 (10 15 20 30 40 50)"
cleanup; set_quota 1
