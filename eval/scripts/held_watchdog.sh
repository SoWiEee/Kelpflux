#!/usr/bin/env bash
# 釋放卡在 launch-failed-requeued-held 的 job，讓 eval harness 不因 node 打嗝空等。
export KUBECONFIG=$HOME/.kube/config
LOG="${1:-/tmp/held_watchdog.log}"
echo "[wd] start $(date +%T)" > "$LOG"
while true; do
  out=$(kubectl exec -n slurm slurm-controller-0 -- bash -lc '
    ids=$(squeue -h -t PD -o "%i|%r" 2>/dev/null | awk -F"|" "tolower(\$2) ~ /launch|held|requeue/ {print \$1}")
    if [ -n "$ids" ]; then
      for j in $ids; do scontrol release "$j" 2>/dev/null; scontrol update jobid="$j" state=pending 2>/dev/null; done
      echo "released: $ids"
    fi' 2>/dev/null)
  [ -n "$out" ] && echo "[wd $(date +%T)] $out" >> "$LOG"
  sleep 40
done
