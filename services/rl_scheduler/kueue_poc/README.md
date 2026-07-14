# Kueue PoC — RDSAC as a K8s-native admission-ordering policy

A minimal, working prototype that plugs the learned scheduling policy into the
**cloud-native** scheduling layer (Kueue), as the K8s-native counterpart to the
Slurm `job_submit.lua` priority path. It turns the paper's "learned *policy* layer
complements the DRA/Kueue *mechanism* layer" argument (§2.4) from prose into a
runnable artifact, and is the concrete start of the §6.3 future-work item.

## What it demonstrates

Kueue admits Workloads from a ClusterQueue in `(priority, creationTimestamp)`
order. Admission order is **not** a pluggable policy — but the Workload
**priority value is mutable**. So a small controller can drive admission order by
watching pending Workloads and patching each one's `spec.priority`. That is
exactly the lever a learned policy needs: *which queued job should run next*.

Smoke result (`run_smoke.sh`, 6 jobs, cpu-gated ClusterQueue quota = 1 so one
admits at a time and the order is a clean total order):

| round | controller | admission order (declared runtime s) |
|---|---|---|
| baseline | off | `50 10 40 20 30 15` (= submission order, Kueue FIFO) |
| rdsac | on (sjf stub) | `10 15 20 30 40 50` (shortest-first — policy reordered) |

The controller changed the admission order end-to-end. This is the same
fail-safe shape as the Slurm path: if the controller is down, Kueue keeps its
default FIFO order and nothing blocks.

## Architecture

```
K8s Jobs (label kueue.x-k8s.io/queue-name: poc-lq)
        │  Kueue webhook suspends each Job, creates a pending Workload
        ▼
Kueue ClusterQueue (StrictFIFO by priority+timestamp, small quota → a queue forms)
        ▲  patch spec.priority
        │
controller.py  ──score──►  sjf stub   (shortest declared runtime first)   [proves loop]
               ──score──►  RDSAC /decide (serve :8003)                     [increment 1]
        │  fail-safe: any scorer/patch error → leave default order, never block
        ▼
Kueue admits highest-priority pending Workload when quota frees → pod is created
```

## Files

- `manifests/kueue-v0.18.3.yaml` — pinned Kueue install (applied server-side).
- `manifests/10-kueue-setup.yaml` — `kueue-poc` namespace, `poc-flavor`
  ResourceFlavor, `poc-cq` ClusterQueue (cpu-gated, StrictFIFO), `poc-lq`
  LocalQueue. Quota is deliberately tiny to force a queue.
- `controller.py` — poll-based ordering controller. `--scorer sjf|serve`,
  `--namespace`, `--serve-url`, `--once`. No k8s-client dependency (shells to
  kubectl), fail-safe throughout.
- `run_smoke.sh` — two-round baseline-vs-rdsac demo using a quota gate for a
  clean total order.

## Run

```bash
export KUBECONFIG=$HOME/.kube/config
# 1. install Kueue (once)
kubectl apply --server-side -f services/rl_scheduler/kueue_poc/manifests/kueue-v0.18.3.yaml
kubectl -n kueue-system rollout status deploy/kueue-controller-manager
# 2. PoC queue objects (once)
kubectl apply -f services/rl_scheduler/kueue_poc/manifests/10-kueue-setup.yaml
# 3. smoke
bash services/rl_scheduler/kueue_poc/run_smoke.sh
```

Kueue's default `manageJobsWithoutQueueName: false` means it only touches Jobs
carrying the queue-name label — the `slurm` namespace and the running platform
are untouched.

## From PoC to research-grade (increments)

This prototype proves the **control loop**. To make it a paper result:

1. **Wire `/decide`** — replace the sjf stub with the RDSAC `/decide` call
   (`--scorer serve`). Needs the Job → 166-dim observation mapping (MPS/VRAM
   request, wait time, SLO urgency, GPU-type one-hot). The fail-safe fallback is
   already in `score_serve`.
2. **Gate on GPU/MPS, not cpu** — model the contended resource as the DRA GPU /
   MPS slot (ResourceClaim + Kueue quota on the device) so the ordering decision
   is over the actual scarce resource, matching §5.3.
3. **Real workload** — swap the `sleep` container for the cuBLAS `gpu_workload`
   (and the LLM hybrid) with a DRA GPU claim, on the same DRA MPS backend as
   §5.3 (no backend confound).
4. **Seed-level paired A/B** — compare **Kueue-native FIFO vs RDSAC-ordered** on
   the same workload seeds, with the drift-robust interleave + seed-level
   one-sample t methodology from §5.1. This upgrades §5.4's SOTA comparison from
   a *sim approximation* of Kueue to the *real* Kueue admission controller,
   removing a §6.1 validity threat.
5. **Watch for a positive** — Kueue's native ordering (FIFO/priority) is weaker
   than Slurm's score+backfill, so "learned beats Kueue FIFO" is a plausible new
   positive result in the ecosystem-relevant setting (analogue of the paper's one
   robust positive, "score beats naive Slurm cons_tres"). A tie is still a fine,
   honest-consistent result — the value is the artifact + the stronger baseline.

> Note: the K8s-native lane runs jobs as native pods, a different execution path
> than the Slurm lane, so absolute JCT is **not** comparable across lanes (same
> backend-confound discipline as §6.1). Compare only *within* the K8s lane.
