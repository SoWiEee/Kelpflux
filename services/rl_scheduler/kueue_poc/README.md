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

## Increment 1 (DONE): real RDSAC ordering + multi-method eval

`rdsac_order.py` turns the RDSAC placement policy into a queue ordering by rolling
out `/act` over the pending batch (pick next job → debit that GPU's free MPS →
repeat; no-op/abstain → FIFO). `controller.py --scorer serve` uses it live;
`eval_ab.py` uses it offline to compare **fifo / sjf / rdsac** over the same
per-seed job stream on a real Kueue queue, seed-level aggregated (Δ mean-JCT vs
FIFO), matching the paper's methodology.

```bash
# serve must be running on :8003 (RDSAC checkpoint loaded)
PYTHONPATH=. .venv-m11/bin/python services/rl_scheduler/kueue_poc/eval_ab.py \
    --seeds 1 2 3 4 5 6 --n-jobs 8 --concurrency 2 --scale 0.25 --out runs/kueue_eval.json
```

## Finding: DRA claims are exclusive → no native GPU lane beside the Slurm workers

The Slurm platform holds **each GPU in a long-lived worker pod via an exclusive
DRA ResourceClaim** (`allocated,reserved`). A second claim on the same GPU is
`Unschedulable` — *"0/2 nodes: cannot allocate all claims"*. MPS sharing in this
deployment happens **inside** the worker pod (among Slurm jobs via `gres/mps`),
**not across separate K8s pods**. So a native-K8s GPU-job lane cannot co-reside
with the Slurm workers on the same GPUs. Consequences:

- The eval executes each job as a **runtime-faithful proxy** (a pod that sleeps
  `runtime * --scale`) carrying the real GPU/MPS features (mps_req, gpu_type,
  runtime) the policy reasons over. This measures **ordering quality** (effect on
  wait time → JCT) on real pods + real Kueue admission + real wall-clock — honest
  that the *compute* is a proxy, not GPU co-location.
- **Real GPU co-location** needs a worker **evicted** for an isolated eval window
  (the gpu-toggle pattern), freeing that GPU's claim; then eval pods run real
  cuBLAS via their own claim. Even then, whether two eval pods can *share* one
  freed GPU via MPS across two claims must be verified empirically (the driver
  may still allocate the device exclusively). Parked as a scoped follow-up.

## Remaining increments

- **Real GPU execution** — evict a worker, run cuBLAS `gpu_workload` pods (§5.3
  backend); measure real JCT under MPS co-location.
- **Seed-level paired A/B vs real Kueue FIFO** — already the shape of `eval_ab.py`;
  scale seeds + add the drift-robust interleave to upgrade §5.4 from a *sim
  approximation* of Kueue to the *real* Kueue admission controller (removes a
  §6.1 threat).
- **Watch for a positive** — Kueue's native ordering (FIFO/priority) is weaker
  than Slurm's score+backfill, so "learned (or sjf) beats Kueue FIFO" is a
  plausible new positive in the ecosystem-relevant setting (analogue of "score
  beats naive Slurm cons_tres"). A tie is still honest-consistent.

> Note: the K8s-native lane runs jobs as native pods, a different execution path
> than the Slurm lane, so absolute JCT is **not** comparable across lanes (same
> backend-confound discipline as §6.1). Compare only *within* the K8s lane.
