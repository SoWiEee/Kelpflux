"""A/B runner for the heavy-tail high-contention live experiment (§4.4 / spec).

Drives the full evaluation: for each σ ∈ {0 (control), 1.0 (main)} and each arm
∈ {score, SAC, RDSAC-mean, RDSAC-cvar} it switches the served checkpoint via
serve.py /reload + /shadow (no pod restart), submits the SAME heavy-tail stream
(per-job CRN → identical across arms), waits for drain, collects per-job JCT from
sacct, discards a warmup round, and reports the tail panel + paired deltas vs score.

The evaluation contract (eval-writeup §4.4):
  * σ=0 control → all arms should tie (proves any σ=1 difference is from uncertainty).
  * σ=1.0 → RDSAC-cvar should cut p99 / CVaR vs SAC if the sim σ-finding transfers.

The report builder (`build_report`) is pure and unit-tested. The orchestration
(`run`) is cluster-only: it talks to serve over HTTP and shells out via the
live_ab_heavytail wrappers. Use --dry-run to print the plan without a cluster.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib import request as _urlrequest

from eval.scripts.live_ab_heavytail import (
    WorkloadSpec,
    collect_sacct,
    decide_node,
    gen_workload,
    join_records,
    parse_free_mps,
    submit_stream,
    wait_drain,
)
from eval.scripts.tail_metrics import paired_delta, summarize

ARMS = ("score", "SAC", "RDSAC-mean", "RDSAC-cvar", "CrossQ")  # score first = paired baseline


# ── pure report builder (unit-tested) ─────────────────────────────────────────

def build_report(records_by_arm: dict, *, sigma: float, family: str, beta: float = 0.25) -> dict:
    """{arm: [join_records...]} → per-arm tail panel + paired deltas vs score.

    Pairs arms on the common (round, job_id) set so ΔJCT%/Δp99%/ΔCVaR% compare the
    SAME jobs. score is the baseline; learned arms are diffed against it.
    """
    panels: dict = {}
    jct_by_arm: dict = {}
    for arm, recs in records_by_arm.items():
        keyed = {(r["round"], r["job_id"]): r for r in recs}
        jct_by_arm[arm] = keyed
        jcts = [r["jct"] for r in recs]
        trues = [r["true_runtime_s"] for r in recs]
        panels[arm] = summarize(jcts, true_runtimes=trues, beta=beta)
        panels[arm]["completed"] = len(recs)

    report: dict = {"sigma": sigma, "family": family, "beta": beta,
                    "panels": panels, "paired_vs_score": {}}
    if "score" not in jct_by_arm:
        return report
    score_keyed = jct_by_arm["score"]
    for arm, keyed in jct_by_arm.items():
        if arm == "score":
            continue
        common = sorted(set(score_keyed) & set(keyed))
        if not common:
            continue
        s = [score_keyed[k]["jct"] for k in common]
        m = [keyed[k]["jct"] for k in common]
        report["paired_vs_score"][arm] = paired_delta(s, m, beta=beta)
    return report


def render_summary(reports: list[dict]) -> str:
    lines = ["# Heavy-tail live A/B — SUMMARY", ""]
    for rep in reports:
        lines.append(f"## σ={rep['sigma']}  family={rep['family']}")
        lines.append("")
        lines.append("| arm | n | mean | p95 | p99 | CVaR | slowdown_p99 |")
        lines.append("|---|--:|--:|--:|--:|--:|--:|")
        for arm, pan in rep["panels"].items():
            lines.append(
                f"| {arm} | {pan.get('completed', pan['n'])} | {pan['mean']:.1f} | "
                f"{pan['p95']:.1f} | {pan['p99']:.1f} | {pan['cvar']:.1f} | "
                f"{pan.get('slowdown_p99', float('nan')):.2f} |")
        lines.append("")
        if rep["paired_vs_score"]:
            lines.append("| arm vs score | ΔJCT% | Δp99% | ΔCVaR% | t-test p |")
            lines.append("|---|--:|--:|--:|--:|")
            for arm, d in rep["paired_vs_score"].items():
                lines.append(
                    f"| {arm} | {d['djct_pct']:+.1f} | {d['dp99_pct']:+.1f} | "
                    f"{d['dcvar_pct']:+.1f} | {d.get('ttest_p', float('nan')):.3g} |")
            lines.append("")
    return "\n".join(lines)


# ── live orchestration (cluster-only) ─────────────────────────────────────────

def _post(serve_url: str, path: str, payload: dict, *, dry_run: bool) -> dict:
    if dry_run:
        print(f"  POST {path} {payload}")
        return {"ok": True, "dry_run": True}
    data = json.dumps(payload).encode()
    req = _urlrequest.Request(serve_url.rstrip("/") + path, data=data,
                              headers={"Content-Type": "application/json"}, method="POST")
    with _urlrequest.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _configure_arm(serve_url: str, arm: str, ckpts: dict, *, dry_run: bool,
                   placement: bool = False) -> None:
    """score → boost OFF (shadow). learned → load its checkpoint.

    In submit-time placement mode the RL treatment is the node choice (``-w``),
    not the /decide priority boost, so learned arms also run shadow=ON (boost
    off) — the checkpoint is still loaded so /act answers the node decision."""
    if arm == "score":
        _post(serve_url, "/shadow", {"shadow": True}, dry_run=dry_run)
        return
    ckpt = ckpts.get(arm)
    if not ckpt:
        raise ValueError(f"no checkpoint configured for arm {arm!r}")
    _post(serve_url, "/reload", {"checkpoint": ckpt}, dry_run=dry_run)
    _post(serve_url, "/shadow", {"shadow": placement}, dry_run=dry_run)


def _query_free_mps(exec_prefix, node_names: list[str]) -> list[int]:
    """Run `scontrol show node <names> -o` inside the cluster, parse free MPS per
    node (configured − allocated). Best-effort: missing node → full capacity."""
    import subprocess
    out_by_node = {}
    cmd = (exec_prefix or []) + ["scontrol", "show", "node", ",".join(node_names), "-o"]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            m = re.search(r"NodeName=(\S+)", line)
            if m:
                out_by_node[m.group(1)] = parse_free_mps(line)
    return [out_by_node.get(n, 100) for n in node_names]


def _make_place_fn(arm: str, *, serve_url: str, node_names: list[str], exec_prefix,
                   placement: bool, dry_run: bool):
    """RL arms in placement mode → a closure that picks a node per job via /act.
    score arm, non-placement mode, or dry_run → None (Slurm places)."""
    if not placement or arm == "score" or dry_run:
        return None

    def _place(job):
        free = _query_free_mps(exec_prefix, node_names)
        return decide_node(serve_url, job, node_free_mps=free, node_names=node_names)

    return _place


def run(args) -> int:
    ckpts = {"SAC": args.sac_ckpt, "RDSAC-mean": args.rdsac_mean_ckpt,
             "RDSAC-cvar": args.rdsac_cvar_ckpt, "CrossQ": args.crossq_ckpt}
    arms = [a for a in ARMS if a == "score" or ckpts.get(a)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sigmas = [float(s) for s in args.sigmas]
    mps_buckets = [int(b) for b in args.mps_buckets.split(",") if b.strip()] or None
    all_records: list[dict] = []
    reports: list[dict] = []

    import shlex as _shlex
    exec_prefix = ([*_shlex.split(args.kubectl), "exec", "-n", args.namespace,
                    args.login_pod, "--"] if args.login_pod else None)
    # Real-CUDA workload (replaces sleep): same spec across all arms → fair.
    workload = (WorkloadSpec(bin_path=args.workload_bin, dim=args.workload_dim,
                             vram_mb=args.workload_vram_mb,
                             iters_per_sec=args.iters_per_sec,
                             time_factor=args.workload_time_factor)
                if args.cuda_workload else None)
    if workload is not None:
        print(f"[ab] CUDA workload: {workload}", flush=True)
    gpu_nodes = [n for n in (args.gpu_nodes or "").split(",") if n]
    if args.placement and not gpu_nodes:
        print("error: --placement requires --gpu-nodes node0,node1 (index ↔ RL node_j)",
              file=sys.stderr)
        return 2

    for si, sigma in enumerate(sigmas):
        # Same stream for every method at this σ (CRN: identical generator inputs).
        jobs = gen_workload(args.family, args.n_jobs, seed=args.seed, sigma=sigma,
                            target_max_s=args.target_max_s, mps_oversub=args.mps_oversub,
                            arrival_mode=args.arrival_mode, mps_buckets=mps_buckets)
        records_by_arm: dict = {a: [] for a in arms}

        def _do(arm: str, rnd: int, is_warm: bool) -> None:
            tag = "warmup" if is_warm else f"r{rnd - args.warmup + 1}"
            print(f"[ab] σ={sigma} {arm} {tag}: configure + submit {len(jobs)} jobs", flush=True)
            _configure_arm(args.serve_url, arm, ckpts, dry_run=args.dry_run,
                           placement=args.placement)
            # sacct -S is read in the cluster timezone (UTC); use UTC with a back-margin
            # so host/cluster clock skew can't push it into the future. round is in the
            # job_name, so an over-wide window is harmless (join filters by exact name).
            since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - 300))
            place_fn = _make_place_fn(arm, serve_url=args.serve_url,
                                      node_names=gpu_nodes, exec_prefix=exec_prefix,
                                      placement=args.placement, dry_run=args.dry_run)
            submit_stream(jobs, arm, rnd, dry_run=args.dry_run,
                          partition=args.partition, exec_prefix=exec_prefix,
                          place_fn=place_fn, placement=args.placement,
                          workload=workload, exclusive_gpu=args.exclusive_gpu)
            if args.dry_run:
                return
            wait_drain(kubectl=args.kubectl, namespace=args.namespace,
                       controller_pod=args.controller_pod, timeout_s=args.drain_timeout_s)
            parsed = collect_sacct(since, kubectl=args.kubectl, namespace=args.namespace,
                                   controller_pod=args.controller_pod)
            rr = join_records(jobs, parsed, arm, rnd)
            if not is_warm:               # discard warmup round(s)
                records_by_arm[arm].extend(rr)
                all_records.extend(rr)

        total = args.warmup + args.rounds
        if args.interleave:
            # round-robin: each round runs every method once, rotating who goes first
            # so drift over the run averages out equally across methods (§4.4.1).
            for rnd in range(total):
                is_warm = rnd < args.warmup
                k = rnd % len(arms)
                order = arms[k:] + arms[:k]
                for arm in order:
                    _do(arm, rnd, is_warm)
        else:
            # block design: all rounds of one method, then the next (drift-prone).
            arm_order = arms if si % 2 == 0 else list(reversed(arms))
            for arm in arm_order:
                for rnd in range(total):
                    _do(arm, rnd, rnd < args.warmup)

        if not args.dry_run:
            reports.append(build_report(records_by_arm, sigma=sigma, family=args.family,
                                        beta=args.beta))

    if not args.dry_run:
        (out_dir / "records.json").write_text(json.dumps(all_records, indent=2))
        (out_dir / "reports.json").write_text(json.dumps(reports, indent=2))
        (out_dir / "SUMMARY.md").write_text(render_summary(reports))
        print(f"[ab] wrote {out_dir}/SUMMARY.md")
    else:
        print("[ab] dry-run complete (no cluster calls executed)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Heavy-tail live A/B runner")
    p.add_argument("--serve-url", default="http://localhost:8002")
    p.add_argument("--sac-ckpt", default=None)
    p.add_argument("--rdsac-mean-ckpt", default=None)
    p.add_argument("--rdsac-cvar-ckpt", default=None)
    p.add_argument("--crossq-ckpt", default=None,
                   help="CrossQ checkpoint (adds a CrossQ live arm)")
    p.add_argument("--family", choices=["philly", "ali"], default="philly")
    p.add_argument("--n-jobs", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sigmas", nargs="*", default=["0.0", "1.0"])
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--target-max-s", type=float, default=180.0)
    p.add_argument("--mps-oversub", type=float, default=4.0)
    p.add_argument("--mps-buckets", default="",
                   help="comma-separated MPS-slot buckets, e.g. '25,50,75,100' "
                        "(1/2/3/4 of the 4-slot card; 100=whole GPU). Assigns each "
                        "job a clean fraction by size-rank instead of the scaled "
                        "demand-peak value. Empty = legacy oversub scaling.")
    p.add_argument("--arrival-mode", choices=["burst", "poisson"], default="burst")
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--partition", default="gpu-rtx4070")
    # Real-CUDA workload (replaces sleep): a normal user-style CUDA sbatch job so
    # MPS interference / VRAM / heterogeneous-card speed surface in JCT.
    p.add_argument("--cuda-workload", action="store_true",
                   help="run real CUDA jobs (gpu_workload sgemm) instead of sleep N")
    p.add_argument("--exclusive-gpu", action="store_true",
                   help="each job takes the whole GPU (--gres=mps:100, no co-residency); "
                        "sidesteps MPS multiplexing, keeps heterogeneity + queueing")
    p.add_argument("--workload-bin", default="/shared/bin/gpu_workload",
                   help="path to the compiled gpu_workload binary (on shared NFS)")
    p.add_argument("--workload-dim", type=int, default=4096,
                   help="sgemm matrix dim (compute intensity + base VRAM)")
    p.add_argument("--workload-vram-mb", type=int, default=512,
                   help="extra VRAM scratch per job (independent VRAM pressure)")
    p.add_argument("--iters-per-sec", type=float, default=145.0,
                   help="idle reference-card (4070) iters/s at --workload-dim; "
                        "iters = true_runtime_s × this (CRN by job_id)")
    p.add_argument("--workload-time-factor", type=float, default=20.0,
                   help="--time = true_runtime_s × this (slow-card × contention "
                        "margin so jobs aren't killed; uniform across arms → fair)")
    p.add_argument("--placement", action="store_true",
                   help="submit-time RL node placement: RL arms pick a node via "
                        "/act and submit with -w <node> (Slurm 21.08 can't re-pin "
                        "post-submit). Requires --gpu-nodes; learned arms run boost-off.")
    p.add_argument("--gpu-nodes", default="",
                   help="comma-separated GPU node names in RL node_j index order, "
                        "e.g. slurm-worker-gpu-rtx4070-0,slurm-worker-gpu-rtx3080-0")
    p.add_argument("--kubectl", default="kubectl")
    p.add_argument("--namespace", default="slurm")
    p.add_argument("--controller-pod", default="slurm-controller-0")
    p.add_argument("--login-pod", default=None,
                   help="pod to run sbatch in via kubectl exec (e.g. pod/slurm-login-xxx); "
                        "omit to run sbatch locally")
    p.add_argument("--drain-timeout-s", type=float, default=3600.0)
    p.add_argument("--out-dir", default=f"runs/htab_{time.strftime('%Y%m%d-%H%M%S')}")
    p.add_argument("--interleave", action="store_true",
                   help="round-robin method order each round (averages out cluster drift, §4.4.1); "
                        "default is block design")
    p.add_argument("--dry-run", action="store_true", help="print plan, no cluster calls")
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
