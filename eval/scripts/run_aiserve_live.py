"""Live A/B of schedulers on the AI-serving workload (SLO-discriminating).

Runs the aiserve dual-class workload (latency-SLO inference + best-effort
training, real cuBLAS, MPS co-resident) on the real 2×1 cluster and measures
the SLO-violation rate / inference latency that separate schedulers (§4.6).

Heuristic arms (score / multifactor / fcfs) switch the live Slurm policy via
``baseline_switch`` with the RL hook OFF, so Slurm-native ordering governs.
The DRL arm is handled separately (needs the live obs to carry SLO features);
this runner covers the heuristics (Phase 1). All arms run the SAME CRN workload
(same --seed) → paired. Production (score + RL on) is restored at the end.

Usage:
    python -m eval.scripts.run_aiserve_live \
      --arms score,multifactor,fcfs --login-pod slurm-login-xxx \
      --n-jobs 60 --rounds 2 --warmup 1 --load 0.8 --out-dir runs/aiserve_live_$(date +%s)
"""
from __future__ import annotations

import argparse
import json
import shlex
import statistics as st
import time
from pathlib import Path

from eval.scripts import baseline_switch
from eval.scripts.live_ab_heavytail import (
    BertWorkloadSpec, WorkloadSpec, collect_sacct, gen_aiserve_workload,
    join_records, submit_stream, wait_drain,
)
from sim.metrics import _pct

HEURISTICS = ("score", "multifactor", "fcfs", "packing")


def slo_summary(records: list[dict]) -> dict:
    jcts = [r["jct"] for r in records]
    inf = [r["jct"] for r in records if r["job_class"] == "inference"]
    tr = [r["jct"] for r in records if r["job_class"] == "training"]
    slo = [r for r in records if r["slo_s"] > 0]
    viol = [r for r in slo if r["jct"] > r["slo_s"]]
    return {
        "n": len(records),
        "jct_mean": st.fmean(jcts) if jcts else 0.0,
        "inf_jct": st.fmean(inf) if inf else 0.0,
        "inf_p99": _pct(inf, 99) if inf else 0.0,
        "train_jct": st.fmean(tr) if tr else 0.0,
        "slo_n": len(slo),
        "slo_viol": len(viol),
        "slo_viol_rate": (len(viol) / len(slo)) if slo else 0.0,
    }


def render(summaries: dict) -> str:
    lines = ["# aiserve LIVE — scheduler SLO comparison", "",
             "| arm | n | jct_mean | inf_jct | inf_p99 | train_jct | SLO_viol% |",
             "|---|--:|--:|--:|--:|--:|--:|"]
    for arm, s in summaries.items():
        lines.append(f"| {arm} | {s['n']} | {s['jct_mean']:.1f} | {s['inf_jct']:.1f} | "
                     f"{s['inf_p99']:.1f} | {s['train_jct']:.1f} | {s['slo_viol_rate']*100:.1f} |")
    return "\n".join(lines)


def run(args) -> int:
    exec_prefix = ([*shlex.split(args.kubectl), "exec", "-n", args.namespace,
                    args.login_pod, "--"] if args.login_pod else None)
    if args.workload == "bert":
        workload = BertWorkloadSpec(batch_size=args.bert_batch_size, seq_len=args.bert_seq_len,
                                    infer_rate=args.bert_infer_rate, train_rate=args.bert_train_rate)
    else:
        workload = WorkloadSpec(bin_path=args.workload_bin, dim=args.workload_dim,
                                vram_mb=args.workload_vram_mb, iters_per_sec=args.iters_per_sec,
                                time_factor=args.workload_time_factor)
    arms = [a for a in args.arms.split(",") if a]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records_by_arm: dict[str, list] = {a: [] for a in arms}

    for arm_idx, arm in enumerate(arms):
        print(f"\n=========== ARM {arm} ===========", flush=True)
        if arm in HEURISTICS:
            baseline_switch.switch(arm, dry_run=args.dry_run)   # RL off, Slurm-native
        else:
            print(f"[aiserve-live] arm {arm!r} not a heuristic; skip (DRL arm = Phase 2)")
            continue
        total = args.warmup + args.rounds
        for rnd in range(total):
            is_warm = rnd < args.warmup
            tag = "warmup" if is_warm else f"r{rnd - args.warmup + 1}"
            jobs = gen_aiserve_workload(args.n_jobs, seed=args.seed, n_gpus=2, load=args.load,
                                        inf_runtime_s=args.inf_runtime, train_runtime_s=args.train_runtime,
                                        inf_slo_s=args.inf_slo_s, gang_frac=args.gang_frac)
            print(f"[aiserve-live] {arm} {tag}: submit {len(jobs)} jobs", flush=True)
            since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - 300))
            # Label jobs "score" so sbatch_cmd's `arm != "score"` default does NOT
            # add -H (hold). The actual scheduling policy is set by baseline_switch
            # (Slurm-native, RL off); the job label is only for sacct name-matching.
            # round is offset per-arm so names stay unique across arms in sacct.
            rkey = arm_idx * 100 + rnd
            submit_stream(jobs, "score", rkey, dry_run=args.dry_run, partition=args.partition,
                          exec_prefix=exec_prefix, workload=workload, exclusive_gpu=False)
            if args.dry_run:
                continue
            wait_drain(kubectl=args.kubectl, namespace=args.namespace,
                       controller_pod=args.controller_pod, timeout_s=args.drain_timeout_s)
            parsed = collect_sacct(since, kubectl=args.kubectl, namespace=args.namespace,
                                   controller_pod=args.controller_pod)
            recs = join_records(jobs, parsed, "score", rkey)
            for r in recs:
                r["arm"] = arm   # relabel to the real scheduler arm
            if not is_warm:
                records_by_arm[arm].extend(recs)
        print(f"[aiserve-live] {arm} done: {len(records_by_arm[arm])} records", flush=True)

    summaries = {a: slo_summary(records_by_arm[a]) for a in arms if records_by_arm[a]}
    if not args.dry_run:
        (out / "records.json").write_text(json.dumps(records_by_arm, indent=2))
        report = render(summaries)
        (out / "SUMMARY.md").write_text(report)
        print("\n" + report)
        # restore production: score heuristic + RL hook on
        print("\n=========== RESTORE PRODUCTION ===========", flush=True)
        baseline_switch.switch("score", rl_enabled=True)
        print(f"[aiserve-live] DONE -> {out}/SUMMARY.md")
    else:
        print("[aiserve-live] dry-run complete")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="aiserve live scheduler SLO comparison")
    p.add_argument("--arms", default="score,multifactor,fcfs")
    p.add_argument("--n-jobs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--load", type=float, default=0.8)
    p.add_argument("--inf-runtime", type=float, default=4.0)
    p.add_argument("--train-runtime", type=float, default=30.0)
    p.add_argument("--inf-slo-s", type=float, default=0.0,
                   help="fixed inference SLO deadline (s); 0 = idle-ref × 4. Live "
                        "cuBLAS overhead makes the relative deadline unachievable; "
                        "use a fixed live-calibrated target (e.g. 75)")
    p.add_argument("--gang-frac", type=float, default=0.0,
                   help="fraction of TRAINING jobs that are 2-GPU gang (both nodes) "
                        "→ head-of-line blocking; backfill (multifactor/score) escapes "
                        "it, FCFS does not")
    p.add_argument("--partition", default="gpu")
    p.add_argument("--workload", choices=["cuda", "bert"], default="cuda",
                   help="cuda = sgemm proxy; bert = real BERT infer/fine-tune")
    p.add_argument("--bert-batch-size", type=int, default=16)
    p.add_argument("--bert-seq-len", type=int, default=128)
    p.add_argument("--bert-infer-rate", type=float, default=20.0)
    p.add_argument("--bert-train-rate", type=float, default=6.0)
    p.add_argument("--workload-bin", default="/shared/bin/gpu_workload")
    p.add_argument("--workload-dim", type=int, default=4096)
    p.add_argument("--workload-vram-mb", type=int, default=512)
    p.add_argument("--iters-per-sec", type=float, default=145.0)
    p.add_argument("--workload-time-factor", type=float, default=20.0)
    p.add_argument("--kubectl", default="kubectl")
    p.add_argument("--namespace", default="slurm")
    p.add_argument("--controller-pod", default="slurm-controller-0")
    p.add_argument("--login-pod", default=None)
    p.add_argument("--drain-timeout-s", type=float, default=1800.0)
    p.add_argument("--out-dir", default=f"runs/aiserve_live_{time.strftime('%Y%m%d-%H%M%S')}")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
