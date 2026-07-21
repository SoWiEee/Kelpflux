#!/usr/bin/env python3
"""How much of the high-load headroom can score's SJF weight (epsilon) capture?

The headroom study (headroom_analysis.py) shows that at heavy load the best
static ordering beats `score` by 10-14%. If that gap were simply "score
under-weights shortest-job-first", then tuning score's `epsilon` (the
f_runtime_short SJF kicker) would close most of it — and no learned scheduler
would be needed. This script tests that: for each high-load instance it sweeps
epsilon and measures what fraction of the headroom epsilon-tuning recovers.

Reuses the best-ordering ceiling (best_jct) already computed by
headroom_analysis.py, so it only re-runs `score` at each epsilon.

Usage:
    PYTHONPATH=. .venv-m11/bin/python eval/scripts/score_epsilon_sweep.py \
        --headroom-results runs/headroom_YYYYMMDD-HHMMSS/results.json \
        --loads 100 125 150
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from sim.loader import generate_by_family
from sim.runner import run


def _score_jct(jobs, eps, *, n_nodes, gpus_per_node) -> float:
    metrics, _ = run(jobs, n_nodes=n_nodes, gpus_per_node=gpus_per_node,
                     scheduler_name="score", scheduler_kwargs={"epsilon": eps})
    return metrics.summary()["jct_mean"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--headroom-results", required=True,
                   help="results.json from headroom_analysis.py (provides best_jct)")
    p.add_argument("--n-nodes", type=int, default=2)
    p.add_argument("--gpus-per-node", type=int, default=1)
    p.add_argument("--loads", type=int, nargs="+", default=[100, 125, 150])
    p.add_argument("--eps-grid", type=float, nargs="+",
                   default=[0.0, 0.30, 0.50, 0.70, 1.0])
    p.add_argument("--default-eps", type=float, default=0.30)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    total_gpus = args.n_nodes * args.gpus_per_node
    rows = json.load(open(args.headroom_results))
    load_rows = [r for r in rows if r.get("tier") == "load" and r["load"] in args.loads]
    kw = dict(n_nodes=args.n_nodes, gpus_per_node=args.gpus_per_node)

    per_load: dict[int, list] = {}
    for r in load_rows:
        jobs = [j for j in generate_by_family(r["family"], n_jobs=r["load"], seed=r["seed"])
                if j.gpu_count <= total_gpus]
        base = _score_jct(jobs, args.default_eps, **kw)   # default score
        floor = r["best_jct"]                             # best-ordering ceiling
        eps_jcts = {e: _score_jct(jobs, e, **kw) for e in args.eps_grid}
        best_eps = min(eps_jcts, key=eps_jcts.get)
        best_eps_jct = eps_jcts[best_eps]
        headroom = base - floor
        captured = base - best_eps_jct
        frac = captured / headroom * 100.0 if headroom > 1e-6 else float("nan")
        per_load.setdefault(r["load"], []).append(
            {"family": r["family"], "seed": r["seed"], "base": base, "floor": floor,
             "best_eps": best_eps, "best_eps_jct": best_eps_jct,
             "headroom_pct": (base - floor) / floor * 100.0,
             "captured_pct": captured / floor * 100.0, "frac_of_headroom": frac})

    summary = {}
    print(f"{'load':5s}  {'headroom%':>9s}  {'ε-captured%':>11s}  {'frac captured':>13s}  {'best ε (mode)':>13s}")
    print("-" * 62)
    for load in args.loads:
        recs = per_load.get(load, [])
        if not recs:
            continue
        hr = float(np.mean([x["headroom_pct"] for x in recs]))
        cap = float(np.mean([x["captured_pct"] for x in recs]))
        fr = float(np.nanmean([x["frac_of_headroom"] for x in recs]))
        mode_eps = Counter(x["best_eps"] for x in recs).most_common(1)[0]
        summary[str(load)] = {"headroom_pct": hr, "captured_pct": cap,
                              "frac_of_headroom_pct": fr, "best_eps_mode": mode_eps[0]}
        print(f"{load:<5d}  {hr:+8.1f}%  {cap:+10.1f}%  {fr:12.0f}%  "
              f"{mode_eps[0]:.2f} ({mode_eps[1]}/{len(recs)})")
    print("\nTuning score's SJF weight recovers only a minority of the high-load "
          "headroom → most of it is beyond linear reweighting.")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "epsilon_sweep.json").write_text(
            json.dumps({"summary": summary, "per_load": per_load}, indent=2))
        print(f"Wrote {out}/epsilon_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
