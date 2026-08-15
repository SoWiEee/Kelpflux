#!/usr/bin/env python3
"""Aggregate the heavy-load (§6.3) aimix 6-arm A/B — baseline = BACKFILL, no score.

score is fully removed from the evaluation. The learned config runs with --no-score
(SAC / RDSAC-mean / RDSAC-cvar / RLPD only); the two Slurm-native configs each emit a
single vanilla-placement panel (internally still labelled "score" by run_heavytail_ab,
but under the fcfs / backfill Slurm config it IS that scheduler). BACKFILL is the paired
reference every other arm is measured against.

  learned run dir : runs/<tag>_learned_s<seed>_<stamp>/reports.json
    → SAC / RDSAC-mean / RDSAC-cvar / RLPD panels (no score panel)
  slurm run dirs  : runs/<tag>_{fcfs,backfill}_s<seed>_<stamp>/reports.json
    → one "score"-labelled panel = that config's vanilla-Slurm metric

Primary significance = seed-level paired ΔJCT% vs backfill (one-sample t over seeds),
ΔJCT% = 100·(backfill_jct − arm_jct)/backfill_jct  (+ = arm is faster than backfill).

Usage (positional, called by the harness):
    aggregate_heavy150_vs_backfill.py <stamp> <tag> "<learned prefixes>" <seed> [<seed> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# learned ckpt-prefix → reports.json panel key
PREFIX_TO_PANEL = {"sac": "SAC", "rdsac_mean": "RDSAC-mean", "rdsac_cvar": "RDSAC-cvar",
                   "rlpd_cvar": "RLPD"}
METRICS = ("mean", "makespan", "p95", "p99")  # JCT / Makespan / P95 / P99


def _load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _panel_metric(entries, arm: str, metric: str):
    """Mean of `metric` over all report entries whose `arm` panel completed."""
    vals = [e["panels"][arm][metric] for e in entries
            if arm in e.get("panels", {}) and e["panels"][arm].get("completed")
            and metric in e["panels"][arm]]
    return float(np.mean(vals)) if vals else None


def _fmt_seed_stat(d: dict[int, float]):
    """d: seed→ΔJCT% → 'mean±sd', 'pos/n', one-sample-t p vs 0."""
    a = np.array([d[s] for s in sorted(d)], float)
    if a.size < 2:
        return f"(n={a.size})", "—", "—"
    pos = int((a > 0).sum())
    p = stats.ttest_1samp(a, 0.0).pvalue
    return f"{a.mean():+.1f}±{np.std(a, ddof=1):.1f}", f"{pos}/{a.size}", f"{p:.3f}"


def _fmt_abs(d: dict[int, float]):
    a = np.array([d[s] for s in sorted(d)], float)
    if a.size == 0:
        return "—"
    sd = np.std(a, ddof=1) if a.size >= 2 else 0.0
    return f"{a.mean():.1f}±{sd:.1f}"


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    stamp, tag, learned_str, *seed_strs = argv
    learned = [x for x in learned_str.split() if x]
    seeds = [int(s) for s in seed_strs]
    learned_panels = [PREFIX_TO_PANEL[p] for p in learned if p in PREFIX_TO_PANEL]

    abs_metrics: dict[str, dict[str, dict[int, float]]] = {}
    arm_jct: dict[str, dict[int, float]] = {}   # arm → seed → JCT mean

    # learned arms — absolute panels from the --no-score learned run
    for s in seeds:
        entries = _load(Path(f"runs/{tag}_learned_s{s}_{stamp}/reports.json"))
        if not entries:
            continue
        for arm in learned_panels:
            for m in METRICS:
                v = _panel_metric(entries, arm, m)
                if v is not None:
                    abs_metrics.setdefault(arm, {}).setdefault(m, {})[s] = v
            jm = _panel_metric(entries, arm, "mean")
            if jm is not None:
                arm_jct.setdefault(arm, {})[s] = jm

    # Slurm-native arms — the single "score"-labelled panel IS that config's metric
    for cfg in ("fcfs", "backfill"):
        for s in seeds:
            entries = _load(Path(f"runs/{tag}_{cfg}_s{s}_{stamp}/reports.json"))
            if not entries:
                continue
            for m in METRICS:
                v = _panel_metric(entries, "score", m)
                if v is not None:
                    abs_metrics.setdefault(cfg, {}).setdefault(m, {})[s] = v
            jm = _panel_metric(entries, "score", "mean")
            if jm is not None:
                arm_jct.setdefault(cfg, {})[s] = jm

    # ----- baseline = backfill; ΔJCT% vs backfill, seed-paired -----
    base = arm_jct.get("backfill", {})
    djct: dict[str, dict[int, float]] = {}
    for arm, seed_jct in arm_jct.items():
        if arm == "backfill":
            continue
        for s, j in seed_jct.items():
            if s in base and base[s] > 0:
                djct.setdefault(arm, {})[s] = 100.0 * (base[s] - j) / base[s]

    # ----- render -----
    row_order = ["fcfs", "backfill", *learned_panels]
    label = {"fcfs": "fcfs (naive Slurm)", "backfill": "backfill (naive Slurm) [baseline]"}
    out = [
        f"# Heavy-load aimix 6-arm A/B ({tag}) — {stamp}", "",
        f"seeds={sorted(base)} (n={len(base)}). n_jobs≈150 regime (§5.7 headroom window "
        "125–150, ≈10–14% over score, ~16% ε-reachable). Arms: fcfs, backfill, "
        + ", ".join(learned_panels) + ". score REMOVED from evaluation. All arms one DRA MPS "
        "backend, aimix (Qwen/BERT/ResNet/cuBLAS), oversub 2.0. ΔJCT% vs BACKFILL "
        "(+=faster than backfill); seed_t=one-sample t over seeds (primary significance).", "",
        "| arm | JCT(s) | Makespan(s) | P95(s) | P99(s) | ΔJCT% vs backfill | seed+ | seed_t p |",
        "|---|--:|--:|--:|--:|--:|:--:|--:|",
    ]
    for arm in row_order:
        if arm not in abs_metrics:
            continue
        am = abs_metrics[arm]
        jct = _fmt_abs(am.get("mean", {}))
        mk = _fmt_abs(am.get("makespan", {}))
        p95 = _fmt_abs(am.get("p95", {}))
        p99 = _fmt_abs(am.get("p99", {}))
        if arm == "backfill":
            dj, pos, pv = "baseline", "—", "—"
        else:
            dj, pos, pv = _fmt_seed_stat(djct.get(arm, {}))
        out.append(f"| {label.get(arm, arm)} | {jct} | {mk} | {p95} | {p99} | {dj} | {pos} | {pv} |")

    Path(f"runs/{tag}_{stamp}_TABLES.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n[agg] runs/{tag}_{stamp}_TABLES.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
