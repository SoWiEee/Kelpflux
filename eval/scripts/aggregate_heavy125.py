#!/usr/bin/env python3
"""Aggregate the heavy-load (§6.3) live A/B into the §5.3 hybrid 4-metric table.

Reads the per-seed run dirs produced by run_heavy125_5arm.sh and emits, per arm:
JCT / Makespan / P95 / P99 (seed-averaged absolutes) plus the primary significance
test — seed-level paired ΔJCT% vs score (one-sample t over training seeds).

  learned run dir: runs/<tag>_learned_s<seed>_<stamp>/reports.json
    → score panel + SAC/RDSAC-mean/RDSAC-cvar panels + paired_vs_score deltas
  slurm run dirs : runs/<tag>_{fcfs,backfill}_s<seed>_<stamp>/reports.json
    → score panel only (vanilla Slurm placement); ΔJCT% computed vs learned-config score

Usage (called by the harness; args positional):
    aggregate_heavy125.py <stamp> <tag> "<learned prefixes>" <seed> [<seed> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# learned ckpt-prefix → reports.json panel key
PREFIX_TO_PANEL = {"sac": "SAC", "rdsac_mean": "RDSAC-mean", "rdsac_cvar": "RDSAC-cvar"}
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


def _paired_djct(entries, arm: str):
    vals = [e["paired_vs_score"][arm]["djct_pct"] for e in entries
            if arm in e.get("paired_vs_score", {})]
    return float(np.mean(vals)) if vals else None


def _fmt_seed_stat(d: dict[int, float]):
    """d: seed→value → 'mean±sd', 'pos/n', 'p' (one-sample t vs 0)."""
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

    # per-arm per-metric absolute (seed→value); plus score & paired deltas
    abs_metrics: dict[str, dict[str, dict[int, float]]] = {}
    djct: dict[str, dict[int, float]] = {}
    score_jct: dict[int, float] = {}

    for s in seeds:
        entries = _load(Path(f"runs/{tag}_learned_s{s}_{stamp}/reports.json"))
        if not entries:
            continue
        sj = _panel_metric(entries, "score", "mean")
        if sj is not None:
            score_jct[s] = sj
        for arm in ["score", *learned_panels]:
            for m in METRICS:
                v = _panel_metric(entries, arm, m)
                if v is not None:
                    abs_metrics.setdefault(arm, {}).setdefault(m, {})[s] = v
        for arm in learned_panels:
            dv = _paired_djct(entries, arm)
            if dv is not None:
                djct.setdefault(arm, {})[s] = dv

    # naive-Slurm arms: score panel from their own run → absolute metrics + ΔJCT% vs learned-score
    for cfg in ("fcfs", "backfill"):
        for s in seeds:
            entries = _load(Path(f"runs/{tag}_{cfg}_s{s}_{stamp}/reports.json"))
            if not entries:
                continue
            for m in METRICS:
                v = _panel_metric(entries, "score", m)
                if v is not None:
                    abs_metrics.setdefault(cfg, {}).setdefault(m, {})[s] = v
            sj = _panel_metric(entries, "score", "mean")
            if sj is not None and s in score_jct and score_jct[s] > 0:
                djct.setdefault(cfg, {})[s] = 100.0 * (score_jct[s] - sj) / score_jct[s]

    # ----- render -----
    row_order = ["score", *learned_panels, "fcfs", "backfill"]
    label = {"score": "score (backfill+multifactor+Lua)",
             "fcfs": "fcfs (naive Slurm)", "backfill": "backfill (naive Slurm)"}
    out = [
        f"# Heavy-load live A/B ({tag}) — {stamp}", "",
        f"seeds={sorted(score_jct)} (n={len(score_jct)}). n_jobs≈125 regime (§5.7 headroom "
        "≈10%). Learned arms: " + ", ".join(learned_panels) + ". All arms one DRA MPS "
        "backend, Qwen hybrid, oversub 2.0. ΔJCT% vs score (+=faster); seed_t=one-sample t "
        "over training seeds (primary significance).", "",
        "| arm | JCT(s) | Makespan(s) | P95(s) | P99(s) | ΔJCT% vs score | seed+ | seed_t p |",
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
        if arm == "score":
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
