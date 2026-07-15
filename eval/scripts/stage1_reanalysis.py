#!/usr/bin/env python3
"""Stage-1 statistical re-analysis of the real-machine A/B evals (no new runs).

The paper's real-machine results (§5.3) are mostly a NEGATIVE / equivalence claim
("learned placement does not robustly beat score; they sit within a ±5% band").
For that claim, chasing significance is the wrong frame — the credible tools are:

  * seed-level analysis (avoid job-level pseudoreplication that inflates p),
  * multiple-comparison correction across the arm family (Holm-Bonferroni),
  * bootstrap 95% CIs on the per-seed ΔJCT%,
  * TOST equivalence test against the ±5% practical-equivalence margin,
  * a power / minimum-detectable-effect (MDE) statement for n=8.

This recomputes all of the above from the EXISTING per-seed run outputs, for both
real workloads (cuBLAS low-load and Hybrid LLM high-load). ΔJCT% is per seed,
paired vs the score arm on the same CRN job stream: (score - arm)/score*100, so
POSITIVE = arm faster than score.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

import numpy as np
from scipy import stats

LEARNED_ARMS = ["SAC", "RDSAC-mean", "RDSAC-cvar", "UXP-RL", "RLPD"]
SLURM_ARMS = ["fcfs", "backfill"]
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]
MARGIN = 5.0   # ±5% practical-equivalence band (§5.5)


def _run_dir(tag: str, config: str, seed: int) -> Path | None:
    hits = sorted(glob.glob(f"runs/{tag}_{config}_s{seed}_*"))
    return Path(hits[0]) if hits else None


def _panel_mean(tag: str, config: str, seed: int, arm: str) -> float | None:
    d = _run_dir(tag, config, seed)
    if d is None or not (d / "reports.json").exists():
        return None
    reps = json.loads((d / "reports.json").read_text())
    vals = [r["panels"][arm]["mean"] for r in reps
            if arm in r.get("panels", {}) and r["panels"][arm].get("completed")]
    return float(np.mean(vals)) if vals else None


def per_seed_delta(tag: str, arm: str) -> list[float]:
    """Per-seed ΔJCT% of an arm vs the (learned-config) score arm. + = faster."""
    config = "learned" if arm in LEARNED_ARMS else arm
    out = []
    for s in SEEDS:
        score = _panel_mean(tag, "learned", s, "score")
        val = _panel_mean(tag, config, s, "score" if arm in SLURM_ARMS else arm)
        if score and val:
            out.append(100.0 * (score - val) / score)
    return out


def bootstrap_ci(x: np.ndarray, n_boot=10000, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, x.size, replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def tost(x: np.ndarray, margin: float) -> tuple[float, bool]:
    """TOST equivalence vs ±margin. Returns (tost_p, equivalent@0.05).

    Equivalence declared iff the 90% CI (1-2α) lies within [-margin, margin]."""
    n = x.size
    m, sd = x.mean(), x.std(ddof=1)
    se = sd / np.sqrt(n)
    if se == 0:
        return (0.0, abs(m) < margin)
    df = n - 1
    t_low = (m - (-margin)) / se     # H0: mu <= -margin
    t_up = (m - margin) / se         # H0: mu >= +margin
    p_low = stats.t.sf(t_low, df)    # upper tail
    p_up = stats.t.cdf(t_up, df)     # lower tail
    p = max(p_low, p_up)
    # 90% CI within margin?
    tcrit = stats.t.ppf(0.95, df)
    lo, hi = m - tcrit * se, m + tcrit * se
    return float(p), bool(lo > -margin and hi < margin)


def mde(sd: float, n: int, alpha=0.05, power=0.8) -> float:
    """Two-sided paired minimum detectable effect (same units as sd, i.e. %)."""
    df = n - 1
    return (stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) * sd / np.sqrt(n)


def analyse(tag: str, label: str) -> dict:
    arms = LEARNED_ARMS + SLURM_ARMS
    rows, raw_p = [], []
    for arm in arms:
        d = np.array(per_seed_delta(tag, arm), dtype=float)
        if d.size < 2:
            continue
        m, sd = d.mean(), d.std(ddof=1)
        t, p = stats.ttest_1samp(d, 0.0)
        lo, hi = bootstrap_ci(d)
        tp, equiv = tost(d, MARGIN)
        rows.append({"arm": arm, "n": int(d.size), "mean": m, "sd": sd,
                     "raw_p": float(p), "ci": (lo, hi), "tost_p": tp, "equiv": equiv})
        raw_p.append((arm, float(p)))
    # Holm-Bonferroni across the arm family
    holm = holm_bonferroni([p for _, p in raw_p])
    for row, (_, _), hp in zip(rows, raw_p, holm):
        row["holm_p"] = hp
    sd_pool = float(np.sqrt(np.mean([r["sd"] ** 2 for r in rows])))
    return {"tag": tag, "label": label, "rows": rows,
            "mde": mde(sd_pool, len(SEEDS)), "sd_pool": sd_pool}


def holm_bonferroni(ps: list[float], alpha=0.05) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * ps[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def print_report(res: dict) -> None:
    print(f"\n### {res['label']}  ({res['tag']}, n={len(SEEDS)} seeds)")
    print(f"{'arm':11s} {'ΔJCT%':>9s} {'raw p':>8s} {'Holm p':>8s} "
          f"{'boot 95% CI':>18s} {'TOST p':>8s} {'±5%等價':>8s}")
    for r in res["rows"]:
        ci = f"[{r['ci'][0]:+.1f},{r['ci'][1]:+.1f}]"
        print(f"{r['arm']:11s} {r['mean']:+8.1f}±{r['sd']:<3.0f} {r['raw_p']:8.3f} "
              f"{r['holm_p']:8.3f} {ci:>18s} {r['tost_p']:8.3f} {'是' if r['equiv'] else '否':>7s}")
    print(f"  pooled SD={res['sd_pool']:.1f}% → MDE(n=8, power .8)≈{res['mde']:.1f}%  "
          f"(能可靠偵測的最小效應；<此值的差異在 n=8 檢力不足)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cublas", default="cublas8")
    ap.add_argument("--hybrid", default="full8")
    ap.add_argument("--out", default="runs/stage1_reanalysis.json")
    args = ap.parse_args()
    results = [analyse(args.cublas, "實機 DRA cuBLAS（低負載）"),
               analyse(args.hybrid, "實機 DRA Hybrid（高負載 LLM serving）")]
    for r in results:
        print_report(r)
    print("\n判讀：Holm 校正後仍顯著者才是穩健差異；TOST『是』= 90%CI 落在 ±5% 內"
          "（證實等價）；兩者皆否 + CI 跨 ±5% = 此規模解析度內不可區分。")
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"[stage1] wrote {args.out}")


if __name__ == "__main__":
    main()
