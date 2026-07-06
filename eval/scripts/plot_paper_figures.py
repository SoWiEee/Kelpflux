"""Generate the two paper figures for docs/paper.md.

Both figures visualize honest-negative findings that were previously only in
prose/tables:

  fig_drift.png  — the apparent single-pass advantage is an artifact of
                   run-order (GPU warm-up) drift: ΔJCT% vs score grows with
                   how *late* a method ran, not with the method itself
                   (source: paper §4.2 table 2, 3 seeds × 3 heuristics).

  fig_scale.png  — no scale crossover: ΔJCT% vs score does NOT trend toward 0
                   as the cluster grows 1×1 → 2×1 → 2×2, so "value requires
                   scale" is unsupported (source: runs/review_scale_{1x1,2x1,2x2}
                   SUMMARY.md, σ=1.0, 40k steps — see §4.5 caveats).

English axis labels avoid CJK-font dependence; captions live in paper.md.

    PYTHONPATH=. .venv-m11/bin/python eval/scripts/plot_paper_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def plot_drift() -> None:
    """§4.2 table 2: (run-order position, ΔJCT% vs score) for 3 heuristics."""
    # (position, ΔJCT%) per method, across seed42/43/44
    data = {
        "FCFS":        [(4, 5.0), (1, 0.5), (3, -0.4)],
        "packing":     [(3, 1.6), (2, 0.8), (1, -0.6)],
        "multifactor": [(2, 1.0), (3, 0.9), (4, -0.7)],
    }
    markers = {"FCFS": "o", "packing": "s", "multifactor": "^"}

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    xs_all, ys_all = [], []
    for name, pts in data.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        xs_all += xs
        ys_all += ys
        ax.scatter(xs, ys, marker=markers[name], s=70, label=name, zorder=3)

    # least-squares trend line over the pooled points
    n = len(xs_all)
    sx = sum(xs_all); sy = sum(ys_all)
    sxx = sum(x * x for x in xs_all); sxy = sum(x * y for x, y in zip(xs_all, ys_all))
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    xline = [0.7, 4.3]
    ax.plot(xline, [slope * x + intercept for x in xline], "--", color="0.4",
            zorder=2, label=f"trend (slope={slope:+.2f}%/pos)")

    ax.axhline(0, color="0.6", lw=0.8, zorder=1)
    ax.set_xlabel("Run-order position (1 = earliest, 4 = latest)")
    ax.set_ylabel("ΔJCT% vs score  (+ = faster)")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_title("Apparent advantage tracks run-order, not method")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_drift.png")
    plt.close(fig)


def plot_scale() -> None:
    """review_scale_*: ΔJCT% vs score at 1×1/2×1/2×2 (σ=1.0, 40k). No crossover."""
    scales = ["1×1", "2×1", "2×2"]
    x = [1, 2, 3]
    # None = policy collapsed to 0% completion (ΔJCT% undefined)
    series = {
        "SAC (philly)":        [-97.5, -24.4, -116.5],
        "SAC (ali)":           [-95.0, -13.9, -65.6],
        "RDSAC-cvar (philly)": [None,  -72.1, -89.8],
        "RDSAC-cvar (ali)":    [None,  -51.3, -136.4],
    }
    styles = {
        "SAC (philly)":        dict(color="C0", marker="o", ls="-"),
        "SAC (ali)":           dict(color="C0", marker="o", ls="--"),
        "RDSAC-cvar (philly)": dict(color="C3", marker="s", ls="-"),
        "RDSAC-cvar (ali)":    dict(color="C3", marker="s", ls="--"),
    }

    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    for name, ys in series.items():
        xs = [xi for xi, y in zip(x, ys) if y is not None]
        yv = [y for y in ys if y is not None]
        ax.plot(xs, yv, label=name, **styles[name], zorder=3)
    # both RDSAC-cvar cells collapse at 1×1 → mark once, annotate to the right
    ax.scatter([1], [-150], marker="x", color="C3", s=70, zorder=4)
    ax.annotate("RDSAC-cvar collapsed 0% done", (1, -150), xytext=(1.9, -150),
                fontsize=7.5, ha="left", va="center", color="C3",
                arrowprops=dict(arrowstyle="->", color="C3", lw=0.8))

    ax.set_xlim(0.8, 3.25)
    ax.axhline(0, color="0.5", lw=1.0, zorder=1)
    ax.text(0.85, 3, "score baseline (ΔJCT%=0)", fontsize=8, color="0.4",
            va="bottom", ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_xlabel("Cluster scale (nodes × GPUs/node)")
    ax.set_ylabel("ΔJCT% vs score  (+ = faster)")
    ax.set_title("No scale crossover: learned arms stay below score")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_scale.png")
    plt.close(fig)


def plot_stabilizer() -> None:
    """value-clip ablation: collapse count (/6 trace×seed) clip-off vs clip-on
    for the two distributional arms (source: value_clip_ablation TABLES.md)."""
    arms = ["RDSAC-mean", "RDSAC-cvar"]
    off = [2, 0]   # collapsed cells (<20% done) out of 6
    on = [0, 2]

    x = list(range(len(arms)))
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    b1 = ax.bar([xi - w / 2 for xi in x], off, w, label="clip-off",
                color="C0", zorder=3)
    b2 = ax.bar([xi + w / 2 for xi in x], on, w, label="clip-on (b=10)",
                color="C3", zorder=3)
    ax.bar_label(b1, labels=[f"{v}/6" for v in off], padding=2, fontsize=9)
    ax.bar_label(b2, labels=[f"{v}/6" for v in on], padding=2, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("collapsed cells (<20% done, /6)")
    ax.set_ylim(0, 3)
    ax.set_title("Return-clip rescues RDSAC-mean, conflicts with CVaR")
    ax.legend(fontsize=9, loc="upper center")
    # arrows telling the two opposite stories
    ax.annotate("", xy=(-w / 2, 0.15), xytext=(-w / 2, 1.9),
                arrowprops=dict(arrowstyle="->", color="0.3", lw=1.4))
    ax.text(0, 2.35, "mean: 2→0 ✓", ha="center", fontsize=8, color="0.2")
    ax.annotate("", xy=(1 + w / 2, 1.9), xytext=(1 + w / 2, 0.15),
                arrowprops=dict(arrowstyle="->", color="0.3", lw=1.4))
    ax.text(1, 2.35, "cvar: 0→2 ✗", ha="center", fontsize=8, color="0.2")
    fig.tight_layout()
    fig.savefig(OUT / "fig_stabilizer.png")
    plt.close(fig)


def main() -> int:
    plot_drift()
    plot_scale()
    plot_stabilizer()
    for f in ("fig_drift.png", "fig_scale.png", "fig_stabilizer.png"):
        print(f"[out] {OUT / f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
