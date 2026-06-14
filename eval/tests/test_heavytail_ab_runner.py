"""Unit tests for the heavy-tail A/B runner's pure report builder."""
import numpy as np

from eval.scripts.run_heavytail_ab import build_report, render_summary


def _recs(arm, jcts, *, rnd=1, true=50.0):
    return [{"job_id": f"j{i}", "arm": arm, "round": rnd, "jct": float(j),
             "true_runtime_s": true, "reported_runtime_s": true, "mps_req": 20,
             "wait": 1.0, "state": "COMPLETED"}
            for i, j in enumerate(jcts)]


def test_build_report_panels_and_paired():
    rng = np.random.default_rng(0)
    score = rng.uniform(50, 150, 60).tolist()
    # RDSAC caps the tail at p75 → CVaR drops more than mean
    p75 = float(np.percentile(score, 75))
    rdsac = np.minimum(score, p75).tolist()
    rec = {
        "score": _recs("score", score),
        "RDSAC-cvar": _recs("RDSAC-cvar", rdsac),
    }
    rep = build_report(rec, sigma=1.0, family="philly")

    assert set(rep["panels"]) == {"score", "RDSAC-cvar"}
    assert rep["panels"]["score"]["completed"] == 60
    d = rep["paired_vs_score"]["RDSAC-cvar"]
    assert d["dcvar_pct"] > d["djct_pct"]          # tail gain > mean gain
    assert "ttest_p" in d


def test_build_report_pairs_on_common_jobs_only():
    # score has j0..j2, model has j1..j3 → paired on the intersection {j1,j2}
    rec = {
        "score": _recs("score", [10, 20, 30])[:3],
        "SAC": [r for r in _recs("SAC", [0, 21, 33, 40]) if r["job_id"] in {"j1", "j2", "j3"}],
    }
    rep = build_report(rec, sigma=0.0, family="ali")
    assert rep["paired_vs_score"]["SAC"]["n"] == 2  # only j1, j2 shared


def test_render_summary_runs():
    rep = build_report({"score": _recs("score", [10, 20, 30])}, sigma=0.0, family="philly")
    md = render_summary([rep])
    assert "σ=0.0" in md and "score" in md
