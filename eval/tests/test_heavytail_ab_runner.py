"""Unit tests for the heavy-tail A/B runner's pure report builder."""
import numpy as np

from eval.scripts.run_heavytail_ab import build_report, render_summary
from eval.scripts.tail_metrics import mean_makespan


def _recs(arm, jcts, *, rnd=1, true=50.0):
    # submit at t=0, end at t=jct so per-round makespan = max(jct).
    return [{"job_id": f"j{i}", "arm": arm, "round": rnd, "jct": float(j),
             "true_runtime_s": true, "reported_runtime_s": true, "mps_req": 20,
             "wait": 1.0, "state": "COMPLETED",
             "submit_ts": 0.0, "end_ts": float(j)}
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


def test_makespan_is_max_end_minus_min_submit_per_round():
    # round 1: submit 0, ends {10,20,30} → makespan 30; round 2: ends {40,60} → 60.
    recs = _recs("score", [10, 20, 30], rnd=1) + _recs("score", [40, 60], rnd=2)
    assert mean_makespan(recs) == (30.0 + 60.0) / 2

    # missing timestamps → that round contributes no span; all-missing → nan.
    no_ts = [{"job_id": "j0", "round": 1, "jct": 5.0, "submit_ts": None, "end_ts": None}]
    assert np.isnan(mean_makespan(no_ts))


def test_build_report_includes_7_metrics():
    rep = build_report({"score": _recs("score", [10, 20, 30])}, sigma=0.0,
                       family="philly", total_mps=100.0)
    p = rep["panels"]["score"]
    for k in ("mean", "p95", "p99", "makespan", "gpu_util", "slowdown_mean", "sla_viol"):
        assert k in p, f"missing metric {k}"
    # util = Σ(mps·jct)/(cap·makespan) = 20·(10+20+30)/(100·30) = 1200/3000 = 0.4
    assert abs(p["gpu_util"] - 0.4) < 1e-9
    # no SLO jobs (slo_s absent → 0) → sla_viol is nan
    assert np.isnan(p["sla_viol"])


def test_sla_violation_rate():
    from eval.scripts.tail_metrics import sla_violation_rate
    recs = [{"round": 1, "jct": 8.0, "slo_s": 10.0},   # ok
            {"round": 1, "jct": 15.0, "slo_s": 10.0},  # late
            {"round": 1, "jct": 5.0, "slo_s": 0.0}]    # best-effort (ignored)
    assert sla_violation_rate(recs) == 0.5              # 1 of 2 SLO jobs late
    assert np.isnan(sla_violation_rate([{"round": 1, "jct": 5.0, "slo_s": 0.0}]))


def test_render_summary_runs():
    rep = build_report({"score": _recs("score", [10, 20, 30])}, sigma=0.0, family="philly")
    md = render_summary([rep])
    assert "σ=0.0" in md and "score" in md
    # 7-metric header
    for col in ("平均JCT(s)", "P95(s)", "P99(s)", "Makespan(s)", "GPU利用率", "Slowdown", "SLA違反率"):
        assert col in md, f"missing column {col}"
