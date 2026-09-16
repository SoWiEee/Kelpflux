import pytest

from eval.scripts.tail_metrics import summarize_gpu_samples, summarize_live_records


def test_active_job_filter_ignores_terminal_squeue_rows(monkeypatch):
    from eval.scripts import scontrol_ab

    monkeypatch.setattr(
        scontrol_ab,
        "_exec",
        lambda *_args, **_kwargs: "1|COMPLETED\n2|RUNNING\n3|PENDING\n",
    )
    assert scontrol_ab._active_job_ids() == {"2", "3"}


def test_live_records_report_slowdown_and_fairness():
    records = [
        {"jct": 60.0, "wait": 10.0, "runtime": 60.0,
         "submit_ts": 0.0, "end_ts": 60.0},
        {"jct": 120.0, "wait": 20.0, "runtime": 120.0,
         "submit_ts": 1.0, "end_ts": 121.0},
    ]
    metrics = summarize_live_records(records, n_jobs=3)
    assert metrics["completed"] == 2
    assert metrics["completion_rate"] == pytest.approx(2 / 3)
    assert metrics["slowdown_mean"] == pytest.approx(1.0)
    assert metrics["jain_slowdown"] == pytest.approx(1.0)
    assert metrics["wait_p95"] == pytest.approx(19.5)


def test_gpu_metrics_mark_missing_node_incomplete():
    samples = [
        {"node": "gpu-a", "gpu_util_pct": 20.0, "memory_pressure_pct": 40.0},
        {"node": "gpu-a", "gpu_util_pct": 80.0, "memory_pressure_pct": 60.0},
    ]
    metrics = summarize_gpu_samples(samples, ["gpu-a", "gpu-b"])
    assert metrics["gpu_util_mean_pct"] == pytest.approx(50.0)
    assert metrics["gpu_memory_pressure_peak_pct"] == pytest.approx(60.0)
    assert metrics["gpu_telemetry_nodes"] == ["gpu-a"]
    assert metrics["gpu_telemetry_complete"] is False


def test_gpu_metrics_complete_when_all_nodes_observed():
    samples = [
        {"node": "gpu-a", "gpu_util_pct": 50.0, "memory_pressure_pct": 25.0},
        {"node": "gpu-b", "gpu_util_pct": 70.0, "memory_pressure_pct": 75.0},
    ]
    metrics = summarize_gpu_samples(samples, ["gpu-a", "gpu-b"])
    assert metrics["gpu_telemetry_complete"] is True
