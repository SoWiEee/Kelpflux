from sim.latency import LatencyModel
from sim.loader import Job, MPS_PER_GPU


def _job(job_id="j", latency_class=""):
    return Job(job_id, "u", 1, "rtx4070", 0.0, 10.0, 0.0, MPS_PER_GPU, latency_class)


def test_live_basic_gpu_cold_then_warm():
    model = LatencyModel.live_basic(gpu_warm_seconds=3.0, gpu_cold_seconds=15.0, cold_after_seconds=300.0)
    first = _job("first")
    assert model.start_latency(first, 0.0) == 15.0
    model.record_start(first)
    model.record_end(first, 25.0)
    second = _job("second")
    assert model.start_latency(second, 100.0) == 3.0
    assert model.start_latency(second, 400.0) == 15.0


def test_hard_placement_latency_overrides_warm_cold():
    model = LatencyModel.live_basic(gpu_warm_seconds=3.0, gpu_cold_seconds=15.0, hard_placement_seconds=22.0)
    assert model.start_latency(_job("held", "hard_placement"), 0.0) == 22.0


def test_explicit_gpu_warm_class_overrides_idle_cold_inference():
    model = LatencyModel.live_basic(gpu_warm_seconds=4.0, gpu_cold_seconds=99.0)
    assert model.start_latency(_job("warm", "gpu_warm"), 0.0) == 4.0
