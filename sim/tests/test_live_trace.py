import json
import subprocess
import sys
from pathlib import Path

from sim.live_trace import sacct_to_normalized


SACCT = """JobID|User|JobName|Partition|State|Submit|Start|End|ElapsedRaw|NodeList|AllocTRES|ReqTRES
129|root|dsac-place-test-hard|gpu-rtx4070|COMPLETED|2026-06-03T08:58:01|2026-06-03T08:58:17|2026-06-03T08:58:22|5|slurm-worker-gpu-rtx4070-1|billing=2,cpu=2,gres/mps=25,node=1|cpu=2,gres/mps=25
129.batch|root|batch||COMPLETED|2026-06-03T08:58:17|2026-06-03T08:58:17|2026-06-03T08:58:22|5|slurm-worker-gpu-rtx4070-1|cpu=2,gres/mps=25,mem=0,node=1|
135|root|otel-openmp-joined|cpu|COMPLETED|2026-06-03T13:23:50|2026-06-03T13:23:50|2026-06-03T13:25:05|75|slurm-worker-cpu-0|billing=4,cpu=4,node=1|cpu=4
"""


def test_sacct_to_normalized_gpu_mps_trace():
    jobs, stats = sacct_to_normalized(SACCT)
    assert stats.raw_rows == 3
    assert stats.emitted_jobs == 1
    assert stats.skipped_steps == 1
    assert stats.skipped_cpu == 1
    job = jobs[0]
    assert job["job_id"] == "129"
    assert job["gpu_count"] == 1
    assert job["mps_req"] == 25
    assert job["gpu_type"] == "rtx4070"
    assert job["submit_ts"] == 0.0
    assert job["runtime"] == 5.0
    assert job["live_wait"] == 16.0
    assert job["latency_class"] == "gpu_warm"


def test_collector_script_input_mode(tmp_path):
    raw = tmp_path / "sacct.txt"
    out = tmp_path / "trace.json"
    summary = tmp_path / "latency.json"
    raw.write_text(SACCT)
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/collect-live-trace.py",
            "--input",
            str(raw),
            "--output",
            str(out),
            "--latency-summary",
            str(summary),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    meta = json.loads(proc.stdout)
    assert meta["emitted_jobs"] == 1
    trace = json.loads(out.read_text())
    assert trace[0]["job_id"] == "129"
    latency = json.loads(summary.read_text())
    assert latency["gpu_warm"]["mean"] == 16.0
