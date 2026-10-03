"""Serve the dashboard with live Slurm and Kubernetes telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent
_NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}
_TERMINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED",
}
_PID_JOB_SCRIPT = r'''
for start do
  pid="$start"
  job=""
  depth=0
  while [ -n "$pid" ] && [ "$pid" -gt 1 ] 2>/dev/null && [ "$depth" -lt 32 ]; do
    if [ -r "/proc/$pid/environ" ]; then
      job=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^SLURM_JOB_ID=//p' | sed -n '1p')
    fi
    if [ -z "$job" ] && [ -r "/proc/$pid/cgroup" ]; then
      job=$(sed -n 's/.*job[_-]\([0-9][0-9]*\).*/\1/p' "/proc/$pid/cgroup" | sed -n '1p')
    fi
    if [ -z "$job" ] && [ -r "/proc/$pid/cmdline" ]; then
      job=$(tr '\0' ' ' < "/proc/$pid/cmdline" | sed -n 's/.*slurmstepd: \[\([0-9][0-9]*\)\..*/\1/p')
    fi
    [ -n "$job" ] && break
    pid=$(sed -n 's/^PPid:[[:space:]]*//p' "/proc/$pid/status" 2>/dev/null)
    depth=$((depth + 1))
  done
  printf '%s|%s\n' "$start" "$job"
done
'''.strip()


def _number(value: object) -> float | None:
    if isinstance(value, dict):
        value = value.get("number", value.get("value"))
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _int(value: object, default: int = 0) -> int:
    number = _number(value)
    return int(number) if number is not None else default


def _timestamp(value: object) -> float | None:
    number = _number(value)
    if number is not None and number > 0:
        return number
    return None


def _iso_timestamp(value: str) -> float | None:
    if not value or value in {"Unknown", "None", "N/A"}:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _first(value: object) -> object:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _mps_value(*values: object) -> int | None:
    """Parse Slurm's mps:25, gres/mps=25, and mps:rtx3080:100 forms."""
    pattern = re.compile(r"(?:gres/)?mps(?:/[^:=,\s]+)?(?::[^:=,\s]+)?[:=](\d+)", re.I)
    for value in values:
        if value is None:
            continue
        text = ",".join(str(item) for item in value) if isinstance(value, list) else str(value)
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def _node_from_job(job: dict[str, Any]) -> str:
    resources = job.get("job_resources") or {}
    node = resources.get("nodes") or job.get("nodes") or ""
    return str(node).split(",", 1)[0].strip()


def _gpu_from_node(node: str, mps: int | None) -> dict[str, object]:
    match = re.search(r"gpu-([^-]+)-(\d+)$", node)
    if match:
        return {"type": match.group(1), "index": int(match.group(2)), "count": 1}
    return {"count": 1 if mps is not None else 0}


def _normalise_live_job(raw: dict[str, Any]) -> dict[str, object]:
    requested = _mps_value(raw.get("tres_req_str"), raw.get("tres_per_node"))
    allocated = _mps_value(raw.get("tres_alloc_str"), raw.get("gres_detail"))
    node = _node_from_job(raw)
    state = _first(raw.get("job_state")) or raw.get("state") or "UNKNOWN"
    submit_ts = _timestamp(raw.get("submit_time"))
    return {
        "job_id": str(raw.get("job_id", raw.get("id", ""))),
        "state": str(state),
        "node": node,
        "gpu": _gpu_from_node(node, requested or allocated),
        "mps": {"requested": requested, "allocated": allocated},
        "submit_ts": submit_ts,
        "start_ts": _timestamp(raw.get("start_time")),
        "job_name": str(raw.get("name", "")),
        "command": str(raw.get("command", raw.get("submit_line", ""))),
        "resource_usage": None,
        "reason": raw.get("state_reason", ""),
        "partition": raw.get("partition", ""),
    }


def _parse_sacct_history(text: str, limit: int = 50) -> list[dict[str, object]]:
    fields = (
        "job_id", "job_name", "state", "submit", "start", "end", "elapsed",
        "node", "allocated_tres", "requested_tres", "partition", "command",
    )
    jobs = []
    for line in text.splitlines():
        values = (line[:-1] if line.endswith("|") else line).split("|", len(fields) - 1)
        if len(values) != len(fields):
            continue
        row = dict(zip(fields, values))
        state = row["state"].split()[0].split("+")[0].upper()
        if state not in _TERMINAL_STATES:
            continue
        submit = _iso_timestamp(row["submit"])
        start = _iso_timestamp(row["start"])
        end = _iso_timestamp(row["end"])
        requested = _mps_value(row["requested_tres"])
        allocated = _mps_value(row["allocated_tres"])
        jobs.append({
            "job_id": row["job_id"],
            "job_name": row["job_name"],
            "state": state,
            "node": row["node"],
            "gpu": _gpu_from_node(row["node"], requested or allocated),
            "mps": {"requested": requested, "allocated": allocated},
            "submit_ts": submit,
            "start_ts": start,
            "end_ts": end,
            "jct_seconds": end - submit if submit is not None and end is not None else None,
            "wait_seconds": start - submit if submit is not None and start is not None else None,
            "runtime_seconds": end - start if start is not None and end is not None else None,
            "elapsed_seconds": _number(row["elapsed"]),
            "partition": row["partition"],
            "command": row["command"],
        })
    jobs.sort(key=lambda job: float(job["end_ts"] or 0), reverse=True)
    return jobs[:limit]


def _parse_pmon(text: str) -> list[dict[str, object]]:
    processes = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        columns = line.split()
        if len(columns) < 10 or not columns[1].isdigit() or "C" not in columns[2]:
            continue
        processes.append({
            "gpu_index": _int(columns[0], -1),
            "pid": int(columns[1]),
            "process_type": columns[2],
            "sm_percent": _number(columns[3]),
            "vram_used_mib": _number(columns[9]),
            "command": " ".join(columns[11:]) if len(columns) > 11 else "",
        })
    return processes


def _aggregate_job_usage(
    processes: list[dict[str, object]],
    pid_jobs: dict[int, str],
    gpu_metrics: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    gpus = {int(item["gpu_index"]): item for item in gpu_metrics if "gpu_index" in item}
    shared_sm = {
        int(process["gpu_index"]): _number(process.get("sm_percent"))
        for process in processes
        if str(process.get("command", "")).startswith("nvidia-cuda-mps")
        and _number(process.get("sm_percent")) is not None
    }
    usage: dict[str, dict[str, object]] = {}
    for process in processes:
        job_id = pid_jobs.get(int(process["pid"]))
        if not job_id:
            continue
        gpu = gpus.get(int(process["gpu_index"]), {})
        item = usage.setdefault(job_id, {
            "sm_percent": None,
            "vram_used_mib": None,
            "gpu_uuid": gpu.get("gpu_uuid", ""),
            "pids": [],
        })
        sm = _number(process.get("sm_percent"))
        vram = _number(process.get("vram_used_mib"))
        if sm is not None:
            item["sm_percent"] = min(100.0, float(item["sm_percent"] or 0) + sm)
        if vram is not None:
            item["vram_used_mib"] = float(item["vram_used_mib"] or 0) + vram
        item["pids"].append(process["pid"])
        total = _number(gpu.get("memory_total_mib"))
        if total and item["vram_used_mib"] is not None:
            item["vram_percent"] = float(item["vram_used_mib"]) / total * 100.0
        if item["sm_percent"] is None and int(process["gpu_index"]) in shared_sm:
            item["shared_gpu_sm_percent"] = shared_sm[int(process["gpu_index"])]
    return usage


def _parse_key_values(line: str) -> dict[str, str]:
    return {key: value for key, value in re.findall(r"(\w+)=([^\s]+)", line)}


def _gpu_type_from_gres(gres: str) -> str:
    match = re.search(r"gpu:([^:,\s]+)", gres, re.I)
    return match.group(1) if match else "GPU"


def _gpu_count_from_gres(gres: str) -> int:
    match = re.search(r"gpu:[^:,\s]+:(\d+)", gres, re.I)
    return int(match.group(1)) if match else 1


def _parse_prometheus(text: str, name: str, default: float | None = None) -> float | None:
    pattern = re.compile(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+({_NUMBER_RE})$")
    values = []
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            values.append(float(match.group(1)))
    return values[-1] if values else default


class LiveCollectionError(RuntimeError):
    """Raised when the live Slurm source cannot be queried."""


class LiveCollector:
    """Read the existing Slurm, worker, and scheduler endpoints for the GUI."""

    def __init__(
        self,
        *,
        kubeconfig: str | None = None,
        namespace: str = "slurm",
        controller: str = "slurm-controller-0",
        scheduler: str = "deploy/rl-scheduler",
        exporter: str = "deploy/slurm-exporter",
        timeout: float = 8.0,
    ) -> None:
        self.kubeconfig = kubeconfig or os.environ.get("KUBECONFIG")
        self.namespace = namespace
        self.controller = controller
        self.scheduler = scheduler
        self.exporter = exporter
        self.timeout = timeout

    def _kubectl(self, *args: str) -> str:
        command = ["kubectl"]
        if self.kubeconfig:
            command.extend(["--kubeconfig", self.kubeconfig])
        command.extend(args)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LiveCollectionError(f"kubectl failed: {exc}") from exc
        if result.returncode:
            detail = result.stderr.strip() or f"exit {result.returncode}"
            raise LiveCollectionError(detail)
        return result.stdout

    def _exec(self, target: str, *command: str) -> str:
        return self._kubectl("-n", self.namespace, "exec", target, "--", *command)

    def _prometheus(self, target: str, port: int) -> str:
        script = (
            "import urllib.request; "
            f"print(urllib.request.urlopen('http://127.0.0.1:{port}/metrics', timeout=3).read().decode())"
        )
        return self._exec(target, "python3", "-c", script)

    def _gpu_metrics(self, node: dict[str, str]) -> list[dict[str, object]]:
        name = node.get("NodeName", "")
        gpu_type = _gpu_type_from_gres(node.get("Gres", ""))
        count = _gpu_count_from_gres(node.get("Gres", ""))
        query = (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,utilization.memory,memory.used,"
            "memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        )
        try:
            output = self._exec(name, *query)
        except LiveCollectionError as exc:
            return [
                {
                    "node": name,
                    "gpu_type": gpu_type,
                    "gpu_index": index,
                    "available": False,
                    "error": str(exc),
                }
                for index in range(count)
            ]

        metrics: list[dict[str, object]] = []
        for row in csv.reader(output.splitlines(), skipinitialspace=True):
            if len(row) < 9:
                continue
            used = _number(row[5])
            total = _number(row[6])
            vram = (used / total * 100.0) if used is not None and total else None
            metrics.append(
                {
                    "node": name,
                    "gpu_type": gpu_type,
                    "gpu_index": _int(row[0]),
                    "gpu_uuid": row[1].strip(),
                    "available": True,
                    "gpu_util_percent": _number(row[3]),
                    "memory_util_percent": _number(row[4]),
                    "memory_used_mib": used,
                    "memory_total_mib": total,
                    "vram_percent": vram,
                    "power_w": _number(row[7]),
                    "temperature_c": _number(row[8]),
                }
            )
        return metrics

    def _job_gpu_usage(
        self, node_name: str, gpu_metrics: list[dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        try:
            processes = _parse_pmon(self._exec(node_name, "nvidia-smi", "pmon", "-c", "1", "-s", "um"))
            if not processes:
                return {}
            pids = [str(process["pid"]) for process in processes]
            mapped = self._exec(node_name, "sh", "-c", _PID_JOB_SCRIPT, "sh", *pids)
        except LiveCollectionError:
            return {}
        pid_jobs = {}
        for line in mapped.splitlines():
            pid, separator, job_id = line.partition("|")
            if separator and pid.isdigit() and job_id.isdigit():
                pid_jobs[int(pid)] = job_id
        return _aggregate_job_usage(processes, pid_jobs, gpu_metrics)

    def collect(self) -> dict[str, object]:
        try:
            queue = json.loads(self._exec(self.controller, "squeue", "--json"))
            node_text = self._exec(self.controller, "scontrol", "show", "node", "--oneliner")
            history_text = self._exec(
                self.controller,
                "sacct", "-X", "-P", "-n", "-S", "now-1day",
                "-o", "JobIDRaw,JobName,State,Submit,Start,End,ElapsedRaw,NodeList,AllocTRES,ReqTRES,Partition,SubmitLine",
            )
        except (json.JSONDecodeError, LiveCollectionError) as exc:
            raise LiveCollectionError(f"Slurm query failed: {exc}") from exc

        jobs = [
            job
            for job in (_normalise_live_job(item) for item in queue.get("jobs", []))
            if str(job["state"]).upper() in _ACTIVE_STATES
        ]
        nodes = [_parse_key_values(line) for line in node_text.splitlines() if line.strip()]
        nodes = [node for node in nodes if node.get("NodeName")]
        gpu_metrics: list[dict[str, object]] = []
        usage_by_job: dict[str, dict[str, object]] = {}
        node_views: list[dict[str, object]] = []
        for node in nodes:
            capacity = _mps_value(node.get("CfgTRES"), node.get("Gres")) or 0
            allocated = _mps_value(node.get("AllocTRES")) or 0
            node_gpu_metrics = []
            if "gpu:" in node.get("Gres", ""):
                node_gpu_metrics = self._gpu_metrics(node)
                gpu_metrics.extend(node_gpu_metrics)
                usage_by_job.update(self._job_gpu_usage(node["NodeName"], node_gpu_metrics))
            node_views.append(
                {
                    "name": node.get("NodeName"),
                    "state": node.get("State", "UNKNOWN"),
                    "partitions": node.get("Partitions", ""),
                    "mps_allocated": allocated,
                    "mps_capacity": capacity,
                    "free_mps": max(capacity - allocated, 0),
                    "gpu_metrics": node_gpu_metrics,
                }
            )

        for job in jobs:
            job["resource_usage"] = usage_by_job.get(str(job["job_id"]))

        scheduler_text = ""
        exporter_text = ""
        try:
            scheduler_text = self._prometheus(self.scheduler, 8002)
        except LiveCollectionError:
            pass
        try:
            exporter_text = self._prometheus(self.exporter, 9341)
        except LiveCollectionError:
            pass

        scheduler = {
            "ready": _parse_prometheus(scheduler_text, "rl_scheduler_ready"),
            "shadow_mode": _parse_prometheus(scheduler_text, "rl_scheduler_shadow_mode"),
            "snapshot_age_s": _parse_prometheus(scheduler_text, "rl_scheduler_snapshot_age_seconds"),
            "pending_jobs": _parse_prometheus(scheduler_text, "rl_scheduler_snapshot_pending_jobs"),
            "free_mps": _parse_prometheus(scheduler_text, "rl_scheduler_snapshot_free_mps"),
            "priority_boost": _parse_prometheus(scheduler_text, "rl_scheduler_last_priority_boost"),
            "policy_value": _parse_prometheus(scheduler_text, "rl_scheduler_policy_value"),
            "policy_entropy": _parse_prometheus(scheduler_text, "rl_scheduler_policy_entropy"),
            "last_job_index": _parse_prometheus(scheduler_text, "rl_scheduler_last_job_index"),
            "last_node_index": _parse_prometheus(scheduler_text, "rl_scheduler_last_node_index"),
            "last_gpu_index": _parse_prometheus(scheduler_text, "rl_scheduler_last_gpu_index"),
        }
        queue_metrics = {
            "oldest_wait_s": _parse_prometheus(exporter_text, "slurm_job_queue_oldest_wait_seconds"),
            "average_wait_s": _parse_prometheus(exporter_text, "slurm_job_queue_avg_wait_seconds"),
            "scheduler_cycle_s": _parse_prometheus(exporter_text, "slurm_scheduler_cycle_last_seconds"),
            "backfill_queue": _parse_prometheus(exporter_text, "slurm_backfill_queue_length"),
        }
        now = datetime.now(timezone.utc)
        return {
            "updated_at": now.isoformat(),
            "source": "live-slurm-kubernetes",
            "jobs": jobs,
            "history": _parse_sacct_history(history_text),
            "nodes": node_views,
            "gpu_metrics": gpu_metrics,
            "scheduler": scheduler,
            "queue_metrics": queue_metrics,
        }


class Handler(SimpleHTTPRequestHandler):
    payload_provider: Callable[[], dict[str, object]] = LiveCollector().collect

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if urlsplit(self.path).path.rstrip("/") == "/api/jobs":
            try:
                payload = self.payload_provider()
                status = 200
            except LiveCollectionError as exc:
                payload = {"error": str(exc), "jobs": [], "source": "live-slurm-kubernetes"}
                status = 503
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, format: str, *args: object) -> None:
        print(f"[gui] {self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Kelpflux GUI with live telemetry")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--kubeconfig", default=None)
    parser.add_argument("--namespace", default="slurm")
    parser.add_argument("--controller", default="slurm-controller-0")
    args = parser.parse_args()

    collector = LiveCollector(
        kubeconfig=args.kubeconfig,
        namespace=args.namespace,
        controller=args.controller,
    )
    Handler.payload_provider = collector.collect

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Kelpflux GUI listening at http://{args.host}:{args.port}/ (live Slurm/Kubernetes telemetry)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
