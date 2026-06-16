"""Slurm-safe DSAC hard placement controller.

This controller turns the DSAC `/act` placement intent into a Slurm-native
hold/release operation:

1. inspect held pending jobs with `squeue --json`;
2. inspect GPU worker state with `scontrol show node`;
3. call rl-scheduler `/act` for `(job_i, node_j, gpu_k)`;
4. set `ReqNodeList=<selected-node>` on the selected held job;
5. release the job so Slurm starts it under the requested node constraint.

It deliberately does not run inside `job_submit.lua`. Jobs must already carry
their GRES/MPS request, for example `sbatch -H -p gpu-rtx4070 --gres=mps:25`.
Slurm then enforces the MPS allocation when the job starts.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Sequence


LOG = logging.getLogger("rl_placement_controller")
GPU_TYPES = ("rtx4070", "rtx3080", "generic")
TOP_K = 16


@dataclass(frozen=True)
class SlurmJob:
    job_id: str
    name: str
    state: str
    reason: str
    mps_req: int
    gpu_count: int
    gpu_type: str
    runtime: float
    submit_ts: float


@dataclass(frozen=True)
class SlurmNode:
    name: str
    free_mps: int
    running_jobs: int
    gpu_type: str
    available: bool = True


@dataclass(frozen=True)
class PlacementDecision:
    job_id: str | None
    node_name: str | None
    gpu_index: int | None
    action: int
    value: float
    entropy: float
    applied: bool
    reason: str


class CommandRunner:
    def __init__(self, prefix: str = "", timeout: float = 15.0):
        self.prefix = shlex.split(prefix) if prefix else []
        self.timeout = timeout

    def run(
        self,
        args: Sequence[str],
        *,
        input_text: str | None = None,
        timeout: float | None = None,
    ) -> str:
        cmd = [*self.prefix, *args]
        proc = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout or self.timeout,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            stdout = proc.stdout.strip()
            raise RuntimeError(
                f"command failed ({proc.returncode}): {' '.join(cmd)}"
                f"{' stderr=' + stderr if stderr else ''}"
                f"{' stdout=' + stdout if stdout else ''}"
            )
        return proc.stdout.strip()


def _state_text(raw: Any) -> str:
    if isinstance(raw, list):
        return " ".join(str(x) for x in raw)
    return str(raw or "")


def _number(raw: Any, default: float = 0.0) -> float:
    if isinstance(raw, dict):
        raw = raw.get("number", default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _parse_tres_int(text: str, names: tuple[str, ...], default: int = 0) -> int:
    if not text:
        return default
    for name in names:
        patterns = (
            rf"(?:^|,){re.escape(name)}=(\d+)",
            rf"(?:^|,){re.escape(name)}:(?:[^,:]+:)?(\d+)",
            rf"(?:^|,)gres/{re.escape(name)}=(\d+)",
        )
        for pattern in patterns:
            found = re.search(pattern, text, flags=re.IGNORECASE)
            if found:
                return int(found.group(1))
    return default


def _gpu_type_from_text(text: str) -> str:
    lower = text.lower()
    for gpu_type in GPU_TYPES:
        if gpu_type in lower:
            return gpu_type
    return "rtx4070"


def parse_squeue_jobs(raw: str, *, default_runtime: float, default_mps: int) -> list[SlurmJob]:
    data = json.loads(raw or "{}")
    jobs: list[SlurmJob] = []
    now = time.time()
    for item in data.get("jobs", []):
        state = _state_text(item.get("job_state")).upper()
        if "PENDING" not in state:
            continue
        tres = ",".join(str(item.get(k, "") or "") for k in ("tres_req_str", "tres_per_node", "gres"))
        mps_req = _parse_tres_int(tres, ("mps",), default=0) or default_mps
        gpu_count = _parse_tres_int(tres, ("gpu",), default=0) or int(_number(item.get("gpus_total"), 1)) or 1
        runtime = _number(item.get("time_limit"), 0.0) * 60.0
        if runtime <= 0:
            runtime = default_runtime
        jobs.append(
            SlurmJob(
                job_id=str(item.get("job_id", "")),
                name=str(item.get("name") or item.get("job_name") or ""),
                state=state,
                reason=str(item.get("state_reason") or item.get("reason") or ""),
                mps_req=int(mps_req),
                gpu_count=int(gpu_count),
                gpu_type=_gpu_type_from_text(tres),
                runtime=float(runtime),
                submit_ts=_number(item.get("submit_time"), now),
            )
        )
    return jobs


def filter_held_jobs(jobs: Sequence[SlurmJob]) -> list[SlurmJob]:
    return [job for job in jobs if "JOBHELD" in job.reason.upper() or "JOB_HELD" in job.reason.upper()]


def parse_scontrol_node(raw: str, node_name: str, *, mps_per_gpu: int) -> SlurmNode:
    cfg = ""
    alloc = ""
    state = ""
    for token in raw.replace("\n", " ").split():
        if token.startswith("CfgTRES=") or token.startswith("Gres="):
            cfg += "," + token.split("=", 1)[1]
        elif token.startswith("AllocTRES="):
            alloc += "," + token.split("=", 1)[1]
        elif token.startswith("State="):
            state = token.split("=", 1)[1].upper()

    total_mps = _parse_tres_int(cfg, ("mps",), default=0) or mps_per_gpu
    alloc_mps = _parse_tres_int(alloc, ("mps",), default=0)
    running = 1 if any(flag in state for flag in ("ALLOCATED", "MIXED", "COMPLETING")) else 0
    unavailable = any(flag in state for flag in ("DOWN", "DRAIN", "NOT_RESPONDING", "FAIL"))
    return SlurmNode(
        name=node_name,
        free_mps=max(0, total_mps - alloc_mps),
        running_jobs=running,
        gpu_type=_gpu_type_from_text(cfg + "," + node_name),
        available=not unavailable,
    )


def build_act_payload(
    jobs: Sequence[SlurmJob],
    nodes: Sequence[SlurmNode],
    *,
    mps_per_gpu: int,
    now: float | None = None,
) -> dict[str, Any]:
    ts = time.time() if now is None else now
    return {
        "now": ts,
        "pending_jobs": [
            {
                "job_id": j.job_id,
                "mps_req": j.mps_req,
                "gpu_count": j.gpu_count,
                "gpu_type": j.gpu_type,
                "runtime": j.runtime,
                "submit_ts": j.submit_ts,
                "can_fit": any(n.free_mps >= j.mps_req for n in nodes),
            }
            for j in sorted(jobs, key=lambda item: item.submit_ts)[:TOP_K]
        ],
        "nodes": [
            {"gpus": [{"free_mps": n.free_mps, "running_jobs": n.running_jobs, "gpu_type": n.gpu_type}]}
            for n in nodes
        ],
        "n_nodes": len(nodes),
        "gpus_per_node": 1,
        "mps_per_gpu": mps_per_gpu,
    }


def post_act(
    payload: dict[str, Any],
    *,
    scheduler_url: str,
    scheduler_exec_prefix: str = "",
    timeout: float = 10.0,
) -> dict[str, Any]:
    body = json.dumps(payload)
    if scheduler_exec_prefix:
        runner = CommandRunner(scheduler_exec_prefix, timeout=timeout)
        raw = runner.run(
            [
                "curl",
                "-fsS",
                "-X",
                "POST",
                "-H",
                "Content-Type: application/json",
                "--data-binary",
                body,
                scheduler_url.rstrip("/") + "/act",
            ],
            timeout=timeout,
        )
    else:
        req = urllib.request.Request(
            scheduler_url.rstrip("/") + "/act",
            data=body.encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
    return json.loads(raw)


def get_scheduler_healthz(
    *,
    scheduler_url: str,
    scheduler_exec_prefix: str = "",
    timeout: float = 10.0,
) -> dict[str, Any]:
    if scheduler_exec_prefix:
        runner = CommandRunner(scheduler_exec_prefix, timeout=timeout)
        raw = runner.run(["curl", "-fsS", scheduler_url.rstrip("/") + "/healthz"], timeout=timeout)
    else:
        with urllib.request.urlopen(scheduler_url.rstrip("/") + "/healthz", timeout=timeout) as resp:
            raw = resp.read().decode()
    return json.loads(raw)


def trim_nodes_to_model_topology(
    node_names: Sequence[str],
    *,
    scheduler_url: str,
    scheduler_exec_prefix: str = "",
) -> list[str]:
    health = get_scheduler_healthz(scheduler_url=scheduler_url, scheduler_exec_prefix=scheduler_exec_prefix)
    n_actions = health.get("n_actions")
    if not isinstance(n_actions, int) or n_actions <= 1:
        return list(node_names)
    supported_placements = max(1, (n_actions - 1) // TOP_K)
    if supported_placements >= len(node_names):
        return list(node_names)
    kept = list(node_names[:supported_placements])
    LOG.warning(
        "trimming node list from %d to %d to match DSAC model n_actions=%s; retrain/deploy a larger topology checkpoint for multi-node placement",
        len(node_names),
        len(kept),
        n_actions,
    )
    return kept


def read_jobs(runner: CommandRunner, *, default_runtime: float, default_mps: int) -> list[SlurmJob]:
    return parse_squeue_jobs(
        runner.run(["squeue", "--json"]),
        default_runtime=default_runtime,
        default_mps=default_mps,
    )


def read_nodes(runner: CommandRunner, node_names: Sequence[str], *, mps_per_gpu: int) -> list[SlurmNode]:
    nodes: list[SlurmNode] = []
    for node_name in node_names:
        nodes.append(parse_scontrol_node(runner.run(["scontrol", "show", "node", node_name]), node_name, mps_per_gpu=mps_per_gpu))
    return nodes


def apply_hard_placement(
    runner: CommandRunner,
    *,
    job_id: str,
    node_name: str,
    release: bool,
) -> None:
    runner.run(["scontrol", "update", f"JobId={job_id}", f"ReqNodeList={node_name}"])
    if release:
        runner.run(["scontrol", "release", job_id])


def choose_and_apply(
    *,
    slurm_runner: CommandRunner,
    scheduler_url: str,
    node_names: Sequence[str],
    scheduler_exec_prefix: str = "",
    job_id: str | None = None,
    job_name_prefix: str = "",
    mps_per_gpu: int = 100,
    default_mps: int = 100,
    default_runtime: float = 600.0,
    release: bool = True,
    shadow: bool = True,
    auto_trim_model_topology: bool = True,
    require_held: bool = True,
) -> PlacementDecision:
    jobs = read_jobs(slurm_runner, default_runtime=default_runtime, default_mps=default_mps)
    if require_held:
        jobs = filter_held_jobs(jobs)
    if job_id:
        jobs = [job for job in jobs if job.job_id == job_id]
    if job_name_prefix:
        jobs = [job for job in jobs if job.name.startswith(job_name_prefix)]
    if not jobs:
        return PlacementDecision(None, None, None, -1, 0.0, 0.0, False, "no_matching_pending_jobs")

    discovered_nodes = read_nodes(slurm_runner, node_names, mps_per_gpu=mps_per_gpu)
    available_pairs = [(name, node) for name, node in zip(node_names, discovered_nodes) if node.available]
    if not available_pairs:
        return PlacementDecision(None, None, None, -1, 0.0, 0.0, False, "no_available_nodes")

    available_names = [name for name, _node in available_pairs]
    effective_nodes = (
        trim_nodes_to_model_topology(
            available_names,
            scheduler_url=scheduler_url,
            scheduler_exec_prefix=scheduler_exec_prefix,
        )
        if auto_trim_model_topology
        else available_names
    )
    nodes_by_name = {name: node for name, node in available_pairs}
    nodes = [nodes_by_name[name] for name in effective_nodes]
    payload = build_act_payload(jobs, nodes, mps_per_gpu=mps_per_gpu)
    act = post_act(
        payload,
        scheduler_url=scheduler_url,
        scheduler_exec_prefix=scheduler_exec_prefix,
    )

    selected_id = act.get("selected_job_id")
    node_index = act.get("node_j")
    gpu_index = act.get("gpu_k")
    if selected_id is None or node_index is None or gpu_index is None:
        return PlacementDecision(
            None,
            None,
            None,
            int(act.get("action", -1)),
            float(act.get("value", 0.0)),
            float(act.get("entropy", 0.0)),
            False,
            "dsac_no_op",
        )
    if int(node_index) < 0 or int(node_index) >= len(effective_nodes):
        return PlacementDecision(
            str(selected_id),
            None,
            int(gpu_index),
            int(act.get("action", -1)),
            float(act.get("value", 0.0)),
            float(act.get("entropy", 0.0)),
            False,
            f"node_index_out_of_range:{node_index}",
        )

    node_name = effective_nodes[int(node_index)]
    if not shadow:
        apply_hard_placement(slurm_runner, job_id=str(selected_id), node_name=node_name, release=release)

    return PlacementDecision(
        str(selected_id),
        node_name,
        int(gpu_index),
        int(act.get("action", -1)),
        float(act.get("value", 0.0)),
        float(act.get("entropy", 0.0)),
        not shadow,
        "applied" if not shadow else "shadow",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply DSAC hard placement via Slurm hold/release.")
    parser.add_argument("--scheduler-url", default=os.getenv("RL_SCHEDULER_URL", "http://rl-scheduler:8002"))
    parser.add_argument("--scheduler-exec-prefix", default=os.getenv("SCHEDULER_EXEC_PREFIX", ""))
    parser.add_argument("--slurm-exec-prefix", default=os.getenv("SLURM_EXEC_PREFIX", ""))
    parser.add_argument("--node-name", action="append", required=True, help="Slurm GPU worker node name; repeat for multiple nodes.")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--job-name-prefix", default="")
    parser.add_argument("--mps-per-gpu", type=int, default=int(os.getenv("MPS_PER_GPU", "100")))
    parser.add_argument("--default-mps", type=int, default=int(os.getenv("DEFAULT_MPS", "100")))
    parser.add_argument("--default-runtime", type=float, default=float(os.getenv("DEFAULT_RUNTIME_SECONDS", "600")))
    parser.add_argument("--interval", type=float, default=float(os.getenv("PLACEMENT_INTERVAL_SECONDS", "5")))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--release", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shadow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-trim-model-topology", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-held", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")
    runner = CommandRunner(args.slurm_exec_prefix)

    while True:
        decision = choose_and_apply(
            slurm_runner=runner,
            scheduler_url=args.scheduler_url,
            scheduler_exec_prefix=args.scheduler_exec_prefix,
            node_names=args.node_name,
            job_id=args.job_id or None,
            job_name_prefix=args.job_name_prefix,
            mps_per_gpu=args.mps_per_gpu,
            default_mps=args.default_mps,
            default_runtime=args.default_runtime,
            release=args.release,
            shadow=args.shadow,
            auto_trim_model_topology=args.auto_trim_model_topology,
            require_held=args.require_held,
        )
        LOG.info(
            "decision job=%s node=%s gpu=%s action=%s value=%.3f entropy=%.3f applied=%s reason=%s",
            decision.job_id,
            decision.node_name,
            decision.gpu_index,
            decision.action,
            decision.value,
            decision.entropy,
            decision.applied,
            decision.reason,
        )
        if args.once:
            return 0 if decision.reason not in {"no_matching_pending_jobs", "dsac_no_op"} else 2
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
