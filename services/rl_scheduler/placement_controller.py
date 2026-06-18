"""Slurm-safe DSAC hard placement controller (slurmrestd path).

This controller turns the DSAC ``/act`` placement intent into a Slurm-native
hold/release operation, talking to **slurmrestd over JWT** (the same auth path
``snapshot_agent`` already uses) — no ``squeue``/``scontrol`` CLI, no munge
socket, no ``kubectl exec``:

1. ``GET /slurm/<v>/jobs``  — find PENDING + held jobs (``JobHeld*`` reason);
2. ``GET /slurm/<v>/nodes`` — read per-node free MPS / availability;
3. ``POST`` rl-scheduler ``/act`` for ``(job_i, node_j, gpu_k)``;
4. ``POST /slurm/<v>/job/<id>`` with ``required_nodes`` + a release priority so
   Slurm starts the held job under the requested node constraint.

It deliberately does not run inside ``job_submit.lua``. Jobs must already be
submitted held with their GRES/MPS request, e.g.
``sbatch -H -p gpu-rtx4070 --gres=mps:25``. Slurm then enforces the MPS
allocation when the job starts.

Release semantics: ``scontrol release`` sends ``priority=INFINITE`` so the
multifactor plugin recomputes the job's priority and lifts the user hold. We
mirror that with ``RELEASE_PRIORITY`` (default ``INFINITE`` = 0xFFFFFFFF);
override via ``--release-priority`` / ``RELEASE_PRIORITY`` if a deployment's
slurmrestd version expects a different sentinel.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence

from services.rl_scheduler.snapshot_agent import (
    _gpu_type_from_text,
    _node_tres_text,
    _number,
    _parse_tres_int,
    _read_jwt_key,
    _state_tokens,
    http_json,
)

LOG = logging.getLogger("rl_placement_controller")
TOP_K = 16

# Slurm INFINITE sentinel — `scontrol release` sets priority=INFINITE, which
# tells slurmctld to recompute the priority and drop the hold.
RELEASE_PRIORITY = 0xFFFFFFFF

# Node config TRES keys that may carry the gpu/mps counts across slurmrestd
# versions (mirrors snapshot_agent.node_view).
_NODE_CFG_KEYS = ("tres", "cfg_tres", "gres", "gres_detail")
_NODE_ALLOC_KEYS = ("alloc_tres", "alloc_gres")


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


def _slurm_base(rest_url: str, api_version: str) -> str:
    return f"{rest_url.rstrip('/')}/slurm/{api_version}"


# ── Parse slurmrestd docs ──────────────────────────────────────────────────

def parse_jobs(
    jobs_doc: dict[str, Any],
    *,
    default_runtime: float,
    default_mps: int,
    now: float | None = None,
) -> list[SlurmJob]:
    now = time.time() if now is None else now
    jobs: list[SlurmJob] = []
    for item in jobs_doc.get("jobs", []):
        if "PENDING" not in _state_tokens(item.get("job_state")):
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
                state="PENDING",
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
    return [job for job in jobs if "JOBHELD" in job.reason.upper().replace("_", "")]


def parse_nodes(nodes_doc: dict[str, Any], *, mps_per_gpu: int) -> list[SlurmNode]:
    nodes: list[SlurmNode] = []
    for node in nodes_doc.get("nodes", []):
        name = str(node.get("name") or node.get("node_name") or node.get("hostname") or "")
        cfg = _node_tres_text(node, _NODE_CFG_KEYS)
        # Skip non-GPU nodes (cpu workers) — they can never host an MPS job.
        if not re.search(r"(gpu|mps)", cfg + "," + name, flags=re.IGNORECASE):
            continue
        alloc = _node_tres_text(node, _NODE_ALLOC_KEYS)
        states = _state_tokens(node.get("state") or node.get("state_flags"))
        total_mps = _parse_tres_int(cfg, ("mps",), default=0) or mps_per_gpu
        alloc_mps = _parse_tres_int(alloc, ("mps",), default=0)
        running = 1 if states & {"ALLOCATED", "MIXED", "COMPLETING"} else 0
        unavailable = bool(states & {"DOWN", "DRAIN", "NOT_RESPONDING", "FAIL"})
        nodes.append(
            SlurmNode(
                name=name,
                free_mps=max(0, total_mps - alloc_mps),
                running_jobs=running,
                gpu_type=_gpu_type_from_text(cfg + "," + name),
                available=not unavailable,
            )
        )
    return nodes


# ── rl-scheduler /act + /healthz ───────────────────────────────────────────

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


def post_act(payload: dict[str, Any], *, scheduler_url: str, timeout: float = 10.0) -> dict[str, Any]:
    return http_json("POST", scheduler_url.rstrip("/") + "/act", body=payload, timeout=timeout)


def get_scheduler_healthz(*, scheduler_url: str, timeout: float = 10.0) -> dict[str, Any]:
    return http_json("GET", scheduler_url.rstrip("/") + "/healthz", timeout=timeout)


def trim_nodes_to_model_topology(
    node_names: Sequence[str],
    *,
    scheduler_url: str,
) -> list[str]:
    health = get_scheduler_healthz(scheduler_url=scheduler_url)
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


# ── Apply (slurmrestd job update) ──────────────────────────────────────────

def apply_hard_placement(
    *,
    rest_base: str,
    jwt_key: bytes | None,
    job_id: str,
    node_name: str,
    release: bool,
    release_priority: int = RELEASE_PRIORITY,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """POST a job update: pin ``required_nodes`` and (optionally) release the hold."""
    body: dict[str, Any] = {"required_nodes": node_name}
    if release:
        body["priority"] = int(release_priority)
    return http_json("POST", f"{rest_base}/job/{job_id}", jwt_key=jwt_key, body=body, timeout=timeout)


# ── Orchestration ──────────────────────────────────────────────────────────

def choose_and_apply(
    *,
    rest_url: str,
    api_version: str,
    scheduler_url: str,
    jwt_key: bytes | None,
    node_names: Sequence[str] | None = None,
    node_name_prefix: str = "",
    job_id: str | None = None,
    job_name_prefix: str = "",
    mps_per_gpu: int = 100,
    default_mps: int = 100,
    default_runtime: float = 600.0,
    release: bool = True,
    shadow: bool = True,
    release_priority: int = RELEASE_PRIORITY,
    auto_trim_model_topology: bool = True,
    require_held: bool = True,
    timeout: float = 10.0,
) -> PlacementDecision:
    base = _slurm_base(rest_url, api_version)

    jobs = parse_jobs(
        http_json("GET", f"{base}/jobs", jwt_key=jwt_key, timeout=timeout),
        default_runtime=default_runtime,
        default_mps=default_mps,
    )
    if require_held:
        jobs = filter_held_jobs(jobs)
    if job_id:
        jobs = [job for job in jobs if job.job_id == job_id]
    if job_name_prefix:
        jobs = [job for job in jobs if job.name.startswith(job_name_prefix)]
    if not jobs:
        return PlacementDecision(None, None, None, -1, 0.0, 0.0, False, "no_matching_pending_jobs")

    discovered = parse_nodes(
        http_json("GET", f"{base}/nodes", jwt_key=jwt_key, timeout=timeout),
        mps_per_gpu=mps_per_gpu,
    )
    available = [n for n in discovered if n.available and n.name]
    if node_names:
        allow = set(node_names)
        available = [n for n in available if n.name in allow]
    if node_name_prefix:
        available = [n for n in available if n.name.startswith(node_name_prefix)]
    if not available:
        return PlacementDecision(None, None, None, -1, 0.0, 0.0, False, "no_available_nodes")

    available_names = [n.name for n in available]
    effective_names = (
        trim_nodes_to_model_topology(available_names, scheduler_url=scheduler_url)
        if auto_trim_model_topology
        else available_names
    )
    nodes_by_name = {n.name: n for n in available}
    nodes = [nodes_by_name[name] for name in effective_names]

    act = post_act(build_act_payload(jobs, nodes, mps_per_gpu=mps_per_gpu), scheduler_url=scheduler_url)

    selected_id = act.get("selected_job_id")
    node_index = act.get("node_j")
    gpu_index = act.get("gpu_k")
    if selected_id is None or node_index is None or gpu_index is None:
        return PlacementDecision(
            None, None, None,
            int(act.get("action", -1)), float(act.get("value", 0.0)), float(act.get("entropy", 0.0)),
            False, "dsac_no_op",
        )
    if int(node_index) < 0 or int(node_index) >= len(effective_names):
        return PlacementDecision(
            str(selected_id), None, int(gpu_index),
            int(act.get("action", -1)), float(act.get("value", 0.0)), float(act.get("entropy", 0.0)),
            False, f"node_index_out_of_range:{node_index}",
        )

    node_name = effective_names[int(node_index)]
    if not shadow:
        apply_hard_placement(
            rest_base=base, jwt_key=jwt_key, job_id=str(selected_id),
            node_name=node_name, release=release, release_priority=release_priority, timeout=timeout,
        )

    return PlacementDecision(
        str(selected_id), node_name, int(gpu_index),
        int(act.get("action", -1)), float(act.get("value", 0.0)), float(act.get("entropy", 0.0)),
        not shadow, "applied" if not shadow else "shadow",
    )


def drain_and_apply(*, max_placements: int = 256, **kwargs) -> list[PlacementDecision]:
    """Place EVERY currently-placeable held job in a single poll cycle.

    ``choose_and_apply`` commits one job per call (the model's top pick over the
    held set). Calling it once per poll means a burst of N held jobs needs N
    intervals to drain — which would unfairly throttle the RL arm of the 2-node
    placement A/B against the unheld ``score`` arm. This loops until a cycle
    applies nothing (held set drained, shadow, or the model no-ops), capped at
    ``max_placements`` so a stuck job can't spin forever.

    Returns the per-iteration decisions (the trailing one is non-applied).
    """
    decisions: list[PlacementDecision] = []
    for _ in range(max(1, max_placements)):
        decision = choose_and_apply(**kwargs)
        decisions.append(decision)
        if not decision.applied:
            break
    return decisions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply DSAC hard placement via slurmrestd hold/release.")
    parser.add_argument("--rest-url", default=os.getenv("SLURM_REST_URL", "http://slurm-restapi.slurm.svc.cluster.local:6820"))
    parser.add_argument("--api-version", default=os.getenv("SLURM_REST_API_VERSION", "v0.0.37"))
    parser.add_argument("--scheduler-url", default=os.getenv("RL_SCHEDULER_URL", "http://rl-scheduler:8002"))
    parser.add_argument("--jwt-key-path", default=os.getenv("SLURM_JWT_KEY_PATH", ""))
    parser.add_argument("--node-name", action="append", default=None, help="Restrict to these Slurm GPU worker node names; repeat for multiple. Omit to auto-discover from /nodes.")
    parser.add_argument("--node-name-prefix", default=os.getenv("NODE_NAME_PREFIX", ""))
    parser.add_argument("--job-id", default="")
    parser.add_argument("--job-name-prefix", default=os.getenv("JOB_NAME_PREFIX", ""))
    parser.add_argument("--mps-per-gpu", type=int, default=int(os.getenv("MPS_PER_GPU", "100")))
    parser.add_argument("--default-mps", type=int, default=int(os.getenv("DEFAULT_MPS", "100")))
    parser.add_argument("--default-runtime", type=float, default=float(os.getenv("DEFAULT_RUNTIME_SECONDS", "600")))
    parser.add_argument("--interval", type=float, default=float(os.getenv("PLACEMENT_INTERVAL_SECONDS", "5")))
    parser.add_argument("--release-priority", type=int, default=int(os.getenv("RELEASE_PRIORITY", str(RELEASE_PRIORITY))))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--release", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shadow", action=argparse.BooleanOptionalAction, default=_env_bool("PLACEMENT_SHADOW", False))
    parser.add_argument("--auto-trim-model-topology", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-held", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drain", action=argparse.BooleanOptionalAction,
                        default=_env_bool("PLACEMENT_DRAIN", True),
                        help="place every placeable held job each cycle (default) "
                             "instead of one-per-poll; --no-drain restores the "
                             "legacy single-placement behaviour")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")
    jwt_key = _read_jwt_key(args.jwt_key_path)

    cycle_kwargs = dict(
        rest_url=args.rest_url,
        api_version=args.api_version,
        scheduler_url=args.scheduler_url,
        jwt_key=jwt_key,
        node_names=args.node_name,
        node_name_prefix=args.node_name_prefix,
        job_id=args.job_id or None,
        job_name_prefix=args.job_name_prefix,
        mps_per_gpu=args.mps_per_gpu,
        default_mps=args.default_mps,
        default_runtime=args.default_runtime,
        release=args.release,
        shadow=args.shadow,
        release_priority=args.release_priority,
        auto_trim_model_topology=args.auto_trim_model_topology,
        require_held=args.require_held,
    )

    while True:
        try:
            if args.drain:
                decisions = drain_and_apply(**cycle_kwargs)
                applied = sum(1 for d in decisions if d.applied)
                decision = decisions[-1]
                LOG.info(
                    "cycle drained: %d placed; last job=%s node=%s applied=%s reason=%s",
                    applied, decision.job_id, decision.node_name, decision.applied, decision.reason,
                )
            else:
                decision = choose_and_apply(**cycle_kwargs)
                LOG.info(
                    "decision job=%s node=%s gpu=%s action=%s value=%.3f entropy=%.3f applied=%s reason=%s",
                    decision.job_id, decision.node_name, decision.gpu_index, decision.action,
                    decision.value, decision.entropy, decision.applied, decision.reason,
                )
        except Exception as exc:  # pragma: no cover - exercised in cluster
            LOG.warning("placement cycle failed: %s", exc)
            decision = None

        if args.once:
            if decision is None:
                return 1
            return 0 if decision.reason not in {"no_matching_pending_jobs", "dsac_no_op"} else 2
        time.sleep(max(1.0, args.interval))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


if __name__ == "__main__":
    raise SystemExit(main())
