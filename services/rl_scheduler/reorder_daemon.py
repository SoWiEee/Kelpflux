#!/usr/bin/env python3
"""Continuously rank eligible pending GPU/MPS jobs for Slurm."""
from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time

from services.rl_scheduler.placement_controller import (
    RELEASE_PRIORITY,
    SlurmJob,
    SlurmNode,
    _slurm_base,
    build_act_payload,
    get_scheduler_healthz,
    parse_jobs,
    parse_nodes,
    post_act,
)
from services.rl_scheduler.snapshot_agent import (
    _number,
    _parse_tres_int,
    _read_bearer_token,
    _read_jwt_key,
    _state_tokens,
    http_json,
)
from sim.gym_env import TOP_K, env_dims

LOG = logging.getLogger("rl_reorder_daemon")
RELEASE_BASE, RELEASE_SPACING = 5_000_000, 10_000
MAX_PENDING_JOBS = 150
MIN_OWNED_PRIORITY = RELEASE_BASE - (MAX_PENDING_JOBS - 1) * RELEASE_SPACING
RANK_DEADLINE_SECONDS = 25.0


def _job_from_item(item, default_runtime):
    parsed = parse_jobs({"jobs": [item]}, default_runtime=default_runtime, default_mps=0)
    if not parsed:
        return None
    job = parsed[0]
    tres = ",".join(str(item.get(k, "") or "") for k in ("tres_req_str", "tres_per_node", "gres"))
    if _parse_tres_int(tres, ("mps",), default=0) <= 0:
        return None
    return job


def parse_eligible_jobs(
    jobs_doc, *, gpu_partitions, default_runtime=600.0, job_name_prefix="",
):
    """Accept only pending, unheld, single-GPU jobs with an explicit MPS request."""
    eligible = []
    for item in jobs_doc.get("jobs", []):
        if "PENDING" not in _state_tokens(item.get("job_state")):
            continue
        reason = str(item.get("state_reason") or item.get("reason") or "")
        if "JOBHELD" in reason.upper().replace("_", "") or _number(item.get("priority"), 1) == 0:
            continue
        partition = str(item.get("partition") or "")
        if partition not in gpu_partitions:
            continue
        job = _job_from_item(item, default_runtime)
        if job is None or job.gpu_count != 1:
            continue
        if job_name_prefix and not job.name.startswith(job_name_prefix):
            continue
        eligible.append(job)
    return sorted(eligible, key=lambda job: job.submit_ts)


def _pending_job_priorities(jobs_doc):
    return {
        str(item["job_id"]): int(_number(item.get("priority"), -1))
        for item in jobs_doc.get("jobs", [])
        if item.get("job_id") is not None
        and "PENDING" in _state_tokens(item.get("job_state"))
    }


def _coerce_jobs(pending):
    jobs = []
    for item in pending:
        if isinstance(item, SlurmJob):
            jobs.append(item)
        else:
            jid, mps, submit_ts = item
            jobs.append(SlurmJob(
                str(jid), "", "PENDING", "", int(mps), 1, "rtx4070", 0.0,
                float(submit_ts),
            ))
    return jobs


def _coerce_nodes(nodes, free):
    result = []
    for item in nodes:
        if isinstance(item, SlurmNode):
            result.append(SlurmNode(
                item.name, int(free.get(item.name, item.free_mps)), item.running_jobs,
                item.gpu_type, item.available,
            ))
        else:
            name = str(item)
            result.append(SlurmNode(
                name, int(free.get(name, 0)), 0,
                "rtx3080" if "3080" in name else "rtx4070",
            ))
    return result


def rank_pending(
    pending, free, *, mps_per_gpu, nodes, serve, post=post_act,
    strict=False, deadline_at=None,
):
    """Greedily query the rolling top-16 policy and return a dispatch order.

    Tuple jobs/nodes retain the evaluation interface. Production passes parsed
    SlurmJob/SlurmNode objects and uses strict mode so policy failures never
    become accidental priority writes.
    """
    remaining = {job.job_id: job for job in _coerce_jobs(pending)}
    localfree = dict(free)
    order = []
    while remaining:
        if deadline_at is not None and time.monotonic() >= deadline_at:
            raise TimeoutError("policy rank deadline exceeded")
        candidates = sorted(remaining.values(), key=lambda job: job.submit_ts)[:TOP_K]
        fnodes = _coerce_nodes(nodes, localfree)
        payload = build_act_payload(candidates, fnodes, mps_per_gpu=mps_per_gpu)
        eligible = {j["job_id"]: set(j["eligible_node_indices"])
                    for j in payload["pending_jobs"]}
        try:
            act = post(payload, scheduler_url=serve)
        except Exception:
            if strict:
                raise
            act = {}
        selected = str(act.get("selected_job_id") or "")
        node_j = act.get("node_j")
        pick = node = None
        if selected in remaining and node_j is not None:
            try:
                node_index = int(node_j)
                if 0 <= node_index < len(fnodes) and node_index in eligible.get(selected, set()):
                    candidate = fnodes[node_index]
                    if candidate.available and candidate.free_mps >= remaining[selected].mps_req:
                        pick, node = selected, candidate.name
            except (TypeError, ValueError):
                pass
        if pick is None:
            if strict:
                try:
                    is_noop = (
                        not selected
                        and int(act.get("action", -1)) == TOP_K * len(fnodes)
                    )
                except (TypeError, ValueError):
                    is_noop = False
                if not is_noop:
                    raise RuntimeError("policy returned no feasible job/node action")
                order.extend(
                    job.job_id
                    for job in sorted(remaining.values(), key=lambda item: item.submit_ts)
                )
                break
            for candidate_job in sorted(remaining.values(), key=lambda job: job.submit_ts):
                for node_index, candidate_node in enumerate(fnodes):
                    if (candidate_node.available
                            and node_index in eligible.get(candidate_job.job_id, set())
                            and localfree.get(candidate_node.name, 0) >= candidate_job.mps_req):
                        pick, node = candidate_job.job_id, candidate_node.name
                        break
                if pick:
                    break
        if pick is None:
            if strict:
                raise RuntimeError("no pending job fits the current GPU/MPS state")
            order.extend(job.job_id for job in sorted(remaining.values(), key=lambda j: j.submit_ts))
            break
        order.append(pick)
        localfree[node] -= remaining.pop(pick).mps_req
    return order


def reorder_step(get_state, set_priorities, *, mps_per_gpu, nodes, serve, post=post_act):
    pending, free = get_state()
    if not pending:
        return []
    order = rank_pending(pending, free, mps_per_gpu=mps_per_gpu, nodes=nodes,
                         serve=serve, post=post)
    set_priorities({jid: RELEASE_BASE - i * RELEASE_SPACING for i, jid in enumerate(order)})
    return order


def reorder_loop(get_state, set_priorities, is_done, *, interval, mps_per_gpu, nodes,
                 serve, post=post_act, on_error=None):
    cycles = 0
    while not is_done():
        try:
            reorder_step(get_state, set_priorities, mps_per_gpu=mps_per_gpu, nodes=nodes,
                         serve=serve, post=post)
        except Exception as exc:
            if on_error:
                on_error(exc)
        cycles += 1
        time.sleep(interval)
    return cycles


def _read_state(rest_base, jwt_key, *, node_names, gpu_partitions,
                mps_per_gpu, default_runtime, job_name_prefix, timeout):
    jobs_doc = http_json("GET", f"{rest_base}/jobs", jwt_key=jwt_key, timeout=timeout)
    nodes_doc = http_json("GET", f"{rest_base}/nodes", jwt_key=jwt_key, timeout=timeout)
    jobs = parse_eligible_jobs(
        jobs_doc, gpu_partitions=gpu_partitions, default_runtime=default_runtime,
        job_name_prefix=job_name_prefix,
    )
    discovered = {node.name: node for node in parse_nodes(nodes_doc, mps_per_gpu=mps_per_gpu)}
    nodes = []
    for name in node_names:
        node = discovered.get(name)
        if node is None:
            gpu_type = "rtx3080" if "3080" in name else "rtx4070"
            node = SlurmNode(name, 0, 0, gpu_type, available=False)
        elif not node.available:
            node = SlurmNode(node.name, 0, node.running_jobs, node.gpu_type, False)
        nodes.append(node)
    return jobs, nodes


def _state_signature(jobs, nodes):
    return (
        tuple((j.job_id, j.mps_req, j.gpu_count, j.partition, j.required_nodes,
               j.submit_ts, j.priority) for j in jobs),
        tuple((n.name, n.gpu_type, n.free_mps, n.available) for n in nodes),
    )


def _set_priority(rest_base, jwt_key, job_id, priority, timeout):
    result = http_json(
        "POST", f"{rest_base}/job/{job_id}", jwt_key=jwt_key,
        body={"priority": int(priority)}, timeout=timeout,
    )
    errors = result.get("errors") or []
    if errors:
        raise RuntimeError(f"Slurm rejected priority update for job {job_id}: {errors}")


def reset_owned_priorities(rest_base, jwt_key, *, timeout=2.0):
    jobs_doc = http_json("GET", f"{rest_base}/jobs", jwt_key=jwt_key, timeout=timeout)
    owned = [
        job_id for job_id, priority in _pending_job_priorities(jobs_doc).items()
        if MIN_OWNED_PRIORITY <= priority <= RELEASE_BASE
    ]
    for job_id in owned:
        _set_priority(rest_base, jwt_key, job_id, RELEASE_PRIORITY, timeout)
    if owned:
        verify_doc = http_json("GET", f"{rest_base}/jobs", jwt_key=jwt_key, timeout=timeout)
        pending = _pending_job_priorities(verify_doc)
        remaining = [
            job_id for job_id in owned
            if job_id in pending and (
                pending[job_id] <= 0
                or pending[job_id] == RELEASE_PRIORITY
                or MIN_OWNED_PRIORITY <= pending[job_id] <= RELEASE_BASE
            )
        ]
        if remaining:
            raise RuntimeError(f"could not restore Slurm priorities for jobs {remaining}")
    return owned


def _restore_cycle_priorities(rest_base, jwt_key, original_priorities):
    if not original_priorities:
        return
    jobs_doc = http_json("GET", f"{rest_base}/jobs", jwt_key=jwt_key, timeout=2.0)
    pending = _pending_job_priorities(jobs_doc)
    for job_id, priority in original_priorities.items():
        current = pending.get(job_id)
        if current is not None and current != priority:
            _set_priority(rest_base, jwt_key, job_id, priority, 2.0)
    verify_doc = http_json("GET", f"{rest_base}/jobs", jwt_key=jwt_key, timeout=2.0)
    pending = _pending_job_priorities(verify_doc)
    mismatches = [
        job_id for job_id, priority in original_priorities.items()
        if job_id in pending and pending[job_id] != priority
    ]
    if mismatches:
        raise RuntimeError(f"could not roll back Slurm priorities for jobs {mismatches}")


def reorder_cycle(
    *, rest_base, jwt_key, scheduler_url, node_names, gpu_partitions,
    mps_per_gpu=100, max_jobs=MAX_PENDING_JOBS, deadline_seconds=RANK_DEADLINE_SECONDS,
    default_runtime=600.0, job_name_prefix="", shadow=False, api_token=None,
):
    if not 1 <= max_jobs <= MAX_PENDING_JOBS:
        raise ValueError(f"max_jobs must be between 1 and {MAX_PENDING_JOBS}")
    deadline = time.monotonic() + deadline_seconds
    get_state = lambda: _read_state(
        rest_base, jwt_key, node_names=node_names, gpu_partitions=gpu_partitions,
        mps_per_gpu=mps_per_gpu, default_runtime=default_runtime,
        job_name_prefix=job_name_prefix, timeout=2.0,
    )
    jobs, nodes = get_state()
    if not jobs:
        return []
    if len(jobs) > max_jobs:
        raise RuntimeError(f"eligible queue has {len(jobs)} jobs; limit is {max_jobs}")
    health = get_scheduler_healthz(scheduler_url=scheduler_url, timeout=2.0)
    obs_dim, n_actions = env_dims(n_nodes=len(node_names), gpus_per_node=1)
    if health.get("obs_dim") != obs_dim or health.get("n_actions") != n_actions:
        raise RuntimeError(
            f"checkpoint topology mismatch: expected {obs_dim}/{n_actions}, "
            f"got {health.get('obs_dim')}/{health.get('n_actions')}"
        )
    free = {node.name: node.free_mps for node in nodes}
    order = rank_pending(
        jobs, free, mps_per_gpu=mps_per_gpu, nodes=nodes, serve=scheduler_url,
        strict=True, deadline_at=deadline,
        post=lambda payload, scheduler_url: post_act(
            payload, scheduler_url=scheduler_url,
            timeout=max(0.1, min(2.0, deadline - time.monotonic())),
            api_token=api_token,
        ),
    )
    if time.monotonic() >= deadline:
        raise TimeoutError("priority write deadline exceeded")
    current_jobs, current_nodes = get_state()
    if _state_signature(jobs, nodes) != _state_signature(current_jobs, current_nodes):
        raise RuntimeError("queue or GPU state changed while ranking; skipping this cycle")
    priorities = {jid: RELEASE_BASE - i * RELEASE_SPACING for i, jid in enumerate(order)}
    current = {job.job_id: job.priority for job in current_jobs}
    if shadow:
        LOG.info("shadow order: %s", order)
        return order
    written = []
    try:
        for job_id, priority in priorities.items():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("priority write deadline exceeded")
            if current.get(job_id) == priority:
                continue
            _set_priority(rest_base, jwt_key, job_id, priority, min(2.0, remaining))
            written.append(job_id)
        verify_doc = http_json("GET", f"{rest_base}/jobs", jwt_key=jwt_key,
                               timeout=max(0.1, min(2.0, deadline - time.monotonic())))
        verified = _pending_job_priorities(verify_doc)
        mismatches = [
            jid for jid, priority in priorities.items()
            if jid in verified and verified[jid] != priority
        ]
        if mismatches:
            raise RuntimeError(f"Slurm priority round-trip mismatch for jobs {mismatches}")
    except Exception:
        try:
            _restore_cycle_priorities(rest_base, jwt_key, current)
        except Exception:
            LOG.exception("could not roll back priorities from the failed cycle")
        try:
            reset_owned_priorities(rest_base, jwt_key)
        except Exception:
            LOG.exception("could not reset reserved priorities after the failed cycle")
        raise
    return order


def main(argv=None):
    parser = argparse.ArgumentParser(description="Continuously reorder unheld Slurm GPU/MPS jobs.")
    parser.add_argument("--rest-url", default=os.getenv(
        "SLURM_REST_URL", "http://slurm-restapi.slurm.svc.cluster.local:6820"))
    parser.add_argument("--api-version", default=os.getenv("SLURM_REST_API_VERSION", "v0.0.39"))
    parser.add_argument("--scheduler-url", default=os.getenv("RL_SCHEDULER_URL", "http://rl-scheduler:8002"))
    parser.add_argument("--jwt-key-path", default=os.getenv("SLURM_JWT_KEY_PATH", ""))
    parser.add_argument("--node-name", action="append", required=True)
    parser.add_argument("--gpu-partition", action="append", default=[])
    parser.add_argument("--mps-per-gpu", type=int, default=100)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--max-jobs", type=int, default=MAX_PENDING_JOBS)
    parser.add_argument("--deadline-seconds", type=float, default=RANK_DEADLINE_SECONDS)
    parser.add_argument("--job-name-prefix", default="")
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)
    if (len(args.node_name) != 2 or args.mps_per_gpu <= 0
            or not 1 <= args.max_jobs <= MAX_PENDING_JOBS):
        parser.error(
            f"the bundled checkpoint requires two GPU nodes and max-jobs in 1..{MAX_PENDING_JOBS}"
        )
    if not args.gpu_partition:
        parser.error("at least one --gpu-partition is required")

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")
    jwt_key = _read_jwt_key(args.jwt_key_path)
    if not jwt_key:
        parser.error("a readable Slurm JWT key is required")
    api_token = _read_bearer_token(os.getenv("RL_API_TOKEN_FILE", ""))
    rest_base = _slurm_base(args.rest_url, args.api_version)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    try:
        if not args.shadow:
            reset_owned_priorities(rest_base, jwt_key)
        while not stop.is_set():
            try:
                order = reorder_cycle(
                    rest_base=rest_base, jwt_key=jwt_key, scheduler_url=args.scheduler_url,
                    node_names=args.node_name, gpu_partitions=set(args.gpu_partition),
                    mps_per_gpu=args.mps_per_gpu, max_jobs=args.max_jobs,
                    deadline_seconds=args.deadline_seconds,
                    job_name_prefix=args.job_name_prefix, shadow=args.shadow,
                    api_token=api_token,
                )
                if order:
                    LOG.info("ranked %d pending GPU/MPS jobs; top=%s", len(order), order[:5])
            except Exception as exc:
                LOG.warning("cycle skipped: %s", exc)
                if not args.shadow:
                    try:
                        reset_owned_priorities(rest_base, jwt_key)
                    except Exception as reset_exc:
                        LOG.error("could not restore owned priorities: %s", reset_exc)
            stop.wait(max(1.0, args.interval))
    finally:
        if not args.shadow:
            try:
                reset_owned_priorities(rest_base, jwt_key)
            except Exception as exc:
                LOG.error("shutdown priority cleanup failed: %s", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
