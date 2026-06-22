"""Periodic Slurm REST -> rl-scheduler snapshot updater.

The live DSAC /decide endpoint intentionally abstains when its cached cluster
snapshot is stale. This agent keeps that snapshot fresh by polling slurmrestd
and posting a compact view to /snapshot.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

LOG = logging.getLogger("rl_snapshot_agent")
GPU_TYPES = ("rtx4070", "rtx3080", "generic")


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def make_jwt_token(key: bytes, username: str = "root", lifetime: int = 3600) -> str:
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    now = int(time.time())
    payload = _b64url(json.dumps({"exp": now + lifetime, "iat": now, "sun": username}).encode())
    signing_input = f"{header}.{payload}".encode()
    sig = hmac.new(key, signing_input, hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url(sig)}"


def _read_jwt_key(path: str) -> bytes | None:
    if not path:
        return None
    try:
        return open(path, "rb").read().strip()
    except OSError as exc:
        LOG.warning("cannot read JWT key %s: %s", path, exc)
        return None


def http_json(method: str, url: str, *, jwt_key: bytes | None = None, body: dict[str, Any] | None = None, timeout: float = 10.0) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode()
    headers = {"Accept": "application/json", "X-SLURM-USER-NAME": "root"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if jwt_key is not None:
        headers["X-SLURM-USER-TOKEN"] = make_jwt_token(jwt_key)
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw or b"{}")


def _state_tokens(raw: Any) -> set[str]:
    if isinstance(raw, list):
        return {str(x).upper() for x in raw}
    return {x.upper() for x in str(raw or "").replace(",", " ").split()}


def _number(v: Any, default: float = 0.0) -> float:
    if isinstance(v, dict):
        v = v.get("number", default)
    try:
        return float(v)
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
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                return int(m.group(1))
    return default


def _gpu_type_from_text(text: str) -> str:
    lower = (text or "").lower()
    for gpu_type in GPU_TYPES:
        if gpu_type in lower:
            return gpu_type
    return "rtx4070"


def job_view(job: dict[str, Any], *, now: float, default_runtime: float, default_mps: int) -> dict[str, Any] | None:
    if "PENDING" not in _state_tokens(job.get("job_state")):
        return None
    tres = ",".join(str(job.get(k, "") or "") for k in ("tres_req_str", "tres_per_node", "gres"))
    submit_ts = _number(job.get("submit_time"), now)
    runtime = _number(job.get("time_limit"), 0.0) * 60.0
    if runtime <= 0:
        runtime = default_runtime
    mps_req = _parse_tres_int(tres, ("mps",), default=0) or default_mps
    gpu_count = _parse_tres_int(tres, ("gpu",), default=0) or int(_number(job.get("gpus_total"), 1.0)) or 1
    return {
        "job_id": str(job.get("job_id", "")),
        "mps_req": int(mps_req),
        "gpu_count": int(gpu_count),
        "gpu_type": _gpu_type_from_text(tres),
        "runtime": float(runtime),
        "submit_ts": float(submit_ts),
        "can_fit": True,
    }


def _node_name(node: dict[str, Any]) -> str:
    return str(node.get("name") or node.get("node_name") or node.get("hostname") or "")


def _node_state(node: dict[str, Any]) -> set[str]:
    return _state_tokens(node.get("state") or node.get("state_flags"))


def _node_tres_text(node: dict[str, Any], keys: tuple[str, ...]) -> str:
    vals: list[str] = []
    for key in keys:
        val = node.get(key)
        if isinstance(val, dict):
            val = ",".join(f"{k}={v}" for k, v in val.items())
        elif isinstance(val, list):
            val = ",".join(str(x) for x in val)
        if val:
            vals.append(str(val))
    return ",".join(vals)


def node_view(node: dict[str, Any], *, mps_per_gpu: int, default_gpus_per_node: int) -> dict[str, Any] | None:
    states = _node_state(node)
    if states & {"DOWN", "DRAIN", "NOT_RESPONDING", "FAIL"}:
        return None
    cfg = _node_tres_text(node, ("tres", "cfg_tres", "gres", "gres_detail", "features", "active_features", "available_features"))
    if not re.search(r"(gpu|gres/mps|mps)", cfg + "," + _node_name(node), flags=re.IGNORECASE):
        return None
    # Allocated TRES field name varies by slurmrestd API version: older builds
    # expose `alloc_tres`/`alloc_gres`; v0.0.37+ uses `tres_used` (clean
    # `gres/mps=<slots>` form). Without `tres_used` the alloc parse silently
    # returns 0 → free_mps stuck at the configured total (the "Free MPS always
    # 200" dashboard bug). `gres_used` is intentionally excluded: its
    # `mps:<type>:<n>` counts jobs, not slots, and would mis-parse.
    alloc = _node_tres_text(node, ("alloc_tres", "alloc_gres", "tres_used"))
    parsed_gpu_count = _parse_tres_int(cfg, ("gpu",), default=0) or default_gpus_per_node
    # Slurm/NVIDIA MPS may expose logical GPU replicas in TRES. The DSAC
    # checkpoint shape is tied to the configured physical GPUs per node, so
    # cap the live snapshot to that configured topology.
    gpu_count = min(max(1, parsed_gpu_count), max(1, default_gpus_per_node))
    total_mps = _parse_tres_int(cfg, ("mps",), default=0) or parsed_gpu_count * mps_per_gpu
    alloc_mps = _parse_tres_int(alloc, ("mps",), default=0)
    free_total = max(0, total_mps - alloc_mps)
    per_gpu = max(0, min(mps_per_gpu, free_total // max(1, gpu_count)))
    running_jobs = 1 if states & {"ALLOCATED", "MIXED", "COMPLETING"} else 0
    gpu_type = _gpu_type_from_text(cfg or _node_name(node))
    return {
        "gpus": [
            {"free_mps": int(per_gpu), "running_jobs": running_jobs, "gpu_type": gpu_type}
            for _ in range(max(1, gpu_count))
        ]
    }


def build_snapshot(
    jobs_doc: dict[str, Any],
    nodes_doc: dict[str, Any],
    *,
    now: float | None = None,
    mps_per_gpu: int = 100,
    default_gpus_per_node: int = 1,
    default_runtime: float = 600.0,
) -> dict[str, Any]:
    now = time.time() if now is None else now
    pending_jobs = [
        view for job in jobs_doc.get("jobs", [])
        if (view := job_view(job, now=now, default_runtime=default_runtime, default_mps=mps_per_gpu)) is not None
    ]
    nodes = [
        view for node in nodes_doc.get("nodes", [])
        if (view := node_view(node, mps_per_gpu=mps_per_gpu, default_gpus_per_node=default_gpus_per_node)) is not None
    ]
    if not nodes:
        nodes = [{"gpus": [{"free_mps": mps_per_gpu, "running_jobs": 0, "gpu_type": "rtx4070"}]}]
    gpus_per_node = max((len(n.get("gpus", [])) for n in nodes), default=default_gpus_per_node)
    return {
        "ts": now,
        "now": now,
        "pending_jobs": pending_jobs,
        "nodes": nodes,
        "n_nodes": len(nodes),
        "gpus_per_node": gpus_per_node,
        "mps_per_gpu": mps_per_gpu,
    }


def run_once(*, rest_url: str, api_version: str, scheduler_url: str, jwt_key: bytes | None, mps_per_gpu: int, default_gpus_per_node: int, default_runtime: float) -> dict[str, Any]:
    base = f"{rest_url.rstrip('/')}/slurm/{api_version}"
    jobs = http_json("GET", f"{base}/jobs", jwt_key=jwt_key)
    nodes = http_json("GET", f"{base}/nodes", jwt_key=jwt_key)
    snap = build_snapshot(jobs, nodes, mps_per_gpu=mps_per_gpu, default_gpus_per_node=default_gpus_per_node, default_runtime=default_runtime)
    http_json("POST", scheduler_url.rstrip("/") + "/snapshot", body=snap)
    return snap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rest-url", default=os.getenv("SLURM_REST_URL", "http://slurm-restapi.slurm.svc.cluster.local:6820"))
    parser.add_argument("--api-version", default=os.getenv("SLURM_REST_API_VERSION", "v0.0.37"))
    parser.add_argument("--scheduler-url", default=os.getenv("RL_SCHEDULER_URL", "http://rl-scheduler:8002"))
    parser.add_argument("--jwt-key-path", default=os.getenv("SLURM_JWT_KEY_PATH", ""))
    parser.add_argument("--interval", type=float, default=float(os.getenv("SNAPSHOT_INTERVAL_SECONDS", "10")))
    parser.add_argument("--mps-per-gpu", type=int, default=int(os.getenv("MPS_PER_GPU", "100")))
    parser.add_argument("--gpus-per-node", type=int, default=int(os.getenv("GPUS_PER_NODE", "1")))
    parser.add_argument("--default-runtime", type=float, default=float(os.getenv("DEFAULT_RUNTIME_SECONDS", "600")))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    jwt_key = _read_jwt_key(args.jwt_key_path)
    while True:
        try:
            snap = run_once(
                rest_url=args.rest_url,
                api_version=args.api_version,
                scheduler_url=args.scheduler_url,
                jwt_key=jwt_key,
                mps_per_gpu=args.mps_per_gpu,
                default_gpus_per_node=args.gpus_per_node,
                default_runtime=args.default_runtime,
            )
            free_mps = sum(g.get("free_mps", 0) for n in snap["nodes"] for g in n.get("gpus", []))
            LOG.info("snapshot pushed: pending=%d nodes=%d free_mps=%d", len(snap["pending_jobs"]), snap["n_nodes"], free_mps)
        except Exception as exc:  # pragma: no cover - exercised in cluster
            LOG.warning("snapshot push failed: %s", exc)
        if args.once:
            return 0
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
