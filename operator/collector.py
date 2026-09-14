"""Cluster state collection.

PartitionConfigLoader  — reads PARTITIONS_JSON env var and builds PartitionConfig list.
ClusterStateCollector  — queries Slurm (via REST or kubectl exec) to build PartitionState
                         for each pool.  REST path is preferred; exec is the fallback.
"""

from __future__ import annotations

import json
import os
import re

from k8s import K8sClient
from models import Config, PartitionConfig, PartitionState
from slurm import SlurmRestClient

_MEMORY_TRES = re.compile(r"^(\d+(?:\.\d+)?)([KMGTPE]?)([cn]?)$", re.IGNORECASE)


def _tres_per_node_fits(tres: str, partition: PartitionConfig) -> bool:
    """Reject scale-ups that cannot make an oversized per-node request fit."""
    requested_cpu = 0
    memory_tres: tuple[str, str, str] | None = None
    for item in tres.split(","):
        key, sep, value = item.partition("=")
        if not sep:
            continue
        if key == "cpu":
            try:
                requested_cpu = int(value)
            except ValueError:
                pass
        elif key == "mem":
            match = _MEMORY_TRES.fullmatch(value)
            if not match:
                continue
            memory_tres = match.groups()

    requested_memory_mb = 0.0
    if memory_tres:
        amount, unit, scope = memory_tres
        factor = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024,
                  "T": 1024**2, "P": 1024**3, "E": 1024**4}[unit.upper()]
        requested_memory_mb = float(amount) * factor
        if scope.lower() == "c":
            requested_memory_mb *= max(requested_cpu, 1)

    return not (
        (partition.cpus_per_node > 0 and requested_cpu > partition.cpus_per_node)
        or (partition.memory_mb_per_node > 0
            and requested_memory_mb > partition.memory_mb_per_node)
    )


def _can_scale_for_job(fields: dict[str, str], partition: PartitionConfig) -> bool:
    reason = fields.get("Reason", "").strip().strip("()").lower()
    return (
        reason == "resources"
        and _tres_per_node_fits(fields.get("TresPerNode", ""), partition)
    )


class PartitionConfigLoader:
    @staticmethod
    def load(cfg: Config) -> list[PartitionConfig]:
        raw = os.getenv("PARTITIONS_JSON", "").strip()
        if not raw:
            return [
                PartitionConfig(
                    partition=cfg.default_partition,
                    worker_statefulset=cfg.default_worker_statefulset,
                    min_replicas=cfg.default_min_replicas,
                    max_replicas=cfg.default_max_replicas,
                    scale_up_step=cfg.default_scale_up_step,
                    scale_down_step=cfg.default_scale_down_step,
                    scale_down_cooldown=cfg.default_scale_down_cooldown,
                    checkpoint_path=cfg.default_checkpoint_path,
                    max_checkpoint_age_seconds=cfg.default_max_checkpoint_age_seconds,
                    checkpoint_grace_seconds=cfg.default_checkpoint_grace_seconds,
                    drain_timeout_seconds=cfg.default_drain_timeout_seconds,
                    fallback=True,
                )
            ]

        payload = json.loads(raw)
        if not isinstance(payload, list) or not payload:
            raise ValueError("PARTITIONS_JSON must be a non-empty JSON array")

        partitions: list[PartitionConfig] = []
        for item in payload:
            partitions.append(
                PartitionConfig(
                    partition=item["partition"],
                    worker_statefulset=item["worker_statefulset"],
                    min_replicas=int(item.get("min_replicas", 1)),
                    max_replicas=int(item.get("max_replicas", cfg.default_max_replicas)),
                    scale_up_step=int(item.get("scale_up_step", 1)),
                    scale_down_step=int(item.get("scale_down_step", 1)),
                    scale_down_cooldown=int(item.get("scale_down_cooldown", 60)),
                    checkpoint_path=item.get("checkpoint_path", ""),
                    max_checkpoint_age_seconds=int(item.get("max_checkpoint_age_seconds", 600)),
                    checkpoint_grace_seconds=int(item.get("checkpoint_grace_seconds", 0)),
                    drain_timeout_seconds=int(item.get(
                        "drain_timeout_seconds", cfg.default_drain_timeout_seconds
                    )),
                    match_features=tuple(item.get("match_features", [])),
                    match_gres=tuple(item.get("match_gres", [])),
                    cpus_per_node=int(item.get("cpus_per_node", 0)),
                    memory_mb_per_node=int(item.get("memory_mb_per_node", 0)),
                    fallback=bool(item.get("fallback", False)),
                )
            )
        return partitions


class ClusterStateCollector:
    def __init__(self, client: K8sClient, partition_cfgs: list[PartitionConfig],
                 rest: SlurmRestClient | None = None):
        self.client = client
        self.partition_cfgs = partition_cfgs
        self.pool_order = list(partition_cfgs)
        self._rest = rest

    @staticmethod
    def _csv_set(value: str) -> set[str]:
        if not value or value == "(null)" or value == "N/A":
            return set()
        return {part for part in value.split(",") if part and part != "(null)"}

    @staticmethod
    def _gres_match(value: str, needles: tuple[str, ...]) -> bool:
        if not value or value == "(null)":
            return False
        return any(needle in value for needle in needles)

    def _classify_job(self, fields: dict[str, str]) -> PartitionConfig | None:
        """Map a job's fields to its pool, in priority order:
        1. NodeList prefix match (running jobs already placed on a pool's nodes)
        2. Feature match
        3. GRES match
        4. Fallback pool
        """
        node_list = fields.get("NodeList", "")
        if node_list and node_list != "(null)":
            for pool in self.pool_order:
                if node_list.startswith(pool.worker_statefulset):
                    return pool

        features = set()
        for key in ("Features", "Feature", "Constraints"):
            features |= self._csv_set(fields.get(key, ""))

        gres_blob = " ".join(
            fields.get(key, "") for key in ("TresPerNode", "TresPerJob", "TresBind", "TRES")
        )

        for pool in self.pool_order:
            if pool.match_features and any(f in features for f in pool.match_features):
                return pool
            if pool.match_gres and self._gres_match(gres_blob, pool.match_gres):
                return pool

        for pool in self.pool_order:
            if pool.fallback:
                return pool
        return None

    def _jobs_by_pool_and_state(self, partition: str) -> dict[str, dict[str, list[dict[str, str]]]]:
        """Fetch all active jobs for a partition.

        Uses REST API when available; falls back to a single squeue exec otherwise.
        Returns {worker_statefulset: {"PENDING": [...], "RUNNING": [...]}}.
        COMPLETING jobs are counted in the RUNNING bucket because scaling down
        their worker can leave Slurm stuck waiting for the final completion ack.
        """
        if self._rest is not None:
            return self._jobs_by_pool_and_state_rest(partition)
        return self._jobs_by_pool_and_state_exec(partition)

    def _jobs_by_pool_and_state_rest(self, partition: str) -> dict[str, dict[str, list[dict[str, str]]]]:
        result: dict[str, dict[str, list[dict[str, str]]]] = {
            p.worker_statefulset: {"PENDING": [], "SCALE_PENDING": [], "RUNNING": []}
            for p in self.partition_cfgs
            if p.partition == partition
        }
        for job in self._rest.list_jobs(partition):
            state = job.get("job_state", "")
            if state not in ("PENDING", "RUNNING", "COMPLETING"):
                continue
            fields = SlurmRestClient._normalize_job(job)
            pool = self._classify_job(fields)
            if pool is None:
                continue
            bucket = result.setdefault(
                pool.worker_statefulset, {"PENDING": [], "SCALE_PENDING": [], "RUNNING": []}
            )
            normalized_state = "RUNNING" if state == "COMPLETING" else state
            bucket[normalized_state].append(fields)
            if normalized_state == "PENDING" and _can_scale_for_job(fields, pool):
                bucket["SCALE_PENDING"].append(fields)
        return result

    def _jobs_by_pool_and_state_exec(self, partition: str) -> dict[str, dict[str, list[dict[str, str]]]]:
        output = self.client.exec_in_controller(
            f"squeue -h -p {partition} -t PENDING,RUNNING,COMPLETING -o '%i|%T|%N|%f|%b|%R' || true"
        )
        result: dict[str, dict[str, list[dict[str, str]]]] = {
            p.worker_statefulset: {"PENDING": [], "SCALE_PENDING": [], "RUNNING": []}
            for p in self.partition_cfgs
            if p.partition == partition
        }
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 5)
            if len(parts) < 6:
                continue
            _jobid, state, nodelist, features, tres_per_node, reason = parts
            fields = {
                "NodeList": nodelist,
                "Features": features,
                "TresPerNode": tres_per_node,
                "Reason": reason,
            }
            pool = self._classify_job(fields)
            if pool is None:
                continue
            bucket = result.setdefault(
                pool.worker_statefulset, {"PENDING": [], "SCALE_PENDING": [], "RUNNING": []}
            )
            if state in ("PENDING", "RUNNING", "COMPLETING"):
                normalized_state = "RUNNING" if state == "COMPLETING" else state
                bucket[normalized_state].append(fields)
                if normalized_state == "PENDING" and _can_scale_for_job(fields, pool):
                    bucket["SCALE_PENDING"].append(fields)
        return result

    def get_busy_nodes(self, partition_cfg: PartitionConfig) -> int:
        if self._rest is not None:
            return self._get_busy_nodes_rest(partition_cfg)
        return self._get_busy_nodes_exec(partition_cfg)

    def _get_busy_nodes_rest(self, partition_cfg: PartitionConfig) -> int:
        prefix = partition_cfg.worker_statefulset
        count = 0
        for node in self._rest.list_nodes():
            name = node.get("name", "")
            if not name.startswith(prefix):
                continue
            if SlurmRestClient._node_states(node) & SlurmRestClient._BUSY_STATES:
                count += 1
        return count

    def _get_busy_nodes_exec(self, partition_cfg: PartitionConfig) -> int:
        prefix = partition_cfg.worker_statefulset
        output = self.client.exec_in_controller(
            rf"sinfo -h -p {partition_cfg.partition} -N -o '%N %T' 2>/dev/null | awk 'tolower($1) ~ /^{prefix}(-|$)/ && tolower($2) ~ /allocated|mixed|completing/ {{count++}} END {{print count+0}}'"
        )
        # Take the last non-empty line in case sinfo emits warnings before the count.
        lines = [ln for ln in (output or "").splitlines() if ln.strip()]
        return int(lines[-1]) if lines else 0

    def get_checkpoint_age_seconds(self, checkpoint_path: str) -> int | None:
        if not checkpoint_path:
            return None
        command = (
            f"if [ -f '{checkpoint_path}' ]; then "
            f"now=$(date +%s); mtime=$(stat -c %Y '{checkpoint_path}'); "
            "echo $((now - mtime)); else echo -1; fi"
        )
        output = self.client.exec_in_controller(command)
        age = int(output or "-1")
        return None if age < 0 else age

    def collect_all_partition_states(self) -> dict[str, PartitionState]:
        """Collect state for all pools with minimal squeue calls.

        Jobs are fetched once per unique partition name, so pools that share a
        partition only trigger one squeue exec instead of one per pool. With
        the cpu / gpu-rtx4070 / gpu-rtx4080 partition split, each pool now has
        its own partition, so this is effectively one squeue per pool — the
        de-duplication still applies if a future config collapses pools.
        """
        jobs_by_partition = {
            partition: self._jobs_by_pool_and_state(partition)
            for partition in {p.partition for p in self.partition_cfgs}
        }
        return {
            p.worker_statefulset: PartitionState(
                partition=p.partition,
                worker_statefulset=p.worker_statefulset,
                current_replicas=self.client.get_replicas(p.worker_statefulset),
                pending_jobs=len(
                    jobs_by_partition[p.partition].get(p.worker_statefulset, {}).get("PENDING", [])
                ),
                running_jobs=len(
                    jobs_by_partition[p.partition].get(p.worker_statefulset, {}).get("RUNNING", [])
                ),
                busy_nodes=self.get_busy_nodes(p),
                scale_pending_jobs=len(
                    jobs_by_partition[p.partition].get(p.worker_statefulset, {}).get("SCALE_PENDING", [])
                ),
            )
            for p in self.partition_cfgs
        }
