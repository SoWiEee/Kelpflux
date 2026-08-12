"""Sim-backed pull function for the M9 weight-tuner bandits.

The bandit wants a `pull(arm, context) -> reward` callable. An arm is
a tuple of score weights (alpha, beta, delta, epsilon); context is a
3-vector summarising the trace. We turn that into one `sim.runner.run`
call and return -mean_JCT_hours as the reward (higher = better).

Caching: the same (arm, context_key) tuple should always yield the
same reward (sim is deterministic given the trace). We cache so the
bandit can revisit arms without paying the simulation cost twice.
The context key is the (trace_family, seed) pair the caller stashes
in `_context_index` — we keep them aligned by index.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

# Make `import sim.*` work whether we're invoked from repo root or not.
_HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.loader import MPS_PER_GPU, generate_by_family  # noqa: E402


Arm = Tuple[float, ...]


@dataclass
class TraceSpec:
    family: str
    seed: int
    n_jobs: int = 1000


def context_vector(spec: TraceSpec) -> Tuple[float, float, float]:
    """3-dim context: normalised job count + mean mps + mean gpu_count.

    Built from the synthetic generator's outputs, so we can cheaply
    score the context without running the simulator. All features land
    roughly in [0, 1] for the trace families we ship.
    """
    jobs = generate_by_family(spec.family, n_jobs=spec.n_jobs, seed=spec.seed)
    n = len(jobs)
    mean_mps = sum(j.mps_req for j in jobs) / max(n, 1) / float(MPS_PER_GPU)
    mean_gpu = sum(j.gpu_count for j in jobs) / max(n, 1) / 8.0
    return (n / 2000.0, mean_mps, mean_gpu)


def default_arm_grid() -> List[Arm]:
    """3×3×3 = 27 arms over (alpha, delta, epsilon).

    Picked to span the M8 sensitivity grid plus an extra epsilon axis
    that M8's 5×5 didn't sweep. Beta stays at 0.20 (M8 default).
    """
    arms: List[Arm] = []
    for a in (0.10, 0.40, 0.70):
        for d in (0.05, 0.20, 0.40):
            for e in (0.00, 0.30, 0.60):
                arms.append((a, d, e))
    return arms
