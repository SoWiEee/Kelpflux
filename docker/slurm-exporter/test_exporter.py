"""Small regression test for Slurm node-state precedence."""

import importlib.util
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "slurm_exporter", Path(__file__).with_name("exporter.py")
)
assert _SPEC and _SPEC.loader
_EXPORTER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EXPORTER)


def test_unavailable_state_wins_over_idle_and_busy() -> None:
    assert _EXPORTER._classify_node_state({"idle", "down"}) == "down"
    assert _EXPORTER._classify_node_state({"allocated", "drain"}) == "down"
    assert _EXPORTER._classify_node_state({"mixed", "not_responding"}) == "down"
    assert _EXPORTER._classify_node_state({"idle"}) == "idle"
    assert _EXPORTER._classify_node_state({"allocated"}) == "alloc"
