from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import OperatorApp  # noqa: E402


def test_operator_reconnects_rest_after_startup_fallback():
    rest = SimpleNamespace(available=False, calls=0, timeout=10)

    def ping(*, retries=3):
        assert retries == 1
        assert rest.timeout == 2
        rest.calls += 1
        return rest.available

    rest.ping = ping
    events = []
    app = OperatorApp.__new__(OperatorApp)
    app.rest = None
    app._configured_rest = rest
    app._rest_retry_at = 0
    app.collector = SimpleNamespace(_rest=None)
    app.logger = SimpleNamespace(emit=lambda *args, **kwargs: events.append((args, kwargs)))

    app._restore_rest_if_available()
    assert app.rest is None
    assert rest.calls == 1

    rest.available = True
    app._rest_retry_at = 0
    app._restore_rest_if_available()
    assert app.rest is rest
    assert rest.timeout == 10
    assert app.collector._rest is rest
    assert events == [(('slurm_rest_recovered',), {'query_mode': 'rest'})]

    app._restore_rest_if_available()
    assert rest.calls == 2
