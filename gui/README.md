# Kelpflux job monitor

This is a dependency-free monitoring GUI for the active Slurm queue. It polls
the configured API every 4 seconds and renders job state, placement, MPS
request/allocation, and any resource usage explicitly returned by the API.

## Run the demo

From the repository root:

```sh
python3 gui/server.py --port 8080
```

Open <http://127.0.0.1:8080/>. The server serves the static files and a
synthetic `GET /api/jobs` response. The synthetic response intentionally has
no per-job utilization values.

To serve only the static files without the mock endpoint:

```sh
python3 -m http.server 8080 --directory gui
```

The GUI will show a reconnect/error state until an API is available. Set an
API base in the control at the top of the page, or use a query parameter such
as `/?api=https%3A%2F%2Fmonitor.example%2Fapi`. The value is stored in browser
local storage. `/api` requests `/api/jobs`; a base ending in `/jobs` is used as
the endpoint as-is.

## API schema

The preferred response is:

```json
{
  "updated_at": "2026-09-10T08:00:00Z",
  "jobs": [
    {
      "job_id": "8421",
      "state": "RUNNING",
      "node": "slurm-worker-gpu-rtx4070-0",
      "gpu": {"type": "rtx4070", "index": 0},
      "mps": {"requested": 25, "allocated": 25},
      "submit_ts": 1789027200,
      "resource_usage": {
        "sm_percent": 61,
        "source": "job-aware exporter"
      }
    }
  ]
}
```

`resource_usage` is optional. It may contain `sm_percent` (or
`sm_utilization`/`gpu_utilization`) or `mps_used`; without one of those fields
the GUI displays `Not reported`.

For convenience, the GUI also accepts a direct Slurm REST jobs response
(`{"jobs": [...]}`), using `job_state`, `nodes`, `gres_detail`,
`tres_req_str`, `tres_alloc_str`, and `submit_time` when present. A browser
normally needs a same-origin adapter or reverse proxy for slurmrestd, both for
CORS and for `X-SLURM-USER-*` authentication headers.

Per-job SM utilization is deliberately not derived from allocated GPU/MPS
values. The existing DCGM exporter reports device-level metrics, so an
adapter must provide an explicit job-to-device/resource mapping before those
values can be shown as job-level usage.
