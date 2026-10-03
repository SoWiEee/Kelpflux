# Kelpflux job monitor

This is a dependency-free monitoring GUI for the active Slurm queue. It polls
the live Kelpflux collector every 4 seconds and renders job state, placement,
MPS request/allocation, GPU telemetry, and scheduler signals.

## Run the demo

From the repository root:

```sh
KUBECONFIG=/home/acane/.kube/config python3 gui/server.py --port 8080
```

Open <http://127.0.0.1:8080/>. The server queries Slurm and Kubernetes GPU
workers directly; `kubectl` access and the `slurm` namespace are required.

For a different cluster context, set `KUBECONFIG` or pass the kubeconfig path:

```sh
python3 gui/server.py --kubeconfig /path/to/kubeconfig --port 8080
```

The API endpoint is `GET /api/jobs`; the GUI also accepts an external API base
in the control at the top of the page for deployments behind a reverse proxy.

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
the GUI displays `尚無工作層級資料`.

For convenience, the GUI also accepts a direct Slurm REST jobs response
(`{"jobs": [...]}`), using `job_state`, `nodes`, `gres_detail`,
`tres_req_str`, `tres_alloc_str`, and `submit_time` when present. A browser
normally needs a same-origin adapter or reverse proxy for slurmrestd, both for
CORS and for `X-SLURM-USER-*` authentication headers.

Per-job SM utilization is deliberately not derived from allocated GPU/MPS
values. The existing DCGM exporter reports device-level metrics, so an
adapter must provide an explicit job-to-device/resource mapping before those
values can be shown as job-level usage.

## 即時資料

GUI 每 4 秒讀取一次 `/api/jobs`，後端即時彙整：

- 「工作要求的 GPU 分享量」與「已配置 GPU 分享量」來自 Slurm 工作資料，代表工作要求或拿到的 MPS 百分比，不是實際算力使用率。
- `gpu_metrics`：由 GPU worker 的 `nvidia-smi` 回報 SM、VRAM、功耗與溫度。
- `scheduler`：由 RL scheduler 的 Prometheus metrics 回報就緒狀態、快照年齡、可用 MPS 與最近動作。
- `queue_metrics`：由 Slurm exporter 回報等待時間、排程週期與 backfill queue。
- 「實際使用量」只有在 API 的工作資料中提供 `resource_usage.sm_percent`、`sm_utilization`、`gpu_utilization` 或 `mps_used` 時才會顯示。
- 介面中的「資料範圍」是說明文字，不是另一個監控指標；它提醒使用者配置量與硬體實際使用量是兩件事。

DCGM exporter 可以即時提供整張 GPU 的 SM、VRAM、功耗等裝置層級指標，但預設沒有
`slurm_job_id`。若要顯示單一工作使用率，後端還需要透過 cgroup/PID 或 GPU UUID
將 Slurm 工作與 GPU 指標做對應，再把上述 `resource_usage` 欄位回傳給 GUI。
因此目前工作列表可能顯示「尚無工作層級資料」；這代表尚未建立工作到
GPU UUID 的對應，不代表整張 GPU 的即時 telemetry 無法取得。
