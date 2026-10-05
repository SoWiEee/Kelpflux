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
「最近完成」列表會替含 OpenTelemetry context 的工作顯示「查看追蹤」。後端透過
`GET /api/traces/<trace_id>` 向 Tempo 查詢該工作；瀏覽器不需 Grafana 憑證。
示範完整工作生命週期時，先提交暫停的 CPU 工作，等 operator 觀察到等待狀態後再釋放：

```sh
kubectl -n slurm exec slurm-controller-0 -- sbatch --parsable -H --partition=cpu --job-name=otel-lifecycle-demo --wrap='sleep 15'
kubectl -n slurm exec slurm-controller-0 -- scontrol release <job-id>
```

工作完成後，從「最近完成」開啟追蹤。立即啟動的工作可能沒有 `queue_wait` span；
暫停後釋放的工作應會顯示等待階段。Tempo 資料有保留期限。

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

`resource_usage` is optional. It may contain `sm_percent`, `vram_used_mib`,
`vram_percent`, or `mps_used`; without one of those fields the GUI displays
`尚無工作層級資料`.

For convenience, the GUI also accepts a direct Slurm REST jobs response
(`{"jobs": [...]}`), using `job_state`, `nodes`, `gres_detail`,
`tres_req_str`, `tres_alloc_str`, and `submit_time` when present. A browser
normally needs a same-origin adapter or reverse proxy for slurmrestd, both for
CORS and for `X-SLURM-USER-*` authentication headers.

Per-job SM/VRAM utilization is never derived from allocated GPU/MPS values.
The live collector samples `nvidia-smi pmon` on each GPU worker and maps its
compute PID to `SLURM_JOB_ID` through the process environment, cgroup, or
`slurmstepd` ancestry. Processes without a Slurm job ID are excluded. Under
MPS, NVIDIA exposes SM utilization for the shared MPS server rather than each
client, so the API labels it `shared_gpu_sm_percent`; VRAM remains attributed
to the client job PID.

## 即時資料

GUI 透過 `/api/events` 接收 Slurm queue 變更的 SSE 通知並立即重讀 `/api/jobs`，另每 4 秒校正完整狀態；後端即時彙整：

- 「工作要求的 GPU 分享量」與「已配置 GPU 分享量」來自 Slurm 工作資料，代表工作要求或拿到的 MPS 百分比，不是實際算力使用率。
- `gpu_metrics`：由 GPU worker 的 `nvidia-smi` 回報 SM、VRAM、功耗與溫度。
- `resource_usage`：由 GPU worker 的 MPS client PID 對應至 Slurm job，回報 Job VRAM；MPS 下的 SM 以「共享 SM」標示，不冒充單一 job 用量。
- `history`：由 `sacct` 讀取最近 24 小時、最多 50 筆終止工作；JCT 定義為完成時間減提交時間。
- `trace_id`：從 Slurm `AdminComment` 的 OTel context 取得；有值時可直接查看 Tempo 工作時間線。
- `scheduler`：由 RL scheduler 的 Prometheus metrics 回報就緒狀態、快照年齡、可用 MPS 與最近動作。
- `queue_metrics`：由 Slurm exporter 回報等待時間與排程週期。
- 「實際使用量」只顯示可由 PID 明確歸屬的工作，不會用 MPS 配置量推估。
- 介面中的「資料範圍」是說明文字，不是另一個監控指標；它提醒使用者配置量與硬體實際使用量是兩件事。

工作方塊支援滑鼠 hover 與鍵盤 focus，可查看工作類型、提交命令、partition、提交時間與即時用量。
Slurm 23.11 的歷史命令欄位使用 `sacct SubmitLine`；若上游沒有保存該欄位，介面會明確顯示未提供。
