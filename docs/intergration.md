# Integration Guide: Add One CPU/GPU Node

本文件記錄在目前單機 Linux + k3s + GPU 環境中，再加入第二台電腦時需要調整的地方。目標是讓第二台機器同時提供 CPU worker capacity 與 GPU/MPS capacity，並讓 DRL scheduler 的 snapshot 與 checkpoint topology 對齊實際 cluster。

> 檔名保留 `intergration.md` 是因為目前文件入口使用這個拼法；若之後要修正拼字，建議另開一次文件 rename，避免連結混在功能變更裡。

## 1. 加入前確認

第二台電腦建議先準備成和第一台一致的 runtime baseline：

| 項目 | 要求 |
|------|------|
| OS | Ubuntu 24.04 或與第一台相容的 Linux |
| Network | 能連到第一台 k3s server 的 `6443`，也能存取 NFS server |
| GPU driver | 已安裝 NVIDIA driver |
| Container runtime | 已安裝 NVIDIA Container Toolkit，k3s agent 可使用 nvidia runtime |
| Storage | 能 mount 第一台提供的 NFS export，例如 `/srv/nfs/k8s` |
| Time sync | 建議 NTP/chrony 正常，避免 Slurm accounting / log 時間混亂 |

目前 `scripts/deploy-1.sh` / `scripts/deploy-2.sh` 假設 k3s cluster 已存在。第二台加入 cluster 後，部署腳本仍在第一台 server 執行。

## 2. 在第一台取得 join token

在第一台 k3s server 上：

```bash
export KUBECONFIG=~/.kube/config
sudo cat /var/lib/rancher/k3s/server/node-token
# K10f426aafcfab99a36047cb9ce0b00e29ab28ce22b7414dca085a80f968eeee42e::server:9a5e3e481545eb74945c29fa74b32acf
hostname -I
# 192.168.0.111
kubectl get nodes -o wide
```

記下：

- `<SERVER_IP>`：第一台的 LAN IP，例如 `192.168.0.111`
- `<NODE_TOKEN>`：`node-token` 內容

## 3. 在第二台安裝 GPU runtime 並加入 k3s

在第二台電腦上：

```bash
# 安裝 driver / nvidia-container-toolkit；若這台不作為 k3s server，不加 --k3s。
bash scripts/setup-linux-gpu.sh

# 加入第一台 k3s server。gpu-host-class 依實際 GPU 型號調整。
curl -sfL https://get.k3s.io | \
  K3S_URL=https://192.168.0.111:6443 \
  K3S_TOKEN=K10f426aafcfab99a36047cb9ce0b00e29ab28ce22b7414dca085a80f968eeee42e::server:9a5e3e481545eb74945c29fa74b32acf \
  INSTALL_K3S_EXEC='agent --node-label gpu-host-class=rtx4070' \
  sh -
```

回到第一台驗證：

```bash
kubectl get nodes -o wide
kubectl get nodes --show-labels | grep gpu-host-class
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'
```

若第二台忘了帶 label，可在第一台補：

```bash
kubectl label node <second-node-name> gpu-host-class=rtx4070 --overwrite
```

## 4. NFS 與共享資料

目前 chart 使用 NFS subdir provisioner 與 RWX PVC。第二台加入後，worker pod 可能被排到第二台，因此第二台必須能連到 NFS server。

在第二台確認：

```bash
showmount -e 192.168.0.111
sudo mkdir -p /mnt/kelpflux-nfs-test
sudo mount -t nfs 192.168.0.111:/srv/nfs/k8s /mnt/kelpflux-nfs-test
touch /mnt/kelpflux-nfs-test/node2-write-test
sudo umount /mnt/kelpflux-nfs-test
```

若 NFS export 限制了 subnet，回第一台調整 `/etc/exports` 後執行：

```bash
sudo exportfs -ra
sudo exportfs -v
```

## 5. GPU Operator 與 node label

`deploy-2.sh` 會安裝 GPU Operator，但一般平台升級會設定 `gpu.autoLabel=false`，避免每次升級都重跑一次性 label hook。加入第二台後，需要確保 GPU node label 已存在：

```bash
kubectl label node <second-node-name> gpu-host-class=rtx4070 --overwrite
kubectl label node <second-node-name> nvidia.com/device-plugin.config=rtx4070-mps --overwrite
kubectl -n gpu-operator rollout status daemonset/nvidia-device-plugin-daemonset --timeout=180s
```

確認第二台 GPU 被 Kubernetes 看見：

```bash
kubectl describe node <second-node-name> | grep -A8 'Allocatable:'
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'
```

注意：MPS sharing 會讓 `nvidia.com/gpu` 顯示 replicas / share slots，不一定等於 physical GPU 張數。Slurm GRES 與 DSAC topology 要以你希望建模的資源單位為準。

## 6. Slurm worker pool 調整

加入第二台後，有兩種常見目標。

### 6.1 只增加可用 worker capacity

如果你只是希望多一台機器可承載更多 worker pod，通常不需要改 chart topology。k3s scheduler 會把 `slurm-worker-*` pod 排到可用 node 上。

部署與驗證：

```bash
bash scripts/deploy-2.sh
bash scripts/verify-live.sh
kubectl -n slurm get pods -o wide | grep slurm-worker
kubectl -n slurm exec slurm-controller-0 -- sinfo -Nel
```

### 6.2 讓 DRL topology 變成 2 nodes × 1 GPU

如果你希望 DRL scheduler 看到兩個 GPU placement nodes，每台一張 GPU，則需要讓 Helm values、snapshot agent、DSAC checkpoint 三者一致。

建議建立 overlay，例如 `chart/values-2node-1gpu.yaml`：

```yaml
rlScheduler:
  snapshotTtlSeconds: 30
  snapshotAgent:
    enabled: true
    intervalSeconds: 10
    gpusPerNode: 1
    mpsPerGpu: 100

slurm:
  workers:
    gpu:
      rtx4070:
        replicas: 2
```

實際 key 需以目前 chart 的 worker pool schema 為準；改完後用 render 檢查：

```bash
helm template slurm-platform ./chart \
  -f chart/values-k3s.yaml \
  -f chart/values-2node-1gpu.yaml \
  --set slurm.jobSubmit.enabled=true \
  --set rlScheduler.enabled=true \
  --set rlScheduler.lua.enabled=true \
  --set rlScheduler.shadowMode=false \
  >/tmp/kelpflux-2node-render.yaml
```

部署：

```bash
VALUES_FILE=chart/values-k3s.yaml bash scripts/deploy-2.sh
helm upgrade --install slurm-platform ./chart \
  -f chart/values-k3s.yaml \
  -f chart/values-2node-1gpu.yaml \
  -n slurm \
  --set gpu.autoLabel=false \
  --set slurm.jobSubmit.enabled=true \
  --set rlScheduler.enabled=true \
  --set rlScheduler.lua.enabled=true \
  --set rlScheduler.shadowMode=false
```

## 7. DSAC checkpoint 必須對齊 topology

目前 live image 內的 checkpoint 是 1 node × 1 GPU topology，`/healthz` 會顯示：

```json
{"obs_dim":192,"n_actions":17}
```

若第二台加入後仍用 1×1 checkpoint，`rl-snapshot-agent` 應該維持送 1×1 snapshot，否則 `/decide` 會因 shape mismatch 安全 abstain。若要讓模型真的在 2 nodes × 1 GPU 上做 placement-aware decision，必須重新訓練 DSAC：

```bash
PYTHONPATH=. .venv-m11/bin/python -m services.rl_scheduler.sim_train \
  --n-nodes 2 \
  --gpus-per-node 1 \
  --trace philly burst ali \
  --total-steps 500000 \
  --out-dir runs/dsac_2node_1gpu_$(date +%Y%m%d)
```

訓練完成後：

1. 更新 `services/rl_scheduler/Dockerfile` 的 `COPY ... /models/dsac.pt` 指向新 checkpoint。
2. 重新執行 `bash scripts/deploy-2.sh`，讓 image rebuild/import，並 rollout restart RL deployments。
3. 確認 `/healthz` 的 `obs_dim` / `n_actions` 和目標 topology 相符。

## 8. Snapshot agent 設定

`rl-snapshot-agent` 負責讓 `/snapshot` 持續 fresh。加入第二台後要檢查：

```bash
kubectl -n slurm logs deploy/rl-snapshot-agent --tail=50
kubectl -n slurm exec slurm-controller-0 -- \
  curl -fsS http://rl-scheduler:8002/healthz
kubectl -n slurm exec slurm-controller-0 -- sh -lc \
  "curl -fsS http://rl-scheduler:8002/metrics | grep -E 'rl_scheduler_snapshot_age_seconds|rl_scheduler_snapshot_free_mps|rl_scheduler_snapshot_pending_jobs'"
```

判讀：

| Metric | 期望 |
|--------|------|
| `rl_scheduler_snapshot_age_seconds` | 小於 `snapshotAgent.intervalSeconds + scrape jitter`，通常 < 30s |
| `rl_scheduler_snapshot_free_mps` | 依 topology 而定；1×1 約 100，2×1 約 200 |
| `rl_scheduler_snapshot_pending_jobs` | 和 Slurm pending queue 大致一致 |

如果 snapshot age 持續變大，先看：

```bash
kubectl -n slurm logs deploy/rl-snapshot-agent --tail=100
kubectl -n slurm get networkpolicy
kubectl -n slurm get svc slurm-restapi rl-scheduler
```

## 9. 驗證清單

加入第二台並重新部署後，在第一台執行：

```bash
bash scripts/verify-live.sh
kubectl get nodes -o wide
kubectl -n slurm get pods -o wide | grep -E 'slurm-worker|rl-snapshot-agent|rl-scheduler'
kubectl -n slurm exec slurm-controller-0 -- sinfo -Nel
kubectl -n slurm exec slurm-controller-0 -- scontrol show nodes
```

提交 CPU smoke job：

```bash
LOGIN_POD=$(kubectl -n slurm get pod -l app=slurm-login -o jsonpath='{.items[0].metadata.name}')
kubectl -n slurm exec "$LOGIN_POD" -- \
  sbatch --wrap='sleep 3' --job-name='node2-cpu-smoke' -p cpu
```

提交 GPU/MPS smoke job，依目前 partition / GRES 名稱調整：

```bash
kubectl -n slurm exec "$LOGIN_POD" -- \
  sbatch -p gpu-rtx4070 --gres=gpu:rtx4070:1,mps:25 \
  --wrap='nvidia-smi && sleep 3' \
  --job-name='node2-gpu-smoke'
```

## 10. 常見風險

| 風險 | 現象 | 處理 |
|------|------|------|
| 第二台沒有 GPU label | GPU worker pod 不上第二台，或 GPU Operator 不套 MPS config | 補 `gpu-host-class` 與 `nvidia.com/device-plugin.config` label |
| NFS 不通 | worker pod 可啟動但 `/shared` 讀寫失敗 | 檢查 `/etc/exports`、防火牆、NFS mount |
| checkpoint topology 不一致 | `/decide` abstain reason 為 shape mismatch | 重新訓練 DSAC 或把 snapshotAgent topology 維持在 checkpoint 支援的大小 |
| GPU share slots 被當 physical GPUs | snapshot free MPS 或 node count 被放大 | 調整 `snapshotAgent.gpusPerNode`，並確認 agent log 的 `nodes/free_mps` |
| Helm upgrade 被 GPU label hook 擋住 | release 變成 failed | 一般升級用 `deploy-2.sh`，它會設定 `gpu.autoLabel=false`；首次加 node 時手動 label |

## 11. 最小建議路線

若目標只是「加入第二台電腦增加資源」：

1. 第二台跑 `setup-linux-gpu.sh`。
2. 第二台用 k3s agent join 第一台。
3. 補 `gpu-host-class=rtx4070` 與 `nvidia.com/device-plugin.config=rtx4070-mps` labels。
4. 確認 NFS 可 mount。
5. 跑 `bash scripts/deploy-2.sh`。
6. 跑 `bash scripts/verify-live.sh`。

若目標是「DRL policy 真的看到 2-node placement」：

1. 先完成上面的最小路線。
2. 新增 2-node overlay values。
3. 用 `--n-nodes 2 --gpus-per-node 1` 重訓 DSAC。
4. 更新 Dockerfile checkpoint 並重新 `deploy-2.sh`。
5. 確認 `/healthz` shape 與 snapshot metrics 對齊。
