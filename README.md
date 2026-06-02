<div align="center">
  
# 〰️ Kelpflux
 
### Elastic Slurm scheduling on Kubernetes for shared GPU AI workloads.
 
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SoWiEee/Kelpflux)
![Slurm](https://img.shields.io/badge/Slurm-23.11-2E86AB?logo=data:image/svg+xml;base64,)
![Kubernetes](https://img.shields.io/badge/Kubernetes-1.34-326CE5?logo=kubernetes&logoColor=white)
![Helm](https://img.shields.io/badge/Helm-3.16+-0F1689?logo=helm&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-1B4332?logoColor=white)
 
*A resilient forest of compute — scheduled by Slurm, scaled by Kubernetes.*
 
Kelpflux brings HPC-grade batch scheduling to Kubernetes, so AI researchers can
submit `sbatch` jobs against a cloud-native cluster that auto-scales CPU and
GPU pools on demand — with MPS-based GPU sharing, checkpoint-aware draining,
and full Prometheus observability.
 
[快速開始](#-getting-started) ·
[使用者教學](docs/tutorial.md) ·
[叢集規格](docs/cluster.md) ·
[優化排程研究](docs/scheduler.md) ·
[採坑紀錄和實作筆記](docs/note.md)
 
</div>
 
<div align="left">
  
## What is Kelpflux?
 
Kelpflux is a cloud-native AI workload platform that runs **Slurm on Kubernetes**.
Researchers submit jobs with familiar `sbatch` commands; the platform handles
the rest — elastic CPU/GPU pool autoscaling, MPS-based GPU sharing,
checkpoint-aware draining, and end-to-end observability.
 
The name fuses two ideas: *kelp forests*, where many independent fronds anchor
to a shared seabed and grow or retreat with the tides; and *flux*, the
continuous flow of compute demand through GPU pools. Together they describe
exactly what Kelpflux does — independent worker pools sharing a common Slurm
control plane, with job throughput flowing dynamically across resources as
demand rises and falls.
 
</div>

---


# 🌱 Motivation

一台有 CPU 和 GPU 的機器，同時有多種 AI 工作要跑——模型推論、超參數搜尋、fine-tuning、資料前處理。  
沒有好的排程系統時，會發生：

- GPU 跑推論時大量閒置（utilization < 20%），同一張卡只讓一個 process 用。
- 多人共用一台主機互相搶資源，沒有隊列、沒有隔離、先到先得。
- Fine-tuning 跑到一半機器重啟，checkpoint 沒存好，重頭來過。
- 工作量少的時候，worker 進程還是佔著資源不釋放。

這些問題的根源在於：現有工具在**資源彈性**和**排程精準度**之間做了取捨。

| 工具 | 擅長 | 不擅長 |
|------|------|--------|
| Kubernetes | 彈性伸縮、容器管理、雲端原生 | HPC workload 的精細資源語意（CPU affinity、GPU GRES、MPS 分配） |
| Slurm | 批次排程、CPU/GPU 精準分配、叢集治理、多使用者隊列 | 動態節點、雲端彈性、容錯恢復 |

本專案的目標很直接：**讓兩者合作**。把 Slurm 跑在 Kubernetes 上，用 K8s 的彈性伸縮撐起 Slurm 的排程能力，解決硬體資源分配的核心問題：

- **利用率**：透過 Slurm MPS（`--gres=mps:25`）讓多個 AI job 共用同一張 GPU 的 SM，utilization 從 < 20% 提升至 70%+
- **隔離性**：CPU pool 和 GPU pool 獨立 autoscale，不同類型的工作互不競爭
- **彈性**：沒有 job 時 worker pod 自動縮回 0；job 進 queue 時 Operator 自動擴出對應節點
- **容錯**：Checkpoint-aware 縮容保護，確保 fine-tuning job 不被中途打斷；NFS PVC 讓結果跨節點持久化

使用者只需要 SSH 進 login node，用熟悉的 `sbatch` 提交工作，不需要知道底層 K8s 的存在。

---

# 🚀 Getting Started

部署統一使用 Helm；目前實機部署固定以 Linux + k3s + GPU 為目標，主要 values 使用 `chart/values-k3s.yaml`。`chart/values.yaml` 保留為 chart default，不作為目前的實際部署路徑。

> Helm chart 名為 `slurm-platform`，把 namespace、ConfigMap、controller/worker StatefulSet、operator、login、NetworkPolicy、device-plugin-config、monitoring（Prometheus/Grafana/Alertmanager/exporters）、storage（NFS subdir provisioner + RWX PVC）全部納入。GPU Operator 因為 PSS=privileged 需求，透過 `scripts/deploy-2.sh` 裝到自己的 `gpu-operator` namespace。完整背景見 [`docs/note.md §5-A`](docs/note.md)。

> 驗證環境：Ubuntu 24.04 x86\_64 + k3s v1.34 + RTX 4070 + NVIDIA driver 580 ✅️

## 1. 準備 k3s/GPU 部署前置資源

`deploy-1.sh` 會整合原本部署步驟 1~4，並輸出時間戳 log：

- 檢查 Linux、NVIDIA driver、Docker、k3s、kubectl、Helm 與 kubeconfig
- 建置 controller、worker、operator、slurm-exporter 映像
- 匯入映像到 k3s containerd
- 建立或重用 munge、ssh、JWT secrets（由 deploy-1.sh 內建處理）
- 套用 NVIDIA RuntimeClass 與 Slurm accounting backend（mysql + slurmdbd）

```bash
export KUBECONFIG=~/.kube/config
bash scripts/deploy-1.sh
```

若主機尚未完成 Linux + k3s + GPU 基礎安裝，先執行 `sudo bash scripts/setup-linux-gpu.sh --k3s`。一般重跑部署時可用下列環境變數略過已完成的階段：

```bash
SKIP_BUILD=1 SKIP_IMPORT=1 bash scripts/deploy-1.sh
SKIP_SECRETS=1 SKIP_PREREQS=1 bash scripts/deploy-1.sh
REGENERATE_SECRETS=true SKIP_BUILD=1 SKIP_IMPORT=1 SKIP_PREREQS=1 bash scripts/deploy-1.sh
```

## 2. 主機 NFS server + LAN exports (Optional)

```bash
sudo bash scripts/setup-nfs-server.sh
cat /etc/exports                       # 必須含 pod CIDR (10.0.0.0/8) AND LAN subnet
sudo exportfs -ra
```

## 3. 部署平台、GPU Operator 與 DSAC Scheduler

`deploy-2.sh` 會把平台主體、GPU Operator 與 live DSAC scheduler 一次收斂到最終狀態。它會先 build/import `slurm-rl-scheduler:m11`，再用一次 `helm upgrade --install` 部署 `slurm-platform`，直接開啟 DSAC live 設定與 `rl-snapshot-agent` 常駐 snapshot 更新，最後用一次 Helm install/upgrade 收斂 NVIDIA GPU Operator；不需要額外 rollout restart。

```bash
export KUBECONFIG=~/.kube/config
bash scripts/deploy-2.sh
```

一般重跑時可用下列環境變數略過已完成的階段：

```bash
SKIP_BUILD=1 SKIP_IMPORT=1 bash scripts/deploy-2.sh
SKIP_GPU_OPERATOR=1 bash scripts/deploy-2.sh
SKIP_WAIT=1 bash scripts/deploy-2.sh
```

DSAC scheduler 會讓 `job_submit.lua` 在 `sbatch` 時呼叫 `/decide`；`shadowMode=false` 代表 DSAC 回傳的 `priority_boost` 會實際加到 `job_desc.priority`。`rl-snapshot-agent` 會每 10 秒從 Slurm REST API 讀取 jobs/nodes，推送 `/snapshot`，避免 snapshot stale 後所有 decision 都被 guardrail 擋掉。`valueAbstain=-100000` 與 `snapshotTtlSeconds=86400` 是目前單機 live 實驗設定，用來避免 checkpoint value scale 造成誤擋。

目前 `deploy-2.sh` 啟用的是 production-safe 的 DSAC live scheduling：RL 會影響 queue priority，實際 node / GPU / MPS placement 仍由 Slurm `select/cons_tres`、GRES、Kubernetes worker pool 與 NVIDIA runtime 執行。若要啟用研究用的 direct DRL placement daemon，必須在能連到 Slurm controller 且可執行 `squeue`、`scontrol`、`srun` 的環境執行，先用 shadow 模式確認 action 與 log，再關閉 shadow：

```bash
# 1) 確認 production DSAC live boost 已啟用
kubectl -n slurm exec slurm-controller-0 -- \
  curl -fsS http://rl-scheduler:8002/healthz

kubectl -n slurm logs deploy/rl-snapshot-agent --tail=20

kubectl -n slurm exec slurm-controller-0 -- \
  curl -fsS http://rl-scheduler:8002/metrics | grep -E \
  'rl_scheduler_shadow_mode|rl_scheduler_last_node_index|rl_scheduler_last_gpu_index'

# 2) Direct placement daemon shadow run：只記錄 DSAC 選到的 job/node/gpu，不執行 srun
SHADOW_MODE=true uv run python -m services.rl_scheduler.live_daemon \
  --policy-dir runs/eval_mlp_20260514-210824/train \
  --node-name slurm-worker-gpu-0 \
  --gpus-per-node 1 \
  --mps-per-gpu 100 \
  --poll-interval 30 \
  --log-dir live_logs/direct-placement-shadow

# 3) Direct placement daemon live run：DSAC 會執行 explicit --nodelist + --gres=mps:N
SHADOW_MODE=false uv run python -m services.rl_scheduler.live_daemon \
  --policy-dir runs/eval_mlp_20260514-210824/train \
  --node-name slurm-worker-gpu-0 \
  --gpus-per-node 1 \
  --mps-per-gpu 100 \
  --poll-interval 30 \
  --live-warmup 0 \
  --log-dir live_logs/direct-placement-live
```

多節點 / 2x2 cluster 時，把 `--node-name` 改成實際 Slurm GPU node 清單，並把 `--gpus-per-node` 設為每台節點的 GPU 數，例如：

```bash
SHADOW_MODE=true uv run python -m services.rl_scheduler.live_daemon \
  --policy-dir runs/eval_mlp_20260514-210824/train \
  --node-name slurm-worker-gpu-0 slurm-worker-gpu-1 \
  --gpus-per-node 2 \
  --mps-per-gpu 100 \
  --log-dir live_logs/direct-placement-2x2-shadow
```

> Direct placement daemon 目前不是 `deploy-2.sh` 的預設部署元件。它會嘗試用 `srun --jobid ... --nodelist ... --gres=mps:N` hard-bind placement，適合研究驗證與蒐集 RLPD transition；production 預設仍建議使用 DSAC priority boost + Slurm placement contract。

預設行為（k3s overlay）：

- `gpu.enabled=true`：在 `gpu-operator` namespace 放 device-plugin-config ConfigMap + 跨節點 labeler Job
- `monitoring.enabled=true`：Prometheus + Alertmanager + Grafana + kube-state-metrics + slurm-exporter（namespace `monitoring`）
- `storage.enabled=true` + `nfsServer=192.168.0.111`：NFS subdir provisioner + StorageClass `slurm-shared-nfs` + 20Gi RWX PVC

LAN IP 不一樣時用 `VALUES_FILE=<your-values.yaml>` 或 Helm values 檔調整 `storage.nfsServer`。GPU Operator 使用 `driver.enabled=false` 與 `toolkit.enabled=false`，因為 host 已經由 `setup-linux-gpu.sh` 裝好驅動與 NVIDIA Container Toolkit。

目前 live scheduler 主線是 **DSAC**。`services/rl_scheduler/smoke_ppo.py` 只是歷史 PPO smoke test，用來快速檢查 simulator API 與 SB3 相容性；不是目前 live cluster 使用的演算法。

> 注意：目前 `slurm-rl-scheduler:m11` 映像會載入 `runs/eval_mlp_20260514-210824/train/dsac.pt`。若要換成新的 DSAC checkpoint，更新 `services/rl_scheduler/Dockerfile` 的 `COPY ... /models/dsac.pt` 後重新執行 `bash scripts/deploy-2.sh`。

**選用功能**（在 `chart/values-k3s.yaml` 開啟）：

| 功能 | 設定 | 說明 |
|------|------|------|
| SSH Login | `login.ssh.authorizedKeys: \|` + 公鑰 | `ssh -p 30022 root@192.168.0.111` |
| OpenTelemetry | `monitoring.otel.enabled: true` | 部署 Tempo + OTel Collector，Grafana 自動加 datasource |

```bash
# 快速加 SSH key（不需重新 helm install）
bash scripts/add-ssh-key.sh add "ssh-ed25519 AAAA... user@laptop"

# 啟用 OTel（helm upgrade）
helm upgrade slurm-platform ./chart -f chart/values-k3s.yaml -n slurm \
  --set monitoring.otel.enabled=true
```

## 4. 驗證 live cluster

`verify-live.sh` 會在 Linux + k3s + GPU live 環境一次完成部署後驗證，涵蓋 chart render、核心 workload rollout、NFS RWX、GPU/GRES、Prometheus/Grafana、DSAC smoke job 與 Lmod 基本檢查。

```bash
export KUBECONFIG=~/.kube/config
bash scripts/verify-live.sh
```

需要略過特定驗證時可用環境變數：

```bash
SKIP_HELM_RENDER=1 bash scripts/verify-live.sh
SKIP_STORAGE=1 SKIP_GPU=1 bash scripts/verify-live.sh
SKIP_MONITORING=1 SKIP_DSAC_SMOKE=1 SKIP_LMOD=1 bash scripts/verify-live.sh
```

## 5. DSAC 訓練與評估

> 以下步驟需要 `.venv-m11`（含 PyTorch）。`PYTHONPATH=.` 確保 `sim/` 和 `services/` 可被找到。

### 快速訓練（本機 CPU）

```bash
# 預設：500k steps, n-step=10, PER, potential shaping, CQL=0.1
PYTHONPATH=. .venv-m11/bin/python -m services.rl_scheduler.sim_train \
    --n-nodes 1 --gpus-per-node 1 \
    --trace philly burst ali \
    --total-steps 500000 \
    --out-dir runs/dsac_sim_$(date +%Y%m%d)

# 加 GPU 加速
PYTHONPATH=. .venv-m11/bin/python -m services.rl_scheduler.sim_train \
    --device cuda --total-steps 500000 \
    --curriculum \
    --out-dir runs/dsac_cuda_$(date +%Y%m%d)

# 2×2 DRL 實驗：需搭配 chart/values-2x2.yaml 與新的 dsac.pt checkpoint。
PYTHONPATH=. .venv-m11/bin/python -m services.rl_scheduler.sim_train \
    --n-nodes 2 --gpus-per-node 2 \
    --trace philly burst ali \
    --total-steps 500000 \
    --out-dir runs/dsac_2x2_$(date +%Y%m%d)
```

### 完整評估（3 families × 5 seeds，對比 score baseline）

```bash
# 完整評估（所有改進開啟，CUDA）
PYTHONPATH=. .venv-m11/bin/python eval/scripts/eval_dsac_placement.py \
    --n-nodes 1 --gpus-per-node 1 \
    --total-steps 500000 \
    --trace-families philly burst ali \
    --seeds 42 43 44 45 46 \
    --device cuda \
    --curriculum \
    --out-dir runs/eval_dsac_$(date +%Y%m%d-%H%M%S)

# 僅 MLP（無 attention，停用 shaping/PER 作為 ablation baseline）
PYTHONPATH=. .venv-m11/bin/python eval/scripts/eval_dsac_placement.py \
    --no-attention --no-per --no-potential-shaping --cql-alpha 0 \
    --total-steps 200000 --device cuda \
    --out-dir runs/eval_mlp_ablation_$(date +%Y%m%d-%H%M%S)

# IQN critic（quantile Huber loss）
PYTHONPATH=. .venv-m11/bin/python eval/scripts/eval_dsac_placement.py \
    --use-iqn --device cuda \
    --out-dir runs/eval_iqn_$(date +%Y%m%d-%H%M%S)

# 載入已有 checkpoint，跳過訓練直接評估
PYTHONPATH=. .venv-m11/bin/python eval/scripts/eval_dsac_placement.py \
    --ckpt runs/dsac_sim/dsac.pt --no-train
```

### 架構與改進 flags 對照

| Flag | 說明 | 預設 |
|------|------|------|
| `--curriculum` | n_jobs 從 10→30→50 漸進 | 關 |
| `--no-per` | 停用 Prioritized Experience Replay | PER 開 |
| `--no-potential-shaping` | 停用 per-step 等待時間 shaping | Shaping 開 |
| `--cql-alpha` | CQL 正則化係數（0=停用） | 0.1 |
| `--use-iqn` | IQN critic（quantile Huber loss） | 關 |
| `--no-attention` | MLP Q-network（非 attention） | 關 |

### 執行單元測試

```bash
.venv-m11/bin/python -m pytest sim/tests/ -q
```

---

## 🗑️ 清理環境

```bash
helm uninstall slurm-platform -n slurm
helm uninstall gpu-operator   -n gpu-operator
kubectl delete -f manifests/core/slurm-accounting.yaml
kubectl delete namespace slurm gpu-operator monitoring nfs-provisioner
# 主機層
/usr/local/bin/k3s-uninstall.sh
sudo systemctl stop nfs-kernel-server
```

> StorageClass 與 gpu-operator namespace 都帶 `helm.sh/resource-policy=keep` 註記，所以 `helm uninstall` 不會自動把它們連同 PV/PVC 拔掉；手動 `kubectl delete namespace` 才會清乾淨。

---

## 部署監控

`monitoring.enabled=true`（k3s overlay 預設打開；chart default 預設關閉）。

```bash
# 存取 Grafana
kubectl -n monitoring port-forward svc/grafana 3000:3000

# 驗證 Prometheus 抓得到 slurm-exporter / operator / kube-state-metrics
bash scripts/verify-live.sh
```

## 使用者操作教學

使用者登入 login node、查看 queue、使用 Lmod、提交 OpenMP / MPI / CUDA / MPS jobs、查看輸出與 Grafana 觀測，請集中參考 [`docs/tutorial.md`](docs/tutorial.md)。

README 的 Getting Started 只保留部署與驗證流程，避免部署步驟和使用者操作混在一起。

---

# 🏗️ System Architecture

<img width="4400" height="2280" alt="圖片" src="https://github.com/user-attachments/assets/5d27ca15-525c-4936-a447-252a8a081934" />


> 完整架構圖請看 [`architecture.html`](assets/architecture.html)

### 主要元件說明

| 元件 | 角色 |
|------|------|
| `slurm-controller` | 執行 `slurmctld`，負責所有排程決策；job 狀態存於獨立 PVC（`slurm-ctld-state`），pod 重啟後 queue 不遺失 |
| `slurm-login` | 使用者入口，提供 `sbatch`、`srun`、`squeue` 等指令 |
| `slurm-worker-*` | 實際執行計算的節點，分 CPU / GPU-A10 / GPU-H100 三個池 |
| `slurm-elastic-operator` | 自製 Python Operator，監控 Queue 狀態並動態調整各 pool 的 replicas；縮容前先 drain 節點，等待 job 完成後才減少 StatefulSet replica |
| `slurmdbd` | Slurm Database Daemon，將 job 會計紀錄（CPU-hours、用戶統計）持久化到 MySQL，為 Fair-Share 排程提供基礎 |
| `mysql` | 後端資料庫（StatefulSet），儲存 slurmdbd 的會計資料，使用 5 Gi PVC |
| NFS + RWX PVC | 跨所有節點的共享磁碟，job 輸出直接寫入 `/shared` |
| `lmod` + modulefile ConfigMaps | HPC 標準模組系統；`module load gcc/11 openmpi/4.1 cuda/12.4` 等指令在 login pod 與 job 內均可用；modulefile 由 Helm chart 產生為 K8s ConfigMap 並掛載到 login/worker pod |

---

# ⚡ Useful Commands

## Slurm Cluster

```bash
# 查看所有 pod 狀態
kubectl -n slurm get pods -o wide

# 查看 Operator 決策日誌（結構化 JSON）
kubectl -n slurm logs deployment/slurm-elastic-operator -f | python3 -m json.tool

# 查看 Slurm controller 日誌
kubectl -n slurm logs statefulset/slurm-controller -f

# 查詢 Operator 寫下的 cooldown 時間戳
kubectl -n slurm get statefulset slurm-worker-cpu \
  -o jsonpath='{.metadata.annotations.slurm\.k8s/last-scale-up-at}'

# 查詢 job 會計紀錄（需要 slurmdbd 正常運行）
kubectl -n slurm exec pod/slurm-controller-0 -- sacct -X --format=JobID,User,State,CPUTime,Start,End

# 確認 slurmdbd / MySQL 狀態
kubectl -n slurm get pods -l app=slurmdbd
kubectl -n slurm get pods -l app=mysql

# 進 login pod 提交 job
kubectl -n slurm exec -it deploy/slurm-login -- bash
```

---

# 🧱 Tech Stack

| 類別 | 工具 |
|------|------|
| 環境 | Ubuntu 24.04 + k3s |
| 容器編排 | Kubernetes |
| HPC 排程器 | Slurm (slurmctld + slurmd)，MpiDefault=pmi2 |
| 節點認證 | Munge |
| Elastic Operator | Python 3.11 + Slurm REST API (slurmrestd) + Kubernetes Python SDK |
| 會計後端 | slurmdbd + MySQL 8.0（job CPU-hours / 使用者統計 / Fair-Share 前置）|
| 共享儲存 | NFS + nfs-subdir-external-provisioner + RWX PVC |
| 網路介面 | Multus CNI + secondary NIC (net2) |
| MPI | OpenMPI 4.1.2 + Slurm PMI2 整合 |
| 模組系統 | Lmod 6.6；modulefile 由 Helm chart 管理，掛載至 login/worker 的 `/opt/modulefiles/` |
| 監控 | Prometheus + Grafana + slurm-exporter + kube-state-metrics + Alertmanager |
| 告警 | 8 條 SLO 規則（provisioning latency、queue wait、flapping 等） |

---

### 7-B：SSH Login ✅ 已完成

使用者可用標準 SSH 直接進入 login pod，不需要安裝 kubectl 或持有 kubeconfig。

```
ssh -p 30022 root@<k3s-host-ip>
       ↓
NodePort :30022 → slurm-login pod
                   ├── sbatch / squeue / sinfo（Slurm 指令即開即用）
                   └── /shared/（NFS 掛載，模型 + 輸出共用）
```

**初次設定 SSH Key（在 k3s host 上執行）：**
```bash
# 1. 生成 key pair（私鑰存在 ~/.ssh/slurm_login_key）
ssh-keygen -t ed25519 -f ~/.ssh/slurm_login_key -C "slurm-login"

# 2. 把公鑰填進 chart/values-k3s.yaml 的 login.ssh.authorizedKeys
cat ~/.ssh/slurm_login_key.pub

# 3. 套用
helm upgrade slurm-platform ./chart -f chart/values-k3s.yaml -n slurm --no-hooks

# 4. 連線
ssh -i ~/.ssh/slurm_login_key -p 30022 root@<k3s-host-ip>
```

**之後新增/移除 key（不需重新 helm install）：**
```bash
bash scripts/add-ssh-key.sh add    "ssh-ed25519 AAAA... user@laptop"
bash scripts/add-ssh-key.sh remove "ssh-ed25519 AAAA... user@laptop"
bash scripts/add-ssh-key.sh list
```

**`chart/values-k3s.yaml` 設定格式：**
```yaml
login:
  ssh:
    nodePort: 30022         # k3s NodePort 範圍 30000-32767；0 = 維持 ClusterIP
    authorizedKeys: |
      ssh-ed25519 AAAA... user@laptop   # 一行一個 key
      ssh-ed25519 AAAA... user@workstation
```

sshd 已加固：`PasswordAuthentication no`、`PermitRootLogin prohibit-password`（key-only）。

---

# 📝 References

- [Slurm Workload Manager Documentation](https://slurm.schedmd.com/)
  - [Slurm Plugin API](https://slurm.schedmd.com/plugins.html)
- [PyTorch Distributed Elastic](https://docs.pytorch.org/docs/stable/distributed.elastic.html)
- [Kubernetes Operator Pythonic Framework (Kopf)](https://github.com/nolar/kopf)
- [Converged Computing: Integrating HPC and Cloud Native](https://www.computer.org/csdl/magazine/cs/2024/03/10770850/22fgId5NFpC)
- [Running Slurm on Amazon EKS with Slinky](https://aws.amazon.com/tw/blogs/containers/running-slurm-on-amazon-eks-with-slinky/)
- [Gang Scheduling](https://kubernetes.io/docs/concepts/scheduling-eviction/gang-scheduling/)
- [Workload Aware Scheduling](https://kubernetes.io/blog/2025/12/29/kubernetes-v1-35-introducing-workload-aware-scheduling/)
- [Slinky Project](https://github.com/slinkyproject)
- [Slonk: Slurm on Kubernetes for ML Research at Character.ai](https://blog.character.ai/slonk/)
- [Prometheus Slurm Exporter](https://github.com/vpenso/prometheus-slurm-exporter)
- [AWS ParallelCluster](https://github.com/aws/aws-parallelcluster)
- [Lmod: An Environment Module System](https://github.com/TACC/Lmod)
- [kube-scheduler Scoring](https://kubernetes.io/docs/reference/scheduling/config/)
- [Grafana](https://grafana.com/)
- [Kube State Metrics](https://github.com/kubernetes/kube-state-metrics)
