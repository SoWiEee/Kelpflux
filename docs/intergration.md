# Integration Guide: Add a Second GPU Node (RTX 3080)

本文件記錄在目前單機 Linux + k3s + GPU 環境中，再加入第二台電腦時需要調整的地方。具體情境：

- **第一台（現況）**：Linux + k3s server，1× RTX 4070（12 GB VRAM），NVIDIA MPS enabled。
- **第二台（要加入）**：Ubuntu 24.04 LTS，1× RTX 3080（**10 GB** VRAM；3080-12G 變體才是 12 GB），k3s agent。

目標是讓第二台機器同時提供 CPU worker capacity 與 GPU/MPS capacity，並讓 DRL scheduler 的 snapshot 與 checkpoint topology 對齊實際 cluster。

> 檔名保留 `intergration.md` 是因為目前文件入口使用這個拼法；若之後要修正拼字，建議另開一次文件 rename，避免連結混在功能變更裡。

## 0. 先決：釐清 topology 到底是「2×2」還是「2 nodes × 1 GPU」（必讀）

口語上常把目標說成 **「2×2」**，但這跟 code 裡的定義不一樣，務必先對齊，否則 checkpoint 與 snapshot 會永遠對不上，`/decide` 一律 abstain。

- 你現在的硬體是 **兩台機器、每台 1 張 GPU** → 在 sim 的術語是 **`n_nodes=2, gpus_per_node=1`（即 2×1）**。
- code comment 與 `chart/values-2x2.yaml` 講的 **「2×2」是「2 nodes × 2 GPUs/node」**（每台兩張卡），那是另一種拓樸，**不是你目前的硬體**。

兩者的 obs/action 維度完全不同（`sim/gym_env.py:84` `env_dims()` 計算；**注意 `JOB_FEAT_DIM` 已收斂為 9**——GPU one-hot 從 4 種縮到 `{rtx4070, rtx3080}` 2 種，見 §0.1）：

| 拓樸 | 說明 | `obs_dim` | `n_actions` | 出處 |
|------|------|-----------|-------------|------|
| 1×1（現況 live） | 1 node × 1 GPU | **160** | **17** | `sim/gym_env.py:46-48`、`/healthz` |
| **2×1（你要的真實拓樸）** | **2 nodes × 1 GPU** | **166** | **33** | `16*9 + 2*1*6 + 4 + 6 = 166`；`16*2*1 + 1 = 33` |
| 2×2（code comment 講的） | 2 nodes × 2 GPUs | 178 | 65 | `sim/gym_env.py:50-52`、`chart/values-2x2.yaml` |

**結論：你要的是 2×1（166 / 33），不是 repo 預設那個 2×2（178 / 65）。**

> ⚠ obs_dim 已從舊的 192/198/210 收斂為 **160/166/178**（GPU 字母表縮成 `{rtx4070, rtx3080}`，`JOB_FEAT_DIM` 11→9）。**現有 1×1 live checkpoint（192-dim）已不相容、`/decide` 會一律 fail-safe 退回 score，直到用新維度重訓。**

> 請先決定你要的是哪一種：
> - 若每台機器只有 1 張卡 → 用 **2×1**：DSAC 訓練帶 `--n-nodes 2 --gpus-per-node 1`，並自行新增一份 `values-2node-1gpu.yaml`（**不要直接用** `values-2x2.yaml`，它會宣告 `gpu count: 2`、`mps:200`，對不上單卡硬體）。
> - 若之後第二台會插到兩張卡才談 2×2 → 那才用 `values-2x2.yaml` 與 `--gpus-per-node 2`。
>
> 後面第 6/7 節以 **2×1** 為主軸撰寫。

## 0.1 GPU 異質性：4070 / 3080 已建模（字母表已收斂，必讀）

GPU 型別字母表已收斂成 **`{rtx4070, rtx3080}`**——sim、score baseline、runtime predictor、snapshot agent 都一致只認這兩種（舊的 `rtx4080`/`a10`/`h100`/`v100`/`p100` 已移除）。剩下要處理的是**拓樸/重訓**，不是建模：

1. **per-job GPU one-hot 已含 3080** — `sim/gym_env.py:38` `GPU_TYPES = ("rtx4070", "rtx3080")`，one-hot 為 **2 維**（不再是 4 維）。`_job_feat()`（`sim/gym_env.py:114`）已對齊，trace 生成器（`sim/loader.py`）也只發 `{rtx4070, rtx3080}` job。連帶 `JOB_FEAT_DIM` 已是 **9**（原 11），`obs_dim` 已收斂為 160（1×1）。
   - **這也是為什麼現有 192-dim checkpoint 不相容**：字母表收斂直接改了 obs 寬度。任何上線都要用新維度重訓。
2. **VRAM 映射已補 3080** — `sim/scheduler/score.py` `_gpu_type_to_vram()` 現為 `{rtx4070→12, rtx3080→10}`，3080 的 10 GB 已參與 `f_vram_fit` 排序（不再走 `None → 0.5` 中性分支）。
3. **異質 VRAM 的 tier 語意** — 4070 是 12 GB、3080 是 10 GB。`vramTiers: [12, 24]`（`chart/values.yaml:270`）與 sim `_DEFAULT_TIERS_GB`（`score.py:24`）目前**仍為 (12, 24)**，未隨字母表收斂；10 GB job 會被歸到 12 GB tier、視為「略微 over-provision」。若要精確建模 10 GB 上限，需新增 10 GB tier 並同步改這兩處（屬選用、影響 score 語意，預設不動）。
4. **同一個 partition 混卡的風險** — 目前 GPU worker pod 只靠 `nvidia.com/gpu` resource request 排程，**沒有把某個 pool 釘到特定實體機器**（`chart/templates/workers.yaml` 無 `nodeSelector`/affinity）。若把 3080 也標成 `gpu-rtx4070` 並丟進同一個 `gpu-rtx4070` partition，Slurm/k3s 會把 4070 的 job 排到 3080 上（反之亦然），VRAM/型別假設就會錯。**建議**：給 3080 一條獨立的 host-class 與（若要嚴格隔離）獨立 partition，見第 5、6 節。

## 1. 加入前確認

第二台電腦建議先準備成和第一台一致的 runtime baseline：

| 項目 | 要求 |
|------|------|
| OS | Ubuntu 24.04 LTS（本案第二台）或與第一台相容的 Linux |
| Network | 能連到第一台 k3s server 的 `6443`，也能存取 NFS server；第二台的 LAN IP 須在 NFS `/etc/exports` 的 allowed clients 內（見 §4）|
| GPU driver | 已安裝 NVIDIA driver；RTX 3080 屬 Ampere，建議 driver >= 535 以相容目前 GPU Operator |
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
# 第二台是 Ubuntu 24.04 + RTX 3080，請確認 driver 支援 Ampere（建議 driver >= 535）。
bash scripts/setup-linux-gpu.sh

# 加入第一台 k3s server。
# 重要：gpu-host-class 要反映真實硬體 → RTX 3080 用 rtx3080，不要沿用 rtx4070，
# 否則 GPU Operator 的 device-plugin MPS config 與 score baseline 的 VRAM 假設都會錯。
curl -sfL https://get.k3s.io | \
  K3S_URL=https://192.168.0.111:6443 \
  K3S_TOKEN=K10f426aafcfab99a36047cb9ce0b00e29ab28ce22b7414dca085a80f968eeee42e::server:9a5e3e481545eb74945c29fa74b32acf \
  INSTALL_K3S_EXEC='agent --node-label gpu-host-class=rtx3080' \
  sh -
```

> k3s 在這台硬體上曾踩過數個 GPU enablement bug（driver / nvidia runtime / device-plugin）。若 GPU 沒被認出，先回頭對照第一台驗證通過的設定（見 repo memory `project-k3s-gpu-verified.md` 與 `scripts/setup-linux-gpu.sh`）再排查。

回到第一台驗證：

```bash
kubectl get nodes -o wide
kubectl get nodes --show-labels | grep gpu-host-class
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'
```

若第二台忘了帶 label，可在第一台補：

```bash
kubectl label node <second-node-name> gpu-host-class=rtx3080 --overwrite
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

**已知地雷（重要）：`/etc/exports` 的 allowed clients 必須涵蓋第二台的 LAN subnet，不能只放 pod CIDR。**
`scripts/setup-nfs-server.sh:15` 預設 `NFS_EXPORT_CLIENTS=172.16.0.0/12`（給 Kind/Docker bridge），這個範圍**不包含**第二台的 LAN IP（例如 `192.168.0.0/24`）。第二台的 kubelet 會用實體 LAN IP 連 NFS，若 exports 沒放它的 subnet，mount 會直接被拒、worker pod 卡在 `ContainerCreating`。

回第一台調整 `/etc/exports`，把第二台的 LAN subnet 加進 allowed clients（保留原本的 pod/bridge subnet）：

```bash
# 直接編 /etc/exports，或用 setup script 重跑並帶上多個 subnet：
sudo NFS_EXPORT_CLIENTS="192.168.0.0/24" bash scripts/setup-nfs-server.sh
# 若要同時保留原本的 bridge subnet，/etc/exports 可放多行或多個 client：
#   /srv/nfs/k8s 192.168.0.0/24(rw,sync,no_subtree_check,no_root_squash,insecure)
#   /srv/nfs/k8s 172.16.0.0/12(rw,sync,no_subtree_check,no_root_squash,insecure)

sudo exportfs -ra
sudo exportfs -v   # 確認第二台 subnet 有列出來
```

> chart 的 NFS server / path 來自 `chart/values-k3s.yaml:33-34`（`nfsServer: 192.168.0.111`、`nfsPath: /srv/nfs/k8s`），由 `chart/templates/storage.yaml:148-158` 注入 PV/provisioner。NFS server 本身仍在第一台。

## 5. GPU Operator 與 node label

`deploy-2.sh` 會安裝 GPU Operator，但一般平台升級會設定 `gpu.autoLabel=false`，避免每次升級都重跑一次性 label hook。加入第二台後，需要確保 GPU node label 已存在。

**先決定 3080 要用哪個 device-plugin config。** 現有的 MPS sharing config `rtx4070-mps`（`chart/values.yaml:375-381`，`replicas: 4`，對應 `MPS_PER_GPU=4`）是「每張卡切 4 個 MPS slot」。3080 也可以重用這個 MPS 切法，但建議新增一個 `rtx3080-mps` config key 並掛上對應的 `nodeAssignments` / `nfdRules`（`chart/values.yaml:384-412`），讓型別語意清楚、未來好調 replicas：

```bash
# 若沿用既有 MPS 切法（最省事，但 host-class 仍標 rtx3080 以利後續區分）：
kubectl label node <second-node-name> gpu-host-class=rtx3080 --overwrite
kubectl label node <second-node-name> nvidia.com/device-plugin.config=rtx4070-mps --overwrite

# 若已在 chart 新增 rtx3080-mps config（建議）：
# kubectl label node <second-node-name> nvidia.com/device-plugin.config=rtx3080-mps --overwrite

kubectl -n gpu-operator rollout status daemonset/nvidia-device-plugin-daemonset --timeout=180s
```

> NFD 規則（`chart/values.yaml:399-412`）是用 PCI device ID 自動套 device-plugin config，目前只列了 Ada（4070/4080）的 ID。RTX 3080（Ampere，PCI device ID `2206`/`2216` 視板型而定）**不在清單裡**，所以 NFD 不會自動幫它套 config，目前只能靠手動 label 或新增一條 `nfdRules` 規則。

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

### 6.2 讓 DRL topology 變成 2 nodes × 1 GPU（2×1，你的真實拓樸）

如果你希望 DRL scheduler 看到兩個 GPU placement nodes、每台一張 GPU，則需要讓 Helm values、snapshot agent、DSAC checkpoint 三者一致對齊 **2×1（obs_dim=166, n_actions=33）**。

> **不要直接套 `chart/values-2x2.yaml`。** 它是「每台 2 張卡」的 experiment profile（`gres count: 2`、`mps: 200`、`maxNodes: 2`、`replicas: 0`，見 `chart/values-2x2.yaml:40-63`），對應 178/65 的 2×2，對不上你的單卡硬體。請另建一份 `values-2node-1gpu.yaml`。

chart 的 worker 是用 **pools list**（`chart/values.yaml:46-118`），Helm 會整段取代 list 而非 merge，所以 overlay 要把整個 `pools` 重列一次，把 GPU pool 調成「每個 pool replica 1 張卡、可上到 2 個 replica（兩台機器各一）」。重點欄位：

```yaml
# chart/values-2node-1gpu.yaml（請以 chart/values.yaml 的 pools schema 為準）
pools:
  - id: cpu
    # ...照抄 values.yaml 的 cpu pool...
    fallback: true

  - id: gpu-rtx4070            # 仍用同一個 partition，但每個 worker pod 只 1 張卡
    statefulset: slurm-worker-gpu-rtx4070
    appLabel: slurm-worker-gpu-rtx4070
    workerClass: gpu-rtx4070
    partition: gpu-rtx4070
    minReplicas: 1
    maxReplicas: 2             # 兩台機器各一個 GPU worker pod
    replicas: 2               # ← 關鍵：拉到 2，讓 Slurm 看到兩個 GPU node
    maxNodes: 2
    features: [gpu, topology-2x1]
    gres:
      - name: gpu
        type: rtx4070          # 注意：型別異質，見下方警告
        count: 1               # ← 每個 pod 1 張卡（不是 2）
      - name: mps
        count: 100             # ← 每張卡 100（不是 200）
    matchGres: ["gpu:rtx4070", "mps"]
    devicePluginConfig: rtx4070-mps

slurm:
  jobSubmit:
    mpsPerNode: 100            # 每張卡 100 MPS slot（per-GPU，不是 per-node 總和）
```

> **型別異質的取捨（重要）**：上面為了簡單沿用單一 `gpu-rtx4070` partition + `type: rtx4070`，但實體上其中一個 worker pod 會落在 RTX 3080。由於 chart 沒有把 pool 釘到特定機器（`workers.yaml` 無 nodeSelector），Slurm 會把這兩個 GPU node 當成同質的 `rtx4070`。
> - 若你**只在乎 placement 容量、不在乎型別精度** → 沿用上面寫法即可，但要知道 score 的 `f_vram_fit` 會把 3080 也當 12 GB。
> - 若你**要嚴格區分 4070 / 3080** → 應新增一個獨立 `gpu-rtx3080` pool（`type: rtx3080`、獨立 partition、`devicePluginConfig: rtx3080-mps`），並在 `slurm.jobSubmit.helper.partition.rules`（`chart/values.yaml:325-331`）加一條 `gpu:rtx3080 → gpu-rtx3080` 路由。此路線下兩個 partition 各 1 個 GPU node，DRL 的 2×1 仍成立，但型別語意正確。

render 檢查（確認 NodeName 出現兩個 GPU node、gres.conf 的 mps Count 正確）：

```bash
helm template slurm-platform ./chart \
  -f chart/values-k3s.yaml \
  -f chart/values-2node-1gpu.yaml \
  --set slurm.jobSubmit.enabled=true \
  --set rlScheduler.enabled=true \
  --set rlScheduler.lua.enabled=true \
  --set rlScheduler.shadowMode=false \
  >/tmp/kelpflux-2node-render.yaml
# 檢查重點：slurm.nodes.conf 有兩條 GPU NodeName；gres.conf 每個 GPU node 有 mps Count=100
grep -E 'NodeName=.*gpu|Name=mps' /tmp/kelpflux-2node-render.yaml
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

> **現況（GPU 字母表收斂後）**：live image 內**仍是舊的 192-dim checkpoint**（收斂前訓練），`/healthz` 會回報該 checkpoint 自帶的 `{"obs_dim":192,"n_actions":17}`；但收斂後的程式碼**現在組的是 160-dim obs**（1×1），兩者**維度不符 → `/decide` forward shape mismatch → 一律 fail-safe 退回 score**。要恢復 RL boost，得先用新維度（1×1=160 / 2×1=166）重訓並重新烘進 image。

**checkpoint 與 topology 的對齊邏輯**：`serve.py` 用 `DSACAgent.load()` 從 checkpoint 還原 `obs_dim`/`n_actions`，而 `/decide` handler 依 `req.n_nodes`/`req.gpus_per_node` 即時組 obs 與 mask。一旦 obs 寬度（topology 或 **GPU 字母表**）與 checkpoint 維度不一致，agent forward 會 shape mismatch；`rl_hook.lua` 把整個呼叫包在 `pcall` 裡，任何失敗都 fail-safe 退回 score baseline（`chart/lua/rl_hook.lua:89-102`）。**所以收斂/retopology 後不重訓、不換 checkpoint，RL 層只會一路 abstain。**

若要讓模型真的在 **2 nodes × 1 GPU（2×1，目標 `obs_dim=166, n_actions=33`）** 上做 placement-aware decision，必須**從頭重新訓練** DSAC（既有 checkpoint 不相容，輸入/輸出 shape 不同）：

```bash
PYTHONPATH=. .venv-m11/bin/python -m services.rl_scheduler.sim_train \
  --n-nodes 2 \
  --gpus-per-node 1 \
  --trace philly ali \
  --total-steps 500000 \
  --out-dir runs/dsac_2node_1gpu_$(date +%Y%m%d)
```

> **重訓前先處理 sim 端的拓樸與型別常數**（對照 `CLAUDE.md` 的 "Key Constants (sim/gym_env.py)" 一節，`sim/gym_env.py:54-61`）：
> 1. 若改用 module 預設而非 CLI 旗標，把 `sim/gym_env.py:60-61` 的 `N_NODES`/`N_GPUS` 改成你要的拓樸（2×1 → `N_NODES=2, N_GPUS=1`）。注意 `gym_env.py:54-58` 的 comment 範例寫的是 2×2（`N_NODES=2, N_GPUS=2`），別照抄。
> 2. **GPU 型別建模已完成**（見 §0.1）：`GPU_TYPES`、`_gpu_type_to_vram`、`sim/loader.py` 生成器、predictor/snapshot 字母表都已收斂成 `{rtx4070, rtx3080}`，`JOB_FEAT_DIM` 已是 9。重訓直接吃這個維度即可，不用再動字母表。
> 3.（選用）若要精確區分 3080 的 10 GB 上限，補 `_DEFAULT_TIERS_GB`（`score.py:24`）與 `chart/values.yaml:270` 的 10 GB tier；預設維持 (12, 24)。
> 4. 同步 `rlpd_finetune.py` 與（已封存的）`hierarchical.py` 的 CLI 預設拓樸，避免 fine-tune 時又退回 1×1。

訓練完成後（持久路線，正式上線建議）：

1. 更新 `services/rl_scheduler/Dockerfile` 的 `COPY ... /models/dsac.pt` 指向新 checkpoint。
2. 重新執行 `bash scripts/deploy-2.sh`，讓 image rebuild/import，並 rollout restart RL deployments。
3. 確認 `/healthz` 的 `obs_dim` / `n_actions` 顯示 **166 / 33**（2×1），且與 snapshot agent 送的 topology 相符。

> **快速熱換（不重啟 pod）**：`serve.py` 現有 `POST /reload`（`services/rl_scheduler/serve.py:451`，原子換 agent、維度不符會保留舊的並回錯）與 `POST /shadow`（`:497`，runtime 切 SHADOW_MODE）。把新 checkpoint 放進 pod 後 `curl -X POST .../reload -d '{"ckpt_path":"/models/<new>.pt"}'` 即可即時換模，適合 A/B 或迭代驗證；但 pod 重建後會退回 image 內的 checkpoint，所以**正式上線仍要走上面 Dockerfile 持久路線**。

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
| `rl_scheduler_snapshot_free_mps` | 依 topology 而定；1×1 約 100，2×1（兩台各 1 卡、每卡 100）約 200 |
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

提交 GPU/MPS smoke job，依目前 partition / GRES 名稱調整。若沿用單一 `gpu-rtx4070` partition（§6.2 預設路線），3080 也走這個 partition：

```bash
kubectl -n slurm exec "$LOGIN_POD" -- \
  sbatch -p gpu-rtx4070 --gres=gpu:rtx4070:1,mps:25 \
  --wrap='nvidia-smi && sleep 3' \
  --job-name='node2-gpu-smoke'
```

若你選了「3080 獨立 partition」路線，改用：

```bash
kubectl -n slurm exec "$LOGIN_POD" -- \
  sbatch -p gpu-rtx3080 --gres=gpu:rtx3080:1,mps:25 \
  --wrap='nvidia-smi && sleep 3' \
  --job-name='node2-gpu-smoke-3080'
```

跑完用 `nvidia-smi` 的輸出確認 job 真的落在 RTX 3080 那台（型別/VRAM 應顯示 3080、10 GB）。

## 10. 常見風險

| 風險 | 現象 | 處理 |
|------|------|------|
| 把「2×1」當成「2×2」 | checkpoint(178/65) 與 snapshot(166/33) 永遠對不上，`/decide` 一律 abstain | 用 2×1（166/33）：訓練帶 `--gpus-per-node 1`，**勿**直接套 `values-2x2.yaml`（見 §0） |
| 拿舊 192-dim checkpoint 配收斂後的 160-dim 程式 | obs 寬度不符 → `/decide` shape mismatch → 一律 abstain 退回 score | 用新維度（1×1=160 / 2×1=166）重訓並重烘 image（§7）；3080 字母表建模本身已完成（§0.1）|
| 第二台沒有 GPU label | GPU worker pod 不上第二台，或 GPU Operator 不套 MPS config | 補 `gpu-host-class=rtx3080` 與 `nvidia.com/device-plugin.config` label |
| 4070/3080 混在同一 partition | job 被排到「錯型別」的卡，VRAM 假設失準 | 用獨立 `gpu-rtx3080` pool/partition，或接受同質化處理（見 §6.2）|
| NFS 不通 | worker pod 卡 `ContainerCreating` 或 `/shared` 讀寫失敗 | `/etc/exports` 的 allowed clients 要含第二台 **LAN subnet**（非只 pod CIDR），見 §4 |
| checkpoint topology 不一致 | `/decide` shape mismatch → fail-safe 退回 score | 重新訓練 DSAC，或把 snapshotAgent topology 維持在 checkpoint 支援的大小 |
| GPU share slots 被當 physical GPUs | snapshot free MPS 或 node count 被放大 | 調整 `snapshotAgent.gpusPerNode`，並確認 agent log 的 `nodes/free_mps` |
| Helm upgrade 被 GPU label hook 擋住 | release 變成 failed | 一般升級用 `deploy-2.sh`，它會設定 `gpu.autoLabel=false`；首次加 node 時手動 label |

## 11. 最小建議路線

若目標只是「加入第二台電腦增加資源」（DRL 維持 1×1、abstain 退回 score 也可接受）：

1. 第二台（Ubuntu 24.04 + RTX 3080）跑 `setup-linux-gpu.sh`。
2. 第二台用 k3s agent join 第一台（`--node-label gpu-host-class=rtx3080`）。
3. 補 `gpu-host-class=rtx3080` 與 `nvidia.com/device-plugin.config=rtx4070-mps`（或新增的 `rtx3080-mps`）labels。
4. **回第一台把第二台 LAN subnet 加進 `/etc/exports`**，確認 NFS 可 mount。
5. 跑 `bash scripts/deploy-2.sh`。
6. 跑 `bash scripts/verify-live.sh`。

若目標是「DRL policy 真的看到 2-node placement」（2×1）：

1. 先完成上面的最小路線。
2. 決定 topology = **2×1（166/33）**（RTX 3080 的 GPU 型別建模已完成，見 §0.1；只剩拓樸與 vramTiers 選用調整）。
3. 新增 `values-2node-1gpu.yaml`（GPU pool `count: 1` / `mps: 100` / `replicas: 2`，**非** `values-2x2.yaml`）。
4. 用 `--n-nodes 2 --gpus-per-node 1` 從頭重訓 DSAC。
5. 更新 Dockerfile checkpoint 並重新 `deploy-2.sh`。
6. 確認 `/healthz` 顯示 `obs_dim=166, n_actions=33`，且 snapshot metrics 對齊。
