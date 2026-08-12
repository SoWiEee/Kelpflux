# Integration Guide: Add a Second GPU Node (RTX 3080)

本文件記錄在目前單機 Linux + k3s + GPU 環境中，再加入第二台電腦時需要調整的地方。具體情境：

- **第一台（現況）**：Linux + k3s server，1× RTX 4070（12 GB VRAM），NVIDIA MPS enabled。
- **第二台（要加入）**：Ubuntu 24.04 LTS，1× RTX 3080（**10 GB** VRAM；3080-12G 變體才是 12 GB），k3s agent。

目標是讓第二台機器同時提供 CPU worker capacity 與 GPU/MPS capacity，並讓 DRL scheduler 的 snapshot 與 checkpoint topology 對齊實際 cluster。

> 檔名保留 `intergration.md` 是因為目前文件入口使用這個拼法；若之後要修正拼字，建議另開一次文件 rename，避免連結混在功能變更裡。

## 0. 先決：你的拓樸是 2×1（2 nodes × 1 GPU），對齊 obs_dim=166（必讀）

你的硬體是 **兩台機器、每台 1 張 GPU**（host-1 = RTX 4070、host-2 = RTX 3080）→ 在 sim 的術語是 **`n_nodes=2, gpus_per_node=1`（2×1）**。三者——Helm values、snapshot agent、DSAC checkpoint——的拓樸必須一致，否則 checkpoint 與 snapshot 永遠對不上，`/decide` 一律 abstain。

維度由 `env_dims()`（`sim/gym_env.py`）算出（`JOB_FEAT_DIM=9`，GPU one-hot 已收斂為 `{rtx4070, rtx3080}` 2 維，見 §0.1）：

| 拓樸 | 說明 | `obs_dim` | `n_actions` |
|------|------|-----------|-------------|
| 1×1（現況 live） | 1 node × 1 GPU | **160** | **17** |
| **2×1（你要的）** | **2 nodes × 1 GPU** | **166** | **33** |

`16*9 + 2*1*6 + 4 + 6 = 166`、`16*2*1 + 1 = 33`（已對 `env_dims(2,1)` 驗證）。

> ⚠ obs_dim 已從舊的 192 收斂為 **160/166**（GPU 字母表縮成 `{rtx4070, rtx3080}`，`JOB_FEAT_DIM` 11→9）。現有 1×1 live checkpoint（192-dim）已不相容、`/decide` fail-safe 退回 score，直到用新維度重訓（§7）。

> **chart 設定走單一 overlay：`chart/values-2x1.yaml`**（已建好）。它把整件事做成**宣告式、低維護**：異質 4070 + 3080、每張卡切 **4 個 MPS slot**、兩種卡**各自獨立 partition** 並用 `nodeSelector` 釘到對應實體機，device-plugin 的 MPS 設定由 **NFD 規則自動套用**（PCI 比對、node 重建也會自動收斂，不靠一次性手動 label）。DSAC 訓練帶 `--n-nodes 2 --gpus-per-node 1`。後面 §5/§6/§7 都以這個 overlay 為主軸。
>
> （repo 另有的 `values-2x2.yaml` 是「每台 2 張卡」的另一種拓樸，**與你的硬體無關**，本指南不再使用它。）

## 0.1 GPU 異質性：4070 / 3080 已建模（字母表已收斂，必讀）

GPU 型別字母表已收斂成 **`{rtx4070, rtx3080}`**——sim、score baseline、runtime predictor、snapshot agent 都一致只認這兩種（舊的 `rtx4080`/`a10`/`h100`/`v100`/`p100` 已移除）。剩下要處理的是**拓樸/重訓**，不是建模：

1. **per-job GPU one-hot 已含 3080** — `sim/gym_env.py:38` `GPU_TYPES = ("rtx4070", "rtx3080")`，one-hot 為 **2 維**（不再是 4 維）。`_job_feat()`（`sim/gym_env.py:114`）已對齊，trace 生成器（`sim/loader.py`）也只發 `{rtx4070, rtx3080}` job。連帶 `JOB_FEAT_DIM` 已是 **9**（原 11），`obs_dim` 已收斂為 160（1×1）。
   - **這也是為什麼現有 192-dim checkpoint 不相容**：字母表收斂直接改了 obs 寬度。任何上線都要用新維度重訓。
2. **VRAM 映射已補 3080** — `sim/scheduler/score.py` `_gpu_type_to_vram()` 現為 `{rtx4070→12, rtx3080→10}`，3080 的 10 GB 已參與 `f_vram_fit` 排序（不再走 `None → 0.5` 中性分支）。
3. **異質 VRAM 的 tier 語意** — 4070 是 12 GB、3080 是 10 GB。`vramTiers: [12, 24]`（`chart/values.yaml:270`）與 sim `_DEFAULT_TIERS_GB`（`score.py:24`）目前**仍為 (12, 24)**，未隨字母表收斂；10 GB job 會被歸到 12 GB tier、視為「略微 over-provision」。若要精確建模 10 GB 上限，需新增 10 GB tier 並同步改這兩處（屬選用、影響 score 語意，預設不動）。
4. **卡別隔離（已由 `values-2x1.yaml` 解決）** — 過去 GPU worker pod 只靠 `nvidia.com/gpu` request 排程、無法釘到特定實體機，混卡會讓 4070 的 job 跑到 3080 上（反之亦然），VRAM/型別假設失準。**現在 `values-2x1.yaml` 用獨立 partition + `nodeSelector`（`gpu-host-class`）把 `gpu-rtx4070` / `gpu-rtx3080` 兩個 pool 各自釘到對應實體機**——chart 端已加上 per-pool `nodeSelector` 支援（`chart/templates/workers.yaml`，opt-in、不影響現有 1×1 部署），卡別不再會錯置。這條路也讓 DRL 看到「兩個型別不同的 placement node」，是 2×1 placement 決策面的前提。

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

> 以下是 **2026-06 實際在 RTX 3080（`192.168.0.104`, Ubuntu 22.04）上驗證通過**的流程。
> 四個關卡照順序做完才會成功：**(1) server 防火牆 → (2) node-2 前置 → (3) 版本釘選 join → (4) NFS/映像/部署**。
> 每一關都有對應的踩雷紀錄，照做可一次到位。

### 3.0 node-2 前置（在第二台上）

```bash
# 1) NVIDIA driver：RTX 3080 屬 Ampere，建議 >= 535（實測 580.x 可）。確認：
nvidia-smi

# 2) NFS client 工具（純 client 機器預設沒有 mount.nfs，缺了 PVC 掛載會卡 ContainerCreating）
sudo apt update && sudo apt install -y nfs-common

# 3) 確認 GPU 的 PCI device id，對照 values-2x1.yaml nfdRules（3080 通常是 2206）
lspci -nn | grep -i NVIDIA          # 例：... [10de:2206] ...
```

> 缺 `nfs-common` 是常見地雷：`mount -t nfs` 會回 `bad option ... need a /sbin/mount.<type> helper`，且之後排到這台的 worker pod 會卡在 `ContainerCreating`。

### 3.1 在 server（acane）開放防火牆（**最容易卡、且踩了三次**）

acane 的 ufw 是 `Default: deny (incoming)` 且 `deny (routed)`。單機時無所謂，但第二台從 LAN 連進來會被靜默 DROP（症狀是「卡住逾時」而非 `connection refused`）。一次把 node 互通需要的埠全開（限定 LAN 網段），並放行 cluster 內部轉發：

```bash
# 在 acane 上：
LAN=192.168.0.0/24
sudo ufw allow from $LAN to any port 6443  proto tcp   # k3s apiserver / join
sudo ufw allow from $LAN to any port 8472  proto udp   # flannel VXLAN（跨 node pod 網路）
sudo ufw allow from $LAN to any port 10250 proto tcp   # kubelet（logs / exec / metrics）
sudo ufw allow from $LAN to any port 2049  proto tcp   # NFS（NFSv4；NFSv3 另需 111 tcp/udp）

# 跨 node 的 pod↔pod（含 slurmd↔slurmctld）會經 acane 轉發，必須放行 routed + cluster CIDR：
sudo ufw allow from 10.42.0.0/16                       # pod CIDR
sudo ufw allow from 10.43.0.0/16                       # service CIDR
sudo ufw default allow routed
sudo ufw reload
```

> **為什麼 routed + CIDR 不能省**：只開 6443 能讓 node `Ready`、甚至 Slurm node 短暫 `idle`，但 slurmctld（acane）週期性 ping node-2 的 slurmd 走的是「acane 轉發到 node-2 pod」這條路。`deny (routed)` 會把它擋掉 → Slurm node 變 `DOWN / Not responding`。開了 routed + pod/svc CIDR 才會穩。

### 3.2 加入 k3s（**版本必須釘到跟 server 一致**）

```bash
# 在 acane 上先抓「現在」的 server token 與版本：
sudo cat /var/lib/rancher/k3s/server/node-token
kubectl version --short | grep Server      # 例：v1.34.6+k3s1
```

```bash
# 在 node-2 上 join。INSTALL_K3S_VERSION 要等於 server 版本，不要用 stable channel。
# gpu-host-class 必須反映真實硬體（3080 用 rtx3080），device-plugin MPS config 與 score 的
# VRAM 假設都靠它。
curl -sfL https://get.k3s.io | \
  INSTALL_K3S_VERSION=v1.34.6+k3s1 \
  K3S_URL=https://192.168.0.111:6443 \
  K3S_TOKEN='<上一步的 node-token>' \
  INSTALL_K3S_EXEC='agent --node-label gpu-host-class=rtx3080' \
  sh -
```

> **版本 skew 地雷**：`stable` channel 會隨時間漂移。若 server 是 1.34、`stable` 已滾到 1.35，agent 會裝成「比 server 新」→ kubelet 不能比 apiserver 新 → join 時 `Failed to validate connection`。**兩台都釘明確版本**；要升級時先升 server 再升 agent。
>
> **「No change detected」地雷**：若這台之前裝過 k3s，重跑 installer 會印 `No change detected so skipping service start`，**只 enable 不 start**。乾淨重裝：
> ```bash
> sudo /usr/local/bin/k3s-agent-uninstall.sh
> # 重裝時加 INSTALL_K3S_FORCE_RESTART=true，避免又被跳過
> ```

### 3.3 在 server 驗證已加入

```bash
kubectl get nodes -o wide                                   # 兩台都 Ready、VERSION 相同
kubectl get nodes --show-labels | grep gpu-host-class       # node-2 帶 gpu-host-class=rtx3080
```

若忘了帶 label：`kubectl label node <node-2> gpu-host-class=rtx3080 --overwrite`。

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

## 5. GPU Operator 與 3080 的 4-slot MPS 設定（宣告式，低維護）

`deploy-2.sh` 會安裝 GPU Operator。**3080 要切成 4 個 MPS slot 的設定，已經宣告在 `values-2x1.yaml` 裡，不需要每台手動 label**：

- `deviceConfigs.rtx3080-mps`：`sharing.mps … replicas: 4` → 每張 3080 切 4 個 share（對齊 4070 的 `rtx4070-mps`、`MPS_PER_GPU=4`）。
- `nfdRules`：用 **PCI device ID 自動**把 `rtx3080-mps` 套到 3080 node（Ampere GA102 10 GB 通常是 `2206`）。NFD 會在 node 重建 / k3s 重裝後**自動收斂**，不靠一次性 hook——這就是低維護的關鍵。
- `nodeAssignments`：保留為非 NFD 叢集的 fallback（與 NFD 並存，衝突時 NFD 優先）。

**唯一的一次性人工步驟：確認 3080 的 PCI ID**（板型不同可能是 `2206`/`2216`/`2208`），對不上就改 `values-2x1.yaml` 的 `nfdRules`：

```bash
# 在第二台（host-2, RTX 3080）查實際 PCI device id：
ssh host-2 'lspci -nn | grep -i NVIDIA'
# 例：... [10de:2206] ...  → 對應 values-2x1.yaml nfdRules 的 "2206"，相符即可
```

套用 overlay（§6.2 會一起部署）後，確認 device-plugin 已把 3080 切成 4 share：

```bash
kubectl -n gpu-operator rollout status daemonset/nvidia-device-plugin-daemonset --timeout=180s
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'
# 兩個 GPU node 的 allocatable nvidia.com/gpu 應各為 4（= 4 MPS share）
```

> **fallback（只在 NFD 沒生效時用）**：手動把 config label 補上 ——
> `kubectl label node <host-2> nvidia.com/device-plugin.config=rtx3080-mps --overwrite`。
> 但正常情況下 NFD 規則會自動處理，**不需要**這一步。

> 注意：MPS sharing 會讓 `nvidia.com/gpu` 顯示 share slots（這裡是 4），不等於 physical GPU 張數（1）。Slurm GRES（`mps:100`，job 要 `mps:25`）與 DSAC topology（2×1）才是「資源單位」的權威來源。

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

### 6.2 讓 DRL topology 變成 2 nodes × 1 GPU（2×1）— 套用 `values-2x1.yaml`

要讓 DRL scheduler 看到兩個型別不同的 GPU placement node（4070 + 3080），就把 **`chart/values-2x1.yaml`** 疊在 `values-k3s.yaml` 之上即可。這份 overlay 已經把所有需要的東西做成宣告式（**單一檔、低維護**）：

- `partitions`：`cpu` + `gpu-rtx4070` + `gpu-rtx3080`（兩種卡各一個 partition）。
- `pools`：`gpu-rtx4070` / `gpu-rtx3080` 兩個 GPU pool，各 `count: 1` / `mps: 100` / `replicas: 1`，並用 **`nodeSelector: {gpu-host-class: …}` 釘到對應實體機**（4070→host-1、3080→host-2）。
- `gpu.deviceConfigs.rtx3080-mps`（`replicas: 4`）+ `nfdRules`（PCI 自動套用）+ `nodeAssignments`（fallback）。
- `slurm.jobSubmit.helper.partition.rules`：多一條 `gpu:rtx3080 → gpu-rtx3080` 路由。

> **為什麼是這份而不是 `values-2x2.yaml`**：`values-2x2.yaml` 是「每台 2 張卡」（`count: 2`、`mps: 200`、178/65）的另一種拓樸，跟你的單卡硬體無關。`values-2x1.yaml` 才對齊 166/33。
>
> **Helm list 取代提醒（維護點）**：overlay 用整段取代 list（非 merge），所以它已重列 `partitions` / `pools` / `gpu.nodeAssignments` / `gpu.nfdRules.rules` / `helper.partition.rules`。日後若 `values.yaml` 改這幾個 list，記得同步 `values-2x1.yaml`（檔頭有列出）。

render 檢查（已驗證：兩個 GPU NodeName、各 `mps Count=100`、兩個 GPU pool 帶 nodeSelector、`gpu-rtx3080` partition 與 `rtx3080-mps` config 都在）：

```bash
helm template slurm-platform ./chart \
  -f chart/values-k3s.yaml \
  -f chart/values-2x1.yaml \
  --set slurm.jobSubmit.enabled=true \
  --set rlScheduler.enabled=true \
  >/tmp/kelpflux-2x1-render.yaml
grep -E 'NodeName=.*gpu|Name=mps|gpu-host-class|PartitionName=gpu' /tmp/kelpflux-2x1-render.yaml
```

部署：

```bash
VALUES_FILE=chart/values-k3s.yaml bash scripts/deploy-2.sh
helm upgrade --install slurm-platform ./chart \
  -f chart/values-k3s.yaml \
  -f chart/values-2x1.yaml \
  -n slurm \
  --set gpu.autoLabel=false \
  --set slurm.jobSubmit.enabled=true \
  --set rlScheduler.enabled=true \
  --set rlScheduler.lua.enabled=true \
  --set rlScheduler.shadowMode=false
```

> 部署後確認兩台各自就位：`kubectl -n slurm get pods -o wide | grep gpu` 應看到 `slurm-worker-gpu-rtx4070-0` 落在 host-1、`slurm-worker-gpu-rtx3080-0` 落在 host-2（靠 `nodeSelector`）。`sinfo -Nel` 應列出兩個 GPU partition。

### 6.3 把本地映像匯入 node-2（**必做，否則 GPU worker `ErrImagePull`**）

本專案的 `slurm-*` 映像是**本地 build、用 `k3s ctr images import` 逐節點塞進去**的（沒有 registry）。k3s 每個 node 的 containerd image store 獨立，`deploy-2.sh` 只 build/import RL scheduler image，其餘只存在 acane。`imagePullPolicy: IfNotPresent` 在 node-2 找不到就會去 registry 拉 → 失敗 → `ErrImagePull`。

把 node-2 可能用到的本地映像**合併匯出成一個 tar**（同 tar 會去重共用的 CUDA layer），傳過去匯入：

```bash
# 在 acane 匯出（worker 是 CUDA base，整包約 4–5 GB）
sudo k3s ctr -n k8s.io images export /tmp/slurm-images.tar \
  docker.io/library/slurm-worker:latest \
  docker.io/library/slurm-controller:latest \
  docker.io/library/slurm-exporter:latest \
  docker.io/library/slurm-elastic-operator:latest \
  docker.io/library/slurm-rl-scheduler:m11
sudo chmod 644 /tmp/slurm-images.tar

# 傳到 node-2 並匯入它的 k3s containerd
scp /tmp/slurm-images.tar <user>@192.168.0.104:/tmp/
ssh <user>@192.168.0.104 'sudo k3s ctr -n k8s.io images import /tmp/slurm-images.tar'

# 驗證 + 清理
ssh <user>@192.168.0.104 "sudo k3s ctr -n k8s.io images ls | grep slurm-"
sudo rm -f /tmp/slurm-images.tar
ssh <user>@192.168.0.104 'sudo rm -f /tmp/slurm-images.tar'
```

> 之後每次 rebuild 這些映像都要重做一次 import。長期低維護的解法是架一個兩台都連得到的**本地 registry**，讓 `imagePullPolicy` 正常運作；在那之前，export→import 是最務實的做法。

### 6.4 部署後排錯（實測會遇到的四個狀態）

| 症狀 | 根因 | 修法 |
|---|---|---|
| GPU worker `ErrImagePull`（node-2） | 本地映像沒匯入 node-2 | 做 §6.3，然後 `kubectl -n slurm delete pod slurm-worker-gpu-rtx3080-0`（StatefulSet 重建、IfNotPresent 直接用） |
| Slurm node `INVALID_REG`（`Low socket*core*thread count`） | reconfigure 期間殘留的舊註冊；偵測 CPU 其實 ≥ 設定 | `state=resume` 無效（INVALID_REG 只能靠重新註冊清）→ `kubectl -n slurm delete pod <worker>` 讓 slurmd 重註冊 → 再 `scontrol update nodename=<node> state=resume` 清 DRAIN |
| Slurm node `DOWN / Not responding` | 跨 node 的 slurmctld→slurmd ping 被 acane ufw `deny routed` 擋（見 §3.1） | 開 routed + pod/svc CIDR 後 `scontrol update nodename=<node> state=resume` →（必要時）`scontrol reconfigure` 清 `*` |
| srun `can't find address for host <node-2>`（只跨 node job） | slurm 設定檔是 **subPath 掛載 → pod 建立時凍結、不隨 configmap 更新**；比 topology 變更更早建立的 pod 拿到舊設定、缺新 node 的 NodeAddr | 重啟那些舊 pod 讓它重掛當前 configmap：`kubectl -n slurm delete pod <pod>`。controller/worker 部署時通常已重建，**`slurm-login` 常被遺漏** |

> 驗證跨 node 連通（pod 沒有 ping/nc 時用 bash `/dev/tcp`）：
> ```bash
> kubectl -n slurm exec slurm-controller-0 -- bash -c \
>   'timeout 3 bash -c "echo > /dev/tcp/<node-2-pod-ip>/6818" && echo OK || echo FAIL'
> ```
>
> 端到端 smoke（MPS 分配，落到對的卡）：
> ```bash
> kubectl -n slurm exec <slurm-login-pod> -- bash -lc \
>   'srun -p gpu-rtx3080 --gres=mps:25 -t 1 nvidia-smi -L'   # → NVIDIA GeForce RTX 3080
> ```

## 7. DSAC checkpoint 必須對齊 topology

> **現況（GPU 字母表收斂後）**：live image 內**仍是舊的 192-dim checkpoint**（收斂前訓練），`/healthz` 會回報該 checkpoint 自帶的 `{"obs_dim":192,"n_actions":17}`；但收斂後的程式碼**現在組的是 160-dim obs**（1×1），兩者**維度不符 → `/decide` forward shape mismatch → 一律 fail-safe 退回 score**。要恢復 RL boost，得先用新維度（1×1=160 / 2×1=166）重訓並重新烘進 image。

> **重訓 / 重評的時機（提醒）**：注意 live 目前**沒有壞**——線上跑的舊映像（舊程式 + 192-dim checkpoint）三者自洽、照常運作；維度不符只在「用收斂後的新程式碼 build 新映像、卻配舊 checkpoint」時才會發生。所以**不急著重 build/部署就不必現在重訓**。又因為 1×1 三方本來就打平（`docs/eval-writeup.md §4.4.2`），1×1 的 RL 即使活著也只是退回 score、不損失可量測的東西。**建議把 160/166-dim 的重訓與重評一次併進即將到來的 2-node + rtx3080 那輪做**（那時 obs_dim 本來就要變、trace 也會重生），而不是現在單獨為 1×1 重跑。eval-writeup §3 的數字是收斂前（192-dim、job 帶 v100/p100）跑的，重評後會位移，但機制結論（A 類打平、B 類待 2-node）不會翻盤。

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
> 4. 同步 `rlpd_finetune.py` 的 CLI 預設拓樸，避免 fine-tune 時又退回 1×1。

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
| 套錯 overlay（拿 `values-2x2.yaml`）| checkpoint(178/65) 與 snapshot(166/33) 永遠對不上，`/decide` 一律 abstain | 用 `values-2x1.yaml`（166/33）+ 訓練帶 `--gpus-per-node 1`（見 §0、§6.2）|
| 拿舊 192-dim checkpoint 配收斂後的 160-dim 程式 | obs 寬度不符 → `/decide` shape mismatch → 一律 abstain 退回 score | 用新維度（1×1=160 / 2×1=166）重訓並重烘 image（§7）；3080 字母表建模本身已完成（§0.1）|
| 3080 沒被切成 4 MPS slot | `nvidia.com/gpu` allocatable 顯示 1 而非 4 → 沒有 share | 多半是 **NFD PCI ID 沒對上**：用 `lspci -nn` 確認 3080 的 id 並更新 `values-2x1.yaml` 的 `nfdRules`（§5）；NFD 沒生效才手動補 `device-plugin.config=rtx3080-mps` label |
| GPU worker 落錯機器（4070 job 跑到 3080）| 卡別/VRAM 假設失準 | 已由 `values-2x1.yaml` 的獨立 partition + `nodeSelector` 解決（§0.1、§6.2）；確認兩台都有 `gpu-host-class` label |
| NFS 不通 | worker pod 卡 `ContainerCreating` 或 `/shared` 讀寫失敗 | `/etc/exports` 含第二台 **LAN subnet**（§4）；node-2 裝 `nfs-common`（§3.0）；acane ufw 開 `2049/tcp`（§3.1）|
| node 加不進來 / Slurm node `DOWN` | join 時 `Failed to validate connection`；或 node Ready 但 Slurm `Not responding` | acane ufw 缺埠：`6443/8472/10250/2049` + **`default allow routed` + pod/svc CIDR**（§3.1）。`deny routed` 會擋跨 node 轉發 |
| agent 版本比 server 新 | join `Failed to validate connection`（kubelet 不能比 apiserver 新）| join 時釘 `INSTALL_K3S_VERSION=<server 版本>`，別用 `stable`（§3.2）|
| GPU worker `ErrImagePull`（node-2）| 本地 `slurm-*` 映像沒匯入 node-2 的 containerd | export→scp→`k3s ctr images import`（§6.3），再 bounce pod |
| srun `can't find address`（跨 node job）| slurm 設定檔 subPath 掛載凍結，舊 pod（常是 `slurm-login`）缺新 node 設定 | 重啟那些舊 pod 重掛 configmap（§6.4）|
| checkpoint topology 不一致 | `/decide` shape mismatch → fail-safe 退回 score | 重新訓練 DSAC，或把 snapshotAgent topology 維持在 checkpoint 支援的大小 |
| GPU share slots 被當 physical GPUs | snapshot free MPS 或 node count 被放大 | 調整 `snapshotAgent.gpusPerNode`，並確認 agent log 的 `nodes/free_mps` |
| Helm upgrade 被 GPU label hook 擋住 | release 變成 failed | 一般升級用 `deploy-2.sh`，它會設定 `gpu.autoLabel=false`；首次加 node 時手動 label |

## 11. 最小建議路線

若目標只是「加入第二台電腦增加資源」（DRL 維持 1×1、abstain 退回 score 也可接受）：

1. 第二台（Ubuntu 24.04 + RTX 3080）跑 `setup-linux-gpu.sh`。
2. 第二台用 k3s agent join 第一台（`--node-label gpu-host-class=rtx3080`）。
3. 確認 host-1 也有 `gpu-host-class=rtx4070` label（nodeSelector / nodeAssignments 都靠它）。3080 的 device-plugin config 走 `values-2x1.yaml` 的 NFD 自動套用，不必手動 label。
4. **回第一台把第二台 LAN subnet 加進 `/etc/exports`**，確認 NFS 可 mount。
5. 跑 `bash scripts/deploy-2.sh`。
6. 跑 `bash scripts/verify-live.sh`。

若目標是「DRL policy 真的看到 2-node placement」（2×1，**你要的**）：

1. 先完成上面的最小路線。
2. 確認 3080 的 PCI ID 與 `values-2x1.yaml` 的 `nfdRules` 相符（§5）；型別建模已完成（§0.1）。
3. 套用 **`chart/values-2x1.yaml`**（已含 partition / pool / `nodeSelector` / `rtx3080-mps` / NFD / 路由，§6.2）——`helm upgrade … -f values-k3s.yaml -f values-2x1.yaml`。
4. 用 `--n-nodes 2 --gpus-per-node 1` 從頭重訓 DSAC（§7）。
5. 更新 Dockerfile checkpoint 並重新 `deploy-2.sh`。
6. 確認 `/healthz` 顯示 `obs_dim=166, n_actions=33`，且 snapshot metrics（`free_mps≈200`、`nodes=2`）對齊。

## 12. node-2 OS 升級：Ubuntu 22.04 → 24.04（修 3080 MPS）

**動機（為什麼要升）**：node-2 被裝成 **Ubuntu 22.04**，但 acane 是 **24.04**。結果 node-2 的 **device-plugin MPS 控制 daemon 的 `config-manager` sidecar 一直 CrashLoopBackOff**（Go panic `index out of range [0]` at `findPidToSignal`），3080 的 MPS 因此**無法多工**——並行 CUDA job 只有 1/N 成功、其餘 `CUDA-capable device(s) is/are busy`。根因是環境差異：

| | acane（node-1, 4070） | node-2（3080） |
|---|---|---|
| OS | Ubuntu 24.04 | **Ubuntu 22.04** |
| NVIDIA driver（host，`driver.enabled=false`）| 580.167.08-1ubuntu1 | **580.159.03-0ubuntu0.22.04.1** |
| device-plugin MPS | ✅ 多工正常（4/4 並行 OK） | ❌ config-manager crash → 不多工 |

`580.167.08` **沒有為 22.04 打包**（只在 24.04 repo），所以單純對齊 driver 不可行；要對齊就得升 OS。升完 24.04 會順帶把 host driver 帶到 580.167.08。詳見 `docs/eval-writeup.md` §4.2 / §5.1 第 4 項與 memory `project-mps-never-functional`。

> ⚠️ **高風險操作**：`do-release-upgrade` 在遠端機（node-2 = `nutn-admin@192.168.0.104`）上跑會 reboot、且可能中途失敗 → 機器可能變不可達。**強烈建議在能實體接觸/有 console（IPMI/螢幕鍵盤）時做**。一定要在 `tmux`/`screen` 裡跑，並讓 release-upgrade 開第二個 sshd（port 1022）當 fallback。

### 12.1 升級前檢查清單（逐項打勾）

**A. 備份 / 記錄當前狀態（在 node-2 上，存到 `/shared` 讓 acane 也讀得到）**
```bash
ssh nutn-admin@192.168.0.104
sudo mkdir -p /shared/node2-preupgrade && cd /shared/node2-preupgrade
# OS / kernel / driver / k3s 版本
{ lsb_release -a; uname -a; cat /proc/driver/nvidia/version; k3s --version; } > versions.txt 2>&1
# 套件清單（回滾比對用）
dpkg -l > dpkg-list.txt
apt-mark showmanual > apt-manual.txt
# k3s agent 設定 + token + 關鍵設定檔
sudo cp -a /etc/rancher/k3s /shared/node2-preupgrade/etc-rancher-k3s 2>/dev/null || true
sudo cp /etc/systemd/system/k3s-agent.service /shared/node2-preupgrade/ 2>/dev/null || true
# NVIDIA / containerd 設定
sudo cp -a /etc/nvidia-container-runtime /shared/node2-preupgrade/ 2>/dev/null || true
nvidia-smi -q > nvidia-smi-q.txt 2>&1
```

**B. 相容性確認（升級後要用的版本，先查清楚）**
- [ ] **k3s 版本**：server（acane）是 `v1.34.6+k3s1`。agent 升級後 **kubelet 不能比 apiserver 新**（§3.2、§10）。24.04 上要**重裝 k3s agent 並釘同一版本**：`curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=v1.34.6+k3s1 K3S_URL=... K3S_TOKEN=... sh -`（token/URL 用 §2 的；node-label `gpu-host-class=rtx3080`）。
- [ ] **NVIDIA driver**：24.04 repo 有 `nvidia-driver-580 580.167.08-1ubuntu1`（= acane）。升完用 `apt-cache policy nvidia-driver-580` 確認 candidate 是 580.167.08。
- [ ] **gpu-operator**：`driver.enabled=false`（用 host driver），所以重點是 host driver + nvidia-container-toolkit 在 24.04 上裝好；gpu-operator 的 MPS daemonset 會自己 reconcile。
- [ ] **NFS client**：24.04 要重裝 `nfs-common`（§3.0），否則 `/shared` PVC 掛不上 → worker `ContainerCreating`。
- [ ] **ufw / 防火牆**：升級可能重置；事後對照 §3.1（acane 端）與 node-2 端的埠（6443/8472/10250/2049 + routed）。

**C. 叢集側的事前準備（在 acane 上）**
- [ ] **drain / cordon node-2**，避免升級期間排程到它：`kubectl cordon nutnadmin-e500-g9-ws760t && kubectl drain nutnadmin-e500-g9-ws760t --ignore-daemonsets --delete-emptydir-data --force`。
- [ ] **記下當前 rl-scheduler image**（目前是實驗用 `slurm-rl-scheduler:htabp1p2`；升級這段可先還原 production `m11`，避免實驗 image 卡住）。
- [ ] **確認 acane（control-plane）健康**，升級期間 4070 + 控制面要能獨撐：`kubectl get nodes`、`scontrol ping`。

**D. 回滾方案（先想好）**
- node-2 是 **agent / worker**，不是 control-plane → **最壞情況可以整台重裝**：22.04 或直接 24.04 全新安裝，再照本指南 §3–§6 重新 join。所以「升級失敗」不會弄壞叢集，只是 node-2 要重來。
- do-release-upgrade 失敗時：機器多半還在舊 OS（升級是先下載再切換）；有 console 就能進 recovery。**沒 console 而 SSH 斷掉 = 要實體去處理**——這就是為什麼建議實體在場。

### 12.2 升級步驟

```bash
# 在 node-2，務必在 tmux 裡（SSH 斷了 upgrade 不會死）
ssh nutn-admin@192.168.0.104
tmux new -s osupg
sudo apt update && sudo apt full-upgrade -y      # 先把 22.04 內更新到最新
sudo apt install -y update-manager-core
sudo do-release-upgrade -d   # -d：22.04→24.04 走第一個釋出階段；會自動開 port 1022 的備援 sshd
# 全程回答 prompt（保留本機改過的設定檔時，k3s/nvidia 相關選 "keep local"）；最後 reboot
```

### 12.3 升級後驗證（關鍵：3080 MPS 要真的多工）

```bash
# 1) node-2 回來、OS/driver 已對齊
ssh nutn-admin@192.168.0.104 'lsb_release -rs; cat /proc/driver/nvidia/version | head -1'
#   期望：24.04 ；NVRM 580.167.08

# 2) 重裝/確認 k3s agent（釘 server 版本），nfs-common，nvidia-container-toolkit
#   （見 12.1-B；若 k3s agent 還在就跳過）

# 3) acane 上：node 回 Ready、uncordon
kubectl uncordon nutnadmin-e500-g9-ws760t
kubectl get nodes -o wide
kubectl -n gpu-operator get pods | grep mps-control      # 期望 2/2 Running（不再 CrashLoop）

# 4) Slurm node 回 UP（升級後通常 DOWN/Not responding；清 phantom gres 要 restart slurmctld）
kubectl -n slurm exec slurm-controller-0 -- scontrol update nodename=slurm-worker-gpu-rtx3080-0 state=RESUME
kubectl -n slurm exec slurm-controller-0 -- sinfo -N | grep 3080   # idle 無 '*'

# 5) ★ 決定性測試：3080 上 4 個並行 CUDA job 全 COMPLETE（MPS 真的多工）
LOGIN=$(kubectl -n slurm get pod -l app=slurm-login -o jsonpath='{.items[0].metadata.name}')
kubectl -n slurm exec "$LOGIN" -- bash -c '
  for k in 1 2 3 4; do
    sbatch -w slurm-worker-gpu-rtx3080-0 -p gpu --gres=mps:25 --time=5 \
      --wrap="/shared/bin/gpu_workload 250 4096 256 $k"; done'
# 等完成後查：4 個都 COMPLETED（升級前是 1 COMPLETED + 3 FAILED）
```

**第 5 步 4/4 COMPLETED = MPS 修好。**（實測 2026-06-21 升完 24.04 後通過。）然後就能跑乾淨的真實算力 real-CUDA 評估（`docs/eval-writeup.md` §5.1 第 4 項）：沿用 `--cuda-workload`、**拿掉 `--exclusive-gpu`**（分數 MPS 共置、跟 sleep 同 sharing），乾淨隔離「真實算力 + 干擾」這一個變數。

> **實測 gotchas（2026-06-21 升級踩到的，照順序）**：
> 1. **anydesk third-party repo 讓 `apt update` 失敗**（TLS handshake 錯）→ do-release-upgrade 第一步 `cache.update()` 中止。升級前先把**所有 non-ubuntu repo 移開**（`/etc/apt/sources.list.d/`），用不到的（如 anydesk）直接刪。
> 2. **非互動升級會卡在 conffile 衝突 prompt**（`*** xrdp.ini (Y/I/N/O/D/Z) [default=N]`），DistUpgradeViewNonInteractive 沒自動帶預設。對策：在 node-2 背景跑一個 `tmux send-keys -t osupg Enter` 的迴圈（每 3s 一次），自動以預設「保留現有設定」答完。
> 3. **Secure Boot 拒簽 DKMS 模組**：別裝 `nvidia-driver-580-server`（走 DKMS → 本地建的 module 沒簽 → `modprobe: Key was rejected by service` → `nvidia-smi` 失敗）。要用 **Canonical 預簽的 prebuilt 模組** `linux-modules-nvidia-580-<kernel>-generic`（`.ko.sig`），**purge 掉 `nvidia-dkms-580*`**（含 -server）後 `--reinstall` 簽好的那顆 + `depmod -a`。driver 維持 24.04 標準 repo 的 580.159.03（580.167 從來不是關鍵變數）。
> 4. **config-manager 仍會 CrashLoopBackOff（同 `findPidToSignal` panic）——但這是 cosmetic**。`mps-control-daemon-ctr` 本身 Running 就會多工；config-manager 只負責「設定變更時 signal reload」，靜態設定下不影響。判定 MPS 好壞**一律以第 5 步的 4/4 並行測試為準**，不要看 config-manager 的 pod readiness。

### 12.4 升級後常見風險（接 §10）

| 風險 | 現象 | 處理 |
|------|------|------|
| k3s agent 比 server 新 | node `NotReady`、kubelet 報版本錯 | 重裝 agent 釘 `INSTALL_K3S_VERSION=v1.34.6+k3s1`（§3.2）|
| driver 沒升到 580.167 | `nvidia-smi` 仍 580.159 或裝不起來 | `apt-cache policy nvidia-driver-580` 確認 24.04 candidate；`apt install nvidia-driver-580` + reboot |
| nfs-common 沒了 | worker 卡 `ContainerCreating`、`/shared` 掛不上 | `apt install nfs-common`（§3.0）|
| Slurm node phantom gres | `AllocTRES=gres/mps=100` 但無 job、新 job `PENDING Resources` | restart slurmctld（`kubectl delete pod slurm-controller-0`）重建 alloc 狀態 |
| ufw 被重置 | 跨 node pod/slurmd 不通、node `Not responding` | 對照 §3.1 重開埠 + `default allow routed` |
