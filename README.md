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
 
[使用者教學](docs/tutorial.md) ·
[叢集規格](docs/cluster.md) ·
[系統架構圖](assets/architecture.html) ·
[優化排程研究](docs/scheduler.md) ·
[論文初稿](docs/paper.md)
 
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

把 Slurm 跑在 K8s 上解決的是**機制**問題（卡內共享、彈性、容錯）。但平台只是載具，真正的研究問題是**排程決策本身**：當一張異質 GPU 叢集同時承載兩類性質相反的工作時，誰該先跑、放在哪張卡、和誰共置？

- **低延遲推論**：有 SLO 期限、常只需部分算力（適合 MPS 分片），但對**尾端延遲（p99）**敏感——偶爾的 straggler 就會違反 SLO。
- **長時間訓練**：吞吐導向、長時間獨佔整卡，對排隊延遲較不敏感。

在異質硬體下（不同世代 GPU 算力可差數倍），這個張力無法用單一靜態規則調好。而雲端原生生態裡的成熟排程器各管一層、卻**都不優化尾端**：Kueue／Volcano 解的是配額與 gang 准入、Kubernetes 1.34 的 DRA 提供的是 GPU 分片**機制**、Kubeflow 管的是工作生命週期——沒有一個是「學習式排序／放置策略」。

本研究就落在這個空隙，並刻意不做「宣稱 DRL 必勝」的研究，而是回答兩個更誠實的問題：

1. **能不能用學習式、風險敏感的策略補上那層缺失的智慧？** 我們以分散式深度強化學習（RDSAC：discrete SAC + IQN，並以 CVaR 風險量度直接優化回報分布的尾端）作為 placement 建議者，透過 Slurm `job_submit.lua` 以**非阻塞、失效即回退**的方式整合進生產路徑——任何服務異常都自動退回既有啟發式，slurmctld 永不被阻塞。此學習式策略與 DRA 並非競爭，而是**互補**：它可在 DRA 的分片機制之上驅動裝置選擇與准入排序。
2. **這套智慧到底什麼時候才真的有用？** 實機評估面臨「一個 step 即一個跑數分鐘的工作、樣本極稀少、量測易受叢集暖機漂移污染」的根本限制。我們提出一套**模擬到實機（sim-to-real）評估方法學**——抗跑序漂移的交錯輪轉、多 seed 配對信賴區間、兼顧平均與尾端（p95／p99／CVaR）及 SLO 違反率——並誠實回報其規模條件：在 2×1 小規模上排程策略統計打平，**智慧排程的價值需要叢集規模與工作負載競爭方能顯現**。

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

目前 `deploy-2.sh` 啟用的是 production-safe 的 DSAC live scheduling：RL 會影響 queue priority，實際 node / GPU / MPS placement 仍由 Slurm `select/cons_tres`、GRES、Kubernetes worker pool 與 NVIDIA runtime 執行。若要讓 DSAC hard-bind placement，可使用 hold-release controller：先讓 job 以 held 狀態進入 queue，controller 呼叫 `/act` 取得 `(job_i, node_j, gpu_k)`，再用 `scontrol update ReqNodeList=<node>` 與 `scontrol release` 讓 Slurm 原生執行該 placement。

```bash
# 1) 確認 production DSAC live boost 已啟用
kubectl -n slurm exec slurm-controller-0 -- \
  curl -fsS http://rl-scheduler:8002/healthz

kubectl -n slurm logs deploy/rl-snapshot-agent --tail=20

kubectl -n slurm exec slurm-controller-0 -- \
  curl -fsS http://rl-scheduler:8002/metrics | grep -E \
  'rl_scheduler_shadow_mode|rl_scheduler_last_node_index|rl_scheduler_last_gpu_index'

# 2) 使用者提交 held GPU/MPS job，讓 controller 做 hard placement 後再 release
kubectl -n slurm exec deploy/slurm-login -- \
  sbatch --hold --parsable -J dsac-place-test \
  -p gpu-rtx4070 --gres=mps:10 --time=00:03:00 \
  --wrap 'hostname; sleep 10'

# 3) Shadow run：只看 DSAC 選到哪個 held job / node / gpu，不更新 Slurm
PYTHONPATH=. python -m services.rl_scheduler.placement_controller \
  --once --job-name-prefix dsac-place-test \
  --node-name slurm-worker-gpu-rtx4070-0 \
  --node-name slurm-worker-gpu-rtx4070-1 \
  --scheduler-url http://rl-scheduler:8002 \
  --scheduler-exec-prefix 'kubectl -n slurm exec pod/slurm-controller-0 --' \
  --slurm-exec-prefix 'kubectl -n slurm exec deploy/slurm-login --'

# 4) Live hard placement：寫入 ReqNodeList 並 release held job
PYTHONPATH=. python -m services.rl_scheduler.placement_controller \
  --once --no-shadow --job-name-prefix dsac-place-test \
  --node-name slurm-worker-gpu-rtx4070-0 \
  --node-name slurm-worker-gpu-rtx4070-1 \
  --scheduler-url http://rl-scheduler:8002 \
  --scheduler-exec-prefix 'kubectl -n slurm exec pod/slurm-controller-0 --' \
  --slurm-exec-prefix 'kubectl -n slurm exec deploy/slurm-login --'
```

> Hard placement controller 目前不是 `deploy-2.sh` 的預設常駐元件。它只處理 held pending jobs，會排除 `DRAIN` / `DOWN` / `NOT_RESPONDING` 節點，並依 `/healthz` 的 `n_actions` 自動修剪 node list 以符合 checkpoint topology。目前 live checkpoint 是 1 node × 1 GPU，因此只能 hard-bind 到一個有效 placement slot；若要讓 DSAC 在兩台 GPU worker 或 2×2 cluster 中真正選擇，必須部署相同 topology 訓練出的 checkpoint。

預設行為（k3s overlay）：

- `gpu.enabled=true`：在 `gpu-operator` namespace 放 device-plugin-config ConfigMap + 跨節點 labeler Job
- `monitoring.enabled=true`：Prometheus + Alertmanager + Grafana + kube-state-metrics + slurm-exporter（namespace `monitoring`）
- `storage.enabled=true` + `nfsServer=192.168.0.111`：NFS subdir provisioner + StorageClass `slurm-shared-nfs` + 20Gi RWX PVC

LAN IP 不一樣時用 `VALUES_FILE=<your-values.yaml>` 或 Helm values 檔調整 `storage.nfsServer`。GPU Operator 使用 `driver.enabled=false` 與 `toolkit.enabled=false`，因為 host 已經由 `setup-linux-gpu.sh` 裝好驅動與 NVIDIA Container Toolkit。

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

## 5. 訓練與評估

> 需要 `.venv-m11`（含 PyTorch），並從 repo 根目錄以 `PYTHONPATH=.` 執行（確保 `sim/`、`services/`、`eval/` 可被找到）。以下為目前的最終工作流。

### 5.1 模擬訓練（目前拓樸 = 2×1，obs_dim=168 / n_actions=33，預設 RDSAC）

目前論文（§5.8）採用的產出模型以 **fairness reward** 訓練：`mo` 完成項（−JCT/S）＋凸公平項（`--fairness-coef 5.0`，壓尾端）＋節點均衡 potential shaping（`--balance-coef 5.0`），並在具**共置干擾**的環境（`--interference 0.3`，實際執行時間隨同卡共置數變慢）下學習「打包 vs 干擾」的權衡。`train_aimix_seeds_fair.sh` 一次訓練 3 臂 × 16 seed：

```bash
# 產出 SAC / RDSAC-mean / RDSAC-cvar × 16 seed → runs/ckpts_aimix16_fair/
# 已用 scripts/gpu-toggle.sh release 釋出本機 4070 時可 DEVICE=cuda；否則 CPU。
DEVICE=cuda STEPS=100000 MAX=4 SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57" \
    bash eval/scripts/train_aimix_seeds_fair.sh
# → runs/ckpts_aimix16_fair/{sac,rdsac_mean,rdsac_cvar}_s{42..57}.pt（48 個 checkpoint）
```

底層等價的單臂指令（可自行調風險/critic 家族）：

```bash
PYTHONPATH=. .venv-m11/bin/python -m services.rl_scheduler.sim_train \
    --n-nodes 2 --gpus-per-node 1 --hetero-cluster --trace aimix \
    --n-jobs 50 --total-steps 100000 --curriculum --device cuda \
    --fixed-alpha --init-alpha 0.05 \
    --fairness-coef 5.0 --balance-coef 5.0 --interference 0.3 \
    --risk-mode cvar \
    --out-dir runs/dsac_2x1_$(date +%Y%m%d)
# vanilla SAC：加 --no-iqn；風險中立 RDSAC：--risk-mode mean
```

### 5.2 實機微調（RLPD，忠於 Ball et al. 2023）

以 shadow-safe 的 `live_daemon` 旁觀收集真實 transition（記錄 Slurm 實際落點 + **sacct 真 JCT**，不干擾生產），再做 RLPD 微調（對稱 50/50 offline/online、LayerNorm 集成 critic、fixed-α 避免離散 SAC 溫度發散）。`live_daemon` 的 obs 直接重用 `gym_env` 的 canonical 特徵抽取（單一真相源，避免 168-d 漂移）；reward 用 sacct `End−Submit`（避開 `MinJobAge` squeue 滯留造成的 JCT 灌水）。

```bash
# 1) 旁觀收集真實 168-d online-log（serve :8003 跑 rdsac_cvar 當 behavior + daemon shadow 記錄）
#    16h 時間界定迴圈送 aimix batch → shadow_logs/transitions_*.jsonl（RLPD 的 raw --online-log）
DURATION_S=57600 N_JOBS=30 bash eval/scripts/collect_aimix_onlinelog.sh

# 2) 16-seed RLPD 微調：各 warm-start rdsac_cvar_sXX（同一 CK 目錄的 base），共用同一份真 online-log。
#    offline sim prior 用 --hetero-cluster（對齊 base/online-log 的 gpu one-hot + free_ram_ratio）；
#    reward 維持 jct_aligned（RLPD critic 從頭訓練，需 offline↔online reward 一致，online-log 記 −JCT/1000）。
#    CPU（本機 4070 被 Slurm/DRA 佔用）；~240s/seed。
#    §5.8 產出模型：CK=runs/ckpts_aimix16_fair（暖啟動自 fairness RDSAC-cvar base）。
CK=runs/ckpts_aimix16_fair \
ONLINE_LOG=shadow_logs/transitions_20260814-203143.jsonl \
    bash eval/scripts/train_rlpd_aimix16.sh
# → runs/ckpts_aimix16_fair/rlpd_cvar_s{42..57}.pt（16 個，serve/eval 可載入）
```

### 5.3 混合工作負載實機評估（早期 campaign，論文表 6：125 工作／8 seed）

由 30% BERT 推論 / 30% ResNet-50 訓練 / 30% Qwen 微調 / 10% 矩陣運算 組成**混合真實工作負載**，跨 8 workload seed，輸出 7 指標（平均 JCT / P95 / P99 / Makespan / GPU 利用率 / Slowdown / SLA 違反率）與 Holm 校正配對 ΔJCT%。此為論文表 6 的 placement-only（RL 綁節點、順序由 Slurm 決定）early campaign；最終 headline 結果見 §5.5（排序致動）。

```bash
# ── Step 1：重訓 3 臂 × 8 seed → /tmp/lckpts_aimix/（早期 mo reward，非 §5.1 的 fairness 版）──
# 已用 scripts/gpu-toggle.sh release 釋出本機 4070 時可 DEVICE=cuda；否則預設 CPU。
DEVICE=cuda STEPS=70000 MAX=6 SEEDS="42 43 44 45 46 47 48 49" \
    bash eval/scripts/train_aimix_seeds.sh
# → /tmp/lckpts_aimix/{sac,rdsac_mean,rdsac_cvar}_s{42..49}.pt（24 個 checkpoint）

# ── Step 2：六臂實機評估（自動換 learned/fcfs/backfill 三套 slurm.conf，結束以 trap 還原）──
# 需 4070+3080 皆在叢集；gpu-toggle release 後須先 restore 並確認 MPS 恢復。
SEEDS="42 43 44 45 46 47 48 49" N_JOBS=30 ROUNDS=3 OVERSUB=2.0 \
    bash eval/scripts/run_aimix6.sh
# → runs/aimix6_<stamp>_TABLES.md（論文表 6：7 指標 × 6 臂，已含 Holm 校正 ΔJCT%）
```

### 5.4 重載六臂實機評估（placement-only，論文表 6b：150 工作／16 seed，基準 = backfill，無 score）

在 §5.7 ceiling 分析指出 headroom 開窗的重載 regime（n_jobs≈150，2-GPU）複核學習式排程。**六臂**：FCFS / Backfill（Slurm-native，剝除 Lua）+ SAC / RDSAC-mean / RDSAC-cvar / RLPD（學習式落點）。**score 已從評估項目移除**；主要顯著性檢定為 **seed-level 配對 ΔJCT% vs backfill**（backfill = Slurm 預設生產排程器，當基準）。三套 slurm.conf 自動切換，結束以 trap 還原。

```bash
# 前置：本地 168-d serve on :8003（eval 逐臂 /reload 對應 checkpoint）
SHADOW_MODE=true PYTHONPATH=. .venv-m11/bin/python -m services.rl_scheduler.serve \
    --policy-dir /tmp/aimix_eval_policy --port 8003 &   # policy-dir 放任一 168-d ckpt 當初始

# 先 SMOKE 驗 wiring（6 jobs / 1 seed，短），再全量
SMOKE=1 bash eval/scripts/run_heavy150_aimix_5arm.sh

# 全量：6 臂 × 16 seed × 150 jobs（real-CUDA；learned 輪帶 --no-score）
CK=runs/ckpts_aimix16 bash eval/scripts/run_heavy150_aimix_5arm.sh
# → runs/heavy150aimix_<stamp>_TABLES.md（JCT/Makespan/P95/P99 + 對 backfill 的 ΔJCT% / seed_t p）
```

> **節點順序 load-bearing。** `GPU_NODES` 內 index 0 須為快卡（4070），與訓練時的 `node_speeds` 一致；顛倒會使學習臂系統性放到慢卡、結果失真。預設 `slurm-worker-gpu-rtx4070-0,slurm-worker-gpu-rtx3080-0` 已正確。

### 5.5 排序致動實機評估（論文 §5.8 headline：ordering-only，真實 CUDA + poisson 負載掃描）

論文的最終結果：讓 RL 掌握**派遣順序**（而非只綁節點）。前展服務中的策略取得完整派遣順序（arrival-aware、rolling top-16，與線上 select 介面一致），再以固定 Slurm `Priority` 交由 Slurm 自身 in-process backfill 排程器致動並自由放置——RL 掌握*順序*、Slurm 掌握*放置與時機*。在真實 CUDA aimix、poisson 到達下做 **oversub=2/4/6 三點負載掃描**，證明排序紅利隨佇列深度成長（淺載與 Backfill 打平、中／深載顯著勝 Backfill 約 −10%～−13%，平均與 P99 同勝）。

```bash
# 前置：本地 168-d serve on :8003（eval 逐臂 /reload 對應 checkpoint）
SHADOW_MODE=true PYTHONPATH=. .venv-m11/bin/python -m services.rl_scheduler.serve \
    --policy-dir /tmp/aimix_eval_policy --port 8003 &   # policy-dir 放任一 168-d ckpt 當初始

# 三點負載掃描 × 10 seed × 6 臂（fcfs/backfill/sac/rdsac_mean/rdsac_cvar/rlpd_cvar）
# 自動切換 fcfs/main 兩套 slurm.conf，結束以 trap 還原；REAL_WORKLOAD=1 用真實 AiMix GPU 工作
OVERSUBS="2 4 6" SEEDS="42 43 44 45 46 47 48 49 50 51" REAL_WORKLOAD=1 \
    CK=runs/ckpts_aimix16_fair \
    bash eval/scripts/run_step3_prio.sh
# → runs/step3prio_<stamp>/ov{2,4,6}/{arm}_s{seed}.json

# 逐負載點配對統計（ΔmeanJCT% / ΔP95 / ΔP99 / P99<bf / Wilcoxon）
PYTHONPATH=. .venv-m11/bin/python -m eval.scripts.aggregate_step3 \
    runs/step3prio_<stamp>/ov6

# 跨負載點的 headroom 曲線（arm × oversub 的 ΔmeanJCT%）
PYTHONPATH=. .venv-m11/bin/python -m eval.scripts.aggregate_step3_sweep \
    runs/step3prio_<stamp>
```

> **節點順序 load-bearing**（同 §5.4）：`NODES` index 0 須為快卡（4070）。**config 還原脆弱**：若中途以 SIGTERM 打斷腳本，trap 可能來不及還原 `slurm.conf`（卡在 fcfs），需從 `/tmp/slurm.conf.s3porig.*` 備份手動還原並重啟 controller。

### 5.6 訓練 flags 對照

| Flag | 說明 | 預設 |
|------|------|------|
| `--curriculum` | n_jobs 從 10→30→50 漸進 | 關 |
| `--no-per` | 停用 Prioritized Experience Replay | PER 開 |
| `--no-potential-shaping` | 停用 per-step 等待時間 shaping | Shaping 開 |
| `--no-iqn` | 改用 scalar twin-Q critic（vanilla SAC）；不加則為預設的 IQN distributional critic | IQN/RDSAC 開 |
| `--risk-mode` | RDSAC 風險扭曲：`mean`（risk-neutral）/`cvar`/`wang`/`cpw`/`msd`（僅 IQN 生效） | `mean` |
| `--fairness-coef` | 凸（平方）per-job JCT 懲罰，壓尾端（改變目標，非 optimum-preserving） | `0`（生效版用 5.0） |
| `--balance-coef` | 節點 free-MPS 均衡 potential shaping（多節點才生效） | `0`（生效版用 5.0） |
| `--interference` | 環境動力學：同卡共置每多一個工作，實際執行時間 ×(1+k·此值) | `0`（生效版用 0.3） |

### 5.7 執行單元測試

```bash
PYTHONPATH=. .venv-m11/bin/python -m pytest sim/tests/ -q
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

---

# 🏗️ System Architecture

<img width="4400" height="2280" alt="圖片" src="https://github.com/user-attachments/assets/5d27ca15-525c-4936-a447-252a8a081934" />

> 完整架構圖請看 [`architecture.html`](assets/architecture.html)

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
