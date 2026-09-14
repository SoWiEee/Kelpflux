<div align="center">

# 〰️ Kelpflux
 
### Elastic Slurm scheduling on Kubernetes for shared GPU AI workloads.
 
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SoWiEee/Kelpflux)
![Slurm](https://img.shields.io/badge/Slurm-23.11-2E86AB?logo=data:image/svg+xml;base64,)
![Kubernetes](https://img.shields.io/badge/Kubernetes-1.35-326CE5?logo=kubernetes&logoColor=white)
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

1. **能不能用學習式、風險敏感的策略補上那層缺失的智慧？** 我們以分散式深度強化學習（RDSAC：discrete SAC + IQN，並以 CVaR 風險量度直接優化回報分布的尾端）從 oldest-first 前 16 個 pending jobs 中選擇 `(job, node, GPU)`。production reorder daemon 將選出的 job 順序寫入 Slurm Priority，Slurm 負責一般 unheld job 的實際 placement；另有 held-job REST controller，可將所選節點寫入 `ReqNodeList` 後 release 工作。Slurm 23.11.4 的 REST v0.0.39 round-trip 已通過 canary 驗證，2×1 Helm overlay 預設啟用該 controller，但僅處理使用者明確以 held 狀態提交的 GPU job。此學習式策略與 DRA 並非競爭，而是**互補**：DRA 提供 GPU/MPS 存取機制，DRL 負責排序或 held-job placement。
2. **這套智慧要以什麼形式、在什麼條件下才真的有用？** 早期只讓 RL **綁定節點**（placement-only、工作*順序*仍由 Slurm 決定）的實機路徑下，學習式策略僅與 Slurm 打平、且以顯著尾端代價換得——但這是**致動路徑**的限制，而非策略無法貢獻。當改讓 RL 掌握**派遣順序**、並以一條可落地的**非阻塞、失效安全的原生致動路徑（Option B：常駐程序週期性重排當前佇列、寫入 Slurm `Priority`，交由 Slurm 原生 backfill 致動）**整合後，在真實 CUDA、poisson 到達的三點負載掃描（oversub=2／4／6）下，學習式策略在**平均 JCT 與尾端 P99 皆穩健顯著勝過生產 Slurm Backfill**（平均約 −11%～−13%，深載尾端 P99 更達 −19%～−22%；配對 Wilcoxon *p*≤0.006、P99 於 10/10 seed 勝過 Backfill）。此實機確認了模擬天花板分析對「ordering headroom 隨負載上升」的預測，並精確界定效益條件：**RL 須掌握*排序*槓桿、致動路徑須原生且連續、負載須足以形成可重排的 backlog**。支撐此結論的是一套**模擬到實機（sim-to-real）評估方法學**——抗跑序漂移的交錯輪轉、多 seed 配對顯著性（Wilcoxon）與信賴區間、兼顧平均與尾端（p95／p99／CVaR）。

## Getting Started

### Requirements

The tested end-to-end deployment targets Linux, k3s, Helm, Docker, and NVIDIA GPU workers. A host NVIDIA driver and GPU runtime must be available; configure an NFS server if using shared storage. The exact cluster versions and hardware are listed in [cluster.md](docs/cluster.md).

The scripts below automate the repository's k3s deployment. Other Kubernetes distributions may require adapting the values and GPU/storage integration; see the [user guide](docs/tutorial.md) before deploying.

### Deploy

From the repository root, point `KUBECONFIG` at the target cluster and run:

```bash
export KUBECONFIG=~/.kube/config
bash scripts/deploy-1.sh
bash scripts/deploy-2.sh
bash scripts/verify-live.sh
```

`deploy-1.sh` prepares the host resources and application images. `deploy-2.sh` installs or upgrades the platform Helm release, NVIDIA GPU components, and scheduler services. `verify-live.sh` checks the resulting cluster, including GPU scheduling and monitoring.

`deploy-2.sh` layers `chart/values-2x1.yaml` over `chart/values-k3s.yaml` for the RTX 4070 + RTX 3080 cluster. Review the k3s storage settings before applying it elsewhere; set `TOPOLOGY_VALUES_FILE` to another overlay to target a different topology.

### Scheduler behavior

The 2×1 profile enables DRL priority reordering for ordinary unheld, explicit-MPS GPU jobs; Slurm still chooses the node and dispatch time. The profile also enables the separate REST placement controller for jobs explicitly submitted with `--hold`; it writes the selected node constraint before release. CPU worker scaling is bounded by the capacity declared in the selected values overlay. See [scheduler.md](docs/scheduler.md) for the control boundaries.

### Check the deployment

```bash
kubectl -n slurm get pods
kubectl -n slurm exec deploy/slurm-login -- sinfo
kubectl -n monitoring port-forward svc/grafana 3000:3000
```

Grafana is then available at `http://127.0.0.1:3000`. See [monitoring.md](docs/monitoring.md) for dashboards and metrics.

### 訓練與評估

以下指令供研究重現使用。Python 環境需包含 PyTorch，並在 repository 根目錄執行。模擬訓練可用 CPU 或可用的 CUDA GPU；實機資料收集與評估會提交／取消 Slurm 工作、切換 Slurm 設定並重啟 controller，**只能在專用測試叢集執行，不要在共用或正式叢集執行**。

#### 模擬訓練

目前 2×1 模型的觀測維度為 168、動作數為 33。下列腳本會訓練 SAC、RDSAC-mean、RDSAC-CVaR 三種方法，各使用 16 個 seed：

```bash
DEVICE=cpu STEPS=100000 MAX=8 \
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57" \
    bash eval/scripts/train_aimix_seeds_fair.sh
# 有可用 GPU 時可改 DEVICE=cuda MAX=4
# checkpoints 輸出至 runs/ckpts_aimix16_fair/
```

單一模型可直接呼叫模擬訓練器：

```bash
PYTHONPATH=. .venv-m11/bin/python -m services.rl_scheduler.sim_train \
    --n-nodes 2 --gpus-per-node 1 --hetero-cluster --trace aimix \
    --n-jobs 50 --total-steps 100000 --curriculum --device cpu \
    --fixed-alpha --init-alpha 0.05 \
    --fairness-coef 5.0 --balance-coef 5.0 --interference 0.3 \
    --risk-mode cvar --out-dir runs/dsac_2x1_$(date +%Y%m%d)
```

#### RLPD 實機微調

需要先以 `collect_aimix_onlinelog.sh` 在專用叢集旁觀收集 16 小時的真實 transition；此步驟會執行實際 GPU 工作：

```bash
DURATION_S=57600 N_JOBS=30 bash eval/scripts/collect_aimix_onlinelog.sh
```

指定收集完成後的 log 檔，從 RDSAC-CVaR checkpoint 微調 16 個 seed：

```bash
CK=runs/ckpts_aimix16_fair \
ONLINE_LOG=shadow_logs/transitions_<收集時間戳>.jsonl \
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57" \
    bash eval/scripts/train_rlpd_aimix16.sh
# 產物：runs/ckpts_aimix16_fair/rlpd_cvar_s{42..57}.pt
```

#### 150-job 實機評估

先在一個終端啟動相容的 168 維 policy server（以既有 checkpoint 作為初始模型）：

```bash
mkdir -p /tmp/aimix_eval_policy
cp runs/ckpts_aimix16_fair/rdsac_cvar_s42.pt /tmp/aimix_eval_policy/dsac.pt
SHADOW_MODE=true PYTHONPATH=. .venv-m11/bin/python -m services.rl_scheduler.serve \
    --policy-dir /tmp/aimix_eval_policy --port 8003
```

在另一個終端執行三種 oversub、10 個 seed、每次 150 jobs 的評估。`REAL_WORKLOAD=1` 使用 AiMix 的 BERT、ResNet、Qwen 與 cuBLAS 工作負載；比較 FCFS、Slurm Backfill、SAC、RDSAC-mean、RDSAC-CVaR 與 RLPD-CVaR：

```bash
ACTUATION=reorder OVERSUBS="2 4 6" N_JOBS=150 \
SEEDS="42 43 44 45 46 47 48 49 50 51" REAL_WORKLOAD=1 \
CK=runs/ckpts_aimix16_fair bash eval/scripts/run_step3_prio.sh
# 結果輸出至 runs/step3prio_<時間戳>/
```

評估結束後彙整結果：

```bash
PYTHONPATH=. .venv-m11/bin/python -m eval.scripts.aggregate_optB_deploy \
    runs/step3prio_<時間戳>
```

訓練與評估旗標、方法細節及結果解讀見 [eval-writeup.md](docs/eval-writeup.md)；排程器實作與實機控制邊界見 [scheduler.md](docs/scheduler.md)。

## Development and Research

- [Scheduler implementation and configuration](docs/scheduler.md)
- [Evaluation methods and results](docs/eval-writeup.md)
- [Paper draft](docs/paper.md)
- [Cluster architecture and operations](docs/cluster.md)

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
