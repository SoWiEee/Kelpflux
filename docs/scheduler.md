# Scheduler Production Spec

本文件描述目前 Kelpflux 上線中的排程與 placement 規格。範圍包含 Slurm 內建排程、`job_submit.lua` submit-time scoring、runtime predictor、weight tuner、DSAC live scheduler、Kubernetes / NVIDIA GPU stack、fallback policy 與可觀測性。歷史開發階段、實驗路線圖與已淘汰設計不再列入本規格。

Kelpflux 的主要貢獻不是取代 Slurm，而是在 Slurm 前後加入一層可訓練、可觀測、可回退的 ML placement control plane：

- 將使用者的 CPU / GPU / VRAM / MPS 需求正規化成 Slurm 能執行的 placement contract。
- 以 submit-time score 作為穩定 fallback，避免 RL service 不可用時影響提交。
- 以 DSAC 在 live traffic 上介入 pending job 的排序，並在 hard placement path 中把模型輸出的 `node_j` / `gpu_k` 轉成 Slurm 原生 placement constraint。
- 以 Slurm GRES、`select/cons_tres`、Kubernetes worker pool、NVIDIA runtime 與 MPS control daemon 共同完成硬體資源配置。
- 以 Prometheus / Grafana / OpenTelemetry 暴露決策、queue、GPU、MPS、placement intent 與 hard placement action，讓非開發者也能看懂系統如何把 job 放到資源上。

## 1. 上線架構

```text
sbatch / srun
    |
    v
Slurm job_submit.lua
    |-- submit helper: 補齊 partition / memory / qos
    |-- score function: MPS / VRAM / fragmentation / runtime signal
    |-- runtime predictor: 可選，用於短工優先與 walltime 建議
    |-- weight tuner: 可選，載入目前最佳 score 係數
    |-- DSAC scheduler: live 模式時可回傳 priority boost + placement action
    v
Slurm priority + backfill + select/cons_tres
    |
    |-- optional hard placement controller: /act -> ReqNodeList -> release
    v
Kubernetes worker pool + NVIDIA runtime + Slurm GRES
    v
physical CPU / GPU / MPS slots
```

核心原則：

- Slurm 仍是最終資源分配與 job lifecycle owner。
- `job_submit.lua` 在提交時調整 priority / metadata / placement contract，不阻塞 Slurm 的基本行為。
- DSAC live scheduler 以 priority boost 介入排序；hard placement controller 會把 `node_j` / `gpu_k` action 轉成 Slurm `ReqNodeList` + release。
- Submit-time path 不在 `job_submit.lua` 內覆蓋 node allocation；正式 hard placement path 透過 Slurm 原生 hold-release 與 `select/cons_tres` 執行。
- 任何 DSAC 失敗都 fallback 到 score / Slurm 原生排程。
- GPU live migration 不在上線規格內；若要處理 running job，只能走 application-level checkpoint + requeue。

## 2. Job Submit 決策流程

以下 Mermaid 圖示意一個 job submit 後，系統如何補齊 job metadata、計算排序訊號、呼叫 DSAC，最後交給 Slurm 分配硬體資源。

```mermaid
flowchart TD
    A["User submits job<br/>sbatch / srun"] --> B["Slurm controller<br/>loads job_submit.lua"]

    B --> C["Submit helper"]
    C --> C1["Fill missing partition<br/>CPU or GPU pool"]
    C --> C2["Fill missing memory<br/>from CPU/GPU request"]
    C --> C3["Fill default QoS<br/>normal or account rule"]

    C1 --> D["Build scheduling signals"]
    C2 --> D
    C3 --> D

    D --> E["Score function"]
    E --> E1["MPS fit<br/>requested slots vs 100-slot GPU"]
    E --> E2["VRAM fit<br/>requested tier vs node tier"]
    E --> E3["Fragmentation penalty<br/>partial MPS usage cost"]
    E --> E4["Runtime signal<br/>optional predictor"]
    E1 --> F["priority_delta<br/>scoreGain * score"]
    E2 --> F
    E3 --> F
    E4 --> F

    D --> G{"RL scheduler enabled?"}
    G -- "no" --> H["Use score priority only"]
    G -- "yes" --> I["POST /decide<br/>rl-scheduler"]
    I --> J{"DSAC returns boost?"}
    J -- "selected" --> K["Add priority_boost"]
    J -- "abstain / no-op / error" --> H

    F --> L["Final job priority"]
    H --> L
    K --> L

    L --> M["Slurm priority queue"]
    M --> N["Backfill scheduler"]
    N --> O["select/cons_tres"]
    O --> P{"Requested hardware"}

    P -- "CPU job" --> Q["CPU worker pod<br/>slurm-worker-cpu-*"]
    P -- "GPU job" --> R["GPU worker pod<br/>slurm-worker-gpu-*"]
    R --> S["NVIDIA runtime<br/>physical GPU access"]
    R --> T["Slurm GRES<br/>gpu + mps slots"]

    Q --> U["Job starts"]
    S --> U
    T --> U

    U --> V["Accounting / metrics"]
    V --> W["Prometheus + Grafana<br/>queue, GPU, DSAC decisions"]
```

決策重點：

- helper 只補缺值，不覆蓋使用者已指定的 partition、memory 或 QoS。
- score function 產生穩定的 submit-time priority delta，是 DSAC 不可用時的主要 fallback。
- DSAC live scheduler 在 submit-time path 會加 priority boost，並回傳 `node_j` / `gpu_k` 作為 placement intent；hard placement controller 會在 held job path 中把同一類 action 落成 `ReqNodeList`。
- GPU job 的硬體資源由 Kubernetes worker pod、NVIDIA runtime、Slurm GRES 與 MPS slot 共同約束，這是目前客製 placement 的執行層。

## 3. 完整排程與客製 Placement Stack

Kelpflux 把「排程」拆成兩個互相銜接的問題：先決定哪個 pending job 應該更早被 Slurm 考慮，再用一組可被 Slurm / Kubernetes / NVIDIA stack 執行的 placement contract 把 job 放到硬體上。

```mermaid
flowchart LR
    U["User intent<br/>sbatch flags"] --> H["Submit helper<br/>normalize request"]
    H --> C["Placement contract<br/>partition / constraint / GRES / MPS / memory / QoS"]
    C --> S["Score fallback<br/>MPS fit / VRAM fit / fragmentation / runtime"]
    C --> R["DSAC live policy<br/>queue + GPU + MPS snapshot"]
    S --> Q["Final priority"]
    R --> Q
    R --> I["Placement action<br/>job_i, node_j, gpu_k"]
    I --> HC["Hard placement controller<br/>held queue only"]
    HC --> NRL["ReqNodeList + release"]
    Q --> P["Slurm priority queue<br/>multifactor + backfill"]
    C --> P
    NRL --> P
    P --> T["select/cons_tres<br/>TRES / GRES allocation"]
    T --> K["Kubernetes worker pool<br/>CPU/GPU StatefulSets"]
    K --> N["NVIDIA GPU Operator<br/>device plugin + MPS"]
    N --> G["Physical resources<br/>CPU cores / GPU / MPS slots"]
    I --> M["Prometheus / Grafana<br/>visualized decision"]
    HC --> M
    G --> M
```

### 3.1 Placement Contract

Submit-time path 中，placement contract 由 submit helper、使用者 sbatch flags 與 Slurm config 共同決定；hard placement path 會在 held job release 前額外寫入 `ReqNodeList`：

| Contract 欄位 | 來源 | 執行者 | 作用 |
|---------------|------|--------|------|
| `partition` | 使用者指定或 submit helper 補齊 | Slurm + Kelpflux operator | 決定 CPU / GPU worker pool，例如 `cpu`、`gpu-rtx4070` |
| `constraint` / features | 使用者指定 | Slurm node feature matching | 約束 VRAM tier、GPU 型號或節點特徵 |
| `ReqNodeList` | hard placement controller | Slurm scheduler / select plugin | 將 DSAC 選到的 worker node 轉成 hard placement constraint |
| `gres/gpu` | `--gres` / `tres_per_node` | Slurm GRES + NVIDIA device plugin | 配置 GPU 類型與數量 |
| `gres/mps` | `--gres` / `tres_per_node` | Slurm GRES + NVIDIA MPS control daemon | 配置單 GPU 上的 MPS slot |
| `memory` | 使用者指定或 helper 估算 | Slurm cgroup / Kubernetes pod resources | 避免 job 在 placement 後因 memory 不足失敗 |
| `qos` / priority | 使用者、helper、score、DSAC | Slurm priority queue | 決定 pending jobs 的排序與 backfill 機會 |
| worker pool size | Slurm pending/running state | Kelpflux operator | pending jobs 觸發 scale-up；running jobs 阻止 scale-down |

換句話說，Kelpflux 的客製 placement 不是單一 API call，而是一條可被多層系統共同執行的約束鏈：job submit 時建立 contract，Slurm 根據 contract 選資源，hard placement controller 可對 held jobs 寫入 `ReqNodeList`，operator 確保 worker pool 存在且不縮掉 running jobs，NVIDIA stack 提供 GPU / MPS isolation。

### 3.2 DSAC 在 Placement 中的角色

DSAC action space 已經是 placement-aware：模型輸出的 flat action 會被解碼成 `(job_i, node_j, gpu_k)`，action mask 只允許選擇 MPS free slots 足夠的 placement。

| DSAC 輸出 | Submit-time live path | Hard placement controller |
|-----------|-----------------------|---------------------------|
| `job_i` | 若選中的 job 是正在 submit 的 job，回傳 positive `priority_boost` | 對 held pending queue 中被選到的 job 執行 placement |
| `node_j` | 記錄為 placement intent，用於 metrics、Grafana 與訓練分析 | 映射到可用 Slurm GPU worker node，寫入 `ReqNodeList=<node>` |
| `gpu_k` | 記錄為 placement intent，用於觀察模型偏好的 GPU slot | 目前 live worker 是 1 GPU / node，因此 `gpu_k=0`；多 GPU node 需要對應 GPU/MPS slot topology |
| `priority_boost` | 加到 `job_desc.priority`，讓 Slurm 更早考慮該 job | 不使用；controller 透過 hold-release 控制何時進入 Slurm placement |
| `abstain` / no-op | guardrail 觸發時不 boost，回到 score + Slurm | 不更新 job，保持 held/pending，等待下一輪或人工處理 |

Submit-time path 仍是低風險預設；hard placement controller 是正式可用的 Slurm-safe placement path，適合需要 DSAC 實際指定 worker / GPU / MPS contract 的實驗與受控上線。它不在 `job_submit.lua` 裡阻塞，而是只處理 held pending jobs，先讓使用者提交 `sbatch --hold ... --gres=mps:N`，再由 controller 寫入 Slurm 原生約束並 release。

#### 演算法：SAC 與 RDSAC 的公式與運作

底層 agent（`services/rl_scheduler/dsac.py`）以一個 `use_iqn` flag 在 **vanilla 離散 SAC**（`use_iqn=False`）與 **risk-sensitive distributional SAC / RDSAC**（`use_iqn=True`，預設）之間切換。flag 存在 checkpoint 裡，train / eval / serve 自動偵測。兩者共用同一套**離散最大熵 RL** 設定，只差 critic 家族與有無 risk distortion。

**共用設定（兩者相同）**

placement 是離散動作，故 actor 是**顯式 masked categorical** `π(a|s;φ)`（非法動作 logits 設 −1e9）。狀態 `s` 為 160 維 obs，動作 `a` 從 `valid(s)` 中取。最大熵目標：最大化

```
J = Σ_t E[ r_t + α·H(π(·|s_t)) ]，  H = −Σ_a π(a|s)·log π(a|s)
```

temperature `α` 由 SAC 自動調（離散 target entropy `H̄ = ratio·log(n_valid)`）：

```
L_α = E_s Σ_a π(a|s)·[ −α·( log π(a|s) + H̄ ) ]
```

target network 用 soft update（τ=0.005）。差別從這裡開始：

**(A) vanilla SAC（`use_iqn=False`）— scalar twin-Q**

critic 是一對 scalar Q `Q_θ1, Q_θ2`（twin，取 min 抑制高估）。soft 狀態值與 MSE soft-Bellman target：

```
V(s') = Σ_a' π(a'|s')·[ min_i Q_θi(s',a') − α·log π(a'|s') ]
y     = r + γ·(1−d)·V(s')                      # 用 target net
L_Q   = Σ_i E[ ( Q_θi(s,a) − y )² ]            # MSE soft-Bellman
```

actor 目標（最小化）：

```
L_π = E_s Σ_a π(a|s)·[ α·log π(a|s) − min_i Q_θi(s,a) ]
```

回報只是**點估計**，沒有分布、沒有風險概念 — 對應 Christodoulou 2019 的離散 SAC。

**(B) RDSAC（`use_iqn=True`，預設）— 雙分布 IQN critic + risk distortion**

依 Ma et al. 2020/2025〈DSAC: Distributional Soft Actor-Critic for Risk-Sensitive RL〉(arXiv:2004.14547) §4.1 的離散動作轉寫。把 scalar Q 換成 **IQN**(Dabney et al. 2018) 的**分位數分布**，並把 soft return 拆成兩個 head：reward 回報分布 `Z_R` 與 entropy 回報分布 `Z_H`（共用 trunk、cosine 嵌入分位數 `τ~U(0,1)`）：

```
θ_τ(s,a) = critic(s, a, φ_cos(τ))             # 第 τ 分位數
Q(s,a)   = E_τ[Z_R(s,a)] + α·E_τ[Z_H(s,a)]    # α-external 組合值
```

分布式 soft-Bellman target（`a' ~ π(·|s')`，twin 取 min）：

```
T Z_R(s,a) = r + γ·(1−d)·Z_R(s',a')                       # reward 回報
T Z_H(s,a) = −log π(a'|s') + γ·(1−d)·Z_H(s',a')           # 純 entropy 回報
```

critic 用 **quantile Huber loss**（κ=1）回歸 `Tθ` 對 `θ`，對分位數對 `(τ_i, τ_j)` 做不對稱加權 `|τ_i − 1{δ<0}|`，`δ = Tθ_τj − θ_τi`。

**risk distortion `ρ` 只作用在 reward 分布 `Z_R`**（entropy 保持期望），由 `risk_mode` 選：

```
mean    : ρ[Z_R] = E[Z_R]                                  # risk-neutral
cvar_β  : ρ[Z_R] = (1/β)∫₀^β F⁻¹_{Z_R}(u) du               # 下尾 β 比例分位數平均
          ≈ 取最低 β 比例的分位數取平均  （β = risk_beta，預設 0.25）
```

（`wang / cpw / msd` 為另三種 distortion，見 `services/rl_scheduler/distortion.py`。）distortion 同時進 **actor 目標**與 serve 回傳的 `value`：

```
L_π   = E_s Σ_a π(a|s)·[ α·log π(a|s) − ρ[Z_R(s,a)] − α·E[Z_H(s,a)] ]   # actor loss
value = Σ_a π(a|s)·( ρ[Z_R(s,a)] + α·E[Z_H(s,a)] )                       # /decide 回傳、餵 §8.3 guardrail
```

`mean` 退化成穩定性導向的 distributional SAC（有分布、不規避風險）；`cvar` 偏好**下尾較不嚴重**的 placement，對應排程的 straggler / cold worker / long-tail runtime 風險。

**SAC vs RDSAC 一覽**

| 面向 | vanilla SAC | RDSAC |
|---|---|---|
| critic 家族 | scalar twin-Q | 雙 head IQN（`Z_R` + `Z_H`）|
| 回報表示 | 點估計 `Q(s,a)` | 分位數分布 |
| Bellman loss | MSE soft-Bellman | quantile Huber（κ=1）|
| risk 機制 | 無 | distortion `ρ[Z_R]`（cvar/wang/cpw/msd）|
| actor 目標 | `α logπ − min_i Q_i` | `α logπ − ρ[Z_R] − α E[Z_H]` |
| flag | `use_iqn=False` | `use_iqn=True`（預設）|

兩條路徑都帶 twin-Q（取 min）與 soft target update。`risk_mode=mean` 是 RDSAC 預設；要尾部感知（p95/p99 JCT、tail slowdown，見 `sim/metrics.py`）時改 `cvar`。

> **命名注意**：此處「DSAC」指**院內組裝的 distributional + 離散 SAC**（Christodoulou 離散 SAC + Dabney IQN critic + CVaR distortion 的組合），**不是** Duan et al. 2021 把回報建模成單一 Gaussian 的連續控制 DSAC，勿假設與該論文對等。兩種 critic 家族都使用 MLP trunk。

### 3.3 Submit 到硬體分配的時序

```mermaid
sequenceDiagram
    participant U as User
    participant L as job_submit.lua
    participant R as DSAC /decide + /act
    participant P as placement_controller
    participant S as Slurm queue
    participant O as Kelpflux operator
    participant K as K8s worker pool
    participant G as NVIDIA / MPS
    participant M as Metrics

    U->>L: sbatch / srun with CPU/GPU/MPS intent
    L->>L: Fill missing partition, memory, QoS
    L->>L: Compute score fallback
    L->>R: POST /decide with job + latest snapshot
    R-->>L: priority_boost + job_i/node_j/gpu_k + safety state
    L->>S: Submit final priority + placement contract
    P->>R: POST /act for held placement queue
    R-->>P: selected job_i/node_j/gpu_k
    P->>S: scontrol update ReqNodeList + release
    S->>S: Multifactor priority + backfill
    S->>O: Pending job reveals required pool
    O->>K: Scale CPU/GPU worker StatefulSet
    K->>G: RuntimeClass + device plugin + MPS control daemon
    S->>G: Allocate GRES gpu/mps and start job
    R->>M: decision/value/entropy/placement action
    G->>M: GPU utilization and MPS free slots
```

### 3.4 Hard Placement Controller

`services/rl_scheduler/placement_controller.py` 是 Slurm-safe hard placement 執行路徑，且**現在預設常駐啟用**（`rlScheduler.placementController.enabled=true`，由 `deploy-2.sh` 一併部署為 `rl-placement-controller` Deployment）。它透過 **slurmrestd（JWT，與 `snapshot_agent` 同一條認證路徑）** 輪詢 held queue 與 GPU worker state，呼叫 DSAC `/act`，把 action 解碼出的 `node_j` 映射到可用 worker，接著對選中的 held job 送一個 job-update：

```
POST /slurm/<v>/job/<job_id>   { "required_nodes": "<selected_node>", "priority": <INFINITE> }
```

`required_nodes` 即 `ReqNodeList`；`priority=INFINITE`（0xFFFFFFFF）等同 `scontrol release`——讓 multifactor plugin 重算優先序、解除 user hold。改走 REST 是為了避免在 service 容器裡塞 `squeue`/`scontrol` CLI、munge socket 或 `kubectl exec` RBAC。`shadow=false`（預設）才真的送 job-update；`shadow=true` 只記 log。

執行規則：

- 預設只處理 `JobHeldUser` / `JobHeldAdmin` pending jobs，避免碰到一般使用者已排隊的 job。
- 使用者的 GPU/MPS 需求必須在提交時已存在，例如 `sbatch --hold -p gpu-rtx4070 --gres=mps:10 ...`；controller 不在 release 後修改 GRES。
- 會排除 `DRAIN`、`DOWN`、`NOT_RESPONDING`、`FAIL` 的 worker node。
- 會讀 `/healthz` 的 `n_actions`，若 node list 大於 checkpoint topology 支援的 placement 數，會自動修剪並在 log 中提示需要重新訓練 / 部署更大 topology checkpoint。
- DSAC no-op 時不更新 job，維持 held/pending。

Live 手動驗證已確認 controller 能把 DSAC action 寫入 Slurm：job `131` 被更新為 `ReqNodeList=slurm-worker-gpu-rtx4070-1`，Slurm 隨後配置 `NodeList=slurm-worker-gpu-rtx4070-1`、`BatchHost=slurm-worker-gpu-rtx4070-1`、`TRES=gres/mps=10`。後續已修正 elastic operator worker lifecycle guard：只要 pool 內 `running_jobs > 0`，policy 會回 `running_jobs_block_scale_down`，scale action 層也會阻止 drain / replica patch，因此 hard placement job 不會再因 `pending_jobs=0` 被 operator scale-down eviction。

`services/rl_scheduler/live_daemon.py` 保留為研究 / legacy 原型；它以 `srun --jobid ... --nodelist ...` 嘗試直接執行，適合離線比較與 RLPD transition 蒐集。正式 hard placement path 是 `services/rl_scheduler/placement_controller.py`，因為它使用 Slurm 原生 hold-release 與 `ReqNodeList`，可被 Slurm priority、backfill、GRES/MPS accounting 和 operator lifecycle guard 正常約束。

### 3.5 訓練管線：sim 訓練 → RLPD 微調至真實環境

上面的 SAC / RDSAC **不是、也無法直接在 live cluster 上從頭訓練**。RL 要數十萬～數百萬個 transition，而真實 cluster 一個 placement 決策對應一個跑數分鐘～小時的 job，湊滿樣本要等數月；且訓練初期的隨機策略會直接破壞系統，真實環境又無法 `reset()` 重放同一條 trace 做配對比較。因此採 **sim-to-real 兩段式**：

```
① sim 大量訓練（便宜 / 安全 / 可 reset / 可配對評估）  →  checkpoint（warm start）
② live shadow-mode 部署，記錄真實 (obs, act, rew, …) 到 JSONL
③ RLPD 用真實資料把 sim checkpoint 微調成真實環境策略
```

| 階段 | 環境 | 做什麼 | 為何不能在 live 直接做 | 產出 | 程式 |
|---|---|---|---|---|---|
| **① sim 訓練** | `KubefluxSchedEnv`（gym） | 從頭訓練 100k–150k 步，curriculum + PER + n-step + shaping | live 一個 step = 一個跑數分鐘～小時的 job，湊樣本要數月 | checkpoint | `services/rl_scheduler/sim_train.py` |
| **② live 蒐集** | k3s + Slurm + GPU/MPS | shadow-mode 跑 checkpoint，`/act` 記錄真實 `(obs, act, rew, next_obs)` | 訓練初期隨機策略會破壞系統 → 只敢 shadow + fail-safe 回退 score | 真實 transition JSONL | `services/rl_scheduler/live_daemon.py` |
| **③ RLPD 微調** | 用 ② 的真實 JSONL + sim transition | 以 sim checkpoint 為 prior，混真實資料 fine-tune | 從零 RLPD = 退回 ① 的 sample-complexity 與破壞性探索 | 真實環境策略 | `services/rl_scheduler/rlpd_finetune.py` |

**目前 checkpoint 位置（2×1、168-dim）：**
- **① sim 訓練產出**：`runs/ckpts_aimix16_fair/{sac,rdsac_mean,rdsac_cvar}_s{42..57}.pt`（§5.1 fairness reward：`--reward-mode mo --fairness-coef 5.0 --balance-coef 5.0 --interference 0.3`，reward_scale=20000）。訓練腳本 `eval/scripts/train_aimix_seeds_fair.sh`。
- **② live 蒐集產出**：`shadow_logs/transitions_*.jsonl`（168-dim、sacct 真 JCT、rew=−JCT/1000）。蒐集腳本 `eval/scripts/collect_aimix_onlinelog.sh`。
- **③ RLPD 產出**：`runs/ckpts_aimix16_fair/rlpd_cvar_s{42..57}.pt`（暖啟動自同目錄 `rdsac_cvar` base、微調 reward `jct_aligned` −JCT/1000 以對齊 online-log）。腳本 `eval/scripts/train_rlpd_aimix16.sh`。

sim **不是 live 的替代品**，而是唯一能做大量學習的地方；它的產出是**「機制性洞察 + 一個可用的 warm start」**。對應有**兩種可轉移性差很多的主張**：sim 的**絕對 JCT/勝幅不轉移**（live 1×1 三方打平已印證），只有**機制性結論轉移**（如「auto-α 是壓垮 SAC 的 artifact」「分布式 critic 才是主因、CVaR 是尾部加成」）。完整模擬與實機數據見 `docs/eval-writeup.md`。

#### RLPD 是什麼、怎麼運作

**RLPD = Reinforcement Learning with Prior Data**（Ball et al. 2023, arXiv:2302.02948）。它要解的問題：sim 訓練好的 checkpoint 是個 warm start，但 sim≠real（train-serve skew），需要用真實資料**適應**，同時避免兩個失敗模式——(a) 真實資料太少時災難性遺忘 sim prior、(b) 純離線（只靠舊資料）無法超越資料本身。RLPD 的核心機制：

- **對稱取樣（symmetric sampling）**：維持兩個 buffer——**prior buffer**（sim transition）與 **online buffer**（live_daemon 收的真實 transition）。每個 gradient step **各取一半**組 batch。這讓 sim transition 留在**每個** batch 裡 → 防遺忘，又讓小量真實資料有槓桿。
- **高 UTD（update-to-data ratio）**：每收一筆真實 transition 做多次更新，把稀少的真實資料榨乾，因為真實樣本昂貴。
- **Critic LayerNorm**：論文最強調的一項——critic 加 LayerNorm 防止對 OOD 動作的值外推發散。本專案 critic 預設 `layer_norm=True`（`dsac.py`），符合論文。沿用 §3.2 的分布式 RDSAC critic（論文用 scalar SAC，此為專案擴充，與 RLPD 正交）。

> 關鍵理解：RLPD **不是丟掉 sim、重練一個**——sim prior 是每個 batch 的一半（對稱取樣），讓小量真實資料有槓桿又不遺忘。live trace collector 管線（`live_daemon.py` → JSONL → `rlpd_finetune.py`）已就緒，等拓樸匹配的多節點 checkpoint 後啟動。

**與原論文的符合度（已對程式碼核對）**：**每步更新的配方忠實照論文**——對稱取樣 50/50（`mixed_batch`，`online_ratio=0.5`）、critic LayerNorm（`layer_norm=True`）、高 UTD（預設 20）、**不加 CQL/顯式 pessimism**（論文明確主張不需要，程式也確實沒有）。但**訓練機制偏離**：論文 RLPD 是**真正線上**（更新迴圈持續用當前 policy 在真環境互動、新 transition 進 online buffer），本實作的 `rlpd_train` 迴圈**只有 `agent.update`、無 `env.step`**，兩個 buffer 在訓練前就凍結 → 實際上是「sim+real 混合 buffer 的**離線**微調」；且 online 資料目前是預收 shadow log 或用 score baseline 示範動作回放的 live trace（非學習 policy 自己跑的 on-policy 資料），critic 是 twin（2）而非論文的大 ensemble，並從 `dsac.pt` warm-start（論文從零訓練、prior 全靠 buffer）。會這樣是因為「真線上微調」需要多節點 live 部署讓 policy 實際互動，尚未上線。詳見 `services/rl_scheduler/rlpd_finetune.py`。

## 4. Slurm 基礎排程設定

上線使用 Slurm 原生能力作為穩定底座：

| 設定 | 上線值 / 行為 | 用途 |
|------|---------------|------|
| `SchedulerType` | `sched/backfill` | 允許不延後高優先 job 的前提下安排短 job 插隊 |
| `SelectType` | `select/cons_tres` | 以 TRES 表示 CPU / GPU / MPS 資源 |
| `SelectTypeParameters` | `CR_Core` | CPU core 級資源選擇；submit-time 與 hard placement 都仍交由 Slurm 管理配置 |
| `PriorityType` | `priority/multifactor` | 保留 Slurm age / job size / partition / qos 等基本排序 |
| `AccountingStorageTRES` | `gres/gpu,gres/mps` | 讓 GPU 與 MPS usage 進 accounting |
| `PreemptType` | 預設關閉 | 上線不主動踢 running job |

上線邊界：

| 能力 | 狀態 |
|------|------|
| 新 job 進來時依 priority 重新排序 | 支援 |
| backfill 短 job | 支援 |
| GPU / MPS 作為 Slurm GRES | 支援 |
| GPU runtime live migration | 不支援 |
| 跨節點 GPU memory 搬移 | 不支援 |
| 強制 requeue running job | 非預設上線行為 |

## 5. Submit Helper

`job_submit.lua` 會先執行 submit helper，讓後續 score 與 DSAC 看到較完整的 job 描述。helper 不覆蓋使用者明確指定的欄位。

| Helper | 行為 | 預設 |
|--------|------|------|
| memory | 使用 GPU 數與 CPU 數估算 `--mem` | enabled |
| partition | 依 `tres_per_node` 內容選擇 `cpu` / `gpu-rtx4070` / `gpu-rtx4080` | enabled |
| qos | 依 account rule 或 default 補 `qos` | enabled，default=`normal` |

partition rule 預設：

| Match | Partition |
|-------|-----------|
| `gpu:rtx4080` | `gpu-rtx4080` |
| `gpu:rtx4070` | `gpu-rtx4070` |
| `gpu:` | `gpu-rtx4070` |
| no GPU match | `cpu` |

## 6. Score Function

score function 是 submit-time heuristic，用來產生 priority delta。分數越高，job 越值得被提前考慮。

```text
score(J, P) = α * f_mps_fit(J, P)
            + β * f_vram_fit(J, P)
            + γ * f_topology(J, P)
            - δ * f_fragmentation(J, P)
            + ε * f_pred_runtime(J)

score = clamp(score, 0, 1)
priority_delta = round(scoreGain * score)
```

上線套用方式：

```text
if scoreApply=true and priority_delta > 0 and job_desc.priority is empty:
    job_desc.priority = priority_delta
```

`scoreGain` 預設為 `1000`，用來把 `[0,1]` score 轉成 Slurm priority delta。

### 6.1 係數

chart 預設係數：

| Factor | Symbol | Default | 說明 |
|--------|--------|---------|------|
| MPS fit | α | `0.40` | 偏好 MPS request 與 GPU slot 配適 |
| VRAM fit | β | `0.20` | 避免小 VRAM job 佔用大 VRAM tier |
| Topology | γ | `0.00` | 保留欄位，目前不影響上線結果 |
| Fragmentation cost | δ | `0.20` | 懲罰容易留下 MPS 碎片的 request |
| Predicted runtime | ε | `0.00` | predictor 啟用且係數非 0 時，短 job 會拿較高分 |

若 `weight-tuner` 啟用，Lua plugin load 時會從 `GET /weights` 載入 `(α, δ, ε)`，`β` 固定，`γ` 維持 chart 設定。

### 6.2 `f_mps_fit`

衡量 job MPS request 與單 GPU MPS 容量的配適程度。

| 項目 | 規格 |
|------|------|
| Input | `job_desc.tres_per_node`，例如 `gpu:rtx4070:1,mps:25` |
| MPS 容量 | `slurm.jobSubmit.mpsPerNode`，預設 `100` |
| Formula | `mps_req / mpsPerNode`，clamp 到 `[0,1]` |
| no MPS request | 回傳 `1.0` |
| request 超過容量 | 回傳 `0.0` |

### 6.3 `f_vram_fit`

依 `--constraint` 中的 `vram-*g` 需求選擇最小可用 VRAM tier，避免過度配置。

| 項目 | 規格 |
|------|------|
| Input | `job_desc.features`，例如 `vram-12g+` |
| VRAM tiers | `slurm.jobSubmit.vramTiers`，預設 `[12, 24]` |
| Formula | `1 - (fit_tier - req) / max_tier`，clamp 到 `[0,1]` |
| 無 VRAM constraint | 回傳 `0.5` |
| 無 tier 可容納 | 回傳 `0.0` |

### 6.4 `f_topology`

目前為保留欄位，Lua 回傳中性值 `0.5`。因 chart 預設 `γ=0`，此因子不影響上線 priority。

### 6.5 `f_fragmentation`

目前使用 submit-time proxy，不讀 live cluster state。它懲罰最容易留下碎片的 MPS request。

| 項目 | 規格 |
|------|------|
| Input | `mps_req` |
| Formula | `4 * x * (1 - x)`，其中 `x = mps_req / mpsPerNode` |
| `mps_req <= 0` | 回傳 `0.0` |
| `mps_req >= mpsPerNode` | 回傳 `0.0` |
| `mps_req = 50%` | 回傳接近 `1.0`，碎片化代價最高 |

### 6.6 `f_pred_runtime`

runtime predictor 啟用時，短 job 取得較高 score。predictor 不可用時回傳中性值。

| 項目 | 規格 |
|------|------|
| Endpoint | `POST /predict`，預設 `http://runtime-predictor:8080/predict` |
| Timeout | `slurm.jobSubmit.predictor.timeoutMs`，預設 `200ms` |
| Formula | `1 - pred_seconds / fallback_seconds`，clamp 到 `[0,1]` |
| fallback seconds | `fallbackHours * 3600`，預設 `4h` |
| predictor disabled | 回傳 `0.5` |
| timeout / 5xx / invalid body | 回傳 `0.5` |

Predictor request body：

```json
{
  "user": "alice",
  "partition": "gpu-rtx4070",
  "gpu_count": 1,
  "mps_req": 25,
  "gpu_type": "rtx4070",
  "user_time_limit_seconds": 3600
}
```

Predictor response：

```json
{
  "pred_seconds": 1146.28,
  "pred_minutes": 19.10,
  "model_version": "lgbm-v1",
  "bootstrap": false,
  "latency_ms": 1.12
}
```

`applyTimeLimit=true` 時，Lua 可把 `job_desc.time_limit` 改成預測值；上線建議只有在 predictor 經過校準後才開啟，避免模型低估造成 job timeout。

## 7. Weight Tuner

`weight-tuner` 是可選 FastAPI service，用 UCB1 在離散 arm 空間中調整 score function 的 `(α, δ, ε)`。

| Endpoint | 說明 |
|----------|------|
| `GET /weights` | 回傳目前 best arm 與統計資料 |
| `POST /feedback` | 以 reward 更新指定 arm |
| `GET /stats` | 回傳所有 arms 的 pulls 與 mean reward |
| `GET /healthz` | health check |

行為：

- Lua plugin 只在 load 時抓一次 `/weights`。
- 抓取失敗時沿用 chart 預設係數。
- `β` 不由 tuner 調整。
- live reward 以 completed jobs 的 mean JCT 轉換為負 reward。

## 8. DSAC Live Scheduler

`rl-scheduler` 是 FastAPI service（`services/rl_scheduler/serve.py`，容器內 `--policy-dir /models --port 8002`），載入 policy-dir 內的 DSAC checkpoint，提供 Slurm Lua hook 與 controller 查詢。

**目前 checkpoint 狀態。** 研究產出的 checkpoint 為 `runs/ckpts_aimix16_fair/`（2×1 拓樸、**obs_dim=168 / n_actions=33**、§5.1 fairness reward；RLPD 臂 `rlpd_cvar_*` 見 §3.5）。`use_iqn` / `risk_mode` 存在 checkpoint 內，serve 載入時自動偵測（vanilla SAC vs RDSAC-cvar）。image 以 Dockerfile `COPY … /models/dsac.pt` 烘入一個 checkpoint；換 checkpoint 時更新該 COPY 來源後重跑 `bash scripts/deploy-2.sh`，或在跑起來後對 serve `POST /reload` 指定新的絕對路徑（eval 逐臂即用此法熱換）。

| Endpoint | 說明 |
|----------|------|
| `GET /healthz` | model readiness、obs/action shape（`obs_dim`/`n_actions`）、snapshot age、shadow mode |
| `POST /snapshot` | 由 `rl-snapshot-agent` 定期更新 cached cluster snapshot |
| `POST /decide` | 對提交中的 job 回傳 priority boost / abstain / selected placement（submit-time，Lua hook 用） |
| `POST /act` | 給 controller/eval：回傳解碼後的 `(action, job_i, node_j, gpu_k, selected_job_id, value, entropy)`；`action==TOP_K·n_placements` 為 no-op（abstain） |
| `POST /reload` | 熱換 checkpoint（指定絕對路徑）；train/eval/serve 三處共用同一 checkpoint 格式 |
| `POST /shadow` | 切換 shadow mode（只記錄不套用） |
| `GET /metrics` | Prometheus metrics |

目前 live 介入方式分成兩層：submit-time **priority boost** 是預設低風險路徑；hold-release **hard placement controller** 是正式可用的受控 placement 路徑。

```text
job_submit.lua -> POST /decide
    if rl_selected=true and priority_boost>0:
        job_desc.priority += priority_boost
```

在 submit-time `/decide` 路徑中，DSAC 不直接執行 `srun --nodelist`，也不在 `job_submit.lua` 內覆蓋 Slurm placement；`node_j` 與 `gpu_k` 會回傳並記錄為 placement intent，實際 placement 仍交給 Slurm `select/cons_tres`。

DSAC 的 placement action 要真正生效，靠的是 **`rl-placement-controller`**（`services/rl_scheduler/placement_controller.py`，§3.4），它**現在預設常駐啟用**：對 held pending jobs 呼叫 `/act`，把 `node_j` 映射到 GPU worker，透過 slurmrestd job-update 寫 `required_nodes` 並以 `priority=INFINITE` release。注意 checkpoint topology 必須與 live node/GPU topology 一致——不一致時 `/act` 會 abstain，controller 隨即 no-op，Slurm 照常 `select/cons_tres` 放置（fail-safe）。目前研究產出的 checkpoint 為 2×1 拓樸的 168-dim（`runs/ckpts_aimix16_fair/`）；上線時 image 烘入的 checkpoint 維度與 live node/GPU 拓樸須一致（2 node × 1 GPU → n_placements=2），否則 `/act` abstain、controller 不動 job。換拓樸即需重訓（見 §3.5、`docs/intergration.md §7`）。

### 8.1 Snapshot Schema

```json
{
  "now": 0,
  "pending_jobs": [],
  "nodes": [
    {
      "gpus": [
        {"free_mps": 100, "running_jobs": 0, "gpu_type": "rtx4070"}
      ]
    }
  ],
  "n_nodes": 1,
  "gpus_per_node": 1,
  "mps_per_gpu": 100
}
```

`rl-snapshot-agent` 會定期從 Slurm REST API 讀取 jobs/nodes 並 POST `/snapshot`，讓 cached snapshot 保持 fresh。`/decide` 仍會在 snapshot 缺失或超過 `snapshotTtlSeconds` 時 abstain，避免使用過期 cluster state 做 live boost。

### 8.2 Decision Schema

Lua hook 送出的 request：

```json
{
  "job_id": "123",
  "mps_req": 25,
  "gpu_count": 1,
  "gpu_type": "rtx4070",
  "runtime_s": 3600,
  "now": 0
}
```

Service response：

```json
{
  "priority_boost": 1000,
  "rl_selected": true,
  "abstain": false,
  "abstain_reason": null,
  "rl_selected_job_id": "123",
  "node_j": 0,
  "gpu_k": 0,
  "value": -180.06,
  "entropy": 0.0,
  "shadow": false
}
```

`value` 為策略下的期望 risk-adjusted action value `Σ_a π(a|s)·(ρ[Z_R(s,a)] + α·E[Z_H(s,a)])`（由 `DSACAgent.action_values` 計算，受 `risk_mode` 影響）；`entropy` 為 categorical policy `π(a|s)` 的熵。兩者餵給下方 §8.3 的 `valueAbstain` / `entropyAbstain` guardrail。

### 8.3 Live Safety Gates

| Gate | 行為 |
|------|------|
| `shadowMode=true` | service 回傳 shadow decision，不實際 boost |
| stale snapshot | abstain，`priority_boost=0` |
| low value | 若 `value < valueAbstain`，abstain |
| high entropy | 若 `entropy > entropyAbstain`，abstain |
| no-op action | 不 boost |
| invalid / masked action | 不 boost |
| network / parse / Lua error | Lua hook no-op，submission 繼續 |

目前 live deployment 使用 DSAC checkpoint，`shadowMode=false` 時會實際套用 positive `priority_boost`。hard placement controller（預設常駐）另以 `/act` 執行 held job placement，**不受 Lua `shadowMode` 控制**；它自己用 `rlScheduler.placementController.shadow`（→ `--shadow / --no-shadow`，預設 `false`）決定是否真的送 slurmrestd job-update。兩條路徑互補：`/decide` 影響 priority 排序，controller 決定 held job 落在哪個 node。

## 9. Boundary Policy

| Failure | 行為 |
|---------|------|
| `job_submit.lua` Lua error | `pcall` 保護；回傳 `slurm.SUCCESS`，priority 不動 |
| predictor timeout / malformed response | `f_pred_runtime=0.5` |
| weight tuner unavailable | 使用 chart 預設 weights |
| RL scheduler unavailable | `rl_apply` no-op；submission 不失敗 |
| score < 0 | clamp 到 0 |
| score > 1 | clamp 到 1 |
| `scoreGain=0` | score 只記錄，不改 priority |
| `scoreApply=false` | score 只記錄，不改 priority |
| DSAC abstain | submit-time 不加 boost；hard placement path 不 release / 不更新 placement |
| DSAC no-op | submit-time 不 boost；hard placement job 維持 held/pending |
| hard placement 選到 unavailable node | controller 過濾 DRAIN/DOWN/NOT_RESPONDING/FAIL，不更新該節點 |
| operator scale-down during running job | policy/action guard 阻止 scale-down，保留 running/COMPLETING worker |

## 10. Monitoring Metrics

`rl-scheduler` 暴露 Prometheus metrics，Grafana dashboard `Scheduler Live Resource View` 會使用這些指標。

| Metric | 說明 |
|--------|------|
| `rl_scheduler_ready` | DSAC model 是否載入 |
| `rl_scheduler_shadow_mode` | 1=shadow，0=live |
| `rl_scheduler_decisions_total{result}` | selected / no_boost / abstain 次數 |
| `rl_scheduler_priority_boost_total` | positive boost 累積次數 |
| `rl_scheduler_last_priority_boost` | 最近一次 boost |
| `rl_scheduler_policy_value` | 最近一次 value estimate |
| `rl_scheduler_policy_entropy` | 最近一次 entropy |
| `rl_scheduler_snapshot_age_seconds` | snapshot age |
| `rl_scheduler_snapshot_pending_jobs` | snapshot pending jobs |
| `rl_scheduler_snapshot_free_mps` | snapshot free MPS slots |
| `rl_scheduler_last_action` | 最近一次 flat action index |
| `rl_scheduler_last_job_index` | 最近一次 selected job slot |
| `rl_scheduler_last_node_index` | 最近一次 selected node index |
| `rl_scheduler_last_gpu_index` | 最近一次 selected GPU index |

## 11. Deployment Knobs

常用 Helm values：

```yaml
slurm:
  jobSubmit:
    enabled: true
    scoreApply: true
    scoreGain: 1000
    mpsPerNode: 100
    predictor:
      enabled: false
      applyTimeLimit: false

rlScheduler:
  enabled: true
  shadowMode: false
  snapshotTtlSeconds: 86400
  valueAbstain: -100000
  entropyAbstain: 1.5
  priorityBoost: 1000
  lua:
    enabled: true

weightTuner:
  enabled: false
```

Live smoke check：

```bash
kubectl -n slurm exec slurm-controller-0 -- \
  curl -fsS http://rl-scheduler:8002/healthz

kubectl -n slurm exec slurm-controller-0 -- \
  curl -fsS http://rl-scheduler:8002/metrics | grep rl_scheduler

LOGIN_POD=$(kubectl -n slurm get pod -l app=slurm-login -o jsonpath='{.items[0].metadata.name}')
kubectl -n slurm exec "$LOGIN_POD" -- \
  sbatch --wrap='sleep 3' --job-name='dsac-live-smoke' -p cpu

kubectl -n slurm logs slurm-controller-0 --tail=500 | grep -E '\[rl\]|\[score'
```

Hard placement smoke check：

```bash
JOB_ID=$(kubectl -n slurm exec deploy/slurm-login -- \
  sbatch --hold --parsable -J dsac-place-smoke \
  -p gpu-rtx4070 --gres=mps:10 --time=00:03:00 \
  --wrap 'hostname; sleep 10')

PYTHONPATH=. python -m services.rl_scheduler.placement_controller \
  --once --no-shadow --job-id "$JOB_ID" \
  --node-name slurm-worker-gpu-rtx4070-0 \
  --node-name slurm-worker-gpu-rtx4070-1 \
  --scheduler-url http://rl-scheduler:8002 \
  --scheduler-exec-prefix 'kubectl -n slurm exec pod/slurm-controller-0 --' \
  --slurm-exec-prefix 'kubectl -n slurm exec deploy/slurm-login --'

kubectl -n slurm exec deploy/slurm-login -- \
  scontrol show job "$JOB_ID" | grep -E 'ReqNodeList|NodeList|BatchHost|TRES'
```

## 12. Files

| Path | Purpose |
|------|---------|
| `chart/templates/configmap-job-submit.yaml` | Generates `job_submit.lua` |
| `chart/lua/rl_hook.lua` | Lua client for DSAC `/decide` |
| `services/rl_scheduler/serve.py` | DSAC FastAPI service |
| `services/rl_scheduler/snapshot_agent.py` | Periodic Slurm REST snapshot updater for `/snapshot` |
| `services/rl_scheduler/placement_controller.py` | Slurm-safe DSAC hard placement controller using hold-release and `ReqNodeList` |
| `services/rl_scheduler/live_daemon.py` | Direct placement research / legacy prototype; not recommended for Slurm-safe production placement |
| `services/rl_scheduler/dsac.py` | RDSAC (Ma et al.) — dual Z_R/Z_H IQN critic + categorical actor |
| `services/rl_scheduler/distortion.py` | Risk distortions (CVaR/Wang/CPW/MSD) |
| `services/runtime_predictor/` | Runtime prediction service |
| `services/weight_tuner/` | UCB1 weight tuner |
| `chart/dashboards/scheduler-live.json` | Live scheduler Grafana dashboard |
| `docs/monitoring.md` | Monitoring metrics and dashboard details |
