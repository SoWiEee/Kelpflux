# 基於 Slurm 與 Kubernetes 架構下 AI 伺服器 GPU 工作負載智慧排程技術之研究

### Intelligent GPU Workload Scheduling Techniques for AI Servers under a Slurm-on-Kubernetes Architecture

**作者一¹、作者二²**
¹○○大學 ○○系　²○○大學 ○○系
{author1, author2}@stumail.nutn.edu.tw

---

## 摘要

隨著大型語言模型與生成式 AI 快速發展，GPU 已成為 AI 工作負載的主要運算資源 [1][4]。然而，多數實驗室及中小型叢集通常由不同世代 GPU 所組成，且 NVIDIA Multi-Process Service (MPS) 允許多個工作共享單張 GPU [5]，使得傳統 Slurm 排程器 [11] 難以同時兼顧資源利用率、工作完成時間與尾端延遲。

現有 GPU 排程方法多著重於單純 GPU placement、同質 GPU 叢集，或未納入真實 Slurm 排程流程的模擬環境，較少同時考慮異質 GPU、MPS 配置與實機部署後的 sim-to-real 落差。此缺口使學習式排程方法即使在模擬中表現良好，也難以判定其效益能否穩健轉移到真實異質叢集。

本研究提出一套支援異質 GPU 與 NVIDIA MPS 的智慧排程框架，透過深度強化學習在 MPS 剩餘容量的約束下決定佇列工作的派遣與 GPU placement。排程代理人根據工作資訊、GPU 使用率、MPS 剩餘容量、佇列狀態與歷史回饋，決定要從佇列中派出哪個工作、並將其放置到哪張 GPU；工作本身攜帶的 MPS fraction 需求（25%、50%、75% 或 100%）由 state 特徵與 action mask 納入考量（遮蔽 MPS 剩餘不足的放置），而非由策略自由選擇。系統以 Slurm 作為排程核心 [11]，Kubernetes 僅負責容器化部署、服務生命週期與健康監控 [10]；當學習式決策服務逾時或異常時，系統會自動回退至啟發式排程策略。

本研究於 RTX 4070 與 RTX 3080 實機環境建置 Slurm 平台，並參考 Alibaba GPU Trace 與 Microsoft Philly Trace 的工作負載特性進行 trace replay 與真實 AI 工作負載評估 [1][4]，和 FCFS、Backfill、Discrete SAC [6]、RDSAC [7][16] 及 RLPD [8] 等方法比較。結果顯示，學習式排程在模擬環境可區分策略差異，但在本研究小規模實機叢集上尚未穩健勝過啟發式方法；此結果凸顯異質 GPU 與 MPS 排程必須同時重視真實部署、統計檢定與場景依賴性。

**關鍵詞**：GPU 排程、異質 GPU、NVIDIA MPS、Slurm、深度強化學習

## Abstract

The rapid growth of large language models and generative AI has made GPUs a primary compute resource for AI workloads [1][4]. However, many laboratory and small-scale clusters consist of heterogeneous GPU generations, and NVIDIA Multi-Process Service (MPS) enables multiple jobs to share a single GPU [5]. These properties make it difficult for traditional Slurm scheduling [11] to jointly optimize GPU utilization, job completion time, and tail latency.

Existing GPU scheduling methods often focus on GPU placement alone, homogeneous GPU environments, or simulation-only evaluation without the real Slurm scheduling path. Few studies jointly consider heterogeneous GPUs, MPS allocation, and real deployment effects. This gap makes it unclear whether learned scheduling policies that perform well in simulation can transfer robustly to real heterogeneous clusters.

This study proposes an intelligent scheduling framework for heterogeneous GPUs with NVIDIA MPS. The framework uses deep reinforcement learning to decide, in an MPS-aware manner, which queued job to dispatch and onto which GPU to place it. Given job features, GPU utilization, remaining MPS capacity, queue state, and historical feedback, the agent selects a job and its placement; each job carries its own MPS fraction request (25%, 50%, 75%, or 100%), which the scheduler respects through state features and an action mask that hides placements with insufficient free MPS, rather than choosing the fraction as a free action. Slurm remains the scheduling core [11], while Kubernetes is used only as the deployment and lifecycle-management platform [10]. If the learned decision service times out or fails, the system automatically falls back to a heuristic scheduling policy.

We implement the framework on a real RTX 4070 and RTX 3080 testbed, and evaluate it using workload characteristics derived from Alibaba GPU Trace and Microsoft Philly Trace [1][4], together with real AI workloads. The proposed framework is compared with FCFS, Backfill, Discrete SAC [6], RDSAC [7][16], and RLPD [8]. Results show that learned policies can distinguish scheduling strategies in simulation, but do not robustly outperform the heuristic baseline on the small real cluster used in this study. These findings show that heterogeneous GPU and MPS scheduling must be evaluated with real deployment, statistical rigor, and explicit attention to scenario dependence.

---

## 1. 緒論

### 1.1 AI 工作負載快速增加

近年大型語言模型、生成式 AI 與深度學習服務快速發展，使 GPU 成為 AI 訓練與推論最重要的運算資源 [1][4]。相較於傳統批次運算，AI 叢集經常同時承載短時間推論、模型微調、長時間訓練與矩陣運算等不同工作，這些工作在執行時間、記憶體需求、延遲敏感度與 GPU 使用型態上皆有明顯差異 [1][4]。

在大學實驗室與中小型研究叢集中，GPU 資源通常有限，且硬體常由不同世代 GPU 漸進式擴充而成。例如 RTX 4070 與 RTX 3080 在算力、記憶體容量與功耗上皆不同。若排程器只把 GPU 視為同質資源，便可能讓小型推論工作佔用高效能 GPU，或讓長時間訓練阻塞後續短工作，造成 GPU utilization、工作完成時間 (JCT) 與 queue delay 之間的取捨更加困難。

NVIDIA MPS 提供另一個重要槓桿：多個 CUDA 工作可共享同一張 GPU，使 GPU 不再只能以整張卡為單位分配 [5]。然而 MPS 也讓排程問題從「選哪張 GPU」變成「選哪張 GPU 與分配多少 **MPS fraction**」。因此，在異質 GPU 與 MPS 共存的環境中，GPU scheduling 已成為影響叢集效能的核心問題。

### 1.2 現有方法限制

Slurm 是高效能運算環境常用的工作排程系統，支援 FCFS、Backfill、multifactor priority 與 GRES/TRES GPU 資源管理 [11]。然而，Slurm 的傳統策略多以固定規則為主，對異質 GPU、MPS 分片、工作類型與長期回報之間的互動感知有限。

現有方法主要存在四項限制：

1. **不充分考慮 GPU 差異**：許多排程方法把 GPU 視為同質資源，較少將不同世代 GPU 的算力、記憶體與執行時間差異納入 placement 決策。
2. **不充分考慮 MPS fraction**：部分研究討論 GPU sharing 或 GPU partition，但未將 **MPS fraction** 作為排程器的顯式動作。
3. **依賴固定規則**：FCFS 與 Backfill 能提供穩定基準，但難以隨工作負載動態調整策略。
4. **缺乏真實 Slurm 流程驗證**：不少學習式排程研究停留在模擬環境，未整合到真實 Slurm job submission path，也未處理服務失效、實機漂移與統計顯著性。

因此，本研究的 gap 是：現有 GPU 排程方法較少同時處理異質 GPU、MPS allocation 與真實 Slurm 排程流程，也較少嚴謹檢驗學習式策略在 sim-to-real 轉移後是否仍具穩健效益。

### 1.3 為什麼使用 DRL

GPU scheduling 不是單次分類問題，而是序列決策問題。一次 placement 會改變 GPU 剩餘 MPS、後續佇列等待時間、工作共置干擾與未來可用資源，因此當下看似最佳的放置，未必能帶來長期最佳的 JCT 或利用率。

此問題具有三個適合使用深度強化學習 (Deep Reinforcement Learning, DRL) 的特徵：

1. **Sequential decision**：每次工作放置會影響後續所有排程決策。
2. **Long-term optimization**：排程目標不只包含當下工作的執行時間，也包含 queue delay、makespan、tail latency 與整體 GPU utilization。
3. **Dynamic environment**：工作到達率、工作類型、GPU 使用率與 MPS 剩餘容量會隨時間改變，固定 heuristic 難以完整涵蓋所有情境。

因此，本研究採用 DRL 作為排程策略學習方法，讓代理人從模擬與實機回饋中學習 GPU placement 與 MPS allocation 的長期效果。不過，本研究並不預設 DRL 必然優於 heuristic，而是以真實環境與統計檢定驗證其效益是否穩健成立。

### 1.4 Contribution

本研究主要貢獻如下：

1. **提出異質 GPU + MPS-aware 排程框架**：以 job 選擇與 GPU placement 為動作空間，並將每個工作攜帶的 MPS fraction 需求（25%、50%、75%、100%）納入 state 特徵與 action mask 的可行性約束，使排程器在尊重 MPS 配額的前提下決定派哪個工作、放到哪張 GPU。
2. **設計 DRL Scheduler**：以 Discrete SAC [6]、RDSAC [7][16] 與 RLPD [8] 等方法學習序列排程策略，並將 state、action、reward 對應到真實 Slurm 叢集。
3. **完成 Slurm 整合**：透過 Slurm job submission path 連接 RL decision service，並提供 fail-safe fallback，避免學習式服務異常時阻塞排程核心。
4. **完成真實部署**：於 RTX 4070 與 RTX 3080 異質 GPU 環境中部署 Slurm、NVIDIA MPS 與監控流程；Kubernetes 僅作為部署與生命週期管理平台。
5. **完成實機驗證與統計分析**：使用 trace replay 與真實 AI workload 評估 FCFS、Backfill、heuristic、SAC、RDSAC 與 RLPD，並以多 seed 配對、多重比較校正與等價檢定檢驗效益。

## 2. 相關研究

### 2.1 GPU Sharing

GPU sharing 技術的目標是在單張或多張 GPU 上提高資源使用率，避免小型工作獨佔整張 GPU。NVIDIA MPS 允許多個 CUDA process 同時共享同一張 GPU 的運算資源 [5]；NVIDIA MIG 則在硬體層面將 GPU 切分成隔離的 GPU instance，提供較強的效能隔離與資源邊界 [21]。

近年研究也開始探討更細緻的 GPU partition 與 spatio-temporal sharing。例如 Serving Heterogeneous Machine Learning Models on Multi-GPU Servers with Spatio-Temporal Sharing 討論多模型服務如何在多 GPU 伺服器上進行時間與空間共享 [32]；Hierarchical Resource Partitioning on Modern GPUs 則研究現代 GPU 上階層式資源分割 [33]。

這些研究顯示 GPU sharing 能提升利用率，但多聚焦於 GPU partition 或 inference serving 本身，較少把 GPU sharing 作為 Slurm 排程器的顯式決策變數。本研究的差異在於：**MPS fraction** 是 action space 的一部分，排程器必須同時決定 GPU placement 與 **MPS fraction**。

### 2.2 GPU Scheduling

傳統 GPU scheduling 方法包含 FCFS、Backfill、Best Fit、packing、multifactor priority 與 Slurm 的 cons_tres/GRES 資源配置 [11]。這些方法具有實作穩定、可解釋與部署成本低等優點，因此仍是 HPC 與研究叢集的重要基準。

然而，傳統方法主要依賴固定規則。FCFS 容易受到 head-of-line blocking 影響；Backfill 能減少資源閒置，但對異質 GPU 與 MPS 共置干擾缺乏細緻感知；Best Fit 能改善裝箱效率，卻可能放大尾端延遲或造成局部碎片。Slurm 雖支援 GPU GRES 與 MPS GRES，但預設策略仍不會主動學習不同工作在不同 GPU 與 MPS 配額下的長期影響。

針對深度學習工作負載，也有專門的叢集排程系統：Gandiva [2] 以 introspective 排程與工作遷移提升利用率，Tiresias [3] 則以近似 age-based 的優先序縮短 JCT；此外亦有研究引入多資源感知、線上遷移與雲端自適應資源配置 [19][20]。邊緣運算場景同樣有專門的 GPU 排程探討 [22][24]，惟其資源與延遲約束與資料中心叢集不同。這些方法多假設整卡分配或同質 GPU，較少將 MPS fraction 納入排程決策。

因此，傳統 GPU scheduling 可作為穩定 baseline，但在異質 GPU + MPS 的設定下，仍需要能同時感知 GPU 差異、MPS 容量、工作特徵與佇列狀態的策略層。

### 2.3 RL Scheduler

強化學習已被應用於 cluster 排程、GPU 排程、network contention control 與 AI serving [23][28][30]。Discrete SAC 適合離散動作空間，可用於選擇節點、GPU 或資源配額 [6]；RDSAC 進一步以分布式評論家與風險敏感目標處理尾端延遲 [7][16]；RLPD 則可將既有真實資料納入 offline-to-online 微調，以降低從零開始探索的成本 [8]。

此外，UXP-RL、Network Contention RL、Kubernetes inference auto-scaling 與多種 DRL scheduler 皆顯示 RL 能處理動態資源分配問題 [23][25][26][28][30]。然而，現有 RL scheduler 多假設單一 GPU、同質 GPU、大型模擬叢集，或將 Kubernetes 當作主要排程器（如 Kubeflow [12]、Volcano [13]、Kueue [14] 與 NVIDIA KAI [31] 等 Kubernetes-native 批次／佇列系統）；較少探討在真實 Slurm 流程中，如何讓 RL 同時決定異質 GPU placement 與 MPS allocation，並在服務失效時維持排程安全。

本研究因此將 RL 放在 Slurm 策略層，而非取代 Slurm 或 Kubernetes。與將 Slurm 整合進 Kubernetes 的 Slinky [27] 等方向互補，本研究讓 Slurm 維持佇列與資源配置語意 [11]，RL 只負責對 GPU 與 MPS 配額提出決策。

### 2.4 Discussion

表 1 彙整本研究與主要研究類型的定位差異。

表 1. 相關研究定位比較

| 類型 | 是否考慮異質 GPU | 是否考慮 MPS fraction | 是否整合真實 Slurm | 主要限制 |
|---|:--:|:--:|:--:|---|
| FCFS / Backfill | 部分 | 部分 | 是 | 固定規則，難以學習長期效果 |
| GPU sharing / partition 研究 | 部分 | 部分 | 否 | 多聚焦 partition 機制，較少處理排程流程 |
| RL scheduler 模擬研究 | 部分 | 少 | 否 | sim-to-real 效益不明 |
| Kubernetes GPU scheduler | 部分 | 部分 | 否 | Kubernetes 是排程主體，不處理 Slurm 工作流 |
| 本研究 | 是 | 是 | 是 | 目前實機規模仍小，需擴大驗證 |

由表 1 可知，本研究的核心位置是「異質 GPU + MPS + 真實 Slurm 流程」的交會點。Kubernetes 僅是部署平台 [10]，不是本文的主要排程貢獻。

## 3. 研究目的與系統架構

### 3.1 System Overview

本研究建立一套以 Slurm 為排程核心的異質 GPU + MPS 智慧排程平台。系統流程如圖 1 所示：

```mermaid
flowchart TD
    A[Job Submission] --> B[Slurm Controller]
    B --> C{RL Scheduler<br/>via job_submit hook}
    C -->|Observe State| D[GPU + MPS fraction]
    D --> E[Execute Job on GPU]
    E --> F[Monitoring<br/>JCT, Util, Queue Delay]
    F --> G[Replay Buffer]
    G --> C
    C -.->|Timeout / Invalid Action| H[Heuristic Fallback<br/>Score-based]
    H --> B
```

**圖 1. 系統架構與排程流程。**  
虛線框為 fail-safe fallback 路徑；RL Scheduler 以 Slurm `job_submit.lua` hook 介入，決策輸出為 `(job, GPU placement)`（工作的 MPS fraction 依其資源請求，非策略輸出）；Monitoring 週期 1 秒蒐集 GPU/SM/memory utilization、MPS fraction 使用量、queue depth、job events，寫入 Replay Buffer 供 offline RL / RLPD 使用。

工作提交後，Slurm 透過 job submission hook 呼叫 RL scheduler。RL scheduler 讀取目前工作資訊、GPU 狀態、MPS 剩餘容量與佇列資訊，輸出所選工作與其 GPU placement（工作的 MPS fraction 依其請求分配、非策略輸出）。工作執行期間，監控服務收集 GPU utilization、SM utilization、memory usage、queue delay、JCT 與 reward，並將資料寫入 replay buffer 供後續訓練或 RLPD 微調使用。

本平台使用 Kubernetes (k3s) 部署 Slurm controller、worker、RL scheduler、monitoring service 與相關容器 [10]。**Kubernetes 在本文中不負責排程決策**；它只提供容器化部署、服務健康檢查、網路與生命週期管理。**此設計避免讓 Kubernetes 成為研究主角，並保留 Slurm 在 HPC batch scheduling 中成熟的佇列語意 [11]。**

### 3.2 Problem Definition

本研究將異質 GPU + MPS 排程建模為馬可夫決策過程。每次有工作可排程時，代理人觀察叢集狀態，選擇一個 GPU 與 **MPS fraction**，並在工作完成後根據 JCT、queue delay 與 GPU utilization 取得 reward。

**State.** 狀態包含四類資訊：

1. **Job features**：工作類型、預估 runtime、GPU memory 需求、MPS 需求、SLO 緊迫度與等待時間。
2. **GPU features**：GPU 型號、GPU utilization、SM utilization、memory usage、可用 MPS、目前共置工作數。
3. **Queue features**：佇列長度、前 K 個工作的需求、等待時間分布與到達率。
4. **History features**：近期完成工作 JCT、slowdown、SLO violation 與各 GPU 的負載變化。

**Action.** 動作為「從佇列前 K 個工作中選一個」與「將其放置到哪張 GPU」的聯合離散決策，另含一個 no-op（暫不派遣）：

```text
Action = (選 job_i ∈ top-K 佇列) × (放置到 node_j / gpu_k)  ∪  {no-op}
```

在本研究 2×1 平台（K=16、2 個 placement）中，動作空間為 16×2+1 = 33 個離散動作（obs 維度 168）。**MPS fraction 不是動作維度**：每個工作攜帶自身的 MPS fraction 需求（25%/50%/75%/100%），排程器透過 state 特徵感知、並由 action mask 遮蔽 MPS 剩餘容量不足的放置，因此策略是在尊重工作 MPS 需求的前提下做 job 選擇與 placement，而非自由指派 fraction。

**Reward.** Reward 需同時反映使用者等待時間與叢集效率。本研究實作採用以 JCT 為核心的多目標形式（`reward_mode="mo"`，即表 4 學習臂的訓練目標）：

```text
R = w_jct · (−JCT / reward_scale) + w_util · GPUUtil
    （另加小權重 placement shaping、可選 SLO 逾時懲罰）
```

其中 w_jct=1.0、w_util=0.05、reward_scale=1000。JCT = 完成時間 − 提交時間，已把 queue delay 內含（故無獨立的 queue-delay 加權項）；GPUUtil 鼓勵 MPS 共置以免策略學到保守閒置；SLO 逾時懲罰為 opt-in，只對延遲-class 且超過期限的工作生效。此外可選用 potential-based reward shaping [9] 提供密集訊號而不改變最優策略。此 reward 設計讓 DRL 不只最佳化單一工作，而是學習長期排程結果。

### 3.3 Research Goal

本研究的目標是在異質 GPU 與 NVIDIA MPS 環境中，學習一個能在 MPS 配額約束下決定工作派遣與 GPU placement 的排程策略，以降低 average JCT、P95/P99 JCT、queue delay 與 slowdown，同時提升 GPU utilization 並控制 SLO violation。

更具體而言，本研究要回答三個問題：

1. DRL 是否能在模擬環境中學到比固定規則更好的 GPU + MPS 排程策略？
2. 此策略能否透過 Slurm job submission path 部署到真實異質 GPU 叢集？
3. 學習式策略在真實環境中是否能穩健勝過 FCFS、Backfill 與啟發式 baseline？

## 4. 排程技術

本章介紹本研究的 scheduler，而非先介紹強化學習原理。重點是 state 如何收集、action 如何映射到 Slurm 與 MPS、reward 如何設計，以及 DRL agent 如何嵌入排程流程。

### 4.1 State

排程器在每次決策時收集以下 state：

| 類別 | 特徵 |
|---|---|
| GPU | GPU 型號、memory capacity、memory usage、GPU utilization、SM utilization、目前 MPS 使用量 |
| MPS | 每張 GPU 剩餘 **MPS fraction**、目前分配比例、共置工作數、**MPS fragmentation** |
| Job | job type、runtime estimate、memory request、MPS request、SLO、submit time |
| Queue | queue length、前 K 個等待工作、等待時間、短工作比例、長工作比例 |
| History | 近期 JCT、P95/P99 JCT、slowdown、GPU load balance、SLO violation |

此 state 設計讓 agent 能區分「把短推論工作放到高效 GPU 的 25% MPS」與「把長訓練工作放到低效 GPU 的 100% MPS」這類決策差異。

### 4.2 Action

Action 是「選擇佇列前 K 個工作中的哪一個」與「將其放置到哪張 GPU」的聯合離散決策，另含 no-op。MPS fraction 為工作自身的資源請求，不在動作空間內：

```text
Action = job_i (top-K 佇列) × placement (node_j / gpu_k)  ∪  {no-op}
```

選定 action 後，scheduler 將結果轉換為 Slurm 可執行的資源請求與節點限制：把選中的工作導向對應節點/GPU，並依該工作請求的 MPS fraction 設定 GRES/MPS。若某放置的 GPU 剩餘 **MPS fraction** 不足以容納該工作的請求，action mask 會在推論前遮蔽該選項，避免產生不可執行決策。

### 4.3 Reward

Reward 的設計目標是降低使用者感受到的等待與完成時間，同時提高 GPU 使用效率。本文使用 JCT 作為主要負向訊號，並加入 GPU utilization 項；queue delay 已內含於 JCT，SLO violation 則為可選的 opt-in 懲罰。

此設計有三個原因：

1. **降低 JCT**：JCT 是 GPU scheduling 論文最常見的核心指標，能反映工作從提交到完成的整體時間。
2. **內含 queue delay**：JCT 以「完成 − 提交」計，已把佇列等待納入，故排程器最佳化 JCT 時同時反映 head-of-line blocking 與飢餓，無需獨立的 queue-delay 加權項。
3. **提升 GPU utilization**：MPS 的目的是讓小工作共用 GPU；若 reward 不納入 utilization，agent 可能學到保守但浪費資源的策略。
4. **控制尾端風險**：P95/P99 JCT 與 SLO violation 對推論與服務型工作特別重要，因此 RDSAC 使用風險敏感目標處理尾端延遲。

### 4.4 DRL Algorithm

本研究比較以下 DRL 方法：

1. **Discrete SAC**：將 Soft Actor-Critic 延伸到離散 action space，適合 job 選擇與 GPU placement 這類有限離散動作 [6]。
2. **RDSAC-mean**：以分布式 critic 建模回報分布，但 actor 主要依平均回報決策 [7][16]。
3. **RDSAC-cvar**：在 RDSAC 上加入 CVaR 風險敏感目標，使策略更重視尾端 JCT 與 SLO violation [16]。
4. **RLPD**：使用真實叢集收集的 transition 對模擬訓練出的模型進行微調，以縮小 sim-to-real gap [8]。

RDSAC 採用雙頭 IQN critic 建模 reward return 與 entropy return [7]，並使用 masked categorical actor 避免選到不可執行 action（即所選放置的 GPU 剩餘 **MPS fraction** 不足以容納該工作請求的動作）。訓練流程包含 prioritized replay、n-step return、potential-based reward shaping [9] 與 heuristic warm start。需澄清命名：本研究的 RDSAC 為自組的「distributional + discrete SAC」，以離散動作空間搭配 IQN 分位數 critic 建構 [7][16]，與 Duan 等人 [17] 針對連續控制、將回報建模為單一高斯分布的 Distributional Soft Actor-Critic 不同，兩者不應混淆。

表 X 彙整學習臂的關鍵訓練超參數；所有臂共用網路與最佳化設定，僅 critic 家族與風險目標不同。

表 X. DRL 訓練超參數

| 參數 | 值 |
|---|---|
| optimizer / learning rate（actor, critic, α） | Adam / 3×10⁻⁴ |
| 折扣 γ / 目標軟更新 τ | 0.99 / 0.005 |
| 隱藏層（MLP trunk） | (256, 256) + LayerNorm |
| batch size / UTD ratio | 256 / 4 |
| n-step return | 10 |
| replay | Prioritized（SumTree）+ score warm start |
| IQN 分位數 N_QUANT / cosine 維度 | 32 / 64（RDSAC，`use_iqn=True`） |
| 風險目標 / tail mass β | CVaR / 0.25（RDSAC-cvar 臂） |
| 溫度 α | 自動調整（init 0.1）；vanilla SAC 另設 fixed-α 控制組 |
| reward | 多目標 `mo`（w_jct=1.0, w_util=0.05），reward_scale=1000 |
| 訓練長度 | 每臂約 1×10⁵ curriculum env-steps |

**收斂性。** 圖 1 為三個學習臂在 aimix-family 課程訓練下的 episode reward 與 critic loss（跨 seed、rolling window w=80 平滑）。三臂 reward 皆從約 0 隨課程階段上升並趨於穩定；RDSAC-mean/cvar 收斂較高且平穩，vanilla SAC 在自動-α 下方差明顯較大（與其已知的溫度控制不穩定一致）。critic loss 皆穩定收斂（RDSAC 分位數 Huber 損失量級約 10⁻¹、SAC 標量 critic 約 10⁻³），未見發散，佐證 §5.4 對 convergence 的評估。

![圖 1. 學習臂訓練收斂曲線](../assets/figures/training_convergence.png)

圖 1. 學習臂訓練收斂（SAC / RDSAC-mean / RDSAC-cvar，aimix-family 課程，跨 seed 平均、rolling window w=80）。原始逐 episode log 見 `runs/aimix_{sac,rdsac_mean,rdsac_cvar}_s*/sim_train.jsonl`。

### 4.5 Scheduler Workflow

完整排程流程如下：

```text
Job Arrival
    |
Observe State
    |
Mask Invalid Actions
    |
RL Policy / Heuristic Fallback
    |
Select Job x GPU Placement
    |
Submit to Slurm
    |
Run on GPU with MPS
    |
Collect JCT / Utilization / Queue Delay
    |
Compute Reward
    |
Store Transition in Replay Buffer
```

為避免學習式服務影響 Slurm 穩定性，系統採 fail-safe 設計。若 RL service 逾時、回傳無效 action 或健康檢查失敗，Slurm 會回退至啟發式策略。此設計使 DRL scheduler 可在真實排程路徑中部署，而不需修改 Slurm 核心。

## 5. 實驗與評估方法

### 5.1 Environment

本研究使用一個小規模異質 GPU 實機叢集進行部署與評估：

| 項目 | 設定 |
|---|---|
| GPU | NVIDIA RTX 4070、NVIDIA RTX 3080 |
| 排程器 | Slurm with GRES/TRES and MPS [11] |
| 部署平台 | Kubernetes/k3s，僅負責容器部署與服務生命週期 [10] |
| GPU sharing | NVIDIA MPS，**MPS fraction** 為 25%、50%、75%、100% [5] |
| 系統 | Ubuntu、CUDA、NVIDIA driver、PyTorch |
| 監控 | GPU utilization、SM utilization、memory usage、job event、queue delay |

此環境刻意保留 GPU 世代差異，讓排程器必須面對異質 GPU placement 的問題。由於硬體規模只有兩張 GPU，本研究將結果定位為實機 proof-of-concept 與方法學驗證，不將結論直接外推至大型生產叢集。

### 5.2 Dataset

工作負載包含三類來源：

1. **Alibaba GPU Trace**：用於參考生產 MLaaS 工作的到達率、工作長度與資源需求分布 [4]。
2. **Microsoft Philly Trace**：用於參考多租戶 GPU training workload 的佇列與 JCT 特性 [1]。
3. **Real Job Replay**：在本研究叢集上執行 cuBLAS、BERT inference、ResNet training、Qwen fine-tuning 與矩陣運算等真實 AI 工作。

模擬環境使用 trace-derived workload 大量訓練 DRL agent；實機環境則以 replay 方式提交相同 seed 的工作序列，使不同 scheduler 可在相同工作負載下進行配對比較。

### 5.3 Baseline

本研究比較以下 baseline 與學習式方法：

| 類型 | 方法 |
|---|---|
| 傳統排程 | FCFS、Backfill |
| 裝箱啟發式 | score heuristic |
| DRL | Discrete SAC [6]、RDSAC-mean [7][16]、RDSAC-cvar [7][16] |
| Sim-to-real | RLPD [8] |

score heuristic 以 **MPS fraction** fit、VRAM fit 與 fragmentation penalty 為主，代表可部署、可解釋且低成本的生產啟發式方法。DRL 方法則用於檢驗學習式策略是否能在異質 GPU + MPS 情境下超越固定規則。

需注意 score 在兩個評估管線的係數設定不同：模擬（表 2）使用含 SJF-like runtime kicker 的完整版本（ε=0.30），而實機部署（表 3、表 4）因 runtime predictor 未上線而停用該項（ε=0），僅保留 MPS-fit 與 VRAM-fit。兩者為同一評分函式的兩種強度設定，因此表 2 與表 3/4 的 score 絕對表現不宜直接跨表比較；各表內部的相對比較（同一管線下 score vs. 其他 arm）則不受影響。

### 5.4 Metrics

本研究不只使用 average JCT，而是依 GPU scheduling 論文常見評估方向分成四類指標：

| 類別 | 指標 |
|---|---|
| Performance | Average JCT、P95 JCT、P99 JCT、Makespan |
| GPU | GPU Utilization、SM Utilization、MPS Utilization |
| Scheduling | Queue Delay、Slowdown、Load Balance、SLO Violation |
| RL | Average Reward、Convergence、Sample Efficiency |

Average JCT 能反映整體完成效率；P95/P99 JCT 與 SLO violation 能捕捉少數被拖慢的工作；GPU utilization 與 MPS utilization 則檢驗排程器是否真正提高共享效率；average reward 與 convergence 用於分析 DRL 訓練是否穩定。

### 5.5 Evaluation Methodology

為避免實機量測誤判，本研究採用以下方法：

1. **Common random numbers**：不同 scheduler 使用相同 seed 與相同工作序列，降低比較變異。
2. **Drift-robust interleaving**：不同方法交錯執行，避免 GPU 暖機、快取或系統狀態漂移與特定方法混淆。
3. **Seed-level paired statistics**：以 seed 為分析單位，而非把每個 job 當獨立樣本，避免偽重複。
4. **Multiple-comparison correction**：多個 scheduler 同時比較時使用 Holm-Bonferroni 校正。
5. **Equivalence testing**：當差異很小時使用 TOST 檢定判斷是否可視為實務等價。

此外，實機學習臂評估（表 3、表 4）的每個 seed 對應一次獨立訓練：學習臂在該 seed 下從頭訓練得到專屬 checkpoint（`{arm}_s{seed}.pt`），再於同一 seed 生成的工作串流上評估，故 8 seeds = 8 次獨立「訓練 + 評估」配對，且訓練 seed 與評估 workload seed 綁定同一數值。因此 seed 間變異同時涵蓋訓練隨機性與工作負載隨機性（一個較保守、不刻意分離兩者的設計）；而同一 seed 內所有 arm 共用相同工作串流（CRN），確保配對比較在該 seed 下的公平性。

此方法學的目的不是讓 DRL 看起來較好，而是誠實判定觀察到的差異是否穩健。

### 5.6 Results

**模擬結果。** 在 trace-derived AI workload 中，啟發式與學習式策略可明顯區分 FCFS、multifactor 與 score 等方法。模擬結果顯示，排程器若能感知工作大小、MPS 配置與 SLO，可降低推論工作 JCT 與 SLO violation。這說明模擬器本身具備區分策略的能力。

表 2. 模擬環境下 AI-serving 工作負載的排程器比較（8 個 held-out workload seed，mean ± std；score 使用 sim 預設 ε=0.30，含 SJF-like kicker）

| 排程器 | 平均 JCT (s) | 推論 JCT (s) | SLO 違反 (%) | 使用率 |
|---|--:|--:|--:|--:|
| FCFS | 2199 ± 1201 | 1847 ± 1151 | 66.5 ± 23.2 | 0.58 |
| multifactor | 1108 ± 398 | 461 ± 264 | 41.1 ± 20.1 | 0.63 |
| score | 1129 ± 398 | 520 ± 331 | 40.7 ± 19.0 | 0.63 |

seed 間變異雖大（工作串流隨機性所致），但 SLO 違反率上 FCFS（66.5%）明顯高於 size-aware 的 multifactor/score（≈41%），此區隔在計入 seed 變異後仍成立——這是「模擬器能區分策略」的正面證據。原始資料見 `runs/aiserve_eval_s42/SUMMARY.md`（seeds 100–108）。

**實機 cuBLAS 低負載共置。** 在 RTX 4070 與 RTX 3080 實機上，以 cuBLAS 工作與 **MPS fraction** 共置進行 8 seed 評估。結果顯示，FCFS、Backfill 與 RDSAC-mean 大致落在 score heuristic 的 ±5% 範圍內，代表此低負載 regime 的策略空間接近平坦；RDSAC-cvar、RLPD 與 SAC 未穩健勝出。

表 3. 實機 cuBLAS 工作負載評估（8 seeds，mean ± std；ΔJCT% 為 seed-level paired difference vs. score heuristic）

| Arm | JCT (s) | P99 (s) | CVaR (s) | ΔJCT% vs. Score |
|---|--:|--:|--:|--:|
| Score heuristic | 6.8 ± 0.7 | 22.1 ± 1.3 | 14.9 ± 1.7 | baseline |
| FCFS | 6.8 ± 0.6 | 21.9 ± 1.4 | 14.8 ± 1.8 | −0.1 ± 3.0 |
| Backfill | 6.8 ± 0.6 | 22.2 ± 1.3 | 14.8 ± 1.8 | −0.1 ± 2.1 |
| RDSAC-mean | 7.0 ± 0.8 | 24.0 ± 3.7 | 15.8 ± 2.5 | −1.7 ± 3.0 |
| RLPD | 7.1 ± 0.9 | 23.9 ± 3.8 | 15.9 ± 2.4 | −3.6 ± 4.8 |
| RDSAC-cvar | 7.2 ± 0.8 | 25.2 ± 4.5 | 16.2 ± 2.9 | −4.6 ± 4.9 |
| SAC | 7.2 ± 0.7 | 24.3 ± 3.6 | 16.1 ± 2.2 | −5.7 ± 7.1 |

**統計註記**：Holm-Bonferroni 校正後，所有 learning-based arms 均未顯著優於 Score (adjusted *p* > 0.05)；TOST ±5% 顯示 FCFS、Backfill、RDSAC-mean 與 Score 統計等價。

**實機混合 AI 工作負載（真四路，load-125）。** 在 BERT inference、ResNet training、Qwen fine-tuning 與 cuBLAS 矩陣運算四路混合、每 seed 125 個工作的高負載下，六個排程器在 average JCT 上彼此接近：score 的平均 JCT 數值最低，SAC 幾乎持平（−1.0%），RDSAC-cvar 次之（−4.1%），FCFS、Backfill 與 RDSAC-mean 則方向上落後 9–14%。經 Holm-Bonferroni 校正後，沒有任何 arm 與 score 有顯著差異（最接近者為 FCFS，*p*<sub>adj</sub> = 0.075）。此為本研究負向結果在真四路、重負載下的直接驗證：學習式 placement 未能勝過 score，且 score 相對於樸素 baseline 的領先在此 seed 數下亦不具統計穩健性。為避免尾端工作被靜默丟棄而造成倖存者偏差，本評估對每個 arm 逐工作記錄終態並依工作類別普查完成率（`states.json`）；六個 arm 均完成 124–125/125 個工作、四類工作（含 Qwen fine-tuning）皆正常結束，故表 4 各項平均可排除倖存者偏差。

表 4. 實機混合 AI 工作負載評估（真四路，每 seed 125 個工作，8 seeds，mean ± std；「完成」為 COMPLETED 工作數 / 125；ΔJCT% 為 seed-level paired difference vs. Score heuristic，正值表示快於 score）

| Arm | 完成 | Avg JCT (s) | P95 (s) | P99 (s) | Makespan (s) | GPU 利用率 | Slowdown | SLA Viol. | ΔJCT% vs. Score |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Score heuristic | 125 | 125.2 ± 19.3 | 433.5 ± 146.4 | 517.6 ± 122.2 | 675.2 ± 42.2 | 8.25 ± 1.30 | 27.57 ± 4.58 | 1.00 ± 0.01 | baseline |
| SAC | 124 | 126.2 ± 19.8 | 471.2 ± 86.2 | 566.5 ± 49.9 | 680.1 ± 37.7 | 8.00 ± 1.23 | 28.81 ± 4.52 | 1.00 ± 0.01 | −1.0 ± 7.8 |
| RDSAC-cvar | 125 | 130.5 ± 23.3 | 395.0 ± 132.9 | 527.7 ± 110.6 | 692.6 ± 31.0 | 8.32 ± 1.43 | 29.67 ± 5.63 | 1.00 ± 0.01 | −4.1 ± 9.3 |
| Backfill | 125 | 136.2 ± 23.4 | 255.6 ± 52.5 | 269.8 ± 50.1 | 693.9 ± 51.1 | 8.68 ± 1.39 | 30.69 ± 5.18 | 1.00 ± 0.01 | −8.9 ± 10.8 |
| FCFS | 125 | 137.0 ± 18.3 | 241.2 ± 41.7 | 255.0 ± 43.7 | 679.2 ± 39.1 | 8.60 ± 1.24 | 33.08 ± 4.14 | 1.00 ± 0.01 | −10.0 ± 8.8 |
| RDSAC-mean | 124 | 142.3 ± 38.8 | 469.7 ± 152.7 | 600.3 ± 77.1 | 714.1 ± 53.0 | 8.59 ± 1.81 | 32.78 ± 8.94 | 0.98 ± 0.02 | −13.9 ± 26.0 |

**統計註記**：Holm-Bonferroni 校正後，六個 arm 對 score 皆未達顯著（*p*<sub>adj</sub> ∈ [0.075, 0.720]）。TOST 等價檢定（seed-level 90% CI）顯示：SAC 與 score 在 ±10% 內統計等價（90% CI = [−6.2%, +4.2%]）；RDSAC-cvar 僅在 ±15% 內等價（[−10.3%, +2.1%]）；FCFS 與 Backfill 的 90% CI 完全落在 0 以下（分別 [−15.8%, −4.1%]、[−16.1%, −1.6%]），方向上一致慢於 score，但在 n=8 下未達 Holm 顯著。pooled SD = 14.3% → MDE(n=8, power .8) ≈ 16.5%，即本規模僅能偵測大於約 16.5% 的效應。P95/P99 呈現一個取捨：FCFS/Backfill 以嚴格序列執行換得較低尾端（P99 ≈ 255–270 s）但平均 JCT 較差；score 與學習式 arm 以較積極的 MPS 共置壓低平均，代價是較重的尾端（P99 ≈ 518–600 s）。SLA 違反率在此負載下對所有 arm 皆近飽和（≈1.00），不具鑑別力。

**統計解讀。** 多重比較校正後，學習式方法在兩個實機場景中皆未穩健勝過 score heuristic。cuBLAS 場景中，RDSAC-mean、FCFS 與 Backfill 可被視為與 score 在 ±5% 內統計等價；真四路重負載（load-125）場景中，六個 arm 經 Holm 校正後皆與 score 無顯著差異，其中 SAC 與 score 在 ±10% 內 TOST 等價，FCFS 與 Backfill 方向上慢於 score 但未達統計穩健。**所有統計檢定均使用 Holm-Bonferroni 多重比較校正；等價性以 TOST (two one-sided tests) 判定（cuBLAS 場景採 ±5% 邊界，load-125 場景因效應量與變異較大，改報 ±10%/±15% 邊界）。**

整體而言，本研究的主要發現是：DRL 能在模擬中學到可區分的策略，但在小規模真實異質 GPU + MPS 叢集中，學習式策略的優勢尚未穩健轉移；此結論在低負載 cuBLAS 與真四路重負載（load-125）兩個實機場景中一致成立。這不否定 DRL 排程的潛力，而是說明此類研究必須以真實部署、統計檢定與場景依賴性作為必要評估條件。

### 5.7 Ceiling Analysis

前述結果顯示，RDSAC 與 SAC 在本研究測試的實機負載下皆未穩健勝過啟發式。這個負向結果究竟是特定 RL 方法的局限，還是測試 regime 本身就沒有太多空間可贏？為了把這兩種可能性分開，在模擬器中進行一項獨立於任何學習式方法的天花板分析：2×1 叢集上，若 GPU/MPS placement 固定為 cluster allocator 的既定政策（對所有排程器一致），排程器唯一能控制的槓桿只剩下 dispatch **ordering**，一個 schedule 即為工作優先序的一個 permutation。本研究對 {philly, ali, burst} 三個 family、各 10 seeds，以 random-restart + swap local search（以 clairvoyant SJF 為種子）搜尋每個 instance 在所有 ordering 中可達到的最佳平均 JCT，並定義 `headroom% = (score_JCT − best_ordering_JCT) / best_ordering_JCT`。此搜尋法在小型高競爭 instance 上與窮舉所有 permutation（work-conserving priority-list schedule 的精確最優解）100% 吻合，顯示其足以逼近真正上界。由於啟發式本身是動態排程器，此上界理論上仍可能被超越；但實證上 clairvoyant SJF-static 的表現與 score 幾乎相同，故 static ordering 這個較窄的類別已足以作為一個緊致（tight）的可達上界。

表 5 彙整不同負載（2-GPU 叢集上的 n_jobs）下的 headroom，結果顯示 headroom 並非固定值，而是隨負載單調上升。

表 5. Headroom vs. 負載（2-GPU 叢集，3 families × 10 seeds/row，n=30）

| 負載 (n_jobs) | Headroom（mean ± 95% CI） | Max |
|---|--:|--:|
| 40 | +0.1% ± 0.1% | +1.5% |
| 60 | +0.7% ± 0.5% | +5.8% |
| 80 | +2.0% ± 1.1% | +9.7% |
| 100 | +4.1% ± 2.3% | +25.2% |

本研究的 cuBLAS 實機比較與模擬比較所用測試負載約為 n_jobs≈50，對應表 5 中 load 40–60 區間；該區間 headroom 僅 0.1–0.7%，即 score 在此負載下已幾乎位於可達到的排程上界。即使負載提高到 n_jobs=100，headroom 也僅約 4.1%。值得注意的是，表 4 的真四路實機評估已把負載直接推到 n_jobs=125——即表 5 外推 headroom 更大的區間——而學習式方法在該負載下仍未勝過 score（方向上甚至落後）。這說明即使 headroom 隨負載成長使「可贏空間」變大，學習式策略在本叢集規模下仍未能把該空間轉化為實際增益。換言之，第 5.6 節的核心負向結果——學習式 placement 未能穩健勝過 score——在低負載下具結構性成因（幾乎沒有 headroom 可贏），在重負載（load-125）下則是另一種成因（headroom 存在但未被 DRL 捕捉）；兩者皆非 RDSAC 或 SAC 這類特定方法的偶然失敗。

排程另一個可能的槓桿是 GPU/MPS placement 本身，而非 ordering。第 5.8 節的 joint-vs-decoupled 消融顯示，這個槓桿在本研究的叢集規模下同樣接近無效。將本節的 ordering headroom 與該 placement 消融合併來看，兩者共同界定了 score 之上的總可贏空間：在本研究測試的負載下，ordering 貢獻了幾乎全部、但仍極小的可贏空間，而 placement 幾乎不貢獻。整體而言，本節分析把第 5.6 節的負向結果重新定位為結構性而非方法性。可重現性材料見 `runs/headroom_20260721-134003/`，對應腳本為 `eval/scripts/headroom_analysis.py` 與 `sim/scheduler/fixed_priority.py`。

### 5.8 Placement 解耦消融

§5.7 的天花板分析針對 ordering 槓桿；本節以一項解耦消融檢視另一個槓桿——GPU/MPS placement 在學習式策略中貢獻多少。系統的 DRL 動作同時決定「選哪個 job」與「放到哪張 GPU」兩件事。為分離兩者，本研究在 action mask 上加入限制，訓練三個共用相同網路形狀（obs_dim / n_actions 不變）的臂：**joint**（完整動作空間，同時學 job 選擇與 placement）、**placement_only**（job 選擇凍結為 score heuristic 的首選，只學 placement）、**job_only**（placement 凍結為 first-fit，只學 job 選擇）。三臂各以 5 個 training seed、於 2×1 異質叢集（RTX 4070 + RTX 3080，含 node-2 host-RAM 限制）訓練 50k steps，並在 philly 與 ali 兩個 family × 5 個 eval seed 上以貪婪策略評估、取 pooled 平均 JCT；以 training seed 為配對單位對 joint 做配對比較。

表 6. Joint-vs-Decoupled placement 消融（2×1，3 臂 × 5 training seeds，pooled JCT over families × seeds）

| 臂 | 平均 JCT (s) | Δ vs joint | 配對 *p* |
|---|--:|--:|--:|
| joint | 9150 ± 2335 | — | — |
| placement_only | 8443 ± 2631 | −5.3% ± 33.8% | 0.585 |
| job_only | 10540 ± 1358 | +19.6% ± 35.7% | 0.295 |

兩個解耦臂皆未與 joint 產生統計顯著差異（配對 *p* 分別為 0.585 與 0.295，95% CI 皆橫跨 ±34% 的零點）。其中 placement_only 與 joint 實質等價，代表在此規模下 DRL 學到的 job 選擇並未帶來穩健增益——score heuristic 的既有排序已足夠；job_only 則在方向上較差（+19.6%，但不顯著），暗示 placement 相較 job 排序是稍微更有用的槓桿，惟證據不足以定論。整體而言，placement 這個槓桿在 2×1 規模下接近無效，與 §5.7 的 ordering 天花板一致：兩者共同說明 score 之上的總可贏空間在測試負載下極小。

此消融的鑑別力受兩項因素限制：其一，僅 5 個訓練 seed，且尾端指標在此規模下 per-instance 變異大（寬 CI）；其二，本模擬的事件模型在低負載下使每個決策點的待排佇列偏薄，job 選擇的候選數少，因而低估了 job-selection 槓桿在高負載下可能的作用——這與 §5.7「headroom 隨負載上升」的觀察一致。可重現性材料見 `runs/ablation_std_20260720-225206/`，對應腳本為 `eval/scripts/ablation_joint_decoupled.py`。

## 6. 結論與未來工作

### 6.1 結論

本研究驗證了 DRL 能於異質 GPU 與 NVIDIA MPS 環境中學習 GPU placement 與 **MPS fraction** [5]，並能透過 Slurm job submission path 整合到真實排程流程 [11]。相較傳統只選 GPU 或只做固定 rule 的方法，本研究將 GPU 型號差異、MPS 配額、工作特徵、佇列狀態與回饋訊號整合成一個可學習的排程框架。

實驗結果顯示，學習式策略在模擬環境可區分 FCFS、Backfill 與啟發式策略，但在 RTX 4070 與 RTX 3080 的小規模實機環境中，尚未穩健勝過啟發式；此結論在低負載 cuBLAS 與真四路重負載（load-125）兩個場景皆一致成立。在真四路重負載下，score 的平均 JCT 數值最低、SAC 與其在 ±10% 內統計等價；MPS/VRAM 感知的 score 相對 Slurm Backfill 雖方向上較快，但在本 seed 數下未達統計顯著（*p*<sub>adj</sub> = 0.21），故本研究不將其宣稱為穩健的正面結果。整體而言，在異質 GPU + MPS 排程中，策略效益高度依賴工作負載、硬體規模與底層資源分配後端。

### 6.2 限制

本研究仍有以下限制：

1. **叢集規模有限**：目前實機環境只有 RTX 4070 與 RTX 3080 兩張 GPU，無法直接代表數十至數百 GPU 的生產叢集。
2. **只使用 MPS fraction**：本研究尚未納入 MIG，也未處理 MIG 與 MPS 混合 partition [5][21]。
3. **未涵蓋多節點大規模訓練**：目前 focus 在單卡或小規模 job replay，尚未完整處理多 GPU gang scheduling。
4. **真實 transition 數量有限**：RLPD 使用的實機資料量仍不足，可能無法充分縮小 sim-to-real gap。
5. **尾端指標樣本數不足**：P99 與 CVaR 在小規模實機實驗中變異較大，因此本文對尾端結果採保守解讀。

### 6.3 未來工作

未來工作可朝以下方向擴展：

1. **重負載 regime 與多 GPU cluster**：§5.7 的天花板分析顯示啟發式之上的可贏空間隨負載上升——在低負載 cuBLAS 場景（n_jobs≈50）下僅 0.1–0.7%，即使提高到 n_jobs=100 也僅約 4.1%；§5.8 則顯示 placement 槓桿在本規模下接近無效，故此空間主要來自 dispatch ordering。本研究已將真四路實機評估推進到 n_jobs=125（表 4）——即 headroom 外推更大的重負載區間——結果顯示學習式方法在該負載下仍未勝過 score（方向上甚至落後），故「重負載能翻轉負向結論」這個假設，在本叢集規模的現有證據下並不成立。惟須強調三點誠實限制：(a) headroom 是最佳靜態排序的可達上界，僅代表「空間存在」，並不保證 DRL 能實際逼近，而表 4 正說明 load-125 下 DRL 尚未逼近該上界；(b) 本研究的重負載實機證據止於 n_jobs=125、2×1（單節點單 GPU）、8 個 workload seed，尚未涵蓋更大 n_jobs 與更大叢集，故「更大規模是否翻轉」仍為有待驗證的假設，而非已成立的宣稱；(c) 尾端指標在此規模下 per-instance 變異大（pooled SD ≈ 14.3%、MDE ≈ 16.5%），實機驗證需顯著更多 workload seed 方能達到統計顯著性。延伸此軸的另一方向是擴展至更多節點與 GPU，檢驗效益是否隨叢集規模進一步浮現。
2. **MIG + MPS fraction 混合 partition**：同時納入硬體級隔離與軟體級共享，建立更完整的 GPU sharing action space。
3. **Offline RL / RLPD**：收集更大量真實 Slurm transition，以 offline RL 或 RLPD 改善 sim-to-real 轉移 [8]。
4. **Multi-Agent RL**：在多節點與多 GPU 場景中，探索分散式或階層式排程代理人。
5. **Energy-aware scheduling**：將功耗、能效與碳排納入 reward，使排程器不只最佳化效能，也最佳化能源效率。
6. **LLM serving workload**：加入更真實的 LLM serving trace，評估 token latency、throughput、SLO violation 與 batch scheduling 的交互影響；近期如語意感知的 LLM 叢集排程 [29] 顯示 LLM workload 特性可進一步被排程器利用。

## 參考文獻

[1] M. Jeon, S. Venkataraman, A. Phanishayee, et al., "Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads," in *USENIX ATC*, 2019.

[2] W. Xiao, R. Bhardwaj, R. Ramjee, et al., "Gandiva: Introspective Cluster Scheduling for Deep Learning," in *USENIX OSDI*, 2018.

[3] J. Gu, M. Chowdhury, K. G. Shin, et al., "Tiresias: A GPU Cluster Manager for Distributed Deep Learning," in *USENIX NSDI*, 2019.

[4] Q. Weng, W. Xiao, Y. Yu, et al., "MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters," in *USENIX NSDI*, 2022.

[5] NVIDIA Corporation, "Multi-Process Service (MPS)," NVIDIA Documentation, 2024.

[6] P. Christodoulou, "Soft Actor-Critic for Discrete Action Settings," *arXiv:1910.07207*, 2019.

[7] W. Dabney, G. Ostrovski, D. Silver, and R. Munos, "Implicit Quantile Networks for Distributional Reinforcement Learning," in *ICML*, 2018.

[8] P. J. Ball, L. Smith, I. Kostrikov, and S. Levine, "Efficient Online Reinforcement Learning with Offline Data (RLPD)," in *ICML*, 2023.

[9] A. Y. Ng, D. Harada, and S. Russell, "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping," in *ICML*, 1999.

[10] Kubernetes Authors, "Kubernetes," https://kubernetes.io, 2024.

[11] SchedMD, "Slurm Workload Manager," https://slurm.schedmd.com, 2024.

[12] Kubeflow Authors, "Kubeflow: The Machine Learning Toolkit for Kubernetes," https://www.kubeflow.org, 2024.

[13] Volcano Authors, "Volcano: A Cloud Native Batch System for Compute-Intensive Workloads," CNCF, https://volcano.sh, 2024.

[14] Kubernetes SIG-Scheduling, "Kueue: Kubernetes-native Job Queueing," https://kueue.sigs.k8s.io, 2024.

[15] Kubernetes Authors, "Dynamic Resource Allocation (DRA)," Kubernetes Documentation, 2025.

[16] X. Ma, J. Chen, L. Xia, J. Yang, Q. Zhao, and Z. Zhou, "DSAC: Distributional Soft Actor-Critic for Risk-Sensitive Reinforcement Learning," *arXiv:2004.14547*, 2020.

[17] J. Duan, Y. Guan, S. E. Li, Y. Ren, and B. Cheng, "Distributional Soft Actor-Critic: Off-Policy Reinforcement Learning for Addressing Value Estimation Errors," *IEEE Transactions on Neural Networks and Learning Systems*, 2021.

[18] A. Bhatt, D. Palenicek, B. Belousov, M. Argus, A. Amiranashvili, T. Brox, and J. Peters, "CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity," in *ICLR*, 2024.

[19] X. Wang, Y. Li, F. Guo, Y. Xu, and J. C. S. Lui, "Dynamic GPU Scheduling With Multi-Resource Awareness and Live Migration Support," *IEEE Transactions on Cloud Computing*, vol. 11, no. 3, 2023.

[20] H. Sedighi, F. Wuhib, and R. H. Glitho, "Dynamic Task Scheduling and Adaptive GPU Resource Allocation in the Cloud," *IEEE Transactions on Network and Service Management*, vol. 23, 2026.

[21] E. Lipe, N. Karia, C. Espenshade, C. Stein, A. Tantawi, and O. Tardieu, "Energy Efficient Scheduling of AI/ML Workloads on Multi Instance GPUs with Dynamic Repartitioning," in *IEEE CCGrid*, 2025.

[22] M. Tsenos and V. Kalogeraki, "Exploring GPU-Based Workload Scheduling Techniques for Edge Computing," in *IEEE IC2E*, 2025.

[23] Y.-D. Lin, Y.-T. Ling, Y.-C. Lai, and D. Sudyana, "Reinforcement Learning for AI as a Service: CPU-GPU Task Scheduling for Preprocessing, Training, and Inference Tasks," *IEEE Transactions on Network and Service Management*, vol. 22, no. 4, 2025.

[24] A. A. Majeed, M. Meribout, and S. M. Sali, "Scheduling Techniques of AI Models on Modern Heterogeneous Edge GPU: A Critical Review," *IEEE Transactions on Industrial Informatics*, vol. 22, no. 4, 2026.

[25] G. Zhang, W. Guo, Z. Tan, Q. Guan, and H. Jiang, "KIS-S: A GPU-Aware Kubernetes Inference Simulator with RL-Based Auto-Scaling," *arXiv:2507.07932*, 2025.

[26] Q. Wu, P. Chen, and Y. Wang, "Defragmentation Scheduling with Deep Reinforcement Learning in Shared GPU Clusters," in *ACM SoCC*, 2025.

[27] SchedMD, "Slinky: Slurm in Kubernetes," https://github.com/SlinkyProject, 2024.

[28] Y. Dong, X. Zheng, X. Pan, and D. Liu, "A reinforcement learning-based approach for scheduling machine learning training tasks in heterogeneous Kubernetes clusters," *Future Generation Computer Systems*, 2026.

[29] Y. Wang, Y. Hu, A. Klimovic, X. Zhang, Y. Wen, G. Sun, and J. Lin, "Semantic-Aware Scheduling for GPU Clusters with Large Language Models," *arXiv:2510.03334*, 2025.

[30] S. Dongare, R. I. S. Khan, H. Albahar, N. Zhao, D. Melendez Maita, and A. R. Butt, "Hybrid Learning and Optimization-Based Dynamic Scheduling for DL Workloads on Heterogeneous GPU Clusters," in *ACM SoCC*, 2025.

[31] NVIDIA, "KAI Scheduler: A Kubernetes-Native GPU Scheduler for AI Workloads," https://github.com/NVIDIA/KAI-Scheduler, 2025.

[32] S. Choi, S. Lee, Y. Kim, J. Park, Y. Kwon, and J. Huh, "Serving Heterogeneous Machine Learning Models on Multi-GPU Servers with Spatio-Temporal Sharing," in *USENIX ATC*, 2022.

[33] U. Saroliya, E. Arima, D. Liu, and M. Schulz, "Hierarchical Resource Partitioning on Modern GPUs: A Reinforcement Learning Approach," in *IEEE CLUSTER*, 2023.
