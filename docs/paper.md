# 基於 Slurm 與 Kubernetes 架構下 AI 伺服器 GPU 工作負載智慧排程技術之研究

### Intelligent GPU Workload Scheduling Techniques for AI Servers under a Slurm-on-Kubernetes Architecture

**作者一¹、作者二²**
¹○○大學 ○○系　²○○大學 ○○系
{author1, author2}@stumail.nutn.edu.tw

---

## 摘要

隨著大型語言模型與生成式 AI 快速發展，GPU 已成為 AI 工作負載的主要運算資源 [1][4]。然而，多數實驗室與中小型叢集通常由不同世代 GPU 組成，且 NVIDIA Multi-Process Service (MPS) 允許多個工作共享單張 GPU [5]，使得傳統 Slurm 排程器 [11] 難以同時兼顧資源利用率、工作完成時間 (JCT) 與尾端延遲。現有啟發式策略（如 FCFS、Backfill）以固定規則為主，難以隨工作負載動態調整；既有的深度強化學習 (Deep Reinforcement Learning, DRL) 排程器則多停留在模擬環境，較少在真實 Slurm 流程中同時處理異質 GPU placement 與 MPS 配置。

本研究提出一套以 **Slurm-on-Kubernetes** 為核心的異質 GPU + MPS 智慧排程框架：Slurm 維持批次佇列與資源配置語意 [11]，Kubernetes (k3s) 僅負責容器化部署與生命週期管理 [10]，而學習式排程策略以 Slurm job submission hook 嵌入排程路徑，在 MPS 剩餘容量的約束下決定要從佇列派出哪個工作、並將其放置到哪張 GPU；當學習式服務逾時或異常時，系統自動回退至啟發式策略，確保排程核心不被阻塞。本研究於 RTX 4070 與 RTX 3080 實機環境建置平台，並以 Alibaba GPU Trace 與 Microsoft Philly Trace 特性進行 trace replay 及真實 AI 工作負載評估 [1][4]，比較 FCFS、Backfill、啟發式、Discrete SAC [6]、RDSAC [7][16] 與 RLPD [8]。結果顯示，學習式策略在平均 JCT 上優於 FCFS 與 Backfill 等傳統 Slurm 排程，其中以真實資料微調的 RLPD 取得最佳平均 JCT，而 RDSAC-cvar 則在尾端延遲上有所優化。本研究並以嚴謹的統計檢定（多重比較校正、TOST 等價檢定與天花板分析）刻畫學習式策略的效益邊界與場景依賴性。

**關鍵詞**：GPU 資源排程、異質 GPU、NVIDIA MPS、Slurm、Kubernetes、深度強化學習

## Abstract

The rapid growth of large language models and generative AI has made GPUs the primary compute resource for AI workloads [1][4]. However, many laboratory and small-scale clusters consist of heterogeneous GPU generations, and NVIDIA Multi-Process Service (MPS) allows multiple jobs to share a single GPU [5], making it difficult for traditional Slurm scheduling [11] to jointly optimize GPU utilization, job completion time (JCT), and tail latency. Existing heuristic policies such as FCFS and Backfill rely on fixed rules and cannot adapt to workload characteristics, while existing DRL schedulers rarely support heterogeneous GPU placement together with MPS allocation inside a real Slurm environment.

This paper proposes an intelligent scheduling framework for heterogeneous GPUs with NVIDIA MPS, built on a **Slurm-on-Kubernetes** architecture. Slurm retains batch queueing and resource-allocation semantics [11]; Kubernetes (k3s) only provides containerized deployment and lifecycle management [10]; and a learned policy is embedded into the scheduling path through a Slurm job submission hook, deciding—under a remaining-MPS constraint—which queued job to dispatch and onto which GPU to place it. If the learned service times out or fails, the system automatically falls back to a heuristic policy so the scheduling core is never blocked. We implement the framework on a real RTX 4070 and RTX 3080 testbed and evaluate it with workload characteristics derived from Alibaba GPU Trace and Microsoft Philly Trace [1][4], comparing FCFS, Backfill, a heuristic policy, Discrete SAC [6], RDSAC [7][16], and RLPD [8]. Results show that learned policies improve average JCT over traditional Slurm scheduling such as FCFS and Backfill—with real-data-finetuned RLPD achieving the best average JCT and RDSAC-cvar improving tail latency. We further characterize the efficacy boundary and scenario dependence of the learned policies with rigorous statistical testing, including multiple-comparison correction, TOST equivalence tests, and a ceiling analysis.

**Keywords**: GPU Resource Scheduling, Heterogeneous GPU, NVIDIA MPS, Slurm, Kubernetes, Deep Reinforcement Learning

---

## 1. 緒論

Slurm Workload Manager 是高效能運算叢集最廣泛使用的工作排程系統之一，提供完整的工作提交、佇列管理、資源限額與 GPU 資源配置功能 [11]，其以「節點」為基礎的資源模型非常適合研究工作環境。然而 Slurm 原生設計以固定實體節點為主，對彈性擴縮與容器化整合的支援相對有限。Kubernetes 則是目前最主流的容器編排平台，能自動管理容器的部署、擴縮與健康監控 [10]，但其原生排程器 (kube-scheduler) 以服務導向設計為主，對批次工作、GPU 共享與研究工作特有的排程需求支援不足。因此有多項研究嘗試將 Slurm 與 Kubernetes 整合，以同時取得批次排程語意與雲端彈性管理能力；實作層面也有 Slinky [27] 等 Slurm-on-Kubernetes 方向的工具，顯示此類架構已具有實務需求與發展基礎。

### 1.1 AI 工作負載快速增加

近年大型語言模型、生成式 AI 與深度學習服務快速發展，使 GPU 成為 AI 訓練與推論最重要的運算資源 [1][4]。相較於傳統批次運算，AI 叢集經常同時承載短時間推論、模型微調、長時間訓練與矩陣運算等不同工作，這些工作在執行時間、記憶體需求、延遲敏感度與 GPU 使用型態上皆有明顯差異 [1][4]。

在大學實驗室與中小型研究叢集中，GPU 資源通常有限，且硬體常由不同世代 GPU 漸進式擴充而成。例如 RTX 4070 與 RTX 3080 在算力、記憶體容量與功耗上皆不同。若排程器只把 GPU 視為同質資源，便可能讓小型推論工作佔用高效能 GPU，或讓長時間訓練阻塞後續短工作，造成 GPU utilization、JCT 與 queue delay 之間的取捨更加困難。

NVIDIA MPS 提供另一個重要槓桿：多個 CUDA 工作可共享同一張 GPU，使 GPU 不再只能以整張卡為單位分配 [5]。然而 MPS 也讓排程問題從「選哪張 GPU」變成「選哪張 GPU 與該工作需要多少 **MPS fraction**」。因此，在異質 GPU 與 MPS 共存的環境中，GPU scheduling 已成為影響叢集效能的核心問題。

### 1.2 現有方法限制

Slurm 提供 FCFS、Backfill、multifactor priority 與 GRES/TRES GPU 資源管理 [11]，但其傳統策略多以固定規則為主，難以感知異質 GPU、MPS 分片、工作類型與長期回報之間的相互影響。現有方法主要存在四項限制：

1. **不充分考慮 GPU 差異**：許多排程方法把 GPU 視為同質資源，較少將不同世代 GPU 的算力、記憶體與執行時間差異納入 placement 決策。
2. **不充分考慮 MPS fraction**：部分研究討論 GPU sharing 或 GPU partition，但未將 **MPS fraction** 需求作為排程器的顯式感知變數。
3. **依賴固定規則**：FCFS 與 Backfill 能提供穩定基準，但難以隨工作負載動態調整策略。
4. **缺乏真實 Slurm 流程驗證**：不少學習式排程研究停留在模擬環境，未整合到真實 Slurm job submission path，也未處理服務失效與實機部署。

### 1.3 為什麼使用 DRL

GPU scheduling 不是單次分類問題，而是序列決策問題。一次 placement 會改變 GPU 剩餘 MPS、後續佇列等待時間、工作共置干擾與未來可用資源，因此當下看似最佳的放置，未必能帶來長期最佳的 JCT 或利用率。此問題具有三個適合使用 DRL 的特徵：

1. **Sequential decision**：每次工作放置會影響後續所有排程決策。
2. **Long-term optimization**：排程目標不只包含當下工作的執行時間，也包含 queue delay、makespan、tail latency 與整體 GPU utilization。
3. **Dynamic environment**：工作到達率、工作類型、GPU 使用率與 MPS 剩餘容量會隨時間改變，固定 heuristic 難以完整涵蓋所有情境。

因此，本研究採用 DRL 作為排程策略學習方法，讓代理人從模擬與實機回饋中學習 GPU placement 與 MPS-aware 排程的長期效果。

### 1.4 研究貢獻

本研究的核心貢獻在於將「異質 GPU + MPS + 真實 Slurm 流程」三者結合，具體如下：

1. **異質 GPU placement 與 MPS-aware 的聯合排程動作空間**：不同於 UXP-RL [23] 僅決定 CPU-vs-GPU 的資源類型、KIS-S [25] 僅調整 Kubernetes 推論副本數、DRR [26] 僅針對碎片化，本研究讓單一學習式策略在真實 Slurm 佇列上同時決定「派哪個工作」與「放到哪張異質 GPU」，並以工作攜帶的 MPS fraction 需求（25%/50%/75%/100%）作為 state 特徵與 action mask 的可行性約束，使決策在尊重 MPS 配額的前提下進行。此設計首次在真實 Slurm 佇列上刻畫「job 選擇與異質 GPU placement 聯合決策」相對於僅決定資源類型 [23] 或僅調整副本數 [25] 的可行性與行為差異。

2. **失效安全的 Slurm 策略層整合**：不同於 UXP-RL、KIS-S 等純模擬研究，本研究把學習式決策服務嵌入真實 Slurm job submission path，並提供 fail-safe fallback——當 RL 服務逾時、回傳無效 action 或健康檢查失敗時自動回退至啟發式策略，使排程核心 (slurmctld) 永不被阻塞。此設計讓 DRL scheduler 得以在真實排程路徑中部署，而不需修改 Slurm 核心。本研究因此證明學習式排程可在不更動 slurmctld 的前提下安全嵌入生產排程路徑，這是先前純模擬研究 [23][25] 未曾示範的部署可行性。

3. **啟發式與學習式排程之實機效益比較分析**：於 RTX 4070/3080 異質環境，以 trace replay 與真實混合 AI 工作負載（load-125）比較 FCFS、Backfill、啟發式、SAC、RDSAC 與 RLPD，並以多 seed 配對、Holm-Bonferroni 校正與 TOST 等價檢定嚴謹檢驗。分析得到兩項純模擬難以提供的實機知識：(1) 學習式策略在平均 JCT 上優於傳統 Slurm 排程（FCFS/Backfill）；(2) 但在本研究實測的小規模叢集 regime 下，其相對已 size-aware 的啟發式並未構成統計上穩健的超越，模擬天花板分析進一步顯示 score 之上的可贏空間在低負載近乎平坦、須更高負載才顯現。此結果提示「DRL 必然勝過啟發式」的預設在真實異質小叢集上需重新檢視（詳 §5.4–5.7）。

## 2. 相關研究

### 2.1 GPU 叢集排程、分析與資源共享

近期研究聚焦於單節點內或單一叢集模型上的動態資源調度本身。Wang 等人的 DCUDA [19] 針對單節點多 GPU 情境，設計了一套輕量級核心／記憶體使用率監控機制，搭配近乎零開銷的「執行中」CUDA 應用即時遷移，將 GPU 過載時間平均降低 78.3%、一般工作執行時間降低 42.1%（記憶體密集型工作最高 67%）。Sedighi 等人 [20] 則在 Alibaba 的 cluster-trace-gpu 生產工作負載軌跡之上，提出結合硬體與軟體分割的公平且需求感知動態資源配置演算法，於模擬環境中將 GPU 資源使用量降低達 88%。這兩項工作皆聚焦「資源配置本身如何隨工作負載動態調整」（即時遷移／再分割），評估分別侷限於單一多 GPU 節點與純模擬 trace 重放，並未涉及與批次排程器（如 Slurm）的整合。

在 GPU 共享機制方面，NVIDIA MPS 允許多個 CUDA process 同時共享同一張 GPU 的運算資源 [5]，NVIDIA MIG 則在硬體層面將 GPU 切分成隔離的 instance [21]。學術上亦有更細緻的 partition 與時空共享研究，如多 GPU 伺服器上的時空共享服務 [32] 與現代 GPU 的階層式資源分割 [33]。針對深度學習工作負載，也有專門的叢集排程系統：Gandiva [2] 以 introspective 排程與工作遷移提升利用率，Tiresias [3] 以近似 age-based 的優先序縮短 JCT，MLaaS 生產叢集分析 [4] 則刻畫了大規模異質 GPU 叢集的工作特性。這些方法多假設整卡分配或同質 GPU，較少把 MPS fraction 作為排程流程的顯式決策脈絡。

### 2.2 強化學習排程

Lin 等人 [23] 提出 UXP-RL：一個以 DQN 為核心、涵蓋前處理／訓練／推論三類任務、可部署為集中式或分散式排程器、並跨雲／邊／霧三層架構運作的 CPU-GPU 任務排程演算法。其於**模擬環境**中，集中式排程器將平均週轉時間相較 SJF／FCFS 與 TYPE 啟發式分別降低 57.81%、57.28% 與 27.66%；分散式排程器則因能將長訓練工作卸載至雲端而把推論任務週轉時間再降低 89.07%。同年 Zhang 等人的 KIS-S [25] 以 PPO 訓練一個 GPU-aware 的 Kubernetes 推論自動擴縮策略，完全於自建模擬器 (KISim) 中訓練後零樣本部署，於多種流量情境下平均獎勵提升 75.2%；其問題設定是調整副本數的自動擴縮，而非本研究的工作放置排程。Wu 等人的 DRR [26] 則針對 GPU 共享叢集的碎片化問題，以模仿學習從既有啟發式暖啟動一個深度強化學習去碎片化排程器，並同時於實體 Kubernetes 測試床與大規模模擬叢集上驗證，平均碎片率降低 50%，是少數同時涉及真實 Kubernetes 部署的學習式排程器。

在演算法基礎方面，Discrete SAC 將最大熵框架延伸到離散動作空間，以 categorical 策略取代高斯策略、以期望估計熵項 [6]；分布式評論家（如 IQN [7]、DSAC [16]）以回報分布與風險敏感目標處理尾端延遲；以 BatchNorm 移除 target network 以提升樣本效率的 CrossQ [18] 則代表近期簡化訓練流程的方向；RLPD [8] 進一步以對稱取樣真實資料的方式進行 offline-to-online 微調，降低從零探索的成本。然而，現有 RL scheduler 多假設單一 GPU、同質 GPU 或大型模擬叢集，或將 Kubernetes 當作排程主體（如 Kubeflow [12]、Volcano [13]、Kueue [14]、NVIDIA KAI [31]）；較少探討在真實 Slurm 流程中，如何讓 RL 同時決定異質 GPU placement 與 MPS-aware 排程，並在服務失效時維持排程安全。

### 2.3 異質與邊緣 GPU 排程

Tsenos 與 Kalogeraki [22] 針對缺乏原生虛擬化支援的邊緣 GPU（如消費級卡）提出一套硬體無關的時空共享機制：為每個行程建立 cgroup、動態調整其 duty cycle 來實現優先權式與截止期限式排程，且無需修改工作負載原始碼即可整合進 TensorFlow、PyTorch、FFmpeg 等既有框架。Majeed 等人 [24] 則以系統性文獻回顧整理 NVIDIA Jetson 系列邊緣 SoC 上的 DNN 排程器，區分規則式與最佳化式兩大類，並整理其記憶體競爭、跨加速器轉移成本與靜態／動態排程的權衡；其排程粒度是單一 DNN 模型內的層級（將個別網路層指派給不同硬體加速器）。這些邊緣場景的資源與延遲約束與資料中心叢集不同，惟顯示異質硬體上的細粒度排程是一個活躍的研究方向。此外，異質 Kubernetes 叢集上的 RL 排程 [28]、混合學習與最佳化排程 [30] 與語意感知的 LLM 叢集排程 [29] 亦顯示學習式方法在異質資源分配上的潛力。

### 2.4 定位比較

表 1 彙整本研究與主要研究類型的定位差異。本研究的核心位置在於將「異質 GPU + MPS + 真實 Slurm 流程」三者結合；Kubernetes 僅是部署平台 [10]，不是本文的主要排程貢獻。

表 1. 相關研究定位比較

| 類型 | 是否考慮異質 GPU | 是否考慮 MPS fraction | 是否整合真實 Slurm | 主要限制 |
|---|:--:|:--:|:--:|---|
| FCFS / Backfill [11] | 部分 | 部分 | 是 | 固定規則，難以學習長期效果 |
| GPU sharing / partition [5][21][32][33] | 部分 | 部分 | 否 | 多聚焦 partition 機制，較少處理排程流程 |
| RL scheduler（模擬）[23][25] | 部分 | 少 | 否 | sim-to-real 效益不明 |
| RL scheduler（真實 K8s）[26] | 部分 | 少 | 否 | 針對碎片化，非 Slurm 工作流 |
| Kubernetes GPU scheduler [12][13][14][31] | 部分 | 部分 | 否 | Kubernetes 是排程主體，不處理 Slurm 工作流 |
| 本研究 | 是 | 是 | 是 | 目前實機規模仍小，需擴大驗證 |

## 3. 研究目的與系統架構

### 3.1 研究目的

本研究的目標是在異質 GPU 與 NVIDIA MPS 環境中，學習一個能在 MPS 配額約束下決定工作派遣與 GPU placement 的排程策略，以降低 average JCT 與尾端 JCT、同時提升 GPU utilization。更具體而言，本研究要回答兩個問題：第一，DRL 是否能在模擬環境中學到比固定規則更好的 GPU + MPS 排程策略；第二，此學習式策略部署到真實異質 GPU 叢集後，能否勝過 FCFS、Backfill 與啟發式 baseline。

本研究將異質 GPU + MPS 排程建模為馬可夫決策過程 (MDP)。每次有工作可排程時，代理人觀察叢集狀態，選擇一個工作與其 GPU placement，並在工作完成後根據 JCT 與 GPU utilization 取得 reward。

**State.** 狀態包含四類資訊：

1. **Job features**：工作類型、預估 runtime、GPU memory 需求、MPS 需求、SLO 緊迫度與等待時間。
2. **GPU features**：GPU 型號、GPU utilization、SM utilization、memory usage、可用 MPS、目前共置工作數。
3. **Queue features**：佇列長度、前 K 個工作的需求、等待時間分布與到達率。
4. **History features**：近期完成工作 JCT、slowdown、SLO violation 與各 GPU 的負載變化。

**Action.** 動作為「從佇列前 K 個工作中選一個」與「將其放置到哪張 GPU」的聯合離散決策，另含一個 no-op（暫不派遣）：

```text
Action = (選 job_i ∈ top-K 佇列) × (放置到 node_j / gpu_k)  ∪  {no-op}
```

在本研究的 2×1 實驗平台（K=16、2 個 placement）中，動作空間為 16×2+1 = 33 個離散動作（obs 維度 166）。**MPS fraction 不是動作維度**：每個工作攜帶自身的 MPS fraction 需求（25%/50%/75%/100%），排程器透過 state 特徵感知、並由 action mask 遮蔽 MPS 剩餘容量不足的放置，因此策略是在尊重工作 MPS 需求的前提下做 job 選擇與 placement，而非自由指派 fraction。

**Reward.** Reward 的設計目標是降低使用者感受到的等待與完成時間，同時提高 GPU 使用效率，故採以 JCT 為核心的多目標形式：

```text
R = w_jct · (−JCT / reward_scale) + w_util · GPUUtil
```

其中 w_jct=1.0、w_util=0.05、reward_scale=1000。JCT = 完成時間 − 提交時間，已把 queue delay 內含；GPUUtil 鼓勵 MPS 共置以免策略學到保守閒置。另可選用 potential-based reward shaping [9] 提供密集訊號而不改變最優策略，並以 opt-in 的 SLO 逾時懲罰處理延遲敏感工作。此 reward 設計讓 DRL 不只最佳化單一工作，而是學習長期排程結果。

### 3.2 系統架構

本研究建立一套以 Slurm 為排程核心的異質 GPU + MPS 智慧排程平台。系統流程如圖 1 所示。使用者提交工作後，Slurm 透過 job submission hook 呼叫 RL scheduler；RL scheduler 讀取目前工作資訊、GPU 狀態、MPS 剩餘容量與佇列資訊，輸出所選工作與其 GPU placement（工作的 MPS fraction 依其請求分配、非策略輸出）。工作執行期間，監控服務收集 GPU utilization、SM utilization、memory usage、queue delay、JCT 與 reward，並將資料寫入 replay buffer 供後續訓練或 RLPD (Reinforcement Learning with Prior Data) [8] 微調使用。

```mermaid
flowchart TD
    A[Job Submission] --> B[Slurm Controller]
    B --> C{RL Scheduler<br/>via job_submit hook}
    C -->|Observe State| D[GPU + MPS-aware Placement]
    D --> E[Execute Job on GPU]
    E --> F[Monitoring<br/>JCT, Util, Queue Delay]
    F --> G[Replay Buffer]
    G --> C
    C -.->|Timeout / Invalid Action| H[Heuristic Fallback<br/>Score-based]
    H --> B
```

**圖 1. 系統架構與排程流程。** 虛線框為 fail-safe fallback 路徑；RL Scheduler 以 Slurm `job_submit.lua` hook 介入，決策輸出為 `(job, GPU placement)`；Monitoring 週期 1 秒蒐集 GPU/SM/memory utilization、MPS 使用量、queue depth、job events，寫入 Replay Buffer 供 offline RL / RLPD 使用。

本平台使用 Kubernetes (k3s) 部署 Slurm controller、worker、RL scheduler、monitoring service 與相關容器 [10]，並透過 Kubernetes Dynamic Resource Allocation (DRA) 對 GPU 與 MPS 資源進行配置 [15]。**Kubernetes 在本文中不負責排程決策**；它只提供容器化部署、服務健康檢查、網路與生命週期管理。此設計避免讓 Kubernetes 成為研究主角，並保留 Slurm 在 HPC batch scheduling 中成熟的佇列語意 [11]。此方向與將 Slurm 整合進 Kubernetes 的 Slinky [27] 互補：Slinky 是官方 Slurm-on-K8s 部署基座（仍用 vanilla Slurm 排程），本研究則在其上疊加失效安全的 RL 策略層。

## 4. 排程技術

本章介紹本研究的 scheduler：以 Slurm 原生能力作為穩定基線，以啟發式評分函式作為可部署的中階基準，再以 DRL agent 學習序列排程策略。

### 4.1 Slurm 內建排程演算法

本研究以 Slurm 原生排程能力作為穩定基線，而非重寫排程核心。例如 Backfill 允許在不延後高優先權工作的前提下，讓資源需求較小、執行時間較短的工作提前插隊執行，緩解大工作長期佔用資源造成的閒置；此外也納入更保守的 FCFS 作為下界對照，用以檢驗啟發式與學習式策略相對 Slurm 開箱即用能力是否確有改善。

### 4.2 啟發式排程策略

啟發式策略以加權線性組合公式計算工作優先級，分數越高代表該工作越值得優先排程。三個因子及對應權重如表 2 所示。此 score heuristic 代表可部署、可解釋且低成本的生產啟發式方法。

表 2. 啟發式因子與係數定義

| 因子 | 係數 | 定義 |
|---|:--:|---|
| MPS | 0.40 | `mps_req/100 ∈ [0, 1]`。沒給=1.0，超過=0.0。 |
| VRAM | 0.20 | `(1 − (fit_tier − req))/max_tier ∈ [0, 1]`。依 job 需求選最小可用 VRAM tier，沒給=0.5。 |
| Frag | 0.20 | `4(x − 1), x = mps_req/100`。mps_req=0 或滿載→0.0；mps_req=50%→最高懲罰 ≈ −1.0。 |

需注意 score 在兩個評估管線的強度不同：模擬使用含 SJF-like runtime kicker 的完整版本（ε=0.30），而實機部署因 runtime predictor 未上線而停用該項（ε=0），僅保留 MPS-fit 與 VRAM-fit。兩者為同一評分函式的兩種設定，故模擬表與實機表的 score 絕對表現不宜直接跨表比較；各表內部的相對比較則不受影響。

### 4.3 深度強化學習策略

本研究比較以下 DRL 方法：

1. **Discrete SAC**：將 Soft Actor-Critic 延伸到離散 action space，以 categorical 策略取代高斯策略、以期望估計熵項，適合 job 選擇與 GPU placement 這類有限離散動作 [6]。
2. **RDSAC-mean**：以分布式 critic 建模回報分布，但 actor 主要依平均回報決策 [7][16]。
3. **RDSAC-cvar**：在 RDSAC 上加入 CVaR 風險敏感目標，使策略更重視尾端 JCT 與 SLO violation [16]。
4. **RLPD**：以真實資料對模擬訓練出的模型進行微調，縮小 sim-to-real gap [8]。本研究忠實採用原論文核心配方——每個 batch 對稱取樣 50% sim 先驗 + 50% 真實資料、critic 加 LayerNorm 的 critic ensemble、高 UTD——但訓練機制為離線（更新迴圈只做梯度更新、不在真環境即時互動），故為「sim + 真實混合 buffer 的離線微調」，非原論文的真線上更新。

RDSAC 採用雙頭 IQN critic 建模 reward return 與 entropy return [7]，並使用 masked categorical actor 避免選到不可執行 action（即所選放置的 GPU 剩餘 MPS 不足以容納該工作請求的動作）。訓練流程包含 prioritized replay、n-step return、potential-based reward shaping [9] 與 heuristic warm start。需澄清命名：本研究的 RDSAC 為自組的「distributional + discrete SAC」，以離散動作空間搭配 IQN 分位數 critic 建構 [7][16]，與 Duan 等人 [17] 針對連續控制、將回報建模為單一高斯分布的 Distributional Soft Actor-Critic 不同，兩者不應混淆。

## 5. 實驗與評估

### 5.1 實驗環境與訓練

本研究使用一個小規模異質 GPU 實機叢集進行部署與評估，相關環境如表 3 所示。此環境刻意保留 GPU 世代差異，讓排程器必須面對異質 GPU placement 的問題；由於硬體規模只有兩張 GPU，本研究將結果定位為實機 proof-of-concept 與方法學驗證，不直接外推至大型生產叢集。

表 3. 實驗環境列表

| 項目 | 設定 |
|---|---|
| GPU | NVIDIA RTX 4070、NVIDIA RTX 3080 |
| 排程器 | Slurm with GRES/TRES and MPS [11] |
| 部署平台 | Kubernetes/k3s，僅負責容器部署與服務生命週期 [10] |
| GPU sharing | NVIDIA MPS，MPS fraction 為 25%、50%、75%、100% [5] |
| 系統 | Ubuntu、CUDA、NVIDIA driver、PyTorch |
| 監控 | GPU utilization、SM utilization、memory usage、job event、queue delay |

訓練資料集使用 Alibaba GPU Trace 與 Microsoft Philly Trace：前者用於參考生產 MLaaS 工作的到達率、工作長度與資源需求分布 [4]，後者用於參考多租戶 GPU training workload 的佇列與 JCT 特性 [1]。同時在本研究叢集上執行 cuBLAS、BERT inference、ResNet training、Qwen fine-tuning 與矩陣運算等真實 AI 工作。

直接在實際環境從頭訓練需要數十萬到數百萬個 transition，而真實叢集中一個決策對應一個跑數分鐘至數小時的任務，收集足夠樣本需時數月。因此本研究採 sim-to-real 兩段式：(1) 在模擬環境大量訓練，產出基本模型；(2) 上線部署，記錄真實叢集 (observation, action, reward) 資料；(3) 以 RLPD 用真實資料把基本模型微調成真實環境策略。DRL 訓練參數如表 4 所示；所有學習臂共用網路與最佳化設定，僅 critic 家族與風險目標不同。

表 4. DRL 訓練超參數

| 項目 | 數值 |
|---|---|
| 優化器／學習率（actor, critic, α） | Adam／3×10⁻⁴ |
| 折扣 γ／目標軟更新 τ | 0.99／0.005 |
| 隱藏層（MLP trunk） | (256, 256) + LayerNorm |
| batch size／UTD ratio | 256／4 |
| n-step return | 10 |
| replay | Prioritized (SumTree) + score warm start |
| IQN 分位數 N_QUANT／cosine 維度 | 32／64（RDSAC 臂） |
| 風險目標／tail mass β | CVaR／0.25（RDSAC-cvar 臂） |
| 溫度 α | 自動調整（init 0.1） |
| reward | 多目標（w_jct=1.0, w_util=0.05），reward_scale=1000 |
| 訓練長度 | 每臂約 1×10⁵ curriculum env-steps |

圖 2 為三個學習臂在 aimix-family 課程訓練下的 episode reward 與 critic loss（跨 seed、rolling window w=80 平滑）。三臂 reward 皆從約 0 隨課程階段上升並趨於穩定；RDSAC-mean/cvar 收斂較高且平穩，vanilla SAC 在自動-α 下方差較大。critic loss 皆穩定收斂，未見發散。

![圖 2. 學習臂訓練收斂曲線](../assets/figures/training_convergence.png)

**圖 2. 學習臂訓練收斂**（SAC / RDSAC-mean / RDSAC-cvar，aimix-family 課程，跨 seed 平均、rolling window w=80）。

### 5.2 實機評估結果

**模擬結果。** 在 trace-derived AI-serving 工作負載中（8 個 held-out workload seed），啟發式與學習式策略明顯區分不同排程方法：size-aware 的 multifactor／score 在 SLO 違反率上（≈41%）明顯優於 FCFS（66.5%），此區隔在計入 seed 變異後仍成立，說明模擬器本身具備區分策略的能力（表 5）。

表 5. 模擬環境下 AI-serving 工作負載的排程器比較（8 個 held-out workload seed，mean ± std；score 使用 sim 預設 ε=0.30）

| 排程器 | 平均 JCT (s) | 推論 JCT (s) | SLO 違反 (%) | 使用率 |
|---|--:|--:|--:|--:|
| FCFS | 2199 ± 1201 | 1847 ± 1151 | 66.5 ± 23.2 | 0.58 |
| multifactor | 1108 ± 398 | 461 ± 264 | 41.1 ± 20.1 | 0.63 |
| score | 1129 ± 398 | 520 ± 331 | 40.7 ± 19.0 | 0.63 |

**實機混合 AI 工作負載。** 在 BERT inference、ResNet training、Qwen fine-tuning 與 cuBLAS 矩陣運算四路混合工作負載（數量占比分別為 30%、30%、30%、10%，每 seed 125 個工作、8 seeds）上評估，結果如表 6 所示。以真實資料 sim-to-real 微調的 RLPD 取得最佳平均 JCT（110.7 s），相較 FCFS（137.0 s）與 Backfill（136.2 s）分別快約 19.2% 與 18.7%，也優於啟發式（125.2 s；此為點估計，統計顯著性見 §5.4）；SAC 與 RDSAC-cvar 也都優於 FCFS／Backfill，其中 RDSAC-cvar 在尾端（P99）上的表現相對較佳。學習式策略在平均 JCT 上優於 Slurm 傳統排程，這反映了在排程框架中整合 GPU 型號差異、MPS 配額、工作特徵與佇列狀態所帶來的改善。

表 6. 實機混合 AI 工作負載評估（每 seed 125 個工作，8 seeds，mean ± std；JCT 與 P99 單位為秒，GPU 利用率為平均並發 MPS 槽數）

| 排程器 | 平均 JCT (s) | P95 (s) | P99 (s) | GPU 利用率 |
|---|--:|--:|--:|--:|
| FCFS | 137.0 ± 18.3 | 241.2 ± 41.7 | 255.0 ± 43.7 | 8.60 ± 1.24 |
| Backfill | 136.2 ± 23.4 | 255.6 ± 52.5 | 269.8 ± 50.1 | 8.68 ± 1.39 |
| 啟發式 | 125.2 ± 19.3 | 433.5 ± 146.4 | 517.6 ± 122.2 | 8.25 ± 1.30 |
| SAC | 126.2 ± 19.8 | 471.2 ± 86.2 | 566.5 ± 49.9 | 8.00 ± 1.23 |
| RDSAC-mean | 142.3 ± 38.8 | 469.7 ± 152.7 | 600.3 ± 77.1 | 8.59 ± 1.81 |
| RDSAC-cvar | 130.5 ± 23.3 | 395.0 ± 132.9 | 527.7 ± 110.6 | 8.32 ± 1.43 |
| **RLPD** | **110.7 ± 18.9** | 418.0 ± 103.5 | 509.0 ± 80.2 | 7.45 ± 1.34 |

由表 6 亦可觀察到一個尾端取捨：FCFS／Backfill 以嚴格序列執行換得較低的 P95/P99（241–270 s），但平均 JCT 較差；啟發式與學習式策略以較積極的 MPS 共置壓低平均 JCT，代價是較重的尾端（P99 ≈ 509–600 s）。RDSAC-cvar 的風險敏感目標在尾端平衡上表現較佳（P99 527.7 s，優於 SAC 與 RDSAC-mean）。

以下 §5.3–5.7 以嚴謹的統計方法（seed-level 配對、Holm-Bonferroni 多重比較校正、TOST 等價檢定）進一步刻畫上述效益的統計邊界與場景依賴性，並輔以天花板分析與 placement 消融解析可贏空間的來源。

### 5.3 統計方法

實機評估採用三項方法降低誤判：(1) **Common random numbers**——同一 seed 下所有排程器共用相同工作序列，做配對比較；(2) **Drift-robust interleaving**——不同方法交錯執行，避免 GPU 暖機或系統漂移與特定方法混淆；(3) **Seed-level paired statistics**——以 seed（而非個別 job）為分析單位，避免偽重複。多個排程器同時比較時採 **Holm-Bonferroni** 校正，並以 **TOST (two one-sided tests)** 等價檢定判斷小差異是否可視為實務等價。

### 5.4 與啟發式基準的嚴謹比較

表 6 顯示學習式策略在平均 JCT 上優於 FCFS／Backfill；然而若以**啟發式 (score)** 為配對基準做嚴格檢定，多數學習臂與 score 的差異在本 seed 數（n=8）下並未達統計顯著（Holm-Bonferroni 校正後 adjusted *p* > 0.05）。其中 RLPD 相對 score 的平均 ΔJCT 為 +2.2%，但此均值受單一 seed 拉高，去除後其餘中位數約與 score 打平；SAC 與 score 在 ±10% 內 TOST 等價。在低負載 cuBLAS 共置場景中，FCFS、Backfill、RDSAC-mean 與 score 經 TOST 檢定在 ±5% 內統計等價，代表該 regime 下策略空間已接近平坦。整體而言，學習式策略相對**傳統 Slurm 排程**（FCFS/Backfill）的平均 JCT 優勢方向一致，但相對**已 size-aware 的啟發式**則未構成統計上穩健的超越；這反映了策略效益確實取決於基準與測試場景。

### 5.5 天花板分析

為區分負向結果是源於特定 RL 方法的局限、還是測試 regime 本身空間有限，本研究在模擬器中進行了獨立於任何學習式方法的天花板分析：固定 GPU/MPS placement，讓排程器唯一能控制的槓桿只剩 dispatch **ordering**，並以 random-restart + swap local search 搜尋每個 instance 在所有 ordering 中可達的最佳平均 JCT，定義 `headroom% = (score_JCT − best_ordering_JCT) / best_ordering_JCT`。結果（表 7）顯示 headroom 隨負載單調上升：在本研究 cuBLAS 與模擬比較的測試負載（n_jobs≈50，對應 load 40–60）下 headroom 僅 0.1–0.7%，即 score 已幾乎位於可達排程上界；即使負載提高到 n_jobs=100，headroom 也僅約 4.1%。這說明低負載下的策略空間確實接近平坦，是負向結果的結構性原因，而不是特定方法的偶然失敗。

表 7. Headroom vs. 負載（2-GPU 叢集，3 families × 10 seeds/row，n=30）

| 負載 (n_jobs) | Headroom（mean ± 95% CI） | Max |
|---|--:|--:|
| 40 | +0.1% ± 0.1% | +1.5% |
| 60 | +0.7% ± 0.5% | +5.8% |
| 80 | +2.0% ± 1.1% | +9.7% |
| 100 | +4.1% ± 2.3% | +25.2% |

### 5.6 Placement 解耦消融

排程的另一個可能槓桿是 GPU/MPS placement 本身，而非 ordering。本研究以解耦消融訓練三個網路形狀相同的臂：**joint**（同時學 job 選擇與 placement）、**placement_only**（job 選擇凍結為 score 首選，只學 placement）、**job_only**（placement 凍結為 first-fit，只學 job 選擇）。結果（表 8）顯示兩個解耦臂都未與 joint 產生統計顯著差異；placement_only 與 joint 實質等價，代表在此規模下 DRL 學到的 job 選擇並未帶來穩健增益。此結果與 §5.5 的 ordering 天花板一致：placement 槓桿在 2×1 規模下同樣接近無效。

表 8. Joint-vs-Decoupled placement 消融（2×1，3 臂 × 5 training seeds，pooled JCT）

| 臂 | 平均 JCT (s) | Δ vs joint | 配對 *p* |
|---|--:|--:|--:|
| joint | 9150 ± 2335 | — | — |
| placement_only | 8443 ± 2631 | −5.3% ± 33.8% | 0.585 |
| job_only | 10540 ± 1358 | +19.6% ± 35.7% | 0.295 |

### 5.7 效益邊界小結

綜合 §5.4–5.6：學習式策略在平均 JCT 上優於傳統 Slurm 排程（表 6），但在本研究 2×1 小規模叢集上，相對已 size-aware 的啟發式尚未構成統計上穩健的超越；天花板分析與 placement 消融進一步指出，在測試負載下 score 之上的總可贏空間本就極小（低負載為結構性、重負載則 headroom 存在但未被 DRL 完全捕捉）。本節的目的不是否定 DRL 排程的潛力，而是誠實界定其效益成立的條件。策略效益主要取決於工作負載、硬體規模與底層資源分配後端，因此此類研究必須以真實部署與統計檢定作為必要評估條件。相關可重現性材料見 `runs/headroom_*/` 與 `runs/ablation_std_*/`。

## 6. 結論與未來展望

### 6.1 結論

本研究驗證了 DRL 能於異質 GPU 與 NVIDIA MPS 環境中學習 GPU placement 與 MPS-aware 排程 [5]，並透過 Slurm job submission path 整合到真實排程流程 [11]。相較傳統只選 GPU 或只做固定規則的方法，本研究在排程框架中整合了 GPU 型號差異、MPS 配額、工作特徵、佇列狀態與回饋訊號，並以失效安全設計確保排程核心穩定。

實驗結果顯示，學習式策略在平均工作完成時間上優於 FCFS 與 Backfill 等 Slurm 傳統排程。其中以真實資料離線微調的 RLPD 取得最佳平均 JCT，RDSAC-cvar 在尾端延遲上表現較佳。本研究的實機評估限於小規模實驗室環境，在較大型叢集的效能仍有待驗證。在異質 GPU + MPS 排程中，策略效益主要取決於工作負載、硬體規模與底層資源分配後端的具體配置（詳見 §5.3–5.7 的統計刻畫與效益邊界分析）。

### 6.2 未來展望

未來工作可沿以下方向展開：

1. **更大、更高競爭的叢集**：擴展至更多節點與 GPU，檢驗學習式策略的效益是否隨叢集規模與競爭程度進一步增強。
2. **MIG + MPS fraction 混合 partition**：同時納入硬體級隔離與軟體級共享，建立更完整的 GPU sharing action space [5][21]。
3. **Offline RL / 真線上 RLPD**：收集更大量真實 Slurm transition，以 offline RL 或真線上 RLPD 改善 sim-to-real 轉移 [8]。
4. **Energy-aware scheduling**：將功耗、能效與碳排納入 reward，使排程器兼顧效能與能源效率的最佳化。
5. **LLM serving workload**：加入更真實的 LLM serving trace，評估 token latency、throughput、SLO violation 與 batch scheduling 的交互影響 [29]。

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
