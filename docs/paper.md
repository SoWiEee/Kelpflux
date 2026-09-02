# 基於 Slurm 與 Kubernetes 架構下 AI 伺服器 GPU 工作負載智慧排程技術之研究

### Intelligent GPU Workload Scheduling Techniques for AI Servers under a Slurm-on-Kubernetes Architecture

**作者一¹、作者二²**
¹○○大學 ○○系　²○○大學 ○○系
{author1, author2}@stumail.nutn.edu.tw

---

## 摘要

異質 GPU 與 NVIDIA MPS 共享使排程器必須同時考量硬體差異、工作配額與佇列狀態；傳統 Slurm 固定規則難以動態兼顧工作完成時間（JCT）與尾端延遲 [1][2][3][4]。現有深度強化學習（DRL）排程研究多停留於模擬，較少在真實 Slurm 流程中處理 MPS-aware 工作派遣與 GPU placement。

本研究提出以 Slurm-on-Kubernetes 為排程核心的異質 GPU 智慧排程框架；Kubernetes 僅負責部署 [5]。框架以聯合動作介面建模 MPS-aware 工作派遣與 placement。本研究於 RTX 4070 與 RTX 3080 環境，以 trace-derived 混合 AI 工作負載比較 FCFS、Backfill、SAC、RDSAC 與 RLPD [6][7][8][9]。在由 DRL 控制順序、並以可落地的 **Option B（非阻塞週期性重排、失效安全）**致動下，評估**真實 CUDA、poisson 到達**的結果：深負載（oversub=6）下學習式策略的平均 JCT 顯著勝過 Backfill 約 10~12%、尾端 P99 更大幅同勝約 19~22%；且於淺／中／深三個負載點（oversub=2／4／6）平均 JCT 皆穩健勝過 Backfill（≈−11%～−13%，皆顯著）。此結果實機確認了模擬天花板分析對「ordering headroom 隨負載上升」的預測，並顯示效益取決於 RL 是否掌握*順序*槓桿、致動路徑是否原生連續，以及負載形成可重排的 job queue。

**關鍵詞**：GPU 資源排程、異質 GPU、NVIDIA MPS、Slurm、Kubernetes、深度強化學習

## Abstract

The rapid growth of large language models and generative AI has made GPUs the primary compute resource for AI workloads [1][2]. However, many laboratory and small-scale clusters consist of heterogeneous GPU generations, and NVIDIA Multi-Process Service (MPS) allows multiple jobs to share a single GPU [3], making it difficult for traditional Slurm scheduling [4] to jointly optimize GPU utilization, job completion time (JCT), and tail latency. Existing heuristic policies such as FCFS and Backfill rely on fixed rules and cannot adapt to workload characteristics, while existing DRL schedulers rarely perform job selection and heterogeneous GPU placement under explicit MPS-quota constraints inside a real Slurm environment.

This paper proposes an intelligent scheduling framework for heterogeneous GPUs with NVIDIA MPS, using **Slurm as the scheduling core**. Kubernetes (k3s) only provides deployment and lifecycle management [5]. The policy interface models joint queue selection and placement. A placement-only real-machine path calls `/act` before submission and binds the selected node through `sbatch -w`, leaving job *ordering* to Slurm; on an RTX 4070/3080 testbed with trace-derived mixed AI workloads [1][2], learned policies under this path only match or marginally beat FCFS/Backfill in mean JCT and at a large tail cost, without a robust advantage. A held-job controller's post-submission `required_nodes` REST actuation was disabled by the tested Slurm REST API (v0.0.37); we therefore validated a **native ordering actuation path** instead, deployed as **Option B (non-blocking periodic re-prioritization, fail-safe)**: a daemon re-ranks the live pending queue by the served policy every few seconds and writes those ranks as administrator `Priority` on unheld jobs, letting Slurm's own in-process backfill actuate at native speed and place freely. Under **real-CUDA aimix and poisson arrival** (10 seeds, 150 jobs), at deep load (oversub=6) every learned policy significantly beats production Backfill by **~10–12% in mean JCT and ~19–22% in tail P99** (paired Wilcoxon *p*≤0.006; P99 below Backfill in 10/10 seeds) — at this load Backfill's own aggressive mean-optimizing reorder starves some jobs into the worst tail of all arms, exactly what RL ordering avoids. Across a three-point load sweep (oversub=2/4/6) the learned policies beat Backfill in mean JCT at **all three loads** (≈−11% to −13%, all significant); the *mechanism* is load-dependent (a mild-SJF ordering headroom at deep/medium load versus tail-starvation avoidance from continuous re-ranking at shallow load — the latter attribution pending a B-path dispatch-order re-run). This is a real-machine confirmation of the simulator ceiling analysis's prediction that ordering headroom grows with load, and shows the benefit hinges on the RL agent controlling *ordering* and a native, continuous actuation path. Multiple-comparison correction, TOST, a ceiling analysis, and a placement ablation characterize the efficacy boundary and remaining uncertainty [6][7][8][9].

**Keywords**: GPU Resource Scheduling, Heterogeneous GPU, NVIDIA MPS, Slurm, Kubernetes, Deep Reinforcement Learning

---

## 1. 緒論

Slurm Workload Manager 是高效能運算叢集最廣泛使用的工作排程系統之一，提供完整的工作提交、佇列管理、資源限額與 GPU 資源配置功能 [4]，其以「節點」為基礎的資源模型非常適合研究工作環境。然而 Slurm 原生設計以固定實體節點為主，對彈性擴縮與容器化整合的支援相對有限。Kubernetes 則是目前最主流的容器編排平台，能自動管理容器的部署、擴縮與健康監控 [5]，但其原生排程器 (kube-scheduler) 以服務導向設計為主，對批次工作、GPU 共享與研究工作特有的排程需求支援不足。因此有多項研究嘗試將 Slurm 與 Kubernetes 整合，以同時取得批次排程語意與雲端彈性管理能力；實作層面也有 Slinky [10] 等 Slurm-on-Kubernetes 方向的工具，顯示此類架構已具有實務需求與發展基礎。

### 1.1 AI 工作負載快速增加

近年大型語言模型、生成式 AI 與深度學習服務快速發展，使 GPU 成為 AI 訓練與推論最重要的運算資源 [1][2]。相較於傳統批次運算，AI 叢集經常同時承載短時間推論、模型微調、長時間訓練與矩陣運算等不同工作，這些工作在執行時間、記憶體需求、延遲敏感度與 GPU 使用型態上皆有明顯差異 [1][2]。

在大學實驗室與中小型研究叢集中，GPU 資源通常有限，且硬體常由不同世代 GPU 漸進式擴充而成。例如 RTX 4070 與 RTX 3080 在算力、記憶體容量與功耗上皆不同。若排程器只把 GPU 視為同質資源，便可能讓小型推論工作佔用高效能 GPU，或讓長時間訓練阻塞後續短工作，造成 GPU utilization、JCT 與 queue delay 之間的取捨更加困難。

NVIDIA MPS 提供另一個重要槓桿：多個 CUDA 工作可共享同一張 GPU，使 GPU 不再只能以整張卡為單位分配 [3]。然而 MPS 也讓排程問題從「選哪張 GPU」變成「選哪張 GPU 與該工作需要多少 **MPS fraction**」。因此，在異質 GPU 與 MPS 共存的環境中，GPU scheduling 已成為影響叢集效能的核心問題。

### 1.2 現有方法限制

Slurm 提供 FCFS、Backfill、multifactor priority 與 GRES/TRES GPU 資源管理 [4]，但其傳統策略多以固定規則為主，難以感知異質 GPU、MPS 分片、工作類型與長期回報之間的相互影響。現有方法主要存在四項限制：

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

1. **MPS 配額約束下的工作派遣與異質 GPU placement**：不同於 UXP-RL [11] 主要決定 CPU-vs-GPU 資源類型、KIS-S [12] 調整 Kubernetes 推論副本數、DRR [13] 處理 GPU 碎片化，本研究的策略介面與模擬環境將「派哪個工作」與「放到哪張異質 GPU」建模為聯合離散動作。每個工作攜帶既定的 MPS fraction 需求（25%/50%/75%/100%）作為 state 特徵與 action mask 的可行性約束，使決策在尊重 MPS 配額的前提下進行。此設計首次在真實 Slurm 佇列上刻畫「job 選擇與異質 GPU placement 聯合決策」相對於僅決定資源類型 [11] 或僅調整副本數 [12] 的可行性與行為差異。

2. **失效安全的 Slurm 策略層整合**：不同於 UXP-RL、KIS-S 等純模擬研究，本研究把學習式決策服務嵌入真實 Slurm job submission path，並提供 fail-safe 回退機制，當 RL 服務逾時、回傳無效 action 或健康檢查失敗時自動回退至啟發式策略，使排程核心 (slurmctld) 永不被阻塞。此設計讓 DRL scheduler 得以在真實排程路徑中部署，而不需修改 Slurm 核心。本研究因此證明學習式排程可在不更動 slurmctld 的前提下安全嵌入生產排程路徑，這是先前純模擬研究 [11][12] 未曾示範的部署可行性。

3. **實機比較與效益邊界分析**：本研究於 RTX 4070/3080 異質環境，以 trace-derived 混合 AI 工作負載比較 FCFS、Backfill、啟發式、SAC、RDSAC 與 RLPD，並採多 seed、Holm-Bonferroni 校正與 TOST 等價檢定。學習式策略在平均 JCT 上優於傳統 Slurm 排程，但兩者皆不足以支持穩健超越 size-aware 啟發式。並且由於實驗環境限於小叢集，經消融實驗分析後因變異過大而無法辨識哪個決策具有穩定優勢。

## 2. 相關研究

### 2.1 GPU 叢集排程、分析與資源共享

近期研究聚焦於單節點內或單一叢集模型上的動態資源調度本身。Wang 等人的 DCUDA [14] 針對單節點多 GPU 情境，設計了一套輕量級核心／記憶體使用率監控機制，搭配近乎零開銷的「執行中」CUDA 應用即時遷移，將 GPU 過載時間平均降低 78.3%、一般工作執行時間降低 42.1%（記憶體密集型工作最高 67%）。Sedighi 等人 [15] 則在 Alibaba 的 cluster-trace-gpu 生產工作負載軌跡之上，提出結合硬體與軟體分割的公平且需求感知動態資源配置演算法，於模擬環境中將 GPU 資源使用量降低達 88%。這兩項工作皆聚焦「資源配置本身如何隨工作負載動態調整」（即時遷移／再分割），評估分別侷限於單一多 GPU 節點與純模擬 trace 重放，並未涉及與批次排程器（如 Slurm）的整合。

在 GPU 共享機制方面，NVIDIA MPS 允許多個 CUDA process 同時共享同一張 GPU 的運算資源 [3]，NVIDIA MIG 則在硬體層面將 GPU 切分成隔離的 instance [16]。學術上亦有更細緻的 partition 與時空共享研究，如多 GPU 伺服器上的時空共享服務 [17] 與現代 GPU 的階層式資源分割 [18]。針對深度學習工作負載，也有專門的叢集排程系統：Gandiva [19] 以 introspective 排程與工作遷移提升利用率，Tiresias [20] 以近似 age-based 的優先序縮短 JCT，MLaaS 生產叢集分析 [2] 則刻畫了大規模異質 GPU 叢集的工作特性。這些方法多假設整卡分配或同質 GPU，較少把 MPS fraction 作為排程流程的顯式決策脈絡。

### 2.2 強化學習排程

Lin 等人 [11] 提出 UXP-RL：一個以 DQN 為核心、涵蓋前處理／訓練／推論三類任務、可部署為集中式或分散式排程器、並跨雲／邊／霧三層架構運作的 CPU-GPU 任務排程演算法。其於**模擬環境**中，集中式排程器將平均週轉時間相較 SJF／FCFS 與 TYPE 啟發式分別降低 57.81%、57.28% 與 27.66%；分散式排程器則因能將長訓練工作卸載至雲端而把推論任務週轉時間再降低 89.07%。同年 Zhang 等人的 KIS-S [12] 以 PPO 訓練一個 GPU-aware 的 Kubernetes 推論自動擴縮策略，完全於自建模擬器 (KISim) 中訓練後零樣本部署，於多種流量情境下平均獎勵提升 75.2%；其問題設定是調整副本數的自動擴縮，而非本研究的工作放置排程。Wu 等人的 DRR [13] 則針對 GPU 共享叢集的碎片化問題，以模仿學習從既有啟發式暖啟動一個深度強化學習去碎片化排程器，並同時於實體 Kubernetes 測試床與大規模模擬叢集上驗證，平均碎片率降低 50%，是少數同時涉及真實 Kubernetes 部署的學習式排程器。

在演算法基礎方面，Discrete SAC 將最大熵框架延伸到離散動作空間，以 categorical 策略取代高斯策略、以期望估計熵項 [6]；分布式評論家（如 IQN [7]、DSAC [8]）以回報分布與風險敏感目標處理尾端延遲；以 BatchNorm 移除 target network 以提升樣本效率的 CrossQ [21] 則代表近期簡化訓練流程的方向；RLPD [9] 進一步以對稱取樣真實資料的方式進行 offline-to-online 微調，降低從零探索的成本。然而，現有 RL scheduler 多假設單一 GPU、同質 GPU 或大型模擬叢集，或將 Kubernetes 當作排程主體（如 Kubeflow [22]、Volcano [23]、Kueue [24]、NVIDIA KAI [25]）；較少探討在真實 Slurm 流程中，如何讓 RL 同時決定異質 GPU placement 與 MPS-aware 排程，並在服務失效時維持排程安全。

### 2.3 異質與邊緣 GPU 排程

Tsenos 與 Kalogeraki [26] 針對缺乏原生虛擬化支援的邊緣 GPU（如消費級卡）提出一套硬體無關的時空共享機制：為每個行程建立 cgroup、動態調整其 duty cycle 來實現優先權式與截止期限式排程，且無需修改工作負載原始碼即可整合進 TensorFlow、PyTorch、FFmpeg 等既有框架。Majeed 等人 [27] 則以系統性文獻回顧整理 NVIDIA Jetson 系列邊緣 SoC 上的 DNN 排程器，區分規則式與最佳化式兩大類，並整理其記憶體競爭、跨加速器轉移成本與靜態／動態排程的權衡；其排程粒度是單一 DNN 模型內的層級（將個別網路層指派給不同硬體加速器）。這些邊緣場景的資源與延遲約束與資料中心叢集不同，惟顯示異質硬體上的細粒度排程是一個活躍的研究方向。此外，異質 Kubernetes 叢集上的 RL 排程 [28]、混合學習與最佳化排程 [29] 與語意感知的 LLM 叢集排程 [30] 亦顯示學習式方法在異質資源分配上的潛力。

### 2.4 定位比較

表 1 彙整本研究與主要研究類型的定位差異。本研究的核心位置在於將「異質 GPU + MPS + 真實 Slurm 流程」三者結合；Kubernetes 僅是部署平台 [5]，不是本文的主要排程貢獻。

表 1. 相關研究定位比較

| 類型 | 是否考慮異質 GPU | 是否考慮 MPS fraction | 是否整合真實 Slurm | 主要限制 |
|---|:--:|:--:|:--:|---|
| FCFS / Backfill [4] | 部分 | 部分 | 是 | 固定規則，難以學習長期效果 |
| GPU sharing / partition [3][16][17][18] | 部分 | 部分 | 否 | 多聚焦 partition 機制，較少處理排程流程 |
| RL scheduler（模擬）[11][12] | 部分 | 少 | 否 | sim-to-real 效益不明 |
| RL scheduler（真實 K8s）[13] | 部分 | 少 | 否 | 針對碎片化，非 Slurm 工作流 |
| Kubernetes GPU scheduler [22][23][24][25] | 部分 | 部分 | 否 | Kubernetes 是排程主體，不處理 Slurm 工作流 |
| 本研究 | 是 | 是 | 是 | 目前實機規模仍小，需擴大驗證 |

## 3. 研究目的與系統架構

### 3.1 研究目的

基於上述背景，本研究聚焦於以下三項問題：

1. 在每個工作的 MPS 配額與 GPU 型號差異（異質 GPU）同時存在時，聯合考慮工作選擇與 GPU 放置能否改善工作完成時間？
2. 學習式策略接入真實 Slurm 排程機制後，，能否在保有失效回退機制的條件下，與 FCFS 及 Backfill 進行公平比較？
3. 在多種工作負載情境下，當學習式策略是否能進一步改善平均 JCT與尾端 JCT？

本研究的核心貢獻在於將「異質 GPU + MPS + 真實 Slurm 流程」三者結合，主要貢獻有三：

- **MPS 配額約束下的工作派遣與異質 GPU placement**。不同於 UXP-RL [4] 主要決定 CPU-vs-GPU 資源類型、KIS-S [5] 調整 Kubernetes 推論副本數、DRR[6] 處理 GPU 碎片化，本研究的策略介面與模擬環境將「派哪個工作」與「放到哪張異質 GPU」建模為聯合離散動作。每個工作攜帶既定的 MPS 配額需求作為 state 特徵與 action mask 的可行性約束，使決策在尊重 MPS 配額的前提下進行。此設計在模擬環境刻畫「job 選擇與異質 GPU placement 聯合決策」，實機則聚焦提交時 placement 相對於僅決定資源類型 [4] 或僅調整副本數 [5] 的可行性與行為差異。
- **失效安全的Slurm 策略層整合**。不同於 UXP-RL、KIS-S 等純模擬研究，本研究把學習式決策服務嵌入 Slurm job submission path，並提供 fail-safe 回退機制，當學習式排程服務逾時就自動回退至 Slurm 原生策略，在評估量測中未觀察到逾時，且策略服務失效時具備回退機制。此設計讓客製化 DRL 排程器得以在真實排程路徑中部署，而不需修改 Slurm 核心，這是先前純模擬研究 [4, 5] 未曾示範的部署可行性。
- **實機比較與效益分析**。本研究於 RTX 4070與 RTX 3080 異質環境，以 trace-derived 混合 AI 工作負載比較 FCF、Backfill、SAC、RDSAC 與 RLPD，並使用統計檢定。結果顯示，學習式策略的工作完成時間優於 FCFS 與 Backfill，但是在尾端部分仍比 FCFS 略差。

**State.** 狀態包含四類資訊：

1. **Job features**：工作類型、預估 runtime、GPU memory 需求、MPS 需求、SLO 緊迫度與等待時間。
2. **GPU features**：GPU 型號、GPU utilization、SM utilization、memory usage、可用 MPS、目前共置工作數。
3. **Queue features**：佇列長度、前 K 個工作的需求、等待時間分布與到達率。
4. **History features**：近期完成工作 JCT、slowdown、SLO violation 與各 GPU 的負載變化。

**Action.** 動作為「從佇列前 K 個工作中選一個」與「將其放置到哪張 GPU」的聯合離散決策，另含一個 no-op（暫不派遣）：

$$
\text{Action} \;=\; \big(\text{選 job}_i \in \text{top-}K\ \text{佇列}\big) \,\times\, \big(\text{放置至 node}_j / \text{gpu}_k\big) \;\cup\; \{\text{no-op}\}
$$

在本研究的 2×1 實驗平台（K=16、2 個 placement）中，動作空間為 16×2+1 = 33 個離散動作，observation 維度為 168。每個工作攜帶自身的 MPS fraction 需求（25%/50%/75%/100%），排程器透過 state 特徵感知、並由 action mask 遮蔽 MPS 剩餘容量不足的放置，因此策略是在符合工作 MPS 需求的前提下做 job 選擇與 placement

**Reward.** Reward 的設計目標是降低使用者感受到的等待與完成時間、同時抑制尾端延遲並促進異質節點間的負載均衡。本研究之產出模型（§5.8 之 fairness-reward checkpoint）以下列**單一**每步回報訓練（不再區分單／多目標；S 為 `reward_scale`）：

$$
\begin{aligned}
r_t \;=&\; \underbrace{\sum_{j \in \mathcal{C}(t)} \left[ -\frac{\mathrm{JCT}_j}{S} \;-\; \lambda_{\mathrm{fair}}\left(\frac{\mathrm{JCT}_j}{S}\right)^{2} \right]}_{\text{工作完成時計入}} \;+\; \underbrace{\big[\, \gamma\,\phi(s_{t+1}) - \phi(s_t) \,\big]}_{\text{每步 potential shaping}}, \\[4pt]
\phi(s) \;=&\; -\frac{1}{S\,N}\sum_{i \in \mathcal{P}(s)} \mathrm{wait}_i \;-\; \lambda_{\mathrm{bal}}\cdot \mathrm{imbalance}(s), \\[4pt]
\mathrm{imbalance}(s) \;=&\; \frac{\mathrm{std}_n\big(\mathrm{freeMPS}_n\big)}{\mathrm{MPSperGPU}},
\end{aligned}
$$

其中 $\mathcal{C}(t)$ 為在第 $t$ 步完成的工作集合、$\mathcal{P}(s)$ 為狀態 $s$ 下的待排（pending）工作集合。訓練參數：**$\lambda_{\mathrm{fair}}=5.0$、$\lambda_{\mathrm{bal}}=5.0$、$\gamma=0.99$、$S=20000$（reward_scale）、$N=$ 每 episode 工作數**。各項意義如下：

- **完成項**（工作完成時計入）：$-\mathrm{JCT}_j/S$，其中 $\mathrm{JCT}=$ 完成時間 $-$ 提交時間、已含 queue delay，$S$ 使該項落於 $O(0.1)$ 尺度。此為主要吞吐訊號。
- **公平／抗飢餓項**：$-\lambda_{\mathrm{fair}}(\mathrm{JCT}_j/S)^{2}$，對單一工作 JCT 的**凸（平方）懲罰**。其效果為將目標由純平均改寫為

$$
\min \sum_j \Big(\mathrm{JCT}_j + \lambda_{\mathrm{fair}}\,\mathrm{JCT}_j^{2}\Big) \;=\; \underbrace{\textstyle\sum_j \mathrm{JCT}_j}_{\text{平均}} \;+\; \underbrace{\lambda_{\mathrm{fair}}\textstyle\sum_j \mathrm{JCT}_j^{2}}_{\text{尾端／變異項}},
$$

  是一個**改變目標**的項（非 optimum-preserving），刻意以少許平均換取有界的最差情況——這正是純平均-JCT reward 無法表達、而 §5.8 尾端結果所需者。
- **Potential-based shaping** $\gamma\,\phi(s')-\phi(s)$ [31]：提供密集訊號且**不改變最優策略**（Ng et al. 1999）。$\phi$ 含兩部分——(i) 待排工作的累計等待（排入工作即降低總等待 $\to$ 每步正向 bonus）；(ii) **節點負載均衡項** $-\lambda_{\mathrm{bal}}\cdot\mathrm{imbalance}$，懲罰把負載集中於單一節點，於 2×1 異質叢集促進 free-MPS 平衡。

需澄清一點：**interference（干擾）並非 reward 項而是環境動力學**。訓練環境設 `interference=0.3`，即工作的實際執行時間為

$$
\mathrm{runtime}_{\text{real}} \;=\; \mathrm{runtime}_{\text{nominal}} \times \big(1 + 0.3\,k\big), \qquad k = \text{同一 GPU 上的共置工作數},
$$

使 MPS 過度打包付出真實的執行變慢代價；此設定讓上述 fairness／balance 訊號在「打包 vs 干擾」的真實張力下學習，而非 reward 公式的一部分。此 reward 設計讓 DRL 不只最佳化單一工作，而是學習兼顧平均、尾端與節點均衡的長期排程結果。

### 3.2 系統架構

本研究建立一套以 Slurm 為排程核心的異質 GPU + MPS 智慧排程平台（圖 1）。學習式策略以服務形式接入 Slurm 的工作提交流程：策略讀取工作、GPU、MPS 剩餘容量與佇列狀態，輸出建議的工作與 GPU placement，並由系統於提交時據以綁定節點（工作的 MPS fraction 依其請求分配、非策略輸出）。若策略服務逾時或回傳不可行決策，系統自動回退至啟發式路徑，確保排程核心不被阻塞。工作執行期間，監控服務收集資源使用、queue delay、JCT 與 reward，寫入 replay buffer 供後續訓練或 RLPD (Reinforcement Learning with Prior Data) [9] 微調使用。

需說明的是，本文的主要實機評估聚焦於 **GPU placement** 的效果：系統於提交時取得策略建議的節點並綁定，藉此在真實叢集上比較不同 placement 決策；由策略即時從佇列挑選下一個工作（順序）並以 held-job 控制器透過 `required_nodes` REST 呼叫致動，在受測 slurmrestd（v0.0.37）中被停用而未生效，故該 REST 致動路徑不納入本文效能宣稱。§5.8 進一步驗證並採用一條替代的**原生排序致動路徑**：前展服務中的策略取得完整派遣順序，再以固定 Slurm `Priority` 交由 Slurm 自身 in-process 排程器致動，使 RL 掌握順序與節點、Slurm 掌握時機；此路徑經實測有效並用於 §5.8 之重載評估。

```mermaid
flowchart TD
    A[Job Submission] --> B[Slurm Controller]
    B --> C[job_submit.lua<br/>Score / optional priority intent]
    B --> D[Held Pending Queue]
    D --> E[Placement Controller<br/>via slurmrestd]
    E --> F{RL Scheduler /act}
    F -->|job, node, GPU| E
    E -.->|required_nodes REST<br/>disabled, NOT validated| B
    E -->|precompute order via /act drain<br/>→ fixed Priority, §5.8| K[Slurm in-process<br/>backfill actuates by Priority]
    K --> B
    B --> G[Execute Job on GPU]
    G --> H[Monitoring<br/>JCT, Util, Queue Delay]
    H --> I[Replay Buffer]
    I --> F
    F -.->|Timeout / Invalid / No-op| J[Leave Held or Use<br/>Slurm / Heuristic Path]
    J --> B
```

**圖 1. 系統架構與排程流程。** 學習式策略接入 Slurm 提交流程，於提交時提供節點綁定建議；透過 `required_nodes` REST 的即時致動在受測環境未生效（圖中虛線），改採 §5.8 驗證的原生路徑：前展策略取得派遣順序後，以固定 Priority 交由 Slurm 自身 in-process 排程器致動。逾時、不可行決策或 no-op 時回退至 Slurm／啟發式路徑。監控服務週期蒐集資源使用、queue delay 與 job events，寫入 Replay Buffer。

本平台使用 Kubernetes (k3s) 部署 Slurm controller、worker、RL scheduler、monitoring service 與相關容器 [5]。底層 GPU 基礎建設是透過 Kubernetes Dynamic Resource Allocation (DRA) 宣告與取得裝置 [32]，工作層級的 MPS 配額仍由 Slurm `gres/mps` 與對應的 MPS 執行環境落實。**Kubernetes 在本文中不負責工作排程決策**，只提供容器化部署、服務健康檢查、網路與生命週期管理。此設計保留 Slurm 在 HPC batch scheduling 中成熟的佇列語意 [4]。此方向與將 Slurm 整合進 Kubernetes 的 Slinky [10] 互補：Slinky 提供 Slurm-on-Kubernetes 部署基礎，本研究則在 Slurm 排程路徑上加入具失效回退機制的學習式策略層。

## 4. 排程技術

本章介紹本研究的 scheduler：以 Slurm 原生能力 Backfill 作為穩定 baseline，再以 DRL agent 學習序列排程策略。

### 4.1 Slurm 內建排程演算法

本研究以 Slurm 原生排程能力作為穩定基線，而非重寫排程核心。例如 Backfill 允許在不延後高優先權工作的前提下，讓資源需求較小、執行時間較短的工作提前插隊執行，緩解大工作長期佔用資源造成的閒置；此外也納入更保守的 FCFS 作為基準對照（為現成排程器基準，非理論下界），用以檢驗學習式策略相對 Slurm 開箱即用能力是否確有改善。

### 4.2 深度強化學習策略

本研究比較以下 DRL 方法：

1. **Discrete SAC**：將 Soft Actor-Critic 延伸到離散 action space，以 categorical 策略取代高斯策略、以期望估計熵項，適合 job 選擇與 GPU placement 這類有限離散動作 [6]。
2. **RDSAC-mean**：以分布式 critic 建模回報分布，但 actor 主要依平均回報決策 [7][8]。
3. **RDSAC-cvar**：在 RDSAC 上加入 CVaR 風險敏感目標，使策略更重視尾端 JCT 與 SLO violation [8]。
4. **RLPD**：以真實資料對模擬訓練出的模型進行微調，縮小 sim-to-real gap [9]。本研究忠實採用原論文核心配方——每個 batch 對稱取樣 50% sim 先驗 + 50% 真實資料、critic 加 LayerNorm 的 critic ensemble、高 UTD——但訓練機制為離線（更新迴圈只做梯度更新、不在真環境即時互動），故為「sim + 真實混合 buffer 的離線微調」，非原論文的真線上更新。具體作法為：**暖啟動自對應 campaign 的 RDSAC-cvar base**（§5.2 重載複核用早期 RDSAC-cvar base、§5.8 用該節之 fairness-reward RDSAC-cvar base），offline 先驗為同一異質 regime 的 sim rollouts、online 半批為 §5.1 所述之真實 Slurm 線上日誌（168 維、2 786 筆 transition、以 sacct 真實 JCT 計 reward）。關鍵在於 **RLPD 微調階段的 reward 刻意採 `jct_aligned`（$-\mathrm{JCT}/1000$）而非 base 的 mo＋公平 reward**：RLPD 的 critic 從頭學起，需要 offline↔online 兩半批的 reward *定義一致*，而線上日誌記錄的即是 $-\mathrm{JCT}/1000$，故 offline 半批亦以同尺度的 `jct_aligned` 計，而非與 base 的訓練目標一致（此為與其他三臂在 reward 上的**刻意差異**）。微調配置：offline-steps=50 000、gradient updates=200、UTD=20、固定 $\alpha=0.05$。

RDSAC 採用雙頭 IQN critic 建模 reward return 與 entropy return [7]，並使用 masked categorical actor 避免選到不可執行 action（即所選放置的 GPU 剩餘 MPS 不足以容納該工作請求的動作）。訓練流程包含 prioritized replay、n-step return、potential-based reward shaping [31] 與 heuristic warm start。需澄清命名：本研究的 RDSAC 為自組的「distributional + discrete SAC」，以離散動作空間搭配 IQN 分位數 critic 建構 [7][8]，與 Duan 等人 [33] 針對連續控制、將回報建模為單一高斯分布的 Distributional Soft Actor-Critic 不同，兩者不應混淆。

## 5. 實驗與評估

### 5.1 實驗環境與訓練

本研究使用一個小規模異質 GPU 實機叢集進行部署與評估，相關環境如表 2 所示。此環境刻意保留 GPU 世代差異，讓排程器必須面對異質 GPU placement 的問題；由於硬體規模只有兩張 GPU，本研究將結果定位為實機 proof-of-concept 與方法學驗證，不直接外推至大型生產叢集。

表 2. 實驗環境列表

| 項目 | 設定 |
|---|---|
| 節點 1（控制平面） | Intel Core i7-10700（16 執行緒）、64 GB RAM、NVIDIA RTX 4070 |
| 節點 2（工作節點） | Intel Core i7-9700、8 GB RAM、NVIDIA RTX 3080 |
| 作業系統 | Ubuntu 24.04.4 LTS（kernel 7.0.0 / 6.8.0） |
| NVIDIA driver／CUDA | 580.167.08／CUDA 13.0 |
| 容器平台 | k3s v1.34.6、containerd 2.2.2；僅負責容器部署與服務生命週期 [5] |
| GPU 資源宣告 | Kubernetes Dynamic Resource Allocation (DRA) driver v0.4.1 [32] |
| 排程器 | Slurm 23.11.7 with GRES/TRES and MPS（slurmrestd REST API v0.0.37）[4] |
| GPU sharing | NVIDIA MPS，MPS fraction 為 25%、50%、75%、100% [3] |
| 深度學習框架 | PyTorch |
| 網路 | 同一區域網路，節點間 RTT ≈ 0.16 ms（ping 量測，可忽略） |
| 監控 | SM 利用率、memory usage、job event、queue delay |

訓練資料集使用 Alibaba GPU Trace 與 Microsoft Philly Trace：前者用於參考生產 MLaaS 工作的到達率、工作長度與資源需求分布 [2]，後者用於參考多租戶 GPU training workload 的佇列與 JCT 特性 [1]。同時在本研究叢集上執行 cuBLAS、BERT inference、ResNet training、Qwen fine-tuning 與矩陣運算等真實 AI 工作。

為確保比較公平，各方法在同一評估中取得一致的工作資訊：FCFS／Backfill 使用提交時的 Slurm time-limit；模擬中 score 的 SJF 項與學習式策略的 runtime 特徵皆來自同一組執行時間估計（模擬為 oracle），無單一方法獨享的未來資訊；實機部署因 runtime predictor 未上線，score 停用 SJF 項（ε=0），僅使用 MPS-fit 與 VRAM-fit。

直接在實際環境從頭訓練需要數十萬到數百萬個 transition，而真實叢集中一個決策對應一個跑數分鐘至數小時的任務，收集足夠樣本需時數月。因此本研究採 sim-to-real 兩段式：(1) 在模擬環境大量訓練，產出基本模型；(2) 上線部署，記錄真實叢集 (observation, action, reward) 資料；(3) 以 RLPD 用真實資料把基本模型微調成真實環境策略。DRL 訓練參數如表 3 所示；所有學習臂共用網路與最佳化設定，僅 critic 家族與風險目標不同。

表 3. DRL 訓練超參數

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
| 溫度 α | 固定 0.05 |
| reward（§3.1） | mo：−JCT/S 完成項＋凸公平項 λ_fair=5.0＋potential shaping（等待＋節點均衡 λ_bal=5.0）；S=reward_scale=20000；環境 interference=0.3 |
| 訓練長度 | 每臂約 1×10⁵ curriculum env-steps |

圖 2 為三個學習臂在 aimix-family 課程訓練下的 episode reward 與 critic loss（跨 seed、rolling window w=80 平滑）。三臂皆使用固定溫度 α=0.05；曲線在課程切換後維持有限值，未出現數值發散。由於不同 critic 的 loss 尺度不可直接比較，圖 2 僅用於檢查訓練穩定性，不據此判定策略優劣。

![圖 2. 學習臂訓練收斂曲線](../assets/figures/training_convergence.png)

**圖 2. 學習臂訓練收斂**（SAC / RDSAC-mean / RDSAC-cvar，aimix-family 課程，跨 seed 平均、rolling window w=80）。

### 5.2 實機評估結果

在 BERT 推論、ResNet 訓練、Qwen 微調與 cuBLAS 矩陣運算四路混合工作負載（數量占比分別為 30%、30%、30%、10%），每個 seed 提交 150 個工作，使用 10 seeds 進行評估，到達採 poisson，：inter-arrival 為指數分佈、平均間隔 = mean(runtime)/oversub，故到達比服務快 `oversub` 倍、job queue 隨時間持續堆積，runtime 經壓縮使 p95≈`target_max`=20 s。排序只有在存在可重排的 backlog 時才有意義，故本節取三個負載點 poisson **oversub=2、4、6**（淺／中／深佇列）並置成三點負載掃描，以檢驗 RL 在不同 job queue 的效益，結果分別如 6, 7, 8 所示。

> JCT／Makespan／P95／P99 為各 seed 內平均之未加權平均 ± 標準差，以 Slurm 內建 Backfill 作為 seed-level 配對基準；三點負載掃描之 ΔmeanJCT% 彙總見表 7。

表 4. 重載混合工作負載評估結果（oversub=6，**Option B 非阻塞週期性重排致動**）

| 排程策略 | 平均 JCT (s) | P50 (s) | P95 (s) | P99 (s) | ΔmeanJCT% [95% CI] | Wilcoxon *p* | P99<bf |
|---|--:|--:|--:|--:|--:|--:|--:|
| FCFS | 352.1 ± 21.8 | 355.3 ± 30.4 | 650.8 ± 26.8 | 675.7 ± 27.9 | +14.9 [+10.4, +19.4] | 0.002 | 10/10 |
| Backfill | 306.7 ± 12.7 | 289.6 ± 22.4 | 686.3 ± 52.5 | 750.1 ± 21.4 | — | — | — |
| SAC | 270.0 ± 17.7 | 269.0 ± 22.1 | 561.4 ± 26.1 | 583.2 ± 25.2 | −12.0 [−15.1, −8.8] | 0.002 | 10/10 |
| RDSAC-mean | 269.9 ± 16.6 | 269.1 ± 23.7 | 565.0 ± 29.5 | 590.5 ± 30.7 | −11.9 [−14.8, −9.1] | 0.002 | 10/10 |
| RDSAC-cvar | 277.1 ± 27.3 | 269.4 ± 31.4 | 582.7 ± 55.1 | 607.5 ± 57.7 | −9.7 [−14.7, −4.7] | 0.006 | 10/10 |
| RLPD | 269.6 ± 17.9 | 270.9 ± 26.5 | 562.5 ± 28.1 | 585.9 ± 28.8 | −12.1 [−15.1, −9.1] | 0.002 | 10/10 |

> [!NOTE]
> 在深負載下以可落地的 Option B 致動評估，所有學習式策略的平均 JCT 皆顯著勝過 Backfill 約 9.7~12.1%（皆 *p*≤0.006），且**尾端 P99 亦全數同勝**（P99<bf 皆 10/10；學習式 P99 ≈ 583~608 s vs Backfill 750 s）——與靜態致動探針相比，B 的平均紅利相當而尾端明顯更佳。FCFS 則顯著慢於 Backfill（+14.9%）。

**致動路徑與可落地部署（Option B）。** RL 掌握*順序*後，有三條把該順序落到 Slurm 的致動路徑：(i) **靜態致動**——一次性前展全佇列取得派遣順序 → 固定 Slurm Priority，是排序品質的探針，但需完整工作集、非開放式生產可直接部署；(ii) **held-job 線上致動（Option C）**——事件驅動、忠實於聯合（選×放）動作，但工作須先 held（失效即阻塞、營運脆弱）且殘餘致動延遲使其小輸靜態（初步 n=3：平均約 −8%）；(iii) **Option B——非阻塞週期性重排**：工作以 unheld 正常提交，一個常駐程序每數秒讀取**當前** pending 佇列、以策略重排並寫入 Slurm Priority（`direct_set_prio`），交由 Slurm 原生 backfill 於原生速度致動並自由放置；B 每輪對**真實完成／到達**重新排序（線上反應），且**失效安全**（程序中止時工作仍以 Slurm 原生優先序執行、不被阻塞）。三者中**唯一可落地者（非阻塞的 Option B）同時是效能最佳者**，故**本節之三點負載掃描（表 4–7）一律以 Option B 致動量測**，作為本框架建議部署形態的實測。相對於靜態致動探針，B 的平均紅利相當、尾端則明顯更佳（見各表 P99），因每輪對真實佇列重排能更主動地規避 Backfill 的重排飢餓。

> [!NOTE]
> B 相對靜態致動探針的尾端優勢目前部分為**跨 campaign 比較**（靜態探針與 B 部署為不同 campaign，Backfill 基準接近但非 seed-paired）；嚴謹的 seed-level 同-campaign B-vs-靜態 head-to-head 待後續補測。B 之量測為 oversub=2／4／6 三點、單一 workload 家族（aimix）、2×1 小叢集。

表 5. 中載混合工作負載評估結果（oversub=4，**Option B 非阻塞週期性重排致動**）

| 排程策略 | 平均 JCT (s) | P50 (s) | P95 (s) | P99 (s) | ΔmeanJCT% [95% CI] | Wilcoxon *p* | P99<bf |
|---|--:|--:|--:|--:|--:|--:|--:|
| FCFS | 284.3 ± 20.2 | 287.6 ± 28.6 | 523.3 ± 26.0 | 541.3 ± 29.1 | +7.8 [+4.5, +11.0] | 0.002 | 9/10 |
| Backfill | 264.0 ± 18.5 | 241.7 ± 27.1 | 562.6 ± 104.1 | 720.4 ± 81.4 | — | — | — |
| SAC | 235.3 ± 20.5 | 234.8 ± 31.2 | 487.5 ± 33.7 | 507.6 ± 35.7 | −10.9 [−13.4, −8.4] | 0.002 | 10/10 |
| RDSAC-mean | 234.2 ± 18.8 | 234.5 ± 20.7 | 492.2 ± 27.0 | 509.6 ± 29.0 | −11.3 [−13.6, −9.0] | 0.002 | 10/10 |
| RDSAC-cvar | 235.5 ± 19.1 | 233.0 ± 34.7 | 493.3 ± 30.7 | 510.7 ± 34.4 | −10.8 [−13.3, −8.3] | 0.002 | 10/10 |
| RLPD | 236.9 ± 18.1 | 244.5 ± 17.5 | 494.7 ± 29.3 | 513.9 ± 31.0 | −10.3 [−12.3, −8.2] | 0.002 | 10/10 |

> [!NOTE]
> 在中負載下以 Option B 致動評估，所有學習式策略的平均 JCT 皆顯著勝過 Backfill 約 10.3~11.3%（皆 *p*=0.002），且**尾端 P99 亦全數同勝**（學習式 P99 ≈ 508~514 s vs Backfill 720 s，P99<bf 皆 10/10）；FCFS 則顯著慢於 Backfill（+7.8%，P99<bf 9/10）。

表 6. 淺載混合工作負載評估結果（oversub=2，**Option B 非阻塞週期性重排致動**）

| 排程策略 | 平均 JCT (s) | P50 (s) | P95 (s) | P99 (s) | ΔmeanJCT% [95% CI] | Wilcoxon *p* | P99<bf |
|---|--:|--:|--:|--:|--:|--:|--:|
| FCFS | 171.4 ± 28.9 | 176.6 ± 33.1 | 301.6 ± 53.5 | 313.9 ± 56.3 | +17.7 [+12.6, +22.8] | 0.002 | 8/10 |
| Backfill | 145.8 ± 25.4 | 115.5 ± 35.2 | 429.9 ± 194.0 | 599.7 ± 184.8 | — | — | — |
| SAC | 127.3 ± 22.9 | 121.8 ± 24.4 | 259.7 ± 45.9 | 269.7 ± 46.3 | −12.7 [−15.8, −9.5] | 0.002 | 10/10 |
| RDSAC-mean | 125.7 ± 24.4 | 119.0 ± 27.8 | 255.0 ± 49.1 | 267.5 ± 50.4 | −14.0 [−18.0, −9.9] | 0.002 | 10/10 |
| RDSAC-cvar | 124.7 ± 23.1 | 125.8 ± 29.2 | 250.6 ± 46.3 | 260.8 ± 47.2 | −14.5 [−18.2, −10.9] | 0.002 | 10/10 |
| RLPD | 126.3 ± 23.3 | 125.6 ± 33.6 | 256.4 ± 51.6 | 267.9 ± 52.3 | −13.4 [−17.4, −9.5] | 0.002 | 10/10 |

> [!NOTE]
> 在淺負載下以可落地的 Option B 致動評估，**所有學習式策略的平均 JCT 顯著勝過 Backfill 約 12.7~14.5%**（皆 *p*=0.002，10/10 seed 皆負），且**尾端 P99 亦大幅同勝**（學習式 P99 ≈ 261~270 s vs Backfill 599.7 s，P99<bf 皆 10/10）。此與靜態致動探針「淺載與 Backfill 打平」的結果不同：學習臂的中位數（P50 ≈ 120 s）其實與 Backfill（115 s）相近，勝出主要來自**避開 Backfill 貪婪短工作重排造成的長工作尾端飢餓**（Backfill P99 高達 600 s、seed 間變異極大 ±185 s），而該尾端飢餓在此 regime 連 Backfill 的*平均*都被拉高——B 的連續重排壓制了此飢餓。FCFS 則顯著慢於 Backfill（+17.7%）。此淺載勝出的**致動 vs 排序機制歸因待 B 路徑之派遣順序重跑補實**（見 §5.8 機制段）。

表 7. 三點 poisson 負載掃描：各臂相對 Backfill 之 seed-level 配對 ΔmeanJCT%（負值＝快於 Backfill；每格為 10 seed 配對差之平均 ± 標準差；括號為 Backfill 該負載點之絕對平均 JCT，s）

| 策略 | oversub=2（Backfill 145.8 s） | oversub=4（Backfill 264.0 s） | oversub=6（Backfill 306.7 s） |
|---|--:|--:|--:|
| FCFS | +17.7 ± 7.8 | +7.8 ± 5.0 | +14.9 ± 6.8 |
| SAC | −12.7 ± 4.8 | −10.9 ± 3.9 | −12.0 ± 4.9 |
| RDSAC-mean | −14.0 ± 6.2 | −11.3 ± 3.5 | −11.9 ± 4.4 |
| RDSAC-cvar | −14.5 ± 5.6 | −10.8 ± 3.8 | −9.7 ± 7.6 |
| RLPD | −13.4 ± 6.0 | −10.3 ± 3.2 | −12.1 ± 4.6 |

### 5.3 系統行為量測

除排程品質外，本研究亦量測學習式決策路徑的系統行為，以檢驗其失效安全整合是否非侵入。在 2×1 平台上以 8 個工作負載 seed（每 seed 125 個工作）重放排程序列，於控制平面（CPU）逐次計時策略決策，並依線上服務的判定門檻（低信心 value／entropy）將每次決策分類為 RL 主導、低信心回退，或暫不派遣（no-op），結果如表 8。

決策延遲為次毫秒級（p99 0.27 ms、最大 7.3 ms），較 Lua hook 的 fail-safe 逾時門檻（150 ms）低約三個數量級；在 76,099 次決策中無任一次逾時，顯示 RL 路徑對 slurmctld 幾乎零額外負擔，逾時型回退不會因決策過慢而觸發。在實際放置決策中約 12% 因低信心回退至啟發式基準、其餘由 RL 主導；其中大量的 no-op 反映離散事件下多數時間步並無可派工作，屬正常等待行為。

表 8. 系統行為量測（RDSAC-cvar 策略，2×1，8 seeds × 125 工作，控制平面 CPU）

| 指標 | 數值 |
|---|--:|
| 決策延遲 mean／p50／p95／p99／max (ms) | 0.19／0.18／0.26／0.27／7.27 |
| 逾時（> 150 ms fail-safe 門檻）比例 | 0.00%（0 / 76,099） |
| 放置決策中低信心回退比例 | 12.0%（115 / 959） |
| RL 主導放置比例 | 88.0%（844 / 959） |

### 5.4 統計方法

§5.2 之實機負載掃描（表 4–7）採用三項方法降低誤判：

- Common random numbers：同一 seed 下的所有排程器共用相同工作序列（相同到達時刻與 runtime），使 seed 間變異在配對比較中相互抵消。
- Drift-robust interleaving：同一 campaign 內各臂交錯執行，降低 GPU 暖機或系統漂移與特定方法混淆。
- Seed-level paired statistics：以 seed 為分析單位（每個負載點 n=10 seeds，seed 42–51），避免偽重複；每個非基準臂（FCFS、SAC、RDSAC-mean、RDSAC-cvar、RLPD）皆與 **Slurm 內建 Backfill** 於**同一 seed** 上配對比較。配對顯著性以 seed-level **Wilcoxon 符號秩檢定**（雙尾、n=10）計算——此檢定不假設常態、對尾端敏感，適合本節之偏態 JCT 分佈。

**ΔmeanJCT% 與分析單位**：ΔmeanJCT% 由表 4–7 之 seed-mean JCT 計

$$
\Delta\mathrm{meanJCT}\% \;=\; \frac{\mathrm{JCT}_{\text{arm}} - \mathrm{JCT}_{\text{Backfill}}}{\mathrm{JCT}_{\text{Backfill}}} \times 100 \qquad (\text{負值} = \text{快於 Backfill}),
$$

故可由表 4–7 **逐格還原**；各表的「平均 JCT」為各 seed 內平均 JCT 之未加權平均 ± 標準差（n=10），非 pooled jobs。95% CI 由 seed-level 配對差得出（$\bar{d}\pm1.96\,\mathrm{SE}$）；「P99<bf」為該臂 P99 勝過 Backfill 之 seed 計數（10 中之幾）。RLPD 與其餘學習臂同屬本 campaign、同 10 seed 配對，故其對 Backfill 之比較為配對推論。

**多重比較**：每個負載點對 Backfill 有 5 項比較（FCFS 與四個學習臂）。本節所報之學習臂 *p*=0.002 為 Wilcoxon 於 n=10 的最小可達 *p* 值，遠低於 Holm-Bonferroni 於 5 項 family 校正後之門檻，故顯著性於多重比較下仍穩健。

因單一顯著性檢定於 n=10 之檢定力有限，本研究不僅依賴 *p* 值，而是以點估計、95% CI、逐 seed P99 勝負計數（P99<bf），以及 §5.6 天花板分析交叉佐證效益邊界。

> 上述配對與 interleaving 保證僅適用於各自 campaign 內；跨 campaign（如 §4.2 之 RLPD live 量測與本節之 Option B 部署掃描）不作配對推論。百分比若由彙總平均值計算僅描述點估計，推論性結論一律以同一 seed 內的配對差為準。

### 5.6 天花板分析

此節用於量測 RL 方法的提升效益天花板，因此在模擬器中進行了獨立於任何學習式方法的天花板分析：固定 GPU/MPS placement，讓排程器唯一能控制的槓桿只有**分派順序**，並以 random-restart + swap local search 搜尋每個 instance 在所有 ordering 中可達的最佳平均 JCT，定義

$$
\text{headroom}\% \;=\; \frac{\mathrm{JCT}_{\text{score}} - \mathrm{JCT}_{\text{best-ordering}}}{\mathrm{JCT}_{\text{best-ordering}}},
$$

結果如表 9 所示，代表 headroom 會隨著工作負載的數目有單調遞增的趨勢，即工作越多、學習式的效益越明顯。

表 9. Headroom vs. 負載（2-GPU 叢集，3 families × 10 seeds/row，n=30）

| 負載 (n_jobs) | Headroom（mean ± 95% CI） |
|---|--:|
| 40 | +0.1% ± 0.1% |
| 60 | +0.7% ± 0.5% |
| 80 | +2.0% ± 1.1% |
| 100 | +4.1% ± 2.3% |
| 125 | 10.3% ± 5.3% |
| 150 | 14.0% ± 4.6 |

### 5.8 排序致動的實機驗證：ordering headroom 於重載下可被學習式策略捕捉

前述實機評估（§5.2）與天花板分析（§5.6）留下一個關鍵的歸因問題：§5.6 的模擬天花板分析預測「ordering headroom 隨負載上升」，但 §5.2 的實機路徑（提交時以 `sbatch -w` 綁定 RL 所選節點、**工作順序仍由 Slurm 決定**）在重載下不但未捕捉此 headroom，尾端 P99 反而約為 Slurm-native 的 2.4 倍。此負向結果究竟源於「學習式策略無法捕捉 ordering headroom」，還是源於「該實機路徑並未賦予 RL 對*順序*的控制權」？本節以一條經驗證的**原生排序致動路徑**分離此二因。

**致動路徑與方法。** §3.2 所述的 held-job 控制器，其提交後 `required_nodes` REST 致動在受測的 slurmrestd（v0.0.37）被停用而未生效（摘要與 §3.2 所述）。本研究改採一條原生路徑並先行驗證其正確性：以服務中的策略對一個確定性、**arrival-aware** 的雙節點消耗過程做前展（in-memory drain，重複呼叫 `/act`，容量與 first-fit 回退皆與線上路徑一致；工作僅在其到達時刻後方進入佇列，策略每步只看**當前已到達佇列的 rolling top-16 視窗**，與線上 top-16 select 介面完全一致），讀出策略對該 seed 全部 150 個工作的完整**派遣順序**；隨後將每個工作於其到達時刻以 unheld 提交，提交後即以 `scontrol update Priority` 依前展所得 rank 設定 Slurm 優先序（管理員 `direct_set_prio`，於 unheld 工作設定方能生效；於 held 狀態設定會被 multifactor 重算清除）。Slurm **自身**的 in-process backfill 排程器即以此固定優先序在原生速度下排序並放置工作——RL 掌握*順序*、Slurm 掌握*放置與時機*。

**結果（表 4，深載 oversub=6）。** 在此真實 CUDA、poisson、深負載 regime 下、經可落地的 Option B 致動，**所有學習式策略在平均 JCT 與尾端 P99 上均顯著勝過 Backfill**。平均：ΔmeanJCT −9.7% 至 −12.1%，五臂之 Wilcoxon 符號秩檢定（配對，10 seed）皆 *p*≤0.006。**尾端亦同勝**：學習式 P99 ≈ 583–608 s vs Backfill 750 s（ΔP99 −19% 至 −22%），且 10 seed 中 10 個的 P99 皆勝過 Backfill。值得注意的是，**深佇列下 Backfill 自身的 P99（750 s）為所有臂之最差**——其積極的平均導向重排把部分工作餓入尾端；FCFS 以嚴格序列換得較低尾端（P99 676 s）但平均最高（+14.9%，*p*=0.002，顯著慢於 Backfill）；學習式策略則**同時**取得較低平均（270–277 s vs 307 s）與較低尾端（583–608 s vs 750 s），在兩個軸上皆優於生產 Backfill——確認 Backfill 為較強基準，而學習式排序在兩軸皆勝之。


**中／淺負載複核（oversub=4、2）與三點負載掃描。** 為檢驗上述優勢是否為 oversub=6 單一負載點之特例，於同一 Option B 部署路徑、同 10 個 workload seed（42–51）下再取兩個較淺負載點 **oversub=4**（到達比服務快 4 倍）與 **oversub=2**（快 2 倍，佇列近乎不堆積）。中負載結果如**表 5（§5.2）**：**所有學習式策略在 oversub=4 顯著勝過 Backfill**，平均 ΔmeanJCT −10.3% 至 −11.3%（五臂 Wilcoxon 皆 *p*=0.002），尾端亦全數同勝（學習式 P99 ≈ 508–514 s vs Backfill 720 s，P99<bf 皆 10/10）；FCFS 則顯著慢於 Backfill（+7.8%，*p*=0.002）。**淺負載結果如表 6（§5.2）——與早期靜態致動探針的「打平」結論不同**：在可落地的 Option B 下，**四個學習式臂在 oversub=2 亦顯著勝過 Backfill**，平均 ΔmeanJCT −12.7% 至 −14.5%（四臂 Wilcoxon 皆 *p*=0.002，10/10 seed 皆負、95% CI 皆遠離 0），尾端同勝（學習式 P99 ≈ 261–270 s vs Backfill 600 s，P99<bf 皆 10/10）。故三點並置（**表 7**）顯示：**在部署形態（B）下，學習式對 Backfill 的平均 JCT 優勢於三個負載點皆穩健存在**（≈−13% @ oversub=2、≈−11% @ oversub=4、≈−11% @ oversub=6），而非隨負載單調浮現。

**淺載勝出的來源——尾端飢餓規避，而非 SJF 排序 headroom。** 關鍵在於淺載勝出的*機制*與深載不同：學習臂的**中位數**（P50 ≈ 120 s）其實與 Backfill（115 s）相近甚或略高，其平均優勢**幾乎全數來自尾端**——Backfill 於淺佇列的貪婪短工作優先重排把少數長工作餓入極端尾端（P99 高達 600 s、seed 間變異達 ±185 s），而在此 regime 該飢餓工作足以把 Backfill 的*平均*一併拉高；學習式較公平的排序將長工作施以有界延遲、把 P99 壓到 ≈ 265 s，於是*平均*也隨之較低。這也解釋了為何**靜態致動探針**在淺載呈現「打平」而**Option B** 卻顯著勝出：靜態一次性設定的優先序，在淺佇列大量閒置容量下仍會被 Slurm 自身 backfill 於 gap 中重排而**重新引入**飢餓；B 每數秒持續重申優先序，壓縮了 backfill 偏離的空間，使無飢餓的順序真正被執行。此「**致動連續性 vs 排序品質**」的歸因，其定量佐證（B 路徑實際派遣順序的 ρ(rank,runtime) 與長工作 rank-pct）**待 B 路徑之派遣順序重跑補實**（見下方機制段之待補標註）。就 §5.6 天花板分析而言：該分析量測的是*靜態* best-ordering 於*平均 JCT* 上的 headroom（隨負載單調上升，於淺載 n 小時 ≈ 0），仍成立且不受此處影響；Option B 在淺載額外取得的紅利屬*尾端規避*而非靜態排序 headroom，兩者為不同機制。

（本節 oversub=4／2 與三點掃描之數據表見 §5.2 表 5、表 6、表 7，不在此重列。）

**學習式策略學到了什麼（排序機制分析）。** 為打開上述行為結果的黑箱，將每個 checkpoint 的派遣順序與同一工作流上的兩個參考排序比較：FCFS（依到達）與純 SJF（依真實 runtime、短工作優先，亦即 Backfill 短工作重排所近似、且會把長工作餓入尾端者）。

> [!IMPORTANT]
> **機制分析路徑待更新（B 路徑重跑待補）。** 以下 ρ 與 rank-pct 係以**靜態致動的 arrival-aware 前展順序**（`analyze_rl_order.py` + `precompute_schedule`）讀出，用以解釋*靜態致動探針*的行為；但本節表格已改採可落地的 **Option B**。由於 B 的實際派遣順序受線上重排與 Slurm backfill 交互影響（尤其淺載 ov2 的勝出疑似源於 B 連續重申優先序、壓制 Backfill 飢餓的**致動連續性**，而非排序 ρ 本身），下列淺載「無序可排 → 打平」的靜態機制**不再對應 B 的淺載勝出**。**B 路徑的實際派遣順序機制分析（記錄真實 start time → ρ(rank,runtime)、長工作 rank-pct，B vs 靜態對照）待補跑**（`collect_jct` 需加 per-job 記錄）。以下靜態結果僅供對照。

- **淺負載（oversub=2，靜態順序）**：$\rho(\text{rank},\text{runtime})\approx+0.02$、$\rho(\text{rank},\text{arrival})\approx+0.99$——佇列近空、top-16 視窗過小，靜態前展順序近乎 FCFS。此解釋*靜態探針*淺載打平；惟 Option B 於同負載顯著勝出（表 6），差異疑在致動連續性（待 B 路徑重跑確認）。
- **深負載（oversub=6，靜態順序）**：$\rho(\text{rank},\text{runtime})$ 升至 $\approx+0.25$、$\rho(\text{rank},\text{arrival})$ 降至 $\approx0.6\text{–}0.8$——持續 backlog 給出可重排視窗，策略浮現**溫和的 SJF 傾向**（短工作／inference 類最先，per-class rank-pct：inference $\approx0.31$、llm/batch $\approx0.63$），壓低平均 JCT。
- **關鍵——尾端保護**：最長 20% 工作在 RL 下的平均 rank-percentile $\approx0.54$，而純 SJF 會把它們排到 $\approx0.90$（近佇列末端）。即策略**並非全 SJF**，而是對長工作施加**有界延遲**（$\Delta\approx-0.36$），使其不被餓入尾端。

這解釋了 §5.8 何以能*同時*取得低平均（SJF 傾向）與低尾端（長工作有界延遲）：學習式策略等效於一個**負載自適應、溫和 SJF 但帶尾端上限**的排序——深載在 SJF 與公平之間取一個 Backfill 的貪婪重排達不到的折衷（圖見 `runs/rl_order_analysis/`）。深載此機制在靜態與 B 下一致；淺載的 B-vs-靜態差異則待上述重跑補實。

**詮釋與界限。** 三點結論：（1）重載下 Backfill 之上的 ordering headroom 不僅存在，且**可被現有學習式策略在真實 CUDA、realistic poisson 到達下捕捉**——前提是 RL 須經由一條原生致動路徑取得對*順序*的控制權。這修正了 §5.2／§5.7 的歸因：placement-only 的負向結果與 2.4 倍尾端，相當程度上是「未賦予 RL 順序控制權」與「進程外綁定／節點綁定致動」之限制，而非「RL 無法貢獻」之證明。（2）此結果亦**修正**了本文早期以 wait-dominated 代理所得的暫時性判斷「尾端結構性受限、fairness reward 動不了 P99」：在深佇列、真實 CUDA 下，學習式策略的 P99 反而**穩定低於** Backfill（10/10）——因為此 regime 的尾端主要來自 Backfill 為衝平均而產生的重排飢餓，正是 RL 排序可避免者。（3）界限須明列：本結果為**三個負載點**（oversub=2／4／6）之量測（表 7）——在可落地的 Option B 部署形態下，平均 JCT 上學習式於**三個負載點皆顯著勝過 Backfill**（≈−11% 至 −13%，皆 *p*=0.002/≤0.006），尾端 P99 亦於三點全數同勝。惟勝出*機制*隨負載而異：深／中載主要來自持續 backlog 下的溫和 SJF 排序 headroom（§5.6 之預測對象，該靜態 headroom 隨負載單調上升），淺載則主要來自 B 連續重排對 Backfill 尾端飢餓的規避（**致動連續性**效應，非靜態排序 headroom；此淺載歸因待 B 路徑派遣順序重跑補實）。效益之負載相依性主要體現於*機制*而非*勝負*；惟三點皆於同一 2×1 小叢集、單一 workload 家族（aimix）量得，跨叢集規模與工作負載組成之外推仍待驗證。本節隔離*排序*效益，RL *placement* 之效果另見 §5.2／§5.7。

> 相關材料見 `runs/step3prio_*/ov{2,4,6}/`（真實 CUDA poisson oversub=2／4／6）與 `eval/scripts/{scontrol_ab.py,run_step3_prio.sh,aggregate_step3.py,aggregate_step3_sweep.py}`；三點負載掃描由 `aggregate_step3_sweep.py` 彙整。排序機制分析（RL order vs SJF/FCFS 的 Spearman、長工作 rank-pct、per-class）由 `eval/scripts/analyze_rl_order.py` 產出至 `runs/rl_order_analysis/`（含散點圖）。

### 5.9 效益邊界小結

綜合 §5.2–5.8，效益邊界可依「RL 掌握何種槓桿、以何路徑致動」而清楚劃分。在 placement 為主的實機路徑（§5.2，RL 只選節點、順序由 Slurm 決定）下，部分學習式策略（SAC、RDSAC-cvar）的平均 JCT 僅略優於或打平 FCFS／Backfill，且未形成相對 size-aware 啟發式的穩健優勢；更關鍵的是這些平均值是以顯著的尾端代價換得——該路徑下學習式 P99（約 500–600 s，重載達 640–668 s）約為 FCFS／Backfill（255–270 s）的兩倍。

然而 §5.8 顯示，當 RL 經一條**原生排序致動路徑**取得對*順序*的控制權、並以可落地的 **Option B（非阻塞週期性重排）**致動後，在**真實 CUDA、realistic poisson 到達、深負載（oversub=6）**下，所有學習式策略不僅平均 JCT **顯著勝過生產 Backfill 約 10–12%**（Wilcoxon 皆 *p*≤0.006），**尾端 P99 亦大幅同勝約 19–22%**（10 seed 中 10/10 的 P99 低於 Backfill）。此結果一方面**實機確認**了 §5.6 天花板分析對「ordering headroom 隨負載上升」的預測（且**可被現有 DRL 策略捕捉**，修正了先前「出現 headroom 但 DRL 未能捕捉」的暫時性判斷），另一方面也指出 §5.2 的負向結果與 2.4 倍尾端相當程度是**致動路徑**（進程外綁定、節點綁定序列化、順序不由 RL 掌握）之限制，而非策略本身無法貢獻。

因此，現有證據支持的、更精確的結論是：**學習式排程的效益取決於 RL 是否掌握*排序*槓桿、致動路徑是否原生、以及負載是否足以形成可重排的 backlog**——三者具備時，重載下平均 JCT *與*尾端 P99 皆可穩健勝過已充分調校的生產 Slurm 排程。此亦**修正**了本文早期以 wait-dominated 代理所得的暫時性判斷「尾端結構性受限、fairness reward 動不了 P99」：在深佇列、真實 CUDA 下，尾端主要來自 Backfill 為衝平均而生的重排飢餓，正是 RL 排序可避免者，故學習式 P99 反而穩定較低。在可落地的 Option B 部署形態下，完整的 poisson 三點負載掃描（oversub=2／4／6）顯示平均 JCT 優勢於**三個負載點皆穩健存在**（≈−11%～−13%，皆顯著；表 7），惟其*機制*隨負載而異——深／中載主要來自 backlog 下的溫和 SJF 排序 headroom（§5.6 之預測對象，隨負載單調上升），淺載則主要來自 B 連續重排對 Backfill 尾端飢餓的規避（致動連續性效應，此淺載歸因待 B 路徑派遣順序重跑補實）；placement 消融（§5.7）則因變異過大而無法提供確定歸因。策略效益整體仍取決於工作負載、負載強度、硬體規模與底層資源分配後端。

> 相關材料見 `runs/headroom_*/` 與 `runs/ablation_std_*/`

## 6. 結論與未來展望

### 6.1 結論

本研究實作了可在異質 GPU 與 NVIDIA MPS 配額約束下輸出工作選擇與 GPU placement 的 DRL 策略，並透過 Slurm job submission path 整合到真實排程流程。相較傳統只選 GPU 或只做固定規則的方法，本研究在排程框架中整合了 GPU 型號差異、MPS 配額、工作特徵、佇列狀態與回饋訊號，並以失效安全設計確保排程核心穩定。

實驗結果顯示，學習式策略的效益取決於其掌握的排程槓桿、致動路徑與負載強度。在僅控制 placement（節點綁定、順序由 Slurm 決定）的實機路徑下，部分學習式策略（SAC、RDSAC-cvar）在平均 JCT 上僅略優於或打平 FCFS／Backfill，且以顯著尾端代價換得（P99 約為 Slurm-native 的兩倍），未形成相對 size-aware 啟發式的穩健優勢。然而，當策略經一條經驗證的**原生排序致動路徑**、並以可落地的 **Option B（非阻塞週期性重排、失效安全）**致動取得對*順序*的控制權後，在**真實 CUDA、poisson 到達、深負載（oversub=6）**下，所有學習式策略的平均 JCT **顯著勝過生產 Backfill 約 10–12%**、尾端 P99 更**大幅同勝約 19–22%**（Wilcoxon 皆 *p*≤0.006，P99 於 10/10 seed 低於 Backfill）。此結果實機確認了模擬天花板分析對「ordering headroom 隨負載上升」的預測，並顯示先前的負向結果相當程度上係致動路徑（進程外綁定、節點綁定序列化）之限制而非策略本身無法貢獻。

此結果亦修正了本文早期以 wait-dominated 代理所得的暫時性判斷「尾端結構性受限、fairness reward 動不了 P99」：在深佇列、真實 CUDA 下，尾端主要來自 Backfill 為衝平均而生的重排飢餓，正是 RL 排序可避免者，故學習式 P99 反而穩定較低。在可落地的 Option B 部署形態下，本文之 poisson 三點負載掃描（oversub=2／4／6）顯示平均 JCT 優勢於**三個負載點皆穩健存在**（≈−11%～−13%，皆顯著），惟勝出*機制*隨負載而異（深／中載為 backlog 下的排序 headroom、淺載為對 Backfill 尾端飢餓的規避，後者之定量歸因待 B 路徑派遣順序重跑補實）；跨較大型叢集與其他工作負載組成之外推仍待後續驗證，placement 消融亦因變異過大而無法形成確定歸因。

### 6.2 未來展望

未來工作可沿以下方向展開：

1. **更大、更高競爭的叢集**：擴展至更多節點與 GPU，檢驗學習式策略的效益是否隨叢集規模與競爭程度進一步增強。
2. **MIG + MPS fraction 混合 partition**：同時納入硬體級隔離與軟體級共享，建立更完整的 GPU sharing action space [3][16]。
3. **Offline RL / 真線上 RLPD**：收集更大量真實 Slurm transition，以 offline RL 或真線上 RLPD 改善 sim-to-real 轉移 [9]。
4. **Energy-aware scheduling**：將功耗、能效與碳排納入 reward，使排程器兼顧效能與能源效率的最佳化。
5. **LLM serving workload**：加入更真實的 LLM serving trace，評估 token latency、throughput、SLO violation 與 batch scheduling 的交互影響 [30]。

## 參考文獻

[1] M. Jeon, S. Venkataraman, A. Phanishayee, et al., "Analysis of large-scale multi-tenant GPU clusters for DNN training workloads," in *USENIX ATC*, 2019.

[2] Q. Weng, W. Xiao, Y. Yu, et al., "MLaaS in the wild: workload analysis and scheduling in large-scale heterogeneous GPU clusters," in *USENIX NSDI*, 2022.

[3] NVIDIA Corporation, "Multi-Process service (MPS)," NVIDIA Documentation, 2024.

[4] SchedMD, "Slurm workload manager," https://slurm.schedmd.com, 2024.

[5] Kubernetes Authors, "Kubernetes," https://kubernetes.io, 2024.

[6] P. Christodoulou, "Soft actor-critic for discrete action settings," *arXiv:1910.07207*, 2019 (unpublished).

[7] W. Dabney, G. Ostrovski, D. Silver, and R. Munos, "Implicit quantile networks for distributional reinforcement learning," in *ICML*, 2018.

[8] X. Ma, J. Chen, L. Xia, J. Yang, Q. Zhao, and Z. Zhou, "DSAC: distributional soft actor-critic for risk-sensitive reinforcement learning," *Journal of Artificial Intelligence Research*, vol. 83, 2025.

[9] P. J. Ball, L. Smith, I. Kostrikov, and S. Levine, "Efficient online reinforcement learning with offline data (RLPD)," in *ICML*, 2023.

[10] SchedMD, "Slinky: Slurm in Kubernetes," https://github.com/SlinkyProject, 2024.

[11] Y.-D. Lin, Y.-T. Ling, Y.-C. Lai, and D. Sudyana, "Reinforcement learning for AI as a service: CPU-GPU task scheduling for preprocessing, training, and inference tasks," *IEEE Transactions on Network and Service Management*, vol. 22, no. 4, 2025.

[12] G. Zhang, W. Guo, Z. Tan, Q. Guan, and H. Jiang, "KIS-S: a GPU-aware Kubernetes inference simulator with RL-based auto-scaling," *arXiv:2507.07932*, 2025 (unpublished).

[13] Q. Wu, P. Chen, and Y. Wang, "Defragmentation scheduling with deep reinforcement learning in shared GPU clusters," in *ACM SoCC*, 2025.

[14] X. Wang, Y. Li, F. Guo, Y. Xu, and J. C. S. Lui, "Dynamic GPU scheduling with multi-resource awareness and live migration support," *IEEE Transactions on Cloud Computing*, vol. 11, no. 3, 2023.

[15] H. Sedighi, F. Wuhib, and R. H. Glitho, "Dynamic task scheduling and adaptive GPU resource allocation in the cloud," *IEEE Transactions on Network and Service Management*, vol. 23, 2026.

[16] E. Lipe, N. Karia, C. Espenshade, C. Stein, A. Tantawi, and O. Tardieu, "Energy efficient scheduling of AI/ML workloads on Multi-Instance GPUs with dynamic repartitioning," in *IEEE CCGrid*, 2025.

[17] S. Choi, S. Lee, Y. Kim, J. Park, Y. Kwon, and J. Huh, "Serving heterogeneous machine learning models on multi-GPU servers with spatio-temporal sharing," in *USENIX ATC*, 2022.

[18] U. Saroliya, E. Arima, D. Liu, and M. Schulz, "Hierarchical resource partitioning on modern GPUs: a reinforcement learning approach," in *IEEE CLUSTER*, 2023.

[19] W. Xiao, R. Bhardwaj, R. Ramjee, et al., "Gandiva: introspective cluster scheduling for deep learning," in *USENIX OSDI*, 2018.

[20] J. Gu, M. Chowdhury, K. G. Shin, et al., "Tiresias: a GPU cluster manager for distributed deep learning," in *USENIX NSDI*, 2019.

[21] A. Bhatt, D. Palenicek, B. Belousov, M. Argus, A. Amiranashvili, T. Brox, and J. Peters, "CrossQ: batch normalization in deep reinforcement learning for greater sample efficiency and simplicity," in *ICLR*, 2024.

[22] Kubeflow Authors, "Kubeflow: the machine learning toolkit for Kubernetes," https://www.kubeflow.org, 2024.

[23] Volcano Authors, "Volcano: a cloud native batch system for compute-intensive workloads," CNCF, https://volcano.sh, 2024.

[24] Kubernetes SIG-Scheduling, "Kueue: Kubernetes-native job queueing," https://kueue.sigs.k8s.io, 2024.

[25] NVIDIA, "KAI scheduler: a Kubernetes-native GPU scheduler for AI workloads," https://github.com/NVIDIA/KAI-Scheduler, 2025.

[26] M. Tsenos and V. Kalogeraki, "Exploring GPU-based workload scheduling techniques for edge computing," in *IEEE IC2E*, 2025.

[27] A. A. Majeed, M. Meribout, and S. M. Sali, "Scheduling techniques of AI models on modern heterogeneous edge GPU: a critical review," *IEEE Transactions on Industrial Informatics*, vol. 22, no. 4, 2026.

[28] S. Dong, B. Zheng, L. Pan, and S. Liu, "A reinforcement learning-based approach for scheduling machine learning training tasks in heterogeneous Kubernetes clusters," *Future Generation Computer Systems*, vol. 182, art. 108459, 2026.

[29] S. Dongare, R. I. S. Khan, H. Albahar, N. Zhao, D. Melendez Maita, and A. R. Butt, "Hybrid learning and optimization-based dynamic scheduling for DL workloads on heterogeneous GPU clusters," in *ACM SoCC*, 2025.

[30] Y. Wang, Y. Hu, A. Klimovic, X. Zhang, Y. Wen, G. Sun, and J. Lin, "Semantic-aware scheduling for GPU clusters with large language models," *arXiv:2510.03334*, 2025 (unpublished).

[31] A. Y. Ng, D. Harada, and S. Russell, "Policy invariance under reward transformations: theory and application to reward shaping," in *ICML*, 1999.

[32] Kubernetes Authors, "Dynamic resource allocation (DRA)," Kubernetes Documentation, 2025.

[33] J. Duan, Y. Guan, S. E. Li, Y. Ren, and B. Cheng, "Distributional soft actor-critic: off-policy reinforcement learning for addressing value estimation errors," *IEEE Transactions on Neural Networks and Learning Systems*, 2021.
