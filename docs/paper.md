<!--
TANET 投稿雛形 — 主題三「網際網路與雲端技術應用」
子題：雲端運算、邊緣運算、雲霧整合運算 ／ 分散式系統
格式對齊：TANET 全文 4–6 頁、定稿 PDF；本檔為 Markdown 雛形，
定稿時請套用大會 Word/odt/LaTeX 範本（字體字級依範本：標題標楷體、
內文新細明體/Times New Roman，雙欄，勿編頁碼）。
作者與單位欄位請自行填入。
-->

# 基於 Slurm 與 Kubernetes 的 AI 伺服器 GPU 工作負載智慧排程：風險敏感深度強化學習與模擬到實機評估

### Intelligent GPU Workload Scheduling for AI Servers on Slurm-over-Kubernetes: Risk-Sensitive Deep Reinforcement Learning and a Sim-to-Real Evaluation

**作者一¹、作者二²**
¹○○大學 ○○系　²○○大學 ○○系
{author1, author2}@example.edu.tw

---

## 摘要

異質 GPU 叢集上的 AI 工作負載（推論與訓練混合）排程，直接影響資源使用率、工作完成時間（Job Completion Time, JCT）與服務水準目標（Service Level Objective, SLO）的達成。本研究設計並實作一套**以 Kubernetes（k3s）部署、以 Slurm 為排程核心**的 AI 伺服器 GPU 排程研究平台，整合 NVIDIA Multi-Process Service（MPS）達成卡內細粒度共享，並在 Slurm 的工作提交掛鉤（`job_submit.lua`）中嵌入一個**非阻塞、失效即回退（fail-safe）**的強化學習決策端點：以風險敏感的分散式深度強化學習（RDSAC，結合 discrete SAC 與 Implicit Quantile Network，並以 CVaR 風險量度優化尾端延遲）作為放置（placement）建議者，任何服務異常皆自動退回既有啟發式分數排程，slurmctld 永不被阻塞。為了在「實機樣本稀少、單一節點即一個跑數分鐘的工作」的限制下取得可信的結論，本研究提出一套**模擬到實機（sim-to-real）評估方法學**：離散事件模擬器訓練、實機配對 A/B、抗跑序漂移（drift-robust）的交錯輪轉、多 seed 配對信賴區間，並同時報告平均與尾端指標（p95／p99／CVaR）及 SLO 違反率。在一座雙節點異質叢集（RTX 4070＋RTX 3080）上的實測顯示：**模擬環境可清楚區分排程策略，但在 2×1 規模的真實叢集上，無論啟發式或深度強化學習排程，彼此皆統計打平**；經抗漂移與多 seed 校正後，原本看似的優勢被證實為跑序漂移與小樣本雜訊的假象。進一步地，即使在為風險機制注入不確定性的隨機模擬中，風險敏感 DRL 於多 seed 下亦未穩健勝過純量基準，且單 seed 模擬結果被證實會誤導。本研究的貢獻在於：一套可重現的雲端 GPU 排程平台、一個失效安全的 RL 整合架構，以及一個誠實校正後的 sim-to-real 評估方法學——在此規模排程策略統計等價，而「智慧排程的價值是否需要更大叢集規模方能顯現」則被界定為尚待驗證的假設（本研究的 1×1–2×2 掃描未能證實其交叉）。

**關鍵詞**：GPU 排程、Kubernetes、Slurm、深度強化學習、MPS、邊緣運算、模擬到實機評估

## Abstract

Scheduling mixed AI workloads (inference and training) on heterogeneous GPU clusters directly affects utilization, Job Completion Time (JCT), and Service Level Objective (SLO) attainment. We design and implement a GPU scheduling research platform that is **deployed on Kubernetes (k3s) with Slurm as the scheduling core**, integrates NVIDIA Multi-Process Service (MPS) for intra-GPU fine-grained sharing, and embeds a **non-blocking, fail-safe** reinforcement-learning decision endpoint into Slurm's job-submit hook (`job_submit.lua`). A risk-sensitive distributional deep RL policy (RDSAC: discrete SAC + Implicit Quantile Network, optimized with a CVaR risk measure for tail latency) acts as a placement advisor; any service fault transparently falls back to the existing score heuristic, so slurmctld is never blocked. Because real-cluster samples are scarce (one node hosts a job running for minutes), we propose a **sim-to-real evaluation methodology**: discrete-event simulation for training, paired live A/B, drift-robust interleaving, multi-seed paired confidence intervals, and joint reporting of mean and tail metrics (p95/p99/CVaR) plus SLO violation. On a two-node heterogeneous cluster (RTX 4070 + RTX 3080), the simulator cleanly separates scheduling policies, **but on the real 2×1 cluster, neither heuristic nor deep-RL schedulers differ statistically**; after drift-robust and multi-seed correction, an apparent advantage is shown to be a run-order-drift and small-sample artifact. Moreover, even in a stochastic simulation designed to activate the risk machinery, risk-sensitive DRL does not robustly beat a scalar baseline under multi-seed evaluation, and single-seed simulation results are shown to be misleading. Our contributions are a reproducible cloud GPU scheduling platform, a fail-safe RL integration architecture, and an honestly-corrected sim-to-real methodology. Its key insight is that scheduling-policy differences are statistically equivalent at this scale; whether the value of intelligent scheduling requires larger cluster scale is framed as an open hypothesis that our 1×1–2×2 sweep did not confirm.

**Keywords**: GPU scheduling, Kubernetes, Slurm, deep reinforcement learning, MPS, edge computing, sim-to-real evaluation

---

## 1. 前言

生成式 AI 與深度學習的普及，使學術與企業的 GPU 叢集需同時承載**低延遲推論**與**長時間訓練**兩類性質迥異的工作負載：前者須在服務水準目標（SLO）期限內完成、且常僅需部分 GPU 算力，後者則長時間獨佔整張卡。當底層硬體有限且**異質**——不同世代 GPU 的算力差異可達數倍——如何將兩類工作妥善排程與放置，以兼顧工作完成時間（JCT）、資源使用率與 SLO 達成，已成為雲端與邊緣運算基礎設施的核心議題 [1][4]。

傳統 HPC 排程器（如 Slurm）以靜態優先權與回填（backfill）為主，對 GPU 卡內共享與工作負載特性的感知有限。學界雖已提出多種以強化學習（RL）優化叢集排程的方法 [2][3]，但多數僅止於模擬評估，**鮮少在真實叢集上以嚴謹的統計方法檢驗其效益是否成立**；尤其「模擬中可區分的策略差異能否轉移到實機」這一 sim-to-real 問題，至今缺乏系統性的探討。實機評估之所以困難，在於單一節點即承載一個跑數分鐘至數小時的工作，使可用樣本極度稀少，且量測易受叢集暖機等時變因素干擾而產生假性排名。

針對上述缺口，本研究以一座雙節點異質叢集（RTX 4070＋RTX 3080，配備 NVIDIA MPS）為實驗平台，建構從模擬訓練到實機驗證的完整流程，並以抗漂移、多 seed 配對統計的方法學誠實檢驗排程策略的真實效益。本研究的目標與貢獻如下：

1. **雲端 GPU 排程平台**：以 k3s 部署、Slurm 為排程核心、NVIDIA MPS 達成卡內共享，並以 Helm 完整封裝，可重現部署於異質節點。
2. **失效安全的 RL 整合架構**：於 Slurm 工作提交掛鉤嵌入非阻塞的 RL 決策端點，服務異常即回退啟發式，兼顧研究彈性與生產可靠性。
3. **風險敏感深度強化學習放置策略與其模擬行為分析**：以分散式 RL（RDSAC）建模回報分布並以 CVaR 風險量度；經多 seed 消融誠實揭示其分布式評論家在不確定性下高變異、易崩潰，而 CVaR 風險扭曲的實質作用是**完成率穩定器**而非速度優勢——即使在模擬中亦未穩健勝過純量基準。
4. **模擬到實機評估方法學與規模洞見**：提出抗漂移、多 seed、配對信賴區間並兼顧尾端指標的評估流程，誠實呈現「2×1 規模下排程策略統計打平」的校正後結論，並指出**單 seed 模擬結果會誤導**這一方法學教訓。

## 2. 相關研究

**GPU 叢集排程與分析。** Jeon 等人對微軟 Philly 叢集的大規模多租戶 GPU 工作負載進行分析 [1]，揭示了排隊延遲與資源碎裂問題。Gandiva [2] 與 Tiresias [3] 分別利用 DL 工作的可遷移性與分布感知排程降低 JCT。Weng 等人對阿里巴巴 PAI 叢集的研究 [4] 指出生產 MLaaS 工作高度分片化、以單卡短工作為主。本研究的合成工作負載即以 [1][4] 的統計特性為依據。

**GPU 共享。** NVIDIA MPS [5] 允許多個行程共享單張 GPU 的計算資源，是在小型叢集上提升使用率的關鍵機制；本研究以 MPS 槽（每卡 4 槽，對應 25／50／75／100％）建模卡內共享。

**強化學習排程。** 以 RL 進行資源排程的研究多採用 actor–critic 或值函數方法。本研究的決策核心 RDSAC 為三項技術的整合：離散動作 Soft Actor-Critic [6]、Implicit Quantile Network 分布式評論家 [7]，以及以 CVaR 為風險量度的尾端敏感優化 [7]。為弭平模擬與實機落差，亦採用離線到線上的 RLPD 微調概念 [8]。獎勵塑形採用保證最優策略不變的位能塑形 [9]。

**雲端原生 GPU 排程生態系。** 在 Kubernetes 生態中，數個成熟系統處理 GPU／批次工作負載，但分屬不同抽象層：Kubeflow [12] 負責 ML 工作的生命週期（分散式訓練、超參數搜尋、模型服務），其本身不做排程，而將決策**委派**給批次排程器；Volcano [13] 提供 pod 群組的 gang 排程與 DRF／binpack 等規則式外掛；Kueue [14] 實作 job 級佇列、配額借還（ClusterQueue／ResourceFlavor／Cohort）、fair-share 與 gang 准入，但以暫停（suspend）控制准入、**不負責 pod 放置**；Kubernetes 1.34 起正式釋出（GA）的動態資源分配（Dynamic Resource Allocation, DRA）[15] 則將 GPU 分片、MIG、time-slicing 等以 ResourceClaim／ResourceSlice／DeviceClass **宣告式**地納入 API，成為一等公民。表 0 依抽象層整理這些系統與本研究的定位。

表 0. 雲端原生 GPU 排程系統的層級定位

| 系統 | 所在層 | 機制形態 | 學習式策略 | 尾端／SLO 目標 | GPU 分片 |
|---|---|---|:--:|:--:|:--:|
| Kubeflow [12] | 工作生命週期 | 委派給批次排程器 | ✗ | ✗ | 委派 |
| Volcano [13] | Pod 群組排程 | 規則式 heuristic（gang/DRF/binpack） | ✗ | ✗ | time-slice（無策略） |
| Kueue [14] | Job 級佇列／配額 | 規則式＋約束求解（不做放置） | ✗ | ✗（fair-share 非尾端） | ResourceFlavor 標記 |
| K8s DRA [15] | 裝置分配**機制** | 約束匹配 | ✗ | ✗ | ✓（宣告式一等公民） |
| **本研究（RDSAC）** | 排序／放置**策略** | **學習式＋風險敏感（CVaR）** | ✓ | ✓（直接優化尾端） | MPS-aware 策略 |

**與既有工作的差異。** 上述系統皆為**規則式或約束求解**：Kueue／Volcano 解的是配額與 gang 准入、DRA 提供的是分片**機制**、Kubeflow 管的是生命週期；沒有任何一個是「**學習式、且以尾端延遲（tail latency）為目標的排序／放置策略**」。本研究正落於此空隙：RDSAC 對回報分布以 CVaR 優化 p99／SLO 尾端，是生產系統皆未優化的量。值得強調的是，DRA 提供的是「如何表達要 0.25 張 GPU」的*機制*，而非「該把哪些工作打包、用什麼順序以壓低尾端」的*策略*——因此本研究的學習式策略與 DRA 並非競爭，而是**互補**：一個尾端敏感的策略可在 DRA 之上驅動裝置選擇與准入排序。此外，既有 RL 排程研究多止於模擬；本研究的重點不在宣稱 RL 必勝，而在**建立一套能在真實異質叢集上、以統計嚴謹方式檢驗排程策略效益的方法學**，並誠實回報其規模條件。

## 3. 實驗目的與系統架構

### 3.1 整體架構

平台分為兩個鬆耦合層：**基礎設施層**（Slurm on Kubernetes）與**排程研究層**（模擬器＋深度強化學習）。

叢集以 k3s 部署，控制節點（RTX 4070）兼任 control-plane，工作節點（RTX 3080，算力約前者 0.25×）為異質 GPU 來源。Slurm 控制器、登入節點與 GPU 工作節點皆以容器化 StatefulSet 部署，GPU 經 NVIDIA device plugin 與 MPS control daemon 暴露為可分片資源。

### 3.2 失效安全的 RL 整合

整合點為 Slurm 的 `job_submit.lua`：工作提交時，掛鉤 `rl_hook.lua` 以 HTTP 呼叫 RL 推論服務的 `POST /decide`，取得放置／優先權建議；若服務逾時或異常，掛鉤**靜默回退**至既有啟發式分數排程（以 MPS 適配、VRAM 適配、短工作優先三因子加權）。此設計確保 slurmctld 永不被第三方服務阻塞，使研究用的 RL 元件可安全運行於生產路徑。生產部署中 RL 僅設定佇列優先權，實際放置仍由 Slurm `select/cons_tres` 與 GRES 決定。

### 3.3 模擬器與強化學習環境

離散事件模擬器以「提交／結束」事件驅動，建模 Node → GPU → MPS 槽的階層資源與異質算力。其上以 Gymnasium 介面封裝為 RL 環境：觀測為佇列前 K 個工作的特徵（GPU 數、MPS 需求、等待時間、SLO 緊迫度、工作類別、GPU 型別 one-hot 等），於 2×1 拓樸下維度為 166；動作為「放置於某節點某 MPS 槽」或「暫不放置」。

### 3.4 風險敏感深度強化學習（RDSAC）

決策策略為自行整合的 **discrete 分布式 SAC**（本文稱 RDSAC）：雙頭 IQN 評論家分別建模回報分布（reward 回報 $Z_R$ 與 entropy 回報 $Z_H$），以 quantile Huber loss 學習，搭配 twin-Q、軟更新（τ=0.005）與遮罩式 categorical actor。風險敏感性透過在 actor 目標與動作價值上對回報分布套用 CVaR 扭曲 $\rho[Z_R]$ 達成，對應排程中的長尾 runtime／慢節點（straggler）風險。訓練採優先經驗回放（PER）、n-step 回報、分數暖啟動與位能獎勵塑形。RDSAC「分布式 SAC」之名承襲自 Duan 等人的 Distributional Soft Actor-Critic（DSAC）[16]；惟本研究為**離散動作**、雙頭 IQN 的自組版本，是離散 SAC [6]、IQN 分布式評論家 [7] 與 CVaR 風險量度的組合，**並非** [16] 連續控制版本（將回報建模為單一高斯）的 1:1 重現，兩者不宜逕行對照。

## 4. 實驗與評估方法

### 4.1 評估方法學

實機評估面臨「一個 step 即一個跑數分鐘的工作、湊滿樣本需時甚久」的根本限制，故採用 sim-to-real 流程：模擬器訓練 → 凍結 checkpoint → 實機配對 A/B。為取得可信結論，方法學包含四項要件：

- **抗跑序漂移（drift-robust）**：GPU 隨運行暖機、快取轉熱會使「越晚跑的越快」。若各方法依序整段跑完，跑序會與方法混淆。本研究以交錯輪轉（interleave）讓每個方法跨多輪輪過各個跑序位置，並丟棄暖機輪，將漂移誤差平均化。
- **多 seed 配對信賴區間**：以共用隨機數（CRN）讓各方法跑相同工作負載並配對相減，再以多個訓練 seed 重複，以配對 t 檢定回報 95% 信賴區間與 p 值。
- **尾端與 SLO 指標**：除平均 JCT 外，同時報告 p95／p99／CVaR 與 SLO 違反率，以捕捉飢餓與 straggler。
- **真實工作負載**：除合成負載外，建置可攜式 PyTorch 環境於共享儲存，以真實 BERT 推論／微調作為實機工作。

### 4.2 主要結果

**模擬環境可區分策略（方法學正面結果）。** 在離散事件模擬中，以對 SLO 敏感的 AI 伺服器工作負載（2×1 拓樸、offered load ρ≈0.7 的中度競爭、8 個 held-out seed）評估，具尺寸感知的啟發式相較先到先服務（FCFS）顯著降低 SLO 違反率（表 1），顯示模擬器具備區分排程策略的能力——前提是工作負載與指標選對了維度（時序排序、SLO 感知），而非需要規模的多節點裝箱。此工作負載的 SLO 定義為：推論工作（短、佔 1 GPU 的 25／50％ MPS）帶有延遲期限 `slo_s` = runtime × 4，訓練工作（長、獨佔或大 MPS，含少量 2-GPU 跨節點 gang）為 best-effort（無期限）；SLO 違反率即帶期限工作中 JCT 超過 `slo_s` 的比例。

表 1. 模擬環境下 AI 伺服器工作負載的排程器區分（2×1、ρ≈0.7、8 seed 平均）

| 排程器 | 平均 JCT (s) | 推論 JCT (s) | SLO 違反 (%) | 使用率 |
|---|--:|--:|--:|--:|
| FCFS | 2199 | 1847 | 66.5 | 0.58 |
| multifactor | 1108 | 461 | 41.1 | 0.63 |
| score | 1129 | 520 | 40.7 | 0.63 |

**跑序漂移會污染單趟排名。** 在真實叢集上，單趟（block design）量測一度顯示 FCFS「顯著贏」score 達 +5.0%。然而其改善幅度與「跑第幾位」完美單調相關（表 2）：同一個 FCFS，跑最後一位時 +5.0%、跑最先時則僅 +0.5% 甚至轉為 −0.4%（不顯著）。此為叢集隨時間暖機的漂移假象，而非排程器差異。

表 2. FCFS 的「優勢」隨跑序位置變動（揭示漂移）

| 排程器 | seed42 位置 | seed43 位置 | seed44 位置 | ΔJCT% vs score（各 seed）|
|---|--:|--:|--:|---|
| FCFS | 4（最後）| 1（最先）| 3 | +5.0 / +0.5 / −0.4 |
| packing | 3 | 2 | 1 | +1.6 / +0.8 / −0.6 |
| multifactor | 2 | 3 | 4 | +1.0 / +0.9 / −0.7 |

![圖 1](figures/fig_drift.png)

圖 1. 將三個啟發式跨 3 seed 的 ΔJCT% 對「跑序位置」作圖：正斜率的趨勢線（+0.62%/位）顯示表面「優勢」隨越晚跑而增大，證實其為叢集暖機漂移的假象、而非排程器本身的效果。

**真實 2×1 叢集：啟發式統計打平。** 以 3 seed × 3 種臂順序校正跑序位置後的 cross-seed 聚合如表 3，三個 ΔJCT% 的 mean±std **全部跨越 0**，即生產 score 與 Slurm 原生 FCFS／multifactor／packing 在真實 2×1 上統計打平，無任何排程器具優勢。

表 3. 真實 2×1 叢集抗漂移、多 seed 聚合（mean±std）

| 排程器 | 平均 JCT (s) | p95 | p99 | CVaR | ΔJCT% vs score |
|---|--:|--:|--:|--:|--:|
| score | 182.7±31.7 | 369.1±46.6 | 409.8±37.7 | 349.2±43.0 | — |
| multifactor | 181.8±30.3 | 364.5±42.7 | 405.9±35.6 | 346.7±39.4 | +0.4±1.0 |
| packing | 181.5±30.7 | 365.9±42.1 | 405.2±33.5 | 346.3±38.3 | +0.6±1.1 |
| FCFS | 179.7±32.0 | 362.5±37.2 | 402.4±28.2 | 343.2±32.8 | +1.7±2.9 |

**深度強化學習放置：精確量測下顯著小輸。** 在暴露 GPU 異質性的放置實驗中，深度強化學習 checkpoint 一度名目領先 +3.9%（p=0.116，不顯著）；於非飽和 regime 降噪並將樣本三倍化（n=246）後，三個學習型策略全部反轉為**顯著小輸** score（表 4），且此結論跨兩個訓練 seed 一致。以真實 BERT 工作橫掃低／高負載與 2-GPU gang（head-of-line blocking）三種競爭機制，亦僅見落在雜訊內的微弱訊號。

表 4. 實機放置精確四方比較（n=246，配對 t 檢定）

| 排程器 | 平均 JCT (s) | p99 | CVaR | ΔJCT% | t 檢定 p |
|---|--:|--:|--:|--:|--:|
| score | 167.8 | 425.6 | 374.7 | （基準）| — |
| SAC | 174.0 | 432.4 | 379.5 | −3.7 | 3.7e-16 |
| RDSAC-mean | 174.5 | 435.8 | 379.5 | −4.0 | 2.1e-17 |
| RDSAC-cvar | 175.5 | 432.8 | 381.6 | −4.6 | 5.3e-12 |

須釐清**統計顯著與實務顯著的區別**：表 4 的 p 值極小（≈1e-12～1e-17）源於 n=246 的大樣本，代表「可偵測地變慢」而非「大幅變慢」；三個學習臂的 ΔJCT% 落在 −3.7～−4.6%，仍位於 §4.5 界定的 ±5% 實務等價帶內（僅偏於其負緣）。換言之，學習型策略在此規模是**可統計偵測地、但非實務顯著地**遜於 score——與後續「排程策略空間近乎是平的」洞見一致，而非與之矛盾。

### 4.3 模擬多 seed 消融：風險敏感 DRL 在模擬中亦未勝出

§4.2 表 1 顯示模擬能區分**啟發式**，但那並未檢驗**學習式**策略是否有效。為此，我們在注入 mean-preserving 對數常態 runtime 不確定性（σ=1.0，模擬 straggler 與預測誤差）的隨機模擬中，以固定溫度（fixed-α=0.05）、100k 步、3 個訓練 seed（42／43／44）分別訓練三個學習臂——純量 SAC、風險中立 RDSAC-mean、風險敏感 RDSAC-cvar——並以共用隨機數配對評估其相對 score 的 ΔJCT%（表 5）。

表 5. 模擬 σ=1.0 多 seed（3 訓練 seed × 100k 步，fixed-α=0.05）ΔJCT% vs score

| 學習臂 | philly ΔJCT% | ali ΔJCT% | 完成率 | 穩定性 |
|---|--:|--:|--:|---|
| SAC（純量 twin-Q） | −8.6±13.4 | −17.2±21.8 | 全 seed 100% | 穩定，學習臂中最佳 |
| RDSAC-cvar（風險敏感） | −24.2±16.4 | −48.9±43.5 | 全 seed 100% | 穩定，但一致落後 score 與 SAC |
| RDSAC-mean（分布式） | +70.1† | +89.0† | 0–80%（半數 seed 崩潰） | 雙峰：偶佳，常崩成 0% |

†RDSAC-mean 的正值為**低完成率假象**：ΔJCT% 僅在已完成的工作上配對計算，而該策略於半數 seed 崩潰為 0% 完成（退化為放棄困難工作的 no-op），故其表面優勢並非真實改善。

此結果有三點意涵。第一，**單一 seed 的模擬結果會嚴重誤導**：同一 RDSAC 設定在單 seed 下曾呈現大幅領先 score，但於多 seed 下該領先不可重現——那不過是抽到未崩潰的幸運 seed。第二，**即使在為風險機制量身打造的隨機模擬中，風險敏感 DRL 於多 seed 下亦未穩健勝過純量 SAC 或 score 啟發式**；連此「模擬救援」都失敗，與 §4.2 的實機負結論方向一致（三個訓練 seed 的 RDSAC-cvar ΔJCT% 在兩個 trace 上**一致為負**，非隨機雜訊）。

第三，也是本消融唯一站得住的**正面**發現：**CVaR 風險扭曲的實質作用是穩定訓練／完成率，而非提升速度**。表 6 逐格列出兩個分布式臂的完成率：風險中立的 RDSAC-mean 在 6 個（trace×seed）格中有 4 格崩潰（完成率 ≤20%，退化為 no-op），而加了 CVaR 尾端扭曲的 RDSAC-cvar **6 格全部 100% 完成、無一崩潰**。這指向一個可推廣的觀察——**風險敏感性可作為分布式評論家在環境隨機性下對抗策略崩潰的穩定器**，即使它並未帶來 JCT 上的優勢。

表 6. 兩個分布式臂各 (trace×seed) 完成率——CVaR 消除崩潰

| 訓練 seed | RDSAC-mean（philly／ali） | RDSAC-cvar（philly／ali） |
|---|--:|--:|
| 42 | 80%／100% | 100%／100% |
| 43 | 0%／0% | 100%／100% |
| 44 | 0%／20% | 100%／100% |

### 4.4 與雲端原生 SOTA 基準的對照（強化基準）

§4.2 的區分實驗僅對照自家啟發式，可能招致「未與引用的 SOTA 比較」之質疑。為此，我們將 §2 所述的兩個雲端原生排程器近似納入同一模擬對照——Kueue 式 fair-share（跨使用者 max-min 交錯）與 Volcano 式 binpack（最大需求優先）——在高競爭的 1×1、**佇列飽和** regime（offered load 拉高至系統飽和；GPU 使用率仍約 0.6，屬**佇列**飽和而非**算力**飽和，aiserve 工作負載，8 seed）下量測（表 7）。此處的拓樸（1×1）與競爭度（飽和）皆與表 1（2×1、ρ≈0.7 中度競爭）不同，故 JCT 絕對值明顯較高（如 FCFS 2640 vs 表 1 的 2199、score 1887 vs 1129）；兩表各自檢驗其 regime 內的**相對排名**，跨表的絕對 JCT 不宜直接相減。

表 7. 強化基準：雲端原生 SOTA 近似納入模擬對照（aiserve，8 seed，1×1 佇列飽和）

| 排程器 | 平均 JCT (s) | SLO 違反 (%) | 使用率 |
|---|--:|--:|--:|
| FCFS | 2640 | 59.7 | 0.59 |
| Volcano-binpack | 2515 | 45.3 | 0.60 |
| score（生產） | 1887 | 40.1 | 0.60 |
| multifactor | 1720 | 38.8 | 0.60 |
| Kueue-fairshare | 1722 | 38.8 | 0.60 |

此結果有兩點意涵。第一，**區分確實存在但邊界清楚**：FCFS 與純 binpack（優先塞入大型訓練工作、延誤延遲敏感的推論）明顯較差，而 fair-share／multifactor／score 三者叢聚於同一最佳帶（SLO 違反約 39–40%）。第二，**引用的 SOTA 近似並未勝過生產 score**——Kueue-fairshare 與 multifactor 打平、score 落在同帶。這把「合理排程策略空間狹窄」的結論從自家啟發式延伸到雲端原生 SOTA，回應了「只比自家 heuristic」的質疑（實作為 sim 內排序近似，非完整 Kueue 准入控制器／Volcano 節點評分外掛，見 §4.6 威脅）。

### 4.5 討論

上述結果指向一致的洞見：**在 2×1 規模，整個排程策略空間近乎是平的**——不僅深度強化學習未能穩健勝過啟發式（實機統計打平、模擬多 seed 亦未勝出），連生產 score 對 FCFS 等簡單基準亦僅打平。就實機聚合（表 3）而言，各啟發式相對 score 的 ΔJCT% 落在約 ±0.4～1.7%、且信賴區間跨越 0，在 ±5% 的實務等價界（practical-equivalence margin）內可視為**統計等價**（其中 FCFS 因變異較大而區間較寬，等價宣稱較弱）；換言之這是「證實無實務差異」，而非僅「未偵測到差異」。此 ±5% 等價界同樣涵蓋學習型策略：表 4 中三個學習臂的 −3.7～−4.6% 雖因大樣本而**統計顯著**，其幅度仍落在等價帶內，故整個「策略空間近乎是平的」判斷橫跨啟發式與學習式兩類排程器，而非僅指前者。

**關於「規模」的誠實界定。** 本研究據此**推測**排程策略的差異需要更大規模或更高競爭方能顯現，但必須強調：這是**尚未被證實的假設，而非本研究的結果**。我們的初步規模掃描（1×1／2×1／2×2，§4.3 之外的補充實驗）**並未**呈現「效益隨規模上升」的交叉趨勢（圖 2）：學習臂相對 score 的 ΔJCT% 在各規模皆為負、且隨規模非單調（2×1 反而最接近 score，2×2 又拉開），並無朝 0 收斂的跡象；RDSAC-cvar 在 1×1 甚至崩潰為 0% 完成。此掃描受限於較低訓練預算（40k 步）與跨尺度觀測空間不可直接比較，尚不足以支持或否證此假設。

![圖 2](figures/fig_scale.png)

圖 2. 規模掃描（σ=1.0、40k 步）下學習臂相對 score 的 ΔJCT%。若「效益隨規模浮現」成立，曲線應隨規模趨近 0（score 基準）；實測反而全程為負且非單調，故此假設未獲支持（受訓練預算與跨尺度不可比之限制，僅作為 open question 的方向性證據，見 §4.6）。

將「效益隨規模浮現」從斷言降級為 open question，正是本研究誠實立場的一部分。此外，§4.3 揭示了一個方法學教訓：**單 seed 的模擬評估可能報告不可重現的假性優勢**，凸顯多 seed 配對統計的必要性——這與 §4.2 中「單趟平均值會得到錯誤排名」互為印證。

### 4.6 有效性威脅（Threats to Validity）

作為一篇以方法學與誠實負結論為主軸的研究，本節明列可能削弱結論的因素及其處置：

- **單 seed 脆弱性（已處置）：** 早期單 seed 結果曾呈現 RDSAC 大幅領先，經 3 訓練 seed 重跑後被推翻（§4.3）。所有學習型結論均以多 seed mean±std 呈現。
- **訓練預算 confound（部分處置）：** 決定性的 CVaR 消融已用 100k 步（§4.3）；但規模掃描仍為 40k 步，較複雜的 IQN 評論家可能欠訓練，故規模結論僅作為 open question，不作定論。
- **跨尺度不可比：** 1×1／2×1／2×2 的觀測維度與動作空間不同，checkpoint 不相容、須各自重訓，跨尺度的絕對數值不宜直接相減。
- **合成 trace：** 訓練與模擬工作負載為依 Philly／Alibaba 公開統計特性合成（非原始逐筆），且其 runtime 與特徵獨立，屬預測性最差的保守情境。
- **模擬與實機落差：** 模擬以離散事件近似 MPS 干擾與 runtime 不確定性，未完整建模快取、記憶體頻寬與 kernel 級競爭。
- **評估與部署路徑差異：** 實機放置實驗以顯式 `srun` 綁定放置，而生產路徑中 RL 僅設定佇列優先權、放置仍由 Slurm 執行；兩者不完全等同。
- **小樣本：** 實機單一工作即耗時數分鐘，樣本稀少；已以配對 CRN、抗漂移輪轉與多 seed 盡量降低變異，但統計檢力仍受限。
- **規模上限：** 最大僅測至 2×2，結論不宜外推至數十至數百 GPU 的生產叢集。
- **SOTA 近似（§4.4）：** Kueue／Volcano 對照為**排序層級**的近似（fair-share max-min／binpack 最大需求優先），非完整的 Kueue 准入控制器或 Volcano 節點評分外掛；等價結論限於排序策略層級，未涵蓋配額借還、gang 准入等機制。

## 5. 結論與未來工作

本研究設計並實作了一套以 Kubernetes 部署、Slurm 為核心、整合 MPS 與失效安全 RL 決策的 AI 伺服器 GPU 排程平台，並提出一套兼顧抗漂移、多 seed 配對統計與尾端指標的模擬到實機評估方法學。實測誠實顯示在 2×1 異質叢集上排程策略統計等價，且風險敏感 DRL 即使在注入不確定性的模擬中、經多 seed 檢驗亦未勝出；我們將「智慧排程的效益是否需要更大規模方能顯現」明確界定為**尚待驗證的假設**——本研究的 1×1–2×2 掃描並未證實其交叉。唯一站得住的正面發現是 CVaR 風險扭曲可作為分布式評論家的完成率穩定器。未來工作包含：（1）對照更強的基準——Slurm 原生 `gres/shard`＋backfill＋multifactor，以及模擬中的 Kueue 式 fair-share／Volcano 式 binpack——以鞏固「等價」結論；（2）擴展至更大、更高競爭的叢集以檢驗規模假設；（3）強化分布式評論家的訓練穩定性、克服其在不確定性下的崩潰／退化問題；（4）以線上 RLPD 微調縮小模擬與實機落差。

## 致謝

（依大會格式填寫，如計畫編號、單位支持等。）

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

[15] Kubernetes Authors, "Dynamic Resource Allocation (DRA)," Kubernetes Documentation (GA in v1.34), https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/, 2025.

[16] J. Duan, Y. Guan, S. E. Li, Y. Ren, Q. Sun, and B. Cheng, "Distributional Soft Actor-Critic: Off-Policy Reinforcement Learning for Addressing Value Estimation Errors," *IEEE Transactions on Neural Networks and Learning Systems*, 2021. arXiv:2004.14547.
