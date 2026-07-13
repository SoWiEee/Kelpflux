# 基於 Slurm 與 Kubernetes 架構下 AI 伺服器 GPU 工作負載智慧排程技術之研究

### Intelligent GPU Workload Scheduling Techniques for AI Servers under a Slurm-on-Kubernetes Architecture

**作者一¹、作者二²**
¹○○大學 ○○系　²○○大學 ○○系
{author1, author2}@stumail.nutn.edu.tw

---

## 摘要

一般大學實驗室常見多人共用少量 GPU 主機的情境：資源閒置與利用率不足並存、批次工作排隊與管理繁瑣，叢集內 GPU 世代混雜（算力可差數倍），且須同時容納低延遲推論與長時間訓練兩類性質迥異的負載，使排程決策更為複雜。

本研究以此為背景，建構一套 Slurm-on-Kubernetes 平台：以 NVIDIA MPS 達成單張 GPU 內的多工作共享，交由 Kubernetes 負責容器化部署、健康監控與（視需要）節點擴縮等生命週期管理，並在 Slurm 的 job 提交路徑上整合一個失效安全（fail-safe）的強化學習決策端點——RDSAC（discrete 分布式 SAC，搭配雙頭 IQN 評論家與 CVaR 風險敏感優化）——服務逾時或異常時靜默回退既有啟發式排程（score）。為誠實檢驗此類學習式策略的真實效益，本研究另提出一套抗跑序漂移、多 seed 配對統計、於同一 GPU 分配後端統一重測的 sim-to-real 評估方法學。

以此方法學在同一乾淨的 Kubernetes 動態資源分配（DRA）後端上，對三個真實硬體場景——低負載 cuBLAS 共置、高負載真實 LLM serving、SLO serving——並跨平均工作完成時間（JCT）與服務水準目標（SLO）違反率兩類指標統一重測後：學習式放置策略在三個場景中皆未穩健勝過生產啟發式 score，亦未勝過未加智慧放置的原生 Slurm；唯一穩健的正面發現，是 score 啟發式在高負載下確實勝過原生 Slurm 的 cons_tres 放置。本研究的貢獻因此定位為一套可重現的失效安全 RL 排程整合架構，以及一套能誠實揭露「排程結論高度依賴評估場景與底層 GPU 分配後端」的 sim-to-real 評估方法學，而非宣稱學習式排程必然優越。

未來工作包含：在更大規模、更高競爭的叢集上重新檢驗排程效益是否隨規模浮現的假設；並將 RDSAC 接成 Kubernetes 原生排程 policy（例如 Kueue 的准入排序或 DRA 的裝置選擇外掛）的概念驗證，把「學習式策略層與 DRA 機制層互補」的論述落實為可運行的雛型。

**關鍵詞**：GPU 資源排程、Kubernetes、Slurm、深度強化學習、sim-to-real 評估方法學。

## Abstract

Shared GPU hosts in a typical university lab commonly face idle resources coexisting with low utilization, cumbersome batch-job management, heterogeneous GPU generations differing in compute by several factors, and a mix of latency-sensitive inference and long-running training workloads that together complicate scheduling.

Motivated by this, we build a Slurm-on-Kubernetes platform: NVIDIA MPS lets multiple jobs share a single GPU, Kubernetes handles containerized deployment, health monitoring, and (where needed) node scaling as part of lifecycle management, and a fail-safe reinforcement-learning decision endpoint — RDSAC, a discrete distributional Soft Actor-Critic with a dual-head IQN critic and CVaR risk-sensitive optimization — is integrated into Slurm's job-submission path, silently falling back to the existing score heuristic on timeout or failure. To honestly assess the real-world benefit of such learned policies, we further propose a sim-to-real evaluation methodology built on drift-robust interleaving, multi-seed paired statistics, and unified re-testing on a single GPU-allocation backend.

Using this methodology, we re-tested three real-hardware scenarios — low-load cuBLAS co-location, high-load real LLM serving, and SLO-oriented serving — on the same clean Kubernetes Dynamic Resource Allocation (DRA) backend, across both mean job completion time (JCT) and SLO-violation-rate metrics. Learned placement policies failed to robustly beat the production score heuristic in any of the three scenarios, and also failed to beat naive Slurm without intelligent placement. The only robust positive finding is that the score heuristic outperforms naive Slurm's cons_tres placement under high load. We therefore position this work's contribution as a reproducible fail-safe RL integration architecture together with an evaluation methodology that honestly exposes how scheduling conclusions depend heavily on the evaluation scenario and the underlying GPU-allocation backend — rather than a claim that learned scheduling is superior.

Future work includes re-examining whether scheduling benefits emerge at larger scale and higher contention on a bigger cluster, and building a proof-of-concept that wires RDSAC into a Kubernetes-native scheduling policy (e.g., Kueue admission ordering or a DRA device-selection plugin), turning the "learned policy layer complements the DRA mechanism layer" argument into a working prototype.

---

## 1. 緒論

生成式 AI 與深度學習的普及，使學術與企業的 GPU 叢集需同時承載**低延遲推論**與**長時間訓練**兩類性質迥異的工作負載：前者須在服務水準目標（SLO）期限內完成、且常僅需部分 GPU 算力，後者則長時間獨佔整張卡。當底層硬體有限且**異質**——不同世代 GPU 的算力差異可達數倍——如何將兩類工作妥善排程與放置，以兼顧工作完成時間（JCT）、資源使用率與 SLO 達成，已成為雲端與邊緣運算基礎設施的核心議題 [1][4]。

傳統 HPC 排程器（如 Slurm）以靜態優先權與回填（backfill）為主，對 GPU 卡內共享與工作負載特性的感知有限。學界雖已提出多種以強化學習（RL）優化叢集排程的方法 [2][3]，但多數僅止於模擬評估，**鮮少在真實叢集上以嚴謹的統計方法檢驗其效益是否成立**；尤其「模擬中可區分的策略差異能否轉移到實機」這一 sim-to-real 問題，至今缺乏系統性的探討。實機評估之所以困難，在於單一節點即承載一個跑數分鐘至數小時的工作，使可用樣本極度稀少，且量測易受叢集暖機等時變因素干擾而產生假性排名。

針對上述缺口，本研究以一座雙節點異質叢集（RTX 4070＋RTX 3080）為實驗平台，建構從模擬訓練到實機驗證的完整流程，並以抗漂移、多 seed 配對統計的方法學誠實檢驗排程策略的真實效益。

### 1.1 高效能運算排程器 Slurm

Slurm Workload Manager 是目前高效能運算（HPC）叢集最廣泛使用的工作排程系統之一。它提供完整的工作提交、佇列管理、資源限額與帳務管理功能，並支援 GPU 資源分配。其以「節點」為基礎的資源模型非常適合研究工作環境，且已有廣泛的使用者社群與文件支援。然而 Slurm 原生設計以固定實體節點為主，對彈性擴縮與容器化整合的支援相對有限。

### 1.2 容器化與 Kubernetes

Kubernetes 是目前最主流的容器編排平台，能夠自動管理容器的部署、擴縮與健康監控。Kubernetes 的 Cluster Autoscaler 可根據工作負載自動增減節點，非常適合需要彈性資源的場景。然而 Kubernetes 原生的排程器 (kube-scheduler) 以服務導向設計為主，對批次工作、GPU 共享與研究工作特有的排程需求支援不足。因此有多項研究嘗試將 Slurm 與 Kubernetes 整合，以結合兩者優點。在 Slurm 與 Kubernetes 整合方面，既有研究已指出高效能運算與雲端原生系統的匯流可以同時取得批次排程語意與雲端彈性管理能力；實作層面也有 Slinky 等 Slurm-on-Kubernetes 方向的工具，以及 AWS ParallelCluster 等雲端 HPC 叢集部署方案，顯示此類架構已具有實務需求與發展基礎。

### 1.3	GPU 資源共享技術

NVIDIA 提供多種 GPU 資源共享技術，包括 Time-Slicing、Multi-Process Service (MPS) 以及 Multi-Instance GPU (MIG)。MPS 允許多個 CUDA 程式同時共享一張 GPU 的運算資源；MIG 則在硬體層面將 GPU 切分成獨立的分區，提供更強的隔離性。本系統目前採用基於 MPS 的方式，讓多個較小的工作共享同一張 GPU，以提升整體使用效率。

除了本系統採用的 MPS 之外，近年也有研究探討在 MIG 等機制上進行動態重新分割與能源效率最佳化 [21]；這類方法能提供較強隔離性，但也需要額外硬體支援、分割粒度與工作遷移成本，因此本專題目前先以部署門檻較低的 MPS 作為共享 GPU 的主要實作方式


## 2. 相關研究

**GPU 叢集排程與分析。** Jeon 等人對微軟 Philly 叢集的大規模多租戶 GPU 工作負載進行分析 [1]，揭示了排隊延遲與資源碎裂問題。Gandiva [2] 與 Tiresias [3] 分別利用 DL 工作的可遷移性與分布感知排程降低 JCT。Weng 等人對阿里巴巴 PAI 叢集的研究 [4] 指出生產 MLaaS 工作高度分片化、以單卡短工作為主。本研究的合成工作負載即以 [1][4] 的統計特性為依據。

近期研究則更聚焦於單節點內或單一叢集模型上的動態資源調度本身。Wang 等人的 DCUDA [19] 針對單節點多 GPU 情境，設計了一套輕量級核心／記憶體使用率監控機制，搭配近乎零開銷的「執行中」CUDA 應用即時遷移，將 GPU 過載時間平均降低 78.3%、一般工作執行時間降低 42.1%（記憶體密集型工作最高 67%）。Sedighi 等人 [20] 則在 Alibaba 的 cluster-trace-gpu 生產工作負載軌跡之上，提出結合硬體與軟體分割的公平且需求感知動態資源配置演算法，於模擬環境中將 GPU 資源使用量降低達 88%。這兩項工作皆聚焦「資源配置本身如何隨工作負載動態調整」（即時遷移／再分割），評估分別侷限於單一多 GPU 節點與純模擬 trace 重放，並未涉及與批次排程器（如 Slurm）的整合或真實異質叢集上的統計驗證；本研究的排程決策則作用於 Slurm 的工作提交路徑之上、決定「哪個工作該去哪張卡的哪個 MPS 槽」，並在真實異質叢集上以失效安全與抗漂移統計方法落地檢驗，兩者的問題設定屬於不同但互補的抽象層次。

**GPU 共享。** NVIDIA MPS [5] 允許多個行程共享單張 GPU 的計算資源，是在小型叢集上提升使用率的關鍵機制；本研究以 MPS 槽（每卡 4 槽，對應 25／50／75／100％）建模卡內共享。另一條路線聚焦硬體隔離式的 MIG 動態重分割：Lipe 等人 [21] 針對單張 A100 GPU 的 MIG 切片，先以 Earliest-Deadline-First–Slowest-Slice（EDF-SS）演算法處理切片內的工作排程，再以強化學習（DQN）決定何時、要重分割成 12 種切片組態中的哪一種，於「能耗＋延誤」的多目標指標上優於雙日重分割（26%）、靜態分割（31%）與完全不分割（68%）。相較於 MIG 的硬體級隔離與較高的重分割／遷移成本，本研究採用 MPS 的理由正是部署門檻更低——無需重新配置硬體分區，即可在既有消費級 GPU（RTX 4070／3080，皆不支援 MIG）上即時生效卡內共享，這也是本研究能以校園實驗室既有異質硬體直接驗證的前提。

**強化學習排程。** 以 RL 進行資源排程的研究多採用 actor–critic 或值函數方法。本研究的決策核心 RDSAC 為三項技術的整合：離散動作 Soft Actor-Critic [6]、Implicit Quantile Network 分布式評論家 [7]，以及以 CVaR 為風險量度的尾端敏感優化 [7]。為弭平模擬與實機落差，亦採用離線到線上的 RLPD 微調概念 [8]。獎勵塑形採用保證最優策略不變的位能塑形 [9]。

以 RL 排程 AI 服務型任務的研究路線與本研究最為鄰近。Lin 等人 [23] 提出 UXP-RL：一個以 DQN 為核心、涵蓋前處理／訓練／推論三類任務、可部署為集中式或分散式排程器、並跨雲／邊／霧三層架構運作的 CPU-GPU 任務排程演算法。其於模擬環境中，集中式排程器將平均週轉時間相較 SJF／FCFS 與 TYPE 啟發式（依 GPU 需求高低分類任務）分別降低 57.81％、57.28％與 27.66％；分散式排程器則因能將長訓練工作卸載至雲端而釋放邊／霧資源，把推論任務週轉時間相較集中式再降低 89.07%。同屬 2025 年的近作中，Zhang 等人的 KIS-S [25] 以 PPO 訓練一個 GPU-aware 的 Kubernetes 推論自動擴縮策略（KIScaler），完全於自建模擬器（KISim）中訓練後即零樣本部署，於多種流量情境下平均獎勵提升 75.2%、p95 延遲相較 CPU 基準降低最多 6.7 倍；其問題設定是**調整副本數**的自動擴縮，而非本研究的**工作放置**排程。Wu 等人的 DRR [26]（ACM SoCC ’25）則針對 GPU 共享叢集因分享機制、工作異質性與非同步生命週期造成的碎片化問題，以模仿學習（imitation learning）從既有啟發式暖啟動一個深度強化學習去碎片化排程器，並輔以多尺度策略最佳化平衡探索與利用；其同時於實體 Kubernetes 測試床與大規模模擬叢集上驗證，平均碎片率降低 50%——是本節所列文獻中少數同時涉及真實 Kubernetes 部署的學習式排程器。

**與 [23]（RL for AIaaS）的定位差異。** [23] 與本研究同屬「以強化學習排程 AI 服務型任務」的問題設定，是最貼近本研究、也最可能被質疑「novelty 重疊」的對照組，值得正面處理其區隔：（1）**評估場所與統計嚴謹性**——[23] 完全於自建模擬環境中，以合成任務到達率與 17 個 DNN 模型的離線量測執行時間為輸入評估，並未涉及真實叢集部署，也未處理 sim-to-real 落差、叢集暖機漂移等真實硬體量測特有的混淆因子；本研究的核心方法學貢獻正是把「模擬中可分辨的策略差異能否轉移到實機」系統性地檢驗——以抗漂移交錯輪轉、多 seed 配對統計（seed 層級 one-sample t 檢定）、同一 GPU 分配後端統一重測，並誠實回報「差異不轉移」與「後端本身混淆結論」兩項負結果（§5.3）。（2）**目標函數**——UXP-RL 以最小化平均週轉時間（排隊＋執行）為單一目標；本研究的 RDSAC 是**風險敏感**的：雙頭 IQN 評論家對回報分布套用 CVaR 扭曲，直接優化 p95／p99／SLO 違反率等尾端量，因為 AI serving 情境下「多數請求正常、少數被拖很慢」的尾端體感往往比平均值更貼近使用者實際感受（§5.2）。（3）**失效安全的生產整合**——[23] 的 RL 排程器是排程決策的唯一來源；本研究將 RL 決策整合進 Slurm 既有的 `job_submit.lua` 提交路徑，服務逾時或異常時**靜默回退**至既有 score 啟發式，確保排程核心（slurmctld）永不被研究用元件阻塞——這是把學習式排程放進生產路徑必須解決、但 [23] 未觸及的工程問題（§3.3）。（4）**誠實的負結論**——本研究誠實揭露：在乾淨統一的 DRA 後端上，RDSAC／SAC 等學習式放置在三個真實硬體場景（低負載共置、高負載真實 LLM serving、SLO serving）皆未穩健勝過生產 score 啟發式（§5.3），這與 [23] 及多數既有 RL 排程文獻報告的正面結果形成對比。本文將此差異本身視為方法學貢獻的一部分：RL 排程效益高度依賴評估場景與統計嚴謹程度，一個只在模擬中量測、未做多 seed 配對顯著性檢定的正面結果，未必能在真實部署中複現。DRR [26] 雖已跨出模擬、於真實 Kubernetes 測試床驗證，但其目標仍是聚合碎片率而非尾端 SLO，亦未見多 seed 配對統計或抗漂移設計；本研究的貢獻正補上這塊空缺——把「學習式排程在真實叢集上是否穩健勝過生產基準」的問題，以統計嚴謹的方法學正面回答（即使答案是誠實的「尚未」）。

**異質／邊緣 GPU 排程。** 本研究的叢集本身即異質且非資料中心等級（RTX 4070＋RTX 3080，皆不支援 MIG／vGPU），這使邊緣與異質 GPU 排程文獻格外相關。Tsenos 與 Kalogeraki [22] 針對缺乏原生虛擬化支援的邊緣 GPU（如 RTX 4090、GTX 1080Ti 等消費級卡）提出一套硬體無關的時空共享機制：為每個行程建立 cgroup、動態調整其「duty cycle」（週期性凍結／解凍佔用 GPU 的時間比例）來實現優先權式與截止期限（laxity）式排程，且無需修改工作負載原始碼即可整合進 TensorFlow、PyTorch、FFmpeg 等既有框架。此工作與本研究處境相近——皆是非資料中心等級、不支援硬體分片的消費級 GPU——但其排程單位是**單節點**上的行程級 duty cycle 調整，不涉及叢集級的佇列、backfill 或跨節點放置決策，亦未整合學習式策略，可視為與本研究互補的節點內機制。Majeed 等人 [24] 則以系統性文獻回顧整理 NVIDIA Jetson AGX 系列邊緣 SoC（CPU＋GPU＋深度學習加速器 DLA＋可程式視覺加速器 PVA＋視訊影像合成器 VIC）上的 DNN 排程器，區分規則式（如 Jedi、CP-CNN、Herald、H2H、HaX-CoNN）與最佳化式（線性規劃、AxoNN、遺傳演算法、以 Z3 SMT 求解器動態重排的 D-HaX-CoNN）兩大類，並整理其記憶體競爭、跨加速器轉移成本與靜態／動態排程的權衡。其排程粒度是**單一 DNN 模型內的層級**（將個別網路層指派給不同硬體加速器），與本研究**工作／任務級**的叢集放置決策不在同一抽象層次，可作為異質邊緣排程景觀（landscape）的引用，界定本研究「叢集批次排程」與此類「模型內加速器排程」研究之間的分工。

**雲端原生 GPU 排程生態系。** 在 Kubernetes 生態中，數個成熟系統處理 GPU／批次工作負載，但分屬不同抽象層：Kubeflow [12] 負責 ML 工作的生命週期（分散式訓練、超參數搜尋、模型服務），其本身不做排程，而將決策**委派**給批次排程器；Volcano [13] 提供 pod 群組的 gang 排程與 DRF／binpack 等規則式外掛；Kueue [14] 實作 job 級佇列、配額借還（ClusterQueue／ResourceFlavor／Cohort）、fair-share 與 gang 准入，但以暫停（suspend）控制准入、**不負責 pod 放置**；Kubernetes 1.34 起正式釋出（GA）的動態資源分配（Dynamic Resource Allocation, DRA）[15] 則將 GPU 分片、MIG、time-slicing 等以 ResourceClaim／ResourceSlice／DeviceClass **宣告式**地納入 API，成為一等公民。表 1 依抽象層整理這些系統與本研究的定位。

表 1. 雲端原生 GPU 排程系統的層級定位

| 系統 | 所在層 | 機制形態 | 學習式策略 | 尾端／SLO 目標 | GPU 分片 |
|---|---|---|:--:|:--:|:--:|
| Kubeflow [12] | 工作生命週期 | 委派給批次排程器 | ✗ | ✗ | 委派 |
| Volcano [13] | Pod 群組排程 | 規則式 heuristic（gang/DRF/binpack） | ✗ | ✗ | time-slice（無策略） |
| Kueue [14] | Job 級佇列／配額 | 規則式＋約束求解（不做放置） | ✗ | ✗（fair-share 非尾端） | ResourceFlavor 標記 |
| K8s DRA [15] | 裝置分配**機制** | 約束匹配 | ✗ | ✗ | ✓（宣告式一等公民） |
| **本研究（RDSAC）** | 排序／放置**策略** | **學習式＋風險敏感（CVaR）** | ✓ | ✓（直接優化尾端） | MPS-aware 策略 |

**與既有工作的差異。** 上述系統皆為**規則式或約束求解**：Kueue／Volcano 解的是配額與 gang 准入、DRA 提供的是分片**機制**、Kubeflow 管的是生命週期；沒有任何一個是「**學習式、且以尾端延遲（tail latency）為目標的排序／放置策略**」。本研究正落於此空隙：RDSAC 對回報分布以 CVaR 優化 p99／SLO 尾端，是生產系統皆未優化的量。要強調的是，DRA 提供的是「如何表達要 0.25 張 GPU」的*機制*，而非「該把哪些工作打包、用什麼順序以壓低尾端」的*策略*——因此本研究的學習式策略與 DRA 並非競爭，而是**互補**：一個尾端敏感的策略可在 DRA 之上驅動裝置選擇與准入排序。既有 RL 排程研究多止於模擬；本研究的重點不在宣稱 RL 必勝，而在**建立一套能在真實異質叢集上、以統計嚴謹方式檢驗排程策略效益的方法學**，並誠實回報其規模條件。

值得一提，本節所引 6 篇 IEEE 相關研究中，多數（DCUDA [19]、Sedighi 等人 [20]、MIG 動態重分割 [21]、邊緣 duty-cycle 排程 [22]、UXP-RL [23]、Jetson 排程回顧 [24]）皆作用於單節點執行期、或評估侷限於未與 Kubernetes 整合的獨立模擬環境；僅 KIS-S [25] 與 DRR [26] 是 Kubernetes 原生系統，但分別位於**自動擴縮**（依流量調整推論副本數）與**去碎片化重排程**這兩個子層，皆非表 1 所比較的「排序／放置策略」層級。這進一步凸顯本研究在雲端原生 GPU 排程生態系中的定位空隙：一個作用於 Slurm-on-Kubernetes 排程／放置層、以學習式策略直接優化尾端 SLO 的失效安全整合，其鄰近文獻或止步於單節點／純模擬（[19]–[24]），或雖已部署於 Kubernetes 但作用於相鄰子層（[25][26]），未見與本研究直接重疊者。

### 2.3 Soft Actor Critic (SAC)

> TBA。原始論文的簡介。


### 2.4 風險敏感深度強化學習 (RDSAC)

決策策略為自行整合的 **discrete 分布式 SAC**（本文稱 RDSAC）：雙頭 IQN 評論家分別建模回報分布（reward 回報 $Z_R$ 與 entropy 回報 $Z_H$），以 quantile Huber loss 學習，搭配 twin-Q、軟更新（τ=0.005）與遮罩式 categorical actor。風險敏感性透過在 actor 目標與動作價值上對回報分布套用 CVaR 扭曲 $\rho[Z_R]$ 達成，對應排程中的長尾 runtime／慢節點（straggler）風險。訓練採優先經驗回放（PER）、n-step 回報、分數暖啟動與位能獎勵塑形。RDSAC「風險敏感分布式 SAC」之名承襲自 Ma 等人以回報分布做風險敏感優化的 DSAC [16]；惟本研究為**離散動作**、雙頭 IQN 的自組版本，是離散 SAC [6]、IQN 分布式評論家 [7] 與 CVaR 風險量度的組合，非 [16] 連續控制版本的 1:1 重現。須留意另有**同名但不同**的 Duan 等人 DSAC [17]（將回報建模為單一高斯、以抑制 Q 值高估為目標、風險中立、連續控制），與本研究的風險敏感取向不同，不宜混淆。

## 3. 研究目的與系統架構

### 3.1 研究目的

本研究主要探討以下幾個問題：

- 如何讓多個較小的 GPU 工作共享同一張顯示卡，以提升使用效率。
- 如何利用深度強化學習模型，協助系統決定哪些工作應優先執行，以及應使用哪些硬體資源。
- 如何在模型判斷不可靠時，讓系統自動回到較穩定的基本排程方式。

### 3.2 整體架構

平台分為兩個鬆耦合層：**基礎設施層**（Slurm on Kubernetes）與**排程研究層**（模擬器＋深度強化學習）。

叢集以 k3s 部署，控制節點（RTX 4070）兼任 control-plane，工作節點（RTX 3080，算力約前者 0.25×）為異質 GPU 來源。Slurm 控制器、登入節點與 GPU 工作節點皆以容器化 StatefulSet 部署，GPU 經 NVIDIA device plugin 與 MPS control daemon 暴露為可分片資源。

**為何 Slurm-on-Kubernetes（而非純 K8s 排程器）。** 一個合理的質疑是「既然已在 K8s 上，為何不直接用 Kueue＋DRA＋自訂 kube-scheduler plugin，而要疊一層 Slurm？」本研究選擇 Slurm-on-K8s 有三個理由：（1）**成熟的 HPC 排程語意**——backfill、multifactor 優先權、gang、`gres/mps` 卡內分片皆是 Slurm 開箱即用且經生產驗證的一等公民；在 K8s 側要湊齊等價能力需 Kueue＋Volcano＋DRA 多元件拼裝，且 DRA 於本研究進行時甫 GA（K8s 1.34），生態未穩。（2）**K8s 提供部署與生命週期、Slurm 提供排程核心**，兩層鬆耦合、各司其職：k3s 負責容器化、網路、儲存（NFS RWX）、可觀測性，Slurm 負責佇列與放置決策；這讓平台既可攜（Helm 一鍵部署於異質節點）又保有 HPC 級排程。（3）**研究載具**——`job_submit.lua`／slurmrestd 是穩定、非侵入的策略注入點（§3.2），可在**不 fork slurmctld、不改 kube-scheduler** 的前提下熱插拔學習式策略並失效即回退；相較於維護一個自訂 scheduler plugin，這大幅降低了研究迭代成本。與生態的關係上，本研究的學習式策略與 K8s DRA **互補**（§2）：DRA 給的是分片*機制*，本研究給的是尾端敏感的排序／放置*策略*，未來可在 DRA 之上驅動裝置選擇（§6）。

### 3.3 失效安全的 RL 整合

整合點為 Slurm 的 `job_submit.lua`：工作提交時，掛鉤 `rl_hook.lua` 以 HTTP 呼叫 RL 推論服務的 `POST /decide`，取得放置／優先權建議；若服務逾時或異常，掛鉤**靜默回退**至既有啟發式分數排程（以 MPS 適配、VRAM 適配為主，短工作優先因子在生產環境預設關閉，細節見 §4.2）。此設計確保 slurmctld 永不被第三方服務阻塞，使研究用的 RL 元件可安全運行於生產路徑。

生產部署的 RL 作動有兩條路徑，皆為失效安全（fail-safe）：（1）**優先權微調**——`job_submit.lua` 呼叫 `/decide`，僅提升被選中工作的佇列優先權（`select/cons_tres` 與 GRES 仍決定實際落點）；（2）**顯式節點綁定（explicit placement）**——`/act` 回傳節點選擇 `(node_j, gpu_k)`，於**提交時**將該節點寫入工作的必要節點（`ReqNodeList`），Slurm 遂將工作排到 RL 選定節點、MPS 由 GRES 強制。本研究的實機放置實驗（§5.3）即以路徑 (2) 為對象——由評估 harness 呼叫 `/act` 後以 `sbatch -w` 提交——並於本平台 2×1 叢集驗證其正確釘選（`sbatch -w` 與 `scontrol update ReqNodeList` 皆能把 held／pending 工作釘到指定節點）。將此提交時綁定接入 `job_submit.lua`（呼叫 `/act` 後設 `job_desc.req_nodes`）即為 RL 顯式放置的生產路徑。平台另實作一個非同步 **placement controller**（`services/rl_scheduler/placement_controller.py`，以 slurmrestd hold→pin→release 對已提交 held 工作事後釘節點）作為不需改提交端的替代，惟其經 slurmrestd v0.0.37 job-update 寫入節點約束之實際生效仍在硬化中（實機測試觀察到該 REST 欄位未被套用），故**目前評估與生產皆以提交時節點綁定為準**。`/act` 若 abstain（如 checkpoint 拓樸 ≠ 實機）則 no-op，退回 Slurm 原生放置。

### 3.4 模擬器與強化學習環境

離散事件模擬器以「提交／結束」事件驅動，建模 Node → GPU → MPS 槽的階層資源與異質算力。其上以 Gymnasium 介面封裝為 RL 環境：觀測為佇列前 K 個工作的特徵（GPU 數、MPS 需求、等待時間、SLO 緊迫度、工作類別、GPU 型別 one-hot 等），於 2×1 拓樸下維度為 166；動作為「放置於某節點某 MPS 槽」或「暫不放置」。

## 4. 排程技術

### 4.1 Slurm 內建排程演算法

本平台以 Slurm 原生排程能力作為穩定底座，而非重寫排程核心。`SchedulerType=sched/backfill` 允許在不延後高優先權工作的前提下，讓資源需求較小、執行時間較短的工作提前插隊執行，緩解大工作長期佔用資源造成的閒置；`PriorityType=priority/multifactor` 依工作年齡、規模、partition、QoS 等因子計算靜態優先權，決定佇列排序；`SelectType=select/cons_tres`（`CR_Core`）以 TRES（Trackable RESources）粒度追蹤 CPU／GPU／MPS 資源，並將 GPU 與 MPS 使用量計入 accounting（`gres/gpu`、`gres/mps`）。卡內資源切分透過 `gres/mps` 這個 GRES 型別表達：單一 GPU 依 MPS 槽（每卡 4 槽，對應 25／50／75／100％）分割給多個工作共享，由 Slurm 的 GRES 機制強制配置與計費。

本研究以 backfill＋multifactor 這組 Slurm 原生設定作為評估基準之一（表 3–5、表 6–8 的 `backfill`／`multifactor` 臂），並另納入更保守的 `sched/builtin`＋`priority/basic`（嚴格 FIFO，即 `fcfs` 臂）作為「未加任何智慧排程」的下界對照，用以檢驗啟發式與學習式策略相對 Slurm 開箱即用能力是否確有加值（§5.3.2 觀察到的「score 亦勝過 vanilla Slurm」即源於此對照）。

### 4.2	啟發式排程策略（score）

score 是 submit-time 的加權線性啟發式，於 `job_submit.lua` 提交當下計算，分數越高代表該工作越值得優先排程；其值以 `scoreGain`（預設 `1000`）換算成 priority delta，疊加於 `priority/multifactor` 之上：

```text
score(J, P) = α·f_mps_fit(J, P) + β·f_vram_fit(J, P) + γ·f_topology(J, P)
            − δ·f_fragmentation(J, P) + ε·f_pred_runtime(J)

score = clamp(score, 0, 1)
priority_delta = round(scoreGain × score)
```

三個核心因子及 chart 預設係數如下表：

| 因子 | 意義 | 係數（預設） |
|---|---|---|
| `f_mps_fit` | 衡量工作 MPS 請求與單 GPU MPS 容量（預設 100）的配適程度——bin-pack 式的卡內裝箱，愈不浪費槽位者分數愈高 | α = 0.40 |
| `f_vram_fit` | 依工作 VRAM 需求（`vram-*g` constraint）挑選可容納的最小 VRAM tier，避免小顯存工作佔用大顯存卡 | β = 0.20 |
| `f_fragmentation` | 懲罰最容易留下 MPS 碎片的請求（`4x(1−x)`，x 為 MPS 佔比；x=50% 時碎片代價最高） | δ = 0.20（懲罰項） |
| `f_pred_runtime` | 依 runtime predictor 預測執行時間換算，SJF（Shortest-Job-First）式地讓短工作取得較高分數，需 predictor 服務可用 | ε = 0.00（預設關閉） |
| `f_topology` | 保留欄位，回傳中性值 0.5，不影響上線結果 | γ = 0.00 |

若 weight-tuner（以 UCB1 bandit 在離散 arm 空間中線上調整 `(α, δ, ε)`）啟用，`job_submit.lua` 會在 Lua plugin 載入時以 `GET /weights` 覆寫這三個係數；`β` 恆固定，`γ` 維持 chart 設定不受 tuner 影響。

**生產部署現況（誠實揭露）。** 目前實際上線的係數為 α=0.40、β=0.20、δ=0.20，但 **γ=0、ε=0**——SJF 短工作優先因子在生產環境**未啟用**，score 是 runtime-blind 的；同時 runtime predictor 與 weight-tuner 服務**皆未部署**於生產叢集，三係數為 chart 靜態預設值，並非 UCB1 動態調校。故生產 score 實際上僅為 `clamp(0.4·f_mps_fit + 0.2·f_vram_fit − 0.2·f_fragmentation, 0, 1)`——一個純粹依 MPS／VRAM 配適與碎片懲罰運作的裝箱式啟發式，這也是全文「score 最穩健」核心賣點的具體機制來源。

### 4.3 深度強化學習策略（RDSAC）

RDSAC 的演算法設計已於 §2.4 詳述——discrete 分布式 SAC：雙頭 IQN 評論家（reward 回報 $Z_R$／entropy 回報 $Z_H$）+ CVaR 風險扭曲 + 遮罩式 categorical actor，訓練搭配 PER、n-step 回報、分數暖啟動與位能獎勵塑形。本節聚焦其輸出如何具體轉化為排程動作，並嵌入 §3.3 所述的失效安全整合架構，避免與 §2.4 重複演算法細節。

RDSAC 的策略輸出經兩條生產路徑之一作動：（1）**優先權微調**——`job_submit.lua` 呼叫推論服務的 `POST /decide`，將建議轉為 Slurm 佇列優先權加成，實際落點仍由 `select/cons_tres` 與 GRES 決定；（2）**顯式節點綁定**——評估／部署 harness 呼叫 `POST /act` 取得節點選擇 `(node_j, gpu_k)`，於提交時寫入工作的 `ReqNodeList`（`sbatch -w`），使 Slurm 直接排到指定節點與 MPS 槽。任一路徑下，若推論服務逾時、異常，或（路徑 2）checkpoint 觀測拓樸與實機不符而 abstain，皆靜默回退至 §4.2 的 score 排程，確保 slurmctld 不被研究用元件阻塞。

訓練採 sim-to-real 兩段式：先在離散事件模擬器中以 PER、n-step 回報與分數暖啟動大量訓練出基本模型；再以 `live_daemon` 旁觀模式收集真實叢集的 (observation, action, reward) transition，以 RLPD（Reinforcement Learning with Prior Data）微調成貼合實機分布的策略，細節見 §5.1、§5.3.1。

## 5. 實驗與評估方法

### 5.1 訓練與評估管線

直接在實際環境從頭訓練的話，強化學習需要數十萬到數百萬個 transition，真實叢集一個編排決策對應一個跑數分鐘至數小時的任務，湊滿樣本要等數月，因此採 sim-to-real 兩段式：

1. 在模擬環境大量訓練，產出基本模型
2. 上線部署，記錄真實叢集 (observation, action, reward) 資料
3. RLPD (Reinforcement Learning with Prior Data) 用真實資料把基本模型微調成真實環境策略

為取得可信結論，添加以下方法來讓結果穩固：

- 抗跑序飄移：GPU 隨運行暖機、快取轉熱會使「越晚跑的越快」。若各方法依序整段跑完，跑序會與方法混淆。本研究以交錯輪轉（interleave）讓每個方法跨多輪輪過各個跑序位置，並丟棄暖機輪，將漂移誤差平均化。
- 多 seed 配對信賴區間：以共用隨機數（CRN）讓各方法跑相同工作負載並配對相減，再以多個訓練 seed 重複，以配對 t 檢定回報 95% 信賴區間與 p 值。
- 尾端與 SLO 指標：除平均 JCT 外，同時報告 p95／p99／CVaR 與 SLO 違反率，以捕捉飢餓與 straggler。
- 真實工作負載：除合成負載外，建置可攜式 PyTorch 環境於共享儲存，以真實 BERT 推論／微調作為實機工作。


### 5.2	評估指標說明

在評估一個排程策略時都有主要指標平均工作完成時間(mean JCT)和尾部延遲指標 p95/p99 JCT、CVaR(0.25)。因為從 mean JCT 難以看出把 straggler、queue starvation、head-of-line blocking 等問題，即「多數 job 正常、少數被拖很慢」的情況。這些慢 job 幾乎不影響 mean JCT，但主導使用者體感。而 p95/p99 指標專門抓「最差 5%/1% 有多慢」，正是 mean 結構上看不到的那段。此外，RDSAC-cvar 的設計目標就是優化回報下尾（= JCT 上尾）；只測量 mean 等於拿不會動的尺去量專門改尾部的方法，結構上必然測不出差異。


### 5.3 主要結果

**模擬環境可區分策略（方法學正面結果）。** 在離散事件模擬中，以對 SLO 敏感的 AI 伺服器工作負載（2×1 拓樸、offered load ρ≈0.7 的中度競爭、8 個 held-out seed）評估，具尺寸感知的啟發式相較先到先服務（FCFS）顯著降低 SLO 違反率（表 2），顯示模擬器具備區分排程策略的能力——前提是工作負載與指標選對了維度（時序排序、SLO 感知），而非需要規模的多節點裝箱。此工作負載的 SLO 定義為：推論工作（短、佔 1 GPU 的 25／50％ MPS）帶有延遲期限 `slo_s` = runtime × 4，訓練工作（長、獨佔或大 MPS，含少量 2-GPU 跨節點 gang）為 best-effort（無期限）；SLO 違反率即帶期限工作中 JCT 超過 `slo_s` 的比例。

表 2. 模擬環境下 AI 伺服器工作負載的排程器區分（2×1、ρ≈0.7、8 seed 平均）

| 排程器 | 平均 JCT (s) | 推論 JCT (s) | SLO 違反 (%) | 使用率 |
|---|--:|--:|--:|--:|
| FCFS | 2199 | 1847 | 66.5 | 0.58 |
| multifactor | 1108 | 461 | 41.1 | 0.63 |
| score | 1129 | 520 | 40.7 | 0.63 |

**跑序漂移會污染單趟排名。** 在真實叢集上，單趟（block design）量測一度顯示 FCFS「顯著贏」score 達 +5.0%。然而其改善幅度與「跑第幾位」完美單調相關（表 3）：同一個 FCFS，跑最後一位時 +5.0%、跑最先時則僅 +0.5% 甚至轉為 −0.4%（不顯著）。此為叢集隨時間暖機的漂移假象，而非排程器差異。

表 3. FCFS 的「優勢」隨跑序位置變動（揭示漂移）

| 排程器 | seed42 位置 | seed43 位置 | seed44 位置 | ΔJCT% vs score（各 seed）|
|---|--:|--:|--:|---|
| FCFS | 4（最後）| 1（最先）| 3 | +5.0 / +0.5 / −0.4 |
| packing | 3 | 2 | 1 | +1.6 / +0.8 / −0.6 |
| multifactor | 2 | 3 | 4 | +1.0 / +0.9 / −0.7 |

![圖 1](figures/fig_drift.png)

圖 1. 將三個啟發式跨 3 seed 的 ΔJCT% 對「跑序位置」作圖：正斜率的趨勢線（+0.62%/位）顯示表面「優勢」隨越晚跑而增大，證實其為叢集暖機漂移的假象、而非排程器本身的效果。

**真實 2×1 叢集：啟發式統計打平。** 以 3 seed × 3 種臂順序校正跑序位置後的 cross-seed 聚合如表 4，三個 ΔJCT% 的 mean±std **全部跨越 0**，即生產 score 與 Slurm 原生 FCFS／multifactor／packing 在真實 2×1 上統計打平，無任何排程器具優勢。

表 4. 真實 2×1 叢集抗漂移、多 seed 聚合（mean±std）

| 排程器 | 平均 JCT (s) | p95 | p99 | CVaR | ΔJCT% vs score |
|---|--:|--:|--:|--:|--:|
| score | 182.7±31.7 | 369.1±46.6 | 409.8±37.7 | 349.2±43.0 | — |
| multifactor | 181.8±30.3 | 364.5±42.7 | 405.9±35.6 | 346.7±39.4 | +0.4±1.0 |
| packing | 181.5±30.7 | 365.9±42.1 | 405.2±33.5 | 346.3±38.3 | +0.6±1.1 |
| FCFS | 179.7±32.0 | 362.5±37.2 | 402.4±28.2 | 343.2±32.8 | +1.7±2.9 |

**深度強化學習放置：精確量測下顯著小輸。** 在暴露 GPU 異質性的放置實驗中，深度強化學習 checkpoint 一度名目領先 +3.9%（p=0.116，不顯著）；於非飽和 regime 降噪並將樣本三倍化（n=246）後，三個學習型策略全部反轉為**顯著小輸** score（表 5），且此結論跨兩個訓練 seed 一致。以真實 BERT 工作橫掃低／高負載與 2-GPU gang（head-of-line blocking）三種競爭機制，亦僅見落在雜訊內的微弱訊號。

表 5. 實機放置精確四方比較（n=246，配對 t 檢定）

| 排程器 | 平均 JCT (s) | p99 | CVaR | ΔJCT% | t 檢定 p |
|---|--:|--:|--:|--:|--:|
| score | 167.8 | 425.6 | 374.7 | （基準）| — |
| SAC | 174.0 | 432.4 | 379.5 | −3.7 | 3.7e-16 |
| RDSAC-mean | 174.5 | 435.8 | 379.5 | −4.0 | 2.1e-17 |
| RDSAC-cvar | 175.5 | 432.8 | 381.6 | −4.6 | 5.3e-12 |

須釐清**統計顯著與實務顯著的區別**：表 5 的 p 值極小（≈1e-12～1e-17）源於 n=246 的大樣本，代表「可偵測地變慢」而非「大幅變慢」；三個學習臂的 ΔJCT% 落在 −3.7～−4.6%，仍位於 §5.5 界定的 ±5% 實務等價帶內（僅偏於其負緣）。換言之，學習型策略在此規模是**可統計偵測地、但非實務顯著地**遜於 score，與後續「排程策略空間近乎是平的」洞見一致，而非與之矛盾。

#### 5.3.1 實機 DRA cuBLAS 評估（低負載共置：策略空間平坦）

真正能觸及這座**異質**叢集放置槓桿的評估，必須讓卡內共享（NVIDIA MPS）與計算異質性反映到 JCT。為此以**真 cuBLAS（`gpu_workload`）＋ MPS 分數共置**（Poisson 到達、mps-oversub **1.0** 的低負載、MPS 分桶 25／50／75／100）跑實機配對 A/B（2×1、提交時 `-w` 顯式放置、drift-robust interleave、**8 seed**、每 seed n_jobs=30×3 rounds、σ=1.0），並以**正確分析層級——seed**——的 one-sample t 檢定每臂 ΔJCT% 是否顯著異於 score，避免把 job 當獨立單位的偽重複。全部臂於**同一 DRA MPS 後端**量測（見下方「後端混淆」）：六個學習／啟發式臂（score／SAC／RDSAC-mean／RDSAC-cvar／CrossQ／**RLPD**）加兩個 **Slurm 原生 baseline**（**fcfs** = `sched/builtin`+`priority/basic`、關 Lua、嚴格 FIFO；**backfill** = `sched/backfill`+`priority/basic`、關 Lua、Slurm 現代預設；皆無 `-w`、由 `select/cons_tres` 選節點），代表「不加智慧放置的 vanilla Slurm」。

**RLPD 臂（實機資料微調）。** 為檢驗「用實機資料線上微調能否縮小 sim-to-real 落差」，以 `live_daemon` **旁觀模式**收集真實 transition（記錄決策時觀測、Slurm 實際落點、實現 −JCT，不干擾生產、產出有效 off-policy 資料），共 **181 筆**；再以**忠於原始 RLPD（Ball 等人 2023）**之實作微調（對稱 50／50 離線／線上取樣、LayerNorm 集成評論家 N=10／隨機子集 M=2 取 target min、離散 SAC actor、固定溫度 α=0.05——被遮罩的離散 SAC 自動-α 會因合法動作數遠小於 log(A) 而發散，故釘死）RDSAC-cvar 的 sim 策略得 **RLPD-v3**，作為第八臂。

**結果：低負載下策略空間近乎平坦，無臂勝出。** 表 6（n=8）：**score ≈ fcfs ≈ backfill**（皆 6.8s、ΔJCT ±0.1%、p≈0.88，統計打平），八臂中的六個學習臂則**一致略差**（−1.7～−5.7%），其中 CrossQ／RDSAC-cvar 達 seed 顯著（p=0.011／0.033）但幅度僅 −3～−5%、落在 §5.5 界定的 ±5% 實務等價帶內。**RLPD 微調亦未翻盤**（−3.6%、p=0.071），與未微調的學習臂同屬略差一檔——181 筆真實 transition 不足以彌合落差，「線上微調可救援」在此資料規模為誠實**否定**結果。換言之，**在低負載真實 cuBLAS＋MPS 共置下，無論學習式（含實機微調）或 vanilla Slurm 都未勝過 score，且彼此皆落在 ±5% 內**：此 regime 的排程策略空間近乎是平的。

表 6. 實機 DRA cuBLAS 放置 A/B（2×1、DRA MPS、oversub 1.0、8 seed；JCT／p99／CVaR 為秒；Δ 為相對 score 的百分比，**＋ = 勝過 score**；seed 為正 = 8 個 seed 中 ΔJCT%>0 的個數；seed-t = ΔJCT% 的 seed 層級 one-sample t）

| arm | JCT(s) | p99(s) | CVaR(s) | ΔJCT% | Δp99% | ΔCVaR% | seed 為正 | seed-t p |
|---|--:|--:|--:|--:|--:|--:|:--:|--:|
| score | 6.8±0.7 | 22.1±1.3 | 14.9±1.7 | —（基準） | — | — | — | — |
| fcfs | 6.8±0.6 | 21.9±1.4 | 14.8±1.8 | −0.1±3.0 | +1.3±2.0 | +0.9±2.1 | 2/8 | 0.891 |
| backfill | 6.8±0.6 | 22.2±1.3 | 14.8±1.8 | −0.1±2.1 | −0.4±3.8 | +0.8±1.8 | 3/8 | 0.876 |
| RDSAC-mean | 7.0±0.8 | 24.0±3.7 | 15.8±2.5 | −1.7±3.0 | −8.4±16.1 | −5.6±8.3 | 1/8 | 0.168 |
| CrossQ | 7.1±0.8 | 23.9±3.6 | 15.7±2.4 | −3.1±2.5 | −8.0±15.5 | −5.4±8.4 | 1/8 | 0.011 |
| RLPD | 7.1±0.9 | 23.9±3.8 | 15.9±2.4 | −3.6±4.8 | −8.1±16.9 | −7.0±9.3 | 2/8 | 0.071 |
| RDSAC-cvar | 7.2±0.8 | 25.2±4.5 | 16.2±2.9 | −4.6±4.9 | −14.1±19.5 | −8.3±11.2 | 1/8 | 0.033 |
| SAC | 7.2±0.7 | 24.3±3.6 | 16.1±2.2 | −5.7±7.1 | −10.1±15.9 | −8.7±11.6 | 2/8 | 0.058 |

**方法學要點一：GPU 分配後端會混淆結論——「正面結果」不跨後端重現。** 本平台於評估期間由 device-plugin 遷移至 **Kubernetes DRA**（`gpu.nvidia.com` ResourceClaim + MPS，見 `docs/dra-migration.md`）。**先前於 device-plugin 後端**、同樣真實 cuBLAS 低負載共置，曾量到學習式**小勝** score（RDSAC-cvar +4.5±4.4%、seed-t p=0.023，n=8）；但**換到乾淨的 DRA 後端後，同一 recipe、同一分析層級反轉為略輸**（cvar −4.6%）。實測 DRA MPS 亦使絕對 JCT 大幅改變（同一 hybrid score 基準由 device-plugin 的 39.9s 降至 DRA 的 18.4s，見 §5.3.2）。**結論：那個 +4.5% 的正面結果是後端相關的、不穩健**；跨後端的絕對數字與方向皆不可直接沿用。此為本文評估方法學的一項重要教訓——**排程結論不僅依賴 workload／負載，也依賴底層 GPU 分配後端**，凸顯「同一後端統一重測」的必要，也是本文將全部臂於同一 DRA 後端重跑（表 6、表 7）的原因。

**方法學要點二：seed 層級與小樣本雜訊。** 實機量測雜訊大：同一 checkpoint 的 ΔJCT% 在不同 run 間可大幅擺盪（先前一組 3-seed run 曾量到某臂 +15%，擴至 n=8 後僅剩 +3.6%，為抽到幸運 seed 的小樣本假象）。故一律以 n=8 的 seed 層級估計為準、僅就其宣稱，不採單 seed 或小樣本大數。

#### 5.3.2 實機 DRA Hybrid 評估（高負載真實 LLM serving：score 最佳）

§5.3.1 為合成 cuBLAS、低負載。為以**真實 AI-serving** job＋高負載檢驗，把 payload 換成 Qwen2.5-0.5B 的批次自迴歸生成（長 prompt、prefill-compute-bound，對應 RAG／摘要類長 context 服務），offered load 拉到 **oversub 2.0**（超過單卡容量、迫使動用兩張卡）。此處揭露一個真實硬體約束：慢卡節點（3080）**host RAM 僅 7.5GB**，每個 LLM job 需先把 torch＋約 954MB 模型載入 host RAM（約 2–3GB），兩個並發即 OOM → 進程卡死 → Slurm drain 該節點。故採 **hybrid workload**：mps 25／50 小 job 走 cuBLAS（自包含、可 4-way 共置），mps 75／100 大 job 走真實 LLM（門檻 75 保證任兩 LLM 需求相加 >100，永不同卡共置、慢卡最多同時載入一個模型）。全部八臂（六學習／啟發式 + fcfs／backfill）於**同一 DRA 後端**、8 workload seed 統一量測。

**結果與 §5.3.1 相反：高負載下 score 最佳，學習式與 naive Slurm 皆較差。** 見表 7（score 基準 JCT=18.4s）。**沒有任何學習臂勝過 score**（ΔJCT −3.9～−11.9%），RDSAC-cvar（−3.9%、p=0.344 不顯著）為最接近打平者、與 backfill（−4.9%、p=0.076）同屬「最不差」一檔；fcfs／RDSAC-mean／RLPD 顯著最差（p≤0.031）。**且 score 亦勝過 vanilla Slurm**：fcfs 顯著落後（−10.8%、p=0.004、0／8）、backfill 邊緣落後（−4.9%、p=0.076）——證明 score 的 bin-pack／SJF 因子相對「數 GPU」式的 cons_tres 放置確有加值，而非只是與學習式互比的空殼基準。

表 7. 實機 DRA Hybrid 放置 A/B（2×1、DRA MPS、oversub 2.0、8 workload seed、mps 25／50→cuBLAS、75／100→真實 LLM；JCT／p99／CVaR 為秒；Δ 為相對 score 的百分比，**＋ = 勝過 score**；seed 為正 = 8 個 seed 中 ΔJCT%>0 的個數；seed-t = ΔJCT% 的 seed 層級 one-sample t）

| arm | JCT(s) | p99(s) | CVaR(s) | ΔJCT% | Δp99% | ΔCVaR% | seed 為正 | seed-t p |
|---|--:|--:|--:|--:|--:|--:|:--:|--:|
| score | 18.4±3.4 | 44.5±14.0 | 32.3±10.1 | —（基準） | — | — | — | — |
| RDSAC-cvar | 19.2±4.4 | 51.5±19.9 | 33.6±10.8 | −3.9±10.7 | −17.4±41.2 | −4.7±13.6 | 3/8 | 0.344 |
| backfill | 19.3±3.5 | 34.1±8.1 | 29.1±6.1 | −4.9±6.6 | +19.7±20.7 | +6.8±15.5 | 2/8 | 0.076 |
| CrossQ | 19.8±3.2 | 49.8±15.1 | 36.1±9.4 | −8.4±10.6 | −13.1±18.4 | −14.0±14.5 | 2/8 | 0.062 |
| SAC | 20.0±3.3 | 55.7±22.2 | 37.1±10.3 | −9.7±15.9 | −30.9±53.0 | −19.6±35.5 | 3/8 | 0.128 |
| fcfs | 20.4±3.5 | 35.7±8.1 | 30.8±5.7 | −10.8±7.3 | +16.0±21.0 | +1.1±16.5 | 0/8 | 0.004 |
| RDSAC-mean | 20.5±4.4 | 53.2±18.0 | 35.9±10.1 | −11.0±11.6 | −21.1±33.0 | −13.0±14.9 | 1/8 | 0.031 |
| RLPD | 20.4±2.7 | 54.1±14.7 | 37.6±7.9 | −11.9±11.1 | −26.3±33.4 | −21.7±30.7 | 1/8 | 0.019 |

**機制可解釋。** 學習臂把只 **35–39%** 的 job 放到慢卡 3080，而 score 放 **47%**——學習式較貪心地偏好快卡 4070。在 oversub 2.0 的高負載下，這種過度集中反而**把 4070 塞爆、排隊變長**，總體 JCT 更差；score 更均衡地把慢卡也用起來，反而較快。

**方法學要點四：尾端呈相反權衡（附帶觀察，不作宣稱）。** naive Slurm（backfill／fcfs）雖平均較差，其 p99／CVaR 反而優於 score（backfill Δp99 +19.7%、ΔCVaR +6.8%；fcfs Δp99 +16.0%）——即 score 以較緊的平均換取較長的尾；惟此處 p99／CVaR 為每 seed 20–30 個完成 job 的估計、變異極大（Δp99 標準差達 ±20～53%），僅作尾端行為的定性觀察，在此小叢集不作統計宣稱。

#### 5.3.3 實機 DRA SLO serving 評估（尾端／SLO 指標：score 亦最佳）

§5.3.1／5.3.2 以 JCT 為主軸。但本平台是 **Slurm-on-Kubernetes 的 AI serving**，最貼合其部署情境的服務指標是**每請求 SLO 違反率**；且風險敏感的 RDSAC-cvar 其 CVaR 目標**正是為壓低尾端而設**——若學習式放置有任何優勢會顯現的地方，理應是 SLO／尾端這個「主場」指標。為給尾端估計足夠的統計檢力（§5.3.2 每 seed 僅 20–30 個完成 job、p99 變異極大），本評估改用 **serving-realistic 高-QPS 工作負載**：120 個短 cuBLAS 請求 ×2 輪、Poisson 到達、oversub 2.0、DRA MPS；每請求帶延遲期限 `slo_s = runtime × 4`，**SLO 違反率 = slowdown > 4 的請求比例**。全部八臂於同一 DRA 後端、8 seed 統一量測（表 8、runner `eval/scripts/run_slo8.sh`）。

**結果與 §5.3.1／5.3.2 一致：即使在 CVaR 的「主場」尾端指標上，score 仍最佳。** score 的 SLO 違反率最低（17.6%），**沒有任何學習臂勝過它**。RDSAC-cvar 為**最不差的學習臂**（−4.4pp、**唯一不顯著**的學習臂，seed-t p=0.276），與 §5.3.2 的 JCT 排名一致（cvar 最穩健、輸最少但非贏家）；SAC／RLPD／CrossQ 顯著較差（−6.0～−7.5pp、p≤0.027）；naive Slurm（fcfs／backfill）最差（−12pp、p≈0.05–0.06）。

表 8. 實機 DRA SLO serving A/B（2×1、DRA MPS、120 短 cuBLAS 請求 ×2 輪、Poisson 到達、oversub 2.0、8 seed；SLO 違反率 = slowdown>4 的請求比例、越低越好；ΔSLOviol = score−arm 的百分點，**＋ = 違反更少 = 勝過 score**；seed-t = 該 Δ 的 seed 層級 one-sample t）

| arm | SLO違反% | p99(s) | slowdown_p99 | JCT(s) | ΔSLOviol(pp) | seed-t p |
|---|--:|--:|--:|--:|--:|--:|
| score | 17.6±8.7 | 41.2 | 19.1 | 7.8 | —（基準） | — |
| RDSAC-cvar | 22.0±13.1 | 40.6 | 18.8 | 8.3 | −4.4 | 0.276 |
| RDSAC-mean | 23.1±10.5 | 37.5 | 17.0 | 8.4 | −5.5 | 0.056 |
| SAC | 23.6±11.6 | 33.4 | 15.4 | 8.8 | −6.0 | 0.017 |
| RLPD | 23.7±10.2 | 35.7 | 16.2 | 8.3 | −6.0 | 0.027 |
| CrossQ | 25.1±12.9 | 37.8 | 15.2 | 8.5 | −7.5 | 0.017 |
| fcfs | 29.7±20.1 | 18.9 | 8.5 | 8.4 | −12.0 | 0.063 |
| backfill | 30.0±19.6 | 41.0 | 19.9 | 10.8 | −12.3 | 0.052 |


**尾端呈相反權衡（附帶觀察，不作宣稱）。** 與 §5.3.2 呼應：score 雖 SLO 違反「率」最低，其**極端尾** p99／slowdown_p99 反而較重（p99 41s、slowdown_p99 19），而 SAC／CrossQ／尤其 fcfs 的最壞情況有界（fcfs p99 僅 18.9s、slowdown_p99 8.5）——即「違反次數少」與「最壞延遲有界」是**兩個不同、甚至相反**的目標：score 把多數請求壓在期限內、卻容忍少數更長的 straggler；FCFS 順序執行使任一請求最壞情況有界、卻有更多請求輕微逾期。此為尾端行為的定性觀察，per-seed 樣本仍小、不作統計宣稱。

**綜合 §5.3.1–§5.3.3（皆同一 DRA 後端）。** 三個真實-硬體場景、跨 JCT 與 SLO 兩類指標，在乾淨統一後端下給出一致的誠實圖像：**低負載 cuBLAS 策略空間平坦**（§5.3.1，全在 ±5% 內、學習式略差）、**高負載 hybrid LLM serving 則 score 最佳**（§5.3.2，學習式與 naive Slurm 皆較差）、**serving SLO／尾端指標 score 亦最佳**（§5.3.3，即使在 CVaR 主場的 SLO 指標上，cvar 仍僅為最不差的學習臂）。**在乾淨統一後端下，學習式放置在三個場景、跨 JCT 與 SLO 兩類指標皆未穩健勝過 score、亦未勝過 naive Slurm**；先前 device-plugin 後端量到的 cuBLAS 小勝（+4.5%）不跨後端重現、屬後端假象；線上 RLPD 微調（181 筆真實 transition）亦未翻盤。這強化本文核心命題：**排程結論高度依賴評估場景與 GPU 分配後端**，而學習式放置的實機效益遠比單一場景所暗示的脆弱。本文據此**不宣稱學習式優越**，改以方法學（抗漂移、多 seed、seed 層級配對、同後端統一重測、存活者偏差消除）與誠實的場景／後端依賴負結論為主要貢獻。

### 5.4 與雲端原生 SOTA 基準的對照（強化基準）

§5.3 的區分實驗僅對照自家啟發式，可能招致「未與引用的 SOTA 比較」之質疑。為此，我們將 §2 所述的兩個雲端原生排程器近似納入同一模擬對照——Kueue 式 fair-share（跨使用者 max-min 交錯）與 Volcano 式 binpack（最大需求優先）——在高競爭的 1×1、**佇列飽和** regime（offered load 拉高至系統飽和；GPU 使用率仍約 0.6，屬**佇列**飽和而非**算力**飽和，aiserve 工作負載，8 seed）下量測（表 9）。此處的拓樸（1×1）與競爭度（飽和）皆與表 2（2×1、ρ≈0.7 中度競爭）不同，故 JCT 絕對值明顯較高（如 FCFS 2640 vs 表 2 的 2199、score 1887 vs 1129）；兩表各自檢驗其 regime 內的**相對排名**，跨表的絕對 JCT 不宜直接相減。

表 9. 強化基準：雲端原生 SOTA 近似納入模擬對照（aiserve，8 seed，1×1 佇列飽和）

| 排程器 | 平均 JCT (s) | SLO 違反 (%) | 使用率 |
|---|--:|--:|--:|
| FCFS | 2640 | 59.7 | 0.59 |
| Volcano-binpack | 2515 | 45.3 | 0.60 |
| score（生產） | 1887 | 40.1 | 0.60 |
| multifactor | 1720 | 38.8 | 0.60 |
| Kueue-fairshare | 1722 | 38.8 | 0.60 |

此結果有兩點意涵。第一，**區分確實存在但邊界清楚**：FCFS 與純 binpack（優先塞入大型訓練工作、延誤延遲敏感的推論）明顯較差，而 fair-share／multifactor／score 三者叢聚於同一最佳帶（SLO 違反約 39–40%）。第二，**引用的 SOTA 近似並未勝過生產 score**——Kueue-fairshare 與 multifactor 打平、score 落在同帶。這把「合理排程策略空間狹窄」的結論從自家啟發式延伸到雲端原生 SOTA，回應了「只比自家 heuristic」的質疑（實作為 sim 內排序近似，非完整 Kueue 准入控制器／Volcano 節點評分外掛）。

### 5.5 結果討論

上述結果指向一個**限定於缺乏卡內共享（等待主導）regime** 的洞見：**在該場景下 2×1 的排程策略空間近乎是平的**——不僅深度強化學習未能穩健勝過啟發式，連生產 score 對 FCFS 等簡單基準亦僅打平。（須強調此「平坦」是**場景限定**的：§5.3.1 已證，一旦換成真實 cuBLAS＋MPS 分數共置、觸及卡內共享的放置槓桿，策略空間不再平坦，學習式放置即在多數工作上更佳於 score。）就實機聚合（表 4）而言，各啟發式相對 score 的 ΔJCT% 落在約 ±0.4～1.7%、且信賴區間跨越 0，在 ±5% 的實務等價界（practical-equivalence margin）內可視為**統計等價**（其中 FCFS 因變異較大而區間較寬，等價宣稱較弱）；換言之這是「證實無實務差異」，而非僅「未偵測到差異」。此 ±5% 等價界同樣涵蓋學習型策略：表 5 中三個學習臂的 −3.7～−4.6% 雖因大樣本而**統計顯著**，其幅度仍落在等價帶內，故整個「策略空間近乎是平的」判斷橫跨啟發式與學習式兩類排程器，而非僅指前者。

**關於「規模」的誠實界定。** 本研究據此**推測**排程策略的差異需要更大規模或更高競爭方能顯現，但必須強調：這是**尚未被證實的假設，而非本研究的結果**。我們的初步規模掃描（1×1／2×1／2×2 的補充實驗）**並未**呈現「效益隨規模上升」的交叉趨勢（圖 2）：學習臂相對 score 的 ΔJCT% 在各規模皆為負、且隨規模非單調（2×1 反而最接近 score，2×2 又拉開），並無朝 0 收斂的跡象；RDSAC-cvar 在 1×1 甚至崩潰為 0% 完成。此掃描受限於較低訓練預算（40k 步）與跨尺度觀測空間不可直接比較，尚不足以支持或否證此假設。

![圖 2](figures/fig_scale.png)

圖 2. 規模掃描（σ=1.0、40k 步）下學習臂相對 score 的 ΔJCT%。若「效益隨規模浮現」成立，曲線應隨規模趨近 0（score 基準）；實測反而全程為負且非單調，故此假設未獲支持（受訓練預算與跨尺度不可比之限制，僅作為 open question 的方向性證據）。

將「效益隨規模浮現」從斷言降級為 open question，正是本研究誠實立場的一部分。此規模掃描也帶出一個方法學教訓：**單 seed 的模擬評估可能報告不可重現的假性優勢**，凸顯多 seed 配對統計的必要性，這與 §5.3 中「單趟平均值會得到錯誤排名」互為印證。


## 6. 結論與未來工作

### 6.1 威脅與限制

作為一篇以方法學與誠實負結論為主軸的研究，本節明列可能削弱結論的因素及其處置狀態：

- **GPU 分配後端混淆（已處置）：** 本平台於評估期間由 device-plugin 遷移至 Kubernetes DRA，實測顯示 GPU 分配後端本身會顯著改變絕對 JCT、甚至反轉相對排名——先前於 device-plugin 後端量到的 cuBLAS 學習式小勝（RDSAC-cvar +4.5±4.4%、seed-t p=0.023，n=8）換到乾淨 DRA 後端後反轉為略輸（cvar −4.6%，§5.3.1）。故本文將全部策略於**同一乾淨 DRA 後端統一重測**（§5.3.1–§5.3.3），所有 ΔJCT%／ΔSLOviol 皆對齊同一 score 基準；跨後端的絕對數字與方向皆不予沿用。
- **存活者偏差（已處置）：** §5.3.2 高負載 hybrid A/B 的第一版曾量到學習臂大幅領先，但那是存活者偏差：score 無顯式放置，其被 Slurm 分到慢卡（3080，host RAM 僅 7.5GB）的 job 會因 OOM／冷載入超時而 FAILED，而彙總只計 COMPLETED，使 score 的完成集被截斷、平均值失真而顯得偏優。三項修正還原公平比較：(1) 提交時 free-MPS 快照改為本地即時追蹤；(2) hybrid workload（大 job 走真實 LLM、小 job 走 cuBLAS）讓慢卡節點不再 OOM／drain；(3) 確認每臂完成數對等。修正後結論方向反轉為 score 最佳（§5.3.2）。
- **單／小樣本脆弱性（已處置）：** 實機量測雜訊大，早期單 seed 或小樣本結果多次呈現不可重現的假性優勢——例如某臂在 3-seed run 曾量到 +15%，擴至 n=8 後僅剩 +3.6%（§5.3.1、§5.3.2 方法學要點二）；模擬規模掃描亦曾見單 seed 假性領先於多 seed 重跑後被推翻的情形（§5.5）。故全文學習型結論一律以多 seed、seed 層級配對統計（ΔJCT%／ΔSLOviol 的 seed 層級 one-sample t 檢定）呈現，不採單 seed 或小樣本大數。
- **訓練預算 confound（部分處置）：** 決定性消融（如風險扭曲與 value-clip 穩定性比較）採 100k 步訓練；但規模掃描（§5.5、圖 2）僅用 40k 步，較複雜的 IQN 評論家可能欠訓練，故「效益隨規模浮現」之規模結論僅作為 open question，不作定論。
- **規模／小叢集限制：** 本研究主要評估規模為 2×1，補充規模掃描達 2×2；結論不宜外推至數十至數百 GPU 的生產叢集，「效益隨規模浮現」為尚未證實的假設（§5.5）。此外 1×1／2×1／2×2 的觀測維度與動作空間不同，checkpoint 不相容、須各自重訓，跨尺度絕對數值不宜直接相減。尾端指標（p99／CVaR）在此規模下亦為每 seed 20–30 個完成 job 的小樣本估計、變異極大（§5.3.2、§5.3.3），本文僅作定性觀察，不作統計宣稱。
- **評估與部署路徑一致性：** 實機放置實驗與建議的生產路徑皆以**提交時**將 RL 選定節點寫入工作的必要節點（`-w`／`ReqNodeList`）達成顯式放置，故評估忠實反映部署（§3.3）。平台另實作一個非同步 placement controller（`services/rl_scheduler/placement_controller.py`）作為不需改提交端的替代，惟其經 slurmrestd job-update 事後釘節點之實際生效仍在硬化中；此不影響 §5.3 之結論——該系列實驗皆走已驗證的提交時綁定。
- **SOTA 為模擬內近似：** §5.4 的 Kueue／Volcano 對照為模擬內**排序層級**近似（fair-share max-min／binpack 最大需求優先），非完整的 Kueue 准入控制器或 Volcano 節點評分外掛；等價結論限於排序策略層級，未涵蓋配額借還、gang 准入等機制。

### 6.2 結論

本研究設計並實作了一套以 Kubernetes 部署、Slurm 為核心、整合 MPS 與失效安全 RL 決策的 AI 伺服器 GPU 排程平台，並提出一套兼顧抗漂移、多 seed 配對統計、尾端指標與**同後端統一重測**的模擬到實機評估方法學。核心發現是**排程結論高度依賴評估場景與底層 GPU 分配後端**：於同一乾淨的 Kubernetes DRA 後端統一重測全部策略後，**低負載真實 cuBLAS ＋ MPS 共置的排程策略空間近乎平坦**（生產 score、Slurm 原生 FCFS／backfill 與學習式含實機微調 RLPD 全落在 ±5% 內、學習式略差），**高負載真實 LLM serving 則 score 最佳**（學習式與 naive Slurm 皆較差），且**改以 serving SLO 違反率為指標軸重測、score 於 JCT 與 SLO 兩軸皆最佳**（§5.3.3，即使在 CVaR 主場的尾端指標上學習式仍未勝出）——在乾淨統一後端下，學習式放置在三個場景、跨兩類指標皆未穩健勝過 score、亦未勝過 naive Slurm（§5.3.1–5.3.3）。一項重要的方法學教訓是：先前於 device-plugin 後端量到的 cuBLAS 小勝（RDSAC-cvar +4.5%）在 DRA 後端不重現（反轉為略輸），凸顯排程結論會被 GPU 分配後端混淆、須以單一一致後端統一重測；另一教訓是單 seed／小樣本會誤導（3-seed 曾量到 +15%，n=8 下不復存在）。唯一穩健的正面觀察是 **score 啟發式的 bin-pack／SJF 因子在高負載下勝過 vanilla Slurm 的 cons_tres 放置**（§5.3.2），顯示手調啟發式相對「數 GPU」式放置確有加值。次要發現為分布式評論家的訓練穩定性可被馴服：CVaR 風險扭曲與（更省的）Duan 式 target return-clip 皆能消除其崩潰、且互為替代，惟兩者皆未使任何學習臂穩健勝過基準。

### 6.3 未來工作

未來工作包含：（1）**擴展至更大、更高競爭的叢集**以檢驗「價值隨規模浮現」假設；（2）**將 RDSAC 接成 Kubernetes 原生排程 policy**——Kueue admission-ordering 或 kube-scheduler／DRA device-selection plugin，把「學習式策略層 × DRA 機制層互補」從論述變為 PoC，並直接落進雲端原生生態系；（3）**擴大實機微調的資料規模**——§5.3.1 已直接檢驗以忠於原論文的線上 RLPD（Ball 等人 2023）微調來縮小 sim-to-real 落差，惟 181 筆真實 transition 不足以翻盤，故後續需收集遠更大量的實機資料或加入 on-policy 修正；（4）延伸 return-clip 穩定器並以更新、更穩定的 off-policy 演算法續攻分布式評論家的崩潰／退化——已將 CrossQ（Bhatt 等人 2024 [18]：BatchNorm 評論家、移除 target network、UTD=1）納入為對照臂，SimbaV2 式的 RL 縮放架構（正規化 + 殘差骨幹）列為進一步方向。

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

[16] X. Ma, J. Chen, L. Xia, J. Yang, Q. Zhao, and Z. Zhou, "DSAC: Distributional Soft Actor-Critic for Risk-Sensitive Reinforcement Learning," *arXiv:2004.14547*, 2020.

[17] J. Duan, Y. Guan, S. E. Li, Y. Ren, and B. Cheng, "Distributional Soft Actor-Critic: Off-Policy Reinforcement Learning for Addressing Value Estimation Errors," *IEEE Transactions on Neural Networks and Learning Systems*, 2021. arXiv:2001.02811.

[18] A. Bhatt, D. Palenicek, B. Belousov, M. Argus, A. Amiranashvili, T. Brox, and J. Peters, "CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity," in *ICLR*, 2024.

[19] X. Wang, Y. Li, F. Guo, Y. Xu, and J. C. S. Lui, "Dynamic GPU Scheduling With Multi-Resource Awareness and Live Migration Support," *IEEE Transactions on Cloud Computing*, vol. 11, no. 3, 2023.

[20] H. Sedighi, F. Wuhib, and R. H. Glitho, "Dynamic Task Scheduling and Adaptive GPU Resource Allocation in the Cloud," *IEEE Transactions on Network and Service Management*, vol. 23, 2026.

[21] E. Lipe, N. Karia, C. Espenshade, C. Stein, A. Tantawi, and O. Tardieu, "Energy Efficient Scheduling of AI/ML Workloads on Multi Instance GPUs with Dynamic Repartitioning," in *IEEE 25th International Symposium on Cluster, Cloud and Internet Computing (CCGrid)*, 2025.

[22] M. Tsenos and V. Kalogeraki, "Exploring GPU-Based Workload Scheduling Techniques for Edge Computing," in *IEEE International Conference on Cloud Engineering (IC2E)*, 2025.

[23] Y.-D. Lin, Y.-T. Ling, Y.-C. Lai, and D. Sudyana, "Reinforcement Learning for AI as a Service: CPU-GPU Task Scheduling for Preprocessing, Training, and Inference Tasks," *IEEE Transactions on Network and Service Management*, vol. 22, no. 4, 2025.

[24] A. A. Majeed, M. Meribout, and S. M. Sali, "Scheduling Techniques of AI Models on Modern Heterogeneous Edge GPU—A Critical Review," *IEEE Transactions on Industrial Informatics*, vol. 22, no. 4, 2026.

[25] G. Zhang, W. Guo, Z. Tan, Q. Guan, and H. Jiang, "KIS-S: A GPU-Aware Kubernetes Inference Simulator with RL-Based Auto-Scaling," *arXiv:2507.07932*, 2025.

[26] Q. Wu, P. Chen, and Y. Wang, "Defragmentation Scheduling with Deep Reinforcement Learning in Shared GPU Clusters," in *Proceedings of the 2025 ACM Symposium on Cloud Computing (SoCC)*, 2025.
