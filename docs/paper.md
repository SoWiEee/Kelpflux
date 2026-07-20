# 基於 Slurm 與 Kubernetes 架構下 AI 伺服器 GPU 工作負載智慧排程技術之研究

### Intelligent GPU Workload Scheduling Techniques for AI Servers under a Slurm-on-Kubernetes Architecture

**作者一¹、作者二²**
¹○○大學 ○○系　²○○大學 ○○系
{author1, author2}@stumail.nutn.edu.tw

---

## 摘要

生成式 AI 與深度學習的普及，使 GPU 已成為學術與企業運算基礎設施的核心資源；然而一般大學實驗室常見多人共用少量 GPU 主機的情境：資源閒置與利用率不足並存、批次工作排隊與管理繁瑣，叢集內 GPU 世代混雜（算力可差數倍），且須同時容納低延遲推論與長時間訓練兩類性質迥異的負載，NVIDIA MPS 雖可讓多工作共享單張 GPU，但傳統 Slurm 排程仍難以同時兼顧使用率與工作完成時間（JCT），使排程決策更為複雜。

然而，既有以強化學習改善此類 GPU 排程的研究多止於模擬環境評估，鮮少採用多 seed 配對統計、多重比較校正或等價檢定，也少有在同一 GPU 分配後端上統一重測；因此「模擬中可分辨的策略效益能否穩健轉移到真實異質叢集」這一問題，至今缺乏統計嚴謹的判定。

本研究以此為背景，在 Slurm 的 job 提交路徑上整合一個失效安全（fail-safe）的強化學習決策端點：以 NVIDIA MPS 達成單張 GPU 內的多工作共享，交由 Kubernetes 負責容器化部署與健康監控等生命週期管理，使得服務逾時或異常時靜默回退既有啟發式排程。為誠實檢驗此類學習式策略的真實效益，本研究另提出一套抗跑序漂移、多 seed 配對統計、於同一 GPU 分配後端統一重測的 sim-to-real 評估方法學。

以此方法學在同一乾淨的 Kubernetes 動態資源分配（DRA）後端上，對三個真實硬體場景——低負載 cuBLAS 共置、高負載真實 LLM serving、SLO serving——並跨平均工作完成時間（JCT）與服務水準目標（SLO）違反率兩類指標統一重測後：學習式放置策略在三個場景中皆未穩健勝過生產啟發式 score，亦未勝過未加智慧放置的原生 Slurm；唯一穩健的正面發現，是 score 啟發式在高負載下確實勝過原生 Slurm 的 cons_tres 放置。本研究的貢獻因此定位為一套可重現的失效安全 RL 排程整合架構，以及一套能誠實揭露「排程結論高度依賴評估場景與底層 GPU 分配後端」的 sim-to-real 評估方法學，而非宣稱學習式排程必然優越。

未來工作包含：在更大規模、更高競爭的叢集上重新檢驗排程效益是否隨規模浮現的假設；並將 RDSAC 接成 Kubernetes 原生排程 policy（例如 Kueue 的准入排序或 DRA 的裝置選擇外掛）的概念驗證，把「學習式策略層與 DRA 機制層互補」的論述落實為可運行的雛型。

**關鍵詞**：GPU 資源排程、Kubernetes、Slurm、深度強化學習

## Abstract

The rise of generative AI and deep learning has made GPUs a core compute resource for academic and industry infrastructure alike; yet shared GPU hosts in a typical university lab commonly face idle resources coexisting with low utilization, cumbersome batch-job management, heterogeneous GPU generations differing in compute by several factors, and a mix of latency-sensitive inference and long-running training workloads. NVIDIA MPS lets multiple jobs share a single GPU, but traditional Slurm scheduling still struggles to jointly optimize utilization and job completion time (JCT), which together complicate scheduling.

Motivated by this, we integrate a fail-safe reinforcement-learning decision endpoint — RDSAC, a discrete distributional Soft Actor-Critic with a dual-head IQN critic and CVaR risk-sensitive optimization — into Slurm's job-submission path: NVIDIA MPS lets multiple jobs share a single GPU, Kubernetes handles containerized deployment and health monitoring as part of lifecycle management, and the endpoint silently falls back to the existing score heuristic on timeout or failure. To honestly assess the real-world benefit of such learned policies, we further propose a sim-to-real evaluation methodology built on drift-robust interleaving, multi-seed paired statistics, and unified re-testing on a single GPU-allocation backend.

Using this methodology, we re-tested three real-hardware scenarios — low-load cuBLAS co-location, high-load real LLM serving, and SLO-oriented serving — on the same clean Kubernetes Dynamic Resource Allocation (DRA) backend, across both mean job completion time (JCT) and SLO-violation-rate metrics. Learned placement policies failed to robustly beat the production score heuristic in any of the three scenarios, and also failed to beat naive Slurm without intelligent placement. The only robust positive finding is that the score heuristic outperforms naive Slurm's cons_tres placement under high load. We therefore position this work's contribution as a reproducible fail-safe RL integration architecture together with an evaluation methodology that honestly exposes how scheduling conclusions depend heavily on the evaluation scenario and the underlying GPU-allocation backend — rather than a claim that learned scheduling is superior.

Future work includes re-examining whether scheduling benefits emerge at larger scale and higher contention on a bigger cluster, and building a proof-of-concept that wires RDSAC into a Kubernetes-native scheduling policy (e.g., Kueue admission ordering or a DRA device-selection plugin), turning the "learned policy layer complements the DRA mechanism layer" argument into a working prototype.

---

## 1. 緒論

生成式 AI 與深度學習的普及，使學術與企業的 GPU 叢集需同時承載**低延遲推論**與**長時間訓練**兩類性質迥異的工作負載：前者須在服務水準目標（SLO）期限內完成、且常僅需部分 GPU 算力，後者則長時間獨佔整張卡。當底層硬體有限且**異質**——不同世代 GPU 的算力差異可達數倍——如何將兩類工作妥善排程與放置，以兼顧工作完成時間（JCT）、資源使用率與 SLO 達成，已成為雲端與邊緣運算基礎設施的核心議題 [1][4]。

傳統 HPC 排程器（如 Slurm）以靜態優先權與回填（backfill）為主，對 GPU 卡內共享與工作負載特性的感知有限。學界雖已提出多種以強化學習（RL）優化叢集排程的方法 [2][3]，在 2024~2026 更已有數項研究跨出模擬、於真實叢集上評估；然而這些真機研究幾乎一致地只報告正向差異，**鮮少以嚴謹的統計方法檢驗該效益是否穩健成立**，包含多 seed 配對檢定、多重比較校正與等價檢定在此文獻中罕見。換言之，當前真正稀缺的並非真機評估本身，而是能判定「觀察到的差異究竟是真實效益、還是實機量測雜訊」的統計嚴謹性；尤其「模擬中可區分的策略差異能否轉移到實機」這一 sim-to-real 問題，至今缺乏系統性的探討。實機評估之所以困難，在於單一節點即承載一個跑數分鐘至數小時的工作，使可用樣本極度稀少，且量測易受叢集暖機等時變因素干擾而產生假性排名。

排程之所以適合以強化學習處理，在於其本質是一個**序列決策**問題：每次放置不僅決定當下工作的完成時間，也改變佇列狀態、影響後續工作可用的資源；叢集負載（到達率、工作組成）又隨時間動態變化（如 §5.1、§5.3.1 所用的 Poisson 到達合成負載），難以窮舉成固定規則。相較於固定規則的啟發式，強化學習原則上具備從長期回報中學習動態調度策略的潛力——這是本研究考慮將 DRL 納入排程決策層、而非僅依賴手調啟發式的出發點；至於此潛力在本研究的小規模異質叢集上是否真正轉化為可觀測的效益，則是後續章節以統計嚴謹方法誠實檢驗的核心問題（結果見 §5.3、§6.1）。

針對上述缺口，本研究以一座雙節點異質叢集（RTX 4070＋RTX 3080）為實驗平台，建構從模擬訓練到實機驗證的完整流程，貢獻包括：

- **異質 GPU＋MPS 感知的排程研究平台**：以雙節點異質叢集為基礎，建構涵蓋模擬訓練與實機驗證的完整流程（§3.2、§3.4）。
- **失效安全（fail-safe）的 RL 排程整合架構**：將強化學習決策端點整合進 Slurm 的 `job_submit.lua` 提交路徑，服務逾時或異常時靜默回退既有啟發式排程，確保 slurmctld 排程核心不受阻塞（§3.3）。
- **RDSAC 演算法**：一套自行整合的離散、分布式、風險敏感 Soft Actor-Critic，以雙頭 IQN 評論家與 CVaR 風險扭曲，設計上針對尾端延遲（p95／p99／SLO 違反率）優化（§4.3.2）。
- **抗漂移、多 seed 配對統計的 sim-to-real 評估方法學**：以交錯輪轉消除跑序漂移、seed 層級配對檢定、多重比較校正與 TOST 等價檢定，判定排程策略效益是否穩健（§5.1、§5.3.3）。
- **三個真實硬體場景的誠實驗證**：於低負載共置、高負載真實 LLM serving、SLO serving 三個場景中誠實揭露學習式放置策略「未穩健勝過」生產啟發式與原生 Slurm 的負結論，以及 score 啟發式相對原生 Slurm 唯一穩健的正面發現（§5.3、§6.1）。

### 1.1 背景

Slurm Workload Manager [11] 是目前高效能運算叢集最廣泛使用的工作排程系統之一。它提供完整的工作提交、佇列管理、資源限額與帳務管理功能，並支援 GPU 資源分配。其以「節點」為基礎的資源模型非常適合研究工作環境，且已有廣泛的使用者社群與文件支援。然而 Slurm 原生設計以固定實體節點為主，對彈性擴縮與容器化整合的支援相對有限。而 Kubernetes [10] 是目前最主流的容器編排平台，能夠自動管理容器的部署、擴縮與健康監控。然而 Kubernetes 原生的排程器 (kube-scheduler) 以服務導向設計為主，對批次工作、GPU 共享與研究工作特有的排程需求支援不足。因此有多項研究嘗試將 Slurm 與 Kubernetes 整合，以結合兩者優點。在 Slurm 與 Kubernetes 整合方面，既有研究已指出高效能運算與雲端原生系統的匯流可以同時取得批次排程語意與雲端彈性管理能力；實作層面也有 SchedMD 官方的 Slinky [27] 等 Slurm-on-Kubernetes 方向的工具，以及 AWS ParallelCluster 等雲端 HPC 叢集部署方案，顯示此類架構已具有實務需求與發展基礎。

### 1.2	GPU 資源共享技術

NVIDIA 提供多種 GPU 資源共享技術，包括 Time-Slicing、Multi-Process Service (MPS) 以及 Multi-Instance GPU (MIG)。MPS 允許多個 CUDA 程式同時共享一張 GPU 的運算資源；MIG 則在硬體層面將 GPU 切分成獨立的分區，提供更強的隔離性。本系統目前採用基於 MPS 的方式，讓多個較小的工作共享同一張 GPU，以提升整體使用效率。

除了本系統採用的 MPS 之外，近年也有研究探討在 MIG 等機制上進行動態重新分割與能源效率最佳化 [21]；這類方法能提供較強隔離性，但也需要額外硬體支援、分割粒度與工作遷移成本，因此本專題目前先以部署門檻較低的 MPS 作為共享 GPU 的主要實作方式，也更貼近一般大學實驗室環境的規格。

## 2. 相關研究

以下依主題而非年代分四類回顧相關研究：GPU 叢集排程、分析與資源共享（§2.1）、強化學習排程（§2.2）、異質／邊緣 GPU 排程（§2.3），以及雲端原生 GPU 排程生態系（§2.4）；§2.4 末並以比較表（表 1）定位本研究於此生態系中的空隙。

### 2.1 GPU 叢集排程、分析與資源共享

Jeon 等人對微軟 Philly 叢集的大規模多租戶 GPU 工作負載進行分析 [1]，揭示了排隊延遲與資源碎裂問題。Gandiva [2] 與 Tiresias [3] 分別利用 DL 工作的可遷移性與分布感知排程降低 JCT。Weng 等人對阿里巴巴 PAI 叢集的研究 [4] 指出生產 MLaaS 工作高度分片化、以單卡短工作為主。本研究的合成工作負載即以 [1][4] 的統計特性為依據。

近期研究則更聚焦於單節點內或單一叢集模型上的動態資源調度本身。Wang 等人的 DCUDA [19] 針對單節點多 GPU 情境，設計了一套輕量級核心／記憶體使用率監控機制，搭配近乎零開銷的「執行中」CUDA 應用即時遷移，將 GPU 過載時間平均降低 78.3%、一般工作執行時間降低 42.1%（記憶體密集型工作最高 67%）。Sedighi 等人 [20] 則在 Alibaba 的 cluster-trace-gpu 生產工作負載軌跡之上，提出結合硬體與軟體分割的公平且需求感知動態資源配置演算法，於模擬環境中將 GPU 資源使用量降低達 88%。這兩項工作皆聚焦「資源配置本身如何隨工作負載動態調整」（即時遷移／再分割），評估分別侷限於單一多 GPU 節點與純模擬 trace 重放，並未涉及與批次排程器 Slurm 的整合或真實異質叢集上的統計驗證。

在 MIG 研究上，Lipe 等人 [21] 針對單張 A100 GPU 的 MIG 切片，先以 Earliest-Deadline-First–Slowest-Slice（EDF-SS）演算法處理切片內的工作排程，再以 DQN 強化學習決定何時、要重分割成 12 種切片組態中的哪一種，於「能耗＋延誤」的多目標指標上優於雙日重分割（26%）、靜態分割（31%）與完全不分割（68%）。相較於 MIG 的硬體級隔離與較高的重分割與遷移成本，本研究採用 MPS 的理由正是部署門檻更低，無需重新配置硬體分區，即可在既有消費級 GPU（RTX 4070／3080）上即時生效卡內共享，這也是本研究能以校園實驗室既有異質硬體直接驗證的前提。

### 2.2 強化學習排程

#### 2.2.1 強化學習排程近作與真機化趨勢

以 RL 排程 AI 服務型任務的研究路線與本研究最為鄰近。Lin 等人 [23] 提出的 UXP-RL策略是一個以 DQN 為核心，涵蓋前處理、訓練和推論三類任務、可部署為集中式或分散式排程器、並跨雲、邊、霧三層架構運作的 CPU-GPU 任務排程演算法。其於模擬環境中，集中式排程器將平均週轉時間相較 SJF／FCFS 與 TYPE 啟發式（依 GPU 需求高低分類任務）分別降低 57.81％、57.28％與 27.66％；分散式排程器則因能將長訓練工作卸載至雲端而釋放邊和霧資源，把推論任務週轉時間相較集中式再降低 89.07%。同屬 2025 年的近作中，Zhang 等人的 KIS-S [25] 以 PPO 訓練一個 GPU-aware 的 Kubernetes 推論自動擴縮策略（KIScaler），完全於自建模擬器中訓練後即零樣本部署，於多種流量情境下平均獎勵提升 75.2%、p95 延遲相較 CPU 基準降低最多 6.7 倍；其問題設定是**調整副本數**的自動擴縮，而非本研究的**工作放置**排程。Wu 等人的 DRR [26]（ACM SoCC ’25）則針對 GPU 共享叢集因分享機制、工作異質性與非同步生命週期造成的碎片化問題，以模仿學習 (imitation learning) 從既有啟發式暖啟動一個深度強化學習去碎片化排程器，並輔以多尺度策略最佳化平衡探索與利用；其同時於實體 Kubernetes 測試床與大規模模擬叢集上驗證，平均碎片率降低 50%，是本節所列文獻中少數同時涉及真實 Kubernetes 部署的學習式排程器。

值得注意的是，在 2025~2026 的 RL 排程研究逐漸往真實環境發展，除前述 DRR [26] 外，Dong 等人的 DRL-MLS [28] 已於**真實 9 節點異質 Kubernetes 叢集**上以改良 DQN 排程機器學習訓練任務，回報平均完成時間降低 22.2%、makespan 降低 5.9%；Dongare 等人的 RLTune [30]（SoCC ’25）則以 RL 驅動優先序結合 MILP 節點映射、於 Philly, Helios, Alibaba 生產 trace 上訓練，回報 GPU 利用率最高提升 20%、排隊延遲降低 81%、JCT 縮短達 70%。惟其評估止於離線 trace，且未涉真實叢集部署。

方法族亦不再侷限於深度 RL：圖神經網路結合多智能體 RL、以及**以大型語言模型輔助排程**的路線同步興起，後者如 Wang 等人的 SchedMate [29] 於 128-GPU 實體叢集上，以 LLM 從原始碼、日誌與歷史紀錄萃取語意訊號（如更準的執行時間估計）餵給**既有**排程器，而非由 LLM 直接做放置決策；亦有以監督式學習從即時遙測預測各候選節點完成時間來排序節點者。就本研究的定位而言，目前尚未見有研究將 LLM 直接置於 `job_submit` 式的**放置決策迴路**中。

#### 2.2.2 與最相近工作的定位差異

前述的 DRL-MLS [28] 是本節所列文獻中，在系統形貌上與本研究最相近者——同為真實**異質** GPU 叢集上的**學習式工作放置**，且同樣建基於 Kubernetes，故一併正面處理其區隔：（1）**排程層級**——[28] 為 Kubernetes 原生排程器，直接以 K8s 為排程主體；本研究則作用於 **Slurm-on-Kubernetes** 架構，Kubernetes 僅為部署基座、排程主體仍是 Slurm，RL 決策整合進 `job_submit.lua` 提交路徑並具失效安全回退（§3.3），面對的是「如何在不阻塞 slurmctld 的前提下把學習式策略放進既有 HPC 排程核心」這一 [28] 不需處理的問題。（2）**目標函數**——[28] 優化平均完成時間與 makespan；本研究的 RDSAC 則以 CVaR 風險扭曲**針對尾端**（p95／p99／SLO 違反率）設計優化目標，並額外納入卡內 MPS 分數共享的放置槓桿。（3）**統計嚴謹性與結論方向**——[28] 回報單向的正向差異，未見多 seed 配對統計、多重比較校正或等價檢定；本研究則以 seed 層級配對檢定、Holm-Bonferroni 校正與 TOST 等價檢定判定差異是否穩健，並誠實回報學習式放置**未穩健勝過**生產啟發式的負結論（§5.3、§5.3.3）。換言之，[28] 與本研究並非同一問題的競爭解法，而是分屬「Kubernetes 原生／平均值導向／正面結果」與「Slurm-on-K8s 整合／尾端導向／統計嚴謹的誠實負結論」兩種不同的取徑；[28] 的存在恰恰印證本研究的核心論點——真機評估已非稀缺，**稀缺的是能判定效益是否穩健的統計方法學**。

前述的 RL for AIaaS [23] 與本研究同屬「以強化學習排程 AI 服務型任務」的問題設定，是最貼近本研究、也最可能被質疑「novelty 重疊」的對照組，值得正面處理其區隔：（1）**評估場所與統計嚴謹性**——[23] 完全於自建模擬環境中，以合成任務到達率與 17 個 DNN 模型的離線量測執行時間為輸入評估，並未涉及真實叢集部署，也未處理 sim-to-real 落差、叢集暖機漂移等真實硬體量測特有的混淆因子；本研究的核心方法學貢獻正是把「模擬中可分辨的策略差異能否轉移到實機」系統性地檢驗——以抗漂移交錯輪轉、多 seed 配對統計（seed 層級 one-sample t 檢定）、同一 GPU 分配後端統一重測，並誠實回報「差異不轉移」與「後端本身混淆結論」兩項負結果（§5.3）。（2）**目標函數**——UXP-RL 以最小化平均週轉時間（排隊＋執行）為單一目標；
本研究的 RDSAC 則是**風險敏感**的：雙頭 IQN 評論家對回報分布套用 CVaR 扭曲，**以** p95／p99／SLO 違反率等尾端量**為優化目標**，因為 AI serving 情境下「多數請求正常、少數被拖很慢」的尾端體感往往比平均值更貼近使用者實際感受（§5.2）；惟須誠實指出，此設計目標在本研究 2×1 規模的實測中**尚未轉化為穩健的尾端優勢**——§5.3.3 以 CVaR 的「主場」SLO 違反率重測，RDSAC-cvar 雖為最不差的學習臂，仍未勝過 score，且 naive backfill／fcfs 的極端尾（p99）反而優於 score 與所有學習臂（§5.3.2、§5.3.3），此張力詳見 §6.2、留作未來工作。（3）**失效安全的生產整合**——[23] 的 RL 排程器是排程決策的唯一來源；本研究將 RL 決策整合進 Slurm 既有的 `job_submit.lua` 提交路徑，服務逾時或異常時**靜默回退**至既有 score 啟發式，確保排程核心（slurmctld）永不被研究用元件阻塞——這是把學習式排程放進生產路徑必須解決、但 [23] 未觸及的工程問題（§3.3）。（4）**誠實的負結論**——本研究誠實揭露：在乾淨統一的 DRA 後端上，RDSAC／SAC 等學習式放置在三個真實硬體場景（低負載共置、高負載真實 LLM serving、SLO serving）皆未穩健勝過生產 score 啟發式（§5.3），這與 [23] 及多數既有 RL 排程文獻報告的正面結果形成對比。本文將此差異本身視為方法學貢獻的一部分：RL 排程效益高度依賴評估場景與統計嚴謹程度，一個只在模擬中量測、未做多 seed 配對顯著性檢定的正面結果，未必能在真實部署中複現。DRR [26] 雖已跨出模擬、於真實 Kubernetes 測試床驗證，但其目標仍是聚合碎片率而非尾端 SLO，亦未見多 seed 配對統計或抗漂移設計；本研究的貢獻正補上這塊空缺——把「學習式排程在真實叢集上是否穩健勝過生產基準」的問題，以統計嚴謹的方法學正面回答（即使答案是誠實的「尚未」）。

**與 [26]（DRR）的定位差異。** DRR [26]（ACM SoCC ’25）是本節文獻中少數同時具備「Kubernetes 原生」、「真實測試床驗證」與「學習式」三特徵者，且與本研究同樣面對 GPU 共享叢集的異質性與碎片化，值得獨立處理其區隔：（1）**優化目標**——DRR 以最小化叢集**聚合碎片率**為目標，屬資源佈局層的效率指標；本研究的 RDSAC 則以 CVaR 風險扭曲**針對尾端延遲**（p95／p99／SLO 違反率）設計優化目標，二者優化的是不同層面的量。（2）**暖啟動策略**——DRR 以模仿學習從既有啟發式暖啟動其去碎片化策略；本研究亦以生產 score 啟發式做分數暖啟動（§4.3），兩者在「用既有啟發式引導學習式策略」的概念上高度相近，惟本研究進一步以 RLPD 檢驗實機資料線上微調能否縮小 sim-to-real 落差（§5.3.1，結果為誠實否定）。（3）**統計嚴謹性**——DRR 於實體測試床與大規模模擬回報平均碎片率降低 50%，但未見多 seed 配對統計、多重比較校正或等價檢定；本研究則以 seed 層級配對檢定、Holm-Bonferroni 校正與 TOST 等價檢定判定差異是否穩健，並誠實回報負結論（§5.3.3）。DRR 與本研究因此分屬「碎片率導向的 Kubernetes 原生去碎片化」與「尾端 SLO 導向的 Slurm-on-K8s 失效安全整合」兩種取徑。

### 2.3 異質／邊緣 GPU 排程

本研究的叢集本身即異質且非資料中心等級（RTX 4070＋RTX 3080，皆不支援 MIG／vGPU），這使邊緣與異質 GPU 排程文獻格外相關。Tsenos 與 Kalogeraki [22] 針對缺乏原生虛擬化支援的邊緣 GPU（如 RTX 4090、GTX 1080Ti 等消費級卡）提出一套硬體無關的時空共享機制：為每個行程建立 cgroup、動態調整其「duty cycle」（週期性凍結／解凍佔用 GPU 的時間比例）來實現優先權式與截止期限（laxity）式排程，且無需修改工作負載原始碼即可整合進 TensorFlow、PyTorch、FFmpeg 等既有框架。此工作與本研究處境相近——皆是非資料中心等級、不支援硬體分片的消費級 GPU——但其排程單位是**單節點**上的行程級 duty cycle 調整，不涉及叢集級的佇列、backfill 或跨節點放置決策，亦未整合學習式策略，可視為與本研究互補的節點內機制。Majeed 等人 [24] 則以系統性文獻回顧整理 NVIDIA Jetson AGX 系列邊緣 SoC（CPU＋GPU＋深度學習加速器 DLA＋可程式視覺加速器 PVA＋視訊影像合成器 VIC）上的 DNN 排程器，區分規則式（如 Jedi、CP-CNN、Herald、H2H、HaX-CoNN）與最佳化式（線性規劃、AxoNN、遺傳演算法、以 Z3 SMT 求解器動態重排的 D-HaX-CoNN）兩大類，並整理其記憶體競爭、跨加速器轉移成本與靜態／動態排程的權衡。其排程粒度是**單一 DNN 模型內的層級**（將個別網路層指派給不同硬體加速器），與本研究**工作／任務級**的叢集放置決策不在同一抽象層次，可作為異質邊緣排程景觀（landscape）的引用，界定本研究「叢集批次排程」與此類「模型內加速器排程」研究之間的分工。

### 2.4 雲端原生 GPU 排程生態系

**雲端原生 GPU 排程生態系。** 在 Kubernetes 生態中，數個成熟系統處理 GPU／批次工作負載，但分屬不同抽象層：Kubeflow [12] 負責 ML 工作的生命週期（分散式訓練、超參數搜尋、模型服務），其本身不做排程，而將決策**委派**給批次排程器；Volcano [13] 提供 pod 群組的 gang 排程與 DRF／binpack 等規則式外掛；Kueue [14] 實作 job 級佇列、配額借還（ClusterQueue／ResourceFlavor／Cohort）、fair-share 與 gang 准入，但以暫停（suspend）控制准入、**不負責 pod 放置**；Kubernetes 1.34 起正式釋出（GA）的動態資源分配（Dynamic Resource Allocation, DRA）[15] 則將 GPU 分片、MIG、time-slicing 等以 ResourceClaim／ResourceSlice／DeviceClass **宣告式**地納入 API，成為一等公民。表 1 依抽象層整理這些系統與本研究的定位。

表 1. 雲端原生 GPU 排程系統的層級定位

| 系統 | 所在層 | 機制形態 | 學習式策略 | 尾端／SLO 目標 | GPU 分片 |
|---|---|---|:--:|:--:|:--:|
| Kubeflow [12] | 工作生命週期 | 委派給批次排程器 | ✗ | ✗ | 委派 |
| Volcano [13] | Pod 群組排程 | 規則式 heuristic（gang/DRF/binpack） | ✗ | ✗ | time-slice（無策略） |
| KAI Scheduler [31] | Pod 群組排程 | 規則式（gang／topology-aware／fair-share） | ✗ | ✗ | GPU 共享（fraction／MPS） |
| Kueue [14] | Job 級佇列／配額 | 規則式＋約束求解（不做放置） | ✗ | ✗（fair-share 非尾端） | ResourceFlavor 標記 |
| K8s DRA [15] | 裝置分配**機制** | 約束匹配 | ✗ | ✗ | ✓（宣告式一等公民） |
| **本研究（RDSAC）** | 排序／放置**策略** | **學習式＋風險敏感（CVaR）** | ✓ | ✓（設計上以尾端為目標＊） | MPS-aware 策略 |

＊RDSAC 是表中唯一將尾端／SLO **設為明確優化目標**（CVaR 風險扭曲）的系統，其餘系統連此目標都未設定；惟此設計意圖在本研究 2×1 規模的實測中**未轉化為穩健的尾端優勢**——即使在 SLO 違反率此一 CVaR「主場」指標上，RDSAC-cvar 仍未勝過 score，且 naive backfill／fcfs 的 p99 反而優於 score 與所有學習臂，詳見 §5.3.3、§6.2。

**與既有工作的差異。** 上述系統皆為**規則式或約束求解**：Kueue／Volcano 解的是配額與 gang 准入、DRA 提供的是分片**機制**、Kubeflow 管的是生命週期；沒有任何一個是「**學習式、且以尾端延遲（tail latency）為目標的排序／放置策略**」。本研究正落於此空隙：RDSAC 對回報分布以 CVaR **為目標**嘗試優化 p99／SLO 尾端——這是生產系統皆未設為優化目標的量，即便此設計目標本身在本研究規模下**尚未轉化為實測尾端優勢**（§5.3.3、§6.2）。要強調的是，DRA 提供的是「如何表達要 0.25 張 GPU」的*機制*，而非「該把哪些工作打包、用什麼順序以壓低尾端」的*策略*——因此本研究的學習式策略與 DRA 並非競爭，而是**互補**：一個尾端敏感的策略可在 DRA 之上驅動裝置選擇與准入排序。須誠實指出，「既有 RL 排程研究止於模擬」的說法在 2025–2026 已不再成立（詳見 §2.2）——但這些真機研究**無一報告多 seed 配對統計、多重比較校正或等價檢定**，其結論皆為單向的正向差異。因此本研究的重點不在宣稱 RL 必勝，也不在「率先上真機」，而在**建立一套能在真實異質叢集上、以統計嚴謹方式判定排程策略效益是否穩健的方法學**，並誠實回報其規模條件與負結論。

值得一提的是，本節所引 6 篇 IEEE 相關研究中，多數（DCUDA [19]、Sedighi 等人 [20]、MIG 動態重分割 [21]、邊緣 duty-cycle 排程 [22]、UXP-RL [23]、Jetson 排程回顧 [24]）皆作用於單節點執行期、或評估侷限於未與 Kubernetes 整合的獨立模擬環境；僅 KIS-S [25] 與 DRR [26] 是 Kubernetes 原生系統，但分別位於**自動擴縮**（依流量調整推論副本數）與**去碎片化重排程**這兩個子層，皆非表 1 所比較的「排序／放置策略」層級。這進一步凸顯本研究在雲端原生 GPU 排程生態系中的定位空隙：一個作用於 Slurm-on-Kubernetes 排程／放置層、以學習式策略**針對**尾端 SLO 設計優化目標的失效安全整合（該尾端優勢在本研究規模的實測中未穩健成立，見 §5.3.3、§6.2）；其鄰近文獻或止步於單節點／純模擬（[19]–[24]），或雖已部署於 Kubernetes 但作用於相鄰子層（[25][26]）。與本研究在系統形貌上最接近者為 DRL-MLS [28]（真實異質 Kubernetes 叢集上的學習式放置），惟其排程主體為 Kubernetes 原生而非 Slurm-on-K8s、目標為平均完成時間而非尾端、且未做多 seed 配對統計與等價檢定而僅報告正向結果，故與本研究分屬不同取徑。

## 3. 研究目的與系統架構

### 3.1 研究目的

本研究主要探討以下幾個問題：

- 如何讓多個較小的 GPU 工作共享同一張顯示卡，以提升使用效率。
- 如何利用深度強化學習模型，協助系統決定哪些工作應優先執行，以及應使用哪些硬體資源。
- 如何在模型判斷不可靠時，讓系統自動回到較穩定的基本排程方式。
- 如何以統計嚴謹的方法（抗跑序漂移、多 seed 配對檢定、於同一 GPU 分配後端統一重測）誠實判定學習式排程策略相對生產基準的效益是否穩健。

彙整而言，本研究的研究目標是：在異質、MPS 卡內共享的 GPU 叢集中，尋找兼顧使用效率與工作完成表現的 GPU 與 MPS 配置決策方式，並以統計嚴謹的方法誠實檢驗其相對生產基準的效益。

### 3.2 整體架構

平台分為兩個鬆耦合層：**基礎設施層**（Slurm on Kubernetes）與**排程研究層**（模擬器＋深度強化學習）。

叢集以 k3s 部署，控制節點（RTX 4070）兼任 control-plane，工作節點（RTX 3080，算力約前者 0.25×）為異質 GPU 來源。Slurm 控制器、登入節點與 GPU 工作節點皆以容器化 StatefulSet 部署，GPU 經 NVIDIA device plugin 與 MPS control daemon 暴露為可分片資源。

**為何 Slurm-on-Kubernetes（而非純 K8s 排程器）。** 一個合理的質疑是「既然已在 K8s 上，為何不直接用 Kueue＋DRA＋自訂 kube-scheduler plugin，而要疊一層 Slurm？」本研究選擇 Slurm-on-K8s 有三個理由：（1）**成熟的 HPC 排程語意**——backfill、multifactor 優先權、gang、`gres/mps` 卡內分片皆是 Slurm 開箱即用且經生產驗證的一等公民；在 K8s 側要湊齊等價能力需 Kueue＋Volcano＋DRA 多元件拼裝，且 DRA 於本研究進行時甫 GA（K8s 1.34），生態未穩。（2）**K8s 提供部署與生命週期、Slurm 提供排程核心**，兩層鬆耦合、各司其職：k3s 負責容器化、網路、儲存（NFS RWX）、可觀測性，Slurm 負責佇列與放置決策；這讓平台既可攜（Helm 一鍵部署於異質節點）又保有 HPC 級排程。（3）**研究載具**——`job_submit.lua`／slurmrestd 是穩定、非侵入的策略注入點（§3.2），可在**不 fork slurmctld、不改 kube-scheduler** 的前提下熱插拔學習式策略並失效即回退；相較於維護一個自訂 scheduler plugin，這大幅降低了研究迭代成本。與生態的關係上，本研究的學習式策略與 K8s DRA **互補**（§2.4）：DRA 給的是分片*機制*，本研究給的是尾端敏感的排序／放置*策略*，未來可在 DRA 之上驅動裝置選擇（§6）。

**與 Slinky 的關係。** 另一個常見追問是：「SchedMD 官方已推出 Slinky [27]——一個以 Kubernetes operator 把 Slurm 各元件 pod 化部署的官方 Slurm-on-K8s 專案——為何不直接採用它？」Slinky 定位為**部署基座**：其 operator 負責 Slurm 元件（`slurmctld`、`slurmd`、`slurmdbd` 等）的 pod 化生命週期管理，但排程核心仍是**未經修改的 vanilla Slurm**，沿用 backfill／multifactor（§4.1）等既有排程演算法，並未內建任何學習式或風險敏感的放置策略。本研究的貢獻——失效安全 RL 決策端點（§3.3）與抗漂移、多 seed 配對的 sim-to-real 評估方法學（§5.1）——作用於**策略層**，與 Slinky 所在的**部署層**在架構上正交，理論上可移植到 Slinky 之上（以其 operator 取代本研究手刻的 Helm chart 部署基礎設施）。本研究選擇自行以 Helm 部署，是為了在研究階段對 `job_submit.lua`、slurmrestd 版本與叢集拓樸保有完全控制，作為研究控制變因，而非否定 Slinky 的工程價值——**連官方參考的 Slurm-on-K8s 部署方案都僅使用 vanilla Slurm 排程**，這進一步凸顯本研究在其上疊加學習式尾端放置策略層的定位空隙，與 §2.4 對 Kubeflow／Volcano／Kueue／DRA 的定位討論呼應。

### 3.3 失效安全的 RL 整合

整合點為 Slurm 的 `job_submit.lua`：工作提交時，掛鉤 `rl_hook.lua` 以 HTTP 呼叫 RL 推論服務的 `POST /decide`，取得放置／優先權建議；若服務逾時或異常，掛鉤**靜默回退**至既有啟發式分數排程（以 MPS 適配、VRAM 適配為主，短工作優先因子在生產環境預設關閉，細節見 §4.2）。此設計確保 slurmctld 永不被第三方服務阻塞，使研究用的 RL 元件可安全運行於生產路徑。

生產部署的 RL 作動有兩條路徑，皆為失效安全（fail-safe）：（1）**優先權微調**——`job_submit.lua` 呼叫 `/decide`，僅提升被選中工作的佇列優先權（`select/cons_tres` 與 GRES 仍決定實際落點）；（2）**顯式節點綁定（explicit placement）**——`/act` 回傳節點選擇 `(node_j, gpu_k)`，於**提交時**將該節點寫入工作的必要節點（`ReqNodeList`），Slurm 遂將工作排到 RL 選定節點、MPS 由 GRES 強制。本研究的實機放置實驗（§5.3）即以路徑 (2) 為對象——由評估 harness 呼叫 `/act` 後以 `sbatch -w` 提交——並於本平台 2×1 叢集驗證其正確釘選（`sbatch -w` 與 `scontrol update ReqNodeList` 皆能把 held／pending 工作釘到指定節點）。將此提交時綁定接入 `job_submit.lua`（呼叫 `/act` 後設 `job_desc.req_nodes`）即為 RL 顯式放置的生產路徑。平台另實作一個非同步 **placement controller**（`services/rl_scheduler/placement_controller.py`，以 slurmrestd hold→pin→release 對已提交 held 工作事後釘節點）作為不需改提交端的替代，惟其經 slurmrestd v0.0.37 job-update 寫入節點約束之實際生效仍在硬化中（實機測試觀察到該 REST 欄位未被套用），故**目前評估與生產皆以提交時節點綁定為準**。`/act` 若 abstain（如 checkpoint 拓樸 ≠ 實機）則 no-op，退回 Slurm 原生放置。

### 3.4 模擬器與強化學習環境

離散事件模擬器以「提交／結束」事件驅動，建模 Node → GPU → MPS 槽的階層資源與異質算力。其上以 Gymnasium 介面封裝為 RL 環境，正式定義如下：

- **狀態（State）**：佇列前 K 個工作的特徵（GPU 數、MPS 需求、等待時間、SLO 緊迫度、工作類別、GPU 型別 one-hot 等），於 2×1 拓樸下觀測維度為 166。
- **動作（Action）**：「放置於某節點某 MPS 槽」或「暫不放置」的離散選擇，即節點×MPS 配額的聯合空間。
- **獎勵（Reward）**：以完成時間為核心的 −JCT 訊號，搭配位能獎勵塑形（potential shaping）提供密集回饋（§4.3.2）；於 §5.3.2 的多元真實工作負載實驗中，進一步擴充為涵蓋 GPU 利用率的多目標獎勵。

## 4. 排程技術

### 4.1 Slurm 內建排程演算法

本平台以 Slurm 原生排程能力作為穩定底座，而非重寫排程核心。`SchedulerType=sched/backfill` 允許在不延後高優先權工作的前提下，讓資源需求較小、執行時間較短的工作提前插隊執行，緩解大工作長期佔用資源造成的閒置；`PriorityType=priority/multifactor` 依工作年齡、規模、partition、QoS 等因子計算靜態優先權，決定佇列排序；`SelectType=select/cons_tres`（`CR_Core`）以 TRES（Trackable RESources）粒度追蹤 CPU／GPU／MPS 資源，並將 GPU 與 MPS 使用量計入 accounting（`gres/gpu`、`gres/mps`）。卡內資源切分透過 `gres/mps` 這個 GRES 型別表達：單一 GPU 依 MPS 槽（每卡 4 槽，對應 25／50／75／100％）分割給多個工作共享，由 Slurm 的 GRES 機制強制配置與計費。

本研究以 backfill＋multifactor 這組 Slurm 原生設定作為評估基準之一，並另納入更保守的**嚴格 FIFO**作為「未加任何智慧排程」的下界對照，用以檢驗啟發式與學習式策略相對 Slurm 開箱即用能力是否確有加值。

### 4.2	啟發式排程策略

score 是 submit-time 的加權線性啟發式，於 `job_submit.lua` 提交當下計算，分數越高代表該工作越值得優先排程；其值以 `scoreGain`（預設 `1000`）換算成 priority delta，疊加於 `priority/multifactor` 之上：

```text
score(J, P) = α·f_mps_fit(J, P) + β·f_vram_fit(J, P) + γ·f_topology(J, P)
            − δ·f_fragmentation(J, P) + ε·f_pred_runtime(J)

score = clamp(score, 0, 1)
priority_delta = round(scoreGain × score)
```

**生產僅啟用三個因子，其餘為停用／保留欄位。** 下表列出 score 的完整因子與 chart 預設係數；須先點明：真正在生產叢集運作的只有 `f_mps_fit`（α）、`f_vram_fit`（β）與 `f_fragmentation`（δ）三項，`f_topology`（γ=0，保留欄位、回傳中性值 0.5）與 `f_pred_runtime`（ε=0，SJF 短工作因子、需 predictor 服務）在上線環境均為停用，故生產 score 是 runtime-blind 的純裝箱啟發式（完整說明見本節末「生產部署現況」）：

| 因子 | 意義 | 係數（預設） |
|---|---|---|
| `f_mps_fit` | 衡量工作 MPS 請求與單 GPU MPS 容量（預設 100）的配適程度——bin-pack 式的卡內裝箱，愈不浪費槽位者分數愈高 | α = 0.40 |
| `f_vram_fit` | 依工作 VRAM 需求（`vram-*g` constraint）挑選可容納的最小 VRAM tier，避免小顯存工作佔用大顯存卡 | β = 0.20 |
| `f_fragmentation` | 懲罰最容易留下 MPS 碎片的請求（`4x(1−x)`，x 為 MPS 佔比；x=50% 時碎片代價最高） | δ = 0.20（懲罰項） |
| `f_pred_runtime` | 依 runtime predictor 預測執行時間換算，SJF（Shortest-Job-First）式地讓短工作取得較高分數，需 predictor 服務可用 | ε = 0.00（預設關閉） |
| `f_topology` | 保留欄位，回傳中性值 0.5，不影響上線結果 | γ = 0.00 |

若 weight-tuner（以 UCB1 bandit 在離散 arm 空間中線上調整 `(α, δ, ε)`）啟用，`job_submit.lua` 會在 Lua plugin 載入時以 `GET /weights` 覆寫這三個係數；`β` 恆固定，`γ` 維持 chart 設定不受 tuner 影響。

**生產部署現況（誠實揭露）。** 目前實際上線的係數為 α=0.40、β=0.20、δ=0.20，但 **γ=0、ε=0**——SJF 短工作優先因子在生產環境**未啟用**，score 是 runtime-blind 的；同時 runtime predictor 與 weight-tuner 服務**皆未部署**於生產叢集，三係數為 chart 靜態預設值，並非 UCB1 動態調校。故生產 score 實際上僅為 `clamp(0.4·f_mps_fit + 0.2·f_vram_fit − 0.2·f_fragmentation, 0, 1)`——一個純粹依 MPS／VRAM 配適與碎片懲罰運作的裝箱式啟發式，這也是全文「score 最穩健」核心賣點的具體機制來源。

### 4.3 深度強化學習策略

#### 4.3.1 SAC 背景

Soft Actor-Critic（SAC）是一種最大熵（maximum entropy）off-policy actor-critic 演算法：除了最大化期望回報外，目標函數額外納入策略熵 $\mathcal{H}(\pi(\cdot|s))$ 作為正則項，鼓勵策略在維持高回報的同時保留探索所需的隨機性，避免過早收斂到次佳的確定性策略，並提升對超參數與初始化的穩健性。原始 SAC 設計於連續動作空間，以高斯策略搭配 reparameterization trick 取樣；Christodoulou [6] 提出的離散動作變體則以 categorical 策略取代高斯策略、以期望而非取樣估計熵項，使最大熵框架得以套用於本研究「選擇節點／MPS 槽」這類離散動作排程問題。RDSAC 即以此離散 SAC 為基礎，進一步將評論家由純量 Q 值擴展為分布式評論家（§4.3.2）。

#### 4.3.2 RDSAC 演算法

決策策略為自行整合的 **discrete 分布式 SAC**（本文稱 RDSAC）：雙頭 IQN 評論家分別建模回報分布（reward 回報 $Z_R$ 與 entropy 回報 $Z_H$），以 quantile Huber loss 學習，搭配 twin-Q、軟更新（τ=0.005）與遮罩式 categorical actor。風險敏感性透過在 actor 目標與動作價值上對回報分布套用 CVaR 扭曲 $\rho[Z_R]$ 達成，對應排程中的長尾 runtime／慢節點（straggler）風險。訓練採優先經驗回放（PER）、n-step 回報、分數暖啟動與位能獎勵塑形。RDSAC「風險敏感分布式 SAC」之名承襲自 Ma 等人以回報分布做風險敏感優化的 DSAC [16]；惟本研究為**離散動作**、雙頭 IQN 的自組版本，是離散 SAC [6]、IQN 分布式評論家 [7] 與 CVaR 風險量度的組合，非 [16] 連續控制版本的 1:1 重現。須留意另有**同名但不同**的 Duan 等人 DSAC [17]（將回報建模為單一高斯、以抑制 Q 值高估為目標、風險中立、連續控制），與本研究的風險敏感取向不同，不宜混淆。

## 5. 實驗與評估方法

本節依序說明評估環境與資料集（§5.1）、評估指標與其設計動機（§5.2），以及跨模擬與三個真實硬體場景的主要結果（§5.3）；所用基準線涵蓋 Slurm 原生排程（FCFS、backfill、multifactor）、生產啟發式 score，以及本研究的學習式策略（SAC、RDSAC-mean、RDSAC-cvar、RLPD 微調）。

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

以下轉入真實叢集量測：表 3、表 4 採 AI 伺服器合成負載於真實 2×1 叢集、未開啟 MPS 卡內共享的**等待主導 regime**（與 §5.3.1 起開啟 MPS 分數共置的場景相對）。**跑序漂移會污染單趟排名。** 在真實叢集上，單趟（block design）量測一度顯示 FCFS「顯著贏」score 達 +5.0%。然而其改善幅度與「跑第幾位」完美單調相關（表 3）：同一個 FCFS，跑最後一位時 +5.0%、跑最先時則僅 +0.5% 甚至轉為 −0.4%（不顯著）。此為叢集隨時間暖機的漂移假象，而非排程器差異。

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

#### 5.3.1 實機 cuBLAS 評估（低負載共置：策略空間平坦）

真正能觸及這座**異質**叢集放置槓桿的評估，必須讓卡內共享（NVIDIA MPS）與計算異質性反映到 JCT。為此以**真 cuBLAS（`gpu_workload`）＋ MPS 分數共置**（Poisson 到達、mps-oversub **1.0** 的低負載、MPS 分桶 25／50／75／100）跑實機配對 A/B（2×1、node order 4070→3080、提交時 `-w` 顯式放置、drift-robust interleave、**8 seed**、每 seed n_jobs=30×3 rounds、σ=1.0），並以**正確分析層級——seed**——的 one-sample t 檢定每臂 ΔJCT% 是否顯著異於 score，避免把 job 當獨立單位的偽重複。

為了檢驗「用實機資料線上微調能否縮小 sim-to-real 落差」，收集了 181 筆真實 transitions，拿來做 RLPD 微調。

從表 5 可以發現，四個學習式策略**一致略差**，其中 RDSAC-cvar 達 seed 顯著，而 RLPD 微調效益不大。無論學習式或 Slurm 內建設定都沒有贏過啟發式，且彼此皆落在 ±5% 內，代表此 regime 的排程策略空間近乎是平的。


表 5. 實機 cuBLAS 工作負載評估

| arm | JCT(s) | p99(s) | CVaR(s) | ΔJCT% | Δp99% | ΔCVaR% | seed 為正 | seed-t p |
|---|--:|--:|--:|--:|--:|--:|:--:|--:|
| score | 6.8±0.7 | 22.1±1.3 | 14.9±1.7 | —（基準） | — | — | — | — |
| fcfs | 6.8±0.6 | 21.9±1.4 | 14.8±1.8 | −0.1±3.0 | +1.3±2.0 | +0.9±2.1 | 2/8 | 0.891 |
| backfill | 6.8±0.6 | 22.2±1.3 | 14.8±1.8 | −0.1±2.1 | −0.4±3.8 | +0.8±1.8 | 3/8 | 0.876 |
| RDSAC-mean | 7.0±0.8 | 24.0±3.7 | 15.8±2.5 | −1.7±3.0 | −8.4±16.1 | −5.6±8.3 | 1/8 | 0.168 |
| RLPD | 7.1±0.9 | 23.9±3.8 | 15.9±2.4 | −3.6±4.8 | −8.1±16.9 | −7.0±9.3 | 2/8 | 0.071 |
| RDSAC-cvar | 7.2±0.8 | 25.2±4.5 | 16.2±2.9 | −4.6±4.9 | −14.1±19.5 | −8.3±11.2 | 1/8 | 0.033 |
| SAC | 7.2±0.7 | 24.3±3.6 | 16.1±2.2 | −5.7±7.1 | −10.1±15.9 | −8.7±11.6 | 2/8 | 0.058 |

#### 5.3.2 實機多元真實工作負載評估

為了更貼近生產的多元真實 AI workload 組合，改用混合工作負載。由 30% BERT 推論、30% ResNet 訓練、30% Qwen2.5 微調、10% 矩陣運算四類真實 job 依比例混合而成，同時涵蓋推論、訓練與生成式微調三種計算特徵。並將學習式的獎勵函數由純 −JCT 改為**多目標獎勵**（−JCT ＋ GPU 利用率）。

從表 6 可以發現，RDSAC-cvar 統計追平 score，是六個策略中唯一未落後者，並且發現 backfill 顯著落後。

表 6. 實機混合工作負載評估（Holm p 為跨臂族多重比較校正後 p 值，粗體 = 通過校正之顯著差異）

| arm | 平均JCT(s) | P95(s) | P99(s) | Makespan(s) | GPU利用率 | Slowdown | SLA違反率 | ΔJCT% vs score | Holm p |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| score | 25.3±5.4 | 51.1±18.4 | 60.1±18.7 | 140.2±17.2 | 1.05±0.23 | 5.87±1.97 | 0.93±0.08 | —（基準） | — |
| RDSAC-cvar | 25.2±6.0 | 53.4±23.2 | 67.4±30.5 | 142.5±21.5 | 1.06±0.27 | 5.69±2.19 | 0.95±0.06 | +0.1±14.3 | 0.982 |
| SAC | 27.0±4.3 | 61.1±17.5 | 76.1±14.9 | 147.4±21.9 | 1.06±0.18 | 6.45±1.88 | 0.95±0.05 | −8.8±18.1 | 0.627 |
| fcfs | 27.9±5.9 | 46.2±13.2 | 50.4±14.3 | 146.8±20.1 | 1.07±0.21 | 7.03±2.66 | 0.96±0.07 | −10.8±9.6 | 0.062 |
| RDSAC-mean | 27.8±5.5 | 63.1±22.1 | 86.3±35.6 | 152.0±32.6 | 1.09±0.25 | 6.51±1.73 | 0.95±0.05 | −12.1±24.8 | 0.627 |
| backfill | 28.4±5.6 | 47.6±12.6 | 52.4±11.8 | 149.0±19.5 | 1.09±0.21 | 6.94±2.34 | 0.96±0.07 | −12.8±8.6 | **0.021** |

> 六臂的 GPU 利用率全部聚集在 1.05~1.09、彼此不可區分，無一臂顯著偏高或偏低。可能原因是利用率項權重（mo_w_util=0.05）過輕，不足以在策略中產生可觀測的裝箱行為改變。

SAC／RDSAC-mean 的 P99（76.1s／86.3s）明顯劣於 score（60.1s），顯示其風險中性目標放大了少數 straggler；RDSAC-cvar 的風險趨避設計則把 P99 拉回 67.4s，且其 slowdown 甚至略優於啟發式，RDSAC-cvar 贏在尾端風險。

**stock Slurm 尾端反而最輕，代價是平均與 slowdown 較差。** fcfs／backfill 的 P95（46.2s／47.6s）為六臂最低——未加裝箱的 vanilla Slurm 不會製造 MPS 共置干擾，也就不會有裝箱帶來的尾端放大；但其平均 JCT／slowdown 明顯較差（slowdown 6.94–7.03，六臂最高）。此為典型的「均值 vs 尾端」取捨：score／RDSAC-cvar 用裝箱換取更好的平均與 slowdown，代價是尾端稍重；stock Slurm 反之。

這強化本文核心命題：**排程結論高度依賴評估場景與 GPU 分配後端**，而學習式放置的實機效益遠比單一場景所暗示的脆弱。本文據此**不宣稱學習式優越**，改以方法學（抗漂移、多 seed、seed 層級配對、同後端統一重測、存活者偏差消除）與誠實的場景／後端依賴負結論為主要貢獻。

#### 5.3.3 統計穩健性複核：多重比較校正、等價檢定與檢力

§5.3.1–5.3.2 對每個場景同時以 one-sample t 檢定 5–6 個臂對 score，單看個別 p 值會高估顯著性，正確工具是**多重比較校正、等價檢定與檢力分析**。我們對既有 per-seed 資料補做以下複核，Holm-Bonferroni 跨臂族校正、ΔJCT% 的 bootstrap 95% CI、對 ±5% 實務等價帶的 TOST 等價檢定，並報告 n=8 的最小可偵測效應（MDE）。結果見表 7。

表 7. 統計穩健性複核（Holm p＝跨臂族多重比較校正後；CI＝bootstrap 95%；±5% 等價＝90% CI 是否落在 ±5% 帶內＝TOST 證實等價）

| 場景 | arm | ΔJCT% | Holm p | boot 95% CI | ±5%等價 |
|---|---|--:|--:|--:|:--:|
| **cuBLAS** 低負載 | SAC | −5.7 | .288 | [−10.5,−1.3] | 否 |
| MDE≈5.1% | RDSAC-mean | −1.7 | .503 | [−3.5,+0.4] | **是** |
|  | RDSAC-cvar | −4.6 | .200 | [−7.9,−1.6] | 否 |
|  | RLPD | −3.6 | .288 | [−6.6,−0.4] | 否 |
|  | fcfs | −0.1 | 1.00 | [−1.9,+1.9] | **是** |
|  | backfill | −0.1 | 1.00 | [−1.5,+1.3] | **是** |
| **aimix** 高負載 | SAC | −8.8 | .627 | [−21.7,+1.8] | 否 |
| MDE≈18.7% | RDSAC-mean | −12.1 | .627 | [−28.7,+3.1] | 否 |
|  | RDSAC-cvar | +0.1 | .982 | [−10.1,+8.4] | 否 |
|  | fcfs | −10.8 | .062 | [−16.9,−4.8] | 否 |
|  | backfill | −12.8 | **.021** | [−18.8,−7.7] | 否 |

從表 7 可以發現：

1. 多重比較校正移除了大部分「學習式較差」的顯著宣稱。cuBLAS, aimix 場景校正後沒有任何學習臂顯著，故「學習式**顯著**較差」在兩場景皆不成立，誠實降為「未穩健勝出」
2. 唯一正面主張通過校正：在混和工作負載中，啟發式顯著贏過 Slurm backfill。
3. cuBLAS 場景 RDSAC-mean／fcfs／backfill 經 TOST 證實**在 ±5% 內與啟發式統計等價**，故「策略空間平坦」是證實的統計等價。而 aimix 場景則因變異大得多，在 8 seeds 下經 Holm 校正下除 backfill 外均不顯著，更多 seed 能做的是**提升檢力以釐清此負向效應是否成立**

## 6. 結論與未來工作

本節依序回答研究問題（§6.1）、說明可能削弱結論的限制範圍（§6.2），並展望未來工作（§6.3）。

### 6.1 結論

本研究設計並實作了一套以 Kubernetes 為部署基座、Slurm 為排程核心、整合 GPU MPS 強化學習策略的 AI 伺服器 GPU 排程平台。核心發現是在小叢集中，學習式放置在三個場景、跨兩類指標**皆未穩健勝過**啟發式排程、亦未勝過 Slurm 內建排程。

唯一穩健的正面觀察是 **啟發式的 bin-pack／SJF 因子在高負載下勝過 Slurm 內建的 cons_tres 放置**，顯示手調啟發式相對「數 GPU」式放置確有加值。

### 6.2 威脅與限制

作為一篇以方法學與誠實負結論為主軸的研究，本節明列可能削弱結論的因素及其處置狀態：

- **規模／小叢集限制：** 本研究主要評估規模為 2×1；結論不宜外推至數十至數百 GPU 的生產叢集，「效益隨規模浮現」為尚未證實的假設。此外 1×1／2×1／2×2 的觀測維度與動作空間不同，checkpoint 不相容、須各自重訓，跨尺度絕對數值不宜直接相減。尾端指標（p99／CVaR）在此規模下亦為每 seed 20–30 個完成 job 的小樣本估計、變異極大（§5.3.2），本文僅作定性觀察，不作統計宣稱。

### 6.3 未來工作

1. 擴展至更大、更高競爭的叢集以檢驗「價值隨規模浮現」假設
2. 擴大實機微調的資料規模。惟 181 筆真實 transition 不足以翻盤，故後續需收集遠更大量的實機資料或加入 on-policy 修正

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

[27] SchedMD, "Slinky: Slurm in Kubernetes," https://github.com/SlinkyProject, 2024.

[28] Y. Dong, X. Zheng, X. Pan, and D. Liu, "A reinforcement learning-based approach for scheduling machine learning training tasks in heterogeneous Kubernetes clusters," *Future Generation Computer Systems*, 2026.

[29] Y. Wang, Y. Hu, A. Klimovic, X. Zhang, Y. Wen, G. Sun, and J. Lin, "Semantic-Aware Scheduling for GPU Clusters with Large Language Models," *arXiv:2510.03334*, 2025.

[30] S. Dongare, R. I. S. Khan, H. Albahar, N. Zhao, D. Melendez Maita, and A. R. Butt, "Hybrid Learning and Optimization-Based Dynamic Scheduling for DL Workloads on Heterogeneous GPU Clusters," in *Proceedings of the 2025 ACM Symposium on Cloud Computing (SoCC)*, 2025. arXiv:2512.10271.

[31] NVIDIA, "KAI Scheduler: A Kubernetes-Native GPU Scheduler for AI Workloads," https://github.com/NVIDIA/KAI-Scheduler, 2025.
