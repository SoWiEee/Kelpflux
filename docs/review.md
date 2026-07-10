# Kelpflux 研究審核報告

> **評估時間：** 2026-06-29
> **評估快照：** main @ `5895ee0`（paper.md §2 加入雲端原生排程生態系層級對照之後）
> **評估範圍：** 研究定位與差異化、與雲端原生排程生態系（Kubeflow／Volcano／Kueue／K8s 1.34 DRA）的關係、可發表性、下一輪改進方向。
> **評估面向（綜合，不分段）：** DevOps、AI Infra、HPC、IEEE 審稿人、Kubernetes maintainer。

## 執行摘要

本研究的平台（Slurm-on-K8s + MPS + 失效安全 RL 整合）工程上扎實，sim-to-real 評估方法學（抗漂移、多 seed 配對、尾端指標）是真正的方法學貢獻。但若以「下一輪 review 能不能過」為標準，**目前最致命的結構性弱點是：RDSAC（乃至任何學習式臂）在整篇論文裡從未在任何地方贏過 baseline**——§4.2 的「正面結果」是 heuristic vs heuristic（不含 RDSAC），唯一出現 RDSAC 的 §4.2 表 4 是顯著小輸。這使「風險敏感學習式策略」這條主貢獻線目前**沒有任何支撐證據**，reviewer 會直接質疑複雜度的正當性。

好消息：補上證據所需的實驗**大多已在 `eval-writeup.md §4.4` 做出**（stochastic sim 下 RDSAC 的風險機制顯著生效），只是（a）尚未折進 TANET `paper.md`，（b）關鍵的 mean-vs-cvar 比較仍有單一訓練 seed 的可信度警訊。真正全新的只有「規模交叉曲線」。

> **更新（2026-07-01）：** P0 的 #1/#2/#3 已全部執行完畢（見 §四）。結果**推翻了 §4.4 的樂觀結論**——多 seed（100k）下 RDSAC 未穩健勝過 SAC 或 score，§4.4 的優勢是單 seed 運氣。此「sim 救援」失敗反而**強化**論文的誠實負結論主軸；已折入 `paper.md` §4.3（表 5）／§4.4 與貢獻 #3。下方 §三仍記錄 review 當下的原始判斷。

---

## 再次審核（2026-07-10）

> **評估快照：** main @ `956b154`（§4.2.1–4.2.4 實機評估、RLPD 負結論、DRA 遷移、live Slurm 原生 baseline 之後）
> **範圍：** 2026-07-01 P0 之後的新進展、DRA 遷移對論文的影響、目前最深的結構性問題與下一輪改進。

### 執行摘要（2026-07-10）

實機評估線大幅充實：§4.2.1（低負載 cuBLAS，學習式小勝 cvar +4.5%）、§4.2.2（高負載 LLM serving，學習式顯著落後 −10～16%）、§4.2.3（RLPD 微調未翻盤）、§4.2.4（統一 DRA 重測 + Slurm 原生 baseline）；平台亦完整遷移至 K8s DRA。**上次審核最大缺口 #4（強 baseline）已在 live 層級補上**，並產出一個更鋒利的正面觀察：**score 啟發式連 naive Slurm（FCFS/backfill）都贏**。

但同一批數據把上次的「最致命問題」推到更尖銳的位置——**在乾淨的統一 DRA 重測（§4.2.4）下沒有任何學習臂勝過 score，唯一穩定的正面結果來自 score 這個手調啟發式**。並帶出兩個新問題：（a）**尾端反轉**——naive backfill/fcfs 的 p99/CVaR 反而優於 score 與所有學習臂，直接侵蝕「風險敏感學習式優化尾端」這條最鋒利的差異軸；（b）**DRA 後端混淆**——DRA MPS 使絕對 JCT 減半（39.9→18.4s），§4.2.1/4.2.2（device-plugin）與 §4.2.4（DRA）不同後端、不可並比。

### 一、自上次審核以來已解決／推進

| 上次項目 | 現況 |
|---|---|
| #4 強 baseline | ✅ **live 層級完成**：§4.2.4 加入 Slurm 原生 FCFS/backfill 實機臂（reconfig slurmctld + restore-trap），統一 DRA 8 seed。發現 score 亦勝 naive Slurm。 |
| #5 eval↔prod placement gap | ✅ **大致解決**：生產與評估皆走提交時 `-w`/ReqNodeList 顯式放置（§3.2/§4.6）。 |
| #6 serving 真實度 | 🟡 **部分**：§4.2.2 用真實 Qwen LLM serving + Poisson 到達 + MPS buckets；SLO 分級仍未做。 |
| #7 DRA 互補 | 🟡 **平台已在 DRA 上**（GPU 分配全走 `gpu.nvidia.com` ResourceClaim），但 RDSAC 仍只驅動提交時放置，未接成 DRA device-selection／Kueue admission-ordering 的 policy plugin。 |
| RLPD 微調救援 | ✅ **已檢驗、負結論**（§4.2.3，181 筆 transition 不足）。 |

### 二、目前最深的問題

1. **（致命，未解）學習式仍從未穩健勝出；唯一正面是手調啟發式。** 統一 DRA 重測下 score 最佳、所有學習臂為負；唯一穩定勝 baseline 的是 **score 啟發式贏 naive Slurm**——贏的是 **bin-pack/SJF 啟發式，不是 RDSAC/RL**。§4.2.1 的 +4.5% cvar 是低負載 cuBLAS、且在已被 DRA 取代的**舊後端**。→ 寫作上須明確重定位：**貢獻是「方法學 + 誠實的場景依賴負結論 + score 相對 vanilla Slurm 的加值」，RDSAC/RL 是被嚴謹研究後在此規模/場景發現未勝的對象**，非賣點。

2. **（新，對主打差異軸有殺傷力）尾端反轉。** §4.2.4 表 4-4：naive backfill Δp99 **+19.7%**、ΔCVaR **+6.8%**；fcfs Δp99 +16.0%——**naive Slurm 尾端比 score 與所有學習臂都好**。上次審核把「學習式 + 風險敏感（CVaR 優化尾端）」列為最鋒利差異軸；但實機顯示 **score 是以較緊的平均換較長的尾，而尾端最好的其實是最笨的 backfill**。→ 兩條路：(a) 誠實把 risk-sensitive 降為「完成率穩定器」（§4.3 已部分如此）、不再宣稱優化 serving 尾端；(b) 深究 backfill 尾端優勢、檢驗 risk-sensitive RL 能否在尾端真的勝 backfill。尾端估計變異極大（±20~53%），任一宣稱都需更多 seed。

3. **（新，威脅有效性）DRA 後端混淆未入 §4.6。** §4.2.1/4.2.2（device-plugin）與 §4.2.4（DRA，JCT 減半）不同後端；§4.6 未提，且 §5 開頭仍以 §4.2.1 的 +4.5% 起頭，與 §4.2.4 的「無臂勝出」並列讀來矛盾。→ (a) §4.6 立即補一條 threats；(b) 理想上把 §4.2.1/4.2.2 也在 DRA 下重測，使全篇同後端。

4. **（承上）§5 與未來工作已 stale。** §5 仍以 +4.5% 正面結果開頭；未來工作仍列 `gres/shard`（本叢集用 gres/mps、N/A）與「補強 baseline」（§4.2.4 已做）。→ 重寫 §5 以場景依賴 + 誠實負結論為主軸，正面結果界定為「窄、低負載、舊後端」，並更新未來工作清單。

### 三、更新後的改進優先級（2026-07-10）

| # | 改進 | 解決的質疑 | 工作量 | 優先 |
|--:|---|---|:--:|:--:|
| A | **重定位貢獻與 §5**：主軸＝方法學 + 場景依賴負結論 + score 啟發式加值；正面結果界定範圍；未來工作去 stale | 「學習式從不贏，複雜度正當性？」 | 低（寫作） | **P0** |
| B | **§4.6 補 DRA 後端 threats + 理想重測 §4.2.1/4.2.2 於 DRA** | 「跨後端不可比、§5 開頭與 §4.2.4 矛盾」 | 中 | **P0** |
| C | **處理尾端反轉**：降 risk-sensitive 為穩定器，或深究 backfill 尾端優勢並檢驗 RL 能否在尾端勝 backfill | 「主打優化尾端，但最笨的 backfill 尾端最好」 | 中 | **P1** |
| D | **RDSAC 接成 DRA/Kueue policy plugin PoC**（platform 已在 DRA、可行性大增）：把「策略層 × DRA 機制層互補」從論述變 demo | novelty 不足、「為何不落進生態系」 | 高 | **P1（最高 novelty 槓桿）** |
| E | **serving SLO 分級**（延續 §4.2.2）：多層 SLO + tail-aware 目標 | SLO 模型仍淺 | 中 | P2 |
| F | **更大叢集**（scale 交叉，硬體受限） | 「value at scale 仍是斷言」 | 高 | P2 |

**定位（2026-07-10）：** 負結論主軸更穩、baseline 缺口已補（連 naive Slurm 都納入且 score 贏它）。下一輪最高價值＝ **A/B（寫作重定位 + DRA 後端一致性）** 鞏固可信度，以及 **D（RDSAC 接 DRA/Kueue policy PoC）** 把「平台已在 DRA 上」的事實轉成 novelty。**C（尾端反轉）** 是對主打差異軸的新威脅，必須誠實處理。

---

> 以下為 **2026-06-29／07-01 版**審查（P0 前後），保留為對照。

## 一、目前領域已有的做法（state of practice）

| 系統 | 所在層 | 機制形態 | 學習式 | 尾端／SLO 目標 | GPU 分片 |
|---|---|---|:--:|:--:|:--:|
| Kubeflow | 工作生命週期 | 委派給批次排程器 | ✗ | ✗ | 委派 |
| Volcano | Pod 群組排程 | 規則式（gang／DRF／binpack） | ✗ | ✗ | time-slice（無策略） |
| Kueue | Job 級佇列／配額 | 規則式＋約束求解（不做放置，用 suspend 控制准入） | ✗ | ✗（fair-share 非尾端） | ResourceFlavor 標記 |
| K8s 1.34 DRA（GA） | 裝置分配**機制** | 約束匹配（scheduler plugin） | ✗ | ✗ | ✓（宣告式一等公民） |
| Slurm（原生） | HPC 批次排程 | backfill＋multifactor＋fairshare＋`gres/shard` | ✗ | ✗ | ✓（shard） |
| **本研究（RDSAC）** | 排序／放置**策略** | **學習式＋風險敏感（CVaR）** | ✓ | ✓（直接優化尾端） | MPS-aware 策略 |

**關鍵觀察：**
- K8s 生態（Kueue／Volcano／DRA／Kubeflow）與 HPC（Slurm）的成熟排程器**全是規則式或約束求解**，沒有一個優化回報分布的尾端（p99／CVaR）。
- DRA 在 1.34 GA 後把 MPS／MIG 分片變成宣告式一等公民——這**侵蝕了「MPS-aware」當核心賣點的空間**，必須改以「策略層」而非「機制層」立論。
- HPC reviewer 會知道 Slurm 本身很強（backfill／multifactor／shard）。「我們贏 FCFS」沒有說服力；對照組必須升級。

## 二、本系統的特色與研究差異（distinctiveness）

站得住、且強弱分明的三點：

1. **學習式 + 風險敏感（CVaR over return distribution）** — 生態系裡沒有一個優化 tail／p99／SLO 尾端。這是最鋒利的差異軸，也是論文應該主打的角度：「生產系統優化的是 mean／fairness／約束滿足，我們優化的是別人不優化的那個量（tail），而 tail 才是 serving SLO 的關鍵」。
2. **失效安全整合架構**（`job_submit.lua` 非阻塞回退） — DevOps 角度是真實的工程價值，能把研究用 RL 安全放進生產路徑。務必保留為賣點。
3. **抗漂移、多 seed 配對的 sim-to-real 方法學 + 誠實的 scale 門檻結論** — 多數排程 paper over-claim，這是可獨立成立的方法學／cautionary 貢獻。

**定位句：** Kubeflow 是生命週期、Volcano／Kueue 是規則式配額＋gang、DRA 是分配機制——四者皆非「學習式、且以尾端延遲為目標的排序／放置策略」。本研究落於此空隙，且與 DRA **互補**（DRA 給的是「如何表達要 0.25 張 GPU」的機制，本研究給的是「該把哪些工作打包、用什麼順序壓低尾端」的策略），可在 DRA 之上驅動裝置選擇與准入排序。

## 三、最致命的問題（next-review 一定會打的點）

**（致命）RDSAC 在整篇論文裡從未贏過 baseline。**
- `paper.md` 表 1 的「模擬可區分策略」是 **FCFS vs multifactor／score**，**完全不含 RDSAC** → 它證明的是「模擬器能區分策略」，不是「學習式策略有效」。
- 表 4 是唯一出現 RDSAC 之處，而它**顯著小輸**（−3.7~−4.6%，p≪1e-11）。
- 後果：主打「風險敏感學習式策略」（貢獻 #3），卻沒有任何實驗顯示其價值。reviewer：「既然 sim 能區分、又從不顯示 RDSAC 贏，那 CVaR／IQN 這套複雜度的正當性在哪？」

**二級問題：**
- **CVaR 缺 ablation 且現有結果不可信。** `eval-writeup §4.4.2` 的 mean-vs-cvar 在**單一訓練 seed**下擺盪極大（同一 cvar 設定 philly 跨不同 run 從 +53.5 → +0.7 → −17.9）。risk-sensitive 是 headline 卻無多 seed mean±std，claim 站不住。
- **baseline 太弱。** §2 引了 Kueue／Volcano／DRA，實驗只比自家 FCFS／score；HPC 角度更要對照 Slurm `gres/shard` + backfill + multifactor。
- **eval ↔ prod placement gap。** 生產路徑 RL 只設**佇列優先權**，但 live 放置實驗用 `live_daemon` 的 explicit `srun` 放置——評估的東西不是部署的東西。需誠實揭露或補上。
- **K8s maintainer 視角：** Slurm-on-K8s 對 K8s 人是「為何要兩個排程器」，需明確正當化（HPC backfill／gang 成熟度 + 研究載具），否則被當「跟平台對著幹」。最有殺傷力的反向建議：把 RDSAC 做成 kube-scheduler scoring plugin 或 Kueue admission ordering plugin，直接落進生態系。

## 四、P0 實驗結果（2026-06-30 ～ 07-01，已執行）

P0 的 #1/#2/#3 已全部跑完（`runs/review_cvar100k_s{42,43,44}`、`runs/review_scale_*`、`runs/review_cvar_s*`），結論一致為**誠實負向**：

| 改進方向 | 狀態 | 結論 |
|---|---|---|
| #3 CVaR 多 seed 消融（σ=1.0，100k，3 seed） | ✅ 完成 | **推翻** §4.4 單 seed 主張。多 seed 下無任何 RDSAC 臂穩健勝 SAC 或 score：SAC philly −8.6±13.4／ali −17.2±21.8（最佳學習臂但仍輸 score）；RDSAC-cvar 一致落後（−24／−49）；RDSAC-mean 雙峰、半數 seed 崩潰 0% 完成，其「+70/+89」為低完成率假象。§4.4 的 +54% 是抽到未崩潰的幸運 seed。 |
| #2 RDSAC 在 stochastic sim 生效 | ✅ 併入 #3 | 由 #3 的 σ=1.0 多 seed 直接檢驗——**未生效**（見上）。 |
| #1 規模交叉曲線 | ✅ 完成 | **無交叉**。1×1／2×1／2×2 下 ΔJCT% 雜訊且多為負（1×1 rdsac-cvar 甚至崩潰 0%），未見「隨規模上升」趨勢。單 seed + 跨尺度 obs 不可比 + 40k 欠訓練為 confound，但方向明確：此量測不支持該假設。 |

**唯一站得住的正面觀察：** CVaR 風險扭曲買到的是**完成率穩定性**（RDSAC-cvar 全 seed 100% 完成，RDSAC-mean 頻繁崩潰），而非速度——風險敏感性作為對抗分布式評論家退化的穩定器。已折入 `paper.md` §4.3（表 5）與 §4.4，貢獻 #3 亦重新定位。

**方法學收穫（升級為貢獻）：** 單 seed 模擬結果會嚴重誤導——同一 RDSAC 設定單 seed 領先、多 seed 不可重現。這強化了論文「嚴謹統計」主軸。

---

## 五、改進優先級表

| # | 改進項目 | 主要角度 | 解決哪個 review 質疑 | 工作量 | 狀態 |
|--:|---|---|---|:--:|:--:|
| **1** | **sim 規模交叉曲線**：RDSAC vs baseline 隨叢集規模的 ΔJCT% | IEEE／AI Infra | 「value requires scale」只是斷言 | 中 | ✅ 完成 → **無交叉**（負向） |
| **2** | **讓 RDSAC 在某處真的贏**：stochastic sim 下 CVaR 勝 heuristic | IEEE／AI Infra | 貢獻 #3 無證據 | 中 | ✅ 完成 → **未生效**（負向） |
| **3** | **CVaR ablation 多 seed**：mean vs cvar，跨 seed 求 mean±std | IEEE | §4.4.2 單 seed 不可信 | 低–中 | ✅ 完成 → **推翻 §4.4**（負向） |
| **4** | **補強 baseline**：對照 Slurm 原生 backfill+multifactor；sim 內加 Kueue-style fair-share／Volcano binpack | HPC／K8s | 「只比自家 heuristic，不比引用的 SOTA」 | 中 | ✅ **實質完成**：(a) sim SOTA 已做（表 7：Kueue/Volcano 在帶內、未勝 score）；(b) Slurm 原生 backfill+multifactor+**gres/mps** 已是 live 對照臂（表 3 multifactor，`baseline_switch.py`），已打平。**gres/shard 不適用**——叢集用 `gres/mps` 非 `gres/shard`（chart 僅渲染 `Name=gpu`+`Name=mps`）。 |
| **5** | **誠實處理 eval↔prod placement gap**：明寫差異，或讓 prod 也走 explicit placement | DevOps／K8s | 評估非部署路徑 | 低 | **P1**（下一輪） |
| **6** | **serving 真實度**：bursty／Poisson 推論到達 + SLO 分級，而非固定 `slo_s` | AI Infra | SLO 模型過淺 | 中 | **P1**（下一輪） |
| **7** | **DRA／Kueue 互補 PoC**：用 RDSAC 當 Kueue admission ordering 或 DRA device-selection 的 policy | K8s／AI Infra | 把「互補」從論述變 demo，大幅提升 novelty | 高 | **P2** |
| **8** | **正當化 Slurm-on-K8s 架構選擇**：為何不用 Kueue+DRA+scheduler plugin，寫進 §3 | K8s maintainer | 「為何兩個排程器」 | 低 | **P2** |

**P0 執行後的定位（2026-07-01）：** #1/#2/#3 全部完成，結果**一致為誠實負向**——「sim 救援」失敗，連模擬多 seed 下 RDSAC 都未勝出。因此論文**不宣稱 RDSAC 優越**，改為全押「方法學 + 誠實負結論」：實機統計打平、模擬多 seed 亦打平/負向、單 seed 會誤導、CVaR 僅提供完成率穩定性。下一輪主攻 **#4（強 baseline）** 與 **#5（placement gap）** 以鞏固負結論的可信度，並視野心考慮 **#7（DRA 互補 PoC）** 提升 novelty。

---

> 前一版審查（2026-06-23，以 sim-to-live 保真度與負結論強健性為主軸）內容已併入 `eval-writeup.md` 各節；本檔自此版起改以「研究定位差異化 + 改進優先級」為主軸。
