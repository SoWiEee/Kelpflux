# Evaluation Report

整理模擬環境與實機環境的評估報告，清楚比較三種排程方式的行為（啟發式 score、、SAC、RDSAC），並在真實 2 Nodes × 1 GPU 環境中，評估 DRL 排程到底有沒有贏過啟發式演算法。

## 訓練與評估管線

在真實叢集中從頭做強化學習訓練需要數十到數百萬個 transition，而真實叢集一個編排決策 (placement) 對應一個跑數分鐘到數小時的工作，湊滿訓練樣本需要等待數個月，因此採用 sim-to-real 兩段式管線：

1. 在模擬環境中大量訓練（便宜/安全/可配對評估），產生基礎模型 (checkpoint)
2. 上線部署到實機，記錄真實 $(obs, act, rew)$ 資料到 JSONL
3. 運用 RLPD 用真實資料把基礎模型微調成真實環境策略

---

## 0. 摘要與結果總覽

欲比較的三種排程方式如下：

- Score：啟發式優先序，作為 baseline
- SAC：原味離散 SAC，scalar twin-Q critic
- RDSAC：分布式 IQN critic，有風險中立 mean 和風險扭曲 cvar 兩種變體

> 在訓練、評估、上線服務階段皆用 `use_iqn` 標籤切換 SAC/RDSAC、`risk_mode` 切換 mean/cvar。

> [!IMPORTANT]
> 目前在實機環境 (RTX 4070 + RTX 3080) 拓樸上，沒有任何 DRL 排程策略打贏啟發式。

評估在拓樸匹配的 **2×1**（obs_dim=166、n_actions=33）做，有分成模擬評估 (§3) 和實機評估 (§4)：

| 評估 | score | SAC | RDSAC-mean | RDSAC-cvar | 判定 |
|---|---|---|---|---|---|
| 模擬 σ-sweep, fixed-α, multi-seed（§3.1，σ=1.0，3 train seed）| 0% | −38.3±12 / −33.6±23 | −12.9±10 / −8.4±3 | −35.4±16 / −42.0±30 | 沒人贏過 score，模型間排名是訓練雜訊 |
| 實機）| 0% | −21.6±9.6 | −13.4±4.4 | −21.2±6.1 | 三個 DRL 策略都輸 Slurm/score；過度集中在 4070 |

（sim 兩個數字 = philly / ali（σ=1.0 mean±std）；live 兩個數字 = σ=1 / σ=0；ΔJCT% vs score，負值=較慢）

### 模擬結果摘要

- 紀錄於 §3.1~§3.4，σ-sweep 部分已做 multi-seed 評估
- 注入校準過的不確定性後，三方逼近打平、但沒人贏過 score
- 3 個訓練 seed 的 mean±std 確認了兩件事：
  - 沒人贏過 score（穩健）
  - 模型間的差異是訓練雜訊（std 5–30 pts、跟 mean 差同量級，同 config 跨 seed 擺盪 30–70 pts）

### 實機結果摘要

- 紀錄於 §4.1（四方）和 §4.2（真實 CUDA）
- 真實 cuBLAS job 含異質 MPS 需求 25/50/75/100，DRL 策略都輸 score，關鍵在於「大 job 沒放對」

---

## 1. 評估對象

### 1.1 Heuristic score scheduler

以加權線性組合公式來計算工作優先級，分數越高代表該工作越值得優先排程。五個因子及對應權重說明如下表：

| 訊號 | 用途 |
|---|---|
| MPS fit | 小 MPS job 是否能塞進目前 GPU 的剩餘 slot，避免大 job 佔滿整張卡。 |
| VRAM fit | job 需求是否符合 GPU tier，避免高階卡被低需求 job 浪費。 |
| Fragmentation | 避免接受一個 job 後讓剩餘 MPS slot 過度碎片化。 |
| Runtime Fit | 若 runtime predictor 有可用估計，短 job 會得到額外 priority boost。 |

最終分數 $score = α·\text{mps_fit} + β·\text{vram_fit} + γ·\text{topology} + δ·\text{frag} + ε·\text{runtime fit}$，以 `score_gain=1000` 加在 Slurm multifactor 優先級上。

> [!NOTE]
> 實機設定目前採用 $α=0.4, β=0.2, δ=0.2$，而剩餘兩項 $γ=0, ε=0$ 是預設關閉。目前 runtime predictor 與 weight-tuner 都沒部署，所以權重是靜態，不會在線上調整。

> 實作位置：模擬器 baseline `sim/scheduler/score.py`（cluster-aware、ε 預設 0.30）；live submit hook `chart/templates/configmap-job-submit.yaml`（無狀態 proxy、ε=0）；Slurm priority path `chart/templates/{slurm-conf,login,workers}.yaml`。

### 1.2 SAC

根據 [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) 實作離散版本（`DSACAgent(use_iqn=False)`）：**scalar twin-Q critic、MSE soft-Bellman、無 IQN 的 Z_R/Z_H 分布式分解、無 risk distortion**。排程 action 是離散且有 mask 的——每一步只能從 pending queue 中可行的 (job, node, GPU/MPS placement) 組合裡選一個。SAC 在本專案作為 RDSAC 的對照基準：用**完全相同的配方**（步數、traces、curriculum、PER、shaping、MLP trunk、2×1 拓樸）訓練，唯一差別是 critic 型別與沒有 risk，用來回答「分布式/風險機制到底有沒有用」。

### 1.3 RDSAC

根據 [Distributional Soft Actor-Critic for Risk-Sensitive Reinforcement Learning](https://arxiv.org/abs/2004.14547) 的**離散動作復刻版**。並把連續控制的 reparameterised Gaussian actor 替換成顯式 categorical actor，其中 critic 與 risk 機制如下：

- 雙分布 critic：把 soft return 拆成 reward 分布 $Z_R$ 與 entropy 分布 $Z_H$，各以 IQN 的 quantile 表示、共用 trunk，quantile Huber 回歸 + twin double learning。
- risk 進策略目標：actor 目標對 reward 分布套用 distortion `ρ`，`risk_mode ∈ {mean, cvar, wang, cpw, msd}`。`mean` 是風險中立；`cvar`（β=0.25）偏好下尾較不嚴重的 placement，對應排程的 straggler / cold worker / long-tail runtime 風險。

> 實作位置：`services/rl_scheduler/dsac.py`、`services/rl_scheduler/distortion.py`。

---

## 2. 實驗與 Benchmark 方法

### 2.0 訓練與評估管線

在真實叢集中從頭做強化學習訓練需要數十到數百萬個 transition，而真實叢集一個編排決策 (placement) 對應一個跑數分鐘到數小時的工作，湊滿訓練樣本需要等待數個月，因此採用 sim-to-real 兩段式。§3 的所有數字都是「在模擬環境裡訓練 + 在模擬環境裡評估」，目的是篩架構與抽洞察，不是預測真實績效；真實績效由 §4 的實機 A/B 評估，最終策略將由 RLPD 微調產出。

| 階段 | 環境 | 做什麼 | 為什麼不能在實機直接做 | 產出 |
|---|---|---|---|---|
| (1) 模擬訓練 | `KubefluxSchedEnv`（gym） | 從頭訓練 100k–150k 步，curriculum，PER + shaping | 實機一個 step = 一個跑數分鐘～小時的 job，湊滿樣本要數月 | checkpoint（warm start） |
| (2) 模擬評估 | 同 sim，配對 + 受控消融 | paired t-test、SAC/RDSAC/cvar/fixed-α 消融 | 實機無法 `reset()` 重放同一 trace → 拿不到 counterfactual | **機制性洞察**（§3） |
| (3) 實機部署 | k3s + Slurm + GPU/MPS | shadow-mode 跑 checkpoint，記錄真實 (obs, act, rew) 資料 | 訓練初期隨機策略會破壞系統 → 只敢 shadow + fail-safe 回退 score | 真實 A/B（§4）+ 微調語料 |
| (4) RLPD 微調 | 用 (3) 的真實 JSONL | 以模擬產出的 checkpoint 為 prior，混合真實資料做微調 | 從頭 RLPD = 退回 (1) 的 sample-complexity 與 (2) 的破壞性探索 | 真實環境策略（future work，§5.1） |

> RLPD (RL with Prior Data) 的前提就是從一個既有 prior 出發再做微調；模擬器的 checkpoint 不是被丟掉，而是 RLPD 站在它肩膀上。實機 trace 收集器（`live_daemon.py` → JSONL → `rlpd_finetune.py`）已就緒。

在模擬環境中「可大量訓練 + 可配對消融」，但代價是模擬環境的絕對數字不轉移。機制性洞察只能當作定性參考，實際會不會生效要看 §4 的實機 A/B 評估。目前從 §4.1 的 multi-seed 結果是 **DRL 顯著輸給啟發式**，即直接使用基礎模型 (sim checkpoint) 的效果不好，可能要修改架構、增強訓練穩定性、RLPD 真實資料微調。

### 2.1 模擬配對測試 Benchmark

相同亂數種子 (seed) 對 DRL 模型與啟發式 score 做配對比較來降低 trace 隨機性造成的誤差。`Δ = (score − model)/score`，負值代表 model 的 JCT 較高、較差。

| 項目 | 值 |
|---|---|
| training | 從頭訓練 100k 步，curriculum n_jobs 10→30→50，fixed-α 0.05 |
| reward_scale | 20000（修復 alpha 觸頂） |
| cluster | RTX 4070 + RTX 3080 (obs_dim=166、n_actions=33) |
| jobs per trace | 50 |
| trace families | `philly`, `ali`（Alibaba PAI）|
| seeds | 每個 $(model, σ)$ 訓練 1 seed；評估配對 5 seed |
| metric | mean JCT（主），p95 / p99 JCT、CVaR（尾部），paired t-test 95% CI / p-value |

> 每個 $(model, σ)$ 只訓練一個 seed；評估階段的 5 seed 是評估隨機性而非訓練隨機性。這是目前最大的方法學限制（§3.3 有同 config 兩跑擺盪 60–90 pts 的鐵證）。

### 2.2 受控變數設計（每個實驗只動一個變因）

| 變因 | 隔離什麼 | 出現於 |
|---|---|---|
| `use_iqn` (SAC/RDSAC) | critic 型別：scalar vs 分布式 IQN | §3.1, §3.3 |
| `risk_mode` (mean/cvar) | 風險扭曲：risk-neutral vs CVaR | §3.1, §3.3 |
| `fixed_alpha` ($α=0.05$) | 溫度控制：避免 auto-α railing（全程釘死）| §3.1–§3.4 |
| `runtime_sigma` / `interference` | 注入 runtime 不確定性 / MPS 共置干擾（opt-in，預設關 → 與確定性 env 逐位元相同）| §3.1–§3.4 |
| `colocation_actions` (PACK/ISOLATE) | 共置是否成為一個動作 | §3.4 |

**隨機性注入模型**：`actual = predicted · exp(σZ − σ²/2)`（mean-preserving lognormal，E=1，只增變異不偏均值；obs 仍顯示 nominal runtime → 真實的結果不確定性）。idiosyncratic 噪音對每個 job common-random（以 `(seed, job_id)` 鍵）→ 同一 job 在每個 policy 下拿到相同乘子 → 配對比較。Harness：`eval/scripts/sweep_stochastic.py`。

### 2.3 σ 校準方法（讓注入的噪音不是憑空挑的）

由注入模型 `σ = std(log(actual/predicted))`，正是 runtime 預測器的 log-殘差標準差。`eval/scripts/measure_predictor_sigma.py` 使用**生產級 LightGBM 預測器**（同 `services/runtime_predictor` 的 features、time-honest 80/20 split、超參）在真實 trace 上量 held-out log-殘差 std，當作要注入模擬器的 σ，結果見 §3.2。

### 2.4 實機 A/B 評估設定

在實機環境提交 job 做配對 A/B 測試。每個 job 用同一條 stream 在四種方法 (score/SAC/RDSAC-mean/RDSAC-cvar) 各重放一次（per-job common-random）。設計重點如下：

- DRL 模型的每個 job 先呼叫 serve `/act` 拿到節點選擇，用 `sbatch -w <node>` 在提交時選好 node；score 策略交給 Slurm 預設編排。
- 去除飄移偏差：GPU 跑久了會變快（MPS 暖、快取熱），若一種方法整段跑完才換下一種，drift 會和方法混淆。改用 round-robin（`--interleave`）交錯方法順序、每方法跨多輪輪過各個位置，把飄移誤差平均掉。

> 目前尚未部署 runtime predictor 和 weight tuner ，因此 score 的 `ε`(SJF) 與線上權重調整都是關的


### 為何多用 p95/p99/CVaR 測量？

每張表格都有主要指標 mean JCT 和 p95/p99 JCT、CVaR(0.25)。納入尾部指標不是裝飾，而是這套評估能不能看見 RDSAC 效應的**前提**的三個理由：

- mean JCT 會把排程病態洗掉：straggler、queue starvation、head-of-line blocking 的典型表現是「多數 job 正常、少數被拖很慢」；這幾個慢 job 攤進整批裡幾乎不動 mean，但主導使用者體感。p95/p99 專抓「最差 5%/1% 有多慢」，正是 mean 結構上看不到的那段。
- 尾部是 RDSAC/CVaR 的靶心：RDSAC-cvar 的設計目標就是優化回報下尾（= JCT 上尾）；只測量 mean 等於拿不會動的尺去量專門改尾部的方法，結構上必然測不出差異。鐵證在 §3.1：RDSAC 對 SAC 的 **mean 差距有限，p99 卻差 6–11×**，優勢全在尾部。
- 領域慣例：GPU cluster scheduling / SLO 文獻裡，tail JCT、tail slowdown（p95/p99）本就是標準指標（p99 幾乎是 SLO 代名詞），mean 是必要但不充分。

---

## 3. 模擬結果（Simulator）

本節是管線的階段 (1)(2)：在模擬環境訓練、在模擬環境裡做配對評估 (obs_dim=166、n_actions=33)。這裡的數字當定性洞察、用來抽機制與做受控消融，並非用來預測真實績效。建立三個消融實驗如下：

- σ-sweep 三方 (§3.1)：注入校準過的 runtime 不確定性（σ 校準到生產預測器的 log-殘差，§3.2），比 score / SAC / RDSAC-mean / RDSAC-cvar。結果三方逼近打平、沒人贏過 score；σ→cvar 只在 σ=0.5 成立、不單調。
- 分布式 critic vs 風險扭曲 (§3.3)：結果兩者都只剩個位數 pts、且看 workload 正負擺盪，沒有單一主因

> [!NOTE]
> 模擬訓練有**高變異**的特性，即相同 config 跨訓練 seed 可擺盪 30~90 pts。在 §3.1 的 σ-sweep 已使用 3 個訓練 seed 並使用 mean±std 量化這個變異（其餘格子仍單 seed，看方向別細讀點估計）。

### 3.1 隨機性消融：注入 runtime 不確定性後，三方怎麼排？

由於 RDSAC 的風險機制 CVaR 是用來「規避下尾風險」的，但風險機制要有風險可管才有意義。在模擬環境的 runtime 是給定狀態與動作，可確定結束時間，reward 分布塌縮成一個點時 (CVaR = mean) 風險機制沒作用。

因此在環境裡加一個**校準過的** runtime 不確定性 σ（mean-preserving lognormal，方法見 §2.3；σ 取自生產預測器的真實 log-殘差，§3.2），讓三方在「有尾部風險可管」時比較策略優劣。

σ 的值是 0.5, 1.0，分別訓練 SAC/RDSAC-mean/RDSAC-cvar (fixed-α 0.05)，各訓練 10,000 步、curriculum、5-seed 配對、philly/ali。並且每個 $(σ, arm)$ 用 3 個訓練 seed (42/43/44) 執行，專門用來打掉單 seed 雜訊。

**ΔJCT% vs score（負=較慢）：**

| σ | family | SAC | RDSAC-mean | RDSAC-cvar |
|---|---|---:|---:|---:|
| 0.5 | philly | −7.1±5.1 | −19.9±9.4 | −7.8±7.5 |
| 0.5 | ali | −15.0±7.5 | −2.4±5.3 | −3.7±3.2 |
| 1.0 | philly | −38.3±12.3 | −12.9±10.1 | −35.4±16.3 |
| 1.0 | ali | −33.6±23.0 | −8.4±3.2 | −42.0±29.8 |

從結果可以有兩個發現：

- 沒有任何 DRL 模型贏過 score：每個 mean 都是負的，連 +1 個 std 也構不到 0。
- 策略間的排名是訓練雜訊：標準差大致落在 5~30，跟策略間的平均差距*同量級甚至更大。

> 評估資料 `runs/mseed_2x1_s{42,43,44}/`，整理成 `runs/mseed_2x1_agg/SUMMARY.txt`。

### 3.2 σ 校準到真實預測誤差：σ=1.0 其實偏保守

由於 §3.1 的弱點：σ=0.5/1.0 若是憑空挑的，「注入噪音 → 抗噪法贏」就近乎套套邏輯。所以用 §2.3 的方法，測量生產 LightGBM 預測器在真實 trace 上的 held-out log-殘差，當作該注入的 σ：

| workload | σ（殘差 std）| 95% CI | 形狀 |
|---|---:|---|---|
| philly | **1.45** | [1.31, 1.58] | near-Gaussian |
| ali | **1.24** | [1.11, 1.38] | near-Gaussian |

可以發現真實 $σ ≈ [1.2, 1.45]$，故 §3.1 用的 $σ=1.0$ 落在真實下限以下、是保守值。以及殘差**近高斯**（excess kurtosis −0.1~+0.4），lognormal 噪音模型沒有低估尾部。

> [!WARNING]
> 這些合成 trace 的 runtime 與特徵無關（corr(log_rt, gpu_count)=0.04，predictor 打不過 predict-the-mean），所以 1.2~1.45 是**最難預測**上界；真實結構化資料上好 predictor 會更低，因此合理 $σ ≈ [0.5, 1.45]$，且 §3.1 測的 {0.5, 1.0} 都落在其中。

### 3.3 拆解：贏的是「分布式 critic」還是「風險扭曲」？

在 §3.1 把 RDSAC 當一整包，但它其實有**分布式 critic**（把 reward 建模成分布，而非單點 Q）與 **風險扭曲**（CVaR）。若要知道哪個在出力，就放第三方 RDSAC-mean（有分布式 critic、但風險中立）來實驗，使用 3 train seed, $σ=1.0$ 三方，計算 mean±std：

| family | SAC | RDSAC-mean | RDSAC-cvar | SAC→mean（分布式）| mean→cvar（風險）|
|---|---:|---:|---:|---:|---:|
| philly | −38.3±12.3 | −12.9±10.1 | −35.4±16.3 | **+25.4±22.6** | **−22.5±10.4** |
| ali | −33.6±23.0 | −8.4±3.2 | −42.0±29.8 | **+25.2±22.4** | **−33.6±27.6** |

可以發現分布式 critic 是有用的那一半，CVaR 風險扭曲反而扣分。

> [!WARNING]
> 目前標準差仍然很大、並且這結論只在 $σ=1.0$ 成立。CVaR 在 $σ=1.0$ 沒幫上忙、甚至扣分，分布式 critic 是相對有用的一半，但是兩者都沒讓 RDSAC 贏過 score。

> 原始檔 `runs/mseed_2x1_s{42,43,44}/`

### 3.5 模擬環境評估統整

| 研究問題 | 模擬評估的答案 |
|---|---|
| 注入不確定性後，RDSAC 會贏嗎？（§3.1）| **沒人贏過 score**（multi-seed 穩健）；arm 間排名是訓練雜訊（std 5–30 pts），單 seed 看到的「cvar 最好」不成立 |
| 贏在分布式 critic 還是風險扭曲？（§3.3）| σ=1.0 下 **分布式 critic 是有用的一半、CVaR 反而扣分**（−22~−34 pts，與設計意圖相反）；但仍沒讓 RDSAC 贏 score |

使用多種子訓練後，有兩個穩健結論：沒人贏過 score、策略間的差異是訓練雜訊。而這個「訓練高變異 + 沒人贏」直接接到實機評估 (§4.1) 的退化：三個 DRL 策略全把負載擠到 4070、全部輸給 score (Slurm)。

實機評估也做了 multi-seed（三個訓練種子 checkpoint 各跑一次四方 A/B）→ 負結果 seed-robust：每個 seed×arm 都輸 score（−7~−31% JCT）、過度集中跨 seed 穩定（§4.1）。所以兩端都固實了。原始檔 `runs/mseed_2x1_*`、`runs/htab_live_mseed_s*`。

### 3.6 改善措施

從前面的討論可以確定瓶頸是**訓練退化導致過度集中與高變異**，不是演算法問題。

- Load-balance shaping (P1)：在 potential 加節點均衡項（`φ -= balance_coef·std(free_mps_per_node)`），根據 Ng et al. 1999 保證不改最優策略、只引導探索時避免擠在單張 GPU）
- Reward normalization (P2)：running-std 回報正規化（PopArt-lite，消手調 `reward_scale` 的脆弱、降 seed 敏感度）

使用 `--balance-coef 5.0 --normalize-reward` 在 σ=1.0、3seed 重新訓練，對比 baseline（§3.1）：

| family | model | JCT(h) | p95(h) | p99(h) | ΔJCT% | Δp99%（vs baseline）|
|---|---|---:|---:|---:|---:|---:|
| philly | **cvar** | 2.1±0.1 | 9.3±1.1 | 30.4±3.8 | **−4.1±4.6** | **−17%** |
| philly | sac | 2.3±0.1 | 11.4±0.5 | 28.4±1.5 | −11.8±4.2 | **−30%** |
| philly | mean | 2.6±0.1 | 11.7±1.2 | 41.0±8.8 | −23.1±7.4 | +39% |
| ali | cvar | 1.1±0.1 | 4.3±0.9 | 20.4±0.6 | −12.0±12.0 | **−15%** |
| ali | sac | 1.2±0.1 | 5.5±1.2 | 21.1±0.3 | −15.2±17.5 | +5% |
| ali | mean | 1.4±0.1 | 6.6±0.5 | 22.7±2.5 | −32.7±10.4 | +6% |

> 指標格式為 mean±std，score baseline 參考 JCT philly 2.1h / ali 1.1h。

**對比 baseline ΔJCT%（§3.1，改善 = 往 0 靠多少）：**

| family | model | baseline ΔJCT% | P1+P2 ΔJCT% | 改善 |
|---|---|---:|---:|---:|
| philly | **cvar** | −35.4±16.3 | **−4.1±4.6** | **+31** |
| philly | sac | −38.3±12.3 | −11.8±4.2 | +27 |
| ali | cvar | −42.0±29.8 | −12.0±12.0 | +30 |
| ali | sac | −33.6±23.0 | −15.2±17.5 | +19 |
| philly | mean | −12.9±10.1 | −23.1±7.4 | −10 |
| ali | mean | −8.4±3.2 | −32.7±10.4 | −24 |

從結果可以發現可以有效改善，三個正面證據如下：

- cvar/SAC 大幅靠近 score 且變異數同步收斂，尾部也跟著改善
- 改善 DRL 過度集中，cvar 的節點分佈跨 3 seed 從 baseline 89% 降到 80/65/67%，可證明 balance shaping 有效
- `philly cvar s44 = +2.3%`，目前首次有 DRL 策略在模擬評估上贏過 score

> [!IMPORTANT]
> 目前還有幾個限制：(a) 仍然沒有人穩健贏過 score；(b) RDSAC-mean 反而退步（balance shaping 與風險中立模型互動不良，待查）；(c) 集中度仍 >50%。
> P1+P2 確認「修訓練、別加花招」方向正確，把退化策略往「均衡、低變異、逼近 score」推進了一大步，值得拿去做真實 CUDA job 的實機檢驗（§5.1 第 4 項）。

> 原始檔 `runs/p1p2_2x1_s{42,43,44}/`、彙總 `runs/p1p2_2x1_agg.txt`。

---

## 4. 實機執行結果

本節是管線的階段 (3)：把 §3 的模擬產出 checkpoint 烘進真實叢集。主要結論如下：

- 三個 DRL 模型（SAC、RDSAC-mean、RDSAC-cvar）全顯著輸 Slurm 預設（p<0.01），且越把負載擠到 4070 輸越多。
- fail-fallback 設計在真實環境驗證有效：`/decide` 失敗或低信心時自動回退 score，slurmctld 從不被擋，這是讓 shadow 部署能安全跑的前提。
- 真實微調語料已開始累積：實機 A/B 測試時 `live_daemon.py` 記錄的真實 (obs, act, rew) 即階段 (4) RLPD 的輸入。

> 主要價值在於用正確的實驗設計（共享 partition + placement + drift-robust interleave）拿到真實的結論，並證明 sim-to-real 工程管線真的能跑，為 multi-seed 固實、RLPD 微調鋪路。

### 4.1 實機配對 A/B 測試

在 submit-時 DRL 挑選 node，跑四種方法進行評估 (score / SAC / RDSAC-mean / RDSAC-cvar)。結果發現所有 DRL 方法都顯著輸 Slurm 預設，且輸的幅度由「把負載擠到 4070 的程度」決定。

> 協定：`run_heavytail_ab --placement --gpu-nodes rtx4070-0,rtx3080-0`、四方（score/SAC/RDSAC-mean/RDSAC-cvar，checkpoint 取 §3.1 的 σ=1.0 三方）、philly、partition `gpu`（橫跨兩台的共享 partition）、n=20/stream、σ∈{0,1}、**`--interleave`**（drift-robust，每方法 4 輪×輪轉位置）、warmup 丟棄、per-job CRN。每方法 **n=80 paired**。原始檔 `runs/htab_live_place4clean_20260618-170651/`。

**ΔJCT% / Δp99% / ΔCVaR% vs score（全部 paired）：**

| σ | 策略 | mean JCT | p99 | CVaR | ΔJCT% | Δp99% | ΔCVaR% | p |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| 0.0 | score | 8.4 | 35.0 | 18.9 | — | — | — | — |
| 0.0 | SAC | 10.6 | 40.2 | 25.2 | −26.3 | −14.9 | −33.3 | <.001 ✓ |
| 0.0 | RDSAC-mean | 10.7 | 41.2 | 25.1 | −28.1 | −17.7 | −32.8 | <.001 ✓ |
| 0.0 | **RDSAC-cvar** | 11.0 | 41.0 | 25.7 | **−31.8** | −17.1 | −36.0 | <.001 ✓ |
| 1.0 | score | 8.4 | 36.5 | 19.6 | — | — | — | — |
| 1.0 | SAC | 9.4 | 40.2 | 22.6 | −12.2 | −10.3 | −15.3 | .009 ✓ |
| 1.0 | RDSAC-mean | 9.9 | 40.2 | 23.1 | −17.4 | −10.3 | −18.4 | .001 ✓ |
| 1.0 | **RDSAC-cvar** | 10.4 | 40.0 | 25.0 | **−24.2** | −9.7 | −27.9 | <.001 ✓ |

補上 `NodeList` 擷取後，把每個決策落在哪台都記下來 (pooled n=160/arm)：

| 策略 | 落在 4070 | 落在 3080 | ΔJCT%（σ=1）|
|---|--:|--:|--:|
| score（Slurm 自選）| **52%** | **48%** | 0（baseline）|
| SAC | 88% | 12% | −12.2 |
| RDSAC-mean | 88% | 12% | −17.4 |
| **RDSAC-cvar** | **92%** | **8%** | **−24.2** |

可以發現所有 DRL 都把負載嚴重擠到 RTX 4070，而 Slurm backfill 是均衡的 52/48，而且越是擠到 4070、就輸越多。

這是因為 job 是固定 `sleep N`（runtime 與落哪台無關），JCT 差異全來自**排隊**：擠到 4070 → 該卡佇列壅塞 → wait 變長。這把 §4.1 前 `/act` 探針的「偏好 node 0」量化成鐵證，也是 §3.5 壓倒性 caveat（**單訓練 seed、no-op 傾向**的 checkpoint）在真實環境的兌現。

**multi-seed 確認（負結果對訓練 seed 穩健）**

> 上面是單一 checkpoint。為了排除「剛好抽到壞 seed」，用 3 個訓練 seed (42/43/44) 的 checkpoint 各跑一次相同評估（σ=1.0、sleep job、philly、n=80/arm/seed）。下表紀錄**絕對指標**（跨 3 seed mean±std，單位小時）：

| arm | JCT(h) | p95(h) | p99(h) | CVaR(h) | slowdown | 落在 4070 |
|---|---:|---:|---:|---:|---:|---:|
| score | **8.3±0.4** | 33.7±0.5 | **35.9±1.4** | **19.4±0.7** | **2.0±0.2** | 48% |
| RDSAC-mean | 9.4±0.3 | 33.1±1.4 | 39.3±1.2 | 22.6±0.7 | 2.4±0.1 | 85% |
| RDSAC-cvar | 10.0±0.6 | 34.0±1.4 | 39.4±1.0 | 23.4±1.6 | 2.6±0.2 | 89% |
| SAC | 10.1±1.1 | 34.8±0.3 | 40.7±0.4 | 24.1±1.8 | 2.6±0.4 | 88% |

**配對差（vs score，mean±std，每格 t-test p 取 3 seed 平均）：**

| arm | ΔJCT% | Δp99% | ΔCVaR% | p | per-seed ΔJCT |
|---|---:|---:|---:|---:|---|
| RDSAC-mean | **−13.4±4.4** | −9.5±1.1 | −16.6±4.4 | 0.062 | −16.7 / −7.1 / −16.3 |
| RDSAC-cvar | **−21.2±6.1** | −9.9±6.6 | −20.6±8.7 | **0.000** | −13.3 / −22.4 / −28.1 |
| SAC | **−21.6±9.6** | −13.5±3.5 | −24.0±7.6 | 0.028 | −8.4 / −30.9 / −25.5 |

**結論：seed-robust，且全指標一致變差。** 每個 (seed × arm) 的 ΔJCT 都是負的（範圍 −7.1 ~ −30.9），**沒有任何 seed 讓 learned arm 贏過 score**；不只 mean JCT——**p99 尾部（+3.4~4.8h）、CVaR（−17~−24%）、slowdown（2.0→2.4~2.6）全部一致變差**。過度集中也跨 seed 穩定（85–89% 擠 4070 vs score 48%）。所以 §4.1 的負結果**不是單 seed 壞運，是這套 train+placement 的結構性退化**。（RDSAC-mean 一致最輕，與 sim multi-seed「mean 是 σ=1.0 最好的 learned arm」一致。）原始檔 `runs/htab_live_mseed_s{42,43,44}/`、彙總 `runs/mseed_live_agg_full.txt`。**剩餘範圍限制**：單 family（philly）、sleep job（見下 workload caveat）。

> [!NOTE]
> 本節評估 job 固定 `sleep N`，不做 GPU 計算，所以 runtime 與 placement 無關、placement 只影響排隊 wait。之後換成真實 CUDA job 會讓測試更完整、也更公平（見 §5.1 第 4 項）；對目前過度集中的 checkpoint 預期會更慘（多付干擾代價），但那才是讓好的 placement 策略有機會展現價值的場域。

### 4.2 真實 CUDA job（異質 MPS 需求）

使用真實 cuBLAS sgemm workload `gpu_workload.cu`，讓排程決策真的有算力後果。MPS 需求有 `--mps-buckets 25,50,75,100`，按照 job 大小指派（大 job 要更多 GPU、CRN 穩定），這樣才會壓到 score 的 `f_mps_fit`/碎片感知。好的策略要會把 `25+75`、`50+50` 打包同卡、把 `mps:100` 的 job 送去閒置的 GPU。

3 train seed、σ=1.0、philly、n=14/round、`--interleave`，264 job 全 COMPLETED，bucket 分佈 {25:72, 50:72, 75:72, 100:48}：

| arm | JCT(s) | ΔJCT% | ΔCVaR% | 落點 4070/3080 | big(≥75)JCT | small JCT |
|---|---:|---:|---:|---:|---:|---:|
| score | 23.2±0.2 | （baseline）| | 42/58 | 38s | 11s |
| **RDSAC-mean** | 24.9±0.3 | **−7.2±1.5** | −0.4±2.8 | 61/39 | **37s** | 15s |
| RDSAC-cvar | 27.9±2.6 | **−20.3±11** | −26.7±19 | 48/52 | 45s | 14s |
| SAC | 28.3±2.4 | **−21.7±10** | −26.1±18 | 52/48 | 45s | 15s |

結果 DRL 仍然沒贏過啟發式 score，但 RDSAC-mean 最接近打平。

大工作（75, 100）主宰 JCT。RDSAC-mean 把大 job 處理得跟 score 一樣好才會接近打平；而 SAC/cvar 把大 job 擺爛，就是那 −20% 的來源，小 job 三方都差不多。也就是說：在有異質 GPU-fraction 需求時，輸贏的關鍵是大 job 有沒有放對。

> 原始檔 `runs/mpsbuckets_s{42,43,44}/`

### 4.3 暴露 GPU 異質性後的實機 A/B（item-1）

§4.2 的 checkpoint 是**同質訓練**——obs 裡的 GPU one-hot 寫死、sim 裡兩張卡同速，所以策略對「4070 快 / 3080 慢」是**盲的**，大 job 才會放錯卡。item-1 修正：obs 暴露每張卡的 `gpu_type`、sim cluster 給每節點速度（4070=1.0×、3080≈0.25×）。sim 配對證實有效（同質 baseline vs 異質訓練、兩者都在異質環境，異質訓練贏 10/12 格；§3 / `eval/scripts/compare_hetero_vs_homo.py`）。本節把**異質感知的 checkpoint（seed-43）**搬上真實 4070+3080。

σ=1.0、philly、real cuBLAS（`target_max_s=60`、buckets 25/50/75/100）、placement 模式、n=96/arm（2 measured round）：

| arm | mean JCT | p95 | p99 | CVaR | ΔJCT% | t-test p |
|---|---:|---:|---:|---:|---:|---:|
| score | 474.9 | 1016.8 | 1108.2 | 944.4 | （baseline）| |
| SAC | 474.2 | 1019.5 | 1093.2 | 949.3 | +0.1 | 0.845 |
| **RDSAC-mean** | **456.2** | 1022.0 | 1102.0 | 956.2 | **+3.9** | **0.116** |
| RDSAC-cvar | 520.7 | 1023.2 | 1093.2 | 956.7 | **−9.7** | **0.0007** |

（正 ΔJCT% = RL 較快；尾端 p95/p99/CVaR 四方差異全在 ±1.4% 內、無顯著差異。）

**發現：暴露異質性把 learned 從「全輸」拉到「打平 / 名目小贏」，但還不是統計顯著的勝利。**
- **§4.2（同質）learned 全輸 −7~−22%** → **§4.3（異質感知）RDSAC-mean 翻成名目 +3.9%、SAC 打平**。方向跟 sim 的 item-1 發現同調（暴露卡片異質性有幫助）。
- 但 RDSAC-mean 的 +3.9% **統計不顯著**（p=0.116），稱不上真實勝出;**RDSAC-cvar 反而顯著變差 −9.7%**（p=0.0007，與 §3.3「CVaR 風險扭曲在某些設定吃虧」呼應）。**⚠️ §4.3.1 降噪 + 加樣本精確重跑後,這個 +3.9% 反轉成顯著的 −3.3% —— 是雜訊假象,非真實勝出。**

**兩個 load-bearing caveats：**（1）**極端競爭**——slowdown_p99 ≈ 460（job 等了 ~460× 自身 runtime）；在這種飽和下卡滿時 RL arm 也常 abstain → 退回 vanilla Slurm，稀釋 RL 效果。（2）**單一 seed-43 checkpoint、單 σ、2 round**。

#### 4.3.1 精確測量（完整四方）：+3.9% 沒撐過,三方全顯著輸 score −3.7~−4.6%

上面的 +3.9% 不顯著（p=0.116），主因是 §4.3 跑在 slowdown_p99≈460 的**極端飽和**、JCT 變異巨大、且只有 n=96。為了「做到統計顯著」,按方法學降噪 + 加樣本重跑（**非飽和 regime**：poisson 到達 + `--mps-oversub 1.0`，slowdown_p99 從 460→**193**；**n 96→246**）。先以 score vs RDSAC-mean 確認反轉,再補齊**完整四方**:

| arm | n | mean JCT | p99 | CVaR | slowdown_p99 | ΔJCT% | t-test p |
|---|--:|--:|--:|--:|--:|--:|--:|
| score | 246 | 167.8 | 425.6 | 374.7 | 193.3 | （baseline）| |
| SAC | 246 | 174.0 | 432.4 | 379.5 | 193.8 | **−3.7** | 3.65e-16 |
| RDSAC-mean | 246 | 174.5 | 435.8 | 379.5 | 194.8 | **−4.0** | 2.07e-17 |
| RDSAC-cvar | 246 | 175.5 | 432.8 | 381.6 | 195.2 | **−4.6** | 5.32e-12 |

**結果反轉且四方一致**：把變異砍掉（slowdown 460→193）+ 樣本三倍化後,§4.3 的「名目 +3.9%」**翻成顯著的小輸**,而且**三個 learned arm 全部顯著輸 score（−3.7~−4.6%、全 p ≪ 1e-11）**,排序 score > SAC ≈ RDSAC-mean > RDSAC-cvar（cvar 最差,與 §3.3 風險扭曲在此 regime 吃虧呼應）。所以 §4.3 的 +3.9% 是**極端飽和 + 小樣本下的雜訊假象**,不是真實勝出。**精確測量下,item-1 的卡片異質感知在實機仍顯著略輸 score**——與全文主軸（2×1 沒有 DRL arm 穩健贏 score）完全一致,並糾正了 §4.3 暫定的「名目小贏」讀法。

**多 seed 穩健性確認**：拿 seed-44 的 checkpoint 重跑同一套四方協定,結論不變——每個 arm × 每個 seed 都顯著輸 score：

| arm vs score | seed-43 ΔJCT% (p) | seed-44 ΔJCT% (p) |
|---|--:|--:|
| SAC | −3.7（3.7e-16）| −4.0（5.5e-10）|
| RDSAC-mean | −4.0（2.1e-17）| −2.0（1.1e-06）|
| RDSAC-cvar | −4.6（5.3e-12）| −3.1（1.8e-14）|

幅度隨 seed 略有擺動（如 RDSAC-mean −4.0 vs −2.0），但**方向與顯著性完全穩健**：兩個訓練 seed、三個 learned arm、全部顯著輸 score（−2~−4.6%、全 p ≪ 1e-5）。三個正交手段（非飽和降噪 + n 三倍化 + 多 seed）一起把「§4.3 名目 +3.9%」徹底釘成**穩健、統計顯著的負結果**。

> 原始檔 `runs/htab_item1_20260622-103714/`（§4.3）、`runs/htab_item1_sig_20260622-183551/`（2-arm 反轉）、`runs/htab_4arm_sig_20260623-001716/`（s43 四方）、`runs/htab_4arm_s44_20260623-041243/`（s44 四方,多 seed）

### 4.4 實機評估的 seed 方法學：能不能自動選 seed？

§4.3.1 是手動挑「sim 沒崩」的 seed-43/44 來上線,這是 **selection bias**——真實部署只會訓一個 seed,而它有機會就是崩掉的那個。所以要問:**能否用一個部署前可算、不碰 live test 的準則自動選 seed?** 最自然的候選是 **sim-validation**(用 sim 分數選),前提是它要能預測 live 排名。直接檢驗:把 sim 裡**崩掉的 seed-42**(hetero sim：cvar −133% / mean −139%,遠差於 43/44 的 −6~−43%)也丟上實機跑同一套四方協定。

**完整四方 × 三 seed 的 live 結果**（每 arm n=246、非飽和 regime、real cuBLAS；JCT/p95/p99/CVaR 單位秒,ΔJCT% 為對 score 配對、負值=較慢）：

**seed-42（sim 崩潰：cvar −133% / mean −139%）**

| arm | JCT(mean) | p95 | p99 | CVaR | ΔJCT% | t-test p |
|---|--:|--:|--:|--:|--:|--:|
| score | 170.5 | 399.8 | 430.9 | 376.8 | —（baseline）| |
| SAC | 176.0 | 404.8 | 429.1 | 382.8 | −3.2 | 2.2e-11 |
| RDSAC-mean | 172.7 | 395.8 | 426.1 | 374.1 | **−1.3** | 0.044 |
| RDSAC-cvar | 175.0 | 404.8 | 426.2 | 382.5 | −2.6 | 7.0e-09 |

**seed-43（sim 正常）**

| arm | JCT(mean) | p95 | p99 | CVaR | ΔJCT% | t-test p |
|---|--:|--:|--:|--:|--:|--:|
| score | 167.8 | 398.0 | 425.6 | 374.7 | —（baseline）| |
| SAC | 174.0 | 402.0 | 432.4 | 379.5 | −3.7 | 3.7e-16 |
| RDSAC-mean | 174.5 | 402.0 | 435.8 | 379.5 | −4.0 | 2.1e-17 |
| RDSAC-cvar | 175.5 | 403.8 | 432.8 | 381.6 | −4.6 | 5.3e-12 |

**seed-44（sim 正常）**

| arm | JCT(mean) | p95 | p99 | CVaR | ΔJCT% | t-test p |
|---|--:|--:|--:|--:|--:|--:|
| score | 170.0 | 399.0 | 432.8 | 375.8 | —（baseline）| |
| SAC | 176.8 | 404.8 | 432.2 | 382.9 | −4.0 | 5.5e-10 |
| RDSAC-mean | 173.4 | 400.0 | 428.0 | 379.4 | −2.0 | 1.1e-06 |
| RDSAC-cvar | 175.4 | 405.5 | 432.2 | 381.7 | −3.1 | 1.8e-14 |

**綜合三 seed（mean ± std，這正是「報全 seed 分布」建議的落地形式）：**

| arm | JCT(mean) | p95 | p99 | CVaR | ΔJCT% vs score |
|---|--:|--:|--:|--:|--:|
| score | 169.4 ± 1.4 | 398.9 ± 0.9 | 429.8 ± 3.7 | 375.8 ± 1.1 | —（baseline）|
| SAC | 175.6 ± 1.4 | 403.9 ± 1.6 | 431.2 ± 1.9 | 381.7 ± 1.9 | **−3.6 ± 0.4** |
| RDSAC-mean | 173.5 ± 0.9 | 399.3 ± 3.2 | 430.0 ± 5.1 | 377.7 ± 3.1 | **−2.4 ± 1.4** |
| RDSAC-cvar | 175.3 ± 0.3 | 404.7 ± 0.9 | 430.4 ± 3.6 | 381.9 ± 0.5 | **−3.4 ± 1.0** |

**綜合推論：**

1. **三個 learned arm 跨 seed 一致輸 score**：ΔJCT% = SAC −3.6±0.4、cvar −3.4±1.0、mean −2.4±1.4——**全部 (mean − std) 仍 < 0**,沒有一個 arm 在任何 seed 翻正。負結果**穩健**,不是單 seed 運氣。
2. **沒有任何方法改善尾端**：p95/p99/CVaR 四方在 std 內幾乎重疊;尤其 **RDSAC-cvar（專為壓尾設計）的 CVaR 反而略高於 score（381.9 vs 375.8）**——風險扭曲在此 regime 沒兌現它的賣點。RL 不但沒贏 mean,連它最該贏的尾端也沒贏。
3. **learned arm 之間的排名是雜訊**：RDSAC-mean（−2.4）名目最不爛,但其 ±1.4 與 SAC（−3.6±0.4）、cvar（−3.4±1.0）重疊——sub-ranking 不可下結論(呼應 §3.1「arm 排名是訓練雜訊」)。
4. **機制解讀**：live 下 RL 訊號很小(飽和 abstain → fallback、非飽和 placement 差異有限),所以異質感知策略只能在 score 的放置上做**小幅擾動,而擾動略微有害**(~−3% JCT、尾端持平或略差),且這小擾動的大小不取決於 sim 長相(故 §4.4 sim 不預測 live)。

**結論（與全文一致、且現在統計顯著 + 跨 seed 穩健）**：真實 2×1 上,**沒有任何 DRL 排程策略贏過 score**;異質感知(item-1)把 §4.2 的大幅落後收斂成「**一致、顯著、但小幅(~−3%)的落後且無尾端紅利**」。要真的贏過 score,瓶頸不在演算法花招,而在(a)讓 RL 訊號在 live 真正起作用(降低 fallback/abstain 比例)、(b)穩定訓練(§6)、(c)用真實 transition 做保守 RLPD(§6,已試 trace-replay 失敗)。

**判定：sim-validation 不能預測 live,自動選 seed 失敗。** sim 排名是 43≈44 ≫ 42（42 崩),但 **live 排名是 42≈43≈44**——seed-42 落在同一個窄帶(−1.3~−3.2%)、甚至是所有 learned arm 裡**最不爛的那個**(RDSAC-mean −1.3%)。**sim 的崩潰沒有轉移到 live**;若按 sim-validation 選 seed,會錯誤丟掉一個實機表現正常的 checkpoint。

**機制**:live 下 RL 訊號很小——飽和時 arm 大量 abstain → 退回 vanilla Slurm,非飽和時 placement 差異也有限——所以**不管 sim 長相如何,所有 learned arm 在 live 都收斂到「比 score 略差 ~−1~−5%」**。sim 端的高變異(seed collapse)在 live 被這個「小訊號 + fallback」洗掉了。

**可解釋、穩定的實機評估協定（結論）：**

1. **測量端控變異**：非飽和 regime(poisson + `--mps-oversub 1.0`)+ 高 n + per-job paired CRN（§4.3.1）——這把「排名不穩」的測量雜訊壓掉,讓「score vs DRL」的 gap 變顯著且可重現。
2. **訓練端不要手動挑 seed,也別靠 sim 自動選**（已證 sim 不預測 live）。改**報全 seed 分布**(mean±std over all seeds,含失敗)——這正好可行,因為 live 對 seed 穩健:三個 seed × 三個 arm 全部落在 −1.3~−4.6%。
3. **不要排名 DRL arm 之間**(SAC vs mean vs cvar 的高低是訓練雜訊);只報**穩健的 top-level 結論**:`score > 所有 DRL arm`,跨 seed、跨 arm、統計顯著。

> 原始檔 `runs/htab_4arm_s42_20260623-101559/`（seed-42 live,sim-validation 檢驗）

### 4.5 Slurm 原生排程 baseline 對照（R3）

先前實機只比「自家 score 啟發式 vs DRL」,審稿人會問:score 贏不贏得了更單純的排程器?這裡把 score 與三個 **Slurm 原生**排程同場上線比(工程價值=所有排程都真的上線,而非 sim):

- **FCFS**：`SchedulerType=sched/builtin` + `PriorityType=priority/basic`（嚴格 FIFO）
- **multifactor**：`PriorityType=priority/multifactor`（age/jobsize 加權,chart 既有權重）
- **packing**：`SelectTypeParameters` 加 `CR_Pack_Nodes`（優先塞滿單節點）
- **SJF 不另比**：Slurm 無原生 SJF plugin,現有 score 的 `f_runtime_short`+predictor 就是 live 版 SJF 近似。

每臂把 Lua 的 `SCORE_APPLY`/`RL_ENABLED` 關掉(讓 Slurm 原生排程純粹接管),改 ConfigMap + **重啟 slurmctld**(`job_submit.lua`/`slurm.conf` 都是 subPath mount,不能熱載),再跑與 §4.3.1 相同的 heavy-tail CRN workload(philly、n=100、poisson、`--mps-oversub 1.0`、mps-buckets 25/50/75/100、partition `gpu`、**無 RL placement**)。

**陷阱:run-order 漂移。** 單 pass(block design、臂依序跑)的結果是 fcfs **+5.0%**、packing +1.6%、multifactor +1.0% 全顯著「贏」score——但改善幅度與**跑的先後順序完美單調相關**,是叢集隨時間變化(GPU restore 後暖機)的漂移,不是排程器差異。為洗掉它,跑 **3 個 seed × 3 種不同臂順序**,讓每臂在不同 pass 落在不同 run-position:

| arm | seed42 位置 | seed43 位置 | seed44 位置 | ΔJCT% vs score（各 seed）|
|---|--:|--:|--:|---|
| fcfs | 4（最後）| 1（最先）| 3 | **+5.0 / +0.5 / −0.4** |
| packing | 3 | 2 | 1 | +1.6 / +0.8 / −0.6 |
| multifactor | 2 | 3 | 4 | +1.0 / +0.9 / −0.7 |

fcfs 的「優勢」隨它跑第幾位從 +5.0(跑第4)掉到 −0.4(跑第3、p=0.068 不顯著)——**排名是漂移假象**。位置對照後的 cross-seed 聚合:

| arm | mean JCT | p95 | p99 | CVaR | ΔJCT% vs score |
|---|--:|--:|--:|--:|--:|
| score | 182.7±31.7 | 369.1±46.6 | 409.8±37.7 | 349.2±43.0 | — |
| multifactor | 181.8±30.3 | 364.5±42.7 | 405.9±35.6 | 346.7±39.4 | **+0.4±1.0** |
| packing | 181.5±30.7 | 365.9±42.1 | 405.2±33.5 | 346.3±38.3 | **+0.6±1.1** |
| fcfs | 179.7±32.0 | 362.5±37.2 | 402.4±28.2 | 343.2±32.8 | **+1.7±2.9** |

**結論:三個 ΔJCT% 的 mean±std 全部跨過 0 → 真實 2×1 上,生產 score 啟發式與 Slurm 原生 FCFS/multifactor/packing 統計打平,沒有任何排程器有優勢。** 這把「沒人贏 score」放進更大的脈絡:不只 DRL 贏不了 score,**連 score 自己對 trivial baseline 都只是打平**——在此規模,整個排程策略空間是平的,瓶頸不在挑排程器,而在叢集規模/工作負載本身的結構。方法學上這也再次示範 §4.4 的教訓:**單 pass 的排名會被 run-order 漂移污染,必須位置對照 + 多 seed**。

> 原始檔 `runs/baseline_sweep_20260624-010350/`（seed-42 forward）、`runs/baseline_passB_s43_*`、`runs/baseline_passC_s44_*`、聚合 `runs/baseline_confirm_20260624-064343/SUMMARY.md`；切換工具 `eval/scripts/baseline_switch.py`、聚合 `eval/scripts/aggregate_baseline.py`。

---

## 5. 結論

| 問題 | 結論 |
|---|---|
| DRL path 能在 2-node 上跑？ | 可以。166-dim checkpoint 上線、共享 `gpu` partition、submit-時 placement A/B 全 job 乾淨完成（§4.1）。 |
| 先前 `alpha` 觸頂是真 bug？修好了？ | 是真 bug（return 尺度壓過 entropy ~300×）。已用 reward_scale 1000→20000 + 放寬 clamp 修好（§2.2）。 |
| RDSAC / SAC 在 2×1 贏過 score？ | **沒有。** sim 三方逼近打平、沒人贏過 score（§3.1）；live 2-node placement 三個 learned arm 全顯著輸 Slurm（§4.1）。在真實 2×1，沒有任何 DRL 策略 在可檢驗的設定贏過 score。 |
| 分布式 / 風險機制有用嗎？(score vs SAC vs RDSAC) | **2×1 下沒有可定論的優勢（multi-seed 確認）。** σ-sweep 三方（3 train seed）沒人贏 score、arm 排名是訓練雜訊（§3.1）；拆解上 σ=1.0 的 CVaR 反而扣分、分布式 critic 是相對有用的一半（§3.3）。live placement 三方全輸 Slurm 且 **seed-robust**（§4.1）。 |
| risk-sensitive(cvar) 優於 risk-neutral(mean)？ | **方向上 sim 內 cvar 較穩、但在雜訊內。** 2×1（§3.1）cvar 是最穩的 learned arm（σ=0.5 最接近打平）；但**三方都仍輸 score**，差異落在單 seed 擺盪內。**而 live placement 反而 cvar 最差**（過度集中 4070 最兇，§4.1）——sim 與 live 對 cvar 的評價相反。 |
| 共置動作（PACK/ISOLATE）有用嗎？ | **沒有，即使有 2 GPU。** 2×1 colocation ON 仍輸 OFF ~20 pts（§3.4），「價值需 ≥2 GPU」被推翻；瓶頸是動作空間加倍（33→65）的 underfit。 |
| 2-node placement 結果？ | **負（四方一致、且 seed-robust）**。單 checkpoint：learned 全顯著輸 Slurm（−12~−32% JCT、p<0.01、drift-robust）。**multi-seed 確認**：3 個 train seed 各跑一次四方，每個 seed×arm 都輸 score（SAC −21.6±9.6、mean −13.4±4.4、cvar −21.2±6.1）。機制：learned 全把負載擠到 4070（85–89% vs score ~50%），不是 seed 運氣（§4.1）。 |
| 退化能修嗎？ | **能、方向對。** 對症加 P1 load-balance shaping + P2 reward normalization（§3.6）：cvar/SAC 改善 +19~+31 pts、variance 收斂（cvar philly `−4.1±4.6`）、過度集中從 ~89% 拉回 ~71%，且**首次有 learned arm 在 sim 贏過 score**（philly cvar +2.3%）。仍未穩健贏 score、mean 反而退步——是「修訓練」方向的有效一步，下一步拿去真實 CUDA job 實機檢驗。 |
| 最穩定上線策略 | 保留 stale snapshot / low confidence / service down 時的 heuristic/Slurm fallback。**在策略證明能穩健均衡放置前，RL placement 不應蓋過 Slurm 預設**。 |

**工程貢獻**：(1) 可上線的 DRL inference path（非僅 notebook/sim）；(2) DRL 對齊 Ma et al. RDSAC，有單元/行為測試；(3) 定位並修好 temperature auto-tune 的 reward-scale 根因；(4) sim + live trace collector 已能支援後續 RLPD；(5) 乾淨的四方受控對照（score/SAC/RDSAC-mean/cvar）+ 隨機性/共置消融；(6) **2-node 上線管線**：共享 `gpu` partition、submit-時 RL placement（`-w`，因 Slurm 21.08 無法 post-submit 重釘節點，§4.1）、外加修掉 4 個只在多節點現形的 chart bug（releasePriority 科學記號 CrashLoop、netpol 漏列、`-H` hold 被 score/rl_hook 覆蓋、controller 一次只放一個 job）。

**核心一句話**：在真實 2×1，DRL 排程 **沒有在任何可檢驗的設定贏過 score，而且這結論對訓練 seed 穩健**——sim σ-sweep（3 train seed）三方逼近打平、沒人贏 score、arm 排名是訓練雜訊（§3.1–3.4）；**live 2-node placement（3 train seed）四方全輸 Slurm**——sleep job −7~−31% JCT（§4.1），**真實 CUDA job（含異質 MPS 需求、整張卡 job 都有）−7~−22%**（§4.2，輸在大 job 沒放對；跨 workload「沒人贏 score」穩健、arm 排名隨 workload 漂移）。瓶頸是**這套 train+placement 會結構性地長出退化的放置策略**——下一步要改的是這個（架構/訓練穩定性，§3.6 的 P1+P2 已是有效一步），而非宣稱「用了 DRL/risk-sensitive」就算贏。

## 6. 後續改進（Future Work）

從前面的實驗證明瓶頸並非演算法花招 (SAC vs. RDSAC)，而是**訓練退化**。策略會結構性地過度集中到單一 GPU 上，且發現**訓練高變異**（跨 seed 擺盪 30~90 pts）。這兩個瓶頸都是退化崩塌，所以改善優先級是**修訓練、別加花招**。

- [X] Load-balance reward shaping：避免過度集中，potential-based 節點均衡項（`balance_coef`）已把 cvar 的過度集中從 ~89% 拉回 ~71%。
- [X] Return normalization：：running-std PopArt-lite 把 cvar variance 從 16→5、整體 +19~+31 pts 靠近 score。
- [X] 重跑 multi-seed 比對：有效讓過度集中收斂、變異數下降、首次有 DRL 策略在模擬環境贏過 score。
- [X] 模擬和實機都用 multi-seed：σ-sweep 三方用 3 個訓練種子 (42/43/44)，實機 A/B 測試也用 3 個 seed 的 checkpoint 各跑一次四方
- [X] 向量化加速模擬：把 `sim_train(--num-envs N)` 接成 N 個 env 並行 rollout、共用同一 learner。注意：vec path 的 score-warmup 退回 random-legal（score-warmup 需 in-process `env._state`），且每 iteration 仍 `utd_ratio` 次更新，所以 UTD 隨 N 稀釋——要維持樣本效率就同步調高 `--utd-ratio`。
- [ ] σ 校準的外部效度：§3.2 的 σ 是合成 trace 的最難預測上界；應在真實結構化 trace（`load_philly()`）上重新量測，並把 σ-sweep 落在實測區間。
- [X] 真實 CUDA job 評估：已把 `sleep N` 換成參數化 cuBLAS workload（`eval/scripts/gpu_workload.cu`）並編譯到 `/shared/bin/gpu_workload`（兩節點），跑出分數 MPS 共置的實機評估 (§4.2)
- [ ] 補強 baseline：目前只比自家 score + vanilla SAC；補 FCFS / SJF（已有 oracle runtime）/ packing 啟發式與近似上界，讓 ΔJCT% 有尺度感。
- [~] 完整 PopArt return normalization：已實作（`dsac.py --use-popart`，輸出保值 rescale、誤差 9.5e-7 驗證過），但 **3-seed 反而比 P2 reward-norm 更差（−23~−48% vs −12~−33%）、變異更大** → **否決**，保留關閉、預設仍用 P2。
- [X] 暴露 GPU 異質性（item-1）：obs 加 per-card `gpu_type`、sim 加 per-node 速度（4070=1.0×/3080≈0.25×）。sim 配對贏 10/12；**實機把 learned 從 §4.2「全輸 −7~−22%」拉到「打平 / 名目 +3.9%」（RDSAC-mean，§4.3）**。seed-43 hetero checkpoint 為目前最佳。
- [ ] **讓 §4.3 的 +3.9% 達到統計顯著**：目前 p=0.116。三個正交手段——(a) **更多配對樣本**（更多 round / job，n 96→300+ 收緊 SE）；(b) **降低 JCT 變異的 regime**（§4.3 是 slowdown_p99≈460 的極端飽和，巨大尾端淹沒訊號；改用非飽和負載讓效果浮出）；(c) **多 seed checkpoint**（跨 3 個訓練 seed 報 mean±std，排除單 seed 運氣）。
- [ ] **RLPD 走真實 transition logging（取代 trace-replay）**：`--online-trace`（把 live sacct trace 當 score demonstrations 回放）**在飽和與非飽和兩個 regime 都退化 prior**（seed-43 −19.6%/−4.5% → −97~−174% / −79%），病根是 demonstration 機制本身（拉向 sim-score + reward/scale 不匹配）非資料飽和。正確路線：部署 `live_daemon`（SHADOW_MODE）記**真實 `(obs,act,rew)` transitions** → `rlpd_finetune --online-log` + **保守 offline RL**（大 offline anchor、低 online-ratio、限制更新步數、可加 CQL 式保守項）。**已試 host-side daemon（`kubectl exec` 包 squeue/scontrol）→ 不可行**：每次 poll 透過 kubectl 來回 ~1-2s 太慢,抓不到 job *完成* 的瞬間（transition 在完成時才寫）→ 只記到決策、0 筆 completion-transition → RLPD online buffer 空 → 一樣退化。**已修一個必要 bug**（`live_daemon` 的 `value_abstain` 寫死 -1.0,seed-43 value≈-10 → 全 abstain、0 transition;改成 env `VALUE_ABSTAIN` 可調）。下一步應把 daemon **部署進叢集**（低延遲 squeue/sacct），收滿真實 transitions 再做保守 offline RL。三次嘗試（trace-replay 飽和/非飽和、host-side transition-log）都退化 prior,**seed-43 base 仍為最佳 checkpoint**。
- [ ] 完整 PopArt 的替代：既然 PopArt 否決，溫度穩定改用「固定 α + 良好 reward_scale」或 per-arm reward 標準化。

---

## 6. 重現指令

**σ 校準（§3.2）**

```bash
PYTHONPATH=. .venv-m11/bin/python eval/scripts/measure_predictor_sigma.py \
  --trace sim/data/philly_subsample.json
```

**2×1 σ-sweep 三方 + 拆解（§3.1 / §3.3）**

```bash
PYTHONPATH=. .venv-m11/bin/python eval/scripts/sweep_stochastic.py \
  --sigmas 0.0 0.5 1.0 --total-steps 100000 --warmup-steps 2000 --n-jobs 50 \
  --seeds 42 43 44 45 46 --trace-families philly ali --risk-modes mean cvar \
  --n-nodes 2 --gpus-per-node 1 --curriculum --fixed-alpha --init-alpha 0.05 \
  --device cuda --out-dir runs/stoch_sweep_2x1_$(date +%Y%m%d-%H%M%S)
# 9 個 arm（3 σ × {SAC, RDSAC-mean, RDSAC-cvar}）逐 σ 存 checkpoint + sweep.json；
# σ=1.0 那組 checkpoint 即 2-node live A/B 的輸入。
# 共置消融（§3.4）：另跑 --sigmas 0.5 --interference 0.3 --risk-mode cvar --no-sac，ON/OFF 各一次（加/不加 --colocation）。
```

**重尾 + 高競爭 2-node placement live A/B（§4.1，submit-時 -w，drift-robust round-robin）**

```bash
# 前置：建 166-dim image（boot model = σ=1.0 cvar + /models/htab/{sac,rdsac_mean,rdsac_cvar}.pt）、
# 部署到 2-node + 共享 gpu partition、port-forward 8002（見 services/rl_scheduler/Dockerfile.htab2x1）
#   docker build -t slurm-rl-scheduler:htab2x1 -f services/rl_scheduler/Dockerfile.htab2x1 .
#   兩台都 import：docker save ... | sudo k3s ctr -n k8s.io images import -（每個 node）
#   helm upgrade ... --set rlScheduler.image.tag=htab2x1（含共享 gpu partition）+ 重啟 slurmctld
#   kubectl port-forward -n slurm svc/rl-scheduler 8002:8002 &
KUBECONFIG=~/.kube/config PYTHONPATH=. .venv-m11/bin/python -m eval.scripts.run_heavytail_ab \
  --serve-url http://localhost:8002 --login-pod <slurm-login-pod> \
  --placement --gpu-nodes slurm-worker-gpu-rtx4070-0,slurm-worker-gpu-rtx3080-0 \
  --partition gpu --interleave --family philly --n-jobs 24 --target-max-s 30 \
  --sigmas 0.0 1.0 --rounds 4 --warmup 1 \
  --sac-ckpt /models/htab/sac.pt \
  --rdsac-mean-ckpt /models/htab/rdsac_mean.pt \
  --rdsac-cvar-ckpt /models/htab/rdsac_cvar.pt \
  --out-dir runs/htab_live_place4_$(date +%Y%m%d-%H%M%S)
# 跑完還原 production image。工程規格見 docs/live-ab-heavytail-spec.md
```

---

## 7. 資料集來源

§3 的 simulator benchmark **並非直接重放原始資料集**，而是用 `sim/loader.py` 的合成生成器，分布參數依下列公開 GPU cluster trace 的已發表統計校準（可離線、無網路重現）。Philly 另提供真實 `cluster_log_data.json` 的載入路徑（`load_philly()`）。

| Trace family | 生成器 | 模仿來源 | 連結 / 論文 |
|---|---|---|---|
| `philly` | `generate_philly_like`（亦支援 `load_philly` 真實重放）| Microsoft Philly GPU cluster trace | github.com/msr-fiddle/philly-traces — Jeon et al., *Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads*, USENIX ATC 2019 |
| `ali` | `generate_ali_like` | Alibaba PAI GPU cluster trace（MPS-fractional、短尾、多單卡）| github.com/alibaba/clusterdata（`cluster-trace-gpu-v2020`）— Weng et al., *MLaaS in the Wild*, USENIX NSDI 2022 |

> 注意：`philly` / `ali` 是「**統計近似**」而非逐筆原始資料；數值反映的是這些 trace 的工作負載**特性**（job 大小分布、到達節奏、runtime 尾部），不等同在原始 production log 上的表現。嚴格對照時建議用 `load_philly()` 載入 msr-fiddle/philly-traces 的真實 trace。
