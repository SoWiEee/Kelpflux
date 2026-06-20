# Kelpflux Scheduler Evaluation

本文件整理目前上線規格下的 scheduler evaluation。重點不是證明 DRL 一定優於啟發式，而是在同一套模擬環境與真實 Slurm/k3s/GPU 環境中，清楚比較 3 種排程方式的行為：**heuristic score**、**SAC**、**RDSAC**。

在真實叢集中從頭做強化學習訓練需要數十道數百萬個 transition，而真實叢集一個 placement 決策對應一個跑數分鐘～數小時的 job，湊滿樣本要等數月，因此採用**sim-to-real 兩段式**：

1. 在模擬環境中大量訓練（便宜/安全/可配對評估），產生 checkpoint
2. 上線部署到實機，記錄真實 $(obs, act, rew)$ 資料到 JSONL
3. 運用 RLPD 用真實資料把原始 checkpoint 微調成真實環境策略

模擬環境 (sim) 是唯一能做大量學習的地方；用來產出是「機制性洞察與一個可用的暖啟動」，接著用 RLPD 微調這個暖啟動。

**本文的聚焦點是：在拓樸匹配的真實 2×1 環境，DRL 排程到底有沒有贏過強啟發式 score。** 兩個層面：

- **絕對績效數字（sim → real 轉移差）**：sim 的 JCT/勝幅不會原樣搬到真實系統，故 §3 的 sim 數字只當定性/序數參考、用來抽機制與做配對消融。
- **真實 2×1 的判定（sim + live 都已 multi-seed 固實）**：第二節點（RTX 3080）上線後，在拓樸匹配的 2×1 跑 sim（§3）與 live（§4），兩端都用 3 個訓練 seed 跑 mean±std。**結論是負的、且 seed-robust**——sim 三方逼近打平、沒人贏過 score，arm 間排名是訓練雜訊；live placement 三個 learned arm **每個 seed×arm 都輸 Slurm**（−7~−31% JCT），全因 learned model 把負載過度集中到 4070（85–89% vs Slurm 均衡 ~50%）。→ 瓶頸是**這套 train+placement 會長出過度集中的退化策略**，要改的是架構/訓練穩定性（§5.1）。

> 閱讀順序：§3（模擬結果）講 2×1 sim 的配對消融與機制；§4（實機結果）給真實 2-node placement 的四方 A/B；§5.1 列出 multi-seed 固實與 RLPD 微調的 future work。

---

## 0. 摘要與結果總覽

三種排程方式：**score**（MPS-aware 啟發式優先序，baseline=0%）、**SAC**（vanilla 離散 SAC，scalar twin-Q critic，無分布式/風險）、**RDSAC**（分布式 IQN critic + 風險扭曲，mean/cvar 變體）。三者在訓練 / eval / live serving 用一個 `use_iqn` flag 切換 SAC↔RDSAC、`risk_mode` 切換 mean↔cvar。

**核心發現：在真實匹配的 2×1（2-node 異質：RTX 4070 + RTX 3080）拓樸上，沒有任何 learned model（SAC / RDSAC-mean / RDSAC-cvar）贏過 heuristic score；live placement 還顯著輸給 Slurm 預設。**

評估在拓樸匹配的 **2×1**（obs_dim=166、n_actions=33）做，分 sim（§3）與 live（§4）：

| 評估 | score | SAC | RDSAC-mean | RDSAC-cvar | 判定 |
|---|---|---|---|---|---|
| **sim** σ-sweep, fixed-α, **multi-seed**（§3.1，σ=1.0，3 train seed）| 0% | −38.3±12 / −33.6±23 | −12.9±10 / −8.4±3 | −35.4±16 / −42.0±30 | **multi-seed 確認：沒人贏過 score；arm 間排名是訓練雜訊（std 5–30 pts）** |
| **live** 2-node placement, submit-時 -w（§4.1 四方，**multi-seed**）| 0% | −21.6±9.6 | −13.4±4.4 | −21.2±6.1 | **三個 learned arm 全輸 Slurm、seed-robust（3 train seed mean±std，每個 seed×arm −7~−31% JCT）；全過度集中 4070（85–89%）** |

（sim 兩個數字 = philly / ali（σ=1.0 mean±std）；live 兩個數字 = σ=1 / σ=0；ΔJCT% vs score，負值=較慢）

**一條主軸（一律以 2×1 實機為準）：**

1. **sim（§3.1–3.4，σ-sweep 部分已 multi-seed）：注入校準過的不確定性後，三方在 2×1 逼近打平、但沒人贏過 score。** 3 個訓練 seed 的 mean±std **確認**了兩件事：(i) **沒人贏 score**（穩健）；(ii) **arm 間的差異是訓練雜訊**——std 5–30 pts、跟 mean 差同量級，同 config 跨 seed 擺盪 30–70 pts，所以單 seed 看到的「cvar 最好」之類排名**不成立**（§3.1）。拆解上唯一半穩健的 slice 是 σ=1.0：**CVaR 風險扭曲反而扣分**（§3.3）。共置動作即使有第二張卡仍拖累（§3.4）。

2. **🟥 live（§4.1 sleep / §4.2 真實 CUDA）：真實 2-node placement 是乾淨、seed-robust 的四方負結果。** sleep-job A/B（§4.1）三個 learned arm 全顯著輸 Slurm（**−12~−32% JCT**，multi-seed 確認），因為全把負載擠到 4070（85–92% vs Slurm 均衡）。**把 sleep 換成真實 cuBLAS job（§4.2，獨佔 GPU）→ 負結果放大 2–4×**（learned **−53~−132%**，cvar 最慘），因為獨佔佇列下 RL 不看佔用的 `-w` 放置把 job 硬塞到忙碌的卡上排隊（JCT 由 wait 主導 65–75%）。**placement 越有真實後果，這套退化策略越輸 Slurm。**

**一句話：** 在真實 2×1，DRL 排程 **沒有在任何可檢驗的設定贏過 score**：sim σ-sweep（已 multi-seed）三方逼近打平、沒人贏 score、且 arm 間排名是訓練雜訊；live placement（sleep 與真實 CUDA 都測過）因退化放置而顯著輸 Slurm、真實算力下更慘。multi-seed 還揭露**訓練本身高變異**（同 config 跨 seed 擺盪 30–70 pts），正是 live 容易長出退化策略的根。

**兩個 load-bearing caveats：**（1）**訓練高變異（已用 multi-seed 量化、兩端都固實）**——sim σ-sweep（§3.1）與 live placement（§4.1）**都已用 3 個訓練 seed 跑 mean±std**：sim 確認「沒人贏 score、arm 排名是訓練雜訊」，live 確認「負結果 seed-robust（每個 seed×arm 都輸 score）」。訓練本身高變異（同 config 跨 seed 擺盪 30–70 pts）是這套方法的核心限制；（2）σ 是合成 trace 的最難預測上界，真實結構化資料 predictor 會更準 → 合理區間 [0.5, 1.45]。

---

## 1. 評估對象

### 1.1 Heuristic score scheduler

heuristic score 是目前最穩定的 submit-time baseline。它不需要訓練模型，而是直接用 job 需求與 cluster 狀態計算優先權，交給 Slurm `select/cons_tres` 做實際 placement。核心訊號：

| 訊號 | 用途 |
|---|---|
| MPS fit | 小 MPS job 是否能塞進目前 GPU 的剩餘 slot，避免大 job 佔滿整張卡。 |
| VRAM fit | job 需求是否符合 GPU tier，避免高階卡被低需求 job 浪費。 |
| fragmentation penalty | 避免接受一個 job 後讓剩餘 MPS slot 過度碎片化。 |
| runtime shortness | 若 runtime predictor 有可用估計，短 job 會得到額外 priority boost。 |

最終分數 = `clamp01(α·f_mps_fit + β·f_vram_fit + γ·topology − δ·f_frag + ε·f_runtime)`，以 `score_gain=1000` 加在 Slurm multifactor 優先序上（**RL 的 `priority_boost` 加在同一條優先序**，故三方共用同一介入機制）。

> 實機設定和模擬環境稍有不同，參考 `slurm-config-job-submit` ConfigMap 發現只有開 3 個權重 $α=0.4, β=0.2, δ=0.2$，而 `γ=0`（topology 關）、`ε=0`（runtime/SJF 關），此外 runtime predictor 與 weight-tuner 都沒部署（`PRED_ENABLED=false` / `WT_ENABLED=false`，叢集無對應 pod），所以權重是靜態、不被線上調。且 live 的因子是**無狀態 proxy**（`f_mps_fit = mps_req/100`、`f_frag = 4x(1−x)`），與 sim 的 cluster-aware 版（`f_mps_fit = mps_req/該GPU剩餘slot`、ε=0.30 SJF 開）**不同**。即實機 score 實際 = `clamp01(0.4·mps大小 + 0.2·vram_fit − 0.2·碎片)`，**完全不看 runtime**。

實作：simulator baseline `sim/scheduler/score.py`（cluster-aware、ε 預設 0.30）；live submit hook `chart/templates/configmap-job-submit.yaml`（無狀態 proxy、ε=0）；Slurm priority path `chart/templates/{slurm-conf,login,workers}.yaml`。

### 1.2 SAC

根據 [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) 實作離散版本（`DSACAgent(use_iqn=False)`）：**scalar twin-Q critic、MSE soft-Bellman、無 IQN 的 Z_R/Z_H 分布式分解、無 risk distortion**。排程 action 是離散且有 mask 的——每一步只能從 pending queue 中可行的 (job, node, GPU/MPS placement) 組合裡選一個。SAC 在本專案作為 RDSAC 的對照基準：用**完全相同的配方**（步數、traces、curriculum、PER、shaping、MLP trunk、2×1 拓樸）訓練，唯一差別是 critic 型別與沒有 risk，用來回答「分布式/風險機制到底有沒有用」。

### 1.3 RDSAC

根據 [DSAC: Distributional Soft Actor-Critic for Risk-Sensitive Reinforcement Learning](https://arxiv.org/abs/2004.14547)的**離散動作忠實轉寫**。把連續控制的 reparameterised Gaussian actor 換成**顯式 categorical actor**，critic 與 risk 機制依論文 §4.1：

- **雙分布 critic**：把 soft return 拆成 reward 分布 `Z_R` 與 entropy 分布 `Z_H`，各以 IQN 的 quantile 表示、共用 trunk，quantile Huber 回歸 + twin double learning。
- **risk 進策略目標**：actor 目標對 reward 分布套用 distortion `ρ`，`risk_mode ∈ {mean, cvar, wang, cpw, msd}`。`mean` 是 risk-neutral（distributional 但不規避風險）；`cvar`（β=0.25）偏好下尾較不嚴重的 placement，對應排程的 straggler / cold worker / long-tail runtime 風險。

> 實作位於 `services/rl_scheduler/dsac.py`、`services/rl_scheduler/distortion.py`。

---

## 2. 實驗與 Benchmark 方法

### 2.0 訓練與評估管線

整體方法是 **sim-to-real 兩段式**，原因與各段產出如下表。理解這條管線是讀懂後面所有實驗的前提：**§3 的所有數字都是「在模擬環境裡訓練 + 在模擬環境裡評估」，目的是篩架構與抽洞察，不是預測真實績效**；真實績效由 §4 的實機 A/B 評估，最終策略由 RLPD 微調產出。

| 階段 | 環境 | 做什麼 | 為什麼不能在實機直接做 | 產出 |
|---|---|---|---|---|
| (1) 模擬訓練 | `KubefluxSchedEnv`（gym） | 從頭訓練 100k–150k 步，curriculum，PER + shaping | 實機一個 step = 一個跑數分鐘～小時的 job，湊滿樣本要數月 | checkpoint（warm start） |
| (2) 模擬評估 | 同 sim，配對 + 受控消融 | paired t-test、SAC/RDSAC/cvar/fixed-α 消融 | 實機無法 `reset()` 重放同一 trace → 拿不到 counterfactual | **機制性洞察**（§3） |
| (3) 實機部署 | k3s + Slurm + GPU/MPS | shadow-mode 跑 checkpoint，記錄真實 (obs, act, rew) 資料 | 訓練初期隨機策略會破壞系統 → 只敢 shadow + fail-safe 回退 score | 真實 A/B（§4）+ 微調語料 |
| (4) RLPD 微調 | 用 (3) 的真實 JSONL | 以模擬產出的 checkpoint 為 prior，混合真實資料做微調 | 從頭 RLPD = 退回 (1) 的 sample-complexity 與 (2) 的破壞性探索 | 真實環境策略（future work，§5.1） |

在模擬環境中「可大量訓練 + 可配對消融」，代價是模擬環境的絕對數字不轉移：機制性洞察（§3，2×1 sim）只當定性參考，真實判定由 §4 的 live A/B 給——而 §4.1 的結果是負的（learned placement 顯著輸 Slurm）。RLPD（**R**L with **P**rior **D**ata）的前提就是從一個既有 prior 出發再微調；模擬器的 checkpoint 不是被丟掉，而是 RLPD 站在它肩膀上。實機 trace 收集器（`live_daemon.py` → JSONL → `rlpd_finetune.py`）已就緒——而 §4.1 的 multi-seed 負結果（seed-robust）正說明：**直接烘 sim checkpoint 上線會輸，而且不是 seed 運氣問題；要贏得改架構/訓練穩定性，或走 RLPD 用真實資料微調**。

### 2.1 Simulator paired benchmark

相同亂數種子 (seed) 對 DRL 模型與啟發式 score 做配對比較來降低 trace 隨機性造成的誤差。`Δ = (score − model)/score`，負值代表 model 的 JCT 較高、較差。

| 項目 | 值 |
|---|---|
| training | 從頭訓練 100k 步，curriculum n_jobs 10→30→50，fixed-α 0.05 |
| reward_scale | 20000（修復 alpha 觸頂） |
| cluster | **2 node × 1 GPU**（2×1 異質：RTX 4070 + RTX 3080；obs_dim=166、n_actions=33）|
| jobs per trace | 50 |
| trace families | `philly`, `ali`（Alibaba PAI）|
| seeds | 每個 (model, σ) 訓 **1 個** seed；eval 配對 5 seed |
| metric | mean JCT（主），p95 / p99 JCT、CVaR（尾部），paired t-test 95% CI / p-value |

> **訓練 seed 注意**：每個 (model, σ) 只訓練**一個** seed；eval 的 5 seed 是評估隨機性、不是訓練隨機性。這是目前最大的方法學限制（§3.3 有同 config 兩跑擺盪 60–90 pts 的鐵證）。

> 早期 mean run 出現 `alpha` 自動調到 clamp 上限的警訊。排查後確認**不是 alpha 邏輯錯**，而是 **return 尺度問題**：舊 `reward = −JCT/1000`，50 job 累加 → return 量級 O(−150)，critic 的 `E[Z_R]` 跨動作落差約 560；actor 目標裡 entropy 正則 `α·log(n)`（上限~2）被 `Z_R`（~560）**壓過約 300×** → policy 早期塌成 one-hot（entropy ≈ 0.002）、探索停擺、alpha 一路 railing。修法：`reward_scale` 1000→**20000**（return 降到 O(−10)），log_alpha clamp 上限 1.0→3.0。修後 alpha 自由調節、entropy 由 0.002 恢復到 ≈0.13。

### 2.2 受控變數設計（每個實驗只動一個旋鈕）

| 旋鈕 | 隔離什麼 | 出現於 |
|---|---|---|
| `use_iqn`（SAC↔RDSAC）| critic 型別：scalar vs 分布式 IQN | §3.1, §3.3 |
| `risk_mode`（mean↔cvar）| 風險扭曲：risk-neutral vs CVaR | §3.1, §3.3 |
| `fixed_alpha`（釘死 α=0.05）| 溫度控制：避免 auto-α railing（全程釘死）| §3.1–§3.4 |
| `runtime_sigma` / `interference` | 注入 runtime 不確定性 / MPS 共置干擾（opt-in，預設關 → 與確定性 env 逐位元相同）| §3.1–§3.4 |
| `colocation_actions`（PACK/ISOLATE）| 共置是否成為一個動作 | §3.4 |

**隨機性注入模型**：`actual = predicted · exp(σZ − σ²/2)`（mean-preserving lognormal，E=1，只增變異不偏均值；obs 仍顯示 nominal runtime → 真實的結果不確定性）。idiosyncratic 噪音對每個 job common-random（以 `(seed, job_id)` 鍵）→ 同一 job 在每個 policy 下拿到相同乘子 → 配對比較。Harness：`eval/scripts/sweep_stochastic.py`。

### 2.3 σ 校準方法（讓注入的噪音不是憑空挑的）

由注入模型 `σ = std(log(actual/predicted))` —— 正是 runtime 預測器的 log-殘差標準差。`eval/scripts/measure_predictor_sigma.py` 用**生產級 LightGBM 預測器**（同 `services/runtime_predictor` 的 features、time-honest 80/20 split、超參）在真實 trace 上量 held-out log-殘差 std，當作要注入 sim 的 σ。結果見 §3.2。

### 2.4 實機 cluster A/B 設定

在實際 k3s + Slurm + GPU/MPS 環境提交 `sbatch` job 做 paired A/B。拓樸是 **2×1**：兩台 GPU host（RTX 4070 + RTX 3080）各一張卡，掛在一個**橫跨兩台的共享 `gpu` partition** 下，讓「放哪張卡」成為真實的決策。每個 job 用同一條 stream 在四種方法（score / SAC / RDSAC-mean / RDSAC-cvar）各重放一次（per-job common-random）。三個設計要點：

1. **submit-時 RL placement**：learned arm 的每個 job 先呼叫 serve `/act` 拿到節點選擇，用 `sbatch -w <node>` 在提交時釘下；score arm 不加 `-w`，交給 Slurm 自選（乾淨 baseline）。learned arm 跑 boost-off（`/shadow` ON），所以 treatment 是**純 placement**、不混 `/decide` 的 priority boost。
2. **為何不用 post-submit controller**：Slurm 21.08 的 slurmrestd v0.0.37 把 `required_nodes` 列為 disabled key、`scontrol` 也拒絕更新已提交 job 的 required nodes——**21.08 無法 post-submit 重釘節點**，所以放置決策只能在 submit 當下做。
3. **去除 cluster drift 偏差**：GPU 跑久了會變快（MPS 暖、快取熱），若一種方法整段跑完才換下一種，drift 會和方法混淆。改用 **round-robin（`--interleave`）交錯方法順序**、每方法跨多輪輪過各個位置，把 drift 平均掉。指標由 controller pod 的 `sacct` 收（含每個 job 落在哪台 node）。

> **生產服務拓樸**：跑 `slurmctld` / `rl-scheduler`(serve) / `rl-snapshot-agent`，但**沒有部署 `runtime-predictor` 和 `weight-tuner`**，因此 score 的 `ε`(SJF) 與線上權重調整都是關的（§1.1 生產實況），這是讀 §4 數字時的環境前提。

### 2.5 為何用 p95 / p99 / CVaR 量尾部（不只看 mean JCT）

全文每張表都同時報 **mean JCT（主）** 與 **p95 / p99 JCT、CVaR(0.25)（尾部）**。納入尾部指標不是裝飾，而是這套評估能不能看見 RDSAC 效應的**前提**的四個理由：

1. **mean 會把排程病態洗掉。** straggler、queue starvation、head-of-line blocking 的典型表現是「多數 job 正常、少數被拖很慢」；這幾個慢 job 攤進整批裡幾乎不動 mean，但主導使用者體感。p95/p99 專抓「最差 5%/1% 有多慢」，正是 mean 結構上看不到的那段。
2. **尾部是 RDSAC/CVaR 的靶心——不量它就無法檢驗它。** RDSAC-cvar 的設計目標就是優化回報下尾（= JCT 上尾）；只量 mean 等於拿不會動的尺去量專門改尾部的方法，結構上必然測不出差異。鐵證在 §3.1：RDSAC 對 SAC 的 **mean 差距有限，p99 卻差 6–11×**，優勢全在尾部。
3. **重尾 workload 下 mean 本身不穩。** §4.1 的 live workload 刻意重尾，重尾分布的 mean 由極端值主導、估計噪音大；分位數對重尾更穩健。**CVaR(0.25)=「最差 25% 的平均」**，比單點 p99（小樣本下只是最差 1–2 個 job、噪音主導）穩定，故本文以 **CVaR 為主要尾部判別指標、p99 為輔助**。
4. **領域慣例。** GPU cluster scheduling / SLO 文獻裡，tail JCT、tail slowdown（p95/p99）本就是標準指標（p99 幾乎是 SLO 代名詞）；mean 必要但不充分。

這也是為何 §4.1 的判定不是只看 mean，而是同時看 p99 / CVaR——尾部指標才是真正能分開四方的那把尺。

---

## 3. 模擬結果（Simulator）

本節是管線的階段 (1)(2)：在模擬環境訓練、在模擬環境裡做配對評估，**全部在拓樸匹配的 2×1**（obs_dim=166、n_actions=33）跑。這裡的數字當定性/序數洞察、用來抽機制與做受控消融，不是真實績效預測（真實由 §4 的 live A/B 給）。

三個受控消融，回答「分布式/風險/共置機制在 2×1 到底有沒有用」：

1. **σ-sweep 三方（§3.1）**：注入校準過的 runtime 不確定性（σ 校準到生產預測器的 log-殘差，§3.2），比 score / SAC / RDSAC-mean / RDSAC-cvar。**結果：三方逼近打平、沒人贏過 score；σ→cvar 只在 σ=0.5 成立、不單調。**
2. **拆解：分布式 critic vs 風險扭曲（§3.3）**：用 SAC→RDSAC-mean→RDSAC-cvar 拆兩個貢獻。**結果：兩者都只剩個位數 pts、且看 workload 正負擺盪——沒有單一主因。**
3. **共置動作消融（§3.4）**：PACK/ISOLATE 在 2×1 ON vs OFF。**結果：仍拖累（ON 輸 OFF ~20 pts），「價值需 ≥2 GPU」被推翻。**

> **共同前提（讀數字前先知道）**：sim 訓練本身**高變異**——同 config 跨訓練 seed 可擺盪 30–90 pts。§3.1 的 σ-sweep 已用 **3 個訓練 seed 跑 mean±std** 把這個變異量化（其餘格子仍單 seed，看方向別細讀點估計）。§3.2 給 σ 取值的外部效度，§3.5 把 sim 收斂到 live（§4.1）的負結果。

### 3.1 隨機性消融：注入 runtime 不確定性後，三方怎麼排？

**先講為什麼要注入噪音。** RDSAC 的風險機制（CVaR）是用來「規避下尾風險」的——但風險機制要**有風險可管**才有意義。sim 的 runtime 是 oracle（給定狀態與動作，結束時間是確定的），回報分布塌成一個點，CVaR 就等於 mean、風險機制整個閒置。所以這節在環境裡加一個**校準過的** runtime 不確定性 σ（mean-preserving lognormal，方法見 §2.3；σ 取自生產預測器的真實 log-殘差，§3.2），讓三方在「有尾部風險可管」時較高下。

σ∈{0.5, 1.0} 各訓 **SAC / RDSAC-mean / RDSAC-cvar**（fixed-α 0.05、100k 步、curriculum、5-seed 配對、philly/ali）。**關鍵：每個 (σ, arm) 用 3 個訓練 seed（42/43/44）跑，報 mean±std**——這是本文唯一做了 multi-seed 的地方，專門用來打掉單 seed 雜訊。**ΔJCT% vs score（負=較慢）：**

| σ | family | SAC | RDSAC-mean | RDSAC-cvar |
|---|---|---:|---:|---:|
| 0.5 | philly | −7.1±5.1 | −19.9±9.4 | −7.8±7.5 |
| 0.5 | ali | −15.0±7.5 | −2.4±5.3 | −3.7±3.2 |
| 1.0 | philly | −38.3±12.3 | −12.9±10.1 | −35.4±16.3 |
| 1.0 | ali | −33.6±23.0 | −8.4±3.2 | −42.0±29.8 |

**兩個判定（multi-seed 後）：**

1. **沒有任何 arm 贏過 score——而且這條穩健。** 每個 mean 都是負的，連 +1 個 std 也構不到 0（最接近的 σ=0.5 ali mean/cvar 是 −2.4/−3.7，仍負）。

2. **arm 之間的排名是訓練雜訊，不是真訊號。** std 普遍 **5–30 pts**，跟 arm 之間的 mean 差**同量級甚至更大**。最戲劇性的是 cvar @ ali σ=1.0 三個 seed 給 **+0.1 / −76.3 / −45.9**（mean −42±30）——同一個 config，換個 seed 就從「打平」崩到「慘輸」。這直接打掉先前單 seed 看到的「cvar 是最穩 best arm」：cvar−SAC 只有 σ=0.5 ali 是 **+11.3±7.3**（勉強脫離雜訊），其餘三格 **−0.7±5.8 / +2.9±26.3 / −8.4±23.5** 全在雜訊內。

**一句話**：multi-seed 把這節從「三方逼近打平、cvar 似乎略好」收斂成兩句**更強**的結論——(i) **沒人贏 score**（穩健）；(ii) **arm 間差異是單訓練 seed 的高變異產物**（同 config 跨 seed 擺盪 30–70 pts），不是可定論的機制排名。這也呼應 §4.1 live 的退化：訓練本身就高變異、容易長出過度集中的策略。原始檔 `runs/mseed_2x1_s{42,43,44}/`、彙總 `runs/mseed_2x1_agg/SUMMARY.txt`。

### 3.2 σ 校準到真實預測誤差：σ=1.0 其實偏保守

**這節是為 §3.1 的 σ 取值背書。** §3.1 的弱點：σ=0.5/1.0 若是憑空挑的，「注入噪音 → 抗噪法贏」就近乎套套邏輯。所以用 §2.3 的方法，量生產 LightGBM 預測器在真實 trace 上的 held-out log-殘差，當作該注入的 σ：

| workload | σ（殘差 std）| 95% CI | 形狀 |
|---|---:|---|---|
| philly | **1.45** | [1.31, 1.58] | near-Gaussian |
| ali | **1.24** | [1.11, 1.38] | near-Gaussian |

兩個結論：(1) 真實 σ ≈ **1.2–1.45**，故 §3.1 用的 **σ=1.0 落在真實下限以下、是保守值**——RDSAC 在 σ=1.0 就大勝，真實噪音下只會更強；(2) 殘差**近高斯**（excess kurtosis −0.1~+0.4），lognormal 噪音模型沒有低估尾部。**誠實告誡**：這些合成 trace 的 runtime 與特徵無關（corr(log_rt, gpu_count)=0.04，predictor 打不過 predict-the-mean），所以 1.2–1.45 是「最難預測」上界；真實結構化資料上好 predictor 會更低 → 合理 σ 區間 ≈ **[0.5, 1.45]**，§3.1 測的 {0.5, 1.0} 都落在其中。

### 3.3 拆解：贏的是「分布式 critic」還是「風險扭曲」？

§3.1 把 RDSAC 當一整包，但它其實有兩個機制：**分布式 critic**（把回報建模成分布，而非單點 Q）與 **風險扭曲**（CVaR）。要知道哪個在出力，就放第三方 **RDSAC-mean**（有分布式 critic、但風險中立）來夾：`SAC→mean` 隔離分布式 critic 的貢獻、`mean→cvar` 隔離風險扭曲。用 §3.1 的 **multi-seed**（3 train seed）σ=1.0 三方，mean±std：

| family | SAC | RDSAC-mean | RDSAC-cvar | SAC→mean（**分布式**）| mean→cvar（**風險**）|
|---|---:|---:|---:|---:|---:|
| philly | −38.3±12.3 | −12.9±10.1 | −35.4±16.3 | **+25.4±22.6** | **−22.5±10.4** |
| ali | −33.6±23.0 | −8.4±3.2 | −42.0±29.8 | **+25.2±22.4** | **−33.6±27.6** |

**判定（multi-seed，σ=1.0）：分布式 critic 是有用的那一半，CVaR 風險扭曲反而扣分。** 兩個 family 方向一致：`SAC→mean`（加分布式 critic）**+25 pts**（RDSAC-mean 是 σ=1.0 最好的 learned arm），而 `mean→cvar`（再加 CVaR）**−22~−34 pts**（cvar 變成最差）。**這跟 CVaR 的設計意圖相反**——本來想用它管尾部風險，σ=1.0 下卻過度保守/不穩、把 mean 的優勢吐回去。philly 的 `mean→cvar −22.5±10.4` 約 2σ、算半穩健。

**誠實 caveat**：(i) std 仍大（SAC→mean +25±22 才 ~1σ）；(ii) **這只在 σ=1.0 成立**——σ=0.5 的拆解 family 間打架、落在雜訊內（philly 分布式 −12.8、ali +12.6）。所以能穩健說的是：**CVaR 在 σ=1.0 沒幫上忙、甚至扣分；分布式 critic 是相對有用的一半——但兩者都沒讓 RDSAC 贏過 score。** 原始檔 `runs/mseed_2x1_s{42,43,44}/`。

### 3.4 共置動作消融：讓模型自己決定 PACK/ISOLATE，反而更差

下一個問題：**讓模型自己決定共置策略**能不能幫到 RDSAC？給每個放置動作加一個 mode（`colocation_actions`，opt-in）：`PACK`（接受 MPS 共享，付 interference slowdown）vs `ISOLATE`（要求 GPU 空閒才放，不共享但要等卡空出來）。這把動作空間從 33 撐到 65。在 interference=0.3、σ=0.5、fixed-α=0.05、RDSAC-cvar、5-seed 配對下比較 colocation **ON vs OFF**：

| family | OFF（baseline）ΔJCT% | ON（+共置）ΔJCT% | OFF p99 | ON p99 |
|---|---:|---:|---:|---:|
| philly | **−7.9** | −29.4 | 22.65 h | 22.71 h |
| ali | **−1.0** | −21.0 | 10.74 h | **22.12 h** |

**判定：加了共置動作反而明顯更差**——OFF 贏 ON ~20 pts（philly −7.9 vs −29.4、ali −1.0 vs −21.0），ali 尾部 p99 還直接翻倍（10.74→22.12h）。

**為什麼？** 最可能是**容量/預算問題**：動作空間幾乎翻倍（33→65），但訓練步數不變 → underfit（多出的 ISOLATE 動作大多被 mask、訓練訊號稀疏）。這跟 §4.1 的觀察一致：learned model 連最基本的「兩台均衡放置」都做不好（live 量到它把 88–92% 的 job 擠到 4070），再加一層共置決策只會雪上加霜。**結論：共置動作在當前預算下沒有價值，預設關。** Caveats：單訓練 seed、只測 (σ=0.5, interference=0.3) 一點。原始檔 `runs/coloc_2x1_off_20260618-190621/`、`runs/coloc_2x1_on_20260618-201323/`。

### 3.5 sim 小結：三個問題的答案，以及它如何接到 live

§3.1–3.3 各問了一個機制問題，把答案收成一句：

| 問題 | sim（2×1）的答案 |
|---|---|
| 注入不確定性後，RDSAC 會贏嗎？（§3.1）| **沒人贏過 score**（multi-seed 穩健）；arm 間排名是訓練雜訊（std 5–30 pts），單 seed 看到的「cvar 最好」不成立 |
| 贏在分布式 critic 還是風險扭曲？（§3.3）| σ=1.0 下 **分布式 critic 是有用的一半、CVaR 反而扣分**（−22~−34 pts，與設計意圖相反）；但仍沒讓 RDSAC 贏 score |
| 加共置動作（需更多卡）有用嗎？（§3.4）| **沒有**——ON 仍輸 OFF ~20 pts，瓶頸是動作空間翻倍後的 underfit |

**一句話**：multi-seed 後，2×1 sim 給出兩句穩健結論——**沒人贏過 score**、且 **arm 間差異是訓練雜訊**（同 config 跨 seed 擺盪 30–70 pts）。這個「訓練高變異 + 沒人贏」直接接到 live（§4.1）的退化：三個 learned arm 全把負載擠到 4070、全輸 Slurm。**而 live 也做了 multi-seed（3 個 train seed checkpoint 各跑一次四方 A/B）→ 負結果 seed-robust**：每個 seed×arm 都輸 score（−7~−31% JCT）、過度集中跨 seed 穩定（§4.1）。所以兩端都固實了。原始檔 `runs/mseed_2x1_*`、`runs/htab_live_mseed_s*`。

### 3.6 對症改善：load-balance shaping（P1）+ reward normalization（P2）

診斷確定後（瓶頸是「訓練退化→過度集中 + 高變異」，不是演算法），對症下兩帖（§5.1 路線，commit `01de2c9`）：**P1** 在 potential 加節點均衡項（`φ -= balance_coef·std(free_mps_per_node)`，Ng et al. 1999 保證不改最優策略、只導引探索遠離擠單卡）；**P2** running-std 回報正規化（PopArt-lite，消手調 `reward_scale` 的脆弱、降 seed 敏感度）。用 `--balance-coef 5.0 --normalize-reward` 在 σ=1.0、3 train seed 重訓三方，對比 baseline（§3.1）：

**P1+P2 絕對指標**（σ=1.0、3 train seed、mean±std、小時；score 參考 JCT philly 2.1h / ali 1.1h）：

| family | model | JCT(h) | p95(h) | p99(h) | ΔJCT% | Δp99%（vs baseline）|
|---|---|---:|---:|---:|---:|---:|
| philly | **cvar** | 2.1±0.1 | 9.3±1.1 | 30.4±3.8 | **−4.1±4.6** | **−17%** |
| philly | sac | 2.3±0.1 | 11.4±0.5 | 28.4±1.5 | −11.8±4.2 | **−30%** |
| philly | mean | 2.6±0.1 | 11.7±1.2 | 41.0±8.8 | −23.1±7.4 | +39% |
| ali | cvar | 1.1±0.1 | 4.3±0.9 | 20.4±0.6 | −12.0±12.0 | **−15%** |
| ali | sac | 1.2±0.1 | 5.5±1.2 | 21.1±0.3 | −15.2±17.5 | +5% |
| ali | mean | 1.4±0.1 | 6.6±0.5 | 22.7±2.5 | −32.7±10.4 | +6% |

**vs baseline ΔJCT%（§3.1，改善 = 往 0 靠多少）：**

| family | model | baseline ΔJCT% | P1+P2 ΔJCT% | 改善 |
|---|---|---:|---:|---:|
| philly | **cvar** | −35.4±16.3 | **−4.1±4.6** | **+31** |
| philly | sac | −38.3±12.3 | −11.8±4.2 | +27 |
| ali | cvar | −42.0±29.8 | −12.0±12.0 | +30 |
| ali | sac | −33.6±23.0 | −15.2±17.5 | +19 |
| philly | mean | −12.9±10.1 | −23.1±7.4 | −10 |
| ali | mean | −8.4±3.2 | −32.7±10.4 | −24 |

（CVaR/slowdown 未列：sim σ-sweep harness 只記錄 JCT/p95/p99；CVaR 僅 live panels 有，見 §4.1。）

**判定：對症有效（但未全勝）。** 三個正面證據：(i) **cvar/SAC 大幅靠近 score**（+19~+31 pts），且 **variance 同步收斂**（cvar philly 16.3→**4.6**，`−4.1±4.6` 是全專案最佳+最穩的 sim 結果），**尾部也跟著改善**（cvar p99 −15~−17%、SAC philly p99 −30%）；(ii) **過度集中真的被拉回**——cvar 的節點分佈跨 3 seed 從 baseline ~89% 降到 **80/65/67%（平均 ~71%）**，證明 balance shaping 在動；(iii) **`philly cvar s44 = +2.3%`——全專案第一次有 learned arm 在 sim 贏過 score**。**誠實限制**：(a) 仍**沒人「穩健」贏過 score**（最佳 mean −4%）；(b) **RDSAC-mean 反而退步**（−10/−24 pts——balance shaping 與風險中立 arm 互動不良，待查）；(c) 集中度仍 >50%。**結論**：P1+P2 確認「修訓練、別加花招」方向正確——把退化策略往「均衡、低變異、逼近 score」推進了一大步，值得拿去做真實 CUDA job 的實機檢驗（§5.1 第 4 項）。原始檔 `runs/p1p2_2x1_s{42,43,44}/`、彙總 `runs/p1p2_2x1_agg.txt`。

---

## 4. 實機執行結果（Live cluster）

本節是管線的階段 (3)：把 §3 的模擬產出 checkpoint 烘進真實叢集，在拓樸匹配的 2×1 跑 paired A/B。**結論是乾淨的四方負結果**：

- **🟥 2-node placement：三個 learned arm 全顯著輸 Slurm（§4.1）**：四方 submit-時 `-w` placement A/B——**SAC / RDSAC-mean / RDSAC-cvar 全顯著輸 Slurm 預設（−12~−32% JCT，每格 p<0.01）**，且**越把負載擠到 4070 輸越多**（learned 全擠 88–92% vs score 均衡 52%，cvar 92% 最慘）。這把 §3.1–3.4 的「sim 逼近打平、單 seed 雜訊」釘成 live 負結果。
- **fail-safe 設計在真實環境驗證有效**：`/decide` 失敗或低信心時自動回退 score，slurmctld 從不被擋——這是讓 shadow 部署能安全跑的前提。
- **真實微調語料已開始累積**：live A/B 期間 `live_daemon.py` 記錄的真實 (obs, act, rew) 即階段 ④ RLPD 的輸入。

換言之，§4 沒有「DRL 在 live 大勝」的故事——**2-node placement 直接輸 Slurm**。價值在於**用正確的實驗設計（共享 partition + submit-時 placement + drift-robust interleave）拿到誠實的結論**，並證明 sim-to-real 工程管線真的能跑，為 multi-seed 固實 / RLPD 微調鋪路。

### 4.1 2-node live placement A/B（submit-時 RL 選 node，四方）—— 真實多節點結果：所有 learned placement 都顯著輸 Slurm，且「越擠 4070 越差」

> **真實 2-node placement 的四方負結果。** 第二節點（RTX 3080）上線後，在拓樸匹配的 2×1 跑 RL placement A/B。四方（score / SAC / RDSAC-mean / RDSAC-cvar）結論直接：**三個 learned arm 的 placement 全顯著輸 Slurm 預設，且輸的幅度由「把負載擠到 4070 的程度」決定。**

**為什麼是 submit-時 -w 而非 post-submit controller**：原設計的 `rl-placement-controller`（held job → slurmrestd 寫 `required_nodes` → release）**在 Slurm 21.08 根本行不通**——slurmrestd v0.0.37 把 `required_nodes` 列為 disabled key（`"Operation not permitted"`），`scontrol` 也拒絕更新已提交 job 的 required nodes。實測一輪輪排除後確認:21.08 無法 post-submit 重釘節點。故改成**submit-時決定**：每個 learned arm 的 job 先呼叫 serve `/act` 拿到節點選擇，用 `sbatch -w <node>` 釘下（score arm 不加 -w → Slurm 自選，乾淨 baseline）。learned arm 跑 boost-off（shadow），所以 treatment 是**純 placement**、不混 priority boost。

協定：`run_heavytail_ab --placement --gpu-nodes rtx4070-0,rtx3080-0`、四方（score/SAC/RDSAC-mean/RDSAC-cvar，checkpoint 取 §3.1 的 σ=1.0 三方）、philly、partition `gpu`（橫跨兩台的共享 partition）、n=20/stream、σ∈{0,1}、**`--interleave`**（drift-robust，每方法 4 輪×輪轉位置）、warmup 丟棄、per-job CRN。每方法 **n=80 paired**。原始檔 `runs/htab_live_place4clean_20260618-170651/`。（註：首跑被一次 host driver 升級造成的 mismatch 汙染，本節用 driver 修復後的乾淨重跑。）

**ΔJCT% / Δp99% / ΔCVaR% vs score（全部 paired，負=比 score 慢）：**

| σ | arm | mean JCT | p99 | CVaR | ΔJCT% | Δp99% | ΔCVaR% | p |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| 0.0 | score | 8.4 | 35.0 | 18.9 | — | — | — | — |
| 0.0 | SAC | 10.6 | 40.2 | 25.2 | −26.3 | −14.9 | −33.3 | <.001 ✓ |
| 0.0 | RDSAC-mean | 10.7 | 41.2 | 25.1 | −28.1 | −17.7 | −32.8 | <.001 ✓ |
| 0.0 | **RDSAC-cvar** | 11.0 | 41.0 | 25.7 | **−31.8** | −17.1 | −36.0 | <.001 ✓ |
| 1.0 | score | 8.4 | 36.5 | 19.6 | — | — | — | — |
| 1.0 | SAC | 9.4 | 40.2 | 22.6 | −12.2 | −10.3 | −15.3 | .009 ✓ |
| 1.0 | RDSAC-mean | 9.9 | 40.2 | 23.1 | −17.4 | −10.3 | −18.4 | .001 ✓ |
| 1.0 | **RDSAC-cvar** | 10.4 | 40.0 | 25.0 | **−24.2** | −9.7 | −27.9 | <.001 ✓ |

**判定：三個 learned placement arm 全顯著輸 Slurm（每格 p<0.01）**，mean JCT −12~−32%、尾部（p99/CVaR）同樣全負。**σ 不改變結論**（σ=1 略好於 σ=0 但排名不動）。

**機制（這次直接量到、且乾淨）**：補上 `NodeList` 擷取後，**每個 arm 落在哪台都記下來**（pooled n=160/arm）：

| arm | 落在 4070 | 落在 3080 | ΔJCT%（σ=1）|
|---|--:|--:|--:|
| score（Slurm 自選）| **52%** | **48%** | 0（baseline）|
| SAC | 88% | 12% | −12.2 |
| RDSAC-mean | 88% | 12% | −17.4 |
| **RDSAC-cvar** | **92%** | **8%** | **−24.2** |

**核心發現：所有 learned model 都把負載嚴重擠到 4070（88–92%），而 Slurm backfill 是均衡的 52/48；而且「越擠 4070、輸越多」——cvar 擠最兇（92%）就輸最慘（−24%）、SAC 擠最少（88%）輸最少（−12%），單調對應。** 因為 job 是固定 `sleep N`（runtime 與落哪台無關），JCT 差異**全來自 wait（排隊）**：擠到 4070 → 該卡佇列壅塞 → wait 變長。這把 §4.1 前 `/act` 探針的「偏好 node 0」量化成鐵證，也是 §3.5 壓倒性 caveat（**單訓練 seed、no-op 傾向**的 checkpoint）在真實環境的兌現。

**意外的反轉**：sim §3.1 裡 cvar 是「最穩的 learned arm」；**到了真實 placement，cvar 反而最差**——因為它最積極地把 job 往 node 0 集中。risk-sensitivity 在「選哪台卡」這個決策上變成**過度集中**的壞習慣。

**multi-seed 確認（負結果對訓練 seed 穩健）**：上面是單一 checkpoint。為了排除「剛好抽到壞 seed」，用 3 個訓練 seed（42/43/44）的 checkpoint 各跑一次**同樣的四方 placement A/B**（σ=1.0、sleep job、philly、n=80/arm/seed）。下表給**絕對指標**（跨 3 seed mean±std，單位小時），與模擬表格同規格：

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

> **workload caveat（重要）**：本 A/B 的 job 是 `sleep N`、**不做 GPU compute**，所以 runtime 與 placement 無關、placement 只影響排隊 wait——這把「共置干擾、VRAM 限制、異質算力」三個真實 placement 槓桿**結構性抹掉**了（sim 反而有用 `interference=0.3` 建模，是個 sim↔live 不一致）。換成真實 CUDA job 會讓測試更完整、也更公平（見 §5.1 第 4 項）；對目前過度集中的 checkpoint 預期會更慘（多付干擾代價），但那才是讓**好的** placement 策略有機會展現價值的場域。

> 一句話：真實 2-node placement 是**乾淨、且對訓練 seed 穩健的四方負結果**——score 均衡放置（~50/50），三個 learned arm 全把負載擠到 4070（85–92%）、跨 3 個 seed 全顯著輸 Slurm（每個 seed×arm 都是 −7~−31% JCT），沒有任何 seed 翻盤。不是「2-node 解鎖 RL」，而是「**這套 train+placement 會結構性地長出過度集中的退化策略，比 Slurm 均衡放置差**」。要再談 RL placement，得先改掉這個過度集中（multi-seed 確認過、不是 seed 運氣問題）。

### 4.2 真實 CUDA job（獨佔 GPU）：placement 有真實算力後果時，負結果放大 2–4×

§4.1 的 caveat 是 job 用 `sleep`、placement 只影響排隊。這節把 workload 換成**真實 cuBLAS sgemm job**（`gpu_workload.cu`，§6），讓 placement 真的有算力後果。**因為 node-2 的 device-plugin MPS 壞了（gpu-operator `config-manager` CrashLoopBackOff，需對齊 node-2 OS 才能修，§5.1）**，這節用**獨佔 GPU 模式**（`--gres=mps:100`，一卡一 job、無共置）——犧牲「MPS 共置干擾」訊號（sim §3.3/§3.6 已涵蓋），但保留 sleep job 測不到的兩個真實槓桿：**4070 vs 3080 的 ~4× 異質算力**、與**獨佔佇列**（一卡同時只跑一 job）。3 train seed、σ=1.0、philly、n=14/round、`--interleave`：

| arm | JCT(s) | p99(s) | ΔJCT% | 落點 4070/3080 | wait 佔 JCT |
|---|---:|---:|---:|---:|---:|
| score | 13.1±0.1 | 26.4±1 | （baseline）| 20/80 | 65% |
| RDSAC-mean | 20.1±3.2 | 52.6±4 | **−52.9±25** | 42/58 | 69% |
| SAC | 22.2±2.7 | 58.4±7 | **−68.7±21** | 48/52 | 71% |
| RDSAC-cvar | 30.5±11.4 | 92.0±47 | **−132.3±86** | 52/48 | 75% |

**判定：負結果放大 2–4×、且 seed-robust。** 三個 learned arm 全輸 score——SAC −69%、mean −53%、**cvar −132%**（vs §4.1 sleep 的 −12~−32%）。**cvar 最慘、變異最大**（±86），與 sim（§3.3 σ=1.0 CVaR 扣分、§3.6 cvar 高變異）一致。

**機制（為何放大）：JCT 由排隊 wait 主導（65–75%），不是 run time。** 拆 JCT = wait + run：score wait 8.5s，learned wait 14–23s；run 只差 4.6 vs 6–8s。關鍵在**獨佔佇列**——learned arm 雖然看起來 ~50/50，但**實際丟到 4070 的 job 數是 score 的 ~2.5×**（learned n≈28–34 vs score n=13）。獨佔模式下一卡只能跑一 job，4070 被塞太多 → 排成長隊（**4070 上的 job JCT 30–49s vs 3080 上的 11–13s**）→ wait 爆掉。score 反而把大宗 job 卸到 3080、讓兩卡的佇列都短。**也就是：在「placement 有真實後果」的場域，RL 的 submit-時 `-w` 放置不看當下節點佔用 → 把 job 硬塞到忙碌的卡上排隊；score（backfill）只放到空閒的卡 → 排隊短。**

**範圍限制**：獨佔模式（無共置）→ **沒測到 MPS 干擾**這個槓桿（node-2 MPS 修好後可補；sim §3.3/§3.6 已涵蓋干擾），單 family（philly）。但**四方一致、放大顯著、seed-robust**——「真實算力後果讓 RL placement 的退化更嚴重」是穩健結論。原始檔 `runs/excl_live_s{42,43,44}/`、彙總 `runs/excl_live_agg.txt`。

> 一句話：把 sleep 換成真實 CUDA job（獨佔 GPU）後，§4.1 的負結果**放大 2–4 倍**（learned −53~−132% vs sleep 的 −12~−32%）——因為獨佔佇列下，RL 不看佔用的 `-w` 放置把 job 塞到忙碌的卡上排隊，wait 爆掉。**placement 越有真實後果，這套退化策略越輸 Slurm。**

---

## 5. 結論

| 問題 | 結論 |
|---|---|
| DRL path 能在 2-node 上跑？ | 可以。166-dim checkpoint 上線、共享 `gpu` partition、submit-時 placement A/B 全 job 乾淨完成（§4.1）。 |
| 先前 `alpha` 觸頂是真 bug？修好了？ | 是真 bug（return 尺度壓過 entropy ~300×）。已用 reward_scale 1000→20000 + 放寬 clamp 修好（§2.2）。 |
| RDSAC / SAC 在 2×1 贏過 score？ | **沒有。** sim 三方逼近打平、沒人贏過 score（§3.1）；live 2-node placement 三個 learned arm 全顯著輸 Slurm（§4.1）。**淨答案：在真實 2×1，沒有任何 learned model 在可檢驗的設定贏過 score。** |
| 分布式 / 風險機制有用嗎？(score vs SAC vs RDSAC) | **2×1 下沒有可定論的優勢（multi-seed 確認）。** σ-sweep 三方（3 train seed）沒人贏 score、arm 排名是訓練雜訊（§3.1）；拆解上 σ=1.0 的 CVaR 反而扣分、分布式 critic 是相對有用的一半（§3.3）。live placement 三方全輸 Slurm 且 **seed-robust**（§4.1）。 |
| risk-sensitive(cvar) 優於 risk-neutral(mean)？ | **方向上 sim 內 cvar 較穩、但在雜訊內。** 2×1（§3.1）cvar 是最穩的 learned arm（σ=0.5 最接近打平）；但**三方都仍輸 score**，差異落在單 seed 擺盪內。**而 live placement 反而 cvar 最差**（過度集中 4070 最兇，§4.1）——sim 與 live 對 cvar 的評價相反。 |
| 共置動作（PACK/ISOLATE）有用嗎？ | **沒有，即使有 2 GPU。** 2×1 colocation ON 仍輸 OFF ~20 pts（§3.4），「價值需 ≥2 GPU」被推翻；瓶頸是動作空間加倍（33→65）的 underfit。 |
| 2-node placement 結果？ | **負（四方一致、且 seed-robust）**。單 checkpoint：learned 全顯著輸 Slurm（−12~−32% JCT、p<0.01、drift-robust）。**multi-seed 確認**：3 個 train seed 各跑一次四方，每個 seed×arm 都輸 score（SAC −21.6±9.6、mean −13.4±4.4、cvar −21.2±6.1）。機制：learned 全把負載擠到 4070（85–89% vs score ~50%），不是 seed 運氣（§4.1）。 |
| 退化能修嗎？ | **能、方向對。** 對症加 P1 load-balance shaping + P2 reward normalization（§3.6）：cvar/SAC 改善 +19~+31 pts、variance 收斂（cvar philly `−4.1±4.6`）、過度集中從 ~89% 拉回 ~71%，且**首次有 learned arm 在 sim 贏過 score**（philly cvar +2.3%）。仍未穩健贏 score、mean 反而退步——是「修訓練」方向的有效一步，下一步拿去真實 CUDA job 實機檢驗。 |
| 最穩定上線策略 | 保留 stale snapshot / low confidence / service down 時的 heuristic/Slurm fallback。**在策略證明能穩健均衡放置前，RL placement 不應蓋過 Slurm 預設**。 |

**工程貢獻**：(1) 可上線的 DRL inference path（非僅 notebook/sim）；(2) DRL 對齊 Ma et al. RDSAC，有單元/行為測試；(3) 定位並修好 temperature auto-tune 的 reward-scale 根因；(4) sim + live trace collector 已能支援後續 RLPD；(5) 乾淨的四方受控對照（score/SAC/RDSAC-mean/cvar）+ 隨機性/共置消融；(6) **2-node 上線管線**：共享 `gpu` partition、submit-時 RL placement（`-w`，因 Slurm 21.08 無法 post-submit 重釘節點，§4.1）、外加修掉 4 個只在多節點現形的 chart bug（releasePriority 科學記號 CrashLoop、netpol 漏列、`-H` hold 被 score/rl_hook 覆蓋、controller 一次只放一個 job）。

**核心一句話**：在真實 2×1，DRL 排程 **沒有在任何可檢驗的設定贏過 score，而且這結論對訓練 seed 穩健**——sim σ-sweep（3 train seed）三方逼近打平、沒人贏 score、arm 排名是訓練雜訊（§3.1–3.4）；**live 2-node placement（3 train seed）四方全輸 Slurm**——sleep job −7~−31% JCT（§4.1），**換真實 CUDA job（獨佔 GPU）負結果放大到 −53~−132%**（§4.2，placement 有真實後果就更慘）。瓶頸是**這套 train+placement 會結構性地長出退化的放置策略**——下一步要改的是這個（架構/訓練穩定性，§3.6 的 P1+P2 已是有效一步），而非宣稱「用了 DRL/risk-sensitive」就算贏。

### 5.1 未來工作（Future Work）

> **改善路線（針對診斷出來的病灶）。** 實驗證明瓶頸**不是演算法花招**（SAC vs RDSAC vs critic 型別 ≈ 噪音；CVaR/共置甚至扣分），而是**訓練退化**：策略會結構性地過度集中到單卡（live 85–92% vs Slurm ~50%），且訓練高變異（跨 seed 擺盪 30–90 pts）。**過度集中與高變異是同一個病**（退化崩塌）。所以改善優先序是「修訓練、別加花招」：
>
> - ✓ **(P1) 反過度集中——load-balance reward shaping（已實作 + 有效，§3.6）**：potential-based 節點均衡項（`balance_coef`）把 cvar 的過度集中從 ~89% 拉回 ~71%。
> - ✓ **(P2) 訓練穩定——return normalization（已實作 + 有效，§3.6）**：running-std PopArt-lite 把 cvar variance 從 16→5、整體 +19~+31 pts 靠近 score。
> - ✓ **重跑 multi-seed 比對（已完成，§3.6）**：有效——過度集中收斂、variance 下降、首次有 learned arm 在 sim 贏 score（philly cvar +2.3%）。**剩**：mean 退步待查；**下一步換真實 CUDA job（下方第 4 項）做最終實機檢驗**。

依「擋住結論的程度」排序：

**讓現有結論站得住（方法學門檻）**

1. ✓ **sim + live multi-seed（已完成，§3.1/§3.3/§4.1）。** sim σ-sweep 三方用 3 個訓練 seed（42/43/44）跑 mean±std → 確認「沒人贏 score」穩健、arm 排名是訓練雜訊；live placement A/B 也用 3 個 seed 的 checkpoint 各跑一次四方 → 負結果 **seed-robust**（每個 seed×arm 都輸 score −7~−31%、過度集中跨 seed 穩定）。**剩下**：共置/σ=0 等 sim 格子尚未 multi-seed、live 只測 philly。原始檔 `runs/mseed_2x1_*`、`runs/htab_live_mseed_s*`、彙總 `runs/mseed_*_agg*`。
2. **σ 校準的外部效度。** §3.2 的 σ 是合成 trace 的最難預測上界；應在真實結構化 trace（`load_philly()`）上重量，並把 σ-sweep 落在實測區間。
3. **向量化 / 加速 sim（已實作）。** 純 Python 離散事件 ~10 steps/s 是多 seed 研究的算力牆（一個 σ 區塊 ~4.6h）。已加入 `sim/vec_env.py`（`SyncVectorSchedEnv` 參考實作 + `AsyncVectorSchedEnv` 多進程，autoreset 語義一致、async≡sync 經測試），並把 `sim_train(--num-envs N)` 接成 N 個 env 並行 rollout、共用同一 learner——多核近線性提升 rollout 吞吐，讓上面兩項（multi-seed、σ-sweep）在算力上可行。注意：vec path 的 score-warmup 退回 random-legal（score-warmup 需 in-process `env._state`），且每 iteration 仍 `utd_ratio` 次更新，所以 UTD 隨 N 稀釋——要維持樣本效率就同步調高 `--utd-ratio`。

**已完成的 2-node 上線管線（§4.1）**

✓ **2-node placement A/B + 工程基建（已完成，§4.1）。** 第二節點上線後，跑出真實四方 placement A/B（drift-robust `--interleave`），結論是負的（learned 全輸 Slurm）。沿途打通的基建：共享 `gpu` partition、submit-時 RL placement（`-w`，因 Slurm 21.08 無法 post-submit 重釘節點）、四個只在多節點現形的 chart bug 修復。工程規格見 `docs/live-ab-heavytail-spec.md`、`docs/intergration.md`。

**讓 live 測試更真實（最可能改變 placement 結論的方向）**

4. ✓ **真實 CUDA job（部分完成，§4.2）+ 待補：MPS 共置干擾。** 已把 `sleep N` 換成參數化 cuBLAS workload（`eval/scripts/gpu_workload.cu`），跑出**獨佔 GPU**的真實四方 A/B（§4.2）：負結果放大 2–4×（learned −53~−132%）。**已驗證的預期**——真實算力後果讓退化放置更慘。**剩兩塊**：
   - **MPS 共置干擾**還沒測。發現重大事實：**叢集的 MPS 從來沒真正多工過**（兩張卡 Exclusive_Process、但沒跑 MPS 控制 daemon；所有舊 live 結果都用 sleep job、無 CUDA context，所以一直沒現形）。4070 的 device-plugin MPS 可用、但 **node-2（3080）的 gpu-operator `config-manager` 一直 CrashLoopBackOff**（`findPidToSignal` panic）。根因是 **node-2 是 Ubuntu 22.04 / driver 580.159、acane 是 24.04 / 580.167**，且 `580.167` 沒為 22.04 打包 → 安全的 driver 對齊不可行。**修法：把 node-2 對齊到 Ubuntu 24.04**（有實體接觸時、獨立排程做），之後就能跑「共置 + 干擾」的 real-CUDA A/B（沿用 `--cuda-workload` 不加 `--exclusive-gpu`）。
   - **VRAM 限制**（4070 12GB vs 3080 10GB）也還沒推到綁定——獨佔模式 VRAM 沒成為約束。
   - 管線已就緒：`WorkloadSpec`/`--cuda-workload`/`--exclusive-gpu` 都實作 + 測過，binary 已編到 `/shared/bin/gpu_workload`（兩節點）。
5. **補強 baseline。** 目前只比自家 score + vanilla SAC；補 FCFS / SJF（已有 oracle runtime）/ packing 啟發式與近似上界，讓 ΔJCT% 有尺度感。

**演算法與韌性**

6. **return normalization（PopArt）** 取代手調 reward_scale，讓單一 α 控制器穩定（消掉本文必須釘 α=0.05 的 caveat）。
7. **機制 ablation**（PER / potential shaping / 雙頭 Z_R/Z_H）；以及把 critic 換成 Duan et al. 原版 DSAC（高斯回報 + 抑制 Q overestimation）對照 SAC——是公平 ablation，但 §3.3 已測過 critic 型別 ≈ 噪音，不太可能翻盤。
8. **per-model 各自調好的溫度**（本輪只釘單一 α=0.05）與 **held-out workload split**（如 train philly、test ali）證明泛化。

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
