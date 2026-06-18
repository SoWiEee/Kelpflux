# Kelpflux Scheduler Evaluation

本文件整理目前上線規格下的 scheduler evaluation。重點不是證明 DRL 一定優於啟發式，而是在同一套模擬環境與真實 Slurm/k3s/GPU 環境中，清楚比較 3 種排程方式的行為：**heuristic score**、**SAC**、**RDSAC**。

在真實叢集中從頭做強化學習訓練需要數十道數百萬個 transition，而真實叢集一個 placement 決策對應一個跑數分鐘～數小時的 job，湊滿樣本要等數月，因此採用**sim-to-real 兩段式**：

1. 在模擬環境中大量訓練（便宜/安全/可配對評估），產生 checkpoint
2. 上線部署到實機，記錄真實 $(obs, act, rew)$ 資料到 JSONL
3. 運用 RLPD 用真實資料把原始 checkpoint 微調成真實環境策略

模擬環境 (sim) 是唯一能做大量學習的地方；用來產出是「機制性洞察與一個可用的暖啟動」，接著用 RLPD 微調這個暖啟動。

**本文的聚焦點是：真實環境到底反映出了哪些 sim 推論。** 據此把 sim 結論分成三類（這也是讀本文的主軸）：

- **絕對績效數字（sim → real 轉移差）**：sim 的 JCT/勝幅不會原樣搬到真實系統，故只當定性/序數參考。
- **A 類——已被 live 反映的機制結論**：「**確定性 1×1 placement 退化 → 沒有學習法贏過強啟發式 → 應該打平**」（連帶「SAC 最差是 auto-α 假象」）。§4 的 live 1×1 三方乾淨打平**正面印證**了這條——這是本文**唯一已在真實環境兌現**的轉移。
- **B 類——sim 內成立、但 live 1×1 結構上反映不出來的推論**：「oracle runtime → CVaR≈mean → 風險機制空轉」「注入校準過的不確定性後 RDSAC > SAC、分布式 critic 才是主因、CVaR 是尾部加成」「共置要 ≥2 GPU」。這些原理上跟演算法本身綁定、可轉移，**但本系統的 live 1×1（makespan-bound + 生產 score runtime-blind）沒有可表現的決策面 → §4 量不到 → 本文不宣稱已轉移**，列為**待 2-node 驗證**（§5.1）。

> 閱讀順序對應管線：§3（模擬結果）先講 **sim 訓練產出了哪些洞察**；§4（實機結果）再講 **這些洞察與真實環境的關聯/發現**；§5.1 列出 RLPD 微調與多節點驗證的 future work。

---

## 0. 摘要與結果總覽

三種排程方式：**score**（MPS-aware 啟發式優先序，baseline=0%）、**SAC**（vanilla 離散 SAC，scalar twin-Q critic，無分布式/風險）、**RDSAC**（分布式 IQN critic + 風險扭曲，mean/cvar 變體）。三者在訓練 / eval / live serving 用一個 `use_iqn` flag 切換 SAC↔RDSAC、`risk_mode` 切換 mean↔cvar。

**核心發現：三方排名隨「環境有沒有 runtime 不確定性」整個翻轉。**

| 實驗條件 | score | SAC | RDSAC-cvar | 判定 | live 是否反映 |
|---|---|---|---|---|---|
| 確定性 1×1 sim, auto-α（§3.2, 30-seed）| 0% | −106/−312% | −25/−121% | 表面 score > cvar > SAC，**但誤導** | **✓ 反映**（§4 三方打平）|
| 確定性 1×1 sim, fixed-α（§3.3）| 0% | −17/−24% | −25/−121% | **SAC 翻身 ≈/贏 cvar**；淨增益 ≈0 | **✓ 反映**（§4 三方打平）|
| Live 1×1（§4）| 0% | ≈0% | ≈0% | **三方統計打平**；drift-robust 重跑（§4.4.2）確認、score 在 CVaR 甚至微幅最好 | —（即 live 本身）|
| 隨機 sim σ=1.0, fixed-α（§3.6 三方）| 0% | −107/−155% | −4.7/−14.2% | **RDSAC ≫ SAC**；分布式 critic 為主因 | **✗ 未反映**（1×1 makespan-bound + 生產 score runtime-blind，結構上測不到；**待 2-node 驗證**，§5.1）|
| 隨機 sim **2×1**, fixed-α（§3.9 三方）| 0% | −2.8/−5.0（σ0）·−26.4/−17.8（σ1）| −26.3/−12.8（σ0）·−10.2/−30.3（σ1）| **gap 收掉、逼近打平但未贏 score**；2×1 是 **cvar>mean**（§3.6 反向）| —（即 2-node sim 本身；live 見下）|
| **Live 2-node** placement, submit-時 -w（§4.5）| 0% | — | **−15.8（σ0）/ −16.6（σ1）** | **RL placement 顯著輸 Slurm ~16% JCT、尾部更差、σ 無影響（p<0.005）** | —（即 2-node live 本身）|

（兩個數字 = philly / ali；ΔJCT% vs score，負值=較慢。**§3.6 列是 1×1 sim 推論，§4 的 live 1×1 結構上無法檢驗；§3.9 列是其 2-node 真實拓樸的第一次檢驗——部分證實、單訓練 seed**）

**兩個時期、兩個故事：**

1. **沒有不確定性時（確定性 sim + live）→ 三方分不出高下。** 確定性 sim 表面上 SAC 墊底，但 fixed-α 對照（§3.3）證明那幾乎全是共用 auto-α 控制器 railing 的**假象**；釘死 α 後 SAC 追平甚至贏過 RDSAC-cvar。Live 1×1 三方都在 ±1% 雜訊內、模型幾乎全 abstain 回退 score。根因：oracle runtime 讓回報塌成點 → CVaR≈mean → 風險機制結構性閒置。

2. **加入真實 runtime 不確定性 → 清楚排名浮現（sim 推論，live 1×1 尚無法驗證）。** 注入的 σ 已**校準到生產預測器的真實 log-殘差**（§3.5：σ≈1.2–1.45，故 §3.4 的 σ=1.0 偏保守）。RDSAC−SAC 差距**隨 σ 單調拉開**（philly/ali 平均 −65→+47→+199 pts，§3.4），σ=1.0 下 RDSAC 甚至贏過 score、p99 尾部低 5–9×。三方拆解（§3.6）指出**贏的主因是「把回報建模成分布」（分布式 critic，SAC→mean +108~125 pts），CVaR 風險扭曲只是尾部專用小加成**。RDSAC 內部則 cvar > mean（§3.1）。**⚠ 這整組（§3.1、§3.4–3.7）是 sim 內的機制推論——§4 的 live 1×1 因 makespan-bound + 生產 score runtime-blind 而結構上反映不出來，須等 2-node 拓樸（§5.1）才能在真實環境檢驗，目前不可當成已轉移的結論。**

**一句話：** 1×1 確定性環境（含 live）三方打平、換演算法贏不了強啟發式（**這一條 sim 與 live 互相印證**）；補上真實不確定性後 **sim 內** RDSAC > SAC，且贏在分布式 critic 本身、CVaR 為尾部加成（**這一條 live 1×1 尚反映不出、待 2-node**）。

**三個 load-bearing caveats：**（1）**單訓練 seed**——同 cvar config 兩跑擺盪 50–90 pts，故 SAC→mean 大效應穩健、mean-vs-cvar 細排名需 multi-seed；（2）σ 是合成 trace 的最難預測上界（特徵與 runtime 無關），真實結構化資料 predictor 會更準 → 合理區間 [0.5, 1.45]；（3）**全部 1×1**，placement 是退化決策，真正檢驗要等拓樸匹配的多節點 checkpoint。

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

根據 [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) 實作離散版本（`DSACAgent(use_iqn=False)`）：**scalar twin-Q critic、MSE soft-Bellman、無 IQN 的 Z_R/Z_H 分布式分解、無 risk distortion**。排程 action 是離散且有 mask 的——每一步只能從 pending queue 中可行的 (job, node, GPU/MPS placement) 組合裡選一個。SAC 在本專案作為 RDSAC 的對照基準：用**完全相同的配方**（步數、traces、curriculum、PER、shaping、MLP trunk、1×1）訓練，唯一差別是 critic 型別與沒有 risk，用來回答「分布式/風險機制到底有沒有用」。

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

在模擬環境中「可大量訓練 + 可配對消融」，代價是模擬環境的絕對數字不轉移；機制性結論裡，**只有 A 類（1×1 退化 → 打平）已在實機反映**，B 類（σ → RDSAC、分布式 critic、共置）**原理上可轉移但實機 1×1 測量不到，待 2-node 驗證**。RLPD（**R**L with **P**rior **D**ata）的前提就是從一個既有 prior 出發再微調；模擬器的 checkpoint 不是被丟掉，而是 RLPD 站在它肩膀上。實機 trace 收集器（`live_daemon.py` → JSONL → `rlpd_finetune.py`）已就緒，等拓樸匹配的多節點 checkpoint 後啟動 (4)。

### 2.1 Simulator paired benchmark

相同亂數種子 (seed) 對 DRL 模型與啟發式 score 做配對比較來降低 trace 隨機性造成的誤差。`Δ = (score − model)/score`，負值代表 model 的 JCT 較高、較差。

| 項目 | 值 |
|---|---|
| training | 從頭訓練 150k 步（§3.4 起的隨機性實驗為 100k），curriculum n_jobs 10→30→50 |
| reward_scale | 20000（修復 alpha 觸頂） |
| cluster | 1 node × 1 GPU（§3.1–§3.7 執行當時 obs 192 / 17 actions；**GPU 字母表收斂為 {rtx4070, rtx3080} 後程式現為 obs 160**——160-dim 重跑見 §3.8）|
| jobs per trace | 50 |
| trace families | `philly`, `ali`（Alibaba PAI）|
| seeds（eval） | 確定性實驗 30 seed；隨機性實驗 5 seed |
| metric | mean JCT（主），p95 / p99 JCT（尾部），paired t-test 95% CI / p-value |

> **訓練 seed 注意**：除非另註，每個 (model, 條件) 只訓練**一個** seed。eval 的 30/5 seed 是評估隨機性，不是訓練隨機性。這是目前最大的方法學限制（§3.6 有同 config 兩跑擺盪 50–90 pts 的鐵證）。

> 早期 mean run 出現 `alpha` 自動調到 clamp 上限的警訊。排查後確認**不是 alpha 邏輯錯**，而是 **return 尺度問題**：舊 `reward = −JCT/1000`，50 job 累加 → return 量級 O(−150)，critic 的 `E[Z_R]` 跨動作落差約 560；actor 目標裡 entropy 正則 `α·log(n)`（上限~2）被 `Z_R`（~560）**壓過約 300×** → policy 早期塌成 one-hot（entropy ≈ 0.002）、探索停擺、alpha 一路 railing。修法：`reward_scale` 1000→**20000**（return 降到 O(−10)），log_alpha clamp 上限 1.0→3.0。修後 alpha 自由調節、entropy 由 0.002 恢復到 ≈0.13。

### 2.2 受控變數設計（每個實驗只動一個旋鈕）

| 旋鈕 | 隔離什麼 | 出現於 |
|---|---|---|
| `use_iqn`（SAC↔RDSAC）| critic 型別：scalar vs 分布式 IQN | §3.2, §3.6 |
| `risk_mode`（mean↔cvar）| 風險扭曲：risk-neutral vs CVaR | §3.1, §3.6 |
| `fixed_alpha`（auto↔釘死）| 溫度控制：拆穿 auto-α railing 假象 | §3.3, §3.4, §3.6 |
| `runtime_sigma` / `interference` | 注入 runtime 不確定性 / MPS 共置干擾（opt-in，預設關 → 與確定性 env 逐位元相同）| §3.4–§3.7 |
| `colocation_actions`（PACK/ISOLATE）| 共置是否成為一個動作 | §3.7 |

**隨機性注入模型**：`actual = predicted · exp(σZ − σ²/2)`（mean-preserving lognormal，E=1，只增變異不偏均值；obs 仍顯示 nominal runtime → 真實的結果不確定性）。idiosyncratic 噪音對每個 job common-random（以 `(seed, job_id)` 鍵）→ 同一 job 在每個 policy 下拿到相同乘子 → 配對比較。Harness：`eval/scripts/sweep_stochastic.py`。

### 2.3 σ 校準方法（讓注入的噪音不是憑空挑的）

由注入模型 `σ = std(log(actual/predicted))` —— 正是 runtime 預測器的 log-殘差標準差。`eval/scripts/measure_predictor_sigma.py` 用**生產級 LightGBM 預測器**（同 `services/runtime_predictor` 的 features、time-honest 80/20 split、超參）在真實 trace 上量 held-out log-殘差 std，當作要注入 sim 的 σ。結果見 §3.5。

### 2.4 實機 cluster A/B 設定

在實際 k3s + Slurm + GPU/MPS 環境提交 `sbatch` job 做 paired A/B。學習方法 `shadowMode=false`（RL boost 生效）、score 方法 `shadowMode=true`（boost 強制 0），用 serve 的 `/reload`+`/shadow` 切換而不重啟 pod。關鍵穩定化：

1.`gpu-rtx4070` pool 設 `min_replicas=1`（warm pool）—— 消除冷啟動 race，且「恰好 1 個 healthy GPU node」讓 snapshot `nodes=1` 與 1×1 checkpoint 拓樸匹配、`/decide` 正常 boost。
2. **去除共用單 GPU 的偏差**：丟棄 warmup round 只消得掉**一次性冷啟動**，但整個跑程還有**緩慢 drift**（GPU 越跑越快），block 設計會把 drift 和方法混淆，必須用 **round-robin 交錯方法順序**才能平均掉（證據與解法見 §4.4.1–4.4.2）。live 環境：namespace `slurm`、controller `slurm-controller-0`、GPU partition `gpu-rtx4070`、GRES `gpu:rtx4070:1,mps:rtx4070:100`。指標由 controller pod 的 `sacct` 收。**生產服務拓樸**：跑 `slurmctld` / `rl-scheduler`(serve) / `rl-snapshot-agent`，但**沒有部署 `runtime-predictor` 和 `weight-tuner`**，因此 score 的 `ε`(SJF) 與線上權重調整都是關的（§1.1 生產實況），這是讀 §4 數字時的環境前提。

### 2.5 為何用 p95 / p99 / CVaR 量尾部（不只看 mean JCT）

全文每張表都同時報 **mean JCT（主）** 與 **p95 / p99 JCT、CVaR(0.25)（尾部）**。納入尾部指標不是裝飾，而是這套評估能不能看見 RDSAC 效應的**前提**的四個理由：

1. **mean 會把排程病態洗掉。** straggler、queue starvation、head-of-line blocking 的典型表現是「多數 job 正常、少數被拖很慢」；這幾個慢 job 攤進整批裡幾乎不動 mean，但主導使用者體感。p95/p99 專抓「最差 5%/1% 有多慢」，正是 mean 結構上看不到的那段。
2. **尾部是 RDSAC/CVaR 的靶心——不量它就無法檢驗它。** RDSAC-cvar 的設計目標就是優化回報下尾（= JCT 上尾）；只量 mean 等於拿不會動的尺去量專門改尾部的方法，結構上必然測不出差異。鐵證在 §3.4：RDSAC 對 SAC 的 **mean 差距有限，p99 卻差 6–11×**，優勢全在尾部。
3. **重尾 workload 下 mean 本身不穩。** §4.4 的 workload 刻意重尾，重尾分布的 mean 由極端值主導、估計噪音大；分位數對重尾更穩健。**CVaR(0.25)=「最差 25% 的平均」**，比單點 p99（n=208 下只是最差 1–2 個 job、噪音主導，§4.4.2）穩定，故本文以 **CVaR 為主要尾部判別指標、p99 為輔助**。
4. **領域慣例。** GPU cluster scheduling / SLO 文獻裡，tail JCT、tail slowdown（p95/p99）本就是標準指標（p99 幾乎是 SLO 代名詞）；mean 必要但不充分。

這也是為何 §4.4.2 的最終判定不是只看 mean，而是「mean/p95 全平、但**穩定的尾部指標 CVaR** 上 score 微幅最好」——尾部指標才是真正能分開（或在 1×1 上確認分不開）三方的那把尺。

---

## 3. 模擬結果（Simulator）

本節是管線的階段 (1)(2)，在模擬環境訓練、在模擬環境裡做配對評估，因此這這裡的數字當定性/序數洞察，並非真實績效預測。模擬訓練產出的機制性洞察分成兩類：

**A. 已被 live 1×1 反映的洞察（sim ↔ live 互相印證）**

1. **確定性 1×1 placement 是退化決策 → 沒有學習法贏得過強啟發式 → 應該打平**（§3.2–3.3）。sim 全 rollout 下無任何 learned model 贏過 score，§4 的 live 1×1 三方乾淨打平**正面印證**。連帶 **auto-α 是壓垮 SAC 的 artifact**（§3.3，釘死 α 後 SAC 翻身）也與 live「SAC 同樣打平、非結構墊底」一致。→ 真實部署要 pin α，不要照搬為 cvar 尺度調的 auto-α 控制器。

**B. sim 內成立、但 live 1×1 結構上反映不出來的推論（⚠ sim 半已在 2×1 部分檢驗，§3.9；live 半待執行，§5.1）**

2. **確定性 oracle runtime → 回報塌成點 → CVaR≈mean → 風險機制閒置；補上校準過的真實不確定性後，贏的主因是分布式 critic、CVaR 是尾部加成**（§3.1、§3.4–3.6）。
3. **共置動作的價值需 ≥2 GPU**（§3.7）。

> **為什麼 B 類在 live 1×1 反映不出來**：(i) 1×1 只有一張 GPU，JCT 被 makespan 綁死，排序動不了它；(ii) 生產 score 的 `ε=0`（runtime-blind，§1.1），注入的 σ 對 score baseline 完全無作用。兩者都讓「σ 越大 RDSAC 越贏」這條 sim 機制在 1×1 沒有可表現的決策面。**所以 B 類在本文只當 sim 推論，不宣稱已轉移**。第二節點上線後 **§3.9 已在真實 2×1 拓樸跑完 σ-sweep，是 B 類的第一次檢驗——部分證實**（1×1 災難級差距收掉、逼近打平但仍未贏 score；2×1 下變 cvar>mean，與 §3.6 反向），但仍單訓練 seed，且 live 半（烘 checkpoint 上 2-node 跑 A/B）待執行（§5.1 第 5 項）。

以下 §3.1–3.7 是支撐這些洞察的受控實驗；§4 再看 A 類如何被真實環境反映、B 類為何尚不能。

### 3.1 確定性 sim：risk-neutral(mean) vs risk-sensitive(cvar)

> **⚠ B 類（sim 推論，live 1×1 未能反映、待 2-node 驗證）。** cvar > mean 是 sim 內結論；live 三方打平（§4.4.2）並未反映此排名。

兩 run 只差 `--risk-mode`（reward_scale、步數、seed 全相同），5-seed（後由 §3.2 30-seed 取代雜訊）：

**mean（risk-neutral）**

| Family | RDSAC JCT | Score JCT | Δ | 95% CI | p | 顯著 | p95 | p99 |
|---|---:|---:|---:|---:|---:|:--:|---:|---:|
| philly | 7.537 h | 2.621 h | −187.6% | [−261.3, −113.9]% | 0.0021 | **是** | 27.04 h | 37.84 h |
| ali | 2.342 h | 1.383 h | −69.4% | [−134.9, −3.8]% | 0.0424 | **是** | 7.59 h | 22.17 h |

**cvar（β=0.25，下尾風險敏感）**

| Family | RDSAC JCT | Score JCT | Δ | 95% CI | p | 顯著 | p95 | p99 |
|---|---:|---:|---:|---:|---:|:--:|---:|---:|
| philly | 2.783 h | 2.621 h | −6.2% | [−63.1, +50.7]% | 0.777 | 否 | 9.42 h | 22.64 h |
| ali | 2.301 h | 1.383 h | −66.4% | [−111.5, −21.3]% | 0.0150 | **是** | 7.14 h | 18.43 h |

**結論**：同配方、同探索強度下，**CVaR 明確優於 risk-neutral mean**——把 philly 拉到與 score 統計打平（−6.2%、p=0.78），mean 則 −187.6%、p=0.002 顯著差；尾部 p95 大致砍半（philly 9.42 vs 27.04h）。**誠實限制**：cvar 在 ali 仍顯著差（−66%）、且 cvar 仍未在任何 family 上**贏過** score——只是把差距縮到不顯著。這支持把 cvar 而非 mean 烘進 live image 的設計選擇，**但此優勢屬 B 類、尚未在真實環境兌現**。

### 3.2 確定性 sim：三方對照 score / SAC / RDSAC（30-seed）

加入 vanilla SAC 第三方，30-seed 重評（取代 §3.1 的 5-seed 雜訊）：

| family | score | SAC | RDSAC-cvar | RDSAC-mean |
|---|---:|---:|---:|---:|
| philly | 0% | −106.4% | **−24.6%** | −117.3% |
| ali | 0% | −311.6% | **−120.9%** | −128.4% |

原始排名（好→差）**score > RDSAC-cvar > RDSAC-mean ≳ SAC**，vanilla SAC 全面墊底、比 cvar 差 3–4×。**但這個排名有誤導性**（見 §3.3）。兩個 finding：(i) 1×1 sim 全 rollout 下**沒有任何 learned model 贏過 score**（**此條 A 類，§4 live 已反映**）；(ii) cvar 泛化遠勝 mean（−24.6% vs −121%，約 5×）。

### 3.3 fixed-α 受控對照：拆穿「SAC 最差」的 auto-α 假象

§3.2 的 SAC 慘敗到底是 scalar critic 真的差，還是共用 auto-α 控制器在 SAC 上 railing 的副作用？把 SAC 改成忠實公開實作（Christodoulou 2019 / `toshikwa/sac-discrete`），其餘配方與 §3.2 相同，**唯一掃描的變數是溫度**：`alpha` 釘死 {0.01, 0.05, 0.20} vs auto-α。

| family | auto-α SAC | **fixedA 0.01** | fixedA 0.05 | fixedA 0.20 | RDSAC-cvar |
|---|---:|---:|---:|---:|---:|
| philly | −106.4% | **−16.8%** | −20.8% | +1.5% | −24.6% |
| ali | −311.6% | **−23.8%** | −69.2% | −22.6% | −120.9% |

三個重點：(1) **「SAC 最差」基本上是 auto-α 假象**——*任何一個*釘死的 α 都把 SAC 從墊底拉到領先 ~90–290 pts，並**追平甚至贏過** RDSAC-cvar；(2) **失效點是控制器本身**：忠實版的 auto-α 仍從 0.1 railing 到 **2.58**（entropy 0.081），ΔJCT 停在 −75.9/−132.4%——忠實化救不了它，只有釘死 α 才行；(3) **分布式/風險的優勢因此大幅縮水**——給定穩定溫度，IQN+CVaR 相對「調好的 scalar SAC」在**確定性** 1×1 的淨增益接近零（**此「確定性 1×1 淨增益≈0」屬 A 類，與 §4 live 打平一致**）。誠實限制：fixed-α 各值的細排名是單訓練 seed 雜訊，方向（任何 fixed α ≫ auto-α）穩健。

> **「fixed-α 還是 auto-α 才是正規 SAC？」——澄清**：兩者是**同一份實作**的一個開關（`DSACAgent(fixed_alpha=...)`，`dsac.py:175`／溫度自動調在 `dsac.py:401-410`），不是兩種互斥的演算法。**auto-α（最大熵溫度自動調，Haarnoja 2018b）才是 SAC 的 canonical 預設**，本文與生產預設都走它；fixed-α（SAC-v1 把 α 當超參）同樣合法，但這裡的用途是**受控對照（control group）**：SAC 與 RDSAC **共用同一個 auto-α 控制器**，而該控制器的 `target_entropy_ratio`／clamp 是照 RDSAC-cvar 的 O(10) 回報尺度調的，套到 scalar SAC 上會把 α railing 到 2.58、entropy 壓到 0.08。所以本節釘死 α 不是「換一種實作」，而是把溫度從混淆變數裡隔離出來，證明 §3.2 的「SAC 墊底」來自 mis-tuned 的共用控制器、而非 scalar critic 本身。**對生產的含意**：live 要嘛 pin α、要嘛為 SAC 的回報尺度單獨重調控制器，別把為 cvar 調的 auto-α 直接套到 SAC。

> §3.1–§3.3 的共同結論：**在確定性 1×1 sim，分布式/風險機制的淨增益 ≈0**。但這有一個被忽略的前提——sim 的 runtime 是 oracle（`gym_env.step`：`end_ts = now + runtime`），轉移確定性 → 回報分布 `Z_R` 塌成 point mass → **CVaR ≈ mean → 風險機制結構性閒置**。§3.4 起把這個前提拿掉。

### 3.4 隨機性消融：注入 runtime 不確定性

> **⚠ B 類（sim 推論，live 1×1 未能反映、待 2-node 驗證）。** 本節「σ 越大 RDSAC 越贏」是 sim 內機制；§4.4.2 已說明 live 1×1（makespan-bound + 生產 score runtime-blind）結構上測不到此效應。

在 `KubefluxSchedEnv` 加 opt-in 的 mean-preserving lognormal runtime 噪音 σ（方法見 §2.3）。在匹配的 σ∈{0, 0.5, 1.0} 下各訓練 SAC 與 RDSAC-cvar（100k 步、curriculum、5 seed、philly/ali），三者透過同一隨機環境配對評估。

**結果 1 — auto-α 下，RDSAC−SAC 的 ΔJCT% 差距隨 σ 單調拉開**（正 = RDSAC 較優）：

| σ | philly | ali | 平均 |
|---|---:|---:|---:|
| 0.0 | −113 | −16 | **−65**（RDSAC 較差）|
| 0.5 | +52 | +42 | **+47**（反超）|
| 1.0 | +65 | +333 | **+199**（壓倒）|

σ=0 時 RDSAC 反比 SAC 差（沒有尾部可優化）；σ=0.5 起全面反超；σ=1.0 時 SAC 崩潰、RDSAC 相對穩健。

**結果 2 — fixed-α 受控對照（σ=1.0、α 釘死 0.05）**，排除 auto-α 干擾：

| family | SAC ΔJCT% | RDSAC ΔJCT% | SAC p99 | RDSAC p99 |
|---|---:|---:|---:|---:|
| philly | −57.0 | **+53.5** | 69.6 h | **6.2 h** |
| ali | −68.7 | **+73.8** | 44.3 h | **7.7 h** |

fixed-α 下 RDSAC 仍贏 SAC（+110~143 pts，排除 α 假象），σ=1.0 時 RDSAC 甚至**贏過 score**（philly +54%、ali +74%）並把 p99 壓低 6–11×。**消融結論**：§3.3「淨增益≈0」的根因是**兩個一起**——(a) oracle runtime 零不確定性、(b) auto-α 壓垮策略；兩個都修正後，風險機制在「有尾部風險可管理」時確實有價值（**sim 內；live 兌現待 2-node**）。原始檔 `runs/stoch_sweep_*/`、`runs/stoch_fixedA_*/`。

### 3.5 σ 校準到真實預測誤差：σ=1.0 其實偏保守

> **⚠ B 類支援節（為 §3.4 的 σ 取值背書；同屬 live 1×1 未能反映、待 2-node）。**

§3.4 的弱點：σ=0.5/1.0 若是憑空挑的，「注入噪音 → 抗噪法贏」近乎套套邏輯。用 §2.4 的方法量生產 LightGBM 預測器的 held-out log-殘差：

| workload | σ（殘差 std）| 95% CI | 形狀 |
|---|---:|---|---|
| philly | **1.45** | [1.31, 1.58] | near-Gaussian |
| ali | **1.24** | [1.11, 1.38] | near-Gaussian |

兩個結論：(1) 真實 σ ≈ **1.2–1.45**，故 §3.4 用的 **σ=1.0 落在真實下限以下、是保守值**——RDSAC 在 σ=1.0 就大勝，真實噪音下只會更強；(2) 殘差**近高斯**（excess kurtosis −0.1~+0.4），lognormal 噪音模型沒有低估尾部。**誠實告誡**：這些合成 trace 的 runtime 與特徵無關（corr(log_rt, gpu_count)=0.04，predictor 打不過 predict-the-mean），所以 1.2–1.45 是「最難預測」上界；真實結構化資料上好 predictor 會更低 → 合理 σ 區間 ≈ **[0.5, 1.45]**，§3.4 測的 {0.5, 1.0} 都落在其中。

### 3.6 拆解：贏的是「分布式 critic」還是「風險扭曲」？

> **⚠ B 類（sim 推論，live 1×1 未能反映、待 2-node 驗證）。** 「分布式 critic 為主因」是 sim 內的拆解結論；live 1×1 全 abstain，未能反映。

§3.4 比的是 SAC（scalar）vs RDSAC-cvar（分布式 + 風險），把兩個貢獻綁死。加入第三方 **RDSAC-mean**（分布式但**風險中立**）即可拆開（σ=1.0、fixed-α=0.05、5 seed、philly/ali）：

| family | SAC | RDSAC-mean | RDSAC-cvar | SAC→mean（**分布式**）| mean→cvar（**風險**）|
|---|---:|---:|---:|---:|---:|
| philly | −107.4 | +0.7 | −4.7 | **+108.1** | −5.4 |
| ali | −154.7 | −29.4 | −14.2 | **+125.2** | +15.2 |

**核心發現：絕大部分增益來自分布式 critic，不是風險扭曲。** SAC→RDSAC-mean（風險中立）就吃掉 **+108~+125 pts**，幾乎是整個 SAC↔RDSAC 差距——**在有噪音時，把回報「建模成分布」本身才是關鍵**（scalar critic 的單點 Q 在高回報變異下是較差的學習目標，quantile critic 穩健得多）。CVaR（mean→cvar 僅 −5/+15 pts）是**較小、看 workload 的尾部專用加成**：ali 小幅正向（+15 pts）、philly 略負（−5）——在 philly/ali 上 CVaR 的尾部加成不大且依 workload 而定。

**強 caveat（單訓練 seed，有鐵證）**：同一組 config（σ=1.0、fixed-α=0.05、cvar）在 §3.4 fixed-α 對照給 philly **+53.5**/ali **+73.8**，本輪卻是 −4.7/−14.2 —— 同設定兩跑，cvar 擺盪 ~60–90 pts。因此：SAC→mean 的 +108~125 pts 太大、seed 雜訊吃不掉，「**分布式 critic 是主因**」（在 sim 內）穩健；mean→cvar 較小、落在單 seed 雜訊內，**CVaR 淨增益需 multi-seed 才能定論**。原始檔 `runs/item1_calib_*/`。

### 3.7 共置動作消融：PACK/ISOLATE 在 1×1 反而拖累（負結果）

> **⚠ B 類（sim 負結果；其「價值需 ≥2 GPU」的結論本身就指向 2-node，live 1×1 無從檢驗）。**

把不確定性補回後，下一問題：**讓模型自己決定共置策略**能否在單卡下進一步幫到 RDSAC？給每個放置動作加一個 mode（`colocation_actions`，opt-in，預設關 → 動作空間與舊版逐位元相同）：`PACK`（接受 MPS 共享，付 §3.4 的 interference slowdown）vs `ISOLATE`（要求 GPU 空閒才放，不共享但要等卡空出來）。動作空間 17→33。在 interference=0.3、σ=0.5、fixed-α=0.05、RDSAC-only 下比較 colocation **ON vs OFF**：

| family | OFF（baseline）ΔJCT% | ON（+共置）ΔJCT% | OFF p99 | ON p99 |
|---|---:|---:|---:|---:|
| philly | **+35.8** | −6.0 | **12.2 h** | 23.0 h |
| ali | **+66.0** | −3.1 | **7.1 h** | 15.0 h |

共置動作在 philly/ali 都明顯更差（OFF 贏 ~42/~69 pts）、尾部 p99 也變糟。兩個推測成因：

1. 動作空間加倍、訓練預算不變 → underfit（多出的 ISOLATE 大多被 mask、稀疏）；這是「相同預算」非「能力天花板」比較
2. 單卡下 ISOLATE = 讓 GPU 閒置等佇列堆積，對 JCT 通常比擠進去吃干擾更糟，interference=0.3 還不夠重到讓「等獨佔」划算。**意涵**：共置作為決策的價值需 **≥2 GPU**（單卡沒有「放哪張卡」的真實選擇），正好接到第二節點（`docs/intergration.md` 的 RTX 3080）。程式保留為 opt-in、預設關。Caveats：每種方法單一訓練 seed、只測 (σ=0.5, interference=0.3) 一點。原始檔 `runs/b_coloc_*/`。

### 3.8 160-dim 字母表重跑 + 決策來源拆分：模型其實「幾乎不下放置指令」

> **A 類（DSAC 在確定性 1×1 輸給強啟發式，與 §4 live 打平方向一致）；這裡第一次量到崩塌的 *機制*，並用受控對照證明它對訓練配方穩健（recipe-robust）。**

GPU 字母表收斂成 `{rtx4070, rtx3080}`（obs 192→160）後，在 160-dim 下跑**兩個配方**（都 curriculum n_jobs 10→30→50、向量化 `--num-envs 8`、500k steps、5 seeds、philly+ali，各約 53 min）：

- **配方 A（預設）**：RDSAC-mean、**auto-α**、random-legal 暖機。
- **配方 B（受控）**：RDSAC-mean、**fixed-α 0.05**（依 §3.3 釘死 α）、**score-warmup**（暖機用 score 啟發式種子）——把 §3.3 建議的兩個修正一次補上。

同時 `eval_dsac_placement.py` 新增**決策來源帳**（report-only，不改行為）：把每個 greedy 決策歸類成 **DRL 放置指令 / no-op / 回退 score**（後者＝若套 live serve guardrail `value<−1.0 或 entropy>2.5` 會 abstain → score baseline 接手）。

| 配方 | philly ΔJCT% | ali ΔJCT% | philly p99 | ali p99 |
|---|---:|---:|---:|---:|
| A 預設（auto-α / random-warmup）| **−28.1**（p=.03 ✓）| **−124.3**（p=.13）| 19.8 h | 68.5 h |
| B 受控（fixed-α 0.05 / score-warmup）| **−30.9**（p=.13）| **−59.3**（p=.02 ✓）| 22.2 h | 20.1 h |

| 配方 | philly：DRL指令 / no-op / →score | ali：DRL指令 / no-op / →score | 訓練末 entropy |
|---|---|---|---:|
| A 預設 | 0.0% / **98.1%** / 1.9% | 0.0% / **98.9%** / 1.1% | ~0.13 |
| B 受控 | 0.0% / **99.6%** / 0.4% | 0.0% / **99.6%** / 0.4% | ~0.005 |

**主結論（recipe-robust 的 no-op 崩塌）**：兩個配方下，訓練後的策略都**幾乎不下任何真實放置指令**（DRL 指令 ≈ 0%）、而是 **~98–99.6% 選 no-op**——主動「等」而非放置。關鍵：**補上 score-warmup + fixed-α（B）並沒有救回放置行為**——no-op 比例反而更高（98→99.6%）、entropy 收得更低（0.13→0.005）。所以最初「no-op 是無 score-warmup 的暖機假象」的猜想**被對照推翻**：在確定性 1×1，DSAC 收斂到 no-op 是**對訓練配方穩健的退化**，不是某個 recipe bug。（800-step 煙霧測試一度顯示 score-warmup→DRL指令 100%，那只是暖機**當下**由 score 啟發式直接驅動；學習一接手，policy 就退回 no-op。B 的 fixed-α 確實把 ali 差距從 −124% 收到 −59%，但**仍輸、仍 99.6% no-op**。）

**為什麼 no-op 會贏？** 確定性 1×1 的 placement 本身退化（單卡單槽、放哪都一樣），reward 對「放置 vs 等」幾乎沒有可學的梯度，policy 就塌到「安全」的 no-op。模型既不放置、也幾乎不 abstain（→score 僅 0.4–1.9%，因為它**自信地** no-op），等於「放著不管」→ 輸給會主動 bin-pack 的 score，尾部 p99 也爆掉（job 堆積）。**這正是 A 類「確定性 1×1 沒有學習法贏得過強啟發式」的具體機制**，並與 §4.4.2 live 1×1（模型 abstain/no-op ~90–100%、三方打平）同構。

**意涵**：1×1 不是調 recipe 能翻盤的——要有真正的 placement 決策面（§3.7、第 5 項的 **≥2 GPU / 2-node**）才談得上 DRL 是否贏。原始檔：A `runs/eval_160dim_20260617-151916/`、B `runs/eval_160dim_fixedA_sw_20260617-170056/`。

### 3.9 2-node（2×1 異質）σ-sweep：B 類推論的第一次真實拓樸檢驗

> **B 類驗證（部分證實，非乾淨勝利）。** §3.1–3.7 的 B 類推論（σ→RDSAC、分布式 critic 為主因、共置需 ≥2 GPU）此前都標「待 2-node 驗證」。RTX 3080 第二節點上線後（`docs/intergration.md`），本節在**真實匹配的 2×1 拓樸**（obs_dim=166、n_actions=33）首次跑 σ-sweep。**結論是混合的：1×1 的災難級差距在 2×1 收掉了，但 §3.4「σ→RDSAC 贏過 score」與 §3.6「分布式 critic 為主因」都沒有乾淨轉移。**

協定與 §3.4/§3.6 對齊、只換拓樸：`sweep_stochastic.py --n-nodes 2 --gpus-per-node 1`，σ∈{0, 0.5, 1.0} 各訓 SAC / RDSAC-mean / RDSAC-cvar，fixed-α 0.05、curriculum、100k steps、5-seed 配對評估、philly+ali。**ΔJCT% vs score（負=較慢），粗體=該列最佳 learned arm：**

| σ | family | SAC | RDSAC-mean | RDSAC-cvar | cvar p99 / SAC p99 (h) |
|---|---|---:|---:|---:|---|
| 0.0 | philly | **−2.8** | −0.4 | −26.3 | 27.98 / 26.64 |
| 0.0 | ali | **−5.0** | −48.2 | −12.8 | 9.04 / 9.32 |
| 0.5 | philly | −7.4 | −28.9 | **−3.3** | 21.11 / 22.59 |
| 0.5 | ali | −5.6 | −27.6 | **−1.8** | 10.55 / 9.20 |
| 1.0 | philly | −26.4 | −17.9 | **−10.2** | **29.37 / 39.06** |
| 1.0 | ali | −17.8 | −26.6 | −30.3 | 21.52 / 21.13 |

**三個發現：**

1. **✅（方向性，A 類延伸）2×1 把 1×1 的災難級差距收掉了，但仍未贏過 score。** 1×1 fixed-α 下 learned arm 還在 −17~−24%（§3.3），auto-α 更是 −106~−312%（§3.2）；到 2×1，最好的格子已逼近打平——RDSAC-mean −0.4%（philly σ=0）、RDSAC-cvar −1.8%（ali σ=0.5）。這正面支持 §5.1 第 5 項「要有真正的 placement 決策面（≥2 GPU）DRL 才談得上競爭」。**誠實限制：沒有任何一格真的贏過 score（全 Δ 為負）——是「逼近打平」，不是「翻盤」。**

2. **◐（§3.4 部分證實、非單調）σ→cvar 的方向對，但壓倒性沒重現。** σ=0 時 RDSAC-cvar ≤ SAC（沒尾部可優化，符合「確定性→CVaR≈mean」預期）；σ>0 後 RDSAC-cvar 在 4 格中 3 格反超 SAC，且 **σ=1.0 philly 把 p99 從 39.1→29.4h 砍 25%**——尾部風險故事在真實拓樸上看得到。**但不是 §3.4 在 1×1 那種乾淨單調**：σ=1.0 ali 反而最差（−30.3），且 §3.4-result2 在 1×1「σ=1 RDSAC 贏過 score +53/+74%」**在 2×1 完全沒重現**（cvar 仍 −10/−30%）。推測：動作空間 17→33、訓練預算不變 → 2×1 underfit；加上單 seed 雜訊。

3. **✗（§3.6 不轉移、甚至反向）「分布式 critic 為主因」在 2×1 翻盤成「風險扭曲才關鍵」。** §3.6 在 1×1 結論是 SAC→RDSAC-mean 吃掉幾乎全部增益、cvar 只是小加成。2×1 **正好相反**：RDSAC-mean（純分布式、風險中立）在 σ>0 是**最差**的 learned arm（−28.9/−27.6 @ σ=0.5、−17.9/−26.6 @ σ=1.0），而 RDSAC-cvar（加風險）才是最好。**在真實 2-node 上，是 CVaR 風險扭曲而非裸分布式 critic 在扛**——與 §3.6 的 1×1 拆解相反。

**⚠ 壓倒性 caveat（與 §3.6 同級）**：每個 arm **單一訓練 seed**（eval 才 5 seeds）。§3.6 已有同 config 兩跑擺盪 60–90 pts 的鐵證，故本節**所有細排名都可能翻**——尤其 σ=0 RDSAC-cvar philly −26.3（比自己 σ=0.5 的 −3.3 還差，明顯離群）、σ=0 RDSAC-mean ali −48.2 都像單 seed 壞點。**方向性發現（gap 收掉、cvar>mean@2×1）比點估計穩；要把 §3.9 升級成定論需 multi-seed（§5.1 第 1 項）。** 三個 σ=1.0 checkpoint 已存，直接餵後續 2-node live A/B（§5.1 第 5 項）。原始檔 `runs/stoch_sweep_2x1_20260618-003742/`。

---

## 4. 實機執行結果（Live cluster）

本節是管線的階段 (3)：把 §3 的模擬產出 checkpoint 烘進真實叢集跑 paired A/B。重點是檢驗 §3 機制性洞察與真實環境的關聯，也就是哪些 sim 推論被真實環境反映、哪些反映不出來：

- **A 類被正面反映、「絕對數字不轉移」當場成立**：sim 表面排名 `score > RDSAC > SAC`，但 live 1×1 三方統計打平（Δ 全在 ±1% 雜訊內）、學習模型 abstain 88–100% fail-safe 回退 score。這**正面印證** §3 的 A 類洞察（**1×1 退化 placement → 沒有學習法贏過強啟發式 → 打平**），sim 的勝幅本就不該轉移到這裡。**至於 B 類（σ → RDSAC、分布式 critic、共置），live 1×1 結構上反映不出來**（makespan-bound + 生產 score runtime-blind，§4.4.2），故本文不列為已轉移、留待 2-node。
- **fail-safe 設計在真實環境驗證有效**：`/decide` 失敗或低信心時自動回退 score，slurmctld 從不被擋——這是讓階段 ③ 能安全 shadow 部署的前提，也是 §2.0 不敢在 live 直接訓練的另一面。
- **真實微調語料已開始累積**：live A/B 期間 `live_daemon.py` 記錄的真實 (obs, act, rew) 即階段 ④ RLPD 的輸入；live 環境本身已驗證可收 `sacct` 指標、可逐 (round, idx) 配對。

換言之，§4 沒有「DRL 在 live 大勝」的故事——它的價值是**證明模擬環境洞察的方向正確（1×1 該打平、就打平了）、且 sim-to-real 的工程管線（shadow + fail-safe + trace collector）真的能跑**，為多節點上線與 RLPD 微調鋪好路。

### 4.1 RDSAC 實機 A/B（cvar-v2 checkpoint）

把 §3 的 cvar-v2 RDSAC checkpoint 烘進 `slurm-rl-scheduler:m11` 部署到實機，與 score-only 做配對 A/B（14 個 MPS sleep job × 3 輪）。結果：**RDSAC 能在 production 正確上線並與 score 持平**（ΔJCT −0.2%，遠小於逐對雜訊）。原因與 §3.3 一致——1×1 強 baseline 下 RDSAC 對每個到達 job 幾乎均勻 boost（`selected=14/14`），佇列排序等同 score。（細表已併入下方擴大版 §4.2。）

### 4.2 擴大評估：128 job/方法 + 受控方法順序（暖機假象的源頭）

§4.1 的 42-job 單 block 太少。擴大到 128 job/方法、6×6 參數網格（`mps∈{20,25,34,50,67,75}`、runtime∈{8,14,20,28,36,45}s），operator 全程暫停把拓樸釘在 1 個 GPU node，並**交換方法順序**：

| 跑次 | 方法順序 | aggregate ΔJCT | warm-subset ΔJCT |
|---|---|---:|---:|
| v2_123417 | rdsac → score | **−26.9%**（RDSAC 較差）| −2.3%（r3–8）|
| v2swap | score → rdsac（+warmup）| **+8.6%**（RDSAC 較好）| +0.9%（r2–8）|

aggregate ΔJCT 隨方法順序翻號 → 先跑的那個方法吃到一次性 GPU/MPS 暖機懲罰（rdsac round-1 wait 106.8s vs score 14.4s），所以那 ±20%+ 是**暖機假象、非排程效果**。暖機後逐輪 Δ 在 0 附近抖動，兩次 warm 估計平均 ≈ **−0.7%**。**結論**：1×1 live RDSAC 與 score 真正打平，比 §4.1 多 3× 資料、控掉暖機混淆後依然成立。RDSAC 方法的 RL 在 88–100% 提交上 abstain（snapshot 時效性 + 單 GPU placement 太瑣碎主導）→ as-deployed RDSAC 大多回退成 score。原始檔 `runs/live_ab/SUMMARY_v2.md`。

### 4.3 三方 live：SAC 也打平

把 vanilla SAC 也部署 live（`variant:"SAC"`，serve `/healthz` 回報），同樣的擴大 + 受控方法順序協定。結果與 RDSAC 一致：**1×1 live 下 score ≈ RDSAC ≈ SAC，三方分不出**——每個 learned model 都 abstain ~90–100% 而 fail-safe 回 score。原始檔 `runs/live_ab/SUMMARY_sac.md`。

> §4 一致的故事：1×1 太小，placement 端到端無法表現出優勢、也無法表現出明顯劣勢。sim 全 rollout 會放大 checkpoint 不足（略輸 score，§3.2）；live 因 abstain 回退 + 單 GPU placement 瑣碎而把差異洗掉（持平）。真正的增益／檢驗要等拓樸匹配的多節點 checkpoint。

### 4.4 重尾 + 高競爭 live A/B —— 在 1×1 上拆開三方的條件（已執行）

§4.1–4.3 在 1×1 全部打平，但**打平的原因不只是「1×1 太小」，也是 workload 本身沒有給出可測的決策**。先前 A/B 用 14 個 `sleep N` job、`runtime∈{8,16,26,40}s`、範圍窄又**確定**——有 slack、沒有尾部風險。在這種 workload 下三方**結構上必然打平**，跟模型好壞無關。

要在**現有 1×1 硬體**上（不碰 sim、不等多節點）評估 score/SAC/RDSAC 的差異，正確的施力點是**改 live A/B 的 workload 與量測，而非改模型或拓樸**。1×1 上 live 只剩兩個真實槓桿：**job 排序**（priority boost）與**單 GPU 的 MPS 打包程度**。設計三根支柱讓這兩個決策產生可測的 JCT 差異：

| 支柱 | 做法 | 拆開誰 | 理由 |
|---|---|---|---|
| **(A) 高競爭 / 深佇列 + MPS 超賣** | 一次注入遠多於單 GPU 能舒服容納的 job（peak MPS 需求 ≫ 100），佇列真的堆積 | **score vs SAC/RDSAC** | 沒競爭就沒排序問題，SJF 立刻解完 → 必然打平 |
| **(B) 重尾 runtime + σ-noisy 估計** | 真實 sleep 時間抽自重尾分布；餵給 `/decide` 與排序的是**加噪的估計**（`reported = true·exp(σZ−σ²/2)`），σ 校準到 §3.5 量到的預測器 log-殘差（≈1.2–1.45）| **SAC vs RDSAC** | RDSAC 的優勢全在「不確定下的尾部風險」。若 runtime 確定（`sleep N` 已知）→ CVaR≈mean → RDSAC 與 SAC **結構上必然打平**，改什麼都沒用。這是 §3.4–3.6「σ 越大 RDSAC 越贏」的 **live 對應物** |
| **(E) 尾部量測** | 除 mean JCT，加 p95/p99 JCT、tail slowdown、JCT 的 CVaR、最差 straggler、完成率 | 看得見 RDSAC 的尾部優勢 | RDSAC-cvar 直接優化尾部；只看 mean 會把差異洗掉 |

**配對協定**：同一條 job stream（per-job common-random：相同 arrival / true / reported / mps）在四種方法各重放，丟棄 warmup round；**只用 philly/ali**（兩者皆 trace 衍生、天生重尾，正合支柱 B）。**資料集適用性（誠實）**：我們**不做字面重放**——`gpu_count≤1` 過濾 + 時間壓縮把 runtime 映到可測尺度但**保留尾部形狀**，所以檢驗的是「philly/ali 形狀的競爭+尾部下三方是否分得開」，是有界、已聲明的範圍。工程規格見 `docs/live-ab-heavytail-spec.md`。

#### 4.4.1 首輪（block 設計）暴露的 drift × 順序混淆

2026-06-15 首輪用 **block 設計**（一種方法跑完所有輪次才換下一種）。harness 端到端可用（過程中修掉 4 個只在實機現形的 bug：sacct 時區、`wait_drain` 把 slurmctld socket-timeout 的空查詢誤判為排空、sbatch 偶發 timeout 掉 job、serve 映像的 `dsac.py` 載不了 scalar-critic 的 SAC checkpoint）。**首輪表面上 RDSAC 砍 19% p99，但那是假象**——下面解釋為什麼，§4.4.2 用正確設計重跑直接推翻它。原始檔 `runs/htab_live_20260615-003638/`。

##### 為什麼會這樣：drift × 執行順序的混淆（confounding）

- **drift（漂移）**：GPU/叢集的速度不是整段時間都固定的；跑了 ~50 分鐘，同樣的工作後來會變快（MPS 狀態穩定、快取暖、背景負載變動）。我們**直接量到**這件事——同一個 score、排同一批工作，早跑 p99=153.5、晚跑 p99=124.5。
- **執行順序的問題**：1×1 只有一張 GPU，四種排程方法**不能同時跑、只能一個接一個**。本輪用的是 **block 設計**——把一種方法的所有輪次跑完才換下一種。於是某個方法被綁在「慢時段」、另一個被綁在「快時段」。
- **混淆**：晚跑的方法看起來比較好，但**分不清是它真的比較好、還是只是跑在比較快的時段**——drift 和方法**糾纏在一起**。本輪 RDSAC 剛好排在較後（較快）位置，就揹上了那 ~19% 的假優勢。單一 warmup-round 丟棄消的是**一次性冷啟動**，**消不掉整個跑程的緩慢趨勢**。

##### 解法：round-robin 交錯排程方法的執行順序（已執行，結果見 §4.4.2）

不要「一種方法跑到底再換下一種」，改成**每一輪讓四種方法各跑一次，而且每輪輪換誰先跑**：

```
第1輪: score  SAC    mean   cvar
第2輪: SAC    mean   cvar   score
第3輪: mean   cvar   score  SAC
第4輪: cvar   score  SAC    mean
```

這樣**每種方法都會分到一些早期（慢）輪次和一些晚期（快）輪次**，drift 對四者的影響被**平均掉、互相抵消**。跑夠多輪後若仍有差異，才是**真實**差異而非位置造成。實作為 runner 的 `--interleave` 模式（每輪 `/reload` 切方法 + 每輪輪轉順序）。

#### 4.4.2 drift-robust 結果（round-robin, philly, n=30, σ∈{0,1}, 每方法 4 輪×4 位置）

2026-06-15 用 `--interleave` 重跑（每方法跨 4 輪各輪過一次先後位置），原始檔 `runs/htab_live_rr_20260615-024419/`。**drift 確實被消掉**——關鍵驗證：同一個 score 方法跨 σ 區塊現在**穩定**了：

| | block 設計（首輪）| round-robin（本輪）|
|---|---|---|
| score mean JCT 跨 σ | 49.2 → 46.8（漂移 5%）| 43.5 → 43.7（**穩 0.5%**）|
| score p99 跨 σ | 153.5 → 124.5（漂移 19%）| 125.0 → 127.0（**穩 1.6%**）|

drift 消掉後的三方比較（pooled 兩 σ，每方法 n=208）：

| 排程方法 | mean | p95 | p99 | CVaR(0.25) |
|---|--:|--:|--:|--:|
| **score** | 43.6 | 124.0 | 126.0 | **79.0** |
| SAC | 44.0 | 124.0 | 152.0 | 81.2 |
| RDSAC-mean | 43.9 | 125.0 | 132.9 | 80.7 |
| RDSAC-cvar | 44.0 | 124.0 | 151.0 | 81.1 |

**判定：乾淨的 null——四種方法在 1×1 上打平，沒有任何學習方法贏過 score。**

1. **mean、p95 全部打平**（mean 43.6–44.0，差 <1%；p95 124–125）。
2. **在穩定的尾部指標 CVaR 上，score 反而（微幅）最好**（79.0 vs 學習方法 80.7–81.2，學習方法差 ~2–3%，但都 **non-significant**，paired p 0.34–0.91）。
3. **首輪那個「RDSAC −19% p99」確實是 drift 假象**——drift 消掉後方向甚至反轉（RDSAC 尾部若有差異是**略差**）。p99 本身在 n=208 下只是最差 1–2 個 job，噪音主導（同方法跨 σ 在 126–153 間跳），不可細讀。

**意涵（誠實）**：這**不是**「sim 的 σ-發現被推翻」，而是**1×1 makespan-bound 讓排序根本動不了 JCT**——score 與 RDSAC 的決策都無從表現（不論好壞）。sim §3.4–3.6 的機制要在 live 兌現，需要**有真實決策面的拓樸**（2-node，§5.1）。本輪的價值是：**用正確的實驗設計（round-robin 去 drift）拿到一個站得住腳的 1×1 三方打平**，而不是被位置效應誤導成「RDSAC 贏」。

> **環境細節 caveat（σ 對 score 無作用）**：支柱 (B) 假設排程器用加噪的 runtime 估計排序，但**生產 score 的 `ε=0`（§1.1，SJF/predictor 關）→ score 完全不看 runtime**，所以注入的 σ 對 score 這條 baseline **沒有任何作用**（它純按 mps 大小/vram/碎片排）。σ 只進到 RL 的 `/decide` obs。因此「σ=0 vs σ=1 對 score 無差異」是**預期內**的，不是反證；要在 live 真正檢驗 σ 機制，得先把生產 `ε` 開起來 + 部署 predictor，或限定結論在「score 不用 runtime」前提下。

> 一句話：drift-robust 重跑後，**1×1 三方乾淨打平、score 甚至在 CVaR 微幅最好**；首輪的 RDSAC 尾部優勢證實是 drift 假象。1×1 排序動不了 makespan-bound 的 JCT——決定性檢驗仍需 2-node。

### 4.5 2-node live placement A/B（submit-時 RL 選 node）—— 第一個真實多節點結果：RL placement 顯著輸給 Slurm

> **A 類的延伸檢驗（負結果）。** §4.4.2 之前所有 live 都在 1×1（排序動不了 makespan）；第二節點（RTX 3080）上線後，這是**第一次在真實多節點上測 RL placement**。結論直接：**σ=1.0 cvar checkpoint 的 placement 顯著比 Slurm 自己的放置差**。

**為什麼是 submit-時 -w 而非 post-submit controller**：原設計的 `rl-placement-controller`（held job → slurmrestd 寫 `required_nodes` → release）**在 Slurm 21.08 根本行不通**——slurmrestd v0.0.37 把 `required_nodes` 列為 disabled key（`"Operation not permitted"`），`scontrol` 也拒絕更新已提交 job 的 required nodes。實測一輪輪排除後確認:21.08 無法 post-submit 重釘節點。故改成**submit-時決定**：RL arm 每個 job 先呼叫 serve `/act` 拿到節點選擇，用 `sbatch -w <node>` 釘下（score arm 不加 -w → Slurm 自選，乾淨 baseline）。learned arm 跑 boost-off（shadow），所以 treatment 是**純 placement**、不混 priority boost。

協定：`run_heavytail_ab --placement --gpu-nodes rtx4070-0,rtx3080-0`、philly、partition `gpu`（橫跨兩台的共享 partition）、n=20/stream、σ∈{0,1}、**`--interleave`**（drift-robust，每方法 4 輪×輪轉位置）、warmup 丟棄、per-job CRN。每方法 **n=80 paired**。原始檔 `runs/htab_live_place_20260618-105123/`。

| σ | arm | mean JCT | p95 | p99 | CVaR(0.25) | tail-slowdown p99 |
|---|---|--:|--:|--:|--:|--:|
| 0.0 | score | 8.9 | 34.0 | 36.2 | 19.8 | 4.5 |
| 0.0 | **RDSAC-cvar** | 10.3 | 34.0 | 40.2 | 24.4 | **8.9** |
| 1.0 | score | 8.7 | 34.0 | 36.0 | 19.9 | 5.0 |
| 1.0 | **RDSAC-cvar** | 10.2 | 35.0 | 40.0 | 24.4 | **10.0** |

| σ | ΔJCT% vs score | Δp99% | ΔCVaR% | paired t-test p |
|---|--:|--:|--:|--:|
| 0.0 | **−15.8** | −11.0 | −23.5 | 0.0035 ✓ |
| 1.0 | **−16.6** | −11.1 | −22.7 | 0.0028 ✓ |

**判定：RL placement 顯著、穩定地輸給 Slurm 預設放置**——mean JCT −16%、p99 −11%、CVaR −23%、tail-slowdown 約 **2× 更差**（8.9 vs 4.5 / 10.0 vs 5.0），**兩個 σ 都 p<0.005 顯著**，且 drift-robust。**σ 完全不改變結果**（σ=0 ≈ σ=1），與 §4.4.2「生產 score `ε=0` runtime-blind」一致。

**誠實解讀**：這**推翻了「≥2 GPU 就能讓 RL 翻盤」的樂觀預期**（§5.1 第 5 項）——至少對這個 checkpoint。成因是**負載不均**：補上 `NodeList` 擷取後的確認跑（σ=1、n=24/方法，`runs/htab_live_nodes_20260618-111715/`）**直接量到** RL 把 job 偏放 4070、相對閒置 3080：

| arm | 落在 4070 | 落在 3080 |
|---|--:|--:|
| score（Slurm 自選）| 58% | 42% |
| **RDSAC-cvar** | **71%** | **29%** |

RL 比 Slurm **多 +13pp 倒向 4070**。因為本實驗的 job 是固定 `sleep N`（runtime 與落在哪台無關），JCT 的差異**全來自 wait（排隊）**——RL 把較多 job 擠到 4070 → 該卡佇列更壅塞 → wait 更長 → JCT −16%。這與 §4.5 前 live `/act` 探針觀察到的「偏好 node 0、node 0 滿了就 no-op」一致，是 §3.9 壓倒性 caveat（**單訓練 seed、no-op 傾向**的 checkpoint）在真實環境的兌現。**誠實校正**：偏斜是**真實但溫和**（+13pp），不是「完全閒置 3080」——RL 仍有 29% 放 3080；單 seed 退化策略造成的**輕度失衡**就足以在真實 placement 上由「sim 逼近打平」翻成「live 顯著淨負」。

**範圍限制**：單一 checkpoint（σ=1.0 cvar）、單 family（philly）、JCT n=80 / 節點分佈確認 n=24；要下「RL placement 一定輸」的普遍結論需 multi-seed checkpoint + SAC/mean 三方 + ali。但**方向（這個 production-候選 checkpoint 的 placement 顯著輸 Slurm、且確實偏放 4070）穩健且顯著**。

> 一句話：第一個真實 2-node placement 結果是**乾淨的負結果**——RL（cvar, submit-時 -w）顯著輸 Slurm 預設放置 ~16% JCT、尾部更差，且 σ 無影響。不是「2-node 解鎖 RL」，而是「**單 seed、no-op 傾向的 checkpoint 一旦真的去選 node，就把負載擠歪、比 Slurm 差**」。這把 §3.9 的 sim 內 caveat 變成 live 實證，也讓「先 multi-seed 固實再談 placement」成為下一步的硬前提。

---

## 5. 結論

| 問題 | 結論 |
|---|---|
| DRL path 能在 live 上跑？ | 可以。warm-pool 穩定化後 RDSAC cvar-v2 live A/B 全 job 乾淨完成、RL boost 確實生效（§4.1）。 |
| 先前 `alpha` 觸頂是真 bug？修好了？ | 是真 bug（return 尺度壓過 entropy ~300×）。已用 reward_scale 1000→20000 + 放寬 clamp 修好（§2.2）。 |
| RDSAC 在標準 sim benchmark 打贏 score？ | **確定性 1×1 下還沒有**（§3.2，30-seed philly/ali 都不優於 score；**此條 A 類，live 已反映**）。注入真實不確定性後 σ=1.0 fixed-α 可贏過 score（§3.4）——**但屬 B 類：sim 內成立，live 1×1 未能反映、待 2-node 驗證**。 |
| 分布式 / 風險機制有用嗎？(score vs SAC vs RDSAC) | **看環境有沒有不確定性**。確定性 1×1（含 live）淨增益≈0，且「SAC 最差」是 auto-α 假象（§3.3）——**此條 A 類，與 live 打平一致**。一旦注入**校準過的**真實不確定性（§3.5 σ≈1.2–1.45），RDSAC−SAC 差距隨 σ **單調拉開**（philly/ali 平均 −65→+199 pts，§3.4），三方拆解（§3.6）指出**增益主要來自「分布式 critic」**（SAC→mean +108~125 pts），CVaR 是尾部專用小加成——**這整串屬 B 類：sim 內成立，§4 live 1×1 結構上反映不出（makespan-bound + 生產 score runtime-blind），待 2-node 驗證**。CVaR 淨增益另需 multi-seed 才能定論（單 seed 擺盪大）。 |
| risk-sensitive(cvar) 優於 risk-neutral(mean)？ | **sim 內是**（30-seed 下 cvar 泛化遠勝 mean，−24.6% vs −121%，約 5×，§3.2），支持把 cvar 烘進 live image——**但屬 B 類，live 1×1 三方打平並未反映此排名**。 |
| live A/B 已能公平比較 DRL vs score？ | 可以。重尾 + 高競爭 + **round-robin 去 drift** 後（§4.4.2）拿到**乾淨的 1×1 三方打平**：mean/p95 全平，CVaR 上 score 甚至微幅最好，沒有學習方法贏過 score（全 non-significant）。首輪 block 設計的「RDSAC −19% p99」證實是執行順序 × cluster drift 的假象（score 自身 p99 漂移 153→125）。根因：1×1 JCT 被 makespan 綁死、排序動不了它——非 σ-發現被推翻，而是缺真實決策面。 |
| 最穩定上線策略 | DRL live scheduler 保持 enabled + GPU warm pool，並保留 stale snapshot / low confidence / service down 時的 heuristic/Slurm fallback。 |

**工程貢獻**：(1) 可上線的 DRL inference path（非僅 notebook/sim）；(2) DRL 對齊 Ma et al. RDSAC，有單元/行為測試；(3) 定位並修好 temperature auto-tune 的 reward-scale 根因；(4) sim + live trace collector 已能支援後續 RLPD；(5) 乾淨的三方受控對照（score/SAC/RDSAC，一個 flag 切換）+ 隨機性消融把「分布式/風險機制何時有用」的條件講清楚。

**核心一句話**：在 1×1 確定性環境（含 live）三方打平、換演算法贏不了強啟發式——**這條 sim 與 live 互相印證、是本文唯一已轉移的結論**；補上**校準過的**真實 runtime 不確定性後 **sim 內** RDSAC > SAC，且贏在「把回報建模成分布」這件事本身、CVaR 風險扭曲是尾部專用加成——**但這條因 live 1×1 makespan-bound + 生產 score runtime-blind 而尚未在真實環境反映，列為待 2-node 驗證的推論**。不是只宣稱「用了 DRL / risk-sensitive」就算贏。

### 5.1 未來工作（Future Work）

依「擋住結論的程度」排序：

**讓現有結論站得住（方法學門檻）**

1. **多訓練 seed（≥3–5）重跑 §3.4 / §3.6 關鍵點。** 目前每 cell 單一訓練 seed，§3.6 已有同 config 兩跑擺盪 50–90 pts 的鐵證；per-family 數字要 mean±std 才能下定論，尤其 mean-vs-cvar 的細排名。
2. **σ 校準的外部效度。** §3.5 的 σ 是合成 trace 的最難預測上界；應在真實結構化 trace（`load_philly()`）上重量，並把 σ-sweep 落在實測區間。
3. **向量化 / 加速 sim（已實作）。** 純 Python 離散事件 ~10 steps/s 是多 seed 研究的算力牆（一個 σ 區塊 ~4.6h）。已加入 `sim/vec_env.py`（`SyncVectorSchedEnv` 參考實作 + `AsyncVectorSchedEnv` 多進程，autoreset 語義一致、async≡sync 經測試），並把 `sim_train(--num-envs N)` 接成 N 個 env 並行 rollout、共用同一 learner——多核近線性提升 rollout 吞吐，讓上面兩項（multi-seed、σ-sweep）在算力上可行。注意：vec path 的 score-warmup 退回 random-legal（score-warmup 需 in-process `env._state`），且每 iteration 仍 `utd_ratio` 次更新，所以 UTD 隨 N 稀釋——要維持樣本效率就同步調高 `--utd-ratio`。

**讓 sim 結論能轉移到 live**

✓ **重尾 + 高競爭 live A/B（已完成，§4.4.1–4.4.2）。** 首輪 block 設計被 cluster drift 汙染（score 自身 p99 漂移 153→125）；改用 `--interleave`（round-robin 交錯排程方法順序、每方法跨 4 輪輪過 4 位置）後 drift 消掉（score 跨 σ 穩定到 0.5%），得到**乾淨的 1×1 三方打平**——沒有學習方法贏過 score。**剩下的不是再跑一次 1×1**（makespan-bound 天花板已確認），而是第 5 項的 **2-node** 才有真實 placement 決策面。工程規格見 `docs/live-ab-heavytail-spec.md`。

4. **修 train/serve 動作落差（path 已補上，待對齊驗證）。** sim 訓練的是 placement policy（job, node, gpu），而 submit-time `/decide` 只把 RL 選擇轉成 priority boost、Slurm 仍做真正 allocation——這是落差來源。**現在 `rl-placement-controller` 已預設常駐**（透過 slurmrestd 對 held job 寫 `required_nodes`＋release，§3.4），所以「live 真接 explicit placement」這條路已經接上、不再是 sim 獨有。但落差**在 1×1 是退化的**（單 node 只有一個合法目標，explicit placement ≡ priority），要到第 5 項的 **2-node** 才看得出差異；且需先用相符維度的 checkpoint 重訓，否則 `/act` abstain → controller no-op。對齊驗證併入 2-node 那輪做。
5. **拓樸匹配的多節點 checkpoint（RTX 3080 第二節點，`docs/intergration.md`）。** 單卡 placement 退化；2-node（2×1 異質）才讓「放哪張卡」「共置與否」（§3.7）成為真實決策，也才能在 live 分出高下。（`rtx3080` 已建模進 `GPU_TYPES` / `_gpu_type_to_vram`）。
   - ◐ **sim 半已執行（§3.9）。** 第二節點實體上線後，已在真實匹配的 2×1（obs_dim=166、n_actions=33）跑完 σ-sweep。結果**部分證實 B 類**：1×1 的災難級差距收掉、逼近打平（但仍未贏過 score），且 cvar>mean 在 2×1 成立；**但 §3.4「σ→贏過 score」與 §3.6「分布式 critic 為主因」沒有乾淨轉移**（後者甚至反向）。仍是單訓練 seed，需第 1 項 multi-seed 升級成定論。
   - ✗ **live 半已執行（§4.5，負結果）。** §3.9 的 σ=1.0 cvar checkpoint 烘進 166-dim image、部署到 2-node cluster 跑 submit-時 -w placement A/B（post-submit controller 路在 Slurm 21.08 行不通，見 §4.5）。結果：**RL placement 顯著輸 Slurm 預設 ~16% JCT、尾部更差、σ 無影響（p<0.005）**。**B 類在真實環境的第一次檢驗是負的**——單 seed、no-op 傾向的 checkpoint 去選 node 反而把負載擠歪。→ 硬前提變成**先 multi-seed 固實 checkpoint（第 1 項）再談 placement**。
6. **補強 baseline。** 目前只比自家 score + vanilla SAC；補 FCFS / SJF（已有 oracle runtime）/ packing 啟發式與近似上界，讓 ΔJCT% 有尺度感。

**演算法與韌性**

7. **return normalization（PopArt）** 取代手調 reward_scale，讓單一 α 控制器跨 SAC/RDSAC 都穩（消掉 §3.3 必須釘 α 的 caveat）。
8. **機制 ablation**（PER / potential shaping / 雙頭 Z_R/Z_H）與 **score-warmup on/off**（驗證 live abstain ~90% 是否來自 imitate score）。
9. **per-model 各自調好的溫度**（本輪只釘單一 α=0.05）與 **held-out workload split**（如 train philly、test ali）證明泛化。

---

## 6. 重現指令

**確定性 sim 受控對照（§3.1–3.2，mean 與 cvar 只差 `--risk-mode`）**

```bash
PYTHONPATH=. .venv-m11/bin/python eval/scripts/eval_dsac_placement.py \
  --total-steps 150000 --warmup-steps 2000 --n-jobs 50 \
  --n-nodes 1 --gpus-per-node 1 \
  --trace-families philly ali --train-trace philly ali \
  --seeds 42 43 44 45 46 --curriculum --reward-scale 20000 \
  --risk-mode cvar --risk-beta 0.25 --device cuda \
  --out-dir runs/rdsac_eval_cvar_v2     # mean: --risk-mode mean
```

**σ 校準（§3.5）**

```bash
PYTHONPATH=. .venv-m11/bin/python eval/scripts/measure_predictor_sigma.py \
  --trace sim/data/philly_subsample.json
```

**隨機性消融 + 三方拆解（§3.4, §3.6）**

```bash
PYTHONPATH=. .venv-m11/bin/python eval/scripts/sweep_stochastic.py \
  --sigmas 1.0 --total-steps 100000 --n-jobs 50 --curriculum \
  --seeds 42 43 44 45 46 --trace-families philly ali \
  --risk-modes mean cvar --fixed-alpha --init-alpha 0.05 --device cuda
# 共置消融（§3.7）：加 --colocation --interference 0.3 --no-sac
```

**2-node（2×1 異質）σ-sweep（§3.9，B 類的真實拓樸檢驗）**

```bash
# 與上面 §3.4/§3.6 同協定，只把拓樸換成 2×1（obs_dim=166, n_actions=33）
PYTHONPATH=. .venv-m11/bin/python eval/scripts/sweep_stochastic.py \
  --sigmas 0.0 0.5 1.0 --total-steps 100000 --warmup-steps 2000 --n-jobs 50 \
  --seeds 42 43 44 45 46 --trace-families philly ali --risk-modes mean cvar \
  --n-nodes 2 --gpus-per-node 1 --curriculum --fixed-alpha --init-alpha 0.05 \
  --device cuda --out-dir runs/stoch_sweep_2x1_$(date +%Y%m%d-%H%M%S)
# 9 個 arm（3 σ × {SAC, RDSAC-mean, RDSAC-cvar}）逐 σ 存 checkpoint + sweep.json；
# σ=1.0 那組 checkpoint 即後續 2-node live A/B 的輸入。
```

**重尾 + 高競爭 live A/B（§4.4，drift-robust round-robin）**

```bash
# 前置：建含當前 serve.py/dsac.py + σ-checkpoints 的映像、部署、port-forward 8002
#   docker build -f Dockerfile.htab -t slurm-rl-scheduler:htab2 .   # FROM m11 + 覆蓋 serve.py/dsac.py/distortion.py + COPY 3 個 ckpt 到 /models/htab/
#   docker save slurm-rl-scheduler:htab2 | sudo k3s ctr images import -
#   kubectl set image deploy/rl-scheduler serve=slurm-rl-scheduler:htab2 -n slurm
#   kubectl port-forward -n slurm svc/rl-scheduler 8002:8002 &
KUBECONFIG=~/.kube/config PYTHONPATH=. .venv-m11/bin/python -m eval.scripts.run_heavytail_ab \
  --serve-url http://localhost:8002 --login-pod pod/<slurm-login-pod> \
  --interleave --family philly --n-jobs 30 --target-max-s 30 \
  --sigmas 0.0 1.0 --rounds 4 --warmup 1 \
  --sac-ckpt /models/htab/sac.pt \
  --rdsac-mean-ckpt /models/htab/rdsac_mean.pt \
  --rdsac-cvar-ckpt /models/htab/rdsac_cvar.pt \
  --out-dir runs/htab_live_rr_$(date +%Y%m%d-%H%M%S)
# 跑完 kubectl set image ... serve=slurm-rl-scheduler:m11 還原 production。工程規格見 docs/live-ab-heavytail-spec.md
```

---

## 7. 資料集來源

§3 的 simulator benchmark **並非直接重放原始資料集**，而是用 `sim/loader.py` 的合成生成器，分布參數依下列公開 GPU cluster trace 的已發表統計校準（可離線、無網路重現）。Philly 另提供真實 `cluster_log_data.json` 的載入路徑（`load_philly()`）。

| Trace family | 生成器 | 模仿來源 | 連結 / 論文 |
|---|---|---|---|
| `philly` | `generate_philly_like`（亦支援 `load_philly` 真實重放）| Microsoft Philly GPU cluster trace | github.com/msr-fiddle/philly-traces — Jeon et al., *Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads*, USENIX ATC 2019 |
| `ali` | `generate_ali_like` | Alibaba PAI GPU cluster trace（MPS-fractional、短尾、多單卡）| github.com/alibaba/clusterdata（`cluster-trace-gpu-v2020`）— Weng et al., *MLaaS in the Wild*, USENIX NSDI 2022 |

> 注意：`philly` / `ali` 是「**統計近似**」而非逐筆原始資料；數值反映的是這些 trace 的工作負載**特性**（job 大小分布、到達節奏、runtime 尾部），不等同在原始 production log 上的表現。嚴格對照時建議用 `load_philly()` 載入 msr-fiddle/philly-traces 的真實 trace。
