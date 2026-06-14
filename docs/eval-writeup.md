# Kelpflux Scheduler Evaluation

本文件整理目前上線規格下的 scheduler evaluation。重點不是證明 DRL 一定優於啟發式，而是在同一套 simulator 與 live Slurm/k3s/GPU 環境中，清楚比較三種排程方式的行為：**heuristic score**、**SAC**、**RDSAC**。

**訓練與評估管線（先讀這段）：** 本系統的策略**不是、也無法直接在 live cluster 上從頭訓練**——RL 要數十萬～數百萬個 transition，而真實 cluster 一個 placement 決策對應一個跑數分鐘～數小時的 job，湊滿樣本要等數月；且訓練初期的隨機策略會直接破壞系統，真實環境又無法 `reset()` 重放同一條 trace 做配對比較。因此採 **sim-to-real 兩段式**：

```
① sim 大量訓練（便宜 / 安全 / 可 reset / 可配對評估）  →  checkpoint
② live shadow-mode 部署，記錄真實 (obs, act, rew) 到 JSONL
③ RLPD 用真實資料把 sim checkpoint 微調成真實環境策略
```

sim **不是 live 的替代品**，而是唯一能做大量學習的地方；它的產出是**「機制性洞察 + 一個可用的 warm start」**，RLPD 微調的對象正是這個 warm start。對應地，本文有**兩種可轉移性差很多的主張**要分清楚：

- **絕對績效數字（sim → real 轉移差）**：sim 的 JCT/勝幅不會原樣搬到真實系統（§4 的 live 1×1 三方打平已當場印證），故只當定性/序數參考。
- **機制性結論（sim → real 轉移好）**：「auto-α 是壓垮 SAC 的 artifact」「oracle runtime → CVaR≈mean → 風險機制空轉」「分布式 critic 才是主因、CVaR 是尾部加成」「共置要 ≥2 GPU 才有意義」——這些是**演算法本身**的性質，跟 sim 擬真度無關，會跟著 checkpoint 進真實系統，是 §3 真正值錢的產出。

**閱讀順序對應管線：** §3（模擬結果）先講 **sim 訓練產出了哪些洞察**；§4（實機結果）再講 **這些洞察與真實環境的關聯/發現**；§5.1 列出 RLPD 微調與多節點驗證的 future work。

文件結構：§0 摘要與結果總覽 → §1 評估對象 → §2 實驗與 benchmark 方法（含 §2.0 管線總覽）→ §3 模擬結果（sim 洞察）→ §4 實機執行結果（與真實環境的關聯）→ §5 結論 + future work → §6 重現指令 → §7 資料集來源。

---

## 0. 摘要與結果總覽

三種排程方式：**score**（MPS-aware 啟發式優先序，baseline=0%）、**SAC**（vanilla 離散 SAC，scalar twin-Q critic，無分布式/風險）、**RDSAC**（分布式 IQN critic + 風險扭曲，mean/cvar 變體）。三者在訓練 / eval / live serving 用一個 `use_iqn` flag 切換 SAC↔RDSAC、`risk_mode` 切換 mean↔cvar。

**核心發現：三方排名隨「環境有沒有 runtime 不確定性」整個翻轉。**

| 實驗條件 | score | SAC | RDSAC-cvar | 判定 |
|---|---|---|---|---|
| 確定性 1×1 sim, auto-α（§3.2, 30-seed）| 0% | −106/−158/−312% | −25/−31/−121% | 表面 score > cvar > mean ≳ SAC，**但誤導** |
| 確定性 1×1 sim, fixed-α（§3.3）| 0% | −17/+2/−24% | −25/−31/−121% | **SAC 翻身 ≈/贏 cvar**；淨增益 ≈0 |
| Live 1×1（§4）| 0% | ≈0% | ≈0%（−0.2~−0.7%）| **三方統計打平**；模型 abstain 88–100% 回退 score |
| 隨機 sim σ=1.0, fixed-α（§3.6 三臂）| 0% | −107/−135/−155% | −4.7/+9.2/−14.2% | **RDSAC ≫ SAC**；分布式 critic 為主因 |

（三個數字 = philly / burst / ali；ΔJCT% vs score，負值=較慢）

**兩個時期、兩個故事：**

1. **沒有不確定性時（確定性 sim + live）→ 三方分不出高下。** 確定性 sim 表面上 SAC 墊底，但 fixed-α 對照（§3.3）證明那幾乎全是共用 auto-α 控制器 railing 的**假象**；釘死 α 後 SAC 追平甚至贏過 RDSAC-cvar。Live 1×1 三方都在 ±1% 雜訊內、模型幾乎全 abstain 回退 score。根因：oracle runtime 讓回報塌成點 → CVaR≈mean → 風險機制結構性閒置。

2. **加入真實 runtime 不確定性 → 清楚排名浮現。** 注入的 σ 已**校準到生產預測器的真實 log-殘差**（§3.5：σ≈1.2–1.45，故 §3.4 的 σ=1.0 偏保守）。RDSAC−SAC 差距**隨 σ 單調拉開**（−73→+47→+196 pts，§3.4），σ=1.0 下 RDSAC 甚至贏過 score、p99 尾部低 5–9×。三臂拆解（§3.6）指出**贏的主因是「把回報建模成分布」（分布式 critic，SAC→mean +108~125 pts），CVaR 風險扭曲只是尾部專用小加成**（burst p99 46→20h）。RDSAC 內部則 cvar > mean（§3.1）。

**一句話：** 1×1 確定性環境（含 live）三方打平、換演算法贏不了強啟發式；補上真實不確定性後 RDSAC > SAC，且贏在分布式 critic 本身、CVaR 為尾部加成。

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

實作：simulator baseline `sim/scheduler/score.py`；live submit hook `chart/templates/configmap-job-submit.yaml`；Slurm priority path `chart/templates/{slurm-conf,login,workers}.yaml`。

### 1.2 SAC

vanilla 離散 Soft Actor-Critic（`DSACAgent(use_iqn=False)`）：**scalar twin-Q critic、MSE soft-Bellman、無 IQN 的 Z_R/Z_H 分布式分解、無 risk distortion**。排程 action 是離散且有 mask 的——每一步只能從 pending queue 中可行的 (job, node, GPU/MPS placement) 組合裡選一個。SAC 在本專案作為 RDSAC 的對照基準：用**完全相同的配方**（步數、traces、curriculum、PER、shaping、MLP trunk、1×1）訓練，唯一差別是 critic 型別與沒有 risk，用來回答「分布式/風險機制到底有沒有用」。

### 1.3 RDSAC

RDSAC 是 Ma et al. 2020/2025〈DSAC: Distributional Soft Actor-Critic for Risk-Sensitive Reinforcement Learning〉(arXiv:2004.14547) 的**離散動作忠實轉寫**。把連續控制的 reparameterised Gaussian actor 換成**顯式 categorical actor**，critic 與 risk 機制依論文 §4.1：

- **雙分布 critic**：把 soft return 拆成 reward 分布 `Z_R` 與 entropy 分布 `Z_H`，各以 IQN 的 quantile 表示、共用 trunk，quantile Huber 回歸 + twin double learning。
- **risk 進策略目標**：actor 目標對 reward 分布套用 distortion `ρ`，`risk_mode ∈ {mean, cvar, wang, cpw, msd}`。`mean` 是 risk-neutral（distributional 但不規避風險）；`cvar`（β=0.25）偏好下尾較不嚴重的 placement，對應排程的 straggler / cold worker / long-tail runtime 風險。

實作 `services/rl_scheduler/dsac.py`、`services/rl_scheduler/distortion.py`。

---

## 2. 實驗與 Benchmark 方法

### 2.0 訓練與評估管線（sim 訓練 → RLPD 微調至真實環境）

整體方法是 **sim-to-real 兩段式**，原因與各段產出如下表。理解這條管線是讀懂後面所有實驗的前提：**§3 的所有數字都是「在 sim 裡訓練 + 在 sim 裡評估」，目的是篩架構與抽洞察，不是預測真實績效**；真實績效由 §4 的 live A/B 量，最終策略由 RLPD 微調產出。

| 階段 | 環境 | 做什麼 | 為什麼不能在 live 直接做 | 產出 |
|---|---|---|---|---|
| **① sim 訓練** | `KubefluxSchedEnv`（gym） | 從頭訓練 100k–150k 步，curriculum，PER + shaping | live 一個 step = 一個跑數分鐘～小時的 job，湊滿樣本要數月 | checkpoint（warm start） |
| **② sim 評估** | 同 sim，配對 + 受控消融 | paired t-test、SAC/RDSAC/cvar/fixed-α 消融 | live 無法 `reset()` 重放同一 trace → 拿不到 counterfactual | **機制性洞察**（§3） |
| **③ live 部署** | k3s + Slurm + GPU/MPS | shadow-mode 跑 checkpoint，記錄真實 (obs, act, rew) | 訓練初期隨機策略會破壞系統 → 只敢 shadow + fail-safe 回退 score | 真實 A/B（§4）+ 微調語料 |
| **④ RLPD 微調** | 用 ③ 的真實 JSONL | 以 sim checkpoint 為 prior，混真實資料 fine-tune | 從零 RLPD = 退回 ① 的 sample-complexity 與 ② 的破壞性探索 | 真實環境策略（future work，§5.1） |

**兩段式的取捨**：sim 換來「可大量訓練 + 可配對消融」，代價是擬真度——所以 sim 的**絕對數字不轉移**（§4 live 1×1 三方打平已印證），只有**機制性結論轉移**。RLPD（**R**L with **P**rior **D**ata）的前提就是從一個既有 prior 出發再微調；sim checkpoint 不是被丟掉，而是 RLPD 站在它肩膀上——沒有它，連微調的起點都沒有。live trace collector（`live_daemon.py` → JSONL → `rlpd_finetune.py`）已就緒，等拓樸匹配的多節點 checkpoint 後啟動 ④。

### 2.1 Simulator paired benchmark

相同 seed 對 learned model 與 heuristic score 做 paired comparison，降低 trace 隨機性造成的誤差。`Δ = (score − model)/score`，**負值代表 model 的 JCT 較高、較差**。

| 項目 | 值 |
|---|---|
| training | 從頭訓練 150k 步（§3.4 起的隨機性實驗為 100k），curriculum n_jobs 10→30→50 |
| reward_scale | **20000**（修復 alpha 觸頂，見 2.2） |
| cluster | 1 node × 1 GPU（obs 192 / 17 actions） |
| jobs per trace | 50 |
| trace families | `philly`, `burst`, `ali` |
| seeds（eval） | 確定性實驗 30 seed；隨機性實驗 5 seed |
| metric | mean JCT（主），p95 / p99 JCT（尾部），paired t-test 95% CI / p-value |

> **訓練 seed 注意**：除非另註，每個 (model, 條件) 只訓練**一個** seed。eval 的 30/5 seed 是評估隨機性，不是訓練隨機性。這是目前最大的方法學限制（§3.6 有同 config 兩跑擺盪 50–90 pts 的鐵證）。

### 2.2 reward_scale 修復（alpha 觸頂根因）

早期 mean run 出現 `alpha` 自動調到 clamp 上限的警訊。排查後確認**不是 alpha 邏輯錯**，而是 **return 尺度問題**：舊 `reward = −JCT/1000`，50 job 累加 → return 量級 O(−150)，critic 的 `E[Z_R]` 跨動作落差約 560；actor 目標裡 entropy 正則 `α·log(n)`（上限~2）被 `Z_R`（~560）**壓過約 300×** → policy 早期塌成 one-hot（entropy ≈ 0.002）、探索停擺、alpha 一路 railing。修法：`reward_scale` 1000→**20000**（return 降到 O(−10)），log_alpha clamp 上限 1.0→3.0。修後 alpha 自由調節、entropy 由 0.002 恢復到 ≈0.13。

### 2.3 受控變數設計（每個實驗只動一個旋鈕）

| 旋鈕 | 隔離什麼 | 出現於 |
|---|---|---|
| `use_iqn`（SAC↔RDSAC）| critic 型別：scalar vs 分布式 IQN | §3.2, §3.6 |
| `risk_mode`（mean↔cvar）| 風險扭曲：risk-neutral vs CVaR | §3.1, §3.6 |
| `fixed_alpha`（auto↔釘死）| 溫度控制：拆穿 auto-α railing 假象 | §3.3, §3.4, §3.6 |
| `runtime_sigma` / `interference` | 注入 runtime 不確定性 / MPS 共置干擾（opt-in，預設關 → 與確定性 env 逐位元相同）| §3.4–§3.7 |
| `colocation_actions`（PACK/ISOLATE）| 共置是否成為一個動作 | §3.7 |

**隨機性注入模型**：`actual = predicted · exp(σZ − σ²/2)`（mean-preserving lognormal，E=1，只增變異不偏均值；obs 仍顯示 nominal runtime → 真實的結果不確定性）。idiosyncratic 噪音對每個 job common-random（以 `(seed, job_id)` 鍵）→ 同一 job 在每個 policy 下拿到相同乘子 → 配對比較。Harness：`eval/scripts/sweep_stochastic.py`。

### 2.4 σ 校準方法（讓注入的噪音不是憑空挑的）

由注入模型 `σ = std(log(actual/predicted))` —— 正是 runtime 預測器的 log-殘差標準差。`eval/scripts/measure_predictor_sigma.py` 用**生產級 LightGBM 預測器**（同 `services/runtime_predictor` 的 features、time-honest 80/20 split、超參）在真實 trace 上量 held-out log-殘差 std，當作要注入 sim 的 σ。結果見 §3.5。

### 2.5 Live cluster A/B 設定

在實際 k3s + Slurm + GPU/MPS 環境提交 `sbatch` job 做 paired A/B。learned 臂 `shadowMode=false`（RL boost 生效）、score 臂 `shadowMode=true`（boost 強制 0）。兩個關鍵穩定化：(1) `gpu-rtx4070` pool 設 `min_replicas=1`（warm pool）—— 消除冷啟動 race，且「恰好 1 個 healthy GPU node」讓 snapshot `nodes=1` 與 1×1 checkpoint 拓樸匹配、`/decide` 正常 boost；(2) **方法論教訓**：共用單 GPU 的 block A/B 必須丟棄每臂 ≥1 個 warmup round 並交換 arm 順序（否則 aggregate 會被一次性 GPU/MPS 暖機懲罰帶風向）。live 環境：namespace `slurm`、controller `slurm-controller-0`、GPU partition `gpu-rtx4070`、GRES `gpu:rtx4070:1,mps:rtx4070:100`。指標由 controller pod 的 `sacct` 收。

---

## 3. 模擬結果（Simulator）

本節是管線的**階段 ①②**：在 sim 裡訓練、在 sim 裡配對評估。**請把這裡的數字當定性/序數洞察，不是真實績效預測**（真實績效見 §4）。sim 訓練實際產出的、會跟著 checkpoint 進真實系統的**機制性洞察**有四條：

1. **auto-α 是壓垮 SAC 的 artifact，不是 scalar critic 本身爛**（§3.3）——釘死 α 後 SAC 翻身。→ 真實部署要 pin α，不要照搬為 cvar 尺度調的 auto-α 控制器。
2. **確定性 oracle runtime → 回報塌成點 → CVaR≈mean → 風險機制結構性閒置**（§3.1–3.3 與 live 打平互相印證）。→ RDSAC 只有在環境**有 runtime 不確定性**時才有意義。
3. **補上校準過的真實不確定性後，贏的主因是「把回報建模成分布」（分布式 critic），CVaR 只是尾部專用加成**（§3.4–3.6）。→ 真實系統若要砍 straggler 尾部才需 cvar，一般情形 distributional critic 已吃下大部分增益。
4. **共置動作在 1×1 反而拖累，價值需 ≥2 GPU**（§3.7）。→ 共置決策的真正檢驗要等多節點拓樸。

以下 §3.1–3.7 是支撐這四條的受控實驗；§4 再看它們與真實環境的關聯。

### 3.1 確定性 sim：risk-neutral(mean) vs risk-sensitive(cvar)

兩 run 只差 `--risk-mode`（reward_scale、步數、seed 全相同），5-seed（後由 §3.2 30-seed 取代雜訊）：

**mean（risk-neutral）**

| Family | RDSAC JCT | Score JCT | Δ | 95% CI | p | 顯著 | p95 | p99 |
|---|---:|---:|---:|---:|---:|:--:|---:|---:|
| philly | 7.537 h | 2.621 h | −187.6% | [−261.3, −113.9]% | 0.0021 | **是** | 27.04 h | 37.84 h |
| burst | 8.007 h | 3.541 h | −126.1% | [−223.8, −28.4]% | 0.0231 | **是** | 37.15 h | 55.66 h |
| ali | 2.342 h | 1.383 h | −69.4% | [−134.9, −3.8]% | 0.0424 | **是** | 7.59 h | 22.17 h |

**cvar（β=0.25，下尾風險敏感）**

| Family | RDSAC JCT | Score JCT | Δ | 95% CI | p | 顯著 | p95 | p99 |
|---|---:|---:|---:|---:|---:|:--:|---:|---:|
| philly | 2.783 h | 2.621 h | −6.2% | [−63.1, +50.7]% | 0.777 | 否 | 9.42 h | 22.64 h |
| burst | 4.320 h | 3.541 h | −22.0% | [−130.6, +86.7]% | 0.604 | 否 | 16.16 h | 65.66 h |
| ali | 2.301 h | 1.383 h | −66.4% | [−111.5, −21.3]% | 0.0150 | **是** | 7.14 h | 18.43 h |

**結論**：同配方、同探索強度下，**CVaR 明確優於 risk-neutral mean**——把 philly 拉到與 score 統計打平（−6.2%、p=0.78），mean 則 −187.6%、p=0.002 顯著差；尾部 p95 大致砍半（philly 9.42 vs 27.04h）。**誠實限制**：cvar 在 ali 仍顯著差（−66%）、burst 的 p99 反比 mean 差（65.66 vs 55.66h，單一壞 seed）、且 cvar 仍未在任何 family 上**贏過** score——只是把差距縮到不顯著。這支持把 cvar 而非 mean 烘進 live image 的設計選擇。

### 3.2 確定性 sim：三方對照 score / SAC / RDSAC（30-seed）

加入 vanilla SAC 第三臂，30-seed 重評（取代 §3.1 的 5-seed 雜訊）：

| family | score | SAC | RDSAC-cvar | RDSAC-mean |
|---|---:|---:|---:|---:|
| philly | 0% | −106.4% | **−24.6%** | −117.3% |
| burst | 0% | −157.5% | **−31.1%** | −159.2% |
| ali | 0% | −311.6% | **−120.9%** | −128.4% |

原始排名（好→差）**score > RDSAC-cvar > RDSAC-mean ≳ SAC**，vanilla SAC 全面墊底、比 cvar 差 3–4×。**但這個排名有誤導性**（見 §3.3）。兩個 finding：(i) 1×1 sim 全 rollout 下**沒有任何 learned model 贏過 score**；(ii) cvar 泛化遠勝 mean（−24.6% vs −117%，約 5×）。

### 3.3 fixed-α 受控對照：拆穿「SAC 最差」的 auto-α 假象

§3.2 的 SAC 慘敗到底是 scalar critic 真的差，還是共用 auto-α 控制器在 SAC 上 railing 的副作用？把 SAC 改成忠實公開實作（Christodoulou 2019 / `toshikwa/sac-discrete`），其餘配方與 §3.2 相同，**唯一掃描的變數是溫度**：`alpha` 釘死 {0.01, 0.05, 0.20} vs auto-α。

| family | auto-α SAC | **fixedA 0.01** | fixedA 0.05 | fixedA 0.20 | RDSAC-cvar |
|---|---:|---:|---:|---:|---:|
| philly | −106.4% | **−16.8%** | −20.8% | +1.5% | −24.6% |
| burst | −157.5% | **+2.2%** | −36.8% | −18.3% | −31.1% |
| ali | −311.6% | **−23.8%** | −69.2% | −22.6% | −120.9% |

三個重點：(1) **「SAC 最差」基本上是 auto-α 假象**——*任何一個*釘死的 α 都把 SAC 從墊底拉到領先 80–290 pts，並**追平甚至贏過** RDSAC-cvar；(2) **失效點是控制器本身**：忠實版的 auto-α 仍從 0.1 railing 到 **2.58**（entropy 0.081），ΔJCT 停在 −75.9/−88.2/−132.4%——忠實化救不了它，只有釘死 α 才行；(3) **分布式/風險的優勢因此大幅縮水**——給定穩定溫度，IQN+CVaR 相對「調好的 scalar SAC」在**確定性** 1×1 的淨增益接近零。誠實限制：fixed-α 各值的細排名是單訓練 seed 雜訊，方向（任何 fixed α ≫ auto-α）穩健。

> §3.1–§3.3 的共同結論：**在確定性 1×1 sim，分布式/風險機制的淨增益 ≈0**。但這有一個被忽略的前提——sim 的 runtime 是 oracle（`gym_env.step`：`end_ts = now + runtime`），轉移確定性 → 回報分布 `Z_R` 塌成 point mass → **CVaR ≈ mean → 風險機制結構性閒置**。§3.4 起把這個前提拿掉。

### 3.4 隨機性消融：注入 runtime 不確定性

在 `KubefluxSchedEnv` 加 opt-in 的 mean-preserving lognormal runtime 噪音 σ（方法見 §2.3）。在匹配的 σ∈{0, 0.5, 1.0} 下各訓練 SAC 與 RDSAC-cvar（100k 步、curriculum、5 seed、3 family），三者透過同一隨機環境配對評估。

**結果 1 — auto-α 下，RDSAC−SAC 的 ΔJCT% 差距隨 σ 單調拉開**（正 = RDSAC 較優）：

| σ | philly | burst | ali | 平均 |
|---|---:|---:|---:|---:|
| 0.0 | −113 | −89 | −16 | **−73**（RDSAC 較差）|
| 0.5 | +52 | +47 | +42 | **+47**（反超）|
| 1.0 | +65 | +189 | +333 | **+196**（壓倒）|

σ=0 時 RDSAC 反比 SAC 差（沒有尾部可優化）；σ=0.5 起全面反超；σ=1.0 時 SAC 崩潰、RDSAC 相對穩健。

**結果 2 — fixed-α 受控對照（σ=1.0、α 釘死 0.05）**，排除 auto-α 干擾：

| family | SAC ΔJCT% | RDSAC ΔJCT% | SAC p99 | RDSAC p99 |
|---|---:|---:|---:|---:|
| philly | −57.0 | **+53.5** | 69.6 h | **6.2 h** |
| burst | −103.6 | **−13.4** | 53.6 h | **16.9 h** |
| ali | −68.7 | **+73.8** | 44.3 h | **7.7 h** |

fixed-α 下 RDSAC 仍贏 SAC（+90~143 pts，排除 α 假象），σ=1.0 時 RDSAC 甚至**贏過 score**（philly +54%、ali +74%）並把 p99 壓低 5–9×。**消融結論**：§3.3「淨增益≈0」的根因是**兩個一起**——(a) oracle runtime 零不確定性、(b) auto-α 壓垮策略；兩個都修正後，風險機制在「有尾部風險可管理」時確實有價值。原始檔 `runs/stoch_sweep_*/`、`runs/stoch_fixedA_*/`。

### 3.5 σ 校準到真實預測誤差：σ=1.0 其實偏保守

§3.4 的弱點：σ=0.5/1.0 若是憑空挑的，「注入噪音 → 抗噪法贏」近乎套套邏輯。用 §2.4 的方法量生產 LightGBM 預測器的 held-out log-殘差：

| workload | σ（殘差 std）| 95% CI | 形狀 |
|---|---:|---|---|
| philly | **1.45** | [1.31, 1.58] | near-Gaussian |
| burst | **1.42** | [1.27, 1.56] | near-Gaussian |
| ali | **1.24** | [1.11, 1.38] | near-Gaussian |

兩個結論：(1) 真實 σ ≈ **1.2–1.45**，故 §3.4 用的 **σ=1.0 落在真實下限以下、是保守值**——RDSAC 在 σ=1.0 就大勝，真實噪音下只會更強；(2) 殘差**近高斯**（excess kurtosis −0.1~+0.4），lognormal 噪音模型沒有低估尾部。**誠實 caveat**：這些合成 trace 的 runtime 與特徵無關（corr(log_rt, gpu_count)=0.04，predictor 打不過 predict-the-mean），所以 1.2–1.45 是「最難預測」上界；真實結構化資料上好 predictor 會更低 → 合理 σ 區間 ≈ **[0.5, 1.45]**，§3.4 測的 {0.5, 1.0} 都落在其中。

### 3.6 拆解：贏的是「分布式 critic」還是「風險扭曲」？

§3.4 比的是 SAC（scalar）vs RDSAC-cvar（分布式 + 風險），把兩個貢獻綁死。加入第三臂 **RDSAC-mean**（分布式但**風險中立**）即可拆開（σ=1.0、fixed-α=0.05、5 seed、3 family）：

| family | SAC | RDSAC-mean | RDSAC-cvar | SAC→mean（**分布式**）| mean→cvar（**風險**）|
|---|---:|---:|---:|---:|---:|
| philly | −107.4 | +0.7 | −4.7 | **+108.1** | −5.4 |
| burst | −135.2 | −20.0 | +9.2 | **+115.2** | +29.2 |
| ali | −154.7 | −29.4 | −14.2 | **+125.2** | +15.2 |

**核心發現：絕大部分增益來自分布式 critic，不是風險扭曲。** SAC→RDSAC-mean（風險中立）就吃掉 **+108~+125 pts**，幾乎是整個 SAC↔RDSAC 差距——**在有噪音時，把回報「建模成分布」本身才是關鍵**（scalar critic 的單點 Q 在高回報變異下是較差的學習目標，quantile critic 穩健得多）。CVaR（mean→cvar 僅 −5/+29/+15 pts）是**較小、看 workload 的尾部專用加成**：burst p99 從 46.3→**20.3h**（砍半多）、mean +29 pts；ali 小幅正向；philly 略負。

**強 caveat（單訓練 seed，有鐵證）**：同一組 config（σ=1.0、fixed-α=0.05、cvar）在 §3.4 fixed-α 對照給 philly **+53.5**/burst −13.4/ali **+73.8**，本輪卻是 −4.7/+9.2/−14.2 —— 同設定兩跑，cvar 擺盪 ~50–90 pts。因此：SAC→mean 的 +110~125 pts 太大、seed 雜訊吃不掉，「**分布式 critic 是主因**」穩健；mean→cvar 較小、落在單 seed 雜訊內，**CVaR 淨增益需 multi-seed 才能定論**（唯一例外是 burst 的 p99 尾部改善夠大夠一致）。原始檔 `runs/item1_calib_*/`。

### 3.7 共置動作消融：PACK/ISOLATE 在 1×1 反而拖累（負結果）

把不確定性補回後，下一問題：**讓模型自己決定共置策略**能否在單卡下進一步幫到 RDSAC？給每個放置動作加一個 mode（`colocation_actions`，opt-in，預設關 → 動作空間與舊版逐位元相同）：`PACK`（接受 MPS 共享，付 §3.4 的 interference slowdown）vs `ISOLATE`（要求 GPU 空閒才放，不共享但要等卡空出來）。動作空間 17→33。在 interference=0.3、σ=0.5、fixed-α=0.05、RDSAC-only 下比較 colocation **ON vs OFF**：

| family | OFF（baseline）ΔJCT% | ON（+共置）ΔJCT% | OFF p99 | ON p99 |
|---|---:|---:|---:|---:|
| philly | **+35.8** | −6.0 | **12.2 h** | 23.0 h |
| burst | −41.9 | **−9.2** | 18.5 h | 20.4 h |
| ali | **+66.0** | −3.1 | **7.1 h** | 15.0 h |

**負結果，方向相反**：共置動作在 philly/ali 明顯更差（OFF 贏 ~42/~69 pts），只有 burst 上 ON 較好。兩個成因：(1) 動作空間加倍、訓練預算不變 → underfit（多出的 ISOLATE 大多被 mask、稀疏）；這是「相同預算」非「能力天花板」比較；(2) 單卡下 ISOLATE = 讓 GPU 閒置等佇列堆積，對 JCT 通常比擠進去吃干擾更糟，interference=0.3 還不夠重到讓「等獨佔」划算。**意涵**：共置作為決策的價值需 **≥2 GPU**（單卡沒有「放哪張卡」的真實選擇），正好接到第二節點（`docs/intergration.md` 的 RTX 3080）。程式保留為 opt-in、預設關。Caveats：每臂單一訓練 seed、只測 (σ=0.5, interference=0.3) 一點。原始檔 `runs/b_coloc_*/`。

---

## 4. 實機執行結果（Live cluster）

本節是管線的**階段 ③**：把 §3 的 sim checkpoint 烘進真實 k3s + Slurm + GPU/MPS 跑 paired A/B。重點是檢驗 §3 機制性洞察與真實環境的**關聯**，而非期待 sim 勝幅重現。三條關聯/發現：

- **「絕對數字不轉移」當場成立**：sim 表面排名 `score > RDSAC > SAC`，但 live 1×1 三方統計打平（Δ 全在 ±1% 雜訊內）、學習模型 abstain 88–100% fail-safe 回退 score。這**正面印證** §3 洞察 ②——1×1 是退化 placement，幾乎沒有決策空間，sim 的勝幅本就不該轉移到這裡。
- **fail-safe 設計在真實環境驗證有效**：`/decide` 失敗或低信心時自動回退 score，slurmctld 從不被擋——這是讓階段 ③ 能安全 shadow 部署的前提，也是 §2.0 不敢在 live 直接訓練的另一面。
- **真實微調語料已開始累積**：live A/B 期間 `live_daemon.py` 記錄的真實 (obs, act, rew) 即階段 ④ RLPD 的輸入；live 環境本身已驗證可收 `sacct` 指標、可逐 (round, idx) 配對。

換言之，§4 沒有「DRL 在 live 大勝」的故事——它的價值是**證明 sim 洞察的方向正確（1×1 該打平、就打平了）、且 sim-to-real 的工程管線（shadow + fail-safe + trace collector）真的能跑**，為多節點上線與 RLPD 微調鋪好路。

### 4.1 RDSAC live A/B（cvar-v2 checkpoint）

把 §3 的 cvar-v2 RDSAC checkpoint 烘進 `slurm-rl-scheduler:m11` 部署 live，與 score-only 做配對 A/B。設定：兩臂各 3 輪 × 14 個 MPS sleep job（`mps∈{20,34,50}`、runtime∈{8,16,26,40}s），逐 (round, idx) 工作負載完全相同。

| 指標 | RDSAC (priority-boost) | score-only | Δ |
|---|---:|---:|---:|
| JCT mean | 86.1s | 86.2s | **−0.1s (−0.2%)** |
| JCT median | 90.5s | 90.0s | +0.5s |
| JCT p95 | 153.0s | 153.6s | −0.6s |
| WAIT mean | 64.1s | 64.2s | −0.1s |
| 配對 ΔJCT (n=42) | **−0.1s**（sd 27.2） | better 20 / tie 11 / worse 11 | — |

**解讀**：1×1 live 下 RDSAC 與 score **統計上無法區分**（−0.2%，遠小於 ±27s 逐對雜訊）。原因與 §3.3 一致——1×1 強 baseline 下 RDSAC 幾乎對每個到達 job 都均勻 boost（每輪 `selected=14/14`），佇列排序等同 score。這證明 **RDSAC 能在 production 正確上線並與啟發式持平**。

### 4.2 擴大評估：128 job/arm + 受控 arm 順序

§4.1 的 42-job 單 block 太少。擴大到 128 job/arm、6×6 參數網格（`mps∈{20,25,34,50,67,75}`、runtime∈{8,14,20,28,36,45}s），operator 全程暫停把拓樸釘在 1 個 GPU node，並**交換 arm 順序**：

| 跑次 | arm 順序 | aggregate ΔJCT | warm-subset ΔJCT |
|---|---|---:|---:|
| v2_123417 | rdsac → score | **−26.9%**（RDSAC 較差）| −2.3%（r3–8）|
| v2swap | score → rdsac（+warmup）| **+8.6%**（RDSAC 較好）| +0.9%（r2–8）|

aggregate ΔJCT 隨 arm 順序翻號 → 先跑的那臂吃到一次性 GPU/MPS 暖機懲罰（rdsac round-1 wait 106.8s vs score 14.4s），所以那 ±20%+ 是**暖機假象、非排程效果**。暖機後逐輪 Δ 在 0 附近抖動，兩次 warm 估計平均 ≈ **−0.7%**。**結論**：1×1 live RDSAC 與 score 真正打平，比 §4.1 多 3× 資料、控掉暖機混淆後依然成立。RDSAC 臂的 RL 在 88–100% 提交上 abstain（snapshot 時效性 + 單 GPU placement 太瑣碎主導）→ as-deployed RDSAC 大多回退成 score。原始檔 `runs/live_ab/SUMMARY_v2.md`。

### 4.3 三方 live：SAC 也打平

把 vanilla SAC 也部署 live（`variant:"SAC"`，serve `/healthz` 回報），同樣的擴大 + 受控 arm 順序協定。結果與 RDSAC 一致：**1×1 live 下 score ≈ RDSAC ≈ SAC，三方分不出**——每個 learned model 都 abstain ~90–100% 而 fail-safe 回 score。原始檔 `runs/live_ab/SUMMARY_sac.md`。

> §4 一致的故事：1×1 太小，placement 端到端無法表現出優勢、也無法表現出明顯劣勢。sim 全 rollout 會放大 checkpoint 不足（略輸 score，§3.2）；live 因 abstain 回退 + 單 GPU placement 瑣碎而把差異洗掉（持平）。真正的增益／檢驗要等拓樸匹配的多節點 checkpoint。

### 4.4 設計中（未執行）：重尾 + 高競爭 live A/B —— 在 1×1 上拆開三方的條件

§4.1–4.3 在 1×1 全部打平，但**打平的原因不只是「1×1 太小」，也是 workload 本身沒有給出可測的決策**。先前 A/B 用 14 個 `sleep N` job、`runtime∈{8,16,26,40}s`、範圍窄又**確定**——有 slack、沒有尾部風險。在這種 workload 下三方**結構上必然打平**，跟模型好壞無關。

要在**現有 1×1 硬體**上（不碰 sim、不等多節點）評估 score/SAC/RDSAC 的差異，正確的施力點是**改 live A/B 的 workload 與量測，而非改模型或拓樸**。1×1 上 live 只剩兩個真實槓桿：**job 排序**（priority boost）與**單 GPU 的 MPS 打包程度**。設計三根支柱讓這兩個決策產生可測的 JCT 差異：

| 支柱 | 做法 | 拆開誰 | 理由 |
|---|---|---|---|
| **(A) 高競爭 / 深佇列 + MPS 超賣** | 一次注入遠多於單 GPU 能舒服容納的 job（peak MPS 需求 ≫ 100），佇列真的堆積 | **score vs SAC/RDSAC** | 沒競爭就沒排序問題，SJF 立刻解完 → 必然打平 |
| **(B) 重尾 runtime + σ-noisy 估計** | 真實 sleep 時間抽自重尾分布；餵給 `/decide` 與排序的是**加噪的估計**（`reported = true·exp(σZ−σ²/2)`），σ 校準到 §3.5 量到的預測器 log-殘差（≈1.2–1.45）| **SAC vs RDSAC** | RDSAC 的優勢全在「不確定下的尾部風險」。若 runtime 確定（`sleep N` 已知）→ CVaR≈mean → RDSAC 與 SAC **結構上必然打平**，改什麼都沒用。這是 §3.4–3.6「σ 越大 RDSAC 越贏」的 **live 對應物** |
| **(E) 尾部量測** | 除 mean JCT，加 p95/p99 JCT、tail slowdown、JCT 的 CVaR、最差 straggler、完成率 | 看得見 RDSAC 的尾部優勢 | RDSAC-cvar 直接優化尾部；只看 mean 會把差異洗掉 |

**配對協定**：同一條 job stream（per-job common-random：相同 arrival / true-runtime / reported-runtime / mps）在 score / SAC / RDSAC 各重放一次；沿用 §2.5/§4.2 的方法學教訓——每臂丟棄 ≥1 warmup round 並交換 arm 順序，避免共用單 GPU 的一次性暖機懲罰帶風向。並記錄每臂的 `/decide` abstain 率：若高競爭下模型仍大量 abstain，那本身是發現，後續才考慮放寬 §8.3 的 `valueAbstain`/`entropyAbstain`。

**為什麼只用 philly 和 ali**：兩者都是**由真實叢集 trace 衍生**、runtime **天生重尾**，正好是支柱 (B) 需要的形狀；burst 是合成尖峰壓力 pattern、非 trace 衍生，對「差異是否會轉移到真實 workload」的論證較弱，故略去。

**資料集適用性（誠實說明）**：philly/ali 的**強項**正是重尾 runtime（符合 (B) 的需求）；**落差**在規模——原 trace 是大叢集、job 跑數小時～數天，且含多 GPU job。我們**不做字面重放**，而是 (1) `gpu_count ≤ 1` 過濾（單 GPU 跑不了多卡 job）、(2) **時間壓縮**把 runtime 映到可測的 live 尺度（目標最長 job ~數分鐘）但**保留尾部形狀與相對次序**。因此這個 live A/B 檢驗的是「**philly/ali 形狀的競爭 + 尾部**下三方是否分得開」，**不是**生產級字面重放——這是有界、已聲明的範圍，不是隱藏假設。

**預期與價值**：(a) 高競爭下 score vs DRL 若仍打平 → 證實 1×1 即使加載也觸頂；(b) σ-noisy + 尾部量測下 RDSAC-cvar 若砍低 p99/CVaR → **sim 的 σ-發現轉移到 live**（本論文 thesis 第一次在真環境兌現）；(c) 若加噪加載後 RDSAC 仍 = SAC → **sim σ-結果不轉移 live**，這是同等有價值的負結果。無論結果如何，這比再跑一輪打平的 A/B 有資訊量。

> **天花板 caveat**：即使如此，1×1 上限仍低（單 GPU 上 score 的 SJF-ish 排序已接近最佳，DRL 上行有限），效果量可能小；決定性檢驗仍是 2-node 異質拓樸。但這是 1×1 能做的最佳、最便宜評估。工程規格見 `docs/live-ab-heavytail-spec.md`。

---

## 5. 結論

| 問題 | 結論 |
|---|---|
| DRL path 能在 live 上跑？ | 可以。warm-pool 穩定化後 RDSAC cvar-v2 live A/B 全 job 乾淨完成、RL boost 確實生效（§4.1）。 |
| 先前 `alpha` 觸頂是真 bug？修好了？ | 是真 bug（return 尺度壓過 entropy ~300×）。已用 reward_scale 1000→20000 + 放寬 clamp 修好（§2.2）。 |
| RDSAC 在標準 sim benchmark 打贏 score？ | **確定性 1×1 下還沒有**（§3.2，30-seed 三族都不優於 score）。但加入真實不確定性後 σ=1.0 fixed-α 可贏過 score（§3.4）。 |
| 分布式 / 風險機制有用嗎？(score vs SAC vs RDSAC) | **看環境有沒有不確定性**。確定性 1×1（含 live）淨增益≈0，且「SAC 最差」是 auto-α 假象（§3.3）。一旦注入**校準過的**真實不確定性（§3.5 σ≈1.2–1.45），RDSAC−SAC 差距隨 σ **單調拉開**（−73→+196 pts，§3.4），fixed-α 下仍成立。三臂拆解（§3.6）指出**增益主要來自「分布式 critic」**（SAC→mean +108~125 pts），CVaR 是尾部專用小加成。CVaR 淨增益需 multi-seed 才能定論（單 seed 擺盪大）。 |
| risk-sensitive(cvar) 優於 risk-neutral(mean)？ | **是**。30-seed 下 cvar 泛化遠勝 mean（−24.6% vs −117%，約 5×，§3.2），支持把 cvar 烘進 live image。 |
| live A/B 已能公平比較 DRL vs score？ | 可以（warm-pool + 三方驗證 + 受控 arm 順序）。但 **1×1 live 分不出模型**——learned model 都 abstain ~90–100% fail-safe 回 score（§4）。 |
| 最穩定上線策略 | DRL live scheduler 保持 enabled + GPU warm pool，並保留 stale snapshot / low confidence / service down 時的 heuristic/Slurm fallback。 |

**工程貢獻**：(1) 可上線的 DRL inference path（非僅 notebook/sim）；(2) DRL 對齊 Ma et al. RDSAC，有單元/行為測試；(3) 定位並修好 temperature auto-tune 的 reward-scale 根因；(4) sim + live trace collector 已能支援後續 RLPD；(5) 乾淨的三方受控對照（score/SAC/RDSAC，一個 flag 切換）+ 隨機性消融把「分布式/風險機制何時有用」的條件講清楚。

**核心一句話**：在 1×1 確定性環境（含 live）三方打平、換演算法贏不了強啟發式；補上**校準過的**真實 runtime 不確定性後 RDSAC > SAC，且贏在「把回報建模成分布」這件事本身，CVaR 風險扭曲是尾部專用加成。不是只宣稱「用了 DRL / risk-sensitive」就算贏。

### 5.1 未來工作（Future Work）

依「擋住結論的程度」排序：

**讓現有結論站得住（方法學門檻）**

1. **多訓練 seed（≥3–5）重跑 §3.4 / §3.6 關鍵點。** 目前每 cell 單一訓練 seed，§3.6 已有同 config 兩跑擺盪 50–90 pts 的鐵證；per-family 數字要 mean±std 才能下定論，尤其 mean-vs-cvar 的細排名。
2. **σ 校準的外部效度。** §3.5 的 σ 是合成 trace 的最難預測上界；應在真實結構化 trace（`load_philly()`）上重量，並把 σ-sweep 落在實測區間。
3. **向量化 / 加速 sim。** 純 Python 離散事件 ~10 steps/s 是多 seed 研究的算力牆（一個 σ 區塊 ~4.6h），是上面兩項可行的前置工程。

**讓 sim 結論能轉移到 live**

★ **（現有 1×1 即可做、最近期）重尾 + 高競爭 live A/B。** 改 live A/B 的 workload（深佇列 + MPS 超賣 + 重尾 runtime + σ-noisy 估計）與量測（p95/p99/CVaR 尾部），檢驗 §3.4–3.6 的「σ 越大 RDSAC 越贏」是否轉移到 live。不碰 sim、不等硬體。設計理由見 §4.4，工程規格見 `docs/live-ab-heavytail-spec.md`。

4. **修 train/serve 動作落差。** sim 訓練 placement policy（job, node, gpu），live 只把 RL 選擇轉成 priority boost、Slurm 仍做真正 allocation；兩者對齊（live 真接 explicit placement，或 sim 改學 priority/selection）後 sim 結論才能宣稱轉移到 live。
5. **拓樸匹配的多節點 checkpoint（RTX 3080 第二節點，`docs/intergration.md`）。** 單卡 placement 退化；2-node（2×1 異質）才讓「放哪張卡」「共置與否」（§3.7）成為真實決策，也才能在 live 分出高下。需先補 `rtx3080` 進 `GPU_TYPES` / `_gpu_type_to_vram`。
6. **補強 baseline。** 目前只比自家 score + vanilla SAC；補 FCFS / SJF（已有 oracle runtime）/ packing 啟發式與近似上界，讓 ΔJCT% 有尺度感。

**演算法與韌性**

7. **return normalization（PopArt）** 取代手調 reward_scale，讓單一 α 控制器跨 SAC/RDSAC 都穩（消掉 §3.3 必須釘 α 的 caveat）。
8. **機制 ablation**（PER / potential shaping / 雙頭 Z_R/Z_H）與 **score-warmup on/off**（驗證 live abstain ~90% 是否來自 imitate score）。
9. **per-model 各自調好的溫度**（本輪只釘單一 α=0.05）與 **held-out workload split**（train philly+burst、test ali）證明泛化。

---

## 6. 重現指令

**確定性 sim 受控對照（§3.1–3.2，mean 與 cvar 只差 `--risk-mode`）**

```bash
PYTHONPATH=. .venv-m11/bin/python eval/scripts/eval_dsac_placement.py \
  --total-steps 150000 --warmup-steps 2000 --n-jobs 50 \
  --n-nodes 1 --gpus-per-node 1 \
  --trace-families philly burst ali --train-trace philly burst ali \
  --seeds 42 43 44 45 46 --no-attention --curriculum --reward-scale 20000 \
  --risk-mode cvar --risk-beta 0.25 --device cuda \
  --out-dir runs/rdsac_eval_cvar_v2     # mean: --risk-mode mean
```

**σ 校準（§3.5）**

```bash
PYTHONPATH=. .venv-m11/bin/python eval/scripts/measure_predictor_sigma.py \
  --trace sim/data/philly_subsample.json
```

**隨機性消融 + 三臂拆解（§3.4, §3.6）**

```bash
PYTHONPATH=. .venv-m11/bin/python eval/scripts/sweep_stochastic.py \
  --sigmas 1.0 --total-steps 100000 --n-jobs 50 --curriculum \
  --seeds 42 43 44 45 46 --trace-families philly burst ali \
  --risk-modes mean cvar --fixed-alpha --init-alpha 0.05 --device cuda
# 共置消融（§3.7）：加 --colocation --interference 0.3 --no-sac
```

**Live A/B（§4）**

```bash
sudo kubectl exec -n slurm deploy/slurm-login -- bash -lc '
for spec in smallA:25:4 smallB:25:4 fullA:100:12 smallC:25:4 halfA:50:6 smallD:25:4; do
  IFS=: read -r name mps secs <<< "$spec"
  sbatch --parsable -p gpu-rtx4070 --gres=mps:${mps} -c 1 --mem=512M --time=00:03:00 \
    -J "bench-${name}" --wrap "echo start=\$(date -Is) host=\$(hostname); sleep ${secs}; echo end=\$(date -Is)"
done'
# 收集：sudo kubectl exec -n slurm slurm-controller-0 -- sacct -X -P -j <ids> --format=JobID,State,Submit,Start,End,ElapsedRaw,AllocTRES%120
```

---

## 7. 資料集來源

§3 的 simulator benchmark **並非直接重放原始資料集**，而是用 `sim/loader.py` 的合成生成器，分布參數依下列公開 GPU cluster trace 的已發表統計校準（可離線、無網路重現）。Philly 另提供真實 `cluster_log_data.json` 的載入路徑（`load_philly()`）。

| Trace family | 生成器 | 模仿來源 | 連結 / 論文 |
|---|---|---|---|
| `philly` | `generate_philly_like`（亦支援 `load_philly` 真實重放）| Microsoft Philly GPU cluster trace | github.com/msr-fiddle/philly-traces — Jeon et al., *Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads*, USENIX ATC 2019 |
| `ali` | `generate_ali_like` | Alibaba PAI GPU cluster trace（MPS-fractional、短尾、多單卡）| github.com/alibaba/clusterdata（`cluster-trace-gpu-v2020`）— Weng et al., *MLaaS in the Wild*, USENIX NSDI 2022 |
| `burst` | `generate_burst_heavy` | **非具名公開資料集**：沿用 `philly` 的 job-size 組合，疊加日週期爆發到達，作為到達突發壓力測試 | —（合成壓力模式）|

> 注意：`philly` / `ali` 是「**統計近似**」而非逐筆原始資料；數值反映的是這些 trace 的工作負載**特性**（job 大小分布、到達節奏、runtime 尾部），不等同在原始 production log 上的表現。嚴格對照時建議用 `load_philly()` 載入 msr-fiddle/philly-traces 的真實 trace。
