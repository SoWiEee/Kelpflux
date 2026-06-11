# Kelpflux Scheduler Evaluation

本文件整理目前上線規格下的 scheduler evaluation。重點不是證明 DRL 一定優於啟發式，而是清楚比較三種做法在同一套 simulator 與 live Slurm/k3s/GPU 環境中的行為：heuristic score、SAC、RDSAC。

> 註：DRL 實作已從早期的 discrete SAC 改寫為 Ma et al. 的 risk-sensitive distributional SAC（RDSAC）。本文 §3 為改寫後的新結果；§4 live cluster 結果是 RDSAC 改寫前用舊 checkpoint 跑的，保留作為基礎設施驗證紀錄。

## 1. 評估對象

### 1.1 Heuristic score scheduler

heuristic score 是目前最穩定的 submit-time baseline。它不需要訓練模型，而是直接用 job 需求與 cluster 狀態計算優先權，交給 Slurm `select/cons_tres` 做實際 placement。

核心直覺：

| 訊號 | 用途 |
|---|---|
| MPS fit | 小 MPS job 是否能塞進目前 GPU 的剩餘 slot，避免大 job 佔滿整張卡。 |
| VRAM fit | job 需求是否符合 GPU tier，避免高階卡被低需求 job 浪費。 |
| fragmentation penalty | 避免接受一個 job 後讓剩餘 MPS slot 過度碎片化。 |
| runtime shortness | 若 runtime predictor 有可用估計，短 job 會得到額外 priority boost。 |

相關實作：

| 層級 | 檔案 |
|---|---|
| simulator baseline | `sim/scheduler/score.py` |
| live submit hook | `chart/templates/configmap-job-submit.yaml` |
| Slurm priority path | `chart/templates/slurm-conf.yaml`、`chart/templates/login.yaml`、`chart/templates/workers.yaml` |

### 1.2 SAC

SAC 是 Soft Actor-Critic。它的目標不是只最大化 reward，而是同時最大化 reward 與 policy entropy，因此會保留探索能力。標準 SAC 通常用在連續 action space，例如控制機器人的力矩或速度。

Kelpflux 的排程 action 是離散且有 mask 的：每一步只能從 pending queue 中可行的 job、node、GPU/MPS placement 組合裡選一個，不能選資源不足的 action。因此目前沒有把「連續 SAC」直接作為 live scheduler；SAC 在本專案中主要作為 DSAC 的概念來源與比較基準。

### 1.3 RDSAC

RDSAC 是 Ma et al. 2020/2025〈DSAC: Distributional Soft Actor-Critic for Risk-Sensitive Reinforcement Learning〉(arXiv:2004.14547) 的**離散動作忠實轉寫**，取代早期的 discrete SAC。它把連續控制的 reparameterised Gaussian actor 換成**顯式 categorical actor**（placement 是離散動作），critic 與 risk 機制依論文 §4.1：

- **雙分布 critic**：把 soft return 拆成 reward 分布 `Z_R` 與 entropy 分布 `Z_H`，各以 IQN 的 quantile 表示、共用 trunk，quantile Huber 回歸 + twin double learning。
- **risk 進策略目標**：actor 目標對 reward 分布套用 distortion `ρ`，`risk_mode ∈ {mean, cvar, wang, cpw, msd}`。`mean` 是 risk-neutral，等同穩定性導向的 distributional SAC；`cvar` 偏好下尾較不嚴重的 placement，對應排程的 straggler / cold worker / long-tail runtime 風險。

實作在 `services/rl_scheduler/dsac.py` 與 `services/rl_scheduler/distortion.py`，演算法與單元/行為測試見 commit `fe899ec`。本文 §3 報告 `mean`（risk-neutral）與 `cvar`（β=0.25）兩個變體在**完全相同配置、僅 `risk_mode` 不同**下的受控對照，這也是本次評估的主軸：驗證 risk distortion 對 placement 是否真的有用。

## 2. Benchmark 方法

### 2.1 Simulator paired benchmark

標準 simulator benchmark 使用相同 seed 對 RDSAC 與 heuristic score 做 paired comparison，降低 trace 隨機性造成的誤差。

本次設定（mean / cvar 兩 run 共用，只差 `--risk-mode`）：

| 項目 | 值 |
|---|---|
| agent | RDSAC，`risk_mode ∈ {mean, cvar}`（cvar β=0.25） |
| critic trunk | MLP（`--no-attention`） |
| training | 從頭訓練 150k 步，curriculum n_jobs 10→30→50 |
| reward_scale | **20000**（修復 alpha 觸頂，見下方說明） |
| checkpoint | `runs/rdsac_eval_mean_v2/`、`runs/rdsac_eval_cvar_v2/` |
| cluster | 1 node × 1 GPU |
| jobs per trace | 50 |
| trace families | `philly`, `burst`, `ali` |
| seeds | 42, 43, 44, 45, 46 |
| metric | mean JCT（主），p95 / p99 JCT（尾部） |

#### reward_scale 修復（alpha 觸頂根因）

上一輪 mean run 出現 `alpha` 自動調到 clamp 上限 2.718（=e¹）的警訊。排查後確認**不是 alpha 邏輯錯**（符號為標準 Christodoulou discrete-SAC 形式），而是 **return 尺度問題**：

- 舊 `reward = −JCT / 1000`，50 個 job 累加 → 每個 episode 的 return 量級約 **O(−150)**，critic 學到的 `E[Z_R]` 跨動作落差約 **560**。
- actor 目標 `Σ π·(α·logπ − ρ[Z_R])` 中，entropy 正則項 `α·log(n)` 上限僅約 2，被 `Z_R`（~560）**壓過約 300×**。
- 結果：policy 在訓練早期就塌成 one-hot（量到的 actor entropy ≈ 0.002，遠低於 target 0.1·log(n)），**探索停擺**；alpha auto-tune 看到 entropy ≪ target 就一路把 temperature 推到 clamp 上限仍無力回天。

修法（最小幅度，走既有 `sim_train`）：把 `reward_scale` 由 1000 提到 **20000**，使 return 量級降到 O(−10)、讓 entropy 項與 Q 同量級可競爭；同時把 log_alpha clamp 上限由 1.0 放寬到 3.0（α≤~20），給 auto-tune 餘量；並在 `sim_train.jsonl` 補記 alpha/entropy 以便驗證。修復後量測：alpha 不再釘頂、自由調節（mean run 收斂於 α≈1.4、cvar run α≈0.66），actor entropy 由 0.002 恢復到 **≈0.13**，回到 target 附近。

> 與更早版本差異：最早的 §3 數字來自 500k 步 discrete-SAC checkpoint，**不應**與本次 RDSAC 150k 步直接比較。本次 mean 與 cvar 則是同步驟、同 reward_scale、同 seed 的乾淨受控對照。

重現指令見 §6。

### 2.2 Live cluster smoke A/B

使用者要求 benchmark 要跑在 live cluster 上，因此本次也在實際 k3s + Slurm + GPU/MPS 環境提交 `sbatch` job。這不是大樣本統計 benchmark，而是 live path smoke A/B，用來驗證 DSAC live scheduler、Slurm placement、worker lifecycle 與 MPS allocation 是否真的能協同運作。

live 環境：

| 項目 | 值 |
|---|---|
| namespace | `slurm` |
| login | `deploy/slurm-login` |
| controller | `slurm-controller-0` |
| GPU partition | `gpu-rtx4070` |
| GPU worker | `slurm-worker-gpu-rtx4070-0`, `slurm-worker-gpu-rtx4070-1` |
| GRES | `gpu:rtx4070:1,mps:rtx4070:100` per worker |

live workload：6 個短 GPU/MPS sleep jobs，依序要求 `mps=25,25,100,25,50,25`。這組 workload 可以檢查小 MPS job 是否能共用 GPU，以及 full-GPU job 是否會排到獨立 worker。

## 3. Simulator 結果

reward_scale 修復後重跑，150k 步、5 seeds，`mean` 與 `cvar` 兩變體（`Δ` 為 `(score − dsac)/score`，負值代表 RDSAC 的 JCT 較高、較差）。

### 3.1 risk_mode = mean（risk-neutral）

| Family | RDSAC mean JCT | Score mean JCT | Δ | 95% CI | p-value | 顯著 | p95 JCT | p99 JCT |
|---|---:|---:|---:|---:|---:|:--:|---:|---:|
| philly | 7.537 h | 2.621 h | -187.6% | [-261.3%, -113.9%] | 0.0021 | **是** | 27.04 h | 37.84 h |
| burst | 8.007 h | 3.541 h | -126.1% | [-223.8%, -28.4%] | 0.0231 | **是** | 37.15 h | 55.66 h |
| ali | 2.342 h | 1.383 h | -69.4% | [-134.9%, -3.8%] | 0.0424 | **是** | 7.59 h | 22.17 h |

### 3.2 risk_mode = cvar（β=0.25，下尾風險敏感）

| Family | RDSAC mean JCT | Score mean JCT | Δ | 95% CI | p-value | 顯著 | p95 JCT | p99 JCT |
|---|---:|---:|---:|---:|---:|:--:|---:|---:|
| philly | 2.783 h | 2.621 h | -6.2% | [-63.1%, +50.7%] | 0.777 | 否 | 9.42 h | 22.64 h |
| burst | 4.320 h | 3.541 h | -22.0% | [-130.6%, +86.7%] | 0.604 | 否 | 16.16 h | 65.66 h |
| ali | 2.301 h | 1.383 h | -66.4% | [-111.5%, -21.3%] | 0.0150 | **是** | 7.14 h | 18.43 h |

### 3.3 解讀：risk distortion 才是 RDSAC 的價值來源

這組受控對照（兩 run 只差 `risk_mode`，reward_scale、步數、seed 全相同）得到一個清楚、且有點反直覺的結論：

- **修好 alpha 反而讓 risk-neutral（mean）變差，並從「未顯著」掉到「顯著低於 baseline」**。上一輪 alpha 釘在 ceiling、entropy≈0 的「壞」狀態，等效於全程近乎 deterministic greedy，反而碰巧較穩（philly 4.48h、未顯著）；一旦 entropy/探索恢復，mean 變體在這些 trace 上**訓練不穩定**——n_jobs=50 階段的 avg_jct 在約 10k–49k 秒之間劇烈震盪、不收斂——greedy eval 取到的策略因此更差。換句話說，risk-neutral SAC 的探索在 1×1 的窄 action space + 強 baseline 下沒有帶來增益。
- **CVaR 才是讓 RDSAC 真正發揮的關鍵**，這正是當初採用 Ma et al. risk-sensitive DSAC 的初衷。同樣 reward_scale、同樣探索強度下，cvar：
  - 把 **philly 拉到與強 score baseline 統計打平**（−6.2%，p=0.78，CI 跨 0），mean 則是 −187.6%、p=0.002 顯著差；
  - **尾部 p95 大致砍半**：philly 9.42h vs mean 27.04h、burst 16.16h vs mean 37.15h；philly per-seed 也更集中（1.86–4.48h vs mean 4.70–9.19h）；
  - 收斂在較低的 α≈0.66（mean≈1.4），即較 exploitative、訓練較穩。
- **誠實的限制**：(1) cvar 在 **ali 仍顯著差**（−66%，p=0.015），這個 trace 對 RDSAC 一直最難；(2) burst 的 **p99 反而比 mean 差**（65.66h vs 55.66h）——p95 改善但最尾端有單一壞 seed，CVaR 對「平均尾部」有效不代表壓得住最極端的離群；(3) 整體上 cvar 仍未在任何 family 上**贏過** score，只是把差距縮到不顯著。1×1 + 已調好的啟發式，這個結果與先前一致：換演算法不會憑空贏過強 baseline，但 risk distortion 明確優於 risk-neutral。

結論一句話：**alpha 修復本身不是效能銀彈，但它讓 risk 機制能正常運作；在能正常運作後，CVaR 對 mean JCT 與尾部都顯著優於 risk-neutral mean**，這支持了採用 risk-sensitive DSAC 的設計選擇。

## 4. Live cluster 結果

> §4.1–4.3 的 live 結果是 RDSAC 改寫前、用舊 discrete-SAC checkpoint 跑的。它們驗證的是 serving / Slurm placement / worker lifecycle / MPS allocation 等基礎設施路徑能否運作，與演算法版本無關，因此保留。**§4.4 是換上 RDSAC cvar-v2 checkpoint、warm-pool 穩定化後重跑的配對 A/B。**

### 4.1 DSAC live scheduler enabled

提交 job：`136-141`。結果全部 `COMPLETED`。

| Job | Name | Req MPS | State | Submit | Start | End | Wait | Runtime | JCT | Node |
|---:|---|---:|---|---|---|---|---:|---:|---:|---|
| 136 | bench-dsaclive-smallA | 25 | COMPLETED | 07:28:10 | 07:28:13 | 07:28:18 | 3s | 5s | 8s | gpu-rtx4070-0 |
| 137 | bench-dsaclive-smallB | 25 | COMPLETED | 07:28:10 | 07:28:13 | 07:28:18 | 3s | 5s | 8s | gpu-rtx4070-0 |
| 138 | bench-dsaclive-fullA | 100 | COMPLETED | 07:28:10 | 07:28:18 | 07:28:30 | 8s | 12s | 20s | gpu-rtx4070-0 |
| 139 | bench-dsaclive-smallC | 25 | COMPLETED | 07:28:10 | 07:28:25 | 07:28:29 | 15s | 4s | 19s | gpu-rtx4070-1 |
| 140 | bench-dsaclive-halfA | 50 | COMPLETED | 07:28:10 | 07:28:25 | 07:28:31 | 15s | 6s | 21s | gpu-rtx4070-1 |
| 141 | bench-dsaclive-smallD | 25 | COMPLETED | 07:28:11 | 07:28:29 | 07:28:33 | 18s | 4s | 22s | gpu-rtx4070-1 |

Summary：

| Metric | Value |
|---|---:|
| completed jobs | 6 / 6 |
| mean wait | 9.83s |
| mean runtime | 6.33s |
| mean JCT | 14.67s |
| observed placement | first two 25-MPS jobs co-located on worker 0; 100-MPS job isolated during execution; later 25/50/25-MPS jobs co-located on worker 1 |

這次 live run 證明 DSAC scheduler enabled 時，job submit、priority decision、Slurm placement、MPS GRES allocation 與 worker pod 都能完成一輪實際運作。

### 4.2 Score-only fallback run

為了比較 fallback 行為，測試中暫時把 `rl-scheduler` scale 到 0，提交相同 workload。提交 job：`142-147`。

結果：score-only fallback run 沒有形成可比較的完成樣本。前三個 job 進入 worker 後變成 `NODE_FAIL` / `COMPLETING`，後三個 job 因 GPU worker unavailable/resource busy 留在 pending，之後已用 `scancel` 清掉 pending jobs，並用 Slurm `DOWN` / `RESUME` 流程清空 queue。`rl-scheduler` 已恢復為 1 replica。由於 elastic operator 在沒有 GPU job 時會把 GPU worker StatefulSet 縮到 0，Slurm 端後續會看到 GPU nodes 為 `idle*` / not responding，直到下一次 GPU workload 觸發 worker scale-up。

`sacct` 摘要：

| Job | Name | Req MPS | State | Start/End 狀態 | Node |
|---:|---|---:|---|---|---|
| 142 | bench-scoreonly-smallA | 25 | NODE_FAIL / COMPLETING | Start 曾在 07:30:21；End 07:30:32 | gpu-rtx4070-0 |
| 143 | bench-scoreonly-smallB | 25 | NODE_FAIL / COMPLETING | Start 曾在 07:30:21；End 07:30:33 | gpu-rtx4070-0 |
| 144 | bench-scoreonly-fullA | 100 | NODE_FAIL / COMPLETING | End 07:30:31；Slurm show job StartTime 為 Unknown | gpu-rtx4070-1 |
| 145 | bench-scoreonly-smallC | 25 | PENDING, then cancelled | no node assigned | none |
| 146 | bench-scoreonly-halfA | 50 | PENDING, then cancelled | no node assigned | none |
| 147 | bench-scoreonly-smallD | 25 | PENDING, then cancelled | no node assigned | none |

當時 node 狀態：

```text
slurm-worker-gpu-rtx4070-0  IDLE+COMPLETING+NOT_RESPONDING
slurm-worker-gpu-rtx4070-1  IDLE+COMPLETING+NOT_RESPONDING
```

這表示 live 對照組遇到 worker lifecycle / Slurm completion acknowledgement 問題。GPU worker pod 在測試期間被重新建立，導致 Slurm 將 node 視為 not responding，job 完成回報不乾淨。測試後已將 queue 清空；GPU worker StatefulSet 依目前 scale-to-zero 策略回到 0 replica。這個結果不能用來主張 score 比 DSAC 差；它只能說明目前 live benchmark 若要做嚴格 A/B，需要先固定 worker lifecycle 或在每個 phase 前重置成相同 warm/cold 狀態。

### 4.3 P0 fix follow-up smoke

後續修正兩個 live worker lifecycle 問題：

1. operator 在 scale-down 成功 patch replicas 後，會把被移除的 Slurm worker nodes 標成 `DOWN`；下一次 scale-up 會先 `RESUME` 目標 StatefulSet ordinals。
2. chart 不再於 k3s pod 的 GPU `gres.conf` 條目渲染 `Cores=`。live worker 曾因 `Cores=0-3` 和容器可見 CPU topology 不一致而在 slurmd log 出現 `Invalid GRES data for gpu, Cores=0-3`，導致 job stuck `COMPLETING`。

部署後重新提交 GPU/MPS smoke job `149`：

| Job | Name | Req MPS | State | Submit | Start | End | Runtime | Node |
|---:|---|---:|---|---|---|---|---:|---|
| 149 | p0-gres-smoke | 25 | COMPLETED | 07:56:55 | 07:56:59 | 07:57:15 | 16s | gpu-rtx4070-0 |

測試後 queue 為空；GPU worker StatefulSet 回到 scale-to-zero 狀態，Slurm GPU nodes 保持 `DOWN` / `DRAIN`，等待下一批 GPU workload 由 operator scale-up 並 resume。

### 4.4 RDSAC checkpoint live A/B（重跑，cvar-v2）

把 §3 的 **cvar-v2 RDSAC checkpoint** 烘進 `slurm-rl-scheduler:m11` 部署到 live，與 score-only 啟發式做配對 A/B。要做到「RDSAC 真的有作用」的公平比較，先排除兩個讓比較失真的問題：

1. **拓樸不匹配 → `/decide` 靜默 abstain**：1×1 checkpoint（`n_actions=17`）遇到 2 個 healthy GPU worker 時，snapshot 回報 `nodes=2`、serve 以 33 個 action 建 mask → shape mismatch → 每次決策都早期 abstain、`priority_boost_total` 不動，RL 看似在跑實際完全不影響排程（詳見 `docs/note.md` #17.2）。
2. **worker lifecycle / 冷啟動 race**：GPU pool `scale-to-0` 後冷啟動喚醒會與 job dispatch 競爭而 `NODE_FAIL`（即 §4.2 暴露、`docs/note.md` #16 的死結）。

**穩定化做法**：把 `gpu-rtx4070` pool 設成 `min_replicas=1`（warm pool）。一顆常駐 GPU 節點同時解掉兩件事 —— 沒有冷啟動 race，且「恰好 1 個 healthy GPU node」讓 snapshot `nodes=1` 與 1×1 checkpoint 匹配、`/decide` 正常 boost。Operator 另補兩個 hardening：scale-up in-flight gate（避免 0→1→2 overshoot 打死剛落地的 job）與 provisioning-complete 後 `scontrol reconfigure`（清掉換 pod IP 造成的 `NOT_RESPONDING`）。詳見 `docs/cluster.md §4`。

**A/B 設定**：兩臂各 3 輪 × 14 個 MPS sleep job（`mps∈{20,34,50}`、runtime∈{8,16,26,40}s），逐 (round, idx) 工作負載完全相同。RDSAC 臂 `shadowMode=false`（RL boost 生效）、score 臂 `shadowMode=true`（boost 強制 0、純啟發式）。指標由 controller pod 的 `sacct` 收。

| 指標 | RDSAC (priority-boost) | score-only | Δ |
|---|---:|---:|---:|
| JCT mean | 86.1s | 86.2s | **−0.1s (−0.2%)** |
| JCT median | 90.5s | 90.0s | +0.5s |
| JCT p95 | 153.0s | 153.6s | −0.6s |
| WAIT mean | 64.1s | 64.2s | −0.1s |
| 配對 ΔJCT (n=42) | **−0.1s**（sd 27.2） | better 20 / tie 11 / worse 11 | — |

**解讀**：1×1 live 下 RDSAC 與 score **統計上無法區分**（ΔJCT −0.2%，遠小於 ±27s 逐對雜訊）。原因與 §3 一致 —— 在 1×1、強啟發式 baseline 下，RDSAC 幾乎對每個到達的 job 都均勻 boost（每輪 `selected=14/14`），佇列排序等同 score 的排序，沒有重排效果。這證明 **RDSAC 能在 production 正確上線並與啟發式持平**，但真正的增益要等「拓樸匹配的多節點 checkpoint、placement 選擇有意義」時才會出現；單純把 1×1 checkpoint 丟到 live 不會贏。

### 4.5 RDSAC 擴大評估：更多資料、受控 order（2026-06-11）

§4.4 的 42-job 單一 block 樣本太少、CI 太寬，不足以下定論。本節用**更多資料**重評，並補上一個 §4.4 沒控到的混淆因子。原始檔：`runs/live_ab/SUMMARY_v2.md`、`runs/rdsac_eval30_*/SUMMARY.md`。

**(a) Live：擴大到 128 job/arm、6×6 參數網格，並交換 arm 順序。** 兩次跑各 8 輪 × 16 job（`mps∈{20,25,34,50,67,75}`、runtime∈{8,14,20,28,36,45}s，掃滿 36 種組合），operator 全程暫停把拓樸釘在 1 個 GPU node。**aggregate ΔJCT 會隨 arm 順序翻號**：

| 跑次 | arm 順序 | aggregate ΔJCT | warm-subset ΔJCT |
|---|---|---:|---:|
| v2_123417 | rdsac → score | **−26.9%**（RDSAC 較差） | −2.3%（r3–8） |
| v2swap | score → rdsac（+warmup） | **+8.6%**（RDSAC 較好） | +0.9%（r2–8） |

先跑的那一臂吃到一次性的 GPU/MPS 暖機懲罰（run #1 rdsac round-1 wait = 106.8s vs score 14.4s），所以那個看似顯著的 ±20%+ aggregate **是 order/warmup 假象、不是排程效果**。暖機後逐輪 Δ 在 0 附近抖動（run #2 r3–8：+0.8/+0.5/−1.2/−0.9/+1.1/−0.4 s，工作 60–100s），兩次 warm 估計平均 ≈ **−0.7%**。**結論：1×1 live 下 RDSAC 與 score 真正打平**，比 §4.4 −0.2% 多 3× 資料、且控掉了 block 設計的暖機混淆後依然成立。另外 RDSAC 臂的 RL 在 88–100% 的提交上 abstain（受 snapshot 時效性 + 單 GPU placement 太瑣碎主導，逐跑高變異），更印證「as-deployed RDSAC 大多回退成 score、無可測差異」。

**(b) Sim：30 seed 重評（取代舊的 5 seed）。** 用兩個 v2 checkpoint 以 as-deployed risk 模式（cvar→cvar、mean→mean）在 1×1、`n_jobs=50` 跑配對 eval。5 seed 的 CI 寬到無意義（舊 cvar philly −6.2%，CI[−63%,+51%]，p=0.78），30 seed 後 CI 收斂、符號穩定：

| checkpoint | philly | burst | ali |
|---|---:|---:|---:|
| **cvar_v2**（live） | −24.6%（不顯著，p≈0.12） | −31.1%（顯著） | −120.9%（顯著） |
| **mean_v2** | −117.3%（顯著） | −159.2%（顯著） | −128.4%（顯著） |

兩個 finding：**(i)** 在 1×1 sim 全 rollout 下**兩個 checkpoint 都贏不了 score**——舊 5-seed 的「接近持平」是雜訊；**(ii)** **cvar 的泛化遠勝 mean**（−24.6% vs −117%，約 5×），這正是把 cvar 而非 mean 烘進 live image 的經驗依據。

**(a)(b) 一致的故事**：1×1 太小，placement 策略端到端**無法表現出優勢、也無法表現出明顯劣勢**。sim 全 rollout 會放大 checkpoint 的不足（略輸 score）；live 端因 abstain 回退 + 單 GPU placement 瑣碎而把差異洗掉（持平）。真正的增益／檢驗要等拓樸匹配的多節點 checkpoint。**方法論教訓**：共用單 GPU 的 block A/B 必須丟棄每臂 ≥1 個 warmup round 並交換 arm 順序（或逐輪 interleave），否則 aggregate 會被一次性暖機帶風向（→ `docs/note.md` #18）。

## 5. 結論

目前可誠實下的結論：

| 問題 | 結論 |
|---|---|
| DRL path 是否能在 live 上跑？ | 可以。舊 checkpoint job `136-141` 全部完成；**RDSAC cvar-v2 live A/B 已重跑**（§4.4），warm-pool 穩定化後 84 個 job 全乾淨完成、RL boost 確實生效。 |
| 先前 `alpha` 觸頂是真 bug 嗎？已修好？ | 是真 bug（return 尺度壓過 entropy 項 ~300×，policy 塌縮、探索停擺）。已用 reward_scale 1000→20000 + 放寬 clamp 修好：alpha 不再釘頂、entropy 由 0.002 恢復到 ≈0.13。 |
| RDSAC 是否在標準 simulator benchmark 打贏 score？ | 還沒有。**30-seed 重評**（§4.5b，取代舊 5-seed 雜訊）下 cvar 三個 family 都不優於 score（philly −24.6% 不顯著、burst/ali 顯著較差），mean 更全面顯著較差。1×1 sim 全 rollout 下沒有任何 family 贏過 score。 |
| risk-sensitive(cvar) 是否優於 risk-neutral(mean)？ | **是，明確優於**。30-seed 下 cvar 的泛化遠勝 mean（philly −24.6% vs −117%，約 5×），支持把 cvar 而非 mean 烘進 live image 的設計選擇；唯絕對值仍未過 score baseline。 |
| live A/B 是否已能公平比較 DRL vs score？ | **可以了，且已用更多資料驗證**。warm-pool（`min_replicas=1`）消除冷啟動 race 並讓 1×1 checkpoint 拓樸匹配。§4.5a 擴大到 128 job/arm 並**交換 arm 順序**：aggregate 會隨順序翻號（−27% ↔ +8.6%），證明那是一次性暖機假象；warm-subset 平均 ≈ −0.7%，**RDSAC 與 score 在 1×1 真正打平**（confirms §4.4 −0.2%）。 |
| 目前最穩定的上線策略 | DRL live scheduler 保持 enabled + GPU warm pool（`min_replicas=1`），並保留 stale snapshot、low confidence、service down 時的 heuristic/Slurm fallback。 |

工程貢獻目前比較明確的是：

1. 已有可上線的 DRL inference path，而不是只停在 notebook/simulator。
2. DRL 實作已對齊 Ma et al. RDSAC（雙分布 critic + categorical actor + risk distortion），並有單元/行為測試覆蓋；risk 機制是相對舊 discrete-SAC 的主要新能力。
3. 已定位並修好 temperature auto-tune 的 reward-scale 根因，並在 `sim_train.jsonl` 加上 alpha/entropy instrumentation，後續訓練可直接觀測收斂品質。
4. simulator 與 live trace collector 已能支援後續 RLPD：先用 live `sacct` / normalized trace 蒐集真實 transition，再混合 simulator replay 做 fine-tuning。
5. benchmark 給出乾淨的 risk-mode 受控對照：30-seed 重評（§4.5b）下 RDSAC 在 1×1 + 強 baseline 下仍未取勝，但 **CVaR 泛化遠勝 risk-neutral mean**（約 5×），這是把 cvar 烘進 live image 的依據。RDSAC live A/B 也已用**更多資料 + 受控 arm 順序**重跑（§4.5a）：warm-pool 穩定化後 RDSAC 能在 production 正確運作、與 score 在 1×1 真正打平（aggregate 隨順序翻號證明為暖機假象，warm Δ≈−0.7%）。後續方向：穩定 mean 變體訓練、改善 ali 與 burst 最尾端、提升 reward fidelity，並訓練拓樸匹配的多節點 checkpoint 讓 placement 選擇真正有意義。不是只宣稱「用了 DRL / risk-sensitive」就算贏。

## 6. 本次驗證指令

Simulator（本次 §3 受控對照，mean 與 cvar 只差 `--risk-mode`）：

```bash
# risk-neutral mean
PYTHONPATH=. .venv-m11/bin/python eval/scripts/eval_dsac_placement.py \
  --total-steps 150000 --warmup-steps 2000 --n-jobs 50 \
  --n-nodes 1 --gpus-per-node 1 \
  --trace-families philly burst ali --train-trace philly burst ali \
  --seeds 42 43 44 45 46 \
  --no-attention --curriculum --reward-scale 20000 \
  --risk-mode mean --device cuda \
  --out-dir runs/rdsac_eval_mean_v2

# risk-sensitive cvar (β=0.25)
PYTHONPATH=. .venv-m11/bin/python eval/scripts/eval_dsac_placement.py \
  --total-steps 150000 --warmup-steps 2000 --n-jobs 50 \
  --n-nodes 1 --gpus-per-node 1 \
  --trace-families philly burst ali --train-trace philly burst ali \
  --seeds 42 43 44 45 46 \
  --no-attention --curriculum --reward-scale 20000 \
  --risk-mode cvar --risk-beta 0.25 --device cuda \
  --out-dir runs/rdsac_eval_cvar_v2
```

Live DSAC phase：

```bash
sudo kubectl exec -n slurm deploy/slurm-login -- bash -lc '
phase=dsaclive
for spec in smallA:25:4 smallB:25:4 fullA:100:12 smallC:25:4 halfA:50:6 smallD:25:4; do
  IFS=: read -r name mps secs <<< "$spec"
  sbatch --parsable -p gpu-rtx4070 --gres=mps:${mps} -c 1 --mem=512M --time=00:03:00 \
    -J "bench-${phase}-${name}" \
    --wrap "echo phase=${phase} name=${name} mps=${mps} start=\$(date -Is) host=\$(hostname); sleep ${secs}; echo end=\$(date -Is)"
done
'
```

Live result collection：

```bash
sudo kubectl exec -n slurm slurm-controller-0 -- bash -lc \
  'sacct -X -P -j 136,137,138,139,140,141,142,143,144,145,146,147 \
   --format=JobID,JobName,Partition,State,Submit,Start,End,ElapsedRaw,NodeList,AllocTRES%120,ReqTRES%120'
```

## 7. 資料集來源

§3 的 simulator benchmark **並非直接重放原始資料集**，而是用 `sim/loader.py` 的合成生成器，其分布參數（GPU 數量比例、到達過程、log-normal runtime 與尾部）依下列公開 GPU cluster trace 的已發表統計校準，因此可離線、無網路重現。其中 Philly 另提供真實 `cluster_log_data.json` 的載入路徑（`load_philly()`），可選擇用原始 trace 重放。

| Trace family | 生成器 | 模仿來源 | 資料集連結 | 對應論文 |
|---|---|---|---|---|
| `philly` | `generate_philly_like`（亦支援 `load_philly` 真實重放） | Microsoft Philly GPU cluster trace | https://github.com/msr-fiddle/philly-traces | Jeon et al., *Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads*, USENIX ATC 2019 — https://www.usenix.org/conference/atc19/presentation/jeon |
| `ali` | `generate_ali_like` | Alibaba PAI GPU cluster trace（MPS-fractional、短尾、多單卡） | https://github.com/alibaba/clusterdata（`cluster-trace-gpu-v2020`） | Weng et al., *MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters*, USENIX NSDI 2022 — https://www.usenix.org/conference/nsdi22/presentation/weng |
| `burst` | `generate_burst_heavy` | **非具名公開資料集**：沿用 `philly` 的 job-size 組合，疊加日週期爆發到達（`burst_concentration` 集中於 active window），作為到達突發壓力測試 | —（合成壓力模式） | — |

> 注意：`philly` / `ali` 是「**統計近似**」而非逐筆原始資料；數值結果反映的是這些 trace 的工作負載**特性**（job 大小分布、到達節奏、runtime 尾部），不等同在原始 production log 上的表現。要做嚴格對照時，建議改用 `load_philly()` 載入 msr-fiddle/philly-traces 的真實 `cluster_log_data.json`。
