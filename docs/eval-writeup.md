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

實作在 `services/rl_scheduler/dsac.py` 與 `services/rl_scheduler/distortion.py`，演算法與單元/行為測試見 commit `fe899ec`。本文 §3 報告 `risk_mode=mean`（risk-neutral baseline）的結果；`cvar` 的尾部對照仍待補（見 §3 末）。

## 2. Benchmark 方法

### 2.1 Simulator paired benchmark

標準 simulator benchmark 使用相同 seed 對 RDSAC 與 heuristic score 做 paired comparison，降低 trace 隨機性造成的誤差。

本次設定：

| 項目 | 值 |
|---|---|
| agent | RDSAC，`risk_mode=mean`（risk-neutral） |
| critic trunk | MLP（`--no-attention`） |
| training | 從頭訓練 150k 步，curriculum n_jobs 10→30→50 |
| checkpoint | `runs/rdsac_eval_mean/train/dsac.pt` |
| cluster | 1 node × 1 GPU |
| jobs per trace | 50 |
| trace families | `philly`, `burst`, `ali` |
| seeds | 42, 43, 44, 45, 46 |
| metric | mean JCT（主），p95 / p99 JCT（尾部） |

重現指令：

```bash
PYTHONPATH=. .venv-m11/bin/python eval/scripts/eval_dsac_placement.py \
  --total-steps 150000 --warmup-steps 2000 --n-jobs 50 \
  --n-nodes 1 --gpus-per-node 1 \
  --trace-families philly burst ali \
  --seeds 42 43 44 45 46 \
  --no-attention --curriculum --risk-mode mean --device cuda \
  --out-dir runs/rdsac_eval_mean
```

> 與舊版差異：早期 §3 數字來自一個 500k 步的 discrete-SAC checkpoint；本次是 RDSAC 從頭訓練 **150k 步**。步數較少，因此**不應**把兩組數字當作同條件直接比較。

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

RDSAC `risk_mode=mean`，150k 步，5 seeds（`Δ` 為 `(score − dsac)/score`，負值代表 RDSAC 的 JCT 較高、較差）：

| Family | RDSAC mean JCT | Score mean JCT | Δ | 95% CI | p-value | p95 JCT | p99 JCT | 結論 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| philly | 4.475 h | 2.621 h | -70.8% | [-178.4%, +36.8%] | 0.142 | 21.24 h | 44.73 h | RDSAC 較差，未顯著 |
| burst | 3.541 h | 3.541 h | -0.0% | [-118.2%, +118.2%] | 1.000 | 10.50 h | 23.04 h | 打平 |
| ali | 1.986 h | 1.383 h | -43.6% | [-141.5%, +54.3%] | 0.284 | 5.50 h | 29.68 h | RDSAC 較差，未顯著 |

解讀：

- **RDSAC(mean) 在 1×1 simulator 上仍未打贏 heuristic score**。philly、ali 方向是較差但**都未達統計顯著**（p=0.142、0.284，seed 間變異大）；burst 與 score 打平。
- 這與舊 discrete-SAC checkpoint 的結論一致：1×1 action space 的可學結構有限，而 score baseline 已把 MPS fit、短工作偏好與碎片化懲罰寫得很強。換演算法本身不會憑空在 1×1 上贏過一個已調好的啟發式。
- **訓練品質警訊**：本次 run 的 `alpha` 自動調到 clamp 上限 2.718（=e¹），代表 policy entropy 持續低於 target、temperature 一路被推到頂。這通常是 `target_entropy_ratio` / alpha clamp 需要調，或 150k 步尚未充分收斂；下一輪應先處理這點再下效能結論。
- RDSAC 相對舊版的價值在 **risk-sensitive（cvar）對尾部的處理**，而上表是 `risk_mode=mean`（risk-neutral），尚未動用 risk 機制。p95/p99 欄位先記錄下來，作為之後 cvar 對照的 baseline。

### 待補：cvar 尾部對照

本輪只完成 `risk_mode=mean`。RDSAC 的核心主張——CVaR 用犧牲一點平均換取較好的尾部——需要再跑一個同配置的 `--risk-mode cvar` run，比較 p95/p99 JCT 與 tail slowdown 是否優於上表的 mean。演算法層面已用單元 + 行為測試確認 CVaR 會把策略推向低變異動作（commit `fe899ec`），但 simulator trace 上的尾部效益尚未量測。重現：

```bash
PYTHONPATH=. .venv-m11/bin/python eval/scripts/eval_dsac_placement.py \
  --total-steps 150000 --warmup-steps 2000 --n-jobs 50 \
  --trace-families philly burst ali --seeds 42 43 44 45 46 \
  --no-attention --curriculum --risk-mode cvar --risk-beta 0.25 --device cuda \
  --out-dir runs/rdsac_eval_cvar
```

## 4. Live cluster 結果

> 以下 live 結果是 RDSAC 改寫前、用舊 discrete-SAC checkpoint 跑的。它們驗證的是 serving / Slurm placement / worker lifecycle / MPS allocation 等基礎設施路徑能否運作，與演算法版本無關，因此保留。換上 RDSAC checkpoint 的 live A/B 尚未重跑。

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

## 5. 結論

目前可誠實下的結論：

| 問題 | 結論 |
|---|---|
| DRL path 是否能在 live 上跑？ | 可以。job `136-141` 全部完成，MPS allocation 與 worker placement 有實際生效（舊 checkpoint；RDSAC live A/B 待重跑）。 |
| RDSAC(mean) 是否在標準 simulator benchmark 打贏 score？ | 還沒有。philly、ali 方向較差但**未達顯著**，burst 打平；且本次 `alpha` 觸頂，訓練尚未調好。 |
| RDSAC 的 risk-sensitive(cvar) 在尾部是否有效？ | simulator 上尚未量測。演算法已用單元 + 行為測試確認 CVaR 會把策略推向低變異動作，但 trace 上的 p95/p99 / tail slowdown 對照待補。 |
| live A/B 是否已能公平比較 DRL vs score？ | 還不能。score-only phase 暴露 worker lifecycle / Slurm completion acknowledgement 問題。 |
| 目前最穩定的上線策略 | DRL live scheduler 保持 enabled，但保留 stale snapshot、low confidence、service down 時的 heuristic/Slurm fallback。 |

工程貢獻目前比較明確的是：

1. 已有可上線的 DRL inference path，而不是只停在 notebook/simulator。
2. DRL 實作已對齊 Ma et al. RDSAC（雙分布 critic + categorical actor + risk distortion），並有單元/行為測試覆蓋；risk 機制是相對舊 discrete-SAC 的主要新能力。
3. simulator 與 live trace collector 已能支援後續 RLPD：先用 live `sacct` / normalized trace 蒐集真實 transition，再混合 simulator replay 做 fine-tuning。
4. benchmark 顯示 RDSAC(mean) 尚未優於 heuristic score，這讓後續方向更清楚：先修 alpha 自動調 / 收斂問題，再跑 cvar 尾部對照，並改善訓練資料、reward fidelity 與 live latency/worker lifecycle model。不是只宣稱「用了 DRL / risk-sensitive」就算贏。

## 6. 本次驗證指令

Simulator（RDSAC mean，本次 §3 結果）：

```bash
PYTHONPATH=. .venv-m11/bin/python eval/scripts/eval_dsac_placement.py \
  --total-steps 150000 --warmup-steps 2000 --n-jobs 50 \
  --n-nodes 1 --gpus-per-node 1 \
  --trace-families philly burst ali \
  --seeds 42 43 44 45 46 \
  --no-attention --curriculum --risk-mode mean --device cuda \
  --out-dir runs/rdsac_eval_mean
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
