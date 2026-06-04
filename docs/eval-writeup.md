# Kelpflux Scheduler Evaluation

本文件整理目前上線規格下的 scheduler evaluation。重點不是證明 DRL 一定優於啟發式，而是清楚比較三種做法在同一套 simulator 與 live Slurm/k3s/GPU 環境中的行為：heuristic score、SAC、DSAC。

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

### 1.3 DSAC

DSAC 是 Discrete SAC。它把 SAC 的 maximum-entropy actor-critic 形式改成離散 action space，並支援 action masking。Kelpflux 目前使用 DSAC 作為 DRL scheduler/placement 的主要研究實作。

目前 DSAC 的 live path 是：

1. `rl-snapshot-agent` 定期把 Slurm pending/running/node/GPU/MPS 狀態送到 `rl-scheduler`。
2. `rl-scheduler` 用 DSAC checkpoint 對 masked action space 做推論。
3. policy 若選中 job，live scheduler 回傳 priority boost 與 placement hints。
4. hard placement controller 會把可安全介入的 placement 寫回 Slurm，使指定 job 傾向落到指定 worker/GPU/MPS slot。
5. 若 snapshot stale、模型不可用、action confidence 不足或 placement 不安全，系統 abstain，回到 heuristic/Slurm fallback。

相關實作：

| 層級 | 檔案 |
|---|---|
| DSAC agent | `services/rl_scheduler/dsac.py` |
| masked scheduling env | `sim/gym_env.py` |
| simulator training | `services/rl_scheduler/sim_train.py` |
| live inference API | `services/rl_scheduler/serve.py` |
| live daemon / placement path | `services/rl_scheduler/live_daemon.py`、`services/rl_scheduler/placement_controller.py` |
| replay buffer / RLPD support | `services/rl_scheduler/replay_buffer.py`、`scripts/collect-live-trace.py` |

## 2. Benchmark 方法

### 2.1 Simulator paired benchmark

標準 simulator benchmark 使用相同 seed 對 DSAC 與 heuristic score 做 paired comparison，降低 trace 隨機性造成的誤差。

本次設定：

| 項目 | 值 |
|---|---|
| checkpoint | `runs/eval_mlp_20260514-210824/train/dsac.pt` |
| mode | eval only, no training |
| cluster | 1 node × 1 GPU |
| jobs per trace | 50 |
| trace families | `philly`, `burst`, `ali` |
| seeds | 42, 43, 44, 45, 46 |
| metric | mean job completion time, lower is better |

重現指令：

```bash
PYTHONPATH=. .venv-m11/bin/python eval/scripts/eval_dsac_placement.py \
  --ckpt runs/eval_mlp_20260514-210824/train/dsac.pt \
  --no-train \
  --n-nodes 1 --gpus-per-node 1 \
  --n-jobs 50 \
  --trace-families philly burst ali \
  --seeds 42 43 44 45 46 \
  --out-dir /tmp/kelpflux-dsac-score-bench-standard
```

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

| Family | DSAC mean JCT | Score mean JCT | Score − DSAC | 95% CI | p-value | 結論 |
|---|---:|---:|---:|---:|---:|---|
| philly | 4.815 h | 2.621 h | -83.7% | [-118.1%, -49.4%] | 0.002 | DSAC 顯著較差 |
| burst | 5.025 h | 3.541 h | -41.9% | [-75.0%, -8.8%] | 0.025 | DSAC 顯著較差 |
| ali | 1.991 h | 1.383 h | -44.0% | [-113.4%, +25.4%] | 0.153 | DSAC 較差但未達顯著 |

解讀：目前這個 MLP DSAC checkpoint 在 1×1 simulator 上沒有打贏 heuristic score。philly 與 burst 的差距達統計顯著；ali 因 seed 間變異較大，方向仍是 DSAC 較差，但 p-value 未達 0.05。

這個結果符合目前系統狀態：1×1 action space 的可學習結構有限，而 score baseline 已經把 MPS fit、短工作偏好與碎片化懲罰寫得很強。DSAC 的價值目前主要在於提供可上線的 DRL decision path 與後續 RLPD/live trace 訓練基礎，而不是目前 checkpoint 已經優於啟發式。

## 4. Live cluster 結果

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

## 5. 結論

目前可誠實下的結論：

| 問題 | 結論 |
|---|---|
| DSAC 是否能在 live 上跑？ | 可以。job `136-141` 全部完成，MPS allocation 與 worker placement 有實際生效。 |
| DSAC checkpoint 是否在標準 simulator benchmark 打贏 score？ | 還沒有。`philly`、`burst` 顯著輸給 heuristic score，`ali` 方向也是輸但不顯著。 |
| live A/B 是否已能公平比較 DSAC vs score？ | 還不能。score-only phase 暴露 worker lifecycle / Slurm completion acknowledgement 問題。 |
| 目前最穩定的上線策略 | DSAC live scheduler 保持 enabled，但保留 stale snapshot、low confidence、service down 時的 heuristic/Slurm fallback。 |

工程貢獻目前比較明確的是：

1. 已有可上線的 DSAC inference path，而不是只停在 notebook/simulator。
2. live path 能把 queue、node、GPU/MPS 狀態轉成模型輸入，並把決策回接到 Slurm submit/placement 流程。
3. simulator 與 live trace collector 已能支援後續 RLPD：先用 live `sacct` / normalized trace 蒐集真實 transition，再混合 simulator replay 做 fine-tuning。
4. benchmark 顯示目前 DSAC checkpoint 尚未優於 heuristic score，這讓後續研究方向更清楚：不是只宣稱「用了 DRL」，而是要改善訓練資料、reward fidelity 與 live latency/worker lifecycle model。

## 6. 後續改進方向

| 優先級 | 改進 | 原因 |
|---|---|---|
| P0 | 修正 live benchmark 的 worker lifecycle guard，讓 score-only 與 DSAC phase 都能在同樣 warm/cold 條件完成 | 沒有穩定 live A/B，就無法可信比較 scheduler。 |
| P1 | 使用 `scripts/collect-live-trace.py` 長期收集 `sacct` normalized trace | DSAC 需要真實 arrival、runtime、wait、node placement 資料，而不是只依賴 synthetic simulator。 |
| P1 | RLPD fine-tuning：score/Slurm trajectory 作為 demonstration replay，DSAC online replay 作為探索資料 | 可以降低純 simulator 訓練和 live 行為之間的落差。 |
| P1 | 2×2 cluster benchmark | 目前 1×1 topology 太小，heuristic 很容易接近最佳；多 worker/GPU/MPS 才能凸顯 placement-aware policy 的價值。 |
| P2 | latency model 納入 warm/cold worker、hard placement、pod startup、Slurm completing/not responding | simulator 若沒模擬這些 live failure mode，模型會學不到真正的部署成本。 |
| P2 | residual DSAC：DSAC 學 score baseline 的修正量，而不是完全取代 score | 可以保留強 baseline，降低 DRL policy 早期不穩定造成的風險。 |

## 7. 本次驗證指令

Simulator：

```bash
PYTHONPATH=. .venv-m11/bin/python eval/scripts/eval_dsac_placement.py \
  --ckpt runs/eval_mlp_20260514-210824/train/dsac.pt \
  --no-train \
  --n-nodes 1 --gpus-per-node 1 \
  --n-jobs 50 \
  --trace-families philly burst ali \
  --seeds 42 43 44 45 46 \
  --out-dir /tmp/kelpflux-dsac-score-bench-standard
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
