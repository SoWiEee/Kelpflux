# 規格：重尾 + 高競爭 Live A/B Workload 產生器 + 尾部量測

**狀態**：設計中（未實作）。目標是在**現有 1×1 cluster**（單 RTX 4070、k3s + Slurm + MPS）上，用一條重尾、高競爭、估計不確定的 workload，評估 `score / SAC / RDSAC` 在 live 的真實差異。**不碰 sim、不等多節點硬體。**

設計理由與科學動機見 `docs/eval-writeup.md` §4.4；本文件只談**工程規格**。

---

## 0. 為什麼需要這支產生器（一句話）

先前 live A/B（§4.1–4.3）用 14 個短、確定 runtime 的 sleep job → 有 slack、沒尾部風險 → 三方**結構上必然打平**。要拆開三方，workload 必須同時具備：**高競爭**（讓排序/打包決策有意義，拆 score vs DRL）+ **重尾 runtime 且估計不確定**（讓尾部風險出現，拆 SAC vs RDSAC）+ **尾部量測**（看得見差異）。

---

## 1. 名詞與不變式

| 名詞 | 定義 |
|---|---|
| `true_runtime` | job 實際 `sleep` 的秒數（決定真實 JCT）|
| `reported_runtime` | 餵給 Slurm `--time` 與 `/decide` 的**估計**值；= `true_runtime · exp(σZ − σ²/2)`，`Z~N(0,1)` |
| `mps_req` | job 要求的 MPS slot（單 GPU 容量 = 100）|
| arm | 一個排程器設定（`score` / `SAC` / `RDSAC-mean` / `RDSAC-cvar`）跑完整條 stream 一次 |
| round | 同一條 stream 在同一 arm 的一次重放（首個 round 為 warmup，丟棄）|

**不變式（配對比較的前提）**：
- **per-job common-random**：同一 `job_id` 在**每個 arm、每個 round** 都拿到**相同** `(arrival_offset, true_runtime, reported_runtime, mps_req)`。以 `job_id` 為種子（`zlib.crc32`）抽噪，與 sim 的 CRN 對齊。
- `σ=0` ⇒ `reported == true`（退化成確定 runtime，可作對照組）。

---

## 2. Workload 產生器規格

### 2.1 介面

```
eval/scripts/live_ab_heavytail.py
  gen_workload(family, n_jobs, seed, *, sigma, compress, mps_scale,
               arrival_mode, target_max_s) -> list[LiveJob]
```

`LiveJob` 欄位：`job_id, arrival_offset_s, true_runtime_s, reported_runtime_s, mps_req, gpu_count(=1)`。

### 2.2 參數

| 參數 | 預設 | 說明 |
|---|---|---|
| `family` | — | `philly` 或 `ali`（**只此二者**，見 §4.4 理由）|
| `n_jobs` | 300 | 深佇列；遠多於先前的 14（高競爭來源之一）|
| `sigma` | 1.0 | estimate 噪音；校準到 §3.5 的 ≈1.2–1.45，預設取保守 1.0。`0` = 確定對照 |
| `compress` | auto | 時間壓縮因子，使最長 job ≈ `target_max_s` |
| `target_max_s` | 180 | 壓縮後最長 job 的目標秒數（讓整條 A/B 在可測時間內跑完）|
| `mps_scale` | auto | 縮放 `mps_req` 使 **peak 並發需求 ≈ 3–5× 100**（MPS 超賣 → 強制排隊/打包）|
| `arrival_mode` | `burst` | `burst`（短窗口內全送，佇列瞬間堆積）/ `poisson`（到達率 > 服務率，佇列穩定堆積）|
| `seed` | 42 | trace 生成種子 |

### 2.3 產生步驟

1. `jobs = sim.loader.generate_by_family(family, n_jobs=n_jobs, seed=seed)`（**只讀 loader，不改 sim**）。
2. **過濾**：`jobs = [j for j in jobs if j.gpu_count <= 1]`（單 GPU 跑不了多卡 job）。
3. **時間壓縮**（保留尾部形狀）：`true = clamp(j.runtime * compress, MIN_S, target_max_s_softcap)`；`compress` 取自 `target_max_s / max(runtime)`。**不硬截尾**——尾部用 soft cap，保留重尾相對結構。
4. **估計加噪**：`reported = true * exp(sigma*Z - sigma²/2)`，`Z` 由 `crc32(job_id)` 種子抽（per-job CRN）。
5. **MPS 超賣**：`mps_req = clamp(round(j.mps_req * mps_scale), 1, 100)`，`mps_scale` 調到 peak 並發需求 ≈ 3–5×。
6. **到達**：`burst` → `arrival_offset` 集中在前 ~10% 窗口；`poisson` → rate 設成 `> Σtrue/容量`。

### 2.4 提交（每個 job → 一個 sbatch）

```bash
sbatch --job-name=htab_<arm>_<round>_<job_id> \
       --gres=mps:<mps_req> \
       --time=<ceil(reported_runtime_s)> \
       --comment='{"job_id":..,"true":..,"reported":..,"mps":..}' \
       --wrap='sleep <true_runtime_s>'
```

- `--time` 用 **reported**（scheduler 看到的估計，會錯）；實際 `sleep` 用 **true**。排序按 reported → 排錯 → 懲罰真實尾部。
- `/decide` 收到的 `runtime_s` 也是 **reported**（與 §8.2 schema 一致）。
- `--comment` 攜帶 ground truth，供事後 join 與量測。

---

## 3. A/B 協定

```
for arm in [score, SAC, RDSAC-mean, RDSAC-cvar]:        # arm 順序每批交換
    deploy/reload checkpoint(arm)                        # 見 §3.1
    for round in [warmup, r1, r2, ...]:
        submit_stream(jobs)                              # 同一條 stream（CRN）
        wait_drain()                                     # 等全部 job 結束
        collect_sacct(arm, round)
    discard(warmup)                                      # 丟首個 round
```

- **arm 切換**：score 臂 `shadowMode=true`（boost=0）；learned 臂載對應 checkpoint。learned 臂之間切換用 §3.1 的 `/reload`（避免 pod 重啟）。
- **warmup 丟棄 + arm 順序交換**：沿用 §2.5/§4.2 教訓，消除共用單 GPU 的一次性暖機懲罰偏差。
- **paired 單位**：以 `(round, job_id)` 對齊跨 arm 的同一 job。

### 3.1 serve.py `/reload` endpoint（熱載，免 pod 重啟）

```
POST /reload  {"checkpoint": "<path>", "variant": "RDSAC-cvar"}
  → 載入新 DSAC checkpoint 到 serve 的 model slot（atomic swap）
  → 回 {"ok": true, "obs_dim":.., "n_actions":.., "use_iqn":.., "risk_mode":..}
  → 失敗（dims 不符 / 檔案缺）→ 保留舊 model、回 4xx、不中斷服務
```

切 arm = 一次 `/reload`，避免每次重啟 `rl-scheduler` pod 的冷啟動 race。

---

## 4. 尾部量測規格

每 `(arm, round)` 由 controller pod 的 `sacct` 收每 job 的 `Submit/Start/End`，算：

| 指標 | 定義 |
|---|---|
| `jct` | `End − Submit`（per job）|
| `wait` | `Start − Submit` |
| `slowdown` | `jct / true_runtime` |
| **mean_jct** | 既有主指標 |
| **p50 / p95 / p99 jct** | 尾部 |
| **cvar_jct(β)** | 最差 `β` 比例 job 的 JCT 平均（β=0.25，對齊 RDSAC-cvar 優化目標）|
| **p95 / p99 slowdown** | 尾部延展 |
| **max_jct** | 最差 straggler |
| **completed_frac** | 窗口內完成比例 |
| **abstain_rate** | 該 arm `/decide` 回 abstain 的比例（learned 臂）|

**配對統計**（vs score，以 job 為單位、或以 round 為單位 bootstrap）：

| Δ 指標 | 公式 |
|---|---|
| ΔJCT% | `(score − model)/score` |
| Δp99% | p99 JCT 相對改善 |
| ΔCVaR% | cvar_jct 相對改善（**RDSAC 的主戰場**）|
| 顯著性 | paired t-test / bootstrap 95% CI；報 p-value |

輸出：`runs/htab_<ts>/SUMMARY.md` + `metrics.json`（per arm/round 陣列）+ `jobs.csv`（per-job raw，供重算）。

---

## 5. 預期結果與判讀

| 觀察 | 判讀 |
|---|---|
| 高競爭下 score ≈ SAC ≈ RDSAC | 1×1 即使加載仍觸頂 → 決定性檢驗只能等 2-node |
| RDSAC-cvar 顯著砍低 p99/CVaR、mean 持平或小贏 | **sim σ-發現轉移到 live**（thesis 首次真環境兌現）|
| 加噪加載後 RDSAC 仍 = SAC | **sim σ-結果不轉移 live** → 同等有價值的負結果 |
| learned 臂 abstain 仍 ~高 | 高競爭未改變模型信心 → 後續實驗放寬 §8.3 gate（記錄為條件）|

天花板 caveat：單 GPU 上 score 的 SJF-ish 排序已接近最佳，DRL 上行有限，效果量可能小。本實驗是 1×1 能做的最佳評估，非決定性。

---

## 6. 實作邊界（明確「不做什麼」）

- **不改 sim**：只 `import sim.loader.generate_by_family` 讀 trace。
- **不動 production checkpoint**（`slurm-rl-scheduler:m11`）；arm 切換用 `/reload` 載指定檔。
- **不字面重放**：trace 經 gpu≤1 過濾 + 時間壓縮（已聲明範圍，見 §4.4）。
- **只用 philly / ali**；不含 burst。
- 新增檔：`eval/scripts/live_ab_heavytail.py`（產生器 + 提交 + 收集）、`serve.py` 加 `/reload`、尾部量測可重用/擴充 `sim/metrics.py` 的 summary。
