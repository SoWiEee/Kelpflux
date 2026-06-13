# Kelpflux 系統審查報告（v7）

> **評估時間：** 2026-06-13
> **評估快照：** main @ `c26b268`（A/B sim-fidelity 實驗併入後）
> **評估視角：** IEEE 期刊/會議審稿人 + AI Infra 專家 + ML/Model 專家
> **評估範圍：** RDSAC 演算法與評估方法學、sim-to-live 保真度、單卡/異質叢集基礎設施、研究可發表性。
> **與 v6 的關係：** v6（HPC/SRE/ML-systems 視角）對「平台工程完整度」的判斷大致仍成立；v7 不重複那些，改用三個更嚴格的專家視角重審 **研究主張的可辯護性**，並把本輪 sim 隨機性消融（eval §4.4–4.5）的新證據納入。

---

## 0. 執行摘要

**一句話判斷：** Kelpflux 的系統工程已經成熟（v6 已肯定），本輪最大的進展是**把「分布式/風險機制到底有沒有用」這個核心研究問題從「測不出來」推進到「測得出來、且有方向性結論」**——但這個結論目前**只在模擬器內、單一訓練 seed、人工注入的噪音下成立**，距離可發表/可上線還有三道關卡（噪音的真實性、訓練 seed 的統計顯著性、sim-to-live 的動作落差）。

本輪確立的事實（eval §4.3–4.5）：

1. **確定性 1×1 sim 下三者打平**，根因是**兩個一起**：oracle runtime（零不確定性 → CVaR≈mean，風險機制結構性閒置）+ auto-α 控制器 railing 壓垮策略。
2. **注入 runtime 不確定性後，RDSAC 隨 σ 單調拉開對 SAC 的差距**（−73 → +47 → +196 pts），且 **fixed-α 對照排除 α 假象**（仍 +90～+143 pts），σ=1.0 時甚至**贏過 score** 並把 p99 尾部壓低 5–9×。
3. **共置動作（PACK/ISOLATE）在 1×1 是負結果**——動作空間加倍在相同預算下 underfit，且單卡隔離=閒置；其價值需 ≥2 GPU。

**三個專家視角的共同結論：** 目前的瓶頸已經**不是工程，而是方法學與測試平台規模**。三個視角各自指出的最高優先改進（詳見 §2）：

| 視角 | 最高優先改進 | 為什麼擋住結論 |
|---|---|---|
| **IEEE 審稿人** | 把 σ **校準到 runtime predictor 的真實殘差分布**，並補**多訓練 seed**（≥3–5）| 否則「注入噪音 → 抗噪方法贏」近乎套套邏輯；單 seed 的 per-family 數字審稿人不會接受 |
| **AI Infra** | 修正 **train/serve 動作落差**（sim 學 placement，live 只套 priority boost）| 否則 sim 結論無法外推到 live；live 永遠只能 demo fallback |
| **ML/Model** | 在 σ-sweep 補 **RDSAC-mean 臂**，拆開「分布式 critic」與「風險扭曲」兩個貢獻 | 目前 SAC vs RDSAC-cvar 把兩件事混在一起，無法歸因 |

---

## 1. 自 v6 以來的狀態變更（delta）

| 項目 | v6（2026-05-31） | v7（2026-06-13） |
|---|---|---|
| 演算法命名 | 「DSAC」 | **RDSAC**：雙頭 IQN 分布式 critic（Z_R/Z_H）+ categorical actor + 風險扭曲；`use_iqn` flag 切換 vanilla SAC ↔ RDSAC。**注意命名陷阱**：這不是 Duan et al. 2021 的連續控制 DSAC，是 Christodoulou 2019 + Dabney 2018 + 風險變體的自組版本 |
| 三方對照 | 缺 | 已有 score / SAC / RDSAC 三方（eval §4.3），靠單一 flag 切換 |
| auto-α 假象 | 未診斷 | 已定位並用 fixed-α 對照拆解（§4.3.1）——SAC 墊底大半是共用 α 控制器 railing |
| 評估種子數 | 5-seed | eval 用 30-seed paired；**但訓練仍單 seed**（最大方法學缺口）|
| 不確定性消融 | 無 | **新增 §4.4**：opt-in runtime σ + 干擾模型，證明風險機制在有尾部時有用 |
| 共置動作 | 無 | **新增 §4.5**：PACK/ISOLATE，1×1 負結果，待 2-node |
| 第二節點 | 「待建 2×2」 | RTX 3080（Ubuntu 24.04）即將加入 → 2-node **異質 2×1**；`docs/intergration.md` 已備 runbook（並標出 3080 在 codebase 完全缺席的異質性缺口）|

v6 列的 P1-1（建 2-node 環境）正在硬體層發生；P1-2（eval matrix）部分由 §4.4 隨機性消融推進；P1-3（score-residual RDSAC）仍未做，且 §4.5 的負結果反而提高了它的優先序。

---

## 2. 三方專家審視

### 2.1 IEEE 審稿人視角：這份結果能不能過 peer review？

審稿人會先肯定**誠實的負結果**（§4.3.1 自我修正、§4.5 負結果）——這在系統論文裡是加分而非減分。但會在以下五點要求 major revision：

| # | 審稿意見 | 嚴重度 | 改進 |
|---|---|:---:|---|
| R1 | **注入噪音的真實性未經校準。** 核心正結果是「在人工 mean-preserving lognormal σ 下，風險敏感法贏」。σ=0.5/1.0 從何而來？若噪音是任意選的，這個結果近乎套套邏輯（加風險 → 抗風險法贏）。 | **Critical** | 你們**已經有 LightGBM runtime predictor**——量它在真實 trace 上的 log-residual 分布，用**那個** σ（與形狀）當作 sim 噪音。把「σ 來自實測預測誤差」寫進方法學，這個結果才站得住。 |
| R2 | **訓練單 seed。** eval 用 30 種子是評估隨機性，但每個 (algo, σ) 只訓練**一次**。doc 自己承認 §4.3.1 細排名是單訓練 seed 雜訊。per-family 的 +52/+189/+333 不能只報點值。 | **Critical** | 每個 cell 至少 3–5 個訓練 seed，報 mean±std 或 IQR；對「RDSAC−SAC gap 隨 σ 單調」做跨 seed 的顯著性。沒有這個，所有 §4.4/§4.5 的數字都是 anecdote。 |
| R3 | **baseline 太單薄。** 只比自家 `score` 啟發式 + vanilla SAC。 | High | 補：FCFS、SJF（你有 oracle runtime，SJF 是強 baseline）、Tetris/packing 啟發式；理想上一個已發表的 RL scheduler（Decima / DeepRM 改編）。再加一個近似上界（clairvoyant SJF 或離線排程 LP）讓 ΔJCT% 有尺度感。 |
| R4 | **外部效度 / sim-to-real gap 未量化。** 全部正結果在 sim；live 1×1 全數打平（模型 abstain ~90–100%）。 | High | 明確分離兩個主張：(a) sim 內的演算法比較（需 R1/R2 才成立）、(b) live 的**韌性**（fail-safe，不是效能）。寫一個 Threats to Validity：單卡天花板、注入噪音、單訓練 seed、僅啟發式 baseline、消費級 GPU 與桌面遊戲共享造成的熱/競爭干擾。 |
| R5 | **命名與定位。** 「RDSAC/DSAC」與已發表的 Duan 2021 DSAC 撞名，審稿人第一眼就會混淆。 | Medium | 換一個乾淨名稱（如 *discrete risk-sensitive distributional SAC, dRSAC*）或在標題/摘要就重度 caveat。貢獻定位建議照 v6 §6：主軸是 **safe + observable ML-assisted scheduling platform**，RL 超越 baseline 是「在有尾部風險時」的條件式加分，不是無條件主張。 |

**審稿人總評（模擬）：** 系統貢獻（OTel trace bridge、sim-to-live、MPS-aware、誠實的消融）足以撐一篇 systems track。但若標題押「risk-sensitive RL beats heuristic GPU scheduling」，以目前證據會被打回——因為它在**校準噪音 + 多 seed + 多 baseline** 下尚未驗證。R1+R2 是過關門檻。

### 2.2 AI Infra 專家視角：這套東西在真實基礎設施上站得住嗎？

| # | 觀察 | 嚴重度 | 改進 |
|---|---|:---:|---|
| I1 | **train/serve 動作落差。** sim 訓練的是 placement policy（job, node, gpu），但 live 的 Lua hook 只把 RL 的選擇轉成 **priority boost**——Slurm 仍做真正的 allocation，`node_j/gpu_k` 在 live 並未被強制執行。等於訓練一個放置策略、上線只用了它的「選哪個 job」那一半。 | **Critical** | 兩條路擇一：(a) 把 live 真正接上 explicit placement（`live_daemon` 已有 `srun` 明確放置的雛形，但預設 SHADOW），讓 serve 的 placement 真的被執行；或 (b) 把 sim 的 action 改成只學 priority/job-selection，與 live 對齊。現在的不一致讓任何 sim 結論都無法宣稱能轉移到 live。 |
| I2 | **MPS 不是生產級共享原語。** 無記憶體/故障隔離：一個 job OOM 可拖垮共置者。§4.4 的干擾模型是 `1 + k·factor` 線性近似，真實干擾取決於 kernel overlap、記憶體頻寬、L2 競爭，可能是非線性甚至災難式。 | High | 消費卡無 MIG，可接受用 MPS demo，但要在 Threats 寫明；干擾模型至少做一次**實測校準**（在真卡上跑 2 個 compute-bound job 量 slowdown 分布），別只用線性假設。生產敘事需提 MIG/時間片的取捨。 |
| I3 | **固定拓樸的 obs/action 空間 = 反彈性。** `gym_env` 的 obs_dim / n_actions 綁死 N_NODES×N_GPUS，節點 join/leave 就 checkpoint 不相容、要從零重訓。對「cloud-native 彈性叢集」的宣稱是根本性矛盾——這也是為什麼加一台 3080 就要重訓。 | High | 中期應走 **permutation-invariant / 可變長度** 的 obs（attention trunk 已是 set-based，往「節點數無關」的編碼推進），讓單一 policy 跨拓樸。否則每次擴縮都重訓，無法營運。 |
| I4 | **submit-path 同步延遲。** Lua 在 `slurm_job_submit` 同步呼叫 `/decide`（150ms timeout）。負載高時這是 submit 關鍵路徑。 | Medium | 量 `/decide` 的 p99 與在 batch submit 風暴下的 slurmctld 影響；serve 確保 CPU 推論 + warm pool（memory 已記 `min_replicas=1`）。把 decision latency、snapshot age、abstain rate 設成有 alert 的 SLO。 |
| I5 | **測試平台規模 vs 宣稱。** 整個「叢集」是一張與 Steam 遊戲共享的消費級 RTX 4070 上的 k3s 單節點；第二台是異質消費卡 3080。 | Medium | 這不是缺陷、是限制——但要在論文/報告誠實標示，且 live 數字要排除遊戲佔卡造成的熱節流/競爭（本輪訓練就曾因 `wwm.exe` 佔卡使 CUDA 不可用）。理想上把實驗挪到專用機或雲端 spot GPU 做最終數據。 |
| I6 | **單副本關鍵服務 + operator 無 leader election**（v6 已記，仍未解）。 | Medium | 維持 v6 建議：operator active-passive leader election；rl-scheduler 可 2 replicas + PDB。fail-safe 讓 submit 不壞，但 snapshot pusher / live_daemon 仍是 SPOF。 |

**Infra 總評：** 可觀測性與 fail-safe 設計是真強項（v6 已肯定）。但 **I1（train/serve 動作落差）是把 sim 成果接到 live 的根本障礙**，沒解掉，live 永遠只能展示「壞 checkpoint 也不會讓 JCT 變差」，無法展示「RL 讓 JCT 變好」。I3 則決定這套東西能不能叫「彈性叢集」。

### 2.3 ML / Model 專家視角：建模與方法本身是否扎實？

| # | 觀察 | 嚴重度 | 改進 |
|---|---|:---:|---|
| M1 | **歸因未拆乾淨。** σ-sweep 比的是 SAC（scalar）vs RDSAC-**cvar**（分布式 + 風險扭曲），把「分布式 critic」與「風險扭曲」兩個貢獻綁在一起。 | **Critical** | 在 σ-sweep 補 **RDSAC-mean 臂**（分布式但風險中立）。三方 SAC / RDSAC-mean / RDSAC-cvar 才能回答「贏是因為 distributional critic 還是因為 CVaR」。這是最便宜、最高價值的下一個實驗。 |
| M2 | **score-warmup 可能是 live abstain ~90% 的元兇。** 訓練前期用 score 排程器產生種子 transition，模型可能直接 clone 了 score（1×1 近最優）→ 學不到偏離 score 的放置 → 上線就 no-op。 | High | 做 warmup on/off 的 ablation；或改成「衰減式」warmup（前期高、後期關）。若關掉 warmup 後 live abstain 率下降，這就是 1×1 live 全打平的真正機制（而非「策略本來就該 no-op」）。 |
| M3 | **auto-α 修法是 band-aid。** 需要 reward_scale=20000 + 放寬 clamp 才不 railing，說明根因是 return 尺度 vs entropy 項失衡；discrete-SAC 的 target-entropy 啟發式本就 finicky。 | High | 用 **return normalization（PopArt / reward 標準化）**讓單一 α 控制器跨 SAC/RDSAC 都穩，不必 per-algorithm 調 reward_scale。這也讓 §4.3.1 必須釘 α 的 caveat 消失。 |
| M4 | **§4.5 共置負結果是預算混淆。** ON 有 2× 動作但同 100k 步 → underfit。結論其實是「相同預算下大動作空間學不好」，不是「共置沒用」。 | High | 給 ON 臂**配對的訓練資訊量**（按 log|A| 放大步數，或對 ISOLATE 的稀疏 mask 做 action-embedding / factorized policy）。否則 §4.5 應明確標為「budget-confounded」，不能下「共置無用」的強結論（doc 已部分標註，可再強化）。 |
| M5 | **無 held-out workload。** 在 philly/burst/ali 混合訓練、又在同三族評估。 | Medium | 做 workload split（train philly+burst、test ali）證明泛化而非記住 trace 統計。對「能應付沒見過的 workload」的宣稱是必要的。 |
| M6 | **機制堆疊缺 ablation。** n-step、PER、potential shaping、score warmup、雙頭 Z_R/Z_H 全開，沒有逐項 ablation。 | Medium | 至少對 PER、potential shaping、雙頭分解各做一次開關。尤其**雙頭 Z_R/Z_H**（RDSAC 特有的熵回報分離）增加複雜度——若單頭分布式 critic（熵折進 V）效果相當，應簡化。 |
| M7 | **sim 是純 Python 離散事件、~10 steps/s = 多 seed 研究的算力牆。** 本輪一個 σ 區塊就要 ~4.6h。 | High（間接擋住 M1/R2）| 向量化 / 編譯 sim（numba、或重寫熱路徑），把吞吐拉到 10²–10³ steps/s，多訓練 seed（R2）與三方臂（M1）才在算力上可行。**這是解開 R2/M1 的前置條件。** |

**Model 總評：** RDSAC 的實作對齊參考文獻、有測試覆蓋（本輪 +9 測試），σ 消融的方向性結論在理論上合理。但 **M1（拆 distributional vs risk）與 M2（warmup 是否造成 abstain）是兩個最關鍵、最便宜的待答問題**，而 M7（算力牆）是讓 R2/M1 變得可行的前置工程。

---

## 3. 整合改進清單（依「擋住結論的程度」重排）

> 原則：先做能讓**研究結論成立**的事（P0），再做能讓**結論轉移到 live**的事（P1），最後才是工程韌性與清理（P2，多數沿用 v6）。

**已解決，不再列入待辦（自 v6 / 本輪）：** 三方 score/SAC/RDSAC 對照（§4.3）、auto-α 診斷 + fixed-α 受控對照（§4.3.1）、30-seed paired eval、隨機性消融能力與 σ-sweep（§4.4）、共置動作能力與其 1×1 負結果（§4.5）、2-node 加入 runbook（`docs/intergration.md`）、`chart/values-2x2.yaml`、runtime predictor 測試（`services/runtime_predictor/tests/` 已含 feature/cold-start/retrain 三項，對應 v6 P2-2）、7 個 CI workflow（v6 C1/S2）。

### P0 — 沒做就無法下結論（方法學門檻）

| ID | 項目 | 對應視角 | 產出 |
|---|---|---|---|
| P0-1 | **σ 校準到真實 predictor 殘差** | R1 | 量 LightGBM 的 log-residual 分布，σ 用實測值；方法學寫明 |
| P0-2 | **多訓練 seed（≥3–5）重跑 §4.4/§4.5 關鍵點** | R2, M4 | 每 cell mean±std；gap 單調性的跨 seed 顯著性 |
| P0-3 | **σ-sweep 補 RDSAC-mean 臂** | M1 | 三方拆解 distributional vs risk 貢獻 |
| P0-4 | **向量化 / 加速 sim**（前置工程）| M7 | steps/s ↑ 一個數量級，讓 P0-2/P0-3 可行 |

### P1 — 讓 sim 結論能轉移到 live / 更強的對照

| ID | 項目 | 對應視角 | 備註 |
|---|---|---|---|
| P1-1 | **修 train/serve 動作落差** | I1 | 二選一：live 真接 explicit placement，或 sim 改學 priority/selection |
| P1-2 | **score-residual RDSAC**（`final = score + RL_delta`，bounded）| v6 P1-3, R5 | §4.5 負結果後優先序升高；學「何時修正啟發式」而非從零學 |
| P1-3 | **warmup on/off ablation**（驗證 live abstain 成因）| M2 | 可能直接解釋 1×1 live 全打平 |
| P1-4 | **補強 baseline**：FCFS / SJF / packing / 近似上界 | R3 | ΔJCT% 才有尺度感 |
| P1-5 | **2-node（4070+3080）異質實驗** | I3, §4.5 | 補 `rtx3080` 進 `GPU_TYPES` / `_gpu_type_to_vram`（10GB）；共置動作在真 2-GPU 重評 |
| P1-6 | **干擾模型實測校準** | I2 | 真卡量 2-job slowdown 分布，取代線性假設 |

### P2 — 工程韌性與清理（多沿用 v6，仍有效）

| ID | 項目 | 來源 |
|---|---|---|
| P2-1 | return normalization（PopArt）取代 reward_scale 手調 | M3 |
| P2-2 | 機制 ablation（PER / shaping / 雙頭 Z_R/Z_H）| M6 |
| P2-3 | held-out workload split | M5 |
| P2-4 | operator leader election；rl-scheduler 2 replicas + PDB | v6 P1-5/P2-1, I6 |
| P2-5 | permutation-invariant obs（往跨拓樸 policy）| I3 |
| P2-6 | fragmentation 維持 shadow + 加 progress penalty；NFS mount tuning | v6 P2-3/P2-5 |
| P2-7 | submit-path chaos 實機數據；decision-latency/abstain SLO + alert | v6 P1-4, I4 |

---

## 4. 更新後評分卡

| 面向 | v6 | v7 | 變動說明 |
|---|:---:|:---:|---|
| 工程完整度 | 4/5 | 4/5 | 維持；A/B 工具 + 測試再增 |
| Live 安全性 | 4/5 | 4/5 | fail-safe 已多次實證；仍缺實機 chaos 數據 |
| 可觀測性 | 5/5 | 5/5 | 最強項，未變 |
| 生產化韌性 | 3/5 | 3/5 | leader election / 單副本仍未解 |
| **DRL 研究方法學** | 2.5/5 | **3/5** | 隨機性消融 + fixed-α 對照是實質方法學進步；但單訓練 seed、未校準噪音、未拆 distributional/risk 仍壓住分數 |
| **可發表性（IEEE）** | — | **2.5/5** | 系統貢獻夠；演算法主張需 P0-1/P0-2/P0-3 才過門檻 |
| **sim-to-live 保真度** | — | **2/5** | I1 動作落差是最大未解項 |

**結論：** 本輪把核心研究問題從「測不出」推進到「有方向性答案」，是真進展。但三個專家視角一致指向同一個瓶頸——**結論目前活在「模擬器 + 單訓練 seed + 人工噪音」的三重溫室裡**。接下來最該投資的不是更多功能，而是 **P0 四項**（校準噪音、多 seed、拆 distributional/risk、加速 sim）把溫室拆掉；其次是 **P1-1** 把 sim 與 live 的動作對齊。把這些做掉，這份研究才從「我們做了一個能跑的平台」升級成「我們有一個站得住的結論」。
