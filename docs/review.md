# Kelpflux 系統審查報告（v8 對齊更新）

> **評估時間：** 2026-06-23
> **評估快照：** main @ `1d96ad2`（item-1 live A/B 完整四方 + 跨 seed 綜合 + PopArt 否決 + RLPD 三次嘗試後）
> **評估視角：** IEEE 期刊/會議審稿人 + AI Infra 專家 + ML/Model 專家
> **評估範圍：** RDSAC 演算法與評估方法學、sim-to-live 保真度、單卡/異質叢集基礎設施、研究可發表性。
>
> **本次對齊更新（相對前一版 v8 快照）：** 把 v8 寫作當下「待跑 / 進行中 / anecdote」的幾件事對齊成已定案的結果，多數新結果是**負的或否決**：(1) **item-1 已 live 驗證**——完整四方 live A/B，最初名目 +3.9% 在非飽和 regime + n 三倍化 + 配對 CRN 下翻轉成三方全顯著輸 score（−3.7~−4.6%、seed 42/43/44 一致）；(2) **full PopArt 已 3-seed 評估 → 否決**（比 P2 reward-norm 更差、變異更大），預設仍用 P2；(3) **RLPD 三次嘗試全失敗**，seed-43 base 仍為最佳 checkpoint；(4) **負結果的統計顯著性已達成**（n=246、配對 CRN、多 seed、跨 seed mean±std）；(5) **node-2 曾一次性掉線、已復原**，3080 MPS 通過 4-concurrent 真實 CUDA 測試，異質感知 live A/B 已跑完。核心判斷不變：真實 2×1 上沒有任何 DRL 臂贏 score。

---

## 0. 執行摘要

**一句話判斷：** Kelpflux 的系統工程持續成熟，但**核心負結論不變**：在真實 2×1（RTX 4070 + RTX 3080）上，**沒有任何 DRL 臂在任何可檢驗設定贏過 score 啟發式**——sim 多 seed、sleep live、真實 CUDA live、**異質感知 live A/B（item-1）四端一致**（eval §3.1/§4.1/§4.2/§4.3）。本輪三件工程（item-1 異質性曝光、PopArt、host 監控 + 3080 MPS 解鎖）都是**縮小 gap、穩定訓練、補測試平台缺口**，不是「贏過 score」的主張——而本次對齊更新後，多數結果已定案為**負或否決**。

本輪確立的事實：

1. **item-1（異質性曝光，sim + live 皆已驗證，結論為顯著負）**：先前策略**對卡型完全失明**——sim runtime 無 per-node 速度因子、obs 把 GPU one-hot 硬編成 `[1,0,0]`（一律當 4070）。現在 obs 曝露真實 gpu_type、sim cluster 模 node 速度（4070=1.0×、3080≈0.25×）。sim 配對（同異質環境裡 homo-baseline vs hetero-trained，皆 vs score）**hetero-trained 贏 10/12 格**，直接對應 §4.2「大 job 沒放對」的失敗。**現已在 live 異質叢集驗證（§4.3/§4.3.1）：上線後做了完整四方 live A/B（score/SAC/RDSAC-mean/RDSAC-cvar），最初的名目 +3.9% 在精確測量（非飽和 regime + 樣本三倍化 n=246 + 配對 CRN）下翻轉成三方全顯著輸 score——SAC −3.7%、RDSAC-mean −4.0%、RDSAC-cvar −4.6%（p 全 ≪ 0.05），seed 42/43/44 一致，§4.4 有跨 seed 綜合 mean±std。** 結論：item-1「縮小 §4.2 大幅落後」這個宣稱成立，但收斂成「一致、顯著、但小幅（~−3%）、無尾端紅利」的落後，沒贏 score。
2. **full PopArt（已 3-seed 評估 → 否決）**：曾把 P2 的「PopArt-lite」（reward 前正規化）換成 critic 內的正規 PopArt（van Hasselt 2016）——reward head 出正規化回報、running μ/σ、output-preserving 重縮放（驗證逐位元 9.5e-7）、Z_R 在所有消費點反正規化。`--use-popart` flag、寫進 checkpoint、預設關（54 測試通過）。**但 3-seed 評估反而比 P2 reward-norm 更差（−23~−48% vs −12~−33%）、變異更大 → 否決，預設關閉、仍用 P2**（eval §6）。它沒解決 auto-α 問題，替代方案是固定 α + 良好 reward_scale 或 per-arm reward 標準化。
3. **node-2 OS 升級 22.04→24.04（已完成）**：解鎖 3080 的 MPS（先前是打包缺口——driver 580 沒為 22.04 build）。3080 現通過 4-concurrent MPS 測試。**伴隨發現：MPS 先前根本沒在多工**——sleep job 不建 CUDA context 所以掩蓋了壞掉的 MPS，真 cuBLAS job 才暴露 → 現在真實-CUDA eval（§4.2）才成立。
4. **host-metrics 監控盲區補上（已完成）**：加 node-exporter（hostNetwork DaemonSet，per-node host CPU/mem/disk/net）+「Node / Host」Grafana dashboard。起因是 node-2 那次掉線前無 host 遙測——這盲區現已覆蓋（既有 DCGM/Slurm dashboard 早已 multi-node-aware）。

**狀態 caveat（歷史敘述）：** node-2 曾**一次性掉線**（疑似硬體/網路故障），現**已復原**——3080 通過 4-concurrent 真實 CUDA 測試，異質性感知的 live A/B **已經跑完**（§4.3/§4.3.1/§4.4）。item-1 應記為 **sim + live 皆已驗證、結論為顯著負**。

**三個專家視角的共同結論：** 瓶頸仍是**方法學與測試平台規模**，本輪把幾項待跑/進行中的事釘成已定案結果（多為負），但沒有翻動核心判斷。三個視角各自的最高優先改進（詳見 §2）：

| 視角 | 最高優先改進 | 為什麼擋住結論 |
|---|---|---|
| **IEEE 審稿人** | **live 負結果的統計顯著性已達成**（n=246、配對 CRN、非飽和 regime、seed 42/43/44），剩下未達標的只有 item-1 的 sim 端細節與已被否決的 PopArt；新增門檻是**多 baseline**（FCFS/SJF/packing） | 「啟發式勝出」要別人站得上去，需要比自家 score 更廣的 baseline；負結果本身的顯著性已不再是門檻 |
| **AI Infra** | **異質性感知的 live A/B 已跑完**（node-2 復原後），結論為顯著負；剩下是 **train/serve 動作落差**已關後的後續（讓 RL 訊號在 live 真正起作用、降 abstain/fallback） | live 已是真實裁判且已裁定 score 勝出；下一步價值在「讓 RL 訊號可起作用」而非再跑一次 A/B |
| **ML/Model** | **PopArt 已 3-seed 否決**，溫度穩定改走固定 α + 良好 reward_scale 或 per-arm reward 標準化；**RLPD 三次嘗試全失敗**，正確路線是把 daemon 部署進叢集收真實 transition 再做保守 offline RL | auto-α 仍是 band-aid，PopArt 這條路已證走不通；RLPD 需要真實 transition 而非 trace-replay |

---

## 1. 自 v7 以來的狀態變更（delta）

| 項目 | v7（2026-06-13） | v8（2026-06-22） |
|---|---|---|
| 測試平台拓樸 | 異質 2×1「即將上線」 | **已上線並跑出多 seed 結果**：sim σ-sweep（3 train seed）+ live placement A/B（3 train seed）+ 真實 CUDA job（§3.1/§4.1/§4.2）。核心負結論在三端固實 |
| 卡型異質性 | 策略**對卡型完全失明**（runtime 無 per-node 速度、obs GPU one-hot 硬編 `[1,0,0]`）| **item-1 已修（sim + live 皆驗證，結論顯著負）**：obs 曝露真實 gpu_type、sim 模 node 速度（4070=1.0×/3080≈0.25×）；hetero-trained 在同異質環境贏 homo-baseline 10/12 格（commit `bc69c0e`/`6db9a36`）。obs_dim 仍 166（dim-preserving）。**live 四方 A/B（§4.3.1）：最初名目 +3.9% 在非飽和 regime + n=246 + 配對 CRN 下翻成三方全顯著輸 score −3.7~−4.6%，seed 42/43/44 一致** |
| Return normalization | PopArt-lite（reward 前正規化，§3.6 P2）| **full PopArt 已 3-seed 評估 → 否決**（critic 內、van Hasselt 2016；output-preserving 重縮放驗證 9.5e-7；`--use-popart`，預設關，54 測試通過，commit `14f824f`）。**3-seed 反而比 P2 reward-norm 更差（−23~−48% vs −12~−33%）、變異更大 → 否決，預設仍用 P2**（eval §6）|
| 3080 MPS | runbook 已備、但實際**未驗證多工** | **已解鎖**：node-2 22.04→24.04 升級修好 device-plugin config-manager CrashLoop，3080 通過 4-concurrent MPS（`intergration.md §12`）。**發現：MPS 先前根本沒多工**——sleep job 無 CUDA context 掩蓋了它 |
| Host 監控 | 只有 GPU(DCGM)/Slurm/k8s/RL，**無 host-level metrics** | **已補**：node-exporter（hostNetwork DaemonSet）+「Node / Host」Grafana dashboard（per-node CPU/mem/disk/net），commit `d0f5523`。起因 node-2 掉線前無 host 遙測 |
| 真實 CUDA eval | sleep job（runtime 與 placement 無關）| **已換真實 cuBLAS**（`gpu_workload.cu`，異質 MPS 需求 25/50/75/100，§4.2）；輸贏關鍵=大 job 有沒有放對 |
| node-2 狀態 | — | 曾**一次性掉線**（疑硬體/網路故障）、**已復原**；3080 通過 4-concurrent 真實 CUDA 測試 → 異質性感知的 **live A/B 已跑完**（§4.3/§4.3.1/§4.4，結論顯著負）|

v7 的 P0/P1 推進情況：P0-2（多 seed）已做（sim+live 皆 3 train seed）；P0-3（拆 distributional vs risk）已做（§3.3，結論：分布式 critic 是有用的一半、CVaR 反扣分）；P0-4（向量化）已落地（`--num-envs N`）；P0-1（σ 校準）已做（§3.2）。本輪新增 item-1（已 live 驗證為顯著負）與 full PopArt（3-seed 後否決）對應 v7 的 I3/P2-1/P2-5（彈性/正規化）方向。

---

## 2. 三方專家審視

### 2.1 IEEE 審稿人視角：這份結果能不能過 peer review？

審稿人會先肯定**誠實的負結果**（§3.5 自我修正、§4.1 seed-robust 負結果、§4.2 大 job 失敗剖析）——這在系統論文裡是加分而非減分。R1（σ 校準）與 R2（多 seed）自 v7 起已大幅解決，但仍有未竟之處：

| # | 審稿意見 | 嚴重度 | 狀態 / 改進 |
|---|---|:---:|---|
| R1 | **注入噪音的真實性。** 核心結果依賴注入的 mean-preserving lognormal σ。 | ~~Critical~~ **已解** | **DONE**：§3.2 用生產 LightGBM 量真實 trace 的 log-殘差 std（philly 1.45 / ali 1.24），證明 §3.1 用的 σ=1.0 是保守下界、且殘差近高斯。剩餘：σ 是合成 trace 的最難預測上界，應在真實結構化 trace 上重量（v7 P0-1 的尾巴）。 |
| R2 | **訓練單 seed。** | ~~Critical~~ **已解** | **DONE**：sim σ-sweep 與 live A/B 都用 3 train seed（42/43/44）報 mean±std。**item-1 的 live 負結果現在統計顯著**（§4.3.1，n=246、配對 CRN、非飽和 regime）、跨 seed 穩健（§4.4 有綜合 mean±std：SAC −3.6±0.4、cvar −3.4±1.0、mean −2.4±1.4，全部 mean−std 仍 < 0），這正是 R2 要的東西，對 live 負結果已達標。剩下未達標的只有 item-1 的 sim 端速度建模（見 M8）與已被否決的 PopArt（3-seed 反而更差 → 否決，不再追）。 |
| R3 | **baseline 太單薄。** 只比自家 `score` 啟發式 + vanilla SAC。 | High（未解）| **在實機上補可切換的 Slurm 內建 baseline**（工程價值來自所有排程同場上線比較，而非只留在 sim）：FCFS ≈ `SchedulerType=sched/builtin` + `PriorityType=priority/basic`；multifactor ≈ `PriorityType=priority/multifactor`（chart 已設權重）；packing ≈ `SelectTypeParameters=CR_Pack_Nodes`（對 vs `CR_LLN`）——這三個都是 Slurm 內建、改設定即可切換，應在實機上跟現有 score/DRL 同場比較。**SJF 沒有 Slurm 原生 plugin**，現有 score 啟發式的 SJF-inspired factor（`f_runtime_short` + runtime_predictor）就是 live 版 SJF 近似，不是還沒做的 baseline。理想上再加一個已發表的 RL scheduler（Decima / DeepRM 改編）與近似上界讓 ΔJCT% 有尺度感。 |
| R4 | **外部效度 / sim-to-real gap。** | High（部分解）| sim-to-live gap 現有**真數據**：sim 與 live placement 對 cvar 評價相反（sim 較穩 vs live 最差）。**§4.4 新增一個強的方法學發現：sim-validation 不能預測 live**——sim 排名是 seed 43≈44 ≫ 42（42 在 sim 崩塌 cvar −133%/mean −139%），但 **live 排名是 42≈43≈44**（seed-42 落在同一窄帶 −1.3~−3.2%、甚至是所有 learned arm 裡最不爛的 RDSAC-mean −1.3%）。sim 的崩潰沒轉移到 live，**自動依 sim 選 seed 不穩健，應報全 seed 分布（含失敗 seed）**。機制：live RL 訊號很小（飽和 abstain → fallback、非飽和 placement 差異有限），把 sim 端的高變異洗掉了。明確分離兩個主張：(a) sim 內演算法比較、(b) live 的**負結果 + 韌性（fail-safe）**。Threats to Validity 要寫：訓練高變異（跨 seed 擺盪 30–90 pts）、item-1 sim 速度建模為一階純量近似（M8）、消費級 GPU 與桌面遊戲共享干擾、**sim 不預測 live 故不能用 sim-validation 選 seed**。 |
| R5 | **命名與定位。** 「RDSAC/DSAC」與 Duan 2021 DSAC 撞名。 | Medium（未解）| 換乾淨名稱（如 *discrete risk-sensitive distributional SAC, dRSAC*）或標題/摘要重度 caveat。定位主軸應是 **safe + observable ML-assisted scheduling platform 的誠實負結果 + 機制剖析**，而非「RL 贏 heuristic」。 |
| R6（新）| **item-1 的因果宣稱過強。** 「曝露異質性 → 贏 10/12」聽起來像正結果，但 12 格全是「vs score 的負值往 0 靠」、非真贏 score（只有 ali cvar 1 格 +3.4%），且含 1 個崩塌 seed。 | ~~High~~ **已 live 驗證，結論為顯著負** | **DONE**：item-1 已框成「**縮小 §4.2 的 gap**」而非「贏」，並**在 live 異質叢集驗證**（node-2 復原後，§4.3/§4.3.1）。完整四方 live A/B 在非飽和 regime + n=246 + 配對 CRN + 多 seed（42/43/44）下，最初名目 +3.9% **翻成三方全顯著輸 score（−3.7~−4.6%、全 p ≪ 1e-11）**；item-1 把 §4.2 的大幅落後收斂成「一致、顯著、但小幅（~−3%）、無尾端紅利」的落後——縮 gap 成立、贏 score 不成立。 |

**審稿人總評（模擬）：** 系統貢獻（OTel trace bridge、sim-to-live、MPS-aware、誠實的多 seed 負結果、異質性曝光的機制修正）足以撐一篇 systems track，且 R1/R2 的補強讓方法學顯著變硬。但若標題押「risk-sensitive / heterogeneity-aware RL beats heuristic GPU scheduling」，以目前證據仍會被打回——因為**沒人贏 score** 是本研究最穩健的事實。可發表的故事是「我們用嚴謹的多 seed + 校準噪音的設計，誠實地證明在此規模下啟發式仍勝出，並剖析了 DRL 退化的機制」。

### 2.2 AI Infra 專家視角：這套東西在真實基礎設施上站得住嗎？

| # | 觀察 | 嚴重度 | 狀態 / 改進 |
|---|---|:---:|---|
| I1 | **train/serve 動作落差。** sim 訓練 placement（job, node, gpu），live 過去只把 RL 選擇轉成 priority boost。 | ~~Critical~~ **已解** | **DONE**：live A/B 現走 explicit placement（`run_heavytail_ab --placement`，submit-時 `-w` 釘節點，因 Slurm 21.08 無法 post-submit 重釘）。serve 的 node 選擇真的被執行——這也是為什麼 §4.1 能量化「learned 全把負載擠到 4070（85–89%）」這個鐵證。落差已關上，留下的是**結果本身是負的**。 |
| I2 | **MPS 不是生產級共享原語 + 干擾模型未實測。** | High（部分解）| **進展**：§4.2 已用真實 cuBLAS job 取代 sleep（線性干擾假設不再是唯一證據）；且本輪發現 **MPS 先前根本沒在多工**（sleep 無 CUDA context 掩蓋之），node-2 升級後 3080 才真的 4-concurrent。仍未解：消費卡無 MIG/故障隔離要在 Threats 寫明；2-job slowdown 分布的正式實測校準仍缺。 |
| I3 | **固定拓樸的 obs/action 空間 = 反彈性。** obs_dim / n_actions 綁死 N_NODES×N_GPUS，節點 join/leave 就 checkpoint 不相容。 | High（部分解）| **進展**：item-1 讓 GPU one-hot 反映真實卡型（4070→`[1,0,0]`/3080→`[0,1,0]`），**dim-preserving（obs_dim 仍 166）**——所以策略現在「看得到」異質性而不需改維度。但這只解了「同拓樸內的異質性」；**節點數變動仍要重訓**。中期仍應走 permutation-invariant / set-based obs（attention pooling），讓單一 policy 跨拓樸。 |
| I4 | **submit-path 同步延遲。** Lua 在 `slurm_job_submit` 同步呼叫 `/decide`（timeout）。 | Medium（未解）| 量 `/decide` 的 p99 與 batch submit 風暴下的 slurmctld 影響；把 decision latency、snapshot age、abstain rate 設成有 alert 的 SLO。**本輪補的 node-exporter host metrics 是這方向的前置**——現在至少能看到 node 的 CPU/load/mem 壓力。 |
| I5 | **測試平台規模 vs 宣稱。** 消費級 4070（與 Steam 共享）+ 異質 3080。 | Medium（持續）| 這是限制不是缺陷，但要誠實標示。node-2 那次**一次性掉線（已復原）**正是消費級測試平台脆弱性的活證據——理想上最終數據挪到專用機/雲端 spot。live 數字要排除遊戲佔卡的熱節流（曾因 `wwm.exe` 佔卡使 CUDA 不可用）。 |
| I6 | **可觀測性盲區 + 單副本關鍵服務。** | Medium（部分解）| **進展**：node-exporter（hostNetwork DaemonSet）+「Node / Host」dashboard 補上 host-level CPU/mem/disk/net 盲區（先前只有 GPU/Slurm/k8s/RL）——node-2 那次掉線前的記憶體壓力/load 之類現在看得到。**仍未解**：operator 無 leader election；rl-scheduler 單副本；snapshot pusher / live_daemon 仍是 SPOF。 |

**Infra 總評：** I1（train/serve 動作落差）這個 v7 的最大障礙**本輪已關上**——live 真的執行 RL 的 placement，代價是讓負結果無所遁形（learned 全擠 4070）。可觀測性（I6）與測試平台脆弱性（I5）本輪都有進展（host metrics、3080 MPS 解鎖）；node-2 那次一次性掉線已復原，**「異質性感知 live A/B」這個最關鍵的下一步已經跑完**（§4.3/§4.3.1/§4.4），結論為三方全顯著輸 score。I3 的彈性問題只解了一半（同拓樸異質性 OK，跨拓樸仍重訓）。

### 2.3 ML / Model 專家視角：建模與方法本身是否扎實？

| # | 觀察 | 嚴重度 | 狀態 / 改進 |
|---|---|:---:|---|
| M1 | **歸因未拆乾淨。** 分布式 critic 與風險扭曲綁在一起。 | ~~Critical~~ **已解** | **DONE**：§3.3 補了 RDSAC-mean 臂（分布式但風險中立），三方 3-seed。結論：σ=1.0 下**分布式 critic 是有用的一半（SAC→mean +25 pts）、CVaR 風險扭曲反而扣分（mean→cvar −22~−34 pts）**——與設計意圖相反，但兩者都沒讓 RDSAC 贏 score。歸因問題解了，答案是「設計的風險機制沒帶來預期收益」。 |
| M2 | **score-warmup 可能造成 live 過度集中 / no-op。** | High（部分轉向）| 1×1 的 abstain 問題在 2×1 已被「過度集中到 4070」取代（learned 不再 no-op，而是學壞）。§3.6 的診斷指向**訓練退化**（過度集中 + 高變異）而非單純 clone score。warmup on/off ablation 仍值得做，但 P1 balance-shaping 已把集中度從 89%→~71%（§3.6），方向對。 |
| M3 | **auto-α 修法是 band-aid（reward_scale=20000 手調）。** | High（PopArt 路線已否決）| **PopArt 已 3-seed 評估 → 否決**：full PopArt（critic 內 return normalization，van Hasselt 2016）已實作 + 驗證 output-preserving（9.5e-7），目標本是消掉手調 reward_scale 與固定 α=0.05 的 caveat；**但 3-seed 反而比 P2 reward-norm 更差（−23~−48% vs −12~−33%）、變異更大 → 否決，預設仍用 P2**（eval §6）。PopArt 沒解決 auto-α，替代方案是**固定 α + 良好 reward_scale 或 per-arm reward 標準化**。先前的 PopArt-lite（§3.6 P2）仍是已驗證有效的那個，故 M3 本身仍 High（auto-α 的乾淨解未定）。 |
| M4 | **§3.4 共置負結果是預算混淆。** ON 有 2× 動作但同步數 → underfit。 | High（持平）| 給 ON 臂配對的訓練資訊量（按 log\|A\| 放大步數），或對 ISOLATE 稀疏 mask 做 action-embedding。否則維持「budget-confounded」標註，不下「共置無用」強結論。 |
| M5 | **無 held-out workload。** | Medium（未解）| workload split（train philly、test ali）證明泛化而非記住 trace 統計。item-1 之後更該做——「異質性曝露幫助大 job 放置」是否泛化到沒見過的 workload？ |
| M6 | **機制堆疊缺 ablation。** n-step、PER、shaping、warmup、雙頭 Z_R/Z_H 全開。 | Medium（部分解）| **進展**：§3.6 對 balance-shaping + reward-norm 做了開關對比（有效，cvar/SAC +19~+31 pts）。仍缺：PER、雙頭 Z_R/Z_H 的逐項 ablation。尤其雙頭分解增加複雜度——若單頭（熵折進 V）效果相當應簡化。 |
| M7 | **sim 算力牆。** 純 Python 離散事件。 | ~~High~~ **已解（第一階段）** | **DONE**：`sim/vec_env.py` + `sim_train --num-envs N` 多進程並行 rollout 已落地（caveat：vec path 的 score-warmup 退回 random-legal，UTD 隨 N 稀釋需同步調高 `--utd-ratio`）。多 seed（R2）與三方臂（M1）已在算力上可行並已執行。 |
| M8（新）| **item-1 的 sim 速度建模太粗。** 3080≈0.25× 是固定純量縮放，真實異質性（不同 kernel 對記憶體頻寬/SM 數的敏感度不同）非單一純量。 | Medium | 把 0.25× 標為一階近似；理想上 per-workload 量真實 4070-vs-3080 的 runtime 比，餵進 sim。否則 item-1 的「贏 10/12」帶著一個未校準的速度假設。 |
| M9（新）| **RLPD sim-to-real fine-tune 三次嘗試全失敗。** 想用 live transition 把 seed-43 prior 再 fine-tune，但三條路都退化 prior。 | High | **現況（eval §6）**：(a) trace-replay 飽和、(b) trace-replay 非飽和、(c) host-side transition-log（`kubectl exec` 包 squeue/scontrol）三次嘗試**全退化 seed-43 prior**（−19.6%/−4.5% → −97~−174%/−79%）。trace-replay 的病根是 demonstration 機制本身（拉向 sim-score + reward/scale 不匹配）；host-side daemon 透過 `kubectl exec` 每 poll ~1-2s 太慢，抓不到 job *完成* 瞬間的 completion-transition → online buffer 空 → 一樣退化。**已修一個必要 bug**（`live_daemon` 的 `value_abstain` 寫死 −1.0，seed-43 value≈−10 → 全 abstain、0 transition；改成 env `VALUE_ABSTAIN` 可調）。**seed-43 base 仍為最佳 checkpoint**。正確路線：把 daemon **部署進叢集**（低延遲 squeue/sacct）收滿真實 `(obs,act,rew)` transitions → 保守 offline RL（大 offline anchor、低 online-ratio、限步數、CQL 式保守項）。 |

**Model 總評：** v7 的兩個關鍵歸因問題本輪都有答案：M1 已拆（分布式有用、CVaR 反扣分）、M7 算力牆已解。本次對齊後：**M3 的 PopArt 路線已 3-seed 否決**（比 P2 更差），auto-α 的乾淨解仍未定；**M9 的 RLPD 三次嘗試全失敗**，正確路線是把 daemon 部署進叢集收真實 transition 做保守 offline RL；M8（item-1 速度建模校準）仍開著。整體圖像未變：機制層面沒有任何花招讓 DRL 贏 score，瓶頸是**訓練退化（過度集中 + 高變異）**——item-1 縮了 gap 但在 live 仍顯著略輸，PopArt 與 RLPD 兩條「修訓練」的支線都已撞牆（否決 / 退化）。

---

## 3. 整合改進清單（依「擋住結論的程度」重排）

> 原則：先做能讓**研究結論成立**的事（P0），再做能讓**結論轉移到 live**的事（P1），最後才是工程韌性與清理（P2，多數沿用 v6）。

**已解決，不再列入待辦（自 v7 / 本輪）：** 三方 score/SAC/RDSAC 對照 + auto-α 診斷 + fixed-α 對照（§3.1）、σ 校準到真實 predictor 殘差（§3.2，✅ v7 P0-1）、σ-sweep 補 RDSAC-mean 臂拆解 distributional/risk（§3.3，✅ v7 P0-3）、sim+live 雙端 3 train seed（§3.1/§4.1，✅ v7 P0-2）、向量化 sim（`--num-envs N`，✅ v7 P0-4）、**train/serve 動作落差**（live 真接 explicit placement，§4.1，✅ v7 P1-1 I1）、**真實 CUDA job eval**（§4.2 取代 sleep，✅ v7 P1-6 部分）、**item-1 異質性曝光（sim + live 皆驗證）**（live 四方 A/B 三方全顯著輸 score、跨 seed 穩健，§4.3/§4.3.1/§4.4，✅ 本輪）、**live 負結果的統計顯著性 + 跨 seed mean±std**（n=246、配對 CRN、非飽和 regime、seed 42/43/44，✅ 本輪，對應 R2/P0-1 的 live 端）、**node-2 復原 + 3080 MPS 解鎖**（`intergration.md §12`，✅ 本輪 I2 部分）、**host-metrics 監控**（node-exporter + dashboard，✅ 本輪 I6 部分）。

**已評估後否決 / 撞牆（不再追原路線）：** **full PopArt**（3-seed 反而比 P2 reward-norm 更差 −23~−48% vs −12~−33%、變異更大 → 否決，預設仍用 P2，eval §6）；**RLPD 三條路徑**（trace-replay 飽和/非飽和、host-side transition-log 三次全退化 seed-43 prior，eval §6）——這兩條「修訓練」支線的原方法已證走不通，替代路線見 P0-5（RLPD 改收真實 transition）與 P0-6（auto-α 改走固定 α）。

### P0 — 沒做就無法下結論（方法學門檻）

| ID | 項目 | 對應視角 | 狀態 |
|---|---|---|---|
| P0-1 | **item-1 的多訓練 seed 確認**（live）| R2, R6 | **DONE**：item-1 live A/B 已跨 seed 42/43/44 報 mean±std（§4.4），三方全顯著輸 score、跨 seed 穩健。負結果的統計顯著性已達標 |
| P0-2 | **異質性感知的 live A/B**（item-1 上線檢驗）| I1, R6 | **DONE**：node-2 復原後已跑完完整四方 live A/B（§4.3/§4.3.1），結論為三方全顯著輸 score −3.7~−4.6%（最初名目 +3.9% 在非飽和 + n=246 + 配對 CRN 下翻轉）|
| P0-3 | **PopArt 路線評估** | M3 | **DONE → 否決**：3-seed 反而比 P2 reward-norm 更差（−23~−48% vs −12~−33%）、變異更大 → 否決，預設仍用 P2。auto-α 乾淨解改走 P0-6 |
| P0-4 | **真實結構化 trace 上重量 σ** | R1 尾巴 | **OPEN**：§3.2 的 σ 是合成 trace 最難預測上界，應在 `load_philly()` 真實 trace 上重量 |
| P0-5 | **RLPD 走真實 transition（取代 trace-replay）** | M9 | **OPEN（原三路徑已撞牆）**：三次嘗試全退化 prior；正確路線是把 `live_daemon`（SHADOW_MODE）**部署進叢集**收真實 `(obs,act,rew)` transitions → 保守 offline RL（大 offline anchor、低 online-ratio、限步數、CQL 式保守項）。`value_abstain` 寫死 −1.0 的 bug 已修（env `VALUE_ABSTAIN` 可調）|
| P0-6 | **auto-α 的乾淨解（非 PopArt）** | M3 | **OPEN**：PopArt 已否決；改走固定 α + 良好 reward_scale 或 per-arm reward 標準化 |

### P1 — 讓 sim 結論能轉移到 live / 更強的對照

| ID | 項目 | 對應視角 | 狀態 / 備註 |
|---|---|---|---|
| P1-1 | **score-residual RDSAC**（`final = score + RL_delta`，bounded）| R5, M2 | **OPEN**：過度集中負結果後優先序更高；學「何時修正啟發式」而非從零學、且天然 fail-safe 回 score |
| P1-2 | **在實機補可切換的 Slurm 內建 baseline**：FCFS / multifactor / packing | R3 | **OPEN（改在 live 做）**：工程價值來自所有排程同場上線比較，故改設定即可切換的 Slurm 內建 baseline 直接做實機評估，不留在 sim——FCFS ≈ `sched/builtin`+`priority/basic`、multifactor ≈ `priority/multifactor`（chart 已設權重）、packing ≈ `CR_Pack_Nodes`（對 `CR_LLN`），三者在實機跟現有 score/DRL 同場比。**SJF 無 Slurm 原生 plugin**，現有 score 的 SJF-inspired factor（`f_runtime_short`+predictor）就是 live 版 SJF 近似，不是待做的 baseline。再加近似上界讓 ΔJCT% 有尺度感 |
| P1-3 | **共置動作在真 2-GPU 重評** | M4, §3.4 | **部分**：2×1 已上線但共置仍 budget-confounded；MPS 多工已解鎖，可在真共置下重評 |
| P1-4 | **干擾模型實測校準** | I2 | **OPEN**：真卡量 2-job slowdown 分布，取代線性假設（MPS 現已能真共置 → 可量了）|
| P1-5 | **item-1 速度建模校準** | M8 | **OPEN**：3080≈0.25× 是固定純量；per-workload 量真實 4070-vs-3080 runtime 比 |

### P2 — 工程韌性與清理（多沿用 v7，仍有效）

| ID | 項目 | 來源 | 狀態 |
|---|---|---|---|
| P2-1 | 機制 ablation（PER / shaping / 雙頭 Z_R/Z_H）| M6 | 部分（§3.6 做了 shaping/reward-norm 開關）|
| P2-2 | held-out workload split | M5 | OPEN |
| P2-3 | operator leader election；rl-scheduler 2 replicas + PDB | I6 | OPEN（snapshot pusher / live_daemon 仍 SPOF）|
| P2-4 | permutation-invariant obs（往跨拓樸 policy）| I3 | OPEN（item-1 只解同拓樸異質性，跨拓樸仍重訓）|
| P2-5 | fragmentation 維持 shadow + 加 progress penalty；NFS mount tuning | v6 P2-3/P2-5 | OPEN |
| P2-6 | submit-path chaos 實機數據；decision-latency/abstain SLO + alert | I4 | 部分（host metrics 已補，SLO/alert 仍缺）|

---

## 4. 更新後評分卡

| 面向 | v6 | v7 | v8 | 變動說明 |
|---|:---:|:---:|:---:|---|
| 工程完整度 | 4/5 | 4/5 | **4.5/5** | item-1 + 2-node 上線管線 + 真實 CUDA eval + 異質 live A/B；接近滿分，缺的是 baseline 廣度 |
| Live 安全性 | 4/5 | 4/5 | 4/5 | fail-safe 多次實證；仍缺實機 chaos 數據 |
| 可觀測性 | 5/5 | 5/5 | 5/5 | host-metrics 盲區補上，鞏固最強項 |
| 生產化韌性 | 3/5 | 3/5 | **3.5/5** | host monitoring 補上、3080 MPS 解鎖、node-2 復原；leader election / 單副本 SPOF 仍未解 |
| **DRL 研究方法學** | 2.5/5 | 3/5 | **3.75/5** | σ 校準 + 多 seed（sim+live）+ 拆 distributional/risk + 向量化全到位；**本次再 +：live 負結果統計顯著（n=246、配對 CRN）+ 跨 seed mean±std + 「sim 不預測 live」的方法學發現**。扣分轉到：缺多 baseline、item-1 速度建模未校準、PopArt/RLPD 兩條修訓練支線撞牆 |
| **可發表性（IEEE）** | — | 2.5/5 | **3/5** | R1/R2/I1/R6 已解，誠實負結果 + 機制剖析 + 跨 seed 顯著性是可發表故事；仍缺多 baseline（R3，現規劃在 live 補 Slurm 內建 baseline）|
| **sim-to-live 保真度** | — | 2/5 | **3.5/5** | I1 動作落差已關（live 真執行 placement）；真實 CUDA + 3080 MPS 解鎖；**異質 live A/B 已跑完（node-2 復原）且結論統計顯著**；扣分轉到「sim 不預測 live」（§4.4）這個保真度限制本身已被量化暴露 |

**結論：** v7 把研究問題從「測不出」推進到「有方向性答案」；v8 把那答案釘成三端一致的穩健負結論（sim 多 seed + sleep live + 真實 CUDA live：**沒人贏 score**）；**本次對齊更新再加上第四端——異質性感知的 live A/B（item-1）**：完整四方、非飽和 regime、n=246、配對 CRN、跨 seed 42/43/44，最初名目 +3.9% 翻轉成三方全顯著輸 score（−3.7~−4.6%），且 §4.4 還證明「sim-validation 不能預測 live」。v8 寫作當下還活在溫室裡的兩個改進，本次都已定案且多為負：**item-1 已 live 驗證為顯著負、full PopArt 3-seed 後否決、RLPD 三次嘗試全失敗**。所以接下來最該投資的 **P0** 已從「補 seed / 跑 live」轉成「補多 baseline（在實機切換 Slurm 內建）、把 RLPD 改成部署進叢集收真實 transition、auto-α 改走非 PopArt 的乾淨解」。這份研究的定位已是「我們有一個被多 seed 統計支撐、且在 live 異質叢集驗證過的負結論 + 一個可觀測的 fail-safe 平台」。**切記：item-1 是縮 gap 的工程，不是贏過 score 的主張——「真實 2×1 上沒有任何 DRL 臂贏 score」目前不成立任何反例，且是本研究最穩健的事實。本次多數新結果是負的或否決，不是正面進展。**

---

## 5. 跨領域異議網絡

### 5.1 跨領域盲點

1. **「贏 score」這個勝負判準從頭到尾沒被質疑過，它本身可能是這份研究最大的未檢驗假設。** 全文（含 §2 三位審稿人）都把「DRL 有沒有贏過啟發式」當成隱含的成敗線，所有工程都繞著它轉。但沒有人停下來問：對一個跑在兩張消費級卡（其中一張你還要拿去打遊戲）上的 2×1 叢集，「贏過一個已經調得不錯的啟發式」是不是一個結構上幾乎不可能、且就算贏了也沒人會在乎幅度的目標？當勝負線本身可疑時，繞著它做幾個月工程，得到的精度再高也是錨在錯的問題上。真正該被擺上桌的是判準本身，而不是判準下的小數點。

2. **「把負結果做到統計顯著」這件事，背後到底是科學需求還是沉沒成本，文件沒有區分，而這個區分決定了它值不值得。** §4.3 到 §4.3.1，動用三個正交手段（非飽和降噪、樣本三倍化、多 seed）只為了把一個名目 +3.9% 釘成顯著的 −3.3%，這在科學上是嚴謹的；但「我已經投了幾個月，所以要把它做到無懈可擊」與「這個 −3.3% 的顯著性會改變任何人的決策」是兩件事。前者是承諾升級（commitment escalation），後者才是科學價值。一份釘得極死的 −3% 負結果，和一份釘得馬虎的 −3% 負結果，對讀者的行動影響可能完全一樣：都是「別在這個規模用 DRL」。值得追問的是，顯著性的最後幾個百分點，是為了知識，還是為了讓自己甘心收手。

3. **機會成本與止損線從未被命名。** 文件密集記錄「做了什麼、解了什麼、還欠什麼」，卻沒有任何一處寫下「在什麼條件下我會停」。RLPD 三次嘗試全失敗、node-2 掉線、核心負結論反覆出現「不變」，這些在技術視角裡是「待解的 OPEN 項」，但在資源配置視角裡是訊號：你正在為一個邊際報酬遞減的問題持續投入，而沒有預設的退出條件。沒有止損線不代表該停，但「從沒想過止損線長什麼樣」本身是盲點。一個健康的研究計畫應該能回答「怎樣算輸到該收」，即使答案是「還沒到」。

4. **受眾是誰，決定了「做完」長什麼樣，而這個問題沒被回答。** 如果受眾是口試委員，那「做完」可能是「一個完整、自洽、能在 30 分鐘內講清楚並守住的故事」，此時負結果加機制剖析已經接近完成，再投入是邊際裝飾。如果受眾是未來雇主，那「做完」是「一個能展示工程深度與誠實的作品集條目」，此時平台與管線比結論更值錢。如果受眾是你自己（為了搞懂 ML-for-systems 到底行不行），那永遠沒有「做完」，只有「學夠了沒」。這三個受眾指向完全不同的收尾方式，而文件沒有挑明它在替誰寫，於是所有改進清單都缺一個對焦的錨。

5. **這份研究最有價值的產出，可能不是任何 checkpoint，而是「一個誠實、可重現的領域級負結果 + 一個異質 MPS-aware 的可觀測平台」，但全文的敘事重心押在前者最弱的地方。** ML-for-systems 這個領域充斥著「我們的 RL 贏了 baseline」的正結果論文，而能被嚴謹重現的負結果稀缺到反而是公共財：它幫整個領域省下重複踩坑的成本。十年後真正會被引用、被記得的，大概率是「在小規模異質叢集上，校準噪音 + 多 seed 證明啟發式仍勝出」這個可被站上去的結論，以及那套 OTel-bridge / sim-to-live / fail-safe 平台，而不是 seed-43 的權重檔。弔詭的是，文件花最多力氣的（讓 DRL 縮小落後）恰好是最不會留下的部分。

6. **「過程本身是否還是你想做的事」這個問題，被工程動能蓋過去了。** 寫一個 GPU 開關腳本好讓出卡來打遊戲，這個細節透露了一件技術視角完全不會記錄的事：你和這個專案的關係已經帶著張力。當一個研究計畫需要你在「跑實驗」和「過生活」之間排程同一張顯卡時，繼續與否就不只是科學判斷，也是「我還想不想每天做這個」的判斷。這不是軟弱，是資源（包含你的注意力與熱情）配置的一部分。技術審稿人不會問你還愛不愛這件事，但這個答案實際上比任何 P0 項都更能預測這份研究的結局。

7. **「繼續補強想贏」與「承認輸了收尾」之外，至少還有兩個沒被列出的選項。** 文件的隱含選擇空間是二元的：要嘛把 P0 做完繼續追平/追贏，要嘛承認負結果收掉。但局外人會看到第三條路（把負結果本身當頭條：不再試圖贏，而是把「啟發式為何在此規模難被 DRL 超越」做成一個機制完整、baseline 豐富、可重現的研究主張，這反而需要補的是 §2 的 R3 baseline 而非追平），以及第四條路（把這套異質 MPS-aware、可觀測、fail-safe 的平台從「驗證 DRL 排程」這個它一直輸的問題，轉去解一個 RL 真有結構性勝算的問題：例如不是和調好的啟發式比 JCT，而是去做啟發式根本沒在處理的事，如多目標權衡、線上漂移適應、或人類偏好對齊的排程）。把選項從二元擴成四元，往往比在二元裡糾結更接近出路。

8. **負結果的「韌性 / fail-safe」敘事，可能在無意間把一個設計上的勝利講成了安慰獎。** 全文反覆強調 fail-safe 回退 score 經多次實證，這在工程上是真功勞。但從局外人角度，這裡藏著一個未被點破的張力：一個「只要表現不好就自動退回啟發式」的系統，它的安全性恰恰來自於它不真正信任自己的 RL 決策。換句話說，這套架構的穩健，部分是建立在「預期 RL 會失敗」之上的。這不是缺陷，但它意味著「平台很安全」和「RL 有價值」這兩個賣點之間有內在拉扯：越強調前者，越暗示後者還沒兌現。值得想清楚你要賣的是哪一個。

### 5.2 觀點對撞（不調和）

**對撞一：止損/轉向 vs 先把現有故事出貨 vs 重構整個問題框架。**
一種聲音說：邊際報酬已經遞減（RLPD 三連敗、live 反覆 −3%、卡還要分時打遊戲），該畫止損線，把平台轉去一個 RL 有勝算的問題。另一種聲音針鋒相對：現在收掉等於把幾個月變成沒有交付物，最務實的是先把手上這個「誠實負結果 + 可觀測平台」包成一個能守住的完整成品出貨（口試 / 作品集 / 一篇 systems 短文），出貨之後再談轉向。第三種聲音說兩者都錯：問題不在繼續或收尾，而在框架本身選錯了，即「DRL 對打調好的啟發式」這個擂台，無論你站著還是離場都是輸，真正該做的是換一個 RL 不必和啟發式正面對撞的問題重新立題。這三者無法調和，因為它們對「沉沒的幾個月」估值不同：第一種視為已沉沒、應忽略；第二種視為待回收、應變現；第三種視為學費、買到的是「別再選這種擂台」的教訓。

**對撞二：追求負結果的顯著性，是沉沒成本的合理化，還是正當且稀缺的科學貢獻。**
一邊：§4.3.1 那種「動用三個手段把 +3.9% 釘成顯著 −3.3%」是教科書級的承諾升級：當一個結果需要你這麼用力才能釘死，市場（讀者的注意力）通常已經用沉默告訴你它不重要了，再精修是對自己交代，不是對知識交代。另一邊：這恰恰相反，是在做一件多數人偷懶不做、因而稀缺的事，即把負結果做到可重現、統計顯著、跨 seed 穩健，正是 ML-for-systems 再現性危機最缺的公共財；嫌它「不重要」的人，用的是「正結果才算貢獻」的偏見尺。這組對撞無解，因為它觸到一個更深的分歧：科學的價值，是由它改變了多少人的行動來定義，還是由它本身的嚴謹與誠實來定義。

**對撞三：受眾是口試委員（一份要守得住的論文），還是這是領域級的公共知識（一個要被別人站上去的結論）。**
若受眾是口試委員，最優策略是收斂、自洽、可防禦：別再開新戰線（PopArt 已否決就讓它否決、RLPD 別再試第四次），把現有負結果與機制剖析打磨成一個 30 分鐘講得完、問不倒的故事，多餘的 baseline 和 live 重跑都是風險而非加分。若這是公共知識，最優策略恰好相反：要開戰線，即補滿 FCFS/SJF/packing/已發表 RL baseline（否則「啟發式勝出」這個結論別人沒法站上去，因為你只比了自家 score），要報全 seed 分布含失敗 seed，要把平台與資料開源。前者要你關門收斂，後者要你開門擴張；前者的多餘工作是後者的必要工作。你不可能同時最佳化「守得住」和「站得上去」，因為一個要的是封閉，一個要的是暴露。

