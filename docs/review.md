# Kelpflux 系統審查報告

> **評估時間：** 2026-06-23
> **評估快照：** main @ `1d96ad2`（工作目錄現況：item-1 live A/B 完整四方 + 跨 seed 綜合 + PopArt 否決 + RLPD 三次嘗試後）
> **評估範圍：** RDSAC 演算法與評估方法學、sim-to-live 保真度、單卡/異質叢集基礎設施、研究可發表性。
> **評估面向：** IEEE 審稿人、AI Infra 工程師、ML 專家、跨領域。

## 執行摘要

**核心判斷（最穩健的事實）：** 在真實 2×1（RTX 4070 + RTX 3080）上，**沒有任何 DRL 臂在任何可檢驗設定贏過 score 啟發式**——sim 多 seed、sleep live、真實 CUDA live、異質感知 live A/B（item-1）**四端一致**（eval §3.1/§4.1/§4.2/§4.3）。

item-1 異質性曝光把 §4.2 的大幅落後收斂成「**一致、顯著、但小幅（~−3%）、無尾端紅利**」的落後：完整四方 live A/B（score/SAC/RDSAC-mean/RDSAC-cvar）在非飽和 regime + 樣本三倍化 n=246 + 配對 CRN 下，最初的名目 +3.9% 翻轉成三方全顯著輸 score——SAC −3.7%、RDSAC-mean −4.0%、RDSAC-cvar −4.6%（p 全 ≪ 0.05），seed 42/43/44 一致（§4.3.1/§4.4）。「縮小 gap」成立，「贏 score」不成立。

**更大的脈絡（§4.5,新）：** 把 score 與 **Slurm 原生 FCFS/multifactor/packing** 實機對照（3 seed + run-position 漂移對照),**四個排程器全部統計打平**（ΔJCT% vs score 全在 ±0.4~1.7±std、跨 0）。所以不只 DRL 贏不了 score——**連 score 自己對 trivial baseline 都沒優勢**;在此規模整個排程策略空間是平的,瓶頸不在挑排程器/演算法花招,而在叢集規模與工作負載結構本身。

研究的真正定位是「**一個被多 seed 統計支撐、且在 live 異質叢集驗證過的負結論 + 一個可觀測的 fail-safe 平台**」，而非「RL 贏 heuristic」。剩餘的改進多屬「讓結論更站得住（多 baseline、Threats）」與「修訓練的兩條支線已撞牆（PopArt 否決、RLPD 退化）後的替代路線」。

---

## IEEE 審稿人

審稿人會先肯定**誠實的負結果**（§3.5 自我修正、§4.1 seed-robust 負結果、§4.2 大 job 失敗剖析），這在系統論文裡是加分。已解決的幾項：**注入噪音真實性**（原本核心結果依賴 σ=1.0 的 mean-preserving lognormal，受真實性質疑；現用生產 LightGBM 量真實 trace 的 log-殘差 std（philly 1.45 / ali 1.24）證明 σ=1.0 是保守下界、殘差近高斯，§3.2）、**訓練單 seed**（原本只跑單 seed；現 sim+live 皆 seed 42/43/44 報 mean±std，live 負結果 n=246 + 配對 CRN 已統計顯著，§4.3.1/§4.4）、**item-1 因果宣稱過強**（原本「曝露異質性 → 贏 10/12」聽起來像正結果；現 live 四方 A/B 證實是顯著負、屬縮 gap 而非贏，§4.3.1）。仍需改進：

### baseline 太單薄（已大幅補強：Slurm 原生 baseline 已實機對照）

原本只比自家 `score` 啟發式 + vanilla SAC，「啟發式勝出」這個結論別人沒法站上去。**已在實機補上三個 Slurm 原生 baseline**（改設定即可切換、同場上線比，§4.5）：FCFS（`sched/builtin`+`priority/basic`）、multifactor（`priority/multifactor`）、packing（`CR_Pack_Nodes`）；每臂關掉 Lua 的 score/RL hook 讓 Slurm 原生排程接管，跑與 §4.3.1 相同的 heavy-tail CRN workload。**結果（3 seed、run-position 對照）：score 與三個 Slurm 原生排程全部統計打平**（ΔJCT% vs score：multifactor +0.4±1.0、packing +0.6±1.1、fcfs +1.7±2.9，mean±std 全跨 0）。方法學插曲:單 pass 一度顯示 fcfs +5%，但那與 run-order 完美單調相關（GPU restore 後暖機漂移），3 seed × 3 種臂順序對照後消失（fcfs 隨位置在 +5.0/+0.5/−0.4 擺盪）。**SJF 沒有 Slurm 原生 plugin**，score 的 `f_runtime_short`+predictor 就是 live 版 SJF 近似，不另比。這把核心結論放進更大脈絡:**不只 DRL 贏不了 score，連 score 自己對 trivial baseline 都只是打平——在此規模整個排程策略空間是平的**。仍可選做:加一個已發表 RL scheduler（Decima / DeepRM 改編）與近似上界讓 ΔJCT% 有絕對尺度感。

### 外部效度與「sim 不預測 live」

§4.4 量化出一個強的方法學限制：**sim-validation 不能預測 live**。sim 排名是 seed 43≈44 ≫ 42（42 在 sim 崩塌 cvar −133%/mean −139%），但 live 排名是 42≈43≈44（seed-42 落在同一窄帶 −1.3~−3.2%，甚至是所有 learned arm 裡最不爛的 RDSAC-mean −1.3%）。sim 的崩潰沒轉移到 live，**自動依 sim 選 seed 不穩健，應報全 seed 分布（含失敗 seed）**。機制：live RL 訊號很小（飽和時 abstain → fallback、非飽和時 placement 差異有限），把 sim 端的高變異洗掉了。論文要明確分離兩個主張：(a) sim 內演算法比較、(b) live 的負結果 + 韌性（fail-safe）。Threats to Validity 要寫全：訓練高變異（跨 seed 擺盪 30–90 pts）、item-1 sim 速度建模為一階純量近似、消費級 GPU 與桌面遊戲共享干擾、sim 不預測 live 故不能用 sim-validation 選 seed。

### 命名與定位撞名

「RDSAC/DSAC」與 Duan 2021 DSAC 撞名。換乾淨名稱（如 *discrete risk-sensitive distributional SAC, dRSAC*）或在標題/摘要重度 caveat。定位主軸應是 **safe + observable ML-assisted scheduling platform 的誠實負結果 + 機制剖析**，而非「RL 贏 heuristic」。

### σ 仍只在合成 trace 上量

§3.2 的 σ 是合成 trace 的最難預測上界，屬保守下界的論證；應在 `load_philly()` 真實結構化 trace 上重量，把噪音校準從合成推進到真實。

---

## AI Infra 工程師

已解決的幾項：**train/serve 動作落差**（原本 sim 訓 placement、live 只把 RL 選擇轉成 priority boost；現走 explicit placement，submit-時 `-w` 釘節點（Slurm 21.08 無法 post-submit 重釘），serve 的 node 選擇真的被執行——這也是 §4.1 能量化「learned 全把負載擠到 4070 85–89%」的原因）、**3080 MPS**（原本 driver 580 沒為 22.04 build，MPS 根本沒多工、被無 CUDA context 的 sleep job 掩蓋；升 24.04 後 3080 通過 4-concurrent 真實 CUDA，§4.2 才成立）、**host 監控盲區**（原本只有 GPU(DCGM)/Slurm/k8s/RL；補 node-exporter（hostNetwork DaemonSet）+「Node / Host」dashboard）、**異質 live A/B**（node-2 復原後已跑完，結論顯著負）。仍需改進：

### MPS 干擾模型未實測校準

§4.2 已用真實 cuBLAS job 取代 sleep（線性干擾假設不再是唯一證據），但 2-job slowdown 分布的正式實測校準仍缺。MPS 現已能真共置 → 應在真卡量 2-job slowdown 分布取代線性假設。消費卡無 MIG/故障隔離，要在 Threats 寫明。

### 固定拓樸的 obs/action 空間 = 反彈性

`obs_dim`/`n_actions` 綁死 `N_NODES × N_GPUS`，節點 join/leave 就 checkpoint 不相容。item-1 讓 GPU one-hot 反映真實卡型（4070→`[1,0,0]`/3080→`[0,1,0]`）且 dim-preserving（obs_dim 仍 166），**只解了「同拓樸內的異質性」**；節點數變動仍要重訓。中期應走 permutation-invariant / set-based obs（attention pooling），讓單一 policy 跨拓樸。

### submit-path 同步延遲與 SLO 缺口

Lua 在 `slurm_job_submit` 同步呼叫 `/decide`（帶 timeout）。應量 `/decide` 的 p99 與 batch submit 風暴下對 slurmctld 的影響，把 decision latency、snapshot age、abstain rate 設成有 alert 的 SLO。node-exporter host metrics 是這方向的前置（現在看得到 node 的 CPU/load/mem 壓力），但 SLO/alert 與 submit-path chaos 實機數據仍缺。

### 關鍵服務單副本 SPOF

operator 無 leader election；rl-scheduler 單副本；snapshot pusher / live_daemon 仍是 SPOF。host 監控盲區已補，但這幾個關鍵服務的高可用性未解（operator leader election、rl-scheduler 2 replicas + PDB）。

### 消費級測試平台脆弱性

消費級 4070（與 Steam 共享）+ 異質 3080 是限制不是缺陷，但要誠實標示。node-2 曾一次性掉線（已復原）正是脆弱性的活證據；live 數字要排除遊戲佔卡的熱節流（曾因 `wwm.exe` 佔卡使 CUDA 不可用）。理想上最終數據挪到專用機 / 雲端 spot。

---

## ML 專家

已解決的幾項：**歸因拆乾淨**（原本 distributional critic 與風險扭曲綁在一起；補 RDSAC-mean 臂（分布式但風險中立）後證實**分布式 critic 是有用的一半（SAC→mean +25 pts）、CVaR 風險扭曲反而扣分（mean→cvar −22~−34 pts）**，與設計意圖相反，§3.3）、**sim 算力牆**（原本純 Python 離散事件；加 `sim/vec_env.py` + `--num-envs N` 多進程並行 rollout，多 seed 與三方臂在算力上可行）。**兩條「修訓練」支線已撞牆**：full PopArt（原本想用 critic 內 return normalization 取代手調 reward_scale=20000，3-seed 反而比 P2 reward-norm 更差 −23~−48% vs −12~−33% → 否決，eval §6）、RLPD（原本想用 trace-replay 把 seed-43 prior fine-tune，三次全退化 → 撞牆）。仍需改進：

### auto-α 仍是 band-aid（PopArt 路線已否決）

溫度自動調整目前靠手調 `reward_scale` 與固定 α=0.05 撐住。full PopArt 已實作 + 驗證 output-preserving（逐位元 9.5e-7），目標本是消掉這個 caveat，**但 3-seed 反而比 P2 reward-norm 更差、變異更大 → 否決，預設仍用 P2**（先前的 PopArt-lite 才是已驗證有效的那個）。auto-α 的乾淨解仍未定，替代路線是**固定 α + 良好 reward_scale 或 per-arm reward 標準化**。

### RLPD 需改收真實 transition

想用 live transition 把 seed-43 prior 再 fine-tune，三條路全退化 prior（−19.6%/−4.5% → −97~−174%/−79%）：(a) trace-replay 飽和、(b) trace-replay 非飽和、(c) host-side transition-log。trace-replay 病根是 demonstration 機制本身（拉向 sim-score + reward/scale 不匹配）；host-side daemon 透過 `kubectl exec` 每 poll ~1-2s 太慢，抓不到 job 完成瞬間的 completion-transition → online buffer 空 → 一樣退化。**已修一個必要 bug**（`live_daemon` 的 `value_abstain` 寫死 −1.0，seed-43 value≈−10 → 全 abstain、0 transition；改成 env `VALUE_ABSTAIN` 可調）。**seed-43 base 仍為最佳 checkpoint。** 正確路線：把 daemon 部署進叢集（低延遲 squeue/sacct）收滿真實 `(obs,act,rew)` transitions → 保守 offline RL（大 offline anchor、低 online-ratio、限步數、CQL 式保守項）。

### score-warmup 過度集中

1×1 的 abstain/no-op 問題在 2×1 已被「過度集中到 4070」取代（learned 不再 no-op，而是學壞）。診斷指向**訓練退化**（過度集中 + 高變異）而非單純 clone score。P1 balance-shaping 已把集中度從 89%→~71%（§3.6），方向對，但 warmup on/off ablation 仍值得做。

### 共置負結果是預算混淆

§3.4 的 ON 臂有 2× 動作但同步數 → underfit。應給 ON 臂配對的訓練資訊量（按 log|A| 放大步數），或對 ISOLATE 稀疏 mask 做 action-embedding；否則維持「budget-confounded」標註，不下「共置無用」強結論。MPS 多工已解鎖，可在真共置下重評。

### 無 held-out workload

目前沒有 workload split，無法區分泛化與記住 trace 統計。應做 train philly / test ali 的 split；item-1 之後尤其該驗「異質性曝露幫助大 job 放置」是否泛化到沒見過的 workload。

### 機制堆疊缺 ablation

n-step、PER、shaping、warmup、雙頭 Z_R/Z_H 全開。§3.6 已對 balance-shaping + reward-norm 做開關對比（有效，cvar/SAC +19~+31 pts），但 PER、雙頭 Z_R/Z_H 的逐項 ablation 仍缺。雙頭分解增加複雜度，若單頭（熵折進 V）效果相當應簡化。

### item-1 的 sim 速度建模太粗

3080≈0.25× 是固定純量縮放，但真實異質性（不同 kernel 對記憶體頻寬/SM 數的敏感度不同）非單一純量。應把 0.25× 標為一階近似，理想上 per-workload 量真實 4070-vs-3080 的 runtime 比餵進 sim，否則 item-1 帶著一個未校準的速度假設。

---

## 跨領域

§2 三個技術面向問的是「這套東西做得對不對、夠不夠硬」，它們的座標系預設了「你要繼續、且勝負判準是贏不贏得了 score」。本面向換一組座標系，問的是「這件事值不值得繼續、該往哪走、做完長什麼樣、十年後留下什麼」。價值在對撞與盲點，不在完整；它故意不替你做決定，只把視線之外的地圖攤開。

### 「贏 score」這個勝負判準本身沒被質疑

全文（含三位技術審稿人）都把「DRL 有沒有贏過啟發式」當成隱含的成敗線，所有工程都繞著它轉。但沒有人停下來問：對一個跑在兩張消費級卡（其中一張你還要拿去打遊戲）上的 2×1 叢集，「贏過一個已經調得不錯的啟發式」是不是一個結構上幾乎不可能、且就算贏了也沒人會在乎幅度的目標？當勝負線本身可疑時，繞著它做幾個月工程，得到的精度再高也是錨在錯的問題上。真正該被擺上桌的是判準本身，而不是判準下的小數點。

### 統計顯著性：科學需求還是沉沒成本

§4.3 到 §4.3.1，動用三個正交手段（非飽和降噪、樣本三倍化、多 seed）只為了把名目 +3.9% 釘成顯著的負值，這在科學上是嚴謹的；但「我已經投了幾個月，所以要把它做到無懈可擊」與「這個顯著性會改變任何人的決策」是兩件事。一份釘得極死的 −3% 負結果，和一份釘得馬虎的 −3% 負結果，對讀者的行動影響可能完全一樣：都是「別在這個規模用 DRL」。值得追問的是，顯著性的最後幾個百分點，是為了知識，還是為了讓自己甘心收手。

### 機會成本與止損線從未被命名

文件密集記錄「做了什麼、解了什麼、還欠什麼」，卻沒有任何一處寫下「在什麼條件下我會停」。RLPD 三次嘗試全失敗、node-2 曾掉線、核心負結論反覆出現「不變」，這些在技術視角裡是「待解的 OPEN 項」，在資源配置視角裡卻是訊號：你正在為一個邊際報酬遞減的問題持續投入，而沒有預設的退出條件。沒有止損線不代表該停，但「從沒想過止損線長什麼樣」本身是盲點。一個健康的研究計畫應該能回答「怎樣算輸到該收」，即使答案是「還沒到」。

### 受眾是誰決定了「做完」長什麼樣

如果受眾是口試委員，「做完」可能是「一個完整、自洽、能在 30 分鐘內講清楚並守住的故事」，此時負結果加機制剖析已接近完成，再投入是邊際裝飾。如果受眾是未來雇主，「做完」是「一個能展示工程深度與誠實的作品集條目」，此時平台與管線比結論更值錢。如果受眾是你自己（為了搞懂 ML-for-systems 到底行不行），那永遠沒有「做完」，只有「學夠了沒」。這三個受眾指向完全不同的收尾方式，而文件沒有挑明它在替誰寫，於是所有改進清單都缺一個對焦的錨。

### 最有價值的產出可能不是 checkpoint

這份研究最有價值的產出，可能不是任何 checkpoint，而是「一個誠實、可重現的領域級負結果 + 一個異質 MPS-aware 的可觀測平台」。ML-for-systems 充斥著「我們的 RL 贏了 baseline」的正結果論文，而能被嚴謹重現的負結果稀缺到反而是公共財：它幫整個領域省下重複踩坑的成本。十年後真正會被引用、被記得的，大概率是「在小規模異質叢集上，校準噪音 + 多 seed 證明啟發式仍勝出」這個可被站上去的結論，以及那套 OTel-bridge / sim-to-live / fail-safe 平台，而不是 seed-43 的權重檔。弔詭的是，文件花最多力氣的（讓 DRL 縮小落後）恰好是最不會留下的部分。

### 過程本身是否還是你想做的事

寫一個 GPU 開關腳本好讓出卡來打遊戲，這個細節透露了一件技術視角完全不會記錄的事：你和這個專案的關係已經帶著張力。當一個研究計畫需要你在「跑實驗」和「過生活」之間排程同一張顯卡時，繼續與否就不只是科學判斷，也是「我還想不想每天做這個」的判斷。這不是軟弱，是資源（包含你的注意力與熱情）配置的一部分。技術審稿人不會問你還愛不愛這件事，但這個答案實際上比任何待改進項都更能預測這份研究的結局。

### 二元選項外的第三、第四條路

文件的隱含選擇空間是二元的：要嘛把待改進項做完繼續追平/追贏，要嘛承認負結果收掉。但局外人會看到第三條路（把負結果本身當頭條：不再試圖贏，而是把「啟發式為何在此規模難被 DRL 超越」做成一個機制完整、baseline 豐富、可重現的研究主張，這反而需要補的是多 baseline 而非追平），以及第四條路（把這套異質 MPS-aware、可觀測、fail-safe 的平台從「驗證 DRL 排程」這個它一直輸的問題，轉去解一個 RL 真有結構性勝算的問題：不是和調好的啟發式比 JCT，而是去做啟發式根本沒在處理的事，如多目標權衡、線上漂移適應、或人類偏好對齊的排程）。把選項從二元擴成四元，往往比在二元裡糾結更接近出路。

### fail-safe 敘事可能把設計勝利講成安慰獎

全文反覆強調 fail-safe 回退 score 經多次實證，這在工程上是真功勞。但從局外人角度，這裡藏著一個未被點破的張力：一個「只要表現不好就自動退回啟發式」的系統，它的安全性恰恰來自於它不真正信任自己的 RL 決策。換句話說，這套架構的穩健，部分是建立在「預期 RL 會失敗」之上的。這不是缺陷，但它意味著「平台很安全」和「RL 有價值」這兩個賣點之間有內在拉扯：越強調前者，越暗示後者還沒兌現。值得想清楚你要賣的是哪一個。

### 觀點對撞（不調和）

**對撞一：止損/轉向 vs 先把現有故事出貨 vs 重構整個問題框架。** 一種聲音說：邊際報酬已經遞減（RLPD 三連敗、live 反覆 −3%、卡還要分時打遊戲），該畫止損線，把平台轉去一個 RL 有勝算的問題。另一種聲音針鋒相對：現在收掉等於把幾個月變成沒有交付物，最務實的是先把手上這個「誠實負結果 + 可觀測平台」包成一個能守住的完整成品出貨（口試 / 作品集 / 一篇 systems 短文），出貨之後再談轉向。第三種聲音說兩者都錯：問題不在繼續或收尾，而在框架本身選錯了，即「DRL 對打調好的啟發式」這個擂台，無論你站著還是離場都是輸，真正該做的是換一個 RL 不必和啟發式正面對撞的問題重新立題。這三者無法調和，因為它們對「沉沒的幾個月」估值不同：第一種視為已沉沒、應忽略；第二種視為待回收、應變現；第三種視為學費、買到的是「別再選這種擂台」的教訓。

**對撞二：追求負結果的顯著性，是沉沒成本的合理化，還是正當且稀缺的科學貢獻。** 一邊：那種「動用三個手段把名目正值釘成顯著負值」是教科書級的承諾升級，當一個結果需要你這麼用力才能釘死，讀者的注意力通常已經用沉默告訴你它不重要了。另一邊：這恰恰相反，是在做一件多數人偷懶不做、因而稀缺的事，即把負結果做到可重現、統計顯著、跨 seed 穩健，正是 ML-for-systems 再現性危機最缺的公共財；嫌它「不重要」的人，用的是「正結果才算貢獻」的偏見尺。這組對撞無解，因為它觸到一個更深的分歧：科學的價值，是由它改變了多少人的行動來定義，還是由它本身的嚴謹與誠實來定義。

**對撞三：受眾是口試委員（一份要守得住的論文），還是這是領域級的公共知識（一個要被別人站上去的結論）。** 若受眾是口試委員，最優策略是收斂、自洽、可防禦：別再開新戰線（PopArt 已否決就讓它否決、RLPD 別再試第四次），把現有負結果與機制剖析打磨成一個 30 分鐘講得完、問不倒的故事，多餘的 baseline 和 live 重跑都是風險而非加分。若這是公共知識，最優策略恰好相反：要開戰線，即補滿 FCFS/SJF/packing/已發表 RL baseline（否則「啟發式勝出」這個結論別人沒法站上去，因為你只比了自家 score），要報全 seed 分布含失敗 seed，要把平台與資料開源。前者要你關門收斂，後者要你開門擴張；前者的多餘工作是後者的必要工作。你不可能同時最佳化「守得住」和「站得上去」，因為一個要的是封閉，一個要的是暴露。

---

## 待改進總表

> 優先級：P0 = 不做就無法下結論（方法學門檻）；P1 = 讓結論能轉移到 live / 更強的對照；P2 = 工程韌性與清理。

| 編號 | 改進項目 | 面向 | 優先級 | 現況 / 下一步 |
|---|---|---|:---:|---|
| 1 | σ 在真實結構化 trace 上重量 | IEEE | P0 | 現用合成 trace 最難預測上界（保守下界）；應在 `load_philly()` 真實 trace 重量 |
| 2 | auto-α 的乾淨解（非 PopArt） | ML | P0 | PopArt 已 3-seed 否決；改走固定 α + 良好 reward_scale 或 per-arm reward 標準化 |
| 3 | RLPD 改部署 daemon 進叢集收真實 transition | ML | P0 | 三次 trace-replay/host-side 全退化；正路是低延遲 squeue/sacct + 保守 offline RL（CQL 式） |
| 4 | ~~補多 baseline（實機切換 Slurm 內建）~~ → **已做**；剩已發表 RL scheduler + 近似上界 | IEEE | P2 | **DONE（§4.5）**：FCFS/multifactor/packing 已實機對照,3 seed + run-position 對照 → 全部與 score 統計打平。剩可選:Decima/DeepRM 改編 + 近似上界給 ΔJCT% 絕對尺度 |
| 5 | Threats to Validity 寫全 | IEEE | P1 | §4.4 已量化「sim 不預測 live」；連同訓練高變異、消費卡干擾正式寫進效度威脅 |
| 6 | score-residual RDSAC（`final = score + bounded RL_delta`） | ML / IEEE | P1 | 學「何時修正啟發式」、天然 fail-safe；過度集中負結果後優先序升高 |
| 7 | item-1 sim 速度建模校準 | ML | P1 | 3080≈0.25× 是一階純量；per-workload 量真實 4070-vs-3080 ratio 餵進 sim |
| 8 | 共置動作在真 2-GPU 重評 | ML | P1 | 去 budget-confounded；MPS 多工已解鎖可量真共置 |
| 9 | MPS 干擾模型實測校準 | Infra | P1 | 真卡量 2-job slowdown 分布取代線性假設 |
| 10 | held-out workload split | ML | 中 | train philly / test ali，證明泛化而非記住 trace 統計 |
| 11 | 命名與定位（RDSAC 撞 Duan 2021） | IEEE | 中 | 換 dRSAC 或重 caveat；定位為 safe+observable platform 的負結果 |
| 12 | 測試平台搬專用機 / 雲端 spot | Infra | 中 | 消費卡脆弱性（node-2 曾掉線、遊戲熱節流） |
| 13 | warmup / PER / 雙頭 Z_R/Z_H ablation | ML | P2 | balance-shaping、reward-norm 已做；其餘逐項缺 |
| 14 | 跨拓樸 permutation-invariant obs | Infra | P2 | item-1 只解同拓樸異質性，節點數變動仍重訓 |
| 15 | 關鍵服務 SPOF 收斂 | Infra | P2 | operator leader election、rl-scheduler 2 replicas + PDB；snapshot/live_daemon 仍 SPOF |
| 16 | decision-latency / abstain SLO + alert | Infra | P2 | host metrics 已補；SLO/alert 與 submit-path chaos 數據仍缺 |
| 17 | fragmentation progress penalty；NFS mount tuning | Infra | P2 | 沿用前版，仍有效 |
