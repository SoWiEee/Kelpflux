# 論文審核報告 — 第二輪（術語統一與圖表自說性完成後）

## 執行摘要
- **整體狀態**：論文骨架與故事線扎實，**可進入「補實驗/補統計」階段**。剩餘缺口集中在 **證據鏈完整性**，而非寫作表達。

---

## 剩餘待改進項目（依優先級排序）

### P0 — 必須在投稿前完成（統計嚴謹度 / Baseline 完整性）

| # | 項目 | 現狀 | 具體行動 | 預估工時 |
|---|------|------|----------|----------|
| 1 | **補充材料統計表 (Table S1, S2)** | 正文只寫「見補充材料」，尚無實際檔案 | 產出 `supplementary/tables.tex`：<br>• ANOVA / paired-t 原始 p-value<br>• Holm-Bonferroni adjusted p-value<br>• Cohen's d / Hedge's g<br>• TOST equivalence bounds ±5% / ±10%<br>• Power analysis (post-hoc) | 2-3 hr |
| 2 | **MPS-aware Backfill baseline** | 僅有 vanilla Backfill | 在模擬器加入 `backfill+mpx`（Slurm 原生 `gres/mpx`），並納入表 3/4 對比 | 4-6 hr |
| 3 | **KAI / Volcano scheduler baseline（模擬）** | 完全缺失 | 用 Kubernetes 模擬器跑 KAI-Scheduler / Volcano，對比「K8s 原生排程」vs「Slurm+RL」 | 8-12 hr |
| 4 | **Ablation: Joint vs. Decoupled** | 缺失 | 三組實驗：(a) GPU placement only (fixed 100% MPS) (b) MPS fraction only (fixed GPU) (c) Joint — 同 seed、同 workload | 6-8 hr |

---

### P1 — 強烈建議完成（證據深度 / 可重現性）

| # | 項目 | 現狀 | 具體行動 | 預估工時 |
|---|------|------|----------|----------|
| 5 | **RLPD sim-to-real gap 量化圖** | 只文字說明 | 繪製：x=training steps, y=sim/real JCT gap；疊加 policy KL divergence、value error 曲線 | 4-6 hr |
| 6 | **Motivating Example / 反例** | Introduction 缺乏「為何三者同時存在才需要 DRL」的具體場景 | §1.1 或 §1.2 加入一個 concrete toy case：<br>• 2 GPU (A=fast, B=slow)<br>• 3 jobs (short-inf, long-train, medium)<br>• 展示 heuristic 在 MPS fragmentation 下陷入局部最優，DRL 能跳出 | 1-2 hr |
| 7 | **Artifact Appendix / Reproducibility** | 無 | 準備：<br>• GitHub repo (MIT/Apache-2.0)<br>• Docker image (Slurm + MPS + RL service)<br>• Trace subset (Alibaba/Philly 1k jobs)<br>• `run_all.sh` 一鍵重現表 3/4<br>• Compute budget 表 (GPU-hours, CO2e) | 4-6 hr |

---

### P2 — 若有餘力可加分（擴大外部效度）

| # | 項目 | 備註 |
|---|------|------|
| 9 | **LLM serving workload (vLLM/TGI)** | 使用 ShareGPT/Orca trace，評估 token latency、throughput、batch scheduling |

---

## 正文微調建議（低優先級，可併入下一輪修稿）

1. **§3.1 架構圖 caption** 補上：`job_submit.lua` hook 名稱、monitoring 週期 (1s)、fallback 條件 (timeout > 200ms / invalid action / health check fail)。
2. **§4.4 RDSAC 雙頭 IQN** 補一句：「critic 輸出 32 分位數 (τ∈{0.03,…,0.97})，actor 以 CVaR<sub>0.1</sub> 為目標」。
3. **§5.5 Drift-robust interleaving** 具體化：「以 ABCD... 輪換執行 4 種 scheduler，每輪間隔 5 min 冷卻」。
4. **表 2 (模擬結果)** 補上 **95% CI** 與 **effect size vs FCFS**。
5. **結論段落** 加入：「本研究開源之最小實機平台（Slurm+MPS+RL hook）可作為社群基準設施，歡迎擴展至更大叢集。」

---

## 風險提示

| 風險 | 影響 | 緩解策略 |
|------|------|----------|
| P0 項目 2/3 需額外模擬器開發 | 可能延遲投稿 | 先用簡化模擬器 (discrete-event, 非完整 K8s) 跑 baseline，註明為「模擬器近似」 |
| 統計檢定結果仍不顯著 | 結論需更保守 | 誠實報告「無足夠證據拒絕 H0」，強調 **等價性檢定** 與 **效應量** 而非 p-value |
| Reviewer 質疑 2 GPU 太小 | 外部效度受限 | 在 Limitations 明確定位為「最小可驗證平台」，並引用 SC'23/ATC'24 類似 2-4 GPU 實機論文 |
