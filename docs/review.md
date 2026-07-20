# 論文審核報告 — 第二輪（術語統一與圖表自說性完成後）

## 執行摘要
- **整體狀態**：論文骨架與故事線扎實，**可進入「補實驗/補統計」階段**。剩餘缺口集中在 **證據鏈完整性**，而非寫作表達。
- **本輪修訂（2026-07-20）**：移除三類不當項目並強化其餘——
  - **事實／術語錯誤**：`backfill+mpx（gres/mpx）`（Slurm 無此功能，叢集用 `gres/mps`）、`CVaR₀.₁`（實際尾端質量 `risk_beta = 0.25`）、RLPD sim-to-real gap 曲線（實機僅 181 筆 transition，無法支撐平滑曲線）。
  - **會削弱誠實負結論主軸／重新引入已刪弱對照**：Volcano/KAI 模擬對照、暗示「DRL 跳出局部最優」的引例、風險表「簡化模擬器 + 註明近似」的緩解策略。
  - 新增「已定案、刻意不追」區塊，保留先前決策理由，避免重工。

---

## P0 — 投稿前建議完成（直接強化核心主張）

| # | 項目 | 現狀 | 具體行動 | 預估工時 |
|---|------|------|----------|----------|
| 1 | **Ablation: Joint vs. Decoupled 放置決策** | 缺失；論文核心賣點「GPU 放置 × MPS 分配需**聯合**決策」目前無直接證據 | sim-only、同 seed CRN 配對、沿用既有 `gym_env`／`score`：<br>(a) 只放置（MPS 固定滿槽）<br>(b) 只 MPS 分配（GPU 固定）<br>(c) 聯合<br>**若三者差異落在 ±5% 內，誠實回報「本規模下聯合決策未帶來可測增益」——與全文負結論一致，不損及貢獻定位。** | 6-8 hr |

> **為何是唯一 P0：** 清單中 CP 值最高的一項——直接檢驗論文最想主張的東西，純模擬、零硬體成本。唯一但書：實機 MPS 從未真正多工（見部署紀錄），故此消融結論須限定在 sim scope，不外推至實機。

---

## P1 — 強烈建議完成（可重現性 / 補充材料打包）

| # | 項目 | 現狀 | 具體行動 | 預估工時 |
|---|------|------|----------|----------|
| 2 | **補充材料統計表 (Table S1, S2)** | **統計已算完並寫入 §5.3**（Holm-Bonferroni、TOST ±5%、Cohen's d，正文已引用表 S1/S2）；僅缺實際檔案 | **投稿前必須存在、但屬打包非統計缺口**：把 §5.3 已回報的數字整理成 `supplementary/tables.tex`（原始 + adjusted *p*、Cohen's *d*、TOST 邊界、MDE） | 1-2 hr |
| 3 | **Artifact Appendix / Reproducibility** | repo 已存在；缺對外打包 | Docker image (Slurm + MPS + RL service)、trace subset (Alibaba/Philly 1k jobs)、`run_all.sh` 一鍵重現表 3/4、compute budget 表 (GPU-hours, CO2e) | 4-6 hr |

---

## P2 — 若有餘力可加分（擴大外部效度）

| # | 項目 | 備註 |
|---|------|------|
| 4 | **LLM serving workload (vLLM/TGI)** | 使用 ShareGPT/Orca trace，評估 token latency、throughput、batch scheduling；與 §6.3 未來工作對齊 |

---

## 正文微調建議（低優先級，可併入下一輪修稿）

1. **§3.1 架構圖 caption** 補上：`job_submit.lua` hook 名稱、monitoring 週期、fallback 條件（timeout / invalid action / health check fail）。〔數值請填入實際部署值〕
2. **§4.4 RDSAC 雙頭 IQN** 補一句：critic 輸出分位數、actor 以 **CVaR 尾端質量 `risk_beta = 0.25`** 為目標（對齊 `dsac.py`，切勿寫成 0.1）。
3. **§5.5 Drift-robust interleaving** 具體化：以 ABCD… 輪換執行 4 種 scheduler，每輪間隔〔填入實際冷卻時間〕。
4. **表 2（模擬結果）** 補上 **95% CI** 與 **effect size vs FCFS**（部分場景已補，統一格式即可）。
5. **結論段落** 加入：「本研究開源之最小實機平台（Slurm+MPS+RL hook）可作為社群基準設施，歡迎擴展至更大叢集。」

---

## 已定案、刻意不追（保留理由，避免重工）

| 項目 | 決策 | 理由 |
|------|------|------|
| 階段二加 seed 到 n=16–24 | 不追 | hybrid 均值已在 ±5% 等價帶外、數學上無法證等價；cuBLAS 已證實等價。成本 ~25h 訓練 32 個 checkpoint，CP 值過低。階段一即統計上正確的收尾點。 |
| §5.4 sim-internal SOTA 對照（Volcano/KAI/Kueue 模擬） | 移除 | 模擬器內近似非公平比較、可信度不足；曾寫入後刪除。**勿再以「簡化模擬器 + 註明近似」重新引入。** |
| Kueue 真 GPU 2-node 對照 | 擱置（open） | DRA 的 GPU claim 為獨佔式，Slurm 與 Kueue 無法在同一張卡並存，無法照搬 §5.3 的 A/B 方法。已有可運行 PoC 驗證整合路徑，完整真 GPU 評估列為開放項目。 |
| 表 5（BERT 放置，n=246 job 層級） | 刪除 | job 當獨立單位＝偽重複、p 值被灌水（1e-12～1e-17），與 §5.3.1 seed 層級分析矛盾。結論一律以 seed 層級為準。 |

---

## 風險提示

| 風險 | 影響 | 緩解策略 |
|------|------|----------|
| 統計檢定結果仍不顯著 | 結論需更保守 | 誠實報告「無足夠證據拒絕 H0」，強調 **等價性檢定** 與 **效應量** 而非 p-value |
| Reviewer 質疑 2 GPU 太小 | 外部效度受限 | 在 Limitations 明確定位為「最小可驗證平台」，並引用類似 2-4 GPU 實機論文〔需查證確有此類論文再引，勿臆造引用〕 |
