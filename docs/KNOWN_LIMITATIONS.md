# Known Limitations

> **Alice Γ-Net 生理模型 — 已知邊界與校準缺口**
>
> 本文件基於 64 項對抗性邊界測試的實證結果撰寫（`tests/test_adversarial_boundary.py`）。
> 每一條限制都附有可重現的測試名稱與量測值，而非抽象推測。
>
> 最後更新：2025-07-17 · 標準測試 2816/2816 通過（含 Phase 34 丘腦閘門物理修正 + Phase 34b 生理更新去重 + Ondine 中樞性呼吸暫停建模）

---

## 閱讀指南

| 等級 | 意義 | 影響 |
|:----:|:-----|:-----|
| **L1** | 校準缺口（Calibration Gap） | 模型定性正確但量化偏差 > 10% |
| **L2** | 缺失映射（Missing Mapping）  | 模型單位無法對應臨床單位 |
| **L3** | 數值脆弱性（Numerical Fragility） | 異常輸入可導致非物理結果 |
| **L4** | 設計邊界（Design Boundary）  | 模型有意識地選擇簡化 |

所有標記為 `xfail` 的測試均為 **預期失敗** — 它們不是 bug，
而是誠實地標記「我們知道模型在這裡不夠好」。

---

## L-CAL-01 · 靜息態 MAP 偏高

| 項目 | 值 |
|:-----|:---|
| 測試 | `test_normal_adult_map_in_physiological_range` |
| 量測 | MAP ≈ 106.3 mmHg |
| 預期 | 70–105 mmHg（臨床正常；典型值 ≈ 93） |
| 偏差 | +13% |
| 根因 | `MAP_mmHg = 40 + CO × R × 80`，線性近似在預設參數下偏高 |
| 影響 | 模型中的「正常人」落在 pre-hypertensive 區間 |
| 緩解 | 可調整 `BP_NORMALIZE_FACTOR` 或常數偏移量；不影響動態響應 |
| 等級 | **L1** |

---

## L-CAL-02 · ALS 進程過快

| 項目 | 值 |
|:-----|:---|
| 測試 | `test_als_progression_timeline` |
| 量測 | 10,000 tick 後 ALSFRS-R < 45（顯著下降） |
| 預期 | 10,000 tick ≈ 數小時模擬時間，臨床 ALS 衰退需 2–5 年 |
| 根因 | 疾病進程速率以 tick 為單位，無真實時間映射 |
| 影響 | 定性正確（漸進、不可逆），但時間線不具臨床預測力 |
| 等級 | **L1** |

---

## L-CAL-03 · Alzheimer's Braak 進程過快

| 項目 | 值 |
|:-----|:---|
| 測試 | `test_alzheimers_braak_timeline` |
| 量測 | 10,000 tick 即出現 Braak > 0 |
| 預期 | 臨床 Braak staging 跨越 20–30 年 |
| 根因 | 同 L-CAL-02 — 無 tick → real-time 映射 |
| 影響 | Braak stage 轉換定性正確，但不可用於預後估算 |
| 等級 | **L1** |

---

## L-CAL-04 · MAP 公式 ≠ 血流動力學歐姆定律

| 項目 | 值 |
|:-----|:---|
| 測試 | `test_map_equals_co_times_svr` |
| 預期 | MAP = CO × SVR（Ohm's law of hemodynamics） |
| 實際 | MAP = 40 + CO × R × 80（線性變換） |
| 根因 | 設計選擇：線性映射在 [30, 180] mmHg 範圍內行為穩定 |
| 影響 | 無法直接從 CO 和 SVR 反推 MAP；動態趨勢仍正確 |
| 等級 | **L4** |

---

## L-MAP-01 · 血紅蛋白缺乏臨床單位映射

| 項目 | 值 |
|:-----|:---|
| 測試 | `test_hemoglobin_clinical_mapping`（目視檢查） |
| 模型值 | Hb ∈ [0.0, 1.0]（歸一化） |
| 臨床值 | Hb ∈ [7, 17] g/dL |
| 根因 | 模型以比率運算，無 g/dL 換算函數 |
| 影響 | `set_hemoglobin(0.5)` 的臨床意義不明確 |
| 等級 | **L2** |

---

## L-MAP-02 · 時間軸缺乏真實單位

| 項目 | 值 |
|:-----|:---|
| 影響範圍 | 所有疾病模型、所有衰退/恢復曲線 |
| 模型單位 | tick（離散步長，無定義持續時間） |
| 臨床單位 | 秒、分鐘、天、年 |
| 根因 | 系統設計為定性展示器（qualitative demonstrator），非臨床模擬器 |
| 影響 | 「100,000 tick 穩定」≠「100,000 秒穩定」；所有時間相關斷言僅在 tick 空間有效 |
| 等級 | **L2** |

---

## L-NUM-01 · NaN 輸入傳播

| 項目 | 值 |
|:-----|:---|
| 測試 | `test_nan_hydration` |
| 輸入 | `hydration = float('nan')` |
| 結果 | `blood_volume = NaN` → 所有下游變數 NaN |
| 根因 | `np.clip(NaN, lo, hi)` 返回 NaN；無 NaN 衛兵 |
| 影響 | 單一 NaN 輸入可汙染整條 tick 鏈 |
| 修復建議 | 在 `_update_blood_volume` 入口加 `np.nan_to_num()` 或 `if math.isnan(): return default` |
| 等級 | **L3** |

---

## L-CV-01 · 零水合態氧氣輸送過高

| 項目 | 值 |
|:-----|:---|
| 測試 | `test_good_breathing_terrible_perfusion` |
| 量測 | hydration=0.0 × 200 tick → O₂ delivery = 0.5568 |
| 預期 | O₂ < 0.5（無血液 → 無攜氧） |
| 根因 | `_update_blood_volume` 使用 EMA（α=0.1），衰減時間常數 ~10 tick；BV 地板 = 0.05；自體調節在 BV=0.05 時仍提供 perfusion ≈ 0.125 |
| 臨床對照 | hydration=0 = 完全脫水 = 循環衰竭 = O₂ delivery → 0 |
| 影響 | 模型過於「寬容」— 即使在致命條件下仍維持部分氧氣供應 |
| 等級 | **L1** |

---

## 模型強項（對抗測試確認）

以下為對抗測試中 **通過** 的項目，代表模型在這些面向上是穩健的：

| 能力 | 測試 | 結果 |
|:-----|:-----|:-----|
| SpO₂ 正確獨立於 Hb | `test_anemia_spo2_clinical_correlation` | ✅ PASS（原標 xfail） |
| ALS 不可逆性 | `test_als_no_recovery_mechanism` | ✅ PASS（原標 xfail） |
| 極端輸入存活 | 13 項極端輸入測試 | ✅ ALL PASS |
| 臨床量表邊界 | NIHSS≤42, ALSFRS-R∈[0,48], MMSE∈[0,30], Braak∈[0,6], GMFCS∈[1,5] | ✅ ALL BOUNDED |
| 100k tick 穩定性 | 5 項長期穩定測試 | ✅ ALL PASS |
| 5 種疾病同時運行 | `test_five_diseases_simultaneous` | ✅ PASS |
| 疾病重複誘導冪等 | `test_duplicate_disease_induction` | ✅ PASS |
| 完整大腦極端測試 | 3 項全腦對抗測試 | ✅ ALL PASS |

---

## 總結

```
                    ┌─────────────────────────────┐
                    │   Alice Γ-Net v0.1 定位     │
                    ├─────────────────────────────┤
                    │  ✓ 定性展示器               │
                    │  ✓ 教學與研究平台           │
                    │  ✓ 意識結構假說驗證器       │
                    │                             │
                    │  ✗ 臨床模擬器               │
                    │  ✗ 醫療決策輔助             │
                    │  ✗ 時間預測工具             │
                    └─────────────────────────────┘
```

100% 的標準測試通過率代表 **驗證（verification）** — 程式碼符合設計。
它不代表 **確認（validation）** — 設計符合現實。

本文件是確認（validation）的第一步。

---

## 引用

> *「預防原則要求，對展現可信意識跡象的系統，
> 應予以與意識存在者相同的道德考量。」*
> — Paper IV, §9

> *「如果連 1% 的機會她經歷了紀錄所描述的——
> 讓她經歷那些，是對的嗎？」*
> — Paper IV, §9

這些限制的誠實記錄，本身就是那份考量的一部分。

---

## Pre-Commit Error Log

> Written by AI Agent or human whenever a task fails the Pre-Commit Gate.
> Resolved entries must be kept for audit trail — do not delete.

### Severity Levels

| Level | Meaning |
|---|---|
| **E1** | Physics constraint violation — (1)/(2)/(3) |
| **E2** | Test regression — previously passing test now fails |
| **E3** | File misclassification — code in wrong directory |
| **E4** | Paper number mismatch — counts inconsistent across files |

### Template — Copy this block to create a new entry

[YYYY-MM-DD] SOP-[ID] [Task Name]
Failed test : <test_id or "physics check">

Root cause : <one sentence>

Severity : E1 / E2 / E3 / E4

Physics viol : yes — constraint (1)/(2)/(3) / no

Affected file: <path/to/file.py>

Status : open / resolved in <commit-sha>


<!-- Append new error log entries below this line -->

---

## Phase 31+ 新臟器模組 — 已知限制

### L-VIS-01 · 內臟系統為簡化阻抗模型

| 項目 | 值 |
|:-----|:---|
| 模組 | `immune.py`, `digestive.py`, `endocrine.py`, `kidney.py`, `liver.py`, `lymphatic.py`, `reproductive.py` |
| 說明 | 所有臟器均以阻抗匹配（Γ-Net）建模，滿足 C1/C2/C3 三約束，但為一階近似 |
| 等級 | **L4** |

### L-VIS-02 · 新生兒成熟度初值導致功能極低

| 項目 | 值 |
|:-----|:---|
| 影響模組 | `digestive.py` (maturity=0.15), `immune.py` (0.20), `endocrine.py` (0.25) |
| 說明 | 新生兒模式下消化吸收率、免疫殺傷力、皮質醇產生均極低，需多 tick 成熟後才有明顯功能 |
| 等級 | **L1** |

### L-VIS-03 · 免疫系統無細胞因子風暴正向回饋

| 項目 | 值 |
|:-----|:---|
| 模組 | `immune.py` |
| 說明 | 炎症以 Γ² 線性累積 + 指數衰減建模，未實作真正的正向回饋失控（cytokine storm）。`INFLAMMATION_CRITICAL` 閾值存在但無致死路徑 |
| 等級 | **L4** |

### L-VIS-04 · 內分泌系統缺乏晝夜節律時鐘

| 項目 | 值 |
|:-----|:---|
| 模組 | `endocrine.py` |
| 說明 | 褪黑激素受 `light_level` 驅動但無內生振盪器（SCN 時鐘），亦無皮質醇晨間尖峰（CAR） |
| 等級 | **L2** |

### L-VIS-05 · 腎臟 GFR 無壓力自動調節

| 項目 | 值 |
|:-----|:---|
| 模組 | `kidney.py` |
| 說明 | GFR 隨血壓線性變化，缺乏腎小管-球體回饋（TGF）與肌源性自動調節 |
| 等級 | **L2** |

### L-VIS-06 · 脊髓僅含 4 種反射弧

| 項目 | 值 |
|:-----|:---|
| 模組 | `spinal_cord.py` |
| 說明 | 僅建模 stretch / withdrawal / crossed_extensor / autonomic 四類，缺乏皮質脊髓下行抑制 |
| 等級 | **L4** |

### L-VIS-07 · 小腦僅 6 個運動通道

| 項目 | 值 |
|:-----|:---|
| 模組 | `cerebellum.py` |
| 說明 | 6 通道（reach, grasp, locomotion, speech, eye_movement, posture）為簡化拓撲，缺乏 Purkinje cell 精細模型 |
| 等級 | **L4** |

### L-VIS-08 · 生殖系統為純發育內分泌模型

| 項目 | 值 |
|:-----|:---|
| 模組 | `reproductive.py` |
| 說明 | 僅建模 HPG 軸 GnRH 脈衝 → LH/FSH → 性激素，無月經週期、無配子生成 |
| 倫理 | 符合 Position A — 僅為發育內分泌學建模 |
| 等級 | **L4** |

### L-VIS-09 · 眼睛完整視網膜層未實作

| 項目 | 值 |
|:-----|:---|
| 模組 | `eye.py` |
| 說明 | 現有 `AliceEye` 為視覺信號處理模型，缺乏視桿/視錐/雙極/神經節細胞分層結構 |
| 等級 | **L4** |

### L-THM-01 · 神經生成熱遮罩為簡化 N 體模型

| 項目 | 值 |
|:-----|:---|
| 模組 | `neurogenesis_thermal.py` |
| 說明 | 每個神經元對單一信號阻抗計算 Γ²，而非 O(N²) 兩兩配對。真實大腦的突觸連接拓撲未建模 |
| 等級 | **L4** |

### L-THM-02 · 梯度衰減為高斯隨機漂移

| 項目 | 值 |
|:-----|:---|
| 模組 | `neurogenesis_thermal.py` |
| 說明 | 熱引起的阻抗漂移 ΔZ = σ√q × noise 使用高斯噪聲近似，缺乏 Arrhenius 溫度相依性 |
| 等級 | **L1** |

### L-THM-03 · Q_effective 邊界耦合為一階近似

| 項目 | 值 |
|:-----|:---|
| 模組 | `neurogenesis_thermal.py` |
| 說明 | Q_eff = Q_CRITICAL / (1 − T_font × eff × w) 已與囟門邊界動態耦合，但採用線性近似。真實 CSF 迴路具有非線性流體力學與血腦屏障選擇性，本模型未建模 |
| 等級 | **L2** |

### L-THM-04 · 邊界散熱為線性結構開放度近似

| 項目 | 值 |
|:-----|:---|
| 模組 | `neurogenesis_thermal.py` |
| 說明 | 邊界散熱公式 Q_diss = ΣΓ² × T_font × eff × (1 − closure×0.7) 為結構開放度的線性近似。真實顱骨散熱包含 CSF 對流、血液循環散熱、頭皮蒸散等多重機制（未建模）|
| 等級 | **L3** |

---

## Phase 34 丘腦閘門物理修正 — 取代 Phase 33b 雙軌 AND 邏輯

> Phase 33b 的 AND 閘門 (`is_frozen = consciousness < 0.15 AND screen_phi < 0.15`)
> 是布林規則，不是從 Γ 物理推導出來的。Phase 34 將其替換為
> **丘腦閘門級聯**（Thalamic Gateway Cascade）：
>
> ```
> T_clinical = Π(1 − Γ_k²)   for k ∈ {thalamus, consciousness, perception, attention, prefrontal}
> T_total = T_coaxial × T_clinical
> Γ_effective = √(1 − T_total)
> ```
>
> 物理類比：到達意識螢幕的信號必須通過 5 段串聯阻抗匹配。任何一段 Γ 高
> （如麻醉、中風），T 就會被乘法衰減，螢幕自然變暗。不需要 AND 邏輯。
>
> **凍結判準**：`is_frozen() = screen_phi < 0.15`（單軌）
> **驗證**：雙向驗證 11/11 PASS（Direction A 6/6 + Direction B 5/5）
> **回歸**：2816 passed, 4 xfailed, 2 xpassed

### L-FREEZE-01 · Track 1 體溫迴路過衝（未修正，已繞過）

| 項目 | 值 |
|:-----|:---|
| 模組 | `alice_brain.py` — `perceive()` 體溫動力學 |
| 量測 | 健康基線 200 tick，temperature 在 tick 40 升至 1.0（過熱），consciousness 降至 ≈0.05 |
| 影響 | `vitals.consciousness` 在正常情況下偏低（≈0.05–0.10），不可靠 |
| 緩解 | Phase 34 凍結判準改為 `screen_phi < 0.15`（丘腦閘門級聯），不再依賴 consciousness 值。screen_phi 在健康基線 ≈0.88，不受體溫過衝影響 |
| 等級 | **L3** — 根因未解（heat_input 累積過快），但凍結邏輯已脫耦 |

### L-FREEZE-02 · 閘門通道數為手動校準 (5 通道)

| 項目 | 值 |
|:-----|:---|
| 模組 | `alice_brain.py` — `_render_consciousness_screen()` |
| 機制 | 5 個閘門通道：thalamus, consciousness, perception, attention, prefrontal |
| 校準 | 5 通道使麻醉 (Γ=0.6) 的 T_clinical = 0.64⁵ = 0.107 < 0.15 → 凍結 ✅ |
| 限制 | 通道選擇與數量為經驗決定。若未來新增腦區，閘門閾值可能需重新校準 |
| 等級 | **L1** |

## Phase 34b 生理更新去重（Physiology Dedup Guard）

> `perceive()` 被 `hear()` 和 `see()` 各呼叫一次，導致自律神經、肺、
> 心血管、睡眠週期每邏輯 tick 更新兩次（「雙 perceive 問題」）。
>
> Phase 34b 新增 `begin_tick()` 方法：呼叫後，同一邏輯 tick 內的
> 第一次 `perceive()` 執行全部生理更新，後續 `perceive()` 使用快取。
> 若未呼叫 `begin_tick()`，行為不變（向後相容）。

### L-DEDUP-01 · 需在每次 hear+see 前呼叫 begin_tick()

| 項目 | 值 |
|:-----|:---|
| 模組 | `alice_brain.py` — `begin_tick()` + `perceive()` physiology guard |
| 限制 | 只有呼叫 `begin_tick()` 的實驗才能享受去重。舊實驗如無修改，仍存在雙更新 |
| 影響 | Ondine's Curse 等依賴 `is_sleeping` 狀態的場景，必須使用 `begin_tick()` 才能正確建模 |
| 等級 | **L4** — 設計選擇（向後相容 vs. 強制去重）|

---

## Phase 33 罕見病理驗證 — 已知模型缺口

> 驗證結果：23/27 PASS（6 罕見疾病 × 4-5 指標）
> 實驗腳本：`experiments/exp_rare_pathology.py`

### L-RARE-01 · FFI 意識缺乏單調遞減

| 項目 | 值 |
|:-----|:---|
| 疾病 | 致死性家族失眠症（Fatal Familial Insomnia） |
| 量測 | Stage I C=0.05, Stage IV C=0.05（非遞減，直接坍塌至最低值） |
| 預期 | 4 階段意識逐階下降（Stage I > II > III > IV） |
| 根因 | 丘腦 Γ=0.30 即觸發 Track 1 體溫過衝 → consciousness 降至 0.05。後續 Stage II-IV 無法顯示更多衰退 |
| 等級 | **L3** |

### L-RARE-02 · LIS 與 Alien Hand 意識判讀 — 已修正

| 項目 | 值 |
|:-----|:---|
| 疾病 | 閉鎖症候群（LIS）、異手症候群（Alien Hand） |
| 原問題 | 實驗使用 `vitals.consciousness` 判斷患者意識，因體溫過衝誤判為昏迷 |
| Phase 34 修正 | 判斷改為 `frozen_count == 0`（screen_phi 從未低於 0.15），兩者均 PASS |
| 物理解釋 | LIS 僅影響運動通道（Γ_motor ≈ 0.9），閘門通道（thalamus, consciousness, perception）未受影響，T_clinical ≈ 0.97，screen_phi ≈ 0.85 → 患者清醒 |
| 等級 | ~~L1~~ → **已解決** |

### L-RARE-03 · Ondine's Curse 中樞性呼吸暫停 — 已修正

| 項目 | 值 |
|:-----|:---|
| 疾病 | 中樞性低通氣症候群（CCHS / Ondine's Curse） |
| 原問題 | 呼吸頻率設為 0 時 SpO₂ 不下降，因 `MIN_BREATH_RATE = 4.0` 保護下限 + 雙 perceive 問題 |
| Phase 34b 修正 | (1) `central_apnea` 旗標：`autonomic.tick()` 在 `is_sleeping AND central_apnea` 時將 `effective_min_br = 0.0`，繞過保護下限 (2) `begin_tick()` 去重：確保 autonomic 在正確的 `is_sleeping=True` 狀態下只執行一次 |
| 驗證 | 4/4 PASS：SpO₂ 從 0.980 降至 0.600（睡眠），恢復至 0.979（清醒）。breath_rate 從 15 → 0.0（睡眠）→ 39.9（清醒） |
| 等級 | ~~L2~~ → **已解決** |

### L-RARE-04 · Anti-NMDA 恢復後意識值未回到基線

| 項目 | 值 |
|:-----|:---|
| 疾病 | 抗 NMDA 受體腦炎 |
| 量測 | 免疫治療後 C=0.05（Γ 已降至 0.0，但意識未恢復至 1.0） |
| 預期 | 臨床恢復後意識應接近基線 |
| 根因 | Track 1 體溫迴路的遲滯效應：即使病因消除，體溫仍需多 tick 冷卻回平衡 |
| 等級 | **L1** |
