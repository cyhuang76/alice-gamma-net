# Known Limitations

> **Alice Γ-Net 生理模型 — 已知邊界與校準缺口**
>
> 本文件基於 64 項對抗性邊界測試的實證結果撰寫（`tests/test_adversarial_boundary.py`）。
> 每一條限制都附有可重現的測試名稱與量測值，而非抽象推測。
>
> 最後更新：2025-06-25 · 標準測試 2106/2106 通過

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
