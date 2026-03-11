# 湧現標準 — Emergence Standard

> **Γ-Net 的終極檢驗：行為是從 C2 長出來的，還是你寫進去的？**
>
> *"If you had to write a flag, it didn't emerge."*

---

## 問題陳述

Alice Γ-Net 聲稱所有行為源自阻抗物理：

$$\mathcal{A}[\Gamma] = \int_0^T \sum_i \Gamma_i^2(t)\, dt \;\to\; \min$$

$$\Delta Z = -\eta \cdot \Gamma \cdot x_{\rm in} \cdot x_{\rm out} \quad (\text{C2})$$

如果一個現象（疾病、情緒、意識狀態）**必須依靠疾病專屬參數、專屬旗標、
或人工設定的 Γ 閾值**才能出現——它就不是從 C2 湧現的，
而是被腳本「寫」進去的。

此文件定義三個嚴格等級，作為所有模型的品質判準。

---

## 三個湧現等級

### E0 — 真湧現（True Emergence）

> **定義**：僅使用 C1/C2/C3 + 組織初始條件 + 外部事件序列，
> 行為自動出現。**零疾病專屬程式碼。**

| 條件 | 說明 |
|:-----|:-----|
| 零旗標 | 無 `is_ptsd=True`、無 `disease_type="ALS"`、無專屬 if/else |
| 零專屬公式 | 無 `z *= 0.995`、無自定義衰減函數 |
| 零手動閾值 | 無 `if gamma_sq > 0.7: stage = "severe"` |
| 參數 | 僅允許組織層級（`z_ref`, `z_min`, `z_max`, `eta`），不允許疾病層級 |
| 驗證方式 | 改變初始條件 → 不同疾病表現自然出現 → 與臨床文獻定性一致 |

**範例（理想態）**：
```
channel = ImpedanceChannel(z_init=50.0, z_ref=50.0)   # 健康神經

# 事件：重複高強度衝擊 (trauma)
for _ in range(1000):
    channel.remodel(x_in=1.0, x_out=1.0, eta=0.01, z_target=500.0)

# 結果：Γ → 1, T → 0  →  「通道崩潰」 →  臨床上對應 PTSD / ALS / ...
# 不需要 is_ptsd 旗標——崩潰行為是 C2 的數學必然
```

**E0 是最終目標。** 目前整個臨床層0%達到此等級。

---

### E1 — 參數化湧現（Parameterized Emergence）

> **定義**：使用 C2 + ImpedanceChannel，但需要**每疾病參數**
> （`z_coeff`, `severity`, `treatment_factor`）來設定初始條件與目標。
> C2 仍執行所有 Z 更新——但初始條件是手動給的。

| 條件 | 說明 |
|:-----|:-----|
| Z 更新 | **全部**通過 `channel.remodel()` — C2 公式 |
| 參數 | 允許 `z_base`, `z_coeff`, `severity`, `treatment_factor` |
| 態度 | 「腳本化假說」— 有用的工程模型，非物理湧現 |
| 驗證方式 | C2 驅動 Z 演化、C1 每 tick 成立、但行為仍由參數決定 |

**範例（當前工廠疾病）**：
```python
Hypertension = make_template_disease(
    "Hypertension", z_base=50, z_coeff=0.3,
    default_severity=0.5, treatment_factor=0.5, ...
)
# C2 驅動 Z，但 z_coeff=0.3 是「人類寫的處方」
```

**E1 是當前的工程現實。** 12 個工廠疾病達到此等級。

---

### E2 — 腳本模型（Scripted Model）

> **定義**：Z 更新不遵循 C2，使用 ad-hoc 公式、直接賦值、
> 或條件分支。行為完全由程式碼邏輯決定。

| 條件 | 說明 |
|:-----|:-----|
| Z 更新 | `z *= 0.995`、`z = f(severity)`、`z += (t - z) * r` |
| 旗標 | `is_ptsd=True`、`stage_override`、... |
| 態度 | 「展示用原型」— 可用於演示，不代表物理湧現 |
| 目標 | 逐步遷移至 E1（使用 ImpedanceChannel），長期目標 E0 |

**E2 是目前 ~90 個唯一疾病的現狀。** 需要逐步遷移。

---

## 判定流程圖

```
                    你的疾病模型
                        │
            ┌───────────┴───────────┐
            │ Z 更新是否全部通過     │
            │ channel.remodel()?     │
            └───────┬───────┬───────┘
                   YES     NO
                    │       │
                    │       └──→ 🔴 E2 腳本模型
                    │
            ┌───────┴───────────┐
            │ 是否需要疾病專屬   │
            │ 參數 (z_coeff等)?  │
            └───────┬───────┬───┘
                   YES     NO
                    │       │
                    │       └──→ 🟢 E0 真湧現
                    │
                    └──→ 🟡 E1 參數化湧現
```

---

## 與不可約維度定理的關係

Paper 0 證明了：

> 異質阻抗網路存在幾何代價下界 $A_{\rm cut}$，無法被學習消除。
> 中繼節點的出現是熱力學必然。

**這意味著**：
- 某些 Γ 不可能歸零 → 系統永遠帶有殘餘反射
- 殘餘反射模式（pattern）由拓撲決定，不由參數決定
- **E0 湧現的疾病**應當是不可約維度定理的直接後果：
  拓撲限制 → 特定 Γ 模式 → 觀察到「疾病」

## 神經同軸線模型（Neural Cable Model）

> 根據 Γ-Net 物理學，神經元 = 同軸傳輸線。
> 同一條 ImpedanceChannel，同一個 C2，
> Γ 從 0 → 1 的連續光譜映射整個神經病理學。

```
Γ ≈ 0.0   健康匹配        → 正常傳導
Γ ≈ 0.1   熱雜訊升高      → 發燒、發炎
Γ ≈ 0.3   反射回饋迴路    → PTSD（反覆紀錄重播）
Γ ≈ 0.7   通道逐步崩潰    → ALS（進行性傳導衰退）
Γ → 1.0   開路 + 回波     → 幻肢痛（源頭斷裂但反射仍在）
```

**關鍵洞見**：這四種疾病不是四個不同的模型——
它們是**同一條物理軸**（Γ）上的四個點。
差異僅在：初始條件 + 外部事件序列。

如果這個模型能夠僅從 C2 + 不同初始條件產出這四種行為，
它就是 **E0 真湧現**的第一個參考實現。

---

## 開發規則

### 對 AI 助手（Copilot / Claude / GPT）

1. **所有新疾病模型必須標明湧現等級**（E0 / E1 / E2）
2. **E1 是最低可接受標準**（Z 更新全部通過 `channel.remodel()`）
3. **新模型不得為 E2**（ad-hoc Z 更新）— 舊 E2 可以保留但須註記
4. **不得為了通過測試而添加疾病專屬旗標**
5. **追求 E0**：如果你發現某行為可以從 C2 + 初始條件自然產生，
   移除多餘的參數

### 對人類開發者

1. 每個新模型的 PR 必須在 docstring 標註湧現等級
2. E0 模型需附帶「只改初始條件就能產生不同行為」的測試
3. E1 → E0 的遷移是持續性目標
4. 90 個 E2 模型的遷移計劃見 OPEN_ISSUES.md ISSUE-2

---

## 當前狀態

| 模組 | E0 | E1 | E2 | 合計 | E1 疾病名 |
|:-----|:--:|:--:|:--:|:----:|:----------|
| Cardiology | 0 | 0 | 10 | 10 | — |
| Endocrinology | 0 | 2 | 8 | 10 | Acromegaly, ThyroidStorm |
| ENT | 0 | 2 | 8 | 10 | Tinnitus, Sinusitis |
| Gastroenterology | 0 | 0 | 9 | 9 | — |
| Dermatology | 0 | 3 | 7 | 10 | Psoriasis, Urticaria, Acne |
| Ophthalmology | 0 | 1 | 9 | 10 | DryEye |
| Obstetrics | 0 | 1 | 8 | 9 | GDM |
| Pulmonology | 0 | 0 | 9 | 9 | — |
| Orthopedics | 0 | 1 | 9 | 10 | Tendinitis |
| Oncology | 0 | 3 | 8 | 11 | LungCancer, CRC, PancreaticCancer |
| Immunology | 0 | 2 | 8 | 10 | RA, AllergicRhinitis |
| **合計** | **0** | **15** | **102** | **117** | — |

### Ad-hoc Z 更新類型（E2）

| 類型 | 模式 | 數量 | 範例 |
|:----:|:-----|-----:|:-----|
| TYPE A | `z = f(severity)` | ~7 | 直接函數賦值 |
| TYPE B | `z *= (1 ± rate)` | ~40 | 指數衰減/增長 |
| TYPE C | `z += (target − z) × rate` | ~20 | 線性內插 |
| TYPE D | `z = base × (1 + param × coeff)` | ~35 | 參數化公式 |

---

## 引用

> *「在 Γ-Net 的理想版本裡，疾病行為必須是從 C2 ＋ 不可約維度定理
> 測試出來，而不是靠疾病專屬參數調出來。」*
>
> *「如果一個現象需要特殊旗標、客製化 C2 變體、或手動 Γ 閾值，
> 它只能被標記為『腳本化假說』，而不是『從 C2 湧現』。」*

---

*最後更新：2026-03-11*
*文件版本：v1.0*
