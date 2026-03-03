# Lab-Γ 診斷引擎規格書
# Lab-Γ Diagnostic Engine Specification

**版本 Version**: v0.1 Draft — 2026-03-03  
**作者 Author**: Hsi-Yu Huang  
**狀態 Status**: Design Phase → Phase 1 Implementation

---

## 1. 系統概述 System Overview

### 1.1 問題陳述 Problem Statement

患者在醫院完成抽血後，通常需要等待數天才能取得報告解讀。  
Lab-Γ 診斷引擎將檢驗數值即時轉換為 12 維器官阻抗向量，  
與 120 個疾病 Γ 模板進行模式匹配，在秒級內給出鑑別診斷。

### 1.2 核心原理 Core Principle

```
Lab Values → Z Mapping → Γ Calculation → Disease Template Matching → Differential Diagnosis
                                                                          │
                                              Doctor Feedback ◄────────────┘
                                                   │
                                              C2 Hebbian Update → Z Mapping (closed loop)
```

每個檢驗值偏離正常範圍 = 對應器官的阻抗偏離正常值。  
多個器官的 Γ 組合形成獨特的「疾病指紋」，對應到 120 個已知疾病模板。

### 1.3 物理基礎 Physics Foundation

$$\Gamma_{organ} = \frac{Z_{patient} - Z_{normal}}{Z_{patient} + Z_{normal}}$$

- $\Gamma = 0$：完美匹配，該器官系統正常
- $\Gamma \to \pm 1$：嚴重失配，該器官系統異常
- $|\Gamma|^2$：反射功率比例 = 「疾病嚴重度」的物理量度

**能量守恆 (C1)**：$\Gamma^2 + T = 1$，器官傳輸效率 $T = 1 - \Gamma^2$

---

## 2. 12 器官系統定義 Organ System Definitions

| # | 系統 ID | 中文名 | 英文名 | Z_normal (Ω) | 主要檢驗值來源 |
|---|---------|--------|--------|--------------|---------------|
| 1 | `cardiac` | 心血管 | Cardiovascular | 50 | Trop, BNP, CK-MB, HR, BP |
| 2 | `pulmonary` | 肺臟 | Pulmonary | 60 | SpO₂, PaO₂, PaCO₂, pH |
| 3 | `hepatic` | 肝臟 | Hepatic | 65 | AST, ALT, ALP, GGT, Bil, Alb, PT |
| 4 | `renal` | 腎臟 | Renal | 70 | Cr, BUN, eGFR, K, Na, UACR |
| 5 | `endocrine` | 內分泌 | Endocrine | 75 | TSH, FT4, Cortisol, HbA1c, Glucose |
| 6 | `immune` | 免疫 | Immune | 75 | WBC, CRP, ESR, PCT, Ig, CD4 |
| 7 | `heme` | 血液 | Hematologic | 55 | Hb, RBC, MCV, Plt, Retic, Ferritin |
| 8 | `GI` | 腸胃 | Gastrointestinal | 65 | Amylase, Lipase, Alb, Stool OB |
| 9 | `vascular` | 血管 | Vascular | 45 | TC, LDL, HDL, TG, HbA1c, CRP, Hcy |
| 10 | `bone` | 骨骼 | Musculoskeletal | 120 | Ca, P, ALP, Vit-D, UA, CRP |
| 11 | `neuro` | 神經 | Neurological | 80 | Glu, Na, Ca, NH₃, Lactate, B12 |
| 12 | `repro` | 生殖 | Reproductive | 95 | FSH, LH, E2, Testosterone, βhCG, AMH |

---

## 3. Lab → Z 映射表 Lab-to-Impedance Mapping

### 3.1 正常化公式 Normalization

每個檢驗值先正常化為無量綱偏差 $\delta$：

**線性正常化 (大多數項目)**：
$$\delta = \frac{x - \mu}{\sigma}$$

**分段飽和正常化 (有上下界的項目)**：
$$\delta = \begin{cases}
\text{clip}(\frac{x - ref_{low}}{ref_{low} - critical_{low}}, -1, 0) & \text{if } x < ref_{low} \\
0 & \text{if } ref_{low} \leq x \leq ref_{high} \\
\text{clip}(\frac{x - ref_{high}}{critical_{high} - ref_{high}}, 0, 1) & \text{if } x > ref_{high}
\end{cases}$$

### 3.2 檢驗值 → 器官 Z 偏移 Lab-to-Z Shift

$$Z_{organ} = Z_{normal} \times \left(1 + \sum_j w_j \cdot |\delta_j| \right)$$

其中 $w_j$ 是檢驗值 $j$ 對該器官的權重。

### 3.3 完整映射表 Full Mapping Table

#### CBC 全血球計數

| 檢驗項目 | 單位 | 參考區間 (成人) | critical_low | critical_high | 主要器官 (權重) |
|---------|------|----------------|-------------|--------------|----------------|
| WBC | 10³/μL | 4.0–11.0 | 1.0 | 30.0 | immune(0.40), heme(0.20) |
| RBC | 10⁶/μL | M:4.5–5.5, F:4.0–5.0 | 2.0 | 7.0 | heme(0.35), cardiac(0.10) |
| Hemoglobin (Hb) | g/dL | M:13.5–17.5, F:12.0–16.0 | 6.0 | 20.0 | heme(0.40), cardiac(0.15), neuro(0.05) |
| Hematocrit (Hct) | % | M:38–50, F:36–44 | 20 | 60 | heme(0.30) |
| MCV | fL | 80–100 | 60 | 120 | heme(0.25) |
| Platelets (Plt) | 10³/μL | 150–400 | 20 | 1000 | heme(0.30), hepatic(0.10), immune(0.10) |
| Neutrophils | % | 40–70 | 5 | 90 | immune(0.30) |
| Lymphocytes | % | 20–40 | 5 | 80 | immune(0.30) |

#### BMP 基礎代謝

| 檢驗項目 | 單位 | 參考區間 | critical_low | critical_high | 主要器官 (權重) |
|---------|------|---------|-------------|--------------|----------------|
| Sodium (Na) | mEq/L | 136–145 | 120 | 160 | renal(0.30), neuro(0.25) |
| Potassium (K) | mEq/L | 3.5–5.0 | 2.5 | 6.5 | renal(0.30), cardiac(0.25) |
| Chloride (Cl) | mEq/L | 98–106 | 80 | 120 | renal(0.20) |
| CO₂ (Bicarb) | mEq/L | 22–29 | 10 | 40 | renal(0.20), pulmonary(0.15) |
| BUN | mg/dL | 7–20 | 2 | 100 | renal(0.35), hepatic(0.10) |
| Creatinine (Cr) | mg/dL | M:0.7–1.3, F:0.6–1.1 | 0.2 | 10.0 | renal(0.45) |
| Glucose (Glu) | mg/dL | 70–100 (fasting) | 40 | 500 | endocrine(0.40), neuro(0.15), vascular(0.10) |
| Calcium (Ca) | mg/dL | 8.5–10.5 | 6.0 | 14.0 | bone(0.25), neuro(0.15), endocrine(0.10) |

#### LFT 肝功能

| 檢驗項目 | 單位 | 參考區間 | critical_low | critical_high | 主要器官 (權重) |
|---------|------|---------|-------------|--------------|----------------|
| AST (GOT) | U/L | 10–40 | — | 1000 | hepatic(0.35), cardiac(0.10) |
| ALT (GPT) | U/L | 7–56 | — | 1000 | hepatic(0.40) |
| ALP | U/L | 44–147 | — | 1000 | hepatic(0.20), bone(0.20) |
| GGT | U/L | M:8–61, F:5–36 | — | 500 | hepatic(0.25) |
| Total Bilirubin | mg/dL | 0.1–1.2 | — | 20.0 | hepatic(0.30), heme(0.15) |
| Direct Bilirubin | mg/dL | 0.0–0.3 | — | 10.0 | hepatic(0.25) |
| Albumin (Alb) | g/dL | 3.5–5.0 | 1.5 | — | hepatic(0.25), renal(0.10), GI(0.15) |
| Total Protein | g/dL | 6.0–8.3 | 3.0 | 12.0 | hepatic(0.15), immune(0.15) |
| PT / INR | sec / ratio | 11–13.5 / 0.8–1.2 | — | 5.0 (INR) | hepatic(0.25), heme(0.15) |

#### TFT 甲狀腺功能

| 檢驗項目 | 單位 | 參考區間 | critical_low | critical_high | 主要器官 (權重) |
|---------|------|---------|-------------|--------------|----------------|
| TSH | mIU/L | 0.4–4.0 | 0.01 | 50.0 | endocrine(0.45) |
| Free T4 | ng/dL | 0.8–1.8 | 0.1 | 5.0 | endocrine(0.35) |
| Free T3 | pg/mL | 2.3–4.2 | 0.5 | 10.0 | endocrine(0.25) |

#### Lipid Panel 脂質

| 檢驗項目 | 單位 | 參考區間 | critical_low | critical_high | 主要器官 (權重) |
|---------|------|---------|-------------|--------------|----------------|
| Total Cholesterol | mg/dL | <200 | — | 400 | vascular(0.25) |
| LDL | mg/dL | <130 | — | 300 | vascular(0.35), cardiac(0.10) |
| HDL | mg/dL | >40 (M), >50 (F) | 15 | — | vascular(0.25) (inverse: low=bad) |
| Triglycerides | mg/dL | <150 | — | 1000 | vascular(0.20), endocrine(0.10) |

#### Inflammatory / Cardiac Markers 發炎/心臟

| 檢驗項目 | 單位 | 參考區間 | critical_low | critical_high | 主要器官 (權重) |
|---------|------|---------|-------------|--------------|----------------|
| CRP | mg/L | <3.0 | — | 200 | immune(0.30), vascular(0.10), bone(0.05) |
| ESR | mm/hr | M:<15, F:<20 | — | 120 | immune(0.25) |
| Procalcitonin (PCT) | ng/mL | <0.05 | — | 100 | immune(0.35) |
| Troponin I | ng/mL | <0.04 | — | 50 | cardiac(0.50) |
| BNP | pg/mL | <100 | — | 5000 | cardiac(0.40) |
| CK-MB | U/L | <25 | — | 300 | cardiac(0.30) |
| D-Dimer | ng/mL | <500 | — | 10000 | vascular(0.20), pulmonary(0.15) |

#### Metabolic / Other 代謝/其他

| 檢驗項目 | 單位 | 參考區間 | critical_low | critical_high | 主要器官 (權重) |
|---------|------|---------|-------------|--------------|----------------|
| HbA1c | % | 4.0–5.6 | — | 14.0 | endocrine(0.40), vascular(0.15) |
| Uric Acid | mg/dL | M:3.5–7.2, F:2.6–6.0 | — | 15.0 | bone(0.20), renal(0.15) |
| Ferritin | ng/mL | M:20–500, F:20–200 | 5 | 2000 | heme(0.30), hepatic(0.10) |
| Vitamin D | ng/mL | 30–100 | 5 | — | bone(0.25), immune(0.10) |
| Vitamin B12 | pg/mL | 200–900 | 100 | — | neuro(0.20), heme(0.15) |
| Folate | ng/mL | >3.0 | 1.0 | — | heme(0.15) |
| Amylase | U/L | 28–100 | — | 1000 | GI(0.35) |
| Lipase | U/L | 0–160 | — | 1000 | GI(0.40) |
| Ammonia (NH₃) | μg/dL | 15–45 | — | 200 | hepatic(0.20), neuro(0.20) |
| Lactate | mmol/L | 0.5–2.2 | — | 10.0 | neuro(0.15), cardiac(0.10) |
| Homocysteine | μmol/L | 5–15 | — | 50 | vascular(0.20), neuro(0.10) |

#### Reproductive 生殖 (選填)

| 檢驗項目 | 單位 | 參考區間 | critical_low | critical_high | 主要器官 (權重) |
|---------|------|---------|-------------|--------------|----------------|
| FSH | mIU/mL | F-follicular:3–10 | — | 100 | repro(0.35) |
| LH | mIU/mL | F-follicular:2–15 | — | 100 | repro(0.30) |
| Estradiol (E2) | pg/mL | F-follicular:30–120 | — | 500 | repro(0.25) |
| βhCG | mIU/mL | <5 (非孕) | — | 100000 | repro(0.45) |
| AMH | ng/mL | 1.0–10.0 | 0.1 | — | repro(0.30) |
| Testosterone | ng/dL | M:300–1000 | 50 | 1500 | repro(0.30), endocrine(0.10) |

**合計：53 個檢驗項目 → 12 器官系統**

---

## 4. Γ 計算引擎 Gamma Calculation Engine

### 4.1 單器官 Γ 計算

```python
def compute_organ_gamma(z_patient: float, z_normal: float) -> float:
    """Γ = (Z_patient - Z_normal) / (Z_patient + Z_normal)"""
    return (z_patient - z_normal) / (z_patient + z_normal)
```

### 4.2 12 維 Γ 向量

```python
@dataclass
class PatientGammaVector:
    cardiac: float    = 0.0
    pulmonary: float  = 0.0
    hepatic: float    = 0.0
    renal: float      = 0.0
    endocrine: float  = 0.0
    immune: float     = 0.0
    heme: float       = 0.0
    GI: float         = 0.0
    vascular: float   = 0.0
    bone: float       = 0.0
    neuro: float      = 0.0
    repro: float      = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([...])  # 12D vector
    
    @property
    def total_gamma_squared(self) -> float:
        """ΣΓ² — 全身疾病負擔 Total disease burden"""
        return sum(g**2 for g in self.to_array())
    
    @property
    def health_index(self) -> float:
        """H = Π(1 - Γ_i²) — 全身健康指數"""
        return np.prod(1 - self.to_array()**2)
```

### 4.3 數值範例 Worked Example

**案例：急性肝炎患者**

| 檢驗 | 值 | 正常 | δ | 主要器官 |
|------|-----|------|---|---------|
| AST | 480 U/L | 25 | +4.75σ | hepatic |
| ALT | 520 U/L | 31 | +4.89σ | hepatic |
| Bil | 3.2 mg/dL | 0.65 | +4.64σ | hepatic |
| Alb | 2.8 g/dL | 4.25 | −1.93σ | hepatic |
| PT INR | 1.8 | 1.0 | +2.00σ | hepatic |

計算：
$$Z_{hepatic} = 65 \times (1 + 0.35 \times |4.75| + 0.40 \times |4.89| + 0.30 \times |4.64| + 0.25 \times |1.93| + 0.25 \times |2.00|)$$
$$Z_{hepatic} = 65 \times (1 + 1.66 + 1.96 + 1.39 + 0.48 + 0.50) = 65 \times 6.99 = 454\,\Omega$$

$$\Gamma_{hepatic} = \frac{454 - 65}{454 + 65} = \frac{389}{519} = 0.750$$

其他器官 Γ 較低（cardiac 可能 0.05 因 AST 有小權重），形成「肝臟主導」的 Γ 模式。

---

## 5. 疾病 Γ 模板 Disease Gamma Templates

### 5.1 模板結構

```python
@dataclass
class DiseaseTemplate:
    disease_id: str           # e.g. "hepatitis_acute"
    specialty: str            # e.g. "gastroenterology"
    display_name: str         # e.g. "急性肝炎 Acute Hepatitis"
    gamma_signature: dict[str, float]  # 12 organ Γ expected values
    primary_organs: list[str] # dominant organs
    clinical_scale: str       # e.g. "ALT level / HAI score"
    key_labs: list[str]       # most discriminating lab values
    severity_thresholds: dict  # Γ → mild/moderate/severe
    suggested_tests: list[str] # tests to confirm/exclude
```

### 5.2 模板範例 Template Examples

```json
{
  "hepatitis_acute": {
    "specialty": "gastroenterology",
    "display_name": "急性肝炎 Acute Hepatitis",
    "gamma_signature": {
      "cardiac": 0.05, "pulmonary": 0.02, "hepatic": 0.75,
      "renal": 0.08, "endocrine": 0.05, "immune": 0.25,
      "heme": 0.10, "GI": 0.15, "vascular": 0.03,
      "bone": 0.02, "neuro": 0.08, "repro": 0.00
    },
    "primary_organs": ["hepatic", "immune"],
    "suggested_tests": ["Hepatitis A/B/C serology", "Liver US", "Ceruloplasmin"]
  },
  "myocardial_infarction": {
    "specialty": "cardiology",
    "display_name": "心肌梗塞 Myocardial Infarction",
    "gamma_signature": {
      "cardiac": 0.85, "pulmonary": 0.15, "hepatic": 0.10,
      "renal": 0.08, "endocrine": 0.10, "immune": 0.20,
      "heme": 0.05, "GI": 0.03, "vascular": 0.35,
      "bone": 0.02, "neuro": 0.05, "repro": 0.00
    },
    "primary_organs": ["cardiac", "vascular"],
    "suggested_tests": ["ECG", "Serial Troponin", "Coronary angiography"]
  },
  "diabetic_ketoacidosis": {
    "specialty": "endocrinology",
    "display_name": "糖尿病酮酸中毒 DKA",
    "gamma_signature": {
      "cardiac": 0.15, "pulmonary": 0.10, "hepatic": 0.05,
      "renal": 0.35, "endocrine": 0.80, "immune": 0.10,
      "heme": 0.05, "GI": 0.10, "vascular": 0.20,
      "bone": 0.05, "neuro": 0.30, "repro": 0.00
    },
    "primary_organs": ["endocrine", "renal", "neuro"],
    "suggested_tests": ["Blood gas (pH, AG)", "Urine ketones", "Serum osmolality"]
  }
}
```

### 5.3 120 個疾病的 Γ 特徵概覽

每個臨床專科已有 10 個疾病模型 (見 `SYSTEM_MANUAL.md §6`)，
Γ 模板需定義每個疾病在 12 器官系統的期望 Γ 值。

**建模依據**：
- 主要受累器官 Γ > 0.5（嚴重失配）
- 次要受累器官 Γ = 0.1–0.3（中度影響）
- 無關器官 Γ < 0.05（正常）
- 系統性疾病（如 SLE、Sepsis）：多器官 Γ 普遍升高

---

## 6. 匹配演算法 Matching Algorithm

### 6.1 距離度量 Distance Metric

**加權歐氏距離 (主要方法)**：
$$d(P, D_k) = \sqrt{\sum_{i=1}^{12} w_i \cdot (\Gamma_{P,i} - \Gamma_{D_k,i})^2}$$

其中 $w_i$ 是器官系統的全域權重（預設=1，可由回饋調整）。

**相似度轉換**：
$$S_k = \frac{1}{1 + d(P, D_k)}$$

**信心度 Confidence**：
$$C_k = \frac{S_k}{\sum_{j=1}^{120} S_j}$$

### 6.2 輸出格式

```python
@dataclass
class DiagnosisCandidate:
    rank: int
    disease_id: str
    display_name: str
    specialty: str
    confidence: float          # 0–1
    distance: float            # Γ-space distance
    primary_deviations: list   # top contributing Γ deviations
    suggested_tests: list[str] # recommended follow-up
    severity: str              # mild / moderate / severe / critical
```

### 6.3 可解釋性 Explainability

每個診斷候選附帶：
1. **哪些器官 Γ 最匹配**：`"hepatic: patient=0.75, template=0.75 (match)"`
2. **哪些檢驗值貢獻最大**：`"AST=480 (↑12×), ALT=520 (↑17×), Bil=3.2 (↑5×)"`
3. **排除理由**：`"cardiac Γ=0.05 vs MI template=0.85 → mismatch"`

---

## 7. 閉環回饋 Closed-Loop Feedback (Phase 2)

### 7.1 回饋記錄格式

```python
@dataclass
class DoctorFeedback:
    timestamp: datetime
    patient_gamma_vector: PatientGammaVector
    lab_values: dict[str, float]
    engine_top5: list[str]          # engine's predictions
    confirmed_disease: str | None    # doctor's actual diagnosis
    overruled: list[str]             # wrong predictions
    missed: list[str]               # diseases engine missed
```

### 7.2 C2 Hebbian 更新 (未來)

$$\Delta w_j^{organ} = -\eta \cdot \Gamma_{error} \cdot x_{lab_j} \cdot x_{feedback}$$

- $\Gamma_{error}$：引擎預測 vs 實際診斷的 Γ 差異
- 醫師確認 → 強化該疾病模板
- 醫師修正 → 調整 Lab→Z 映射權重
- η 初始值 = 0.001（極保守，避免過度調整）

---

## 8. API 介面設計 API Interface

### 8.1 CLI 介面

```bash
# 輸入 JSON 檔
python -m alice.lab_diagnosis --input patient_labs.json

# 互動模式
python -m alice.lab_diagnosis --interactive
```

### 8.2 REST API (Phase 3)

```
POST /api/lab-diagnosis
Content-Type: application/json

Request:
{
  "patient_info": {"age": 45, "sex": "M"},
  "lab_values": {
    "AST": 480, "ALT": 520, "Bil_total": 3.2,
    "Alb": 2.8, "INR": 1.8, "WBC": 12.5, "CRP": 45
  }
}

Response:
{
  "gamma_vector": {
    "cardiac": 0.05, "pulmonary": 0.02, "hepatic": 0.75,
    ...
  },
  "total_gamma_squared": 0.68,
  "health_index": 0.42,
  "diagnoses": [
    {
      "rank": 1,
      "disease": "acute_hepatitis",
      "display_name": "急性肝炎 Acute Hepatitis",
      "specialty": "gastroenterology",
      "confidence": 0.72,
      "severity": "severe",
      "top_contributors": ["AST ↑12×", "ALT ↑17×", "Bil ↑5×"],
      "suggested_tests": ["Hep A/B/C serology", "Liver US"]
    },
    ...
  ]
}
```

---

## 9. 檔案結構 File Structure

```
alice/
├── diagnostics/                   # 新子系統
│   ├── __init__.py
│   ├── lab_mapping.py             # Lab → Z 映射引擎
│   ├── gamma_engine.py            # Γ 計算 + 匹配
│   ├── disease_templates.py       # 120 疾病 Γ 模板
│   ├── disease_templates.json     # 模板資料
│   ├── diagnosis_report.py        # 報告生成
│   └── feedback.py                # 回饋記錄 (Phase 2)
├── ...
tests/
├── test_lab_mapping.py
├── test_gamma_engine.py
├── test_disease_matching.py
└── ...
experiments/
├── exp_lab_diagnosis_demo.py      # CLI 示範
└── ...
```

---

## 10. 開發路線 Development Roadmap

| Phase | 內容 | 預期產出 |
|-------|------|---------|
| **Phase 1** | Lab→Z 映射 + Γ 引擎 + 120 模板 + CLI | 可運行原型 |
| **Phase 2** | 回饋閉環 + Hebbian 更新 | 自適應系統 |
| **Phase 3** | REST API + Web UI (Γ 雷達圖) | 用戶介面 |
| **Phase 4** | 臨床驗證 + 文獻案例對照 | 信效度報告 |
| **Phase 5** | Paper V: "Lab-Γ Diagnostic Engine" | 論文 |

---

## 11. 倫理與限制 Ethics & Limitations

1. **不是醫療設備**：本系統是研究工具，不替代醫師診斷
2. **參考區間簡化**：v0.1 使用通用成人值，未區分年齡/性別/種族
3. **權重為先驗值**：初始權重基於醫學文獻，尚未經大規模臨床驗證
4. **必須結合臨床**：Γ 匹配只是「提示」，最終診斷由醫師決定
5. **不含影像/病理**：此版本僅使用血液檢驗值，不包含影像或組織病理

---

*Lab-Γ Diagnostic Engine Specification v0.1 — 2026-03-03*  
*Physics-constrained clinical decision support through impedance matching*
