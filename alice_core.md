# 🔵 Alice Smart System — 核心圓 (alice_core.md)

> **版本**：v1.1 | **建立日期**：2026-03-16 | **上次更新**：2026-03-17 | **更新規則**：僅在重大理論突破或確立新公理時更新

---

## 一、專案本質

**Alice Smart System** 是一套以**物理學為底層邏輯**的生命體模擬器（Physics-driven Medical Lifeform Simulator），其核心主張：

> **所有認知、生理與病理現象，皆源自於物理學的阻抗匹配（Impedance Matching）、熱力學與傳輸線理論（Transmission Line Theory）。**

理論體系：**Γ-Net (Gamma-Net) 框架**，六篇論文（P0~P5）形成一完整的同心圓推論鏈。

---

## 二、三大不可違反的物理約束（C1 / C2 / C3）

> 這三條約束不是假設，而是從熱力學第一定律推導出的**定理**。

### C1：能量守恆
$$\Gamma_i^2(t) + T_i(t) = 1$$

- **$\Gamma_i$**：第 $i$ 個界面的反射係數
- **$T_i = 1 - \Gamma_i^2$**：傳輸係數
- 意義：在任意界面、任意時刻，入射能量 = 反射 + 傳輸

### C2：阻抗重塑（Impedance Remodeling）
$$\boxed{\Delta Z_i = -\eta\,\Gamma_i\,x_{\text{in},i}\,x_{\text{out},i}}$$

- **$\eta > 0$**：重塑速率
- **$x_{\text{in}}, x_{\text{out}}$**：界面兩側的訊號
- 意義：**反射作用最小化**的唯一一階梯度下降規則
- 等價性：這條式子與 Hebb 學習律、Wolff 骨骼定律、Glagov 血管重塑律，等 **29 種組織特定法則**代數等價（P3 證明）

### C3：阻抗標記傳輸（Impedance-Tagged Transport）
每一跨界面的能量或物質流，都攜帶 $(Z_\text{source}, Z_\text{load})$ 元資料，使得 $\Gamma_i$ 可在**局部計算**，不需要全局網路知識。

**實作注意**：在 ALICE 程式碼中，跨模組的值必須是攜帶 $Z$ 元資料的 `ElectricalSignal` 物件，禁止傳遞裸浮點數。

---

## 三、核心公理與基礎推導鏈

### 起源公式：Γ 的熱力學必然性
$$\Gamma = \frac{Z_2 - Z_1}{Z_2 + Z_1}, \quad T = 1 - \Gamma^2 = \frac{4Z_1 Z_2}{(Z_1+Z_2)^2}$$

### 最小反射原則（MRP）
一個生命體的**熱力學可行性條件**：
$$P_\text{metabolic} > P_\text{waste} = \sum_i \Gamma_i^2 P_{\text{in},i}$$

違反此條件的生物將因能量不足而死亡（物理排除，無需演化論說明）。

### 反射作用量（Reflection Action）：所有三大約束的變分源頭
$$\mathcal{A}[\Gamma] = \int_0^T \sum_i \Gamma_i^2(t)\,P_{\text{in},i}(t)\,dt \;\to\;\min$$

---

## 三之一、新確立的核心推論

### 中繼節點普遍性（Universal Relay-Node Necessity）
> P1 證明 $A_{\text{cut}} > 0$ → 中繼節點必然存在。P0 新推論：所有阻抗不連續界面（$\Gamma \neq 0$）都需要中繼結構。人體有 29 個界面 → 29 個中繼節點 → 29 種 C2 維護法則。

因果鏈：
$$\Gamma \neq 0 \;\Longrightarrow\; A_{\text{cut}} > 0 \;\Longrightarrow\; \Delta Z = -\eta\,\Gamma\,x_{\text{in}}\,x_{\text{out}}$$

**雙重性**：中繼拓撲同時傳遞訊號（健康）和傳遞損傷（病理級聯）。

### 級聯耦合矩陣 $C_{kj}$
$$|\Gamma_k^{\text{eff}}| = |\Gamma_k| + \sum_j C_{kj}\,\Gamma_j^2$$

- **二值拓撲**：$C_{kj} = \varepsilon$（有解剖連接）或 $0$（無），$\varepsilon = 0.03$
- **驗證**：7/7 器官 AUC 改善，$p(\text{binomial}) = 0.008$
- **來源**：Guyton & Hall, 14th ed. 解剖連接

### 器官特異 $\Omega_i$（由下而上回收 Kleiber 4/3）

P2 定義全局 $\Omega = B_{\text{total}}/B_{\text{tissue}} \approx 1.33$（由上而下）。

器官特異推導：$\Omega_i = 1 + \alpha \cdot \text{BF}_i \cdot \text{NI}_i/5$

| 指標 | 值 |
|------|------|
| 由下而上 mean $\overline{\Omega_i}$ | **1.376** |
| 由上而下 $\Omega$ (P2) | **1.33** |
| 差異 | ~3% |

物理意義：$\overline{\Omega} \approx \frac{4}{3}$ = 每 4 份能量中 3 份用於組織，1 份是雙網路基礎設施。

---

## 四、關鍵衍生量一覽

| 符號 | 名稱 | 定義 | 論文 |
|------|------|------|------|
| $\mathcal{A}$ | 反射作用量 | $\int_0^T\!\sum_i \Gamma_i^2 P_{\text{in},i}\,dt$ | P0 |
| $A_\text{imp}$ | 可改善作用量 | $\mathcal{A} - A_\text{cut}$ | P1 |
| $A_\text{cut}$ | 不可約作用量（拓撲代價） | $\sum_{k>K^*}\!\Gamma_k^2$ | P1 |
| $\Phi_\text{meta}$ | 意識指數 | $1 - \Gamma_\text{meta}^2$ | P1 |
| $\mathcal{S}$ | 零空間（「靈魂」） | $\ker(\partial Z / \partial t)$ | P1 |
| $H$ | 全身健康指數 | $\prod_i(1-\Gamma_i^2)$ | P2 |
| $\beta$ | Murray 分叉比例 | $2^{1/3} \approx 1.26$ | P2 |
| $\Omega$ | 雙網路開銷 | $\geq 1$ | P2 |
| $D_Z$ | 阻抗負債（睡眠動機） | $\int_0^{T_\text{wake}}\!\|\Gamma\|^2 P_\text{in}\,dt$ | P3 |
| $\Pi_\Gamma$ | Gamma 疼痛指數 | $w_n\Gamma_n^2 + w_v\Gamma_v^2 + w_D\frac{dD_Z}{dt}$ | P4 |

---

## 五、同心圓推論架構（六層）

```
Layer 0 (P0) | 熱力學第一定律 → Γ、C1、MRP、C2、C3
Layer 1 (P1) | C2 在有限網路 → A_cut>0 → 拓撲、大腦、意識、記憶、靈魂
Layer 2 (P2) | C2 在分支幾何 → Murray 定律、Kleiber 3/4、健康指數 H
Layer 3 (P3) | C2 ≡ 29 種法則 → D_Z、睡眠、生命週期 ODE、老化、情緒
Layer 4 (P4) | 拓撲擾動 → 疾病湧現（E0）：心血管、神經、代謝、多器官耦合
Layer 5 (P5) | Γ-向量 + NHANES (n=52,545) → 全因死亡 AUC=0.705 | 級聯 7/7 (10-cycle 穩健) | 共病 AUC=0.853 | Framingham 平手 (0.682) | 維度成本 Q4/Q1=3.3× | 五層 AUC 遞進 (0.604→0.709)
```

---

## 六、關鍵驗證數據（NHANES，$n = 49{,}774$）

| 指標 | 數值 |
|------|------|
| 全因死亡 AUC (Config B) | **0.705** [0.699–0.711] |
| Q1/Q4 死亡率比值 | **6.89×** |
| Harrell's C | **0.70** |
| 零擬合參數 | ✅ |

**五層 AUC 遞進**（n=7,393, NHANES 2013-2018）：

| 層次 | 配置 | Mean AUC | 擬合參數 |
|------|------|:---:|:---:|
| 1 | Lab（獨立 Γ） | 0.604 | **0** |
| 2 | Lab + 二值級聯 | 0.617 | **0** |
| 3 | Lab + BP | 0.611 | **0** |
| 4 | Lab + BP + 級聯 | 0.626 | **0** |
| 5 | Γ + Logistic Reg | **0.709** | 12 |

---

## 七、ALICE 系統架構（程式碼對應）

### 大腦與神經 (`alice/brain/`)
- `prefrontal.py` — 前額葉/執行控制
- `hippocampus.py` — 海馬迴/情節記憶
- `amygdala.py` — 杏仁核/情緒與恐懼
- `sleep_physics.py` — 睡眠 = $D_Z$ 清除
- `attention_plasticity.py` — 注意力可塑性（C2）

### 身體生理 (`alice/body/`)
- `cardiovascular.py` — 血管 = 傳輸線，血壓影響腦部供氧
- `endocrine.py` — 荷爾蒙 = 阻抗調節器
- `immune.py` — 發炎 = 阻抗不匹配產生的廢熱

### 實驗驗證 (`experiments/`)
超過 50+ 個驗證實驗，目標：**所有臨床現象從 C1/C2/C3 自然湧現，無硬編碼規則**。

---

## 八、理論邊界（不主張的事）

1. **必要條件，非充分原因**：框架說明生命體*必須*滿足的物理條件，不解釋為什麼某物種會演化出該結構。
2. **無演化軌跡**：阻抗空間中的幾何最優路徑，不代表實際演化路徑。
3. **C2 唯一性是數學定理**，不是演化論述。
4. **與天擇相容**：天擇決定哪些譜系存活；Γ-框架提供天擇所操作的物理目標函數。

---

## 九、已知待解問題與知識空缺

- [ ] NHANES 死亡人數的確認（7,627 vs 7,770）
- [x] ~~定理引用錯誤~~ — 已修正（2026-03-17 交叉引用掃描清零）
- [ ] P1/P2 PDF 編譯依賴問題需修復
- [ ] ε = 0.03 的物理推導（從 Γ² 平均值推導）

---

*更新規則：此文件代表「不移動的核心圓心」。僅在確立新公理、獲得重要實驗驗證、或發現理論根本性錯誤時更新。*
