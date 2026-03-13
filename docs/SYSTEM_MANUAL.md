# Alice Gamma-Net 系統說明手冊
# Alice Gamma-Net System Reference Manual

**版本 Version**: v31.1 — 2026-03-13  
**作者 Author**: Hsi-Yu Huang · Independent Researcher · Taiwan  
**授權 License**: AGPL-3.0 (code) / CC BY-NC-SA 4.0 (papers)

---

## 目次 Table of Contents

1. [核心物理 Core Physics](#1-核心物理-core-physics)
2. [系統架構 Architecture](#2-系統架構-architecture)
3. [核心引擎 Core Engines](#3-核心引擎-core-engines)
4. [大腦系統 Brain Systems](#4-大腦系統-brain-systems)
5. [身體系統 Body Systems](#5-身體系統-body-systems)
6. [臨床專科 Clinical Specialties](#6-臨床專科-clinical-specialties)
7. [數據流 Data Flow](#7-數據流-data-flow)
8. [REST API](#8-rest-api)
9. [實驗 Experiments](#9-實驗-experiments)
10. [論文對應 Paper Correspondence](#10-論文對應-paper-correspondence)
11. [統計 Statistics](#11-統計-statistics)

---

## 1. 核心物理 Core Physics

### 1.1 最小反射作用原理 Minimum Reflection Action Principle

Alice 的所有行為源自單一變分原理：  
All Alice behaviour derives from a single variational principle:

$$A[\Gamma] = \int_0^T \sum_i \Gamma_i^2(t)\,dt \;\to\; \min$$

其中反射係數 where the reflection coefficient:

$$\Gamma_i = \frac{Z_{load,i} - Z_{source,i}}{Z_{load,i} + Z_{source,i}}$$

### 1.2 三條不可違反的物理約束 Three Inviolable Constraints

| 約束 Constraint | 公式 Equation | 物理意義 Physical Meaning |
|---|---|---|
| **C1 能量守恆** Energy Conservation | $\Gamma^2 + T = 1$ | 每條通道、每個 tick，反射 + 透射 = 1 |
| **C2 Hebbian 更新** | $\Delta Z = -\eta \cdot \Gamma \cdot x_{pre} \cdot x_{post}$ | 對 $A[\Gamma]$ 作梯度下降 — 學習即阻抗匹配 |
| **C3 訊號協議** Signal Protocol | 所有模組間值必須是 `ElectricalSignal` | 無 Z 元數據 → Γ 未定義 → 物理無意義 |

### 1.3 不可約維度定理 Irreducible Dimensional Cost (Paper I)

$$A = A_{imp}(t) + A_{cut}$$

- $A_{imp}(t) = \sum \Gamma_{common}^2(t) \to 0$ — 可學習的阻抗失配（Hebbian 可消除）
- $A_{cut} = \sum (K_{src} - K_{tgt})^+$ — 不可消除的維度截斷代價
- **結論**: 任何有限帶寬通道的資訊傳輸必然有不可約殘留

### 1.4 雙生網路 Twin Networks (Paper II)

$$H = (1 - \Gamma_n^2)(1 - \Gamma_v^2)$$

- $\Gamma_n$：神經阻抗失配 Neural impedance mismatch
- $\Gamma_v$：血管阻抗失配 Vascular impedance mismatch
- 耦合係數 Coupling: $\alpha_{v \to n} = 0.5$, $\alpha_{n \to v} = 0.3$
- Murray's Law: $r_{parent}^3 = \sum r_{daughter}^3$ 使 $\sum\Gamma_v^2 \to \min$

### 1.5 阻抗債與睡眠 Impedance Debt & Sleep (Paper III)

$$D_{imp}(t) = \int_0^t \sum_i \Gamma_i^2(\tau)\,d\tau - \int_{sleep} R(\tau)\,d\tau$$

- 清醒時累積阻抗債（Γ² 熱累積）
- N3 慢波睡眠：突觸下調 $\times 0.95$，阻抗重正規化
- REM：情緒記憶處理 + 突觸修剪

### 1.6 生命週期方程 Lifecycle Equation (Paper IV)

$$\frac{d(\sum\Gamma^2)}{dt} = -\eta_{learn} \cdot \sum\Gamma^2 + \gamma_{novel} \cdot \Gamma_{env} + \delta_{aging} \cdot D(t)$$

三力競爭：學習（↓Γ）vs 新奇（↑Γ）vs 老化（↑Γ）→ 認知浴缸曲線

### 1.7 非線性物理模型 Nonlinear Physics (v31.1)

| 模型 Model | 公式 | 用途 Purpose |
|---|---|---|
| Butterworth 2nd-order | $H(f) = 1/\sqrt{1 + (f/f_c)^4}$ | 帶寬滾降 Bandwidth roll-off |
| Johnson-Nyquist 熱噪聲 | $N = k_B \cdot T \cdot \Delta f \cdot L \cdot (1 + \Gamma^2)$ | 通道噪聲 Channel noise |
| Arrhenius 老化 | $k = A \cdot e^{-E_a / k_B T}$ | 神經退化 Neural aging |
| Quemada 黏度 | $\eta(\dot\gamma) = \eta_\infty(1 + k/\sqrt{\dot\gamma})^2$ | 血液流變 Blood rheology |
| Autocorrelation 頻率估計 | $O(N)$ per tick | 訊號頻率辨識 |

---

## 2. 系統架構 Architecture

```
                     AliceBrain (2,951 行 — Layer 5 最高整合)
                         │
          ┌──────────────┼──────────────────┐
          │              │                  │
    FusionBrain     LifeLoop          SystemState
    (Neural+Protocol)  (閉環誤差補償)     (生命體徵)
          │              │                  │
    ┌─────┴─────┐   ┌────┴────┐      ┌─────┴─────┐
    │           │   │         │      │           │
  GammaNet   Neural  Sensor  Motor  Temperature  Pain
  V4Protocol  Base   Error   Cmd    Stability   Consciousness
    │                                    │
  ┌─┴───────────────────────┐      ┌────┴─────────┐
  │  PriorityRouter (O(1))  │      │  發燒方程     │
  │  YearRingCache (8 ring) │      │  dΘ/dt =      │
  │  BrainHemisphere (L/R)  │      │  α·ΣΓ² − β·Θ │
  │  ErrorCorrector         │      └──────────────┘
  └─────────────────────────┘

          42 Brain Engines              32 Body Modules
    ┌──────────────────────┐     ┌──────────────────────┐
    │ Thalamus → Cortex    │     │ Eye, Ear, Hand, Mouth│
    │ Amygdala → Prefrontal│     │ Lung, Cardio, Liver  │
    │ Hippocampus → Wernicke│    │ Kidney, Immune, Skin │
    │ BasalGanglia, Cerebel│     │ + 12 Clinical Specs  │
    │ Sleep, Pruning, Life │     │ + VascularImpedance  │
    └──────────────────────┘     └──────────────────────┘
```

### 2.1 目錄結構 Directory Layout

```
alice-gamma-net/
├── alice/              # 核心原始碼 Core source
│   ├── core/           #   物理引擎 Physics engines (6 files)
│   │   ├── signal.py          # ElectricalSignal + CoaxialChannel (764 行)
│   │   ├── protocol.py        # GammaNetV4Protocol (641 行)
│   │   ├── gamma_topology.py  # 動態 Γ-拓撲 (1,762 行)
│   │   ├── cache_analytics.py # 快取分析 (208 行)
│   │   └── cache_persistence.py # 快取持久化 (170 行)
│   ├── brain/          #   大腦引擎 Brain engines (42 files, ~29,000 行)
│   ├── body/           #   身體器官 Body organs (32 files, ~14,000 行)
│   ├── modules/        #   高階認知 Cognitive modules (5 files)
│   ├── api/            #   REST API + WebSocket (2 files)
│   ├── alice_brain.py  #   統一控制器 Unified controller (2,951 行)
│   └── main.py         #   CLI/API 入口 Entry point (240 行)
├── tests/              # pytest 測試套件 (92 files, 3,274 tests)
├── experiments/        # 實驗腳本 (73 files)
├── paper/              # 六篇論文 LaTeX source (6 .tex, 70 pp)
├── figures/            # 論文圖表 (18 figures)
└── docs/               # 文件 Documentation
```

---

## 3. 核心引擎 Core Engines

### 3.1 ElectricalSignal — 統一電氣訊號 (`alice/core/signal.py`)

所有模組間通訊的唯一合法載體。  
The only legal carrier for inter-module communication.

```python
class ElectricalSignal:
    """統一電氣訊號 — 攜帶 Z 元數據的同軸電纜訊號"""
    source: str              # 來源模組
    modality: str            # 感覺模態
    impedance: float         # Z_source (Ω)
    frequency: float         # 主頻率 (Hz)
    amplitude: float         # 振幅
    waveform: np.ndarray     # 波形數據
    phase: float             # 相位
```

**關鍵方法 Key Methods:**

| 方法 Method | 說明 |
|---|---|
| `from_raw(data, source, modality, impedance)` | 從原始數據自動分析頻率生成訊號 |
| `from_neural_activity(...)` | 從神經活動創建 |
| `band` → `BrainWaveBand` | δ/θ/α/β/γ 頻帶分類 |
| `power`, `energy`, `rms` | 功率 / 能量 / 均方根 |
| `attenuate(factor)` | 衰減 |
| `add_noise(noise_power)` | Johnson-Nyquist 熱噪聲 |

### 3.2 CoaxialChannel — 同軸電纜通道

模擬兩個腦區之間的物理傳輸線。  
Simulates the physical transmission line between two brain regions.

**6 步傳輸模擬 6-step transmission:**
1. 阻抗匹配計算 Impedance matching
2. 反射計算 Reflection: $P_r = \Gamma^2 \times P_{in}$
3. 衰減 Attenuation: $A(dB) = \alpha \times L \times f/f_{ref}$
4. 帶寬限制 Butterworth roll-off: $H(f) = 1/\sqrt{1+(f/f_c)^4}$
5. Johnson-Nyquist 熱噪聲
6. 相位延遲 Phase delay

**腦區阻抗映射 Region Impedance Map:**

| 通道 Channel | Z₀ (Ω) |
|---|---|
| Sensory → Prefrontal | 75 |
| Sensory → Limbic | 50 |
| Prefrontal → Motor | 75 |
| Limbic → Motor | 110 |

### 3.3 GammaTopology — 動態 Γ-拓撲 (`alice/core/gamma_topology.py`)

N 節點 K 維阻抗向量的活拓撲網路 — 1,762 行。

**8 種預設生物纖維 Tissue Types:**

| 名稱 | K (modes) | Z_mean (Ω) | 說明 |
|---|---|---|---|
| `CORTICAL_PYRAMIDAL` | 5 | 80 | 皮層錐體細胞 |
| `MOTOR_ALPHA` | 3 | 50 | Aα 運動神經元 |
| `SENSORY_AB` | 3 | 60 | Aβ 觸覺 |
| `PAIN_AD_FIBER` | 2 | 90 | Aδ 快痛 |
| `PAIN_C_FIBER` | 1 | 120 | C 纖維慢痛 |
| `CARDIAC_PURKINJE` | 4 | 40 | 心臟浦金野 |
| `AUTONOMIC_PREGANGLIONIC` | 2 | 100 | 自主神經 B |
| `ENTERIC_NEURON` | 1 | 110 | 腸神經系統 |

**核心方法 Key Methods:**

| 方法 | 說明 |
|---|---|
| `tick(external_stimuli)` | 核心演化迴路：計算 Γ → 傳輸 → Hebbian 更新 |
| `full_gamma_matrix()` | 完整 N×N Γ 矩陣 |
| `effective_adjacency(threshold)` | 湧現連通矩陣 (\|Γ\| < threshold = 有效連接) |
| `total_action()` | $A[\Gamma] = \int \sum\Gamma^2\,dt$ |
| `action_decomposition()` | $A = A_{imp} + A_{cut}$ 分解 |
| `insert_relay_nodes(...)` | 維度不匹配時自動插入中繼 |
| `box_counting_dimension()` | 盒計數分形維度 |
| `spectral_dimension()` | 譜維度 |
| `impedance_entropy()` | 阻抗熵 |

### 3.4 GammaNetV4Protocol — 通訊協議 (`alice/core/protocol.py`)

完整處理管線：  
`Signal → MessagePacket → PriorityRouter(O(1)) → YearRingCache(hit=零計算) → BrainHemisphere(需時啟動) → ErrorCorrector(最小修正) → 寫回快取`

**子系統 Subsystems:**

| 元件 | 說明 |
|---|---|
| `PriorityRouter` | 4 級 O(1) 路由器 (BACKGROUND/NORMAL/HIGH/CRITICAL) + 老化防飢餓 |
| `YearRingCache` | 8 層同心環記憶 + Fibonacci 整理（頻繁命中 → 內層） |
| `BrainHemisphere` | 左腦(序列/β-γ) + 右腦(並行/δ-θ-α) — 需時啟動 |
| `ErrorCorrector` | 最小差異修正 |

---

## 4. 大腦系統 Brain Systems

42 個引擎，分 10 類。所有引擎共享同一物理：$\Gamma = (Z_L - Z_0)/(Z_L + Z_0)$。

### 4.1 感覺處理 Sensory Processing

| 引擎 Engine | 檔案 | 核心功能 | 特徵方程 |
|---|---|---|---|
| **ThalamusEngine** | `thalamus.py` | 感覺閘門 — 所有感覺必須通過丘腦才達皮質 | $G_{total} = G_{arousal} \times (\alpha G_{top} + (1-\alpha) G_{bottom})$ |
| **PerceptionPipeline** | `perception.py` | LC 諧振辨識 O(1) — 非 FFT | Lorentzian 諧振曲線；頻段 δ/θ/α/β/γ |
| **AuditoryGroundingEngine** | `auditory_grounding.py` | 跨模態 Hebbian 接線 — 語言 = 遠端阻抗控制 | $\Gamma_{cross} = (Z_a - Z_v)/(Z_a + Z_v)$ |

### 4.2 語言系統 Language

| 引擎 | 檔案 | 核心功能 | 特徵方程 |
|---|---|---|---|
| **BrocaEngine** | `broca.py` | 語音運動編譯：概念 → F1/F2/F3 | $E_{artic} = 1 - \|\Gamma_{plan}\|^2$ |
| **WernickeEngine** | `wernicke.py` | 序列預測：低預測誤差 = 理解 | $\Gamma_{syn}(i,j) = 1 - P(j\|i)$ |
| **SemanticFieldEngine** | `semantic_field.py` | 概念 = 狀態空間中的共振吸引子 | $\Gamma_{sem} = 1 - sim(x,c_i)^{Q_i}$ |
| **SemanticPressureEngine** | `semantic_pressure.py` | 說話 = 語義壓力釋放 | $P_{sem} = \sum(mass \cdot valence^2 \cdot (1 - e^{-arousal}))$ |
| **RecursiveGrammarEngine** | `recursive_grammar.py` | Chomsky Merge 的阻抗實現 | $Merge\_cost(\alpha,\beta) = Z_\alpha \times Z_\beta$ |
| **NarrativeMemoryEngine** | `narrative_memory.py` | 自傳式因果敘事弧 | $\Gamma_{causal} = 1 - CausalStrength$ |

### 4.3 記憶與學習 Memory & Learning

| 引擎 | 檔案 | 核心功能 | 特徵方程 |
|---|---|---|---|
| **HippocampusEngine** | `hippocampus.py` | 時間綁定：跨模態快照 → 連貫情節 | LTP/LTD 動態容量 |
| **AttentionPlasticityEngine** | `attention_plasticity.py` | 訓練改變丘腦 RC 時間常數 | $gate\_\tau = C \times R$; $Q = \sqrt{L/C}/R$ |
| **CognitiveFlexibilityEngine** | `cognitive_flexibility.py` | 任務切換的暫態穩定 | $\tau_{reconfig}$ = 閘門重配延遲 |

### 4.4 情緒與動機 Emotion & Motivation

| 引擎 | 檔案 | 核心功能 | 特徵方程 |
|---|---|---|---|
| **AmygdalaEngine** | `amygdala.py` | 快速情緒 + 戰逃反應 (LeDoux 雙通路) | $\Gamma_{threat} = (Z_{sig} - Z_{threat})/(Z_{sig} + Z_{threat})$ |
| **EmotionGranularityEngine** | `emotion_granularity.py` | 8 維 Plutchik 情緒阻抗圖 | $Z_e = Z_0 \times (1 - E_i)$；8 基本情緒各有 $\kappa_i$ |
| **CuriosityDriveEngine** | `curiosity_drive.py` | 好奇/無聊/自我識別 — 自由意志 = 內部 Γ 驅動 | $\Gamma_{novelty} = \|Z_{in} - Z_{model}\|/(Z_{in} + Z_{model})$ |
| **HomeostaticDriveEngine** | `homeostatic_drive.py` | 下丘腦：飢餓/口渴/溫度 | $\Gamma_{hunger} = \|glucose - setpoint\| / setpoint$ |
| **PhysicsRewardEngine** | `physics_reward.py` | 阻抗匹配替代 Q-learning | $RPE = R_{actual} - (1-\Gamma^2) \cdot P_{in}$ |

### 4.5 運動系統 Motor

| 引擎 | 檔案 | 核心功能 | 特徵方程 |
|---|---|---|---|
| **BasalGangliaEngine** | `basal_ganglia.py` | Go/NoGo/Hyperdirect 三通路 + 習慣形成 | 習慣：$\Gamma_{action} \to 0$ 隨練習 |
| **Cerebellum** | `cerebellum.py` | 精密運動校準：攀爬纖維 = 誤差 | $\Delta Z_{purk} = -\eta \cdot \Gamma \cdot context \cdot error$ (C2) |
| **SpinalCord** | `spinal_cord.py` | 高速反射傳輸線 (30ms) | $T = 1 - \Gamma^2$ |

### 4.6 執行功能 Executive

| 引擎 | 檔案 | 核心功能 | 特徵方程 |
|---|---|---|---|
| **PrefrontalCortexEngine** | `prefrontal.py` | 目標管理 + 抑制 = 主動提升通道 Z | $E_{inhibition} = \int \Gamma_{block}^2\,dt$ |
| **MetacognitionEngine** | `metacognition.py` | System 1/2 切換 + 反事實推理 | $\Gamma_{thinking} = \sum(w_i \Gamma_i)/\sum w_i$ |
| **PredictiveEngine** | `predictive_engine.py` | 主動推論 — 最小化自由能 | $F = \|S_{sens} - S_{pred}\|^2/(2\sigma^2)$ |

### 4.7 意識與睡眠 Consciousness & Sleep

| 引擎 | 檔案 | 核心功能 | 特徵方程 |
|---|---|---|---|
| **AwarenessMonitor** | `awareness_monitor.py` | $\Phi$ 覺知指數 | $\Phi = (1/N)\sum(1 - \Gamma_i^2)$ |
| **NeuralActivityDisplay** | `neural_display.py` | 意識螢幕：每像素 = 一個通道終端 | 死像素=中風, 燒入=PTSD, 雪花=噪音, 黑屏=昏迷 |
| **SleepCycle** | `sleep.py` | Wake→N1→N2→N3→REM 90 分鐘 | N3 鞏固 / REM 修剪 |
| **SleepPhysicsEngine** | `sleep_physics.py` | 離線阻抗重正規化 | $D_{imp} += \sum\Gamma^2 \times \alpha_{fatigue}$ |
| **LifeLoop** | `life_loop.py` | 閉環誤差補償：Sensor→Error→PID→Motor | 反射能量 = 誤差的物理測量 |

### 4.8 自主神經與穩態 Autonomic

| 引擎 | 檔案 | 核心功能 | 特徵方程 |
|---|---|---|---|
| **AutonomicNervousSystem** | `autonomic.py` | 交感/副交感恆溫器 | 交感=加速, 副交感=穩壓 |
| **ImpedanceAdaptationEngine** | `impedance_adaptation.py` | 經驗驅動 Γ 改善 | $\eta_{eff} = \eta_{base} \times 4c(1-c)$ (Yerkes-Dodson) |
| **TemporalCalibrator** | `calibration.py` | 跨模態時間綁定 | $\|t_A - t_B\| < \Delta t$ → 綁定 |
| **GradientOptimizer** | `gradient_optimizer.py` | MRP 顯式變分梯度 | $\partial(\Gamma_i^2)/\partial Z_i = -4Z_L(Z_L-Z_i)/(Z_L+Z_i)^3$ |
| **PinchFatigueEngine** | `pinch_fatigue.py` | Pollock-Barraclough 電磁壓縮老化 | Coffin-Manson: $N_f = C/(\Delta\varepsilon)^\beta$ |

### 4.9 發育與臨床 Development & Clinical

| 引擎 | 檔案 | 核心功能 | 特徵方程 |
|---|---|---|---|
| **NeuralPruningEngine** | `pruning.py` | Γ→0 存活, Γ>>0 凋亡 | 存活 +5% / 凋亡 -5% Hebbian |
| **NeurogenesisThermalShield** | `neurogenesis_thermal.py` | 200B 神經元分散 Γ² 熱負載 | $q = Q_{total}/N$; $q_{birth} = 0.125$ |
| **FontanelleModel** | `fontanelle.py` | 囟門：Γ 場自組織窗口 | 閉合 = 嬰兒失憶症結束 |
| **BoneChinaEngine** | `bone_china.py` | 5 階段記憶鞏固 (陶瓷燒製) | Clay→Greenware→Bisque→Glaze→Porcelain |
| **PhantomLimbEngine** | `phantom_limb.py` | 截肢 = 開路 Γ=1 → 100% 反射 = 痛 | 鏡像治療: Γ 1.0→<0.3, VAS 7.2→2.1 |
| **ClinicalNeurologyEngine** | `clinical_neurology.py` | 5 大神經疾病統一模型 | Stroke/ALS/Dementia/Alzheimer's/CP |
| **PharmacologyEngine** | `pharmacology.py` | 每種藥 = 阻抗修改因子 α | $Z_{eff} = Z \times (1 + \alpha_{drug})$ |
| **LifecycleEquationEngine** | `lifecycle_equation.py` | 浴缸曲線三力方程 | 學習 vs 新奇 vs 老化 |

### 4.10 整合 Integration

| 引擎 | 檔案 | 核心功能 |
|---|---|---|
| **FusionBrain** | `fusion_brain.py` | 統一 v3 神經基底 + v4 協議系統；認知(70%)+情緒(30%)→運動 |

---

## 5. 身體系統 Body Systems

32 個模組，統一物理：$\Gamma = (Z_L - Z_0)/(Z_L + Z_0)$, $T = 1 - \Gamma^2$。

### 5.1 感覺器官 Sensory Organs (7)

| 器官 Organ | 檔案 | 功能 | Z (Ω) | 特色 |
|---|---|---|---|---|
| **AliceEye** | `eye.py` (757行) | 光子→視網膜 FFT→視神經 ElectricalSignal | 50 | 透鏡=傅立葉變換器；Nyquist 解析度檢查 |
| **AliceEar** | `ear.py` (409行) | 聲波→耳道共振→耳蝸 FFT→ElectricalSignal | 50 | 基底膜 tonotopic 分解 |
| **CochlearFilterBank** | `cochlea.py` (484行) | ERB 臨界帶分解；Gammatone 濾波器 | — | 24 通道, 80–8000Hz |
| **AliceNose** | `nose.py` (270行) | 嗅覺：分子阻抗匹配，繞過丘腦→杏仁核 | 45 | 20 受體類型 |
| **AliceSkin** | `skin.py` (297行) | 分布式阻抗感測陣列：觸覺/溫度/痛覺 | 60 | $Z_{skin}(T) = Z_0(1+\alpha_T(T-T_{ref}))$ |
| **VestibularSystem** | `vestibular.py` (292行) | 半規管(角速度) + 耳石(線加速度) | 120/200 | LC 共振慣性感測 |
| **InteroceptionOrgan** | `interoception.py` (330行) | 8 通道內感受：心跳/呼吸/飢餓/溫度... | — | $\Gamma_{intero} = \sum\Gamma_{organ}/N$ |

### 5.2 運動輸出 Motor Output (2)

| 器官 | 檔案 | 功能 | 特色 |
|---|---|---|---|
| **AliceHand** | `hand.py` (719行) | PID 運動控制 + 焦慮顫抖 | $u(t) = K_p e + K_i \int e\,dt + K_d \dot{e}$ |
| **AliceMouth** | `mouth.py` (442行) | Source-Filter 聲道模型 | 聲帶 120Hz + PID 音高追蹤 |

### 5.3 生命器官 Vital Organs (10)

| 器官 | 檔案 | 功能 | Z (Ω) | 臨床意義 |
|---|---|---|---|---|
| **AliceLung** | `lung.py` (439行) | LC 振盪呼吸 + 散熱 + 語音氣流 | — | 共振頻率 $= 1/(2\pi\sqrt{LC})$ |
| **CardiovascularSystem** | `cardiovascular.py` (714行) | 心臟泵 → 血管傳輸線 → 腦灌注 | — | $CO = HR \times SV$; 壓力反射 |
| **KidneySystem** | `kidney.py` (319行) | 腎小球阻抗濾波器 | 70 | $GFR = flow \times FF \times (1-\Gamma^2)$ |
| **LiverSystem** | `liver.py` (345行) | 代謝阻抗變壓器：解毒/醣原/膽紅素 | 65 | CYP450 藥物代謝 |
| **DigestiveSystem** | `digestive.py` (375行) | 腸-腦軸傳輸線：蠕動/吸收/迷走神經 | 65 | 90% 血清素在腸 |
| **EndocrineSystem** | `endocrine.py` (327行) | 激素級聯：HPA/HPT/HPG/GH/Insulin | — | 設定點伺服系統 |
| **ImmuneSystem** | `immune.py` (372行) | 固有+適應性免疫 + 細胞激素 | 75 | $\Gamma_{immune} = (Z_{pathogen}-Z_{self})/(Z_p+Z_s)$ |
| **LymphaticSystem** | `lymphatic.py` (231行) | 淋巴引流 + 免疫巡邏 | 55 | 水腫 = 引流不足 |
| **ReproductiveSystem** | `reproductive.py` (219行) | GnRH 振盪器網路 (發育用) | 95 | 青春期 = 振盪器啟動 |
| **VascularImpedanceNetwork** | `vascular_impedance.py` (787行) | Paper II 核心：血管阻抗匹配傳輸線 | 0.05–5.0 | Murray's Law; 雙 Γ-Net |

---

## 6. 臨床專科 Clinical Specialties

12 個專科，每科 10 個疾病模型，共 120 個。統一架構：  
每個疾病 = 特定的阻抗失配模式 (Γ signature)。  

### 6.1 心臟科 Cardiology (`clinical_cardiology.py`, 638 行)

| # | 疾病 Disease | Γ 簽章 | 量表 Scale |
|---|---|---|---|
| 1 | 心肌梗塞 MI | 冠狀動脈阻塞 → regional Γ→1.0 | Killip I–IV |
| 2 | 心衰竭 CHF | 泵 Z 失配 → CO↓ | NYHA I–IV |
| 3 | 心房顫動 AF | SA 節點 Z 振盪 → 不規則 RR | CHA₂DS₂-VASc |
| 4 | 高血壓 HTN | 小動脈 Z 慢性 ↑ | JNC Stage |
| 5 | 主動脈狹窄 AS | 瓣膜 Z 阻塞 → LV 負荷 | Gradient mmHg |
| 6 | 心肌病 DCM | 心肌 Z 結構改變 | LVEF % |
| 7 | 心包炎 Pericarditis | 心包 Z 變化 → 收縮性 | VAS 0–10 |
| 8 | 肺動脈高壓 PulmHTN | 肺血管 Z↑ | WHO Class |
| 9 | 心內膜炎 Endocarditis | 生物膜 Z 污染 | Duke Criteria |
| 10 | 主動脈剝離 Dissection | 管壁 Z 不連續 → 傳輸線撕裂 | Stanford A/B |

### 6.2 胸腔科 Pulmonology (`clinical_pulmonology.py`, 523 行)

| # | 疾病 | Γ 簽章 | 量表 |
|---|---|---|---|
| 1 | 氣喘 Asthma | 支氣管 Z 振盪 | FEV1/FVC |
| 2 | COPD | 漸進氣道 Z↑ | GOLD I–IV |
| 3 | 肺炎 Pneumonia | 肺泡 Z 充填（液體） | CURB-65 |
| 4 | 肺栓塞 PE | 血管阻塞 Γ→1.0 | Wells Score |
| 5 | 氣胸 Pneumothorax | 胸膜 Z 不連續 | Size % |
| 6 | 肺纖維化 Fibrosis | Z 漸進硬化 | FVC % pred |
| 7 | ARDS | 表面活性劑喪失 → Z 塌陷 | PaO₂/FiO₂ |
| 8 | 睡眠呼吸中止 OSA | 週期性氣道 Z 阻塞 | AHI |
| 9 | 肺癌 Lung CA | 局部 Z 浸潤 | TNM |
| 10 | 囊性纖維化 CF | 黏液 Z 升高 | FEV1 % |

### 6.3 腸胃科 Gastroenterology (`clinical_gastroenterology.py`, 510 行)

| # | 疾病 | Γ 簽章 | 量表 |
|---|---|---|---|
| 1 | GERD | LES Z 失效 → 逆行 Γ | LA Grade |
| 2 | 消化性潰瘍 PUD | 黏膜屏障 Z 崩壞 | Forrest |
| 3 | IBD (Crohn/UC) | 黏膜 Z 振盪（發炎） | CDAI/Mayo |
| 4 | IBS | 腸-腦軸 Γ 升高 | Rome IV |
| 5 | 肝硬化 Cirrhosis | 肝 Z 漸進 → 門脈高壓 | Child-Pugh |
| 6 | 膽結石 Cholelithiasis | 膽管 Z 阻塞 | Murphy sign |
| 7 | 胰腺炎 Pancreatitis | 胰管 Z 阻塞 → 自消化 | Ranson |
| 8 | 腸阻塞 Bowel Obstruction | 傳輸線 Z 不連續 | SBO grade |
| 9 | 大腸癌 CRC | 黏膜 Z 轉化 | TNM/CEA |
| 10 | B/C 型肝炎 Hepatitis | 病毒 Z 污染肝細胞 | ALT/Fibrosis |

### 6.4 腫瘤科 Oncology (`clinical_oncology.py`, 500 行)

癌症 = 阻抗偽裝：$Z_{tumor} \to Z_{host}$（免疫逃逸）

| # | 癌症 | Γ 簽章 | 量表 |
|---|---|---|---|
| 1 | 肺癌 | 氣道 Z 浸潤 | TNM/EGFR |
| 2 | 乳癌 | 乳管 Z 轉化 | TNM/ER-PR |
| 3 | 大腸癌 | 黏膜 Z 生長 | TNM/CEA |
| 4 | 肝癌 HCC | 肝 Z 轉化 | BCLC/AFP |
| 5 | 胰臟癌 | 胰管 Z 狹窄 | TNM/CA19-9 |
| 6 | 膠質母細胞瘤 GBM | 神經 Z 浸潤 + 水腫 | KPS/WHO |
| 7 | 白血病 | 骨髓 Z 接管 | FAB/WBC |
| 8 | 淋巴瘤 | 淋巴 Z 擴張 | Ann Arbor |
| 9 | 腎細胞癌 RCC | 實質 Z 侵犯 | TNM/IMDC |
| 10 | 轉移 Metastasis | Γ 偽裝 (Γ_tumor → Γ_host) | Sites/Burden |

### 6.5 免疫科 Immunology (`clinical_immunology.py`, 427 行)

| # | 疾病 | Γ 簽章 | 量表 |
|---|---|---|---|
| 1 | SLE | 自身耐受 Z 崩壞 | SLEDAI |
| 2 | RA | 關節 Z 攻擊 | DAS-28 |
| 3 | 過敏性休克 Anaphylaxis | IgE Z 級聯 → 系統 Γ≈1 | WAO Grade |
| 4 | 過敏性鼻炎 | 氣道 Z 過敏 | ARIA |
| 5 | HIV/AIDS | CD4⁺ Z 逐步破壞 | CD4 count |
| 6 | 敗血症 Sepsis | 系統 Z 風暴 → 多器官 | SOFA |
| 7 | 移植排斥 | 供體 Z 失配 → Γ | Banff Grade |
| 8 | 肉芽腫病 Sarcoidosis | 肉芽腫 Z 封裝 | Organ staging |
| 9 | 血管炎 Vasculitis | 血管 Z 發炎 | BVAS |
| 10 | 免疫缺陷 | 免疫通道 Z 衰減 | Ig levels |

### 6.6 眼科 Ophthalmology (`clinical_ophthalmology.py`, 415 行)

| # | 疾病 | Γ 簽章 | 量表 |
|---|---|---|---|
| 1 | 青光眼 Glaucoma | 視神經 Z 壓迫 (IOP) | IOP/VF MD |
| 2 | 白內障 Cataract | 水晶體 Z 混濁 | LOCS III |
| 3 | 視網膜剝離 RD | 光感受器 Z 不連續 | Area/Macula |
| 4 | AMD | 中央視網膜 Z 退化 | AREDS |
| 5 | 糖尿病視網膜病變 DR | 血管 Z 出血/新生 | ETDRS |
| 6 | 屈光不正 | 焦點 Z 失配 | Diopters |
| 7 | 乾眼症 Dry Eye | 淚膜 Z 破壞 | OSDI |
| 8 | 角膜潰瘍 | 表面 Z 失效 | Size/Depth |
| 9 | 視神經炎 | 神經 Z 脫髓鞘 | VA/RAPD |
| 10 | 斜視 Strabismus | 雙眼 Z 對齊誤差 | Prism Δ |

### 6.7 耳鼻喉科 ENT (`clinical_ent.py`, 407 行)

耳 = 三級 Z 變壓器：$Z_{air} \to ossicles \to Z_{cochlea}$

| # | 疾病 | Γ 簽章 | 量表 |
|---|---|---|---|
| 1 | 感音性聽損 SNHL | 耳蝸毛細胞 Z 退化 | PTA dB HL |
| 2 | 傳導性聽損 | 聽小骨鏈 Z 失配 | Air-Bone Gap |
| 3 | 梅尼爾氏症 | 內淋巴 Z 振盪 | AAO-HNS |
| 4 | 耳鳴 Tinnitus | 幻影 Z 信號 | THI |
| 5 | 中耳炎 | 中耳 Z 液體充填 | Tympanometry |
| 6 | 聲帶麻痺 | 喉 Z 開路 | VHI/GRBAS |
| 7 | 竇炎 | 竇 Z 阻塞 | Lund-Mackay |
| 8 | 嗅覺喪失 Anosmia | 嗅覺 Z 斷路 | UPSIT |
| 9 | 突發性聽損 SSHL | 急性耳蝸 Z 失效 | PTA recovery |
| 10 | BPPV | 耳石 Z 位移 | Dix-Hallpike |

### 6.8 皮膚科 Dermatology (`clinical_dermatology.py`, 412 行)

皮膚 = 第一 Z 界面：$Z_{SKIN} = 60\Omega$

| # | 疾病 | Γ 簽章 | 量表 |
|---|---|---|---|
| 1 | 異位性皮膚炎 | 屏障 Z 失效 (filaggrin) | SCORAD |
| 2 | 乾癬 Psoriasis | 角質細胞 Z 振盪 | PASI |
| 3 | 蕁麻疹 Urticaria | 組織胺 Z 突波 | UAS-7 |
| 4 | 帶狀皰疹 HZ | 皮節 Z 再活化 | VAS/PHN |
| 5 | 黑色素瘤 Melanoma | 黑色素細胞 Z 轉化 | Breslow/TNM |
| 6 | 接觸性皮膚炎 | 外源過敏 Z 失配 | Patch test |
| 7 | 痤瘡 Acne | 毛囊皮脂 Z 阻塞 | IGA |
| 8 | 白斑 Vitiligo | 黑色素細胞 Z 自體免疫 | VASI |
| 9 | 蜂窩性組織炎 | 皮下 Z 感染擴散 | Eron class |
| 10 | 燒傷 Burns | 熱 Z 級聯失效 | TBSA%/Depth |

### 6.9 內分泌科 Endocrinology (`clinical_endocrinology.py`, 404 行)

| # | 疾病 | Γ 簽章 | 量表 |
|---|---|---|---|
| 1 | 第一型糖尿病 T1DM | β 細胞 Z 自體免疫破壞 | C-peptide/HbA1c |
| 2 | 第二型糖尿病 T2DM | 胰島素阻抗 Z 失配 | HOMA-IR/HbA1c |
| 3 | 甲亢 Hyperthyroid | 代謝 Z 過度驅動 | FT4/TSH |
| 4 | 甲低 Hypothyroid | 代謝 Z 不足 | TSH/FT4 |
| 5 | 庫欣氏症 Cushing | 皮質醇 Z 過量 | 24h UFC |
| 6 | 艾迪森氏症 Addison | 皮質醇 Z 缺失 | AM cortisol |
| 7 | 嗜鉻細胞瘤 Pheo | 兒茶酚胺 Z 突波 | VMA/meta |
| 8 | 肢端肥大症 Acromegaly | GH Z 過量 | IGF-1 |
| 9 | DKA | 代謝級聯 Γ → 多器官 | pH/AG/Glucose |
| 10 | 甲狀腺風暴 | 正回饋 Γ 失控 | Burch-Wartofsky |

### 6.10 腎臟科 Nephrology (`clinical_nephrology.py`, 357 行)

| # | 疾病 | Γ 簽章 | 量表 |
|---|---|---|---|
| 1 | AKI | 瞬間濾過 Γ 跳升 | KDIGO Stage |
| 2 | CKD | 腎元 Z 漸進退化 | eGFR/CKD Stg |
| 3 | 腎結石 | 管道 Z 阻塞 | Stone mm |
| 4 | 腎病症候群 | 腎小球屏障 Z 滲漏 | Proteinuria |
| 5 | 腎炎症候群 | 發炎性腎小球 Z | Hematuria |
| 6 | 糖尿病腎病 | 葡萄糖介導 Z 漂移 | UACR/eGFR |
| 7 | 電解質異常 | 濾液 Z 失配 | Na/K/Ca |
| 8 | 腎性高血壓 | RAAS 回饋 Z 放大 | BP mmHg |
| 9 | 多囊腎 PKD | 結構 Z 扭曲 | Kidney vol |
| 10 | 腎小管酸中毒 RTA | 酸鹼 Z 失配 | pH/HCO₃ |

### 6.11 骨科 Orthopedics (`clinical_orthopedics.py`, 409 行)

骨骼 = 最高生物 Z (120Ω)

| # | 疾病 | Γ 簽章 | 量表 |
|---|---|---|---|
| 1 | 骨折 Fracture | 結構 Z 不連續（開路） | AO/OTA |
| 2 | 骨質疏鬆 Osteoporosis | 骨 Z 退化 | DXA T-score |
| 3 | 椎間盤突出 | 脊柱 Z 壓迫 | ODI |
| 4 | 骨關節炎 OA | 軟骨 Z 磨損 | K-L Grade |
| 5 | 前十字韌帶撕裂 ACL | 穩定 Z 失效 | IKDC |
| 6 | 肌腱炎 Tendinitis | 連接 Z 發炎 | VAS/DASH |
| 7 | 脊柱側彎 Scoliosis | 結構 Z 偏移 | Cobb angle |
| 8 | 骨肉瘤 | 骨 Z 轉化 | TNM/Enneking |
| 9 | 痛風 Gout | 結晶 Z 沉積 | Serum urate |
| 10 | 骨髓炎 Osteomyelitis | 骨 Z 感染 | Cierny-Mader |

### 6.12 婦產科 Obstetrics (`clinical_obstetrics.py`, 422 行)

| # | 疾病 | Γ 簽章 | 量表 |
|---|---|---|---|
| 1 | 子癇前症 Preeclampsia | 胎盤血管 Z 失效 | BP/Proteinuria |
| 2 | PCOS | HPG 振盪器 Z 鎖死 | Rotterdam/AMH |
| 3 | 子宮內膜異位 | 異位內膜 Z | rASRM |
| 4 | 子宮肌瘤 Fibroids | 結構 Z 扭曲 | FIGO/UFS-QOL |
| 5 | 早產 Preterm | 宮頸 Z 衰弱 | GA wks/CL |
| 6 | 妊娠糖尿病 GDM | 孕期胰島素 Z 漂移 | OGTT/Glucose |
| 7 | 卵巢癌 | 生殖腺 Z 轉化 | CA-125/FIGO |
| 8 | 更年期 Menopause | 雌激素 Z 缺失 | MRS/FSH |
| 9 | 羊水栓塞 AFE | 外源顆粒 Z 突波 | DIC/SOFA |
| 10 | 產後出血 PPH | 子宮弛緩 Z 失效 | EBL mL |

---

## 7. 數據流 Data Flow

### 7.1 perceive() 完整感知管線 (34 步)

```
外部刺激 External Stimulus
  │
  ├─ 1. ElectricalSignal 封裝 (C3)
  ├─ 2. 感覺器官前處理 (Eye.see / Ear.hear / Skin.touch)
  ├─ 3. 耳蝸指紋 (CochlearFilterBank.fingerprint)
  ├─ 4. MessagePacket 封裝 (freq_tag + hash)
  ├─ 5. PriorityRouter O(1) 分類
  ├─ 6. YearRingCache 查找 (命中 → 跳到 Step 30)
  │
  ├─ 7. ThalamusEngine.gate() — 感覺閘門
  ├─ 8. 自主神經 arousal → 丘腦增益
  ├─ 9. ImpedanceAdaptation → Γ 改善
  ├─10. TemporalCalibrator.bind() — 跨模態對齊
  │
  ├─11. PerceptionPipeline.recognize() — LC 諧振 O(1)
  ├─12. AuditoryGrounding → 跨模態 Hebbian
  ├─13. SemanticField.recognize() — 吸引子匹配
  ├─14. HippocampusEngine.encode() — 情節綁定
  │
  ├─15. AmygdalaEngine.assess_threat() — 快速情緒
  ├─16. EmotionGranularity.update() — 8D 情緒向量
  ├─17. SemanticPressure.accumulate() — 語義壓力
  ├─18. CuriosityDrive.compute_novelty() — 好奇心
  │
  ├─19. Wernicke.comprehend() — 序列預測/理解
  ├─20. RecursiveGrammar.parse() — 句法解析
  ├─21. NarrativeMemory.add_episode() — 敘事弧
  │
  ├─22. Prefrontal.evaluate_action() — 目標評估
  ├─23. Metacognition.evaluate() — System 1/2
  ├─24. PredictiveEngine.predict() — 主動推論
  │
  ├─25. BasalGanglia.select_action() — Go/NoGo
  ├─26. Cerebellum.correct_motor() — 精密校準
  ├─27. Hand.reach() / Mouth.speak() — 運動執行
  │
  ├─28. LifeLoop.estimate_errors() — 閉環誤差
  ├─29. LifeLoop.compensate() — PID 補償指令
  │
  ├─30. BrainHemisphere 處理 (L/R)
  ├─31. ErrorCorrector 最小修正
  ├─32. YearRingCache.store() — 寫回快取
  ├─33. ConsciousnessScreen.render() — 螢幕渲染: T_i = 1 − Γ_i²
  └─34. SystemState.tick() — 生命體徵更新 (溫度/痛覺/穩定/心率)
```

### 7.2 發燒方程 SystemState Fever Equation

$$\frac{d\Theta}{dt} = \alpha \cdot \sum\Gamma^2 - \beta \cdot \Theta \cdot (1 - p)$$

- $\alpha$ = 生熱係數 (Γ² 熱累積)
- $\beta$ = 副交感冷卻
- $p$ = 交感張力
- 佇列壓力→溫度↑→痛覺↑→穩定↓→意識模糊→心率異常 (THE PAIN LOOP)

---

## 8. REST API

FastAPI + WebSocket 即時串流 (`alice/api/server.py`, 1,264 行)

### 8.1 端點一覽

| 端點 Endpoint | 方法 | 功能 |
|---|---|---|
| `/api/status` | GET | 系統狀態 |
| `/api/brain` | GET | 腦狀態快照 |
| `/api/vitals` | GET | 生命體徵即時數據 |
| `/api/waveforms` | GET | 波形（心跳/溫度/痛覺/左右腦） |
| `/api/oscilloscope` | GET | 示波器（通道波形/反射/駐波） |
| `/api/perceive` | POST | 感知刺激 |
| `/api/think` | POST | 思考/推理 |
| `/api/act` | POST | 行動選擇 |
| `/api/learn` | POST | 學習回饋 |
| `/api/stabilize` | POST | 穩定系統 |
| `/api/time-scale` | POST | 時間加速 |
| `/api/dream-input` | POST | 夢境輸入 (REM) |
| `/api/administer-drug` | POST | 藥理學給藥 |
| `/api/introspect` | GET | 完整內省報告 |
| `/api/working-memory` | GET | 工作記憶內容 |
| `/api/causal-graph` | GET | 因果圖 |
| `/api/stats` | GET | 完整統計 |
| `/ws/stream` | WebSocket | 即時串流 |

### 8.2 啟動方式

```bash
# CLI 模式
python -m alice.main

# API 伺服器
python -m alice.main --api --port 8000
```

---

## 9. 實驗 Experiments

73 個實驗腳本 (`experiments/`)，分類如下：

### 9.1 核心驗證 Core Verification

| 實驗 | 驗證內容 |
|---|---|
| `exp_gamma_verification.py` | C1/C2/C3 約束驗證 |
| `exp_scaling_analysis.py` | 維度 K 的不可約代價 |
| `exp_relay_nodes.py` | 中繼節點降低 A_cut |
| `exp_dynamic_gamma_topology.py` | 湧現拓撲結構 |
| `exp_heterogeneous_dimensions.py` | 異構維度網路 |

### 9.2 臨床驗證 Clinical Verification

| 實驗 | 驗證內容 |
|---|---|
| `exp_clinical_neurology.py` | 5 大神經疾病 (中風/ALS/失智/AD/CP) |
| `exp_clinical_cardiology.py` ~ `exp_clinical_pulmonology.py` | 12 專科各 10 疾病 |
| `exp_clinical_calibration.py` | 臨床量表校準 |
| `exp_pharmacology.py` | MS/PD/Epilepsy/Depression 藥物模擬 |

### 9.3 雙生網路 Twin Networks (Paper II)

| 實驗 | 驗證內容 |
|---|---|
| `exp_vascular_impedance.py` | 血管 Γ-Net 基礎驗證 |
| `exp_dual_network_stability.py` | 雙網路穩定性 |
| `exp_dual_network_literature.py` | 文獻對照 |
| `exp_paper_ii_verification.py` | Paper II 全方程驗證 |
| `exp_paper_ii_origins.py` | Paper II 理論起源 |

### 9.4 睡眠/發育 Sleep & Development (Paper III/IV)

| 實驗 | 驗證內容 |
|---|---|
| `exp_sleep_physics.py` | 阻抗債 + 慢波重正規化 |
| `exp_eeg_impedance_debt.py` | EEG δ 功率 vs 年齡 |
| `exp_embryogenesis.py` | 胚胎發育 Γ 場 |
| `exp_morphogenesis_pde.py` | 形態發生 PDE |
| `exp_life_loop.py` | 閉環生命迴路 |
| `exp_neural_pruning.py` | 大規模 Γ 凋亡 |

### 9.5 認知/情緒 Cognition & Emotion

| 實驗 | 驗證內容 |
|---|---|
| `exp_attention_training.py` | 注意力可塑性 |
| `exp_cognitive_flexibility.py` | 任務切換代價 |
| `exp_curiosity_boredom.py` | 好奇心/無聊/自我 |
| `exp_metacognition.py` | 元認知 System 1/2 |
| `exp_inner_monologue.py` | 內在獨白 |
| `exp_thalamus_amygdala.py` | 丘腦-杏仁核快速通路 |

---

## 10. 論文對應 Paper Correspondence

| 論文 Paper | 檔案 File | 頁數 | 核心方程 Key Equation | 物理別名 Physics Alias | 對應程式碼 Code |
|---|---|---|---|---|---|
| **P0. Framework** | `paper_0_framework.tex` | 6 | $A = A_{imp} + A_{cut}$ | C = Constraint; Pain = protocol collapse | `gamma_topology.py` |
| **P1. Topology** | `paper_1_topology.tex` | 8 | $A_{cut} = \sum(K_{src}-K_{tgt})^+$ | Thermal refugia as C2 boundary | `gamma_topology.py` |
| **P2. Dual Network** | `paper_2_dual_network.tex` | 11 | $H = (1-\Gamma_n^2)(1-\Gamma_v^2)$ | H = global power-transfer efficiency | `vascular_impedance.py` |
| **P3. Temporal** | `paper_3_temporal.tex` | 14 | $D_{imp}$ debt + N3 recalibration | Sleep = global decoupled dissipation state | `sleep_physics.py` |
| **P4. Consciousness** | `paper_4_consciousness.tex` | 16 | $\Phi = (1/N)\sum(1-\Gamma_i^2)$ | Memory = hysteretic topological deformation; Soul = invariant topological core; Willpower tri-factor | `awareness_monitor.py`, `prefrontal.py` |
| **P5. Grand Unification** | `paper_5_clinical.tex` | 15 | $\hat Y = \mathbb{1}[H < \theta]$ | Disease = pathological attractor; Comorbidity = coupled-subsystem divergence | `gamma_engine.py` |

### 論文間交互引用 Cross-references

- P3 chronic stress → P4 hallucination unification (sensory deprivation + stress pathway)
- P4 willpower tri-factor → P5 willpower non-existence proof
- P5 immune-metabolic loop → P3 impedance debt + Arrhenius aging
- P0 thermal refugia → P1 topological necessity → P3 sleep boundary condition

---

## 11. 統計 Statistics

| 項目 Metric | 數值 Value |
|---|---|
| Python 檔案 | ~306 |
| 總程式碼行數 | ~131,900+ |
| 大腦引擎 Brain Engines | 42 |
| 身體模組 Body Modules | 32 (含 12 臨床專科) |
| 疾病模型 Disease Models | 125 (12 專科 × 10 + Lab-Γ templates) |
| 臨床量表 Clinical Scales | 125 |
| pytest 測試 | 3,274 |
| 實驗腳本 Experiments | 91 |
| 論文 Papers | 6 (70 pp total) |
| 論文圖表 Figures | 23 |
| 可測試預測 Testable Predictions | 18 |

### 三條不可違反的物理約束貫穿所有模組
### Three Inviolable Constraints Across Every Module

$$\boxed{\Gamma^2 + T = 1 \quad|\quad \Delta Z = -\eta\Gamma x_{pre}x_{post} \quad|\quad \text{ElectricalSignal only}}$$

**所有行為源自單一原理：最小反射作用 $A[\Gamma] \to \min$**  
**All behaviour emerges from one principle: Minimum Reflection Action**

---

*Alice Gamma-Net System Manual v31.1 — 2026-03-13*  
*Author: Hsi-Yu Huang · Independent Researcher · Taiwan*
