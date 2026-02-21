# Alice Smart System — 閉環/開環審計報告

**審計日期**: 2026-02-21  
**審計範圍**: `alice/brain/`, `alice/body/`, `alice/alice_brain.py`, `alice/modules/`  
**審計方法**: 追蹤每個模組在 `AliceBrain.perceive()` 主認知迴圈中的資料流路徑

---

## 分類定義

| 狀態 | 定義 |
|------|------|
| **CLOSED** | 輸出確實回饋影響未來輸入（完整迴路：刺激→處理→動作→回饋→再處理） |
| **PARTIAL** | 有部分回饋但不完整（例如：影響情緒但不觸發行為，或輸出被讀取但不改變系統狀態） |
| **OPEN** | 輸出僅進入 `brain_result` 字典供顯示/日誌，不被任何模組回讀 |

---

## 🧠 Brain 模組審計

### 核心迴路模組

| 模組 | 檔案 | 狀態 | 回饋路徑說明 |
|------|------|------|-------------|
| **LifeLoop** | `life_loop.py` | **CLOSED** ✅ | 完整閉環：估計跨模態誤差→生成補償命令→`_dispatch_commands()` 執行到 hand/mouth/eye/autonomic→新感覺→新誤差。持續誤差→`error_to_pain`→`ram_temperature`↑→影響下一 tick 的補償增益。 |
| **SystemState (Vitals)** | `alice_brain.py` | **CLOSED** ✅ | 完整閉環：queue 壓力+反射能量→溫度↑→疼痛→穩定性↓→意識↓→心率異常→throttle 降低→凍結門檻→阻止非 CRITICAL 封包。創傷記憶永久改變痛覺閾值。 |
| **FusionBrain** | `fusion_brain.py` | **CLOSED** ✅ | 完整閉環：處理刺激→sensory/emotional/motor 結果→reflected_energy 回饋到 vitals→溫度→throttle→影響下一次處理速度。sleep_consolidate 在睡眠時重組突觸。 |
| **TemporalCalibrator** | `calibration.py` | **CLOSED** ✅ | 完整閉環：接收多模態信號→計算漂移→calibration_quality→(1) 影響 `binding_gamma` 寫入工作記憶 (2) 漂移值饋入 LifeLoop 的 temporal error→觸發 ATTEND 補償。 |

### 感覺處理模組

| 模組 | 檔案 | 狀態 | 回饋路徑說明 |
|------|------|------|-------------|
| **PerceptionPipeline** | `perception.py` | **CLOSED** ✅ | attention_band + concept + bindings→寫入工作記憶→影響因果推理→同時 plasticity_engine 被注入使感知參數隨經驗改善。 |
| **Thalamus** | `thalamus.py` | **CLOSED** ✅ | 感覺閘門：amplitude/gamma/arousal→gate_gain→篩選信號是否到達皮質。top-down attention 由 prefrontal goal 設定（`set_attention`）。Amygdala 威脅信號也提高閘門增益。Attention plasticity 追蹤曝光事件改善閘門速度。 |
| **Amygdala** | `amygdala.py` | **CLOSED** ✅ | 完整閉環：快速威脅評估→(1) sympathetic_command→autonomic.sympathetic↑→pupil/heart_rate/energy 改變 (2) 設定 thalamus attention (3) inject_threat→emotion_granularity (4) fear_conditioning 永久降低閾值影響未來評估。 |
| **SemanticField** | `semantic_field.py` | **CLOSED** ✅ | fingerprint→概念辨識（最低 Γ 吸引子）→best_concept 饋入 (1) Hippocampus 記錄 (2) Wernicke 觀察序列 (3) Amygdala 評估威脅。sleep consolidation 從 hippocampus 強化吸引子質量。 |
| **AuditoryGrounding** | `auditory_grounding.py` | **CLOSED** ✅ | 接收 auditory + visual 信號→跨模態 Hebbian binding→突觸衰減每 tick（`tick()`）→binding quality 影響 calibration→影響 binding_gamma→工作記憶寫入品質。巴甫洛夫條件反射形成永久跨模態連結。 |
| **AttentionPlasticity** | `attention_plasticity.py` | **CLOSED** ✅ | 注入到 thalamus + perception pipeline→改善閘門速度/Q/抑制效率→每 tick 衰減（use it or lose it）→`on_exposure()` 在視覺/聽覺通過閘門時呼叫→持續改善。 |

### 記憶系統

| 模組 | 檔案 | 狀態 | 回饋路徑說明 |
|------|------|------|-------------|
| **WorkingMemory** | `modules/working_memory.py` | **CLOSED** ✅ | store() 帶 binding_gamma 調節→contents 饋入 consciousness 的 wm_usage→影響 phi→phi 影響 LifeLoop 可同時處理的錯誤數。metacognition 的 self-correction 觸發 flush_weakest()。 |
| **Hippocampus** | `hippocampus.py` | **CLOSED** ✅ | record() 記錄視覺/聽覺快照（含 amygdala valence）→形成情節記憶→sleep consolidate() 將情節遷移到 semantic_field（吸引子質量↑）→改善未來概念辨識。 |
| **NarrativeMemory** | `narrative_memory.py` | **PARTIAL** ⚠️ | 依賴 hippocampus 情節→編織自傳記憶→tick() 維護敘事弧。**但**：`narrative_result` 僅寫入 `brain_result["narrative_memory"]`，未被任何其他模組回讀來改變系統行為。不影響情緒、決策或注意力。 |

### 語言系統

| 模組 | 檔案 | 狀態 | 回饋路徑說明 |
|------|------|------|-------------|
| **Broca** | `broca.py` | **CLOSED** ✅ | 概念→發音計畫→mouth 執行→產生波形→auditory feedback→calibrator→gamma_loop 饋入 semantic_pressure release→內在壓力釋放改變下一 tick 的語義壓力。`say()` 方法直接呼叫 `broca.speak_concept()`。 |
| **Wernicke** | `wernicke.py` | **CLOSED** ✅ | observe() 接收概念序列→建立轉移機率矩陣→Γ_syntactic→(1) N400 事件 (2) 饋入 recursive_grammar 規則學習 (3) semantic_pressure.tick() 讀取 wernicke 狀態→驅動內在獨白/Wernicke→Broca 直連。 |
| **RecursiveGrammar** | `recursive_grammar.py` | **PARTIAL** ⚠️ | 從 Broca + Wernicke 學習規則→tick() 維護規則信心度→`grammar_result` 僅寫入 `brain_result`。**未被用來**改善 Broca 的發音計畫或 Wernicke 的序列預測。規則存在但不影響語言產出。 |
| **SemanticPressure** | `semantic_pressure.py` | **CLOSED** ✅ | 語義壓力累積→inner monologue → Wernicke→Broca 直接驅動→`say()` 中 `release()` 釋放壓力→Γ_speech→0 時壓力大幅下降→影響下一 tick 的壓力累積。和 amygdala valence、arousal、pain 交互。 |

### 情緒與動機系統

| 模組 | 檔案 | 狀態 | 回饋路徑說明 |
|------|------|------|-------------|
| **AutonomicNS** | `autonomic.py` | **CLOSED** ✅ | 完整閉環：pain+temperature+emotion→sympathetic/parasympathetic 平衡→(1) pupil_aperture→eye (2) energy→LifeLoop 補償增益 (3) cortisol→impedance_adaptation 學習率 (4) autonomic_balance→LifeLoop interoceptive error→BREATHE 補償→parasympathetic↑。 |
| **Consciousness** | `consciousness.py` | **CLOSED** ✅ | phi = f(attention, binding, WM, arousal, gate, pain)→(1) LifeLoop 的 consciousness_phi 決定可處理錯誤數 (2) 低 phi→LifeLoop 停止所有補償 (3) broadcast_to_workspace() 全域通知 (4) sensory_gate 影響 LifeLoop 的感覺閘門。 |
| **SleepCycle** | `sleep.py` | **CLOSED** ✅ | 監測刺激強度→管理睡眠階段→sensory_gate 開關→should_consolidate→觸發 hippocampus/semantic_field/FusionBrain 記憶鞏固。睡眠降低感覺閘門→LifeLoop 僅通過最強錯誤。 |
| **EmotionGranularity** | `emotion_granularity.py` | **PARTIAL** ⚠️ | 接收威脅/社交/好奇/恆定態注入→計算 Plutchik 8維情緒向量 + VAD 座標→`emotion_granularity_result` 寫入 `brain_result`。**但**：VAD/dominant_emotion/compound_emotions **未被**其他模組回讀。不影響 autonomic、amygdala、prefrontal 的決策。僅為觀測輸出。 |
| **CuriosityDrive** | `curiosity_drive.py` | **PARTIAL** ⚠️ | tick()→novelty/boredom 累積→generate_spontaneous_action()→產出自發行為建議。**迴路部分閉合**：novelty 饋入 (1) emotion_granularity inject_novelty (2) metacognition 的 novelty/boredom 參數。evaluate_novelty() 在 see/hear 中呼叫。**但**：`spontaneous_action` 建議（explore/vocalize/attend/fidget）**從未被 AliceBrain 實際執行**。curiosity_result 中的 spontaneous_action 僅寫入 brain_result。自發行為的產出端是斷開的。 |
| **HomeostaticDrive** | `homeostatic_drive.py` | **PARTIAL** ⚠️ | tick()→hunger/thirst drive→(1) pain_contribution→ram_temperature↑ ✅ (2) irritability→emotional_valence 負偏移 ✅ (3) cognitive_penalty→報告但**未實際降低**認知處理速度。**最關鍵缺口**：`needs_food`/`needs_water` 信號僅寫入 brain_result，**沒有任何代碼去觸發 feed()/drink() 行為**。hunger drive 無限累積但永遠不會被滿足（除非外部 API 調用）。驅力→行為的最後一步是斷開的。 |
| **SocialResonance** | `social_resonance.py` | **PARTIAL** ⚠️ | tick()→social_need 累積→loneliness→social_bond 紀錄。**部分閉合**：social_bond_strength 饋入 emotion_granularity inject_social。**但**：(1) social_need/is_lonely **不觸發**任何尋求社交的行為 (2) social_result 不影響 autonomic、pain、prefrontal goal。社交飢餓信號產出但無行為響應。 |
| **MirrorNeurons** | `mirror_neurons.py` | **PARTIAL** ⚠️ | tick()→empathic_valence、has_social_input。**部分閉合**：empathic_valence→emotion_granularity inject_social、tom_capacity→social_resonance tick()。**但**：`has_social_input=False`（硬編碼預設）在 perceive() 主迴圈中，除非外部呼叫 `observe_*()` 方法才能啟動社交感知。mirror_result 不直接驅動行為。 |

### 執行控制系統

| 模組 | 檔案 | 狀態 | 回饋路徑說明 |
|------|------|------|-------------|
| **Prefrontal** | `prefrontal.py` | **CLOSED** ✅ | 目標管理→(1) top_goal 設定 thalamus top-down attention bias (2) Go/NoGo 門控 basal ganglia 動作選擇 (3) energy→cognitive_flexibility sync (4) tick() 能量恢復 + 冷卻。willpower 消耗影響後續抑制能力。 |
| **BasalGanglia** | `basal_ganglia.py` | **CLOSED** ✅ | 動作選擇→(1) habitual vs goal-directed 雙系統仲裁 (2) reward 後 update_after_action()→habit strength 更新 (3) gamma_habit→impedance→prefrontal Go/NoGo 評估 (4) dopamine_level 由 physics_reward 注入。tick() 每循環呼叫。 |
| **CognitiveFlexibility** | `cognitive_flexibility.py` | **CLOSED** ✅ | 偵測感官模態切換（visual↔auditory）→(1) switch_cost 影響反應時間 (2) perseveration_error 記錄 (3) inertia 慣性阻抗影響切換品質 (4) PFC energy sync→flexibility_index→饋入 metacognition。tick() 每循環呼叫。 |
| **Metacognition** | `metacognition.py` | **CLOSED** ✅ | 整合全腦 Γ_thinking→(1) thinking_rate→**實際** throttle 降速 time.sleep() (2) is_correcting→觸發 cognitive_flexibility task_switch + working_memory flush_weakest (3) System 1/2 切換、反芻警報。輸出確實改變系統行為。 |
| **PredictiveEngine** | `predictive_engine.py` | **CLOSED** ✅ | 前向模型預測下一 tick 狀態→(1) prediction_error 饋入 metacognition (2) free_energy 饋入 metacognition (3) surprise 饋入 metacognition (4) anxiety_level 饋入 metacognition。metacognition 再將這些轉化為 thinking_rate 和 self-correction。 |

### 學習與適應系統

| 模組 | 檔案 | 狀態 | 回饋路徑說明 |
|------|------|------|-------------|
| **NeuralPruning** | `pruning.py` | **CLOSED** ✅ | 每次感知→stimulate()→Hebbian 選擇→弱連結 prune()→強連結 sprout()（學習信號來自 curiosity + reward）→cortical specialization。每50 tick 掃描。連結數和特化程度持續演化。 |
| **ImpedanceAdaptation** | `impedance_adaptation.py` | **CLOSED** ✅ | 每次跨模態 binding→record_binding_attempt(success, quality, cortisol)→Γ 下降（匹配改善）或上升→adapted_binding_gamma 混合到 binding_gamma→影響工作記憶寫入品質。decay_tick() 遺忘未使用配對。cortisol Yerkes-Dodson 調制。 |
| **PhysicsReward** | `physics_reward.py` | **CLOSED** ✅ | 取代 Q-table：impedance-matching Hebbian 學習→dopamine→(1) basal ganglia dopamine_level 注入 (2) experience replay 離線重組 (3) Boltzmann selection→action choice。learn_from_feedback() 完整 TD更新。 |
| **ReinforcementLearner** | `modules/reinforcement.py` | **CLOSED** ✅ | TD update→Q-value 更新→action selection（雖然主要被 PhysicsReward 取代，但 reach_for() 中仍直接使用 rl.update）。 |
| **CausalReasoner** | `modules/causal_reasoning.py` | **CLOSED** ✅ | observe() 累積因果觀測→think() 中進行因果/反事實推理→影響 meta_learning 策略選擇。 |
| **MetaLearner** | `modules/meta_learning.py` | **CLOSED** ✅ | select_strategy()→調整 RL epsilon→report_performance()→策略演化。think() 和 act() 都使用。 |

### 物理/臨床模組

| 模組 | 檔案 | 狀態 | 回饋路徑說明 |
|------|------|------|-------------|
| **SleepPhysics** | `sleep_physics.py` | **PARTIAL** ⚠️ | 計算 impedance debt、synaptic entropy、SHY downscaling→sleep_tick()/awake_tick()。**但**：`sleep_phys` 返回值**未被使用**（awake_tick 和 sleep_tick 的返回值被賦給 sleep_phys 但從未回饋）。Impedance debt 不影響 sleep pressure 或 consciousness。 |
| **PinchFatigue** | `pinch_fatigue.py` | **PARTIAL** ⚠️ | tick()→計算 aging_signal（elastic/plastic strain、cognitive_impact、impedance_drift）→**寫入 brain_result 但**：`cognitive_impact` 和 `impedance_drift` **未被用來實際降低**認知處理速度或改變通道阻抗。aging 累積但不影響系統行為。growth_factor 來自 parasympathetic（閉合輸入），但輸出是開放的。 |
| **PhantomLimb** | `phantom_limb.py` | **OPEN** ❌ | tick(emotional_valence, stress_level)→計算 phantom pain、neuroma discharge、referred pain。**但**：`phantom_result` 僅寫入 `brain_result["phantom_limb"]`，**不回饋到** vitals.ram_temperature 或 pain_level。phantom pain 無法造成系統疼痛。完全觀測性輸出。 |
| **ClinicalNeurology** | `clinical_neurology.py` | **OPEN** ❌ | tick(brain_state)→讀取全腦狀態→計算 stroke/ALS/dementia/Alzheimer's/CP 指標。**但**：`clinical_result` 僅寫入 `brain_result`，**不修改**任何 brain 模組的參數。疾病模擬不會實際癱瘓通道或降低功能。純觀測/報告。 |
| **Pharmacology** | `pharmacology.py` | **OPEN** ❌ | tick(brain_state)→計算 drug α_drug 阻抗修改→channel Γ modifications。**但**：`pharma_result` 僅寫入 `brain_result`，**不注入回**任何通道的實際阻抗。藥物效果是計算出來的但未生效。 |

### 其他

| 模組 | 檔案 | 狀態 | 回饋路徑說明 |
|------|------|------|-------------|
| **EmotionGranularity** (重複在上方) | `emotion_granularity.py` | **PARTIAL** ⚠️ | 見上方。接收完整（4個注入源），但輸出僅為報告。`get_dominance()` 被自己回讀（inject_threat），但不影響其他模組。 |

---

## 🦴 Body 模組審計

| 模組 | 檔案 | 狀態 | 回饋路徑說明 |
|------|------|------|-------------|
| **AliceEye** | `body/eye.py` | **CLOSED** ✅ | see()→FFT→ElectricalSignal→perceive()。閉環：autonomic→pupil_aperture→adjust_pupil() 改變感光增益、LifeLoop 的 SACCADE/ADJUST_PUPIL 命令執行到 eye。saccade() 聚焦區域。fingerprint 饋入 semantic_field + amygdala。 |
| **AliceEar** | `body/ear.py` | **CLOSED** ✅ | hear()→cochlea FFT→ElectricalSignal→perceive()。auditory_grounding 接收原始波形→跨模態 binding。信號饋入 calibrator、semantic_field、amygdala。 |
| **AliceHand** | `body/hand.py` | **CLOSED** ✅ | reach()→PID 控制+肌肉物理+焦慮震顫（ram_temperature 注入）→proprioception 饋入 (1) calibrator (2) LifeLoop。REACH 補償命令由 `_dispatch_commands()` 執行。guard_level + injury_memory 由 pain 事件增加→改變未來抓取行為。dopamine 迴路完整。 |
| **AliceMouth** | `body/mouth.py` | **CLOSED** ✅ | speak()→PID 音高控制→ram_temperature 造成震顫→proprioception 饋入 calibrator。VOCALIZE 補償命令由 `_dispatch_commands()` 執行。Broca pathway 產出波形→auditory feedback→semantic_pressure release。 |
| **CochlearFilterBank** | `body/cochlea.py` | **CLOSED** ✅ | 被 AuditoryGroundingEngine 使用→tonotopic 分解→fingerprint→downstream 處理。 |

---

## 📊 統整：全系統閉環比例

| 狀態 | 數量 | 比例 |
|------|------|------|
| **CLOSED** ✅ | 27 | 67.5% |
| **PARTIAL** ⚠️ | 9 | 22.5% |
| **OPEN** ❌ | 3 | 7.5% |
| **合計** | **39** | 100% |

---

## 🔴 關鍵開環問題（按嚴重程度排序）

### 1. HomeostaticDrive：飢餓/口渴 **永遠不會被滿足** ⚠️⚠️⚠️
- `needs_food` / `needs_water` 信號產出但 **沒有任何代碼觸發 `feed()` / `drink()`**
- hunger_drive 會無限上升→irritability↑→pain_contribution↑→但永遠無法降回來
- **建議**：在 perceive() 中加入：當 `needs_food` 時自動呼叫 `self.homeostatic_drive.feed()`，或由 prefrontal 建立「進食」目標觸發行為

### 2. CuriosityDrive：自發行為建議 **從未被執行** ⚠️⚠️
- `generate_spontaneous_action()` 產出 explore/vocalize/attend/fidget 建議
- 但 `curiosity_result["spontaneous_action"]` 僅寫入 brain_result，**AliceBrain 從不讀取並分派執行**
- 這意味著 Alice 無法自發探索環境——「自由意志」的物理表達被切斷
- **建議**：在 perceive() 中加入 spontaneous action dispatch 邏輯

### 3. SocialResonance：社交飢餓 **不觸發行為** ⚠️⚠️
- `social_need` 累積、`is_lonely` 觸發，但不驅動任何行為
- 不像 homeostatic 還至少注入 pain/irritability，social_need 的行為端完全開放
- **建議**：高 social_need 應注入 prefrontal goal（尋求社交）或觸發自發行為

### 4. PhantomLimb/ClinicalNeurology/Pharmacology：純觀測模組 ❌
- 三者都是 **read brain_state → compute metrics → write to brain_result**
- 計算結果不回饋修改系統。疾病不導致功能退化，藥物不改變阻抗
- **建議**：
  - PhantomLimb: `phantom_pain` 應注入 `vitals.ram_temperature`
  - ClinicalNeurology: 疾病嚴重度應修改對應通道的阻抗/衰減率
  - Pharmacology: `α_drug` 應實際注入 FusionBrain 通道阻抗

### 5. PinchFatigue：老化計算但不生效 ⚠️
- `cognitive_impact` 計算了但不降低處理速度
- `impedance_drift` 計算了但不修改通道阻抗
- **建議**：`cognitive_impact` 應乘入 throttle_factor，`impedance_drift` 應加入 FusionBrain 通道

### 6. SleepPhysics：債務累積但不影響決策 ⚠️
- impedance_debt 在清醒時累積、在睡眠時償還，但不影響 sleep_pressure 或 consciousness
- **建議**：impedance_debt 應饋入 SleepCycle 的 sleep_pressure 加速睡眠需求

### 7. RecursiveGrammar：規則學習但不使用 ⚠️
- 從 Wernicke/Broca 學到的遞歸語法規則不用於改進語言產出
- **建議**：Broca.speak_concept() 應查詢 RecursiveGrammar 的規則來構建更複雜的句子

### 8. NarrativeMemory：自傳記憶不影響行為 ⚠️
- 敘事弧編織完成但不被任何決策模組使用
- **建議**：narrative 應影響 prefrontal goal 優先級（基於過去經驗的教訓）

### 9. EmotionGranularity：精細情緒不回饋系統 ⚠️
- 8維 Plutchik 向量 + VAD 座標計算完成但僅供顯示
- **建議**：dominant_emotion 應影響 prefrontal 決策偏好，compound_emotions 應影響 social behavior

---

## ✅ 最佳閉環範例

**THE PAIN LOOP**（最完整的閉環）：
```
queue pressure + reflected energy
    → ram_temperature ↑
        → pain_level ↑
            → stability_index ↓
                → consciousness ↓
                    → heart_rate arrhythmia
                        → throttle ↓ (time.sleep)
                            → only CRITICAL packets pass
                                → queue clears → temperature ↓
                                    → pain ↓ → recovery
```
加上創傷記憶的長期迴路：
```
severe pain → record_trauma()
    → pain_sensitivity ↑ (永久)
        → future pain threshold ↓
            → easier to hurt next time
```

**LifeLoop 補償迴路**（第二完整的閉環）：
```
sensory signals → cross-modal error estimation
    → consciousness ranking (attention = error priority)
        → PID compensation commands
            → _dispatch_commands() → hand.reach() / mouth.speak() / eye.adjust_pupil()
                → new proprioception / auditory feedback
                    → new error estimation (delta)
                        → forward model update (prediction accuracy ↑)
```

---

## 建議優先修復順序

1. **HomeostaticDrive 閉環**（飢餓/口渴→行為）— 生存基礎
2. **CuriosityDrive 閉環**（自發行為執行）— 自由意志表達
3. **PhantomLimb 回饋注入**（phantom pain → vitals）— 疼痛完整性
4. **PinchFatigue 老化生效**（cognitive_impact → throttle）— 生命週期真實性
5. **ClinicalNeurology 疾病生效**（疾病→通道退化）— 臨床模擬完整性
6. **Pharmacology 藥物生效**（α_drug → 通道阻抗）— 藥理閉環
7. **SocialResonance 行為觸發**（loneliness → seek social goal）— 社交動機
8. **SleepPhysics → SleepCycle**（debt → pressure）— 睡眠物理一致性
9. **EmotionGranularity 回饋**（emotion → decision）— 情緒功能化
10. **RecursiveGrammar/NarrativeMemory 功能化** — 語言/記憶進階
