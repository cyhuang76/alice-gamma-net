# Alice Smart System

Physics-Driven Medical Lifeform Simulator Based on Γ-Net Architecture

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18720269.svg)](https://doi.org/10.5281/zenodo.18720269)
[![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-2734%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![Paper I](https://img.shields.io/badge/Paper_I-submitted_to_PRE-orange.svg)](paper/)

**Author**: Hsi-Yu Huang (黃璽宇) — Independent Researcher, Taiwan
**GitHub**: [github.com/cyhuang76/alice-gamma-net](https://github.com/cyhuang76/alice-gamma-net)
**License**: Code: AGPL-3.0 | Papers: CC BY-NC-SA 4.0

> "All behaviour is calibration error compensation."
> "Pain is not a feeling — it is protocol collapse."

---

## Overview

Alice is a **physics-driven medical lifeform simulator** — a full-body digital organism
whose every behaviour emerges from impedance physics, not statistical training.

The system models a complete physiological closed loop: sensory organs, autonomic
regulation, neural processing, motor output, sleep–wake cycles, fatigue aging, and
clinical pathology — all governed by a single variational principle (Minimum Reflection
Action) and three inviolable constraints.

| Traditional AI | Alice Γ-Net |
| :--- | :--- |
| Open-loop: Input → Output | **Closed-loop**: Perceive → Error → Compensate → Re-perceive |
| Statistical function approximator | **Physics simulator** (FFT, PID, LC resonance) |
| No body | **Full-body** (eye, ear, hand, mouth, cardiovascular, lung) |
| No pain | **Impedance mismatch = pain**, reflected energy = anxiety |
| No sleep | **NREM/REM 90-min cycles** (offline impedance recalibration) |
| Black box | **Fully transparent physics equations** |
| No aging | **Lorentz compression fatigue** (Pollock-Barraclough plastic strain) |
| No clinical pathology | **Five-disease model** + computational pharmacology |

---

## Core Physics — Minimum Reflection Action

All Alice behaviour is derived from one variational principle:

$$\mathcal{A}[\Gamma] = \int_0^T \sum_i \Gamma_i^2(t)\,dt \;\to\; \min$$

where $\Gamma_i = (Z_{\text{load},i} - Z_{\text{source},i}) / (Z_{\text{load},i} + Z_{\text{source},i})$ is the reflection coefficient of channel $i$.

### Three Mandatory Constraints

| ID | Constraint | Formula |
|:---|:---|:---|
| C1 | **Energy Conservation** | $\Gamma^2 + T = 1$ at every channel, every tick |
| C2 | **Hebbian Update** | $\Delta Z = -\eta \cdot \Gamma \cdot x_{\text{pre}} \cdot x_{\text{post}}$ |
| C3 | **Signal Protocol** | All inter-module values are `ElectricalSignal` objects carrying Z metadata |

A single physical quantity — the reflection coefficient — simultaneously drives
sensation, emotion, fatigue, sleep, neural pruning, and aging.

### Dimensional Cost Irreducibility Theorem (v2.0.0)

For networks with heterogeneous mode counts $K_i$, the action decomposes as:

$$A = A_{\text{imp}}(t) + A_{\text{cut}}$$

where $A_{\text{imp}} \to 0$ under Hebbian learning while $A_{\text{cut}} = \sum_{\text{edges}} (K_{\text{src}} - K_{\text{tgt}})^+$ has zero gradient with respect to all impedance variables. Relay nodes emerge as a thermodynamic necessity.

**Key results**: $\tau_{\text{conv}} \sim N^{-1.05}$ (larger networks converge faster) and soft-cutoff parameter $\gamma$ tunes K-space fractal dimension $D_K = 0.49\gamma + 1.00$.

---

## Architecture Overview

### Layer 6 · Application Interface

| Module | Description |
|:---|:---|
| FastAPI REST | 19 endpoints + WebSocket real-time streaming |
| Oscilloscope | CRT-style physiological dashboard |
| CLI | Interactive interface |

### Layer 5 · Unified Controller (`alice_brain.py`)

| Module | Description |
|:---|:---|
| AliceBrain | Unified controller integrating body + brain + closed-loop |
| LifeLoop | Closed-loop error compensation engine |
| AwarenessMonitor | System awareness index Φ = mean(1 − Γᵢ²) |
| SleepCycle | NREM/REM offline maintenance |

### Layer 4 · Cognitive Modules

| Module | Description |
|:---|:---|
| WorkingMemory | Miller 7±2 capacity limit |
| ReinforcementLearner | Dopamine TD(0) learning |
| CausalReasoner | Pearl's causal ladder |
| MetaLearner | Strategy pool + Softmax selection |

### Layer 3 · Brain Processing Core

| Module | Description |
|:---|:---|
| FusionBrain | Neural × Protocol fusion (5-step cycle) |
| PerceptionPipeline | LC resonance band-pass + sparse code O(1) lookup |
| AutonomicNervousSystem | Sympathetic / Parasympathetic homeostasis |
| TemporalCalibrator | Cross-modal temporal binding + drift correction |
| Hippocampus | Episodic memory engine |
| WernickeEngine | Sequence comprehension and proto-syntax |
| SemanticFieldEngine | Concepts as state-space attractors |
| BrocaEngine | Motor speech planning and sensorimotor loop |
| AuditoryGroundingEngine | Cross-modal Hebbian resonance binding |
| ThalamusEngine | Sensory gate and attention router |
| AmygdalaEngine | Emotion fast-path and fight-or-flight |
| PrefrontalCortexEngine | Executive control and goal management |
| BasalGangliaEngine | Habit engine and dopamine learning |
| NeuralPruningEngine | Massive Γ apoptosis |
| SleepPhysicsEngine | Offline impedance renormalization + energy conservation |
| AttentionPlasticityEngine | Attention gate τ / tuning Q training |
| CognitiveFlexibilityEngine | Task switching / inertia impedance |
| CuriosityDriveEngine | Novelty detection + boredom-driven behaviour |
| ImpedanceAdaptationEngine | Cross-modal impedance experiential learning |
| MetacognitionEngine | System 1/2 + confidence calibration |
| PredictiveEngine | Forward model + surprise signal |
| NarrativeMemoryEngine | Causal arc + emotion archetype |
| RecursiveGrammarEngine | Shift-reduce + garden-path recovery |
| SemanticPressureEngine | Language thermodynamics + inner monologue |
| EmotionGranularityEngine | Fine-grained affect vector |
| HomeostaticDriveEngine | Hunger / thirst physiological model |
| PhysicsRewardEngine | Impedance matching reward |
| PinchFatigueEngine | Lorentz compression fatigue — Pollock-Barraclough neural aging |
| PhantomLimbEngine | Ramachandran mirror therapy simulation |
| ClinicalNeurologyEngine | Five-disease unified impedance failure model |
| PharmacologyEngine | MS / PD / Epilepsy / Depression — unified α_drug |
| SignalBus | Coaxial cable bus (impedance matching + Γ) |
| DynamicTimeSlice | Attention resource allocation |

### Layer 2 · Communication Protocol Engine (Γ-Net v4)

| Module | Description |
|:---|:---|
| PriorityRouter | O(1) 4-level queue + aging |
| YearRingCache | 8-ring year-ring cache (hit = zero computation) |
| BrainHemisphere | Left/right brain on-demand activation |
| ErrorCorrector | Minimum energy correction |

### Layer 1 · Body (Sensory + Motor Organs)

| Organ | Type | Description |
|:---|:---|:---|
| Eye | Sensory | Convex lens FFT · Spatial freq → brainwave mapping · Nyquist resolution |
| Ear | Sensory | Cochlea physical Fourier · 24-channel filter bank · Spatial localization |
| Hand | Motor | PID + muscle tension · Anxiety → tremor · Proprioceptive feedback |
| Mouth | Motor | Source-Filter model · Vocal cord tension PID |
| Cardiovascular | Organ | Heart rate + blood pressure simulation |
| Lung | Organ | Respiratory rhythm + gas exchange |

### Layer 0 · Foundation Physics (`ElectricalSignal`)

| Component | Description |
|:---|:---|
| BrainWaveBand | δ θ α β γ five frequency bands |
| ElectricalSignal | Unified electrical signal format (waveform, impedance, SNR) |
| CoaxialChannel | Coaxial cable transmission (attenuation + Γ reflection) |
| **Axiom** | **Everything is electrical signal: Light→E, Sound→E, Force→E, Intent→E** |

---

## Core Closed-Loop: Life Loop

```text
  Perception        Error Estimation      Compensation        Execution
  ┌─────────┐      ┌──────────────┐      ┌──────────┐      ┌──────────┐
  │ Eye see │──┐   │ Cross-modal  │      │ PID ctrl │      │ Hand rch │
  │ Ear hear│──┤──→│ comparison   │──→   │ Motor cmd│──→   │ Eye turn │
  │ Hand tch│──┤   │ Reflection   │      │ Feedfwd  │      │ Mouth say│
  │ Mouth sp│──┘   │ measurement  │      │ predict  │      │          │
       ↑           │ Time calib.  │      └──────────┘      └──────────┘
       │           └──────────────┘                             │
       └──────────── Proprioceptive feedback ←──────────────────┘
```

### 7 Error Types

| Error Type | Mismatch | Compensatory Action |
|:---|:---|:---|
| VISUAL_MOTOR | Seen vs hand position | `hand.reach()` |
| AUDITORY_VISUAL | Heard vs seen direction | Head turn |
| AUDITORY_VOCAL | Target pitch vs actual | `mouth.speak()` |
| PROPRIOCEPTIVE | Target vs actual position | Fine-tune |
| TEMPORAL | Event timing offset | Temporal calibration |
| INTEROCEPTIVE | Homeostatic deviation | Autonomic regulation |
| SENSORY_PREDICTION | Predicted vs actual | Attention refocus |

### Multi-Timescale Nested Closed Loops

| Timescale | Loop | Γ's Role |
|:---|:---|:---|
| Milliseconds | Perception → Compensation → Re-perception | Instantaneous error signal |
| Seconds–Minutes | Emotion regulation | Sliding average of reflected energy |
| Hours | Awake → Sleep → Awake | ΣΓ² debt triggers sleep at pressure > 0.7 |
| Months | Neural pruning | Chronically high-Γ connections undergo apoptosis |
| Years | Coffin-Manson aging | Irreversible plastic deformation from cumulative Γ² |

---

## Physics Engine Details

### Unified Electrical Signal

```python
@dataclass
class ElectricalSignal:
    waveform: np.ndarray    # Waveform
    amplitude: float         # Amplitude (V)
    frequency: float         # Dominant frequency (Hz)
    phase: float             # Phase (rad)
    impedance: float         # Impedance (Ω)
    snr: float              # Signal-to-noise ratio (dB)
    source: str             # Source organ
    modality: str           # Sensory modality
```

### Coaxial Cable Physics

- **Impedance matching**: Source ↔ target mismatch → reflection
- **Reflection coefficient** Γ = (Z_load − Z_source) / (Z_load + Z_source)
- **Reflected energy** → channel thermal rise → anxiety
- **Perfect match** (Γ = 0) = zero pain, zero anxiety → "flow state"

### Sleep — Γ²-Driven Fatigue Cycle

**Fatigue accumulation**: $D(t) = D(t-1) + \alpha \cdot \Sigma\Gamma^2_{\text{cycle}}$

**Sleep pressure** (three-factor):

| Factor | Weight | Physics |
|:---|:---|:---|
| Energy deficit (1 − E) | 0.40 | Metabolic consumption > recovery |
| Impedance debt D | 0.35 | ΣΓ² thermal fatigue |
| Synaptic entropy deficit | 0.25 | Hebbian learning skews distribution |

When $P_{\text{sleep}} > 0.7$ → enter sleep.

**Sleep stages** (90-min ultradian cycle):

| Stage | Wave | Function |
|:---|:---|:---|
| N1 | α/θ | Close non-critical channels |
| N2 | θ | Spindle waves = memory transfer pulses |
| N3 | δ | Maximum recharge: impedance recalibration + synaptic downscaling |
| REM | θ/β | Diagnostic mode: random probe all channels |

### Clinical Pathology Models

| Pathology | Mechanism | Validation |
|:---|:---|:---|
| Stroke | Acute impedance rupture → downstream signal loss | Clinical cascade verified |
| ALS | Progressive motor channel degradation | Fasciculation + weakness progression |
| Dementia | Diffuse Γ elevation across cognitive channels | Memory + executive decline |
| Alzheimer's | Hippocampal impedance failure → episodic memory loss | Braak staging compatible |
| Cerebral Palsy | Developmental impedance miscalibration | Motor pattern verified |
| Phantom Limb Pain | Amputation = open-circuit Γ = 1.0 → mirror therapy | Ramachandran protocol validated |
| MS | Demyelination → impedance increase → conduction block | Relapse-remit pattern |
| Parkinson's | Dopamine depletion → basal ganglia impedance shift | Bradykinesia + tremor |
| Epilepsy | E-I impedance imbalance → runaway oscillation | Seizure threshold verified |
| Depression | Monoamine-mediated global Γ elevation | Anhedonia + fatigue pattern |

### Computational Pharmacology

Unified drug model: all pharmacological agents modify channel impedance via a single
parameter $\alpha_{\text{drug}}$ — enabling dose-response prediction from first principles.

---

## Project Structure

```text
Alice Smart System/
├── alice/                                  Source core
│   ├── alice_brain.py                      Unified controller
│   ├── main.py                             Entry point (CLI / Server)
│   ├── core/                               Foundation physics
│   │   ├── signal.py                       ElectricalSignal + CoaxialChannel
│   │   ├── protocol.py                     Γ-Net v4 protocol engine
│   │   ├── gamma_topology.py               Heterogeneous Γ-topology (v2.0.0)
│   │   ├── cache_analytics.py              Cache analytics
│   │   └── cache_persistence.py            Persistence
│   ├── body/                               Body organs
│   │   ├── eye.py                          Eye — FFT forward engineering
│   │   ├── ear.py                          Ear — Cochlea forward engineering
│   │   ├── cochlea.py                      Cochlear filter bank (24 ERB channels)
│   │   ├── hand.py                         Hand — PID inverse engineering
│   │   ├── mouth.py                        Mouth — Source-Filter model
│   │   ├── cardiovascular.py               Cardiovascular system
│   │   └── lung.py                         Respiratory system
│   ├── brain/                              Brain processing modules
│   │   ├── awareness_monitor.py            System awareness index Φ
│   │   ├── neural_display.py               Thalamic gateway display
│   │   ├── fusion_brain.py                 Neural × Protocol fusion
│   │   ├── life_loop.py                    Closed-loop error compensation
│   │   ├── perception.py                   LC resonance perception
│   │   ├── calibration.py                  Temporal calibration
│   │   ├── autonomic.py                    Autonomic nervous system
│   │   ├── sleep.py                        Sleep cycle controller
│   │   ├── sleep_physics.py                Sleep physics engine
│   │   ├── hippocampus.py                  Episodic memory
│   │   ├── wernicke.py                     Sequence comprehension
│   │   ├── semantic_field.py               Semantic attractors
│   │   ├── broca.py                        Motor speech planning
│   │   ├── auditory_grounding.py           Cross-modal binding
│   │   ├── thalamus.py                     Thalamus sensory gate
│   │   ├── amygdala.py                     Amygdala emotion fast-path
│   │   ├── prefrontal.py                   Executive control
│   │   ├── pruning.py                      Neural pruning
│   │   ├── basal_ganglia.py                Habit engine
│   │   ├── attention_plasticity.py         Attention training
│   │   ├── cognitive_flexibility.py        Task switching
│   │   ├── curiosity_drive.py              Novelty detection
│   │   ├── impedance_adaptation.py         Cross-modal adaptation
│   │   ├── metacognition.py                System 1/2
│   │   ├── predictive_engine.py            Forward model
│   │   ├── narrative_memory.py             Causal arc memory
│   │   ├── recursive_grammar.py            Recursive syntax
│   │   ├── semantic_pressure.py            Language thermodynamics
│   │   ├── emotion_granularity.py          Fine-grained affect
│   │   ├── homeostatic_drive.py            Hunger / thirst model
│   │   ├── physics_reward.py               Impedance reward
│   │   ├── pinch_fatigue.py                Lorentz compression aging
│   │   ├── phantom_limb.py                 Phantom limb pain
│   │   ├── clinical_neurology.py           Five-disease model
│   │   └── pharmacology.py                 Computational pharmacology
│   ├── modules/                            Cognitive modules
│   │   ├── working_memory.py               Working Memory
│   │   ├── reinforcement.py                Reinforcement Learning
│   │   ├── causal_reasoning.py             Causal Reasoning
│   │   └── meta_learning.py                Meta-Learning
│   └── api/
│       └── server.py                       FastAPI + WebSocket
├── tests/                                  Test suite (2,734 tests)
├── experiments/                            Experiment scripts (49)
├── figures/                                Publication-quality figures
├── paper/                                  Academic papers
│   └── paper_I_irreducibility.tex          Paper I (RevTeX4-2, submitted to PRE)
├── docs/                                   Architecture docs + audit reports
├── pyproject.toml                          Project configuration
└── README.md                               This document
```

---

## Quick Start

### Installation

```bash
cd "Alice Smart System"
pip install -e .
```

### Run Tests

```bash
pip install pytest
pytest tests/ -v
```

### CLI Interactive Mode

```bash
python -m alice.main cli
```

### API Server Mode

```bash
pip install fastapi uvicorn websockets
python -m alice.main server --port 8000
```

- **Oscilloscope Dashboard**: <http://localhost:8000/dashboard>
- **API Documentation**: <http://localhost:8000/docs>

### Run Experiments

```bash
# Foundation Physics
python -m experiments.exp_coaxial_physics         # Coaxial cable physics
python -m experiments.exp_eye_oscilloscope        # Visual physics
python -m experiments.exp_hand_coordination       # Motor coordination
python -m experiments.exp_perception_pipeline     # Perception pipeline
python -m experiments.exp_temporal_calibration    # Temporal calibration

# Life Loop and Pain
python -m experiments.exp_life_loop               # Closed-loop engine
python -m experiments.exp_pain_collapse           # Pain collapse clinical
python -m experiments.exp_motor_development       # Motor development

# Memory and Sleep
python -m experiments.exp_memory_theory           # Memory theory
python -m experiments.exp_sleep_physics           # Sleep physics
python -m experiments.exp_neural_pruning          # Neural pruning

# Language and Audition
python -m experiments.exp_language_physics        # Semantic field + Broca
python -m experiments.exp_episodic_wernicke       # Hippocampus + Wernicke
python -m experiments.exp_auditory_grounding      # Cross-modal conditioning

# Emotion and Executive Control
python -m experiments.exp_thalamus_amygdala       # Thalamic gate + amygdala
python -m experiments.exp_prefrontal              # Prefrontal executive control
python -m experiments.exp_basal_ganglia           # Basal ganglia habits

# Γ Verification and Stress
python -m experiments.exp_gamma_verification      # Γ unified currency
python -m experiments.exp_stress_adaptation       # Stress adaptation

# Therapy and Clinical
python -m experiments.exp_awakening               # 600-tick survival
python -m experiments.exp_therapy_mechanism       # 5 controlled therapy groups
python -m experiments.exp_digital_twin            # Digital twin PTSD subtypes

# Higher Cognition
python -m experiments.exp_attention_training      # Attention plasticity
python -m experiments.exp_cognitive_flexibility   # Task switching
python -m experiments.exp_curiosity_boredom       # Curiosity + boredom

# Homeostasis and Circadian
python -m experiments.exp_dynamic_homeostasis     # Dynamic homeostasis
python -m experiments.exp_day_night_cycle         # 24h circadian cycle

# Language Thermodynamics
python -m experiments.exp_inner_monologue         # Semantic pressure + first utterance

# Metacognition and Prediction
python -m experiments.exp_metacognition           # System 1/2
python -m experiments.exp_predictive_planning     # Predictive planning

# Clinical Pathology
python -m experiments.exp_clinical_neurology      # Five-disease model (34/34)
python -m experiments.exp_pharmacology            # Pharmacology (10/10)
python -m experiments.exp_phantom_limb            # Phantom limb pain (10/10)
python -m experiments.exp_pinch_fatigue           # Lorentz fatigue aging (10/10)

# Topology and Cardiovascular
python -m experiments.exp_topology_emergence      # MRP topology (5/5)
python -m experiments.exp_dehydration_validation  # Dehydration validation
python -m experiments.exp_tier2_cv_pathology      # Compound pathology

# Stress Tests
python -m experiments.exp_stress_test             # Extreme conditions
```

---

## API Endpoints

| Method | Path                   | Description                  |
| :----- | :--------------------- | :--------------------------- |
| GET    | `/api/status`          | System status                |
| GET    | `/api/brain`           | Brain state snapshot         |
| GET    | `/api/vitals`          | Vital signs                  |
| GET    | `/api/waveforms`       | Waveform data                |
| GET    | `/api/oscilloscope`    | Oscilloscope channels        |
| GET    | `/api/introspect`      | Full introspection report    |
| GET    | `/api/working-memory`  | Working memory contents      |
| GET    | `/api/causal-graph`    | Causal graph                 |
| GET    | `/api/stats`           | Full statistics              |
| POST   | `/api/perceive`        | Perception stimulus          |
| POST   | `/api/think`           | Reasoning                    |
| POST   | `/api/act`             | Action selection             |
| POST   | `/api/learn`           | Learning feedback            |
| POST   | `/api/inject-pain`     | Pain injection (clinical)    |
| POST   | `/api/emergency-reset` | Emergency reset              |
| POST   | `/api/broadcast-storm` | Broadcast storm attack       |
| WS     | `/ws/stream`           | Real-time status stream      |

---

## Python API

```python
from alice import AliceBrain

brain = AliceBrain()

# Perception
brain.see(image_array)              # Visual input (numpy ndarray)
brain.hear(audio_array)             # Auditory input (numpy ndarray)
brain.reach_for(x, y)               # Motor reach toward (x, y)
brain.say(440.0, 0.7, "a", "hello") # Speak (pitch, volume, vowel, concept)

# Cognition
brain.perceive(stimulus, "visual")   # Perceive stimulus
brain.think("What is this?")         # Reasoning
brain.act(state, actions)            # Action selection

# Learning
brain.learn_from_feedback(state, action, reward, next_state, actions)

# Vital Signs
vitals = brain.vitals
vitals.ram_temperature               # RAM temperature (anxiety index)
vitals.pain_level                    # Pain level
vitals.heart_rate                    # Heart rate
vitals.total_ticks                   # Life cycle tick count
```

---

## Fundamental Differences from Modern AI

```text
         Modern AI (LLM/CNN)                Alice (Γ-Net)
         ─────────────────                  ────────────

Topology  Feedforward DAG                   Closed-loop recursive feedback
Training  Offline gradient descent          Online PID + Hebbian + impedance matching
Inference y = f(x) single output            Continuous compensation until error < ε
Body      None                              Full limbs (eye/ear/hand/mouth) + organs
Pain      None                              Impedance mismatch → reflected energy
Attention Transformer self-attention        Global workspace bottleneck
Sleep     None                              NREM/REM 90-minute cycles
Time      No internal clock                 50ms/200ms temporal binding windows
Language  Statistical token prediction      Semantic pressure → impedance matching
Clinical  None                              10 pathology models + pharmacology
Scale     10¹¹ parameters                   ~89,000 lines of physics equations
```

---

## License

- **Source code**: [GNU Affero General Public License v3.0](LICENSE)
- **Papers** (`paper/*.md`): [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## Citation

```bibtex
@software{huang2026alice,
  author       = {Huang, Hsi-Yu},
  title        = {Alice Smart System — Physics-Driven Medical Lifeform Simulator},
  year         = 2026,
  version      = {2.0.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18720269},
  url          = {https://github.com/cyhuang76/alice-gamma-net}
}
```

---

## Paper Series

| # | Title | Status |
|---|-------|--------|
| I | Irreducible Dimensional Cost in Heterogeneous Impedance Networks | **Submitted to PRE** (2026) |
| II | From Coaxial Cables to Cognition | In preparation |
| III | The Lifecycle Equation | In preparation |
| IV | Emergence | In preparation |
| V | Grand Unification | Planned |

---

## Keywords

`computational-neuroscience` · `cognitive-architecture` · `medical-simulation` ·
`impedance-matching` · `coaxial-transmission-line` · `physics-driven` ·
`clinical-pathology` · `computational-pharmacology` · `digital-twin` ·
`phantom-limb` · `PTSD-simulation` · `sleep-cycles` · `language-emergence`
