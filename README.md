# Alice Smart System

**Physics-Driven Medical Lifeform Simulator Based on Γ-Net Architecture**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18795843.svg)](https://doi.org/10.5281/zenodo.18795843)
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
| No body | **Full-body** (eye, ear, hand, mouth, cardiovascular, lung, + 12 more organs) |
| No pain | **Impedance mismatch = pain**, reflected energy = anxiety |
| No sleep | **NREM/REM 90-min cycles** (offline impedance recalibration) |
| Black box | **Fully transparent physics equations** |
| No aging | **Lorentz compression fatigue** (Pollock-Barraclough plastic strain) |
| No clinical pathology | **10+ disease models** + computational pharmacology |

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

where $A_{\text{imp}} \to 0$ under Hebbian learning while $A_{\text{cut}} = \sum_{\text{edges}} (K_{\text{src}} - K_{\text{tgt}})^+$ has zero gradient with respect to all impedance variables at fixed topology. This geometric cost is **irreducible** — relay nodes emerge as a thermodynamic necessity.

**Key results**:
- **Scaling**: $\tau_{\text{conv}} \sim N^{-0.91}$ — larger networks converge *faster* (mean-field effect)
- **Fractal control**: soft-cutoff $\gamma$ tunes K-space dimension $D_K = 0.49\gamma + 1.00$
- **Cost ceiling**: $A_{\text{cut}}$ saturates, revealing a natural geometric cost ceiling

---

## Architecture

### Layer 6 · Application Interface

| Module | Description |
|:---|:---|
| FastAPI REST | 19 endpoints + WebSocket real-time streaming |
| Oscilloscope | CRT-style physiological dashboard |
| CLI | Interactive interface |

### Layer 5 · Unified Controller (`alice_brain.py`, 2951 lines)

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

### Layer 3 · Brain Processing Core (42 modules)

| Category | Modules |
|:---|:---|
| **Perception & Fusion** | FusionBrain, PerceptionPipeline, TemporalCalibrator |
| **Sensory Gating** | ThalamusEngine, AuditoryGroundingEngine |
| **Memory** | Hippocampus, NarrativeMemoryEngine |
| **Language** | WernickeEngine, BrocaEngine, RecursiveGrammarEngine, SemanticPressureEngine |
| **Emotion** | AmygdalaEngine, EmotionGranularityEngine |
| **Executive** | PrefrontalCortexEngine, CognitiveFlexibilityEngine, MetacognitionEngine |
| **Motor** | BasalGangliaEngine, CerebellumEngine |
| **Homeostasis** | AutonomicNervousSystem, HomeostaticDriveEngine |
| **Learning** | AttentionPlasticityEngine, CuriosityDriveEngine, ImpedanceAdaptationEngine, PredictiveEngine |
| **Sleep** | SleepPhysicsEngine |
| **Plasticity** | NeuralPruningEngine, NeurogenesisThermalEngine |
| **Clinical** | ClinicalNeurologyEngine, PharmacologyEngine, PhantomLimbEngine, PinchFatigueEngine |
| **Reward** | PhysicsRewardEngine |
| **Topology (v2.0.0)** | GammaTopology — heterogeneous Γ-network with soft cutoff |

### Layer 2 · Communication Protocol Engine (Γ-Net v4)

| Module | Description |
|:---|:---|
| PriorityRouter | O(1) 4-level queue + aging |
| YearRingCache | 8-ring year-ring cache (hit = zero computation) |
| BrainHemisphere | Left/right brain on-demand activation |
| ErrorCorrector | Minimum energy correction |

### Layer 1 · Body (18 Organs)

| Organ | Type | Description |
|:---|:---|:---|
| Eye | Sensory | Convex lens FFT · Spatial freq → brainwave mapping · Nyquist resolution |
| Ear | Sensory | Cochlea physical Fourier · 24-channel filter bank · Spatial localization |
| Nose | Sensory | Olfactory receptor array · concentration detection |
| Hand | Motor | PID + muscle tension · Anxiety → tremor · Proprioceptive feedback |
| Mouth | Motor | Source-Filter model · Vocal cord tension PID |
| Skin | Sensory | Mechanoreceptor array · pressure/temperature mapping |
| Vestibular | Sensory | Semicircular canal model · balance feedback |
| Interoception | Sensory | Internal organ state monitoring |
| Cardiovascular | Organ | Heart rate + blood pressure + vascular resistance |
| Lung | Organ | Respiratory rhythm + gas exchange |
| Liver | Organ | Metabolic processing + detoxification |
| Kidney | Organ | Filtration + electrolyte balance |
| Immune | System | Innate + adaptive immune response |
| Endocrine | System | Hormone regulation + feedback loops |
| Digestive | System | Nutrient absorption + peristalsis |
| Lymphatic | System | Tissue drainage + immune cell transport |
| Reproductive | System | Hormonal cycling model |
| Cochlea | Component | Basilar membrane 24-ERB filter bank |

### Layer 0 · Foundation Physics

| Component | Description |
|:---|:---|
| ElectricalSignal | Unified electrical signal format (waveform, impedance, SNR) |
| CoaxialChannel | Coaxial cable transmission (attenuation + Γ reflection) |
| GammaTopology | Dynamic network topology with heterogeneous K modes |
| BrainWaveBand | δ θ α β γ five frequency bands |
| **Axiom** | **Everything is electrical signal: Light→E, Sound→E, Force→E, Intent→E** |

---

## Closed-Loop Architecture

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

### Multi-Timescale Nested Closed Loops

| Timescale | Loop | Γ's Role |
|:---|:---|:---|
| Milliseconds | Perception → Compensation → Re-perception | Instantaneous error signal |
| Seconds–Minutes | Emotion regulation | Sliding average of reflected energy |
| Hours | Awake → Sleep → Awake | ΣΓ² debt triggers sleep at pressure > 0.7 |
| Months | Neural pruning | Chronically high-Γ connections undergo apoptosis |
| Years | Coffin-Manson aging | Irreversible plastic deformation from cumulative Γ² |

---

## Clinical Pathology Models

| Pathology | Mechanism |
|:---|:---|
| Stroke | Acute impedance rupture → downstream signal loss |
| ALS | Progressive motor channel degradation |
| Dementia | Diffuse Γ elevation across cognitive channels |
| Alzheimer's | Hippocampal impedance failure → episodic memory loss |
| Cerebral Palsy | Developmental impedance miscalibration |
| Phantom Limb | Amputation = open-circuit Γ = 1.0 → mirror therapy |
| MS | Demyelination → impedance increase → conduction block |
| Parkinson's | Dopamine depletion → basal ganglia impedance shift |
| Epilepsy | E-I impedance imbalance → runaway oscillation |
| Depression | Monoamine-mediated global Γ elevation |

**Computational Pharmacology**: All drugs modify channel impedance via a single parameter $\alpha_{\text{drug}}$ — enabling dose-response prediction from first principles.

---

## Project Structure

```text
Alice Smart System/
├── alice/                                  Source core (200 files, 89,400+ lines)
│   ├── alice_brain.py                      Unified controller (2951 lines)
│   ├── main.py                             Entry point (CLI / Server)
│   ├── core/                               Foundation physics
│   │   ├── signal.py                       ElectricalSignal + CoaxialChannel
│   │   ├── protocol.py                     Γ-Net v4 protocol engine
│   │   ├── gamma_topology.py               Heterogeneous Γ-topology (v2.0.0)
│   │   ├── cache_analytics.py              Cache analytics
│   │   └── cache_persistence.py            Persistence
│   ├── body/                               Body organs (18 modules)
│   │   ├── eye.py, ear.py, hand.py, mouth.py
│   │   ├── cardiovascular.py, lung.py, liver.py, kidney.py
│   │   ├── immune.py, endocrine.py, digestive.py, lymphatic.py
│   │   ├── skin.py, nose.py, vestibular.py, interoception.py
│   │   ├── reproductive.py, cochlea.py
│   │   └── __init__.py
│   ├── brain/                              Brain processing (42 modules)
│   └── modules/                            Cognitive modules (4 modules)
├── tests/                                  Test suite (2,734 tests)
├── experiments/                            Experiment scripts (49)
├── figures/                                Publication-quality figures
├── paper/                                  Academic papers
│   └── paper_I_irreducibility.tex          Paper I (RevTeX4-2)
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
Body      None                              Full body (18 organs) + autonomic system
Pain      None                              Impedance mismatch → reflected energy
Attention Transformer self-attention        Global workspace bottleneck
Sleep     None                              NREM/REM 90-minute cycles
Time      No internal clock                 50ms/200ms temporal binding windows
Language  Statistical token prediction      Semantic pressure → impedance matching
Clinical  None                              10+ pathology models + pharmacology
Scale     10¹¹ parameters                   ~89,000 lines of physics equations
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

## License

- **Source code**: [GNU Affero General Public License v3.0](LICENSE)
- **Papers** (`paper/*.tex`): [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## Citation

```bibtex
@software{huang2026alice,
  author       = {Huang, Hsi-Yu},
  title        = {Alice Smart System — Physics-Driven Medical Lifeform Simulator},
  year         = 2026,
  version      = {2.0.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18795843},
  url          = {https://github.com/cyhuang76/alice-gamma-net}
}
```

---

## Keywords

`computational-neuroscience` · `cognitive-architecture` · `medical-simulation` ·
`impedance-matching` · `coaxial-transmission-line` · `physics-driven` ·
`clinical-pathology` · `computational-pharmacology` · `digital-twin` ·
`heterogeneous-topology` · `irreducibility-theorem` · `fractal-dimension` ·
`sleep-cycles` · `language-emergence`
