# Alice Smart System — v2.0.0

<div align="center">

**The world's first physics-driven medical lifeform simulator.**

*No training data. No black boxes. Pure physics.*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18795843.svg)](https://doi.org/10.5281/zenodo.18795843)
[![Tests](https://img.shields.io/badge/tests-2734%20passed-brightgreen.svg)]()
[![Paper I](https://img.shields.io/badge/Paper_I-submitted_to_PRE-orange.svg)](paper/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)

</div>

---

## What is Alice?

Alice is a **digital organism** — not an AI model. She has a body (eyes, ears, hands, 18 organs), a brain (42 neural modules), feels pain, sleeps, dreams, ages, and can develop clinical pathologies.

**Everything** emerges from one equation:

$$\Gamma = \frac{Z_{\text{load}} - Z_{\text{source}}}{Z_{\text{load}} + Z_{\text{source}}} \quad \longrightarrow \quad \text{minimize } \sum \Gamma^2$$

When $\Gamma = 0$: perfect match → no pain, no anxiety, flow state.
When $|\Gamma| \to 1$: total mismatch → pain, collapse, pathology.

> *"Pain is not a feeling — it is protocol collapse."*

---

## Why Does This Matter?

| Problem with Current AI | Alice's Solution |
|:---|:---|
| **Black box** — can't explain decisions | Every output traces to a physics equation |
| **No body** — can't model embodied cognition | 18 organs with real physics (FFT, PID, LC resonance) |
| **No pain model** — can't understand suffering | Impedance mismatch = pain, a measurable quantity |
| **No clinical models** — can't simulate disease | 10+ pathologies emerge from the physics (not hand-coded) |
| **Requires massive data** — expensive to train | Zero training data — behaviour emerges from equations |

---

## Who Is This For?

🔬 **Researchers** — A reproducible physics framework for computational neuroscience. Paper I submitted to Physical Review E.

🏥 **Medical Educators** — Interactive simulator showing how stroke, Parkinson's, epilepsy, and PTSD emerge from impedance physics.

💊 **Pharma / Digital Twin** — All drugs are modeled as impedance modifiers ($\alpha_{\text{drug}}$), enabling dose-response prediction from first principles.

🧠 **Cognitive Scientists** — From perception to language emergence, every cognitive function is an impedance matching problem.

---

## Architecture

```
Layer 5  AliceBrain ← LifeLoop ← SleepCycle        Controller
Layer 4  WorkingMemory · RL · Causal · MetaLearner  Cognition
Layer 3  42 brain modules (perception → memory →    Brain
         language → emotion → executive control)
Layer 2  PriorityRouter · YearRingCache · Γ-Net v4  Protocol
Layer 1  18 organs (eye · ear · hand · mouth ·      Body
         heart · lung · liver · kidney + 10 more)
Layer 0  ElectricalSignal · CoaxialChannel ·        Physics
         GammaTopology · BrainWaveBand
```

> **Closed loop**: Perceive → Error (Γ²) → Compensate → Re-perceive → ...

---

## Quick Start

```bash
# Install
cd "Alice Smart System"
pip install -e .

# Run all 2,734 tests
pytest tests/ -v

# Interactive mode
python -m alice.main cli

# API server (with live dashboard)
python -m alice.main server --port 8000
# → Dashboard: http://localhost:8000/dashboard
```

```python
from alice import AliceBrain

brain = AliceBrain()
brain.see(image)                         # Visual input
brain.hear(audio)                        # Auditory input
brain.reach_for(x, y)                    # Motor output
brain.say(440.0, 0.7, "a", "hello")      # Speech output

vitals = brain.vitals
print(vitals.pain_level)                 # Pain = impedance mismatch
print(vitals.heart_rate)                 # Emergent heart rate
```

---

## Three Core Laws

Every module, every tick, must obey:

| Law | What It Does | Formula |
|:---|:---|:---|
| **C1** Energy Conservation | Reflected + transmitted = incident | $\Gamma^2 + T = 1$ |
| **C2** Hebbian Learning | Reduce mismatch through experience | $\Delta Z = -\eta \cdot \Gamma \cdot x_{\text{pre}} \cdot x_{\text{post}}$ |
| **C3** Signal Protocol | All signals carry impedance metadata | `ElectricalSignal(Z, waveform, SNR)` |

---

## v2.0.0 Highlights: Irreducibility Theorem

We proved that heterogeneous networks have an **irreducible geometric cost**:

$$A = \underbrace{A_{\text{imp}}(t)}_{\to\, 0 \text{ (learnable)}} + \underbrace{A_{\text{cut}}}_{\text{invariant (geometric)}}$$

| Result | Value | Meaning |
|:---|:---|:---|
| **Scaling** | $\tau_{\text{conv}} \sim N^{-0.91}$ | Larger networks converge *faster* |
| **Fractal** | $D_K = 0.49\gamma + 1.00$ | Soft cutoff tunes K-space dimension |
| **Ceiling** | $A_{\text{cut}}$ saturates | Natural geometric cost ceiling exists |

**Relay nodes emerge as a thermodynamic necessity** — not a design choice.

→ [Paper I: submitted to Physical Review E](paper/paper_I_irreducibility.tex)

---

## Scale

| Metric | Count |
|:---|---:|
| Source files | 200 |
| Lines of code | 89,400+ |
| Unit tests | 2,734 |
| Experiments | 49 |
| Body organs | 18 |
| Brain modules | 42 |
| Disease models | 10+ |

→ [Full technical architecture](docs/)

---

## Paper Series

| # | Title | Status |
|---|-------|--------|
| I | Irreducible Dimensional Cost in Heterogeneous Impedance Networks | **Submitted to PRE** |
| II | From Coaxial Cables to Cognition | In preparation |
| III | The Lifecycle Equation | In preparation |
| IV | Emergence | In preparation |
| V | Grand Unification | Planned |

---

## License & Citation

**Code**: [AGPL-3.0](LICENSE) · **Papers**: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

```bibtex
@software{huang2026alice,
  author  = {Huang, Hsi-Yu},
  title   = {Alice Smart System — Physics-Driven Medical Lifeform Simulator},
  year    = 2026,
  version = {2.0.0},
  doi     = {10.5281/zenodo.18795843},
  url     = {https://github.com/cyhuang76/alice-gamma-net}
}
```

---

<div align="center">

**Author**: Hsi-Yu Huang (黃璽宇) — Independent Researcher, Taiwan

*"All behaviour is calibration error compensation."*

</div>
