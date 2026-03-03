# Alice Smart System — v3.2.0

<div align="center">

**The world's first physics-driven electronic lifeform simulator.**

*No training data. No black boxes. Pure impedance physics.*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18847656.svg)](https://doi.org/10.5281/zenodo.18847656)
[![Tests](https://img.shields.io/badge/tests-3084%20passed-brightgreen.svg)]()
[![Papers](https://img.shields.io/badge/papers-4%20published-orange.svg)](paper/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)

</div>

---

## What is Alice?

Alice is a **digital organism** — not an AI model. She has a body (eyes, ears, hands, 31 organs), a brain (42 neural modules), feels pain, sleeps, dreams, ages, and can develop 120+ clinical pathologies across 14 medical specialties.

**Everything** emerges from one equation:

$$\mathcal{A}[\Gamma] = \int_0^T \sum_i \Gamma_i^2(t)\, dt \;\to\; \min$$

where $\Gamma_i = \frac{Z_{\text{load},i} - Z_{\text{source},i}}{Z_{\text{load},i} + Z_{\text{source},i}}$

When $\Gamma = 0$: perfect match — no pain, no anxiety, flow state.
When $|\Gamma| \to 1$: total mismatch — pain, collapse, pathology.

> *"Pain is not a feeling — it is protocol collapse."*

---

## Three Core Laws

Every module, every tick, must obey:

| Law | What It Does | Formula |
|:---|:---|:---|
| **C1** Energy Conservation | Reflected + transmitted = incident | $\Gamma^2 + T = 1$ |
| **C2** Hebbian Learning | Reduce mismatch through experience | $\Delta Z = -\eta \cdot \Gamma \cdot x_{\text{pre}} \cdot x_{\text{post}}$ |
| **C3** Signal Protocol | All signals carry impedance metadata | `ElectricalSignal(Z, waveform, SNR)` |

---

## Architecture

```
Layer 6  Lab-Γ Diagnostic Engine (REST API + Web UI)  Diagnostics
Layer 5  AliceBrain · LifeLoop · SleepCycle            Controller
Layer 4  WorkingMemory · RL · Causal · MetaLearner     Cognition
Layer 3  42 brain modules (perception → memory →        Brain
         language → emotion → executive control)
Layer 2  PriorityRouter · YearRingCache · Γ-Net v4     Protocol
Layer 1  31 organs (eye · ear · hand · mouth ·          Body
         heart · lung · liver · kidney + 23 more)
Layer 0  ElectricalSignal · CoaxialChannel ·            Physics
         GammaTopology · BrainWaveBand
```

> **Closed loop**: Perceive → Error (Γ²) → Compensate → Re-perceive → …

---

## Quick Start

```bash
# Install
cd "Alice Smart System"
pip install -e .

# Run all 3,084 tests
pytest tests/ -v

# Lab-Γ Diagnostic API (Swagger: http://localhost:8420/docs)
python -m alice.diagnostics.api

# Lab-Γ Web Dashboard (with Γ radar chart)
streamlit run alice/diagnostics/web_ui.py

# Interactive mode
python -m alice.main cli

# Alice brain API server (with live dashboard)
python -m alice.main server --port 8000
```

---

## Lab-Γ Diagnostic Engine (NEW)

53 laboratory values → 12 organ impedances → Γ vector → 125 disease templates.

```
  Lab values → Z_organ = Z_normal × (1 + Σ w_j·|δ_j|)
             → Γ_organ = (Z_patient − Z_normal) / (Z_patient + Z_normal)
             → Template match → Ranked differential diagnosis
             → Physician feedback → C2 Hebbian weight update
```

| Phase | Feature | Tests |
|:---:|:---|:---:|
| 1 | Lab → Z → Γ → 125 disease matching | 66 |
| 2 | C2 Hebbian feedback (confirm/reject/correct) | 34 |
| 3 | FastAPI REST (10 endpoints) + Streamlit UI + Γ radar chart | 25 |

**API Endpoints:**
| Method | Path | Description |
|:---:|:---|:---|
| POST | `/diagnose` | Lab values → differential diagnosis |
| POST | `/gamma` | Lab values → raw 12-D Γ vector |
| POST | `/feedback` | Physician confirms/rejects/corrects |
| GET | `/templates` | List 125 disease templates |
| GET | `/organs` | 12 organ impedance reference |
| GET | `/labs` | 53 supported laboratory items |
| GET | `/health` | API health check |

---

## Scale

| Metric | Count |
|:---|---:|
| Source files (`.py`) | 283 |
| Lines of code | 113,000+ |
| Unit tests | 3,084 |
| Experiments | 83 |
| Body organs | 31 |
| Brain modules | 42 |
| Disease models | 125 |
| Medical specialties | 14 |
| Lab items mapped | 53 |
| Figures | 29 |

---

## Paper Series

| # | Title | Pages | Status |
|---|-------|:-----:|--------|
| I | Irreducible Dimensional Cost in Heterogeneous Impedance Networks | 5 | **Published** ([DOI](https://doi.org/10.5281/zenodo.18847656)) |
| II | Dual Neural–Vascular Impedance Networks: Architecture & Sleep | 10 | **Complete** |
| III | Impedance Debt, Sleep Homeostasis, and the Evolution of Brains | 13 | **Complete** |
| IV | The Lifecycle Equation: From Embryo to Senescence | 10 | **Complete** |

Key results:
- **Paper I**: Irreducibility theorem — heterogeneous networks have a geometric cost floor $A_{\text{cut}}$ that cannot be learned away. Relay nodes emerge as thermodynamic necessity.
- **Paper II**: Dual neural–vascular coupling — $H = (1-\Gamma_n^2)(1-\Gamma_v^2)$. Vascular debt accumulates on weeks-to-years timescale.
- **Paper III**: Adenosine ≡ impedance debt readout. No-waste corollary (C1 → every molecule has function). Cambrian explosion as Γ phase transition.
- **Paper IV**: Lifecycle equation $L(t) = \prod_i [1 - \Gamma_i^2(t)]$ from embryo to senescence, with Arrhenius aging.

---

## Directory Layout

```
alice-gamma-net/
├── alice/                  # Core source code
│   ├── brain/              # 42 neural modules
│   ├── body/               # 31 body organs
│   ├── core/               # Protocol, signals, physics
│   ├── diagnostics/        # Lab-Γ Engine (API + feedback + UI)
│   ├── modules/            # Shared module infrastructure
│   └── api/                # Brain REST API server
├── tests/                  # 82 test files, 3,084 tests
├── experiments/            # 83 experiment scripts
├── paper/                  # 4 papers (LaTeX source)
├── figures/                # 29 generated figures
├── docs/                   # Architecture docs
│   ├── KNOWN_LIMITATIONS.md
│   ├── SYSTEM_MANUAL.md
│   ├── LAB_GAMMA_ENGINE.md
│   └── ...
├── pyproject.toml          # Project metadata & dependencies
├── CITATION.cff            # Citation metadata
└── LICENSE                 # AGPL-3.0-or-later
```

---

## Nonlinear Physics Models (v31.1)

| Model | Formula | Purpose |
|:---|:---|:---|
| Butterworth roll-off | $H(f) = 1/\sqrt{1+(f/f_0)^4}$ | Bandwidth limiting |
| Johnson-Nyquist noise | $N = k_B T \Delta f \cdot L \cdot (1+\Gamma^2)$ | Thermal noise |
| Arrhenius aging | $\delta = \delta_0 \exp(E_a \cdot \text{stress})$ | Component aging |
| Quemada viscosity | $\eta = \eta_0 (1 + a \phi + b \phi^3)$ | Blood viscosity |
| Autocorrelation freq | $O(N)$, no FFT | Frequency estimation |

---

## License & Citation

**Code**: [AGPL-3.0](LICENSE) · **Papers**: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

```bibtex
@software{huang2026alice,
  author  = {Huang, Hsi-Yu},
  title   = {Alice Smart System — Physics-Driven Electronic Lifeform Simulator},
  year    = 2026,
  version = {3.2.0},
  doi     = {10.5281/zenodo.18847656},
  url     = {https://github.com/cyhuang76/alice-gamma-net}
}
```
