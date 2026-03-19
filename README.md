# Alice Smart System — v3.9.1

<div align="center">

**The world's first physics-driven electronic lifeform simulator.**

*No training data. No black boxes. Pure impedance physics.*

> **「公式是固定的，結果是透明的，驗證權在你手上。」**
> *"Fixed formulas. Transparent results. Verification is yours."*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18751831.svg)](https://doi.org/10.5281/zenodo.18751831)
[![Tests](https://img.shields.io/badge/tests-3274%20passed-brightgreen.svg)]()
[![Papers](https://img.shields.io/badge/papers-6%20published-orange.svg)](paper/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)

**[>> Launch Live Verification Platform (browser-only, zero server) <<](https://cyhuang76.github.io/alice-gamma-net/app/)**

</div>

---

## What is Alice?

Alice is a **digital organism** — not an AI model. She has a body (eyes, ears, hands, 31 organs), a brain (42 neural modules), feels pain, sleeps, dreams, ages, and can develop 125+ clinical pathologies across 13 medical specialties.

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

## Physics Glossary — Code as Physics

Every biology term in this codebase has a strict impedance-physics definition.
If you see a familiar word, it is **not** a metaphor — it is the human-readable alias
for a measurable impedance quantity.

| Biology Term | Physics Definition | Symbol / Equation | Where Defined |
|:---|:---|:---|:---|
| **Sleep** | Global decoupled dissipation state (offline impedance renormalization) | $D_Z^{-} = \int_{sleep} R(\tau)\,d\tau$ | Paper 3 / `sleep_physics.py` |
| **Memory** | Hysteretic topological deformation (no read/write — current flows through shaped Z) | $\partial Z / \partial t \neq 0$ | Paper 1 / `working_memory.py` |
| **Consciousness** | Mean transmission coefficient across monitored channels | $\Phi = (1/N)\sum(1 - \Gamma_i^2)$ | Paper 1 / `awareness_monitor.py` |
| **Soul** | Invariant topological core (null space of impedance change) | $\ker(\partial Z / \partial t)$ | Paper 1 |
| **Emotion** | 8-dimensional Plutchik impedance map (VAD space) | $Z_e = Z_0(1-E_i)$ | Paper 3 / `emotion_granularity.py` |
| **Willpower** | Top-down prefrontal forcing function (three-factor product) | $\eta_{\text{pfc}}^{\text{eff}} = \eta \cdot Q_{\text{gate}} \cdot (1-\langle\Gamma^2_{\text{mem}}\rangle)$ | Paper 4 / `prefrontal.py` |
| **Health** | Global power-transfer efficiency (product of all organ transmissions) | $H = \prod_i(1-\Gamma_i^2)$ | Paper 2 / `gamma_engine.py` |
| **Allostatic load** | Cumulative exogenous set-point forcing | $L_{\text{allo}} = \int f_{\text{ext}}\,dt$ | Paper 5 |
| **Disease** | Pathological attractor (impedance lock-in at $\Gamma \to 1$) | $dZ/dt \to 0,\;\Gamma \gg 0$ | Paper 4 |
| **Comorbidity** | Coupled-subsystem divergence (off-diagonal C matrix) | $\lambda_+ > 0$ when $C_{kj}C_{jk} > \eta^2$ | Paper 4 |
| **Pain** | Protocol collapse (reflected energy at impedance mismatch) | $P_{\text{reflected}} = \Gamma^2 \cdot P_{\text{in}}$ | Paper 0 / `signal.py` |
| **Dopamine** | Impedance-matching improvement signal (reward prediction error) | $\delta = R - (1-\Gamma^2)P_{\text{in}}$ | `basal_ganglia.py` |
| **Aging** | Arrhenius-driven irreversible impedance drift | $\delta(t) = \delta_0 e^{E_a \cdot \text{stress}}$ | Paper 3 / `lifecycle_equation.py` |

> *"C" in C1/C2/C3 = **Constraint** (not "condition"). These are physics axioms, not design choices.*

---

## Gamma-Net Health Check

> Zero-parameter impedance physics &middot; Public verification platform

This is **not** a model you need to trust — it is a **fixed-equation physical law** you can verify.

| | Typical ML Model | Γ-Net |
|:---|:---|:---|
| Parameters | Millions (fitted to data) | **Zero** (fixed by physics) |
| Trust model | "Believe our AUC" | "Run it yourself" |
| Audit | Requires expertise | Anyone with a health report |
| Data collection | Server-side | **Browser-only, offline-capable** |

### Three Walls of Verification

1. **Wall 1 — Self-computation**: Enter your health-check values → the calculator runs entirely in your browser via [Pyodide](https://pyodide.org) (Python-in-WebAssembly). No server, no data collection, no account.
2. **Wall 2 — Textbook side-by-side**: Same inputs produce **6 published clinical risk scores** alongside the 12-organ Γ-vector:
   - **Cardiovascular**: Framingham ATP-III, ASCVD Pooled Cohort (Goff 2014), SCORE2 (ESC 2021)
   - **Renal**: eGFR CKD-EPI 2021 (race-free)
   - **Hepatic**: FIB-4 liver fibrosis (Sterling 2006)
   - **Metabolic**: HOMA-IR + Metabolic Syndrome (ATP-III)
   
   Where they agree ✓. Where they disagree → Γ-Net shows *which organ channel* drives the discordance.
3. **Wall 3 — Falsifiable prediction**: Γ-Net predicts which organ will deteriorate first, and tracks cumulative impedance debt $D_Z = \int \sum \Gamma_i^2\, dt$ over time. Two patients with identical current labs but different $D_Z$ have **different prognoses** — only Γ-Net captures this.

**[>> Open Web App <<](https://cyhuang76.github.io/alice-gamma-net/app/)** — runs 100% in your browser, works offline after first load.

> **Disclaimer**: 本頁結果僅作研究與自我驗證參考，不構成任何醫療建議，請以正式醫療檢查及專業醫師判斷為準。

```bash
# CLI: textbook vs Γ-Net, four clinical cases
py -3.11 experiments/demo_dual_rail.py

# CLI: enter your own health-check data
py -3.11 experiments/demo_dual_rail.py --interactive

# CLI: 5-year longitudinal D_Z impedance debt tracking
py -3.11 experiments/demo_dual_rail.py --timeline
```

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

# Run all 3,274 tests
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
| Source files (`.py`) | 220+ |
| Lines of code | 93,000+ |
| Unit tests | 3,274 |
| Experiments | 27 |
| Body organs | 31 |
| Brain modules | 42 |
| Disease models | 125 |
| Medical specialties | 13 |
| Lab items mapped | 53 |
| Paper figures | 30 |
| Testable predictions | 18 |

---

## Paper Series

| # | Title | Pages | Status |
|---|-------|:-----:|--------|
| 0 | From the First Law of Thermodynamics to Three Inviolable Constraints — Framework Definition | 6 | **Published** ([DOI](https://doi.org/10.5281/zenodo.18751831)) |
| 1 | From Neuron to Mind — Irreducible Topology, Neural Architecture, Memory, Consciousness, and Soul | 15 | **Complete** |
| 2 | Dual Impedance Networks and Metabolic Scaling — Vascular Architecture, Organ Health, and Kleiber's Law | 9 | **Complete** |
| 3 | Temporal Dynamics and Universal Remodeling — Impedance Debt, Sleep, Lifecycle, Aging, and 29 C2 Networks | 21 | **Complete** |
| 4 | Topological Pathology — Disease as Impedance Failure on the Γ-Network | 16 | **Complete** |
| 5 | Complete Empirical Validation — Γ as the Universal Interface Law from Microtubules to Power Grids | 15 | **Complete** |

Key results:
- **Paper 0**: Derives the complete axiomatic framework from the First Law of Thermodynamics. C1/C2/C3 emerge as *theorems*, not postulates. C2 proven as the *unique* first-order activity-gated update rule minimising reflected energy.
- **Paper 1**: **Irreducibility theorem** — heterogeneous networks have a geometric cost floor $A_{\text{cut}}$ that cannot be learned away. Relay nodes emerge as thermodynamic necessity. **Directional asymmetry**: top-down $A_{\text{cut}}=4$, bottom-up $A_{\text{cut}}=0$ ("knowing is easy, doing is hard"). **Bandwidth increase** via multi-stage relay (Chebyshev transformer analogy). **Memory** = hysteretic topological deformation ($\partial Z/\partial t$); **cognitive standing wave** = stable interference pattern encoding concepts. **Consciousness** = $1 - \Gamma_{\text{meta}}^2$ (self-referential loop). **Soul** = Invariant Topological Core ($\ker(\partial Z/\partial t)$). **Brain as $d\Gamma/dt$ detector**: change-detection replaces computation. **Cognitive E0 emergences**: attention, habituation, curiosity, fight-or-flight as C2 resource allocation. **Thermal noise floor** as thermodynamic attractor (Landauer). **Willpower as SNR**. **Cross-network resonance**: empathy = tuning fork effect.
- **Paper 2**: **Dual neural–vascular coupling** — $H = (1-\Gamma_n^2)(1-\Gamma_v^2)$. **Murray's Law** derived as variational minimum. **Kleiber's Law** ($B \propto M^{3/4}$) from fractal impedance cost. **Impedance disparity**: physical basis for extreme Z differences between tissues. **Blood as crosstalk isolator**: BBB = impedance wall preventing vascular Z fluctuations from polluting neural Z. Organ-specific $\Omega_i$ (bottom-up mean 1.376 ≈ top-down 1.33).
- **Paper 3**: **Impedance debt** ($D_Z$) and **sleep** as global decoupled dissipation state; adenosine ≡ $D_Z$ readout. **Lifecycle equation** from embryo to senescence. **29 C2 networks** (Hebb, Wolff, Glagov, Davis, immune memory, …) unified as parameter instances of one gradient-descent rule. PTSD & chronic stress as integral-path equivalence. Morphogenesis PDE, embryogenesis timeline, skill-decay anti-chronology.
- **Paper 4**: **Disease** — from single-organ pathology to multi-organ failure — emerges from impedance physics alone. 5-step dual-network cascade ($\Gamma_v\!\uparrow \to \rho\!\downarrow \to \Gamma_n\!\uparrow \to \text{autonomic}\!\downarrow \to \Gamma_v\!\uparrow\!\uparrow$). Coupled multi-organ dynamics with bifurcation analysis. **Willpower tri-factorization**: $\eta_{\text{pfc}}^{\text{eff}} = \eta \cdot Q_{\text{gate}} \cdot (1 - \langle\Gamma^2_{\text{mem}}\rangle)$. **Four Laws of the Null Space** (parallel to thermodynamics). **Moral Constraint Theorem**: blame on sealed patterns = pure injury.
- **Paper 5**: **Complete Empirical Validation** — Γ as universal interface law. Interface Scale Table (25 nm microtubules → 10⁶ m power grids). NHANES 10-cycle validation ($n=52{,}545$, zero fitted parameters, AUC 0.705). **Three-layer cascade validation**: binary topology (7/7 organs improved, $p=0.008$). **Five-layer AUC progression**: 0.604 → 0.617 (cascade) → 0.626 (BP+cascade) → 0.709 (Γ+LR, 12 weights). **Multi-organ comorbidity** (DM+Cardiac AUC = 0.853). **Framingham zero-parameter match** (0.682 vs 0.682). ε = 0.03 physical derivation. **18 testable predictions** with falsification criteria.

> **Methodological note**: Papers 0–4 include explicit circularity warnings — simulation data confirms the theory it is built from. Paper 5 breaks this circularity with NHANES external epidemiological data (10 cycles, $n=52{,}545$) and zero fitted parameters. Total: 82 pages, 30 figures, 18 testable predictions.

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
├── tests/                  # 93 test files, 3,274 tests
├── experiments/            # Organized experiment scripts
│   ├── figgen/             # 9 figure generators
│   ├── validation/         # 5 validation experiments
│   └── simulation/         # 13 simulation experiments
├── paper/                  # 6 papers (82 pages, LaTeX)
│   └── gammanet.sty        # Shared LaTeX macros
├── figures/                # 30 generated figures (fig_p{N}_*)
├── build_figures.py        # Unified figure generation API
├── docs/                   # Architecture docs
│   ├── app/                # GitHub Pages verification web app
│   ├── KNOWN_LIMITATIONS.md
│   ├── SYSTEM_MANUAL.md
│   ├── LAB_GAMMA_ENGINE.md
│   └── README.md           # Documentation index
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
  version = {3.9.1},
  doi     = {10.5281/zenodo.18751831},
  url     = {https://github.com/cyhuang76/alice-gamma-net}
}
```
