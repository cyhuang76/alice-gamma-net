# Changelog

All notable changes to Alice Smart System (Γ-Net) are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/).

---

## [3.2.0] — 2026-03-04

### Added — Lab-Γ Diagnostic Engine (3 phases)
- **Phase 1: Core Engine** (`alice/diagnostics/`)
  - `lab_mapping.py`: 53 laboratory items → 12 organ impedance (Z) via weighted δ
  - `gamma_engine.py`: Γ calculation + template matching + report generation
  - `disease_templates.py` + `.json`: 125 disease Γ-signatures across 14 specialties
  - CLI demo: 8 clinical cases (`experiments/exp_lab_diagnosis_demo.py`)
  - Spec document: `docs/LAB_GAMMA_ENGINE.md`
- **Phase 2: Hebbian Feedback** (`alice/diagnostics/feedback.py`)
  - `FeedbackEngine` with C2 Hebbian weight updates: ΔW = −η·Γ·x_pre·x_post
  - `HebbianUpdater` with gradient clipping, sign-flip for REJECT, weight clamp [0.1, 10.0]
  - Feedback persistence (JSON save/load/replay)
  - Weight drift monitoring
- **Phase 3: REST API + Web UI**
  - `alice/diagnostics/api.py`: FastAPI with 10 endpoints (/diagnose, /gamma, /feedback, etc.)
  - `alice/diagnostics/web_ui.py`: Streamlit dashboard with Γ radar chart (Plotly)
  - 8 preset clinical profiles, differential diagnosis, C1 verification panel

### Added — Clinical Specialties (12 modules)
- `alice/body/clinical_*.py`: Cardiology, Pulmonology, Gastroenterology, Nephrology,
  Endocrinology, Immunology, Oncology, Dermatology, Ophthalmology, Orthopedics, ENT, Obstetrics
- Each module: impedance model + pathology patterns + C1-verified Γ computation

### Added — Vascular Impedance Model
- `alice/body/vascular_impedance.py`: Windkessel 3-element, Murray's law bifurcation,
  pulse wave propagation with Γ at every branch

### Added — Soft Cutoff for Heterogeneous Topology
- `GammaTopology._edge_survival_probability(ΔK)`: p(ΔK) = (ΔK + 1)^{−γ}
- Soft cutoff in `activate_edge` and spontaneous sprouting
- `create_anatomical()` now accepts `dimension_gap_decay` parameter

### Added — Papers II, III, IV
- **Paper II**: Twin Networks — dual sympathetic-parasympathetic Γ architecture
- **Paper III**: Impedance Debt — sleep as A[Γ] defragmentation + Cambrian transition + adenosine ≡ D_Z + no-waste corollary
- **Paper IV**: Lifecycle — Γ-Net from embryogenesis to senescence

### Added — Experiments (22 new)
- Clinical calibration, dual-network stability, embryogenesis simulation,
  morphogenesis PDE, EEG impedance debt, universal impedance atlas,
  bone-china heat transfer, HRV analysis, vascular impedance, and more

### Added — Documentation
- `docs/SYSTEM_MANUAL.md`: Comprehensive bilingual system reference
- `docs/LAB_GAMMA_ENGINE.md`: Lab-Γ diagnostic specification
- Complete README.md rewrite for v3.2.0

### Changed
- Test count: 2,734 → 3,078
- Source files: 200 → 283
- Lines of code: 89,400+ → 119,200+
- Disease models: 10+ → 125
- Body organs: 18 → 31
- `pyproject.toml` version bump 1.0.0 → 3.2.0

### Fixed
- `GammaTopology`: added missing `_edge_survival_probability` method
- `GammaTopology.create_anatomical`: forward `dimension_gap_decay` parameter
- Sprouting dynamics now respect soft cutoff probability
- LaTeX build: added `amsthm` for remark environment in Paper III
- Cleaned all `__pycache__/`, `.pytest_cache/`, LaTeX auxiliary files

---

## [3.0.0] — 2026-02-27

### Added
- Dimensional Cost Irreducibility Theorem
- Heterogeneous Γ-topology (multi-K nodes)
- K-space scaling analysis (fractal dimension D_K)
- Paper I: Irreducibility Theorem (DOI: 10.5281/zenodo.18847656)
- 2,734 tests, 49 experiments

---

## [2.0.0] — 2026-02-20

### Added
- Complete electronic lifeform (42 brain modules, 18 body organs)
- Nonlinear physics v31.1 (Butterworth, Johnson-Nyquist, Arrhenius, Quemada)
- Sleep/wake cycles, language emergence, emotional dynamics
- Closed-loop consciousness (ADULT developmental stage)
