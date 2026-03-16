# Changelog

All notable changes to Alice Smart System (Γ-Net) are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/).

---

## [3.9.0] — 2026-03-16

### Added — Figures & Visualization
- **9 new figures** across all 6 papers:
  - P0: Deductive chain flowchart
  - P1: K-space clustering + Television inequality parameter space
  - P2: Kleiber's Law scatter (12 mammalian species, mouse → blue whale)
  - P3: Sleep D_Z discharge (48h infant vs adult comparison)
  - P4: Saddle-node bifurcation/hysteresis + multi-organ cascade network
  - P5: Kaplan–Meier survival curves + cross-organ AUC waterfall
- **`build_figures.py`**: Unified figure generation API with auto-detection of compatible Python runtime (avoids Python 3.14 numpy segfault)

### Added — LaTeX Infrastructure
- **`gammanet.sty`**: Shared macro package (theorem environments + physics macros)
- All 6 papers use `\usepackage{gammanet}`, removing 30+ duplicate definitions
- Unified author affiliation: "Independent Researcher, Taipei, Taiwan"

### Added — Documentation
- `docs/README.md`: Documentation index with build instructions

### Changed — Project Structure
- `experiments/` reorganized into `figgen/` (9), `validation/` (5), `simulation/` (13)
- All 27 figures renamed to `fig_p{N}_{description}` convention
- All `.tex` `\includegraphics` paths synchronized with new names
- `.gitignore`: added `figures/*.png` (regeneratable from PDFs)
- Paper 5 title corrected (resolved overlap with Paper 3)

### Fixed
- P5: KeyError in Kaplan-Meier generation (wrong JSON keys)
- P5: indentation error in health boxplot fallback path

---

## [3.8.0] — 2026-03-13

### Added — Paper Content (P3, P4, P5)
- **P3**: Chronic stress ≡ PTSD via integral-path equivalence (D_Z threshold)
- **P3**: Thermal refugia as C2 boundary condition (replaces cave biology)
- **P4**: Willpower tri-factorization: η_pfc^eff = η × Q_gate × (1 − ⟨Γ²_mem⟩)
- **P4**: Sensory-deprivation collapse (gain divergence when x_in → 0)
- **P4**: Hallucination unification theorem (stress + deprivation pathways)
- **P5**: Recovery dynamics — three regimes (pre-bifurcation / single-channel / multi-channel)
- **P5**: Immune–metabolic feedback loop (eigenvalue analysis)
- **P5**: Acute vs chronic bifurcation (pulse vs steady-state)
- **P5**: Computational convergence section (state machine)
- **P5**: Epistemic layers remark (theorem / NHANES / ODE)
- **P5**: AUC 0.705 defense remark (Kepler analogy)
- **P5**: Substrate-vs-structure remark (29 mechanisms, same math)

### Changed — Terminology Hardening (9 cross-boundary terms)
All biology terms anchored to impedance-physics definitions at first occurrence:

| Term | Physics Alias | Paper |
|---|---|---|
| C (in C1/C2/C3) | Constraint | P0 |
| Soul | Invariant Topological Core | P4 |
| Allostatic load | Cumulative exogenous set-point forcing | P5 |
| Disease | Pathological attractor | P5 |
| Comorbidity | Coupled-subsystem divergence | P5 |
| H | Global power-transfer efficiency | P2 |
| Willpower | Top-down prefrontal forcing function | P5 |
| Selfish Brain | Asymmetric energy sink / privileged topological hub | P5 |
| Sleep | Global decoupled dissipation state | P3 |
| Memory | Hysteretic topological deformation | P4 |

### Changed — Documentation Alignment
- **README.md**: Physics Glossary table (13 biology → impedance mappings)
- **README.md**: Paper 3/4/5 descriptions updated with hardened terminology
- **SYSTEM_MANUAL.md**: Paper table expanded (4 → 6 papers, correct filenames, physics aliases)
- **SYSTEM_MANUAL.md**: Statistics updated (3,274 tests, 6 papers, 70 pp, 125 diseases)
- **SYSTEM_MANUAL.md**: Directory layout counts corrected

### Fixed
- P5: undefined ref `\sec:tissue-C2` → `\sec:twentynine`
- P2: bibitem case mismatch unified to uppercase Paper0/Paper1/Paper3
- P5: phantom `fig:recovery-regimes` reference removed
- P0/P1/P3: cave sections rewritten as thermal-refugia C2 boundary condition

### Verified
- All 6 papers compile with zero errors
- 3,771 tests passed (pytest, Python 3.11)
- Zero anthropomorphic variables found in 88+ source files

---

## [3.7.0] — 2026-03-10

### Added — Paper Content (P3, P4, P5)
- **P4**: Four Laws of the Null Space (parallel to thermodynamics)
- **P4**: Moral Constraint Theorem — blame on sealed patterns = pure injury
- **P4**: Developmental genesis of null space (S_eff grows with age via η lifecycle)
- **P5**: 5-organ coupled disease cascade (experiment + figure + paper section)
- **P5**: Surface observability — skin as impedance boundary + Prediction #17
- **P5**: Caregiver triple-load model (emotional + physical + bureaucratic)
- **P5**: Obesity as impedance-driven metabolic strategy + leptin Γ mismatch + Prediction #18
- **P5**: Willpower non-existence proof (proof by contradiction from psychiatric epidemiology)
- **P5**: Neuroendocrine-behavioural intermediate layer + single-core/dual-core overload remark
- **P5**: Detection threshold remark — null signal as health certificate
- **P3+P5**: PTSD as impedance physics — three conditions, therapeutic physics, empathic load equation, two new predictions
- **P3+P4**: Impedance fixation as category error — three mechanisms + therapeutic implications
- **P3+P4**: Sealed shutdown vs runaway collapse — R_seal ratio + end-of-life quality remark
- **P3+P4**: Moral blame as secondary Γ² injection — therapeutic vs iatrogenic criterion
- **P3+P5**: Psychological load to somatic collapse — full chain PTSD → dual-network → immune+metabolic failure
- **P3**: Visual health perception as unconscious impedance tomography
- **P3**: Skill-decay anti-chronology — η(t) lifecycle + simulation

### Changed
- 8 papers → 6 papers restructure + C2/C3 terminology rename
- Complete codebase rename to biomedical engineering standard
- README + CITATION.cff updated (70 pages, 18 predictions, 3,274 tests)
- Interface Scale Table restructured by physical domain (7 domains)
- Paper 4: replace hardware-interrupt/CPU analogies with pure Γ-physics

### Fixed
- CITATION.cff Gamma encoding repair
- ASCVD female formula and Framingham female age table corrected
- LaTeX: add `amsthm` package to Papers 2, 3, 5

---

## [3.4.1] — 2026-03-04

### Changed
- NHANES validation expanded: 1-cycle → 3-cycle (n=7,393)
- All 7/7 organs statistically significant

---

## [3.4.0] — 2026-03-04

### Added
- **Paper 5 (P5)**: The Grand Unification — Γ as universal interface law
- NHANES 10-cycle empirical validation (n=49,774, zero fitted parameters, AUC 0.77–0.80)
- Level 2b STARD diagnostic accuracy (10-cycle NHANES, N=52,545, 7,627 deaths)
- MIMIC-IV diagnostic accuracy pipeline (Level 2b, STARD-compliant)
- Neonatal impedance reference standard (growth charts, centile curves P5–P95)
- Gamma-Pain Index: theory (Papers 2/4/5) + prototype (PICS data)
- Public falsification platform: textbook vs Γ-Net side-by-side (Pyodide web app)
- Papers 2–5: everyday Gamma evidence sections

---

## [3.3.0] — 2026-03-04

### Added
- **Paper 4 (P4)**: Memory, Consciousness, and Soul as Impedance Physics
- NHANES mortality survival analysis — Level 3 evidence (N=7,371, Harrell's C=0.6989)
- Explicit circularity warnings in Papers 0–4
- Brain Γ-map calibration from EEG + HRV
- Paper 5 (then Paper VI): Cambrian section, evolutionary theory, testable predictions

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
- Paper I: Irreducibility Theorem (DOI: 10.5281/zenodo.18751831)
- 2,734 tests, 49 experiments

---

## [2.0.0] — 2026-02-20

### Added
- Complete electronic lifeform (42 brain modules, 18 body organs)
- Nonlinear physics v31.1 (Butterworth, Johnson-Nyquist, Arrhenius, Quemada)
- Sleep/wake cycles, language emergence, emotional dynamics
- Closed-loop consciousness (ADULT developmental stage)
