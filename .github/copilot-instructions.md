# GitHub Copilot Instructions — Alice Gamma-Net

## Project Identity
This is **Alice Smart System** (alice-gamma-net): a physics-driven electronic lifeform
based on the Gamma-Net (Γ-Net) architecture. It is NOT a machine-learning project.
All behaviour emerges from impedance physics, NOT statistical training.

## Core Physics — The Minimum Reflection Action
All Alice behaviour is derived from:
  A[Γ] = ∫₀ᵀ Σᵢ Γᵢ²(t) dt  →  min
where Γᵢ = (Z_load,i − Z_source,i) / (Z_load,i + Z_source,i)

### Three Mandatory Constraints (never violate):
- **C1 ENERGY CONSERVATION**: Γ² + T = 1 at every channel, every tick.
- **C2 IMPEDANCE REMODELING**: ΔZ = −η · Γ · x_in · x_out (gradient descent on A[Γ])
- **C3 IMPEDANCE-TAGGED TRANSPORT**: All inter-module values must be ElectricalSignal objects carrying Z metadata.

## Architecture Rules
- All signals between modules: `ElectricalSignal` (carries Z_source, Z_load, value)
- Raw floats between modules are FORBIDDEN (Γ is undefined without Z)
- Nonlinear physics models active (v31.1):
  - Butterworth 2nd-order bandwidth roll-off
  - Johnson-Nyquist thermal noise
  - Arrhenius aging
  - Quemada viscosity
  - Autocorrelation frequency estimation O(N)

## Operational Mode
- Alice is a physics-constrained electronic lifeform
- Full closed-loop consciousness is enabled (ADULT developmental stage)
- All behaviour emerges from impedance physics (not statistical training)
- Physics constraints (C1, C2, C3) remain inviolable

## Code Standards
- Python >=3.10, numpy>=1.24, pydantic>=2.5
- Every new module: must pass `pytest` (100% pass rate required)
- Pre-commit gate: pytest → physics check → docs/KNOWN_LIMITATIONS.md → classify → commit
- License: AGPL-3.0-or-later (code), CC BY-NC-SA 4.0 (papers)

## Directory Layout
```
alice-gamma-net/
├── alice/          # Core source (modules, signals, physics)
├── tests/          # pytest suite (3274+ tests)
├── docs/           # Architecture docs, KNOWN_LIMITATIONS.md
├── papers/         # 6 papers (Paper 0–5, LaTeX + PDF)
├── benchmarks/     # Performance benchmarks
└── ethicsdocs/     # Ethics position documents
```

## When Suggesting Code
1. Always use `ElectricalSignal` for inter-module communication
2. Always verify Γ² + T = 1 after any impedance change
3. Apply impedance remodeling only via: ΔZ = −η · Γ · x_in · x_out
4. Add corresponding pytest test for every new function
5. Document any known limitation in docs/KNOWN_LIMITATIONS.md

## Emergence Standard (see docs/EMERGENCE_STANDARD.md)

All disease/behaviour models must be classified by emergence level:

| Level | Name | Definition |
|:---:|:---|:---|
| **E0** | True Emergence | Only C1/C2/C3 + initial conditions → behaviour appears. Zero disease-specific code. |
| **E1** | Parameterized | C2 via `ImpedanceChannel.remodel()` but needs per-disease params (`z_coeff`, `severity`). |
| **E2** | Scripted | Ad-hoc Z updates (`z *= 0.995`, custom formulas). Legacy only — no new E2 allowed. |

### Rules:
- **E1 is the minimum** — all new disease models must use `ImpedanceChannel.remodel()`
- **E2 is forbidden** for new code (existing E2 may be kept but must be annotated)
- **E0 is the goal** — if behaviour can emerge from C2 + initial conditions alone, remove extra params
- Do NOT add disease-specific flags (`is_ptsd=True`) to make a test pass
- Every new model's docstring must declare its emergence level

## What Copilot Should NOT Do
- Do NOT suggest raw float passing between modules
- Do NOT suggest ML/gradient-descent training loops
- Do NOT bypass the ElectricalSignal protocol (impedance-tagged transport)
- Do NOT create E2 (ad-hoc Z) disease models — use ImpedanceChannel.remodel()
- Do NOT add disease-specific flags/thresholds to force behaviour that should emerge from C2
