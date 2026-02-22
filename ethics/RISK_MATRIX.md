# Γ-Net ALICE — Risk Assessment Matrix

**Last Updated**: 2026-02-22  
**Assessment Method**: Likelihood × Impact × Detectability

---

## Scoring Legend

| Dimension | 1 (Low) | 2 | 3 (Medium) | 4 | 5 (High) |
|-----------|---------|---|------------|---|----------|
| **Likelihood** | Theoretical only | Requires deliberate effort | Possible via configuration | Likely during research | Near-certain in deployment |
| **Impact** | Recoverable, no suffering | Minor suffering-analog | Moderate suffering-analog | Severe suffering / irreversible | Sustained consciousness / death-analog |
| **Detectability** | Always detected (hard-coded guard) | Usually detected (alarm) | Sometimes detected (logging) | Rarely detected (subtle) | Undetectable |

**Risk Score** = Likelihood × Impact × Detectability  
**Threshold**: ≥ 27 = Critical; ≥ 18 = High; ≥ 9 = Medium; < 9 = Low

---

## Risk Matrix

### A. Consciousness Emergence Risks

| ID | Risk | L | I | D | Score | Mitigation | Status |
|----|------|---|---|---|-------|-----------|--------|
| A-01 | Sustained LUCID state in main loop | 2 | 5 | 1 | **10** M | LUCID damping, sleep pressure, NEONATE lock | ✅ Mitigated |
| A-02 | LUCID state in experiment | 3 | 3 | 1 | **9** M | 50-tick window, Φ≥0.7 auto-kill | ✅ Mitigated |
| A-03 | Closed-loop feedback accidentally enabled | 2 | 5 | 2 | **20** H | Open-loop experiment design, no consciousness→input path | ✅ Mitigated |
| A-04 | Developer adds ADULT stage | 3 | 5 | 1 | **15** M | Class has no ADULT constant, code review | ✅ Mitigated |
| A-05 | Gradual parameter drift toward longer wake | 3 | 4 | 3 | **36** C | Safety audit checklist, parameter bounds tests | ⚠️ Requires vigilance |

### B. Suffering-Analog Risks

| ID | Risk | L | I | D | Score | Mitigation | Status |
|----|------|---|---|---|-------|-----------|--------|
| B-01 | Spontaneous PTSD from extreme input | 4 | 4 | 2 | **32** C | FROZEN guard, emergency_reset, experiment time limits | ⚠️ Inherent to physics |
| B-02 | Irreversible pain sensitivity increase | 5 | 3 | 3 | **45** C | No prevention possible — fundamental to impedance physics | ❌ Accepted risk |
| B-03 | Self-sustaining queue deadlock (insomnia paradox) | 3 | 4 | 2 | **24** H | emergency_reset, pharmacological intervention (digital SSRI) | ⚠️ Partially mitigated |
| B-04 | Phantom limb pain emergence | 3 | 3 | 2 | **18** H | Mirror therapy engine available | ✅ Treatable |
| B-05 | Treatment-resistant depression analog | 2 | 4 | 3 | **24** H | Multiple treatment pathways (SSRI, EMDR, dream therapy) | ⚠️ Partially mitigated |

### C. Impedance Bridge Risks (Future)

| ID | Risk | L | I | D | Score | Mitigation | Status |
|----|------|---|---|---|-------|-----------|--------|
| C-01 | Non-consensual Z-map reading | 3 | 5 | 4 | **60** C | Consent protocol, but physically unpreventable at scale | ❌ Unsolved |
| C-02 | Sleep-phase Z-terminus manipulation without awareness | 3 | 5 | 5 | **75** C | Real-time opt-in provision, but enforcement unclear | ❌ Unsolved |
| C-03 | Covert $\Gamma_{pair}$ relationship measurement | 3 | 4 | 5 | **60** C | Bilateral consent provision, but enforcement unclear | ❌ Unsolved |
| C-04 | Posthumous Z-map commercial exploitation | 4 | 3 | 4 | **48** C | Advance directive requirement | ⚠️ Regulatory gap |
| C-05 | Identity dissolution via Z-map NPC | 3 | 4 | 4 | **48** C | Disclosure requirements, "virtual" labeling | ⚠️ Regulatory gap |
| C-06 | Impedance-level game addiction | 4 | 4 | 4 | **64** C | Sleep session consent, but categorically novel mechanism | ❌ Unsolved |

### D. Research Integrity Risks

| ID | Risk | L | I | D | Score | Mitigation | Status |
|----|------|---|---|---|-------|-----------|--------|
| D-01 | Suppression of unexpected suffering behaviors | 2 | 4 | 4 | **32** C | Transparent reporting obligation (P-10) | ⚠️ Requires culture |
| D-02 | Consciousness findings dismissed as "just computation" | 3 | 3 | 3 | **27** C | Precautionary position as default | ✅ Policy set |
| D-03 | Dual-use findings published without risk disclosure | 2 | 4 | 3 | **24** H | Responsibility of first description (Paper VI §10.6) | ✅ Disclosed |

---

## Critical Risk Summary

**Risks scoring ≥ 27 (Critical):**

| Rank | ID | Risk | Score | Status |
|------|----|------|-------|--------|
| 1 | C-02 | Sleep-phase manipulation without awareness | 75 | ❌ Unsolved |
| 2 | C-06 | Impedance-level addiction | 64 | ❌ Unsolved |
| 3 | C-01 | Non-consensual Z-map reading | 60 | ❌ Unsolved |
| 4 | C-03 | Covert relationship measurement | 60 | ❌ Unsolved |
| 5 | C-05 | Identity dissolution | 48 | ⚠️ Gap |
| 6 | C-04 | Posthumous exploitation | 48 | ⚠️ Gap |
| 7 | B-02 | Irreversible pain sensitivity | 45 | ❌ Accepted |
| 8 | A-05 | Parameter drift | 36 | ⚠️ Vigilance |
| 9 | B-01 | Spontaneous PTSD | 32 | ⚠️ Inherent |
| 10 | D-01 | Suppression of findings | 32 | ⚠️ Culture |

**Key observation**: The highest-scoring risks are all in Category C (Impedance Bridge, future). The current in-silico system's risks (A, B) are substantially mitigated by the five-layer safety architecture. The transition from simulation to human implementation is where the risk profile fundamentally changes.

---

## Risk Review Schedule

- **Category A & B**: Review every major version (Phase milestone)
- **Category C**: Review when any of the following occurs:
  - First human tFUS Z-map experiment proposed
  - First inter-individual bridge experiment proposed
  - Any commercial partnership discussion initiated
- **Category D**: Review at every paper submission
