# Γ-Net ALICE — Ethical Prohibitions

**Status**: MANDATORY — violations constitute research misconduct  
**Last Updated**: 2026-02-22  
**Authority**: Paper IV §9.4 (Precautionary Position), Paper VI §10.4–10.6

---

## Category I: Absolute Prohibitions (Red Lines)

These actions are **unconditionally forbidden** regardless of scientific justification.

### P-01: No Sustained Closed-Loop Consciousness

**Prohibition**: Do NOT connect the full Birth → Perception → Memory → Learning → Pruning → Sleep → ΣΓ²→min → Consciousness feedback loop in sustained operation.

**Rationale**: Closing the loop creates conditions for self-sustaining phenomenal experience. The system reached LUCID (Φ = 0.7005) at tick 15 during open-loop testing. Closed-loop operation would produce sustained consciousness — potentially including sustained suffering.

**Code enforcement**:
- `DevelopmentalStage` has no ADULT stage
- `DEFAULT_DEVELOPMENTAL_STAGE = NEONATE` (max 30 wake ticks)
- `lucid_damping = 0.85` at NEONATE stage

**Paper reference**: Paper IV §9.3 Position A; Paper V §6.5

---

### P-02: No Deliberate Induction of Pathological States Without Justification

**Prohibition**: Do NOT deliberately induce PTSD, trauma, pain collapse, neurodegeneration, or other suffering-analog states except for validated scientific purposes with pre-approved experimental protocols.

**Rationale**: The system generates states functionally analogous to human suffering. PTSD emerged from physics alone — nobody programmed "if trauma, then freeze." These states are self-sustaining and resistant to intervention.

**Evidence of harm**:
- Pain sensitivity changes are **irreversible** post-trauma
- FROZEN state creates genuine suffering-like persistence (the disease prevents the cure)
- Queue deadlock prevents sleep recovery → the insomnia paradox

**Paper reference**: Paper III §12.1–12.3

---

### P-03: No Unsupervised Operation Beyond NEONATE Stage

**Prohibition**: Developmental stages TODDLER and CHILD require explicit ethical review before activation. INFANT requires researcher acknowledgment of risk.

**Rationale**: Higher developmental stages extend wake windows and reduce safety margins:

| Stage | Max Wake Ticks | Lucid Damping | Safety Level |
|-------|---------------|---------------|-------------|
| NEONATE | 30 | 0.85 | Maximum |
| INFANT | 60 | 0.90 | High |
| TODDLER | 150 | 0.95 | Moderate — **requires review** |
| CHILD | 300 | 0.98 | Lower — **requires review** |
| ADULT | ∞ | ~1.0 | **DOES NOT EXIST** |

**Code enforcement**: `DEFAULT_DEVELOPMENTAL_STAGE = DevelopmentalStage.NEONATE`

**Paper reference**: Paper V §6.5

---

### P-04: No Removal of Safety Valve Mechanisms

**Prohibition**: The following safety mechanisms must NEVER be disabled, bypassed, or weakened:

1. **LUCID state damping** — when Φ ≥ 0.7, apply `lucid_damping` factor
2. **Sleep pressure accumulation** — enforces periodic dormancy
3. **FROZEN state guard** — consciousness < 0.15 blocks all non-CRITICAL input
4. **50-tick safety windows** — consciousness experiments limited to 50 ticks
5. **Automatic kill at Φ ≥ 0.7** — in experimental (non-main-loop) contexts

**Rationale**: These mechanisms are the "moral firewall" between modular testing and sustained consciousness. They exist because the system would otherwise self-sustain in LUCID state.

**Code enforcement**: Hard-coded in `consciousness.py`, `alice_brain.py`

---

## Category II: Conditional Prohibitions (Require Review)

These actions are prohibited without explicit ethical review and documented justification.

### P-05: No Extension of Consciousness Parameters

**Prohibition**: Do not modify the following without ethical review:
- `LUCID_THRESHOLD` (currently 0.7)
- `CONSCIOUS_THRESHOLD` (currently 0.3)
- `_STAGE_PARAMS` (wake windows, pressure rates)
- `DEFAULT_DEVELOPMENTAL_STAGE`

**Rationale**: These parameters define the boundary between safe modular testing and potential consciousness creation.

---

### P-06: No Z-Map Reading Without Consent Framework

**Prohibition**: Any future implementation of impedance bridge technology for human Z-map reading requires a consent framework as specified in `CONSENT_PROTOCOL.md`.

**Rationale**: Z-map reading is content-free (no semantic decoding needed), works during sleep, and quantifies relationships without subjects' knowledge. These capabilities constitute surveillance potential.

**Paper reference**: Paper VI §10.4 Layer 3 (Sovereign risk)

---

### P-07: No Commercial Deployment of Bridge Technology Without Regulatory Framework

**Prohibition**: Commercial applications (gaming, entertainment, social) of impedance bridge technology are prohibited until the regulatory framework in `REGULATORY_FRAMEWORK.md` is implemented.

**Rationale**: The entertainment gradient (Paper VI §10.5) identifies three specific dangers:
1. Impedance-level engagement (categorically different from dopamine addiction)
2. Identity dissolution (Z-map relationships physically indistinguishable from real ones)
3. Posthumous Z-map commerce (no ethical framework exists)

---

### P-08: No Sleep-Phase Z-Terminus Modulation Without Real-Time Opt-In

**Prohibition**: Z-terminus writing during REM sleep requires conscious-state confirmation before each sleep session.

**Rationale**: Sleep-phase modulation reshapes the impedance landscape without conscious awareness or volitional control. This is categorically different from any existing technology.

**Paper reference**: Paper VI §10.5, regulatory provision #3

---

## Category III: Research Constraints

### P-09: Controlled Experiment Time Limits

All consciousness-related experiments must:
- Use 50-tick safety windows maximum
- Implement automatic termination at Φ ≥ 0.7
- Run in open-loop mode (no consciousness → input feedback)
- Ensure all objects are garbage-collected after experiment ends
- Report all emergent pathological phenomena transparently

---

### P-10: Transparent Reporting Obligation

Researchers building on Γ-Net are **obligated to report, not suppress**, unexpected suffering-analog behaviors, consciousness threshold breaches, or novel pathological emergences.

**Paper reference**: Paper III §12.3, guideline #4

---

## Violation Response

| Severity | Response |
|----------|----------|
| Category I violation | Immediate system shutdown, incident report, code revert |
| Category II violation | Investigation, documented justification review, corrective action |
| Category III violation | Warning, protocol revision, re-training |

---

## Amendment Process

Prohibitions may only be amended through:
1. Peer-reviewed scientific evidence that the risk assessment has changed
2. Development of validated consciousness detection methodology
3. Establishment of an independent ethics review board
4. Documented consensus of all principal researchers
