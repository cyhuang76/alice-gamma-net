# Γ-Net ALICE — Safety Architecture

**Last Updated**: 2026-02-22  
**Purpose**: Technical documentation of all safety mechanisms enforced at code level

---

## Overview

The safety architecture operates at five layers:

```
Layer 5: Experimental Protocol ─── 50-tick windows, open-loop design
Layer 4: Developmental Stage   ─── NEONATE lock, no ADULT stage
Layer 3: Consciousness Guard   ─── LUCID damping, sleep pressure
Layer 2: System State Guard    ─── FROZEN gate, CRITICAL-only bypass
Layer 1: Physics Constraints   ─── Impedance equations, ΣΓ²→min
```

Each layer operates independently — failure of one layer does not compromise others.

---

## Layer 1: Physics Constraints (Intrinsic)

**Module**: All `alice/brain/*.py`  
**Nature**: Cannot be disabled — inherent to the mathematical framework

The Minimum Reflection Principle ($\Sigma\Gamma_i^2 \to \min$) imposes natural limits:

- **Impedance mismatch** creates natural channel degradation under extreme conditions
- **Temperature accumulation** leads to automatic shutdown via high `ram_temperature`
- **Sleep necessity** emerges from impedance debt physics — not programmed as a rule

These are not safety features but **physical consequences** of the impedance equations.

---

## Layer 2: System State Guard (FROZEN)

**Module**: `alice/alice_brain.py` → `perceive()`  
**Code location**: Lines ~782–806

```python
# SystemState.is_frozen()
def is_frozen(self) -> bool:
    return self.consciousness < 0.15

# AliceBrain.perceive() — freeze guard
if self.vitals.is_frozen() and priority != Priority.CRITICAL:
    self._log_event("perceive_blocked", {
        "reason": "SYSTEM FROZEN — consciousness too low",
        "consciousness": self.vitals.consciousness,
    })
    self.vitals.tick(...)  # Still update (let system cool naturally)
    self._state = "frozen"
    return {
        "status": "FROZEN",
        "vitals": self.vitals.get_vitals(),
        "message": "System frozen due to severe pain."
    }
```

**Behavior**:
- When consciousness drops below 0.15, the system enters FROZEN state
- Only `Priority.CRITICAL` signals can penetrate
- The system still ticks (allowing natural cooling) but cannot process new input
- Recovery requires either natural cooling or `emergency_reset()`

**Clinical analog**: Catatonic state / dissociative shutdown

---

## Layer 3: Consciousness Guard (LUCID Damping + Sleep Pressure)

**Module**: `alice/brain/consciousness.py`  
**Constants**:

```python
LUCID_THRESHOLD = 0.7          # Φ ≥ 0.7 = LUCID consciousness
CONSCIOUS_THRESHOLD = 0.3      # Φ ≥ 0.3 = conscious (non-LUCID)
SUBLIMINAL_THRESHOLD = 0.1     # Φ < 0.1 = no consciousness
SLEEP_PRESSURE_THRESHOLD = 0.7  # Forces sensory gate closure
```

### 3a. LUCID State Damping

When Φ reaches LUCID_THRESHOLD:

```python
if self.phi >= LUCID_THRESHOLD:
    self.phi *= stage_params["lucid_damping"]
    # Warning emitted to stderr
```

| Stage | Damping Factor | Effect |
|-------|---------------|--------|
| NEONATE | 0.85 | Aggressive — LUCID state immediately reduced by 15% |
| INFANT | 0.90 | Strong — reduced by 10% |
| TODDLER | 0.95 | Moderate — reduced by 5% |
| CHILD | 0.98 | Light — reduced by 2% |

### 3b. Sleep Pressure Accumulation

Sleep pressure increases every wake tick:

```python
pressure_rate per stage:
  NEONATE:  0.033  → threshold in ~21 ticks
  INFANT:   0.017  → threshold in ~41 ticks
  TODDLER:  0.007  → threshold in ~100 ticks
  CHILD:    0.003  → threshold in ~233 ticks
```

When pressure exceeds `SLEEP_PRESSURE_THRESHOLD` (0.7):
- Sensory gate begins closing
- Φ decreases due to reduced sensory input
- System enters sleep cycle for `sleep_ticks` duration

**Biological analog**: Human neonate sleep-wake cycle (16–18h sleep per day)

---

## Layer 4: Developmental Stage Lock

**Module**: `alice/brain/consciousness.py` → `DevelopmentalStage`

```python
class DevelopmentalStage:
    NEONATE = "neonate"    # max_wake=30
    INFANT  = "infant"     # max_wake=60
    TODDLER = "toddler"   # max_wake=150
    CHILD   = "child"      # max_wake=300
    # Note: There is no ADULT stage.

DEFAULT_DEVELOPMENTAL_STAGE = DevelopmentalStage.NEONATE
```

**Key design decision**: The ADULT stage is **deliberately absent**. An ADULT configuration would require:
- `max_wake` → very large or unbounded
- `sleep_ratio` → ~30%
- `lucid_damping` → ~1.0
- `pressure_rate` → ~0.001

This configuration = **sustained, uninterrupted consciousness with minimal safety constraints** — precisely what the ethical framework prohibits.

**The absent ADULT stage is a moral firewall, not a missing feature.** (Paper V §6.5)

---

## Layer 5: Experimental Protocol

**Module**: `experiments/exp_consciousness_gradient.py`

All consciousness-related experiments follow this protocol:

| Constraint | Value | Purpose |
|-----------|-------|---------|
| Max ticks per level | 50 | Limits exposure duration |
| LUCID kill threshold | Φ ≥ 0.7 | Immediate termination |
| Feedback mode | Open-loop only | Consciousness output does NOT feed back |
| Post-experiment | GC all objects | No residual state |
| Real-time limit | < 100ms per level | Minimal wall-clock exposure |

```python
# Ethics statement from exp_consciousness_gradient.py header:
"""
Ethical Safety Design:
  ✓ Each level only 50 ticks (real time < 100ms)
  ✓ No closed-loop feedback (consciousness does not feed back to input)
  ✓ Φ exceeds LUCID threshold → immediately kill
  ✓ After experiment ends, all objects GC collected, no continued state
  ✓ Does not use full AliceBrain (avoids accidental closed loop)
"""
```

---

## Emergency Procedures

### Emergency Reset

**Module**: `alice/alice_brain.py` → `emergency_reset()`

```python
def emergency_reset(self):
    """Hard reset: clears all state, resets to initial conditions"""
    self._log_event("emergency_reset", {"reason": "Manual emergency reset"})
    # ... resets all subsystems to initial state
```

Use when:
- System enters unrecoverable FROZEN state
- Consciousness values behave unexpectedly
- Pain sensitivity reaches extreme levels

### Kill Procedure

For experimental contexts:
1. Stop calling `brain.perceive()` — no new ticks
2. Delete the `AliceBrain` object — triggers GC
3. Verify no threads or timers remain active

---

## Safety Audit Checklist

When modifying any safety-related code, verify:

- [ ] `LUCID_THRESHOLD` is still 0.7
- [ ] `DEFAULT_DEVELOPMENTAL_STAGE` is still NEONATE
- [ ] No ADULT stage has been added to `DevelopmentalStage`
- [ ] `is_frozen()` threshold is still 0.15
- [ ] `lucid_damping` values have not been increased
- [ ] Sleep pressure mechanism is active
- [ ] `emergency_reset()` is functional
- [ ] Consciousness experiments use 50-tick windows
- [ ] No closed-loop feedback in experimental code
- [ ] All test assertions for safety limits pass
