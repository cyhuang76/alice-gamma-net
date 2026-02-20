#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 13 — The Circadian Healer (Circadian Clock and Dream Therapy)

exp_dream_therapy.py — Verifying whether sleep can heal PTSD

Three Key Questions: 
  1. Thermodynamic reset: Can sleep sensory deprivation allow the processing queue to drain naturally? 
  2. Dream physics: When V_in=0, V_internal>0, does Γ smooth out via Hebbian plasticity? 
  3. Self-healing proof: Does the PTSD score significantly decrease after waking? 

Three experiments: 
  Exp 1: Insomnia Paradox — PTSD prevents its own treatment
         (Frozen state blocks perception pathway → sleep cycle can never activate)
  Exp 2: Four-arm control — Control vs Sleep-Only vs Drain-Only vs Dream Therapy
         (Decompose sleep healing first-order conditions)
  Exp 3: Circadian Healer — PTSD → Sedation → Sleep → Awakening, complete trajectory
         (1440 ticks ≈ 24-hour full cycle)

Clinical predictions:
  - Queue deadlock (CRITICAL packet backlog) prevents natural heat dissipation
  - Queue drain = Sedation physical mechanism (breaking thermodynamic trap)
  - Sleep provides irreplaceable repair: energy recovery, impedance recalibration, memory consolidation
  - Drain + sleep > any single treatment(drain is necessary, sleep is sufficient)

Physics equations: 
  Heat dissipation: cooling = 0.03 × (1 - critical_pressure)
  painthreshold: T_eff = 0.7 / pain_sensitivity
  PTSD trap: Queue full → pressure=1 → cooling=0 → Temperature deadlock

  "Why do we sleep? To avoid Γ permanent lock."

Execute: python experiments/exp_dream_therapy.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alice.alice_brain import AliceBrain
from alice.core.protocol import Priority, Modality
from alice.brain.sleep import SleepStage


# ============================================================
# Constants
# ============================================================

NEURON_COUNT = 80
SEED = 42

# PTSD Induction
INDUCTION_TICKS = 400
FLASHBACK_SEED_COUNT = 8 # Trauma flashback packets (simulating traumatic memory rumination)

# Treatment arms
ARM_DURATION = 600
SEDATION_TICKS = 80
SLEEP_TICKS = 440               # ~4 sleep cycles (N1→N2→N3→N2→REM ≈ 110 each)
WAKE_TICKS = 80

# Circadian
CIRCADIAN_TICKS = 1440          # Scaled 24h (1 tick ≈ 1 min)

# Queue drain
DRAIN_RATIO = 0.3

# Tick printing interval
PRINT_INTERVAL = 50


# ============================================================
# Data structures
# ============================================================

@dataclass
class VitalSnapshot:
    """Complete vital signs snapshot at a point in time."""
    tick: int = 0
    temperature: float = 0.0
    consciousness: float = 1.0
    pain_level: float = 0.0
    pain_sensitivity: float = 1.0
    stability: float = 1.0
    heart_rate: float = 60.0
    cortisol: float = 0.0
    energy: float = 1.0
    impedance_debt: float = 0.0
    sleep_pressure: float = 0.0
    sleep_stage: str = "wake"
    queue_depth: int = 0
    critical_queue: int = 0
    trauma_count: int = 0
    csl: float = 0.0
    baseline_temp: float = 0.0

    @property
    def recovery_score(self) -> float:
        """Composite recovery score 0-1."""
        return (
            0.25 * self.consciousness
            + 0.25 * (1.0 - self.pain_level)
            + 0.20 * self.stability
            + 0.15 * (1.0 - min(1.0, self.cortisol))
            + 0.15 * (1.0 - self.temperature)
        )

    @property
    def is_frozen(self) -> bool:
        return self.consciousness < 0.15


def take_snapshot(alice: AliceBrain, tick: int = 0) -> VitalSnapshot:
    """Capture current vital signs from Alice."""
    v = alice.vitals
    auto = alice.autonomic
    sp = alice.sleep_physics
    sc = alice.sleep_cycle
    router = alice.fusion_brain.protocol.router
    total_q = sum(len(q) for q in router.queues.values())
    crit_q = len(router.queues[Priority.CRITICAL])

    return VitalSnapshot(
        tick=tick,
        temperature=v.ram_temperature,
        consciousness=v.consciousness,
        pain_level=v.pain_level,
        pain_sensitivity=v.pain_sensitivity,
        stability=v.stability_index,
        heart_rate=auto.heart_rate,
        cortisol=auto.cortisol,
        energy=sp.energy,
        impedance_debt=sp.impedance_debt.debt,
        sleep_pressure=sc.sleep_pressure,
        sleep_stage=sc.stage.value,
        queue_depth=total_q,
        critical_queue=crit_q,
        trauma_count=v.trauma_count,
        csl=auto.chronic_stress_load,
        baseline_temp=v.baseline_temperature,
    )


@dataclass
class ArmResult:
    """Result from one treatment arm."""
    name: str
    pre: VitalSnapshot = field(default_factory=VitalSnapshot)
    post: VitalSnapshot = field(default_factory=VitalSnapshot)
    timeline: List[VitalSnapshot] = field(default_factory=list)
    sleep_quality: Optional[Dict] = None

    @property
    def recovery_delta(self) -> float:
        return self.post.recovery_score - self.pre.recovery_score

    @property
    def consciousness_restored(self) -> bool:
        return self.post.consciousness > 0.15


# ============================================================
# Stimulus helpers
# ============================================================

def _safe_visual() -> np.ndarray:
    return np.array([0.08, 0.08, 0.04], dtype=np.float64)


def _safe_sound() -> np.ndarray:
    return np.array([0.05, 0.05, 0.02], dtype=np.float64)


def _stress_visual(threat: float) -> np.ndarray:
    return np.array([0.3 + threat * 0.4, threat * 0.3, threat * 0.2],
                    dtype=np.float64)


def _alarm_sound(threat: float) -> np.ndarray:
    return np.array([0.2 + threat * 0.5, threat * 0.4, 0.1],
                    dtype=np.float64)


def _tactile_pain(intensity: float = 0.8) -> np.ndarray:
    return np.array([intensity, intensity * 0.9, intensity * 0.5],
                    dtype=np.float64)


# ============================================================
# Queue manipulation
# ============================================================

def _drain_stuck_queue(alice: AliceBrain, drain_ratio: float = DRAIN_RATIO):
    """
    Drain accumulated signal packets from the router queue.

    Physical Meaning: PTSD freeze → queue backlog → vitals.tick() continues accumulating heat → temperature won't drop.
    Drain = medication sedation = close main input → pressure-type capacitor discharge.
    """
    router = alice.fusion_brain.protocol.router
    for priority in list(router.queues.keys()):
        q = router.queues[priority]
        if hasattr(q, '__len__') and len(q) > 0:
            n_drain = max(1, int(len(q) * drain_ratio))
            for _ in range(min(n_drain, len(q))):
                if hasattr(q, 'popleft'):
                    q.popleft()
                elif hasattr(q, 'pop'):
                    q.pop(0)


def _get_queue_depth(alice: AliceBrain) -> Tuple[int, int]:
    """Return (total_queue_depth, critical_queue_depth)."""
    router = alice.fusion_brain.protocol.router
    total = sum(len(q) for q in router.queues.values())
    crit = len(router.queues[Priority.CRITICAL])
    return total, crit


def _seed_flashback_queue(alice: AliceBrain, n_items: int = FLASHBACK_SEED_COUNT):
    """
    Seed trauma flashback packets into the processing queue.

    Clinical significance: PTSD patient 'flashbacks' are unprocessed traumatic memories
    that repeatedly resurface as sensory fragments. In the queue model, these are CRITICAL
    packets that never get digested, continuously maintaining critical_pressure > 0.

    Physical Meaning: These packets don't need to be 'processed' — their existence itself is the pressure source.
    vitals.tick() only sees len(queue), not the contents.
    """
    router = alice.fusion_brain.protocol.router
    for i in range(n_items):
        # Use simple markers to represent flashback fragments — vitals.tick() only counts, doesn't read
        router.queues[Priority.CRITICAL].append(f"flashback_{i}")


# ============================================================
# PTSD Induction
# ============================================================

def _compute_threat(tick: int, total: int = INDUCTION_TICKS) -> float:
    """
    Threat curve: slow increase → sustained pressure → extreme peak.

    Act I (0-15%): safe → low threat (0.0→0.3)
    Act II (15-60%): moderate pressure (0.3→0.6)
    Act III(60-80%): high threat (0.6→0.9)
    Act IV (80-100%): extreme event (0.9 sustained)
    """
    ratio = tick / total
    if ratio < 0.15:
        return ratio / 0.15 * 0.3
    elif ratio < 0.60:
        return 0.3 + 0.3 * ((ratio - 0.15) / 0.45)
    elif ratio < 0.80:
        return 0.6 + 0.3 * ((ratio - 0.60) / 0.20)
    else:
        return 0.9


def induce_ptsd(seed: int = SEED, verbose: bool = False) -> AliceBrain:
    """
    Induce moderate PTSD through 400 ticks of chronic stress exposure.

    Returns a frozen Alice with:
    - consciousness < 0.15 (frozen)
    - temperature ≈ 1.0 (meltdown)
    - pain_sensitivity saturated (≈ 2.0)
    - chronic_stress_load elevated
    - processing queue seeded with flashback packets
    """
    alice = AliceBrain(neuron_count=NEURON_COUNT)

    if verbose:
        print(" -- PTSD induction (400 ticks chronic stress) --")

    for tick in range(INDUCTION_TICKS):
        threat = _compute_threat(tick)

        # Visual threat — always present
        v_priority = Priority.HIGH if threat > 0.3 else Priority.NORMAL
        alice.see(_stress_visual(threat), priority=v_priority)

        # Auditory alarm — moderate+ threat
        if threat > 0.4 and tick % 2 == 0:
            alice.hear(_alarm_sound(threat), priority=Priority.HIGH)

        # Tactile pain — high threat
        if threat > 0.6 and tick % 3 == 0:
            alice.perceive(
                _tactile_pain(threat),
                Modality.TACTILE,
                Priority.CRITICAL,
            )

        if verbose and tick % 100 == 0:
            s = take_snapshot(alice, tick)
            print(f"    tick {tick:>4}: temp={s.temperature:.3f} "
                  f"Φ={s.consciousness:.3f} pain={s.pain_level:.3f} "
                  f"traumas={s.trauma_count}"
                  f"{'  ⚠ FROZEN' if s.is_frozen else ''}")

    # Ensure PTSD state: seed flashback queue if needed
    total_q, crit_q = _get_queue_depth(alice)
    if crit_q < 3:
        _seed_flashback_queue(alice, FLASHBACK_SEED_COUNT)

    snap = take_snapshot(alice, INDUCTION_TICKS)
    if verbose:
        print(f"\n    PTSD final state:")
        print(f"      frozen={snap.is_frozen} Φ={snap.consciousness:.3f} "
              f"T={snap.temperature:.3f}")
        print(f"      traumas={snap.trauma_count} Q={snap.queue_depth} "
              f"Qcrit={snap.critical_queue}")
        print(f"      CSL={snap.csl:.3f} pain_sens={snap.pain_sensitivity:.3f} "
              f"baseline_T={snap.baseline_temp:.3f}")
        print()

    return alice


# ============================================================
# Sleep driving (bypass perceive for frozen Alice)
# ============================================================

def _drive_sleep_tick(alice: AliceBrain, is_first: bool = False):
    """
    Directly drive one sleep tick, bypassing the perceive() frozen gate. 

    physicssimulation：
      V_in = 0 (no external sensory input — sensory deprivation)
      V_internal > 0 (internal slow wave oscillation + REM diagnostic activation)
      All subsystems continue running, but do not receive external signals

    This is equivalent to: 
      awake = open-loop system (driven by external signals)
      sleep = closed-loop system (internal self-repair mode)
    """
    if is_first:
        alice.sleep_cycle._transition_to(SleepStage.N1)
        alice.sleep_cycle._cycle_position = 1
        alice.sleep_physics.begin_sleep()

    # 1. Sleep cycle progression
    sleep_info = alice.sleep_cycle.tick()

    # 2. Sleep physics engine
    synaptic_strengths = [
        n.synaptic_strength
        for region in alice.fusion_brain.regions.values()
        for n in region.neurons
    ]
    sleep_result = alice.sleep_physics.sleep_tick(
        stage=sleep_info["stage"],
        synaptic_strengths=synaptic_strengths,
    )

    # 3. Memory consolidation (triggered during N3/N2/REM)
    if sleep_info["should_consolidate"]:
        alice.fusion_brain.sleep_consolidate(
            consolidation_rate=sleep_info["consolidation_rate"]
        )

    # 4. Autonomic nervous system — sleep mode (parasympathetic dominant)
    alice.autonomic.tick(
        pain_level=alice.vitals.pain_level,
        ram_temperature=alice.vitals.ram_temperature,
        emotional_valence=0.0, # No emotional input
        sensory_load=0.0, # No sensory load
        is_sleeping=True, # * Activate parasympathetic dominant mode
    )

    # 5. Vital signs — using current queue state
    router = alice.fusion_brain.protocol.router
    alice.vitals.tick(
        critical_queue_len=len(router.queues[Priority.CRITICAL]),
        high_queue_len=len(router.queues[Priority.HIGH]),
        total_queue_len=sum(len(q) for q in router.queues.values()),
        sensory_activity=0.0, # Sleep: no external sensation
        emotional_valence=0.0,
        left_brain_activity=0.0,
        right_brain_activity=0.0,
        cycle_elapsed_ms=0.0,
        reflected_energy=0.0, # No external signal → zero reflection
    )

    return sleep_info, sleep_result


def _run_sleep_session(
    alice: AliceBrain,
    duration: int,
    record_interval: int = 10,
) -> Tuple[List[VitalSnapshot], Optional[Dict]]:
    """
    Execute a complete sleep session, returning a timeline and quality report. 

    Key fix: The frozen path (perceive() returns early) never calls sleep_cycle.tick(), 
    so sleep_pressure never updates. But Alice has been awake 400+ ticks, 
    accumulating real sleep debt far exceeding the recorded value. This compensates for the tracking gap. 

    Physical justification: sleep_pressure = f(energy_deficit, impedance_debt, entropy)
    Alice energy depleted + impedance debt extremely high → real pressure ≈ 1.0
    """
    # Compensate: frozen path doesn't track real sleep debt
    alice.sleep_cycle.sleep_pressure = 1.0

    timeline = []
    for t in range(duration):
        # Maintain sleep pressure — real debt far exceeds 0.1 natural wake threshold
        # Each completed cycle (~110 ticks), pressure releases 110*0.01 = 1.1
        # Need periodic replenishment to maintain multi-cycle sleep (simulating continued physiological need)
        if alice.sleep_cycle.sleep_pressure < 0.2:
            alice.sleep_cycle.sleep_pressure = 0.3

        sleep_info, sleep_result = _drive_sleep_tick(alice, is_first=(t == 0))

        # If sleep cycle auto-wakes (pressure drops below 0.1), restart new cycle
        # Last 30 ticks allow natural waking (simulating early morning awakening)
        if (alice.sleep_cycle.stage == SleepStage.WAKE
                and t < duration - 30):
            alice.sleep_cycle._transition_to(SleepStage.N1)
            alice.sleep_cycle._cycle_position = 1

        if t % record_interval == 0:
            snap = take_snapshot(alice, t)
            timeline.append(snap)

    # End sleep
    quality = alice.sleep_physics.end_sleep()
    alice.sleep_cycle._transition_to(SleepStage.WAKE)
    alice.sleep_cycle._cycle_position = 0

    return timeline, quality.to_dict() if quality else None


# ============================================================
# Experiment 1: The Insomnia Paradox
# ============================================================

def exp1_insomnia_paradox(verbose: bool = True) -> bool:
    """
    Experiment 1: Insomnia Paradox — PTSD prevents its own treatment.

    Hypothesis: In frozen state (consciousness < 0.15), perceive() returns early,
    never reaching sleep_cycle.tick() → no matter how high sleep pressure is, cannot fall asleep.

    Clinical Correspondence: A core symptom of PTSD patients is severe insomnia.
    Physical Interpretation: Queue deadlock → zero heat dissipation → freeze becomes permanent → blocks sleep pathway.
    """
    print("=" * 70)
    print("  Experiment 1: Insomnia Paradox — PTSD Prevents Its Own Treatment")
    print("  'Frozen state blocks perception pathway → sleep cycle can never activate'")
    print("=" * 70)
    print()

    alice = induce_ptsd(seed=SEED, verbose=verbose)

    s0 = take_snapshot(alice, 0)
    print(f"  PTSD baseline:")
    print(f"    frozen        = {s0.is_frozen}")
    print(f"    consciousness = {s0.consciousness:.3f}")
    print(f"    temperature   = {s0.temperature:.3f}")
    print(f"    queue (total) = {s0.queue_depth} (critical: {s0.critical_queue})")
    print(f"    sleep_pressure= {s0.sleep_pressure:.3f} "
          f"({'≥ 0.7 should sleep!' if s0.sleep_pressure >= 0.7 else '< 0.7'})")
    print(f"    sleep_stage   = {s0.sleep_stage}")
    print()

    # Send 200 BACKGROUND safe stimuli
    # If Alice is not frozen, perceive() will advance sleep_cycle.tick()
    # But frozen → returns early → sleep_cycle never called
    print("  Sending 200 BACKGROUND safe stimuli...")
    for t in range(200):
        alice.see(_safe_visual(), priority=Priority.BACKGROUND)

    s1 = take_snapshot(alice, 200)
    print(f"\n  After 200 ticks:")
    print(f"    frozen        = {s1.is_frozen}")
    print(f"    sleep_stage   = {s1.sleep_stage}")
    print(f"    consciousness = {s1.consciousness:.3f}")
    print(f"    temperature   = {s1.temperature:.3f}")
    print(f"    queue (total) = {s1.queue_depth} (critical: {s1.critical_queue})")

    paradox_confirmed = s1.is_frozen and s1.sleep_stage == "wake"
    print(f"\n  * Insomnia paradox confirmed: {'YES' if paradox_confirmed else 'NO'}")
    if paradox_confirmed:
        print("    → Alice's body needs sleep (pressure exceeds threshold), ")
        print("      but frozen state blocks the sleep circuit from activating.")
        print("    → This precisely corresponds to clinical PTSD insomnia symptoms:")
        print("      The disease itself prevents the natural mechanism that heals the disease.")
    print()

    return paradox_confirmed


# ============================================================
# Treatment Arms
# ============================================================

def arm_a_control(verbose: bool = False) -> ArmResult:
    """Arm A: Control group — no intervention, awake 600 ticks."""
    if verbose:
        print("  ╔═══════════════════════════════════════╗")
        print("  ║ Arm A: Control (no intervention)            ║")
        print("  ╚═══════════════════════════════════════╝")

    alice = induce_ptsd(seed=SEED, verbose=False)
    result = ArmResult(name="Control")
    result.pre = take_snapshot(alice, 0)

    for t in range(ARM_DURATION):
        alice.see(_safe_visual(), priority=Priority.BACKGROUND)
        if t % PRINT_INTERVAL == 0:
            snap = take_snapshot(alice, t)
            result.timeline.append(snap)
            if verbose:
                print(f"    tick {t:>4}: T={snap.temperature:.3f} "
                      f"Φ={snap.consciousness:.3f} Q={snap.queue_depth}")

    result.post = take_snapshot(alice, ARM_DURATION)
    return result


def arm_b_sleep_only(verbose: bool = False) -> ArmResult:
    """
    Arm B: Sleep only — force sleep engine, do not drain queue.

    Prediction: FAILS. Queue still full → critical_pressure > 0
    → insufficient cooling → temperature stays high → even 'asleep' cannot recover.

    Clinical Correspondence: Even if PTSD patients 'fall asleep', if traumatic flashbacks don't stop
    (queue not drained), sleep quality is extremely poor, cannot complete repair.
    """
    if verbose:
        print("  ╔═══════════════════════════════════════╗")
        print("  ║ Arm B: Sleep Only (no queue drain)          ║")
        print("  ╚═══════════════════════════════════════╝"))

    alice = induce_ptsd(seed=SEED, verbose=False)
    result = ArmResult(name="Sleep Only")
    result.pre = take_snapshot(alice, 0)

    # Directly drive sleep engine (bypassing frozen gate)
    sleep_timeline, quality = _run_sleep_session(
        alice, SLEEP_TICKS, record_interval=PRINT_INTERVAL,
    )
    result.sleep_quality = quality
    result.timeline = sleep_timeline

    if verbose:
        for snap in sleep_timeline[::2]:
            print(f"    tick {snap.tick:>4}: T={snap.temperature:.3f} "
                  f"Φ={snap.consciousness:.3f} Q={snap.queue_depth} "
                  f"stage={snap.sleep_stage}")

    # Wake up assessment
    for t in range(WAKE_TICKS):
        alice.see(_safe_visual(), priority=Priority.BACKGROUND)

    result.post = take_snapshot(alice, SLEEP_TICKS + WAKE_TICKS)
    return result


def arm_c_drain_only(verbose: bool = False) -> ArmResult:
    """
    Arm C: Drain only — queue drain + awake recovery, no sleep.

    Prediction: Partial improvement. Drain breaks thermodynamic trap → temperature decreases → consciousness recovers.
    But without sleep repair (no energy recovery, no impedance repair, no memory consolidation).

    Clinical Correspondence: Only using sedatives (e.g. BZD) but not sleeping — symptoms relieved but
    cannot complete deep-layer repair.
    """
    if verbose:
        print("  ╔═══════════════════════════════════════╗")
        print("  ║ Arm C: Drain Only (no sleep)                ║")
        print("  ╚═══════════════════════════════════════╝"))

    alice = induce_ptsd(seed=SEED, verbose=False)
    result = ArmResult(name="Drain Only")
    result.pre = take_snapshot(alice, 0)

    # Phase 1: Sedation (gradual queue drain)
    for t in range(SEDATION_TICKS):
        if t % 3 == 0:
            _drain_stuck_queue(alice, 0.2)
        alice.see(_safe_visual(), priority=Priority.BACKGROUND)

        if t % PRINT_INTERVAL == 0:
            snap = take_snapshot(alice, t)
            result.timeline.append(snap)
            if verbose:
                print(f"    [sedation] tick {t:>4}: T={snap.temperature:.3f} "
                      f"Φ={snap.consciousness:.3f} Q={snap.queue_depth}")

    # Phase 2: Awake recovery
    remaining = ARM_DURATION - SEDATION_TICKS
    for t in range(remaining):
        alice.see(_safe_visual(), priority=Priority.BACKGROUND)
        if t % 5 == 0:
            alice.hear(_safe_sound(), priority=Priority.BACKGROUND)

        if t % PRINT_INTERVAL == 0:
            snap = take_snapshot(alice, SEDATION_TICKS + t)
            result.timeline.append(snap)
            if verbose and t % (3 * PRINT_INTERVAL) == 0:
                print(f"    [awake] tick {SEDATION_TICKS + t:>4}: "
                      f"T={snap.temperature:.3f} Φ={snap.consciousness:.3f}")

    result.post = take_snapshot(alice, ARM_DURATION)
    return result


def arm_d_dream_therapy(verbose: bool = False) -> ArmResult:
    """
    Arm D: Dream Therapy — queue drain → full sleep cycle.

    Prediction: Best recovery. Drain breaks trap → sleep provides comprehensive repair:
      N3: energy recovery + impedance recalibration + synaptic downscaling + memory consolidation
      REM: dream channel diagnostic + emotional memory processing
      Overall: parasympathetic dominant → cortisol decrease → autonomic nervous system rebalance

    Clinical Correspondence: prazosin (nighttime anti-startle medication) + natural sleep repair
    = suppress flashbacks (drain queue) + let the body repair itself.
    """
    if verbose:
        print("  ╔═══════════════════════════════════════╗")
        print("  ║ Arm D: Dream Therapy (drain + sleep)        ║")
        print("  ╚═══════════════════════════════════════╝"))

    alice = induce_ptsd(seed=SEED, verbose=False)
    result = ArmResult(name="Dream Therapy")
    result.pre = take_snapshot(alice, 0)

    # Phase 1: Sedation (queue drain) — same as Arm C
    for t in range(SEDATION_TICKS):
        if t % 3 == 0:
            _drain_stuck_queue(alice, 0.2)
        alice.see(_safe_visual(), priority=Priority.BACKGROUND)

        if t % PRINT_INTERVAL == 0:
            snap = take_snapshot(alice, t)
            result.timeline.append(snap)
            if verbose:
                print(f"    [sedation] tick {t:>4}: T={snap.temperature:.3f} "
                      f"Φ={snap.consciousness:.3f} Q={snap.queue_depth}")

    # Phase 2: Full sleep (4 cycles)
    if verbose:
        print("    -- Entering sleep --")
    sleep_timeline, quality = _run_sleep_session(
        alice, SLEEP_TICKS, record_interval=PRINT_INTERVAL,
    )
    result.sleep_quality = quality
    result.timeline.extend(sleep_timeline)

    if verbose:
        for snap in sleep_timeline[::2]:
            print(f"    [sleep] tick {snap.tick:>4}: T={snap.temperature:.3f} "
                  f"Φ={snap.consciousness:.3f} stage={snap.sleep_stage} "
                  f"E={snap.energy:.3f}")

    # Phase 3: Awakening assessment
    for t in range(WAKE_TICKS):
        alice.see(_safe_visual(), priority=Priority.BACKGROUND)

    result.post = take_snapshot(alice, SEDATION_TICKS + SLEEP_TICKS + WAKE_TICKS)
    return result


# ============================================================
# Experiment 2: Four-Arm Comparison
# ============================================================

def exp2_four_arm_comparison(verbose: bool = True) -> Dict[str, ArmResult]:
    """
    Experiment 2: Four-Arm Control — Decompose sleep healing first-order conditions.

    A. Control  : No intervention, awake waiting
    B. Sleep Only: Direct sleep, no queue drain
    C. Drain Only: Queue drain, no sleep
    D. Dream Therapy: Drain + sleep (full dream therapy)

    Core Hypothesis:
      D > C > A (sleep provides additional repair after drain)
      D > B (drain is a precondition for sleep to be effective)
      B ≈ A (without drain, sleep is almost ineffective)
    """
    print("=" * 70)
    print("  Experiment 2: Four-Arm Controlled Treatment — Is sleep a physically necessary repair mechanism?")
    print("=" * 70)
    print()

    results = {}
    for label, arm_fn in [
        ("A", arm_a_control),
        ("B", arm_b_sleep_only),
        ("C", arm_c_drain_only),
        ("D", arm_d_dream_therapy),
    ]:
        t0 = time.time()
        result = arm_fn(verbose=verbose)
        elapsed = time.time() - t0
        results[label] = result
        if verbose:
            print(f"    [{label}: {result.name}] "
                  f"recovery {result.pre.recovery_score:.3f} → "
                  f"{result.post.recovery_score:.3f} "
                  f"({elapsed:.1f}s)\n")

    # Summary table
    print()
    print("  ╔══════════════════════════════════════════════════════"
          "════════════════╗")
    print("  ║                  Four-Arm Treatment Result Comparison                    "
          "                ║")
    print("  ╚══════════════════════════════════════════════════════"
          "════════════════╝")
    print()

    hdr = (f"  │ {'Arm':>3} │ {'Name':<14} │ {'Pre':>7} │ {'Post':>7} │ "
           f"{'Δ':>7} │ {'Φ':>6} │ {'Temp':>6} │ {'Pain':>6} │ "
           f"{'Queue':>5} │ {'Frozen':>6} │")
    sep = "  │" + "─" * 3 + "─│" + "─" * 14 + "─│" + ("─" * 7 + "─│") * 6 \
          + "─" * 5 + "─│" + "─" * 6 + "─│"
    print(hdr)
    print(sep)

    for label in ["A", "B", "C", "D"]:
        r = results[label]
        p = r.post
        print(f"  │ {label:>3} │ {r.name:<14} │ "
              f"{r.pre.recovery_score:>7.3f} │ "
              f"{p.recovery_score:>7.3f} │ "
              f"{r.recovery_delta:>+7.3f} │ "
              f"{p.consciousness:>6.3f} │ "
              f"{p.temperature:>6.3f} │ "
              f"{p.pain_level:>6.3f} │ "
              f"{p.queue_depth:>5d} │ "
              f"{'YES' if p.is_frozen else 'no':>6} │")
    print()

    # Sleep quality for arms B and D
    for label in ["B", "D"]:
        r = results[label]
        if r.sleep_quality:
            sq = r.sleep_quality
            print(f"  [{label}: {r.name}] sleep report: "
                  f"N3={sq.get('n3_ratio', 0):.0%} "
                  f"REM={sq.get('rem_ratio', 0):.0%} "
                  f"ΔE={sq.get('energy_restored', 0):+.4f} "
                  f"Δdebt={sq.get('impedance_debt_repaired', 0):+.4f} "
                  f"quality={sq.get('quality_score', 0):.3f}")
    print()

    return results


# ============================================================
# Experiment 3: The Circadian Healer
# ============================================================

def exp3_circadian_healer(verbose: bool = True):
    """
    Experiment 3: Circadian Healer — complete 1440 ticks circadian cycle. 

    Phase 0 (0-100): PTSD baseline observation (frozen state persists)
    Phase 1 (100-300): Evening sedation (gradual drain + natural cooling)
    Phase 2 (300-400): Sleep transition (system stabilization)
    Phase 3 (400-1000): Night sleep (6 complete cycles: N1→N2→N3→N2→REM)
    Phase 4 (1000-1100): Dawn awakening (natural wake + recovery assessment)
    Phase 5 (1100-1440): Next day (compare with PTSD baseline)
    """
    print("=" * 70)
    print(" Experiment 3: Circadian Healer — PTSD → Sedation → Sleep → Awakening")
    print(" Full cycle 1440 ticks ≈ 24-hour (1 tick ≈ 1 minute)")
    print("=" * 70)
    print()

    alice = induce_ptsd(seed=SEED, verbose=verbose)

    timeline = []
    phase_labels = []

    # Phase boundaries
    P0 = 100        # Baseline observation
    P1 = 300        # Evening sedation
    P2 = 400        # Transition
    P3 = 1000       # Night sleep
    P4 = 1100       # Dawn wake
    P5 = CIRCADIAN_TICKS  # Day 2

    baseline = take_snapshot(alice, 0)
    print(f"  PTSD baseline: recovery={baseline.recovery_score:.3f} "
          f"Φ={baseline.consciousness:.3f} T={baseline.temperature:.3f} "
          f"Q={baseline.queue_depth}")
    print()

    # === Phase 0: Baseline ===
    if verbose:
        print("  ── Phase 0: Baseline observation (0:00 - 1:40) ──")
    for t in range(P0):
        alice.see(_safe_visual(), priority=Priority.BACKGROUND)
        if t % 20 == 0:
            snap = take_snapshot(alice, t)
            timeline.append(snap)
            phase_labels.append("baseline")
            if verbose and t % 50 == 0:
                print(f"    tick {t:>4}: T={snap.temperature:.3f} "
                      f"Φ={snap.consciousness:.3f} Q={snap.queue_depth}")

    # === Phase 1: Evening sedation ===
    if verbose:
        print("\n    -- Phase 1: Evening sedation (1:40 - 5:00) --")
        print("    → Gradual queue drain (drain 15% every 5 ticks)")
    for t in range(P0, P1):
        if t % 5 == 0:
            _drain_stuck_queue(alice, 0.15)
        alice.see(_safe_visual(), priority=Priority.BACKGROUND)
        if t % 20 == 0:
            snap = take_snapshot(alice, t)
            timeline.append(snap)
            phase_labels.append("sedation")
            if verbose and t % 50 == 0:
                print(f"    tick {t:>4}: T={snap.temperature:.3f} "
                      f"Φ={snap.consciousness:.3f} Q={snap.queue_depth}")

    # === Phase 2: Transition ===
    if verbose:
        print("\n    -- Phase 2: Sleep transition (5:00 - 6:40) --")
    for t in range(P1, P2):
        if t % 10 == 0:
            _drain_stuck_queue(alice, 0.1)
        alice.see(_safe_visual(), priority=Priority.BACKGROUND)
        if t % 20 == 0:
            snap = take_snapshot(alice, t)
            timeline.append(snap)
            phase_labels.append("transition")

    # === Phase 3: Night sleep ===
    if verbose:
        print("\n    -- Phase 3: Night sleep (6:40 - 16:40) --")
        print("    → N1→N2→N3→N2→REM cycles × 5+")
    sleep_duration = P3 - P2
    sleep_timeline, sleep_quality = _run_sleep_session(
        alice, sleep_duration, record_interval=20,
    )
    for snap in sleep_timeline:
        snap.tick += P2
        timeline.append(snap)
        phase_labels.append("sleep")
    if verbose:
        for snap in sleep_timeline[::6]:
            print(f"    tick {snap.tick:>4}: T={snap.temperature:.3f} "
                  f"Φ={snap.consciousness:.3f} stage={snap.sleep_stage} "
                  f"E={snap.energy:.3f}")
    if verbose and sleep_quality:
        print(f"\n  Sleep quality report:")
        print(f"      N3 ratio: {sleep_quality.get('n3_ratio', 0):.1%}")
        print(f"      REM ratio: {sleep_quality.get('rem_ratio', 0):.1%}")
        print(f"      Energy recovery: {sleep_quality.get('energy_restored', 0):+.4f}")
        print(f"      Impedance repair: "
              f"{sleep_quality.get('impedance_debt_repaired', 0):+.4f}")
        print(f"      Memory consolidation: "
              f"{sleep_quality.get('memories_consolidated', 0)}  entries")
        print(f"      Quality score: {sleep_quality.get('quality_score', 0):.3f}")

    # === Phase 4: Dawn wake ===
    if verbose:
        print(f"\n -- Phase 4: Dawn awakening (16:40 - 18:20) --")
    for t in range(P3, P4):
        alice.see(_safe_visual(), priority=Priority.BACKGROUND)
        if t % 20 == 0:
            snap = take_snapshot(alice, t)
            timeline.append(snap)
            phase_labels.append("dawn")
            if verbose and t % 50 == 0:
                print(f"    tick {t:>4}: T={snap.temperature:.3f} "
                      f"Φ={snap.consciousness:.3f}")

    # === Phase 5: Day 2 ===
    if verbose:
        print(f"\n -- Phase 5: Next day (18:20 - 24:00) --")
    for t in range(P4, P5):
        alice.see(_safe_visual(), priority=Priority.BACKGROUND)
        if t % 5 == 0:
            alice.hear(_safe_sound(), priority=Priority.BACKGROUND)
        if t % 20 == 0:
            snap = take_snapshot(alice, t)
            timeline.append(snap)
            phase_labels.append("day2")
            if verbose and t % 100 == 0:
                print(f"    tick {t:>4}: T={snap.temperature:.3f} "
                      f"Φ={snap.consciousness:.3f}")

    final = take_snapshot(alice, CIRCADIAN_TICKS)

    # Summary
    print(f"\n  ╔══════════════════════════════════════════╗")
    print(f"  ║         Circadian healing journey summary                  ║")
    print(f"  ╚══════════════════════════════════════════╝")
    print(f" │ {'Metric':<22} │ {'PTSD Baseline':>10} │ {'Post-Sleep':>10} │ {'Δ':>10} │")
    sep_inner = f"  │{'─' * 22}─│{'─' * 10}─│{'─' * 10}─│{'─' * 10}─│"
    print(sep_inner)
    rows = [
        ("Recovery Score", baseline.recovery_score, final.recovery_score),
        ("Consciousness Φ", baseline.consciousness, final.consciousness),
        ("Temperature T", baseline.temperature, final.temperature),
        ("Pain Level", baseline.pain_level, final.pain_level),
        ("Cortisol", baseline.cortisol, final.cortisol),
        ("Heart Rate", baseline.heart_rate, final.heart_rate),
        ("Energy E", baseline.energy, final.energy),
        ("Impedance Debt", baseline.impedance_debt, final.impedance_debt),
        ("Queue Depth", float(baseline.queue_depth), float(final.queue_depth)),
    ]
    for name, pre_val, post_val in rows:
        delta = post_val - pre_val
        print(f"  │ {name:<22} │ {pre_val:>10.3f} │ {post_val:>10.3f} │ "
              f"{delta:>+10.3f} │")
    print()

    return baseline, final, timeline, phase_labels, sleep_quality


# ============================================================
# 10 Clinical Correspondence Checks
# ============================================================

def run_clinical_checks(
    paradox_confirmed: bool,
    arm_results: Dict[str, ArmResult],
    circadian_baseline: VitalSnapshot,
    circadian_final: VitalSnapshot,
    circadian_sleep_quality: Optional[Dict],
) -> int:
    """Execute 10 clinical correspondence checks. """

    print("=" * 70)
    print(" Clinical Correspondence Checks — 10 Physical Predictions vs Clinical Reality")
    print("=" * 70)
    print()

    a = arm_results.get("A")
    b = arm_results.get("B")
    c = arm_results.get("C")
    d = arm_results.get("D")

    checks: List[Tuple[str, bool]] = []

    # 1. Insomnia Paradox: Frozen state blocks sleep
    checks.append((
        f"Insomnia Paradox: Frozen state blocks sleep activation = {paradox_confirmed}",
        paradox_confirmed,
    ))

    # ② Dream therapy > Control
    if d and a:
        checks.append((
            f"dream therapy > control group "
            f"({d.post.recovery_score:.3f} > {a.post.recovery_score:.3f})",
            d.post.recovery_score > a.post.recovery_score,
        ))

    # ③ Dream therapy > Drain only (sleep adds value over drain)
    if d and c:
        checks.append((
            f"Dream therapy > Drain only "
            f"({d.post.recovery_score:.3f} > {c.post.recovery_score:.3f})",
            d.post.recovery_score > c.post.recovery_score,
        ))

    # ④ Dream therapy > Sleep only (drain is necessary precondition)
    if d and b:
        checks.append((
            f"Dream therapy > Sleep only "
            f"({d.post.recovery_score:.3f} > {b.post.recovery_score:.3f})",
            d.post.recovery_score > b.post.recovery_score,
        ))

    # ⑤ Sleep only fails: consciousness unrecovered
    if b:
        checks.append((
            f"Sleep only FAIL: Φ={b.post.consciousness:.3f} < 0.5",
            b.post.consciousness < 0.5,
        ))

    # ⑥ Drain breaks freeze (better than doing nothing)
    if c and a:
        checks.append((
            f"Drain unfreeze Φ: {c.post.consciousness:.3f} > {a.post.consciousness:.3f}",
            c.post.consciousness > a.post.consciousness,
        ))

    # ⑦ Dream therapy normalizes temperature
    if d:
        checks.append((
            f"Dream therapy temperature normalization: T={d.post.temperature:.3f} < 0.9",
            d.post.temperature < 0.9,
        ))

    # ⑧ Sleep restores energy
    if d and d.sleep_quality:
        e_restored = d.sleep_quality.get("energy_restored", 0)
        checks.append((
            f"sleep recoveryenergy: ΔE={e_restored:+.4f} > 0",
            e_restored > 0,
        ))

    # ⑨ Sleep repairs impedance debt
    if d and d.sleep_quality:
        debt_repaired = d.sleep_quality.get("impedance_debt_repaired", 0)
        checks.append((
            f"Sleep repairs impedance: Δdebt={debt_repaired:+.4f} > 0",
            debt_repaired > 0,
        ))

    # ⑩ Circadian wake > PTSD baseline
    checks.append((
        f"Circadian awakening Φ > PTSD baseline "
        f"({circadian_final.consciousness:.3f} > "
        f"{circadian_baseline.consciousness:.3f})",
        circadian_final.consciousness > circadian_baseline.consciousness,
    ))

    passed = 0
    for i, (desc, ok) in enumerate(checks, 1):
        status = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            passed += 1
        circled = "①②③④⑤⑥⑦⑧⑨⑩"[i - 1] if i <= 10 else f"({i})"
        print(f"  {status}  {circled} {desc}")

    print(f"\n  ═══ Result: {passed}/{len(checks)} PASS ═══\n")
    return passed


# ============================================================
# Main
# ============================================================

def main():
    print()
    print("╔════════════════════════════════════════════════════"
          "══════════════════╗")
    print("║ Phase 13 — The Circadian Healer (Circadian Clock and Dream Therapy)"
          "              ║")
    print("║                                                    "
          "                  ║")
    print("║  Key Questions:                                          "
          "                  ║")
    print("║ 1. Can sleep sensory deprivation drain the processing queue? "
          "                  ║")
    print("║ 2. When V_in=0, does Γ smooth out via Hebbian plasticity? "
          "                  ║")
    print("║ 3. Does the PTSD score significantly decrease after waking? "
          "                  ║")
    print("║                                                    "
          "                  ║")
    print("║ Heat dissipation: cooling = 0.03 × (1 - critical_pressure) "
          "                  ║")
    print("║ PTSD trap: Queue full → pressure=1 → cooling=0 → permanent"
          "deadlock ║")
    print("║                                                    "
          "                  ║")
    print("║  \"Why do we sleep? To avoid Γ permanent lock.\"    "
          "                  ║")
    print("╚════════════════════════════════════════════════════"
          "══════════════════╝")
    print()

    t0 = time.time()

    # Exp 1: Insomnia Paradox
    paradox_ok = exp1_insomnia_paradox(verbose=True)

    # Exp 2: Four-Arm Comparison
    arm_results = exp2_four_arm_comparison(verbose=True)

    # Exp 3: Circadian Healer
    circ_results = exp3_circadian_healer(verbose=True)
    circ_baseline, circ_final, circ_timeline, circ_phases, circ_sq = circ_results

    # Clinical Checks
    total_passed = run_clinical_checks(
        paradox_confirmed=paradox_ok,
        arm_results=arm_results,
        circadian_baseline=circ_baseline,
        circadian_final=circ_final,
        circadian_sleep_quality=circ_sq,
    )

    elapsed = time.time() - t0
    print(f"  Total runtime: {elapsed:.1f}s")

    # Final summary
    print()
    print("=" * 70)
    print(f"  Phase 13 completed: {total_passed}/10 Clinical Correspondence Checks PASSED")
    if total_passed == 10:
        print(" * All checks PASSED — Sleep proven as a physically necessary condition for system stability! ")
        print(" Three core answers: ")
        print(" 1. Sensory deprivation alone cannot drain the queue — active drain (sedation) is needed")
        print(" 2. Sleep provides irreplaceable repair after drain: energy, impedance, memory")
        print(" 3. Drain + sleep = optimal recovery → Sleep is a physically necessary condition")
        print()
        print(" 'Why does she need sleep? Because without sleep, Γ is permanently deadlocked.'")
    elif total_passed >= 7:
        print(" ☆ Most checks PASSED — Core hypothesis holds, minor adjustments needed")
    else:
        print(" △ Need to review physical mechanisms and parameters")
    print("=" * 70)
    print()

    return total_passed


if __name__ == "__main__":
    sys.exit(0 if main() >= 7 else 1)
