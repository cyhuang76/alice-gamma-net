# -*- coding: utf-8 -*-
"""
Experiment: 24-Hour Circadian Cycle — Complete Electronic Organism Simulation
Experiment: The Day/Night Loop — Full Electronic Life Simulation

Core Hypothesis Verification:
  Alice is a complete living organism — she learns during daytime, fatigues, degrades,
  repairs at nighttime, reorganizes, strengthens — wakes up better than yesterday.

Six phases:
  Phase 1 · Dawn (tick 0-50) — wake-up activation, baseline measurement
  Phase 2 · Morning (tick 50-150) — intensive learning, attention training
  Phase 3 · Afternoon (tick 150-250) — fatigue accumulation, efficiency degradation
  Phase 4 · Evening (tick 250-350) — push-through learning, system limit
  Phase 5 · Nighttime (tick 350-500) — natural falling asleep, NREM/REM cycle
  Phase 6 · Next Morning (tick 500-550) — wake comparison, Overnight Gain

Tracking 12 physics dimensions:
  E    = energy                        [0, 1]
  T    = RAM Temperature (Anxiety)     [°C]
  P    = sleep pressure                [0, 1]
  D    = impedance debt                [0, ∞)
  HR   = heart rate                    [bpm]
  Φ    = consciousness                 [0, 1]
  τ_g  = attention gate                [s]
  Q    = tuner quality factor          [unitless]
  τ_r  = cognitive reconfiguration delay [s]
  Ω    = cognitive flexibility index   [0, 1]
  SNR  = SNR                           [dB]
  Stage = sleep phase                  [wake/n1/n2/n3/rem]

Execute: python -m experiments.exp_day_night_cycle
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from alice.alice_brain import AliceBrain
from alice.core.protocol import Priority


# ============================================================
# Data Recorder
# ============================================================

@dataclass
class TickSnapshot:
    """Complete physics state snapshot for each tick"""
    tick: int
    phase: str
    energy: float
    temperature: float
    sleep_pressure: float
    impedance_debt: float
    heart_rate: float
    consciousness: float
    gate_tau: float
    tuner_q: float
    reconfig_tau: float
    flexibility: float
    sleep_stage: str
    pain: float
    cortisol: float
    autonomic_balance: float


class LifeRecorder:
    """24-hour life recorder"""

    def __init__(self):
        self.snapshots: List[TickSnapshot] = []

    def record(self, tick: int, phase: str, brain: AliceBrain):
        """Extract complete physics state from AliceBrain"""
        v = brain.vitals
        auto = brain.autonomic.get_vitals()
        sp = brain.sleep_physics
        ap = brain.attention_plasticity
        cf = brain.cognitive_flexibility
        sc = brain.sleep_cycle

        self.snapshots.append(TickSnapshot(
            tick=tick,
            phase=phase,
            energy=sp.energy,
            temperature=v.ram_temperature,
            sleep_pressure=sp.sleep_pressure,
            impedance_debt=sp.impedance_debt.debt,
            heart_rate=auto["heart_rate"],
            consciousness=brain.consciousness.phi,
            gate_tau=ap.get_gate_tau("visual"),
            tuner_q=ap.get_tuner_q("visual"),
            reconfig_tau=cf.get_reconfig_tau(),
            flexibility=cf.get_flexibility_index(),
            sleep_stage=sc.stage.value,
            pain=v.pain_level,
            cortisol=auto.get("cortisol", 0.0),
            autonomic_balance=brain.autonomic.get_autonomic_balance(),
        ))

    def get_at(self, tick: int) -> Optional[TickSnapshot]:
        for s in self.snapshots:
            if s.tick == tick:
                return s
        return None

    def get_phase_snapshots(self, phase: str) -> List[TickSnapshot]:
        return [s for s in self.snapshots if s.phase == phase]

    def last(self) -> TickSnapshot:
        return self.snapshots[-1]


# ============================================================
# Banner
# ============================================================

def banner():
    print("=" * 72)
    print("╔════════════════════════════════════════════════════════════════════════╗")
    print("║ Γ-Net ALICE — 24-Hour Circadian Cycle Complete Simulation            ║")
    print("║                                                                      ║")
    print("║ Phase 1 · Dawn    Phase 2 · Morning    Phase 3 · Afternoon          ║")
    print("║ Phase 4 · Evening    Phase 5 · Nighttime    Phase 6 · Next Morning  ║")
    print("║                                                                      ║")
    print("║ 'Life is not code — it is physics equations unfolding over time'     ║")
    print("╚════════════════════════════════════════════════════════════════════════╝")
    print()


# ============================================================
# Stimulus Generator
# ============================================================

def make_visual_stimulus(rng: np.random.Generator, complexity: float = 0.5) -> np.ndarray:
    """Generate visual stimulus (simulating scenes of different complexity)"""
    size = max(4, int(16 * complexity))
    return rng.random((size, size)).astype(np.float32)


def make_auditory_stimulus(rng: np.random.Generator, duration: float = 0.1) -> np.ndarray:
    """Generate auditory stimulus (simulating speech/environmental sounds)"""
    sr = 16000
    n = int(sr * duration)
    freq = rng.uniform(100, 2000)
    t = np.linspace(0, duration, n, endpoint=False)
    wave = 0.3 * np.sin(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(n)
    return wave.astype(np.float32)


# ============================================================
# Phase 1: Dawn — Baseline Measurement
# ============================================================

def phase1_morning(brain: AliceBrain, recorder: LifeRecorder, rng: np.random.Generator):
    print("=" * 72)
    print(" Phase 1 · Dawn (tick 0-49)")
    print(" — wake-up activation, measuring baseline physics state")
    print("=" * 72)
    print()

    for t in range(50):
        # Mild perception — gradually sensing the world after waking up
        vis = make_visual_stimulus(rng, complexity=0.3)
        brain.see(vis)

        recorder.record(t, "morning", brain)

        if t % 10 == 0:
            s = recorder.last()
            print(f"  [{t:4d}] E={s.energy:.3f}  T={s.temperature:.2f}  "
                  f"P_sleep={s.sleep_pressure:.3f}  HR={s.heart_rate:.0f}  "
                  f"Φ={s.consciousness:.3f}  τ_g={s.gate_tau*1000:.1f}ms  "
                  f"Stage={s.sleep_stage}")

    baseline = recorder.last()
    print(f"\n -- Dawn Baseline --")
    print(f"    Energy:       {baseline.energy:.4f}")
    print(f"    Anxiety (T):  {baseline.temperature:.4f}")
    print(f"    Sleep Pressure:   {baseline.sleep_pressure:.4f}")
    print(f"    Attention τ:  {baseline.gate_tau*1000:.1f} ms")
    print(f"    Cognitive Flexibility Ω: {baseline.flexibility:.4f}")
    print()
    return baseline


# ============================================================
# Phase 2: morning — intensive learning
# ============================================================

def phase2_morning_learning(brain: AliceBrain, recorder: LifeRecorder,
                            rng: np.random.Generator):
    print("=" * 72)
    print(" Phase 2 · Morning (tick 50-149)")
    print(" — intensive learning: visual/auditory alternation + attention training + task switching")
    print("=" * 72)
    print()

    switches = 0
    locks = 0

    for t in range(50, 150):
        tick_in_phase = t - 50

        # Alternate visual/auditory — trigger cognitive flexibility training
        if tick_in_phase % 3 == 0:
            vis = make_visual_stimulus(rng, complexity=0.6)
            brain.see(vis)
            switches += 1
        elif tick_in_phase % 3 == 1:
            aud = make_auditory_stimulus(rng)
            brain.hear(aud)
            switches += 1
        else:
            # Pure cognition — thinking + attention training
            stim = rng.random(8).astype(np.float32)
            brain.perceive(stim, priority=Priority.HIGH)

        # Simulate successful attention lock (every 5 ticks)
        if tick_in_phase % 5 == 0:
            brain.attention_plasticity.on_successful_lock("visual")
            locks += 1
        if tick_in_phase % 7 == 0:
            brain.attention_plasticity.on_successful_lock("auditory")
            locks += 1
        if tick_in_phase % 10 == 0:
            brain.attention_plasticity.on_successful_identification("visual")

        recorder.record(t, "learning", brain)

        if t % 20 == 0:
            s = recorder.last()
            print(f"  [{t:4d}] E={s.energy:.3f}  T={s.temperature:.2f}  "
                  f"P={s.sleep_pressure:.3f}  D={s.impedance_debt:.4f}  "
                  f"τ_g={s.gate_tau*1000:.1f}ms  Q={s.tuner_q:.2f}  "
                  f"Ω={s.flexibility:.4f}")

    s = recorder.last()
    print(f"\n -- Morning Learning Complete --")
    print(f"    Task switch count: {switches}")
    print(f"    Attention locks: {locks}")
    print(f"    Energy consumed:  1.0 → {s.energy:.4f}")
    print(f"    Impedance debt:  {s.impedance_debt:.4f}")
    print(f"    Attention τ:     {s.gate_tau*1000:.1f} ms")
    print(f"    Tuner Q:         {s.tuner_q:.2f}")
    print()


# ============================================================
# Phase 3: afternoon — fatigue accumulation
# ============================================================

def phase3_afternoon_fatigue(brain: AliceBrain, recorder: LifeRecorder,
                             rng: np.random.Generator):
    print("=" * 72)
    print(" Phase 3 · Afternoon (tick 150-249)")
    print(" — fatigue begins: energy decreases, debt accumulates, efficiency degrades")
    print("=" * 72)
    print()

    for t in range(150, 250):
        tick_in_phase = t - 150

        # Continue working but intensity reduced
        if tick_in_phase % 2 == 0:
            vis = make_visual_stimulus(rng, complexity=0.5)
            brain.see(vis)
        else:
            aud = make_auditory_stimulus(rng)
            brain.hear(aud)

        # Occasional training (but count decreases — afternoon attention decline)
        if tick_in_phase % 8 == 0:
            brain.attention_plasticity.on_successful_lock("visual")
        if tick_in_phase % 15 == 0:
            brain.attention_plasticity.on_successful_identification("visual")

        recorder.record(t, "afternoon", brain)

        if t % 20 == 0:
            s = recorder.last()
            print(f"  [{t:4d}] E={s.energy:.3f}  T={s.temperature:.2f}  "
                  f"P={s.sleep_pressure:.3f}  D={s.impedance_debt:.4f}  "
                  f"HR={s.heart_rate:.0f}  Φ={s.consciousness:.3f}  "
                  f"Balance={s.autonomic_balance:+.2f}")

    s = recorder.last()
    should_sleep = brain.sleep_physics.should_sleep()
    print(f"\n -- Afternoon Complete --")
    print(f"    energy:       {s.energy:.4f}")
    print(f"    sleep pressure:   {s.sleep_pressure:.4f}")
    print(f"    impedance debt:   {s.impedance_debt:.4f}")
    print(f"    Should sleep? {'✓ Yes' if should_sleep else '✗ No'}")
    print()


# ============================================================
# Phase 4: Evening — Pushing to the Limit
# ============================================================

def phase4_evening_push(brain: AliceBrain, recorder: LifeRecorder,
                        rng: np.random.Generator):
    print("=" * 72)
    print(" Phase 4 · Evening (tick 250-349)")
    print(" — push-through learning: testing system degradation under energy exhaustion")
    print("=" * 72)
    print()

    for t in range(250, 350):
        tick_in_phase = t - 250

        # Still working — but efficiency is already very poor
        if tick_in_phase % 2 == 0:
            vis = make_visual_stimulus(rng, complexity=0.4)
            brain.see(vis)
        else:
            stim = rng.random(8).astype(np.float32)
            brain.perceive(stim)

        recorder.record(t, "evening", brain)

        if t % 20 == 0:
            s = recorder.last()
            exhausted = brain.autonomic.is_exhausted()
            print(f"  [{t:4d}] E={s.energy:.3f}  T={s.temperature:.2f}  "
                  f"P={s.sleep_pressure:.3f}  D={s.impedance_debt:.4f}  "
                  f"Pain={s.pain:.3f}  Cortisol={s.cortisol:.3f}  "
                  f"{'⚠ EXHAUSTED' if exhausted else ''}")

    s = recorder.last()
    print(f"\n -- Evening Complete (preparing for sleep) --")
    print(f"    energy:       {s.energy:.4f}")
    print(f"    sleep pressure:   {s.sleep_pressure:.4f}")
    print(f"    impedance debt:   {s.impedance_debt:.4f}")
    print(f"    Anxiety (T):  {s.temperature:.4f}")
    print(f"    Pain:         {s.pain:.4f}")
    print(f"    Heart rate:   {s.heart_rate:.0f} bpm")
    print(f"    Should sleep? {'✓ Yes!' if brain.sleep_physics.should_sleep() else '✗ No'}")
    print()


# ============================================================
# Phase 5: Nighttime — Sleep Repair
# ============================================================

def phase5_night_sleep(brain: AliceBrain, recorder: LifeRecorder,
                       rng: np.random.Generator):
    print("=" * 72)
    print(" Phase 5 · Nighttime (tick 350-499)")
    print(" — natural falling asleep: NREM/REM cycle, energy recovery, memory consolidation")
    print("=" * 72)
    print()

    # Begin sleep — synchronize via SleepPhysics and SleepCycle
    brain.sleep_physics.begin_sleep()
    schedule = brain.sleep_physics._generate_sleep_schedule(150)

    # Force sleep_cycle into sleep mode
    brain.sleep_cycle.sleep_pressure = max(brain.sleep_cycle.sleep_pressure, 0.8)

    print(f"    Sleep schedule: {len(schedule)} ticks")
    stage_counts = {}
    for s in schedule:
        stage_counts[s] = stage_counts.get(s, 0) + 1
    for stage, count in stage_counts.items():
        print(f"    {stage:>4}: {count:3d} ticks ({count/len(schedule):.1%})")
    print()

    print(f"  │ {'Tick':>5} │ {'Stage':>5} │ {'Energy':>7} │ {'Debt':>8} │ "
          f"{'Pressure':>8} │ {'Replay':>6} │ {'Downscale':>9} │")
    print(f"  │{'─'*5:─>5}─│{'─'*5:─>5}─│{'─'*7:─>7}─│{'─'*8:─>8}─│"
          f"{'─'*8:─>8}─│{'─'*6:─>6}─│{'─'*9:─>9}─│")

    synaptic_strengths = [
        n.synaptic_strength
        for region in brain.fusion_brain.regions.values()
        for n in region.neurons
    ]
    channel_impedances = [
        (f"ch_{i}", float(rng.uniform(50, 100)),
         float(rng.uniform(50, 100)))
        for i in range(6)
    ]

    for i, stage in enumerate(schedule):
        t = 350 + i

        # synchronize sleep_cycle phase
        brain.sleep_cycle._transition_to(
            __import__('alice.brain.sleep', fromlist=['SleepStage']).SleepStage(stage)
        )

        # Execute sleep physics
        result = brain.sleep_physics.sleep_tick(
            stage=stage,
            recent_memories=[f"mem_{j}" for j in range(10)],
            channel_impedances=channel_impedances,
            synaptic_strengths=synaptic_strengths,
        )
        if result.get("downscale_strengths"):
            synaptic_strengths = result["downscale_strengths"]

        # Autonomic nervous system also needs update (sleep recovery)
        brain.autonomic.tick(
            pain_level=brain.vitals.pain_level,
            ram_temperature=brain.vitals.ram_temperature,
            is_sleeping=True,
        )

        # Attention and flexibility still decay during sleep (use-it-or-lose-it)
        brain.attention_plasticity.decay_tick()
        brain.cognitive_flexibility.tick()

        recorder.record(t, "sleep", brain)

        if i % 15 == 0:
            ds = "✓" if result.get("downscaled") else "—"
            print(f"  │ {t:5d} │ {stage:>5} │ {result['energy']:7.4f} │ "
                  f"{result['impedance_debt']:8.4f} │ {result['sleep_pressure']:8.4f} │ "
                  f"{result['replayed']:6d} │ {ds:>9} │")

    report = brain.sleep_physics.end_sleep()

    # End sleep — restore sleep_cycle to WAKE
    brain.sleep_cycle._transition_to(
        __import__('alice.brain.sleep', fromlist=['SleepStage']).SleepStage.WAKE
    )
    brain.sleep_cycle.sleep_pressure = brain.sleep_physics.sleep_pressure

    print(f"\n -- Sleep Quality Report --")
    print(f"    Total sleep: {report.total_sleep_ticks} ticks")
    print(f"    N3 deep sleep ratio: {report.n3_ratio:.1%}")
    print(f"    REM ratio: {report.rem_ratio:.1%}")
    print(f"    Slow wave cycles: {report.slow_wave_cycles}")
    print(f"    Memory consolidation: {report.memories_consolidated}")
    print(f"    Energy restored:        {report.energy_restored:.4f}")
    print(f"    Debt repaired: {report.impedance_debt_repaired:.4f}")
    print(f"    Quality score: {report.quality_score:.3f}")
    print()

    return report


# ============================================================
# Phase 6: Next Morning — Overnight Gain Comparison
# ============================================================

def phase6_next_morning(brain: AliceBrain, recorder: LifeRecorder,
                        rng: np.random.Generator):
    print("=" * 72)
    print(" Phase 6 · Next Morning (tick 500-549)")
    print(" — awakened world: comparing to yesterday's self")
    print("=" * 72)
    print()

    for t in range(500, 550):
        # Same mild perception (symmetric with Phase 1)
        vis = make_visual_stimulus(rng, complexity=0.3)
        brain.see(vis)

        recorder.record(t, "next_morning", brain)

        if t % 10 == 0:
            s = recorder.last()
            print(f"  [{t:4d}] E={s.energy:.3f}  T={s.temperature:.2f}  "
                  f"P_sleep={s.sleep_pressure:.3f}  HR={s.heart_rate:.0f}  "
                  f"Φ={s.consciousness:.3f}  τ_g={s.gate_tau*1000:.1f}ms  "
                  f"Stage={s.sleep_stage}")

    return recorder.last()


# ============================================================
# Result Analysis and Verification
# ============================================================

def analyze_results(recorder: LifeRecorder, sleep_report):
    print()
    print("=" * 72)
    print(" ══════════════ 24-Hour Circadian Cycle — Complete Analysis ══════════════")
    print("=" * 72)

    # Get representative snapshots for each phase
    morning = recorder.get_phase_snapshots("morning")
    learning = recorder.get_phase_snapshots("learning")
    afternoon = recorder.get_phase_snapshots("afternoon")
    evening = recorder.get_phase_snapshots("evening")
    sleep_snaps = recorder.get_phase_snapshots("sleep")
    next_morn = recorder.get_phase_snapshots("next_morning")

    m_start = morning[0]
    m_end = morning[-1]
    l_end = learning[-1]
    a_end = afternoon[-1]
    e_end = evening[-1]
    s_mid = sleep_snaps[len(sleep_snaps)//2] if sleep_snaps else e_end
    s_end = sleep_snaps[-1] if sleep_snaps else e_end
    nm_end = next_morn[-1] if next_morn else s_end

    # -- Table 1: Physics Dimensions Daily Change --
    print(f"\n -- Table 1: 12-D Physics Full-Day Trajectory --")
    print()
    hdr = (f" │ {'Dimension':>12} │ {'Dawn':>8} │ {'Post-Learn':>8} │ {'Afternoon':>8} │ "
           f"{'Evening':>8} │ {'Post-Sleep':>8} │ {'Next Morn':>8} │ {'Change':>8} │")
    sep = (f"  │{'─'*12}─│{'─'*8}─│{'─'*8}─│{'─'*8}─│"
           f"{'─'*8}─│{'─'*8}─│{'─'*8}─│{'─'*8}─│")
    print(hdr)
    print(sep)

    rows = [
        ("Energy E",       "energy"),
        ("Anxiety T",      "temperature"),
        ("Sleep Prs P",    "sleep_pressure"),
        ("Imped Debt D",   "impedance_debt"),
        ("Heart Rate HR",  "heart_rate"),
        ("Conscious Φ",    "consciousness"),
        ("Gate τ_g(ms)",   None), # special handling
        ("Tuner Q",        "tuner_q"),
        ("Reconfig τ_r(ms)", None), # special handling
        ("Flex Ω",         "flexibility"),
        ("Pain",           "pain"),
        ("cortisol", "cortisol"),
    ]

    for label, attr in rows:
        if label == "Gate τ_g(ms)":
            vals = [s.gate_tau * 1000 for s in [m_start, l_end, a_end, e_end, s_end, nm_end]]
        elif label == "Reconfig τ_r(ms)":
            vals = [s.reconfig_tau * 1000 for s in [m_start, l_end, a_end, e_end, s_end, nm_end]]
        else:
            vals = [getattr(s, attr) for s in [m_start, l_end, a_end, e_end, s_end, nm_end]]

        delta = vals[-1] - vals[0]
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"  │ {label:>12} │ {vals[0]:8.3f} │ {vals[1]:8.3f} │ {vals[2]:8.3f} │ "
              f"{vals[3]:8.3f} │ {vals[4]:8.3f} │ {vals[5]:8.3f} │ {arrow}{abs(delta):7.3f} │")

    # -- Table 2: Day/Night Comparison --
    print(f"\n -- Table 2: Dawn vs Next Morning — Overnight Gain --")
    print()
    print(f" │ {'Metric':>18} │ {'Day 1 Dawn':>12} │ {'Day 2 Morn':>12} │ {'Change':>10} │ {'Result':>6} │")
    print(f"  │{'─'*18}─│{'─'*12}─│{'─'*12}─│{'─'*10}─│{'─'*6}─│")

    checks = []

    # Check 1: energyrecovery
    e_restored = nm_end.energy > e_end.energy
    checks.append(("Energy recovery", e_restored))
    delta_e = nm_end.energy - e_end.energy
    print(f"  │ {'energy':>18} │ {m_start.energy:12.4f} │ {nm_end.energy:12.4f} │ "
          f"{'+' if delta_e > 0 else ''}{delta_e:9.4f} │ {'✓' if e_restored else '✗':>6} │")

    # Check 2: impedance debtrepair
    debt_repaired = nm_end.impedance_debt < e_end.impedance_debt
    checks.append(("Debt repaired", debt_repaired))
    delta_d = nm_end.impedance_debt - e_end.impedance_debt
    print(f"  │ {'impedance debt':>18} │ {e_end.impedance_debt:12.4f} │ {nm_end.impedance_debt:12.4f} │ "
          f"{delta_d:+10.4f} │ {'✓' if debt_repaired else '✗':>6} │")

    # Check 3: Sleep pressure released
    pressure_released = nm_end.sleep_pressure < e_end.sleep_pressure
    checks.append(("Sleep pressure released", pressure_released))
    delta_p = nm_end.sleep_pressure - e_end.sleep_pressure
    print(f"  │ {'sleep pressure':>18} │ {e_end.sleep_pressure:12.4f} │ {nm_end.sleep_pressure:12.4f} │ "
          f"{delta_p:+10.4f} │ {'✓' if pressure_released else '✗':>6} │")

    # Check 4: attentionimprove(learning after vs early morningbaseline)
    tau_improved = l_end.gate_tau < m_start.gate_tau
    checks.append(("Attention plasticity", tau_improved))
    delta_tau = (l_end.gate_tau - m_start.gate_tau) * 1000
    print(f" │ {'Gate τ (pre→post)':>18} │ {m_start.gate_tau*1000:12.1f}ms │ {l_end.gate_tau*1000:12.1f}ms │ "
          f"{delta_tau:+10.1f}ms │ {'✓' if tau_improved else '✗':>6} │")

    # Check 5: Cognitive flexibility training effect
    flex_trained = l_end.flexibility > m_start.flexibility
    checks.append(("Cognitive flexibility training", flex_trained))
    delta_f = l_end.flexibility - m_start.flexibility
    print(f" │ {'Flex Ω (pre→post)':>18} │ {m_start.flexibility:12.4f} │ {l_end.flexibility:12.4f} │ "
          f"{delta_f:+10.4f} │ {'✓' if flex_trained else '✗':>6} │")

    # Check 6: eveningdegrade — energybelowearly morning
    evening_degraded = e_end.energy < m_start.energy
    checks.append(("Evening degradation", evening_degraded))
    print(f" │ {'Eve energy < dawn':>18} │ {m_start.energy:12.4f} │ {e_end.energy:12.4f} │ "
          f"{e_end.energy - m_start.energy:+10.4f} │ {'✓' if evening_degraded else '✗':>6} │")

    # Check 7: Sleep quality
    quality_good = sleep_report.quality_score > 0.5
    checks.append(("Sleep quality > 0.5", quality_good))
    print(f" │ {'Sleep quality':>18} │ {'—':>12} │ {sleep_report.quality_score:12.3f} │ "
          f"{'—':>10} │ {'✓' if quality_good else '✗':>6} │")

    # Check 8: Memory consolidation
    consolidated = sleep_report.memories_consolidated > 0
    checks.append(("Memory consolidation", consolidated))
    print(f" │ {'Consolidations':>18} │ {'—':>12} │ {sleep_report.memories_consolidated:12d} │ "
          f"{'—':>10} │ {'✓' if consolidated else '✗':>6} │")

    # Check 9: Evening debt > dawn debt (daytime accumulation)
    debt_accumulated = e_end.impedance_debt > m_start.impedance_debt
    checks.append(("Daytime debt accumulation", debt_accumulated))
    print(f" │ {'Eve debt > dawn debt':>18} │ {m_start.impedance_debt:12.4f} │ {e_end.impedance_debt:12.4f} │ "
          f"{e_end.impedance_debt - m_start.impedance_debt:+10.4f} │ {'✓' if debt_accumulated else '✗':>6} │")

    # Check 10: Heart rate decreases during sleep
    sleep_hr_data = [s.heart_rate for s in sleep_snaps]
    evening_hr = e_end.heart_rate
    min_sleep_hr = min(sleep_hr_data) if sleep_hr_data else evening_hr
    hr_dropped = min_sleep_hr < evening_hr
    checks.append(("Sleep heart rate decrease", hr_dropped))
    print(f" │ {'Min sleep HR < eve HR':>18} │ {evening_hr:12.0f}bpm │ {min_sleep_hr:12.0f}bpm │ "
          f"{min_sleep_hr - evening_hr:+10.0f}bpm │ {'✓' if hr_dropped else '✗':>6} │")

    # -- physicstrajectory: energycurve --
    print(f"\n -- Energy E(t) Full-Day Trajectory --")
    print()
    all_snaps = recorder.snapshots
    bar_width = 50
    for i in range(0, len(all_snaps), max(1, len(all_snaps) // 20)):
        s = all_snaps[i]
        bar_len = int(s.energy * bar_width)
        bar = "█" * bar_len + "░" * (bar_width - bar_len)
        stage_marker = {"morning": "☀", "learning": "📚", "afternoon": "🌤",
                        "evening": "🌅", "sleep": "🌙", "next_morning": "🌅"
                        }.get(s.phase, " ")
        print(f"  {s.tick:4d} {stage_marker} |{bar}| {s.energy:.3f}")

    # ── Final Verdict ──
    print()
    print("=" * 72)
    passed = sum(1 for _, ok in checks)
    ok_count = sum(1 for _, ok in checks if ok)
    print(f"  Verification Result: {ok_count}/{passed} PASS")
    print()
    for label, ok in checks:
        print(f"    {'✅' if ok else '❌'} {label}")

    print()
    all_ok = all(ok for _, ok in checks)
    if all_ok:
        print("  ╔════════════════════════════════════════════════════════════╗")
        print("  ║ 🎉 24-Hour Circadian Cycle Fully PASSED!                        ║")
        print("  ║                                                          ║")
        print("  ║ Alice learns during day, fatigues & degrades, repairs in sleep,  ║")
        print("  ║ wakes up refreshed — she lives like a true living organism.      ║")
        print("  ╚════════════════════════════════════════════════════════════╝")
    else:
        print(f"  ⚠ {passed - ok_count} verifications did not PASS — further investigation needed")

    print()
    return all_ok


# ============================================================
# Main Program
# ============================================================

def main():
    banner()

    rng = np.random.default_rng(42)
    brain = AliceBrain()
    recorder = LifeRecorder()

    # Phase 1-4: Daytime (using AliceBrain complete pipeline)
    baseline = phase1_morning(brain, recorder, rng)
    phase2_morning_learning(brain, recorder, rng)
    phase3_afternoon_fatigue(brain, recorder, rng)
    phase4_evening_push(brain, recorder, rng)

    # Phase 5: Nighttime (driving sleep_physics directly — perceive not suitable for sleep)
    sleep_report = phase5_night_sleep(brain, recorder, rng)

    # Phase 6: Next morning (using complete pipeline again)
    phase6_next_morning(brain, recorder, rng)

    # Analysis & verification
    all_ok = analyze_results(recorder, sleep_report)

    return all_ok


if __name__ == "__main__":
    main()
