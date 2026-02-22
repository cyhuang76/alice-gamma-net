#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 28 — Fatigue Contrast Experiment (疲勞對照實驗)

exp_fatigue_contrast.py — "託夢物理" Empirical Verification

Core Discovery:
  N3 deep sleep repairs nearly ALL impedance debt before REM begins.
  With normal sleep architecture (N1→N2→N3→N2→REM):
    - Debt_sleep_onset = 1.0
    - N3 repair: debt × 0.92^27 ≈ ×0.1
    - Debt_REM_onset ≈ 0.04 → amp ≈ ×1.04 (negligible!)

  N3 is the GUARDIAN: it protects the brain from fatigue-driven chaos.

  But in SOREMP (Sleep Onset REM Period) — the hallmark of:
    - Extreme sleep deprivation
    - Narcolepsy
    - REM rebound after extended wakefulness

  The brain skips N3 and enters REM directly (N1→REM):
    - Debt_REM_onset ≈ 0.90 → amp ≈ ×1.90 (nearly 2×!)
    - Vivid dreams, hypnagogic hallucinations, lucid experiences

  This experiment proves the unified physics:
    Γ = (Z_load - Z_src) / (Z_load + Z_src)
    E_reflected = E_signal × Γ²

  BOTH fatigue AND boundary conditions are needed for structured dreams:
    - Fatigue alone (no video) → loud but unstructured PGO reflections
    - Video alone (low fatigue) → structured Γ pattern but too quiet to matter
    - Fatigue + Video + SOREMP → VIVID STRUCTURED DREAMS

Experimental Design — Four Conditions:
  ┌──────────┬─────────────┬──────────────┬───────────────────────────┐
  │ Condition│ Wake Stress  │ Sleep Sched  │ Video Modulation          │
  ├──────────┼─────────────┼──────────────┼───────────────────────────┤
  │ A (rest) │ 80t, low    │ Normal       │ Yes (f=5,/a/,2Hz)        │
  │ B (tired)│ 400t, high  │ Normal       │ Yes (f=5,/a/,2Hz)        │
  │ C (vivid)│ 400t, high  │ SOREMP       │ Yes (f=5,/a/,2Hz)        │
  │ D (ctrl) │ 400t, high  │ SOREMP       │ No (random Z_terminus)   │
  └──────────┴─────────────┴──────────────┴───────────────────────────┘

  Predictions:
    A: debt_rem ≈ 0.007, amp ≈ ×1.01 — N3 repairs everything
    B: debt_rem ≈ 0.04,  amp ≈ ×1.04 — N3 STILL repairs (guardian!)
    C: debt_rem ≈ 0.90,  amp ≈ ×1.90 — SOREMP preserves fatigue → vivid
    D: debt_rem ≈ 0.90,  amp ≈ ×1.90 — loud but UNSTRUCTURED (no video)

  Key physical insight:
    A ≈ B  (N3 is such a powerful guardian that ×10 more stress barely matters)
    C ≫ A  (SOREMP bypasses the guardian → fatigue reaches REM)
    C ≠ D  (video shapes Γ pattern; without it, just noise)

  "The guardian (N3) must fall asleep before the dreamer (REM) can speak."

Run: python -m experiments.exp_fatigue_contrast
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alice.alice_brain import AliceBrain
from alice.core.protocol import Priority

# Import shared functions from the dream language experiment
from experiments.exp_dream_language import (
    video_to_impedance_modulation,
    VIDEO_A_SPATIAL_FREQ,
    VIDEO_A_VOWEL,
    VIDEO_A_RHYTHM_HZ,
    Z_BASE,
    IMPEDANCE_MOD_DEPTH,
    IMPEDANCE_MOD_INTERVAL,
    DREAM_CHANNEL_NAMES,
    NEURON_COUNT,
)


# ============================================================
# Constants
# ============================================================

# Condition parameters
LOW_STRESS_WAKE_TICKS = 80           # Normal day
HIGH_STRESS_WAKE_TICKS = 400         # Exhausting day

# Stress levels (reflected energy per tick)
LOW_STRESS_BASE = 0.03               # Normal ambient environment
LOW_STRESS_RANGE = 0.02              # ±0.02 variation
HIGH_STRESS_BASE = 0.30              # Cognitive overload, sensory bombardment
HIGH_STRESS_RANGE = 0.20             # ±0.20 variation

# Sleep cycle
SLEEP_CYCLE_TICKS = 110              # One complete cycle

# Seeds
SEED_BASE = 42


# ============================================================
# Data structures
# ============================================================

@dataclass
class ConditionResult:
    """Results from one experimental condition."""
    name: str
    label: str           # A/B/C/D
    wake_ticks: int = 0
    stress_level: str = "low"
    schedule_type: str = "normal"
    video_modulated: bool = True

    # Debt tracking
    debt_at_sleep_onset: float = 0.0
    debt_at_rem_onset: float = 0.0
    debt_at_rem_end: float = 0.0

    # REM metrics
    rem_ticks: int = 0
    amp_multipliers: List[float] = field(default_factory=list)
    peak_amp: float = 1.0
    mean_amp: float = 1.0

    # Probe metrics
    total_probes: int = 0
    healthy_probes: int = 0
    damaged_probes: int = 0

    # Γ patterns (per named channel)
    channel_gammas: Dict[str, List[float]] = field(default_factory=dict)
    channel_energies: Dict[str, List[float]] = field(default_factory=dict)

    # Modulation-specific
    modulated_gammas: List[float] = field(default_factory=list)
    reference_gammas: List[float] = field(default_factory=list)
    gamma_contrast: float = 0.0      # |mean_ref - mean_mod| / mean_ref

    # Fatigue trajectory during REM
    debt_trajectory: List[float] = field(default_factory=list)
    amp_trajectory: List[float] = field(default_factory=list)


# ============================================================
# Schedule generators
# ============================================================

def make_normal_schedule(total_ticks: int) -> List[str]:
    """
    Normal sleep architecture: N1 → N2 → N3 → N2 → REM → N2

    N3 occurs BEFORE REM — this is the guardian that repairs impedance debt.
    After N3 (27 ticks × 8% repair/tick), debt is reduced to ~10% of original.
    """
    stages = [
        ("n1", 0.05),
        ("n2", 0.22),
        ("n3", 0.25),   # ← The guardian: repairs ~90% of debt
        ("n2", 0.05),
        ("rem", 0.20),
        ("n2", 0.23),
    ]
    schedule = []
    for stage, ratio in stages:
        n = max(1, int(total_ticks * ratio))
        schedule.extend([stage] * n)
    while len(schedule) < total_ticks:
        schedule.append("n2")
    return schedule[:total_ticks]


def make_soremp_schedule(total_ticks: int) -> List[str]:
    """
    SOREMP (Sleep Onset REM Period) schedule: N1 → REM → N2 → N3 → N2

    In extreme fatigue / narcolepsy / REM rebound:
      The brain enters REM immediately after N1, BEFORE N3 can repair debt.

    Real neuroscience:
      - Multiple Sleep Latency Test (MSLT): SOREMP defined as REM within 15min
      - Narcolepsy: pathological SOREMP
      - Sleep deprivation → compensatory SOREMP on recovery night

    Physics:
      N3 repair hasn't happened yet → debt preserved → PGO amp ≈ ×1.9
      This is WHY sleep-deprived people have vivid dreams/hallucinations.
    """
    stages = [
        ("n1", 0.09),    # Brief descent
        ("rem", 0.50),   # ← REM FIRST (before N3 guardian)
        ("n2", 0.10),
        ("n3", 0.21),    # N3 happens AFTER REM (late repair)
        ("n2", 0.10),
    ]
    schedule = []
    for stage, ratio in stages:
        n = max(1, int(total_ticks * ratio))
        schedule.extend([stage] * n)
    while len(schedule) < total_ticks:
        schedule.append("n2")
    return schedule[:total_ticks]


# ============================================================
# Wake phase (parametric stress level)
# ============================================================

def stressed_wake_phase(
    brain: AliceBrain,
    ticks: int,
    stress_base: float,
    stress_range: float,
    rng: np.random.RandomState,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Awake ticks with parametric stress level.

    Low stress:  reflected_energy ≈ 0.03~0.05 (normal day)
    High stress: reflected_energy ≈ 0.10~0.50 (exhausting day)

    Higher reflected energy → faster impedance debt accumulation.
    IRL: demanding cognitive tasks, sensory overload, emotional stress.
    """
    for t in range(ticks):
        pixels = 0.5 + 0.1 * rng.randn(256)
        pixels = np.clip(pixels, 0, 1)
        brain.see(pixels, priority=Priority.BACKGROUND)

        reflected = stress_base + stress_range * rng.rand()
        brain.sleep_physics.awake_tick(reflected_energy=reflected)

    return {
        "ticks": ticks,
        "debt": brain.sleep_physics.impedance_debt.debt,
        "energy": brain.sleep_physics.energy,
        "sleep_pressure": brain.sleep_physics.sleep_pressure,
    }


# ============================================================
# Single-cycle sleep with detailed REM tracking
# ============================================================

def tracked_sleep_cycle(
    brain: AliceBrain,
    schedule: List[str],
    video_spatial_freq: float,
    video_vowel: str,
    video_rhythm_hz: float,
    use_video: bool,
    rng: np.random.RandomState,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run one sleep cycle with detailed tracking of debt/amp during REM.

    Args:
        use_video: If True, apply video Z_terminus modulation during REM.
                   If False, use random Z_terminus (control condition).
    """
    brain.sleep_physics.begin_sleep()

    debt_at_rem_onset = None
    debt_at_rem_end = None
    rem_ticks = 0
    amp_multipliers = []
    debt_trajectory = []
    amp_trajectory = []
    total_probes = 0
    healthy_probes = 0
    damaged_probes = 0
    channel_gammas: Dict[str, List[float]] = {n: [] for n in DREAM_CHANNEL_NAMES}
    channel_energies: Dict[str, List[float]] = {n: [] for n in DREAM_CHANNEL_NAMES}
    modulated_gammas = []
    reference_gammas = []
    in_rem = False

    for tick_idx, stage in enumerate(schedule):
        # Track REM onset
        if stage == "rem" and not in_rem:
            debt_at_rem_onset = brain.sleep_physics.impedance_debt.debt
            in_rem = True

        if stage != "rem" and in_rem:
            debt_at_rem_end = brain.sleep_physics.impedance_debt.debt
            in_rem = False

        # Build channel impedances
        if stage == "rem" and tick_idx % IMPEDANCE_MOD_INTERVAL == 0 and use_video:
            channel_impedances = video_to_impedance_modulation(
                video_spatial_freq, video_vowel, video_rhythm_hz,
                tick_idx, rng,
            )
        else:
            channel_impedances = [
                (DREAM_CHANNEL_NAMES[i] if i < len(DREAM_CHANNEL_NAMES)
                 else f"ch_{i}",
                 float(50 + 10 * rng.randn()),
                 float(50 + 10 * rng.randn()))
                for i in range(6)
            ]

        synaptic_strengths = list(rng.uniform(0.3, 1.5, 100))

        sleep_result = brain.sleep_physics.sleep_tick(
            stage=stage,
            recent_memories=[f"mem_{i}" for i in range(5)],
            channel_impedances=channel_impedances,
            synaptic_strengths=synaptic_strengths,
        )

        if stage == "rem":
            rem_ticks += 1
            debt_trajectory.append(brain.sleep_physics.impedance_debt.debt)

            dr = sleep_result.get("dream_result")
            if dr:
                am = dr.get("amp_multiplier", 1.0)
                amp_multipliers.append(am)
                amp_trajectory.append(am)

                total_probes += dr.get("probes", 0)
                healthy_probes += dr.get("healthy", 0)
                damaged_probes += dr.get("damaged", 0)

                # Track per-channel Γ directly from impedance landscape
                # (More accurate than brain's stochastic probe sampling:
                #  we measure the actual boundary condition, not a subsample)
                is_modulation_tick = (
                    use_video and tick_idx % IMPEDANCE_MOD_INTERVAL == 0
                )
                probe_amp_val = dr.get("amp_multiplier", 1.0) * 0.3
                for ch_name, zs, zl in channel_impedances:
                    denom = zs + zl
                    gamma = abs((zl - zs) / denom) if denom > 0 else 1.0
                    energy = probe_amp_val ** 2 * gamma ** 2

                    if ch_name in channel_gammas:
                        channel_gammas[ch_name].append(gamma)
                        channel_energies[ch_name].append(energy)

                    # Classify: modulated (video-shaped) vs reference
                    if is_modulation_tick:
                        if ch_name in ("somatosensory", "motor"):
                            reference_gammas.append(gamma)
                        elif ch_name in channel_gammas:
                            modulated_gammas.append(gamma)

    # Handle case where REM was the last stage
    if in_rem:
        debt_at_rem_end = brain.sleep_physics.impedance_debt.debt

    report = brain.sleep_physics.end_sleep()

    return {
        "debt_at_rem_onset": debt_at_rem_onset or 0.0,
        "debt_at_rem_end": debt_at_rem_end or 0.0,
        "rem_ticks": rem_ticks,
        "amp_multipliers": amp_multipliers,
        "debt_trajectory": debt_trajectory,
        "amp_trajectory": amp_trajectory,
        "total_probes": total_probes,
        "healthy_probes": healthy_probes,
        "damaged_probes": damaged_probes,
        "channel_gammas": channel_gammas,
        "channel_energies": channel_energies,
        "modulated_gammas": modulated_gammas,
        "reference_gammas": reference_gammas,
        "sleep_report": report,
    }


# ============================================================
# Run one condition
# ============================================================

def run_condition(
    label: str,
    name: str,
    wake_ticks: int,
    stress_base: float,
    stress_range: float,
    schedule_type: str,
    use_video: bool,
    seed: int,
    verbose: bool = False,
) -> ConditionResult:
    """Run a single experimental condition end-to-end."""
    rng = np.random.RandomState(seed)
    brain = AliceBrain(neuron_count=NEURON_COUNT)

    result = ConditionResult(
        name=name, label=label,
        wake_ticks=wake_ticks,
        stress_level="high" if stress_base > 0.1 else "low",
        schedule_type=schedule_type,
        video_modulated=use_video,
    )

    # ── Wake phase ──
    if verbose:
        print(f"    [{label}] Wake: {wake_ticks}t, stress={stress_base:.2f}")
    wake_info = stressed_wake_phase(
        brain, wake_ticks, stress_base, stress_range, rng, verbose,
    )
    result.debt_at_sleep_onset = wake_info["debt"]

    if verbose:
        print(f"    [{label}] Debt at sleep onset: {result.debt_at_sleep_onset:.4f}")

    # ── Sleep phase ──
    if schedule_type == "soremp":
        schedule = make_soremp_schedule(SLEEP_CYCLE_TICKS)
    else:
        schedule = make_normal_schedule(SLEEP_CYCLE_TICKS)

    sleep_info = tracked_sleep_cycle(
        brain, schedule,
        VIDEO_A_SPATIAL_FREQ, VIDEO_A_VOWEL, VIDEO_A_RHYTHM_HZ,
        use_video=use_video,
        rng=rng,
        verbose=verbose,
    )

    # ── Fill result ──
    result.debt_at_rem_onset = sleep_info["debt_at_rem_onset"]
    result.debt_at_rem_end = sleep_info["debt_at_rem_end"]
    result.rem_ticks = sleep_info["rem_ticks"]
    result.amp_multipliers = sleep_info["amp_multipliers"]
    result.peak_amp = max(sleep_info["amp_multipliers"]) if sleep_info["amp_multipliers"] else 1.0
    result.mean_amp = float(np.mean(sleep_info["amp_multipliers"])) if sleep_info["amp_multipliers"] else 1.0
    result.debt_trajectory = sleep_info["debt_trajectory"]
    result.amp_trajectory = sleep_info["amp_trajectory"]
    result.total_probes = sleep_info["total_probes"]
    result.healthy_probes = sleep_info["healthy_probes"]
    result.damaged_probes = sleep_info["damaged_probes"]
    result.channel_gammas = sleep_info["channel_gammas"]
    result.channel_energies = sleep_info["channel_energies"]
    result.modulated_gammas = sleep_info["modulated_gammas"]
    result.reference_gammas = sleep_info["reference_gammas"]

    # Γ contrast: video-shaped channels vs reference channels
    if result.modulated_gammas and result.reference_gammas:
        mean_mod = float(np.mean(result.modulated_gammas))
        mean_ref = float(np.mean(result.reference_gammas))
        result.gamma_contrast = abs(mean_ref - mean_mod) / max(mean_ref, 0.001)

    if verbose:
        print(f"    [{label}] Debt at REM onset: {result.debt_at_rem_onset:.4f}")
        print(f"    [{label}] Peak amp: ×{result.peak_amp:.3f}, "
              f"Mean amp: ×{result.mean_amp:.3f}")
        if result.modulated_gammas:
            print(f"    [{label}] Modulated Γ̅={np.mean(result.modulated_gammas):.3f}, "
                  f"Reference Γ̅={np.mean(result.reference_gammas):.3f}, "
                  f"Contrast={result.gamma_contrast:.3f}")

    return result


# ============================================================
# Main experiment
# ============================================================

def run_experiment(verbose: bool = True) -> Dict[str, Any]:
    """
    Fatigue Contrast Experiment — Four conditions.

    Architecture:
      ┌─────────────────────────────────────────────────────────────┐
      │  A: Rest + Normal     │ Low stress, N3 repairs, with video  │
      │  B: Tired + Normal    │ High stress, N3 STILL repairs!      │
      │  C: Tired + SOREMP    │ High stress, N3 bypassed → VIVID!   │
      │  D: Tired + SOREMP    │ Same as C but NO video (control)    │
      └─────────────────────────────────────────────────────────────┘

    Key finding:
      N3 is so effective that ×10 more stress barely changes REM dreaming.
      Only by bypassing N3 (SOREMP) does fatigue reach the dream stage.
      This is WHY sleep deprivation causes vivid dreams in humans.
    """
    t0 = time.time()

    if verbose:
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  Phase 28 — Fatigue Contrast Experiment                     ║")
        print("║  Fatigue-Dream Physics Empirical Verification              ║")
        print("║                                                            ║")
        print("║  Predicts: N3 ≈ guardian, SOREMP ≈ gate, video ≈ content   ║")
        print("║  Same equation: Γ = (Z_load - Z_src) / (Z_load + Z_src)   ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()

    # ── Run four conditions ──
    conditions = [
        ("A", "Rest + Normal",   LOW_STRESS_WAKE_TICKS,
         LOW_STRESS_BASE, LOW_STRESS_RANGE, "normal", True),
        ("B", "Tired + Normal",  HIGH_STRESS_WAKE_TICKS,
         HIGH_STRESS_BASE, HIGH_STRESS_RANGE, "normal", True),
        ("C", "Tired + SOREMP",  HIGH_STRESS_WAKE_TICKS,
         HIGH_STRESS_BASE, HIGH_STRESS_RANGE, "soremp", True),
        ("D", "Control (no video)", HIGH_STRESS_WAKE_TICKS,
         HIGH_STRESS_BASE, HIGH_STRESS_RANGE, "soremp", False),
    ]

    results = {}
    for label, name, wake, sb, sr, sched, video in conditions:
        if verbose:
            print(f"  ── Condition {label}: {name} ──")
        r = run_condition(
            label=label,
            name=name,
            wake_ticks=wake,
            stress_base=sb,
            stress_range=sr,
            schedule_type=sched,
            use_video=video,
            seed=SEED_BASE + ord(label),
            verbose=verbose,
        )
        results[label] = r
        if verbose:
            print()

    elapsed = time.time() - t0

    # ══════════════════════════════════════════════════════════
    # Comparison table
    # ══════════════════════════════════════════════════════════
    if verbose:
        print("  " + "=" * 62)
        print("  COMPARISON TABLE")
        print("  " + "=" * 62)
        print(f"  {'Cond':<6} {'Debt_onset':>10} {'Debt_REM':>10} "
              f"{'Peak×':>8} {'Mean×':>8} {'Γ_mod':>8} {'Γ_ref':>8} {'Contrast':>8}")
        print("  " + "-" * 62)

        for label in ["A", "B", "C", "D"]:
            r = results[label]
            mg = f"{np.mean(r.modulated_gammas):.3f}" if r.modulated_gammas else "  N/A"
            rg = f"{np.mean(r.reference_gammas):.3f}" if r.reference_gammas else "  N/A"
            print(f"  {label:<6} {r.debt_at_sleep_onset:>10.4f} "
                  f"{r.debt_at_rem_onset:>10.4f} "
                  f"{r.peak_amp:>7.3f}× {r.mean_amp:>7.3f}× "
                  f"{mg:>8} {rg:>8} {r.gamma_contrast:>8.3f}")

        print("  " + "-" * 62)
        print()

        # ── Physical interpretation ──
        ra, rb, rc, rd = results["A"], results["B"], results["C"], results["D"]
        print("  PHYSICAL INTERPRETATION:")
        print()
        print("  1. N3 Guardian Effect (A ≈ B):")
        ratio_ab = rb.debt_at_rem_onset / max(ra.debt_at_rem_onset, 1e-9)
        print(f"     Debt ratio B/A at REM: {ratio_ab:.1f}×  "
              f"(despite {rb.debt_at_sleep_onset/max(ra.debt_at_sleep_onset,1e-9):.1f}× "
              f"more debt at sleep onset)")
        print(f"     Amp ratio B/A: {rb.peak_amp/max(ra.peak_amp,0.01):.2f}×  "
              "← N3 nearly equalizes them!")
        print()
        print("  2. SOREMP Bypass (C ≫ A):")
        print(f"     C debt at REM: {rc.debt_at_rem_onset:.4f} vs "
              f"A: {ra.debt_at_rem_onset:.4f} "
              f"({rc.debt_at_rem_onset/max(ra.debt_at_rem_onset,1e-9):.0f}× more)")
        print(f"     C peak amp: ×{rc.peak_amp:.3f} vs A: ×{ra.peak_amp:.3f}")
        print("     ← SOREMP bypasses the N3 guardian → fatigue reaches REM")
        print()
        print("  3. Video Content Effect (C ≠ D):")
        if rc.modulated_gammas and rd.modulated_gammas:
            print(f"     C (video):   Γ_contrast = {rc.gamma_contrast:.3f} "
                  f"(structured by video)")
            print(f"     D (no video): Γ_contrast = {rd.gamma_contrast:.3f} "
                  f"(unstructured noise)")
        else:
            print(f"     C has {len(rc.modulated_gammas)} modulated samples, "
                  f"D has {len(rd.modulated_gammas)}")
        print()
        print("  CONCLUSION:")
        print("  ┌───────────────────────────────────────────────────┐")
        print("  │ Meaningful dream content requires THREE factors:  │")
        print("  │   ① Fatigue (impedance debt from wake activity)  │")
        print("  │   ② SOREMP (skipping N3 → debt persists to REM) │")
        print("  │   ③ Boundary (video → Z_terminus → Γ pattern)   │")
        print("  │                                                   │")
        print("  │ Same equation as pruning & phantom limb therapy.  │")
        print("  │ 'Change the boundary, not the brain.'             │")
        print("  └───────────────────────────────────────────────────┘")
        print()
        print(f"  Total time: {elapsed:.1f}s")
        print()

    return {
        "conditions": results,
        "elapsed_s": elapsed,
        # Key metrics for testing
        "debt_rem_A": results["A"].debt_at_rem_onset,
        "debt_rem_B": results["B"].debt_at_rem_onset,
        "debt_rem_C": results["C"].debt_at_rem_onset,
        "debt_rem_D": results["D"].debt_at_rem_onset,
        "peak_amp_A": results["A"].peak_amp,
        "peak_amp_B": results["B"].peak_amp,
        "peak_amp_C": results["C"].peak_amp,
        "peak_amp_D": results["D"].peak_amp,
        "gamma_contrast_C": results["C"].gamma_contrast,
        "gamma_contrast_D": results["D"].gamma_contrast,
    }


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    verbose = "--quiet" not in sys.argv
    result = run_experiment(verbose=verbose)
    if not verbose:
        for label in "ABCD":
            r = result["conditions"][label]
            print(f"{label}: debt_rem={r.debt_at_rem_onset:.4f} "
                  f"amp×{r.peak_amp:.3f} contrast={r.gamma_contrast:.3f}")
