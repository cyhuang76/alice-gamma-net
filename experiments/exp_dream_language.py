#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 27 — The Cradle of Language (Dream Incubation & Emergent Communication)

exp_dream_language.py — Dream Language Genesis Experiment

Core Hypothesis:
  Language does not originate in communication — it originates in dreams.
  During REM sleep, PGO waves randomly traverse neural channels.
  External sensory stimulation does NOT enter the sleeping brain.
  Instead, the stimulus changes the TERMINAL IMPEDANCE at the sensory
  boundary — exactly as mirror therapy changes Z_load at a phantom limb.

  Non-invasive impedance modulation (replaces invasive signal injection):
    Phantom limb: Z_load = ∞ (open circuit) → Γ = 1.0 → pain
    Mirror: visual_feedback → Z_load approaches Z_normal → Γ decreases

    Language channels: Z_load = undefined (no experience) → high Γ
    Dream video: changes Z_terminus at retina/cochlea → Γ landscape shifts
    Brain's own PGO waves reflect off modified terminus → dream content

  Key insight: We do NOT inject signals into the sleeping brain.
  This explicitly rejects the invasive-chip approach (Neuralink-style
  direct neural writing). Instead, we change BOUNDARY CONDITIONS
  at the sensory terminus. The brain's own PGO waves do the rest.

  The three faces of the same equation:
    Pruning:      birth Z_random → Hebbian Γ screening → specialisation
    Phantom limb: amputation Z_∞ → mirror Z_match → therapy
    Dream:        fatigue D_imp → video Z_terminus → PGO reflection → content

  All driven by: E_reflected = E_signal × Γ²

Experimental Design:
  Two Alice instances (Alice-α, Alice-β) sleep with a DIFFERENT "video"
  playing near them during REM:

  Video A: spatial freq=5 Hz, vowel /a/, rhythm 2Hz
  Video B: spatial freq=30 Hz, vowel /i/, rhythm 3Hz

  The video does NOT enter through brain.see() / brain.hear().
  Instead, the video's properties modulate the terminal impedance
  of the sensory channels that PGO probes encounter:
  - Visual channels: Z_terminus shaped by spatial frequency
  - Auditory channels: Z_terminus shaped by vowel formants
  - Rhythm: temporal modulation of impedance landscape

  After N dream cycles, let the two Alices communicate via the
  Social Resonance Engine's bidirectional coupling.

  Measurement:
  1. Do PGO probe reflections show structured Γ patterns?
  2. Do semantic field attractors form from impedance fingerprints?
  3. When Alice-α and Alice-β communicate, does Γ_social decrease?
  4. Does inner monologue reference dream-formed patterns?

  "We don't teach her to speak. We let her dream,
   and observe whether she invents words."

Design Principles (Paper II §1.1 compliance):
  1. Physics first: Video → Z_terminus modulation → PGO reflection (NOT injection)
  2. O(1) perception: Impedance modulation is O(1) per channel
  3. Emergence: NO language is programmed. We observe what forms.

Run: python -m experiments.exp_dream_language
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alice.alice_brain import AliceBrain
from alice.core.protocol import Priority, Modality
from alice.body.cochlea import generate_vowel


# ============================================================
# Constants
# ============================================================

NEURON_COUNT = 80
SEED_ALPHA = 42
SEED_BETA = 137

# Dream cycle parameters
AWAKE_TICKS = 80            # Pre-sleep wakefulness (build sleep pressure)
SLEEP_CYCLE_TICKS = 110     # One N1→N2→N3→N2→REM cycle
N_DREAM_CYCLES = 4          # Number of complete sleep cycles per night
N_NIGHTS = 3                # Number of nights of dream incubation

# REM dream video parameters
REM_TICKS_PER_CYCLE = 22    # ~20% of 110 ticks = REM window
IMPEDANCE_MOD_INTERVAL = 2  # Modulate Z_terminus every 2 ticks during REM

# Communication phase
COMM_ROUNDS = 20            # Rounds of bidirectional coupling

# Video A: spatial freq=5, vowel /a/, rhythm 2Hz
VIDEO_A_SPATIAL_FREQ = 5.0
VIDEO_A_VOWEL = "a"
VIDEO_A_RHYTHM_HZ = 2.0

# Video B: spatial freq=30, vowel /i/, rhythm 3Hz
VIDEO_B_SPATIAL_FREQ = 30.0
VIDEO_B_VOWEL = "i"
VIDEO_B_RHYTHM_HZ = 3.0

# Print control
PRINT_INTERVAL = 20

# ============================================================
# Non-invasive impedance modulation constants
# (Rejects invasive chip approach — mirror therapy principle)
# ============================================================

Z_BASE = 75.0                        # Characteristic impedance (Ω)
IMPEDANCE_MOD_DEPTH = 0.4            # Max Z_load modulation fraction
SPATIAL_FREQ_BOUNDARY = 15.0         # Visual channel split point (Hz)

# Vowel formant centres (Hz) — from acoustic phonetics
VOWEL_FORMANTS: Dict[str, Tuple[float, float]] = {
    "a": (700.0, 1220.0),     # /a/ F1≈700, F2≈1220
    "i": (280.0, 2500.0),     # /i/ F1≈280, F2≈2500
    "u": (310.0, 870.0),      # /u/ F1≈310, F2≈870
}

# Auditory channel frequency bands
FORMANT_BAND_F1 = (200.0, 1500.0)
FORMANT_BAND_F2 = (1500.0, 4000.0)

# Named channels — each is a coaxial terminus in the sensory pathway
DREAM_CHANNEL_NAMES = [
    "visual_low",       # Low spatial frequency (retinal wide-field cells)
    "visual_high",      # High spatial frequency (retinal detail cells)
    "auditory_f1",      # First formant band (cochlear low)
    "auditory_f2",      # Second formant band (cochlear high)
    "somatosensory",    # Body sense (unmodulated reference)
    "motor",            # Motor efference (unmodulated reference)
]


# ============================================================
# Data structures
# ============================================================

@dataclass
class DreamRecord:
    """Records what happened during one REM window."""
    night: int
    cycle: int
    video_label: str
    ticks_in_rem: int = 0
    stimuli_presented: int = 0         # Back-compat: counts Z_terminus modulations
    concepts_recognized: List[str] = field(default_factory=list)
    gammas: List[float] = field(default_factory=list)
    wernicke_observations: int = 0
    n400_events: int = 0
    # Fatigue-modulated dreaming (託夢物理)
    fatigue_factors: List[float] = field(default_factory=list)
    amp_multipliers: List[float] = field(default_factory=list)
    peak_fatigue: float = 0.0
    peak_amp_multiplier: float = 1.0
    # Non-invasive impedance modulation (替代入侵式晶片)
    modulations_applied: int = 0       # Z_terminus modulation events
    modulated_gammas: List[float] = field(default_factory=list)  # Γ at modulated channels


@dataclass
class SemanticSnapshot:
    """Snapshot of semantic field state."""
    n_attractors: int = 0
    attractor_labels: List[str] = field(default_factory=list)
    attractor_masses: List[float] = field(default_factory=list)
    total_mass: float = 0.0


@dataclass
class CommunicationRecord:
    """Records one round of Alice-α ↔ Alice-β communication."""
    round_num: int
    gamma_social_ab: float = 0.0
    gamma_social_ba: float = 0.0
    energy_transfer_ab: float = 0.0
    energy_transfer_ba: float = 0.0
    pressure_a_before: float = 0.0
    pressure_a_after: float = 0.0
    pressure_b_before: float = 0.0
    pressure_b_after: float = 0.0
    sync_delta: float = 0.0


# ============================================================
# Video generators (stimulus models — NOT injected into brain)
# ============================================================

def generate_video_frame(
    spatial_freq: float,
    frame_idx: int,
    resolution: int = 256,
) -> np.ndarray:
    """
    Generate a single video frame as a 1D pixel array.

    NOTE: This models the PHYSICAL stimulus in the environment.
    The stimulus is NOT fed to brain.see() (that would be invasive-chip
    injection). Instead, the stimulus properties are used by
    video_to_impedance_modulation() to compute Z_terminus changes.

    Physics: a sinusoidal luminance pattern at a specific spatial frequency.
    Different spatial frequencies → different visual Γ fingerprints.
    Frame index creates temporal variation (phase shift).
    """
    t = np.linspace(0, 1, resolution)
    phase = frame_idx * 0.1  # Slow phase drift across frames
    pixels = 0.5 + 0.4 * np.sin(2 * np.pi * spatial_freq * t + phase)
    return pixels


def generate_audio_frame(
    vowel: str,
    rhythm_hz: float,
    frame_idx: int,
    tick_duration: float = 0.1,
) -> Optional[np.ndarray]:
    """
    Generate audio for one tick, following rhythmic pattern.

    NOTE: This models the PHYSICAL audio in the environment.
    The audio is NOT fed to brain.hear() (that would be invasive-chip
    injection). Instead, the vowel/rhythm properties are used by
    video_to_impedance_modulation() to compute Z_terminus changes.

    Only produces sound at rhythm onset ticks (simulating rhythmic speech).
    Returns None for silent ticks.
    """
    # Rhythm: produce sound every (1/rhythm_hz) seconds
    period_ticks = max(1, int(1.0 / (rhythm_hz * tick_duration)))
    if frame_idx % period_ticks == 0:
        return generate_vowel(vowel, fundamental=150.0, duration=tick_duration)
    return None


# ============================================================
# Non-invasive impedance modulation (the mirror therapy principle)
# ============================================================

def video_to_impedance_modulation(
    spatial_freq: float,
    vowel: str,
    rhythm_hz: float,
    frame_idx: int,
    rng: np.random.RandomState,
) -> List[Tuple[str, float, float]]:
    """
    Convert video parameters to terminal impedance modulation.

    NON-INVASIVE — this rejects the invasive-chip paradigm.

    The video sits at the sensory terminus (retina / cochlea), changing
    the boundary impedance that PGO probes encounter during REM.
    The brain does NOT receive the video as input.
    The brain's OWN PGO waves reflect off the modified terminus.

    This is identical to mirror therapy (Ramachandran 1996):
      Mirror: visual feedback → Z_load at phantom limb changes → Γ↓ → pain↓
      Video:  sensory pattern → Z_terminus at retina/cochlea changes → Γ pattern

    Same equation: Γ = (Z_load - Z_src) / (Z_load + Z_src)
    Same as pruning: matching connections survive, mismatched are pruned.
    Same as fatigue: D_imp drives PGO amplitude for probing.

    Args:
        spatial_freq: Visual spatial frequency of the video
        vowel: Vowel sound in the video (/a/, /i/, /u/)
        rhythm_hz: Temporal rhythm of the video
        frame_idx: Current frame index (temporal phase)
        rng: Random state for natural variation

    Returns:
        List of (channel_name, Z_source, Z_load) tuples
        Modulated channels have Z_load closer to Z_src (lower Γ)
        Unmodulated channels have random Z_load (reference noise)
    """
    formants = VOWEL_FORMANTS.get(vowel, VOWEL_FORMANTS["a"])
    f1, f2 = formants

    # Temporal phase from rhythm — sinusoidal impedance modulation
    phase = np.sin(2 * np.pi * rhythm_hz * frame_idx * 0.01)
    temporal_mod = 0.5 + 0.5 * phase  # [0, 1]

    channels: List[Tuple[str, float, float]] = []

    for name in DREAM_CHANNEL_NAMES:
        z_src = Z_BASE + rng.randn() * 5.0          # Natural source variation
        z_load_base = Z_BASE + rng.randn() * 20.0   # Random terminus

        # Compute match strength: how well this channel resonates
        match = 0.0

        if name == "visual_low" and spatial_freq < SPATIAL_FREQ_BOUNDARY:
            # Low-freq video matches low visual channel
            centre = SPATIAL_FREQ_BOUNDARY / 2.0
            match = max(0.0, 1.0 - abs(spatial_freq - centre) / centre)

        elif name == "visual_high" and spatial_freq >= SPATIAL_FREQ_BOUNDARY:
            # High-freq video matches high visual channel
            centre = (SPATIAL_FREQ_BOUNDARY + 60.0) / 2.0
            half = (60.0 - SPATIAL_FREQ_BOUNDARY) / 2.0
            match = max(0.0, 1.0 - abs(spatial_freq - centre) / half)

        elif name == "auditory_f1":
            lo, hi = FORMANT_BAND_F1
            if lo <= f1 <= hi:
                centre = (lo + hi) / 2.0
                match = max(0.0, 1.0 - abs(f1 - centre) / ((hi - lo) / 2.0))

        elif name == "auditory_f2":
            lo, hi = FORMANT_BAND_F2
            if lo <= f2 <= hi:
                centre = (lo + hi) / 2.0
                match = max(0.0, 1.0 - abs(f2 - centre) / ((hi - lo) / 2.0))

        # somatosensory, motor: match = 0 (no video modulation → reference)

        # Apply modulation: push Z_load toward Z_src (reducing Γ)
        modulation = match * temporal_mod * IMPEDANCE_MOD_DEPTH
        z_load = z_load_base + modulation * (z_src - z_load_base)

        channels.append((name, max(10.0, z_src), max(10.0, z_load)))

    return channels


# ============================================================
# Helper: build sleep stage schedule
# ============================================================

def make_sleep_schedule(total_ticks: int) -> List[str]:
    """
    Generate one sleep cycle: N1→N2→N3→N2→REM

    Proportions: N1(5%) → N2(22%) → N3(25%) → N2(5%) → REM(20%) → N2(23%)
    """
    stages = [
        ("n1", 0.05),
        ("n2", 0.22),
        ("n3", 0.25),
        ("n2", 0.05),
        ("rem", 0.20),
        ("n2", 0.23),
    ]
    schedule = []
    for stage, ratio in stages:
        n = max(1, int(total_ticks * ratio))
        schedule.extend([stage] * n)
    # Pad or trim
    while len(schedule) < total_ticks:
        schedule.append("n2")
    return schedule[:total_ticks]


# ============================================================
# Helper: snapshot semantic field
# ============================================================

def snapshot_semantic_field(brain: AliceBrain) -> SemanticSnapshot:
    """Take a snapshot of the semantic field's current attractor state."""
    sf = brain.semantic_field.field  # Access the underlying SemanticField
    attractors = sf.attractors if hasattr(sf, 'attractors') else {}
    labels = list(attractors.keys())
    masses = [a.total_mass for a in attractors.values()]
    return SemanticSnapshot(
        n_attractors=len(labels),
        attractor_labels=labels,
        attractor_masses=masses,
        total_mass=sum(masses),
    )


# ============================================================
# Helper: snapshot Wernicke transition matrix
# ============================================================

def snapshot_wernicke(brain: AliceBrain) -> Dict[str, Any]:
    """Snapshot Wernicke's transition matrix structure."""
    w = brain.wernicke
    tm = w.transitions
    concepts = list(tm.counts.keys()) if hasattr(tm, 'counts') else []
    total_transitions = 0
    non_zero_transitions = 0
    for from_c in concepts:
        neighbors = tm.counts.get(from_c, {})
        for to_c, count in neighbors.items():
            total_transitions += count
            if count > 0:
                non_zero_transitions += 1
    chunks = w.get_mature_chunks()
    return {
        "n_concepts": len(concepts),
        "concepts": concepts[:20],  # First 20
        "total_transitions": total_transitions,
        "non_zero_transitions": non_zero_transitions,
        "n_mature_chunks": len(chunks),
        "chunks": [str(c) for c in chunks[:10]],
    }


# ============================================================
# Phase 1: Wake phase — build sleep pressure
# ============================================================

def wake_phase(brain: AliceBrain, ticks: int, rng: np.random.RandomState,
               label: str = "", verbose: bool = False) -> Dict[str, Any]:
    """Awake ticks to build sleep pressure naturally."""
    if verbose:
        print(f"    ── Wake phase ({label}): {ticks} ticks")

    for t in range(ticks):
        # Random mild stimuli (ambient environment)
        pixels = 0.5 + 0.1 * rng.randn(256)
        pixels = np.clip(pixels, 0, 1)
        brain.see(pixels, priority=Priority.BACKGROUND)

        # Sleep physics: accumulate debt
        brain.sleep_physics.awake_tick(
            reflected_energy=0.03 + 0.02 * rng.rand(),
        )

    return {
        "ticks": ticks,
        "sleep_pressure": brain.sleep_physics.sleep_pressure,
        "energy": brain.sleep_physics.energy,
    }


# ============================================================
# Phase 2: Dream incubation — non-invasive impedance modulation
# ============================================================

def dream_incubation_night(
    brain: AliceBrain,
    video_spatial_freq: float,
    video_vowel: str,
    video_rhythm_hz: float,
    video_label: str,
    night_idx: int,
    rng: np.random.RandomState,
    verbose: bool = False,
) -> List[DreamRecord]:
    """
    One night of dream incubation via NON-INVASIVE impedance modulation.

    ═══════════════════════════════════════════════════════════════
    DESIGN PRINCIPLE: Reject the invasive-chip paradigm.
    ═══════════════════════════════════════════════════════════════

    We do NOT call brain.see() or brain.hear() during sleep.
    That would be injecting signals into the perception pipeline —
    the exact approach used by invasive BCIs (Neuralink-style).

    Instead, the video modulates the TERMINAL IMPEDANCE at the
    sensory boundary (retina/cochlea), and the brain's own PGO
    waves reflect off these modified boundaries.

    This is identical to mirror therapy (Ramachandran 1996):
      Mirror:  visual feedback at body midline
               → Z_load at phantom terminus changes
               → Γ decreases → pain relief

      Video:   sensory pattern at retina/cochlea
               → Z_terminus at sensory channels changes
               → PGO probes reflect structured Γ pattern
               → dream content emerges from reflection

    Same equation, same physics, no invasion.
    """
    records = []
    brain.sleep_physics.begin_sleep()

    for cycle in range(N_DREAM_CYCLES):
        schedule = make_sleep_schedule(SLEEP_CYCLE_TICKS)
        dream_rec = DreamRecord(
            night=night_idx, cycle=cycle, video_label=video_label,
        )
        frame_idx = cycle * SLEEP_CYCLE_TICKS

        for tick_in_cycle, stage in enumerate(schedule):
            # --- Build channel impedances ---
            # During REM modulation ticks: video shapes the terminus
            # Otherwise: random impedances (no external influence)
            if (stage == "rem"
                    and tick_in_cycle % IMPEDANCE_MOD_INTERVAL == 0):
                # NON-INVASIVE: video → Z_terminus modulation
                # (NOT brain.see / brain.hear — that is the invasive path)
                channel_impedances = video_to_impedance_modulation(
                    video_spatial_freq, video_vowel, video_rhythm_hz,
                    frame_idx + tick_in_cycle, rng,
                )
                dream_rec.stimuli_presented += 1   # backward compat
                dream_rec.modulations_applied += 1

                # Track Γ at modulated channels
                mod_gammas = []
                for _, zs, zl in channel_impedances:
                    denom = zs + zl
                    g = abs((zl - zs) / denom) if denom > 0 else 1.0
                    mod_gammas.append(g)
                dream_rec.modulated_gammas.extend(mod_gammas)
            else:
                channel_impedances = [
                    (f"ch_{i}", float(50 + 10 * rng.randn()),
                     float(50 + 10 * rng.randn()))
                    for i in range(6)
                ]

            synaptic_strengths = list(rng.uniform(0.3, 1.5, 100))

            # --- Sleep physics tick ---
            # PGO probes inside sleep_tick now encounter the
            # video-modulated impedances (non-invasive boundary change)
            sleep_result = brain.sleep_physics.sleep_tick(
                stage=stage,
                recent_memories=[f"mem_{i}" for i in range(5)],
                channel_impedances=channel_impedances,
                synaptic_strengths=synaptic_strengths,
            )

            # --- REM: track fatigue and probe metrics ---
            if stage == "rem":
                dream_rec.ticks_in_rem += 1

                if sleep_result.get("dream_result"):
                    dr = sleep_result["dream_result"]
                    ff = dr.get("fatigue_factor", 0.0)
                    am = dr.get("amp_multiplier", 1.0)
                    dream_rec.fatigue_factors.append(ff)
                    dream_rec.amp_multipliers.append(am)
                    dream_rec.peak_fatigue = max(dream_rec.peak_fatigue, ff)
                    dream_rec.peak_amp_multiplier = max(
                        dream_rec.peak_amp_multiplier, am)

        records.append(dream_rec)

    brain.sleep_physics.end_sleep()
    return records


# ============================================================
# Phase 3: Communication — two Alices interact
# ============================================================

def communication_phase(
    alice_a: AliceBrain,
    alice_b: AliceBrain,
    rounds: int,
    verbose: bool = False,
) -> List[CommunicationRecord]:
    """
    Let Alice-α and Alice-β communicate bidirectionally.

    Each round:
    1. Both Alices' semantic pressure engines tick (pressure accumulates)
    2. Bidirectional social coupling transfers pressure
    3. Observe: do their Γ_social decrease over rounds? (rapport forming)
    4. Observe: do inner monologue events reference dream-formed concepts?
    """
    records = []

    for r in range(rounds):
        # Tick semantic pressure for both
        sp_a = alice_a.semantic_pressure.tick(
            arousal=0.5,
            phi=alice_a.vitals.consciousness,
            semantic_field=alice_a.semantic_field,
            wernicke=alice_a.wernicke,
        )
        sp_b = alice_b.semantic_pressure.tick(
            arousal=0.5,
            phi=alice_b.vitals.consciousness,
            semantic_field=alice_b.semantic_field,
            wernicke=alice_b.wernicke,
        )

        pressure_a = alice_a.semantic_pressure.pressure
        pressure_b = alice_b.semantic_pressure.pressure

        # Bidirectional coupling
        result_ab, result_ba = alice_a.social_resonance.bidirectional_couple(
            agent_a_id="alice_alpha",
            agent_b_id="alice_beta",
            pressure_a=pressure_a,
            pressure_b=pressure_b,
            empathy_a=0.6,
            empathy_b=0.6,
            effort_a=0.7,
            effort_b=0.7,
            phi_a=max(0.3, alice_a.vitals.consciousness),
            phi_b=max(0.3, alice_b.vitals.consciousness),
        )

        # Apply pressure release
        if pressure_a > 0:
            alice_a.semantic_pressure.release(
                gamma_speech=result_ab.gamma_social,
                phi=alice_a.vitals.consciousness,
            )
        if pressure_b > 0:
            alice_b.semantic_pressure.release(
                gamma_speech=result_ba.gamma_social,
                phi=alice_b.vitals.consciousness,
            )

        pressure_a_after = alice_a.semantic_pressure.pressure
        pressure_b_after = alice_b.semantic_pressure.pressure

        rec = CommunicationRecord(
            round_num=r,
            gamma_social_ab=result_ab.gamma_social,
            gamma_social_ba=result_ba.gamma_social,
            energy_transfer_ab=result_ab.energy_transfer,
            energy_transfer_ba=result_ba.energy_transfer,
            pressure_a_before=pressure_a,
            pressure_a_after=pressure_a_after,
            pressure_b_before=pressure_b,
            pressure_b_after=pressure_b_after,
            sync_delta=result_ab.sync_delta,
        )
        records.append(rec)

        if verbose and r % 5 == 0:
            print(f"        Round {r:2d}: "
                  f"Γ_ab={result_ab.gamma_social:.3f} "
                  f"Γ_ba={result_ba.gamma_social:.3f} "
                  f"η_ab={result_ab.energy_transfer:.3f} "
                  f"P_a={pressure_a:.3f}→{pressure_a_after:.3f} "
                  f"P_b={pressure_b:.3f}→{pressure_b_after:.3f}")

    return records


# ============================================================
# Main experiment
# ============================================================

def run_experiment(verbose: bool = True) -> Dict[str, Any]:
    """
    Dream Language Genesis — Full Experiment (Non-Invasive)

    Architecture:  (NO brain.see/hear during sleep — rejects invasive chips)
      ┌────────────────┐     ┌────────────────┐
      │   Alice-α      │     │   Alice-β      │
      │                │     │                │
      │  Video A →     │     │  Video B →     │
      │  Z_terminus    │     │  Z_terminus    │
      │  modulation    │     │  modulation    │
      │  (f=5,/a/,2Hz) │     │  (f=30,/i/,3Hz)│
      │                │     │                │
      │  PGO probes →  │     │  PGO probes →  │
      │  reflect off   │     │  reflect off   │
      │  modified Z    │     │  modified Z    │
      └───────┬────────┘     └───────┬────────┘
              │                      │
              └──── Communication ───┘
                   20 rounds of
                   bidirectional
                   social coupling
                          │
                          ▼
                   Observe:
                   - Γ patterns in PGO probes?
                   - Γ_social trend?
                   - Emergent structure?
    """
    t0 = time.time()

    if verbose:
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  Phase 27 — The Cradle of Language                         ║")
        print("║  Non-Invasive Dream Incubation (Z_terminus modulation)     ║")
        print("║                                                            ║")
        print("║  Rejects invasive-chip paradigm:                           ║")
        print("║   Video → Z_terminus, NOT Video → brain.see()             ║")
        print("║   Same physics as mirror therapy & synaptic pruning.       ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()

    # ── Create two Alice instances ──
    alice_a = AliceBrain(neuron_count=NEURON_COUNT)
    alice_b = AliceBrain(neuron_count=NEURON_COUNT)
    rng_a = np.random.RandomState(SEED_ALPHA)
    rng_b = np.random.RandomState(SEED_BETA)

    if verbose:
        print("  ┌─ Alice-α created (seed=42)")
        print("  └─ Alice-β created (seed=137)")
        print()

    # ── Baseline snapshots ──
    baseline_a = snapshot_semantic_field(alice_a)
    baseline_b = snapshot_semantic_field(alice_b)
    baseline_w_a = snapshot_wernicke(alice_a)
    baseline_w_b = snapshot_wernicke(alice_b)

    all_dream_records_a: List[DreamRecord] = []
    all_dream_records_b: List[DreamRecord] = []

    # ══════════════════════════════════════════════════════════
    # EXP 1: Dream Incubation (Multiple Nights)
    # ══════════════════════════════════════════════════════════
    if verbose:
        print("  ═══ EXP 1: Dream Incubation ═══")

    for night in range(N_NIGHTS):
        if verbose:
            print(f"\n  ── Night {night + 1}/{N_NIGHTS} ──")

        # Wake phase: build sleep pressure
        if verbose:
            print(f"  Alice-α:")
        wake_phase(alice_a, AWAKE_TICKS, rng_a, label="α", verbose=verbose)
        if verbose:
            print(f"  Alice-β:")
        wake_phase(alice_b, AWAKE_TICKS, rng_b, label="β", verbose=verbose)

        # Dream incubation: different videos
        if verbose:
            print(f"    ── Alice-α dreaming (Video A: f={VIDEO_A_SPATIAL_FREQ}, "
                  f"/{VIDEO_A_VOWEL}/, {VIDEO_A_RHYTHM_HZ}Hz)")
        records_a = dream_incubation_night(
            alice_a,
            VIDEO_A_SPATIAL_FREQ, VIDEO_A_VOWEL, VIDEO_A_RHYTHM_HZ,
            video_label="A",
            night_idx=night,
            rng=rng_a,
            verbose=verbose,
        )
        all_dream_records_a.extend(records_a)

        if verbose:
            print(f"    ── Alice-β dreaming (Video B: f={VIDEO_B_SPATIAL_FREQ}, "
                  f"/{VIDEO_B_VOWEL}/, {VIDEO_B_RHYTHM_HZ}Hz)")
        records_b = dream_incubation_night(
            alice_b,
            VIDEO_B_SPATIAL_FREQ, VIDEO_B_VOWEL, VIDEO_B_RHYTHM_HZ,
            video_label="B",
            night_idx=night,
            rng=rng_b,
            verbose=verbose,
        )
        all_dream_records_b.extend(records_b)

        # Night summary
        if verbose:
            total_mod_a = sum(r.modulations_applied for r in records_a)
            total_mod_b = sum(r.modulations_applied for r in records_b)
            mg_a = [g for r in records_a for g in r.modulated_gammas]
            mg_b = [g for r in records_b for g in r.modulated_gammas]
            mean_g_a = np.mean(mg_a) if mg_a else 1.0
            mean_g_b = np.mean(mg_b) if mg_b else 1.0
            print(f"    Summary: α={total_mod_a} Z-modulations (Γ̅={mean_g_a:.3f}) | "
                  f"β={total_mod_b} Z-modulations (Γ̅={mean_g_b:.3f})")
            # Fatigue metrics
            peak_f_a = max((r.peak_fatigue for r in records_a), default=0)
            peak_f_b = max((r.peak_fatigue for r in records_b), default=0)
            peak_m_a = max((r.peak_amp_multiplier for r in records_a), default=1)
            peak_m_b = max((r.peak_amp_multiplier for r in records_b), default=1)
            print(f"    Fatigue: α peak={peak_f_a:.3f} (×{peak_m_a:.2f}) | "
                  f"β peak={peak_f_b:.3f} (×{peak_m_b:.2f})")

    # ── Post-dream snapshots ──
    post_dream_a = snapshot_semantic_field(alice_a)
    post_dream_b = snapshot_semantic_field(alice_b)
    post_dream_w_a = snapshot_wernicke(alice_a)
    post_dream_w_b = snapshot_wernicke(alice_b)

    if verbose:
        print(f"\n  ═══ Dream Incubation Results ═══")
        print(f"  Alice-α semantic field: {baseline_a.n_attractors} → "
              f"{post_dream_a.n_attractors} attractors "
              f"(Δ={post_dream_a.n_attractors - baseline_a.n_attractors})")
        print(f"  Alice-β semantic field: {baseline_b.n_attractors} → "
              f"{post_dream_b.n_attractors} attractors "
              f"(Δ={post_dream_b.n_attractors - baseline_b.n_attractors})")
        print(f"  Alice-α Wernicke: {baseline_w_a['n_concepts']} → "
              f"{post_dream_w_a['n_concepts']} concepts, "
              f"{post_dream_w_a['n_mature_chunks']} chunks")
        print(f"  Alice-β Wernicke: {baseline_w_b['n_concepts']} → "
              f"{post_dream_w_b['n_concepts']} concepts, "
              f"{post_dream_w_b['n_mature_chunks']} chunks")
        if post_dream_a.attractor_labels:
            print(f"  Alice-α concepts: {post_dream_a.attractor_labels[:10]}")
        if post_dream_b.attractor_labels:
            print(f"  Alice-β concepts: {post_dream_b.attractor_labels[:10]}")

    # ══════════════════════════════════════════════════════════
    # EXP 2: Communication Phase
    # ══════════════════════════════════════════════════════════
    if verbose:
        print(f"\n  ═══ EXP 2: Communication Phase ({COMM_ROUNDS} rounds) ═══")

    comm_records = communication_phase(
        alice_a, alice_b, COMM_ROUNDS, verbose=verbose,
    )

    # ── Post-communication snapshots ──
    post_comm_a = snapshot_semantic_field(alice_a)
    post_comm_b = snapshot_semantic_field(alice_b)
    post_comm_w_a = snapshot_wernicke(alice_a)
    post_comm_w_b = snapshot_wernicke(alice_b)

    # ══════════════════════════════════════════════════════════
    # EXP 3: Divergence Analysis
    # ══════════════════════════════════════════════════════════
    if verbose:
        print(f"\n  ═══ EXP 3: Divergence Analysis ═══")

    # Compare concept sets
    concepts_a_set = set(post_comm_a.attractor_labels)
    concepts_b_set = set(post_comm_b.attractor_labels)
    shared = concepts_a_set & concepts_b_set
    only_a = concepts_a_set - concepts_b_set
    only_b = concepts_b_set - concepts_a_set

    if verbose:
        print(f"  Shared concepts: {len(shared)} — {list(shared)[:10]}")
        print(f"  Only Alice-α:    {len(only_a)} — {list(only_a)[:10]}")
        print(f"  Only Alice-β:    {len(only_b)} — {list(only_b)[:10]}")

    # Γ_social trend
    if comm_records:
        first_gamma = (comm_records[0].gamma_social_ab +
                       comm_records[0].gamma_social_ba) / 2
        last_gamma = (comm_records[-1].gamma_social_ab +
                      comm_records[-1].gamma_social_ba) / 2
        gamma_trend = last_gamma - first_gamma
        if verbose:
            print(f"  Γ_social trend: {first_gamma:.4f} → {last_gamma:.4f} "
                  f"(Δ={gamma_trend:+.4f})")

    # Energy transfer trend
    if comm_records:
        first_eta = (comm_records[0].energy_transfer_ab +
                     comm_records[0].energy_transfer_ba) / 2
        last_eta = (comm_records[-1].energy_transfer_ab +
                    comm_records[-1].energy_transfer_ba) / 2
        if verbose:
            print(f"  η trend: {first_eta:.4f} → {last_eta:.4f} "
                  f"(Δ={last_eta - first_eta:+.4f})")

    elapsed = time.time() - t0

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'=' * 64}")
        print(f"  EXPERIMENT COMPLETE — {elapsed:.1f}s")
        print(f"{'=' * 64}")
        print()
        print("  Observations (NOT conclusions — emergence must be observed):")
        total_mod_a = sum(r.modulations_applied for r in all_dream_records_a)
        total_mod_b = sum(r.modulations_applied for r in all_dream_records_b)
        print(f"    Z_terminus modulations:   α={total_mod_a}  β={total_mod_b}")
        print(f"    (Non-invasive: video→Z_terminus, NOT video→brain)")
        # Modulated Gamma stats
        all_mg_a = [g for r in all_dream_records_a for g in r.modulated_gammas]
        all_mg_b = [g for r in all_dream_records_b for g in r.modulated_gammas]
        if all_mg_a:
            print(f"    Modulated Γ: α mean={np.mean(all_mg_a):.3f} "
                  f"min={min(all_mg_a):.3f} max={max(all_mg_a):.3f}")
        if all_mg_b:
            print(f"                 β mean={np.mean(all_mg_b):.3f} "
                  f"min={min(all_mg_b):.3f} max={max(all_mg_b):.3f}")
        # Fatigue-modulated dreaming metrics
        all_fatigue_a = [f for r in all_dream_records_a for f in r.fatigue_factors]
        all_fatigue_b = [f for r in all_dream_records_b for f in r.fatigue_factors]
        all_amp_a = [m for r in all_dream_records_a for m in r.amp_multipliers]
        all_amp_b = [m for r in all_dream_records_b for m in r.amp_multipliers]
        if all_fatigue_a:
            print(f"    Fatigue (託夢物理): α mean={np.mean(all_fatigue_a):.3f} "
                  f"peak={max(all_fatigue_a):.3f} "
                  f"amp ×{np.mean(all_amp_a):.2f}~×{max(all_amp_a):.2f}")
        if all_fatigue_b:
            print(f"                       β mean={np.mean(all_fatigue_b):.3f} "
                  f"peak={max(all_fatigue_b):.3f} "
                  f"amp ×{np.mean(all_amp_b):.2f}~×{max(all_amp_b):.2f}")
        print(f"    Semantic attractors formed: α={post_dream_a.n_attractors}"
              f"  β={post_dream_b.n_attractors}")
        print(f"    Wernicke concepts:  α={post_dream_w_a['n_concepts']}"
              f"  β={post_dream_w_b['n_concepts']}")
        print(f"    Wernicke chunks:    α={post_dream_w_a['n_mature_chunks']}"
              f"  β={post_dream_w_b['n_mature_chunks']}")
        print(f"    Shared concepts after communication: {len(shared)}")
        print()
        print("  'The physics will tell us the answer.'")
        print("  'We change the boundary, not the brain.'")
        print()

    return {
        "elapsed_s": elapsed,

        # Dream incubation
        "dream_records_alpha": all_dream_records_a,
        "dream_records_beta": all_dream_records_b,

        # Semantic evolution
        "baseline_semantic_alpha": baseline_a,
        "baseline_semantic_beta": baseline_b,
        "post_dream_semantic_alpha": post_dream_a,
        "post_dream_semantic_beta": post_dream_b,
        "post_comm_semantic_alpha": post_comm_a,
        "post_comm_semantic_beta": post_comm_b,

        # Wernicke evolution
        "baseline_wernicke_alpha": baseline_w_a,
        "baseline_wernicke_beta": baseline_w_b,
        "post_dream_wernicke_alpha": post_dream_w_a,
        "post_dream_wernicke_beta": post_dream_w_b,
        "post_comm_wernicke_alpha": post_comm_w_a,
        "post_comm_wernicke_beta": post_comm_w_b,

        # Communication
        "communication_records": comm_records,
        "shared_concepts": list(shared),
        "only_alpha_concepts": list(only_a),
        "only_beta_concepts": list(only_b),

        # key metrics
        "n_attractors_alpha": post_comm_a.n_attractors,
        "n_attractors_beta": post_comm_b.n_attractors,
        "n_shared_concepts": len(shared),
        "gamma_social_trend": gamma_trend if comm_records else 0.0,

        # fatigue-modulated dream metrics (託夢物理)
        "peak_fatigue_alpha": max(
            (r.peak_fatigue for r in all_dream_records_a), default=0.0),
        "peak_fatigue_beta": max(
            (r.peak_fatigue for r in all_dream_records_b), default=0.0),
        "peak_amp_alpha": max(
            (r.peak_amp_multiplier for r in all_dream_records_a), default=1.0),
        "peak_amp_beta": max(
            (r.peak_amp_multiplier for r in all_dream_records_b), default=1.0),
    }


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    verbose = "--quiet" not in sys.argv
    result = run_experiment(verbose=verbose)
    if not verbose:
        print(f"attractors α={result['n_attractors_alpha']} "
              f"β={result['n_attractors_beta']} "
              f"shared={result['n_shared_concepts']} "
              f"Γ_trend={result['gamma_social_trend']:+.4f} "
              f"in {result['elapsed_s']:.1f}s")
