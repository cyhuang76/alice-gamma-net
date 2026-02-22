#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 27 — The Cradle of Language (Dream Incubation & Emergent Communication)

exp_dream_language.py — Dream Language Genesis Experiment

Core Hypothesis:
  Language does not originate in communication — it originates in dreams.
  During REM sleep, PGO waves randomly traverse neural channels.
  If visual/auditory stimuli are presented during REM (like mirror therapy
  provides visual feedback to a phantom limb), the sensory signals change
  the impedance landscape of language-related channels.

  Mirror therapy analogy:
    Phantom limb: Z_load = ∞ (open circuit) → Γ = 1.0 → pain
    Mirror: visual_feedback → Z_load approaches Z_normal → Γ decreases

    Language channels: Z_load = undefined (no experience) → high Γ
    Dream video: sensory input during REM → Z_load acquires structure → Γ decreases

  Key insight: We do NOT inject language. We provide raw sensory stimulation
  (video = pixels, audio = waveforms) during REM sleep. The existing
  perception pipeline (eye → semantic field → Wernicke → hippocampus)
  processes these signals normally. What emerges — if anything — is the
  system's own discovery.

Experimental Design:
  Two Alice instances (Alice-α, Alice-β) each watch a DIFFERENT "video"
  during REM sleep across multiple sleep cycles:

  Video A: Repeating visual pattern + vowel /a/ at 2Hz
  Video B: Different visual pattern + vowel /i/ at 3Hz

  Each video is a structured but meaningless sensory stream:
  - Visual: sinusoidal pixel patterns at specific spatial frequencies
  - Auditory: vowel sounds at specific temporal rhythms

  After N dream cycles, let the two Alices communicate via the
  Social Resonance Engine's bidirectional coupling.

  Measurement:
  1. Do semantic field attractors form from dream content?
  2. Do Wernicke transition matrices develop non-uniform structure?
  3. When Alice-α and Alice-β communicate, do their semantic pressures
     change? Do they develop shared or divergent symbol systems?
  4. Does inner monologue reference dream-formed concepts?

  "We don't teach her to speak. We let her dream,
   and observe whether she invents words."

Design Principles (Paper II §1.1 compliance):
  1. Physics first: Video → eye.see() / ear.hear() → existing Γ pipeline
  2. O(1) perception: Stimuli processed within standard perceive() tick
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
VIDEO_INJECT_INTERVAL = 2   # Inject stimuli every 2 ticks during REM

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
# Data structures
# ============================================================

@dataclass
class DreamRecord:
    """Records what happened during one REM window."""
    night: int
    cycle: int
    video_label: str
    ticks_in_rem: int = 0
    stimuli_presented: int = 0
    concepts_recognized: List[str] = field(default_factory=list)
    gammas: List[float] = field(default_factory=list)
    wernicke_observations: int = 0
    n400_events: int = 0


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
# Video generators
# ============================================================

def generate_video_frame(
    spatial_freq: float,
    frame_idx: int,
    resolution: int = 256,
) -> np.ndarray:
    """
    Generate a single video frame as a 1D pixel array.

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

    Only produces sound at rhythm onset ticks (simulating rhythmic speech).
    Returns None for silent ticks.
    """
    # Rhythm: produce sound every (1/rhythm_hz) seconds
    period_ticks = max(1, int(1.0 / (rhythm_hz * tick_duration)))
    if frame_idx % period_ticks == 0:
        return generate_vowel(vowel, fundamental=150.0, duration=tick_duration)
    return None


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
# Phase 2: Dream incubation — REM video injection
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
    One night of dream incubation: multiple sleep cycles,
    with sensory stimulation during REM windows.

    Mirror therapy analogy:
      Mirror therapy tick: visual_feedback = {limb_name: quality}
        → _apply_mirror_therapy() → Z_load approaches Z_normal → Γ decreases

      Dream incubation tick: brain.see(video_frame) + brain.hear(audio_frame)
        → eye/ear transduction → semantic field → Wernicke → hippocampus
        → impedance landscape changes naturally

    We do NOT call any special "inject" function.
    We simply present stimuli through the EXISTING sensory pipeline
    during REM, exactly as mirror therapy presents visual feedback
    through the EXISTING visual pipeline during pain.
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
            # --- Sleep physics tick ---
            channel_impedances = [
                (f"ch_{i}", float(50 + 10 * rng.randn()),
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

            # --- REM window: present video stimuli ---
            if stage == "rem":
                dream_rec.ticks_in_rem += 1

                if tick_in_cycle % VIDEO_INJECT_INTERVAL == 0:
                    # Visual stimulus: video frame through eye
                    vf = generate_video_frame(
                        video_spatial_freq, frame_idx + tick_in_cycle,
                    )
                    see_result = brain.see(vf, priority=Priority.NORMAL)

                    # Auditory stimulus: vowel through ear (if rhythm allows)
                    audio = generate_audio_frame(
                        video_vowel, video_rhythm_hz,
                        frame_idx + tick_in_cycle,
                    )
                    if audio is not None:
                        hear_result = brain.hear(audio, priority=Priority.NORMAL)
                        # Track what Wernicke observed
                        if "wernicke" in hear_result:
                            dream_rec.wernicke_observations += 1
                            w = hear_result["wernicke"]
                            if w.get("is_n400"):
                                dream_rec.n400_events += 1

                    dream_rec.stimuli_presented += 1

                    # Track concept recognition
                    if "semantic" in see_result:
                        bc = see_result["semantic"].get("best_concept")
                        g = see_result["semantic"].get("gamma", 1.0)
                        if bc:
                            dream_rec.concepts_recognized.append(bc)
                            dream_rec.gammas.append(g)

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
    Dream Language Genesis — Full Experiment

    Architecture:
      ┌────────────────┐     ┌────────────────┐
      │   Alice-α      │     │   Alice-β      │
      │                │     │                │
      │  Video A:      │     │  Video B:      │
      │  f=5, /a/, 2Hz │     │  f=30, /i/, 3Hz│
      │                │     │                │
      │  REM × 4 cycles│     │  REM × 4 cycles│
      │  × 3 nights    │     │  × 3 nights    │
      └───────┬────────┘     └───────┬────────┘
              │                      │
              └──── Communication ───┘
                   20 rounds of
                   bidirectional
                   social coupling
                          │
                          ▼
                   Observe:
                   - Shared symbols?
                   - Γ_social trend?
                   - Emergent structure?
    """
    t0 = time.time()

    if verbose:
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  Phase 27 — The Cradle of Language                         ║")
        print("║  Dream Incubation & Emergent Communication                 ║")
        print("║                                                            ║")
        print("║  'We don't teach her to speak.                             ║")
        print("║   We let her dream,                                        ║")
        print("║   and observe whether she invents words.'                  ║")
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
            total_stim_a = sum(r.stimuli_presented for r in records_a)
            total_stim_b = sum(r.stimuli_presented for r in records_b)
            concepts_a = set()
            concepts_b = set()
            for r in records_a:
                concepts_a.update(r.concepts_recognized)
            for r in records_b:
                concepts_b.update(r.concepts_recognized)
            print(f"    Summary: α={total_stim_a} stimuli, "
                  f"{len(concepts_a)} unique concepts | "
                  f"β={total_stim_b} stimuli, "
                  f"{len(concepts_b)} unique concepts")

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
        print(f"    Dream stimuli presented:  α={sum(r.stimuli_presented for r in all_dream_records_a)}"
              f"  β={sum(r.stimuli_presented for r in all_dream_records_b)}")
        print(f"    Semantic attractors formed: α={post_dream_a.n_attractors}"
              f"  β={post_dream_b.n_attractors}")
        print(f"    Wernicke concepts:  α={post_dream_w_a['n_concepts']}"
              f"  β={post_dream_w_b['n_concepts']}")
        print(f"    Wernicke chunks:    α={post_dream_w_a['n_mature_chunks']}"
              f"  β={post_dream_w_b['n_mature_chunks']}")
        print(f"    Shared concepts after communication: {len(shared)}")
        print()
        print("  'The physics will tell us the answer.'")
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
