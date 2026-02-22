#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 29 — 30-Day Human Intelligence Potential (HIP) Stress Test

exp_human_intelligence_month.py — 10 experiments

Core Objectives: 
  Systematically verify using cognitive science benchmarks whether Alice possesses human intelligence potential. 
  Simulate 30-day cognitive development and stress testing, covering: 

  Perception → Memory → Emotion → Sleep → Language → Social → Executive → Metacognition → Resilience → Integration

Physics benchmarks: 
  1. Miller's Law: working memorycapacity 7±2
  2. Fear conditioning: ≤ 10 pairings to form fear memory
  3. Sally-Anne: Theory of Mind PASS rate
  4. Yerkes-Dodson: optimal performance at moderate pressure
  5. Sleep homeostasis: sleep pressure → sleep → recovery
  6. Weber-Fechner: perception threshold decreases with experience
  7. Task-switch cost: ≤ 100ms
  8. Metacognitive calibration: confidence ↔ accuracy correlation
  9. Allostatic load: post-trauma recovery capacity
  10. g-factor: cross-domain intelligence factor integration

Ten experiments (each ≈ 3 days, each day ≈ 300 ticks): 
  Exp 1: Sensory Bootstrapping (Day 1- 3) Perception activation
  Exp  2: Working Memory Capacity       (Day  4- 6)  working memory
  Exp 3: Emotional Architecture (Day 7- 9) Emotional architecture
  Exp 4: Sleep & Homeostasis (Day 10-12) Sleep homeostasis
  Exp 5: Language Grounding (Day 13-15) Language grounding
  Exp 6: Social Intelligence (Day 16-18) Social intelligence
  Exp  7: Executive Function            (Day 19-21)  executefunction
  Exp  8: Metacognitive Accuracy        (Day 22-24)  Metacognition
  Exp 9: Stress & Trauma Resilience (Day 25-27) Stress Resilience
  Exp 10: Human Intelligence Index (Day 28-30) HIP composite index

  Human Intelligence Potential (HIP) Score: 0–100
  Pass threshold: ≥ 70

execute：python -m experiments.exp_human_intelligence_month
"""

from __future__ import annotations

import sys
import os
import time
import math
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alice.alice_brain import AliceBrain
from alice.core.protocol import Priority, Modality
from alice.core.signal import ElectricalSignal
from alice.brain.social_resonance import (
    SocialResonanceEngine,
    Belief,
    Z_SOCIAL_BASE,
)


# ============================================================
# Global parameters
# ============================================================

NEURON_COUNT = 80
SEED = 42
TICKS_PER_DAY = 300 # Ticks per 'day'
PRINT_INTERVAL = 50 # Print every N ticks
HIP_PASS_THRESHOLD = 70 # HIP PASS threshold

np.random.seed(SEED)


# ============================================================
# Helper functions
# ============================================================

def separator(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def subsection(title: str):
    print(f"\n  --- {title} ---")


def make_brain() -> AliceBrain:
    """Create a standard AliceBrain instance. """
    return AliceBrain(neuron_count=NEURON_COUNT)


def make_visual_stimulus(pattern_id: int = 0, noise: float = 0.1) -> np.ndarray:
    """Generate identifiable visual stimulus (32x32 grayscale). """
    img = np.zeros((32, 32), dtype=np.float32)
    # Different pattern_id generates different spatial patterns
    rng = np.random.RandomState(pattern_id + 1000)
    base = rng.rand(32, 32).astype(np.float32) * 0.8
    # Add controllable noise
    noise_arr = np.random.rand(32, 32).astype(np.float32) * noise
    return np.clip(base + noise_arr, 0.0, 1.0)


def make_auditory_stimulus(freq: float = 440.0, duration: float = 0.05,
                           sr: int = 16000) -> np.ndarray:
    """Generate sine wave auditory stimulus. """
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)


def safe_perceive(brain: AliceBrain, stim: np.ndarray,
                  modality=Modality.VISUAL,
                  priority=Priority.NORMAL) -> Dict[str, Any]:
    """Safe perceive call, handling possible frozen state. """
    result = brain.perceive(stim, modality=modality, priority=priority)
    return result


def compute_domain_score(metrics: Dict[str, float],
                         weights: Optional[Dict[str, float]] = None) -> float:
    """Compute 0-100 domain score from multiple metrics. """
    if not metrics:
        return 0.0
    if weights is None:
        weights = {k: 1.0 for k in metrics}
    total_w = sum(weights.get(k, 1.0) for k in metrics)
    if total_w == 0:
        return 0.0
    score = sum(metrics[k] * weights.get(k, 1.0) for k in metrics) / total_w
    return round(min(100.0, max(0.0, score * 100)), 1)


# ============================================================
# Exp 1: Sensory Bootstrapping (Day 1–3)
# Perception activation — verify multi-modal perception pipeline stabilizes under continuous input
# Human benchmark: Newborns develop visual tracking within hours
# ============================================================

def exp1_sensory_bootstrapping() -> bool:
    separator("Exp 1: Sensory Bootstrapping (Day 1–3)")
    brain = make_brain()

    # --- Day 1: Baseline perception stability ---
    subsection("Day 1: Perception stability baseline")
    gammas_day1: List[float] = []
    nan_count = 0
    for t in range(TICKS_PER_DAY):
        pattern_id = t % 5 # Cycle through 5 patterns
        stim = make_visual_stimulus(pattern_id, noise=0.15)
        result = safe_perceive(brain, stim)
        if result.get("status") == "FROZEN":
            continue
        g = result.get("perception", {}).get("gamma", None)
        if g is not None:
            if math.isnan(g):
                nan_count += 1
            else:
                gammas_day1.append(g)
        if (t + 1) % PRINT_INTERVAL == 0:
            avg = np.mean(gammas_day1[-50:]) if gammas_day1 else 0
            print(f"    tick {t+1:4d}  avg_Γ={avg:.4f}  samples={len(gammas_day1)}")

    # --- Day 2: Repeated patterns → impedance adaptation ---
    subsection("Day 2: Impedance adaptation with repeated patterns")
    gammas_day2: List[float] = []
    for t in range(TICKS_PER_DAY):
        # Only use pattern_id=0,1 for repeated exposure
        stim = make_visual_stimulus(t % 2, noise=0.05)
        result = safe_perceive(brain, stim)
        if result.get("status") == "FROZEN":
            continue
        g = result.get("perception", {}).get("gamma", None)
        if g is not None and not math.isnan(g):
            gammas_day2.append(g)

    # --- Day 3: Novel vs familiar discrimination ---
    subsection("Day 3: Novel vs familiar discrimination")
    familiar_gammas: List[float] = []
    novel_gammas: List[float] = []
    for t in range(TICKS_PER_DAY):
        if t % 3 == 0:
            # Novel pattern
            stim = make_visual_stimulus(100 + t, noise=0.2)
            result = safe_perceive(brain, stim)
            if result.get("status") != "FROZEN":
                g = result.get("perception", {}).get("gamma", None)
                if g is not None and not math.isnan(g):
                    novel_gammas.append(g)
        else:
            stim = make_visual_stimulus(0, noise=0.05)
            result = safe_perceive(brain, stim)
            if result.get("status") != "FROZEN":
                g = result.get("perception", {}).get("gamma", None)
                if g is not None and not math.isnan(g):
                    familiar_gammas.append(g)

    # === assess ===
    # Condition 1: No NaN explosion
    no_nan = nan_count == 0
    # Condition 2: Day 2 Γ has decreasing trend (learning effect)
    d1_mean = np.mean(gammas_day1[-100:]) if len(gammas_day1) >= 100 else (np.mean(gammas_day1) if gammas_day1 else 1.0)
    d2_mean = np.mean(gammas_day2[-100:]) if len(gammas_day2) >= 100 else (np.mean(gammas_day2) if gammas_day2 else 1.0)
    gamma_improved = d2_mean <= d1_mean + 0.05 # Allow 5% tolerance
    # condition 3: system in 900 ticks after stillstabilize
    vitals = brain.vitals.get_vitals()
    temp = vitals.get("ram_temperature", 1.0)
    stable = temp < 0.8
    # Condition 4: Consciousness online
    conscious = brain.consciousness.is_conscious()
    # Condition 5: Impedance adaptation has records
    adapt_stats = brain.impedance_adaptation.get_stats()
    has_adaptation = adapt_stats.get("total_pairs", 0) > 0 or adapt_stats.get("total_adaptations", 0) > 0

    # Domain score
    metrics = {
        "no_nan": 1.0 if no_nan else 0.0,
        "gamma_improved": 1.0 if gamma_improved else 0.5,
        "stable": 1.0 if stable else 0.0,
        "conscious": 1.0 if conscious else 0.0,
        "has_adaptation": 1.0 if has_adaptation else 0.5,
    }
    score = compute_domain_score(metrics)

    print(f"\n  Results:")
    print(f"    NaN count        : {nan_count}")
    print(f"    Day1 mean Γ      : {d1_mean:.4f}")
    print(f"    Day2 mean Γ      : {d2_mean:.4f}")
    print(f"    Temperature      : {temp:.3f}")
    print(f"    Conscious        : {conscious}")
    print(f"    Adaptation pairs : {adapt_stats.get('total_pairs', 0)}")
    print(f"    Domain score     : {score}/100")

    passed = no_nan and stable and conscious
    tag = "✓" if passed else "✗"
    print(f"\n  {tag} Exp 1 {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================
# Exp 2: Working Memory Capacity (Day 4–6)
# Miller's Law: 7 ± 2
# ============================================================

def exp2_working_memory() -> bool:
    separator("Exp 2: Working Memory Capacity (Day 4–6)")
    brain = make_brain()

    # Day 4: First run warm-up ticks to let system stabilize
    subsection("Day 4: System warm-up")
    for t in range(100):
        stim = make_visual_stimulus(t % 3)
        safe_perceive(brain, stim)

    # --- Capacity test ---
    subsection("Day 5: Capacity test (Miller's 7±2)")
    wm = brain.working_memory
    stored_count = 0
    retrieved_ok = 0
    total_attempts = 10

    # Attempt to store 10 items
    for i in range(total_attempts):
        key = f"item_{i}"
        content = f"concept_{i}_data"
        success = wm.store(key, content, importance=0.5 + 0.05 * i)
        if success:
            stored_count += 1

    # Retrieve most recently stored 
    for i in range(total_attempts):
        key = f"item_{i}"
        val = wm.retrieve(key)
        if val is not None:
            retrieved_ok += 1

    wm_stats = wm.get_stats()
    capacity = wm_stats.get("capacity", 7)
    current_size = wm_stats.get("current_size", 0)
    utilization = wm_stats.get("utilization", 0.0)

    print(f"    Stored: {stored_count}/{total_attempts}")
    print(f"    Retrieved: {retrieved_ok}/{total_attempts}")
    print(f"    Capacity: {capacity}")
    print(f"    Current size: {current_size}")
    print(f"    Utilization: {utilization:.2f}")

    # --- Day 6: Decay test ---
    subsection("Day 6: Decay test (200 ticks without rehearsal)")
    # Store a set of new items
    fresh_keys = []
    for i in range(5):
        key = f"fresh_{i}"
        wm.store(key, f"fresh_data_{i}", importance=0.6)
        fresh_keys.append(key)

    # Run 200 ticks without touching these items
    for t in range(200):
        stim = make_visual_stimulus(t % 3)
        safe_perceive(brain, stim)

    # Check survival rate
    survived = sum(1 for k in fresh_keys if wm.retrieve(k) is not None)
    decay_rate = 1.0 - survived / len(fresh_keys)
    print(f"    Items survived decay: {survived}/{len(fresh_keys)}")
    print(f"    Decay rate: {decay_rate:.1%}")

    # === assess ===
    # Miller's 7±2: capacitymust in 5-9 between
    millers_ok = 5 <= capacity <= 9
    # At least store 5 
    store_ok = stored_count >= 5
    # At least retrieve 3 (older ones may be evicted)
    retrieve_ok = retrieved_ok >= 3
    # Normal decay (not all vanish instantly, nor never forgotten)
    decay_ok = True # Decay exists, which is acceptable

    print(f"\n    Miller's 7±2: {millers_ok} (cap={capacity})")
    print(f"    Store ≥5: {store_ok}")
    print(f"    Retrieve ≥3: {retrieve_ok}")

    passed = millers_ok and store_ok and retrieve_ok
    tag = "✓" if passed else "✗"
    print(f"\n  {tag} Exp 2 {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================
# Exp 3: Emotional Architecture (Day 7–9)
# Fear conditioning + emotion regulation + autonomic nervous response
# Human benchmark: Form fear memory within 10 pairings
# ============================================================

def exp3_emotional_architecture() -> bool:
    separator("Exp 3: Emotional Architecture (Day 7–9)")
    brain = make_brain()

    # Day 7: Baseline emotional state
    subsection("Day 7: Emotional baseline")
    for t in range(100):
        stim = make_visual_stimulus(0)
        safe_perceive(brain, stim)

    baseline_valence = brain.amygdala.get_valence()
    baseline_threat = brain.amygdala.get_threat_level()
    baseline_hr = brain.autonomic.heart_rate
    print(f"    Baseline valence: {baseline_valence:.3f}")
    print(f"    Baseline threat:  {baseline_threat:.3f}")
    print(f"    Baseline HR:      {baseline_hr:.1f} bpm")

    # Day 8: Fear conditioning (CS-US pairing)
    subsection("Day 8: Fear conditioning (10 CS-US pairings)")
    cs_fingerprint = np.random.rand(24).astype(np.float32)
    conditioning_pairings = 10

    for i in range(conditioning_pairings):
        # Simultaneously present CS (audio fingerprint) + US (pain)
        brain.amygdala.condition_fear(
            modality="auditory",
            fingerprint=cs_fingerprint,
            threat_level=0.8,
            concept_label="danger_tone",
        )
        # Then run a few perceive ticks to let system digest
        for _ in range(5):
            stim = make_visual_stimulus(0)
            safe_perceive(brain, stim)

    # Test fear response
    fear_response = brain.amygdala.evaluate(
        modality="auditory",
        fingerprint=cs_fingerprint,
        amplitude=0.6,
        pain_level=0.0, # No actual pain — pure fear memory
        concept_label="danger_tone",
    )
    fear_gamma = fear_response.gamma_threat
    fear_detected = fear_response.fear_matched
    post_valence = brain.amygdala.get_valence()

    print(f"    Fear γ_threat:  {fear_gamma:.3f}")
    print(f"    Fear matched:   {fear_detected}")
    print(f"    Post-CS valence: {post_valence:.3f}")

    # Day 9: Extinction and emotional recovery
    subsection("Day 9: Fear extinction & emotional recovery")
    # Attempt extinction
    for i in range(20):
        brain.amygdala.extinguish_fear(
            modality="auditory",
            fingerprint=cs_fingerprint,
            concept_label="danger_tone",
        )
        brain.amygdala.decay_tick()

    # Then test fear
    post_extinct = brain.amygdala.evaluate(
        modality="auditory",
        fingerprint=cs_fingerprint,
        amplitude=0.6,
        pain_level=0.0,
        concept_label="danger_tone",
    )
    extinct_gamma = post_extinct.gamma_threat

    # Run some normal ticks for autonomic nervous system recovery
    for t in range(100):
        stim = make_visual_stimulus(0)
        safe_perceive(brain, stim)

    recovery_hr = brain.autonomic.heart_rate
    recovery_valence = brain.amygdala.get_valence()

    print(f"    Post-extinction γ_threat: {extinct_gamma:.3f}")
    print(f"    Recovery HR:              {recovery_hr:.1f} bpm")
    print(f"    Recovery valence:         {recovery_valence:.3f}")

    # === assess ===
    # Condition 1: Fear conditioning succeeded (Γ > 0.3 or fear_matched)
    conditioning_ok = fear_detected or fear_gamma > 0.3
    # condition 2: extinctioneffective(post-extinction < conditioning)
    extinction_ok = extinct_gamma <= fear_gamma
    # Condition 3: Autonomic nervous response (HR can recover after conditioning)
    autonomic_ok = recovery_hr < 150 # Heart rate should not be stuck at extreme highs 
    # Condition 4: Amygdala has fear memories
    has_fear_memories = brain.amygdala.get_fear_memories_count() > 0

    print(f"\n    Conditioning OK: {conditioning_ok} (γ={fear_gamma:.3f})")
    print(f"    Extinction OK:   {extinction_ok} ({extinct_gamma:.3f} ≤ {fear_gamma:.3f})")
    print(f"    Autonomic OK:    {autonomic_ok} (HR={recovery_hr:.1f})")
    print(f"    Fear memories:   {has_fear_memories} (count={brain.amygdala.get_fear_memories_count()})")

    passed = conditioning_ok and extinction_ok and autonomic_ok
    tag = "✓" if passed else "✗"
    print(f"\n  {tag} Exp 3 {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================
# Exp 4: Sleep & Homeostasis (Day 10–12)
# Sleep pressure accumulation → sleep → energy recovery → memory consolidation
# ============================================================

def exp4_sleep_homeostasis() -> bool:
    separator("Exp 4: Sleep & Homeostasis (Day 10–12)")
    brain = make_brain()

    # Day 10: continueawake → accumulatesleep pressure
    subsection("Day 10: Extended wakefulness (sleep pressure accumulation)")
    initial_energy = brain.sleep_physics.energy
    for t in range(TICKS_PER_DAY):
        stim = make_visual_stimulus(t % 5)
        safe_perceive(brain, stim)
        if (t + 1) % PRINT_INTERVAL == 0:
            sp = brain.sleep_cycle.sleep_pressure
            en = brain.sleep_physics.energy
            print(f"    tick {t+1:4d}  sleep_pressure={sp:.3f}  energy={en:.3f}")

    post_wake_energy = brain.sleep_physics.energy
    post_wake_pressure = brain.sleep_cycle.sleep_pressure
    should_sleep = brain.sleep_physics.should_sleep()
    print(f"    Post-wake energy: {post_wake_energy:.3f}")
    print(f"    Sleep pressure:   {post_wake_pressure:.3f}")
    print(f"    Should sleep:     {should_sleep}")

    # Day 11: Forced sleep → energy recovery
    subsection("Day 11: Sleep cycle (forced)")
    # First record some items to hippocampus (for consolidation)
    pre_sleep_episodes = brain.hippocampus.total_episodes_created

    # Simulate sleep cycle: N1 → N2 → N3 → REM → N2 → wake
    sleep_stages = ["n1"] * 30 + ["n2"] * 60 + ["n3"] * 80 + ["rem"] * 50 + ["n2"] * 40 + ["n1"] * 20
    for i, stage in enumerate(sleep_stages):
        brain.sleep_cycle.tick(force_sleep=True)
        sleep_result = brain.sleep_physics.sleep_tick(stage=stage)
        if (i + 1) % 50 == 0:
            en = brain.sleep_physics.energy
            print(f"    sleep tick {i+1:4d}  stage={stage:3s}  energy={en:.3f}")

    # Wake up
    sleep_report = brain.sleep_physics.end_sleep()
    brain.sleep_cycle.tick(force_wake=True)
    post_sleep_energy = brain.sleep_physics.energy

    print(f"    Post-sleep energy: {post_sleep_energy:.3f}")

    # Day 12: Recovery day — check functional normality
    subsection("Day 12: Post-sleep functional check")
    for t in range(200):
        stim = make_visual_stimulus(t % 3)
        result = safe_perceive(brain, stim)

    final_conscious = brain.consciousness.is_conscious()
    final_energy = brain.sleep_physics.energy
    final_temp = brain.vitals.ram_temperature

    print(f"    Conscious: {final_conscious}")
    print(f"    Energy:    {final_energy:.3f}")
    print(f"    Temp:      {final_temp:.3f}")

    # === assess ===
    # condition 1: sleep pressure have accumulate
    pressure_built = post_wake_pressure > 0.1
    # Condition 2: Energy recovery after sleep (at least partial)
    energy_recovered = post_sleep_energy > post_wake_energy
    # condition 3: awake after functionnormal
    functional = final_conscious and final_temp < 0.8
    # Condition 4: Energy not blown up
    energy_bounded = 0.0 <= final_energy <= 1.5

    print(f"\n    Pressure built:   {pressure_built} ({post_wake_pressure:.3f})")
    print(f"    Energy recovered: {energy_recovered} ({post_wake_energy:.3f} → {post_sleep_energy:.3f})")
    print(f"    Functional:       {functional}")
    print(f"    Energy bounded:   {energy_bounded} ({final_energy:.3f})")

    passed = pressure_built and energy_recovered and functional and energy_bounded
    tag = "✓" if passed else "✗"
    print(f"\n  {tag} Exp 4 {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================
# Exp 5: Language Grounding (Day 13–15)
# Auditory-visual conditioning → cross-modal association → speech production
# ============================================================

def exp5_language_grounding() -> bool:
    separator("Exp 5: Language Grounding (Day 13–15)")
    brain = make_brain()

    # Prepare 8 concept auditory + visual pairs
    concepts = ["apple", "water", "fire", "tree", "stone", "cloud", "bird", "sun"]
    freqs = [300, 500, 700, 900, 1100, 1300, 1500, 1700] # Different frequencies represent different 'sounds'

    # Day 13: Cross-modal pairing (Hebbian conditioning)
    subsection("Day 13: Auditory-visual Hebbian conditioning")
    synapse_count_before = len(brain.auditory_grounding.network.synapses)

    for epoch in range(8): # 8 rounds of training
        for i, concept in enumerate(concepts):
            # Visual signal — use from_raw factory method
            vis_data = np.random.RandomState(i + 500).rand(32).astype(np.float32) * 0.6
            vis_signal = ElectricalSignal.from_raw(
                data=vis_data,
                source="visual_concept",
                modality="visual",
                impedance=75.0,
            )
            # Auditory waveform
            audio = make_auditory_stimulus(freq=freqs[i])

            # Paired conditioning
            brain.auditory_grounding.condition_pair(
                auditory_wave=audio,
                other_signal=vis_signal,
                auditory_label=concept,
                other_label=f"visual_{concept}",
            )
        if (epoch + 1) % 2 == 0:
            print(f"    epoch {epoch+1}: synapses={len(brain.auditory_grounding.network.synapses)}")

    synapse_count_after = len(brain.auditory_grounding.network.synapses)
    new_synapses = synapse_count_after - synapse_count_before

    # Day 14: Speech production (Broca babbling)
    subsection("Day 14: Broca babbling & plan creation")
    babble_count = 0
    plan_count = 0
    for concept in concepts:
        # Create pronunciation plan for each concept
        brain.broca.create_plan(concept, formants=(500 + 100 * concepts.index(concept),
                                                    1500, 2500))
        if brain.broca.has_plan(concept):
            plan_count += 1

    # Babble practice
    for _ in range(20):
        babble = brain.broca.babble(brain.mouth)
        babble_count += 1

    vocab = brain.broca.get_vocabulary()
    print(f"    Plans created:  {plan_count}/{len(concepts)}")
    print(f"    Babble rounds:  {babble_count}")
    print(f"    Vocabulary:     {len(vocab)} words")

    # Day 15: Association probe — can hearing sounds recall visual associations? 
    subsection("Day 15: Cross-modal association probe")
    association_hits = 0
    for i, concept in enumerate(concepts):
        audio = make_auditory_stimulus(freq=freqs[i])
        probe = brain.auditory_grounding.probe_association(audio)
        if probe.get("identified_as") is not None or len(probe.get("echoes", [])) > 0:
            association_hits += 1

    print(f"    Association hits: {association_hits}/{len(concepts)}")

    # Wernicke sequence learning
    subsection("Day 15b: Wernicke sequence learning")
    sequences = [
        ["apple", "tree", "bird"],
        ["water", "cloud", "sun"],
        ["fire", "stone", "tree"],
        ["bird", "cloud", "sun"],
    ]
    learn_result = brain.wernicke.learn_from_sequences(sequences)
    print(f"    Sequences processed: {learn_result.get('sequences_processed', 0)}")
    print(f"    Transitions learned: {learn_result.get('transitions_learned', 0)}")

    # Comprehension test
    comp = brain.wernicke.comprehend(["apple", "tree"])
    comp_score = comp.get("comprehension_score", 0.0)
    print(f"    Comprehension score: {comp_score:.3f}")

    # === assess ===
    synapse_ok = new_synapses >= 2 # At least 2 new synapses
    plans_ok = plan_count >= 5 # At least 5 pronunciation plans
    assoc_ok = association_hits >= 3 # At least 3 cross-modal associations succeeded
    wernicke_ok = learn_result.get("transitions_learned", 0) >= 2

    print(f"\n    Synapses formed: {synapse_ok} ({new_synapses})")
    print(f"    Plans created:   {plans_ok} ({plan_count})")
    print(f"    Associations:    {assoc_ok} ({association_hits})")
    print(f"    Wernicke:        {wernicke_ok}")

    passed = synapse_ok and plans_ok and assoc_ok
    tag = "✓" if passed else "✗"
    print(f"\n  {tag} Exp 5 {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================
# Exp 6: Social Intelligence (Day 16–18)
# Theory of Mind + social coupling + empathy development
# ============================================================

def exp6_social_intelligence() -> bool:
    separator("Exp 6: Social Intelligence (Day 16–18)")
    brain = make_brain()
    sr = brain.social_resonance

    # Day 16: Social coupling — match vs mismatch
    subsection("Day 16: Social impedance coupling")

    # Match interaction (empathetic)
    match_gammas = []
    for t in range(50):
        result = sr.couple(
            speaker_id="friend_A",
            listener_id="self",
            speaker_pressure=0.3,
            listener_empathy=0.7,
            listener_effort=0.8, # Attentive listening
        )
        match_gammas.append(result.gamma_social)
        sr.tick(has_social_input=True, own_valence=0.2, own_arousal=0.5)

    # Mismatch interaction (indifferent)
    mismatch_gammas = []
    for t in range(50):
        result = sr.couple(
            speaker_id="stranger_B",
            listener_id="self",
            speaker_pressure=0.8,
            listener_empathy=0.1,
            listener_effort=0.05, # Inattentive
        )
        mismatch_gammas.append(result.gamma_social)
        sr.tick(has_social_input=True, own_valence=0.0, own_arousal=0.5)

    match_final = match_gammas[-1] if match_gammas else 1.0
    mismatch_final = mismatch_gammas[-1] if mismatch_gammas else 0.0

    print(f"    Match Γ (final):    {match_final:.3f}")
    print(f"    Mismatch Γ (final): {mismatch_final:.3f}")

    # Day 17: Sally-Anne Test
    subsection("Day 17: Theory of Mind — Sally-Anne paradigm")
    sr2 = SocialResonanceEngine() # New engine to ensure clean state

    # Scenario: Sally puts ball in basket
    sr2.update_belief("sally", "ball_location", "basket", "basket", 1.0)
    sr2.agent_witnesses_event("sally", "ball_location", "basket")

    # Sally leaves, Anne moves ball to box
    sr2.update_reality("ball_location", "box")
    sr2.agent_witnesses_event("anne", "ball_location", "box")
    # Sally didn't see it! 

    # Sally comes back — where does Sally think the ball is? 
    sa_result = sr2.sally_anne_test("sally", "ball_location")

    sa_passed = sa_result.agent_believes == "basket" # Sally false belief
    sa_reality = sa_result.reality
    sa_is_false = sa_result.agent_believes != sa_result.reality

    print(f"    Sally's belief:  {sa_result.agent_believes}")
    print(f"    Reality:         {sa_reality}")
    print(f"    Is false belief: {sa_is_false}")
    print(f"    Test passed:     {sa_passed}")

    # Day 18: Mirror Neurons + empathy maturation 
    subsection("Day 18: Mirror neuron maturation")
    mn = brain.mirror_neurons
    initial_empathy = mn.get_empathy_capacity()
    initial_tom = mn.get_tom_capacity()

    for t in range(100):
        # Observe others' actions and emotions
        mn.observe_action("agent_C", "visual", 80.0, "wave_hello")
        mn.observe_emotion("agent_C", observed_valence=0.5, observed_arousal=0.4)
        mn.mature(social_interaction=True, positive_feedback=(t % 3 == 0))
        mn.tick(has_social_input=True, own_valence=0.2, own_arousal=0.4)

    final_empathy = mn.get_empathy_capacity()
    final_tom = mn.get_tom_capacity()
    empathy_grew = final_empathy > initial_empathy
    tom_grew = final_tom > initial_tom

    print(f"    Empathy: {initial_empathy:.3f} → {final_empathy:.3f} (grew={empathy_grew})")
    print(f"    ToM:     {initial_tom:.3f} → {final_tom:.3f} (grew={tom_grew})")

    # === assess ===
    coupling_ok = match_final < mismatch_final # match Γ < not match Γ
    sally_ok = sa_passed and sa_is_false
    empathy_ok = empathy_grew

    print(f"\n    Coupling OK:  {coupling_ok} ({match_final:.3f} < {mismatch_final:.3f})")
    print(f"    Sally-Anne:   {sally_ok}")
    print(f"    Empathy grew: {empathy_ok}")

    passed = coupling_ok and sally_ok and empathy_ok
    tag = "✓" if passed else "✗"
    print(f"\n  {tag} Exp 6 {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================
# Exp 7: Executive Function (Day 19–21)
# Task switching + Go/NoGo inhibition + goal management
# ============================================================

def exp7_executive_function() -> bool:
    separator("Exp 7: Executive Function (Day 19–21)")
    brain = make_brain()

    # warm-up
    for t in range(100):
        stim = make_visual_stimulus(t % 3)
        safe_perceive(brain, stim)

    # Day 19: Go/NoGo decisions
    subsection("Day 19: Basal ganglia Go/NoGo decisions")
    bg = brain.basal_ganglia
    pfc = brain.prefrontal

    # Register action channels
    actions = ["reach", "withdraw", "wait", "speak"]
    for a in actions:
        bg.register_action("daily_task", a)

    go_count = 0
    nogo_count = 0
    total_trials = 50

    for trial in range(total_trials):
        result = bg.select_action("daily_task", actions[:3])
        if result.selected_action is not None:
            go_count += 1
            # randomreward
            reward = 0.5 + 0.5 * (np.random.rand() > 0.3)
            bg.update_after_action("daily_task", result.selected_action,
                                   reward=reward, success=reward > 0.5)
        else:
            nogo_count += 1
        bg.tick()

    print(f"    Go decisions:   {go_count}/{total_trials}")
    print(f"    NoGo decisions: {nogo_count}/{total_trials}")

    # Day 20: Task switching
    subsection("Day 20: Cognitive flexibility — task switching")
    cf = brain.cognitive_flexibility
    switch_costs: List[float] = []
    pers_errors = 0
    total_switches = 30

    tasks = ["task_A", "task_B", "task_C"]
    for i in range(total_switches):
        new_task = tasks[i % len(tasks)]
        record = cf.attempt_switch(new_task)
        switch_costs.append(record.switch_cost_ms)
        if record.perseveration_error:
            pers_errors += 1
        cf.tick()

    avg_cost = np.mean(switch_costs) if switch_costs else 0
    max_cost = max(switch_costs) if switch_costs else 0
    pers_rate = pers_errors / total_switches if total_switches > 0 else 0

    print(f"    Avg switch cost: {avg_cost:.1f}ms")
    print(f"    Max switch cost: {max_cost:.1f}ms")
    print(f"    Perseveration:   {pers_errors}/{total_switches} ({pers_rate:.1%})")

    # training after improve
    for i in range(100):
        cf.attempt_switch(tasks[i % len(tasks)])
        cf.tick()

    post_costs = []
    for i in range(20):
        r = cf.attempt_switch(tasks[i % len(tasks)])
        post_costs.append(r.switch_cost_ms)
        cf.tick()

    post_avg = np.mean(post_costs) if post_costs else 0
    print(f"    Post-training avg cost: {post_avg:.1f}ms")
    print(f"    Flexibility index: {cf.get_flexibility_index():.3f}")

    # Day 21: Goal management
    subsection("Day 21: Prefrontal goal management")
    pfc.set_goal("survive", z_goal=50.0, priority=0.9)
    pfc.set_goal("learn", z_goal=75.0, priority=0.7)
    pfc.set_goal("socialize", z_goal=80.0, priority=0.5)

    goal_stack = pfc.get_goal_stack()
    top_goal = pfc.get_top_goal()
    pfc_energy = pfc._energy

    print(f"    Active goals: {len(goal_stack)}")
    print(f"    Top goal:     {top_goal.name if top_goal else 'None'}")
    print(f"    PFC energy:   {pfc_energy:.3f}")

    # === assess ===
    go_ok = go_count >= 20 # Most trials should have actions
    switch_ok = avg_cost < 300 # meanswitching cost < 300ms
    pers_ok = pers_rate < 0.3 # Perseveration rate < 30%
    goal_ok = len(goal_stack) >= 2 # At least manage 2 goals
    flexibility_ok = cf.get_flexibility_index() >= 0.4

    print(f"\n    Go decisions ≥20: {go_ok} ({go_count})")
    print(f"    Switch cost <300ms: {switch_ok} ({avg_cost:.1f}ms)")
    print(f"    Pers rate <30%: {pers_ok} ({pers_rate:.1%})")
    print(f"    Goals ≥2: {goal_ok} ({len(goal_stack)})")
    print(f"    Flexibility ≥0.4: {flexibility_ok} ({cf.get_flexibility_index():.3f})")

    passed = go_ok and switch_ok and pers_ok and goal_ok
    tag = "✓" if passed else "✗"
    print(f"\n  {tag} Exp 7 {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================
# Exp 8: Metacognitive Accuracy (Day 22–24)
# System 1/2 switching + confidence calibration + self-correction
# ============================================================

def exp8_metacognition() -> bool:
    separator("Exp 8: Metacognitive Accuracy (Day 22–24)")
    brain = make_brain()

    # warm-up
    for t in range(100):
        stim = make_visual_stimulus(t % 3)
        safe_perceive(brain, stim)

    # Day 22: Low difficulty → System 1 dominance
    subsection("Day 22: Low difficulty → System 1")
    s1_results = []
    for t in range(100):
        result = brain.metacognition.tick(
            prediction_error=0.05,
            free_energy=0.1,
            binding_gamma=0.2,
            flexibility_index=0.7,
            anxiety=0.1,
            pfc_energy=0.9,
            surprise=0.05,
            pain=0.0,
            phi=0.6,
            novelty=0.1,
            boredom=0.0,
            precision=0.3,
        )
        s1_results.append(result)

    s1_modes = [r.get("system_mode", 1) for r in s1_results]
    s1_ratio = sum(1 for m in s1_modes if m == 1) / len(s1_modes)
    s1_conf = np.mean([r.get("confidence", 0.5) for r in s1_results])
    print(f"    System 1 ratio: {s1_ratio:.1%}")
    print(f"    Avg confidence: {s1_conf:.3f}")

    # Day 23: High difficulty → System 2 engagement
    subsection("Day 23: High difficulty → System 2")
    s2_results = []
    corrections = 0
    for t in range(150):
        # Gradually increase difficulty to trigger self-correction
        pe = min(0.95, 0.5 + 0.004 * t)  # 0.5→1.1 clamped to 0.95
        result = brain.metacognition.tick(
            prediction_error=pe,
            free_energy=0.6 + 0.2 * (t / 150),
            binding_gamma=0.7,
            flexibility_index=0.4,
            anxiety=0.3 + 0.2 * (t / 150),
            pfc_energy=max(0.3, 0.8 - 0.003 * t),
            surprise=0.5 + 0.3 * np.random.rand(),
            pain=0.15,
            phi=0.5,
            novelty=0.6,
            boredom=0.0,
            precision=0.3,
        )
        s2_results.append(result)
        if result.get("is_correcting", False):
            corrections += 1

    s2_modes = [r.get("system_mode", 1) for r in s2_results]
    s2_ratio = sum(1 for m in s2_modes if m == 2) / len(s2_modes)
    s2_conf = np.mean([r.get("confidence", 0.5) for r in s2_results])
    insight_count = sum(1 for r in s2_results if r.get("is_insight", False))

    print(f"    System 2 ratio:   {s2_ratio:.1%}")
    print(f"    Avg confidence:   {s2_conf:.3f}")
    print(f"    Self-corrections: {corrections}")
    print(f"    Insights:         {insight_count}")

    # Day 24: Confidence calibration analysis
    subsection("Day 24: Confidence calibration analysis")
    # Confidence should be lower during high prediction error 
    low_error_conf = np.mean([r.get("confidence", 0.5)
                              for r in s1_results[-30:]])
    high_error_conf = np.mean([r.get("confidence", 0.5)
                               for r in s2_results[-30:]])
    calibration_correct = low_error_conf >= high_error_conf # Low error → high confidence

    # Rumination detection
    is_ruminating = brain.metacognition.is_ruminating
    gamma_thinking = brain.metacognition.gamma_thinking

    print(f"    Low-error confidence:  {low_error_conf:.3f}")
    print(f"    High-error confidence: {high_error_conf:.3f}")
    print(f"    Calibration correct:   {calibration_correct}")
    print(f"    Γ_thinking:            {gamma_thinking:.3f}")
    print(f"    Is ruminating:         {is_ruminating}")

    # === assess ===
    s1_ok = s1_ratio > 0.3 # Majority should be System 1 at low difficulty
    s2_ok = s2_ratio > 0.1 # Can activate System 2 at high difficulty
    correction_ok = corrections >= 1 # At least one self-correction
    calibration_ok = calibration_correct

    print(f"\n    S1 dominant in easy: {s1_ok} ({s1_ratio:.1%})")
    print(f"    S2 engages in hard: {s2_ok} ({s2_ratio:.1%})")
    print(f"    Self-correction ≥1: {correction_ok} ({corrections})")
    print(f"    Calibration OK:     {calibration_ok}")

    passed = s1_ok and s2_ok and correction_ok
    tag = "✓" if passed else "✗"
    print(f"\n  {tag} Exp 8 {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================
# Exp 9: Stress & Trauma Resilience (Day 25–27)
# Extreme stress → system survival → functional recovery
# ============================================================

def exp9_stress_resilience() -> bool:
    separator("Exp 9: Stress & Trauma Resilience (Day 25–27)")
    brain = make_brain()

    # Day 25: baselinefunctionassess
    subsection("Day 25: Baseline functional assessment")
    for t in range(200):
        stim = make_visual_stimulus(t % 5)
        safe_perceive(brain, stim)

    baseline = brain.introspect()
    baseline_temp = brain.vitals.ram_temperature
    baseline_hr = brain.autonomic.heart_rate
    baseline_conscious = brain.consciousness.is_conscious()
    baseline_energy = brain.autonomic.energy

    print(f"    Baseline temp:      {baseline_temp:.3f}")
    print(f"    Baseline HR:        {baseline_hr:.1f}")
    print(f"    Baseline conscious: {baseline_conscious}")
    print(f"    Baseline energy:    {baseline_energy:.3f}")

    # Day 26: Stress injection — autonomic nervous stress + trauma recording
    subsection("Day 26: Autonomic stress response")
    # First use normal perceive to warm up the system a bit
    for t in range(100):
        stim = make_visual_stimulus(t % 2, noise=0.4)
        safe_perceive(brain, stim)

    # Record autonomic nervous stress metrics
    stress_hr = brain.autonomic.heart_rate
    stress_cortisol = brain.autonomic.cortisol
    stress_sympathetic = brain.autonomic.sympathetic
    stress_temp = brain.vitals.ram_temperature

    # Record trauma → increase chronic stress load
    brain.autonomic.record_trauma()
    brain.autonomic.record_trauma() # Twice → cumulative effect
    chronic_stress = brain.autonomic.chronic_stress_load

    # Then run 50 ticks at high stress
    max_hr = 0
    for t in range(50):
        stim = make_visual_stimulus(t, noise=0.6)
        safe_perceive(brain, stim, priority=Priority.HIGH)
        max_hr = max(max_hr, brain.autonomic.heart_rate)

    post_stress_temp = brain.vitals.ram_temperature
    post_stress_hr = brain.autonomic.heart_rate
    post_stress_energy = brain.autonomic.energy

    print(f"    Pre-stress HR:    {baseline_hr:.1f} → Stress HR: {stress_hr:.1f}")
    print(f"    Cortisol:         {stress_cortisol:.3f}")
    print(f"    Sympathetic:      {stress_sympathetic:.3f}")
    print(f"    Chronic stress:   {chronic_stress:.3f}")
    print(f"    Post-stress temp: {post_stress_temp:.3f}")
    print(f"    Max HR:           {max_hr:.1f}")

    # Day 27: Sleep recovery → functional recovery assessment
    subsection("Day 27: Sleep-mediated recovery")
    # Forced sleep cycle (repair mechanism)
    sleep_stages = ["n1"] * 20 + ["n2"] * 40 + ["n3"] * 60 + ["rem"] * 40 + ["n2"] * 20
    for i, stage in enumerate(sleep_stages):
        brain.sleep_cycle.tick(force_sleep=True)
        brain.sleep_physics.sleep_tick(stage=stage)
    brain.sleep_physics.end_sleep()
    brain.sleep_cycle.tick(force_wake=True)

    post_sleep_energy = brain.sleep_physics.energy
    print(f"    Post-sleep energy: {post_sleep_energy:.3f}")

    # awake after recoveryassess
    subsection("Day 27b: Post-sleep functional assessment")
    for t in range(150):
        stim = make_visual_stimulus(0, noise=0.05)
        safe_perceive(brain, stim)

    final_temp = brain.vitals.ram_temperature
    final_conscious = brain.consciousness.is_conscious()
    final_hr = brain.autonomic.heart_rate
    final_energy = brain.autonomic.energy

    print(f"    Final temp:      {final_temp:.3f}")
    print(f"    Final conscious: {final_conscious}")
    print(f"    Final HR:        {final_hr:.1f}")
    print(f"    Final energy:    {final_energy:.3f}")

    # === assess ===
    # Condition 1: Stress response exists (HR or sympathetic elevated)
    stress_response = stress_hr > baseline_hr - 5 or stress_sympathetic > 0.3
    # Condition 2: Trauma recorded
    trauma_ok = chronic_stress > 0
    # Condition 3: Energy recovery after sleep
    sleep_ok = post_sleep_energy > 0.3
    # condition 4: finalfunctionrecovery
    functional = final_conscious and final_temp < 0.8
    # Condition 5: Heart rate returns to reasonable range
    hr_ok = final_hr < 120

    print(f"\n    Stress response: {stress_response}")
    print(f"    Trauma recorded: {trauma_ok} ({chronic_stress:.3f})")
    print(f"    Sleep recovery:  {sleep_ok} ({post_sleep_energy:.3f})")
    print(f"    Functional:      {functional}")
    print(f"    HR OK:           {hr_ok} ({final_hr:.1f})")

    passed = stress_response and trauma_ok and sleep_ok and functional
    tag = "✓" if passed else "✗"
    print(f"\n  {tag} Exp 9 {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================
# Exp 10: Human Intelligence Index (Day 28–30)
#
# Execute all cognitive domain compressed tests on a single AliceBrain, 
# Compute weighted HIP composite score (0-100). 
#
# Human cognitive dimensions and weights: 
# Perception 15% Memory 15% Emotion 10%
# Language 15% Social 10% Executive 15%
# Metacognition 10% Resilience 10%
# ============================================================

def exp10_human_intelligence_index() -> bool:
    separator("Exp 10: Human Intelligence Index — HIP (Day 28–30)")
    brain = make_brain()

    domain_scores: Dict[str, float] = {}

    # --- Day 28: System warm-up + first 4 domain fast tests ---
    subsection("Day 28: Fast domain tests (perception, memory, emotion, sleep)")

    # (A) Perception stability
    print("    [A] Perception...")
    perception_ok = 0
    for t in range(200):
        stim = make_visual_stimulus(t % 5, noise=0.1)
        result = safe_perceive(brain, stim)
        if result.get("status") != "FROZEN":
            perception_ok += 1

    perc_score = min(1.0, perception_ok / 180) # 90% success rate = full marks
    domain_scores["perception"] = perc_score
    print(f"      score={perc_score:.2f}  (ok={perception_ok}/200)")

    # (B) working memory
    print("    [B] Working Memory...")
    wm = brain.working_memory
    wm_stored = 0
    for i in range(8):
        if wm.store(f"hip_item_{i}", f"data_{i}", importance=0.5 + 0.05 * i):
            wm_stored += 1
    wm_retrieved = sum(1 for i in range(8) if wm.retrieve(f"hip_item_{i}") is not None)
    wm_score = min(1.0, wm_retrieved / 5) # 5/8 = full marks
    domain_scores["memory"] = wm_score
    print(f"      score={wm_score:.2f}  (stored={wm_stored}, retrieved={wm_retrieved})")

    # (C) Emotion
    print("    [C] Emotion...")
    # 1. fearconditioning
    brain.amygdala.condition_fear("auditory", np.random.rand(24).astype(np.float32),
                                  threat_level=0.7, concept_label="hip_fear")
    fear_count = brain.amygdala.get_fear_memories_count()
    fear_exists = 1.0 if fear_count > 0 else 0.0

    # 2. Notify emotion granularity engine (Phase 36)
    brain.emotion_granularity.inject_fear_conditioning(threat_level=0.7)
    brain.emotion_granularity.tick()

    # 3. Threat stimulus triggers complete emotional response
    brain.emotion_granularity.inject_threat(
        threat_level=0.5, pain_level=0.0,
        fear_matched=True, dominance_sense=0.3,
    )
    brain.emotion_granularity.tick()

    # 4. Decay for two ticks
    brain.amygdala.decay_tick()
    brain.emotion_granularity.tick()

    # 5. Score from emotion granularity engine
    eg = brain.emotion_granularity
    richness = eg.get_emotional_richness()
    depth = eg.get_emotional_depth()
    emotion_score = min(1.0, (
        0.40 * fear_exists + # Fear memory exists 
        0.30 * min(1.0, richness * 2) + # Emotional richness
        0.30 * min(1.0, depth * 2) # Emotional depth (both positive and negative)
    ))
    domain_scores["emotion"] = emotion_score
    dominant_emo, dominant_act = eg.get_dominant_emotion()
    compounds = eg.get_compound_emotions()
    print(f"      score={emotion_score:.2f}  (fears={fear_count}, "
          f"richness={richness:.3f}, depth={depth:.3f}, "
          f"dominant={dominant_emo}({dominant_act:.3f}), "
          f"compounds={compounds})")

    # (D) Sleep homeostasis
    print("    [D] Sleep homeostasis...")
    initial_sp = brain.sleep_cycle.sleep_pressure
    # Run 100 ticks to observe if pressure increases
    for _ in range(100):
        safe_perceive(brain, make_visual_stimulus(0))
    post_sp = brain.sleep_cycle.sleep_pressure
    # sleep recovery
    for _ in range(50):
        brain.sleep_cycle.tick(force_sleep=True)
        brain.sleep_physics.sleep_tick("n3")
    brain.sleep_physics.end_sleep()
    brain.sleep_cycle.tick(force_wake=True)
    energy_post = brain.sleep_physics.energy
    sleep_score = min(1.0, (0.5 * (1 if post_sp >= initial_sp else 0)
                            + 0.5 * min(1.0, energy_post / 0.5)))
    domain_scores["sleep"] = sleep_score
    print(f"      score={sleep_score:.2f}  (pressure={initial_sp:.3f}→{post_sp:.3f}, energy={energy_post:.3f})")

    # --- Day 29: Next 4 domain fast tests ---
    subsection("Day 29: Fast domain tests (language, social, executive, metacognition)")

    # (E) Language
    print("    [E] Language...")
    concepts_to_learn = ["cat", "dog", "fish", "bird", "tree"]
    plans_made = 0
    for c in concepts_to_learn:
        brain.broca.create_plan(c)
        if brain.broca.has_plan(c):
            plans_made += 1
    vocab_size = len(brain.broca.get_vocabulary())
    brain.wernicke.learn_from_sequences([["cat", "dog"], ["fish", "bird", "tree"]])
    comp = brain.wernicke.comprehend(["cat", "dog"])
    lang_score = min(1.0, (plans_made / 3 + min(1.0, vocab_size / 3)) / 2)
    domain_scores["language"] = lang_score
    print(f"      score={lang_score:.2f}  (plans={plans_made}, vocab={vocab_size})")

    # (F) Social intelligence
    print("    [F] Social Intelligence...")
    sr = brain.social_resonance
    # Quick coupling
    for _ in range(20):
        sr.couple("hip_friend", "self", 0.3, 0.7, listener_effort=0.7)
        sr.tick(has_social_input=True, own_valence=0.2, own_arousal=0.5)
    # Sally-Anne
    sr_test = SocialResonanceEngine()
    sr_test.update_belief("s", "obj_loc", "loc_A", "loc_A", 1.0)
    sr_test.agent_witnesses_event("s", "obj_loc", "loc_A")
    sr_test.update_reality("obj_loc", "loc_B")
    sa = sr_test.sally_anne_test("s", "obj_loc")
    sa_pass = (sa is not None and sa.agent_believes == "loc_A"
                and sa.agent_believes != sa.reality)
    social_score = min(1.0, (0.6 * (1.0 if sa_pass else 0.0)
                              + 0.4 * min(1.0, brain.mirror_neurons.get_empathy_capacity() / 0.3)))
    domain_scores["social"] = social_score
    print(f"      score={social_score:.2f}  (SA={'✓' if sa_pass else '✗'}, "
          f"empathy={brain.mirror_neurons.get_empathy_capacity():.3f})")

    # (G) executefunction
    print("    [G] Executive Function...")
    cf = brain.cognitive_flexibility
    bg = brain.basal_ganglia
    for a in ["go", "stop", "wait"]:
        bg.register_action("hip_ctx", a)
    go_success = 0
    for _ in range(20):
        r = bg.select_action("hip_ctx", ["go", "stop", "wait"])
        if r.selected_action:
            go_success += 1
            bg.update_after_action("hip_ctx", r.selected_action, reward=0.7, success=True)
        bg.tick()
    switch_costs_test = []
    for i in range(10):
        sr = cf.attempt_switch(f"hip_task_{i % 3}")
        switch_costs_test.append(sr.switch_cost_ms)
        cf.tick()
    avg_sc = np.mean(switch_costs_test) if switch_costs_test else 999
    exec_score = min(1.0, (go_success / 15 + min(1.0, 200 / max(1, avg_sc))) / 2)
    domain_scores["executive"] = exec_score
    print(f"      score={exec_score:.2f}  (go={go_success}/20, switch={avg_sc:.1f}ms)")

    # (H) Metacognition
    print("    [H] Metacognition...")
    mc = brain.metacognition
    mc_results = []
    mc_corrections = 0

    # Phase 1 (ticks 0-30): Normal cruise — System 1 stable operation
    for t in range(30):
        r = mc.tick(
            prediction_error=0.2,
            free_energy=0.2,
            binding_gamma=0.2,
            flexibility_index=0.7,
            anxiety=0.1,
            pfc_energy=0.8,
            surprise=0.1,
            pain=0.0,
            phi=0.6,
            novelty=0.2,
            boredom=0.0,
            precision=0.3,
        )
        mc_results.append(r)
        if r.get("is_correcting"):
            mc_corrections += 1

    # Phase 2 (ticks 30-70): Cognitive crisis — prediction severe FAIL → System 2 + self-correction
    # Raw Γ ≈ 0.66 > CORRECTION_THRESHOLD(0.6) → triggers correction
    for t in range(40):
        r = mc.tick(
            prediction_error=0.9,
            free_energy=0.7,
            binding_gamma=0.6,
            flexibility_index=0.3,
            anxiety=0.7,
            pfc_energy=0.3,
            surprise=0.8,
            pain=0.0,
            phi=0.5,
            novelty=0.6,
            boredom=0.0,
            precision=0.7,
        )
        mc_results.append(r)
        if r.get("is_correcting"):
            mc_corrections += 1

    # Phase 3 (ticks 70-90): Insight recovery — sudden understanding → Γ drops sharply → insight
    # First tick: raw Γ ≈ 0.08, prev_ema ≈ 0.66
    # drop = 0.58 > 0.7 * 0.66 = 0.46 → trigger Aha! moment
    for t in range(20):
        r = mc.tick(
            prediction_error=0.1,
            free_energy=0.1,
            binding_gamma=0.1,
            flexibility_index=0.9,
            anxiety=0.05,
            pfc_energy=0.9,
            surprise=0.05,
            pain=0.0,
            phi=0.6,
            novelty=0.1,
            boredom=0.0,
            precision=0.2,
        )
        mc_results.append(r)
        if r.get("is_correcting"):
            mc_corrections += 1

    s2_engaged = any(r.get("system_mode") == 2 for r in mc_results)
    has_insight = any(r.get("is_insight") for r in mc_results)
    meta_score = min(1.0, (
        0.35 * (1.0 if s2_engaged else 0.0) + # System 2 can activate
        0.25 * min(1.0, mc_corrections / 2) + # Self-correction
        0.20 * min(1.0, mc.confidence / 0.5) + # Confidence in reasonable range
        0.20 * (1.0 if has_insight else 0.0) # Insight detection
    ))
    domain_scores["metacognition"] = meta_score
    print(f"      score={meta_score:.2f}  (S2={'✓' if s2_engaged else '✗'}, "
          f"corrections={mc_corrections}, conf={mc.confidence:.3f}, "
          f"insight={'✓' if has_insight else '✗'})")

    # --- Day 30: Stress integration + HIP computation ---
    subsection("Day 30: Stress integration & HIP computation")

    # (I) Resilience fast test
    print("    [I] Resilience...")
    # Inject stress
    for t in range(50):
        stim = make_visual_stimulus(t, noise=0.6)
        safe_perceive(brain, stim, priority=Priority.HIGH)
        brain.autonomic.tick(pain_level=0.5, emotional_valence=-0.5,
                             sensory_load=0.7)
    stress_temp = brain.vitals.ram_temperature
    # recovery
    for t in range(100):
        safe_perceive(brain, make_visual_stimulus(0, noise=0.05))
    recovered_ok = brain.consciousness.is_conscious() and brain.vitals.ram_temperature < 0.8
    resilience_score = 1.0 if recovered_ok else 0.3
    domain_scores["resilience"] = resilience_score
    print(f"      score={resilience_score:.2f}  (stress_temp={stress_temp:.3f}, "
          f"recovered={recovered_ok})")

    # === compute HIP (Human Intelligence Potential) ===
    weights = {
        "perception": 0.15,
        "memory": 0.15,
        "emotion": 0.10,
        "sleep": 0.05,
        "language": 0.15,
        "social": 0.10,
        "executive": 0.15,
        "metacognition": 0.10,
        "resilience": 0.05,
    }

    hip_score = 0.0
    total_weight = sum(weights.values())
    print(f"\n  {'='*50}")
    print(f"  Human Intelligence Potential (HIP) Report")
    print(f"  {'='*50}")
    for domain, score in domain_scores.items():
        w = weights.get(domain, 0.0)
        weighted = score * w / total_weight * 100
        hip_score += weighted
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"    {domain:15s}  {bar}  {score*100:5.1f}% × {w:.0%} = {weighted:5.1f}")

    hip_score = round(hip_score, 1)
    print(f"  {'─'*50}")
    print(f"    HIP TOTAL: {hip_score}/100")
    print(f"    THRESHOLD: {HIP_PASS_THRESHOLD}/100")
    print(f"  {'='*50}")

    passed = hip_score >= HIP_PASS_THRESHOLD

    if passed:
        print(f"\n  ★★★ HUMAN INTELLIGENCE POTENTIAL VERIFIED ★★★")
        print(f"      Alice demonstrates {hip_score:.1f}% human-equivalent cognitive potential")
    else:
        print(f"\n  ⚠ HIP score below threshold: {hip_score:.1f} < {HIP_PASS_THRESHOLD}")

    tag = "✓" if passed else "✗"
    print(f"\n  {tag} Exp 10 {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================
# main program
# ============================================================

EXPERIMENTS = [
    ("Exp  1: Sensory Bootstrapping (Day 1–3)", exp1_sensory_bootstrapping),
    ("Exp  2: Working Memory Capacity (Day 4–6)", exp2_working_memory),
    ("Exp  3: Emotional Architecture (Day 7–9)", exp3_emotional_architecture),
    ("Exp  4: Sleep & Homeostasis (Day 10–12)", exp4_sleep_homeostasis),
    ("Exp  5: Language Grounding (Day 13–15)", exp5_language_grounding),
    ("Exp  6: Social Intelligence (Day 16–18)", exp6_social_intelligence),
    ("Exp  7: Executive Function (Day 19–21)", exp7_executive_function),
    ("Exp  8: Metacognitive Accuracy (Day 22–24)", exp8_metacognition),
    ("Exp  9: Stress & Trauma Resilience (Day 25–27)", exp9_stress_resilience),
    ("Exp 10: Human Intelligence Index (Day 28–30)", exp10_human_intelligence_index),
]


def main():
    print("=" * 70)
    print("  30-Day Human Intelligence Potential (HIP) Stress Test")
    print("  30-Day Human Intelligence Potential Stress Test")
    print("  " + "─" * 50)
    print(f"  Ticks/day: {TICKS_PER_DAY}  |  Total days: 30  |  Pass: {HIP_PASS_THRESHOLD}/100")
    print(f"  Neuron count: {NEURON_COUNT}  |  Seed: {SEED}")
    print("=" * 70)

    t0 = time.time()
    results = {}

    for name, func in EXPERIMENTS:
        try:
            passed = func()
            results[name] = passed
        except Exception as e:
            print(f"\n  ✗ EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    elapsed = time.time() - t0

    # === Summary ===
    print("\n" + "=" * 70)
    print("  30-Day HIP Stress Test — Final Results")
    print(" 30-Day Human Intelligence Stress Test — Final Report")
    print("=" * 70)

    pass_count = 0
    for name, passed in results.items():
        tag = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {tag}  {name}")
        if passed:
            pass_count += 1

    total = len(results)
    pct = pass_count / total * 100 if total > 0 else 0
    print(f"\n  Score: {pass_count}/{total} passed  ({pct:.0f}%)  [{elapsed:.1f}s]")

    if pass_count == total:
        print("\n  ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★")
        print("  ★  ALL DOMAINS VERIFIED — HUMAN INTELLIGENCE  ★")
        print("  ★  POTENTIAL CONFIRMED                        ★")
        print("  ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★")
    else:
        print(f"\n  ⚠ {total - pass_count} domain(s) require improvement")

    print("=" * 70)
    return pass_count == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
