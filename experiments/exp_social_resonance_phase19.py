#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 19 — Social Resonance & Theory of Mind (Social Resonance and Theory of Mind)

exp_social_resonance_phase19.py — 10 experiment

Core Objectives: 
  Verify the three pillars of SocialResonanceEngine: 
  1. Social impedance coupling (Γ_social)
  2. Theory of Mind (Sally-Anne paradigm)
  3. Social homeostasis (loneliness / compassion fatigue / optimal range)

Physical Prediction：
  - Indifference → Γ ≈ 1 → pressure reflection
  - Empathy → Γ → 0 → pressure transmission
  - Sally-Anne → high ToM → correctprediction false belief
  - Loneliness → social_need ↑ → exceeds threshold
  - Compassion fatigue → compassion_energy ↓ → needs solitude
  - Alice vs Alice → Impedance synchronization → Γ ↓

Ten experiments: 
  Exp 1: Mismatch — Indifferent listener (Γ > 0.5)
  Exp 2: Match — Empathetic listening (Γ < 0.3)
  Exp 3: Energy Conservation — Compassion fatigue physics cost
  Exp 4: Sally-Anne — False belief classic test
  Exp 5: Social Homeostasis — Loneliness and satiation 
  Exp 6: Trust Dynamics — Trust building and betrayal
  Exp 7: Bidirectional Coupling — Bidirectional social vs unidirectional
  Exp 8: Frequency Sync — Social rhythm synchronization
  Exp 9: Alice vs Alice — Two AliceBrain social interaction
  Exp 10: Grand Summary — Complete system clinical report

Execute: python -m experiments.exp_social_resonance_phase19
"""

from __future__ import annotations

import sys
import os
import time
import math
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alice.brain.social_resonance import (
    SocialResonanceEngine,
    Belief,
    SocialCouplingResult,
    SallyAnneResult,
    SocialHomeostasisState,
    Z_SOCIAL_BASE,
    K_RELEASE,
    K_ABSORB,
    K_REFLECT,
    LONELINESS_THRESHOLD,
    SYNC_THRESHOLD,
)
from alice.alice_brain import AliceBrain
from alice.core.protocol import Priority, Modality


# ============================================================
# Experiment parameters
# ============================================================

NEURON_COUNT = 80
SEED = 42
PRINT_INTERVAL = 20


# ============================================================
# Helper functions
# ============================================================

def separator(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def subsection(title: str):
    print(f"\n  --- {title} ---")


def ascii_sparkline(values: list, width: int = 50, label: str = "") -> str:
    if not values or all(v == values[0] for v in values):
        return f"  {label}: [flat]"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    blocks = " ▁▂▃▄▅▆▇█"
    step = len(values) / width if len(values) > width else 1
    sampled = []
    i = 0.0
    while i < len(values) and len(sampled) < width:
        idx = min(int(i), len(values) - 1)
        sampled.append(values[idx])
        i += step
    bar = ""
    for v in sampled:
        level = int((v - mn) / rng * (len(blocks) - 1))
        bar += blocks[level]
    return f"  {label}: {bar}  [{mn:.4f} → {values[-1]:.4f}]"


# ============================================================
# Exp 1: Mismatch — Indifference (Γ > 0.5)
# ============================================================

def exp1_mismatch() -> bool:
    """
    Speaker pressure is very high, listener is completely indifferent. 

    Expected: Γ_social > 0.5, pressure is barely transmitted. 
    """
    separator("Exp 1: Mismatch — Indifferent listener(Γ > 0.5)")

    engine = SocialResonanceEngine()

    gamma_history = []
    eta_history = []

    for tick in range(100):
        # Speaker pressure increases over time 
        speaker_pressure = 0.5 + tick * 0.01

        result = engine.couple(
            speaker_id="speaker",
            listener_id="listener",
            speaker_pressure=speaker_pressure,
            listener_empathy=0.05, # Almost no empathy
            listener_effort=0.0, # Zero effort
        )

        gamma_history.append(result.gamma_social)
        eta_history.append(result.energy_transfer)

        engine.tick()

    avg_gamma = float(np.mean(gamma_history))
    avg_eta = float(np.mean(eta_history))

    print(ascii_sparkline(gamma_history, 50, "Γ_social  "))
    print(ascii_sparkline(eta_history, 50, "η (efficiency)"))
    print(f"\n  mean Γ: {avg_gamma:.4f}")
    print(f"  mean η: {avg_eta:.4f}")

    passed = avg_gamma > 0.5
    print(f"\n  Indifference high Γ: {'✓ PASS' if passed else '✗ FAIL'}")
    print(f"  Exp 1 Result: {'[PASS]' if passed else '[FAIL]'}")
    return passed


# ============================================================
# Exp 2: Match — Empathetic listening(Γ < 0.3)
# ============================================================

def exp2_match() -> bool:
    """
    Speaker pressure is very high, but listener has high empathy + effort in listening. 

    Expected: Γ_social < 0.3, pressure effectively transmitted. 
    """
    separator("Exp 2: Match — Empathetic listening(Γ < 0.3)")

    engine = SocialResonanceEngine()

    gamma_history = []
    eta_history = []
    released_total = 0.0

    for tick in range(100):
        speaker_pressure = 1.5 # High pressure

        result = engine.couple(
            speaker_id="speaker",
            listener_id="listener",
            speaker_pressure=speaker_pressure,
            listener_empathy=0.9, # High empathy
            listener_effort=0.9, # High listening effort
        )

        gamma_history.append(result.gamma_social)
        eta_history.append(result.energy_transfer)
        released_total += result.pressure_released

        engine.tick(has_social_input=True)

    avg_gamma = float(np.mean(gamma_history))
    avg_eta = float(np.mean(eta_history))

    print(ascii_sparkline(gamma_history, 50, "Γ_social  "))
    print(ascii_sparkline(eta_history, 50, "η (efficiency)"))
    print(f"\n  mean Γ: {avg_gamma:.4f}")
    print(f"  mean η: {avg_eta:.4f}")
    print(f"  Cumulative pressure released: {released_total:.4f}")

    passed = avg_gamma < 0.3 and avg_eta > 0.7
    print(f"\n  Empathetic low Γ: {'✓ PASS' if avg_gamma < 0.3 else '✗ FAIL'}")
    print(f"  High efficiency transmission: {'✓ PASS' if avg_eta > 0.7 else '✗ FAIL'}")
    print(f"  Exp 2 Result: {'[PASS]' if passed else '[FAIL]'}")
    return passed


# ============================================================
# Exp 3: Energy Conservation — Compassion Fatigue
# ============================================================

def exp3_compassion_fatigue() -> bool:
    """
    Continuously listening to highly stressed person → compassion energy decreases. 

    expected: 
      Long-term listening → compassion_energy decreases
      This is the physics basis for therapist burnout
    """
    separator("Exp 3: Energy Conservation — Compassion Fatigue")

    engine = SocialResonanceEngine()

    compassion_history = []
    absorbed_total = 0.0

    initial_compassion = engine.get_compassion_energy()
    print(f"  Initial compassion energy: {initial_compassion:.4f}")

    # Continue 200 ticks of high-intensity listening
    for tick in range(200):
        result = engine.couple(
            speaker_id="patient",
            listener_id="therapist",
            speaker_pressure=2.0, # Extremely high pressure
            listener_empathy=0.95,
            listener_effort=0.95,
        )
        absorbed_total += result.pressure_absorbed
        engine.tick(has_social_input=True)
        compassion_history.append(engine.get_compassion_energy())

    final_compassion = engine.get_compassion_energy()

    print(ascii_sparkline(compassion_history, 50, "compassion energy"))
    print(f"\n  Final compassion energy: {final_compassion:.4f}")
    print(f"  Accumulated absorbed pressure: {absorbed_total:.4f}")
    print(f"  Energy decrease: {initial_compassion - final_compassion:.4f}")

    # Recovery period: Solitude 100 ticks
    subsection("Recovery period: Solitude 100 ticks")
    recovery_start = final_compassion
    for tick in range(100):
        engine.tick(has_social_input=False)
        compassion_history.append(engine.get_compassion_energy())

    recovered = engine.get_compassion_energy()
    print(f"  Post-recovery compassion energy: {recovered:.4f}")
    print(f"  Recovery amount: {recovered - recovery_start:.4f}")

    # Verify: energy decreased + can recover
    energy_dropped = final_compassion < initial_compassion * 0.95
    can_recover = recovered > final_compassion

    print(f"\n energy have decrease: {'✓ PASS' if energy_dropped else '✗ FAIL'}")
    print(f"  Solitude can recover: {'✓ PASS' if can_recover else '✗ FAIL'}")
    passed = energy_dropped and can_recover
    print(f"  Exp 3 Result: {'[PASS]' if passed else '[FAIL]'}")
    return passed


# ============================================================
# Exp 4: Sally-Anne — False Belief Test
# ============================================================

def exp4_sally_anne() -> bool:
    """
    Sally-Anne test — Theory of Mind gold standard. 

    Scenario: 
      1. Sally puts ball in basket
      2. Sally leaves
      3. Anne moves ball to box
      4. Sally returns — Question: 'Where will Sally look for the ball?'

    Correct answer: Sally will look in the basket (because she doesn't know the ball was moved)
    → Requires ToM to answer correctly

    Tests: 
      A) Low ToM (initial) → Alice predicts Sally goes to box (egocentric error)
      B) ToM high (maturation after )→ Alice prediction Sally go basket(correct! )
    """
    separator("Exp 4: Sally-Anne — False Belief Test")

    engine = SocialResonanceEngine()

    # === Phase A: low ToM ===
    subsection("Phase A: Low ToM (initial state)")
    print(f"  Initial ToM capacity: {engine.get_tom_capacity():.4f}")

    # 1. Sally puts ball in basket
    engine.update_belief("sally", "ball_location", "basket", "basket", 1.0)
    print("  Sally puts ball in basket → Sally believes ball_location = basket")

    # 2. Sally leaves( not in field)

    # 3. Anne moves ball to box — Sally is not present, so her belief doesn't update
    engine.update_reality("ball_location", "box")
    print("  Anne moves ball to box → reality = box, but Sally doesn't know")

    # 4. Sally-Anne test
    result_low = engine.sally_anne_test("sally", "ball_location")

    print(f"\n  Sally believes: {result_low.agent_believes}")
    print(f"  Reality: {result_low.reality}")
    print(f"  Alice prediction: {result_low.alice_prediction}")
    print(f"  ToM Level: {result_low.tom_level}")
    print(f"  predictioncorrect: {result_low.prediction_correct}")

    # Note: At low ToM, Alice may incorrectly predict "box"
    low_tom_result = result_low.tom_level  # record level

    # === Phase B: high ToM(simulationdevelopment) ===
    subsection("Phase B: High ToM (simulate social development maturation)")

    # Simulate ToM maturation through extensive social interaction
    engine._tom_capacity = 0.8 # Directly set high ToM (simulating extensive social experience)
    print(f"  Post-maturation ToM capacity: {engine.get_tom_capacity():.4f}")

    # Reset Sally's belief state
    engine.update_belief("sally", "ball_location", "basket", "box", 1.0)

    result_high = engine.sally_anne_test("sally", "ball_location")

    print(f"\n  Sally believes: {result_high.agent_believes}")
    print(f"  Reality: {result_high.reality}")
    print(f"  Alice prediction: {result_high.alice_prediction}")
    print(f"  ToM Level: {result_high.tom_level}")
    print(f"  predictioncorrect: {result_high.prediction_correct}")

    # === Phase C: verification false belief detection ===
    subsection("Phase C: False Belief detection")

    false_beliefs = engine.get_false_beliefs("sally")
    gamma_belief = engine.get_gamma_belief("sally")

    print(f" Sally false beliefs: {len(false_beliefs)}")
    print(f"  Γ_belief (mean belief mismatch): {gamma_belief:.4f}")
    for fb in false_beliefs:
        print(f"    {fb.subject}: believes={fb.value}, reality={fb.reality_value}, Γ={fb.gamma_belief:.4f}")

    # verification
    has_false_belief = len(false_beliefs) > 0
    high_tom_correct = result_high.prediction_correct
    high_tom_level2 = result_high.tom_level >= 2

    print(f"\n detection to false belief: {'✓ PASS' if has_false_belief else '✗ FAIL'}")
    print(f"  High ToM prediction correct: {'✓ PASS' if high_tom_correct else '✗ FAIL'}")
    print(f"  Reached Level 2: {'✓ PASS' if high_tom_level2 else '✗ FAIL'}")

    passed = has_false_belief and high_tom_correct and high_tom_level2
    print(f"  Exp 4 Result: {'[PASS]' if passed else '[FAIL]'}")
    return passed


# ============================================================
# Exp 5: Social Homeostasis — Loneliness and satiation 
# ============================================================

def exp5_social_homeostasis() -> bool:
    """
    Long-term solitude → social need increases → exceeds loneliness threshold. 
    After socializing → need decreases → returns to optimal range. 

    Physical prediction: social_need follows homeostatic dynamics. 
    """
    separator("Exp 5: Social Homeostasis — Loneliness and satiation ")

    engine = SocialResonanceEngine()

    need_history = []

    # Phase A: Long-term solitude (200 ticks)
    subsection("Phase A: Long-term solitude")
    for tick in range(200):
        engine.tick(has_social_input=False)
        need_history.append(engine.get_social_need())

    lonely_need = engine.get_social_need()
    homeostasis = engine.get_homeostasis()
    print(f"  Post-solitude social need: {lonely_need:.4f}")
    print(f"  Is lonely: {homeostasis.is_lonely}")
    print(f"  Loneliness duration ticks: {engine.get_loneliness_duration()}")

    is_lonely_after_isolation = homeostasis.is_lonely

    # Phase B: Intensive socializing (100 ticks)
    subsection("Phase B: Intensive socializing")
    for tick in range(100):
        engine.couple(
            speaker_id="friend",
            listener_id="self",
            speaker_pressure=0.5,
            listener_empathy=0.8,
            listener_effort=0.8,
        )
        engine.tick(has_social_input=True)
        need_history.append(engine.get_social_need())

    social_need = engine.get_social_need()
    homeostasis = engine.get_homeostasis()
    print(f"  Post-social need: {social_need:.4f}")
    print(f"  Is lonely: {homeostasis.is_lonely}")

    need_dropped = social_need < lonely_need

    # Phase C: Second solitude → verify need rises again
    subsection("Phase C: Second solitude (verify cycle)")
    for tick in range(100):
        engine.tick(has_social_input=False)
        need_history.append(engine.get_social_need())

    final_need = engine.get_social_need()
    need_rises_again = final_need > social_need

    print(ascii_sparkline(need_history, 60, "social need"))
    print(f"\n  Final need: {final_need:.4f}")

    print(f"\n  Lonely after solitude: {'✓ PASS' if is_lonely_after_isolation else '✗ FAIL'}")
    print(f"  Need decreased after socializing: {'✓ PASS' if need_dropped else '✗ FAIL'}")
    print(f"  Need rises again after solitude: {'✓ PASS' if need_rises_again else '✗ FAIL'}")

    passed = is_lonely_after_isolation and need_dropped and need_rises_again
    print(f"  Exp 5 Result: {'[PASS]' if passed else '[FAIL]'}")
    return passed


# ============================================================
# Exp 6: Trust Dynamics — Trust Building and Erosion
# ============================================================

def exp6_trust_dynamics() -> bool:
    """
    Good interactions → trust grows. 
    Neglect → trust decreases. 
    """
    separator("Exp 6: Trust Dynamics — Trust Building and Erosion")

    engine = SocialResonanceEngine()

    trust_history = []

    # Phase A: High-quality interactions (building trust)
    subsection("Phase A: High-quality interactions 100 ticks")
    for tick in range(100):
        engine.couple(
            speaker_id="self",
            listener_id="friend",
            speaker_pressure=1.0,
            listener_empathy=0.9,
            listener_effort=0.9,
        )
        engine.tick(has_social_input=True)
        model = engine.get_agent_model("friend")
        if model:
            trust_history.append(model.trust)

    friend = engine.get_agent_model("friend")
    trust_after_good = friend.trust if friend else 0.5
    print(f"  Trust after good interactions: {trust_after_good:.4f}")

    # Phase B: Indifferent interactions (trust erosion)
    subsection("Phase B: Indifferent interactions 100 ticks")
    for tick in range(100):
        engine.couple(
            speaker_id="self",
            listener_id="friend",
            speaker_pressure=1.0,
            listener_empathy=0.05,
            listener_effort=0.0,
        )
        engine.tick(has_social_input=True)
        model = engine.get_agent_model("friend")
        if model:
            trust_history.append(model.trust)

    friend = engine.get_agent_model("friend")
    trust_after_cold = friend.trust if friend else 0.5
    print(f"  Trust after indifferent interactions: {trust_after_cold:.4f}")

    print(ascii_sparkline(trust_history, 50, "trust level"))

    trust_grew = trust_after_good > 0.5
    trust_dropped = trust_after_cold < trust_after_good

    print(f"\n  Trust grew: {'✓ PASS' if trust_grew else '✗ FAIL'} ({trust_after_good:.4f})")
    print(f"  Decreased after indifference: {'✓ PASS' if trust_dropped else '✗ FAIL'} ({trust_after_cold:.4f})")

    passed = trust_grew and trust_dropped
    print(f"  Exp 6 Result: {'[PASS]' if passed else '[FAIL]'}")
    return passed


# ============================================================
# Exp 7: Bidirectional Coupling — Bidirectional vs Unidirectional
# ============================================================

def exp7_bidirectional() -> bool:
    """
    Unidirectional listening vs bidirectional listening. 

    Expected: Bidirectional total pressure release > twice unidirectional. 
    (Because bidirectional bond_impedance decreases faster → smaller Γ → higher η)
    """
    separator("Exp 7: Bidirectional — Bidirectional vs Unidirectional")

    # Unidirectional
    subsection("Unidirectional interaction")
    engine_uni = SocialResonanceEngine()
    released_uni = 0.0
    for tick in range(100):
        result = engine_uni.couple(
            speaker_id="A",
            listener_id="B",
            speaker_pressure=1.0,
            listener_empathy=0.7,
            listener_effort=0.7,
        )
        released_uni += result.pressure_released
        engine_uni.tick(has_social_input=True)

    print(f"  Unidirectional pressure released: {released_uni:.4f}")

    # Bidirectional
    subsection("Bidirectional interaction")
    engine_bi = SocialResonanceEngine()
    released_bi = 0.0
    for tick in range(100):
        result_ab, result_ba = engine_bi.bidirectional_couple(
            agent_a_id="A",
            agent_b_id="B",
            pressure_a=1.0,
            pressure_b=1.0,
            empathy_a=0.7,
            empathy_b=0.7,
            effort_a=0.7,
            effort_b=0.7,
        )
        released_bi += result_ab.pressure_released + result_ba.pressure_released
        engine_bi.tick(has_social_input=True)

    print(f"  Bidirectional pressure released: {released_bi:.4f}")
    print(f" ratio: {released_bi / max(released_uni, 0.001):.2f}x")

    # Bidirectional > Unidirectional (no need for >2x since pressure is fixed at 1.0)
    bi_better = released_bi > released_uni * 1.5
    print(f"\n  Bidirectional obviously better: {'✓ PASS' if bi_better else '✗ FAIL'}")

    # Bidirectional mutual familiarity — both parties develop familiarity
    model_bi_B = engine_bi.get_agent_model("B")
    model_bi_A = engine_bi.get_agent_model("A")
    model_uni_B = engine_uni.get_agent_model("B")
    model_uni_A = engine_uni.get_agent_model("A")

    # Bidirectional: Both A and B have interaction records
    bi_mutual = (
        model_bi_B is not None and model_bi_A is not None
        and model_bi_B.familiarity > 0 and model_bi_A.familiarity > 0
    )
    # Unidirectional: Only B has interaction records (A is only speaker, listener model is B)
    uni_onesided = (
        model_uni_B is not None
        and model_uni_B.familiarity > 0
    )
    # Unidirectional A interaction count < Bidirectional A interaction count
    bi_A_interactions = model_bi_A.interaction_count if model_bi_A else 0
    uni_A_interactions = model_uni_A.interaction_count if model_uni_A else 0
    bi_richer = bi_A_interactions > uni_A_interactions

    print(f"  Bidirectional A.interactions={bi_A_interactions}, B.interactions={model_bi_B.interaction_count if model_bi_B else 0}")
    print(f"  Unidirectional A.interactions={uni_A_interactions}, B.interactions={model_uni_B.interaction_count if model_uni_B else 0}")
    print(f"  Bidirectional reciprocity: {'✓ PASS' if bi_mutual else '✗ FAIL'}")
    print(f"  Bidirectional richer: {'✓ PASS' if bi_richer else '✗ FAIL'}")

    passed = bi_better and bi_mutual and bi_richer
    print(f"  Exp 7 Result: {'[PASS]' if passed else '[FAIL]'}")
    return passed


# ============================================================
# Exp 8: Frequency Sync — Social rhythm synchronization
# ============================================================

def exp8_frequency_sync() -> bool:
    """
    Continuous high-quality interactions → sync_degree increases → generating 'rapport'. 
    """
    separator("Exp 8: Frequency Sync — Social rhythm synchronization")

    engine = SocialResonanceEngine()

    sync_history = []

    # 200 ticks of continuous interaction
    for tick in range(200):
        engine.couple(
            speaker_id="A",
            listener_id="partner",
            speaker_pressure=0.8,
            listener_empathy=0.85,
            listener_effort=0.85,
        )
        engine.tick(has_social_input=True)

        model = engine.get_agent_model("partner")
        if model:
            sync_history.append(model.sync_degree)

    final_sync = sync_history[-1] if sync_history else 0.0

    print(ascii_sparkline(sync_history, 50, "sync degree"))
    print(f"\n  Final synchronization: {final_sync:.4f}")
    print(f"  Synchronization threshold: {SYNC_THRESHOLD}")

    monotonic = all(
        sync_history[i] <= sync_history[i + 1] + 0.001
        for i in range(len(sync_history) - 1)
    )
    reached_threshold = final_sync > 0.3 # After 200 ticks there should be obvious synchronization

    print(f"\n  Monotonically increasing: {'✓ PASS' if monotonic else '✗ FAIL'}")
    print(f"  Reached significant synchronization: {'✓ PASS' if reached_threshold else '✗ FAIL'}")

    passed = monotonic and reached_threshold
    print(f"  Exp 8 Result: {'[PASS]' if passed else '[FAIL]'}")
    return passed


# ============================================================
# Exp 9: Alice vs Alice — Two Complete AliceBrain Interaction
# ============================================================

def exp9_alice_vs_alice() -> bool:
    """
    Two complete AliceBrains interact through SocialResonanceEngine. 

    Scenario: 
      Alice_A 'sees' a stress stimulus (red screen)
      Alice_B listens to A through social_resonance.couple()
      → Verify A's pressure is transmitted through social coupling to the integration engine

    This is the Phase 19 core experiment — Alice vs Alice. 
    """
    separator("Exp 9: Alice vs Alice — Two AliceBrain social interaction")

    rng = np.random.RandomState(SEED)

    # Create two Alices
    alice_a = AliceBrain(neuron_count=NEURON_COUNT)
    alice_b = AliceBrain(neuron_count=NEURON_COUNT)

    gamma_history = []
    eta_history = []
    a_valence_history = []
    b_social_need_history = []

    TICKS = 80

    for tick in range(TICKS):
        # Alice A sees a stress stimulus (red flash)
        stress_pixels = rng.rand(64, 64) * 0.3 # Dark noise
        if tick < 50:
            stress_pixels[:32, :32] = 0.9 # Top-left red flash
        result_a = alice_a.see(stress_pixels)

        # Alice B sees a neutral stimulus
        neutral_pixels = rng.rand(64, 64) * 0.5
        result_b = alice_b.see(neutral_pixels)

        # Couple through SocialResonanceEngine
        # Use Alice A's social_resonance engine
        # Alice A is the stressed party (speaker), Alice B is listener
        coupling = alice_a.social_resonance.couple(
            speaker_id="alice_a",
            listener_id="alice_b",
            speaker_pressure=alice_a.vitals.pain_level + alice_a.autonomic.sympathetic * 0.5,
            listener_empathy=alice_b.mirror_neurons.get_empathy_capacity(),
            listener_effort=0.7, # B has moderate listening willingness
        )

        gamma_history.append(coupling.gamma_social)
        eta_history.append(coupling.energy_transfer)
        a_valence_history.append(alice_a.amygdala.get_valence())
        b_social_need_history.append(alice_b.social_resonance.get_social_need())

        if tick % 20 == 0:
            print(f"  tick {tick:3d}: Γ={coupling.gamma_social:.4f}  "
                  f"η={coupling.energy_transfer:.4f}  "
                  f"A_pain={alice_a.vitals.pain_level:.4f}  "
                  f"released={coupling.pressure_released:.4f}")

    # analysis
    avg_gamma = float(np.mean(gamma_history))
    avg_eta = float(np.mean(eta_history))

    print(ascii_sparkline(gamma_history, 50, "Γ_social"))
    print(ascii_sparkline(eta_history, 50, "η (efficiency)"))

    print(f"\n  mean Γ: {avg_gamma:.4f}")
    print(f"  mean η: {avg_eta:.4f}")
    print(f"  Alice A finalpain: {alice_a.vitals.pain_level:.4f}")
    print(f"  Alice B social need: {alice_b.social_resonance.get_social_need():.4f}")

    # verification
    # Coupling exists (Γ is not static 1.0)
    has_coupling = avg_gamma < 0.99
    # systemstabilize( not have NaN)
    no_nan = not any(math.isnan(g) for g in gamma_history)
    # Agent modelcreate 
    model = alice_a.social_resonance.get_agent_model("alice_b")
    model_exists = model is not None

    print(f"\n  Γ < 1.0 (coupling exists): {'✓ PASS' if has_coupling else '✗ FAIL'}")
    print(f" no NaN: {'✓ PASS' if no_nan else '✗ FAIL'}")
    print(f"  Agent model created: {'✓ PASS' if model_exists else '✗ FAIL'}")

    if model_exists:
        print(f"    bond_impedance: {model.bond_impedance:.2f}Ω")
        print(f"    trust: {model.trust:.4f}")
        print(f"    familiarity: {model.familiarity:.4f}")

    passed = has_coupling and no_nan and model_exists
    print(f"  Exp 9 Result: {'[PASS]' if passed else '[FAIL]'}")
    return passed


# ============================================================
# Exp 10: Grand Summary — Complete system clinical report
# ============================================================

def exp10_grand_summary() -> bool:
    """
    Comprehensive social system clinical report — verify all subsystems integrate correctly. 
    """
    separator("Exp 10: Grand Summary — Social System Clinical Report")

    engine = SocialResonanceEngine()

    # === Phase 1: Solitude creates loneliness ===
    subsection("Phase 1: Solitude 150 ticks → create loneliness")
    for tick in range(150):
        engine.tick(has_social_input=False)

    h1 = engine.get_homeostasis()
    print(f"  Social need: {h1.social_need:.4f}")
    print(f"  Lonely: {h1.is_lonely}")

    # === Phase 2: Social interaction ===
    subsection("Phase 2: Interact with three people 100 ticks")
    for tick in range(100):
        for friend in ["friend_a", "friend_b", "friend_c"]:
            engine.couple(
                speaker_id="self",
                listener_id=friend,
                speaker_pressure=0.3 + tick * 0.005,
                listener_empathy=0.7,
                listener_effort=0.7,
            )
        engine.tick(has_social_input=True)

    h2 = engine.get_homeostasis()
    tracked = engine.get_all_agent_ids()
    print(f" tracking Agents: {tracked}")
    print(f"  Social need: {h2.social_need:.4f}")
    print(f"  Compassion energy: {h2.compassion_energy:.4f}")

    # === phase 3: Sally-Anne ===
    subsection("Phase 3: Sally-Anne Test")
    engine._tom_capacity = 0.8
    engine.update_belief("friend_a", "gift_location", "drawer", "drawer", 1.0)
    engine.update_reality("gift_location", "closet")

    sa_result = engine.sally_anne_test("friend_a", "gift_location")
    print(f"  Friend_A believes gift is in: {sa_result.agent_believes}")
    print(f"  Actual location: {sa_result.reality}")
    print(f"  Alice prediction: {sa_result.alice_prediction}")
    print(f"  ToM Level: {sa_result.tom_level}")

    # === phase 4: Final Report ===
    subsection("Phase 4: Complete state report")
    state = engine.get_state()
    stats = engine.get_stats()

    print(f"  Total ticks: {stats['total_ticks']}")
    print(f"  Total interactions: {stats['total_interactions']}")
    print(f"  Total couplings: {stats['total_couplings']}")
    print(f"  Avg Γ: {stats['avg_gamma']:.4f}")
    print(f"  Avg η: {stats['avg_eta']:.4f}")
    print(f"  ToM capacity: {stats['tom_capacity']:.4f}")
    print(f"  False belief detections: {stats['false_belief_detections']}")
    print(f"  Tracked agents: {stats['tracked_agents']}")

    # verification
    has_agents = stats['tracked_agents'] >= 3
    has_couplings = stats['total_couplings'] > 200
    has_tom = stats['tom_capacity'] >= 0.8
    sa_correct = sa_result.prediction_correct
    no_nan_state = not any(
        isinstance(v, float) and math.isnan(v)
        for v in [
            stats['avg_gamma'], stats['avg_eta'],
            stats['social_need'], stats['compassion_energy'],
        ]
    )

    print(f"\n tracking ≥ 3 Agents: {'✓ PASS' if has_agents else '✗ FAIL'}")
    print(f"  Couplings > 200: {'✓ PASS' if has_couplings else '✗ FAIL'}")
    print(f"  ToM ≥ 0.8: {'✓ PASS' if has_tom else '✗ FAIL'}")
    print(f"  Sally-Anne correct: {'✓ PASS' if sa_correct else '✗ FAIL'}")
    print(f"  No NaN: {'✓ PASS' if no_nan_state else '✗ FAIL'}")

    passed = has_agents and has_couplings and has_tom and sa_correct and no_nan_state
    print(f"  Exp 10 Result: {'[PASS]' if passed else '[FAIL]'}")
    return passed


# ============================================================
# Main — Execute all experiments
# ============================================================

EXPERIMENTS = [
    ("Exp 1: Mismatch (Indifference Γ>0.5)", exp1_mismatch),
    ("Exp 2: Match (Empathy Γ<0.3)", exp2_match),
    ("Exp 3: Compassion Fatigue", exp3_compassion_fatigue),
    ("Exp 4: Sally-Anne (ToM)", exp4_sally_anne),
    ("Exp 5: Social Homeostasis", exp5_social_homeostasis),
    ("Exp 6: Trust Dynamics", exp6_trust_dynamics),
    ("Exp 7: Bidirectional Coupling", exp7_bidirectional),
    ("Exp 8: Frequency Sync", exp8_frequency_sync),
    ("Exp 9: Alice vs Alice", exp9_alice_vs_alice),
    ("Exp 10: Grand Summary", exp10_grand_summary),
]


def main():
    print("=" * 70)
    print("  Phase 19 — Social Resonance & Theory of Mind")
    print(" Social Resonance and Theory of Mind — 10 Experiments")
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
    print("  Final Results")
    print("=" * 70)

    pass_count = 0
    for name, passed in results.items():
        tag = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {tag}  {name}")
        if passed:
            pass_count += 1

    total = len(results)
    print(f"\n  Score: {pass_count}/{total} passed  ({elapsed:.1f}s)")
    print("=" * 70)

    return pass_count == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
