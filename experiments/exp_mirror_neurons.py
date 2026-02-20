# -*- coding: utf-8 -*-
"""
+======================================================================+
|  Experiment: Mirror Neurons, Empathy & Theory of Mind                 |
| Mirror Neurons, Empathy and Theory of Mind Experiment                 |
|                                                                       |
| 'Empathy is not imitation -- it is impedance resonance'               |
+======================================================================+

This experiment verifies Alice's core social cognition mechanisms:

  Experiment 1: Motor mirroring -- observing others' actions → internal motor model resonance
  Experiment 2: Emotional mirroring -- observing others' emotions → empathy response
  Experiment 3: Empathy development -- from primitive emotional contagion to cognitive empathy maturation
  Experiment 4: Intent inference -- observe behavior sequences → infer intentions (Theory of Mind)
  Experiment 5: Social bonding -- interaction count → impedance reduction → closeness
  Experiment 6: AliceBrain social -- full brain encountering another agent
"""

from __future__ import annotations

import sys
import numpy as np
from typing import List, Dict, Any

from alice.brain.mirror_neurons import (
    MirrorNeuronEngine,
    MirrorLayer,
    MOTOR_MIRROR_IMPEDANCE,
    MIRROR_RESONANCE_THRESHOLD,
    INITIAL_EMPATHY_CAPACITY,
    TOM_INITIAL_CAPACITY,
    SOCIAL_BOND_BASE,
)


def header(title: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def subheader(title: str) -> None:
    print(f"\n  --- {title} ---")


def check(name: str, condition: bool, detail: str = "") -> bool:
    symbol = "[PASS]" if condition else "[FAIL]"
    line = f"  [{symbol}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return condition


# ============================================================================
# Experiment 1: Motor Mirroring
# ============================================================================


def exp1_motor_mirror() -> bool:
    """
    Motor mirroring = impedance resonance when observing others' actions

    Expected:
    - Familiar actions (Z close to self) → high resonance → imitation urge
    - Unfamiliar actions (Z far from self) → low resonance → no imitation urge
    - Repeated observation → learning → resonance elevation
    """
    header("Experiment 1: Motor Mirroring -- Observing = Brain Doing Once")

    engine = MirrorNeuronEngine()
    passed = True

    # === 1a: Familiar action → high resonance ===
    subheader("1a: Familiar action (Z ~ Z_self)")
    familiar = engine.observe_action(
        agent_id="caregiver",
        modality="vocal",
        observed_impedance=MOTOR_MIRROR_IMPEDANCE + 5,  # close to self
        action_label="vocalize",
    )
    print(f"  Z_observed = {familiar.z_observed}, Z_self = {familiar.z_self}")
    print(f"  Gamma_mirror = {familiar.gamma_mirror:.4f}")
    print(f"  Resonance = {familiar.resonance_strength:.4f}")
    print(f"  Imitation urge = {engine.get_imitation_urge():.4f}")
    passed &= check("Familiar action high resonance", familiar.gamma_mirror < MIRROR_RESONANCE_THRESHOLD,
                     f"Gamma = {familiar.gamma_mirror:.4f}")
    passed &= check("Generates imitation urge", engine.get_imitation_urge() > 0.1,
                     f"urge = {engine.get_imitation_urge():.4f}")

    # === 1b: Unfamiliar action → low resonance ===
    subheader("1b: Unfamiliar action (Z = 5 x Z_self)")
    engine2 = MirrorNeuronEngine()
    unfamiliar = engine2.observe_action(
        agent_id="stranger",
        modality="locomotion",
        observed_impedance=MOTOR_MIRROR_IMPEDANCE * 5,
        action_label="backflip",
    )
    print(f"  Z_observed = {unfamiliar.z_observed}, Z_self = {unfamiliar.z_self}")
    print(f"  Gamma_mirror = {unfamiliar.gamma_mirror:.4f}")
    print(f"  Resonance = {unfamiliar.resonance_strength:.4f}")
    passed &= check("Unfamiliar action low resonance", unfamiliar.gamma_mirror > 0.5,
                     f"Gamma = {unfamiliar.gamma_mirror:.4f}")

    # === 1c: Observation learning → resonance elevation ===
    subheader("1c: Observation learning -- repeatedly observing the same action")
    engine3 = MirrorNeuronEngine()
    gammas = []
    target_z = 120.0  # moderately unfamiliar impedance
    for i in range(30):
        event = engine3.observe_action(
            agent_id="teacher",
            modality="manual",
            observed_impedance=target_z,
            action_label="wave",
        )
        gammas.append(event.gamma_mirror)

    print(f"  Trial 1: Gamma = {gammas[0]:.4f}")
    print(f"  Trial 15: Gamma = {gammas[14]:.4f}")
    print(f"  Trial 30: Gamma = {gammas[29]:.4f}")
    passed &= check("Resonance elevated after observation learning", gammas[29] < gammas[0],
                     f"{gammas[0]:.4f} -> {gammas[29]:.4f}")

    return passed


# ============================================================================
# Experiment 2: Emotional Mirroring
# ============================================================================


def exp2_emotional_mirror() -> bool:
    """
    Observing others' emotions → generates resonance within self

    Expected:
    - Observing negative emotions → empathy generates negative valence
    - Observing positive emotions → empathy generates positive valence
    - Empathy intensity positively correlates with observed emotion strength
    """
    header("Experiment 2: Emotional Mirroring -- Seeing Pain Feels Painful")

    engine = MirrorNeuronEngine()
    # first elevate empathy capacity for testing
    for _ in range(30):
        engine.mature(social_interaction=True)
    passed = True

    # === 2a: Observing pain → negative empathy ===
    subheader("2a: Observing pain expression")
    pain_response = engine.observe_emotion(
        agent_id="child",
        observed_valence=-0.8,
        observed_arousal=0.9,
        modality="facial",
    )
    print(f"  Observed valence: -0.8 (pain)")
    print(f"  Empathy intensity: {pain_response.empathy_strength:.4f}")
    print(f"  Resonance valence: {pain_response.resonated_valence:.4f}")
    print(f"  Emotional contagion: {pain_response.emotional_contagion:.4f}")
    print(f"  Empathy type: {pain_response.empathy_type}")
    passed &= check("Pain observation generates empathy", pain_response.empathy_strength > 0.05,
                     f"strength = {pain_response.empathy_strength:.4f}")
    passed &= check("Resonance valence is negative", pain_response.resonated_valence < 0,
                     f"valence = {pain_response.resonated_valence:.4f}")

    # === 2b: Observing joy → positive empathy ===
    subheader("2b: Observing joy expression")
    engine_joy = MirrorNeuronEngine()
    for _ in range(30):
        engine_joy.mature(social_interaction=True)
    joy_response = engine_joy.observe_emotion(
        agent_id="friend",
        observed_valence=0.9,
        observed_arousal=0.7,
        modality="facial",
    )
    print(f"  Observed valence: +0.9 (joy)")
    print(f"  Empathy intensity: {joy_response.empathy_strength:.4f}")
    print(f"  Resonance valence: {joy_response.resonated_valence:.4f}")
    print(f"  Empathy type: {joy_response.empathy_type}")
    passed &= check("Joy observation generates positive resonance", joy_response.resonated_valence > 0,
                     f"valence = {joy_response.resonated_valence:.4f}")

    # === 2c: Emotion intensity positive correlation ===
    subheader("2c: Emotion intensity antagonistic comparison")
    engine_weak = MirrorNeuronEngine()
    for _ in range(30):
        engine_weak.mature(social_interaction=True)
    weak = engine_weak.observe_emotion("person_a", -0.2, 0.3)
    engine_strong = MirrorNeuronEngine()
    for _ in range(30):
        engine_strong.mature(social_interaction=True)
    strong = engine_strong.observe_emotion("person_b", -0.9, 0.9)
    print(f"  Weak emotion (v=-0.2): empathy = {weak.empathy_strength:.4f}")
    print(f"  Strong emotion (v=-0.9): empathy = {strong.empathy_strength:.4f}")
    passed &= check("Strong emotion > weak emotion empathy", strong.empathy_strength > weak.empathy_strength,
                     f"{weak.empathy_strength:.4f} vs {strong.empathy_strength:.4f}")

    return passed


# ============================================================================
# Experiment 3: Empathy Development
# ============================================================================


def exp3_empathy_development() -> bool:
    """
    Empathy capacity matures with social experience
    Infant: only has emotional contagion (motor empathy)
    Child: affective empathy
    Adult: cognitive empathy + Theory of Mind
    """
    header("Experiment 3: Empathy Development -- From Infant to Adult")

    engine = MirrorNeuronEngine()
    passed = True

    # === 3a: Infant period ===
    subheader("3a: Infant period -- only basic emotional contagion")
    initial_empathy = engine.get_empathy_capacity()
    initial_tom = engine.get_tom_capacity()
    print(f"  Empathy capacity: {initial_empathy:.3f}")
    print(f"  Theory of Mind: {initial_tom:.3f}")
    passed &= check("Initial empathy capacity is low", initial_empathy < 0.3,
                     f"capacity = {initial_empathy:.3f}")
    passed &= check("Initial Theory of Mind is low", initial_tom < 0.2,
                     f"tom = {initial_tom:.3f}")

    # Infant empathy type
    baby_response = engine.observe_emotion("mother", -0.7, 0.8)
    print(f"  Infant empathy type: {baby_response.empathy_type}")
    passed &= check("Infant shows motor empathy", baby_response.empathy_type == "motor",
                     f"type = {baby_response.empathy_type}")

    # === 3b: Social experience accumulation ===
    subheader("3b: Growth after 200 social interactions")
    empathy_trace = []
    tom_trace = []
    checkpoints = [0, 50, 100, 150, 200]
    idx = 0

    for i in range(200):
        engine.mature(social_interaction=True, positive_feedback=(i % 3 == 0))
        if idx < len(checkpoints) and i == checkpoints[idx]:
            empathy_trace.append(engine.get_empathy_capacity())
            tom_trace.append(engine.get_tom_capacity())
            idx += 1

    # final values
    empathy_trace.append(engine.get_empathy_capacity())
    tom_trace.append(engine.get_tom_capacity())
    checkpoints.append(200)

    print(f"\n {'Interactions':>10} {'Empathy':>8} {'ToM':>8}")
    print(f"  {'---':>10}  {'---':>8}  {'---':>8}")
    for cp, emp, tom in zip(checkpoints, empathy_trace, tom_trace):
        emp_bar = "#" * int(emp * 20)
        print(f"  {cp:>10}  {emp:>8.3f}  {tom:>8.3f}  {emp_bar}")

    final_empathy = engine.get_empathy_capacity()
    final_tom = engine.get_tom_capacity()
    passed &= check("Empathy capacity grew", final_empathy > initial_empathy,
                     f"{initial_empathy:.3f} -> {final_empathy:.3f}")
    passed &= check("Theory of Mind grew", final_tom > initial_tom,
                     f"{initial_tom:.3f} -> {final_tom:.3f}")

    # === 3c: Empathy type after maturation ===
    subheader("3c: Empathy type after maturation")
    mature_response = engine.observe_emotion("stranger", -0.5, 0.6)
    print(f"  Mature empathy type: {mature_response.empathy_type}")
    print(f"  Empathy intensity: {mature_response.empathy_strength:.4f}")
    passed &= check("Mature type is cognitive empathy",
                     mature_response.empathy_type == "cognitive",
                     f"type = {mature_response.empathy_type}")

    return passed


# ============================================================================
# Experiment 4: Intent Inference (Theory of Mind)
# ============================================================================


def exp4_intent_inference() -> bool:
    """
    Observe behavior sequences → infer intentions

    Expected:
    - Approach behaviors → infer 'approach' intent
    - Communication behaviors → infer 'communicate' intent
    - Threat behaviors → infer 'threaten' intent
    - More evidence → higher confidence
    """
    header("Experiment 4: Intent Inference -- Theory of Mind")

    engine = MirrorNeuronEngine()
    # elevate ToM capacity
    for _ in range(100):
        engine.mature(social_interaction=True, positive_feedback=True)
    passed = True

    # === 4a: Approach intent ===
    subheader("4a: Observe approach behavior sequence")
    approach_behaviors = ["move_toward", "lean_in", "reach", "approach"]
    for b in approach_behaviors:
        engine.infer_intent("person_x", b)

    result_a = engine.infer_intent("person_x", "approach")
    print(f"  Behavior sequence: {approach_behaviors}")
    print(f"  Inferred intent: {result_a.inferred_goal}")
    print(f"  Confidence: {result_a.confidence:.4f}")
    print(f"  Evidence: {result_a.evidence}")
    passed &= check("Approach sequence -> approach",
                     result_a.inferred_goal == "approach",
                     f"goal = {result_a.inferred_goal}")

    # === 4b: Communication intent ===
    subheader("4b: Observe communication behavior sequence")
    engine2 = MirrorNeuronEngine()
    for _ in range(100):
        engine2.mature(social_interaction=True, positive_feedback=True)
    comm_behaviors = ["vocalize", "gesture", "point", "wave", "speak"]
    for b in comm_behaviors:
        engine2.infer_intent("person_y", b)

    result_b = engine2.infer_intent("person_y", "vocalize")
    print(f"  Behavior sequence: {comm_behaviors}")
    print(f"  Inferred intent: {result_b.inferred_goal}")
    print(f"  Confidence: {result_b.confidence:.4f}")
    passed &= check("Communication sequence -> communicate",
                     result_b.inferred_goal == "communicate",
                     f"goal = {result_b.inferred_goal}")

    # === 4c: Threat intent ===
    subheader("4c: Observe threat behavior sequence")
    engine3 = MirrorNeuronEngine()
    for _ in range(100):
        engine3.mature(social_interaction=True, positive_feedback=True)
    # first set up negative emotions
    engine3.observe_emotion("person_z", -0.6, 0.9)
    threat_behaviors = ["growl", "loom", "sudden_move", "loud_sound"]
    for b in threat_behaviors:
        engine3.infer_intent("person_z", b)

    result_c = engine3.infer_intent("person_z", "growl")
    print(f"  Behavior sequence: {threat_behaviors}")
    print(f"  Inferred intent: {result_c.inferred_goal}")
    print(f"  Confidence: {result_c.confidence:.4f}")
    passed &= check("Threat sequence -> threaten",
                     result_c.inferred_goal == "threaten",
                     f"goal = {result_c.inferred_goal}")

    # === 4d: More evidence → higher confidence ===
    subheader("4d: Evidence accumulation -> confidence elevation")
    engine4 = MirrorNeuronEngine()
    for _ in range(100):
        engine4.mature(social_interaction=True, positive_feedback=True)
    confidences = []
    for i in range(8):
        r = engine4.infer_intent("person_w", "demonstrate")
        confidences.append(r.confidence)

    print(f"  Inference 1 confidence: {confidences[0]:.4f}")
    print(f"  Inference 8 confidence: {confidences[-1]:.4f}")
    passed &= check("Confidence increases with evidence", confidences[-1] >= confidences[0],
                     f"{confidences[0]:.4f} -> {confidences[-1]:.4f}")

    return passed


# ============================================================================
# Experiment 5: Social Bonding
# ============================================================================


def exp5_social_bonding() -> bool:
    """
    Social bonding = impedance matching

    Repeated interactions → impedance reduction → smoother signal transmission → increasing closeness
    Like two people becoming more familiar, communication needs less 'explanation'
    """
    header("Experiment 5: Social Bonding -- Interaction = Impedance Reduction = Closeness")

    engine = MirrorNeuronEngine()
    passed = True

    # === 5a: Initial impedance ===
    subheader("5a: Initial social impedance")
    engine.observe_action("friend_a", "vocal", 80.0, "hello")
    engine.observe_action("friend_b", "vocal", 80.0, "hello")

    bonds = engine.get_social_bonds()
    print(f"  friend_a impedance: {bonds['friend_a']}ohm")
    print(f"  friend_b impedance: {bonds['friend_b']}ohm")
    initial_bond_a = bonds["friend_a"]

    # === 5b: Repeated interaction → impedance reduction ===
    subheader("5b: 50 interactions with friend_a")
    bond_trace = [initial_bond_a]
    for i in range(50):
        engine.observe_action("friend_a", "vocal",
                              MOTOR_MIRROR_IMPEDANCE + np.random.normal(0, 5),
                              "chat")
        engine.observe_emotion("friend_a", 0.3, 0.5)
        if (i + 1) % 10 == 0:
            bond_trace.append(engine.get_social_bonds()["friend_a"])

    print(f"\n {'Interactions':>10} {'Impedance(ohm)':>10}")
    print(f"  {'---':>10}  {'---':>10}")
    for idx, b in enumerate(bond_trace):
        cp = idx * 10
        bar = "#" * max(1, int(b / 2))
        print(f"  {cp:>10}  {b:>10.2f}  {bar}")

    final_bond_a = engine.get_social_bonds()["friend_a"]
    final_bond_b = engine.get_social_bonds()["friend_b"]

    passed &= check("Impedance reduced after interaction", final_bond_a < initial_bond_a,
                     f"{initial_bond_a:.2f} -> {final_bond_a:.2f}")
    passed &= check("Non-interacted agent has higher impedance", final_bond_b > final_bond_a,
                     f"a={final_bond_a:.2f} vs b={final_bond_b:.2f}")

    # === 5c: Familiarity ===
    subheader("5c: Familiarity")
    model_a = engine.get_agent_model("friend_a")
    model_b = engine.get_agent_model("friend_b")
    assert model_a is not None and model_b is not None
    print(f"  friend_a familiarity: {model_a.familiarity:.3f} ({model_a.interaction_count} interactions)")
    print(f"  friend_b familiarity: {model_b.familiarity:.3f} ({model_b.interaction_count} interactions)")
    passed &= check("More interactions = more familiar", model_a.familiarity > model_b.familiarity,
                     f"a={model_a.familiarity:.3f} vs b={model_b.familiarity:.3f}")

    return passed


# ============================================================================
# Experiment 6: AliceBrain Social Perception
# ============================================================================


def exp6_alice_social() -> bool:
    """
    Full AliceBrain encountering another Agent:
    1. Observe the other's actions and emotions
    2. Generate mirror resonance and empathy
    3. Create social bonds
    """
    header("Experiment 6: AliceBrain Social Perception -- Meeting Others")

    from alice import AliceBrain
    from alice.core.protocol import Modality

    brain = AliceBrain()
    passed = True

    # === Phase 1: Meeting caregiver ===
    subheader("Phase 1: Meeting caregiver -- observing actions")
    # Caregiver speaking (Alice observes motor pattern similar to her own vocal cords)
    for i in range(5):
        brain.mirror_neurons.observe_action(
            agent_id="caregiver",
            modality="vocal",
            observed_impedance=70.0 + np.random.normal(0, 3),
            action_label="speak_to_baby",
        )
        brain.mirror_neurons.tick(has_social_input=True)

    imitation = brain.mirror_neurons.get_imitation_urge()
    print(f"  Imitation urge: {imitation:.4f}")
    passed &= check("Observing speech generates imitation urge", imitation > 0,
                     f"urge = {imitation:.4f}")

    # === Phase 2: Sensing caregiver emotions ===
    subheader("Phase 2: Sensing caregiver emotions")
    # Caregiver smiling (positive emotion)
    for i in range(10):
        brain.mirror_neurons.observe_emotion(
            agent_id="caregiver",
            observed_valence=0.7,
            observed_arousal=0.5,
            modality="facial",
        )
        brain.mirror_neurons.tick(has_social_input=True)

    empathic_v = brain.mirror_neurons.get_empathic_valence()
    print(f"  Resonance valence: {empathic_v:.4f} (caregiver smiling)")
    passed &= check("Sensed positive emotion", empathic_v > 0,
                     f"valence = {empathic_v:.4f}")

    # === Phase 3: Social bond formation ===
    subheader("Phase 3: Social bond formation")
    bond = brain.mirror_neurons.get_social_bonds()
    print(f"  Social bonds: {bond}")
    caregiver_bond = bond.get("caregiver", SOCIAL_BOND_BASE)
    passed &= check("Bond created with caregiver", caregiver_bond < SOCIAL_BOND_BASE,
                     f"bond Z = {caregiver_bond:.2f} < {SOCIAL_BOND_BASE}")

    # === Phase 4: Theory of Mind ===
    subheader("Phase 4: Inferring caregiver intent")
    # Input enough 'teaching' evidence, to exceed earlier speak_to_baby generated communicate evidence
    teach_behaviors = [
        "demonstrate", "slow_down", "repeat", "point_at",
        "demonstrate", "repeat", "slow_down", "demonstrate",
    ]
    for b in teach_behaviors:
        brain.mirror_neurons.infer_intent("caregiver", b)
    intent = brain.mirror_neurons.infer_intent("caregiver", "demonstrate")
    print(f"  Inferred intent: {intent.inferred_goal}")
    print(f"  Confidence: {intent.confidence:.4f}")
    passed &= check("Inferred teaching intent", intent.inferred_goal == "teach",
                     f"goal = {intent.inferred_goal}")

    # === Phase 5: Full state report ===
    subheader("Phase 5: Mirror system full state")
    state = brain.mirror_neurons.get_state()
    print(f"  Empathy capacity: {state['empathy_capacity']:.4f}")
    print(f"  Theory of Mind: {state['tom_capacity']:.4f}")
    print(f"  Tracked agents: {state['tracked_agents']}")
    print(f"  Motor models: {state['motor_models']}")
    for aid, info in state["agent_models"].items():
        print(f"  [{aid}] goal={info['goal']}, emotion={info['emotion']}, "
              f"bond={info['bond_impedance']}ohm, familiarity={info['familiarity']}")

    stats = brain.mirror_neurons.get_stats()
    print(f"\n  Total mirror events: {stats['total_mirror_events']}")
    print(f"  Total empathy responses: {stats['total_empathy_responses']}")
    print(f"  Total intent inferences: {stats['total_intent_inferences']}")

    passed &= check("Mirror system operating normally", stats['total_mirror_events'] > 0,
                     f"events = {stats['total_mirror_events']}")

    return passed


# ============================================================================
# main program
# ============================================================================


def main():
    print()
    print("+======================================================================+")
    print("| Phase 6.1: Mirror Neurons, Empathy and Theory of Mind -- Experiment Report |")
    print("|                                                                       |")
    print("| 'Empathy is not imitation -- it is impedance resonance'               |")
    print("+======================================================================+")

    experiments = [
        ("Experiment 1: Motor Mirroring", exp1_motor_mirror),
        ("Experiment 2: Emotional Mirroring", exp2_emotional_mirror),
        ("Experiment 3: Empathy Development", exp3_empathy_development),
        ("Experiment 4: Intent Inference (ToM)", exp4_intent_inference),
        ("Experiment 5: Social Bonding", exp5_social_bonding),
        ("Experiment 6: AliceBrain Social", exp6_alice_social),
    ]

    results = []
    for name, func in experiments:
        try:
            result = func()
        except Exception as e:
            print(f"\n  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            result = False
        results.append((name, result))

    # === Summary ===
    print()
    print("=" * 70)
    print("  Final Results")
    print("=" * 70)

    all_passed = True
    for name, result in results:
        symbol = "[PASS]" if result else "[FAIL]"
        print(f"  [{symbol}] {name}")
        all_passed &= result

    total_pass = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\n  Total: {total_pass}/{total} PASS")

    if all_passed:
        print()
        print("  +-----------------------------------------------------+")
        print("  |  Phase 6.1 verification completed                             |")
        print("  |                                                      |")
        print("  |  Alice demonstrates:                                 |")
        print("  |  - Motor mirroring (observing = brain doing once)    |")
        print("  |  - Emotional resonance (seeing pain feels painful)   |")
        print("  |  - Empathy development (from contagion to cognitive) |")
        print("  |  - Intent inference (Theory of Mind)                 |")
        print("  |  - Social bonding (interaction = impedance reduction)|")
        print("  |                                                      |")
        print("  |  Alice is no longer alone -- she understands others. |")
        print("  +-----------------------------------------------------+")
    else:
        print("\n  !! Some experiments did not pass")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
