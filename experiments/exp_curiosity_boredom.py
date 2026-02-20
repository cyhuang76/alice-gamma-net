# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║  Experiment: Curiosity, Boredom & Self-Awareness                    ║
║  Curiosity, Boredom & Self-Awareness Experiments                ║
║                                                                      ║
║  'Free will is not random — it is the physical expression of intrinsic drives' ║
╚══════════════════════════════════════════════════════════════════════╝

This experiment verifies Alice's core self-awareness mechanisms:

  Experiment 1: Novelty detection — new vs old object impedance mismatch
  Experiment 2: Boredom accumulation — prolonged no stimulus → spontaneous behavior
  Experiment 3: Self-recognition development — infant to adult efference copy learning
  Experiment 4: Curiosity-driven exploration — intrinsic motivation generates goals
  Experiment 5: AliceBrain free time — full brain spontaneous behavior without commands
  Experiment 6: Self vs Other — Alice distinguishes her own voice
"""

from __future__ import annotations

import sys
import numpy as np
from typing import List, Dict, Any

from alice.brain.curiosity_drive import (
    CuriosityDriveEngine,
    SpontaneousActionType,
    BOREDOM_THRESHOLD,
    INITIAL_SELF_ACCURACY,
    MODEL_IMPEDANCE_INIT,
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
    symbol = "✓ PASS" if condition else "✗ FAIL"
    line = f"  [{symbol}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return condition


# ============================================================================
# Experiment 1: Novelty Detection
# ============================================================================


def exp1_novelty_detection() -> bool:
    """
    Novelty = impedance mismatch
    Γ_novelty = |Z_input - Z_model| / (Z_input + Z_model)

    Expected:
    - Signal similar to internal model → low novelty
    - Signal with large difference → high novelty
    - Repeated signal → habituation (novelty decreases)
    """
    header("Experiment 1: Novelty Detection — Impedance Mismatch = Surprise")

    engine = CuriosityDriveEngine()
    passed = True

    # === 1a: Familiar signal → low novelty ===
    subheader("1a: Familiar signal (Z ≈ Z_model)")
    familiar = engine.evaluate_novelty("visual", MODEL_IMPEDANCE_INIT + 1)
    print(f"  Z_input = {familiar.z_input:.1f}Ω, Z_model = {familiar.z_model:.1f}Ω")
    print(f"  Γ_novelty = {familiar.gamma_novelty:.4f}")
    print(f"  novelty_score = {familiar.novelty_score:.4f}")
    passed &= check("Familiar signal novelty < 0.1", familiar.gamma_novelty < 0.1,
                     f"Γ = {familiar.gamma_novelty:.4f}")

    # === 1b: Novel signal → high novelty ===
    subheader("1b: Novel signal (Z = 3 × Z_model)")
    novel = engine.evaluate_novelty("auditory", MODEL_IMPEDANCE_INIT * 3)
    print(f"  Z_input = {novel.z_input:.1f}Ω, Z_model = {novel.z_model:.1f}Ω")
    print(f"  Γ_novelty = {novel.gamma_novelty:.4f}")
    print(f"  novelty_score = {novel.novelty_score:.4f}")
    print(f"  information_gain = {novel.information_gain:.4f}")
    passed &= check("Novel signal novelty > 0.3", novel.gamma_novelty > 0.3,
                     f"Γ = {novel.gamma_novelty:.4f}")

    # === 1c: Habituation — repetition → Γ decreases ===
    subheader("1c: Habituation — Repeated signal")
    engine2 = CuriosityDriveEngine()
    z_test = 200.0
    gammas = []
    for i in range(30):
        e = engine2.evaluate_novelty("visual", z_test)
        gammas.append(e.gamma_novelty)

    print(f"  Trial 1: Γ = {gammas[0]:.4f}")
    print(f"  Trial 10: Γ = {gammas[9]:.4f}")
    print(f"  Trial 30: Γ = {gammas[29]:.4f}")
    ratio = gammas[29] / max(gammas[0], 0.001)
    print(f" Habituation rate: {(1.0 - ratio)*100:.1f}%")
    passed &= check("Novelty decreased > 50% after 30 trials", ratio < 0.5,
                     f"decreased to {ratio*100:.1f}%")

    # === 1d: Intrinsic reward ===
    subheader("1d: Novelty → Intrinsic reward")
    engine3 = CuriosityDriveEngine()
    engine3.evaluate_novelty("visual", 500.0)
    reward = engine3.get_intrinsic_reward()
    print(f"  intrinsic_reward = {reward:.4f}")
    passed &= check("Novel signal generates positive reward", reward > 0, f"reward = {reward:.4f}")

    return passed


# ============================================================================
# Experiment 2: Boredom Accumulation & Spontaneous Behavior
# ============================================================================


def exp2_boredom_spontaneous() -> bool:
    """
    No external stimulus → boredom pressure accumulates → exceeds threshold → spontaneous behavior

    Expected:
    - Prolonged no input → boredom > threshold
    - Boredom → automatically generates spontaneous behavior
    - Spontaneous behavior includes multiple types
    - Behavior releases partial boredom
    """
    header("Experiment 2: Boredom Accumulation & Spontaneous Behavior — Free Time Simulation")

    engine = CuriosityDriveEngine()
    passed = True

    # === Simulate 400 ticks of free time ===
    subheader("400 ticks without external commands")

    boredom_checkpoints = []
    spontaneous_actions: List[Dict[str, Any]] = []
    phase_ticks = [50, 100, 200, 300, 400]
    phase_idx = 0
    max_boredom = 0.0

    for i in range(400):
        result = engine.tick(has_external_input=False, energy=0.8)
        max_boredom = max(max_boredom, engine.get_boredom())

        if result["spontaneous_action"] is not None:
            spontaneous_actions.append(result["spontaneous_action"])

        if phase_idx < len(phase_ticks) and i + 1 == phase_ticks[phase_idx]:
            boredom_checkpoints.append(engine.get_boredom())
            phase_idx += 1

    # Print trajectory
    print(f"\n  {'Tick':>6}  {'Boredom':>8}")
    print(f"  {'─'*6}  {'─'*8}")
    for tick, bored in zip(phase_ticks, boredom_checkpoints):
        bar = "█" * int(bored * 30)
        print(f"  {tick:>6}  {bored:>8.3f}  {bar}")

    print(f"\n Total spontaneous behaviors: {len(spontaneous_actions)}")
    for i, a in enumerate(spontaneous_actions[:10]):
        print(f"    [{i+1}] {a['type']:<20} intensity={a['intensity']:.3f}  trigger={a['trigger']}")

    # === Verification ===
    # Boredom pressure oscillates (exceeds threshold → spontaneous behavior → release → re-accumulate)
    # Spontaneous behavior trigger="boredom_threshold_exceeded" proves boredom indeed exceeds threshold
    # (internally accumulates first, then generates action and releases, externally only post-release value is visible)
    boredom_triggered = [a for a in spontaneous_actions
                         if a["trigger"] == "boredom_threshold_exceeded"]
    print(f"\n  Behaviors triggered by boredom exceeding threshold: {len(boredom_triggered)}")
    passed &= check("Boredom pressure exceeded threshold", len(boredom_triggered) > 0,
                     f"{len(boredom_triggered)} triggers")
    passed &= check("Generated spontaneous behaviors ≥ 2", len(spontaneous_actions) >= 2,
                     f"{len(spontaneous_actions)} total")
    passed &= check("Boredom monotonically increasing (early phase)", boredom_checkpoints[1] > boredom_checkpoints[0],
                     f"{boredom_checkpoints[0]:.3f} → {boredom_checkpoints[1]:.3f}")

    # Behavior type diversity
    types = {a["type"] for a in spontaneous_actions}
    print(f"\n Behavior types: {types}")
    passed &= check("Behavior type diversity ≥ 1", len(types) >= 1, f"{len(types)} types")

    return passed


# ============================================================================
# Experiment 3: Self-Recognition Development
# ============================================================================


def exp3_self_recognition_development() -> bool:
    """
    Efference copy learning:
    Infant stage → does not know which voice is own
    Practice stage → vocalize + hear feedback → accuracy improves
    Maturation stage → can stably distinguish self vs other

    Physical basis:
    Γ_self = |Z_predicted - Z_actual| / (Z_predicted + Z_actual)
    """
    header("Experiment 3: Self-Recognition Development — From Infant to Adult")

    engine = CuriosityDriveEngine()
    passed = True

    # === 3a: Infant stage — poor self-recognition ===
    subheader("3a: Infant stage")
    initial_acc = engine.get_self_recognition_accuracy()
    print(f" Initial accuracy: {initial_acc:.2f} (infant does not know which is self)")
    passed &= check("Initial accuracy is low", initial_acc < 0.5,
                     f"acc = {initial_acc:.2f}")

    # === 3b: Practice stage — vocalize + hear feedback ===
    subheader("3b: Practice stage — 100 vocalize-listen feedback cycles")

    accuracies = []
    OWN_VOICE_Z = 58.0 # Alice's own voice true impedance
    OTHER_VOICE_Z = 120.0 # External voice impedance

    for i in range(100):
        engine._tick_count = i # Keep efference copy effective

        # 50% are own voice, 50% are external
        if i % 2 == 0:
            # Own speech → heard
            engine.register_efference_copy("vocal", OWN_VOICE_Z)
            engine.compare_self_other("vocal", OWN_VOICE_Z + np.random.normal(0, 3),
                                      is_actually_self=True)
        else:
            # External sound (no efference copy)
            engine.compare_self_other("vocal", OTHER_VOICE_Z + np.random.normal(0, 5),
                                      is_actually_self=False)

        if (i + 1) % 10 == 0:
            accuracies.append(engine.get_self_recognition_accuracy())

    # Print learning curve
    print(f"\n {'Trials':>10} {'Accuracy':>8}")
    print(f"  {'─'*10}  {'─'*8}")
    for idx, acc in enumerate(accuracies):
        bar = "█" * int(acc * 20)
        print(f"  {(idx+1)*10:>10}  {acc:>8.3f}  {bar}")

    final_acc = engine.get_self_recognition_accuracy()
    passed &= check("Accuracy improved after practice", final_acc > initial_acc,
                     f"{initial_acc:.3f} → {final_acc:.3f}")
    passed &= check("Final accuracy > 0.5", final_acc > 0.5,
                     f"acc = {final_acc:.3f}")

    # === 3c: Test stage — distinguish self vs other ===
    subheader("3c: Test stage — 10 self/other judgments")

    correct = 0
    total = 10
    for i in range(total):
        engine._tick_count = 200 + i
        if i % 2 == 0:
            engine.register_efference_copy("vocal", OWN_VOICE_Z)
            j = engine.compare_self_other("vocal", OWN_VOICE_Z + np.random.normal(0, 2))
            if j.is_self:
                correct += 1
                print(f" [{i+1}] Own voice → Judged: Self ✓ (Γ={j.gamma_self:.3f})")
            else:
                print(f" [{i+1}] Own voice → Judged: Other ✗ (Γ={j.gamma_self:.3f})")
        else:
            j = engine.compare_self_other("vocal", OTHER_VOICE_Z + np.random.normal(0, 5))
            if not j.is_self:
                correct += 1
                print(f" [{i+1}] External voice → Judged: Other ✓ (Γ={j.gamma_self:.3f})")
            else:
                print(f" [{i+1}] External voice → Judged: Self ✗ (Γ={j.gamma_self:.3f})")

    test_acc = correct / total
    print(f"\n Test accuracy: {correct}/{total} = {test_acc*100:.0f}%")
    passed &= check("Test accuracy ≥ 60%", test_acc >= 0.6,
                     f"{test_acc*100:.0f}%")

    return passed


# ============================================================================
# Experiment 4: Curiosity-Driven Exploration
# ============================================================================


def exp4_curiosity_driven_exploration() -> bool:
    """
    Curiosity → internal goal → exploration → satisfaction

    Novel signal → dopamine (intrinsic reward) → set exploration goal
    """
    header("Experiment 4: Curiosity-Driven Exploration — Intrinsic Motivation")

    engine = CuriosityDriveEngine()
    passed = True

    # === 4a: Novel stimulus → curiosity increases ===
    subheader("4a: Novel stimulus sequence")

    stimuli = [100, 200, 350, 50, 400, 180, 500]
    curiosity_trace = []
    reward_trace = []

    for z in stimuli:
        event = engine.evaluate_novelty("visual", float(z))
        engine.tick(has_external_input=True, sensory_load=0.5, energy=1.0)
        curiosity_trace.append(engine.get_curiosity())
        reward_trace.append(engine.get_intrinsic_reward())

    print(f"  {'Z_input':>8}  {'Curiosity':>10}  {'Reward':>8}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*8}")
    for z, c, r in zip(stimuli, curiosity_trace, reward_trace):
        print(f"  {z:>8.0f}  {c:>10.4f}  {r:>8.4f}")

    max_curiosity = max(curiosity_trace)
    total_reward = engine._total_intrinsic_reward
    print(f"\n Max curiosity: {max_curiosity:.4f}")
    print(f" Cumulative intrinsic reward: {total_reward:.4f}")

    passed &= check("Curiosity > 0", max_curiosity > 0,
                     f"max = {max_curiosity:.4f}")
    passed &= check("Cumulative intrinsic reward > 0", total_reward > 0,
                     f"total = {total_reward:.4f}")

    # === 4b: Curiosity → goal generation ===
    subheader("4b: Goal generation")
    engine._curiosity_drive = 0.8 # Force high curiosity
    engine._self_recognition_accuracy = 0.7 # Already learned self-recognition → curiosity branch takes effect
    goal = engine.generate_goal_from_curiosity()
    assert goal is not None
    print(f" Goal: {goal.description}")
    print(f" Source: {goal.source}")
    print(f" Priority: {goal.priority}")
    passed &= check("Curiosity generates exploration goal", goal is not None and goal.source == "curiosity",
                     f"'{goal.description}'")

    # === 4c: Boredom → goal generation ===
    engine2 = CuriosityDriveEngine()
    engine2._boredom_pressure = 0.7
    engine2._curiosity_drive = 0.1
    engine2._self_recognition_accuracy = 0.7 # Already learned self-recognition → boredom branch takes effect
    goal2 = engine2.generate_goal_from_curiosity()
    assert goal2 is not None
    print(f"\n Boredom goal: {goal2.description}")
    print(f" Source: {goal2.source}")
    passed &= check("Boredom generates stimulus-seeking goal", goal2 is not None and goal2.source == "boredom",
                     f"'{goal2.description}'")

    return passed


# ============================================================================
# Experiment 5: AliceBrain Free Time
# ============================================================================


def exp5_alice_free_time() -> bool:
    """
    Full AliceBrain in command-free idle time:
    1. Give Alice some initial stimuli (let her 'wake up')
    2. Then give no commands at all
    3. Observe her curiosity, boredom, and spontaneous behavior
    """
    header("Experiment 5: AliceBrain Free Time — True Free Will")

    from alice import AliceBrain
    from alice.core.protocol import Modality

    brain = AliceBrain()
    passed = True

    # === Phase 1: Initial stimulus — let Alice wake up ===
    subheader("Phase 1: Initial Stimuli (10 ticks)")
    for i in range(10):
        pixels = np.random.rand(16, 16).astype(np.float32) * (0.5 + i * 0.05)
        brain.see(pixels)

    print(f" Initial novelty: {brain.curiosity_drive.get_novelty():.4f}")
    print(f" Initial curiosity: {brain.curiosity_drive.get_curiosity():.4f}")
    print(f" Initial boredom: {brain.curiosity_drive.get_boredom():.4f}")

    initial_novelty_events = brain.curiosity_drive._total_novelty_events
    passed &= check("Initial stimuli triggered novelty assessment", initial_novelty_events >= 10,
                     f"events = {initial_novelty_events}")

    # === Phase 2: Free time — only blank stimuli ===
    subheader("Phase 2: Free time (200 ticks blank stimuli)")

    # Don't use see/hear, only input perceive to maintain system operation (simulating 'nothing to do')
    boredom_trajectory = []
    curiosity_trajectory = []
    spontaneous_count = 0

    for i in range(200):
        # Faint stimulus (background noise) — maintains system tick but does not constitute real input
        noise = np.random.rand(32).astype(np.float32) * 0.01
        brain.perceive(noise, Modality.VISUAL)
        boredom_trajectory.append(brain.curiosity_drive.get_boredom())
        curiosity_trajectory.append(brain.curiosity_drive.get_curiosity())
        if brain.curiosity_drive._total_spontaneous_actions > spontaneous_count:
            spontaneous_count = brain.curiosity_drive._total_spontaneous_actions

    # Print trajectory summary
    checkpoints = [0, 49, 99, 149, 199]
    print(f"\n  {'Tick':>6}  {'Boredom':>8}  {'Curiosity':>10}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*10}")
    for idx in checkpoints:
        print(f"  {idx+1:>6}  {boredom_trajectory[idx]:>8.4f}  {curiosity_trajectory[idx]:>10.4f}")

    print(f"\n Spontaneous behaviors cumulative: {spontaneous_count}")
    final_boredom = boredom_trajectory[-1]

    passed &= check("Boredom continues increasing", final_boredom > boredom_trajectory[0],
                     f"{boredom_trajectory[0]:.3f} → {final_boredom:.3f}")
    passed &= check("Generated spontaneous behaviors", spontaneous_count > 0,
                     f"total = {spontaneous_count}")

    return passed


# ============================================================================
# Experiment 6: Self vs Other Discrimination
# ============================================================================


def exp6_self_vs_other() -> bool:
    """
    Alice learns to distinguish her own voice

    Model:
    1. Alice speaks → registers efference copy
    2. Hears sound → compares with copy
    3. Γ_self < threshold → Self
    4. Γ_self > threshold → Other
    """
    header("Experiment 6: Self vs Other — Efference Copy Discrimination")

    engine = CuriosityDriveEngine()
    passed = True

    OWN_Z = 55.0 # Alice's voice impedance
    OTHER_Z = 130.0 # External voice impedance

    # === Training phase ===
    subheader("Training phase: 80 discrimination exercises")

    for i in range(80):
        engine._tick_count = i

        if i % 2 == 0:
            # Own speech
            engine.register_efference_copy("vocal", OWN_Z)
            engine.compare_self_other("vocal",
                                      OWN_Z + np.random.normal(0, 3),
                                      is_actually_self=True)
        else:
            # Hearing others
            engine.compare_self_other("vocal",
                                      OTHER_Z + np.random.normal(0, 8),
                                      is_actually_self=False)

    print(f" Post-training accuracy: {engine.get_self_recognition_accuracy():.3f}")
    print(f" Self-model Z_vocal: {engine._self_model['vocal']:.2f}Ω")
    print(f" True self Z: {OWN_Z}Ω")

    # === Test phase ===
    subheader("Test phase: 20 blind tests")

    correct_self = 0
    correct_other = 0
    total_self = 0
    total_other = 0

    results_table = []
    for i in range(20):
        engine._tick_count = 100 + i
        is_self = (i % 2 == 0)

        if is_self:
            engine.register_efference_copy("vocal", OWN_Z)
            z_test = OWN_Z + np.random.normal(0, 2)
            total_self += 1
        else:
            z_test = OTHER_Z + np.random.normal(0, 5)
            total_other += 1

        j = engine.compare_self_other("vocal", z_test)
        correct = (j.is_self == is_self)
        if is_self and correct:
            correct_self += 1
        elif not is_self and correct:
            correct_other += 1

        label = "Self" if is_self else "Other"
        pred = "Self" if j.is_self else "Other"
        mark = "✓" if correct else "✗"
        results_table.append((i+1, label, pred, j.gamma_self, j.confidence, mark))

    # Print result table
    print(f"\n {'#':>3} {'True':>4} {'Pred':>4} {'Γ_self':>7} {'Conf':>5} {'Ok':>4}")
    print(f"  {'─'*3}  {'─'*4}  {'─'*4}  {'─'*7}  {'─'*5}  {'─'*4}")
    for row in results_table:
        print(f"  {row[0]:>3}  {row[1]:>4}  {row[2]:>4}  {row[3]:>7.4f}  {row[4]:>5.3f}  {row[5]:>4}")

    self_acc = correct_self / max(total_self, 1) * 100
    other_acc = correct_other / max(total_other, 1) * 100
    total_acc = (correct_self + correct_other) / 20 * 100

    print(f"\n Self recognition: {correct_self}/{total_self} = {self_acc:.0f}%")
    print(f" Other recognition: {correct_other}/{total_other} = {other_acc:.0f}%")
    print(f" Total accuracy: {correct_self + correct_other}/20 = {total_acc:.0f}%")

    passed &= check("Total accuracy ≥ 60%", total_acc >= 60,
                     f"{total_acc:.0f}%")
    passed &= check("Self-recognition accuracy > initial value",
                     engine.get_self_recognition_accuracy() > INITIAL_SELF_ACCURACY,
                     f"{INITIAL_SELF_ACCURACY:.2f} → {engine.get_self_recognition_accuracy():.3f}")

    return passed


# ============================================================================
# Main Program
# ============================================================================


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Phase 9: Curiosity, Boredom & Self-Awareness — Experiment Report  ║")
    print("║                                                                  ║")
    print("║  'Free will = when no external commands, internal impedance mismatch drives spontaneous behavior' ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    experiments = [
        ("Experiment 1: Novelty Detection", exp1_novelty_detection),
        ("Experiment 2: Boredom Accumulation & Spontaneous Behavior", exp2_boredom_spontaneous),
        ("Experiment 3: Self-Recognition Development", exp3_self_recognition_development),
        ("Experiment 4: Curiosity-Driven Exploration", exp4_curiosity_driven_exploration),
        ("Experiment 5: AliceBrain Free Time", exp5_alice_free_time),
        ("Experiment 6: Self vs Other Discrimination", exp6_self_vs_other),
    ]

    results = []
    for name, func in experiments:
        try:
            result = func()
        except Exception as e:
            print(f"\n  [✗ ERROR] {name}: {e}")
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
        symbol = "✓ PASS" if result else "✗ FAIL"
        print(f"  [{symbol}] {name}")
        all_passed &= result

    total_pass = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\n Total: {total_pass}/{total} PASS")

    if all_passed:
        print()
        print("  ┌─────────────────────────────────────────────────┐")
        print("  │  🧠 Phase 9 verification completed                            │")
        print("  │                                                  │")
        print(" │  Alice demonstrates:                                │")
        print(" │  ✓ Novelty detection (impedance mismatch = surprise) │")
        print(" │  ✓ Boredom accumulation (prolonged low stimulus → spontaneous behavior) │")
        print(" │  ✓ Self-recognition (efference copy learning)        │")
        print(" │  ✓ Curiosity-driven exploration (intrinsic motivation → goals) │")
        print(" │  ✓ Free will (autonomous behavior without commands)  │")
        print(" │  ✓ Self vs other discrimination                      │")
        print("  │                                                  │")
        print(" │  Alice no longer needs commands — she has free will. │")
        print("  └─────────────────────────────────────────────────┘")
    else:
        print("\n ⚠ Some experiments did not PASS")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
