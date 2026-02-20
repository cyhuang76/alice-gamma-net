# -*- coding: utf-8 -*-
"""
Phase 22 Experiment: Homeostatic Drive + Physics Reward + End-to-End Lifecycle
HomeostaticDriveEngine + PhysicsRewardEngine + E2E Lifecycle
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alice.alice_brain import AliceBrain
from alice.brain.homeostatic_drive import HomeostaticDriveEngine
from alice.brain.physics_reward import PhysicsRewardEngine
from alice.core.protocol import Modality


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# =================================================================
# Experiment 1: Hunger drive accumulates over time
# =================================================================
def exp_hunger_accumulation():
    separator("Exp 1: Hunger Accumulation Over Time")
    hd = HomeostaticDriveEngine()
    data = []
    for t in range(200):
        sig = hd.tick(sympathetic=0.3, cognitive_load=0.4)
        data.append((t, hd.glucose, hd.hunger_drive, sig.needs_food))
        if sig.needs_food and t > 100:
            hd.eat()
            print(f"  [t={t}] 🍔 EATING! glucose={hd.glucose:.3f}")

    print(f"  Final glucose:  {hd.glucose:.4f}")
    print(f"  Final hunger:   {hd.hunger_drive:.4f}")
    print(f"  Total ticks:    {hd._total_ticks}")
    print(f"  Starvation tks: {hd._starvation_ticks}")
    return data


# =================================================================
# Experiment 2: Thirst drive and dehydration pain
# =================================================================
def exp_thirst_dehydration():
    separator("Exp 2: Thirst Drive & Dehydration Pain")
    hd = HomeostaticDriveEngine()
    for t in range(300):
        sig = hd.tick(sympathetic=0.2, core_temp=38.0)
        if t % 50 == 0:
            print(f"  [t={t:3d}] hydration={hd.hydration:.3f}  "
                  f"thirst={hd.thirst_drive:.3f}  "
                  f"pain={sig.pain_contribution:.3f}")
        if sig.needs_water and t > 150:
            hd.drink()
            print(f"  [t={t}] 💧 DRINKING! hydration={hd.hydration:.3f}")


# =================================================================
# Experiment 3: Sympathetic nervous system accelerates metabolism
# =================================================================
def exp_sympathetic_metabolism():
    separator("Exp 3: Sympathetic Accelerated Metabolism")
    hd_calm = HomeostaticDriveEngine()
    hd_stress = HomeostaticDriveEngine()

    for t in range(100):
        hd_calm.tick(sympathetic=0.1)
        hd_stress.tick(sympathetic=0.9)

    print(f"  Calm glucose:   {hd_calm.glucose:.4f}")
    print(f"  Stress glucose: {hd_stress.glucose:.4f}")
    print(f"  Δ glucose:      {hd_calm.glucose - hd_stress.glucose:.4f}")


# =================================================================
# Experiment 4: Physics reward — impedance learning curve
# =================================================================
def exp_impedance_learning_curve():
    separator("Exp 4: Impedance Learning Curve")
    pr = PhysicsRewardEngine()

    z_history = []
    for epoch in range(50):
        rpe = pr.update("kitchen", "cook", 1.0, "meal_ready")
        ch = pr._channels[("kitchen", "cook")]
        z_history.append(ch.impedance)
        if epoch % 10 == 0:
            print(f"  [epoch={epoch:2d}] Z={ch.impedance:.2f}  "
                  f"Γ={ch.gamma:.4f}  T={ch.transmission:.4f}  RPE={rpe:.4f}")

    print(f"\n  Impedance drop: {z_history[0]:.2f} → {z_history[-1]:.2f}")
    print(f"  Final transmission: {pr._channels[('kitchen','cook')].transmission:.4f}")


# =================================================================
# Experiment 5: Boltzmann action selection distribution
# =================================================================
def exp_boltzmann_distribution():
    separator("Exp 5: Boltzmann Action Selection Distribution")
    pr = PhysicsRewardEngine()

    # Train each action to different degrees
    for _ in range(20):
        pr.update("s1", "best", 1.5, "s2")
    for _ in range(10):
        pr.update("s1", "ok", 0.5, "s2")
    for _ in range(5):
        pr.update("s1", "bad", -0.5, "s2")

    counts = {"best": 0, "ok": 0, "bad": 0}
    for _ in range(1000):
        action, _ = pr.choose_action("s1", ["best", "ok", "bad"])
        counts[action] += 1

    print(f"  Selection over 1000 trials:")
    for a, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {a:>6}: {c:4d} ({c/10:.1f}%)")


# =================================================================
# Experiment 6: Unified dopamine pipeline
# =================================================================
def exp_dopamine_pipeline():
    separator("Exp 6: Unified Dopamine Pipeline")
    brain = AliceBrain(neuron_count=10)

    bg_da_before = brain.basal_ganglia._dopamine_level
    result = brain.learn_from_feedback("s1", "a1", 2.0, "s2")
    bg_da_after = brain.basal_ganglia._dopamine_level

    print(f"  Basal-ganglia DA before:   {bg_da_before:.4f}")
    print(f"  Basal-ganglia DA after:    {bg_da_after:.4f}")
    print(f"  PhysicsReward DA:          {brain.physics_reward._dopamine:.4f}")
    print(f"  RPE from learn_feedback:   {result['dopamine_signal']:.4f}")
    print(f"  DA injected into BG:       {bg_da_after - bg_da_before:.4f}")


# =================================================================
# Experiment 7: Homeostatic ↔ Autonomic interaction
# =================================================================
def exp_homeostatic_autonomic_interaction():
    separator("Exp 7: Homeostatic ↔ Autonomic Interaction")
    brain = AliceBrain(neuron_count=10)

    brain.homeostatic_drive.glucose = 0.2  # Very hungry
    result = brain.perceive(np.random.randn(64), Modality.AUDITORY)

    print(f"  Glucose:          {result['homeostatic_drive']['glucose']:.3f}")
    print(f"  Hunger drive:     {result['homeostatic_drive']['hunger_drive']:.3f}")
    print(f"  Cognitive penalty:{result['homeostatic_drive']['cognitive_penalty']:.3f}")
    print(f"  Irritability:     {result['homeostatic_drive'].get('irritability', 'N/A')}")

    # Normal state
    brain2 = AliceBrain(neuron_count=10)
    result2 = brain2.perceive(np.random.randn(64), Modality.AUDITORY)
    print(f"\n  [Normal] Glucose:  {result2['homeostatic_drive']['glucose']:.3f}")
    print(f"  [Normal] Penalty:  {result2['homeostatic_drive']['cognitive_penalty']:.3f}")


# =================================================================
# Experiment 8: Full lifecycle 200 ticks
# =================================================================
def exp_full_lifecycle_200():
    separator("Exp 8: Full Lifecycle 200-tick")
    brain = AliceBrain(neuron_count=10)

    phases = {"wake": 0, "hungry": 0, "fed": 0}
    for t in range(200):
        stimulus = np.random.randn(64) * 0.3
        result = brain.perceive(stimulus, Modality.AUDITORY)

        hd = result["homeostatic_drive"]
        if hd["needs_food"] and t > 80:
            brain.homeostatic_drive.eat()
            phases["fed"] += 1
        elif hd["hunger_drive"] > 0.1:
            phases["hungry"] += 1
        else:
            phases["wake"] += 1

        # Report every 50 ticks
        if t % 50 == 0:
            v = brain.vitals.get_vitals()
            print(f"  [t={t:3d}] glucose={hd['glucose']:.3f} "
                  f"hunger={hd['hunger_drive']:.3f} "
                  f"consciousness={v['consciousness']:.3f} "
                  f"energy={brain.autonomic.energy:.3f}")

    print(f"\n  Phase distribution: {phases}")
    stats = brain.physics_reward.get_stats()
    print(f"  Reward channels:   {stats['total_channels']}")


# =================================================================
# Experiment 9: Experience replay comparison (with vs without)
# =================================================================
def exp_replay_comparison():
    separator("Exp 9: Experience Replay Comparison")
    pr_replay = PhysicsRewardEngine()
    pr_no_replay = PhysicsRewardEngine()

    for _ in range(100):
        state = f"s{np.random.randint(5)}"
        action = f"a{np.random.randint(3)}"
        reward = np.random.randn()
        next_s = f"s{np.random.randint(5)}"
        pr_replay.update(state, action, reward, next_s)
        pr_no_replay.update(state, action, reward, next_s)

    # Replay
    for _ in range(10):
        pr_replay.replay()

    stats_r = pr_replay.get_stats()
    stats_n = pr_no_replay.get_stats()
    print(f"  With replay:    avg_Γ={stats_r['avg_gamma']:.4f}  "
          f"channels={stats_r['total_channels']}")
    print(f"  Without replay: avg_Γ={stats_n['avg_gamma']:.4f}  "
          f"channels={stats_n['total_channels']}")


# =================================================================
# Experiment 10: Hunger → Cognitive degradation → Eating → Recovery
# =================================================================
def exp_hunger_cognitive_cycle():
    separator("Exp 10: Hunger → Cognitive Degradation → Eat → Recovery")
    brain = AliceBrain(neuron_count=10)
    brain.homeostatic_drive.glucose = 0.15  # Extreme hunger

    print("  [Phase 1: Starving]")
    result = brain.perceive(np.random.randn(64), Modality.AUDITORY)
    print(f"    Cognitive penalty: {result['homeostatic_drive']['cognitive_penalty']:.3f}")

    brain.homeostatic_drive.eat()
    brain.homeostatic_drive.eat()  # Large food intake

    print("  [Phase 2: After eating]")
    for _ in range(30):
        result = brain.perceive(np.random.randn(64), Modality.AUDITORY)

    print(f"    Cognitive penalty: {result['homeostatic_drive']['cognitive_penalty']:.3f}")
    print(f"    Glucose recovered: {brain.homeostatic_drive.glucose:.3f}")

    print("\n  ✅ Phase 22 Experiments Complete")


# =================================================================
# Main
# =================================================================
if __name__ == "__main__":
    exp_hunger_accumulation()
    exp_thirst_dehydration()
    exp_sympathetic_metabolism()
    exp_impedance_learning_curve()
    exp_boltzmann_distribution()
    exp_dopamine_pipeline()
    exp_homeostatic_autonomic_interaction()
    exp_full_lifecycle_200()
    exp_replay_comparison()
    exp_hunger_cognitive_cycle()

    print(f"\n{'='*60}")
    print(f"  ALL 10 EXPERIMENTS COMPLETED")
    print(f"{'='*60}")
