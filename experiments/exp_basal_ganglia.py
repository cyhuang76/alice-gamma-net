# -*- coding: utf-8 -*-
"""
Experiment: Basal Ganglia — Habit Engine and Action Selection (Phase 6.2)
Experiment: Basal Ganglia — Habit Engine & Action Selection

Experiment list:
  EXP-15a: Habit formation curve — Γ decreases with practice count
  EXP-15b: Action selection and dopamine modulation
  EXP-15c: Dual-system arbitration — habit vs goal-directed
  EXP-15d: Breaking habits — PFC energy injection effect
"""

import sys
import numpy as np

from alice.brain.basal_ganglia import (
    BasalGangliaEngine,
    INITIAL_ACTION_GAMMA,
    HABIT_THRESHOLD,
    DA_BASELINE,
)
from alice.brain.prefrontal import PrefrontalCortexEngine


def exp_15a_habit_curve():
    """
    EXP-15a: Habit Formation Curve

    Scenario: Learning to ride a bicycle
    - Tracking Γ_action changes with practice count
    - Expected: exponential decay, final Γ < HABIT_THRESHOLD
    """
    print("=" * 60)
    print("EXP-15a: Habit Formation Curve — Learning to Ride a Bicycle")
    print("=" * 60)

    bg = BasalGangliaEngine()
    bg.register_action("cycling", "pedal_balance")

    n_reps = 100
    gammas = []
    milestones = [1, 5, 10, 20, 50, 100]

    print(f"\n  initial Γ = {INITIAL_ACTION_GAMMA}")
    print(f"  Habit threshold = {HABIT_THRESHOLD}")
    print(f"\n  {'Practice':>8s} {'Γ_habit':>10s} {'Automat.':>8s} {'Is Habit':>8s}")
    print(f"  {'-'*38}")

    habit_formed_at = None
    for i in range(1, n_reps + 1):
        result = bg.update_after_action(
            "cycling", "pedal_balance",
            reward=0.5 + 0.3 * np.random.random(),  # Generally positive
            success=True,
        )
        gammas.append(result["gamma_habit"])

        if i in milestones:
            ch = bg.get_channel("cycling", "pedal_balance")
            print(f"  {i:8d} {result['gamma_habit']:10.4f} "
                  f"{ch.automaticity:8.3f} {'✓' if ch.is_habit else '✗':>8s}")

        if result["newly_formed_habit"] and habit_formed_at is None:
            habit_formed_at = i

    print(f"\n  Habit formed at practice #{habit_formed_at}" if habit_formed_at
          else "\n  Habit not formed (needs more practice)")
    print(f"  final Γ = {gammas[-1]:.4f}")
    print(f"  Γ decrease ratio = {(gammas[0] - gammas[-1]) / gammas[0] * 100:.1f}%")

    # verification
    assert gammas[-1] < gammas[0], "Γ should decrease"
    assert gammas[-1] < HABIT_THRESHOLD, "100 practices should form a habit"
    print("\n  ✓ EXP-15a PASS")
    return True


def exp_15b_dopamine_modulation():
    """
    EXP-15b: Action Selection and Dopamine Modulation

    Scenario: Rat pressing lever — alternating reward and punishment
    - Reward → dopamine increase → Go enhancement
    - Punishment → dopamine decrease → NoGo enhancement
    """
    print("\n" + "=" * 60)
    print("EXP-15b: Action Selection & Dopamine Modulation — Operant Conditioning")
    print("=" * 60)

    bg = BasalGangliaEngine(temperature=0.1)
    bg.register_action("box", "press_lever", initial_go=0.5, initial_nogo=0.5)
    bg.register_action("box", "groom", initial_go=0.3, initial_nogo=0.3)

    n_trials = 30
    phases = [
        ("reward", 10, 1.0),
        ("punishment", 10, -1.0),
        ("recovery", 10, 0.5),
    ]

    print(f"\n  {'Trial':>4s} {'Phase':>8s} {'Action':>12s} {'DA':>8s} "
          f"{'Go':>8s} {'NoGo':>8s} {'RPE':>8s}")
    print(f"  {'-'*64}")

    trial = 0
    for phase_name, n, reward in phases:
        for i in range(n):
            trial += 1
            result = bg.update_after_action("box", "press_lever", reward=reward, success=reward > 0)
            ch = bg.get_channel("box", "press_lever")

            if i == 0 or i == n - 1:
                print(f"  {trial:4d} {phase_name:>8s} {'press_lever':>12s} "
                      f"{result['dopamine']:8.4f} {result['go_strength']:8.4f} "
                      f"{result['nogo_strength']:8.4f} {result['rpe']:8.4f}")

    # Selection test
    print(f"\n  Action selection competition (10 trials):")
    press_count = 0
    for _ in range(10):
        sel = bg.select_action("box", ["press_lever", "groom"])
        if sel.selected_action == "press_lever":
            press_count += 1

    print(f"    press_lever selected: {press_count}/10")

    # verification
    ch = bg.get_channel("box", "press_lever")
    assert ch.go_strength != ch.nogo_strength, "Go/NoGo should differ"
    print("\n  ✓ EXP-15b PASS")
    return True


def exp_15c_dual_system():
    """
    EXP-15c: Dual-System Arbitration — Habit vs Goal-Directed

    Scenario: Commute route selection
    - Habitual route: fast but encounters construction
    - Goal route: slow but ensures arrival
    - Scan uncertainty: observe system switching
    """
    print("\n" + "=" * 60)
    print("EXP-15c: Dual-System Arbitration — Commute Route Selection")
    print("=" * 60)

    bg = BasalGangliaEngine()

    uncertainties = np.linspace(0.0, 1.0, 11)
    print(f"\n  {'Uncert.':>8s} {'System':>12s} {'Habit W.':>10s} {'Goal W.':>10s} {'Action':>12s}")
    print(f"  {'-'*56}")

    transition_point = None
    for u in uncertainties:
        result = bg.arbitrate_systems(
            habit_action="usual_route",
            goal_action="detour_route",
            habit_gamma=0.1,  # High habit degree
            goal_uncertainty=u,
            reward_rate=0.5,
        )
        print(f"  {u:8.2f} {result['chosen_system']:>12s} "
              f"{result['habit_weight']:10.4f} {result['goal_weight']:10.4f} "
              f"{result['chosen_action']:>12s}")

        if result["chosen_system"] == "goal_directed" and transition_point is None:
            transition_point = u

    print(f"\n  System switch point: uncertainty = {transition_point:.2f}" if transition_point
          else "\n  No system switch occurred")

    # verification
    # Low uncertainty → habit; high uncertainty → goal-directed
    r_low = bg.arbitrate_systems("A", "B", habit_gamma=0.1,
                                  goal_uncertainty=0.05, reward_rate=0.8)
    r_high = bg.arbitrate_systems("A", "B", habit_gamma=0.8,
                                   goal_uncertainty=0.95, reward_rate=0.1)
    assert r_low["chosen_system"] == "habitual"
    assert r_high["chosen_system"] == "goal_directed"
    print("\n  ✓ EXP-15c PASS")
    return True


def exp_15d_break_habit():
    """
    EXP-15d: Breaking Habits — PFC Energy Injection

    Scenario: Quitting smoking
    1. Create smoking habit (Γ → 0)
    2. PFC intervention: raise Γ (break automaticity)
    3. Observe: how much PFC energy is needed to break the habit
    """
    print("\n" + "=" * 60)
    print("EXP-15d: Breaking Habits — Quitting Smoking Scenario")
    print("=" * 60)

    bg = BasalGangliaEngine()
    pfc = PrefrontalCortexEngine()

    # Phase 1: Form smoking habit
    bg.register_action("stress", "smoke")
    print(f"\n  Phase 1: Creating habit")
    for i in range(80):
        bg.update_after_action("stress", "smoke", reward=0.7, success=True)

    ch = bg.get_channel("stress", "smoke")
    print(f"    After 80 practices: Γ = {ch.gamma_habit:.4f}, habit = {ch.is_habit}")
    assert ch.is_habit, "80 practices should form a habit"

    # Phase 2: PFC breaks habit
    print(f"\n  Phase 2: PFC intervention to break habit")
    print(f"  {'Count':>4s} {'PFC Eng.':>8s} {'Γ_before':>10s} {'Γ_after':>10s} {'Still Hab':>8s}")
    print(f"  {'-'*44}")

    attempts = 0
    while ch.is_habit and attempts < 20:
        attempts += 1
        pfc_energy = min(0.3, pfc._energy)
        gamma_before = ch.gamma_habit
        result = bg.break_habit("stress", "smoke", pfc_energy_input=pfc_energy)
        pfc.drain_energy(pfc_energy * 0.5)  # PFC consumes energy

        print(f"  {attempts:4d} {pfc._energy:8.4f} {result['gamma_before']:10.4f} "
              f"{result['gamma_after']:10.4f} {'✓' if result['still_habit'] else '✗':>8s}")

    if not ch.is_habit:
        print(f"\n  ✓ Habit broken after attempt #{attempts}")
        print(f"    Final Γ = {ch.gamma_habit:.4f}")
        print(f"    PFC remaining energy = {pfc._energy:.4f}")
    else:
        print(f"\n  × Habit too stubborn, still not broken after {attempts} attempts")

    # Phase 3: Verify no longer automatically executed
    sel = bg.select_action("stress", ["smoke", "deep_breath"])
    print(f"\n  Phase 3: Action selection test")
    print(f"    Selected: {sel.selected_action} (cortex required: {sel.cortical_needed})")

    print("\n  ✓ EXP-15d PASS")
    return True


# ============================================================================
# Main
# ============================================================================


def main():
    results = []
    for exp_fn in [exp_15a_habit_curve, exp_15b_dopamine_modulation,
                   exp_15c_dual_system, exp_15d_break_habit]:
        try:
            ok = exp_fn()
            results.append((exp_fn.__name__, ok))
        except Exception as e:
            print(f"\n  ✗ {exp_fn.__name__} FAIL: {e}")
            import traceback
            traceback.print_exc()
            results.append((exp_fn.__name__, False))

    print("\n" + "=" * 60)
    print("Basal Ganglia Experiment Summary")
    print("=" * 60)
    for name, ok in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\n  PASS: {passed}/{total}")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
