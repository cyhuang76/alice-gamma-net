# -*- coding: utf-8 -*-
"""
Experiment: Prefrontal Cortex — Executive Control & Willpower (Phase 6.1)

Experiment list:
  EXP-14a: Goal stack management — multi-goal competition and sub-goal decomposition
  EXP-14b: Go/NoGo gate — impulse control and energy depletion
  EXP-14c: Planning engine — minimum Γ path
  EXP-14d: Willpower depletion cascade — Baumeister ego depletion
"""

import sys
import numpy as np

from alice.brain.prefrontal import PrefrontalCortexEngine


def exp_14a_goal_stack():
    """
    EXP-14a: Goal Stack Management

    Simulate a day with multiple competing goals:
    1. Set multiple goals (work, fitness, learning, social)
    2. Observe priority sorting
    3. Complete sub-goals → parent goal progress update
    4. Goal overflow → lowest priority eviction
    """
    print("=" * 60)
    print("EXP-14a: Goal Stack Management — Multi-Goal Competition")
    print("=" * 60)

    pfc = PrefrontalCortexEngine(max_goals=4)

    # Set goals
    goals = [
        ("finish_report", 75.0, 0.9),
        ("exercise", 50.0, 0.5),
        ("learn_piano", 80.0, 0.6),
        ("socialize", 60.0, 0.3),
    ]

    for name, z, p in goals:
        result = pfc.set_goal(name, z_goal=z, priority=p)
        print(f" Set goal: {name} (Z={z}Ω, P={p}) → {result['action']}")

    # Decompose main goal
    pfc.decompose_goal("finish_report", [
        {"name": "gather_data", "z_goal": 60.0},
        {"name": "write_draft", "z_goal": 70.0},
        {"name": "review_edit", "z_goal": 75.0},
    ])
    print(f"\n Sub-goal decomposition: finish_report → 3 steps")

    # Display stack
    stack = pfc.get_goal_stack()
    print(f"\n Goal stack (depth={len(stack)}):")
    for g in stack:
        print(f"    {g['name']:20s} P={g['effective_priority']:.3f} "
              f"Z={g['z_goal']:.0f}Ω progress={g['progress']:.1%}")

    # Try adding 5th goal (exceeds capacity)
    result = pfc.set_goal("urgent_task", z_goal=70.0, priority=0.95)
    print(f"\n Add urgent task: {result['action']} (stack={result.get('stack_depth', '?')})")

    # Complete sub-goals
    pfc.update_goal_progress("gather_data", 1.0)
    pfc.update_goal_progress("write_draft", 0.5)
    parent = pfc._goals.get("finish_report")
    print(f"\n  Completed gather_data, write_draft=50%")
    if parent:
        print(f" Parent goal finish_report progress: {parent.progress:.1%}")

    # Final state
    top = pfc.get_top_goal()
    print(f"\n Highest priority goal: {top.name if top else 'None'}")

    assert len(pfc.get_goal_stack()) <= 4, "Stack should not exceed ceiling"
    print("\n  ✓ EXP-14a PASS")
    return True


def exp_14b_go_nogo_gate():
    """
    EXP-14b: Go/NoGo Gate — Impulse Control

    Scenario: student studying in a library
    - Goal: study (Z=75Ω)
    - Interference: phone(Z=200Ω), snack(Z=150Ω), chat with friend(Z=180Ω)
    - Observe: which actions are allowed, which are inhibited
    """
    print("\n" + "=" * 60)
    print("EXP-14b: Go/NoGo Gate — Library Scenario")
    print("=" * 60)

    pfc = PrefrontalCortexEngine()
    pfc.set_goal("study", z_goal=75.0, priority=0.8)

    actions = [
        ("open_textbook", 75.0, "cortical", 0.0), # Matches goal
        ("check_phone", 500.0, "limbic", 0.0), # Conflicts → NoGo
        ("take_notes", 70.0, "cortical", 0.0), # Slightly below match
        ("eat_snack", 500.0, "limbic", 0.0), # Conflicts → NoGo
        ("answer_question", 80.0, "cortical", 0.0), # Close match
        ("fire_alarm", 500.0, "limbic", 0.95), # Emotional override
    ]

    print(f"\n Goal: study (Z_goal=75Ω)")
    print(f" {'Action':20s} {'Z_action':>8s} {'Γ':>8s} {'Decision':>8s} {'Reason':>25s}")
    print(f"  {'-'*72}")

    for name, z, source, emo in actions:
        result = pfc.evaluate_action(
            name, z_action=z, source=source, emotional_override=emo
        )
        print(f"  {name:20s} {z:8.0f}Ω {result.gamma_action:8.4f} "
              f"{result.decision:>8s} {result.reason:>25s}")

    print(f"\n PFC remaining energy: {pfc._energy:.3f}")
    print(f"  Go={pfc._total_go_decisions} NoGo={pfc._total_nogo_decisions} "
          f"Defer={pfc._total_defer_decisions}")

    assert pfc._total_nogo_decisions >= 1, "Should have at least one inhibition"
    assert pfc._total_go_decisions >= 2, "Matching actions should be allowed"
    print("\n  ✓ EXP-14b PASS")
    return True


def exp_14c_planning():
    """
    EXP-14c: Planning Engine — Minimum Γ Path

    Scenario: learning path from beginner (Z=30Ω) to mastery (Z=75Ω)
    """
    print("\n" + "=" * 60)
    print("EXP-14c: Planning Engine — Beginner to Mastery Path")
    print("=" * 60)

    pfc = PrefrontalCortexEngine()
    pfc.set_goal("mastery", z_goal=75.0, priority=0.9)

    # Auto-plan
    plan = pfc.create_plan(z_current=30.0, goal_name="mastery")
    print(f"\n  Z_current=30Ω → Z_goal=75Ω")
    print(f" Steps: {plan['total_steps']}")
    print(f"  {'step':>8s} {'Z_start':>8s} {'Z_end':>8s} {'Γ':>8s} {'effort':>8s} {'energy':>8s}")
    print(f"  {'-'*48}")

    for step in plan["steps"]:
        print(f"  {step['name']:>8s} {step['z_start']:8.1f}Ω {step['z_end']:8.1f}Ω "
              f"{step['gamma']:8.4f} {step['effort']:8.4f} {step['energy']:8.4f}")

    print(f"\n Total energy: {plan['total_energy']:.4f}")
    print(f" Feasibility: {plan['feasible']}")

    # Advance the plan
    print(f"\n Step-by-step advancement:")
    while True:
        result = pfc.advance_plan()
        if result is None:
            break
        print(f"    Completed {result['completed_step']}: "
              f"energy_spent={result['energy_spent']:.4f} "
              f"PFC_energy={result['pfc_energy']:.4f} "
              f"remaining={result['remaining_steps']}")
        if result["plan_complete"]:
            print(f" ✓ Plan completed!")
            break

    assert plan["total_steps"] >= 2, "At least two steps"
    assert plan["total_energy"] > 0, "Energy required"
    print("\n  ✓ EXP-14c PASS")
    return True


def exp_14d_ego_depletion():
    """
    EXP-14d: Willpower Depletion Cascade — Baumeister ego depletion

    Scenario: continuously resisting temptation → energy depletion → impulse breakthrough

    'Baumeister (1998) marshmallow experiment:
     Those who resisted chocolate earlier persisted less on puzzles afterward.
     Willpower is a limited resource — prefrontal energy.'
    """
    print("\n" + "=" * 60)
    print("EXP-14d: Willpower Depletion Cascade — Baumeister Effect")
    print("=" * 60)

    pfc = PrefrontalCortexEngine()
    pfc.set_goal("diet", z_goal=75.0, priority=0.9)

    energies = []
    decisions = []
    n_temptations = 40

    print(f"\n Goal: diet (Z_goal=75Ω)")
    print(f" Temptation: cake (Z=500Ω → Γ≈0.74) × {n_temptations}")
    print(f"\n {'#':>4s} {'Decision':>8s} {'Γ':>8s} {'Energy':>8s} {'Reason':>30s}")
    print(f"  {'-'*62}")

    for i in range(n_temptations):
        result = pfc.evaluate_action(
            f"eat_cake_{i}", z_action=500.0, source="limbic"
        )
        energies.append(pfc._energy)
        decisions.append(result.decision)

        if i < 5 or i > n_temptations - 6 or result.decision == "go":
            print(f"  {i+1:4d} {result.decision:>8s} {result.gamma_action:8.4f} "
                  f"{pfc._energy:8.4f} {result.reason:>30s}")

    # Analysis
    nogo_count = decisions.count("nogo")
    go_count = decisions.count("go")
    first_failure = None
    for i, d in enumerate(decisions):
        if d == "go":
            first_failure = i + 1
            break

    print(f"\n Result statistics:")
    print(f" Successful inhibitions: {nogo_count} times")
    print(f" Impulse breakthroughs: {go_count} times")
    print(f" First failure: temptation #{first_failure}")
    print(f"    Initial energy: {energies[0]:.4f}")
    print(f"    Final energy: {energies[-1]:.4f}")
    print(f" Energy depleted: {energies[0] - energies[-1]:.4f}")

    # Verify
    assert first_failure is not None, "Continuous resistance should eventually fail"
    assert energies[-1] < energies[0], "Energy should decrease"
    assert nogo_count > 0, "First few attempts should succeed in inhibiting"
    print("\n ✓ EXP-14d PASS: Willpower depletion effect successfully reproduced")
    return True


# ============================================================================
# Main
# ============================================================================


def main():
    results = []
    for exp_fn in [exp_14a_goal_stack, exp_14b_go_nogo_gate,
                   exp_14c_planning, exp_14d_ego_depletion]:
        try:
            ok = exp_fn()
            results.append((exp_fn.__name__, ok))
        except Exception as e:
            print(f"\n  ✗ {exp_fn.__name__} FAIL: {e}")
            results.append((exp_fn.__name__, False))

    print("\n" + "=" * 60)
    print("Prefrontal Experiment Summary")
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
