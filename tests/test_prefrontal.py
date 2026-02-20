# -*- coding: utf-8 -*-
"""
Prefrontal Cortex (PFC) Tests — Phase 6.1

Coverage:
  1. Goal management (set, decompose, progress, stack)
  2. Go/NoGo gate (pass, inhibit, impulse breakthrough, emotional override)
  3. Planning engine (path generation, step advancement)
  4. Cognitive control (task switching, perseveration error, emotion regulation)
  5. Energy management (depletion, recovery, willpower)
  6. Tick cycle
"""

import time
import numpy as np
import pytest

from alice.brain.prefrontal import (
    PrefrontalCortexEngine,
    Goal,
    ActionProposal,
    PlanStep,
    GoNoGoDecision,
    TaskSwitchResult,
    GOAL_IMPEDANCE_DEFAULT,
    MAX_GOAL_STACK,
    INHIBITION_ENERGY_COST,
    PFC_MAX_ENERGY,
    PFC_FATIGUE_THRESHOLD,
    GO_THRESHOLD,
    NOGO_THRESHOLD,
    EMOTIONAL_OVERRIDE_THRESHOLD,
    TASK_SWITCH_COST,
    PFC_AMYGDALA_REGULATION,
    GOAL_COMPLETION_THRESHOLD,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def pfc():
    """Fresh prefrontal engine."""
    return PrefrontalCortexEngine()


@pytest.fixture
def pfc_with_goal():
    """Prefrontal with a preset goal."""
    pfc = PrefrontalCortexEngine()
    pfc.set_goal("study", z_goal=75.0, priority=0.8)
    return pfc


@pytest.fixture
def pfc_exhausted():
    """Energy-depleted prefrontal."""
    pfc = PrefrontalCortexEngine()
    pfc._energy = 0.05  # Nearly depleted
    pfc.set_goal("resist_impulse", z_goal=75.0, priority=0.9)
    return pfc


# ============================================================================
# 1. Goal Management Tests
# ============================================================================


class TestGoalManagement:
    """Physical validity of goal management."""

    def test_set_goal_basic(self, pfc):
        """Set a basic goal."""
        result = pfc.set_goal("learn_python", z_goal=50.0, priority=0.7)
        assert result["action"] == "created"
        assert result["goal"] == "learn_python"
        assert result["z_goal"] == 50.0

    def test_goal_stack_depth(self, pfc):
        """Goal stack does not exceed the limit."""
        for i in range(MAX_GOAL_STACK + 3):
            pfc.set_goal(f"goal_{i}", priority=0.1 * i)

        stack = pfc.get_goal_stack()
        assert len(stack) <= MAX_GOAL_STACK

    def test_goal_priority_ordering(self, pfc):
        """Goals are sorted by priority."""
        pfc.set_goal("low", priority=0.2)
        pfc.set_goal("high", priority=0.9)
        pfc.set_goal("mid", priority=0.5)

        stack = pfc.get_goal_stack()
        priorities = [g["effective_priority"] for g in stack]
        assert priorities == sorted(priorities, reverse=True)

    def test_get_top_goal(self, pfc):
        """Get the highest priority goal."""
        pfc.set_goal("A", priority=0.3)
        pfc.set_goal("B", priority=0.9)
        pfc.set_goal("C", priority=0.5)

        top = pfc.get_top_goal()
        assert top is not None
        assert top.name == "B"

    def test_goal_update_existing(self, pfc):
        """Update an existing goal."""
        pfc.set_goal("study", priority=0.5)
        result = pfc.set_goal("study", priority=0.8)
        assert result["action"] == "updated"
        assert result["priority"] == 0.8

    def test_goal_eviction_when_full(self, pfc):
        """Evict lowest priority when stack is full."""
        for i in range(MAX_GOAL_STACK):
            pfc.set_goal(f"goal_{i}", priority=0.5 + 0.05 * i)

        # Add a high priority goal
        pfc.set_goal("urgent", priority=0.99)
        stack = pfc.get_goal_stack()
        names = [g["name"] for g in stack]
        assert "urgent" in names
        assert len(stack) <= MAX_GOAL_STACK

    def test_decompose_goal(self, pfc_with_goal):
        """Sub-goal decomposition."""
        result = pfc_with_goal.decompose_goal("study", [
            {"name": "read_chapter", "z_goal": 60.0},
            {"name": "do_exercises", "z_goal": 70.0},
            {"name": "review", "z_goal": 75.0},
        ])
        assert len(result["sub_goals_created"]) == 3
        assert "read_chapter" in result["sub_goals_created"]

    def test_decompose_nonexistent_goal(self, pfc):
        """Decompose a nonexistent goal."""
        result = pfc.decompose_goal("nonexistent", [])
        assert "error" in result

    def test_goal_progress_update(self, pfc_with_goal):
        """Goal progress update."""
        result = pfc_with_goal.update_goal_progress("study", 0.5)
        assert result["progress"] == 0.5
        assert not result["completed"]

    def test_goal_completion(self, pfc_with_goal):
        """Goal completion."""
        result = pfc_with_goal.update_goal_progress("study", 0.95)
        assert result["completed"]
        assert not result["active"]

    def test_sub_goal_completion_updates_parent(self, pfc_with_goal):
        """Sub-goal completion updates parent progress."""
        pfc_with_goal.decompose_goal("study", [
            {"name": "step1"},
            {"name": "step2"},
        ])
        pfc_with_goal.update_goal_progress("step1", 1.0)

        parent = pfc_with_goal._goals["study"]
        assert parent.progress > 0.0

    def test_goal_effective_priority_decays(self):
        """Goal effective priority decays over time."""
        goal = Goal(name="test", z_goal=75.0, priority=0.8)
        p1 = goal.effective_priority
        # Simulate time passing
        goal.created_at -= 100.0  # Pretend created 100 seconds ago
        p2 = goal.effective_priority
        assert p2 < p1

    def test_no_active_goals_returns_none(self, pfc):
        """Returns None when no active goals."""
        assert pfc.get_top_goal() is None

    def test_empty_goal_stack(self, pfc):
        """Empty goal stack."""
        stack = pfc.get_goal_stack()
        assert stack == []


# ============================================================================
# 2. Go/NoGo Gate Tests
# ============================================================================


class TestGoNoGoGate:
    """Go/NoGo gate impulse control."""

    def test_go_when_goal_aligned(self, pfc_with_goal):
        """Action matches goal → Go."""
        result = pfc_with_goal.evaluate_action(
            "open_textbook", z_action=75.0  # Perfectly matches goal z_goal=75
        )
        assert result.decision == "go"
        assert result.gamma_action < GO_THRESHOLD

    def test_nogo_when_goal_conflict(self, pfc_with_goal):
        """Action conflicts with goal → NoGo."""
        # Γ = |z-75|/(z+75) > 0.7 requires z > 425
        result = pfc_with_goal.evaluate_action(
            "play_games", z_action=500.0  # Far from goal
        )
        assert result.decision == "nogo"
        assert result.inhibited

    def test_defer_when_ambiguous(self, pfc_with_goal):
        """Ambiguous case → Defer."""
        # Find an impedance with Γ between GO_THRESHOLD and NOGO_THRESHOLD
        z_goal = 75.0
        # Γ = |z - 75| / (z + 75) = target → between GO and NOGO
        # e.g. Γ ≈ 0.55 → z ≈ 75 * (1 + 0.55) / (1 - 0.55) ≈ 258
        # But NOGO_THRESHOLD = 0.7, so we need Γ in 0.4~0.7
        # Γ = 0.55 → z = 75 * (1+0.55)/(1-0.55) ≈ 258
        # Let's use a more precise value
        # We need GO_THRESHOLD < Γ < NOGO_THRESHOLD
        # 0.4 < |z-75|/(z+75) < 0.7
        # Take Γ = 0.5 → z = 75*(1+0.5)/(1-0.5) = 225
        result = pfc_with_goal.evaluate_action(
            "check_email", z_action=225.0
        )
        assert result.decision == "defer"

    def test_emotional_override(self, pfc_with_goal):
        """Strong emotion overrides prefrontal → forced Go."""
        result = pfc_with_goal.evaluate_action(
            "run_away", z_action=300.0,
            emotional_override=0.95  # Exceeds threshold
        )
        assert result.decision == "go"
        assert result.reason == "emotional_override"

    def test_inhibition_energy_cost(self, pfc_with_goal):
        """Inhibition consumes energy."""
        e_before = pfc_with_goal._energy
        pfc_with_goal.evaluate_action("play_games", z_action=500.0)
        assert pfc_with_goal._energy < e_before

    def test_inhibition_failure_when_exhausted(self, pfc_exhausted):
        """Energy depleted → inhibition failure → impulse breakthrough."""
        result = pfc_exhausted.evaluate_action(
            "eat_cake", z_action=500.0
        )
        assert result.decision == "go"
        assert "inhibition_failure" in result.reason

    def test_no_goal_defaults_to_go(self, pfc):
        """No goal defaults to Go."""
        result = pfc.evaluate_action("anything", z_action=50.0)
        assert result.decision == "go"

    def test_go_counter(self, pfc_with_goal):
        """Go decision counter."""
        pfc_with_goal.evaluate_action("study_action", z_action=75.0)
        assert pfc_with_goal._total_go_decisions >= 1

    def test_nogo_counter(self, pfc_with_goal):
        """NoGo decision counter."""
        pfc_with_goal.evaluate_action("distraction", z_action=500.0)
        assert pfc_with_goal._total_nogo_decisions >= 1

    def test_inhibition_log(self, pfc_with_goal):
        """Inhibition log is preserved."""
        pfc_with_goal.evaluate_action("play_games", z_action=500.0)
        assert len(pfc_with_goal._inhibition_log) >= 1

    def test_decision_buffer(self, pfc_with_goal):
        """Decision buffer records."""
        pfc_with_goal.evaluate_action("action_1", z_action=75.0)
        pfc_with_goal.evaluate_action("action_2", z_action=200.0)
        assert len(pfc_with_goal._decision_buffer) == 2

    def test_repeated_inhibition_drains_energy(self, pfc_with_goal):
        """Repeated inhibition continuously drains energy."""
        energies = [pfc_with_goal._energy]
        for i in range(5):
            pfc_with_goal.evaluate_action(f"tempt_{i}", z_action=500.0)
            energies.append(pfc_with_goal._energy)

        # Energy should gradually decrease
        assert energies[-1] < energies[0]


# ============================================================================
# 3. Planning Engine Tests
# ============================================================================


class TestPlanning:
    """Planning engine path search."""

    def test_basic_plan_creation(self, pfc_with_goal):
        """Basic plan creation."""
        plan = pfc_with_goal.create_plan(z_current=50.0, goal_name="study")
        assert plan["z_current"] == 50.0
        assert plan["z_goal"] == 75.0
        assert plan["total_steps"] >= 2
        assert len(plan["steps"]) > 0

    def test_plan_steps_have_gamma(self, pfc_with_goal):
        """Each step has a Γ value."""
        plan = pfc_with_goal.create_plan(z_current=30.0)
        for step in plan["steps"]:
            assert "gamma" in step
            assert 0 <= step["gamma"] <= 1

    def test_plan_with_intermediate_states(self, pfc_with_goal):
        """Plan with intermediate states."""
        plan = pfc_with_goal.create_plan(
            z_current=30.0,
            intermediate_states=[40.0, 55.0, 65.0],
        )
        # Should have 4 steps: 30→40→55→65→75
        assert plan["total_steps"] == 4

    def test_plan_no_goal_error(self, pfc):
        """Planning fails with no goal."""
        plan = pfc.create_plan(z_current=50.0)
        assert "error" in plan

    def test_plan_total_energy(self, pfc_with_goal):
        """Plan total energy estimate."""
        plan = pfc_with_goal.create_plan(z_current=30.0)
        assert plan["total_energy"] > 0

    def test_plan_feasibility(self, pfc_with_goal):
        """Plan feasibility assessment."""
        plan = pfc_with_goal.create_plan(z_current=70.0)  # Close to goal
        assert "feasible" in plan

    def test_get_next_step(self, pfc_with_goal):
        """Get next step."""
        pfc_with_goal.create_plan(z_current=50.0)
        step = pfc_with_goal.get_next_plan_step()
        assert step is not None
        assert "name" in step

    def test_advance_plan(self, pfc_with_goal):
        """Advance the plan."""
        pfc_with_goal.create_plan(z_current=50.0)
        total = len(pfc_with_goal._current_plan)
        result = pfc_with_goal.advance_plan()
        assert result is not None
        assert result["remaining_steps"] == total - 1

    def test_advance_empty_plan(self, pfc):
        """Advance empty plan."""
        result = pfc.advance_plan()
        assert result is None

    def test_plan_cache(self, pfc_with_goal):
        """Plan cache."""
        pfc_with_goal.create_plan(z_current=50.0, goal_name="study")
        assert "study" in pfc_with_goal._plan_cache

    def test_plan_counter(self, pfc_with_goal):
        """Plan counter."""
        pfc_with_goal.create_plan(z_current=50.0)
        assert pfc_with_goal._total_plans_created == 1


# ============================================================================
# 4. Cognitive Control Tests
# ============================================================================


class TestCognitiveControl:
    """Cognitive control and task switching."""

    def test_task_switch_basic(self, pfc):
        """Basic task switch."""
        pfc._current_task = "reading"
        result = pfc.switch_task("writing")
        assert result.to_task == "writing"
        assert result.switch_cost > 0
        assert not result.perseveration_error

    def test_task_switch_same_task(self, pfc):
        """Switching to the same task → zero cost."""
        pfc._current_task = "reading"
        result = pfc.switch_task("reading")
        assert result.switch_cost == 0.0

    def test_task_switch_cost_increases_with_fatigue(self, pfc):
        """Fatigue increases switch cost."""
        pfc._current_task = "A"
        result_fresh = pfc.switch_task("B")

        pfc2 = PrefrontalCortexEngine()
        pfc2._current_task = "A"
        pfc2._energy = PFC_FATIGUE_THRESHOLD * 0.5  # Fatigued
        result_tired = pfc2.switch_task("B")

        assert result_tired.switch_cost > result_fresh.switch_cost

    def test_perseveration_error(self):
        """Perseveration error — fatigue + high perseveration strength."""
        pfc = PrefrontalCortexEngine()
        pfc._current_task = "old_task"
        pfc._perseveration_strength = 0.8
        pfc._energy = PFC_FATIGUE_THRESHOLD * 0.3

        result = pfc.switch_task("new_task")
        assert result.perseveration_error

    def test_forced_switch_bypasses_perseveration(self):
        """Forced switch bypasses perseveration."""
        pfc = PrefrontalCortexEngine()
        pfc._current_task = "old_task"
        pfc._perseveration_strength = 0.9
        pfc._energy = PFC_FATIGUE_THRESHOLD * 0.1

        result = pfc.switch_task("new_task", forced=True)
        assert not result.perseveration_error
        assert pfc._current_task == "new_task"

    def test_task_history(self, pfc):
        """Task switch history recording."""
        pfc.switch_task("A")
        pfc.switch_task("B")
        pfc.switch_task("C")
        assert len(pfc._task_history) >= 3

    def test_emotion_regulation_reappraisal(self, pfc):
        """Cognitive reappraisal strategy."""
        result = pfc.regulate_emotion(0.8, strategy="reappraisal")
        assert result["success"]
        assert result["regulated_intensity"] < 0.8

    def test_emotion_regulation_suppression(self, pfc):
        """Expressive suppression strategy."""
        result = pfc.regulate_emotion(0.8, strategy="suppression")
        assert result["success"]
        assert result["energy_cost"] > 0

    def test_emotion_regulation_distraction(self, pfc):
        """Attentional distraction strategy."""
        result = pfc.regulate_emotion(0.8, strategy="distraction")
        assert result["success"]
        assert result["regulated_intensity"] < 0.8

    def test_emotion_regulation_fails_when_exhausted(self, pfc_exhausted):
        """Insufficient energy → emotion regulation fails."""
        result = pfc_exhausted.regulate_emotion(0.8, strategy="suppression")
        assert not result["success"]
        assert result["regulated_intensity"] == 0.8

    def test_unknown_strategy_no_effect(self, pfc):
        """Unknown strategy has no effect."""
        result = pfc.regulate_emotion(0.8, strategy="unknown")
        assert result["regulated_intensity"] == 0.8


# ============================================================================
# 5. Energy Management Tests
# ============================================================================


class TestEnergyManagement:
    """Prefrontal energy (willpower) system."""

    def test_initial_energy(self, pfc):
        """Initial energy = maximum."""
        assert pfc._energy == PFC_MAX_ENERGY

    def test_drain_energy(self, pfc):
        """Drain energy."""
        remaining = pfc.drain_energy(0.3)
        assert remaining == pytest.approx(0.7, abs=0.01)

    def test_drain_energy_floor(self, pfc):
        """Energy does not go below zero."""
        remaining = pfc.drain_energy(5.0)
        assert remaining == 0.0

    def test_restore_energy(self, pfc):
        """Restore energy."""
        pfc.drain_energy(0.5)
        restored = pfc.restore_energy(0.3)
        assert restored == pytest.approx(0.8, abs=0.01)

    def test_restore_energy_ceiling(self, pfc):
        """Energy does not exceed ceiling."""
        restored = pfc.restore_energy(5.0)
        assert restored == PFC_MAX_ENERGY

    def test_tick_recovers_energy(self, pfc):
        """Tick recovers energy."""
        pfc.drain_energy(0.5)
        e_before = pfc._energy
        pfc.tick()
        assert pfc._energy > e_before

    def test_energy_history_tracking(self, pfc):
        """Energy history tracking."""
        for _ in range(5):
            pfc.tick()
        assert len(pfc._energy_history) >= 6  # Initial + 5 ticks

    def test_energy_history_trimming(self, pfc):
        """Energy history trimming."""
        pfc._energy_history = list(range(1500))
        pfc.tick()
        assert len(pfc._energy_history) <= 600


# ============================================================================
# 6. Tick Cycle Tests
# ============================================================================


class TestTickCycle:
    """Prefrontal tick cycle."""

    def test_tick_increments_counter(self, pfc):
        """Tick increments counter."""
        pfc.tick()
        pfc.tick()
        assert pfc._tick_count == 2

    def test_tick_decays_perseveration(self, pfc):
        """Tick decays perseveration."""
        pfc._perseveration_strength = 0.5
        for _ in range(20):
            pfc.tick()
        assert pfc._perseveration_strength < 0.5

    def test_tick_returns_state(self, pfc):
        """Tick returns state."""
        result = pfc.tick()
        assert "tick" in result
        assert "energy" in result
        assert "active_goals" in result


# ============================================================================
# 7. Query Interface Tests
# ============================================================================


class TestQueryInterface:
    """Query interface."""

    def test_get_state(self, pfc_with_goal):
        """Get complete state."""
        state = pfc_with_goal.get_state()
        assert "energy" in state
        assert "current_task" in state
        assert "active_goals" in state
        assert "can_inhibit" in state

    def test_get_stats(self, pfc_with_goal):
        """Get statistics."""
        stats = pfc_with_goal.get_stats()
        assert "total_go" in stats
        assert "total_nogo" in stats
        assert "inhibition_success_rate" in stats

    def test_get_decision_history(self, pfc_with_goal):
        """Get decision history."""
        pfc_with_goal.evaluate_action("test", z_action=75.0)
        history = pfc_with_goal.get_decision_history()
        assert len(history) >= 1
        assert "action" in history[0]
        assert "decision" in history[0]

    def test_get_energy_history(self, pfc):
        """Get energy history."""
        history = pfc.get_energy_history()
        assert len(history) >= 1


# ============================================================================
# 8. Physical Consistency Tests
# ============================================================================


class TestPhysicalConsistency:
    """Γ-Net impedance model physical validity."""

    def test_gamma_range(self, pfc_with_goal):
        """Γ value in physical range [0,1]."""
        for z in [10, 50, 75, 100, 200, 500]:
            result = pfc_with_goal.evaluate_action("test", z_action=float(z))
            assert 0 <= result.gamma_action <= 1.0

    def test_perfect_match_low_gamma(self, pfc_with_goal):
        """Perfect impedance match → Γ ≈ 0."""
        result = pfc_with_goal.evaluate_action("match", z_action=75.0)
        assert result.gamma_action < 0.01

    def test_high_mismatch_high_gamma(self, pfc_with_goal):
        """High impedance mismatch → high Γ."""
        result = pfc_with_goal.evaluate_action("mismatch", z_action=500.0)
        assert result.gamma_action > 0.5

    def test_inhibition_costs_more_than_go(self, pfc_with_goal):
        """Inhibition costs more energy than Go."""
        e_start = pfc_with_goal._energy

        pfc_with_goal.evaluate_action("go_action", z_action=75.0)
        e_after_go = pfc_with_goal._energy

        pfc_with_goal.evaluate_action("nogo_action", z_action=500.0)
        e_after_nogo = pfc_with_goal._energy

        go_cost = e_start - e_after_go
        nogo_cost = e_after_go - e_after_nogo

        assert nogo_cost > go_cost  # Not doing > doing

    def test_ego_depletion_cascade(self):
        """Willpower depletion cascade effect."""
        pfc = PrefrontalCortexEngine()
        pfc.set_goal("focus", z_goal=75.0, priority=0.9)

        inhibition_results = []
        for i in range(50):
            result = pfc.evaluate_action(f"temptation_{i}", z_action=500.0)
            inhibition_results.append(result)

        # Eventually some should fail
        decisions = [r.decision for r in inhibition_results]
        # First few should successfully inhibit
        assert decisions[0] == "nogo"
        # Later ones may have impulse breakthrough
        has_failure = any(
            r.reason == "inhibition_failure_energy_depleted"
            for r in inhibition_results
        )
        assert has_failure, "Sustained inhibition should lead to eventual failure"
