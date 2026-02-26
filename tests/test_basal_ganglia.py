# -*- coding: utf-8 -*-
"""
Basal Ganglia Tests — Phase 6.2

Coverage:
  1. Action channel management (register, get)
  2. Action selection (striatum, softmax, PFC bias)
  3. Habit formation (Γ decrease, automatization, formation events)
  4. Go/NoGo pathway (dopamine modulation, RPE)
  5. Hyperdirect pathway (emergency brake)
  6. Dual-system arbitration (habitual vs goal-directed)
  7. Tick cycle
  8. Habit breaking (PFC energy injection)
  9. Physical consistency
"""

import time
import numpy as np
import pytest

from alice.brain.basal_ganglia import (
    BasalGangliaEngine,
    ActionChannel,
    SelectionResult,
    HabitSnapshot,
    INITIAL_ACTION_GAMMA,
    HABIT_THRESHOLD,
    HABITUATION_RATE,
    DEHABITUATION_RATE,
    DA_BASELINE,
    DA_GO_BOOST,
    DA_NOGO_BOOST,
    HYPERDIRECT_GAIN,
    MIN_ACTIVATION,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def bg():
    """Fresh basal ganglia engine."""
    return BasalGangliaEngine()


@pytest.fixture
def bg_with_actions():
    """Basal ganglia with preset actions."""
    bg = BasalGangliaEngine()
    bg.register_action("kitchen", "make_coffee", initial_go=0.7, initial_nogo=0.3)
    bg.register_action("kitchen", "make_tea", initial_go=0.4, initial_nogo=0.5)
    bg.register_action("kitchen", "eat_snack", initial_go=0.6, initial_nogo=0.4)
    return bg


@pytest.fixture
def bg_with_habit():
    """Basal ganglia with a formed habit."""
    bg = BasalGangliaEngine()
    bg.register_action("morning", "brush_teeth")
    ch = bg._channels["morning"]["brush_teeth"]
    ch.gamma_habit = 0.05  # Already habituated
    ch.execution_count = 200
    ch.success_count = 195
    ch.go_strength = 0.9
    ch.nogo_strength = 0.1
    return bg


# ============================================================================
# 1. Action Channel Management
# ============================================================================


class TestActionChannel:
    """Action channel management."""

    def test_register_action(self, bg):
        """Register an action."""
        ch = bg.register_action("context", "action_a")
        assert ch.action_name == "action_a"
        assert ch.gamma_habit == INITIAL_ACTION_GAMMA

    def test_register_duplicate(self, bg):
        """Duplicate registration returns existing channel."""
        ch1 = bg.register_action("ctx", "act")
        ch2 = bg.register_action("ctx", "act")
        assert ch1 is ch2

    def test_get_channel(self, bg_with_actions):
        """Get action channel."""
        ch = bg_with_actions.get_channel("kitchen", "make_coffee")
        assert ch is not None
        assert ch.go_strength == 0.7

    def test_get_nonexistent_channel(self, bg):
        """Get nonexistent channel."""
        assert bg.get_channel("x", "y") is None

    def test_channel_net_activation(self):
        """Net activation = Go - NoGo."""
        ch = ActionChannel(action_name="test", go_strength=0.8, nogo_strength=0.3)
        assert ch.net_activation == pytest.approx(0.5, abs=0.01)

    def test_channel_net_activation_floor(self):
        """Net activation has a lower bound."""
        ch = ActionChannel(action_name="test", go_strength=0.1, nogo_strength=0.9)
        assert ch.net_activation >= MIN_ACTIVATION

    def test_channel_is_habit(self):
        """Habit determination."""
        ch = ActionChannel(action_name="test", gamma_habit=0.05)
        assert ch.is_habit

        ch2 = ActionChannel(action_name="test2", gamma_habit=0.5)
        assert not ch2.is_habit

    def test_channel_automaticity(self):
        """Automaticity level."""
        ch = ActionChannel(action_name="test", gamma_habit=0.0)
        assert ch.automaticity == pytest.approx(1.0)

        ch2 = ActionChannel(action_name="test2", gamma_habit=1.0)
        assert ch2.automaticity == pytest.approx(0.0)

    def test_channel_success_rate(self):
        """Success rate."""
        ch = ActionChannel(action_name="test", execution_count=10, success_count=7)
        assert ch.success_rate == pytest.approx(0.7)

    def test_channel_success_rate_zero(self):
        """Success rate with zero executions."""
        ch = ActionChannel(action_name="test")
        assert ch.success_rate == 0.0


# ============================================================================
# 2. Action Selection
# ============================================================================


class TestActionSelection:
    """Striatal action selection."""

    def test_basic_selection(self, bg_with_actions):
        """Basic action selection."""
        result = bg_with_actions.select_action(
            "kitchen", ["make_coffee", "make_tea", "eat_snack"]
        )
        assert result.selected_action in ["make_coffee", "make_tea", "eat_snack"]
        assert result.activation > 0

    def test_higher_go_preferred(self):
        """Action with higher Go is more likely to be selected."""
        bg = BasalGangliaEngine(temperature=0.01)  # Near-deterministic
        bg.register_action("test", "strong", initial_go=0.9, initial_nogo=0.1)
        bg.register_action("test", "weak", initial_go=0.1, initial_nogo=0.9)

        selections = {"strong": 0, "weak": 0}
        for _ in range(100):
            result = bg.select_action("test", ["strong", "weak"])
            selections[result.selected_action] += 1

        assert selections["strong"] > selections["weak"]

    def test_pfc_bias_influences_selection(self):
        """PFC bias influences selection."""
        bg = BasalGangliaEngine(temperature=0.01)
        bg.register_action("test", "goal_action", initial_go=0.3, initial_nogo=0.3)
        bg.register_action("test", "other", initial_go=0.5, initial_nogo=0.3)

        # Strong PFC bias should push goal_action
        selections = {"goal_action": 0, "other": 0}
        for _ in range(100):
            result = bg.select_action(
                "test", ["goal_action", "other"],
                pfc_bias={"goal_action": 2.0}  # Strong bias
            )
            selections[result.selected_action] += 1

        assert selections["goal_action"] > selections["other"]

    def test_habitual_action_faster(self, bg_with_habit):
        """Habitual action reacts faster."""
        bg_with_habit.register_action("morning", "novel_action")

        result_habit = bg_with_habit.select_action("morning", ["brush_teeth"])
        result_novel = bg_with_habit.select_action("morning", ["novel_action"])

        assert result_habit.reaction_time < result_novel.reaction_time

    def test_habitual_no_cortical(self, bg_with_habit):
        """Habitual action does not require cortex."""
        result = bg_with_habit.select_action("morning", ["brush_teeth"])
        if result.selected_action == "brush_teeth":
            assert not result.cortical_needed

    def test_novel_needs_cortical(self, bg):
        """Novel action requires cortex."""
        bg.register_action("test", "new_action")
        result = bg.select_action("test", ["new_action"])
        assert result.cortical_needed

    def test_selection_counter(self, bg_with_actions):
        """Selection counter."""
        bg_with_actions.select_action("kitchen", ["make_coffee"])
        assert bg_with_actions._total_selections >= 1

    def test_competing_actions_sorted(self, bg_with_actions):
        """Competing actions sorted by activation."""
        result = bg_with_actions.select_action(
            "kitchen", ["make_coffee", "make_tea", "eat_snack"]
        )
        activations = [c["activation"] for c in result.competing_actions]
        assert activations == sorted(activations, reverse=True)

    def test_emergency_brake_blocks_selection(self, bg_with_actions):
        """Emergency brake blocks all selections."""
        bg_with_actions.emergency_brake()
        result = bg_with_actions.select_action(
            "kitchen", ["make_coffee", "make_tea"]
        )
        assert result.selected_action == "BRAKE"
        assert result.pathway == "hyperdirect"

    def test_auto_register_actions(self, bg):
        """Auto-register unknown actions."""
        result = bg.select_action("new_ctx", ["action_x", "action_y"])
        assert result.selected_action in ["action_x", "action_y"]


# ============================================================================
# 3. Habit Formation
# ============================================================================


class TestHabitFormation:
    """Impedance model of habit formation."""

    def test_positive_reward_decreases_gamma(self, bg):
        """Positive reward decreases Γ → closer to habit."""
        bg.register_action("test", "action")
        gamma_before = bg.get_channel("test", "action").gamma_habit

        bg.update_after_action("test", "action", reward=1.0, success=True)
        gamma_after = bg.get_channel("test", "action").gamma_habit

        assert gamma_after < gamma_before

    def test_negative_reward_increases_gamma(self, bg):
        """Negative reward increases Γ → away from habit."""
        bg.register_action("test", "action")
        # First execute with positive reward to establish baseline
        bg.update_after_action("test", "action", reward=1.0, success=True)
        gamma_before = bg.get_channel("test", "action").gamma_habit

        # Negative reward → RPE = -1.0 - 1.0 = -2.0 → Γ increases
        result = bg.update_after_action("test", "action", reward=-1.0, success=False)
        gamma_after = bg.get_channel("test", "action").gamma_habit

        assert result["rpe"] < 0, f"RPE should be negative, got {result['rpe']}"
        assert gamma_after > gamma_before

    def test_habit_forms_through_repetition(self, bg):
        """Repeated practice forms a habit."""
        bg.register_action("practice", "scales")

        for i in range(100):
            bg.update_after_action("practice", "scales", reward=0.5, success=True)

        ch = bg.get_channel("practice", "scales")
        assert ch.is_habit, f"After 100 reps, gamma={ch.gamma_habit} should be habit"

    def test_habit_formation_event(self, bg):
        """Habit formation event recording."""
        bg.register_action("practice", "chord")

        formed = False
        for i in range(200):
            result = bg.update_after_action("practice", "chord", reward=0.5, success=True)
            if result["newly_formed_habit"]:
                formed = True
                break

        assert formed, "Habit should form after sufficient repetition"
        assert bg._total_habits_formed >= 1

    def test_habit_trajectory_tracking(self, bg):
        """Habit formation trajectory tracking."""
        bg.register_action("test", "walk")
        for _ in range(10):
            bg.update_after_action("test", "walk", reward=0.5)

        traj = bg.get_habit_trajectory("test", "walk")
        assert traj is not None
        assert len(traj["gamma_history"]) == 10
        # Γ should decrease
        assert traj["gamma_history"][-1] < traj["gamma_history"][0]

    def test_success_count_tracking(self, bg):
        """Success count tracking."""
        bg.register_action("test", "action")
        bg.update_after_action("test", "action", reward=0.5, success=True)
        bg.update_after_action("test", "action", reward=-0.5, success=False)

        ch = bg.get_channel("test", "action")
        assert ch.execution_count == 2
        assert ch.success_count == 1

    def test_get_all_habits(self, bg_with_habit):
        """Get all formed habits."""
        habits = bg_with_habit.get_all_habits()
        assert len(habits) >= 1
        assert habits[0]["action"] == "brush_teeth"

    def test_zero_rpe_still_habituates(self, bg):
        """RPE=0 still has mild habituation."""
        bg.register_action("test", "routine")
        # First set baseline
        bg.update_after_action("test", "routine", reward=0.5, success=True)
        gamma_before = bg.get_channel("test", "routine").gamma_habit

        # Second time same reward → RPE ≈ 0
        bg.update_after_action("test", "routine", reward=0.5, success=True)
        gamma_after = bg.get_channel("test", "routine").gamma_habit

        assert gamma_after <= gamma_before


# ============================================================================
# 4. Go/NoGo Pathway
# ============================================================================


class TestGoNoGoPathway:
    """Dopamine modulation of Go/NoGo pathway."""

    def test_positive_rpe_boosts_go(self, bg):
        """Positive RPE strengthens Go pathway."""
        bg.register_action("test", "action")
        go_before = bg.get_channel("test", "action").go_strength

        bg.update_after_action("test", "action", reward=1.0, success=True)
        go_after = bg.get_channel("test", "action").go_strength

        assert go_after > go_before

    def test_negative_rpe_boosts_nogo(self, bg):
        """Negative RPE strengthens NoGo pathway."""
        bg.register_action("test", "action")
        # First positive reward to establish expectation
        bg.update_after_action("test", "action", reward=1.0, success=True)
        nogo_before = bg.get_channel("test", "action").nogo_strength

        # Negative reward
        bg.update_after_action("test", "action", reward=-1.0, success=False)
        nogo_after = bg.get_channel("test", "action").nogo_strength

        assert nogo_after > nogo_before

    def test_dopamine_increases_with_positive_rpe(self, bg):
        """Positive RPE increases dopamine."""
        bg.register_action("test", "action")
        da_before = bg._dopamine_level

        bg.update_after_action("test", "action", reward=1.0, success=True)
        assert bg._dopamine_level > da_before

    def test_dopamine_decreases_with_negative_rpe(self, bg):
        """Negative RPE decreases dopamine."""
        bg.register_action("test", "action")
        # First positive reward to set baseline
        bg.update_after_action("test", "action", reward=1.0, success=True)
        da_before = bg._dopamine_level

        bg.update_after_action("test", "action", reward=-1.0, success=False)
        assert bg._dopamine_level < da_before

    def test_dopamine_clamped(self, bg):
        """Dopamine clamped to [0,1] range."""
        bg.register_action("test", "action")
        for _ in range(50):
            bg.update_after_action("test", "action", reward=10.0)
        assert 0.0 <= bg._dopamine_level <= 1.0

    def test_rpe_reported(self, bg):
        """RPE correctly reported."""
        bg.register_action("test", "action")
        result = bg.update_after_action("test", "action", reward=1.0)
        assert "rpe" in result
        assert "dopamine" in result


# ============================================================================
# 5. Hyperdirect Pathway
# ============================================================================


class TestHyperdirectPathway:
    """Hyperdirect pathway — emergency brake."""

    def test_emergency_brake_activation(self, bg):
        """Emergency brake activation."""
        result = bg.emergency_brake(duration=5)
        assert result["brake_activated"]
        assert bg._emergency_brake

    def test_emergency_brake_blocks_all(self, bg_with_actions):
        """Brake blocks all actions."""
        bg_with_actions.emergency_brake()
        result = bg_with_actions.select_action(
            "kitchen", ["make_coffee", "make_tea"]
        )
        assert result.selected_action == "BRAKE"

    def test_emergency_brake_timeout(self, bg):
        """Brake auto-releases on timeout."""
        bg.emergency_brake(duration=2)
        assert bg._emergency_brake

        bg.tick()
        bg.tick()
        assert not bg._emergency_brake

    def test_emergency_brake_counter(self, bg):
        """Brake counter."""
        bg.emergency_brake()
        bg._emergency_brake = False  # Manual reset
        bg.emergency_brake()
        assert bg._total_emergency_brakes == 2

    def test_nogo_increases_during_brake(self, bg_with_actions):
        """NoGo strengthens during brake."""
        nogo_before = bg_with_actions.get_channel("kitchen", "make_coffee").nogo_strength
        bg_with_actions.emergency_brake()
        nogo_after = bg_with_actions.get_channel("kitchen", "make_coffee").nogo_strength
        assert nogo_after > nogo_before


# ============================================================================
# 6. Dual-System Arbitration
# ============================================================================


class TestDualSystem:
    """Habitual vs goal-directed dual system."""

    def test_stable_environment_favors_habit(self, bg):
        """Stable environment → habitual system."""
        result = bg.arbitrate_systems(
            habit_action="routine_route",
            goal_action="explore_route",
            habit_gamma=0.1,           # High habituation
            goal_uncertainty=0.1,      # Low uncertainty
            reward_rate=0.8,           # High reward rate
        )
        assert result["chosen_system"] == "habitual"

    def test_uncertain_environment_favors_goal(self, bg):
        """Uncertain environment → goal-directed system."""
        result = bg.arbitrate_systems(
            habit_action="routine_route",
            goal_action="explore_route",
            habit_gamma=0.8,           # Low habituation
            goal_uncertainty=0.9,      # High uncertainty
            reward_rate=0.2,           # Low reward rate
        )
        assert result["chosen_system"] == "goal_directed"

    def test_arbitration_weights_sum_to_one(self, bg):
        """Arbitration weights sum to 1."""
        result = bg.arbitrate_systems(
            habit_action="A", goal_action="B",
            habit_gamma=0.5, goal_uncertainty=0.5,
        )
        total = result["habit_weight"] + result["goal_weight"]
        assert total == pytest.approx(1.0, abs=0.01)

    def test_arbitration_returns_chosen(self, bg):
        """Arbitration returns the chosen action."""
        result = bg.arbitrate_systems(
            habit_action="old_way",
            goal_action="new_way",
            habit_gamma=0.1, goal_uncertainty=0.1,
        )
        assert result["chosen_action"] in ["old_way", "new_way"]


# ============================================================================
# 7. Habit Breaking
# ============================================================================


class TestBreakHabit:
    """Habit breaking — PFC energy injection."""

    def test_break_habit_increases_gamma(self, bg_with_habit):
        """Breaking a habit increases Γ."""
        gamma_before = bg_with_habit.get_channel("morning", "brush_teeth").gamma_habit
        result = bg_with_habit.break_habit("morning", "brush_teeth", pfc_energy_input=0.5)
        assert result["gamma_after"] > gamma_before

    def test_break_nonexistent_habit(self, bg):
        """Break nonexistent habit."""
        result = bg.break_habit("nowhere", "nothing")
        assert "error" in result

    def test_break_non_habit(self, bg):
        """Break a non-habitual action."""
        bg.register_action("test", "new_action")
        result = bg.break_habit("test", "new_action")
        assert not result["was_habit"]

    def test_higher_pfc_energy_more_effective(self, bg_with_habit):
        """More PFC energy → more effective breaking."""
        bg2 = BasalGangliaEngine()
        bg2.register_action("morning", "brush_teeth")
        bg2._channels["morning"]["brush_teeth"].gamma_habit = 0.05

        r_low = bg_with_habit.break_habit("morning", "brush_teeth", pfc_energy_input=0.1)
        r_high = bg2.break_habit("morning", "brush_teeth", pfc_energy_input=0.9)

        delta_low = r_low["gamma_after"] - r_low["gamma_before"]
        delta_high = r_high["gamma_after"] - r_high["gamma_before"]

        assert delta_high > delta_low


# ============================================================================
# 8. Tick Cycle
# ============================================================================


class TestTickCycle:
    """Basal ganglia tick cycle."""

    def test_tick_increments(self, bg):
        """Tick increments."""
        bg.tick()
        bg.tick()
        assert bg._tick_count == 2

    def test_dopamine_decays_to_baseline(self, bg):
        """Dopamine decays to baseline."""
        bg._dopamine_level = 0.9
        for _ in range(200):
            bg.tick()
        assert abs(bg._dopamine_level - DA_BASELINE) < 0.1

    def test_dopamine_history_tracking(self, bg):
        """Dopamine history tracking."""
        for _ in range(5):
            bg.tick()
        assert len(bg._dopamine_history) >= 6

    def test_dopamine_history_trimming(self, bg):
        """Dopamine history trimming."""
        bg._dopamine_history = list(range(1500))
        bg.tick()
        assert len(bg._dopamine_history) <= 600

    def test_tick_returns_state(self, bg):
        """Tick returns state."""
        result = bg.tick()
        assert "tick" in result
        assert "dopamine" in result
        assert "emergency_brake" in result


# ============================================================================
# 9. Query Interface
# ============================================================================


class TestQueryInterface:
    """Query interface."""

    def test_get_state(self, bg_with_actions):
        """Get complete state."""
        state = bg_with_actions.get_state()
        assert "dopamine" in state
        assert "total_channels" in state
        assert state["total_channels"] == 3

    def test_get_stats(self, bg_with_actions):
        """Get statistics."""
        bg_with_actions.select_action("kitchen", ["make_coffee"])
        stats = bg_with_actions.get_stats()
        assert "total_selections" in stats
        assert stats["total_selections"] >= 1

    def test_get_habit_trajectory_none(self, bg):
        """No trajectory returns None."""
        assert bg.get_habit_trajectory("x", "y") is None

    def test_get_dopamine_history(self, bg):
        """Get dopamine history."""
        history = bg.get_dopamine_history()
        assert len(history) >= 1


# ============================================================================
# 10. Physical Consistency
# ============================================================================


class TestPhysicalConsistency:
    """Γ-Net impedance model physical validity."""

    def test_gamma_decreases_monotonically_with_practice(self, bg):
        """Sustained practice → Γ monotonically decreases (until habit is formed)."""
        bg.register_action("test", "skill")

        gammas = []
        for _ in range(50):
            bg.update_after_action("test", "skill", reward=0.5, success=True)
            gammas.append(bg.get_channel("test", "skill").gamma_habit)

        # Overall trend should be decreasing (minor fluctuations allowed)
        assert gammas[-1] < gammas[0]

    def test_habit_gamma_in_range(self, bg):
        """Γ_habit in physical range [0,1]."""
        bg.register_action("test", "action")
        for _ in range(200):
            bg.update_after_action("test", "action", reward=np.random.uniform(-1, 1))
            ch = bg.get_channel("test", "action")
            assert 0.0 <= ch.gamma_habit <= 1.0

    def test_automaticity_inversely_proportional_to_gamma(self):
        """Automaticity ∝ 1/Γ."""
        ch = ActionChannel(action_name="test")
        ch.gamma_habit = 0.8
        auto_low = ch.automaticity

        ch.gamma_habit = 0.1
        auto_high = ch.automaticity

        assert auto_high > auto_low

    def test_dual_system_smooth_transition(self, bg):
        """Smooth transition in dual-system arbitration."""
        weights = []
        for uncertainty in np.linspace(0.0, 1.0, 20):
            result = bg.arbitrate_systems(
                habit_action="A", goal_action="B",
                habit_gamma=0.3, goal_uncertainty=uncertainty,
            )
            weights.append(result["goal_weight"])

        # Increasing uncertainty → increasing goal system weight
        assert weights[-1] > weights[0]

    def test_go_nogo_conservation(self, bg):
        """Go + NoGo strengths in reasonable range."""
        bg.register_action("test", "action")
        for _ in range(100):
            reward = np.random.uniform(-1, 1)
            bg.update_after_action("test", "action", reward=reward)

        ch = bg.get_channel("test", "action")
        assert 0.0 <= ch.go_strength <= 1.0
        assert 0.0 <= ch.nogo_strength <= 1.0
