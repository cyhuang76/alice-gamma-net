# -*- coding: utf-8 -*-
"""
Tests for alice.body.hand — Alice's hand (motor output)

Test coverage:
  1. PID controller convergence
  2. Physical displacement validity
  3. Muscle tension & anxiety tremor
  4. Hand-eye coordination error
  5. Dopamine signal
  6. Proprioception electrical signal
  7. Boundary conditions
  8. AliceBrain integration
"""

import math
import pytest
import numpy as np

from alice.body.hand import (
    AliceHand,
    _PIDController,
    PID_KP,
    PID_KI,
    PID_KD,
    REACH_THRESHOLD,
    DOPAMINE_ON_REACH,
    MUSCLE_MAX_FORCE,
    PROPRIOCEPTION_IMPEDANCE,
    TREMOR_FREQ_HZ,
)
from alice.core.signal import ElectricalSignal


# ============================================================================
# PID Controller Tests
# ============================================================================


class TestPIDController:
    """PID controller unit tests."""

    def test_p_term_proportional(self):
        """P term is proportional to error."""
        pid = _PIDController(kp=2.0, ki=0.0, kd=0.0)
        u1 = pid.update(1.0, dt=0.01)
        pid.reset()
        u2 = pid.update(2.0, dt=0.01)
        assert abs(u2 / u1 - 2.0) < 0.01

    def test_i_term_accumulates(self):
        """I term accumulates with persistent error."""
        pid = _PIDController(kp=0.0, ki=1.0, kd=0.0)
        u1 = pid.update(1.0, dt=0.1)
        u2 = pid.update(1.0, dt=0.1)
        assert u2 > u1  # Integral is growing

    def test_d_term_responds_to_change(self):
        """D term responds to rate of error change."""
        pid = _PIDController(kp=0.0, ki=0.0, kd=1.0)
        _ = pid.update(0.0, dt=0.01)
        u = pid.update(1.0, dt=0.01)
        assert u != 0.0  # 𝛥error / dt ≠ 0

    def test_d_term_zero_when_constant_error(self):
        """D term approaches 0 with constant error."""
        pid = _PIDController(kp=0.0, ki=0.0, kd=1.0)
        pid.update(5.0, dt=0.01)
        u = pid.update(5.0, dt=0.01)
        assert abs(u) < 0.01  # Constant error → Δerror ≈ 0

    def test_anti_windup(self):
        """Integrator has anti-windup limit."""
        pid = _PIDController(kp=0.0, ki=1.0, kd=0.0)
        for _ in range(10000):
            pid.update(100.0, dt=1.0)
        assert abs(pid._integral) <= 50.0

    def test_reset(self):
        """Reset returns to initial state."""
        pid = _PIDController(kp=1.0, ki=1.0, kd=1.0)
        pid.update(10.0, dt=0.1)
        pid.reset()
        assert pid._integral == 0.0
        assert pid._last_error == 0.0

    def test_zero_error_zero_output(self):
        """Zero error → zero output (initial state)."""
        pid = _PIDController(kp=1.0, ki=0.0, kd=0.0)
        u = pid.update(0.0, dt=0.01)
        assert u == 0.0


# ============================================================================
# AliceHand Basic Functionality
# ============================================================================


class TestAliceHandBasics:
    """Hand basic functionality tests."""

    def test_init_default_position(self):
        """Initial position at screen center."""
        hand = AliceHand()
        assert hand.x == 960.0
        assert hand.y == 540.0

    def test_init_custom_position(self):
        """Custom initial position."""
        hand = AliceHand(initial_pos=(100.0, 200.0))
        assert hand.x == 100.0
        assert hand.y == 200.0

    def test_init_custom_workspace(self):
        """Custom workspace size."""
        hand = AliceHand(workspace_size=(800.0, 600.0))
        assert hand.workspace_w == 800.0
        assert hand.workspace_h == 600.0
        assert hand.x == 400.0  # Center
        assert hand.y == 300.0

    def test_init_stats_zero(self):
        """Initial statistics are zero."""
        hand = AliceHand()
        assert hand.total_movements == 0
        assert hand.total_reaches == 0
        assert hand.total_dopamine == 0.0


# ============================================================================
# PID-Driven Physical Movement
# ============================================================================


class TestPhysicalMovement:
    """PID controller driven physical movement tests."""

    def test_reach_arrives_at_target(self):
        """Hand can reach the target (PID convergence)."""
        hand = AliceHand(initial_pos=(100.0, 100.0))
        result = hand.reach(500.0, 300.0, ram_temperature=0.0)
        assert result["reached"] is True
        assert result["final_error"] < REACH_THRESHOLD

    def test_reach_non_instant(self):
        """Movement is not instant — requires multiple steps."""
        hand = AliceHand(initial_pos=(0.0, 0.0))
        result = hand.reach(800.0, 600.0, ram_temperature=0.0)
        assert result["steps"] > 1  # Not a single-step arrival

    def test_reach_trajectory_is_smooth(self):
        """Trajectory is physically smooth (no jumps)."""
        hand = AliceHand(initial_pos=(100.0, 100.0))
        result = hand.reach(500.0, 300.0, ram_temperature=0.0)
        trajectory = result["trajectory"]

        # Distance between consecutive points should not be too large
        max_jump = 0.0
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i - 1][0]
            dy = trajectory[i][1] - trajectory[i - 1][1]
            jump = math.sqrt(dx * dx + dy * dy)
            max_jump = max(max_jump, jump)

        # At 60FPS dt=0.016, max velocity won't jump too far
        assert max_jump < 100, f"Trajectory jump too large: {max_jump}"

    def test_reach_updates_position(self):
        """Position is updated after reach."""
        hand = AliceHand(initial_pos=(100.0, 100.0))
        hand.reach(500.0, 300.0, ram_temperature=0.0)
        # Position should be close to target after reaching
        assert abs(hand.x - 500.0) < REACH_THRESHOLD + 1
        assert abs(hand.y - 300.0) < REACH_THRESHOLD + 1

    def test_reach_with_motor_signal(self):
        """Motor cortex signal affects force gain."""
        hand1 = AliceHand(initial_pos=(100.0, 100.0))
        signal = ElectricalSignal(
            waveform=np.sin(np.linspace(0, 2 * np.pi, 64)),
            amplitude=2.0,  # High gain
            frequency=30.0,
            phase=0.0,
            impedance=75.0,
            snr=15.0,
            source="motor_cortex",
            modality="motor",
        )
        result = hand1.reach(500.0, 300.0, motor_signal=signal, ram_temperature=0.0)
        assert result["reached"] is True

    def test_reach_close_target(self):
        """Very close target is reached quickly."""
        hand = AliceHand(initial_pos=(100.0, 100.0))
        result = hand.reach(101.0, 101.0, ram_temperature=0.0)
        assert result["reached"] is True
        assert result["steps"] <= 10

    def test_reach_same_position(self):
        """Target = current position → 0 steps."""
        hand = AliceHand(initial_pos=(100.0, 100.0))
        result = hand.reach(100.0, 100.0, ram_temperature=0.0)
        assert result["reached"] is True
        assert result["steps"] <= 1


# ============================================================================
# Muscle Tension & Anxiety Tremor
# ============================================================================


class TestMuscleTensionAndTremor:
    """Muscle tension + anxiety tremor tests."""

    def test_no_tremor_when_calm(self):
        """ram_temperature=0 means almost no tremor."""
        hand = AliceHand(initial_pos=(100.0, 100.0))
        result = hand.reach(500.0, 300.0, ram_temperature=0.0)
        assert result["tremor_intensity"] < 0.5

    def test_tremor_increases_with_anxiety(self):
        """Higher anxiety means stronger tremor."""
        hand_calm = AliceHand(initial_pos=(100.0, 100.0))
        hand_anxious = AliceHand(initial_pos=(100.0, 100.0))

        r_calm = hand_calm.reach(500.0, 300.0, ram_temperature=0.1)
        r_anxious = hand_anxious.reach(500.0, 300.0, ram_temperature=0.8)

        assert r_anxious["tremor_intensity"] > r_calm["tremor_intensity"]

    def test_high_anxiety_may_fail_reach(self):
        """Extreme anxiety may prevent precise target reaching."""
        hand = AliceHand(initial_pos=(100.0, 100.0))
        # High anxiety + precise target (small workspace)
        result = hand.reach(105.0, 100.0, ram_temperature=0.95, max_steps=50)
        # Not guaranteed to reach, but tremor should be large
        assert result["tremor_intensity"] > 1.0

    def test_peak_tension_positive(self):
        """Muscle tension > 0 during physical movement."""
        hand = AliceHand(initial_pos=(100.0, 100.0))
        result = hand.reach(500.0, 300.0, ram_temperature=0.0)
        assert result["peak_tension"] > 0

    def test_tremor_quadratic_scaling(self):
        """
        Tremor amplitude ∝ temperature² (nonlinear).

        Doubling temperature → approximately 4x tremor (nonlinear runaway).
        """
        hand_low = AliceHand(initial_pos=(100.0, 100.0))
        hand_high = AliceHand(initial_pos=(100.0, 100.0))

        r_low = hand_low.reach(500.0, 300.0, ram_temperature=0.3)
        r_high = hand_high.reach(500.0, 300.0, ram_temperature=0.6)

        # Temperature 0.3→0.6 (doubled), tremor 0.09→0.36 (about 4x)
        # Allow some error (random noise)
        ratio = r_high["tremor_intensity"] / max(r_low["tremor_intensity"], 0.001)
        assert ratio > 2.0, f"Tremor ratio {ratio} should be > 2"

    def test_coordination_drops_with_anxiety(self):
        """High anxiety → decreased coordination → more steps."""
        hand_calm = AliceHand(initial_pos=(100.0, 100.0))
        hand_stressed = AliceHand(initial_pos=(100.0, 100.0))

        r_calm = hand_calm.reach(500.0, 300.0, ram_temperature=0.0, max_steps=300)
        r_stressed = hand_stressed.reach(500.0, 300.0, ram_temperature=0.5, max_steps=300)

        # With anxiety usually more steps needed (or can't reach at all)
        if r_calm["reached"] and r_stressed["reached"]:
            assert r_stressed["steps"] >= r_calm["steps"]


# ============================================================================
# Dopamine Signal
# ============================================================================


class TestDopamine:
    """Dopamine signal tests."""

    def test_dopamine_on_reach(self):
        """Reached target → dopamine > 0."""
        hand = AliceHand(initial_pos=(100.0, 100.0))
        result = hand.reach(500.0, 300.0, ram_temperature=0.0)
        assert result["reached"] is True
        assert result["dopamine"] == DOPAMINE_ON_REACH

    def test_no_dopamine_on_miss(self):
        """Did not reach → dopamine = 0."""
        hand = AliceHand(initial_pos=(100.0, 100.0))
        result = hand.reach(1900.0, 1050.0, ram_temperature=0.0, max_steps=1)
        # Only 1 step given, impossible to reach
        assert result["reached"] is False
        assert result["dopamine"] == 0.0

    def test_cumulative_dopamine(self):
        """Multiple successes → dopamine accumulates."""
        hand = AliceHand(initial_pos=(100.0, 100.0))
        hand.reach(200.0, 200.0, ram_temperature=0.0)
        hand.reach(300.0, 300.0, ram_temperature=0.0)
        assert hand.total_dopamine >= DOPAMINE_ON_REACH * 2

    def test_reach_counter(self):
        """Successful reach counter."""
        hand = AliceHand(initial_pos=(100.0, 100.0))
        hand.reach(200.0, 200.0, ram_temperature=0.0)
        hand.reach(300.0, 300.0, ram_temperature=0.0)
        assert hand.total_reaches == 2
        assert hand.total_movements == 2


# ============================================================================
# Hand-Eye Coordination
# ============================================================================


class TestHandEyeCoordination:
    """Hand-eye coordination error tests."""

    def test_error_when_off_target(self):
        """Off-target has error."""
        hand = AliceHand(initial_pos=(100.0, 100.0))
        coord = hand.compute_hand_eye_error(500.0, 300.0)
        assert coord["error_magnitude"] > 0
        assert coord["reached"] is False
        assert coord["dopamine"] == 0.0

    def test_no_error_when_on_target(self):
        """Near target → reached + dopamine."""
        hand = AliceHand(initial_pos=(100.0, 100.0))
        coord = hand.compute_hand_eye_error(100.5, 100.5)
        assert coord["reached"] is True
        assert coord["dopamine"] == DOPAMINE_ON_REACH
        assert coord["error_magnitude"] < REACH_THRESHOLD

    def test_error_signal_is_electrical(self):
        """Error is encoded as an ElectricalSignal."""
        hand = AliceHand(initial_pos=(100.0, 100.0))
        coord = hand.compute_hand_eye_error(500.0, 300.0)
        sig = coord["error_signal"]
        assert isinstance(sig, ElectricalSignal)
        assert sig.source == "hand_eye_error"
        assert sig.modality == "proprioception"

    def test_error_signal_amplitude_proportional(self):
        """Large error → large signal amplitude."""
        hand = AliceHand(initial_pos=(100.0, 100.0))
        coord_near = hand.compute_hand_eye_error(102.0, 100.0)
        coord_far = hand.compute_hand_eye_error(800.0, 600.0)

        assert coord_far["error_signal"].amplitude > coord_near["error_signal"].amplitude

    def test_error_xy_direction(self):
        """Error XY components have correct direction."""
        hand = AliceHand(initial_pos=(100.0, 200.0))
        coord = hand.compute_hand_eye_error(300.0, 100.0)
        assert coord["error_x"] > 0   # Target is to the right
        assert coord["error_y"] < 0   # Target is above


# ============================================================================
# Proprioception
# ============================================================================


class TestProprioception:
    """Proprioception electrical signal tests."""

    def test_proprioception_returns_signal(self):
        """get_proprioception returns an ElectricalSignal."""
        hand = AliceHand()
        sig = hand.get_proprioception()
        assert isinstance(sig, ElectricalSignal)

    def test_proprioception_source(self):
        """Source is hand."""
        hand = AliceHand()
        sig = hand.get_proprioception()
        assert sig.source == "hand"
        assert sig.modality == "proprioception"

    def test_proprioception_impedance(self):
        """Impedance = proprioception impedance."""
        hand = AliceHand()
        sig = hand.get_proprioception()
        assert sig.impedance == PROPRIOCEPTION_IMPEDANCE

    def test_proprioception_velocity_frequency(self):
        """High velocity → high frequency."""
        hand_still = AliceHand()
        hand_moving = AliceHand()
        hand_moving.vx = 50.0
        hand_moving.vy = 50.0

        sig_still = hand_still.get_proprioception()
        sig_moving = hand_moving.get_proprioception()

        assert sig_moving.frequency > sig_still.frequency

    def test_proprioception_waveform_shape(self):
        """Waveform length = 128."""
        hand = AliceHand()
        sig = hand.get_proprioception()
        assert len(sig.waveform) == 128


# ============================================================================
# tick (Single-Step Update)
# ============================================================================


class TestTick:
    """tick single-step physics update."""

    def test_tick_no_target(self):
        """No target: tick does not move."""
        hand = AliceHand(initial_pos=(100.0, 100.0))
        result = hand.tick()
        assert result["error"] == 0.0
        assert result["reached"] is False

    def test_tick_moves_toward_target(self):
        """With target: tick moves toward target."""
        hand = AliceHand(initial_pos=(100.0, 100.0))
        hand.set_target(500.0, 300.0)

        pos_before = (hand.x, hand.y)
        hand.tick(ram_temperature=0.0)
        pos_after = (hand.x, hand.y)

        # Distance should decrease
        err_before = math.sqrt((500 - pos_before[0])**2 + (300 - pos_before[1])**2)
        err_after = math.sqrt((500 - pos_after[0])**2 + (300 - pos_after[1])**2)
        assert err_after < err_before

    def test_tick_convergence(self):
        """Multiple ticks can reach the target."""
        hand = AliceHand(initial_pos=(100.0, 100.0))
        hand.set_target(300.0, 200.0)

        for _ in range(500):
            result = hand.tick(ram_temperature=0.0)
            if result["reached"]:
                break

        assert result["reached"] is True

    def test_release_clears_target(self):
        """release clears the target."""
        hand = AliceHand()
        hand.set_target(500.0, 300.0)
        hand.release()
        assert hand._target_x is None
        assert hand._target_y is None


# ============================================================================
# Boundary Conditions
# ============================================================================


class TestBoundaryConditions:
    """Boundary condition tests."""

    def test_position_stays_in_workspace(self):
        """Hand does not move outside workspace."""
        hand = AliceHand(workspace_size=(800.0, 600.0), initial_pos=(10.0, 10.0))
        # Target outside boundary
        result = hand.reach(-100.0, -100.0, ram_temperature=0.0)
        assert hand.x >= 0
        assert hand.y >= 0

    def test_target_clamped_to_workspace(self):
        """Target is clamped to workspace."""
        hand = AliceHand(workspace_size=(800.0, 600.0))
        hand.set_target(9999.0, -500.0)
        assert hand._target_x == 800.0
        assert hand._target_y == 0.0

    def test_force_saturation(self):
        """Muscle force has saturation limit."""
        hand = AliceHand(initial_pos=(0.0, 0.0))
        # Very far target → huge error → large PID output → but force is clipped
        result = hand.reach(1900.0, 1050.0, ram_temperature=0.0)
        assert result["peak_tension"] <= MUSCLE_MAX_FORCE + 1.0  # Allow float error


# ============================================================================
# Statistics & Diagnostics
# ============================================================================


class TestStatsAndDiagnostics:
    """Statistics functionality."""

    def test_get_stats(self):
        """get_stats returns complete statistics."""
        hand = AliceHand()
        stats = hand.get_stats()
        assert "total_movements" in stats
        assert "total_reaches" in stats
        assert "reach_rate" in stats
        assert "total_dopamine" in stats
        assert "position" in stats

    def test_reach_rate(self):
        """Success rate is computed correctly."""
        hand = AliceHand(initial_pos=(100.0, 100.0))
        hand.reach(200.0, 200.0, ram_temperature=0.0)  # Should reach
        hand.reach(1900.0, 1050.0, ram_temperature=0.0, max_steps=1)  # Impossible to reach
        stats = hand.get_stats()
        assert stats["reach_rate"] == 0.5

    def test_get_trajectory(self):
        """get_trajectory returns trajectory."""
        hand = AliceHand(initial_pos=(100.0, 100.0))
        hand.reach(300.0, 200.0, ram_temperature=0.0)
        traj = hand.get_trajectory()
        assert len(traj) > 0
        assert isinstance(traj[0], tuple)

    def test_get_muscle_state(self):
        """get_muscle_state returns muscle state."""
        hand = AliceHand()
        state = hand.get_muscle_state(ram_temperature=0.5)
        assert "tremor_level" in state
        assert "coordination" in state
        assert state["tremor_level"] == round(0.5 * 0.5, 4)  # quadratic
        assert state["coordination"] == round(1.0 - 0.4 * 0.5, 4)


# ============================================================================
# AliceBrain Integration
# ============================================================================


class TestAliceBrainIntegration:
    """AliceBrain + AliceHand integration."""

    def test_brain_has_hand(self):
        """AliceBrain has a hand attribute."""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        assert hasattr(brain, "hand")
        assert isinstance(brain.hand, AliceHand)

    def test_reach_for_basic(self):
        """reach_for complete loop."""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        result = brain.reach_for(500.0, 300.0)
        assert "reach" in result
        assert "coordination" in result
        assert "proprioception" in result
        assert "dopamine" in result

    def test_reach_for_reaches_target(self):
        """Target can be reached via AliceBrain.reach_for."""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        brain.hand = AliceHand(initial_pos=(100.0, 100.0))
        result = brain.reach_for(500.0, 300.0)
        assert result["reach"]["reached"] is True

    def test_reach_for_dopamine_feedback(self):
        """Reached target → dopamine > 0."""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        brain.hand = AliceHand(initial_pos=(100.0, 100.0))
        result = brain.reach_for(300.0, 200.0)
        if result["reach"]["reached"]:
            # Dopamine may be TD error, could be positive or negative, but not None
            assert result["dopamine"] is not None

    def test_reach_for_with_anxiety(self):
        """Anxiety affects reach_for."""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        brain.hand = AliceHand(initial_pos=(100.0, 100.0))
        brain.vitals.ram_temperature = 0.8  # High anxiety
        result = brain.reach_for(500.0, 300.0)
        assert result["vitals"]["ram_temperature"] == 0.8

    def test_introspect_has_hand(self):
        """introspect includes hand statistics."""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        intro = brain.introspect()
        assert "hand" in intro["subsystems"]

    def test_get_hand_state(self):
        """get_hand_state returns correctly."""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        state = brain.get_hand_state()
        assert "position" in state
        assert "tremor_level" in state
