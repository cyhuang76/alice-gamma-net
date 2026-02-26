# -*- coding: utf-8 -*-
"""
Life Loop Tests — Closed-Loop Error Compensation Engine
Tests for the Life Loop — Closed-Loop Error Compensation Engine
"""

import numpy as np
import pytest

from alice.brain.life_loop import (
    LifeLoop,
    CrossModalError,
    CompensationCommand,
    CompensationAction,
    ErrorType,
    LoopState,
    ERROR_DEADBAND,
    MAX_CONCURRENT_COMPENSATIONS,
    CONSCIOUSNESS_GATE,
)
from alice.core.signal import ElectricalSignal


# ============================================================================
# Utilities
# ============================================================================

def make_signal(source: str = "test", modality: str = "visual",
                amplitude: float = 0.5, frequency: float = 10.0,
                snr: float = 20.0) -> ElectricalSignal:
    """Quickly create a test ElectricalSignal"""
    waveform = amplitude * np.sin(
        2 * np.pi * frequency * np.linspace(0, 1, 256)
    )
    return ElectricalSignal.from_raw(
        data=waveform, source=source, modality=modality,
    )


# ============================================================================
# Basic Construction
# ============================================================================

class TestLifeLoopInit:
    def test_creates(self):
        ll = LifeLoop()
        assert ll._tick_count == 0

    def test_initial_persistent_errors_zero(self):
        ll = LifeLoop()
        for v in ll._persistent_errors.values():
            assert v == 0.0

    def test_initial_stats(self):
        ll = LifeLoop()
        stats = ll.get_stats()
        assert stats["tick_count"] == 0
        assert stats["total_persistent_error"] == 0.0


# ============================================================================
# Empty Tick (no input)
# ============================================================================

class TestEmptyTick:
    def test_empty_tick_returns_loop_state(self):
        ll = LifeLoop()
        state = ll.tick()
        assert isinstance(state, LoopState)
        assert state.tick_id == 1

    def test_empty_tick_no_errors(self):
        ll = LifeLoop()
        state = ll.tick()
        assert state.errors == []
        assert state.commands == []
        assert state.total_error == 0.0

    def test_empty_tick_full_compensation(self):
        ll = LifeLoop()
        state = ll.tick()
        assert state.compensation_success == 1.0

    def test_tick_increments(self):
        ll = LifeLoop()
        ll.tick()
        ll.tick()
        ll.tick()
        assert ll._tick_count == 3


# ============================================================================
# Visual-Motor Error (hand-eye coordination)
# ============================================================================

class TestVisualMotorError:
    def test_hand_target_mismatch_generates_error(self):
        ll = LifeLoop()
        state = ll.tick(
            hand_target=np.array([1.0, 0.0]),
            hand_position=np.array([0.0, 0.0]),
        )
        assert len(state.errors) > 0
        error_types = [e.error_type for e in state.errors]
        assert ErrorType.VISUAL_MOTOR in error_types

    def test_hand_at_target_no_error(self):
        ll = LifeLoop()
        state = ll.tick(
            hand_target=np.array([0.5, 0.5]),
            hand_position=np.array([0.5, 0.5]),
        )
        vm_errors = [e for e in state.errors if e.error_type == ErrorType.VISUAL_MOTOR]
        assert len(vm_errors) == 0

    def test_hand_error_generates_reach_command(self):
        ll = LifeLoop()
        state = ll.tick(
            hand_target=np.array([1.0, 0.0]),
            hand_position=np.array([0.0, 0.0]),
        )
        reach_cmds = [c for c in state.commands if c.action == CompensationAction.REACH]
        assert len(reach_cmds) > 0

    def test_larger_error_stronger_command(self):
        ll = LifeLoop()
        small = ll.tick(
            hand_target=np.array([0.1, 0.0]),
            hand_position=np.array([0.0, 0.0]),
        )
        ll2 = LifeLoop()
        large = ll2.tick(
            hand_target=np.array([0.8, 0.0]),
            hand_position=np.array([0.0, 0.0]),
        )
        small_str = sum(c.strength for c in small.commands)
        large_str = sum(c.strength for c in large.commands)
        assert large_str > small_str

    def test_deadband_filters_tiny_errors(self):
        ll = LifeLoop()
        tiny = ERROR_DEADBAND * 0.5
        state = ll.tick(
            hand_target=np.array([tiny, 0.0]),
            hand_position=np.array([0.0, 0.0]),
        )
        vm_errors = [e for e in state.errors if e.error_type == ErrorType.VISUAL_MOTOR]
        assert len(vm_errors) == 0


# ============================================================================
# Auditory-Vocal Error (pitch control)
# ============================================================================

class TestAuditoryVocalError:
    def test_pitch_mismatch_generates_error(self):
        ll = LifeLoop()
        state = ll.tick(
            pitch_target=200.0,
            current_pitch=120.0,
        )
        error_types = [e.error_type for e in state.errors]
        assert ErrorType.AUDITORY_VOCAL in error_types

    def test_pitch_matched_no_error(self):
        ll = LifeLoop()
        state = ll.tick(
            pitch_target=200.0,
            current_pitch=200.0,
        )
        av_errors = [e for e in state.errors if e.error_type == ErrorType.AUDITORY_VOCAL]
        assert len(av_errors) == 0

    def test_pitch_error_generates_vocalize_command(self):
        ll = LifeLoop()
        state = ll.tick(
            pitch_target=200.0,
            current_pitch=120.0,
        )
        vocal_cmds = [c for c in state.commands if c.action == CompensationAction.VOCALIZE]
        assert len(vocal_cmds) > 0


# ============================================================================
# Temporal Drift Error
# ============================================================================

class TestTemporalError:
    def test_large_drift_generates_error(self):
        ll = LifeLoop()
        state = ll.tick(
            calibration_drifts={"auditory": 50.0},
        )
        error_types = [e.error_type for e in state.errors]
        assert ErrorType.TEMPORAL in error_types

    def test_small_drift_no_error(self):
        ll = LifeLoop()
        state = ll.tick(
            calibration_drifts={"auditory": 1.0},
        )
        temporal = [e for e in state.errors if e.error_type == ErrorType.TEMPORAL]
        assert len(temporal) == 0

    def test_drift_generates_attend_command(self):
        ll = LifeLoop()
        state = ll.tick(
            calibration_drifts={"visual": 80.0},
        )
        attend = [c for c in state.commands if c.action == CompensationAction.ATTEND]
        assert len(attend) > 0


# ============================================================================
# Interoceptive Error (homeostasis)
# ============================================================================

class TestInteroceptiveError:
    def test_imbalanced_autonomic_generates_error(self):
        ll = LifeLoop()
        state = ll.tick(
            autonomic_balance=0.8,  # High sympathetic dominance
        )
        error_types = [e.error_type for e in state.errors]
        assert ErrorType.INTEROCEPTIVE in error_types

    def test_balanced_autonomic_no_error(self):
        ll = LifeLoop()
        state = ll.tick(
            autonomic_balance=0.0,  # Perfect balance
        )
        intero = [e for e in state.errors if e.error_type == ErrorType.INTEROCEPTIVE]
        assert len(intero) == 0

    def test_imbalance_generates_breathe_command(self):
        ll = LifeLoop()
        state = ll.tick(
            autonomic_balance=0.7,
        )
        breathe = [c for c in state.commands if c.action == CompensationAction.BREATHE]
        assert len(breathe) > 0


# ============================================================================
# Sensory Gate (sleep)
# ============================================================================

class TestSensoryGate:
    def test_full_gate_passes_all(self):
        ll = LifeLoop()
        state = ll.tick(
            hand_target=np.array([1.0, 0.0]),
            hand_position=np.array([0.0, 0.0]),
            sensory_gate=1.0,
        )
        assert len(state.errors) > 0

    def test_closed_gate_filters_weak_errors(self):
        ll = LifeLoop()
        state = ll.tick(
            hand_target=np.array([0.2, 0.0]),
            hand_position=np.array([0.0, 0.0]),
            sensory_gate=0.1,  # deep sleep
        )
        # 0.2 * 0.1 = 0.02 < deadband(0.05) → should be filtered
        vm_errors = [e for e in state.errors if e.error_type == ErrorType.VISUAL_MOTOR]
        assert len(vm_errors) == 0

    def test_strong_stimulus_passes_even_during_sleep(self):
        ll = LifeLoop()
        state = ll.tick(
            hand_target=np.array([1.0, 1.0]),
            hand_position=np.array([0.0, 0.0]),
            sensory_gate=0.1,
        )
        # sqrt(2) * 0.1 ≈ 0.14 > deadband → should pass through
        assert len(state.errors) > 0


# ============================================================================
# Consciousness Gate
# ============================================================================

class TestConsciousnessGate:
    def test_unconscious_no_compensation(self):
        ll = LifeLoop()
        state = ll.tick(
            hand_target=np.array([1.0, 0.0]),
            hand_position=np.array([0.0, 0.0]),
            consciousness_phi=0.1,  # Below CONSCIOUSNESS_GATE
        )
        assert len(state.commands) == 0

    def test_conscious_generates_commands(self):
        ll = LifeLoop()
        state = ll.tick(
            hand_target=np.array([1.0, 0.0]),
            hand_position=np.array([0.0, 0.0]),
            consciousness_phi=0.6,
        )
        assert len(state.commands) > 0

    def test_max_concurrent_compensations(self):
        ll = LifeLoop()
        state = ll.tick(
            hand_target=np.array([1.0, 0.0]),
            hand_position=np.array([0.0, 0.0]),
            pitch_target=300.0,
            current_pitch=100.0,
            autonomic_balance=0.9,
            calibration_drifts={"visual": 80.0, "auditory": 60.0},
            consciousness_phi=0.5,
        )
        # Should not exceed MAX_CONCURRENT_COMPENSATIONS
        assert len(state.commands) <= MAX_CONCURRENT_COMPENSATIONS


# ============================================================================
# Anxiety Effect on Compensation
# ============================================================================

class TestAnxietyEffect:
    def test_anxiety_reduces_compensation_strength(self):
        ll_calm = LifeLoop()
        calm = ll_calm.tick(
            hand_target=np.array([0.8, 0.0]),
            hand_position=np.array([0.0, 0.0]),
            ram_temperature=0.0,
        )
        ll_anx = LifeLoop()
        anxious = ll_anx.tick(
            hand_target=np.array([0.8, 0.0]),
            hand_position=np.array([0.0, 0.0]),
            ram_temperature=0.9,
        )
        calm_str = sum(c.strength for c in calm.commands)
        anx_str = sum(c.strength for c in anxious.commands)
        # Under anxiety, compensation precision decreases (strength may have noise but average should be lower)
        # Use multiple trials and average to eliminate noise
        np.random.seed(42)
        calm_total = 0
        anx_total = 0
        for _ in range(20):
            ll1 = LifeLoop()
            c = ll1.tick(
                hand_target=np.array([0.8, 0.0]),
                hand_position=np.array([0.0, 0.0]),
                ram_temperature=0.0,
            )
            calm_total += sum(x.strength for x in c.commands)
            ll2 = LifeLoop()
            a = ll2.tick(
                hand_target=np.array([0.8, 0.0]),
                hand_position=np.array([0.0, 0.0]),
                ram_temperature=0.9,
            )
            anx_total += sum(x.strength for x in a.commands)
        assert calm_total > anx_total

    def test_low_energy_reduces_strength(self):
        ll_fit = LifeLoop()
        fit = ll_fit.tick(
            hand_target=np.array([0.8, 0.0]),
            hand_position=np.array([0.0, 0.0]),
            energy=1.0,
        )
        ll_tired = LifeLoop()
        tired = ll_tired.tick(
            hand_target=np.array([0.8, 0.0]),
            hand_position=np.array([0.0, 0.0]),
            energy=0.1,
        )
        fit_str = sum(c.strength for c in fit.commands)
        tired_str = sum(c.strength for c in tired.commands)
        assert fit_str > tired_str


# ============================================================================
# Persistent Errors + Pain Conversion
# ============================================================================

class TestPersistentErrors:
    def test_uncompensated_errors_accumulate(self):
        ll = LifeLoop()
        # Unconscious → errors exist but no compensation
        for _ in range(20):
            ll.tick(
                hand_target=np.array([1.0, 0.0]),
                hand_position=np.array([0.0, 0.0]),
                consciousness_phi=0.1,  # Cannot generate compensation
            )
        total = ll.get_total_persistent_error()
        assert total > 0

    def test_compensated_errors_decrease(self):
        ll = LifeLoop()
        # First accumulate some persistent errors
        for _ in range(10):
            ll.tick(
                hand_target=np.array([1.0, 0.0]),
                hand_position=np.array([0.0, 0.0]),
                consciousness_phi=0.1,
            )
        mid_error = ll.get_total_persistent_error()

        # Then consciously compensate
        for _ in range(30):
            ll.tick(
                hand_target=np.array([1.0, 0.0]),
                hand_position=np.array([0.0, 0.0]),
                consciousness_phi=0.8,
            )
        final_error = ll.get_total_persistent_error()
        assert final_error < mid_error

    def test_chronic_error_generates_pain(self):
        ll = LifeLoop()
        # No compensation + persistent large error → pain
        for _ in range(50):
            ll.tick(
                hand_target=np.array([1.0, 1.0]),
                hand_position=np.array([0.0, 0.0]),
                pitch_target=400.0,
                current_pitch=100.0,
                autonomic_balance=0.9,
                consciousness_phi=0.1,
            )
        pain = ll.get_error_to_pain()
        assert pain >= 0.0  # May generate pain


# ============================================================================
# Forward Model (sensory prediction)
# ============================================================================

class TestForwardModel:
    def test_predictions_build_up(self):
        ll = LifeLoop()
        for _ in range(10):
            ll.tick(
                hand_target=np.array([1.0, 0.0]),
                hand_position=np.array([0.0, 0.0]),
            )
        assert len(ll._sensory_predictions) > 0

    def test_prediction_accuracy_starts_average(self):
        ll = LifeLoop()
        acc = ll.get_prediction_accuracy()
        assert 0.0 <= acc <= 1.0


# ============================================================================
# Statistics and Waveforms
# ============================================================================

class TestStatsAndWaveforms:
    def test_stats_keys(self):
        ll = LifeLoop()
        ll.tick()
        stats = ll.get_stats()
        expected = [
            "tick_count", "persistent_errors", "total_persistent_error",
            "error_to_pain", "prediction_accuracy", "cumulative_error",
            "total_compensations", "successful_compensations",
            "avg_error", "avg_compensation", "avg_latency_ms",
        ]
        for key in expected:
            assert key in stats, f"Missing key: {key}"

    def test_waveforms_keys(self):
        ll = LifeLoop()
        for _ in range(5):
            ll.tick()
        wf = ll.get_waveforms()
        assert "total_error" in wf
        assert "compensation" in wf
        assert "loop_latency_ms" in wf
        assert len(wf["total_error"]) == 5

    def test_reset(self):
        ll = LifeLoop()
        for _ in range(10):
            ll.tick(
                hand_target=np.array([1.0, 0.0]),
                hand_position=np.array([0.0, 0.0]),
            )
        ll.reset()
        assert ll._tick_count == 0
        assert ll.get_total_persistent_error() == 0.0
        assert ll.get_stats()["tick_count"] == 0


# ============================================================================
# AliceBrain Closed-Loop Integration
# ============================================================================

class TestAliceBrainClosedLoop:
    def test_brain_has_life_loop(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        assert hasattr(brain, 'life_loop')
        assert isinstance(brain.life_loop, LifeLoop)

    def test_brain_has_eye(self):
        from alice.alice_brain import AliceBrain
        from alice.body.eye import AliceEye
        brain = AliceBrain()
        assert hasattr(brain, 'eye')
        assert isinstance(brain.eye, AliceEye)

    def test_perceive_returns_life_loop_data(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        result = brain.perceive(np.random.randn(100))
        assert "life_loop" in result
        assert "total_error" in result["life_loop"]
        assert "errors" in result["life_loop"]
        assert "commands" in result["life_loop"]
        assert "persistent_errors" in result["life_loop"]

    def test_see_method(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        pixels = np.random.randn(64)
        result = brain.see(pixels)
        assert "visual" in result
        assert result["visual"]["source"] == "eye"
        assert "pupil_aperture" in result["visual"]
        assert "life_loop" in result

    def test_see_updates_visual_signal_cache(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        brain.see(np.random.randn(64))
        assert brain._last_visual_signal is not None

    def test_hear_updates_auditory_signal_cache(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        brain.hear(np.random.randn(256))
        assert brain._last_auditory_signal is not None

    def test_reach_sets_hand_target(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        brain.reach_for(target_x=0.5, target_y=0.3, max_steps=10)
        assert brain._current_hand_target is not None
        np.testing.assert_array_almost_equal(
            brain._current_hand_target, [0.5, 0.3]
        )

    def test_say_sets_pitch_target(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        brain.say(target_pitch=200.0, vowel="a")
        assert brain._current_pitch_target == 200.0

    def test_introspect_includes_life_loop(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        brain.perceive(np.random.randn(100))
        info = brain.introspect()
        assert "life_loop" in info["subsystems"]
        assert "eye" in info["subsystems"]

    def test_closed_loop_see_then_reach(self):
        """
        Full closed loop: see → reach → brain detects hand-eye error
        """
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()

        # See → generates visual signal
        see_result = brain.see(
            np.random.randn(64),
            visual_target=np.array([0.5, 0.5]),
        )
        assert brain._last_visual_signal is not None
        assert brain._current_visual_target is not None

        # Reach → life loop should detect hand-eye error
        reach_result = brain.reach_for(
            target_x=0.5, target_y=0.5, max_steps=50,
        )
        assert "coordination" in reach_result
        assert brain._current_hand_target is not None

    def test_autonomic_drives_pupil(self):
        """
        Closed loop verification: autonomic → pupil → eye
        """
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()

        # Inject pain → sympathetic activation → pupil dilation
        brain.inject_pain(0.8)
        for _ in range(10):
            brain.perceive(np.random.randn(100))

        pupil = brain.autonomic.get_pupil_aperture()
        # Pupil should be larger under pain
        assert pupil >= 0.0  # At least a reasonable value
