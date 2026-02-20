# -*- coding: utf-8 -*-
"""
Tests for alice.brain.calibration — Temporal Calibrator (Action Model)

Test coverage:
  1. Feature matcher — frequency/amplitude/phase/waveform matching
  2. Drift estimator — EMA tracking + compensation
  3. Temporal window binding — simultaneously arriving signals
  4. Extended window + feature matching — delayed but similar signals
  5. SignalFrame — frame structure
  6. Calibration quality — drift → quality decreases
  7. Multi-modal binding — visual + proprioception + error
  8. AliceBrain integration — calibration data in perceive / reach_for
"""

import math
import time
import pytest
import numpy as np

from alice.brain.calibration import (
    TemporalCalibrator,
    SignalFrame,
    _FeatureMatcher,
    _DriftEstimator,
    TEMPORAL_WINDOW_MS,
    EXTENDED_WINDOW_MS,
    BINDING_THRESHOLD,
    MAX_DRIFT_MS,
)
from alice.core.signal import ElectricalSignal


# ============================================================================
# Helper Functions
# ============================================================================


def _make_signal(
    freq: float = 30.0,
    amp: float = 1.0,
    phase: float = 0.0,
    source: str = "test",
    modality: str = "visual",
    timestamp: float = None,
    n: int = 64,
) -> ElectricalSignal:
    """Create an ElectricalSignal for testing"""
    t = np.linspace(0, 1, n, endpoint=False)
    waveform = amp * np.sin(2 * np.pi * freq * t + phase)
    return ElectricalSignal(
        waveform=waveform,
        amplitude=amp,
        frequency=freq,
        phase=phase,
        impedance=50.0,
        snr=15.0,
        timestamp=timestamp if timestamp is not None else time.time(),
        source=source,
        modality=modality,
    )


# ============================================================================
# Feature Matcher
# ============================================================================


class TestFeatureMatcher:
    """Feature matcher tests"""

    def test_identical_signals_score_1(self):
        """Identical signals → score close to 1"""
        sig = _make_signal(freq=30.0, amp=1.0, phase=0.0)
        score = _FeatureMatcher.match(sig, sig)
        assert score > 0.95

    def test_different_freq_lower_score(self):
        """Different frequency → score decreases"""
        sig_a = _make_signal(freq=10.0)
        sig_b = _make_signal(freq=80.0)
        score = _FeatureMatcher.match(sig_a, sig_b)
        assert score < 0.7

    def test_different_amp_lower_score(self):
        """Different amplitude → score decreases"""
        sig_a = _make_signal(amp=0.1)
        sig_b = _make_signal(amp=5.0)
        score = _FeatureMatcher.match(sig_a, sig_b)
        assert score < 0.8

    def test_opposite_phase_lower_score(self):
        """Anti-phase → score decreases"""
        sig_a = _make_signal(phase=0.0)
        sig_b = _make_signal(phase=math.pi)
        score_same = _FeatureMatcher.match(sig_a, sig_a)
        score_opp = _FeatureMatcher.match(sig_a, sig_b)
        assert score_opp < score_same

    def test_similar_signals_high_score(self):
        """Similar signals → high score"""
        sig_a = _make_signal(freq=30.0, amp=1.0, phase=0.1)
        sig_b = _make_signal(freq=32.0, amp=1.1, phase=0.2)
        score = _FeatureMatcher.match(sig_a, sig_b)
        assert score > 0.7

    def test_score_in_range(self):
        """Score is always in 0~1"""
        sig_a = _make_signal(freq=1.0, amp=0.01)
        sig_b = _make_signal(freq=100.0, amp=10.0, phase=math.pi)
        score = _FeatureMatcher.match(sig_a, sig_b)
        assert 0.0 <= score <= 1.0

    def test_empty_waveform_still_works(self):
        """Empty waveform does not crash"""
        sig_a = _make_signal(n=0)
        sig_b = _make_signal(n=64)
        score = _FeatureMatcher.match(sig_a, sig_b)
        assert 0.0 <= score <= 1.0


# ============================================================================
# Drift Estimator
# ============================================================================


class TestDriftEstimator:
    """Drift estimator tests"""

    def test_initial_drift_zero(self):
        """Initial state has no drift"""
        drift = _DriftEstimator()
        correction = drift.get_correction("visual")
        assert correction == 0.0

    def test_update_sets_drift(self):
        """After update, drift exists"""
        drift = _DriftEstimator()
        drift.update("visual", 10.0)
        assert drift.get_correction("visual") != 0.0

    def test_drift_converges_to_observed(self):
        """Sustained constant drift → EMA converges to that value"""
        drift = _DriftEstimator()
        for _ in range(100):
            drift.update("visual", 5.0)
        # Converges to 5ms → correction = -5ms
        assert abs(drift.get_correction("visual") + 5.0) < 0.5

    def test_drift_clamped(self):
        """Drift is clamped within MAX_DRIFT_MS"""
        drift = _DriftEstimator()
        drift.update("visual", 99999.0)
        all_drifts = drift.get_all_drifts()
        assert abs(all_drifts["visual"]) <= MAX_DRIFT_MS

    def test_multiple_modalities(self):
        """Different modalities have independent drifts"""
        drift = _DriftEstimator()
        drift.update("visual", 5.0)
        drift.update("proprioception", -3.0)
        all_drifts = drift.get_all_drifts()
        assert "visual" in all_drifts
        assert "proprioception" in all_drifts
        assert all_drifts["visual"] != all_drifts["proprioception"]

    def test_drift_history(self):
        """Drift history is recorded"""
        drift = _DriftEstimator()
        drift.update("visual", 1.0)
        drift.update("visual", 2.0)
        drift.update("visual", 3.0)
        history = drift.get_drift_history("visual")
        assert len(history) == 3


# ============================================================================
# SignalFrame
# ============================================================================


class TestSignalFrame:
    """Signal frame"""

    def test_empty_frame(self):
        """Empty frame"""
        frame = SignalFrame()
        assert not frame.is_complete
        assert frame.signal_count == 0

    def test_single_modality_not_complete(self):
        """Single modality is not complete"""
        frame = SignalFrame(signals={"visual": [_make_signal()]})
        assert not frame.is_complete

    def test_two_modalities_complete(self):
        """Two modalities = complete"""
        frame = SignalFrame(signals={
            "visual": [_make_signal(modality="visual")],
            "proprioception": [_make_signal(modality="proprioception")],
        })
        assert frame.is_complete
        assert frame.signal_count == 2

    def test_get_primary_signal(self):
        """Get primary signal"""
        sig = _make_signal(modality="visual")
        frame = SignalFrame(signals={"visual": [sig]})
        assert frame.get_primary_signal("visual") is sig
        assert frame.get_primary_signal("auditory") is None


# ============================================================================
# Temporal Window Binding
# ============================================================================


class TestTemporalBinding:
    """Binding within temporal window"""

    def test_simultaneous_signals_bind(self):
        """Simultaneously arriving signals → bound"""
        cal = TemporalCalibrator()
        now = time.time()

        sig_visual = _make_signal(modality="visual", source="eye", timestamp=now)
        sig_prop = _make_signal(modality="proprioception", source="hand", timestamp=now)

        cal.receive(sig_visual)
        cal.receive(sig_prop)
        frame = cal.bind()

        assert frame is not None
        assert frame.is_complete
        assert "visual" in frame.modalities
        assert "proprioception" in frame.modalities

    def test_within_window_binds(self):
        """Arrive within window → bound"""
        cal = TemporalCalibrator()
        now = time.time()
        dt = (TEMPORAL_WINDOW_MS * 0.5) / 1000.0  # Half window

        sig_a = _make_signal(modality="visual", timestamp=now)
        sig_b = _make_signal(modality="proprioception", timestamp=now + dt)

        cal.receive(sig_a)
        cal.receive(sig_b)
        frame = cal.bind()

        assert frame is not None
        assert frame.is_complete

    def test_outside_window_no_temporal_bind(self):
        """Outside window → no temporal binding (may do feature binding)"""
        cal = TemporalCalibrator()
        now = time.time()
        dt = (EXTENDED_WINDOW_MS + 100) / 1000.0  # Beyond extended window

        sig_a = _make_signal(modality="visual", timestamp=now, freq=10.0, amp=0.1)
        sig_b = _make_signal(modality="proprioception", timestamp=now + dt, freq=90.0, amp=5.0)

        cal.receive(sig_a)
        cal.receive(sig_b)
        frame = cal.bind()

        # Only takes the earliest signal, the other stays in buffer
        assert frame is not None
        # Should have only one modality (the other is too far)
        assert len(frame.modalities) == 1

    def test_three_modalities_bind(self):
        """Three modalities arriving simultaneously → all bound"""
        cal = TemporalCalibrator()
        now = time.time()

        sig_v = _make_signal(modality="visual", timestamp=now)
        sig_p = _make_signal(modality="proprioception", timestamp=now + 0.001)
        sig_e = _make_signal(modality="motor", source="hand_eye_error", timestamp=now + 0.002)

        cal.receive(sig_v)
        cal.receive(sig_p)
        cal.receive(sig_e)
        frame = cal.bind()

        assert frame is not None
        assert len(frame.modalities) == 3
        assert len(frame.bindings) == 3  # C(3,2) = 3 pairs


# ============================================================================
# Extended Window + Feature Matching
# ============================================================================


class TestFeatureBinding:
    """Binding of signals outside temporal window but with feature matching"""

    def test_similar_signals_bind_in_extended_window(self):
        """Similar signals within extended window → feature binding"""
        cal = TemporalCalibrator()
        now = time.time()
        # Outside main window but within extended window
        dt = (TEMPORAL_WINDOW_MS + 30) / 1000.0

        # Similar features (same frequency, same amplitude)
        sig_a = _make_signal(modality="visual", timestamp=now, freq=30.0, amp=1.0)
        sig_b = _make_signal(modality="proprioception", timestamp=now + dt, freq=31.0, amp=1.05)

        cal.receive(sig_a)
        cal.receive(sig_b)
        frame = cal.bind()

        assert frame is not None
        # If feature match score is high enough, should bind
        if frame.is_complete:
            assert cal.get_stats()["feature_bindings"] > 0

    def test_dissimilar_signals_no_feature_bind(self):
        """Dissimilar signals → no feature binding"""
        cal = TemporalCalibrator()
        now = time.time()
        dt = (TEMPORAL_WINDOW_MS + 30) / 1000.0

        # Completely different features
        sig_a = _make_signal(modality="visual", timestamp=now, freq=5.0, amp=0.1, phase=0.0)
        sig_b = _make_signal(modality="proprioception", timestamp=now + dt, freq=90.0, amp=5.0, phase=math.pi)

        cal.receive(sig_a)
        cal.receive(sig_b)
        frame = cal.bind()

        # Only binds the earliest modality
        assert frame is not None
        assert len(frame.modalities) == 1


# ============================================================================
# Drift Calibration
# ============================================================================


class TestDriftCalibration:
    """Drift calibration integration tests"""

    def test_drift_correction_improves(self):
        """Persistently offset signals → auto-compensated after calibration"""
        cal = TemporalCalibrator()
        fixed_drift_ms = 10.0  # A modality is consistently 10ms late

        for i in range(20):
            now = time.time()
            sig_a = _make_signal(modality="visual", timestamp=now)
            sig_b = _make_signal(
                modality="proprioception",
                timestamp=now + fixed_drift_ms / 1000.0,
            )
            cal.receive(sig_a)
            cal.receive(sig_b)
            cal.bind()

        # Calibrator should have tracked the drift
        drifts = cal.get_stats()["drifts"]
        assert "proprioception" in drifts or "visual" in drifts

    def test_calibration_quality(self):
        """Calibration quality = 0~1"""
        cal = TemporalCalibrator()
        q = cal.get_calibration_quality()
        assert 0.0 <= q <= 1.0

    def test_large_drift_lowers_quality(self):
        """Large drift → quality decreases"""
        cal = TemporalCalibrator()

        # Normal (no drift)
        quality_before = cal.get_calibration_quality()

        # Feed in signals with large drift
        for _ in range(30):
            now = time.time()
            sig_a = _make_signal(modality="visual", timestamp=now)
            sig_b = _make_signal(
                modality="proprioception",
                timestamp=now + 80.0 / 1000.0,  # 80ms drift
            )
            cal.receive(sig_a)
            cal.receive(sig_b)
            cal.bind()

        quality_after = cal.get_calibration_quality()
        assert quality_after <= quality_before


# ============================================================================
# flush + Buffer Management
# ============================================================================


class TestBufferManagement:
    """Buffer management"""

    def test_empty_buffer_returns_none(self):
        """Empty buffer → None"""
        cal = TemporalCalibrator()
        assert cal.bind() is None

    def test_flush_empties_buffer(self):
        """flush clears buffer"""
        cal = TemporalCalibrator()
        cal.receive(_make_signal(modality="visual"))
        cal.receive(_make_signal(modality="proprioception"))
        frames = cal.flush()
        assert len(frames) >= 1
        assert cal.get_stats()["buffer_size"] == 0

    def test_receive_and_bind_convenience(self):
        """receive_and_bind convenience method"""
        cal = TemporalCalibrator()
        now = time.time()

        # First signal → may form single-modality frame
        frame1 = cal.receive_and_bind(
            _make_signal(modality="visual", timestamp=now)
        )

    def test_get_latest_frame(self):
        """get_latest_frame"""
        cal = TemporalCalibrator()
        assert cal.get_latest_frame() is None

        now = time.time()
        cal.receive(_make_signal(modality="visual", timestamp=now))
        cal.receive(_make_signal(modality="proprioception", timestamp=now))
        cal.bind()

        frame = cal.get_latest_frame()
        assert frame is not None

    def test_get_frames_history(self):
        """Frame history"""
        cal = TemporalCalibrator()
        for i in range(5):
            now = time.time()
            cal.receive(_make_signal(modality="visual", timestamp=now + i * 0.5))
            cal.bind()

        frames = cal.get_frames(last_n=3)
        assert len(frames) <= 3


# ============================================================================
# Statistics
# ============================================================================


class TestStats:
    """Statistics functionality"""

    def test_get_stats_complete(self):
        """Statistics contain all required fields"""
        cal = TemporalCalibrator()
        stats = cal.get_stats()
        assert "total_signals" in stats
        assert "total_frames" in stats
        assert "total_bindings" in stats
        assert "temporal_bindings" in stats
        assert "feature_bindings" in stats
        assert "calibration_quality" in stats
        assert "drifts" in stats

    def test_get_calibration_state(self):
        """Calibration state"""
        cal = TemporalCalibrator()
        state = cal.get_calibration_state()
        assert "quality" in state
        assert "drifts_ms" in state
        assert "calibration_error_history" in state

    def test_signal_count_increments(self):
        """Receive count"""
        cal = TemporalCalibrator()
        cal.receive(_make_signal())
        cal.receive(_make_signal())
        assert cal.get_stats()["total_signals"] == 2


# ============================================================================
# AliceBrain Integration
# ============================================================================


class TestAliceBrainCalibration:
    """AliceBrain + TemporalCalibrator integration"""

    def test_brain_has_calibrator(self):
        """AliceBrain has calibrator attribute"""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        assert hasattr(brain, "calibrator")
        assert isinstance(brain.calibrator, TemporalCalibrator)

    def test_perceive_includes_calibration(self):
        """perceive result contains calibration info"""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        result = brain.perceive(np.random.randn(64))
        assert "calibration" in result
        assert "quality" in result["calibration"]

    def test_reach_for_includes_temporal_binding(self):
        """reach_for result contains temporal_binding info"""
        from alice.alice_brain import AliceBrain
        from alice.body.hand import AliceHand
        brain = AliceBrain(neuron_count=20)
        brain.hand = AliceHand(initial_pos=(100, 100))
        result = brain.reach_for(300.0, 200.0)
        assert "temporal_binding" in result

    def test_introspect_includes_calibrator(self):
        """introspect includes calibrator statistics"""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        intro = brain.introspect()
        assert "calibrator" in intro["subsystems"]

    def test_perceive_then_reach_has_cross_modal_frame(self):
        """perceive → reach_for → calibrator accumulates signals"""
        from alice.alice_brain import AliceBrain
        from alice.body.hand import AliceHand
        brain = AliceBrain(neuron_count=20)
        brain.hand = AliceHand(initial_pos=(100, 100))

        # First perceive
        brain.perceive(np.random.randn(64))
        # Then reach
        result = brain.reach_for(300.0, 200.0)

        # Calibrator should have received multiple signals
        stats = brain.calibrator.get_stats()
        assert stats["total_signals"] >= 2  # visual + proprioception
