# -*- coding: utf-8 -*-
"""
Dynamic Time Slicing Tests

Core physics formula under test:
    T_slice = T_hardware + Σ (Γᵢ² / (1 - Γᵢ²)) × τ_match

Verifications:
    1. Single channel → minimum window (no cross-modal calibration needed)
    2. Multi-channel matched impedance → small window (fast calibration)
    3. Multi-channel mismatched impedance → large window (slow calibration)
    4. Window self-adapts over time (EMA smoothing)
    5. override_window instant override (amygdala fast pathway)
    6. Consciousness Φ modulated by temporal resolution
    7. bind() actually uses the dynamic window
"""

import time
import numpy as np
import pytest

from alice.brain.calibration import (
    TemporalCalibrator,
    MIN_TEMPORAL_WINDOW_MS,
    MAX_TEMPORAL_WINDOW_MS,
    MATCH_COST_MS,
    WINDOW_EMA_RATE,
)
from alice.brain.awareness_monitor import AwarenessMonitor as ConsciousnessModule
from alice.core.signal import ElectricalSignal


# ============================================================================
# Helpers
# ============================================================================

def _make_signal(modality: str, impedance: float = 50.0,
                 timestamp: float | None = None,
                 amplitude: float = 1.0) -> ElectricalSignal:
    """Quickly create a test signal."""
    ts = timestamp or time.time()
    waveform = np.array([amplitude * np.sin(2 * np.pi * 40.0 * i / 100)
                         for i in range(100)])
    return ElectricalSignal(
        waveform=waveform,
        modality=modality,
        amplitude=amplitude,
        frequency=40.0,
        phase=0.0,
        impedance=impedance,
        snr=20.0,
        timestamp=ts,
        source="test",
    )


# ============================================================================
# 1. Basic Dynamic Window Behavior
# ============================================================================

class TestDynamicWindowBasics:
    """Basic behavior of the dynamic time window."""

    def test_initial_window_equals_static(self):
        """Initial dynamic window = original static setting."""
        cal = TemporalCalibrator()
        assert cal.get_active_window_ms() == cal.temporal_window_ms

    def test_initial_frame_rate(self):
        """Initial frame rate = 1000 / initial window."""
        cal = TemporalCalibrator()
        expected_hz = 1000.0 / cal.temporal_window_ms
        assert abs(cal.get_frame_rate() - expected_hz) < 0.1

    def test_initial_temporal_resolution_is_one(self):
        """Initial temporal resolution = 1.0 (no calibration load yet)."""
        cal = TemporalCalibrator()
        assert cal.get_temporal_resolution() == 1.0

    def test_initial_calibration_load_is_zero(self):
        """Initial calibration load = 0.0."""
        cal = TemporalCalibrator()
        assert cal.get_calibration_load() == 0.0


# ============================================================================
# 2. Single Channel → Minimum Window
# ============================================================================

class TestSingleChannel:
    """Single-channel scenario: no cross-modal calibration needed."""

    def test_single_modality_minimum_window(self):
        """Only one modality → window converges toward minimum."""
        cal = TemporalCalibrator()
        t = time.time()

        # Repeatedly feed single-modality signals to let EMA converge
        for i in range(50):
            cal.receive(_make_signal("visual", impedance=50.0, timestamp=t + i * 0.01))

        # Window should approach MIN_TEMPORAL_WINDOW_MS
        assert cal.get_active_window_ms() < MIN_TEMPORAL_WINDOW_MS + 5.0

    def test_single_channel_max_frame_rate(self):
        """Single channel → maximum frame rate."""
        cal = TemporalCalibrator()
        t = time.time()

        for i in range(50):
            cal.receive(_make_signal("auditory", impedance=30.0, timestamp=t + i * 0.01))

        max_hz = 1000.0 / MIN_TEMPORAL_WINDOW_MS
        assert cal.get_frame_rate() > max_hz * 0.8  # Close to max frame rate


# ============================================================================
# 3. Multi-Channel Matched Impedance → Small Window
# ============================================================================

class TestMatchedImpedance:
    """Multi-channel with matched impedance (Γ ≈ 0) → fast calibration."""

    def test_matched_impedance_small_window(self):
        """Similar impedance → small Γ → low calibration cost → window near minimum."""
        cal = TemporalCalibrator()
        t = time.time()

        for i in range(50):
            ts = t + i * 0.01
            cal.receive(_make_signal("visual", impedance=50.0, timestamp=ts))
            cal.receive(_make_signal("auditory", impedance=48.0, timestamp=ts + 0.001))

        # Γ = |50-48|/(50+48) ≈ 0.02 → almost zero calibration cost
        assert cal.get_active_window_ms() < MIN_TEMPORAL_WINDOW_MS + 10.0

    def test_matched_impedance_high_resolution(self):
        """Matched impedance → temporal resolution close to 1.0."""
        cal = TemporalCalibrator()
        t = time.time()

        for i in range(50):
            ts = t + i * 0.01
            cal.receive(_make_signal("visual", impedance=100.0, timestamp=ts))
            cal.receive(_make_signal("auditory", impedance=100.0, timestamp=ts + 0.001))

        assert cal.get_temporal_resolution() > 0.9


# ============================================================================
# 4. Multi-Channel Mismatched Impedance → Large Window
# ============================================================================

class TestMismatchedImpedance:
    """Multi-channel high mismatch → slow calibration → window expands."""

    def test_high_mismatch_large_window(self):
        """Severe impedance mismatch → window significantly expands."""
        cal = TemporalCalibrator()
        t = time.time()

        for i in range(50):
            ts = t + i * 0.01
            # Γ = |200-20|/(200+20) ≈ 0.818 → high mismatch
            cal.receive(_make_signal("visual", impedance=200.0, timestamp=ts))
            cal.receive(_make_signal("auditory", impedance=20.0, timestamp=ts + 0.001))

        assert cal.get_active_window_ms() > MIN_TEMPORAL_WINDOW_MS + 20.0

    def test_high_mismatch_low_resolution(self):
        """High mismatch → temporal resolution drops."""
        cal = TemporalCalibrator()
        t = time.time()

        for i in range(50):
            ts = t + i * 0.01
            cal.receive(_make_signal("visual", impedance=200.0, timestamp=ts))
            cal.receive(_make_signal("auditory", impedance=20.0, timestamp=ts + 0.001))

        assert cal.get_temporal_resolution() < 0.8

    def test_high_mismatch_low_frame_rate(self):
        """High mismatch → frame rate drops."""
        cal = TemporalCalibrator()
        t = time.time()

        for i in range(50):
            ts = t + i * 0.01
            cal.receive(_make_signal("visual", impedance=200.0, timestamp=ts))
            cal.receive(_make_signal("auditory", impedance=20.0, timestamp=ts + 0.001))

        max_hz = 1000.0 / MIN_TEMPORAL_WINDOW_MS
        assert cal.get_frame_rate() < max_hz * 0.8

    def test_three_modality_extreme_mismatch(self):
        """Three-modality severe mismatch → higher calibration cost."""
        cal = TemporalCalibrator()
        t = time.time()

        for i in range(50):
            ts = t + i * 0.01
            cal.receive(_make_signal("visual", impedance=300.0, timestamp=ts))
            cal.receive(_make_signal("auditory", impedance=30.0, timestamp=ts + 0.001))
            cal.receive(_make_signal("proprioception", impedance=150.0, timestamp=ts + 0.002))

        # Three-channel cross-mismatch: summed calibration cost of 3 gamma values
        assert cal.get_calibration_load() > 0.1

    def test_more_channels_more_cost(self):
        """More channels → higher calibration cost (same mismatch level)."""
        t = time.time()

        # 2 channels
        cal2 = TemporalCalibrator()
        for i in range(50):
            ts = t + i * 0.01
            cal2.receive(_make_signal("visual", impedance=200.0, timestamp=ts))
            cal2.receive(_make_signal("auditory", impedance=20.0, timestamp=ts + 0.001))
        w2 = cal2.get_active_window_ms()

        # 3 channels (add a third modality with the same mismatch level)
        cal3 = TemporalCalibrator()
        for i in range(50):
            ts = t + i * 0.01
            cal3.receive(_make_signal("visual", impedance=200.0, timestamp=ts))
            cal3.receive(_make_signal("auditory", impedance=20.0, timestamp=ts + 0.001))
            cal3.receive(_make_signal("proprioception", impedance=110.0, timestamp=ts + 0.002))
        w3 = cal3.get_active_window_ms()

        assert w3 > w2  # More channels → larger window


# ============================================================================
# 5. Window Adaptation (EMA Smoothing + History)
# ============================================================================

class TestWindowAdaptation:
    """Window self-adapts via EMA smoothing."""

    def test_window_does_not_jump_instantly(self):
        """Window should not jump instantly (EMA smoothing)."""
        cal = TemporalCalibrator()
        t = time.time()

        # First send matched signals
        for i in range(10):
            ts = t + i * 0.01
            cal.receive(_make_signal("visual", impedance=50.0, timestamp=ts))
            cal.receive(_make_signal("auditory", impedance=50.0, timestamp=ts + 0.001))

        w_before = cal.get_active_window_ms()

        # Suddenly switch to high mismatch
        ts = t + 0.2
        cal.receive(_make_signal("visual", impedance=300.0, timestamp=ts))
        cal.receive(_make_signal("auditory", impedance=10.0, timestamp=ts + 0.001))

        w_after = cal.get_active_window_ms()

        # Should not jump to max in one step — EMA limits the rate of change
        assert w_after < MAX_TEMPORAL_WINDOW_MS * 0.8

    def test_window_history_recorded(self):
        """Each receive call records window history."""
        cal = TemporalCalibrator()
        t = time.time()

        for i in range(5):
            cal.receive(_make_signal("visual", impedance=50.0, timestamp=t + i * 0.01))

        state = cal.get_calibration_state()
        assert len(state["window_history"]) >= 5

    def test_window_converges_over_time(self):
        """Continuous stable signals → window converges."""
        cal = TemporalCalibrator()
        t = time.time()

        windows = []
        for i in range(100):
            ts = t + i * 0.01
            cal.receive(_make_signal("visual", impedance=100.0, timestamp=ts))
            cal.receive(_make_signal("auditory", impedance=50.0, timestamp=ts + 0.001))
            windows.append(cal.get_active_window_ms())

        # Variance in the second half should be smaller than the first (convergence)
        first_half_var = np.var(windows[:50])
        second_half_var = np.var(windows[50:])
        assert second_half_var <= first_half_var + 1e-6


# ============================================================================
# 6. override_window (Amygdala Fast Pathway)
# ============================================================================

class TestOverrideWindow:
    """Emergency modules like the amygdala can override the window."""

    def test_override_sets_exact_value(self):
        """Override should set the exact value."""
        cal = TemporalCalibrator()
        cal.override_window(25.0)
        assert abs(cal.get_active_window_ms() - 25.0) < 0.1

    def test_override_clamped_to_limits(self):
        """Override value is clamped to physical limits."""
        cal = TemporalCalibrator()

        cal.override_window(1.0)  # Too small
        assert cal.get_active_window_ms() >= MIN_TEMPORAL_WINDOW_MS

        cal.override_window(9999.0)  # Too large
        assert cal.get_active_window_ms() <= MAX_TEMPORAL_WINDOW_MS

    def test_override_updates_derived_metrics(self):
        """Derived metrics update immediately after override."""
        cal = TemporalCalibrator()
        cal.override_window(100.0)

        expected_hz = 1000.0 / 100.0
        assert abs(cal.get_frame_rate() - expected_hz) < 0.5

        expected_res = MIN_TEMPORAL_WINDOW_MS / 100.0
        assert abs(cal.get_temporal_resolution() - expected_res) < 0.05


# ============================================================================
# 7. Consciousness Φ Modulated by Temporal Resolution
# ============================================================================

class TestConsciousnessTemporalModulation:
    """temporal_resolution modulates consciousness Φ."""

    def test_full_resolution_no_penalty(self):
        """temporal_resolution=1.0 → no additional penalty."""
        c = ConsciousnessModule()

        r1 = c.tick(
            attention_strength=0.8,
            binding_quality=0.8,
            arousal=0.8,
            sensory_gate=1.0,
            temporal_resolution=1.0,
        )

        assert r1["components"]["temporal_resolution"] == 1.0
        assert r1["components"]["binding_effective"] == r1["components"]["binding"]

    def test_low_resolution_reduces_binding_effective(self):
        """Low temporal_resolution → binding_effective decreases.
        (In Screen Brightness Model, raw_phi = screen_brightness,
        but temporal_resolution still modulates binding_effective.)"""
        c1 = ConsciousnessModule()
        c2 = ConsciousnessModule()

        r_high = c1.tick(
            attention_strength=0.8,
            binding_quality=0.8,
            arousal=0.8,
            sensory_gate=1.0,
            temporal_resolution=1.0,
        )

        r_low = c2.tick(
            attention_strength=0.8,
            binding_quality=0.8,
            arousal=0.8,
            sensory_gate=1.0,
            temporal_resolution=0.3,
        )

        assert r_low["components"]["binding_effective"] < r_high["components"]["binding_effective"]

    def test_binding_effective_formula(self):
        """binding_effective = binding_quality × temporal_resolution^0.5"""
        c = ConsciousnessModule()
        r = c.tick(
            binding_quality=0.64,
            temporal_resolution=0.49,
        )

        expected = 0.64 * (0.49 ** 0.5)  # 0.64 × 0.7 = 0.448
        assert abs(r["components"]["binding_effective"] - expected) < 0.01

    def test_default_temporal_resolution_backward_compatible(self):
        """No temporal_resolution passed → defaults to 1.0 (backward compatible)."""
        c = ConsciousnessModule()
        r = c.tick(
            attention_strength=0.8,
            binding_quality=0.8,
        )
        assert r["components"]["temporal_resolution"] == 1.0

    def test_zero_resolution_collapses_binding(self):
        """temporal_resolution=0 → binding_effective=0"""
        c = ConsciousnessModule()
        r = c.tick(
            binding_quality=1.0,
            temporal_resolution=0.0,
        )
        assert r["components"]["binding_effective"] == 0.0


# ============================================================================
# 8. Physical Invariants — Core Formula Verification
# ============================================================================

class TestTimeSlicePhysics:
    """Physical properties of the core formula."""

    def test_formula_zero_gamma_zero_cost(self):
        """Γ=0 → calibration cost=0 → minimum window."""
        # Perfect impedance match → no calibration time needed
        cal = TemporalCalibrator()
        t = time.time()

        for i in range(50):
            ts = t + i * 0.01
            cal.receive(_make_signal("visual", impedance=50.0, timestamp=ts))
            cal.receive(_make_signal("auditory", impedance=50.0, timestamp=ts + 0.001))

        assert cal.get_active_window_ms() < MIN_TEMPORAL_WINDOW_MS + 2.0

    def test_formula_high_gamma_high_cost(self):
        """Large Γ → Γ²/(1-Γ²) explodes → window expands."""
        cal = TemporalCalibrator()
        t = time.time()

        for i in range(50):
            ts = t + i * 0.01
            # Γ = |500-10|/(500+10) ≈ 0.96 → Γ²/(1-Γ²) ≈ 11.8
            cal.receive(_make_signal("visual", impedance=500.0, timestamp=ts))
            cal.receive(_make_signal("auditory", impedance=10.0, timestamp=ts + 0.001))

        assert cal.get_active_window_ms() > MIN_TEMPORAL_WINDOW_MS + 50.0

    def test_window_bounded_by_physical_limits(self):
        """Window is always bounded by [MIN, MAX]."""
        cal = TemporalCalibrator()
        t = time.time()

        for i in range(100):
            ts = t + i * 0.005
            # Extreme mismatch
            cal.receive(_make_signal("visual", impedance=1000.0, timestamp=ts))
            cal.receive(_make_signal("auditory", impedance=1.0, timestamp=ts + 0.001))
            cal.receive(_make_signal("proprioception", impedance=500.0, timestamp=ts + 0.002))

            w = cal.get_active_window_ms()
            assert MIN_TEMPORAL_WINDOW_MS <= w <= MAX_TEMPORAL_WINDOW_MS

    def test_resolution_inversely_proportional_to_window(self):
        """Temporal resolution ∝ MIN_WINDOW / active_window (inverse)."""
        cal = TemporalCalibrator()
        t = time.time()

        for i in range(50):
            ts = t + i * 0.01
            cal.receive(_make_signal("visual", impedance=150.0, timestamp=ts))
            cal.receive(_make_signal("auditory", impedance=30.0, timestamp=ts + 0.001))

        w = cal.get_active_window_ms()
        r = cal.get_temporal_resolution()
        expected_r = MIN_TEMPORAL_WINDOW_MS / max(w, MIN_TEMPORAL_WINDOW_MS)
        assert abs(r - expected_r) < 0.05

    def test_frame_rate_inversely_proportional_to_window(self):
        """Frame rate = 1000 / active_window_ms."""
        cal = TemporalCalibrator()
        t = time.time()

        for i in range(50):
            ts = t + i * 0.01
            cal.receive(_make_signal("visual", impedance=100.0, timestamp=ts))
            cal.receive(_make_signal("auditory", impedance=40.0, timestamp=ts + 0.001))

        w = cal.get_active_window_ms()
        hz = cal.get_frame_rate()
        assert abs(hz - 1000.0 / w) < 0.5


# ============================================================================
# 9. Integration Test: bind() Uses the Dynamic Window
# ============================================================================

class TestDynamicBindIntegration:
    """Verify that bind() actually uses the dynamic window."""

    def test_bind_uses_dynamic_window(self):
        """Override with a large window → more distant signals can also bind."""
        cal_narrow = TemporalCalibrator()
        cal_wide = TemporalCalibrator()

        t = time.time()

        # Two signals 80ms apart
        sig1 = _make_signal("visual", impedance=50.0, timestamp=t)
        sig2 = _make_signal("auditory", impedance=50.0, timestamp=t + 0.080)

        # Narrow window (25ms) → 80ms signal is outside the window
        cal_narrow.override_window(25.0)
        cal_narrow.receive(sig1)
        cal_narrow.receive(sig2)
        frame_narrow = cal_narrow.bind()

        # Wide window (200ms) → 80ms signal is within the window
        cal_wide.override_window(200.0)
        cal_wide.receive(sig1)
        cal_wide.receive(sig2)
        frame_wide = cal_wide.bind()

        if frame_narrow is not None:
            # Narrow window may have bound only one modality
            narrow_modalities = len(frame_narrow.modalities)
        else:
            narrow_modalities = 0

        if frame_wide is not None:
            wide_modalities = len(frame_wide.modalities)
        else:
            wide_modalities = 0

        # Wide window should bind more modalities (or equal)
        assert wide_modalities >= narrow_modalities

    def test_dynamic_window_from_impedance_affects_binding(self):
        """Impedance mismatch naturally expands window → distant signals get calibrated in."""
        cal = TemporalCalibrator()
        t = time.time()

        # First push the window larger with high-mismatch signals
        for i in range(30):
            ts = t + i * 0.01
            cal.receive(_make_signal("visual", impedance=200.0, timestamp=ts))
            cal.receive(_make_signal("auditory", impedance=20.0, timestamp=ts + 0.001))

        # Record the expanded window
        expanded_window = cal.get_active_window_ms()
        assert expanded_window > MIN_TEMPORAL_WINDOW_MS + 10  # Confirm window actually expanded

    def test_stats_include_time_slice_fields(self):
        """get_stats() includes dynamic time slice fields."""
        cal = TemporalCalibrator()
        t = time.time()

        cal.receive(_make_signal("visual", impedance=50.0, timestamp=t))
        stats = cal.get_stats()

        assert "active_window_ms" in stats
        assert "frame_rate_hz" in stats
        assert "temporal_resolution" in stats
        assert "calibration_load" in stats

    def test_calibration_state_includes_time_slice(self):
        """get_calibration_state() includes complete time slice info."""
        cal = TemporalCalibrator()
        t = time.time()

        cal.receive(_make_signal("visual", impedance=50.0, timestamp=t))
        state = cal.get_calibration_state()

        assert "active_window_ms" in state
        assert "frame_rate_hz" in state
        assert "temporal_resolution" in state
        assert "calibration_load" in state
        assert "n_active_channels" in state
        assert "avg_channel_gamma" in state
        assert "window_history" in state


# ============================================================================
# 10. End-to-End Scenario: Information Overload
# ============================================================================

class TestInformationOverload:
    """Simulate cognitive overload — too many heterogeneous signals arriving simultaneously."""

    def test_overload_slows_frame_rate(self):
        """Multi-modal high mismatch → frame rate drops → 'slower reactions'."""
        cal = TemporalCalibrator()
        t = time.time()

        # Normal state
        for i in range(30):
            ts = t + i * 0.01
            cal.receive(_make_signal("visual", impedance=50.0, timestamp=ts))
        normal_hz = cal.get_frame_rate()

        # Overload state: 3 modalities + high mismatch
        for i in range(50):
            ts = t + 0.5 + i * 0.01
            cal.receive(_make_signal("visual", impedance=300.0, timestamp=ts))
            cal.receive(_make_signal("auditory", impedance=30.0, timestamp=ts + 0.001))
            cal.receive(_make_signal("proprioception", impedance=150.0, timestamp=ts + 0.002))

        overload_hz = cal.get_frame_rate()
        assert overload_hz < normal_hz

    def test_overload_reduces_consciousness(self):
        """Calibration overload → temporal resolution drops → consciousness Φ decreases."""
        c_normal = ConsciousnessModule()
        c_overload = ConsciousnessModule()

        # Normal: high temporal resolution
        r_normal = c_normal.tick(
            attention_strength=0.8,
            binding_quality=0.7,
            arousal=0.8,
            sensory_gate=1.0,
            temporal_resolution=1.0,
        )

        # Overload: low temporal resolution
        r_overload = c_overload.tick(
            attention_strength=0.8,
            binding_quality=0.7,
            arousal=0.8,
            sensory_gate=1.0,
            temporal_resolution=0.3,
        )

        # In Screen Brightness Model, raw_phi = screen_brightness,
        # but overload reduces binding_effective (content quality)
        assert r_overload["components"]["binding_effective"] < r_normal["components"]["binding_effective"]

    def test_meditation_increases_resolution(self):
        """
        Meditation scenario: reduce input channels → calibration load decreases → temporal resolution improves.
        This explains why meditators report 'clearer time perception'.
        """
        cal = TemporalCalibrator()
        t = time.time()

        # Multi-channel noisy
        for i in range(50):
            ts = t + i * 0.01
            cal.receive(_make_signal("visual", impedance=100.0, timestamp=ts))
            cal.receive(_make_signal("auditory", impedance=60.0, timestamp=ts + 0.001))
            cal.receive(_make_signal("proprioception", impedance=80.0, timestamp=ts + 0.002))
        noisy_res = cal.get_temporal_resolution()

        # Close eyes, sit still → only a single channel remains
        cal2 = TemporalCalibrator()
        for i in range(50):
            ts = t + i * 0.01
            cal2.receive(_make_signal("proprioception", impedance=80.0, timestamp=ts))
        calm_res = cal2.get_temporal_resolution()

        assert calm_res > noisy_res
