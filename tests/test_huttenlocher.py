# -*- coding: utf-8 -*-
"""
Tests for Huttenlocher Developmental Curve (added to pruning.py)

Tests verify:
    1. huttenlocher_curve() function shape
    2. Regional variation (temporal peaks earliest, frontal latest)
    3. NeuralPruningEngine.get_huttenlocher_density()
    4. NeuralPruningEngine.apply_huttenlocher_trajectory()
    5. Curve data retrieval
"""

import pytest
import numpy as np

from alice.brain.pruning import (
    NeuralPruningEngine,
    huttenlocher_curve,
    HUTTENLOCHER_PARAMS,
)


class TestHuttenlocherCurveFunction:
    """Test the standalone huttenlocher_curve function."""

    def test_birth_is_baseline(self):
        """At t=0, density should be baseline (1.0)."""
        val = huttenlocher_curve(0, tau_rise=80, tau_fall=550)
        assert abs(val - 1.0) < 0.01

    def test_peak_exceeds_baseline(self):
        """At some intermediate t, density > 1.0 (exuberant connectivity)."""
        max_val = max(
            huttenlocher_curve(t, tau_rise=80, tau_fall=550)
            for t in range(1, 500)
        )
        assert max_val > 1.3  # Should reach ~150%

    def test_returns_to_baseline(self):
        """At very large t, density → ~1.0 (adult plateau)."""
        val = huttenlocher_curve(5000, tau_rise=80, tau_fall=550)
        assert abs(val - 1.0) < 0.2

    def test_monotonic_rise_then_fall(self):
        """Curve should rise to peak, then fall back."""
        vals = [huttenlocher_curve(t, tau_rise=80, tau_fall=550) for t in range(1, 2000)]
        peak_idx = np.argmax(vals)
        # Peak should not be at the start or end
        assert 0 < peak_idx < len(vals) - 1


class TestRegionalVariation:
    """Different regions peak at different times."""

    def test_temporal_peaks_before_frontal(self):
        """Temporal cortex (auditory) peaks earlier than frontal (motor)."""
        temporal_params = HUTTENLOCHER_PARAMS["temporal"]
        frontal_params = HUTTENLOCHER_PARAMS["frontal_motor"]

        temporal_peak_t = np.argmax([
            huttenlocher_curve(t, **temporal_params) for t in range(1, 2000)
        ])
        frontal_peak_t = np.argmax([
            huttenlocher_curve(t, **frontal_params) for t in range(1, 2000)
        ])
        assert temporal_peak_t < frontal_peak_t

    def test_all_regions_defined(self):
        for region in ["occipital", "temporal", "parietal", "frontal_motor"]:
            assert region in HUTTENLOCHER_PARAMS


class TestEngineHuttenlocherDensity:
    """NeuralPruningEngine.get_huttenlocher_density()."""

    def test_density_at_zero(self):
        engine = NeuralPruningEngine(connections_per_region=100)
        densities = engine.get_huttenlocher_density(0)
        for name, val in densities.items():
            assert abs(val - 1.0) < 0.1

    def test_density_at_peak(self):
        engine = NeuralPruningEngine(connections_per_region=100)
        densities = engine.get_huttenlocher_density(100)
        # At least one region should be above baseline
        assert any(v > 1.05 for v in densities.values())


class TestApplyTrajectory:
    """NeuralPruningEngine.apply_huttenlocher_trajectory()."""

    def test_apply_returns_dict(self):
        engine = NeuralPruningEngine(connections_per_region=100)
        result = engine.apply_huttenlocher_trajectory(50)
        assert isinstance(result, dict)
        assert len(result) == 4  # 4 regions

    def test_apply_has_phase(self):
        engine = NeuralPruningEngine(connections_per_region=100)
        result = engine.apply_huttenlocher_trajectory(50)
        for name, info in result.items():
            assert "phase" in info
            assert info["phase"] in ("synaptogenesis", "pruning")


class TestCurveData:
    """get_huttenlocher_curve_data()."""

    def test_curve_data_structure(self):
        engine = NeuralPruningEngine(connections_per_region=100)
        engine.apply_huttenlocher_trajectory(10)
        data = engine.get_huttenlocher_curve_data()
        assert "targets" in data
        assert "history" in data
        assert "params" in data
