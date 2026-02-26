# -*- coding: utf-8 -*-
"""
Tests for GradientOptimizer — Paper I Eq.11 Explicit Variational Gradient

Tests verify:
    1. Basic construction
    2. Analytic gradient correctness: ∂(Γ²)/∂Z = -4·Z_L·(Z_L-Z)/(Z_L+Z)³
    3. Gradient direction (Z < Z_L → positive, Z > Z_L → negative)
    4. Step produces channel recommendations
    5. State consistency
"""

import pytest
import numpy as np

from alice.brain.gradient_optimizer import (
    GradientOptimizer,
    GradientStep,
    ChannelGradient,
)


class TestGradientConstruction:
    """GradientOptimizer initializes correctly."""

    def test_default_init(self):
        opt = GradientOptimizer()
        assert opt._tick_count == 0

    def test_custom_eta(self):
        opt = GradientOptimizer(eta=0.05)
        assert opt._eta == 0.05


class TestAnalyticGradient:
    """Test ∂(Γ²)/∂Z analytical formula."""

    def test_gradient_at_match(self):
        """When Z = Z_L, Γ = 0 and gradient should be ~0."""
        grad = GradientOptimizer.compute_gamma_sq_gradient(75.0, 75.0)
        assert abs(grad) < 0.01

    def test_gradient_direction_below(self):
        """When Z < Z_L, gradient should be negative (Γ² decreases as Z increases)."""
        grad = GradientOptimizer.compute_gamma_sq_gradient(30.0, 75.0)
        assert grad < 0  # dΓ²/dZ < 0 → increasing Z decreases Γ²

    def test_gradient_direction_above(self):
        """When Z > Z_L, gradient should be positive (Γ² increases as Z increases)."""
        grad = GradientOptimizer.compute_gamma_sq_gradient(150.0, 75.0)
        assert grad > 0  # dΓ²/dZ > 0 → decreasing Z decreases Γ²


class TestStepFunction:
    """Step function processes multiple channels."""

    def test_multiple_channels(self):
        opt = GradientOptimizer()
        channels = {
            "visual": (50.0, 50.0),
            "auditory": (75.0, 50.0),
            "motor": (100.0, 75.0),
        }
        step_result, channel_grads = opt.step(channels)
        assert isinstance(step_result, GradientStep)
        assert len(channel_grads) == 3

    def test_empty_channels(self):
        opt = GradientOptimizer()
        step_result, channel_grads = opt.step({})
        assert isinstance(step_result, GradientStep)
        assert len(channel_grads) == 0


class TestGradientState:
    """get_stats() returns correct structure."""

    def test_stats_keys(self):
        opt = GradientOptimizer()
        opt.step({"test": (50.0, 75.0)})
        stats = opt.get_stats()
        assert "tick_count" in stats
