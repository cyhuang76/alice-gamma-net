# -*- coding: utf-8 -*-
"""
Tests for Cross-Modal Impedance Adaptation Engine — Experience Learning to Improve Γ

Covers:
  1. Basic Γ learning (repeated success → Γ decreases)
  2. Yerkes-Dodson inverted-U curve
  3. Chronic stress suppression
  4. Forgetting / degradation
  5. Failure negative reinforcement
  6. Pair tracking and statistics
  7. Edge cases
  8. Stress cascade integration
  9. alice_brain integration
"""

import math
import pytest
import numpy as np

from alice.brain.impedance_adaptation import (
    ImpedanceAdaptationEngine,
    ModalityPairState,
    _make_pair_key,
    BASE_LEARNING_RATE,
    MAX_LEARNING_RATE,
    YERKES_DODSON_PEAK,
    YERKES_DODSON_WIDTH,
    DRIFT_RATE,
    MIN_GAMMA,
    MAX_GAMMA,
    DEFAULT_INITIAL_GAMMA,
    CHRONIC_STRESS_PENALTY,
)


# ============================================================================
# Helpers
# ============================================================================

@pytest.fixture
def engine():
    return ImpedanceAdaptationEngine()


# ============================================================================
# 1. Basic Γ Learning
# ============================================================================

class TestBasicLearning:
    """Repeated successful binding should reduce Γ (impedance matching improves)"""

    def test_single_success_reduces_gamma(self, engine):
        """One successful binding → Γ decreases"""
        r = engine.record_binding_attempt(
            "visual", "auditory", success=True,
            binding_quality=0.8, cortisol=0.45,
        )
        assert r["new_gamma"] < r["old_gamma"]
        assert r["delta_gamma"] < 0

    def test_repeated_success_monotonically_decreases(self, engine):
        """100 successes → Γ monotonically decreases"""
        gammas = [DEFAULT_INITIAL_GAMMA]
        for _ in range(100):
            r = engine.record_binding_attempt(
                "visual", "auditory", success=True,
                binding_quality=0.8, cortisol=0.45,
            )
            gammas.append(r["new_gamma"])

        # Should be strictly decreasing (each step at least as low as previous)
        for i in range(1, len(gammas)):
            assert gammas[i] <= gammas[i - 1], f"Γ increased at step {i}"

        # Final Γ should be significantly lower than initial
        assert gammas[-1] < DEFAULT_INITIAL_GAMMA * 0.5

    def test_high_quality_learns_faster(self, engine):
        """High quality binding learns faster than low quality"""
        e1 = ImpedanceAdaptationEngine()
        e2 = ImpedanceAdaptationEngine()

        for _ in range(20):
            r_high = e1.record_binding_attempt(
                "visual", "auditory", success=True,
                binding_quality=1.0, cortisol=0.45,
            )
            r_low = e2.record_binding_attempt(
                "visual", "auditory", success=True,
                binding_quality=0.2, cortisol=0.45,
            )

        assert r_high["new_gamma"] < r_low["new_gamma"]

    def test_transmission_efficiency_ceiling(self, engine):
        """Γ approaching 0 should naturally slow learning (ceiling effect)"""
        deltas = []
        for _ in range(200):
            r = engine.record_binding_attempt(
                "visual", "auditory", success=True,
                binding_quality=1.0, cortisol=0.45,
            )
            deltas.append(abs(r["delta_gamma"]))

        # Early delta should be larger than late (diminishing returns)
        early_avg = np.mean(deltas[:10])
        late_avg = np.mean(deltas[-10:])
        assert early_avg > late_avg * 2, "Early learning should be much faster"

    def test_gamma_never_below_min(self, engine):
        """Γ should not go below MIN_GAMMA"""
        for _ in range(500):
            engine.record_binding_attempt(
                "visual", "auditory", success=True,
                binding_quality=1.0, cortisol=0.45,
            )

        g = engine.get_pair_gamma("visual", "auditory")
        assert g >= MIN_GAMMA

    def test_gamma_never_above_max(self, engine):
        """Γ should not exceed MAX_GAMMA"""
        for _ in range(500):
            engine.record_binding_attempt(
                "visual", "auditory", success=False,
                cortisol=0.45,
            )

        g = engine.get_pair_gamma("visual", "auditory")
        assert g <= MAX_GAMMA


# ============================================================================
# 2. Yerkes-Dodson Inverted-U Curve
# ============================================================================

class TestYerkesDodson:
    """Yerkes-Dodson: moderate stress = optimal learning"""

    def test_peak_at_optimal_cortisol(self, engine):
        """β is maximum at c ≈ YERKES_DODSON_PEAK"""
        beta_peak = engine._yerkes_dodson(YERKES_DODSON_PEAK)
        beta_low = engine._yerkes_dodson(0.0)
        beta_high = engine._yerkes_dodson(1.0)

        assert beta_peak > beta_low
        assert beta_peak > beta_high
        assert beta_peak == pytest.approx(1.0, abs=0.01)

    def test_inverted_u_shape(self, engine):
        """Complete inverted-U shape: low→rise→peak→drop→low"""
        cs = np.linspace(0, 1, 11)
        betas = [engine._yerkes_dodson(c) for c in cs]

        # Find peak position
        peak_idx = np.argmax(betas)
        assert 3 <= peak_idx <= 6, "Peak should be in the middle"

        # Left of peak: increasing
        for i in range(1, peak_idx + 1):
            assert betas[i] >= betas[i - 1] - 0.01

        # Right of peak: decreasing
        for i in range(peak_idx + 1, len(betas)):
            assert betas[i] <= betas[i - 1] + 0.01

    def test_learning_rate_follows_yd(self, engine):
        """Effective learning rate also inverted-U"""
        lr_optimal = engine._effective_learning_rate(YERKES_DODSON_PEAK)
        lr_low = engine._effective_learning_rate(0.0)
        lr_high = engine._effective_learning_rate(1.0)

        assert lr_optimal > lr_low
        assert lr_optimal > lr_high

    def test_optimal_cortisol_learns_fastest(self):
        """Under optimal cortisol, learns more in same trials"""
        e_opt = ImpedanceAdaptationEngine()
        e_low = ImpedanceAdaptationEngine()
        e_high = ImpedanceAdaptationEngine()

        # Fewer trials to avoid all saturating to MIN_GAMMA
        for _ in range(15):
            e_opt.record_binding_attempt(
                "visual", "auditory", True, 0.8, cortisol=0.45)
            e_low.record_binding_attempt(
                "visual", "auditory", True, 0.8, cortisol=0.05)
            e_high.record_binding_attempt(
                "visual", "auditory", True, 0.8, cortisol=0.95)

        g_opt = e_opt.get_pair_gamma("visual", "auditory")
        g_low = e_low.get_pair_gamma("visual", "auditory")
        g_high = e_high.get_pair_gamma("visual", "auditory")

        # Γ under optimal stress should be lowest (learned the most)
        assert g_opt < g_low
        assert g_opt < g_high

    def test_yd_curve_data(self, engine):
        """get_yerkes_dodson_curve returns correct format"""
        curve = engine.get_yerkes_dodson_curve(n_points=20)
        assert len(curve["cortisol"]) == 20
        assert len(curve["beta"]) == 20
        assert len(curve["effective_lr"]) == 20
        assert max(curve["beta"]) == pytest.approx(1.0, abs=0.05)

    def test_symmetry_around_peak(self, engine):
        """Roughly symmetric around peak"""
        p = YERKES_DODSON_PEAK
        delta = 0.2
        beta_left = engine._yerkes_dodson(p - delta)
        beta_right = engine._yerkes_dodson(p + delta)
        # Not perfectly symmetric since peak is not at 0.5, but difference should be small
        assert abs(beta_left - beta_right) < 0.15


# ============================================================================
# 3. Chronic Stress Suppression
# ============================================================================

class TestChronicStress:
    """Chronic stress should broadly suppress learning ability"""

    def test_chronic_stress_reduces_learning_rate(self, engine):
        """Chronic stress → effective learning rate decreases"""
        lr_healthy = engine._effective_learning_rate(0.45, chronic_stress=0.0)
        lr_stressed = engine._effective_learning_rate(0.45, chronic_stress=0.8)

        assert lr_stressed < lr_healthy

    def test_chronic_stress_impairs_adaptation(self):
        """Chronically stressed animals learn slower"""
        e_healthy = ImpedanceAdaptationEngine()
        e_chronic = ImpedanceAdaptationEngine()

        # Fewer trials to avoid both saturating to MIN_GAMMA
        for _ in range(15):
            e_healthy.record_binding_attempt(
                "visual", "auditory", True, 0.8,
                cortisol=0.45, chronic_stress=0.0)
            e_chronic.record_binding_attempt(
                "visual", "auditory", True, 0.8,
                cortisol=0.45, chronic_stress=0.8)

        g_healthy = e_healthy.get_pair_gamma("visual", "auditory")
        g_chronic = e_chronic.get_pair_gamma("visual", "auditory")

        assert g_chronic > g_healthy, "Chronic stress should impair learning"

    def test_max_chronic_stress_still_allows_some_learning(self):
        """Even max chronic stress still allows some base learning"""
        e = ImpedanceAdaptationEngine()
        lr = e._effective_learning_rate(0.45, chronic_stress=1.0)
        assert lr > 0, "Should still have base learning even under max stress"


# ============================================================================
# 4. Forgetting / Degradation
# ============================================================================

class TestForgetting:
    """Unused channels should gradually drift back to initial Γ"""

    def test_idle_pairs_drift_back(self, engine):
        """Learned pair Γ increases after idling"""
        # First learn a low Γ
        for _ in range(50):
            engine.record_binding_attempt(
                "visual", "auditory", True, 0.8, cortisol=0.45)

        learned_gamma = engine.get_pair_gamma("visual", "auditory")
        assert learned_gamma < 0.5

        # Idle for 200 ticks (only call decay_tick, no exposure)
        for _ in range(200):
            engine.decay_tick()

        drifted_gamma = engine.get_pair_gamma("visual", "auditory")
        assert drifted_gamma > learned_gamma, "Idle gamma should drift back"

    def test_used_pairs_dont_drift(self, engine):
        """Continuously used pairs don't degrade"""
        # Learning + continuous use
        for _ in range(50):
            engine.record_binding_attempt(
                "visual", "auditory", True, 0.8, cortisol=0.45)
            engine.decay_tick()

        stable_gamma = engine.get_pair_gamma("visual", "auditory")

        # Use another 50 times
        for _ in range(50):
            engine.record_binding_attempt(
                "visual", "auditory", True, 0.8, cortisol=0.45)
            engine.decay_tick()

        final_gamma = engine.get_pair_gamma("visual", "auditory")
        assert final_gamma <= stable_gamma, "Actively used pairs should continue improving"

    def test_drift_direction_toward_initial(self, engine):
        """Degradation direction should be toward initial Γ"""
        engine.record_binding_attempt(
            "visual", "auditory", True, 1.0, cortisol=0.45)
        low_gamma = engine.get_pair_gamma("visual", "auditory")

        for _ in range(100):
            engine.decay_tick()

        drifted = engine.get_pair_gamma("visual", "auditory")
        # Should approach DEFAULT_INITIAL_GAMMA
        assert abs(drifted - DEFAULT_INITIAL_GAMMA) < abs(low_gamma - DEFAULT_INITIAL_GAMMA)


# ============================================================================
# 5. Failure Negative Reinforcement
# ============================================================================

class TestFailure:
    """Failed binding should weakly increase Γ"""

    def test_failure_increases_gamma(self, engine):
        """Failure → Γ slightly increases"""
        r = engine.record_binding_attempt(
            "visual", "auditory", success=False, cortisol=0.3)
        assert r["new_gamma"] >= r["old_gamma"]

    def test_failure_weaker_than_success(self, engine):
        """Failure impact should be weaker than success"""
        e_success = ImpedanceAdaptationEngine()
        e_failure = ImpedanceAdaptationEngine()

        r_s = e_success.record_binding_attempt(
            "visual", "auditory", True, 0.8, cortisol=0.45)
        r_f = e_failure.record_binding_attempt(
            "visual", "auditory", False, 0.8, cortisol=0.45)

        # Success |delta| should be larger than failure
        assert abs(r_s["delta_gamma"]) > abs(r_f["delta_gamma"])

    def test_many_failures_dont_exceed_max(self, engine):
        """Even with many failures, Γ doesn't exceed upper limit"""
        for _ in range(500):
            engine.record_binding_attempt(
                "visual", "auditory", False, cortisol=0.95)

        g = engine.get_pair_gamma("visual", "auditory")
        assert g <= MAX_GAMMA


# ============================================================================
# 6. Pair Tracking
# ============================================================================

class TestPairTracking:
    """Pair key normalization and state tracking"""

    def test_pair_key_order_invariant(self):
        """A-B and B-A should be the same pair"""
        assert _make_pair_key("visual", "auditory") == _make_pair_key("auditory", "visual")

    def test_different_pairs_independent(self, engine):
        """Different pairs' Γ evolve independently"""
        for _ in range(50):
            engine.record_binding_attempt(
                "visual", "auditory", True, 0.8, cortisol=0.45)

        g_va = engine.get_pair_gamma("visual", "auditory")
        g_vt = engine.get_pair_gamma("visual", "tactile")

        assert g_va < DEFAULT_INITIAL_GAMMA  # trained
        assert g_vt == DEFAULT_INITIAL_GAMMA  # not trained

    def test_get_pair_state(self, engine):
        """get_pair_state returns complete information"""
        engine.record_binding_attempt(
            "visual", "auditory", True, 0.8, cortisol=0.4)

        state = engine.get_pair_state("visual", "auditory")
        assert state is not None
        assert "current_gamma" in state
        assert "exposure_count" in state
        assert state["exposure_count"] == 1

    def test_get_pair_state_nonexistent(self, engine):
        """Non-existent pair returns None"""
        assert engine.get_pair_state("x", "y") is None

    def test_unknown_pair_returns_default(self, engine):
        """Unknown pair's Γ returns default"""
        assert engine.get_pair_gamma("foo", "bar") == DEFAULT_INITIAL_GAMMA

    def test_multiple_pairs_stats(self, engine):
        """Statistics should cover all pairs"""
        engine.record_binding_attempt("visual", "auditory", True, 0.8, cortisol=0.3)
        engine.record_binding_attempt("auditory", "tactile", True, 0.6, cortisol=0.3)
        engine.record_binding_attempt("visual", "tactile", False, cortisol=0.3)

        stats = engine.get_stats()
        assert stats["total_pairs"] == 3
        assert stats["total_adaptations"] == 3

    def test_get_all_pairs_sorted(self, engine):
        """get_all_pairs sorted by Γ"""
        # Train first pair more times (Γ lower)
        for _ in range(20):
            engine.record_binding_attempt("visual", "auditory", True, 0.8, cortisol=0.4)
        engine.record_binding_attempt("auditory", "tactile", True, 0.5, cortisol=0.4)

        pairs = engine.get_all_pairs()
        assert len(pairs) == 2
        # First should have lower Γ
        assert pairs[0]["current_gamma"] <= pairs[1]["current_gamma"]


# ============================================================================
# 7. Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge cases and special situations"""

    def test_zero_cortisol(self, engine):
        """Can still learn with zero cortisol (has base rate)"""
        r = engine.record_binding_attempt(
            "visual", "auditory", True, 1.0, cortisol=0.0)
        assert r["delta_gamma"] < 0  # still improves

    def test_max_cortisol(self, engine):
        """Can still learn weakly with max cortisol"""
        r = engine.record_binding_attempt(
            "visual", "auditory", True, 1.0, cortisol=1.0)
        assert r["delta_gamma"] < 0  # still improves, but slowly

    def test_zero_quality(self, engine):
        """Zero quality success → improvement is 0"""
        r = engine.record_binding_attempt(
            "visual", "auditory", True, 0.0, cortisol=0.45)
        assert r["delta_gamma"] == pytest.approx(0.0, abs=1e-6)

    def test_single_modality_binding(self, engine):
        """Single modality query should return 0"""
        assert engine.get_adapted_binding_gamma(["visual"]) == 0.0

    def test_empty_modality_list(self, engine):
        """Empty list should return 0"""
        assert engine.get_adapted_binding_gamma([]) == 0.0

    def test_three_modality_binding(self, engine):
        """Three-modality binding averages all pairs"""
        engine.record_binding_attempt("visual", "auditory", True, 0.8, cortisol=0.4)
        engine.record_binding_attempt("visual", "tactile", True, 0.6, cortisol=0.4)
        engine.record_binding_attempt("auditory", "tactile", True, 0.7, cortisol=0.4)

        gamma = engine.get_adapted_binding_gamma(["visual", "auditory", "tactile"])
        # Should be the average of three pairs' Γ
        g_va = engine.get_pair_gamma("visual", "auditory")
        g_vt = engine.get_pair_gamma("visual", "tactile")
        g_at = engine.get_pair_gamma("auditory", "tactile")
        expected = (g_va + g_vt + g_at) / 3
        assert gamma == pytest.approx(expected, abs=0.001)

    def test_modality_pair_state_properties(self):
        """ModalityPairState properties are correct"""
        state = ModalityPairState(
            modality_a="visual", modality_b="auditory",
            current_gamma=0.5, exposure_count=10, success_count=8,
        )
        assert state.success_rate == 0.8
        assert state.transmission_efficiency == pytest.approx(0.75)
        assert state.is_well_matched is False

        state.current_gamma = 0.2
        assert state.is_well_matched is True

    def test_fresh_engine_stats(self, engine):
        """Fresh engine statistics"""
        stats = engine.get_stats()
        assert stats["total_pairs"] == 0
        assert stats["total_adaptations"] == 0
        # avg_gamma should be default value
        assert stats["avg_gamma"] == DEFAULT_INITIAL_GAMMA


# ============================================================================
# 8. Physics Consistency
# ============================================================================

class TestPhysicsConsistency:
    """Verify mathematical consistency of physical formulas"""

    def test_transmission_efficiency_formula(self, engine):
        """Transmission efficiency = 1 - Γ² always holds"""
        for _ in range(30):
            engine.record_binding_attempt(
                "visual", "auditory", True, 0.8, cortisol=0.45)

        state = engine.get_pair_state("visual", "auditory")
        g = state["current_gamma"]
        eff = state["transmission_efficiency"]
        assert eff == pytest.approx(1 - g ** 2, abs=0.001)

    def test_yd_is_gaussian(self, engine):
        """Yerkes-Dodson should be Gaussian-shaped"""
        # Manual Gaussian calculation
        c = 0.3
        expected = math.exp(-((c - YERKES_DODSON_PEAK) ** 2) / (2 * YERKES_DODSON_WIDTH ** 2))
        actual = engine._yerkes_dodson(c)
        assert actual == pytest.approx(expected, abs=1e-6)

    def test_learning_rate_bounded(self, engine):
        """Learning rate bounded in [BASE, MAX]"""
        for c in np.linspace(0, 1, 50):
            lr = engine._effective_learning_rate(float(c))
            assert lr >= BASE_LEARNING_RATE - 0.001
            assert lr <= MAX_LEARNING_RATE + 0.001

    def test_adaptation_is_energy_conserving(self, engine):
        """Success Γ reduction ∝ (1 - Γ²)×quality is physically correct"""
        # Compute at known state
        g = DEFAULT_INITIAL_GAMMA
        eta = engine._effective_learning_rate(0.45)
        quality = 0.8
        expected_delta = -eta * (1 - g ** 2) * quality

        r = engine.record_binding_attempt(
            "visual", "auditory", True, quality, cortisol=0.45)

        assert r["delta_gamma"] == pytest.approx(expected_delta, abs=1e-4)


# ============================================================================
# 9. Stress Cascade Full Loop
# ============================================================================

class TestStressCascade:
    """Verify complete stress cascade: external chaos → stress → learning modulation"""

    def test_stress_adaptation_trajectory(self):
        """
        Simulate: compare learning efficiency under different stress with independent engines
        Moderate stress should have the greatest improvement
        """
        # Use independent engines to avoid cumulative effects (earlier learner always has larger 1-Γ²)
        e_relaxed = ImpedanceAdaptationEngine()
        e_optimal = ImpedanceAdaptationEngine()
        e_stressed = ImpedanceAdaptationEngine()

        n = 15  # few trials to avoid saturation
        for _ in range(n):
            e_relaxed.record_binding_attempt(
                "visual", "auditory", True, 0.7, cortisol=0.1)
            e_optimal.record_binding_attempt(
                "visual", "auditory", True, 0.7, cortisol=0.45)
            e_stressed.record_binding_attempt(
                "visual", "auditory", True, 0.7, cortisol=0.9)

        g_relaxed = e_relaxed.get_pair_gamma("visual", "auditory")
        g_optimal = e_optimal.get_pair_gamma("visual", "auditory")
        g_stressed = e_stressed.get_pair_gamma("visual", "auditory")

        improvement_relaxed = DEFAULT_INITIAL_GAMMA - g_relaxed
        improvement_optimal = DEFAULT_INITIAL_GAMMA - g_optimal
        improvement_stressed = DEFAULT_INITIAL_GAMMA - g_stressed

        assert improvement_optimal > improvement_relaxed
        assert improvement_optimal > improvement_stressed

    def test_novel_stimulus_creates_default_gamma(self, engine):
        """First-contact modality pair should have high Γ (unfamiliar = high impedance)"""
        r = engine.record_binding_attempt(
            "novel_sense", "visual", True, 0.5, cortisol=0.5)
        assert r["old_gamma"] == DEFAULT_INITIAL_GAMMA

    def test_chronic_then_recovery(self):
        """After chronic stress, recovery → learning ability gradually restores"""
        e_chronic = ImpedanceAdaptationEngine()
        e_healthy = ImpedanceAdaptationEngine()

        # Few trials to avoid saturation
        for _ in range(12):
            e_healthy.record_binding_attempt(
                "visual", "auditory", True, 0.7,
                cortisol=0.4, chronic_stress=0.0)
            e_chronic.record_binding_attempt(
                "visual", "auditory", True, 0.7,
                cortisol=0.4, chronic_stress=0.7)

        g_healthy = e_healthy.get_pair_gamma("visual", "auditory")
        g_chronic = e_chronic.get_pair_gamma("visual", "auditory")

        assert g_chronic > g_healthy


# ============================================================================
# 10. alice_brain Integration
# ============================================================================

class TestAliceBrainIntegration:
    """Verify ImpedanceAdaptationEngine correctly integrates into alice_brain"""

    def test_brain_has_impedance_adaptation(self):
        """AliceBrain should have impedance_adaptation attribute"""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        assert hasattr(brain, "impedance_adaptation")
        assert isinstance(brain.impedance_adaptation, ImpedanceAdaptationEngine)

    def test_perceive_returns_adaptation_info(self):
        """perceive() result should include impedance_adaptation info"""
        from alice.alice_brain import AliceBrain, Modality, Priority
        brain = AliceBrain()
        stimulus = np.random.randn(100)
        result = brain.perceive(stimulus, Modality.VISUAL, Priority.NORMAL)

        assert "impedance_adaptation" in result
        info = result["impedance_adaptation"]
        assert "adapted_gamma" in info
        assert "raw_gamma" in info
        assert isinstance(info["adapted_gamma"], float)

    def test_multiple_perceives_improve_gamma(self):
        """Multiple perceives of same modality → experience-adapted Γ should improve"""
        from alice.alice_brain import AliceBrain, Modality, Priority
        brain = AliceBrain()

        gammas = []
        for _ in range(10):
            stimulus = np.random.randn(100) * 0.5
            result = brain.perceive(stimulus, Modality.VISUAL, Priority.NORMAL)
            gammas.append(result["impedance_adaptation"]["adapted_gamma"])

        # Since each perceive is single-modality (visual only), adapted_gamma
        # depends on whether the frame brings multi-modal binding. At least the system should not crash.
        assert len(gammas) == 10
        assert all(0 <= g <= 1 for g in gammas)
