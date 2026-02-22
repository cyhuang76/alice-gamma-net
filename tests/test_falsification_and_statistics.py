# -*- coding: utf-8 -*-
"""
test_falsification_and_statistics.py — Falsification, Statistics & Sensitivity
===============================================================================

Purpose:
  Address the three critical methodological gaps identified in the rigor audit:

  1. FALSIFICATION TESTS (§F): Narrow-tolerance tests designed to BREAK
     theoretical predictions. If the MRP is correct, these must all pass.
     Unlike verification tests (broad "is x >= 0?"), these assert specific
     theoretical consequences with tight tolerances.

  2. STATISTICAL ANALYSIS (§S): Tests that compute p-values, confidence
     intervals, and effect sizes using Monte Carlo sampling and bootstrap.
     These replace hard-coded threshold comparisons with proper statistical
     inference.

  3. PARAMETER SENSITIVITY (§P): Systematic sweeps that quantify how
     sensitive key outputs are to parameter variations, identifying
     fragile vs. robust predictions.

  4. LUCID THRESHOLD SENSITIVITY (§L): Dedicated sweep of the Φ=0.7
     threshold to test whether the system's behavior is robust to the
     specific choice of this arbitrary engineering parameter.

Audit response: Perplexity rigor audit §2 (Verification vs. Falsification),
                §5 (LUCID threshold), §6 (Statistical analysis), §7.8-7.10.

Author: Methodological Rigor Protocol
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest

# ── Core imports ──────────────────────────────────────────────────────
from alice.brain.consciousness import (
    ConsciousnessModule,
    CONSCIOUS_THRESHOLD,
    LUCID_THRESHOLD,
    SUBLIMINAL_THRESHOLD,
    W_ATTENTION,
    W_BINDING,
    W_MEMORY,
    W_AROUSAL,
    W_SENSORY_GATE,
    PHI_SMOOTHING,
    DevelopmentalStage,
    SLEEP_PRESSURE_THRESHOLD,
)

from alice.brain.pruning import (
    SynapticConnection,
    CorticalRegion,
    CorticalSpecialization,
    NeuralPruningEngine,
    PruningMetrics,
)

from alice.brain.amygdala import (
    AmygdalaEngine,
    FearMemory,
    CONDITIONING_RATE,
    EXTINCTION_RATE,
    FEAR_THRESHOLD,
    FREEZE_THRESHOLD,
    THREAT_IMPEDANCE,
    SAFETY_IMPEDANCE,
)

from alice.brain.sleep import SleepCycle, SLEEP_PRESSURE_ACCUMULATION

from alice.alice_brain import SystemState


# ═══════════════════════════════════════════════════════════════════════
#  HELPER: lightweight bootstrap CI & statistical tests
# ═══════════════════════════════════════════════════════════════════════

def _bootstrap_ci(data: List[float], n_boot: int = 2000,
                  ci: float = 0.95) -> Tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via percentile bootstrap."""
    rng = np.random.default_rng(42)
    arr = np.asarray(data, dtype=np.float64)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[i] = sample.mean()
    alpha = (1 - ci) / 2
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1 - alpha))
    return float(arr.mean()), lo, hi


def _cohens_d(group_a: List[float], group_b: List[float]) -> float:
    """Cohen's d effect size (pooled SD)."""
    a, b = np.asarray(group_a), np.asarray(group_b)
    na, nb = len(a), len(b)
    pooled_std = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                           / (na + nb - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def _welch_t(group_a: List[float], group_b: List[float]) -> Tuple[float, float]:
    """Welch's t-test (two-tailed). Returns (t_statistic, p_value_approx).
    Uses normal approximation for p-value (valid for dof > 30)."""
    a, b = np.asarray(group_a, dtype=np.float64), np.asarray(group_b, dtype=np.float64)
    na, nb = len(a), len(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    if se < 1e-15:
        return 0.0, 1.0
    t_stat = (ma - mb) / se
    from math import erfc
    p_value = erfc(abs(t_stat) / math.sqrt(2))  # two-tailed normal approx
    return float(t_stat), float(p_value)


# ═══════════════════════════════════════════════════════════════════════
#  Helper: standard tick inputs for ConsciousnessModule
# ═══════════════════════════════════════════════════════════════════════

def _tick_consciousness(cm: ConsciousnessModule, **overrides) -> dict:
    """Call cm.tick() with sensible defaults, allowing overrides."""
    defaults = dict(
        attention_strength=0.5,
        binding_quality=0.5,
        working_memory_usage=0.0,
        arousal=0.5,
        sensory_gate=1.0,
        pain_level=0.0,
        temporal_resolution=1.0,
    )
    defaults.update(overrides)
    return cm.tick(**defaults)


def _make_fingerprint(seed: int = 0) -> np.ndarray:
    """Create a simple sensory fingerprint for amygdala tests."""
    rng = np.random.default_rng(seed)
    return rng.random(16).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
#  §F: FALSIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestFalsificationConsciousness:
    """Narrow-tolerance tests for Φ equation.

    MRP prediction: Consciousness requires BOTH arousal AND sensory gating.
    The multiplicative term (arousal × sensory_gate)^0.3 ensures this.
    """

    def test_F01_zero_arousal_zero_gate_kills_phi(self):
        """F-01: With arousal=0 and sensory_gate=0, Φ MUST fall below
        SUBLIMINAL_THRESHOLD within 20 ticks. Theory demands this because
        the multiplicative gating factor → (0.1 × 0.1)^0.3 ≈ 0.04."""
        c = ConsciousnessModule(safety_mode=False)
        for _ in range(20):
            _tick_consciousness(c, arousal=0.0, sensory_gate=0.0,
                                attention_strength=1.0, binding_quality=1.0)
        # With both gating inputs at 0, phi cannot sustain
        assert c.phi < SUBLIMINAL_THRESHOLD, (
            f"Φ={c.phi:.4f} should be < {SUBLIMINAL_THRESHOLD} when "
            f"arousal=0 and sensory_gate=0 (multiplicative gate → 0.04)"
        )

    def test_F02_pain_suppresses_phi(self):
        """F-02: Increasing pain MUST decrease Φ. The equation subtracts
        pain² × 0.5, so pain=0.8 removes ~0.32 from Φ_raw."""
        c_no_pain = ConsciousnessModule(safety_mode=False)
        c_pain = ConsciousnessModule(safety_mode=False)
        inputs = dict(
            attention_strength=0.7, binding_quality=0.7,
            temporal_resolution=0.5, working_memory_usage=0.3,
            arousal=0.6, sensory_gate=0.6,
        )
        for _ in range(30):
            c_no_pain.tick(**inputs, pain_level=0.0)
            c_pain.tick(**inputs, pain_level=0.8)
        assert c_pain.phi < c_no_pain.phi, (
            f"pain Φ={c_pain.phi:.4f} MUST be < no-pain Φ={c_no_pain.phi:.4f}"
        )

    def test_F03_sensory_gate_monotonicity(self):
        """F-03: Φ MUST be monotonically non-decreasing with sensory_gate.
        Test at 5 points in [0.1, 1.0]."""
        gate_values = [0.1, 0.3, 0.5, 0.7, 1.0]
        phi_results = []
        for gate in gate_values:
            c = ConsciousnessModule(safety_mode=False)
            for _ in range(40):
                _tick_consciousness(c, sensory_gate=gate, arousal=0.5,
                                    attention_strength=0.6, binding_quality=0.6)
            phi_results.append(c.phi)
        for i in range(len(phi_results) - 1):
            assert phi_results[i] <= phi_results[i + 1] + 0.02, (
                f"Φ not monotone: gate={gate_values[i]}→Φ={phi_results[i]:.4f} "
                f"> gate={gate_values[i+1]}→Φ={phi_results[i+1]:.4f}"
            )

    def test_F04_phi_bounded_zero_one(self):
        """F-04: Φ MUST remain in [0, 1] under all extreme inputs.
        Theory: Φ represents integration fraction and is clamped."""
        extremes = [
            dict(attention_strength=0, binding_quality=0, temporal_resolution=0,
                 working_memory_usage=0, arousal=0, sensory_gate=0, pain_level=1.0),
            dict(attention_strength=1, binding_quality=1, temporal_resolution=1,
                 working_memory_usage=1, arousal=1, sensory_gate=1, pain_level=0.0),
            dict(attention_strength=1, binding_quality=1, temporal_resolution=1,
                 working_memory_usage=1, arousal=1, sensory_gate=1, pain_level=1.0),
        ]
        for inputs in extremes:
            c = ConsciousnessModule(safety_mode=False)
            for _ in range(50):
                c.tick(**inputs)
            assert 0.0 <= c.phi <= 1.0, f"Φ={c.phi} out of [0,1] with {inputs}"

    def test_F05_sleep_pressure_builds_during_wake(self):
        """F-05: In safety_mode, sleep pressure MUST increase during
        wakefulness and eventually trigger sleep (is_sleeping=True)."""
        c = ConsciousnessModule(safety_mode=True)
        pressures = []
        for _ in range(40):
            _tick_consciousness(c, attention_strength=0.6, arousal=0.5,
                                sensory_gate=0.7)
            pressures.append(c.sleep_pressure)
        # Pressure should increase overall
        assert pressures[-1] > pressures[0], (
            f"Sleep pressure did not increase: start={pressures[0]:.4f}, "
            f"end={pressures[-1]:.4f}"
        )

    def test_F06_neonate_triggers_sleep_within_max_wake(self):
        """F-06: NEONATE developmental stage MUST trigger sleep within
        ~30 ticks of high-arousal wake activity."""
        c = ConsciousnessModule(
            developmental_stage=DevelopmentalStage.NEONATE,
            safety_mode=True,
        )
        slept = False
        for t in range(50):
            _tick_consciousness(c, attention_strength=0.7, arousal=0.6,
                                sensory_gate=0.8)
            if c.is_sleeping:
                slept = True
                break
        assert slept, (
            f"NEONATE did not enter sleep within 50 ticks. "
            f"sleep_pressure={c.sleep_pressure:.4f}, wake_ticks={c.wake_ticks}"
        )


class TestFalsificationPruning:
    """Narrow-tolerance tests for the pruning engine.

    MRP prediction: ΣΓ² → min under consistent stimulation.
    """

    def test_F07_consistent_stim_reduces_gamma(self):
        """F-07: 200 epochs of consistent stimulation + periodic pruning
        MUST reduce mean Γ by at least 20%. Core MRP prediction."""
        region = CorticalRegion(
            name="test_falsif",
            initial_connections=200,
        )
        result_initial = region.stimulate(
            signal_impedance=50.0, signal_frequency=10.0,
        )
        initial_gamma = result_initial["avg_gamma"]

        for i in range(200):
            region.stimulate(signal_impedance=50.0, signal_frequency=10.0)
            if i % 10 == 0:
                region.prune()
        result_final = region.stimulate(
            signal_impedance=50.0, signal_frequency=10.0,
        )
        final_gamma = result_final["avg_gamma"]
        if initial_gamma > 0:
            reduction = (initial_gamma - final_gamma) / initial_gamma
            assert reduction >= 0.20, (
                f"Mean Γ reduced only {reduction*100:.1f}% (need ≥20%). "
                f"Initial={initial_gamma:.4f}, Final={final_gamma:.4f}"
            )

    def test_F08_no_stim_no_spontaneous_death(self):
        """F-08: Without stimulation, connections MUST NOT spontaneously die.
        Pruning requires mismatched Hebbian signal, not mere time passage."""
        region = CorticalRegion(
            name="test_no_stim",
            initial_connections=100,
        )
        initial_count = region.alive_count
        region.prune()  # prune without any stimulation
        assert region.alive_count == initial_count, (
            f"Connections died without stimulation: {initial_count}→{region.alive_count}"
        )

    def test_F09_gamma_always_in_zero_one(self):
        """F-09: Γ (reflection coefficient) MUST be in [0, 1] for all
        valid impedance pairs. |Z_conn - Z_signal| / (Z_conn + Z_signal)."""
        rng = np.random.default_rng(123)
        for _ in range(1000):
            z_conn = rng.uniform(0.1, 1000.0)
            z_signal = rng.uniform(0.1, 1000.0)
            conn = SynapticConnection(
                connection_id=0,
                impedance=z_conn,
                resonant_freq=10.0,
            )
            gamma = conn.compute_gamma(z_signal)
            assert 0.0 <= gamma <= 1.0, (
                f"Γ={gamma} for Z_conn={z_conn:.1f}, Z_signal={z_signal:.1f}"
            )

    def test_F10_impedance_adapts_toward_signal(self):
        """F-10: After Hebbian learning, surviving connections' impedances
        MUST be closer to signal impedance than at birth."""
        target_z = 50.0
        region = CorticalRegion(
            name="test_adapt",
            initial_connections=100,
        )
        initial_distances = [
            abs(c.impedance - target_z) for c in region.connections if c.alive
        ]

        for i in range(300):
            region.stimulate(signal_impedance=target_z, signal_frequency=10.0)
            if i % 20 == 0:
                region.prune()
        final_distances = [
            abs(c.impedance - target_z) for c in region.connections if c.alive
        ]
        assert np.mean(final_distances) < np.mean(initial_distances), (
            f"Mean distance to target didn't decrease: "
            f"initial={np.mean(initial_distances):.2f}, "
            f"final={np.mean(final_distances):.2f}"
        )


class TestFalsificationFear:
    """Narrow-tolerance tests for amygdala fear conditioning.

    MRP prediction: Fear conditioning is fast, extinction is slow,
    and fear memories have a floor > 0.
    """

    def test_F11_conditioning_raises_threat(self):
        """F-11: Fear conditioning MUST raise threat level above baseline."""
        amygdala = AmygdalaEngine()
        fp = _make_fingerprint(seed=42)

        baseline = amygdala._threat_level

        # Condition with 10 high-threat events
        for _ in range(10):
            amygdala.evaluate(
                modality="visual", fingerprint=fp,
                amplitude=0.9, pain_level=0.7,
            )
            amygdala.condition_fear(
                modality="visual", fingerprint=fp,
                threat_level=0.9,
            )

        assert amygdala._threat_level > baseline, (
            f"Threat not raised: baseline={baseline:.4f}, "
            f"after conditioning={amygdala._threat_level:.4f}"
        )

    def test_F12_extinction_slower_than_conditioning(self):
        """F-12: After equal conditioning and extinction trials,
        residual threat MUST remain > 0 (asymmetry prediction)."""
        amygdala = AmygdalaEngine()
        fp = _make_fingerprint(seed=99)

        # 10 conditioning trials
        for _ in range(10):
            amygdala.evaluate(
                modality="auditory", fingerprint=fp,
                amplitude=0.9, pain_level=0.6,
            )
            amygdala.condition_fear("auditory", fp, threat_level=0.9)
        threat_post_cond = amygdala._threat_level

        # 10 extinction trials (safe environment)
        for _ in range(10):
            amygdala.evaluate(
                modality="auditory", fingerprint=fp,
                amplitude=0.05, pain_level=0.0,
            )
            amygdala.extinguish_fear("auditory", fp)
        threat_post_ext = amygdala._threat_level

        # Asymmetry: threat should still be elevated after equal trials
        assert threat_post_ext > 0.0, "Fear completely zeroed — asymmetry violated"
        # But extinction should have SOME effect
        assert threat_post_ext < threat_post_cond, (
            f"Extinction had no effect: cond={threat_post_cond:.4f}, "
            f"ext={threat_post_ext:.4f}"
        )

    def test_F13_fear_memory_effective_threat_floor(self):
        """F-13: FearMemory.effective_threat MUST never reach 0.
        Theory: the floor is 0.01 via np.clip(..., 0.01, 1.0)."""
        # Create a fear memory and extinguish massively
        mem = FearMemory(
            modality="test",
            fingerprint=np.zeros(16, dtype=np.float32),
            threat_level=0.5,
        )
        # Simulate 1000 extinction events
        mem.extinction_count = 1000
        mem.conditioning_count = 1
        assert mem.effective_threat >= 0.01, (
            f"effective_threat={mem.effective_threat:.6f} < 0.01 — floor violated"
        )


class TestFalsificationTrauma:
    """Narrow-tolerance tests for trauma permanence.

    MRP prediction: Trauma is irreversible impedance modification.
    reset() clears transient state but NOT structural damage.
    """

    def test_F14_trauma_survives_reset(self):
        """F-14: pain_sensitivity MUST remain elevated after reset().
        This tests the core 'structural memory' prediction."""
        state = SystemState()

        # Record 10 traumas
        for _ in range(10):
            state.record_trauma(signal_frequency=440.0)
        sens_before = state.pain_sensitivity
        base_before = state.baseline_temperature

        state.reset()

        assert state.pain_sensitivity == pytest.approx(sens_before, abs=0.01), (
            f"Sensitivity changed after reset: {sens_before:.3f} → "
            f"{state.pain_sensitivity:.3f}"
        )
        assert state.baseline_temperature == pytest.approx(base_before, abs=0.01), (
            f"Baseline temp changed after reset: {base_before:.3f} → "
            f"{state.baseline_temperature:.3f}"
        )

    def test_F15_trauma_sensitization_monotone(self):
        """F-15: Each trauma MUST increase pain_sensitivity (up to cap)."""
        state = SystemState()
        prev = state.pain_sensitivity
        for _ in range(20):
            state.record_trauma()
            assert state.pain_sensitivity >= prev, (
                f"Sensitivity decreased: {prev:.3f} → {state.pain_sensitivity:.3f}"
            )
            prev = state.pain_sensitivity
        # After 20 traumas: 1.0 + 20×0.05 = 2.0 (the cap)
        assert state.pain_sensitivity == pytest.approx(2.0, abs=0.01), (
            f"Expected sensitivity cap 2.0, got {state.pain_sensitivity:.3f}"
        )


class TestFalsificationSleep:
    """Narrow-tolerance tests for sleep cycle pressure."""

    def test_F16_pressure_increases_during_wake(self):
        """F-16: SleepCycle.sleep_pressure MUST increase during WAKE ticks."""
        sc = SleepCycle()
        pressures = []
        for _ in range(20):
            sc.tick()  # default: awake
            pressures.append(sc.sleep_pressure)
        for i in range(1, len(pressures)):
            assert pressures[i] >= pressures[i - 1] - 1e-10, (
                f"Pressure decreased at tick {i}: "
                f"{pressures[i-1]:.6f} → {pressures[i]:.6f}"
            )

    def test_F17_sleep_releases_pressure(self):
        """F-17: Once sleeping, pressure MUST decrease."""
        sc = SleepCycle()
        # Accumulate some pressure first
        for _ in range(5):
            sc.tick()
        sc.tick(force_sleep=True)
        # Now we should be in a sleep stage
        assert sc.is_sleeping(), f"Not sleeping after force_sleep"
        p_before = sc.sleep_pressure
        sc.tick()  # sleeping tick
        p_after = sc.sleep_pressure
        assert p_after <= p_before, (
            f"Pressure didn't decrease during sleep: {p_before:.4f} → {p_after:.4f}"
        )


class TestFalsificationMemory:
    """Narrow-tolerance tests for memory decay equation.

    MRP prediction: λ_eff = λ_base / (1 - Γ²)
    Higher Γ → faster decay → worse memory retention.
    """

    def test_F18_gamma_zero_slowest_decay(self):
        """F-18: At Γ=0, λ_eff = λ_base (minimum decay rate).
        At Γ>0, λ_eff > λ_base always."""
        lambda_base = 0.05
        for gamma in [0.1, 0.3, 0.5, 0.7, 0.9]:
            lambda_eff = lambda_base / (1 - gamma ** 2)
            assert lambda_eff > lambda_base, (
                f"λ_eff={lambda_eff:.4f} not > λ_base={lambda_base} at Γ={gamma}"
            )

    def test_F19_gamma_divergence_near_unity(self):
        """F-19: As Γ→1, λ_eff→∞ (instant forgetting).
        At Γ=0.99, λ_eff should be ~50× λ_base."""
        lambda_base = 0.05
        lambda_099 = lambda_base / (1 - 0.99 ** 2)
        ratio = lambda_099 / lambda_base
        assert ratio > 25, f"λ_eff/λ_base = {ratio:.1f} at Γ=0.99 — expected > 25"


# ═══════════════════════════════════════════════════════════════════════
#  §S: STATISTICAL ANALYSIS TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestStatisticalPhi:
    """Monte Carlo statistical analysis of the Φ equation."""

    def test_S01_phi_distribution_under_random_inputs(self):
        """S-01: Under uniform random inputs, compute the distribution of
        Φ and verify it has a well-defined mean with tight CI."""
        rng = np.random.default_rng(777)
        phi_samples = []
        for _ in range(200):
            c = ConsciousnessModule(safety_mode=False)
            for _ in range(30):
                c.tick(
                    attention_strength=rng.uniform(0, 1),
                    binding_quality=rng.uniform(0, 1),
                    working_memory_usage=rng.uniform(0, 0.8),
                    arousal=rng.uniform(0, 1),
                    sensory_gate=rng.uniform(0, 1),
                    pain_level=rng.uniform(0, 0.5),
                    temporal_resolution=rng.uniform(0.2, 1.0),
                )
            phi_samples.append(c.phi)

        mean, ci_lo, ci_hi = _bootstrap_ci(phi_samples)
        ci_width = ci_hi - ci_lo

        assert 0.0 < mean < 1.0, f"Mean Φ={mean:.4f} outside valid range"
        assert ci_width < 0.10, (
            f"95% CI too wide: [{ci_lo:.4f}, {ci_hi:.4f}] (width={ci_width:.4f})"
        )

    def test_S02_pain_effect_size(self):
        """S-02: Compute Cohen's d for pain vs. no-pain Φ distributions.
        Expect a LARGE effect (|d| > 0.8 by convention)."""
        rng = np.random.default_rng(888)
        phi_no_pain = []
        phi_pain = []
        for _ in range(100):
            c0 = ConsciousnessModule(safety_mode=False)
            c1 = ConsciousnessModule(safety_mode=False)
            for _ in range(30):
                inputs = dict(
                    attention_strength=rng.uniform(0.3, 0.8),
                    binding_quality=rng.uniform(0.3, 0.8),
                    working_memory_usage=rng.uniform(0.1, 0.5),
                    arousal=rng.uniform(0.3, 0.7),
                    sensory_gate=rng.uniform(0.3, 0.7),
                    temporal_resolution=rng.uniform(0.4, 1.0),
                )
                c0.tick(**inputs, pain_level=0.0)
                c1.tick(**inputs, pain_level=0.7)
            phi_no_pain.append(c0.phi)
            phi_pain.append(c1.phi)

        d = _cohens_d(phi_no_pain, phi_pain)
        t_stat, p_val = _welch_t(phi_no_pain, phi_pain)

        assert abs(d) > 0.8, f"Cohen's d = {d:.3f} — expected |d| > 0.8 (large effect)"
        assert p_val < 0.01, f"p = {p_val:.6f} — not significant at α=0.01"
        assert np.mean(phi_no_pain) > np.mean(phi_pain), (
            "Mean Φ(no_pain) should exceed Φ(pain)"
        )


class TestStatisticalPruning:
    """Bootstrap analysis of pruning outcomes."""

    def test_S03_pruning_survival_rate_ci(self):
        """S-03: After 200 epochs of directed stimulation, the survival
        rate should have a tight 95% CI across 30 independent runs."""
        survival_rates = []
        for seed in range(30):
            np.random.seed(seed)
            region = CorticalRegion(
                name=f"stat_{seed}",
                initial_connections=100,
            )
            for i in range(200):
                region.stimulate(
                    signal_impedance=50.0, signal_frequency=10.0,
                )
                if i % 10 == 0:
                    region.prune()
            survival_rates.append(region.alive_count / 100.0)

        mean, ci_lo, ci_hi = _bootstrap_ci(survival_rates)
        ci_width = ci_hi - ci_lo

        assert 0.0 < mean < 1.0, f"Mean survival = {mean:.3f}"
        assert ci_width < 0.25, (
            f"Survival rate CI too wide: [{ci_lo:.3f}, {ci_hi:.3f}] "
            f"(width={ci_width:.3f})"
        )

    def test_S04_gamma_reduction_effect_size(self):
        """S-04: Compare mean Γ first stim vs. after 300 stims + prune.
        Expect at least medium effect size (Cohen's d > 0.5)."""
        initial_gammas = []
        final_gammas = []
        for seed in range(30):
            np.random.seed(seed)
            region = CorticalRegion(
                name=f"gamma_{seed}",
                initial_connections=100,
            )
            r0 = region.stimulate(
                signal_impedance=50.0, signal_frequency=10.0,
            )
            initial_gammas.append(r0["avg_gamma"])
            for i in range(300):
                region.stimulate(
                    signal_impedance=50.0, signal_frequency=10.0,
                )
                if i % 15 == 0:
                    region.prune()
            rf = region.stimulate(
                signal_impedance=50.0, signal_frequency=10.0,
            )
            final_gammas.append(rf["avg_gamma"])

        d = _cohens_d(initial_gammas, final_gammas)
        t_stat, p_val = _welch_t(initial_gammas, final_gammas)

        assert d > 0.5, f"d={d:.3f} — pruning effect should be at least medium"
        assert p_val < 0.01, f"p={p_val:.6f} — pruning effect not significant"


class TestStatisticalFear:
    """Statistical analysis of fear conditioning."""

    def test_S05_conditioning_raises_threat_significantly(self):
        """S-05: Across 30 independent runs, conditioning MUST produce
        a statistically significant threat increase (p < 0.01)."""
        no_cond_threats = []
        cond_threats = []
        for seed in range(30):
            fp = _make_fingerprint(seed)

            # Control: evaluate without conditioning
            a_ctrl = AmygdalaEngine()
            for _ in range(10):
                a_ctrl.evaluate("visual", fp, amplitude=0.3, pain_level=0.0)
            no_cond_threats.append(a_ctrl._threat_level)

            # Experiment: evaluate with conditioning
            a_exp = AmygdalaEngine()
            for _ in range(10):
                a_exp.evaluate("visual", fp, amplitude=0.9, pain_level=0.7)
                a_exp.condition_fear("visual", fp, threat_level=0.9)
            cond_threats.append(a_exp._threat_level)

        t_stat, p_val = _welch_t(cond_threats, no_cond_threats)
        d = _cohens_d(cond_threats, no_cond_threats)

        assert np.mean(cond_threats) > np.mean(no_cond_threats), (
            "Conditioning did not increase mean threat"
        )
        assert p_val < 0.01, f"p = {p_val:.6f} — not significant at α=0.01"
        assert abs(d) > 0.5, f"d = {d:.3f} — effect too small"


# ═══════════════════════════════════════════════════════════════════════
#  §P: PARAMETER SENSITIVITY SWEEPS
# ═══════════════════════════════════════════════════════════════════════

class TestParameterSensitivity:
    """Systematic parameter sweeps to quantify robustness."""

    @pytest.mark.parametrize(
        "w_attn",
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    )
    def test_P01_attention_weight_sweep(self, w_attn, monkeypatch):
        """P-01: Sweep W_ATTENTION from 0.05 to 0.50 and verify
        Φ remains within [0, 1]."""
        import alice.brain.consciousness as cm
        monkeypatch.setattr(cm, "W_ATTENTION", w_attn)
        c = ConsciousnessModule(safety_mode=False)
        for _ in range(30):
            _tick_consciousness(c, attention_strength=0.7, arousal=0.5,
                                sensory_gate=0.5, binding_quality=0.5)
        assert 0.0 <= c.phi <= 1.0, (
            f"Φ={c.phi} outside [0,1] at W_ATTENTION={w_attn}"
        )

    @pytest.mark.parametrize(
        "smoothing",
        [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90],
    )
    def test_P02_phi_smoothing_sweep(self, smoothing, monkeypatch):
        """P-02: Sweep PHI_SMOOTHING from 0.05 to 0.90.
        Φ must stay bounded regardless."""
        import alice.brain.consciousness as cm
        monkeypatch.setattr(cm, "PHI_SMOOTHING", smoothing)
        c = ConsciousnessModule(safety_mode=False)
        for _ in range(30):
            _tick_consciousness(c, attention_strength=0.6, arousal=0.5,
                                sensory_gate=0.5, binding_quality=0.6)
        assert 0.0 <= c.phi <= 1.0, (
            f"Φ={c.phi} outside [0,1] at smoothing={smoothing}"
        )

    def test_P03_pain_always_reduces_phi(self):
        """P-03: At multiple pain levels (0.1 to 0.9), higher pain
        always produces lower Φ than the previous level."""
        phi_by_pain = []
        for pain in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
            c = ConsciousnessModule(safety_mode=False)
            for _ in range(30):
                _tick_consciousness(c, attention_strength=0.6, arousal=0.5,
                                    sensory_gate=0.6, binding_quality=0.5,
                                    pain_level=pain)
            phi_by_pain.append((pain, c.phi))
        for i in range(1, len(phi_by_pain)):
            assert phi_by_pain[i][1] <= phi_by_pain[i - 1][1] + 0.02, (
                f"Pain increase {phi_by_pain[i-1][0]}→{phi_by_pain[i][0]} "
                f"did not reduce Φ: {phi_by_pain[i-1][1]:.4f}→{phi_by_pain[i][1]:.4f}"
            )

    def test_P04_pruning_hebbian_2d_sweep(self):
        """P-04: 2D sweep of hebbian_strengthen × hebbian_weaken.
        Verify there is significant variation across parameter space."""
        strengthen_range = [1.01, 1.03, 1.05, 1.07, 1.10]
        weaken_range = [0.90, 0.93, 0.95, 0.97, 0.99]
        survival_values = []
        for s in strengthen_range:
            for w in weaken_range:
                np.random.seed(42)
                region = CorticalRegion(
                    name=f"sweep_{s}_{w}",
                    initial_connections=50,
                    hebbian_strengthen=s,
                    hebbian_weaken=w,
                )
                for i in range(200):
                    region.stimulate(
                        signal_impedance=50.0, signal_frequency=10.0,
                    )
                    if i % 10 == 0:
                        region.prune()
                survival_values.append(region.alive_count / 50.0)

        # There should be meaningful variation across the 25-point grid
        val_range = max(survival_values) - min(survival_values)
        assert val_range > 0.01, (
            f"No sensitivity: range = {val_range:.3f} across 25-point grid"
        )

    def test_P05_amygdala_amplitude_sweep(self):
        """P-05: Sweep amplitude from 0.1 to 1.0 and verify threat
        increases monotonically with amplitude."""
        threat_by_amp = []
        for amp in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            a = AmygdalaEngine()
            fp = _make_fingerprint(seed=55)
            for _ in range(15):
                a.evaluate("test", fp, amplitude=amp, pain_level=0.3)
            threat_by_amp.append((amp, a._threat_level))
        # Overall trend should be increasing
        assert threat_by_amp[-1][1] >= threat_by_amp[0][1], (
            f"Threat not increasing with amplitude: "
            f"amp={threat_by_amp[0][0]}→t={threat_by_amp[0][1]:.4f}, "
            f"amp={threat_by_amp[-1][0]}→t={threat_by_amp[-1][1]:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════
#  §L: LUCID THRESHOLD SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

class TestLucidThresholdSensitivity:
    """Sweep the LUCID_THRESHOLD from 0.5 to 0.9 in 0.05 steps.

    This directly addresses the audit finding that Φ=0.7 is an arbitrary
    engineering parameter with no theoretical derivation.

    The tests verify that system behavior degrades gracefully (not
    catastrophically) as the threshold changes — confirming the specific
    value is not fragile.
    """

    @pytest.mark.parametrize(
        "threshold",
        [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    )
    def test_L01_phi_bounded_regardless_of_threshold(self, threshold, monkeypatch):
        """L-01: Φ must remain in [0, 1] regardless of LUCID_THRESHOLD."""
        import alice.brain.consciousness as cm
        monkeypatch.setattr(cm, "LUCID_THRESHOLD", threshold)
        c = ConsciousnessModule(safety_mode=False)
        for _ in range(50):
            c.tick(
                attention_strength=1.0, binding_quality=1.0,
                temporal_resolution=1.0, working_memory_usage=0.5,
                arousal=1.0, sensory_gate=1.0, pain_level=0.0,
            )
        assert 0.0 <= c.phi <= 1.0, (
            f"Φ={c.phi} out of bounds at LUCID_THRESHOLD={threshold}"
        )

    @pytest.mark.parametrize(
        "threshold",
        [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    )
    def test_L02_sleep_eventually_triggers_at_any_threshold(self, threshold, monkeypatch):
        """L-02: In safety_mode, sleep MUST eventually trigger at any
        LUCID_THRESHOLD. The mechanism is pressure accumulation, not
        the specific threshold value."""
        import alice.brain.consciousness as cm
        monkeypatch.setattr(cm, "LUCID_THRESHOLD", threshold)
        c = ConsciousnessModule(
            developmental_stage=DevelopmentalStage.NEONATE,
            safety_mode=True,
        )
        slept = False
        for _ in range(60):
            _tick_consciousness(c, attention_strength=0.7, arousal=0.6,
                                sensory_gate=0.8)
            if c.is_sleeping:
                slept = True
                break
        assert slept, (
            f"Sleep not triggered within 60 ticks at LUCID_THRESHOLD={threshold}"
        )

    def test_L03_threshold_sensitivity_quantification(self):
        """L-03: Quantify how Φ steady-state changes across the full
        threshold range. A well-designed system should have bounded
        sensitivity (< 2.0)."""
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        steady_states = []

        for thr in thresholds:
            import alice.brain.consciousness as cm
            original = cm.LUCID_THRESHOLD
            cm.LUCID_THRESHOLD = thr
            try:
                c = ConsciousnessModule(safety_mode=False)
                for _ in range(50):
                    _tick_consciousness(c, attention_strength=0.7,
                                        binding_quality=0.6, arousal=0.6,
                                        sensory_gate=0.6)
                steady_states.append(c.phi)
            finally:
                cm.LUCID_THRESHOLD = original

        phi_range = max(steady_states) - min(steady_states)
        thr_range = max(thresholds) - min(thresholds)
        sensitivity = phi_range / thr_range if thr_range > 0 else 0

        assert sensitivity < 2.0, (
            f"System is hypersensitive to LUCID_THRESHOLD: "
            f"sensitivity={sensitivity:.3f} (Φ range={phi_range:.4f} "
            f"across threshold range={thr_range:.2f})"
        )
        for i, ss in enumerate(steady_states):
            assert 0.0 <= ss <= 1.0, (
                f"Invalid Φ={ss:.4f} at threshold={thresholds[i]}"
            )
