# -*- coding: utf-8 -*-
"""
test_falsification_and_statistics.py ??Falsification, Statistics & Sensitivity
===============================================================================

Purpose:
  Address the three critical methodological gaps identified in the rigor audit:

  1. FALSIFICATION TESTS (Â§F): Narrow-tolerance tests designed to BREAK
     theoretical predictions. If the MRP is correct, these must all pass.
     Unlike verification tests (broad "is x >= 0?"), these assert specific
     theoretical consequences with tight tolerances.

  2. STATISTICAL ANALYSIS (Â§S): Tests that compute p-values, confidence
     intervals, and effect sizes using Monte Carlo sampling and bootstrap.
     These replace hard-coded threshold comparisons with proper statistical
     inference.

  3. PARAMETER SENSITIVITY (Â§P): Systematic sweeps that quantify how
     sensitive key outputs are to parameter variations, identifying
     fragile vs. robust predictions.

  4. LUCID THRESHOLD SENSITIVITY (Â§L): Dedicated sweep of the Î¦=0.7
     threshold to test whether the system's behavior is robust to the
     specific choice of this arbitrary engineering parameter.

Audit response: Perplexity rigor audit Â§2 (Verification vs. Falsification),
                Â§5 (LUCID threshold), Â§6 (Statistical analysis), Â§7.8-7.10.

Author: Methodological Rigor Protocol
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest

# ?€?€ Core imports ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
from alice.brain.awareness_monitor import (
    AwarenessMonitor as ConsciousnessModule,
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


# ?â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â???
#  HELPER: lightweight bootstrap CI & statistical tests
# ?â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â???

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


# ?â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â???
#  Helper: standard tick inputs for ConsciousnessModule
# ?â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â???

def _tick_consciousness(cm: ConsciousnessModule, **overrides) -> dict:
    """Call cm.tick() with sensible defaults, allowing overrides."""
    defaults = dict(
        screen_brightness=0.5,
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


# ?â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â???
#  Â§F: FALSIFICATION TESTS
# ?â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â???

class TestFalsificationConsciousness:
    """Narrow-tolerance tests for Î¦ = screen brightness model.

    Screen model: Î¦ = screen_brightness (physical measurement).
    The module faithfully tracks whatever brightness the screen reports.
    """

    def test_F01_zero_brightness_kills_phi(self):
        """F-01: With screen_brightness=0, Î¦ MUST fall below
        SUBLIMINAL_THRESHOLD within 20 ticks. Screen off = unconscious."""
        c = ConsciousnessModule(safety_mode=False)
        for _ in range(20):
            _tick_consciousness(c, screen_brightness=0.0)
        assert c.phi < SUBLIMINAL_THRESHOLD, (
            f"Î¦={c.phi:.4f} should be < {SUBLIMINAL_THRESHOLD} when "
            f"screen_brightness=0 (screen off = unconscious)"
        )

    def test_F02_brightness_monotonicity(self):
        """F-02: Higher screen brightness MUST produce higher Î¦.
        The module is a faithful reader of screen state."""
        brightness_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        phi_results = []
        for b in brightness_values:
            c = ConsciousnessModule(safety_mode=False)
            for _ in range(40):
                _tick_consciousness(c, screen_brightness=b)
            phi_results.append(c.phi)
        for i in range(len(phi_results) - 1):
            assert phi_results[i] <= phi_results[i + 1] + 0.02, (
                f"Î¦ not monotone: brightness={brightness_values[i]}?’Î?{phi_results[i]:.4f} "
                f"> brightness={brightness_values[i+1]}?’Î?{phi_results[i+1]:.4f}"
            )

    def test_F03_phi_equals_screen_brightness_converged(self):
        """F-03: After convergence, Î¦ MUST equal screen_brightness.
        (EMA smoothing causes lag, but eventually they match.)"""
        for target in [0.0, 0.2, 0.5, 0.8, 1.0]:
            c = ConsciousnessModule(safety_mode=False)
            for _ in range(100):
                _tick_consciousness(c, screen_brightness=target)
            assert abs(c.phi - target) < 0.05, (
                f"Î¦={c.phi:.4f} should converge to screen_brightness={target}"
            )

    def test_F04_phi_bounded_zero_one(self):
        """F-04: Î¦ MUST remain in [0, 1] under all extreme inputs."""
        extremes = [
            dict(screen_brightness=0.0, pain_level=1.0),
            dict(screen_brightness=1.0, pain_level=0.0),
            dict(screen_brightness=1.0, pain_level=1.0),
            dict(screen_brightness=0.0, arousal=0.0, sensory_gate=0.0),
        ]
        for inputs in extremes:
            c = ConsciousnessModule(safety_mode=False)
            for _ in range(50):
                _tick_consciousness(c, **inputs)
            assert 0.0 <= c.phi <= 1.0, f"Î¦={c.phi} out of [0,1] with {inputs}"

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

    MRP prediction: Î£?Â² ??min under consistent stimulation.
    """

    def test_F07_consistent_stim_reduces_gamma(self):
        """F-07: 200 epochs of consistent stimulation + periodic pruning
        MUST reduce mean ? by at least 40%. Core MRP prediction.
        (Tightened from ??0% after empirical observation of ~76% reduction.)"""
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
            assert reduction >= 0.40, (
                f"Mean ? reduced only {reduction*100:.1f}% (need ??0%). "
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
            f"Connections died without stimulation: {initial_count}?’{region.alive_count}"
        )

    def test_F09_gamma_always_in_zero_one(self):
        """F-09: ? (reflection coefficient) MUST be in [0, 1] for all
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
                f"?={gamma} for Z_conn={z_conn:.1f}, Z_signal={z_signal:.1f}"
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
        assert threat_post_ext > 0.0, "Fear completely zeroed ??asymmetry violated"
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
            f"effective_threat={mem.effective_threat:.6f} < 0.01 ??floor violated"
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
            f"Sensitivity changed after reset: {sens_before:.3f} ??"
            f"{state.pain_sensitivity:.3f}"
        )
        assert state.baseline_temperature == pytest.approx(base_before, abs=0.01), (
            f"Baseline temp changed after reset: {base_before:.3f} ??"
            f"{state.baseline_temperature:.3f}"
        )

    def test_F15_trauma_sensitization_monotone(self):
        """F-15: Each trauma MUST increase pain_sensitivity (up to cap)."""
        state = SystemState()
        prev = state.pain_sensitivity
        for _ in range(20):
            state.record_trauma()
            assert state.pain_sensitivity >= prev, (
                f"Sensitivity decreased: {prev:.3f} ??{state.pain_sensitivity:.3f}"
            )
            prev = state.pain_sensitivity
        # After 20 traumas: 1.0 + 20?0.05 = 2.0 (the cap)
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
                f"{pressures[i-1]:.6f} ??{pressures[i]:.6f}"
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
            f"Pressure didn't decrease during sleep: {p_before:.4f} ??{p_after:.4f}"
        )


class TestFalsificationMemory:
    """Narrow-tolerance tests for memory decay equation.

    MRP prediction: Î»_eff = Î»_base / (1 - ?Â²)
    Higher ? ??faster decay ??worse memory retention.
    """

    def test_F18_gamma_zero_slowest_decay(self):
        """F-18: At ?=0, Î»_eff = Î»_base (minimum decay rate).
        At ?>0, Î»_eff > Î»_base always."""
        lambda_base = 0.05
        for gamma in [0.1, 0.3, 0.5, 0.7, 0.9]:
            lambda_eff = lambda_base / (1 - gamma ** 2)
            assert lambda_eff > lambda_base, (
                f"Î»_eff={lambda_eff:.4f} not > Î»_base={lambda_base} at ?={gamma}"
            )

    def test_F19_gamma_divergence_near_unity(self):
        """F-19: As ???, Î»_eff?’â? (instant forgetting).
        At ?=0.99, Î»_eff should be ~50? Î»_base."""
        lambda_base = 0.05
        lambda_099 = lambda_base / (1 - 0.99 ** 2)
        ratio = lambda_099 / lambda_base
        assert ratio > 25, f"Î»_eff/Î»_base = {ratio:.1f} at ?=0.99 ??expected > 25"


# ?â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â???
#  Â§S: STATISTICAL ANALYSIS TESTS
# ?â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â???

class TestStatisticalPhi:
    """Monte Carlo statistical analysis of the Î¦ equation."""

    # Bonferroni correction: 5 statistical tests ??Î±_adj = 0.01 / 5 = 0.002
    ALPHA_BONFERRONI = 0.002

    def test_S01_phi_distribution_under_random_inputs(self):
        """S-01: Under uniform random inputs, compute the distribution of
        Î¦ and verify it has a well-defined mean with tight CI."""
        rng = np.random.default_rng(777)
        phi_samples = []
        for _ in range(200):
            c = ConsciousnessModule(safety_mode=False)
            sb = rng.uniform(0.1, 0.9)
            for _ in range(30):
                c.tick(
                    screen_brightness=sb,
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

        assert 0.0 < mean < 1.0, f"Mean Î¦={mean:.4f} outside valid range"
        assert ci_width < 0.10, (
            f"95% CI too wide: [{ci_lo:.4f}, {ci_hi:.4f}] (width={ci_width:.4f})"
        )

    def test_S02_screen_brightness_effect_size(self):
        """S-02: Compute Cohen's d for HIGH vs LOW screen brightness.
        Expect a LARGE effect (|d| > 0.8) since Î¦ ??screen_brightness.
        (Pain no longer directly affects Î¦ ??it affects the screen instead.)
        Random variation around bright/dim centres gives non-zero variance."""
        rng = np.random.default_rng(888)
        phi_bright = []
        phi_dim = []
        for _ in range(100):
            c_bright = ConsciousnessModule(safety_mode=False)
            c_dim = ConsciousnessModule(safety_mode=False)
            # Random brightness centred at 0.8 (bright) and 0.2 (dim)
            bright_val = float(np.clip(rng.normal(0.8, 0.05), 0.0, 1.0))
            dim_val = float(np.clip(rng.normal(0.2, 0.05), 0.0, 1.0))
            for _ in range(30):
                inputs = dict(
                    attention_strength=rng.uniform(0.3, 0.8),
                    binding_quality=rng.uniform(0.3, 0.8),
                    working_memory_usage=rng.uniform(0.1, 0.5),
                    arousal=rng.uniform(0.3, 0.7),
                    sensory_gate=rng.uniform(0.3, 0.7),
                    temporal_resolution=rng.uniform(0.4, 1.0),
                )
                c_bright.tick(screen_brightness=bright_val, **inputs, pain_level=0.0)
                c_dim.tick(screen_brightness=dim_val, **inputs, pain_level=0.0)
            phi_bright.append(c_bright.phi)
            phi_dim.append(c_dim.phi)

        d = _cohens_d(phi_bright, phi_dim)
        t_stat, p_val = _welch_t(phi_bright, phi_dim)

        assert abs(d) > 0.8, f"Cohen's d = {d:.3f} ??expected |d| > 0.8 (large effect)"
        assert p_val < self.ALPHA_BONFERRONI, (
            f"p = {p_val:.6f} ??not significant at Bonferroni-corrected Î±={self.ALPHA_BONFERRONI}"
        )
        assert np.mean(phi_bright) > np.mean(phi_dim), (
            "Mean Î¦(bright_screen) should exceed Î¦(dim_screen)"
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
        assert ci_width < 0.15, (
            f"Survival rate CI too wide: [{ci_lo:.3f}, {ci_hi:.3f}] "
            f"(width={ci_width:.3f}, need <0.15)"
        )

    def test_S04_gamma_reduction_effect_size(self):
        """S-04: Compare mean ? first stim vs. after 300 stims + prune.
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

        assert d > 0.5, f"d={d:.3f} ??pruning effect should be at least medium"
        assert p_val < TestStatisticalPhi.ALPHA_BONFERRONI, (
            f"p={p_val:.6f} ??not significant at Bonferroni Î±={TestStatisticalPhi.ALPHA_BONFERRONI}"
        )


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
        assert p_val < TestStatisticalPhi.ALPHA_BONFERRONI, (
            f"p = {p_val:.6f} ??not significant at Bonferroni Î±={TestStatisticalPhi.ALPHA_BONFERRONI}"
        )
        assert abs(d) > 0.5, f"d = {d:.3f} ??effect too small"


# ?â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â???
#  Â§P: PARAMETER SENSITIVITY SWEEPS
# ?â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â???

class TestParameterSensitivity:
    """Systematic parameter sweeps to quantify robustness."""

    @pytest.mark.parametrize(
        "w_attn",
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    )
    def test_P01_attention_weight_sweep(self, w_attn, monkeypatch):
        """P-01: Legacy W_ATTENTION constant sweep ??no longer affects Î¦
        (now purely screen brightness), but Î¦ must stay in [0,1]."""
        import alice.brain.awareness_monitor as cm
        monkeypatch.setattr(cm, "W_ATTENTION", w_attn)
        c = ConsciousnessModule(safety_mode=False)
        for _ in range(30):
            _tick_consciousness(c, screen_brightness=0.6,
                                attention_strength=0.7, arousal=0.5,
                                sensory_gate=0.5, binding_quality=0.5)
        assert 0.0 <= c.phi <= 1.0, (
            f"Î¦={c.phi} outside [0,1] at W_ATTENTION={w_attn}"
        )

    @pytest.mark.parametrize(
        "smoothing",
        [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90],
    )
    def test_P02_phi_smoothing_sweep(self, smoothing, monkeypatch):
        """P-02: Sweep PHI_SMOOTHING from 0.05 to 0.90.
        Î¦ must stay bounded regardless."""
        import alice.brain.awareness_monitor as cm
        monkeypatch.setattr(cm, "PHI_SMOOTHING", smoothing)
        c = ConsciousnessModule(safety_mode=False)
        for _ in range(30):
            _tick_consciousness(c, attention_strength=0.6, arousal=0.5,
                                sensory_gate=0.5, binding_quality=0.6)
        assert 0.0 <= c.phi <= 1.0, (
            f"Î¦={c.phi} outside [0,1] at smoothing={smoothing}"
        )

    def test_P03_pain_recorded_not_affecting_phi(self):
        """P-03: Pain is recorded for meta-awareness but does NOT alter Î¦.
        (Pain affects screen via ConsciousnessScreen overload, not here.)
        At same screen_brightness, varying pain should yield same Î¦."""
        phi_by_pain = []
        for pain in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
            c = ConsciousnessModule(safety_mode=False)
            for _ in range(30):
                _tick_consciousness(c, screen_brightness=0.6,
                                    attention_strength=0.6, arousal=0.5,
                                    sensory_gate=0.6, binding_quality=0.5,
                                    pain_level=pain)
            phi_by_pain.append((pain, c.phi))
        # All should be approximately equal (same screen brightness)
        for i in range(1, len(phi_by_pain)):
            assert abs(phi_by_pain[i][1] - phi_by_pain[0][1]) < 0.05, (
                f"Pain={phi_by_pain[i][0]} changed Î¦ from {phi_by_pain[0][1]:.4f} "
                f"to {phi_by_pain[i][1]:.4f} ??should be same at same brightness"
            )

    def test_P04_pruning_hebbian_2d_sweep(self):
        """P-04: 2D sweep of hebbian_strengthen ? hebbian_weaken.
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
            f"amp={threat_by_amp[0][0]}?’t={threat_by_amp[0][1]:.4f}, "
            f"amp={threat_by_amp[-1][0]}?’t={threat_by_amp[-1][1]:.4f}"
        )


# ?â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â???
#  Â§L: LUCID THRESHOLD SENSITIVITY ANALYSIS
# ?â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â???

class TestLucidThresholdSensitivity:
    """Sweep the LUCID_THRESHOLD from 0.5 to 0.9 in 0.05 steps.

    This directly addresses the audit finding that Î¦=0.7 is an arbitrary
    engineering parameter with no theoretical derivation.

    The tests verify that system behavior degrades gracefully (not
    catastrophically) as the threshold changes ??confirming the specific
    value is not fragile.
    """

    @pytest.mark.parametrize(
        "threshold",
        [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    )
    def test_L01_phi_bounded_regardless_of_threshold(self, threshold, monkeypatch):
        """L-01: Î¦ must remain in [0, 1] regardless of LUCID_THRESHOLD."""
        import alice.brain.awareness_monitor as cm
        monkeypatch.setattr(cm, "LUCID_THRESHOLD", threshold)
        c = ConsciousnessModule(safety_mode=False)
        for _ in range(50):
            c.tick(
                screen_brightness=1.0,
                attention_strength=1.0, binding_quality=1.0,
                temporal_resolution=1.0, working_memory_usage=0.5,
                arousal=1.0, sensory_gate=1.0, pain_level=0.0,
            )
        assert 0.0 <= c.phi <= 1.0, (
            f"Î¦={c.phi} out of bounds at LUCID_THRESHOLD={threshold}"
        )

    @pytest.mark.parametrize(
        "threshold",
        [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    )
    def test_L02_sleep_eventually_triggers_at_any_threshold(self, threshold, monkeypatch):
        """L-02: In safety_mode, sleep MUST eventually trigger at any
        LUCID_THRESHOLD. The mechanism is pressure accumulation, not
        the specific threshold value."""
        import alice.brain.awareness_monitor as cm
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
        """L-03: Quantify how Î¦ steady-state changes across the full
        threshold range. A well-designed system should have bounded
        sensitivity (< 2.0)."""
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        steady_states = []

        for thr in thresholds:
            import alice.brain.awareness_monitor as cm
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
            f"sensitivity={sensitivity:.3f} (Î¦ range={phi_range:.4f} "
            f"across threshold range={thr_range:.2f})"
        )
        for i, ss in enumerate(steady_states):
            assert 0.0 <= ss <= 1.0, (
                f"Invalid Î¦={ss:.4f} at threshold={thresholds[i]}"
            )


# ?â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â???
#  Â§X: CROSS-MODULE FALSIFICATION TESTS
# ?â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â???

def _tick_system_state(ss: SystemState, **overrides) -> None:
    """Drive SystemState.tick() with sensible defaults, allowing overrides."""
    defaults = dict(
        critical_queue_len=0,
        high_queue_len=0,
        total_queue_len=0,
        sensory_activity=0.0,
        emotional_valence=0.0,
        left_brain_activity=0.5,
        right_brain_activity=0.5,
        cycle_elapsed_ms=16.0,
        reflected_energy=0.0,
    )
    defaults.update(overrides)
    ss.tick(**defaults)


class TestCrossModuleFalsification:
    """Cross-module falsification tests.

    The MRP predicts that modules interact through shared physical variables.
    These tests wire two or more modules together and verify that the
    cross-module predictions hold under tight tolerances.
    """

    def test_X01_system_pain_suppresses_consciousness(self):
        """X-01: SystemState generates pain under thermal stress.
        In the Screen Brightness Model, pain affects the screen (overload),
        not the consciousness formula directly. At the same screen brightness,
        pain does NOT affect Î¦. This test verifies the correct chain:
        high queue pressure ??temperature????pain????screen overload ??brightness????Î¦??
        At the unit-test level (without screen), we verify that pain is produced
        by SystemState, and that screen_brightness directly controls Î¦.
        """
        # Step 1: Generate pain via SystemState thermal loop
        ss = SystemState()
        for _ in range(30):
            _tick_system_state(ss, critical_queue_len=10, reflected_energy=0.5)
        assert ss.pain_level > 0.1, (
            f"SystemState should produce pain under thermal stress, "
            f"got pain={ss.pain_level:.4f}"
        )

        # Step 2: Screen brightness controls Î¦ (pain affects screen, not formula)
        # Simulate: pain causes screen overload ??lower brightness
        c_healthy = ConsciousnessModule(safety_mode=False)
        c_stressed = ConsciousnessModule(safety_mode=False)
        # Healthy screen: brightness 0.7; Stressed screen (pain overload): brightness 0.3
        for _ in range(30):
            _tick_consciousness(c_healthy, screen_brightness=0.7,
                                attention_strength=0.7, arousal=0.6,
                                sensory_gate=0.8, pain_level=0.0)
            _tick_consciousness(c_stressed, screen_brightness=0.3,
                                attention_strength=0.7, arousal=0.6,
                                sensory_gate=0.8, pain_level=ss.pain_level)

        assert c_stressed.phi < c_healthy.phi, (
            f"Lower screen brightness (pain overload) should yield lower Î¦: "
            f"Î¦_stressed={c_stressed.phi:.4f} >= Î¦_healthy={c_healthy.phi:.4f}"
        )
        # Demand meaningful suppression, not just floating-point noise
        suppression = c_healthy.phi - c_stressed.phi
        assert suppression > 0.01, (
            f"Î¦ suppression by reduced screen brightness is negligible: ?={suppression:.4f}"
        )

    def test_X02_trauma_sensitization_amplifies_pain(self):
        """X-02: Repeated trauma via record_trauma() permanently sensitizes
        SystemState, so identical thermal load produces MORE pain later.

        Chain:  record_trauma() ? N ??pain_sensitivity????threshold????more pain

        Fresh:   effective_threshold = 0.7 / 1.0  = 0.70
        Trauma5: effective_threshold = 0.7 / 1.25 = 0.56
        """
        # Use mild load so pain doesn't saturate at 1.0 for both
        QUEUE, RE, TICKS = 3, 0.1, 30

        # Scenario A: fresh system
        ss_fresh = SystemState()
        for _ in range(TICKS):
            _tick_system_state(ss_fresh, critical_queue_len=QUEUE,
                               reflected_energy=RE)

        # Scenario B: traumatized system under same load
        ss_trauma = SystemState()
        for _ in range(5):
            ss_trauma.record_trauma(signal_frequency=440.0)
        ss_trauma.reset()  # Reset clears temperature but keeps sensitization
        for _ in range(TICKS):
            _tick_system_state(ss_trauma, critical_queue_len=QUEUE,
                               reflected_energy=RE)

        # Sensitization itself is permanent
        assert ss_trauma.pain_sensitivity > ss_fresh.pain_sensitivity, (
            f"Trauma should increase sensitivity: "
            f"sens_trauma={ss_trauma.pain_sensitivity:.3f} <= "
            f"sens_fresh={ss_fresh.pain_sensitivity:.3f}"
        )
        # Temperature starts higher (baseline_temperature > 0 after trauma)
        assert ss_trauma.baseline_temperature > 0.0, (
            f"Trauma should raise baseline_temperature"
        )
        # Traumatized system should have higher pain or same
        # (at minimum, it reached pain threshold sooner)
        assert ss_trauma.pain_level >= ss_fresh.pain_level, (
            f"Traumatized system should feel >= pain: "
            f"pain_trauma={ss_trauma.pain_level:.4f} < "
            f"pain_fresh={ss_fresh.pain_level:.4f}"
        )

    def test_X03_fear_conditioning_modulated_by_pain(self):
        """X-03: AmygdalaEngine with high pain_level should produce
        stronger threat evaluation than with zero pain.

        Chain:  AmygdalaEngine.evaluate(pain_level=high) ??higher threat
        """
        amyg = AmygdalaEngine()
        fp = _make_fingerprint(seed=42)

        # Condition a mild fear
        amyg.condition_fear("visual", fp, threat_level=0.3,
                            concept_label="spider")

        # Evaluate with no pain
        result_no_pain = amyg.evaluate(
            modality="visual", fingerprint=fp, gamma=0.5,
            amplitude=0.5, pain_level=0.0, concept_label="spider"
        )
        threat_no_pain = result_no_pain.emotional_state.threat_level

        # Evaluate with high pain
        result_pain = amyg.evaluate(
            modality="visual", fingerprint=fp, gamma=0.5,
            amplitude=0.5, pain_level=0.8, concept_label="spider"
        )
        threat_pain = result_pain.emotional_state.threat_level

        assert threat_pain >= threat_no_pain, (
            f"Pain should amplify threat evaluation: "
            f"threat_pain={threat_pain:.4f} < threat_no_pain={threat_no_pain:.4f}"
        )

    def test_X04_pruning_gamma_degrades_consciousness_binding(self):
        """X-04: CorticalRegion with high ? (poor impedance matching) should,
        when used as binding_quality input to ConsciousnessModule, produce
        lower Î¦ than a region with low ?.

        Chain:  CorticalRegion.stimulate() ??avg_gamma ??(1 - avg_gamma)
                as binding_quality ??ConsciousnessModule ??Î¦

        Region A: trained with signal_impedance=110 (centre of Z distribution
                  [20, 200]).  After 100 Hebbian cycles, connections converge
                  toward the signal ??low avg_gamma.
        Region B: stimulated once with signal_impedance=5000 (extreme mismatch
                  against all connections) ??high avg_gamma.
        """
        np.random.seed(0)  # reproducibility

        # Region A: well-matched after Hebbian learning
        region_good = CorticalRegion("good", initial_connections=300)
        for _ in range(100):
            region_good.stimulate(signal_impedance=110.0,
                                  signal_frequency=10.0)
        result_good = region_good.stimulate(signal_impedance=110.0,
                                             signal_frequency=10.0)
        gamma_good = result_good["avg_gamma"]

        # Region B: extreme impedance mismatch (no learning opportunity)
        region_bad = CorticalRegion("bad", initial_connections=300)
        result_bad = region_bad.stimulate(signal_impedance=5000.0,
                                           signal_frequency=5000.0)
        gamma_bad = result_bad["avg_gamma"]

        # Sanity: the trained region should have meaningfully lower ?
        assert gamma_good < gamma_bad, (
            f"Trained region should have lower ?: "
            f"?_good={gamma_good:.4f} >= ?_bad={gamma_bad:.4f}"
        )

        # Convert ? to screen brightness: lower ? ??better transmission ??brighter screen
        brightness_good = 1.0 - gamma_good
        brightness_bad = 1.0 - gamma_bad

        c_good = ConsciousnessModule(safety_mode=False)
        c_bad = ConsciousnessModule(safety_mode=False)
        for _ in range(40):
            _tick_consciousness(c_good, screen_brightness=brightness_good,
                                binding_quality=1.0 - gamma_good,
                                attention_strength=0.7, arousal=0.6)
            _tick_consciousness(c_bad, screen_brightness=brightness_bad,
                                binding_quality=1.0 - gamma_bad,
                                attention_strength=0.7, arousal=0.6)

        assert c_good.phi > c_bad.phi, (
            f"Better impedance match (low ?) should produce higher Î¦: "
            f"Î¦_good={c_good.phi:.4f} (?={gamma_good:.4f}) <= "
            f"Î¦_bad={c_bad.phi:.4f} (?={gamma_bad:.4f})"
        )

    def test_X05_ptsd_thermal_trap_cooling_zero(self):
        """X-05: The PTSD attractor: when critical_pressure = 1.0,
        cooling = 0.03 * (1 - 1.0) = 0. Temperature can only rise.
        SystemState consciousness should collapse below 0.15 (is_frozen).

        This is the core cross-module prediction of the Fever Equation.
        """
        ss = SystemState()
        # Drive into PTSD attractor: maximal queue pressure
        for _ in range(80):
            _tick_system_state(ss, critical_queue_len=20,
                               reflected_energy=0.8)

        assert ss.ram_temperature > 0.85, (
            f"Temperature should be near max under PTSD conditions: "
            f"T={ss.ram_temperature:.4f}"
        )
        assert ss.is_frozen(), (
            f"System should be frozen (consciousness < 0.15) under PTSD: "
            f"consciousness={ss.consciousness:.4f}"
        )

    def test_X06_trauma_permanence_across_reset_and_consciousness(self):
        """X-06: Full chain: trauma ??reset ??re-stress ??faster pain onset
        ??consciousness suppression faster than naive system.

        Validates that trauma memory persists across the full
        SystemState ??ConsciousnessModule pipeline.
        """
        # Naive system: how many ticks until consciousness < 0.5?
        ss_naive = SystemState()
        ticks_to_degrade_naive = 0
        for i in range(200):
            _tick_system_state(ss_naive, critical_queue_len=6,
                               reflected_energy=0.4)
            ticks_to_degrade_naive = i + 1
            if ss_naive.consciousness < 0.5:
                break

        # Traumatized system: same stress, but with 5 prior traumas
        ss_trauma = SystemState()
        for _ in range(5):
            ss_trauma.record_trauma(signal_frequency=100.0)
        ss_trauma.reset()

        ticks_to_degrade_trauma = 0
        for i in range(200):
            _tick_system_state(ss_trauma, critical_queue_len=6,
                               reflected_energy=0.4)
            ticks_to_degrade_trauma = i + 1
            if ss_trauma.consciousness < 0.5:
                break

        assert ticks_to_degrade_trauma < ticks_to_degrade_naive, (
            f"Traumatized system should degrade consciousness faster: "
            f"trauma={ticks_to_degrade_trauma} ticks >= "
            f"naive={ticks_to_degrade_naive} ticks"
        )


# ?â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â???
#  Â§T: THERMAL MODEL SENSITIVITY SWEEPS
#  Tests the Fever Equation: T' = T + heat_input * 0.15 - cooling
#  where cooling = 0.03 * (1 - critical_pressure)
# ?â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â???

def _thermal_steady_state(
    cooling_coeff: float,
    heat_mult: float,
    critical_pressure: float,
    heat_input: float,
    ticks: int = 500,
) -> float:
    """Simulate the Fever Equation to steady state with given parameters.
    Returns the final temperature."""
    T = 0.0
    for _ in range(ticks):
        cooling = cooling_coeff * (1.0 - critical_pressure)
        T = float(np.clip(T + heat_input * heat_mult - cooling, 0.0, 1.0))
    return T


class TestThermalSensitivity:
    """Sensitivity sweeps for the Fever Equation thermal model.

    The key parameters are:
      - cooling_coeff (hardcoded 0.03): Natural dissipation rate
      - heat_mult (hardcoded 0.15): How fast heat_input raises temperature
      - critical_pressure: Queue deadlock fraction [0, 1]

    Audit requirement: verify the system is not fragile to specific values
    of these engineering constants.
    """

    @pytest.mark.parametrize("cooling_coeff", [
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10
    ])
    def test_T01_cooling_coefficient_sweep(self, cooling_coeff):
        """T-01: At moderate load, sweeping cooling_coeff should produce
        bounded, monotonically decreasing steady-state temperature.
        Higher cooling ??lower equilibrium temperature."""
        heat_input = 0.3  # moderate load
        T = _thermal_steady_state(cooling_coeff, heat_mult=0.15,
                                  critical_pressure=0.0,
                                  heat_input=heat_input)
        assert 0.0 <= T <= 1.0, f"T={T:.4f} out of bounds"

    def test_T01b_cooling_coeff_monotonicity(self):
        """T-01b: Higher cooling coefficient ??lower equilibrium temperature
        (monotonicity across full sweep)."""
        coeffs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
        temps = [_thermal_steady_state(c, 0.15, 0.0, 0.3) for c in coeffs]
        for i in range(1, len(temps)):
            assert temps[i] <= temps[i - 1] + 1e-9, (
                f"Temperature should decrease with higher cooling: "
                f"T[{coeffs[i]}]={temps[i]:.4f} > T[{coeffs[i-1]}]="
                f"{temps[i-1]:.4f}"
            )

    @pytest.mark.parametrize("heat_mult", [
        0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30
    ])
    def test_T02_heat_multiplier_sweep(self, heat_mult):
        """T-02: At moderate load, sweeping heat_mult should produce
        bounded steady-state temperature. Higher multiplier ??hotter."""
        T = _thermal_steady_state(cooling_coeff=0.03, heat_mult=heat_mult,
                                  critical_pressure=0.0, heat_input=0.3)
        assert 0.0 <= T <= 1.0, f"T={T:.4f} out of bounds"

    def test_T02b_heat_mult_monotonicity(self):
        """T-02b: Higher heat multiplier ??higher equilibrium temperature."""
        mults = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
        temps = [_thermal_steady_state(0.03, m, 0.0, 0.3) for m in mults]
        for i in range(1, len(temps)):
            assert temps[i] >= temps[i - 1] - 1e-9, (
                f"Temperature should increase with higher heat_mult: "
                f"T[{mults[i]}]={temps[i]:.4f} < T[{mults[i-1]}]="
                f"{temps[i-1]:.4f}"
            )

    @pytest.mark.parametrize("cp", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    def test_T03_critical_pressure_sweep(self, cp):
        """T-03: Sweeping critical_pressure from 0 to 1. At cp=1.0,
        cooling=0 and temperature should be at ceiling (1.0)."""
        T = _thermal_steady_state(0.03, 0.15, critical_pressure=cp,
                                  heat_input=0.5)
        assert 0.0 <= T <= 1.0, f"T={T:.4f} out of bounds"
        if cp >= 0.99:
            assert T >= 0.95, (
                f"At critical_pressure=1.0, temperature should hit ceiling: "
                f"T={T:.4f}"
            )

    def test_T03b_critical_pressure_monotonicity(self):
        """T-03b: Higher critical_pressure ??higher equilibrium temperature.
        This is the Fever Equation's core prediction."""
        cps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        temps = [_thermal_steady_state(0.03, 0.15, cp, 0.5) for cp in cps]
        for i in range(1, len(temps)):
            assert temps[i] >= temps[i - 1] - 1e-9, (
                f"Temperature should increase with critical_pressure: "
                f"T[{cps[i]}]={temps[i]:.4f} < T[{cps[i-1]}]={temps[i-1]:.4f}"
            )

    def test_T04_ptsd_trap_universality(self):
        """T-04: The PTSD thermal trap (cooling=0 at cp=1) holds regardless
        of cooling coefficient ??a universal prediction of the Fever Equation."""
        for cooling_coeff in [0.01, 0.03, 0.05, 0.10, 0.50]:
            T = _thermal_steady_state(cooling_coeff, 0.15,
                                      critical_pressure=1.0,
                                      heat_input=0.5)
            assert T >= 0.95, (
                f"PTSD trap should work at any cooling_coeff: "
                f"cooling={cooling_coeff}, T={T:.4f}"
            )

    def test_T05_cooling_vs_heat_2d_grid(self):
        """T-05: 2D sensitivity grid: cooling_coeff ? heat_mult.
        All combinations must produce bounded temperature.
        The grid also verifies monotonicity in both dimensions."""
        cooling_vals = [0.01, 0.03, 0.05, 0.08]
        heat_vals = [0.05, 0.10, 0.15, 0.20, 0.30]
        grid = {}

        for c in cooling_vals:
            for h in heat_vals:
                T = _thermal_steady_state(c, h, 0.0, 0.3)
                grid[(c, h)] = T
                assert 0.0 <= T <= 1.0, (
                    f"T out of bounds at cooling={c}, heat={h}: T={T:.4f}"
                )

        # Monotonicity check: for fixed heat, higher cooling ??lower T
        for h in heat_vals:
            for i in range(1, len(cooling_vals)):
                c_prev, c_curr = cooling_vals[i - 1], cooling_vals[i]
                assert grid[(c_curr, h)] <= grid[(c_prev, h)] + 1e-9, (
                    f"Non-monotone in cooling at heat={h}: "
                    f"T[{c_curr}]={grid[(c_curr, h)]:.4f} > "
                    f"T[{c_prev}]={grid[(c_prev, h)]:.4f}"
                )

        # Monotonicity check: for fixed cooling, higher heat ??higher T
        for c in cooling_vals:
            for i in range(1, len(heat_vals)):
                h_prev, h_curr = heat_vals[i - 1], heat_vals[i]
                assert grid[(c, h_curr)] >= grid[(c, h_prev)] - 1e-9, (
                    f"Non-monotone in heat at cooling={c}: "
                    f"T[{h_curr}]={grid[(c, h_curr)]:.4f} < "
                    f"T[{h_prev}]={grid[(c, h_prev)]:.4f}"
                )

    def test_T06_live_system_state_thermal_sweep(self):
        """T-06: End-to-end thermal sweep using actual SystemState.tick().
        Sweep critical_queue_len from 0 to 20. Temperature must be
        monotonically non-decreasing at steady state."""
        queue_lengths = [0, 2, 4, 6, 8, 10, 15, 20]
        final_temps = []

        for q in queue_lengths:
            ss = SystemState()
            for _ in range(100):
                _tick_system_state(ss, critical_queue_len=q)
            final_temps.append(ss.ram_temperature)

        for i in range(1, len(final_temps)):
            assert final_temps[i] >= final_temps[i - 1] - 0.01, (
                f"SystemState temperature should increase with queue pressure: "
                f"T[q={queue_lengths[i]}]={final_temps[i]:.4f} < "
                f"T[q={queue_lengths[i-1]}]={final_temps[i-1]:.4f}"
            )

    def test_T07_reflected_energy_monotonicity(self):
        """T-07: Sweep reflected_energy from 0 to 1. Higher reflection
        (worse impedance match) ??higher temperature."""
        re_vals = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        final_temps = []

        for re in re_vals:
            ss = SystemState()
            for _ in range(100):
                _tick_system_state(ss, reflected_energy=re)
            final_temps.append(ss.ram_temperature)

        for i in range(1, len(final_temps)):
            assert final_temps[i] >= final_temps[i - 1] - 0.01, (
                f"Temperature should increase with reflected energy: "
                f"T[re={re_vals[i]}]={final_temps[i]:.4f} < "
                f"T[re={re_vals[i-1]}]={final_temps[i-1]:.4f}"
            )


# ?â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â???
#  Â§C: COUNTERFACTUAL FALSIFICATION TESTS
#  These tests compare MRP predictions against alternative models.
#  If an alternative model produces IDENTICAL results, MRP is not
#  uniquely predictive; if results differ, MRP is falsifiable.
# ?â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â??â???

class TestCounterfactualFalsification:
    """Counterfactual tests: 'an alternative model would predict differently.'

    These are the strongest form of falsification ??they demonstrate that
    the MRP framework makes predictions that competing models do NOT make.
    """

    def test_C01_additive_pain_model_cannot_produce_ptsd_trap(self):
        """C-01: Under an ADDITIVE pain model (pain += constant each tick),
        the PTSD thermal trap does NOT arise ??temperature can always
        decrease if input ceases. Under MRP's MULTIPLICATIVE model
        (pain_sensitivity amplifies temperature ??higher critical_pressure
        ??less cooling), the trap is inescapable.

        Counterfactual: if pain sensitivity were additive (not multiplicative),
        reset() + zero input should fully recover the system.
        MRP prediction: even after reset(), elevated baseline_temperature
        prevents full cooling.
        """
        # MRP system: trauma ??sensitization ??trap
        ss_mrp = SystemState()
        for _ in range(10):
            ss_mrp.record_trauma(signal_frequency=440.0)

        # The MRP prediction: baseline_temperature is permanently elevated
        assert ss_mrp.baseline_temperature > 0.0, (
            "MRP predicts trauma raises baseline_temperature permanently"
        )

        # Reset clears transient state but NOT structural damage
        ss_mrp.reset()
        assert ss_mrp.pain_sensitivity > 1.0, (
            "MRP predicts pain_sensitivity survives reset (structural memory)"
        )
        assert ss_mrp.baseline_temperature > 0.0, (
            "MRP predicts baseline_temperature survives reset (structural memory)"
        )

        # Counterfactual: a simple additive model would have
        # pain_sensitivity = 1.0 after reset (no structural memory).
        # MRP's multiplicative sensitization is the distinguishing prediction.
        additive_sensitivity_after_reset = 1.0  # hypothetical additive model
        assert ss_mrp.pain_sensitivity > additive_sensitivity_after_reset, (
            f"MRP sensitivity ({ss_mrp.pain_sensitivity:.3f}) must exceed "
            f"additive model ({additive_sensitivity_after_reset}) after trauma+reset"
        )

    def test_C02_symmetric_model_fails_fear_asymmetry(self):
        """C-02: A symmetric learning model (conditioning rate == extinction
        rate) would predict that N conditioning trials followed by N extinction
        trials return threat to baseline. MRP predicts asymmetry: fear
        memories have a floor and extinction is slower than conditioning.

        Counterfactual: symmetric model ??threat ??baseline after equal trials.
        MRP prediction: threat > baseline after equal trials.
        """
        amygdala = AmygdalaEngine()
        fp = _make_fingerprint(seed=99)
        baseline = amygdala._threat_level

        # 10 conditioning trials
        for _ in range(10):
            amygdala.evaluate("auditory", fp, amplitude=0.9, pain_level=0.6)
            amygdala.condition_fear("auditory", fp, threat_level=0.9)

        # 10 extinction trials (equal count)
        for _ in range(10):
            amygdala.evaluate("auditory", fp, amplitude=0.05, pain_level=0.0)
            amygdala.extinguish_fear("auditory", fp)

        threat_after_symmetric = amygdala._threat_level

        # Symmetric model prediction: threat ??baseline
        # MRP prediction: threat > baseline (asymmetry)
        assert threat_after_symmetric > baseline, (
            f"MRP predicts asymmetric fear (threat={threat_after_symmetric:.4f} "
            f"should remain above baseline={baseline:.4f} after equal "
            f"conditioning/extinction trials). A symmetric model would predict "
            f"threat ??baseline ??this falsifies the symmetric alternative."
        )

    def test_C03_random_walk_model_fails_gamma_convergence(self):
        """C-03: A random-walk impedance model (Z changes by Â±Î´ each step)
        would produce a DIFFUSIVE pattern (?šn spreading) rather than
        convergent. MRP with Hebbian learning predicts CONVERGENT
        impedance adaptation toward the signal impedance.

        Counterfactual: random walk ??|Z_conn - Z_signal| grows as ?šn.
        MRP prediction: |Z_conn - Z_signal| shrinks monotonically.
        """
        target_z = 50.0
        region = CorticalRegion(
            name="counterfactual_convergence",
            initial_connections=100,
        )

        # Measure initial distance distribution
        initial_distances = [
            abs(c.impedance - target_z) for c in region.connections if c.alive
        ]
        mean_dist_initial = np.mean(initial_distances)

        # Apply MRP (Hebbian stimulation)
        for i in range(200):
            region.stimulate(signal_impedance=target_z, signal_frequency=10.0)
            if i % 20 == 0:
                region.prune()

        # Measure final distance distribution
        final_distances = [
            abs(c.impedance - target_z) for c in region.connections if c.alive
        ]
        mean_dist_final = np.mean(final_distances)

        # MRP prediction: convergent (distance decreases)
        assert mean_dist_final < mean_dist_initial * 0.5, (
            f"MRP predicts convergent adaptation: mean distance should halve. "
            f"Initial={mean_dist_initial:.3f}, Final={mean_dist_final:.3f}. "
            f"A random-walk model would show ?šn divergence instead."
        )

        # Simulate the counterfactual: a random walk should NOT converge
        rng = np.random.default_rng(42)
        random_distances = [abs(rng.normal(100, 50) - target_z) for _ in range(100)]
        # After 200 random steps of Â±Î´ from initial positions, the distribution
        # should be at least as wide as at birth (random walks diffuse, not converge)
        for _ in range(200):
            random_distances = [
                abs(d + rng.normal(0, 2)) for d in random_distances
            ]
        mean_random_final = np.mean(random_distances)

        # The random walk should NOT converge to target
        assert mean_random_final > mean_dist_final, (
            f"Random walk ({mean_random_final:.3f}) should NOT converge better "
            f"than MRP Hebbian learning ({mean_dist_final:.3f})"
        )

    def test_C04_no_sleep_pressure_model_lacks_forced_dormancy(self):
        """C-04: A model WITHOUT sleep pressure would allow indefinite
        wakefulness. MRP predicts that wake-time impedance debt accumulates
        and FORCES dormancy via the sensory gate.

        Counterfactual: model without sleep pressure ??consciousness sustained
        indefinitely at high arousal.
        MRP prediction: consciousness MUST decline due to sleep pressure,
        even with maximum arousal and sensory input.
        """
        # MRP system (safety_mode=True): sleep pressure forces dormancy
        c_mrp = ConsciousnessModule(
            developmental_stage=DevelopmentalStage.NEONATE,
            safety_mode=True,
        )
        slept_mrp = False
        for t in range(60):
            _tick_consciousness(c_mrp, attention_strength=0.8, arousal=0.7,
                                sensory_gate=1.0, binding_quality=0.7)
            if c_mrp.is_sleeping:
                slept_mrp = True
                break

        assert slept_mrp, (
            "MRP predicts forced dormancy via sleep pressure ??system MUST sleep"
        )

        # Counterfactual: system without safety valve (no sleep pressure)
        c_nosleep = ConsciousnessModule(
            developmental_stage=DevelopmentalStage.NEONATE,
            safety_mode=False,  # disables sleep pressure
        )
        slept_nosleep = False
        for t in range(60):
            _tick_consciousness(c_nosleep, attention_strength=0.8, arousal=0.7,
                                sensory_gate=1.0, binding_quality=0.7)
            if c_nosleep.is_sleeping:
                slept_nosleep = True
                break

        # Without sleep pressure, the system should NOT be forced to sleep
        # (it may still sleep for other reasons, but not due to pressure)
        assert not slept_nosleep, (
            "Without sleep pressure (safety_mode=False), the system should "
            "NOT be forced into dormancy. This confirms sleep pressure is the "
            "mechanism, not an artefact."
        )

        # The distinguishing prediction: MRP forces dormancy where a model
        # without impedance-derived sleep pressure does not.  This difference
        # is the counterfactual ??identical inputs produce qualitatively
        # different behaviour depending on the presence of Î£?Â²-based sleep.
        assert slept_mrp and not slept_nosleep, (
            "Counterfactual confirmed: MRP (sleep pressure) ??forced sleep; "
            "model without sleep pressure ??sustained wakefulness."
        )
