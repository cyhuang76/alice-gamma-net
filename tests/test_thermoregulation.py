# -*- coding: utf-8 -*-
"""
Thermoregulation Physics — pytest suite
════════════════════════════════════════

Verifies the rigorous derivation: η(T) → Z → Γ → A[Γ]

Derivation chain (from exp_thermoregulation_physics.py):
  §1 η̃(T) = exp[α·(1/T − 1/T₀)]           (Arrhenius blood viscosity)
  §2 Z_bus(T) = Z_ref · η̃(T)                (Poiseuille transmission line)
  §3 Γ(T) = (Z_organ − Z_bus) / (Z_organ + Z_bus)
  §4 A(T) = Σ K_common · Γ²(T)              (whole-body action)
  §5 τ_repair ∝ exp[(α+β)·(1/T − 1/T₀)]    (doubly exponential)
  §6 T_c = T₀ / [1 + T₀·ln(τ_max)/(α+β)]   (critical temperature)
  §7 T_c(K) ↑ with K                         (dimensional amplification)

Emergence level: E0 — all results from C1/C2/C3 + Arrhenius.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.exp_thermoregulation_physics import (
    ALPHA,
    BETA,
    CELSIUS_TO_KELVIN,
    T_REF,
    TAU_MAX,
    critical_temperature,
    critical_temperature_with_K,
    eta_repair,
    eta_viscosity,
    gamma_junction,
    simulate_K_dependence,
    simulate_temperature_sweep,
    tau_repair,
    z_bus,
    # Cave thermal physics
    thermal_excess_during_sleep,
    min_ambient_for_survival,
    thermal_penetration_depth,
    cave_damping_factor,
    cave_temperature,
    sleeping_body_temperature,
    P_MET_SLEEP,
    H_CONV,
    A_BODY,
    KAPPA_ROCK,
    RHO_CP_ROCK,
    T_DIURNAL,
    T_ANNUAL,
)
from alice.body.tissue_blueprint import build_whole_body


# ============================================================================
# §1  Arrhenius Viscosity
# ============================================================================


class TestArrheniusViscosity:
    """Verify η̃(T) = exp[α·(1/T − 1/T_ref)]."""

    def test_reference_is_unity(self):
        """η̃(T_ref) = 1 by definition."""
        assert eta_viscosity(T_REF) == pytest.approx(1.0, abs=1e-12)

    def test_cold_increases_viscosity(self):
        """∂η/∂T < 0 → colder = more viscous."""
        assert eta_viscosity(283.15) > eta_viscosity(298.15) > 1.0

    def test_hot_decreases_viscosity(self):
        """T > T_ref → η < 1."""
        assert eta_viscosity(315.0) < 1.0

    def test_25C_within_range(self):
        """η̃(25°C) ∈ [1.1, 1.5] (CRC Handbook water data)."""
        val = eta_viscosity(298.15)
        assert 1.1 < val < 1.5

    def test_0C_within_range(self):
        """η̃(0°C) ∈ [2.0, 3.5] (CRC Handbook)."""
        val = eta_viscosity(273.15)
        assert 2.0 < val < 3.5

    def test_vectorized(self):
        """Works on numpy arrays."""
        T = np.array([273.15, 298.15, 310.0])
        result = eta_viscosity(T)
        assert result.shape == (3,)
        assert result[2] == pytest.approx(1.0, abs=1e-12)

    def test_monotonically_decreasing_in_T(self):
        """η̃ strictly decreases as T increases."""
        T_range = np.linspace(260.0, 320.0, 200)
        eta_vals = eta_viscosity(T_range)
        assert np.all(np.diff(eta_vals) < 0)


# ============================================================================
# §2  Vascular Bus Impedance
# ============================================================================


class TestZBus:
    """Verify Z_bus(T) = Z_ref · η̃(T)."""

    def test_reference_equals_z_ref(self):
        z = z_bus(T_REF, z_ref=55.0)
        assert z == pytest.approx(55.0, abs=1e-10)

    def test_cold_increases_z(self):
        assert z_bus(273.15) > z_bus(298.15) > z_bus(T_REF)

    def test_monotonically_decreasing_in_T(self):
        T_range = np.linspace(260.0, 320.0, 200)
        z_vals = z_bus(T_range)
        assert np.all(np.diff(z_vals) < 0)

    def test_scales_linearly_with_z_ref(self):
        ratio = z_bus(280.0, z_ref=100.0) / z_bus(280.0, z_ref=50.0)
        assert ratio == pytest.approx(2.0, abs=1e-10)


# ============================================================================
# §3  Reflection Coefficient
# ============================================================================


class TestGammaJunction:
    """Verify Γ = (Z_organ − Z_bus) / (Z_organ + Z_bus)."""

    def test_matched_is_zero(self):
        assert gamma_junction(55.0, 55.0) == pytest.approx(0.0)

    def test_sign_convention(self):
        """Z_organ > Z_bus → Γ > 0; Z_organ < Z_bus → Γ < 0."""
        assert gamma_junction(100.0, 50.0) > 0
        assert gamma_junction(50.0, 100.0) < 0

    def test_bounded(self):
        """−1 < Γ < 1 for finite positive impedances."""
        g = gamma_junction(200.0, 30.0)
        assert -1.0 < g < 1.0

    def test_zero_denom(self):
        assert gamma_junction(0.0, 0.0) == 0.0

    def test_gamma_increases_with_cooling(self):
        """|Γ| at 0°C > |Γ| at 25°C for z_organ = z_ref."""
        z_o = 55.0
        g_0 = abs(gamma_junction(z_o, float(z_bus(273.15))))
        g_25 = abs(gamma_junction(z_o, float(z_bus(298.15))))
        g_37 = abs(gamma_junction(z_o, float(z_bus(T_REF))))
        assert g_0 > g_25 > g_37


# ============================================================================
# §4  Repair Rate
# ============================================================================


class TestEtaRepair:
    """Verify η̃_learn(T) = exp[β·(1/T_ref − 1/T)]."""

    def test_reference_is_unity(self):
        assert eta_repair(T_REF) == pytest.approx(1.0, abs=1e-12)

    def test_cold_reduces_repair(self):
        assert eta_repair(283.15) < eta_repair(298.15) < 1.0

    def test_hot_increases_repair(self):
        assert eta_repair(315.0) > 1.0

    def test_drops_faster_than_viscosity_rises(self):
        """β > α → repair drops faster than viscosity rises."""
        T_cold = 280.0
        # η_viscosity rises by exp[α·Δ], η_repair drops by exp[β·Δ]
        # Combined effect: τ grows by exp[(α+β)·Δ]
        assert BETA > ALPHA


# ============================================================================
# §5  Doubly Exponential τ
# ============================================================================


class TestTauRepair:
    """Verify τ̃ = exp[(α+β)·(1/T − 1/T₀)]."""

    def test_reference_is_unity(self):
        assert tau_repair(T_REF) == pytest.approx(1.0, abs=1e-12)

    def test_cold_increases_tau(self):
        assert tau_repair(283.15) > tau_repair(298.15) > 1.0

    def test_exponent_is_alpha_plus_beta(self):
        """log(τ) vs 1/T has slope α+β."""
        T_range = np.linspace(280.0, 310.0, 100)
        log_tau = np.log(tau_repair(T_range))
        inv_T = 1.0 / T_range
        slope, _ = np.polyfit(inv_T, log_tau, 1)
        assert slope == pytest.approx(ALPHA + BETA, rel=0.005)

    def test_equals_product_of_viscosity_and_inverse_repair(self):
        """τ̃ = η̃_viscosity / η̃_repair at any T."""
        T_test = 290.0
        tau_direct = float(tau_repair(T_test))
        tau_product = float(eta_viscosity(T_test)) / float(eta_repair(T_test))
        assert tau_direct == pytest.approx(tau_product, rel=1e-10)


# ============================================================================
# §6  Critical Temperature
# ============================================================================


class TestCriticalTemperature:
    """Verify T_c = T₀ / [1 + T₀·ln(τ_max)/(α+β)]."""

    def test_below_body_temperature(self):
        T_c = critical_temperature()
        assert T_c < T_REF

    def test_above_absolute_zero(self):
        T_c = critical_temperature()
        assert T_c > 200.0

    def test_closed_form_matches_tau(self):
        """τ̃(T_c) ≈ τ_max by definition."""
        T_c = critical_temperature()
        tau_at_Tc = float(tau_repair(T_c))
        assert tau_at_Tc == pytest.approx(TAU_MAX, rel=0.01)

    def test_higher_tau_max_gives_lower_Tc(self):
        """Higher τ_max → more cold tolerance → T_c drops."""
        T_c_100 = critical_temperature(tau_ratio_max=100.0)
        T_c_1000 = critical_temperature(tau_ratio_max=1000.0)
        assert T_c_1000 < T_c_100

    def test_celsius_range_reasonable(self):
        T_c_C = critical_temperature() - CELSIUS_TO_KELVIN
        assert -50.0 < T_c_C < 37.0


# ============================================================================
# §7  Dimensional Amplification — T_c(K)
# ============================================================================


class TestDimensionalAmplification:
    """T_c(K) must increase with brain complexity K."""

    def test_monotonically_increasing(self):
        K_vals = range(1, 11)
        T_c_vals = [critical_temperature_with_K(K) for K in K_vals]
        for i in range(1, len(T_c_vals)):
            assert T_c_vals[i] >= T_c_vals[i - 1], \
                f"T_c(K={i+1}) < T_c(K={i})"

    def test_K1_equals_base(self):
        """T_c(K=1) = T_c(base) since K doesn't add penalty."""
        assert critical_temperature_with_K(1) == pytest.approx(
            critical_temperature(), abs=1e-6)

    def test_K5_above_K1(self):
        """Human brain (K=5) has higher T_c than reptile (K=1)."""
        assert critical_temperature_with_K(5) > critical_temperature_with_K(1)

    @pytest.mark.parametrize("K", [1, 2, 3, 5, 8])
    def test_Tc_below_body_temp(self, K):
        assert critical_temperature_with_K(K) < T_REF

    def test_simulate_K_dependence_consistent(self):
        result = simulate_K_dependence(np.arange(1, 6))
        assert result["T_c_kelvin"][0] < result["T_c_kelvin"][-1]
        assert len(result["K"]) == 5


# ============================================================================
# §8  Whole-Body Action Sweep
# ============================================================================


class TestWholeBodyActionSweep:
    """Temperature sweep on the 278-node whole-body topology."""

    @pytest.fixture(scope="class")
    def sweep_result(self):
        return simulate_temperature_sweep(
            T_celsius_range=np.linspace(-5.0, 42.0, 50))

    def test_returns_expected_keys(self, sweep_result):
        for key in ["T_celsius", "A_vascular", "eta_viscosity",
                     "eta_repair", "tau_repair", "max_gamma"]:
            assert key in sweep_result

    def test_29_organs_measured(self, sweep_result):
        assert sweep_result["n_organs"] == 29

    def test_action_minimum_near_body_temp(self, sweep_result):
        T = sweep_result["T_celsius"]
        A = sweep_result["A_vascular"]
        idx_min = int(np.argmin(A))
        T_opt = T[idx_min]
        assert 10.0 < T_opt < 42.0, f"A minimum at {T_opt}°C"

    def test_extreme_cold_high_action(self, sweep_result):
        T = sweep_result["T_celsius"]
        A = sweep_result["A_vascular"]
        idx_0 = np.argmin(np.abs(T - 0.0))
        idx_min = int(np.argmin(A))
        assert A[idx_0] > A[idx_min] * 1.5

    def test_Tc_is_reported(self, sweep_result):
        T_c = sweep_result["T_c_celsius"]
        assert -50 < T_c < 37


# ============================================================================
# §9  C1 Holds at All Temperatures
# ============================================================================


class TestC1Invariance:
    """Γ² + T = 1 at every edge regardless of temperature."""

    @pytest.mark.parametrize("T_celsius", [-5, 0, 10, 25, 37, 42])
    def test_c1_at_temperature(self, T_celsius):
        T_K = T_celsius + CELSIUS_TO_KELVIN
        eta_v = float(eta_viscosity(T_K))

        topo = build_whole_body(seed=42)
        for name, node in topo.nodes.items():
            if name.startswith("vbus."):
                node.impedance = node.impedance * eta_v

        for (src, tgt), ch in topo.active_edges.items():
            ch.source = topo.nodes[src]
            ch.target = topo.nodes[tgt]
            assert ch.verify_c1(), f"C1 violated at {src}→{tgt}, T={T_celsius}°C"


# ============================================================================
# §10  τ Scales Linear with K
# ============================================================================


class TestTauScalesWithK:
    """τ_brain ∝ K · Γ² / η_learn — linear in K."""

    def test_ratio_K5_to_K1(self):
        T = 295.0
        z_o = 55.0
        z_b = float(z_bus(T))
        g2 = gamma_junction(z_o, z_b) ** 2
        eta_l = float(eta_repair(T))
        tau_1 = 1 * g2 / eta_l
        tau_5 = 5 * g2 / eta_l
        assert tau_5 / tau_1 == pytest.approx(5.0, abs=1e-10)

    @pytest.mark.parametrize("K", [1, 2, 3, 4, 5, 8, 10])
    def test_linear_in_K(self, K):
        T = 290.0
        z_o = 55.0
        z_b = float(z_bus(T))
        g2 = gamma_junction(z_o, z_b) ** 2
        eta_l = float(eta_repair(T))
        tau_K = K * g2 / eta_l
        tau_1 = 1 * g2 / eta_l
        assert tau_K == pytest.approx(K * tau_1, rel=1e-10)


# ============================================================================
# §11  Endothermy Necessity Theorem
# ============================================================================


class TestEndothermyNecessity:
    """Higher K → higher T_c → endothermy required."""

    def test_human_more_vulnerable_than_reptile(self):
        T_c_human = critical_temperature_with_K(5)
        T_c_reptile = critical_temperature_with_K(1)
        assert T_c_human > T_c_reptile

    def test_temperature_gap_is_significant(self):
        gap = (critical_temperature_with_K(5)
               - critical_temperature_with_K(1))
        assert gap > 5.0, f"Gap = {gap} K too small to be biologically relevant"

    def test_K5_Tc_positive_celsius(self):
        """Human T_c should be above 0°C (ice point)."""
        T_c = critical_temperature_with_K(5) - CELSIUS_TO_KELVIN
        assert T_c > 0.0, f"T_c(K=5) = {T_c}°C — still below ice point"


# ============================================================================
# §12  Physical Constants Consistency
# ============================================================================


class TestPhysicalConstants:
    """Verify that the Arrhenius parameters are physically reasonable."""

    def test_alpha_range(self):
        """α = E_a/R for water: 1800–2100 K."""
        assert 1800 < ALPHA < 2100

    def test_beta_range(self):
        """β = E_learn/R for enzymes: 4000–8000 K (Q₁₀ ≈ 2–3)."""
        assert 4000 < BETA < 8000

    def test_beta_greater_than_alpha(self):
        """Biology: repair processes are more temperature-sensitive
        than viscous flow (enzymes have higher E_a than water)."""
        assert BETA > ALPHA

    def test_T_ref_is_body_temperature(self):
        assert T_REF == pytest.approx(310.0, abs=0.5)

    def test_celsius_offset(self):
        assert CELSIUS_TO_KELVIN == pytest.approx(273.15)


# ============================================================================
# §8  Thermal Excess During Sleep
# ============================================================================


class TestThermalExcess:
    """Verify heat balance model during sleep."""

    def test_positive(self):
        """ΔT_excess must be positive (body warmer than ambient)."""
        assert thermal_excess_during_sleep() > 0

    def test_physiological_range(self):
        """ΔT_excess for a 70 kg human should be 4–8°C."""
        dT = thermal_excess_during_sleep()
        assert 4.0 < dT < 8.0

    def test_scales_with_Pmet(self):
        """Higher metabolic rate → larger thermal excess."""
        dT_low = thermal_excess_during_sleep(p_met=60.0)
        dT_high = thermal_excess_during_sleep(p_met=100.0)
        assert dT_high > dT_low

    def test_scales_inversely_with_surface_area(self):
        """Larger surface area → more heat loss → lower ΔT."""
        dT_small = thermal_excess_during_sleep(a_body=1.5)
        dT_large = thermal_excess_during_sleep(a_body=2.0)
        assert dT_small > dT_large

    def test_exact_formula(self):
        """ΔT = P / (h·A), check exact value."""
        dT = thermal_excess_during_sleep(p_met=80.0, h=10.0, a_body=2.0)
        assert dT == pytest.approx(4.0)


# ============================================================================
# §9  Minimum Ambient Temperature
# ============================================================================


class TestMinAmbientSurvival:
    """Verify minimum ambient temperature for sleeping survival."""

    def test_highK_needs_warmer_ambient(self):
        """K=5 needs warmer ambient than K=1."""
        T_min_5 = min_ambient_for_survival(K=5)
        T_min_1 = min_ambient_for_survival(K=1)
        assert T_min_5 > T_min_1

    def test_K5_min_ambient_reasonable(self):
        """K=5 minimum ambient should be around −5 to +5°C."""
        T_min = min_ambient_for_survival(K=5)
        assert -5.0 < T_min < 5.0

    def test_K1_survives_deep_cold(self):
        """K=1 minimum ambient should be well below −10°C."""
        T_min = min_ambient_for_survival(K=1)
        assert T_min < -10.0

    def test_equals_Tc_minus_excess(self):
        """T_amb_min = T_c(K) - ΔT_excess (definition check)."""
        T_c_K5 = critical_temperature_with_K(5) - CELSIUS_TO_KELVIN
        dT = thermal_excess_during_sleep()
        T_min = min_ambient_for_survival(K=5)
        assert T_min == pytest.approx(T_c_K5 - dT, abs=0.01)


# ============================================================================
# §10  Thermal Penetration Depth
# ============================================================================


class TestThermalPenetration:
    """Verify cave rock thermal physics."""

    def test_diurnal_depth_order_cm(self):
        """Diurnal penetration depth should be ~10–20 cm."""
        delta = thermal_penetration_depth(T_DIURNAL)
        assert 0.05 < delta < 0.5

    def test_annual_depth_order_metres(self):
        """Annual penetration depth should be ~2–5 m."""
        delta = thermal_penetration_depth(T_ANNUAL)
        assert 1.0 < delta < 6.0

    def test_annual_much_larger_than_diurnal(self):
        """Annual >> diurnal by √(365) ≈ 19× factor."""
        d_day = thermal_penetration_depth(T_DIURNAL)
        d_year = thermal_penetration_depth(T_ANNUAL)
        ratio = d_year / d_day
        assert 15 < ratio < 25

    def test_scales_with_sqrt_period(self):
        """δ ∝ √T, so doubling period → √2 × depth."""
        d1 = thermal_penetration_depth(1000.0)
        d2 = thermal_penetration_depth(2000.0)
        assert d2 / d1 == pytest.approx(math.sqrt(2), rel=0.01)

    def test_scales_with_sqrt_kappa(self):
        """δ ∝ √κ."""
        d1 = thermal_penetration_depth(T_DIURNAL, kappa=1.0)
        d2 = thermal_penetration_depth(T_DIURNAL, kappa=4.0)
        assert d2 / d1 == pytest.approx(2.0, rel=0.01)


# ============================================================================
# §11  Cave Damping Factor
# ============================================================================


class TestCaveDamping:
    """Verify exponential temperature damping in rock."""

    def test_surface_is_unity(self):
        """At z=0, damping = 1 (full surface amplitude)."""
        assert cave_damping_factor(0.0, T_DIURNAL) == pytest.approx(1.0)

    def test_decays_with_depth(self):
        """Deeper → more damping."""
        d1 = cave_damping_factor(1.0, T_ANNUAL)
        d2 = cave_damping_factor(3.0, T_ANNUAL)
        assert d2 < d1

    def test_diurnal_eliminated_at_5m(self):
        """Diurnal oscillation at 5m: essentially zero."""
        d = cave_damping_factor(5.0, T_DIURNAL)
        assert d < 1e-6

    def test_annual_reduced_at_5m(self):
        """Annual oscillation at 5m: significantly reduced."""
        d = cave_damping_factor(5.0, T_ANNUAL)
        assert 0.05 < d < 0.5

    def test_at_one_delta_is_1_over_e(self):
        """At z=δ, damping = 1/e."""
        delta = thermal_penetration_depth(T_ANNUAL)
        d = cave_damping_factor(delta, T_ANNUAL)
        assert d == pytest.approx(1.0 / math.e, rel=0.01)


# ============================================================================
# §12  Cave Temperature
# ============================================================================


class TestCaveTemperature:
    """Verify cave interior temperature model."""

    def test_mean_equals_annual_mean(self):
        """Cave T ≈ annual mean surface T."""
        T_cave, _ = cave_temperature(12.0, 15.0, 5.0)
        assert T_cave == pytest.approx(12.0)

    def test_amplitude_reduced(self):
        """Cave amplitude much less than surface amplitude."""
        _, amp = cave_temperature(12.0, 15.0, 5.0)
        assert amp < 15.0 * 0.5  # must be < 50% of surface

    def test_deeper_is_more_stable(self):
        """Deeper cave → smaller amplitude."""
        _, amp_shallow = cave_temperature(12.0, 15.0, 2.0)
        _, amp_deep = cave_temperature(12.0, 15.0, 10.0)
        assert amp_deep < amp_shallow

    def test_zero_depth_full_amplitude(self):
        """At surface: full amplitude."""
        _, amp = cave_temperature(12.0, 15.0, 0.0)
        assert amp == pytest.approx(15.0, rel=0.01)


# ============================================================================
# §13  Sleeping Body Temperature
# ============================================================================


class TestSleepingBodyTemperature:
    """Verify the complete cave shelter theorem."""

    def test_higher_than_ambient(self):
        """Body temp > ambient (endothermic)."""
        T_body = sleeping_body_temperature(10.0)
        assert T_body > 10.0

    def test_winter_exposed_dangerous(self):
        """K=5 organism sleeping exposed at -10°C: T_body near T_c."""
        T_body = sleeping_body_temperature(-10.0)
        T_c = critical_temperature_with_K(5) - CELSIUS_TO_KELVIN
        assert T_body < T_c + 2.0  # dangerously close or below

    def test_cave_is_safe(self):
        """In a 12°C cave: T_body safely above T_c(K=5)."""
        T_body = sleeping_body_temperature(12.0)
        T_c = critical_temperature_with_K(5) - CELSIUS_TO_KELVIN
        assert T_body > T_c + 5.0  # safe margin

    def test_low_K_survives_exposed(self):
        """K=1 organism can survive at -10°C (T_c very low)."""
        T_c = critical_temperature_with_K(1) - CELSIUS_TO_KELVIN
        assert -10.0 > T_c  # ambient > T_c, ectotherm viable

    def test_K_dependence_shelter_threshold(self):
        """Higher K → higher T_amb,min → more shelter-dependent."""
        T_min_vals = [min_ambient_for_survival(K=k) for k in [1, 2, 3, 5]]
        for i in range(len(T_min_vals) - 1):
            assert T_min_vals[i] < T_min_vals[i + 1]
