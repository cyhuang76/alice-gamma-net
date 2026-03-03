# -*- coding: utf-8 -*-
"""
pytest wrapper for dual-network stability and pass-rate tests.

Converts the standalone experiment `exp_dual_network_stability.py` into
proper pytest tests so they are covered by the pre-commit gate.

Tests:
  - S0  Baseline convergence (all 10 organs)
  - S0b Perturbation recovery (brain, heart, kidney)
  - S1  Vascular insult cascade (monotone worsening)
  - S2  Coupling asymmetry (no vascular runaway from neural insult)
  - S3  Dual intervention (super-additivity / dual ≥ max single)
  - PR  Randomised pass rate (50 runs, ≥ 60%)
  - PC  Clinical pass rate (50 runs, ≥ 50%)
  - Constants exported from core module
"""

from __future__ import annotations

import sys
import pytest
import numpy as np

sys.path.insert(0, ".")

from alice.body.vascular_impedance import (
    VascularImpedanceNetwork,
    ORGAN_VASCULAR_Z,
    NEURAL_VASCULAR_COUPLING,
    VASCULAR_NEURAL_COUPLING,
    ETA_NEURAL_REPAIR,
    DEFICIT_THRESHOLD,
    simulate_dual_network_cascade,
)

from experiments.exp_dual_network_stability import (
    simulate_dual,
    check_convergence,
    check_recovery,
    check_pass,
    TimeSeriesResult,
    CONVERGENCE_TOL,
    RECOVERY_TOL,
    H_VIABLE,
    PASS_FRACTION_VIABLE,
)


# ============================================================================
# Test: Constants are properly exported from core module
# ============================================================================

class TestCorePhysicsConstants:
    """Verify that the neural self-repair constants are in the core module."""

    def test_eta_neural_repair_exists(self):
        assert ETA_NEURAL_REPAIR == 0.008

    def test_deficit_threshold_exists(self):
        assert DEFICIT_THRESHOLD == 0.05

    def test_coupling_asymmetry(self):
        """α_{v→n} > α_{n→v}"""
        assert VASCULAR_NEURAL_COUPLING > NEURAL_VASCULAR_COUPLING


# ============================================================================
# Test S0: Baseline convergence
# ============================================================================

class TestBaselineConvergence:
    """All 10 organs converge to a stable attractor with no external insult."""

    @pytest.mark.parametrize("organ", sorted(ORGAN_VASCULAR_Z.keys()))
    def test_organ_converges(self, organ: str):
        ts = simulate_dual(organ=organ, n_ticks=500, gamma_n_init=0.05)
        v = check_convergence(ts, label=organ)
        assert v.converged, (
            f"{organ}: var(H)={v.variance_last_100:.6f} >= {CONVERGENCE_TOL}"
        )

    @pytest.mark.parametrize("organ", sorted(ORGAN_VASCULAR_Z.keys()))
    def test_organ_viable(self, organ: str):
        ts = simulate_dual(organ=organ, n_ticks=500, gamma_n_init=0.05)
        assert ts.health[-1] > H_VIABLE, f"{organ}: H={ts.health[-1]}"


# ============================================================================
# Test S0b: Perturbation recovery
# ============================================================================

class TestPerturbationRecovery:
    """Transient insults should not permanently shift the attractor."""

    @pytest.mark.parametrize("organ", ["brain", "heart", "kidney"])
    def test_vascular_perturbation_recovery(self, organ: str):
        ts_base = simulate_dual(organ=organ, n_ticks=600, gamma_n_init=0.05)
        ts_pert = simulate_dual(
            organ=organ, n_ticks=600, gamma_n_init=0.05,
            insult_type="vascular", insult_onset=100,
            insult_duration=100, insult_magnitude=0.20,
        )
        assert check_recovery(ts_base, ts_pert)

    @pytest.mark.parametrize("organ", ["brain", "heart", "kidney"])
    def test_neural_perturbation_recovery(self, organ: str):
        ts_base = simulate_dual(organ=organ, n_ticks=600, gamma_n_init=0.05)
        ts_pert = simulate_dual(
            organ=organ, n_ticks=600, gamma_n_init=0.05,
            insult_type="neural", insult_onset=100,
            insult_duration=100, insult_magnitude=0.15,
        )
        assert check_recovery(ts_base, ts_pert)


# ============================================================================
# Test S1: Vascular cascade monotonicity
# ============================================================================

class TestVascularCascade:
    """Worse vascular stenosis → monotonically worse health."""

    @pytest.mark.parametrize("organ", ["brain", "kidney", "muscle"])
    def test_monotone_health_decline(self, organ: str):
        stenosis_levels = [0.0, 0.20, 0.40, 0.60, 0.80]
        healths = []
        for sten in stenosis_levels:
            ts = simulate_dual(
                organ=organ, n_ticks=800, gamma_n_init=0.05,
                insult_type="vascular" if sten > 0 else "none",
                insult_onset=50, insult_magnitude=sten,
            )
            healths.append(ts.health[-1])
        # Monotone non-increasing (allow small float tolerance)
        for i in range(len(healths) - 1):
            assert healths[i] >= healths[i + 1] - 1e-4, (
                f"{organ}: H[sten={stenosis_levels[i]:.0%}]={healths[i]:.6f} "
                f"< H[sten={stenosis_levels[i+1]:.0%}]={healths[i+1]:.6f}"
            )


# ============================================================================
# Test S2: Coupling asymmetry
# ============================================================================

class TestCouplingAsymmetry:
    """Neural insult should NOT cause runaway vascular failure."""

    @pytest.mark.parametrize("organ", ["brain", "heart", "kidney", "muscle"])
    def test_no_vascular_runaway_from_neural_insult(self, organ: str):
        ts_base = simulate_dual(organ=organ, n_ticks=600, gamma_n_init=0.05)
        ts_neur = simulate_dual(
            organ=organ, n_ticks=600, gamma_n_init=0.05,
            insult_type="neural", insult_onset=50, insult_magnitude=0.50,
        )
        gv_delta = ts_neur.gamma_v_sq[-1] - ts_base.gamma_v_sq[-1]
        assert gv_delta < 0.20, (
            f"{organ}: ΔΓ_v²={gv_delta:.4f} from neural insult (runaway!)"
        )
        assert ts_neur.health[-1] > H_VIABLE, (
            f"{organ}: H={ts_neur.health[-1]} organ non-viable"
        )


# ============================================================================
# Test S3: Dual intervention
# ============================================================================

class TestDualIntervention:
    """Dual intervention should be at least as good as either single arm."""

    @pytest.mark.parametrize("organ", ["brain", "heart", "kidney"])
    def test_dual_at_least_as_good_as_single(self, organ: str):
        N_TICKS = 1000
        ts_none = simulate_dual(
            organ=organ, n_ticks=N_TICKS, gamma_n_init=0.30,
            insult_type="vascular", insult_onset=0, insult_magnitude=0.40,
        )
        ts_fix_n = simulate_dual(
            organ=organ, n_ticks=N_TICKS, gamma_n_init=0.30,
            insult_type="vascular", insult_onset=0, insult_magnitude=0.40,
            intervention_onset=300, intervention_delta_n=0.005,
        )
        ts_fix_v = simulate_dual(
            organ=organ, n_ticks=N_TICKS, gamma_n_init=0.30,
            insult_type="vascular", insult_onset=0, insult_magnitude=0.40,
            intervention_onset=300, intervention_delta_v=0.003,
        )
        ts_both = simulate_dual(
            organ=organ, n_ticks=N_TICKS, gamma_n_init=0.30,
            insult_type="vascular", insult_onset=0, insult_magnitude=0.40,
            intervention_onset=300, intervention_delta_n=0.005,
            intervention_delta_v=0.003,
        )
        assert ts_both.health[-1] >= max(ts_fix_n.health[-1],
                                          ts_fix_v.health[-1]), (
            f"{organ}: dual H={ts_both.health[-1]:.8f} < "
            f"max(single)={max(ts_fix_n.health[-1], ts_fix_v.health[-1]):.8f}"
        )


# ============================================================================
# Test PR: Randomised pass rate
# ============================================================================

class TestPassRate:
    """Statistical pass-rate across randomised parameter sweeps."""

    def test_random_pass_rate_above_60_pct(self):
        rng = np.random.default_rng(42)
        organs = list(ORGAN_VASCULAR_Z.keys())
        n_runs = 50
        n_pass = 0
        for run_id in range(n_runs):
            organ = organs[run_id % len(organs)]
            gn_init = rng.uniform(0.02, 0.20)
            co = rng.uniform(0.60, 1.20)
            bp = rng.uniform(0.70, 1.10)
            visc = rng.uniform(0.90, 1.30)
            sten = rng.uniform(0.0, 0.50)
            ts = simulate_dual(
                organ=organ, n_ticks=600, gamma_n_init=gn_init,
                cardiac_output=co, blood_pressure=bp, blood_viscosity=visc,
                insult_type="vascular" if sten > 0.05 else "none",
                insult_onset=int(rng.integers(0, 200)),
                insult_magnitude=sten,
            )
            passed, _ = check_pass(ts, monitor_start=100)
            if passed:
                n_pass += 1
        rate = n_pass / n_runs
        assert rate >= 0.60, f"Random pass rate {rate:.0%} < 60%"

    def test_clinical_pass_rate_above_50_pct(self):
        rng = np.random.default_rng(43)
        scenarios = []
        for _ in range(10):
            scenarios.append(dict(organ="brain", gn=rng.uniform(0.02, 0.08),
                                  itype="none", imag=0.0, co=rng.uniform(0.9, 1.1)))
        for _ in range(10):
            scenarios.append(dict(organ=rng.choice(["brain", "heart", "kidney"]),
                                  gn=rng.uniform(0.03, 0.10), itype="vascular",
                                  imag=rng.uniform(0.10, 0.25), co=rng.uniform(0.85, 1.05)))
        for _ in range(10):
            scenarios.append(dict(organ=rng.choice(["brain", "kidney", "muscle"]),
                                  gn=rng.uniform(0.10, 0.20), itype="vascular",
                                  imag=rng.uniform(0.25, 0.45), co=rng.uniform(0.80, 1.00)))
        for _ in range(10):
            scenarios.append(dict(organ=rng.choice(["brain", "kidney"]),
                                  gn=rng.uniform(0.08, 0.15), itype="vascular",
                                  imag=rng.uniform(0.15, 0.35), co=rng.uniform(0.40, 0.65)))
        for _ in range(10):
            scenarios.append(dict(organ=rng.choice(["brain", "heart", "kidney", "liver"]),
                                  gn=rng.uniform(0.20, 0.40), itype="vascular",
                                  imag=rng.uniform(0.50, 0.80), co=rng.uniform(0.35, 0.60)))
        n_pass = 0
        for sc in scenarios:
            ts = simulate_dual(
                organ=sc["organ"], n_ticks=600, gamma_n_init=sc["gn"],
                cardiac_output=sc["co"],
                insult_type=sc["itype"], insult_onset=50,
                insult_magnitude=sc["imag"],
            )
            passed, _ = check_pass(ts, monitor_start=100)
            if passed:
                n_pass += 1
        rate = n_pass / len(scenarios)
        assert rate >= 0.50, f"Clinical pass rate {rate:.0%} < 50%"


# ============================================================================
# Test: Core cascade function uses self-repair
# ============================================================================

class TestCoreCascadePhysics:
    """Verify that simulate_dual_network_cascade in the core module
    now uses neural self-repair (Γ_n should not saturate at 0.95)."""

    def test_cascade_gamma_n_not_saturated(self):
        result = simulate_dual_network_cascade(
            organ="brain", n_ticks=500,
            stenosis_at=100, stenosis_fraction=0.3,
        )
        # With self-repair, Γ_n² should stay well below 0.90
        # (without it, it saturates to 0.9025 = 0.95²)
        assert result["final_gamma_n_sq"] < 0.50, (
            f"Γ_n² = {result['final_gamma_n_sq']:.4f} — self-repair not working"
        )

    def test_cascade_health_nonzero(self):
        result = simulate_dual_network_cascade(
            organ="kidney", n_ticks=500,
            stenosis_at=100, stenosis_fraction=0.3,
        )
        assert result["final_health"] > 0, "Organ health collapsed to zero"
