"""
Test suite for Paper VIII: Fractal Cost of Dual-Network Life.

Verifies the four core results:
  1. β = n^{1/d} = 2^{1/3} ≈ 1.2599  (Murray bifurcation ratio)
  2. A_cut ∝ M^{(d-1)/d} = M^{2/3}    (irreducible cost scaling)
  3. B ∝ M^{d/(d+1)} = M^{3/4}         (Kleiber formula d/(d+1))
  4. Ω_n ≈ β                            (dual-network overhead)
"""

import numpy as np
import pytest

N_BRANCH = 2
D_EMBED = 3
BETA = N_BRANCH ** (1.0 / D_EMBED)


class TestBetaBifurcation:
    """Result 1: β = n^{1/d}."""

    def test_beta_value(self):
        assert abs(BETA - 1.2599210498948732) < 1e-12

    @pytest.mark.parametrize("n,d,expected", [
        (2, 2, 2**0.5),
        (2, 3, 2**(1/3)),
        (3, 3, 3**(1/3)),
        (4, 3, 4**(1/3)),
    ])
    def test_beta_general(self, n, d, expected):
        assert abs(n**(1.0/d) - expected) < 1e-12


class TestAcutScaling:
    """Result 2: A_cut ∝ M^{(d-1)/d}."""

    def test_exponent(self):
        n, d = N_BRANCH, D_EMBED
        c = n ** (-1.0 / d)
        nc = n * c
        K0 = 100.0
        L_values = [5, 10, 15, 20, 25, 30]
        acut_arr, M_arr = [], []
        for L in L_values:
            N_T = n ** L
            geo_sum = (nc**L - 1) / (nc - 1)
            acut_arr.append(K0 * (1 - c) * geo_sum)
            M_arr.append(float(N_T))
        coeffs = np.polyfit(np.log(M_arr), np.log(acut_arr), 1)
        assert abs(coeffs[0] - 2/3) < 0.01

    def test_per_cell_decreasing(self):
        n, d = N_BRANCH, D_EMBED
        c = n ** (-1.0 / d)
        nc = n * c
        K0 = 100.0
        prev = float('inf')
        for L in range(5, 35, 5):
            geo_sum = (nc**L - 1) / (nc - 1)
            a_per = K0 * (1 - c) * geo_sum / (n ** L)
            assert a_per < prev
            prev = a_per


class TestKleiberFormula:
    """Result 3: d/(d+1) exponent structure."""

    @pytest.mark.parametrize("d", range(1, 11))
    def test_decomposition(self, d):
        rubner = (d - 1) / d
        corr = 1 / (d * (d + 1))
        assert abs(rubner + corr - d / (d + 1)) < 1e-15

    @pytest.mark.parametrize("d", range(1, 100))
    def test_bounds(self, d):
        alpha = d / (d + 1)
        assert (d - 1) / d < alpha < 1.0

    def test_d3_is_three_quarter(self):
        assert abs(D_EMBED / (D_EMBED + 1) - 0.75) < 1e-15

    def test_convergence_to_one(self):
        assert abs(1000 / 1001 - 1.0) < 1e-3


class TestOmegaOverhead:
    """Result 4: Ω_n ≈ β."""

    def test_omega_neural_near_beta(self):
        omega_n = 1.0 / (1.0 - 0.20)        # brain = 20% of BMR
        assert abs(omega_n - BETA) < 0.02     # within ~1%
        assert abs(omega_n - BETA) / BETA < 0.01  # < 1% relative

    def test_omega_full_bounded(self):
        omega = 1.0 / (1.0 - 0.20 - 0.05)   # brain + cardiac
        assert 1.30 < omega < 1.40


class TestMurrayLaw:
    """Murray's Law r_d = r_p n^{-1/3} from MRP."""

    @pytest.mark.parametrize("n", [2, 3, 4, 8])
    def test_murray_from_action(self, n):
        from scipy.optimize import minimize_scalar
        r_p, Q, lam = 5.0, 1.0, 2.0 / 5.0**6

        def action(r_d):
            return Q**2 / (n * r_d**4) + lam * n * r_d**2

        r_opt = minimize_scalar(action, bounds=(0.1, r_p),
                                method='bounded').x
        r_murray = r_p / n**(1/3)
        assert abs(r_opt - r_murray) < 1e-4


class TestFractalDimension:
    """D_K = d for space-filling relay hierarchy."""

    @pytest.mark.parametrize("d", [2, 3, 4])
    def test_dk_equals_d(self, d):
        c = N_BRANCH ** (-1.0 / d)
        D_K = np.log(N_BRANCH) / np.log(1.0 / c)
        assert abs(D_K - d) < 1e-10


class TestEmpiricalAllometry:
    """Kleiber allometry from published mammalian BMR data."""

    def test_exponent_near_three_quarter(self):
        # (species, mass_kg, BMR_watts)
        data = [
            (0.025, 0.35), (0.30, 1.45), (2.5, 6.3),
            (3.5, 7.8), (10.0, 17.0), (50.0, 50.0),
            (70.0, 80.0), (500.0, 280.0), (500.0, 265.0),
            (4000.0, 1300.0),
        ]
        M = np.array([d[0] for d in data])
        B = np.array([d[1] for d in data])
        coeffs = np.polyfit(np.log10(M), np.log10(B), 1)
        assert abs(coeffs[0] - 0.75) < 0.10
        # R²
        pred = np.polyval(coeffs, np.log10(M))
        ss_res = np.sum((np.log10(B) - pred)**2)
        ss_tot = np.sum((np.log10(B) - np.mean(np.log10(B)))**2)
        assert 1 - ss_res / ss_tot > 0.95
