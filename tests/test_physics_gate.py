# -*- coding: utf-8 -*-
"""Physics compliance gate — verifies C1/C2 enforcement in ImpedanceChannel.

This test ensures the clinical layer's impedance physics is isomorphic
to the core engine (gamma_topology.py):
  C1: Γ² + T = 1  (algebraic identity)
  C2: ΔZ = −η · Γ · x_in · x_out  (only legal Z mutation)
"""

import pytest
from alice.body.clinical_common import ImpedanceChannel


# ---- C1: Γ² + T = 1 -------------------------------------------------------

class TestC1EnergyConservation:
    """C1 must hold for any impedance value."""

    @pytest.mark.parametrize("z_init", [1.0, 10.0, 50.0, 100.0, 250.0, 499.0])
    def test_c1_algebraic_identity(self, z_init):
        ch = ImpedanceChannel(z_ref=50.0, z_init=z_init)
        assert ch.verify_c1()

    def test_c1_at_match(self):
        ch = ImpedanceChannel(z_ref=75.0)
        assert ch.gamma == 0.0
        assert ch.gamma_sq == 0.0
        assert ch.transmission == 1.0

    def test_c1_after_remodel(self):
        ch = ImpedanceChannel(z_ref=50.0, z_init=200.0)
        for _ in range(100):
            ch.remodel(x_in=1.0, x_out=1.0, eta=1.0)
        assert ch.verify_c1()


# ---- C2: remodel is the only Z mutation path --------------------------------

class TestC2ImpedanceRemodeling:
    """C2: ΔZ = −η · Γ(Z, Z_target) · x_in · x_out."""

    def test_remodel_reduces_mismatch(self):
        """Remodel toward z_target must reduce |Γ(Z, Z_target)|."""
        ch = ImpedanceChannel(z_ref=50.0, z_init=50.0)
        z_target = 200.0
        g_before = abs((ch.z - z_target) / (ch.z + z_target))
        ch.remodel(x_in=1.0, x_out=1.0, eta=1.0, z_target=z_target)
        g_after = abs((ch.z - z_target) / (ch.z + z_target))
        assert g_after < g_before

    def test_remodel_toward_healthy(self):
        """Default remodel (no z_target) drives Z toward z_ref."""
        ch = ImpedanceChannel(z_ref=50.0, z_init=200.0)
        z_before = ch.z
        ch.remodel(x_in=1.0, x_out=1.0, eta=1.0)
        assert abs(ch.z - 50.0) < abs(z_before - 50.0)

    def test_remodel_respects_bounds(self):
        ch = ImpedanceChannel(z_ref=50.0, z_init=50.0, z_min=10.0, z_max=200.0)
        for _ in range(1000):
            ch.remodel(x_in=1.0, x_out=1.0, eta=100.0, z_target=500.0)
        assert ch.z <= 200.0
        assert ch.z >= 10.0

    def test_remodel_zero_activity_no_change(self):
        """If x_in or x_out is 0, Z must not change (no gate = no remodel)."""
        ch = ImpedanceChannel(z_ref=50.0, z_init=200.0)
        z_before = ch.z
        ch.remodel(x_in=0.0, x_out=1.0, eta=1.0, z_target=50.0)
        assert ch.z == z_before
        ch.remodel(x_in=1.0, x_out=0.0, eta=1.0, z_target=50.0)
        assert ch.z == z_before

    def test_z_readonly(self):
        """Z property must be read-only — assignment raises AttributeError."""
        ch = ImpedanceChannel(z_ref=50.0)
        with pytest.raises(AttributeError):
            ch.z = 100.0

    def test_remodel_convergence(self):
        """After many remodel steps, Z converges to z_target."""
        ch = ImpedanceChannel(z_ref=50.0, z_init=50.0)
        z_target = 150.0
        for _ in range(500):
            ch.remodel(x_in=1.0, x_out=1.0, eta=2.0, z_target=z_target)
        assert abs(ch.z - z_target) < 5.0  # asymptotic approach


# ---- C1 in ClinicalEngineBase -----------------------------------------------

class TestC1EngineLevelEnforcement:
    """ClinicalEngineBase.tick() must clamp total Γ² ≤ 1.0."""

    def test_reserve_never_negative(self):
        from alice.body.clinical_common import ClinicalEngineBase

        class DummyDisease:
            def tick(self):
                return {"gamma_sq": 0.8}

        engine = ClinicalEngineBase()
        engine.active_diseases["a"] = DummyDisease()
        engine.active_diseases["b"] = DummyDisease()
        with pytest.warns(UserWarning, match="C1 violated"):
            r = engine.tick()
        assert r[engine.RESERVE_KEY] >= 0.0
        assert r["total_gamma_sq"] <= 1.0
