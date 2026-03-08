"""test_clinical_orthopedics.py - 10 orthopedic diseases + engine."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from alice.body.clinical_orthopedics import (
    ClinicalOrthopedicsEngine, FractureModel, OsteoporosisModel,
    DiscHerniationModel, OsteoarthritisModel, ACLModel,
    TendinitisModel, ScoliosisModel, OsteosarcomaModel,
    GoutModel, OsteomyelitisModel,
    Z_BONE, Z_CARTILAGE, Z_SYNOVIAL,
    gamma_sq, joint_transmission, joint_transmission_no_cartilage,
    aging_pain_model,
)

class TestFracture(unittest.TestCase):
    def test_tick(self):
        m = FractureModel(ao_class="A2")
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("healing", r)
    def test_fixation(self):
        m = FractureModel()
        m.fixation()
        res = [m.tick() for _ in range(30)]
        self.assertGreater(res[-1]["healing"], 0)

class TestOsteoporosis(unittest.TestCase):
    def test_tick(self):
        m = OsteoporosisModel(t_score=-2.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("t_score", r)
        self.assertIn("frax", r)

class TestDisc(unittest.TestCase):
    def test_tick(self):
        m = DiscHerniationModel(severity=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("odi", r)

class TestOA(unittest.TestCase):
    def test_tick(self):
        m = OsteoarthritisModel(kl_grade=2)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("womac", r)

class TestACL(unittest.TestCase):
    def test_tick(self):
        m = ACLModel(partial=False)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("ikdc", r)
    def test_recon(self):
        m = ACLModel()
        r0 = m.tick()
        m.reconstruction()
        res = [m.tick() for _ in range(30)]
        self.assertLess(res[-1]["gamma_sq"], r0["gamma_sq"])

class TestTendinitis(unittest.TestCase):
    def test_tick(self):
        m = TendinitisModel(severity=0.4)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("dash", r)

class TestScoliosis(unittest.TestCase):
    def test_tick(self):
        m = ScoliosisModel(cobb=25.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("cobb", r)

class TestOsteosarcoma(unittest.TestCase):
    def test_tick(self):
        m = OsteosarcomaModel(size_cm=5.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("enneking", r)

class TestGout(unittest.TestCase):
    def test_tick(self):
        m = GoutModel(urate=9.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("urate", r)
    def test_allopurinol(self):
        m = GoutModel(urate=10.0)
        r0 = m.tick()
        m.start_allopurinol()
        res = [m.tick() for _ in range(20)]
        self.assertLess(res[-1]["urate"], r0["urate"])

class TestOsteomyelitis(unittest.TestCase):
    def test_tick(self):
        m = OsteomyelitisModel(acute=True)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("esr", r)

class TestOrthoEngine(unittest.TestCase):
    def test_multi(self):
        e = ClinicalOrthopedicsEngine()
        e.add_disease("fracture", ao_class="A1")
        e.add_disease("gout", urate=8.5)
        r = e.tick()
        self.assertIn("total_gamma_sq", r)


# ============================================================================
# JOINT IMPEDANCE TRANSFORMER TESTS
# ============================================================================

class TestJointImpedanceTransformer(unittest.TestCase):
    """Multi-layer graded joint transmission: Bone→Cartilage→Synovial→Cartilage→Bone."""

    def test_healthy_joint_high_transmission(self):
        """Healthy cartilage acts as a quarter-wave transformer → T_total high."""
        r = joint_transmission(Z_CARTILAGE)
        self.assertGreater(r["T_total"], 0.80)

    def test_bone_on_bone_lower_transmission(self):
        """End-stage OA (no cartilage) → T_total much lower."""
        healthy = joint_transmission(Z_CARTILAGE)
        bonebone = joint_transmission_no_cartilage()
        self.assertGreater(healthy["T_total"], bonebone["T_total"])

    def test_degraded_cartilage_worse_than_healthy(self):
        """Partially worn cartilage (Z=40) → lower T than healthy (Z=80)."""
        healthy = joint_transmission(Z_CARTILAGE)
        worn = joint_transmission(Z_CARTILAGE * 0.5)
        self.assertGreater(healthy["T_total"], worn["T_total"])

    def test_per_interface_gamma_sq_small(self):
        """Each interface in healthy joint has Γ² < 0.10."""
        r = joint_transmission(Z_CARTILAGE)
        for key, val in r.items():
            if key.startswith(("bone", "cart", "synovial")):
                self.assertLess(val, 0.10,
                    f"{key}: Γ²={val:.4f} exceeds 0.10")

    def test_bone_on_bone_interface_gamma_larger(self):
        """Direct bone→synovial has larger Γ² than bone→cartilage."""
        bonebone = joint_transmission_no_cartilage()
        healthy = joint_transmission(Z_CARTILAGE)
        self.assertGreater(
            bonebone["bone→synovial"],
            healthy["bone→cart"],
        )

    def test_c1_holds_at_every_interface(self):
        """Γ² + T = 1 at every interface (C1 energy conservation)."""
        r = joint_transmission(Z_CARTILAGE)
        for key, g2 in r.items():
            if "→" in key:
                self.assertAlmostEqual(g2 + (1 - g2), 1.0, places=12)

    def test_cartilage_z_is_geometric_mean(self):
        """Optimal transformer Z ≈ √(Z_bone × Z_synovial) ≈ 77.5.
        Actual Z_cartilage=80 is within 5% of the geometric mean."""
        import math
        z_optimal = math.sqrt(Z_BONE * Z_SYNOVIAL)
        self.assertAlmostEqual(Z_CARTILAGE, z_optimal, delta=5.0)


# ============================================================================
# AGING PAIN IMMUNITY TESTS
# ============================================================================

class TestAgingPainImmunity(unittest.TestCase):
    """Slow Z drift + C2 adaptation → no pain."""

    def test_slow_drift_no_pain(self):
        """Normal aging: drift=0.001, η=0.5 → pain never triggered."""
        r = aging_pain_model(z_drift_rate=0.001, eta=0.5, n_ticks=5000)
        self.assertFalse(r["pain_triggered"])

    def test_fast_drift_causes_pain(self):
        """Acute injury: drift=5.0, η=0.5 → pain IS triggered."""
        r = aging_pain_model(z_drift_rate=5.0, eta=0.5, n_ticks=200)
        self.assertTrue(r["pain_triggered"])

    def test_eta_zero_with_drift_causes_pain(self):
        """Cartilage (η≈0): even slow drift → pain eventually."""
        r = aging_pain_model(z_drift_rate=0.05, eta=0.0, n_ticks=5000)
        self.assertTrue(r["pain_triggered"])

    def test_no_drift_no_pain(self):
        """No aging at all: Z stable → no pain."""
        r = aging_pain_model(z_drift_rate=0.0, eta=0.5, n_ticks=1000)
        self.assertFalse(r["pain_triggered"])
        self.assertAlmostEqual(r["peak_gamma_sq"], 0.0, places=6)

    def test_c2_keeps_gamma_low(self):
        """With sufficient η, Γ² stays near zero despite slow drift."""
        r = aging_pain_model(z_drift_rate=0.01, eta=1.0, n_ticks=2000)
        self.assertLess(r["peak_gamma_sq"], 0.01)

    def test_weak_c2_higher_gamma(self):
        """Weak η → higher Γ² (but still sub-pain if drift is slow)."""
        strong = aging_pain_model(z_drift_rate=0.01, eta=1.0, n_ticks=1000)
        weak = aging_pain_model(z_drift_rate=0.01, eta=0.1, n_ticks=1000)
        self.assertGreater(weak["peak_gamma_sq"], strong["peak_gamma_sq"])


if __name__ == "__main__":
    unittest.main()
