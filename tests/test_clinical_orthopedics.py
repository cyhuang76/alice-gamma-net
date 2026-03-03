"""test_clinical_orthopedics.py - 10 orthopedic diseases + engine."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from alice.body.clinical_orthopedics import (
    ClinicalOrthopedicsEngine, FractureModel, OsteoporosisModel,
    DiscHerniationModel, OsteoarthritisModel, ACLModel,
    TendinitisModel, ScoliosisModel, OsteosarcomaModel,
    GoutModel, OsteomyelitisModel,
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

if __name__ == "__main__":
    unittest.main()
