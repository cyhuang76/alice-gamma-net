"""test_clinical_endocrinology.py - 10 endocrine diseases + engine."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from alice.body.clinical_endocrinology import (
    ClinicalEndocrinologyEngine, T1DMModel, T2DMModel,
    HyperthyroidModel, HypothyroidModel, CushingModel,
    AddisonModel, PheoModel, AcromegalyModel, DKAModel, ThyroidStormModel,
)

class TestT1DM(unittest.TestCase):
    def test_tick(self):
        m = T1DMModel(destruction_rate=0.01)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("glucose", r)
        self.assertIn("c_peptide", r)
    def test_insulin(self):
        m = T1DMModel()
        r0 = m.tick()
        m.start_insulin()
        res = [m.tick() for _ in range(20)]
        self.assertLess(res[-1]["glucose"], r0["glucose"])

class TestT2DM(unittest.TestCase):
    def test_tick(self):
        m = T2DMModel(insulin_resistance=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("homa_ir", r)
    def test_metformin(self):
        m = T2DMModel(insulin_resistance=0.6)
        r0 = m.tick()
        m.start_metformin()
        res = [m.tick() for _ in range(20)]
        self.assertLess(res[-1]["gamma_sq"], r0["gamma_sq"])

class TestHyperthyroid(unittest.TestCase):
    def test_tick(self):
        m = HyperthyroidModel(severity=0.6)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("ft4", r)
        self.assertIn("tsh", r)
    def test_antithyroid(self):
        m = HyperthyroidModel()
        m.start_antithyroid()
        r = m.tick()
        self.assertIn("gamma_sq", r)

class TestHypothyroid(unittest.TestCase):
    def test_tick(self):
        m = HypothyroidModel(severity=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("tsh", r)
    def test_levothyroxine(self):
        m = HypothyroidModel()
        m.start_levothyroxine()
        r = m.tick()
        self.assertIn("gamma_sq", r)

class TestCushing(unittest.TestCase):
    def test_tick(self):
        m = CushingModel(source="pituitary", severity=0.6)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("cortisol_24h", r)
        self.assertIn("acth", r)

class TestAddison(unittest.TestCase):
    def test_tick(self):
        m = AddisonModel(destruction=0.7)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("cortisol_am", r)
    def test_replacement(self):
        m = AddisonModel()
        r0 = m.tick()
        m.start_replacement()
        res = [m.tick() for _ in range(20)]
        self.assertLess(res[-1]["gamma_sq"], r0["gamma_sq"])

class TestPheo(unittest.TestCase):
    def test_tick(self):
        m = PheoModel(severity=0.6)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("vma_24h", r)

class TestAcromegaly(unittest.TestCase):
    def test_tick(self):
        m = AcromegalyModel(severity=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("igf1", r)

class TestDKA(unittest.TestCase):
    def test_tick(self):
        m = DKAModel(severity=0.7)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("ph", r)
        self.assertIn("ag", r)
    def test_drip(self):
        m = DKAModel(severity=0.8)
        r0 = m.tick()
        m.start_insulin_drip()
        res = [m.tick() for _ in range(20)]
        self.assertLess(res[-1]["gamma_sq"], r0["gamma_sq"])

class TestThyroidStorm(unittest.TestCase):
    def test_tick(self):
        m = ThyroidStormModel(severity=0.8)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("BW_score", r)
    def test_treatment(self):
        m = ThyroidStormModel()
        m.start_treatment()
        r = m.tick()
        self.assertIn("gamma_sq", r)

class TestEndoEngine(unittest.TestCase):
    def test_multi(self):
        e = ClinicalEndocrinologyEngine()
        e.add_disease("t2dm", insulin_resistance=0.5)
        e.add_disease("hypothyroid", severity=0.3)
        r = e.tick()
        self.assertIn("total_gamma_sq", r)

if __name__ == "__main__":
    unittest.main()
