"""test_clinical_oncology.py - 10 oncology diseases + engine."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from alice.body.clinical_oncology import (
    ClinicalOncologyEngine, LungCancerModel, BreastCancerModel,
    CRCModel, HCCModel, PancreaticCancerModel, GBMModel,
    LeukemiaModel, LymphomaModel, RCCModel, MetastasisModel,
)

class TestLungCa(unittest.TestCase):
    def test_tick(self):
        m = LungCancerModel(size=2.0, egfr=True)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("stage", r)
    def test_treatment(self):
        m = LungCancerModel(size=3.0, egfr=True)
        r0 = m.tick()
        m.start_treatment()
        res = [m.tick() for _ in range(20)]
        self.assertLess(res[-1]["gamma_sq"], r0["gamma_sq"])

class TestBreastCa(unittest.TestCase):
    def test_tick(self):
        m = BreastCancerModel(size=2.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("ER", r)

class TestCRC(unittest.TestCase):
    def test_tick(self):
        m = CRCModel(size=3.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("cea", r)

class TestHCC(unittest.TestCase):
    def test_tick(self):
        m = HCCModel(size=4.0, cirrhotic=True)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("bclc", r)

class TestPancreatic(unittest.TestCase):
    def test_tick(self):
        m = PancreaticCancerModel(size=3.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("ca19_9", r)

class TestGBM(unittest.TestCase):
    def test_tick(self):
        m = GBMModel(size=4.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("kps", r)

class TestLeukemia(unittest.TestCase):
    def test_tick(self):
        m = LeukemiaModel(subtype="AML", blast=60.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("blasts", r)

class TestLymphoma(unittest.TestCase):
    def test_tick(self):
        m = LymphomaModel(subtype="DLBCL")
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("ldh", r)

class TestRCC(unittest.TestCase):
    def test_tick(self):
        m = RCCModel(size=5.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("imdc", r)

class TestMetastasis(unittest.TestCase):
    def test_tick(self):
        m = MetastasisModel(primary="lung", initial_sites=["brain"])
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("camouflage", r)
    def test_immunotherapy(self):
        m = MetastasisModel(primary="lung", camouflage=0.9)
        r0 = m.tick()
        m.start_immunotherapy()
        res = [m.tick() for _ in range(20)]
        self.assertLess(res[-1]["camouflage"], r0["camouflage"])

class TestOncologyEngine(unittest.TestCase):
    def test_multi(self):
        e = ClinicalOncologyEngine()
        e.add_disease("lung_cancer", size=2.0)
        e.add_disease("metastasis", primary="lung")
        r = e.tick()
        self.assertIn("total_gamma_sq", r)

if __name__ == "__main__":
    unittest.main()
