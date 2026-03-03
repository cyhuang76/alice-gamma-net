"""test_clinical_gastroenterology.py - 10 GI diseases + engine."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from alice.body.clinical_gastroenterology import (
    ClinicalGastroenterologyEngine, GERDModel, PepticUlcerModel, IBDModel,
    IBSModel, CirrhosisModel, CholelithiasisModel, PancreatitisModel,
    BowelObstructionModel, CRCModel, HepatitisModel,
)

class TestGERD(unittest.TestCase):
    def test_tick(self):
        m = GERDModel(les_weakness=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("la_grade", r)

class TestPepticUlcer(unittest.TestCase):
    def test_tick(self):
        m = PepticUlcerModel(h_pylori=True)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("forrest", r)
    def test_ppi(self):
        m = PepticUlcerModel()
        m.start_ppi()
        r = m.tick()
        self.assertIn("gamma_sq", r)

class TestIBD(unittest.TestCase):
    def test_tick(self):
        m = IBDModel(subtype="crohn", severity=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("score", r)
    def test_biologic(self):
        m = IBDModel(subtype="uc", severity=0.5)
        m.start_biologic()
        r = m.tick()
        self.assertIn("gamma_sq", r)

class TestIBS(unittest.TestCase):
    def test_tick(self):
        m = IBSModel(subtype="mixed")
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("gut_brain_gamma", r)

class TestCirrhosis(unittest.TestCase):
    def test_tick(self):
        m = CirrhosisModel(initial_fibrosis=3)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("child_pugh", r)
        self.assertIn("meld", r)

class TestCholelithiasis(unittest.TestCase):
    def test_tick(self):
        m = CholelithiasisModel(stone_size=10.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("stone_mm", r)

class TestPancreatitis(unittest.TestCase):
    def test_tick(self):
        m = PancreatitisModel(cause="gallstone")
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("ranson", r)
    def test_treat(self):
        m = PancreatitisModel()
        m.treat()
        r = m.tick()
        self.assertIn("gamma_sq", r)

class TestBowelObstruction(unittest.TestCase):
    def test_tick(self):
        m = BowelObstructionModel(level="small", complete=True)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("distension", r)
    def test_operate(self):
        m = BowelObstructionModel()
        m.operate()
        r = m.tick()
        self.assertIn("gamma_sq", r)

class TestCRC(unittest.TestCase):
    def test_tick(self):
        m = CRCModel(initial_size=2.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("cea", r)
        self.assertIn("stage", r)

class TestHepatitis(unittest.TestCase):
    def test_tick(self):
        m = HepatitisModel(viral_type="B", initial_load=1e6)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("viral_load", r)
    def test_antiviral(self):
        m = HepatitisModel()
        r0 = m.tick()
        m.start_antiviral()
        res = [m.tick() for _ in range(30)]
        self.assertLess(res[-1]["viral_load"], r0["viral_load"])

class TestGastroEngine(unittest.TestCase):
    def test_multi(self):
        e = ClinicalGastroenterologyEngine()
        e.add_disease("gerd", les_weakness=0.5)
        e.add_disease("cirrhosis", initial_fibrosis=2)
        r = e.tick()
        self.assertIn("total_gamma_sq", r)

if __name__ == "__main__":
    unittest.main()
