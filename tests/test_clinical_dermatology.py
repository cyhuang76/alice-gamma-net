"""test_clinical_dermatology.py - 10 derm diseases + engine."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from alice.body.clinical_dermatology import (
    ClinicalDermatologyEngine, AtopicDermatitisModel, PsoriasisModel,
    UrticariaModel, HerpesZosterModel, MelanomaModel, ContactDermatitisModel,
    AcneModel, VitiligoModel, CellulitisModel, BurnsModel,
)

class TestAD(unittest.TestCase):
    def test_tick(self):
        m = AtopicDermatitisModel(severity=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("scorad", r)
    def test_emollient(self):
        m = AtopicDermatitisModel()
        m.start_emollient()
        r = m.tick()
        self.assertIn("gamma_sq", r)

class TestPsoriasis(unittest.TestCase):
    def test_tick(self):
        m = PsoriasisModel(severity=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("pasi", r)
    def test_biologic(self):
        m = PsoriasisModel(severity=0.5, area=20.0)
        r0 = m.tick()
        m.start_treatment()
        res = [m.tick() for _ in range(20)]
        self.assertLess(res[-1]["gamma_sq"], r0["gamma_sq"])

class TestUrticaria(unittest.TestCase):
    def test_tick(self):
        m = UrticariaModel(severity=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("uas7", r)

class TestZoster(unittest.TestCase):
    def test_tick(self):
        m = HerpesZosterModel(age=70)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("phn_risk", r)

class TestMelanoma(unittest.TestCase):
    def test_tick(self):
        m = MelanomaModel(breslow=1.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("breslow", r)
    def test_excision(self):
        m = MelanomaModel()
        m.excision()
        r = m.tick()
        self.assertIn("gamma_sq", r)

class TestContactDerm(unittest.TestCase):
    def test_tick(self):
        m = ContactDermatitisModel()
        r = m.tick()
        self.assertIn("gamma_sq", r)

class TestAcne(unittest.TestCase):
    def test_tick(self):
        m = AcneModel(severity=0.4)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("iga", r)

class TestVitiligo(unittest.TestCase):
    def test_tick(self):
        m = VitiligoModel(severity=0.4)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("vasi", r)

class TestCellulitis(unittest.TestCase):
    def test_tick(self):
        m = CellulitisModel(severity=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("eron", r)

class TestBurns(unittest.TestCase):
    def test_tick(self):
        m = BurnsModel(tbsa=20.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("fluid_ml", r)
    def test_resuscitation(self):
        m = BurnsModel(tbsa=25.0, depth="full")
        r0 = m.tick()
        m.start_resuscitation()
        res = [m.tick() for _ in range(20)]
        self.assertLessEqual(res[-1]["gamma_sq"], r0["gamma_sq"])

class TestDermEngine(unittest.TestCase):
    def test_multi(self):
        e = ClinicalDermatologyEngine()
        e.add_disease("atopic", severity=0.3)
        e.add_disease("psoriasis", severity=0.3)
        r = e.tick()
        self.assertIn("total_gamma_sq", r)

if __name__ == "__main__":
    unittest.main()
