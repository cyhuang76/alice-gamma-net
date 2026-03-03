"""test_clinical_nephrology.py - 10 renal diseases + engine."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from alice.body.clinical_nephrology import (
    ClinicalNephrologyEngine, AKIModel, CKDModel, NephrolithiasisModel,
    NephroticModel, NephriticModel, DiabeticNephropathyModel,
    ElectrolyteModel, RenalHTNModel, PKDModel, RTAModel,
)

class TestAKI(unittest.TestCase):
    def test_tick(self):
        m = AKIModel(insult_severity=0.7)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("kdigo", r)
        self.assertIn("creatinine", r)
    def test_recovery(self):
        m = AKIModel()
        r0 = m.tick()
        m.start_recovery()
        res = [m.tick() for _ in range(20)]
        self.assertLess(res[-1]["gamma_sq"], r0["gamma_sq"])

class TestCKD(unittest.TestCase):
    def test_tick(self):
        m = CKDModel(initial_gfr=40.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("gfr", r)
        self.assertIn("stage", r)

class TestNephrolithiasis(unittest.TestCase):
    def test_tick(self):
        m = NephrolithiasisModel(stone_size=8.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("stone_mm", r)

class TestNephrotic(unittest.TestCase):
    def test_tick(self):
        m = NephroticModel(barrier_damage=0.6)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("proteinuria", r)
        self.assertIn("albumin", r)

class TestNephritic(unittest.TestCase):
    def test_tick(self):
        m = NephriticModel(inflammation=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("hematuria", r)

class TestDN(unittest.TestCase):
    def test_tick(self):
        m = DiabeticNephropathyModel(hba1c=9.0, duration_years=10)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("uacr", r)

class TestElectrolyte(unittest.TestCase):
    def test_tick(self):
        m = ElectrolyteModel(disorder="hyperkalemia")
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("k", r)
        self.assertIn("na", r)

class TestRenalHTN(unittest.TestCase):
    def test_tick(self):
        m = RenalHTNModel(stenosis=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("bp", r)
    def test_acei(self):
        m = RenalHTNModel(stenosis=0.5)
        r0 = m.tick()
        m.start_ace_inhibitor()
        res = [m.tick() for _ in range(20)]
        self.assertLessEqual(res[-1]["gamma_sq"], r0["gamma_sq"])

class TestPKD(unittest.TestCase):
    def test_tick(self):
        m = PKDModel(initial_volume=600.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("volume_ml", r)

class TestRTA(unittest.TestCase):
    def test_tick(self):
        m = RTAModel(rta_type=1, severity=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("ph", r)
        self.assertIn("hco3", r)

class TestNephrologyEngine(unittest.TestCase):
    def test_multi(self):
        e = ClinicalNephrologyEngine()
        e.add_disease("ckd", initial_gfr=40.0)
        e.add_disease("electrolyte", disorder="hyponatremia")
        r = e.tick()
        self.assertIn("total_gamma_sq", r)

if __name__ == "__main__":
    unittest.main()
