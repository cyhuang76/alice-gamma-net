"""test_clinical_obstetrics.py - 10 OB/GYN diseases + engine."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from alice.body.clinical_obstetrics import (
    ClinicalObstetricsEngine, PreeclampsiaModel, PCOSModel,
    EndometriosisModel, FibroidModel, PretermBirthModel,
    GDMModel, OvarianCancerModel, MenopauseModel, AFEModel, PPHModel,
)

class TestPreeclampsia(unittest.TestCase):
    def test_tick(self):
        m = PreeclampsiaModel(severity=0.5, ga=32.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("bp", r)
        self.assertIn("proteinuria", r)

class TestPCOS(unittest.TestCase):
    def test_tick(self):
        m = PCOSModel(severity=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("amh", r)

class TestEndometriosis(unittest.TestCase):
    def test_tick(self):
        m = EndometriosisModel(severity=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("pain", r)

class TestFibroid(unittest.TestCase):
    def test_tick(self):
        m = FibroidModel(size_cm=5.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("size_cm", r)

class TestPreterm(unittest.TestCase):
    def test_tick(self):
        m = PretermBirthModel(cervical_length=20.0, ga=28.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("cl_mm", r)
    def test_progesterone(self):
        m = PretermBirthModel()
        m.start_progesterone()
        r = m.tick()
        self.assertIn("gamma_sq", r)

class TestGDM(unittest.TestCase):
    def test_tick(self):
        m = GDMModel(severity=0.4)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("fasting", r)

class TestOvarianCancer(unittest.TestCase):
    def test_tick(self):
        m = OvarianCancerModel(stage="II")
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("ca125", r)

class TestMenopause(unittest.TestCase):
    def test_tick(self):
        m = MenopauseModel(years_post=2.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("fsh", r)
    def test_hrt(self):
        m = MenopauseModel()
        r0 = m.tick()
        m.start_hrt()
        res = [m.tick() for _ in range(20)]
        self.assertLess(res[-1]["gamma_sq"], r0["gamma_sq"])

class TestAFE(unittest.TestCase):
    def test_tick(self):
        m = AFEModel(severity=0.8)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("dic", r)

class TestPPH(unittest.TestCase):
    def test_tick(self):
        m = PPHModel(cause="atony", severity=0.6)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("ebl_ml", r)
    def test_uterotonics(self):
        m = PPHModel()
        r0 = m.tick()
        m.give_uterotonics()
        res = [m.tick() for _ in range(10)]
        self.assertLessEqual(res[-1]["gamma_sq"], r0["gamma_sq"])

class TestObstetricsEngine(unittest.TestCase):
    def test_multi(self):
        e = ClinicalObstetricsEngine()
        e.add_disease("preeclampsia", severity=0.4, ga=32.0)
        e.add_disease("gdm", severity=0.3)
        r = e.tick()
        self.assertIn("total_gamma_sq", r)

if __name__ == "__main__":
    unittest.main()
