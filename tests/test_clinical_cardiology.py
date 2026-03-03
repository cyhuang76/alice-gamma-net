"""test_clinical_cardiology.py - 10 cardiac diseases + engine."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from alice.body.clinical_cardiology import (
    ClinicalCardiologyEngine, MIModel, CHFModel, AFModel, HTNModel,
    AorticStenosisModel, CardiomyopathyModel, PericarditisModel,
    PulmHTNModel, EndocarditisModel, AorticDissectionModel,
)

class TestMI(unittest.TestCase):
    def test_tick(self):
        m = MIModel(territory="LAD", occlusion=0.95)
        r = m.tick()
        self.assertIn("total_gamma_sq", r)
        self.assertIn("troponin", r)
        self.assertIn("killip_class", r)
    def test_reperfuse(self):
        m = MIModel()
        m.tick()
        m.reperfuse(tick=1)
        r = m.tick()
        self.assertIn("total_gamma_sq", r)

class TestCHF(unittest.TestCase):
    def test_tick(self):
        m = CHFModel(initial_ef=0.30)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("ef", r)
        self.assertIn("nyha_class", r)
    def test_treatment(self):
        m = CHFModel()
        m.start_treatment()
        r = m.tick()
        self.assertIn("gamma_sq", r)

class TestAF(unittest.TestCase):
    def test_tick(self):
        m = AFModel(risk_factors=3)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("cha2ds2_vasc", r)
    def test_cardiovert(self):
        m = AFModel()
        m.trigger_af()
        m.tick()
        m.cardiovert()
        r = m.tick()
        self.assertIn("gamma_sq", r)

class TestHTN(unittest.TestCase):
    def test_tick(self):
        m = HTNModel(initial_sys=160.0, initial_dia=100.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("systolic", r)
        self.assertIn("diastolic", r)
        self.assertIn("jnc_stage", r)
    def test_treat(self):
        m = HTNModel(initial_sys=170.0)
        r0 = m.tick()
        m.treat()
        res = [m.tick() for _ in range(30)]
        self.assertLess(res[-1]["systolic"], r0["systolic"])

class TestAorticStenosis(unittest.TestCase):
    def test_tick(self):
        m = AorticStenosisModel(initial_valve_z=25.0, rate=0.003)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("gradient_mmhg", r)
        self.assertIn("severity", r)

class TestCardiomyopathy(unittest.TestCase):
    def test_tick(self):
        m = CardiomyopathyModel(subtype="dilated")
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("lvef", r)

class TestPericarditis(unittest.TestCase):
    def test_tick(self):
        m = PericarditisModel(effusion=100.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("effusion_ml", r)
        self.assertIn("tamponade_risk", r)

class TestPulmHTN(unittest.TestCase):
    def test_tick(self):
        m = PulmHTNModel(initial_pap=40.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("mean_pap", r)
        self.assertIn("who_class", r)

class TestEndocarditis(unittest.TestCase):
    def test_tick(self):
        m = EndocarditisModel(valve="mitral", virulence=0.6)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("duke_definite", r)
    def test_antibiotics(self):
        m = EndocarditisModel()
        m.start_antibiotics()
        r = m.tick()
        self.assertIn("gamma_sq", r)

class TestAorticDissection(unittest.TestCase):
    def test_tick(self):
        m = AorticDissectionModel(stanford="A", initial_tear=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("mortality_risk", r)
    def test_surgery(self):
        m = AorticDissectionModel(stanford="A")
        r0 = m.tick()
        m.emergency_surgery()
        res = [m.tick() for _ in range(10)]
        self.assertLess(res[-1]["gamma_sq"], r0["gamma_sq"])

class TestCardiologyEngine(unittest.TestCase):
    def test_multi(self):
        e = ClinicalCardiologyEngine()
        e.add_disease("mi", territory="LAD")
        e.add_disease("hypertension", initial_sys=160.0)
        r = e.tick()
        self.assertIn("total_gamma_sq", r)
        self.assertGreater(r["total_gamma_sq"], 0)

if __name__ == "__main__":
    unittest.main()
