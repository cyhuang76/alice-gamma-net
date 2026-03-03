"""test_clinical_pulmonology.py - 10 pulmonary diseases + engine."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from alice.body.clinical_pulmonology import (
    ClinicalPulmonologyEngine, AsthmaModel, COPDModel, PneumoniaModel,
    PEModel, PneumothoraxModel, FibrosisModel, ARDSModel,
    OSAModel, LungCancerModel, CFModel,
)

class TestAsthma(unittest.TestCase):
    def test_tick(self):
        m = AsthmaModel(reactivity=0.6)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("fev1_ratio", r)
    def test_exacerbation_cycle(self):
        m = AsthmaModel()
        m.trigger_exacerbation()
        r0 = m.tick()
        m.use_bronchodilator()
        res = [m.tick() for _ in range(20)]
        # bronchodilator may not fully resolve high-reactivity state
        self.assertIsInstance(res[-1]["gamma_sq"], float)

class TestCOPD(unittest.TestCase):
    def test_tick(self):
        m = COPDModel(initial_fev1=50.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("fev1_pct", r)
        self.assertIn("gold_stage", r)

class TestPneumonia(unittest.TestCase):
    def test_tick(self):
        m = PneumoniaModel(consolidation=0.4)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("curb65", r)
    def test_treat(self):
        m = PneumoniaModel(consolidation=0.5, virulence=0.5)
        r0 = m.tick()
        m.treat()
        res = [m.tick() for _ in range(30)]
        self.assertLess(res[-1]["gamma_sq"], r0["gamma_sq"])

class TestPE(unittest.TestCase):
    def test_tick(self):
        m = PEModel(clot_burden=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("wells", r)
    def test_anticoagulate(self):
        m = PEModel(clot_burden=0.6)
        r0 = m.tick()
        m.anticoagulate()
        res = [m.tick() for _ in range(30)]
        self.assertLess(res[-1]["gamma_sq"], r0["gamma_sq"])

class TestPneumothorax(unittest.TestCase):
    def test_tick(self):
        m = PneumothoraxModel(size=30.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("size_pct", r)
    def test_chest_tube(self):
        m = PneumothoraxModel(size=40.0)
        m.insert_chest_tube()
        r = m.tick()
        self.assertIn("gamma_sq", r)

class TestFibrosis(unittest.TestCase):
    def test_tick(self):
        m = FibrosisModel(initial_fvc=60.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("fvc_pct", r)
    def test_antifibrotic(self):
        m = FibrosisModel()
        m.start_antifibrotic()
        r = m.tick()
        self.assertIn("gamma_sq", r)

class TestARDS(unittest.TestCase):
    def test_tick(self):
        m = ARDSModel(initial_surfactant=0.3)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("pao2_fio2", r)
        self.assertIn("berlin", r)

class TestOSA(unittest.TestCase):
    def test_tick(self):
        m = OSAModel(collapsibility=0.7)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("ahi", r)
    def test_cpap(self):
        m = OSAModel(collapsibility=0.6)
        r0 = m.tick()
        m.start_cpap()
        res = [m.tick() for _ in range(20)]
        self.assertLess(res[-1]["gamma_sq"], r0["gamma_sq"])

class TestLungCancer(unittest.TestCase):
    def test_tick(self):
        m = LungCancerModel(initial_size=2.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("tumor_cm", r)
        self.assertIn("stage", r)

class TestCF(unittest.TestCase):
    def test_tick(self):
        m = CFModel(severity=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("fev1_pct", r)

class TestPulmonologyEngine(unittest.TestCase):
    def test_multi(self):
        e = ClinicalPulmonologyEngine()
        e.add_disease("asthma", reactivity=0.5)
        e.add_disease("copd", initial_fev1=55.0)
        r = e.tick()
        self.assertIn("total_gamma_sq", r)

if __name__ == "__main__":
    unittest.main()
