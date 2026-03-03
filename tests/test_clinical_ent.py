"""test_clinical_ent.py - 10 ENT diseases + engine."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from alice.body.clinical_ent import (
    ClinicalENTEngine, SNHLModel, ConductiveHLModel, MeniereModel,
    TinnitusModel, OtitisMediaModel, VocalCordParalysisModel,
    SinusitisModel, AnosmiaModel, SSHLModel, BPPVModel,
)

class TestSNHL(unittest.TestCase):
    def test_tick(self):
        m = SNHLModel(severity_db=50.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("pta_db", r)
    def test_hearing_aid(self):
        m = SNHLModel()
        m.apply_hearing_aid()
        r = m.tick()
        self.assertIn("gamma_sq", r)

class TestCHL(unittest.TestCase):
    def test_tick(self):
        m = ConductiveHLModel(cause="otosclerosis")
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("air_bone_gap", r)

class TestMeniere(unittest.TestCase):
    def test_tick(self):
        m = MeniereModel(severity=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("stage", r)

class TestTinnitus(unittest.TestCase):
    def test_tick(self):
        m = TinnitusModel(severity=0.4)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("thi", r)
    def test_trt(self):
        m = TinnitusModel()
        m.start_trt()
        r = m.tick()
        self.assertIn("gamma_sq", r)

class TestOtitisMedia(unittest.TestCase):
    def test_tick(self):
        m = OtitisMediaModel(effusion=True)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("tymp", r)

class TestVocalCord(unittest.TestCase):
    def test_tick(self):
        m = VocalCordParalysisModel()
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("vhi", r)

class TestSinusitis(unittest.TestCase):
    def test_tick(self):
        m = SinusitisModel(chronic=True, severity=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("lund_mackay", r)

class TestAnosmia(unittest.TestCase):
    def test_tick(self):
        m = AnosmiaModel(severity=0.7)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("upsit", r)

class TestSSHL(unittest.TestCase):
    def test_tick(self):
        m = SSHLModel(severity_db=55.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("pta_db", r)
    def test_steroids(self):
        m = SSHLModel()
        r0 = m.tick()
        m.start_steroids()
        res = [m.tick() for _ in range(20)]
        self.assertIn("recovery", res[-1])

class TestBPPV(unittest.TestCase):
    def test_tick(self):
        m = BPPVModel(canal="posterior")
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("nystagmus", r)
    def test_epley(self):
        m = BPPVModel()
        r0 = m.tick()
        m.do_epley()
        res = [m.tick() for _ in range(10)]
        self.assertLess(res[-1]["gamma_sq"], r0["gamma_sq"])

class TestENTEngine(unittest.TestCase):
    def test_multi(self):
        e = ClinicalENTEngine()
        e.add_disease("snhl", severity_db=40.0)
        e.add_disease("tinnitus", severity=0.3)
        r = e.tick()
        self.assertIn("total_gamma_sq", r)

if __name__ == "__main__":
    unittest.main()
