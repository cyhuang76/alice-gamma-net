"""test_clinical_immunology.py - 10 immune diseases + engine."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from alice.body.clinical_immunology import (
    ClinicalImmunologyEngine, SLEModel, RAModel, AnaphylaxisModel,
    AllergicRhinitisModel, HIVModel, SepsisModel, TransplantRejectionModel,
    SarcoidosisModel, VasculitisModel, ImmunodeficiencyModel,
)

class TestSLE(unittest.TestCase):
    def test_tick(self):
        m = SLEModel(severity=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("sledai", r)
    def test_treatment(self):
        m = SLEModel(severity=0.6, flare=True)
        r0 = m.tick()
        m.start_treatment()
        res = [m.tick() for _ in range(20)]
        self.assertLessEqual(res[-1]["gamma_sq"], r0["gamma_sq"])

class TestRA(unittest.TestCase):
    def test_tick(self):
        m = RAModel(severity=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("das28", r)
    def test_dmard(self):
        m = RAModel()
        m.start_dmard()
        r = m.tick()
        self.assertIn("gamma_sq", r)

class TestAnaphylaxis(unittest.TestCase):
    def test_tick(self):
        m = AnaphylaxisModel(allergen_z=300.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("wao", r)
    def test_epinephrine(self):
        m = AnaphylaxisModel(allergen_z=400.0, sensitivity=0.9)
        r0 = m.tick()
        m.give_epinephrine()
        res = [m.tick() for _ in range(5)]
        self.assertLess(res[-1]["gamma_sq"], r0["gamma_sq"])

class TestAllergicRhinitis(unittest.TestCase):
    def test_tick(self):
        m = AllergicRhinitisModel()
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("ige", r)

class TestHIV(unittest.TestCase):
    def test_tick(self):
        m = HIVModel(initial_cd4=300.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("cd4", r)
    def test_art(self):
        m = HIVModel()
        r0 = m.tick()
        m.start_art()
        res = [m.tick() for _ in range(30)]
        self.assertLess(res[-1]["gamma_sq"], r0["gamma_sq"])

class TestSepsis(unittest.TestCase):
    def test_tick(self):
        m = SepsisModel(pathogen_z=300.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("sofa", r)
        self.assertIn("lactate", r)

class TestTransplant(unittest.TestCase):
    def test_tick(self):
        m = TransplantRejectionModel(hla_mismatch=3)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("banff", r)

class TestSarcoidosis(unittest.TestCase):
    def test_tick(self):
        m = SarcoidosisModel(severity=0.4)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("ace", r)

class TestVasculitis(unittest.TestCase):
    def test_tick(self):
        m = VasculitisModel(vessel_size="small")
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("bvas", r)

class TestImmunodeficiency(unittest.TestCase):
    def test_tick(self):
        m = ImmunodeficiencyModel(deficiency="cvid")
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("igg", r)
    def test_ivig(self):
        m = ImmunodeficiencyModel()
        m.start_ivig()
        r = m.tick()
        self.assertIn("gamma_sq", r)

class TestImmunologyEngine(unittest.TestCase):
    def test_multi(self):
        e = ClinicalImmunologyEngine()
        e.add_disease("sle", severity=0.4)
        e.add_disease("ra", severity=0.3)
        r = e.tick()
        self.assertIn("total_gamma_sq", r)

if __name__ == "__main__":
    unittest.main()
