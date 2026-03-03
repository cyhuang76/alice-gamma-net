"""test_clinical_ophthalmology.py - 10 eye diseases + engine."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from alice.body.clinical_ophthalmology import (
    ClinicalOphthalmologyEngine, GlaucomaModel, CataractModel,
    RetinalDetachmentModel, AMDModel, DiabeticRetinopathyModel,
    RefractiveModel, DryEyeModel, CornealUlcerModel,
    OpticNeuritisModel, StrabismusModel,
)

class TestGlaucoma(unittest.TestCase):
    def test_tick(self):
        m = GlaucomaModel(iop=28.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("iop", r)
        self.assertIn("vf_md", r)
    def test_drops(self):
        m = GlaucomaModel(iop=30.0)
        r0 = m.tick()
        m.start_drops()
        res = [m.tick() for _ in range(20)]
        self.assertLess(res[-1]["iop"], r0["iop"])

class TestCataract(unittest.TestCase):
    def test_tick(self):
        m = CataractModel(opacity=0.4)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("opacity", r)
    def test_surgery(self):
        m = CataractModel(opacity=0.5)
        r0 = m.tick()
        m.surgery()
        r = m.tick()
        self.assertLess(r["gamma_sq"], r0["gamma_sq"])

class TestRD(unittest.TestCase):
    def test_tick(self):
        m = RetinalDetachmentModel(initial_area=0.2)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("area", r)

class TestAMD(unittest.TestCase):
    def test_tick(self):
        m = AMDModel(stage=2)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("va", r)

class TestDR(unittest.TestCase):
    def test_tick(self):
        m = DiabeticRetinopathyModel(hba1c=9.0, duration=10)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("stage", r)

class TestRefractive(unittest.TestCase):
    def test_tick(self):
        m = RefractiveModel(diopters=-4.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("va_uc", r)
    def test_correction(self):
        m = RefractiveModel(diopters=-6.0)
        r0 = m.tick()
        m.apply_correction()
        res = [m.tick() for _ in range(5)]
        self.assertLessEqual(res[-1]["gamma_sq"], r0["gamma_sq"])

class TestDryEye(unittest.TestCase):
    def test_tick(self):
        m = DryEyeModel(severity=0.5)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("tbut", r)

class TestCornealUlcer(unittest.TestCase):
    def test_tick(self):
        m = CornealUlcerModel(size=3.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("size_mm", r)

class TestOpticNeuritis(unittest.TestCase):
    def test_tick(self):
        m = OpticNeuritisModel(severity=0.6)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("rapd", r)

class TestStrabismus(unittest.TestCase):
    def test_tick(self):
        m = StrabismusModel(deviation=20.0)
        r = m.tick()
        self.assertIn("gamma_sq", r)
        self.assertIn("deviation", r)

class TestOphthalmologyEngine(unittest.TestCase):
    def test_multi(self):
        e = ClinicalOphthalmologyEngine()
        e.add_disease("glaucoma", iop=26.0)
        e.add_disease("cataract", opacity=0.3)
        r = e.tick()
        self.assertIn("total_gamma_sq", r)

if __name__ == "__main__":
    unittest.main()
