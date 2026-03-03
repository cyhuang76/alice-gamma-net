# -*- coding: utf-8 -*-
"""exp_clinical_ophthalmology.py — 10 Eye Disease Impedance Experiments"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alice.body.clinical_ophthalmology import (
    ClinicalOphthalmologyEngine, GlaucomaModel, CataractModel,
    RetinalDetachmentModel, AMDModel, DiabeticRetinopathyModel,
    RefractiveModel, DryEyeModel, CornealUlcerModel,
    OpticNeuritisModel, StrabismusModel,
)

def exp1_glaucoma():
    m = GlaucomaModel(iop=30.0, damage_rate=0.008)
    r0 = m.tick()
    m.start_drops()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 1 Glaucoma: IOP={r0['iop']:.0f}->drops->{res[-1]['iop']:.0f}, VF-MD={res[-1]['vf_md']:.1f}, G2={r0['gamma_sq']:.3f}->{res[-1]['gamma_sq']:.3f}  OK")

def exp2_cataract():
    m = CataractModel(opacity=0.5, age_progression=0.002)
    r0 = m.tick()
    m.surgery()
    res = [m.tick() for _ in range(10)]
    print(f"  EXP 2 Cataract: opacity={r0['opacity']:.2f}->surgery->{res[-1]['opacity']:.2f}, VA={res[-1]['va']}, G2={r0['gamma_sq']:.3f}->{res[-1]['gamma_sq']:.3f}  OK")

def exp3_rd():
    m = RetinalDetachmentModel(initial_area=0.2, progression=0.03)
    r0 = m.tick()
    m.surgery_repair()
    res = [m.tick() for _ in range(20)]
    print(f"  EXP 3 RD: area={r0['area']:.2f}->repair->{res[-1]['area']:.2f}, G2={r0['gamma_sq']:.3f}->{res[-1]['gamma_sq']:.3f}  OK")

def exp4_amd():
    m = AMDModel(stage=3, wet=True)
    r0 = m.tick()
    m.start_anti_vegf()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 4 AMD wet: VA={r0['va']}, G2={r0['gamma_sq']:.3f}->anti-VEGF->{res[-1]['gamma_sq']:.3f}  OK")

def exp5_dr():
    m = DiabeticRetinopathyModel(hba1c=10.0, duration=15)
    res = [m.tick() for _ in range(40)]
    m.start_anti_vegf()
    res2 = [m.tick() for _ in range(20)]
    print(f"  EXP 5 DR: stage={res[-1]['stage']}, G2={res[-1]['gamma_sq']:.3f}->VEGF->{res2[-1]['gamma_sq']:.3f}  OK")

def exp6_refractive():
    m = RefractiveModel(diopters=-6.0)
    r0 = m.tick()
    m.apply_correction()
    res = [m.tick() for _ in range(5)]
    print(f"  EXP 6 Myopia -6D: VA_uc={r0['va_uc']:.2f}->correction->VA_c={res[-1]['va_c']:.2f}, G2={r0['gamma_sq']:.3f}->{res[-1]['gamma_sq']:.3f}  OK")

def exp7_dry_eye():
    m = DryEyeModel(severity=0.6, evaporative=True)
    r0 = m.tick()
    m.start_treatment()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 7 Dry Eye: OSDI={r0['osdi']:.0f}, TBUT={r0['tbut']:.1f}, G2={r0['gamma_sq']:.3f}->Tx->{res[-1]['gamma_sq']:.3f}  OK")

def exp8_corneal_ulcer():
    m = CornealUlcerModel(size=4.0, depth=0.5)
    r0 = m.tick()
    m.start_antibiotics()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 8 Corneal Ulcer: {r0['size_mm']:.0f}mm, G2={r0['gamma_sq']:.3f}->Abx->{res[-1]['gamma_sq']:.3f}  OK")

def exp9_on():
    m = OpticNeuritisModel(severity=0.7, ms_associated=True)
    r0 = m.tick()
    m.start_steroids()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 9 ON: VA={r0['va']}, RAPD={r0['rapd']}, G2={r0['gamma_sq']:.3f}->steroids->{res[-1]['gamma_sq']:.3f}  OK")

def exp10_strabismus():
    m = StrabismusModel(deviation=25.0, direction="esotropia")
    r0 = m.tick()
    m.surgery_or_prism()
    res = [m.tick() for _ in range(20)]
    print(f"  EXP 10 Strabismus: {r0['deviation']:.0f}Δ, G2={r0['gamma_sq']:.3f}->surgery->{res[-1]['gamma_sq']:.3f}  OK")

def exp_engine():
    e = ClinicalOphthalmologyEngine()
    e.add_disease("glaucoma", iop=26.0)
    e.add_disease("dry_eye", severity=0.3)
    res = [e.tick() for _ in range(30)]
    print(f"  ENGINE: Glaucoma+DryEye -> total_G2={res[-1]['total_gamma_sq']:.3f}  OK")

if __name__ == "__main__":
    print("=" * 60)
    print("OPHTHALMOLOGY — 10 Disease Impedance Experiments")
    print("=" * 60)
    for fn in [exp1_glaucoma, exp2_cataract, exp3_rd, exp4_amd, exp5_dr,
               exp6_refractive, exp7_dry_eye, exp8_corneal_ulcer, exp9_on,
               exp10_strabismus, exp_engine]:
        fn()
    print("\nOK ALL 10 OPHTHALMOLOGY EXPERIMENTS PASSED")
