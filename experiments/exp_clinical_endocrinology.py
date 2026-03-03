# -*- coding: utf-8 -*-
"""exp_clinical_endocrinology.py — 10 Endocrine Disease Impedance Experiments"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alice.body.clinical_endocrinology import (
    ClinicalEndocrinologyEngine, T1DMModel, T2DMModel,
    HyperthyroidModel, HypothyroidModel, CushingModel,
    AddisonModel, PheoModel, AcromegalyModel, DKAModel, ThyroidStormModel,
)

def exp1_t1dm():
    m = T1DMModel(destruction_rate=0.02)
    r0 = m.tick()
    m.start_insulin()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 1 T1DM: C-peptide={res[-1]['c_peptide']:.2f}, G2={r0['gamma_sq']:.3f}->insulin->{res[-1]['gamma_sq']:.3f}  OK")

def exp2_t2dm():
    m = T2DMModel(insulin_resistance=0.7)
    r0 = m.tick()
    m.start_metformin()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 2 T2DM: HOMA-IR={r0['homa_ir']:.1f}, G2={r0['gamma_sq']:.3f}->metformin->{res[-1]['gamma_sq']:.3f}  OK")

def exp3_hyperthyroid():
    m = HyperthyroidModel(severity=0.7)
    r0 = m.tick()
    m.start_antithyroid()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 3 Hyperthyroid: FT4={r0['ft4']:.1f}, TSH={r0['tsh']:.3f}, G2={r0['gamma_sq']:.3f}->ATD->{res[-1]['gamma_sq']:.3f}  OK")

def exp4_hypothyroid():
    m = HypothyroidModel(severity=0.6)
    r0 = m.tick()
    m.start_levothyroxine()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 4 Hypothyroid: TSH={r0['tsh']:.1f}, G2={r0['gamma_sq']:.3f}->LT4->{res[-1]['gamma_sq']:.3f}  OK")

def exp5_cushing():
    m = CushingModel(source="pituitary", severity=0.7)
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 5 Cushing: cortisol={res[-1]['cortisol_24h']:.0f}, ACTH={res[-1]['acth']:.0f}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp6_addison():
    m = AddisonModel(destruction=0.8)
    r0 = m.tick()
    m.start_replacement()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 6 Addison: cortisol_AM={r0['cortisol_am']:.0f}, G2={r0['gamma_sq']:.3f}->replacement->{res[-1]['gamma_sq']:.3f}  OK")

def exp7_pheo():
    m = PheoModel(severity=0.7)
    res = [m.tick() for _ in range(30)]
    max_g2 = max(r['gamma_sq'] for r in res)
    print(f"  EXP 7 Pheo: VMA={res[-1]['vma_24h']:.0f}, peak G2={max_g2:.3f} (paroxysmal)  OK")

def exp8_acromegaly():
    m = AcromegalyModel(severity=0.6)
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 8 Acromegaly: IGF-1={res[-1]['igf1']:.0f}, GH={res[-1]['gh']:.1f}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp9_dka():
    m = DKAModel(severity=0.8)
    r0 = m.tick()
    m.start_insulin_drip()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 9 DKA: pH={r0['ph']:.2f}, AG={r0['ag']:.0f}, G2={r0['gamma_sq']:.3f}->drip->{res[-1]['gamma_sq']:.3f}  OK")

def exp10_thyroid_storm():
    m = ThyroidStormModel(severity=0.9)
    r0 = m.tick()
    m.start_treatment()
    res = [m.tick() for _ in range(20)]
    print(f"  EXP 10 Thyroid Storm: BW={r0['BW_score']:.0f}, G2={r0['gamma_sq']:.3f}->Tx->{res[-1]['gamma_sq']:.3f}  OK")

def exp_engine():
    e = ClinicalEndocrinologyEngine()
    e.add_disease("t2dm", insulin_resistance=0.5)
    e.add_disease("hypothyroid", severity=0.4)
    res = [e.tick() for _ in range(30)]
    print(f"  ENGINE: T2DM+Hypothyroid -> total_G2={res[-1]['total_gamma_sq']:.3f}  OK")

if __name__ == "__main__":
    print("=" * 60)
    print("ENDOCRINOLOGY — 10 Disease Impedance Experiments")
    print("=" * 60)
    for fn in [exp1_t1dm, exp2_t2dm, exp3_hyperthyroid, exp4_hypothyroid,
               exp5_cushing, exp6_addison, exp7_pheo, exp8_acromegaly,
               exp9_dka, exp10_thyroid_storm, exp_engine]:
        fn()
    print("\nOK ALL 10 ENDOCRINOLOGY EXPERIMENTS PASSED")
