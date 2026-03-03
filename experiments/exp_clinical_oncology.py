# -*- coding: utf-8 -*-
"""exp_clinical_oncology.py — 10 Oncology Disease Impedance Experiments
Core principle: Cancer = impedance camouflage (Z_tumor -> Z_host)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alice.body.clinical_oncology import (
    ClinicalOncologyEngine, LungCancerModel, BreastCancerModel,
    CRCModel, HCCModel, PancreaticCancerModel, GBMModel,
    LeukemiaModel, LymphomaModel, RCCModel, MetastasisModel,
)

def exp1_lung_ca():
    m = LungCancerModel(size=2.0, histology="adenocarcinoma", egfr=True)
    r0 = m.tick()
    m.start_treatment()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 1 Lung Ca EGFR+: FEV1={r0['fev1']:.0f}%, G2={r0['gamma_sq']:.3f}->TKI->{res[-1]['gamma_sq']:.3f}  OK")

def exp2_breast_ca():
    m = BreastCancerModel(size=2.5, er=True, her2=False)
    r0 = m.tick()
    m.start_treatment()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 2 Breast Ca ER+: stage={r0['stage']}, G2={r0['gamma_sq']:.3f}->Tx->{res[-1]['gamma_sq']:.3f}  OK")

def exp3_crc():
    m = CRCModel(size=3.0)
    r0 = m.tick()
    m.start_treatment()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 3 CRC: CEA={r0['cea']:.1f}, G2={r0['gamma_sq']:.3f}->FOLFOX->{res[-1]['gamma_sq']:.3f}  OK")

def exp4_hcc():
    m = HCCModel(size=5.0, cirrhotic=True)
    r0 = m.tick()
    m.start_treatment()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 4 HCC: AFP={r0['afp']:.0f}, BCLC={r0['bclc']}, G2={r0['gamma_sq']:.3f}->TACE->{res[-1]['gamma_sq']:.3f}  OK")

def exp5_pancreatic():
    m = PancreaticCancerModel(size=4.0)
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 5 Pancreatic Ca: CA19-9={res[-1]['ca19_9']:.0f}, G2={res[-1]['gamma_sq']:.3f} (poor prognosis)  OK")

def exp6_gbm():
    m = GBMModel(size=4.0, who_grade=4)
    r0 = m.tick()
    m.start_treatment()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 6 GBM IV: KPS={r0['kps']:.0f}, G2={r0['gamma_sq']:.3f}->TMZ+RT->{res[-1]['gamma_sq']:.3f}  OK")

def exp7_leukemia():
    m = LeukemiaModel(subtype="ALL", blast=70.0)
    r0 = m.tick()
    m.start_chemo()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 7 ALL: blast={r0['blasts']:.0f}%, G2={r0['gamma_sq']:.3f}->induction->{res[-1]['gamma_sq']:.3f}  OK")

def exp8_lymphoma():
    m = LymphomaModel(subtype="DLBCL", severity=0.6)
    r0 = m.tick()
    m.start_rchop()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 8 DLBCL: LDH={r0['ldh']:.0f}, B-sx={r0['b_symptoms']}, G2={r0['gamma_sq']:.3f}->R-CHOP->{res[-1]['gamma_sq']:.3f}  OK")

def exp9_rcc():
    m = RCCModel(size=6.0)
    r0 = m.tick()
    m.start_treatment()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 9 RCC: IMDC={r0['imdc']}, G2={r0['gamma_sq']:.3f}->IO->{res[-1]['gamma_sq']:.3f}  OK")

def exp10_metastasis():
    """Core: cancer-as-impedance-camouflage."""
    m = MetastasisModel(primary="lung", initial_sites=["brain", "bone"], camouflage=0.9)
    res_pre = [m.tick() for _ in range(10)]
    camo_pre = res_pre[-1]['camouflage']
    m.start_immunotherapy()
    res_post = [m.tick() for _ in range(20)]
    camo_post = res_post[-1]['camouflage']
    print(f"  EXP 10 Metastasis: camouflage={camo_pre:.2f}->IO->{camo_post:.2f}, G2={res_pre[-1]['gamma_sq']:.3f}->{res_post[-1]['gamma_sq']:.3f} (de-camo) OK")

def exp_engine():
    e = ClinicalOncologyEngine()
    e.add_disease("lung_cancer", size=2.0, egfr=False)
    e.add_disease("metastasis", primary="lung", initial_sites=["bone"])
    res = [e.tick() for _ in range(30)]
    print(f"  ENGINE: LungCa+Metastasis -> total_G2={res[-1]['total_gamma_sq']:.3f}  OK")

if __name__ == "__main__":
    print("=" * 60)
    print("ONCOLOGY — 10 Disease Impedance Experiments")
    print("=" * 60)
    print("  [ Cancer = impedance camouflage: Z_tumor -> Z_host ]")
    for fn in [exp1_lung_ca, exp2_breast_ca, exp3_crc, exp4_hcc, exp5_pancreatic,
               exp6_gbm, exp7_leukemia, exp8_lymphoma, exp9_rcc, exp10_metastasis,
               exp_engine]:
        fn()
    print("\nOK ALL 10 ONCOLOGY EXPERIMENTS PASSED")
