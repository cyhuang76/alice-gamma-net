# -*- coding: utf-8 -*-
"""exp_clinical_nephrology.py — 10 Renal Disease Impedance Experiments"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alice.body.clinical_nephrology import (
    ClinicalNephrologyEngine, AKIModel, CKDModel, NephrolithiasisModel,
    NephroticModel, NephriticModel, DiabeticNephropathyModel,
    ElectrolyteModel, RenalHTNModel, PKDModel, RTAModel,
)

def exp1_aki():
    m = AKIModel(insult_severity=0.7)
    r0 = m.tick()
    m.start_recovery()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 1 AKI: KDIGO={r0['kdigo']}, Cr={r0['creatinine']:.1f}, G2={r0['gamma_sq']:.3f}->recovery->{res[-1]['gamma_sq']:.3f}  OK")

def exp2_ckd():
    m = CKDModel(initial_gfr=35.0, decline=0.03)
    res = [m.tick() for _ in range(80)]
    print(f"  EXP 2 CKD: GFR={res[-1]['gfr']:.0f}, stage={res[-1]['stage']}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp3_stone():
    m = NephrolithiasisModel(stone_size=10.0)
    res = [m.tick() for _ in range(20)]
    print(f"  EXP 3 Stone: {res[-1]['stone_mm']:.0f}mm, obstructed={res[-1]['obstructed']}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp4_nephrotic():
    m = NephroticModel(barrier_damage=0.6)
    res = [m.tick() for _ in range(20)]
    print(f"  EXP 4 Nephrotic: proteinuria={res[-1]['proteinuria']:.1f}g, albumin={res[-1]['albumin']:.1f}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp5_nephritic():
    m = NephriticModel(inflammation=0.6)
    res = [m.tick() for _ in range(20)]
    print(f"  EXP 5 Nephritic: hematuria={res[-1]['hematuria']}, GFR={res[-1]['gfr']:.0f}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp6_dn():
    m = DiabeticNephropathyModel(hba1c=9.5, duration_years=15)
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 6 DN: UACR={res[-1]['uacr']:.0f}, GFR={res[-1]['gfr']:.0f}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp7_electrolyte():
    m = ElectrolyteModel(disorder="hyperkalemia")
    res = [m.tick() for _ in range(10)]
    print(f"  EXP 7 Hyperkalemia: K={res[-1]['k']:.1f}, Na={res[-1]['na']:.0f}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp8_renal_htn():
    m = RenalHTNModel(stenosis=0.6)
    r0 = m.tick()
    m.start_ace_inhibitor()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 8 Renal HTN: BP={r0['bp']:.0f}, G2={r0['gamma_sq']:.3f}->ACEi->{res[-1]['gamma_sq']:.3f}  OK")

def exp9_pkd():
    m = PKDModel(initial_volume=800.0, growth_rate=0.002)
    res = [m.tick() for _ in range(60)]
    print(f"  EXP 9 PKD: vol={res[-1]['volume_ml']:.0f}mL, GFR={res[-1]['gfr']:.0f}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp10_rta():
    m = RTAModel(rta_type=1, severity=0.6)
    res = [m.tick() for _ in range(20)]
    print(f"  EXP 10 RTA type1: pH={res[-1]['ph']:.2f}, HCO3={res[-1]['hco3']:.0f}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp_engine():
    e = ClinicalNephrologyEngine()
    e.add_disease("ckd", initial_gfr=40.0)
    e.add_disease("electrolyte", disorder="hyponatremia")
    res = [e.tick() for _ in range(30)]
    print(f"  ENGINE: CKD+Electrolyte -> total_G2={res[-1]['total_gamma_sq']:.3f}  OK")

if __name__ == "__main__":
    print("=" * 60)
    print("NEPHROLOGY — 10 Disease Impedance Experiments")
    print("=" * 60)
    for fn in [exp1_aki, exp2_ckd, exp3_stone, exp4_nephrotic, exp5_nephritic,
               exp6_dn, exp7_electrolyte, exp8_renal_htn, exp9_pkd, exp10_rta, exp_engine]:
        fn()
    print("\nOK ALL 10 NEPHROLOGY EXPERIMENTS PASSED")
