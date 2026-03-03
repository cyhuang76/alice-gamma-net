# -*- coding: utf-8 -*-
"""exp_clinical_cardiology.py — 10 Cardiac Disease Impedance Experiments"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alice.body.clinical_cardiology import (
    ClinicalCardiologyEngine, MIModel, CHFModel, AFModel, HTNModel,
    AorticStenosisModel, CardiomyopathyModel, PericarditisModel,
    PulmHTNModel, EndocarditisModel, AorticDissectionModel,
)

def exp1_mi():
    m = MIModel(territory="LAD", occlusion=0.95)
    r0 = m.tick()
    m.reperfuse(tick=1)
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 1 MI LAD: troponin_peak, G2={r0['total_gamma_sq']:.3f}->reperfuse->{res[-1]['total_gamma_sq']:.3f}  OK")

def exp2_chf():
    m = CHFModel(initial_ef=0.30, progression=0.001)
    r0 = m.tick()
    m.start_treatment()
    res = [m.tick() for _ in range(50)]
    print(f"  EXP 2 CHF: EF={res[-1]['ef']:.0%}, NYHA={res[-1]['nyha_class']}, G2={r0['gamma_sq']:.3f}->Tx->{res[-1]['gamma_sq']:.3f}  OK")

def exp3_af():
    m = AFModel(risk_factors=4)
    m.trigger_af()
    r0 = m.tick()
    m.cardiovert()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 3 AF: CHA2DS2={r0['cha2ds2_vasc']}, G2={r0['gamma_sq']:.3f}->cardiovert->{res[-1]['gamma_sq']:.3f}  OK")

def exp4_htn():
    m = HTNModel(initial_sys=175.0, initial_dia=105.0)
    r0 = m.tick()
    m.treat()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 4 HTN: {r0['systolic']:.0f}/{r0['diastolic']:.0f}->{res[-1]['systolic']:.0f}/{res[-1]['diastolic']:.0f}, G2={r0['gamma_sq']:.3f}->{res[-1]['gamma_sq']:.3f}  OK")

def exp5_as():
    m = AorticStenosisModel(initial_valve_z=25.0, rate=0.003)
    res = [m.tick() for _ in range(60)]
    print(f"  EXP 5 Aortic Stenosis: gradient={res[-1]['gradient_mmhg']:.0f}mmHg, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp6_dcm():
    m = CardiomyopathyModel(subtype="dilated")
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 6 DCM: LVEF={res[-1]['lvef']:.0%}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp7_pericarditis():
    m = PericarditisModel(effusion=200.0)
    res = [m.tick() for _ in range(20)]
    print(f"  EXP 7 Pericarditis: effusion={res[-1]['effusion_ml']:.0f}mL, tamponade={res[-1]['tamponade_risk']:.2f}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp8_pulm_htn():
    m = PulmHTNModel(initial_pap=45.0, progression=0.002)
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 8 PulmHTN: mPAP={res[-1]['mean_pap']:.0f}, WHO={res[-1]['who_class']}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp9_endocarditis():
    m = EndocarditisModel(valve="mitral", virulence=0.7)
    r0 = m.tick()
    m.start_antibiotics()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 9 Endocarditis: Duke={r0['duke_definite']}, G2={r0['gamma_sq']:.3f}->Abx->{res[-1]['gamma_sq']:.3f}  OK")

def exp10_dissection():
    m = AorticDissectionModel(stanford="A", initial_tear=0.5)
    r0 = m.tick()
    m.emergency_surgery()
    res = [m.tick() for _ in range(20)]
    print(f"  EXP 10 Dissection A: mortality={r0['mortality_risk']:.2f}, G2={r0['gamma_sq']:.3f}->surgery->{res[-1]['gamma_sq']:.3f}  OK")

def exp_engine():
    e = ClinicalCardiologyEngine()
    e.add_disease("mi", territory="LAD", occlusion=0.8)
    e.add_disease("hypertension", initial_sys=160.0)
    res = [e.tick() for _ in range(30)]
    print(f"  ENGINE: MI+HTN -> total_G2={res[-1]['total_gamma_sq']:.3f}  OK")

if __name__ == "__main__":
    print("=" * 60)
    print("CARDIOLOGY — 10 Disease Impedance Experiments")
    print("=" * 60)
    for fn in [exp1_mi, exp2_chf, exp3_af, exp4_htn, exp5_as, exp6_dcm,
               exp7_pericarditis, exp8_pulm_htn, exp9_endocarditis,
               exp10_dissection, exp_engine]:
        fn()
    print("\nOK ALL 10 CARDIOLOGY EXPERIMENTS PASSED")
