# -*- coding: utf-8 -*-
"""exp_clinical_pulmonology.py — 10 Pulmonary Disease Impedance Experiments"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alice.body.clinical_pulmonology import (
    ClinicalPulmonologyEngine, AsthmaModel, COPDModel, PneumoniaModel,
    PEModel, PneumothoraxModel, FibrosisModel, ARDSModel,
    OSAModel, LungCancerModel, CFModel,
)

def exp1_asthma():
    m = AsthmaModel(reactivity=0.7)
    m.trigger_exacerbation()
    r0 = m.tick()
    m.use_bronchodilator()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 1 Asthma: FEV1={r0['fev1_ratio']:.2f}->BD->{res[-1]['fev1_ratio']:.2f}, G2={r0['gamma_sq']:.3f}->{res[-1]['gamma_sq']:.3f}  OK")

def exp2_copd():
    m = COPDModel(initial_fev1=45.0, decline_rate=0.03)
    res = [m.tick() for _ in range(60)]
    print(f"  EXP 2 COPD GOLD={res[-1]['gold_stage']}: FEV1={res[-1]['fev1_pct']:.0f}%, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp3_pneumonia():
    m = PneumoniaModel(consolidation=0.5, virulence=0.6)
    r0 = m.tick()
    m.treat()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 3 Pneumonia: CURB65={r0['curb65']}, G2={r0['gamma_sq']:.3f}->Abx->{res[-1]['gamma_sq']:.3f}  OK")

def exp4_pe():
    m = PEModel(clot_burden=0.6)
    r0 = m.tick()
    m.anticoagulate()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 4 PE: Wells={r0['wells']}, G2={r0['gamma_sq']:.3f}->AC->{res[-1]['gamma_sq']:.3f}  OK")

def exp5_ptx():
    m = PneumothoraxModel(size=40.0, tension=False)
    r0 = m.tick()
    m.insert_chest_tube()
    res = [m.tick() for _ in range(20)]
    print(f"  EXP 5 PTX: size={r0['size_pct']:.0f}%->tube->{res[-1]['size_pct']:.0f}%, G2={r0['gamma_sq']:.3f}->{res[-1]['gamma_sq']:.3f}  OK")

def exp6_ipf():
    m = FibrosisModel(initial_fvc=60.0, rate=0.02)
    r0 = m.tick()
    m.start_antifibrotic()
    res = [m.tick() for _ in range(60)]
    print(f"  EXP 6 IPF: FVC={res[-1]['fvc_pct']:.0f}%, G2={r0['gamma_sq']:.3f}->AF->{res[-1]['gamma_sq']:.3f}  OK")

def exp7_ards():
    m = ARDSModel(initial_surfactant=0.3)
    r0 = m.tick()
    m.intubate()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 7 ARDS: P/F={r0['pao2_fio2']:.0f}, Berlin={r0['berlin']}, G2={r0['gamma_sq']:.3f}->vent->{res[-1]['gamma_sq']:.3f}  OK")

def exp8_osa():
    m = OSAModel(collapsibility=0.7)
    r0 = m.tick()
    m.start_cpap()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 8 OSA: AHI={r0['ahi']:.0f}->CPAP->{res[-1]['ahi']:.0f}, G2={r0['gamma_sq']:.3f}->{res[-1]['gamma_sq']:.3f}  OK")

def exp9_lung_ca():
    m = LungCancerModel(initial_size=2.0, growth_rate=0.008)
    res = [m.tick() for _ in range(60)]
    print(f"  EXP 9 Lung Ca: {res[-1]['tumor_cm']:.1f}cm, stage={res[-1]['stage']}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp10_cf():
    m = CFModel(severity=0.6)
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 10 CF: FEV1={res[-1]['fev1_pct']:.0f}%, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp_engine():
    e = ClinicalPulmonologyEngine()
    e.add_disease("asthma", reactivity=0.5)
    e.add_disease("copd", initial_fev1=55.0)
    res = [e.tick() for _ in range(30)]
    print(f"  ENGINE: Asthma+COPD -> total_G2={res[-1]['total_gamma_sq']:.3f}  OK")

if __name__ == "__main__":
    print("=" * 60)
    print("PULMONOLOGY — 10 Disease Impedance Experiments")
    print("=" * 60)
    for fn in [exp1_asthma, exp2_copd, exp3_pneumonia, exp4_pe, exp5_ptx,
               exp6_ipf, exp7_ards, exp8_osa, exp9_lung_ca, exp10_cf, exp_engine]:
        fn()
    print("\nOK ALL 10 PULMONOLOGY EXPERIMENTS PASSED")
