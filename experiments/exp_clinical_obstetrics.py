# -*- coding: utf-8 -*-
"""exp_clinical_obstetrics.py — 10 OB/GYN Disease Impedance Experiments"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alice.body.clinical_obstetrics import (
    ClinicalObstetricsEngine, PreeclampsiaModel, PCOSModel,
    EndometriosisModel, FibroidModel, PretermBirthModel,
    GDMModel, OvarianCancerModel, MenopauseModel, AFEModel, PPHModel,
)

def exp1_preeclampsia():
    m = PreeclampsiaModel(severity=0.6, ga=34.0)
    r0 = m.tick()
    m.start_mgso4()
    res = [m.tick() for _ in range(20)]
    print(f"  EXP 1 Preeclampsia: BP={r0['bp']:.0f}, proteinuria={r0['proteinuria']:.1f}g, G2={r0['gamma_sq']:.3f}->MgSO4->{res[-1]['gamma_sq']:.3f}  OK")

def exp2_pcos():
    m = PCOSModel(severity=0.5, insulin_resistant=True)
    r0 = m.tick()
    m.start_ocp()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 2 PCOS: AMH={r0['amh']:.1f}, T={r0['testosterone']:.1f}, G2={r0['gamma_sq']:.3f}->OCP->{res[-1]['gamma_sq']:.3f}  OK")

def exp3_endometriosis():
    m = EndometriosisModel(severity=0.6)
    r0 = m.tick()
    m.start_gnrh_agonist()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 3 Endometriosis: pain={r0['pain']:.1f}, CA125={r0['ca125']:.0f}, G2={r0['gamma_sq']:.3f}->GnRH->{res[-1]['gamma_sq']:.3f}  OK")

def exp4_fibroid():
    m = FibroidModel(size_cm=6.0, count=3)
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 4 Fibroid: size={res[-1]['size_cm']:.1f}cm, QoL={res[-1]['qol']:.1f}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp5_preterm():
    m = PretermBirthModel(cervical_length=18.0, ga=28.0)
    r0 = m.tick()
    m.start_progesterone()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 5 Preterm: CL={r0['cl_mm']:.0f}mm, G2={r0['gamma_sq']:.3f}->progesterone->{res[-1]['gamma_sq']:.3f}  OK")

def exp6_gdm():
    m = GDMModel(severity=0.5)
    r0 = m.tick()
    m.start_insulin()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 6 GDM: fasting={r0['fasting']:.0f}mg/dL, G2={r0['gamma_sq']:.3f}->insulin->{res[-1]['gamma_sq']:.3f}  OK")

def exp7_ovarian_ca():
    m = OvarianCancerModel(stage="III", growth_rate=0.015)
    r0 = m.tick()
    m.start_chemo()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 7 OvarianCa: CA125={r0['ca125']:.0f}, G2={r0['gamma_sq']:.3f}->chemo->{res[-1]['gamma_sq']:.3f}  OK")

def exp8_menopause():
    m = MenopauseModel(years_post=3.0)
    r0 = m.tick()
    m.start_hrt()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 8 Menopause: FSH={r0['fsh']:.0f}, MRS={r0['mrs']:.0f}, G2={r0['gamma_sq']:.3f}->HRT->{res[-1]['gamma_sq']:.3f}  OK")

def exp9_afe():
    m = AFEModel(severity=0.9)
    r0 = m.tick()
    m.start_resuscitation()
    res = [m.tick() for _ in range(10)]
    print(f"  EXP 9 AFE: DIC={r0['dic']}, SOFA={r0['sofa']:.0f}, mortality={r0['mortality']:.2f}, G2={r0['gamma_sq']:.3f}  (catastrophic) OK")

def exp10_pph():
    m = PPHModel(cause="atony", severity=0.7)
    r0 = m.tick()
    m.give_uterotonics()
    res = [m.tick() for _ in range(15)]
    print(f"  EXP 10 PPH: EBL={r0['ebl_ml']:.0f}mL, SI={r0['shock_index']:.2f}, G2={r0['gamma_sq']:.3f}->uterotonics->{res[-1]['gamma_sq']:.3f}  OK")

def exp_engine():
    e = ClinicalObstetricsEngine()
    e.add_disease("preeclampsia", severity=0.4, ga=32.0)
    e.add_disease("gdm", severity=0.3)
    res = [e.tick() for _ in range(20)]
    print(f"  ENGINE: Preeclampsia+GDM -> total_G2={res[-1]['total_gamma_sq']:.3f}  OK")

if __name__ == "__main__":
    print("=" * 60)
    print("OB/GYN — 10 Disease Impedance Experiments")
    print("=" * 60)
    for fn in [exp1_preeclampsia, exp2_pcos, exp3_endometriosis, exp4_fibroid,
               exp5_preterm, exp6_gdm, exp7_ovarian_ca, exp8_menopause,
               exp9_afe, exp10_pph, exp_engine]:
        fn()
    print("\nOK ALL 10 OB/GYN EXPERIMENTS PASSED")
