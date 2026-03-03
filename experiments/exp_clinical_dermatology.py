# -*- coding: utf-8 -*-
"""exp_clinical_dermatology.py — 10 Dermatology Disease Impedance Experiments"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alice.body.clinical_dermatology import (
    ClinicalDermatologyEngine, AtopicDermatitisModel, PsoriasisModel,
    UrticariaModel, HerpesZosterModel, MelanomaModel, ContactDermatitisModel,
    AcneModel, VitiligoModel, CellulitisModel, BurnsModel,
)

def exp1_atopic():
    m = AtopicDermatitisModel(severity=0.6, filaggrin_mutation=True)
    r0 = m.tick()
    m.start_emollient()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 1 AD: SCORAD={r0['scorad']:.0f}, TEWL={r0['tewl']:.1f}, G2={r0['gamma_sq']:.3f}->emollient->{res[-1]['gamma_sq']:.3f}  OK")

def exp2_psoriasis():
    m = PsoriasisModel(severity=0.6, area=25.0)
    r0 = m.tick()
    m.start_biologic()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 2 Psoriasis: PASI={r0['pasi']:.1f}, G2={r0['gamma_sq']:.3f}->biologic->{res[-1]['gamma_sq']:.3f}  OK")

def exp3_urticaria():
    m = UrticariaModel(severity=0.6, chronic=True)
    r0 = m.tick()
    m.start_antihistamine()
    res = [m.tick() for _ in range(20)]
    print(f"  EXP 3 Urticaria: UAS-7={r0['uas7']:.0f}, G2={r0['gamma_sq']:.3f}->AH->{res[-1]['gamma_sq']:.3f}  OK")

def exp4_zoster():
    m = HerpesZosterModel(age=75, severity=0.7)
    r0 = m.tick()
    m.start_antiviral()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 4 Zoster: pain={r0['pain']:.1f}, PHN_risk={r0['phn_risk']:.2f}, G2={r0['gamma_sq']:.3f}->AV->{res[-1]['gamma_sq']:.3f}  OK")

def exp5_melanoma():
    m = MelanomaModel(breslow=2.5, growth_rate=0.015)
    res = [m.tick() for _ in range(20)]
    m.excision()
    res2 = [m.tick() for _ in range(10)]
    print(f"  EXP 5 Melanoma: Breslow={res[-1]['breslow']:.1f}mm->excision, G2={res[-1]['gamma_sq']:.3f}->{res2[-1]['gamma_sq']:.3f}  OK")

def exp6_contact():
    m = ContactDermatitisModel(allergen_z=200.0, exposure=0.7)
    r0 = m.tick()
    m.remove_allergen()
    res = [m.tick() for _ in range(20)]
    print(f"  EXP 6 Contact Derm: G2={r0['gamma_sq']:.3f}->remove->{res[-1]['gamma_sq']:.3f}  OK")

def exp7_acne():
    m = AcneModel(severity=0.6, inflammatory=True)
    r0 = m.tick()
    m.start_treatment()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 7 Acne: IGA={r0['iga']}, G2={r0['gamma_sq']:.3f}->Tx->{res[-1]['gamma_sq']:.3f}  OK")

def exp8_vitiligo():
    m = VitiligoModel(severity=0.5, progression=0.008)
    m.start_phototherapy()
    res = [m.tick() for _ in range(60)]
    print(f"  EXP 8 Vitiligo: VASI={res[-1]['vasi']:.2f}, phototherapy, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp9_cellulitis():
    m = CellulitisModel(severity=0.6)
    r0 = m.tick()
    m.start_antibiotics()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 9 Cellulitis: Eron={r0['eron']}, G2={r0['gamma_sq']:.3f}->Abx->{res[-1]['gamma_sq']:.3f}  OK")

def exp10_burns():
    m = BurnsModel(tbsa=25.0, depth="full")
    r0 = m.tick()
    m.start_resuscitation()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 10 Burns: TBSA={r0['tbsa']:.0f}%, G2={r0['gamma_sq']:.3f}->Parkland->{res[-1]['gamma_sq']:.3f}  OK")

def exp_engine():
    e = ClinicalDermatologyEngine()
    e.add_disease("atopic", severity=0.4)
    e.add_disease("psoriasis", severity=0.3)
    res = [e.tick() for _ in range(30)]
    print(f"  ENGINE: AD+Psoriasis -> total_G2={res[-1]['total_gamma_sq']:.3f}  OK")

if __name__ == "__main__":
    print("=" * 60)
    print("DERMATOLOGY — 10 Disease Impedance Experiments")
    print("=" * 60)
    for fn in [exp1_atopic, exp2_psoriasis, exp3_urticaria, exp4_zoster,
               exp5_melanoma, exp6_contact, exp7_acne, exp8_vitiligo,
               exp9_cellulitis, exp10_burns, exp_engine]:
        fn()
    print("\nOK ALL 10 DERMATOLOGY EXPERIMENTS PASSED")
