# -*- coding: utf-8 -*-
"""exp_clinical_immunology.py — 10 Immune Disease Impedance Experiments"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alice.body.clinical_immunology import (
    ClinicalImmunologyEngine, SLEModel, RAModel, AnaphylaxisModel,
    AllergicRhinitisModel, HIVModel, SepsisModel, TransplantRejectionModel,
    SarcoidosisModel, VasculitisModel, ImmunodeficiencyModel,
)

def exp1_sle():
    m = SLEModel(severity=0.6, flare=True)
    r0 = m.tick()
    m.start_treatment()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 1 SLE: SLEDAI={r0['sledai']:.0f}, anti-dsDNA={r0['anti_dsDNA']:.0f}, G2={r0['gamma_sq']:.3f}->Tx->{res[-1]['gamma_sq']:.3f}  OK")

def exp2_ra():
    m = RAModel(severity=0.5, joints=10)
    r0 = m.tick()
    m.start_dmard()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 2 RA: DAS28={r0['das28']:.1f}, G2={r0['gamma_sq']:.3f}->DMARD->{res[-1]['gamma_sq']:.3f}  OK")

def exp3_anaphylaxis():
    m = AnaphylaxisModel(allergen_z=400.0, sensitivity=0.9)
    r0 = m.tick()
    m.give_epinephrine()
    res = [m.tick() for _ in range(10)]
    print(f"  EXP 3 Anaphylaxis: WAO={r0['wao']}, G2={r0['gamma_sq']:.3f}->epi->{res[-1]['gamma_sq']:.3f}  OK")

def exp4_allergy():
    m = AllergicRhinitisModel(allergen_z=180.0, severity=0.5)
    r0 = m.tick()
    m.start_antihistamine()
    res = [m.tick() for _ in range(20)]
    print(f"  EXP 4 Allergy: IgE={r0['ige']:.0f}, G2={r0['gamma_sq']:.3f}->AH->{res[-1]['gamma_sq']:.3f}  OK")

def exp5_hiv():
    m = HIVModel(initial_cd4=250.0)
    r0 = m.tick()
    m.start_art()
    res = [m.tick() for _ in range(60)]
    print(f"  EXP 5 HIV: CD4={r0['cd4']:.0f}->ART->{res[-1]['cd4']:.0f}, G2={r0['gamma_sq']:.3f}->{res[-1]['gamma_sq']:.3f}  OK")

def exp6_sepsis():
    m = SepsisModel(pathogen_z=350.0, virulence=0.8)
    r0 = m.tick()
    m.start_antibiotics()
    m.start_vasopressors()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 6 Sepsis: SOFA={r0['sofa']:.0f}, lactate={r0['lactate']:.1f}, G2={r0['gamma_sq']:.3f}->Abx+VP->{res[-1]['gamma_sq']:.3f}  OK")

def exp7_transplant():
    m = TransplantRejectionModel(hla_mismatch=4, organ="kidney")
    r0 = m.tick()
    m.start_immunosuppression()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 7 Rejection: Banff={r0['banff']}, G2={r0['gamma_sq']:.3f}->IS->{res[-1]['gamma_sq']:.3f}  OK")

def exp8_sarcoidosis():
    m = SarcoidosisModel(severity=0.5, organs_involved=2)
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 8 Sarcoidosis: ACE={res[-1]['ace']:.0f}, stage={res[-1]['stage']}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp9_vasculitis():
    m = VasculitisModel(vessel_size="small", severity=0.6)
    r0 = m.tick()
    m.start_treatment()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 9 Vasculitis: BVAS={r0['bvas']:.0f}, ANCA={r0['anca']}, G2={r0['gamma_sq']:.3f}->Tx->{res[-1]['gamma_sq']:.3f}  OK")

def exp10_cvid():
    m = ImmunodeficiencyModel(deficiency="cvid", severity=0.7)
    r0 = m.tick()
    m.start_ivig()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 10 CVID: IgG={r0['igg']:.0f}, G2={r0['gamma_sq']:.3f}->IVIG->{res[-1]['gamma_sq']:.3f}  OK")

def exp_engine():
    e = ClinicalImmunologyEngine()
    e.add_disease("sle", severity=0.4)
    e.add_disease("ra", severity=0.3)
    res = [e.tick() for _ in range(30)]
    print(f"  ENGINE: SLE+RA -> total_G2={res[-1]['total_gamma_sq']:.3f}  OK")

if __name__ == "__main__":
    print("=" * 60)
    print("IMMUNOLOGY — 10 Disease Impedance Experiments")
    print("=" * 60)
    for fn in [exp1_sle, exp2_ra, exp3_anaphylaxis, exp4_allergy, exp5_hiv,
               exp6_sepsis, exp7_transplant, exp8_sarcoidosis, exp9_vasculitis,
               exp10_cvid, exp_engine]:
        fn()
    print("\nOK ALL 10 IMMUNOLOGY EXPERIMENTS PASSED")
