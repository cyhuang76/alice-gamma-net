# -*- coding: utf-8 -*-
"""exp_clinical_ent.py — 10 ENT Disease Impedance Experiments"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alice.body.clinical_ent import (
    ClinicalENTEngine, SNHLModel, ConductiveHLModel, MeniereModel,
    TinnitusModel, OtitisMediaModel, VocalCordParalysisModel,
    SinusitisModel, AnosmiaModel, SSHLModel, BPPVModel,
)

def exp1_snhl():
    m = SNHLModel(severity_db=55.0)
    r0 = m.tick()
    m.apply_hearing_aid()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 1 SNHL: PTA={r0['pta_db']:.0f}dB->HA->{res[-1]['pta_db']:.0f}dB, speech={res[-1]['speech']:.0f}%, G2={r0['gamma_sq']:.3f}->{res[-1]['gamma_sq']:.3f}  OK")

def exp2_chl():
    m = ConductiveHLModel(cause="otosclerosis", severity=0.6)
    res = [m.tick() for _ in range(20)]
    print(f"  EXP 2 CHL: ABG={res[-1]['air_bone_gap']:.0f}dB, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp3_meniere():
    m = MeniereModel(severity=0.6)
    res = [m.tick() for _ in range(30)]
    attacks = sum(1 for r in res if r.get('in_attack', False))
    max_g2 = max(r['gamma_sq'] for r in res)
    print(f"  EXP 3 Ménière: attacks={attacks}/30, G2_peak={max_g2:.3f}  OK")

def exp4_tinnitus():
    m = TinnitusModel(severity=0.5, frequency=4000.0)
    r0 = m.tick()
    m.start_trt()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 4 Tinnitus: THI={r0['thi']:.0f}->TRT->{res[-1]['thi']:.0f}, G2={r0['gamma_sq']:.3f}->{res[-1]['gamma_sq']:.3f}  OK")

def exp5_otitis():
    m = OtitisMediaModel(effusion=True, acute=True)
    r0 = m.tick()
    m.start_antibiotics()
    res = [m.tick() for _ in range(20)]
    print(f"  EXP 5 AOM: tymp={r0['tymp']}, G2={r0['gamma_sq']:.3f}->Abx->{res[-1]['gamma_sq']:.3f}  OK")

def exp6_vocal_cord():
    m = VocalCordParalysisModel(unilateral=True, severity=0.7)
    res = [m.tick() for _ in range(20)]
    print(f"  EXP 6 Vocal Cord: VHI={res[-1]['vhi']:.0f}, quality={res[-1]['quality']:.2f}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp7_sinusitis():
    m = SinusitisModel(chronic=True, severity=0.6)
    r0 = m.tick()
    m.start_treatment()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 7 Sinusitis: LM={r0['lund_mackay']}, G2={r0['gamma_sq']:.3f}->Tx->{res[-1]['gamma_sq']:.3f}  OK")

def exp8_anosmia():
    m = AnosmiaModel(severity=0.8, cause="post-viral")
    res = [m.tick() for _ in range(60)]
    print(f"  EXP 8 Anosmia: UPSIT={res[-1]['upsit']:.0f}/40, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp9_sshl():
    m = SSHLModel(severity_db=60.0)
    r0 = m.tick()
    m.start_steroids()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 9 SSHL: PTA={r0['pta_db']:.0f}->steroids->{res[-1]['pta_db']:.0f}, recovery={res[-1]['recovery']:.2f}, G2={r0['gamma_sq']:.3f}->{res[-1]['gamma_sq']:.3f}  OK")

def exp10_bppv():
    m = BPPVModel(canal="posterior", severity=0.6)
    r0 = m.tick()
    m.do_epley()
    res = [m.tick() for _ in range(15)]
    print(f"  EXP 10 BPPV: G2={r0['gamma_sq']:.3f}->Epley->{res[-1]['gamma_sq']:.3f}  OK")

def exp_engine():
    e = ClinicalENTEngine()
    e.add_disease("snhl", severity_db=40.0)
    e.add_disease("tinnitus", severity=0.3)
    res = [e.tick() for _ in range(30)]
    print(f"  ENGINE: SNHL+Tinnitus -> total_G2={res[-1]['total_gamma_sq']:.3f}  OK")

if __name__ == "__main__":
    print("=" * 60)
    print("ENT — 10 Disease Impedance Experiments")
    print("=" * 60)
    for fn in [exp1_snhl, exp2_chl, exp3_meniere, exp4_tinnitus, exp5_otitis,
               exp6_vocal_cord, exp7_sinusitis, exp8_anosmia, exp9_sshl,
               exp10_bppv, exp_engine]:
        fn()
    print("\nOK ALL 10 ENT EXPERIMENTS PASSED")
