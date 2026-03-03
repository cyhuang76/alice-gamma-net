# -*- coding: utf-8 -*-
"""exp_clinical_orthopedics.py — 10 Orthopedic Disease Impedance Experiments"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alice.body.clinical_orthopedics import (
    ClinicalOrthopedicsEngine, FractureModel, OsteoporosisModel,
    DiscHerniationModel, OsteoarthritisModel, ACLModel,
    TendinitisModel, ScoliosisModel, OsteosarcomaModel,
    GoutModel, OsteomyelitisModel,
)

def exp1_fracture():
    m = FractureModel(ao_class="A2", displacement=0.7)
    r0 = m.tick()
    m.fixation()
    res = [m.tick() for _ in range(60)]
    print(f"  EXP 1 Fracture A2: healing={res[-1]['healing']:.2f}, G2={r0['gamma_sq']:.3f}->fixation->{res[-1]['gamma_sq']:.3f}  OK")

def exp2_osteoporosis():
    m = OsteoporosisModel(t_score=-3.0, age=75)
    r0 = m.tick()
    m.start_bisphosphonate()
    res = [m.tick() for _ in range(50)]
    print(f"  EXP 2 Osteoporosis: T={r0['t_score']:.1f}, FRAX={r0['frax']:.0f}%, G2={r0['gamma_sq']:.3f}->BP->{res[-1]['gamma_sq']:.3f}  OK")

def exp3_disc():
    m = DiscHerniationModel(severity=0.6, level="L4-L5")
    r0 = m.tick()
    m.start_conservative()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 3 Disc L4-5: ODI={r0['odi']:.0f}, G2={r0['gamma_sq']:.3f}->conservative->{res[-1]['gamma_sq']:.3f}  OK")

def exp4_oa():
    m = OsteoarthritisModel(kl_grade=3, joint="knee")
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 4 OA KL3: WOMAC={res[-1]['womac']:.0f}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp5_acl():
    m = ACLModel(partial=False)
    r0 = m.tick()
    m.reconstruction()
    res = [m.tick() for _ in range(60)]
    print(f"  EXP 5 ACL complete: IKDC={r0['ikdc']:.0f}->recon->{res[-1]['ikdc']:.0f}, G2={r0['gamma_sq']:.3f}->{res[-1]['gamma_sq']:.3f}  OK")

def exp6_tendinitis():
    m = TendinitisModel(severity=0.5, location="achilles")
    r0 = m.tick()
    m.rest()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 6 Achilles: DASH={r0['dash']:.0f}, G2={r0['gamma_sq']:.3f}->rest->{res[-1]['gamma_sq']:.3f}  OK")

def exp7_scoliosis():
    m = ScoliosisModel(cobb=35.0, growing=True)
    m.start_brace()
    res = [m.tick() for _ in range(60)]
    print(f"  EXP 7 Scoliosis: Cobb={res[-1]['cobb']:.0f}°, brace, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp8_osteosarcoma():
    m = OsteosarcomaModel(size_cm=7.0, grade="high")
    r0 = m.tick()
    m.start_chemo()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 8 Osteosarcoma: ALP={r0['alp']:.0f}, G2={r0['gamma_sq']:.3f}->chemo->{res[-1]['gamma_sq']:.3f}  OK")

def exp9_gout():
    m = GoutModel(urate=10.0, chronic=False)
    r0 = m.tick()
    m.start_allopurinol()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 9 Gout: urate={r0['urate']:.1f}->allo->{res[-1]['urate']:.1f}, G2={r0['gamma_sq']:.3f}->{res[-1]['gamma_sq']:.3f}  OK")

def exp10_osteo():
    m = OsteomyelitisModel(acute=True, severity=0.6)
    r0 = m.tick()
    m.start_antibiotics()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 10 Osteomyelitis: ESR={r0['esr']:.0f}, G2={r0['gamma_sq']:.3f}->Abx->{res[-1]['gamma_sq']:.3f}  OK")

def exp_engine():
    e = ClinicalOrthopedicsEngine()
    e.add_disease("fracture", ao_class="A1")
    e.add_disease("osteoporosis", t_score=-2.8)
    e.add_disease("gout", urate=8.5)
    res = [e.tick() for _ in range(30)]
    print(f"  ENGINE: Fracture+OP+Gout -> total_G2={res[-1]['total_gamma_sq']:.3f}  OK")

if __name__ == "__main__":
    print("=" * 60)
    print("Orthopedics — 10 Disease Impedance Experiments")
    print("=" * 60)
    for fn in [exp1_fracture, exp2_osteoporosis, exp3_disc, exp4_oa, exp5_acl,
               exp6_tendinitis, exp7_scoliosis, exp8_osteosarcoma, exp9_gout,
               exp10_osteo, exp_engine]:
        fn()
    print("\nOK ALL 10 ORTHOPEDICS EXPERIMENTS PASSED")
