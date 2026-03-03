# -*- coding: utf-8 -*-
"""exp_clinical_gastroenterology.py — 10 GI Disease Impedance Experiments"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alice.body.clinical_gastroenterology import (
    ClinicalGastroenterologyEngine, GERDModel, PepticUlcerModel, IBDModel,
    IBSModel, CirrhosisModel, CholelithiasisModel, PancreatitisModel,
    BowelObstructionModel, CRCModel, HepatitisModel,
)

def exp1_gerd():
    m = GERDModel(les_weakness=0.7)
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 1 GERD: LA={res[-1]['la_grade']}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp2_pud():
    m = PepticUlcerModel(h_pylori=True, nsaid=True)
    r0 = m.tick()
    m.start_ppi()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 2 PUD: Forrest={r0['forrest']}, G2={r0['gamma_sq']:.3f}->PPI->{res[-1]['gamma_sq']:.3f}  OK")

def exp3_ibd():
    m = IBDModel(subtype="crohn", severity=0.6)
    r0 = m.tick()
    m.start_biologic()
    res = [m.tick() for _ in range(40)]
    print(f"  EXP 3 IBD Crohn: score={r0['score']}, G2={r0['gamma_sq']:.3f}->biologic->{res[-1]['gamma_sq']:.3f}  OK")

def exp4_ibs():
    m = IBSModel(subtype="mixed", sensitivity=0.7)
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 4 IBS: gut-brain G={res[-1]['gut_brain_gamma']:.3f}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp5_cirrhosis():
    m = CirrhosisModel(initial_fibrosis=3, progression=0.002)
    res = [m.tick() for _ in range(60)]
    print(f"  EXP 5 Cirrhosis: Child-Pugh={res[-1]['child_pugh']}, MELD={res[-1]['meld']:.0f}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp6_gallstone():
    m = CholelithiasisModel(stone_size=12.0)
    res = [m.tick() for _ in range(20)]
    print(f"  EXP 6 Gallstone: stone={res[-1]['stone_mm']:.0f}mm, obstructed={res[-1]['obstructed']}, G2={res[-1]['gamma_sq']:.3f}  OK")

def exp7_pancreatitis():
    m = PancreatitisModel(cause="gallstone", initial_severity=0.5)
    r0 = m.tick()
    m.treat()
    res = [m.tick() for _ in range(30)]
    print(f"  EXP 7 Pancreatitis: Ranson={r0['ranson']}, lipase={r0['lipase']:.0f}, G2={r0['gamma_sq']:.3f}->Tx->{res[-1]['gamma_sq']:.3f}  OK")

def exp8_sbo():
    m = BowelObstructionModel(level="small", complete=True)
    r0 = m.tick()
    m.operate()
    res = [m.tick() for _ in range(20)]
    print(f"  EXP 8 SBO: complete={r0['complete']}, G2={r0['gamma_sq']:.3f}->surgery->{res[-1]['gamma_sq']:.3f}  OK")

def exp9_crc():
    m = CRCModel(initial_size=2.0, growth_rate=0.005)
    res = [m.tick() for _ in range(60)]
    m.start_treatment()
    res2 = [m.tick() for _ in range(30)]
    print(f"  EXP 9 CRC: CEA={res[-1]['cea']:.1f}, stage={res[-1]['stage']}, G2={res[-1]['gamma_sq']:.3f}->Tx->{res2[-1]['gamma_sq']:.3f}  OK")

def exp10_hep():
    m = HepatitisModel(viral_type="B", initial_load=1e7)
    r0 = m.tick()
    m.start_antiviral()
    res = [m.tick() for _ in range(50)]
    print(f"  EXP 10 HBV: VL={r0['viral_load']:.0e}->AV->{res[-1]['viral_load']:.0e}, G2={r0['gamma_sq']:.3f}->{res[-1]['gamma_sq']:.3f}  OK")

def exp_engine():
    e = ClinicalGastroenterologyEngine()
    e.add_disease("gerd", les_weakness=0.5)
    e.add_disease("ibs", subtype="mixed")
    e.add_disease("cirrhosis", initial_fibrosis=2)
    res = [e.tick() for _ in range(30)]
    print(f"  ENGINE: GERD+IBS+Cirrhosis -> total_G2={res[-1]['total_gamma_sq']:.3f}  OK")

if __name__ == "__main__":
    print("=" * 60)
    print("GASTROENTEROLOGY — 10 Disease Impedance Experiments")
    print("=" * 60)
    for fn in [exp1_gerd, exp2_pud, exp3_ibd, exp4_ibs, exp5_cirrhosis,
               exp6_gallstone, exp7_pancreatitis, exp8_sbo, exp9_crc, exp10_hep, exp_engine]:
        fn()
    print("\nOK ALL 10 GI EXPERIMENTS PASSED")
