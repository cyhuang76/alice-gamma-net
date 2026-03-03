# -*- coding: utf-8 -*-
"""clinical_nephrology.py — Top-10 Renal Diseases as Impedance-Mismatch Patterns
================================================================================

│ Disease                  │ Γ Physics Mapping                          │ Clinical Scale  │
│──────────────────────────│────────────────────────────────────────────│────────────────│
│ 1. AKI                   │ Sudden filtration Γ jump                   │ KDIGO Stage    │
│ 2. CKD                   │ Progressive nephron Z degradation          │ eGFR / CKD Stg │
│ 3. Nephrolithiasis       │ Ductal Z obstruction                       │ Stone size mm  │
│ 4. Nephrotic Syndrome    │ Glomerular barrier Z leak                  │ Proteinuria    │
│ 5. Nephritic Syndrome    │ Inflammatory glomerular Z                  │ Hematuria      │
│ 6. Diabetic Nephropathy  │ Glucose-mediated Z drift                   │ UACR / eGFR    │
│ 7. Electrolyte Disorder  │ Filtrate impedance mismatch                │ Na/K/Ca levels │
│ 8. Renal Hypertension    │ RAAS feedback Z amplification              │ BP mmHg        │
│ 9. Polycystic Kidney     │ Structural Z distortion                    │ Kidney vol     │
│ 10. RTA                  │ Acid-base Z mismatch                       │ pH / HCO3      │

The nephron is an impedance filter: Z_blood → glomerulus (band-pass) → Z_filtrate.
Disease = filter Z mismatch → wrong things pass, right things don't.
"""

from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Dict

# Physical Constants
Z_BLOOD: float = 70.0
Z_GLOMERULAR: float = 50.0     # Glomerular capillary
Z_TUBULAR: float = 60.0        # Tubular epithelium
Z_COLLECTING: float = 55.0     # Collecting duct
Z_URETER: float = 40.0         # Ureter
GFR_NORMAL: float = 100.0      # mL/min/1.73m²

def gamma_sq(z_l: float, z_0: float) -> float:
    g = (z_l - z_0) / (z_l + z_0)
    return g * g

# ============================================================================
# 1. AKI
# ============================================================================
@dataclass
class AKIState:
    glomerular_z: float = Z_GLOMERULAR
    gfr: float = GFR_NORMAL
    creatinine: float = 1.0
    kdigo_stage: int = 0

class AKIModel:
    def __init__(self, insult_severity: float = 0.6):
        self.state = AKIState()
        self.insult = insult_severity
        self.recovering = False
        self.tick_count = 0

    def start_recovery(self):
        self.recovering = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.recovering:
            st.glomerular_z = max(Z_GLOMERULAR, st.glomerular_z * 0.99)
        elif self.tick_count < 5:
            st.glomerular_z = Z_GLOMERULAR * (1 + self.insult * 3)
        g2 = gamma_sq(st.glomerular_z, Z_GLOMERULAR)
        st.gfr = GFR_NORMAL * (1 - g2)
        st.creatinine = 1.0 / max(st.gfr / GFR_NORMAL, 0.05)
        if st.creatinine < 1.5: st.kdigo_stage = 0
        elif st.creatinine < 2.0: st.kdigo_stage = 1
        elif st.creatinine < 3.0: st.kdigo_stage = 2
        else: st.kdigo_stage = 3
        return {"disease": "AKI", "gfr": st.gfr, "creatinine": st.creatinine,
                "kdigo": st.kdigo_stage, "gamma_sq": g2}

# ============================================================================
# 2. CKD
# ============================================================================
@dataclass
class CKDState:
    nephron_z: float = Z_GLOMERULAR
    gfr: float = GFR_NORMAL
    stage: int = 1

class CKDModel:
    def __init__(self, initial_gfr: float = 55.0, decline: float = 0.02):
        z_ratio = GFR_NORMAL / max(initial_gfr, 5)
        self.state = CKDState(nephron_z=Z_GLOMERULAR * z_ratio, gfr=initial_gfr)
        self.decline = decline
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.nephron_z = min(Z_GLOMERULAR * 10, st.nephron_z * (1 + self.decline / 100))
        g2 = gamma_sq(st.nephron_z, Z_GLOMERULAR)
        st.gfr = max(5, GFR_NORMAL * (1 - g2))
        if st.gfr >= 90: st.stage = 1
        elif st.gfr >= 60: st.stage = 2
        elif st.gfr >= 30: st.stage = 3
        elif st.gfr >= 15: st.stage = 4
        else: st.stage = 5
        return {"disease": "CKD", "gfr": st.gfr, "stage": st.stage, "gamma_sq": g2}

# ============================================================================
# 3. NEPHROLITHIASIS
# ============================================================================
@dataclass
class NephrolithiasisState:
    stone_size_mm: float = 4.0
    ureter_z: float = Z_URETER
    obstructed: bool = False
    pain_vas: float = 0.0

class NephrolithiasisModel:
    def __init__(self, stone_size: float = 6.0):
        self.state = NephrolithiasisState(stone_size_mm=stone_size)
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.obstructed = st.stone_size_mm > 5
        if st.obstructed:
            frac = min(1.0, (st.stone_size_mm - 5) / 10)
            st.ureter_z = Z_URETER + 500 * frac
        else:
            st.ureter_z = Z_URETER
        g2 = gamma_sq(st.ureter_z, Z_URETER)
        st.pain_vas = min(10, g2 * 12)
        return {"disease": "Nephrolithiasis", "stone_mm": st.stone_size_mm,
                "obstructed": st.obstructed, "pain": st.pain_vas, "gamma_sq": g2}

# ============================================================================
# 4. NEPHROTIC SYNDROME
# ============================================================================
@dataclass
class NephroticState:
    barrier_z: float = Z_GLOMERULAR
    proteinuria: float = 0.0   # g/day
    albumin: float = 4.0       # g/dL

class NephroticModel:
    def __init__(self, barrier_damage: float = 0.5):
        self.state = NephroticState()
        self.damage = barrier_damage
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.barrier_z = Z_GLOMERULAR * (1 - self.damage * 0.6)
        g2 = gamma_sq(st.barrier_z, Z_GLOMERULAR)
        st.proteinuria = g2 * 15  # Up to 15 g/day
        st.albumin = max(1.0, 4.0 - st.proteinuria * 0.2)
        return {"disease": "Nephrotic", "proteinuria": st.proteinuria,
                "albumin": st.albumin, "gamma_sq": g2}

# ============================================================================
# 5. NEPHRITIC SYNDROME
# ============================================================================
@dataclass
class NephriticState:
    glomerular_z: float = Z_GLOMERULAR
    hematuria: bool = False
    gfr: float = GFR_NORMAL

class NephriticModel:
    def __init__(self, inflammation: float = 0.5):
        self.state = NephriticState()
        self.inflammation = inflammation
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.glomerular_z = Z_GLOMERULAR * (1 + self.inflammation * 2)
        g2 = gamma_sq(st.glomerular_z, Z_GLOMERULAR)
        st.hematuria = g2 > 0.1
        st.gfr = GFR_NORMAL * (1 - g2 * 0.5)
        return {"disease": "Nephritic", "hematuria": st.hematuria,
                "gfr": st.gfr, "gamma_sq": g2}

# ============================================================================
# 6. DIABETIC NEPHROPATHY
# ============================================================================
@dataclass
class DiabeticNephropathyState:
    glomerular_z: float = Z_GLOMERULAR
    uacr: float = 10.0    # mg/g (normal <30)
    gfr: float = GFR_NORMAL
    hba1c: float = 7.0

class DiabeticNephropathyModel:
    def __init__(self, hba1c: float = 8.5, duration_years: float = 10):
        self.state = DiabeticNephropathyState(hba1c=hba1c)
        self.damage_rate = (hba1c - 5.5) * 0.0005 * math.sqrt(duration_years)
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.glomerular_z = min(Z_GLOMERULAR * 5,
                              st.glomerular_z * (1 + self.damage_rate))
        g2 = gamma_sq(st.glomerular_z, Z_GLOMERULAR)
        st.uacr = 10 + g2 * 3000
        st.gfr = max(10, GFR_NORMAL * (1 - g2))
        return {"disease": "Diabetic Nephropathy", "uacr": st.uacr,
                "gfr": st.gfr, "hba1c": st.hba1c, "gamma_sq": g2}

# ============================================================================
# 7. ELECTROLYTE DISORDER
# ============================================================================
@dataclass
class ElectrolyteState:
    na: float = 140.0    # mEq/L (135–145)
    k: float = 4.0       # mEq/L (3.5–5.0)
    ca: float = 9.5      # mg/dL (8.5–10.5)
    filtrate_z: float = Z_TUBULAR

class ElectrolyteModel:
    def __init__(self, disorder: str = "hyponatremia"):
        self.state = ElectrolyteState()
        self.disorder = disorder
        self.tick_count = 0
        if disorder == "hyponatremia": self.state.na = 125
        elif disorder == "hyperkalemia": self.state.k = 6.5
        elif disorder == "hypocalcemia": self.state.ca = 7.0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        na_dev = abs(st.na - 140) / 140
        k_dev = abs(st.k - 4.0) / 4.0
        ca_dev = abs(st.ca - 9.5) / 9.5
        total_dev = na_dev + k_dev + ca_dev
        st.filtrate_z = Z_TUBULAR * (1 + total_dev * 2)
        g2 = gamma_sq(st.filtrate_z, Z_TUBULAR)
        return {"disease": f"Electrolyte-{self.disorder}", "na": st.na,
                "k": st.k, "ca": st.ca, "gamma_sq": g2}

# ============================================================================
# 8. RENAL HYPERTENSION
# ============================================================================
@dataclass
class RenalHTNState:
    renal_artery_z: float = Z_GLOMERULAR * 1.5
    bp_systolic: float = 160.0
    renin: float = 2.0

class RenalHTNModel:
    def __init__(self, stenosis: float = 0.5):
        z = Z_GLOMERULAR / max(1 - stenosis, 0.01)
        self.state = RenalHTNState(renal_artery_z=z)
        self.on_ace_inhibitor = False
        self.tick_count = 0

    def start_ace_inhibitor(self):
        self.on_ace_inhibitor = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_ace_inhibitor:
            st.renin = max(0.5, st.renin * 0.98)
        g2 = gamma_sq(st.renal_artery_z, Z_GLOMERULAR)
        st.bp_systolic = 120 + g2 * 80
        return {"disease": "Renal HTN", "bp": st.bp_systolic,
                "renin": st.renin, "gamma_sq": g2}

# ============================================================================
# 9. PKD (Polycystic Kidney Disease)
# ============================================================================
@dataclass
class PKDState:
    kidney_volume_ml: float = 300.0  # Normal ~150 mL
    parenchymal_z: float = Z_GLOMERULAR
    gfr: float = GFR_NORMAL

class PKDModel:
    def __init__(self, initial_volume: float = 500.0, growth_rate: float = 0.001):
        self.state = PKDState(kidney_volume_ml=initial_volume)
        self.growth_rate = growth_rate
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.kidney_volume_ml = min(5000, st.kidney_volume_ml * (1 + self.growth_rate))
        # Cysts compress normal parenchyma
        compression = st.kidney_volume_ml / 150 - 1  # >0 when enlarged
        st.parenchymal_z = Z_GLOMERULAR * (1 + compression * 0.3)
        g2 = gamma_sq(st.parenchymal_z, Z_GLOMERULAR)
        st.gfr = max(5, GFR_NORMAL * (1 - g2))
        return {"disease": "PKD", "volume_ml": st.kidney_volume_ml,
                "gfr": st.gfr, "gamma_sq": g2}

# ============================================================================
# 10. RTA (Renal Tubular Acidosis)
# ============================================================================
@dataclass
class RTAState:
    tubular_z: float = Z_TUBULAR
    ph: float = 7.40
    hco3: float = 24.0  # mEq/L (normal 22–26)
    rta_type: int = 1    # Type 1 (distal) / Type 2 (proximal) / Type 4

class RTAModel:
    def __init__(self, rta_type: int = 1, severity: float = 0.5):
        self.state = RTAState(rta_type=rta_type)
        self.severity = severity
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        # Acid-base Z mismatch
        st.tubular_z = Z_TUBULAR * (1 + self.severity)
        g2 = gamma_sq(st.tubular_z, Z_TUBULAR)
        st.hco3 = max(8, 24 - g2 * 16)
        st.ph = 7.40 - g2 * 0.3
        return {"disease": f"RTA Type {st.rta_type}", "ph": st.ph,
                "hco3": st.hco3, "gamma_sq": g2}

# ============================================================================
# UNIFIED ENGINE
# ============================================================================
class ClinicalNephrologyEngine:
    DISEASE_CLASSES = {
        "aki": AKIModel, "ckd": CKDModel, "nephrolithiasis": NephrolithiasisModel,
        "nephrotic": NephroticModel, "nephritic": NephriticModel,
        "diabetic_nephropathy": DiabeticNephropathyModel,
        "electrolyte": ElectrolyteModel, "renal_htn": RenalHTNModel,
        "pkd": PKDModel, "rta": RTAModel,
    }

    def __init__(self):
        self.active_diseases: Dict[str, object] = {}
        self.tick_count = 0

    def add_disease(self, name: str, **kwargs):
        cls = self.DISEASE_CLASSES.get(name)
        if cls:
            self.active_diseases[name] = cls(**kwargs)

    def tick(self) -> Dict:
        self.tick_count += 1
        results = {}
        total_g2 = 0.0
        for name, model in self.active_diseases.items():
            r = model.tick()
            results[name] = r
            total_g2 += r.get("gamma_sq", 0.0)
        results["total_gamma_sq"] = total_g2
        results["renal_reserve"] = max(0.0, 1.0 - total_g2)
        return results
