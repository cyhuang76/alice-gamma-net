# -*- coding: utf-8 -*-
"""clinical_endocrinology.py — Top-10 Endocrine Diseases as Impedance-Mismatch Patterns
========================================================================================

│ Disease                  │ Γ Physics Mapping                          │ Clinical Scale  │
│──────────────────────────│────────────────────────────────────────────│────────────────│
│ 1. Type 1 DM             │ Autoimmune β-cell Z destruction           │ C-peptide/HbA1c│
│ 2. Type 2 DM             │ Insulin resistance Z mismatch             │ HOMA-IR/HbA1c  │
│ 3. Hyperthyroidism       │ Metabolic Z overdrive (low TSH)           │ FT4/TSH        │
│ 4. Hypothyroidism        │ Metabolic Z underdrive (high TSH)         │ TSH/FT4        │
│ 5. Cushing Syndrome      │ Cortisol Z excess                         │ 24h UFC        │
│ 6. Addison Disease       │ Cortisol Z deficit                        │ AM cortisol    │
│ 7. Pheochromocytoma      │ Catecholamine Z surge                     │ VMA/meta (24h) │
│ 8. Acromegaly            │ GH Z excess                               │ IGF-1          │
│ 9. DKA                   │ Metabolic cascade Γ → multi-organ         │ pH/AG/Glucose  │
│ 10. Thyroid Storm        │ Positive feedback Γ run-away              │ Burch-Wartofsky│

Endocrine = set-point controller. Disease = servo Z mismatch → feedback loop failure.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict

# Physical Constants (Ω)
Z_BETA: float = 75.0       # β-cell impedance
Z_RECEPTOR: float = 75.0   # Insulin receptor impedance
Z_THYROID: float = 65.0    # Thyroid follicular cell
Z_ADRENAL: float = 70.0    # Adrenal cortex
Z_PITUITARY: float = 60.0  # Anterior pituitary

def gamma_sq(z_l: float, z_0: float) -> float:
    g = (z_l - z_0) / (z_l + z_0)
    return g * g

# ============================================================================
# 1. TYPE 1 DM
# ============================================================================
@dataclass
class T1DMState:
    beta_cell_z: float = Z_BETA
    beta_cell_mass: float = 1.0
    glucose: float = 100.0     # mg/dL
    hba1c: float = 5.5
    c_peptide: float = 1.5     # ng/mL
    on_insulin: bool = False
    insulin_dose: float = 0.0

class T1DMModel:
    def __init__(self, destruction_rate: float = 0.01):
        self.state = T1DMState()
        self.destruction_rate = destruction_rate
        self.tick_count = 0

    def start_insulin(self, dose: float = 0.5):
        self.state.on_insulin = True
        self.state.insulin_dose = dose

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.beta_cell_mass = max(0.01, st.beta_cell_mass - self.destruction_rate)
        st.beta_cell_z = Z_BETA / max(st.beta_cell_mass, 0.01)
        g2 = gamma_sq(st.beta_cell_z, Z_BETA)
        insulin_eff = st.insulin_dose if st.on_insulin else 0
        st.glucose = 80 + g2 * 300 - insulin_eff * 200
        st.glucose = max(40, min(500, st.glucose))
        st.hba1c = 4.0 + st.glucose / 50
        st.c_peptide = max(0.01, 1.5 * st.beta_cell_mass)
        return {"disease": "T1DM", "glucose": st.glucose, "hba1c": st.hba1c,
                "c_peptide": st.c_peptide, "beta_mass": st.beta_cell_mass, "gamma_sq": g2}

# ============================================================================
# 2. TYPE 2 DM
# ============================================================================
@dataclass
class T2DMState:
    receptor_z: float = Z_RECEPTOR
    glucose: float = 130.0
    hba1c: float = 7.0
    homa_ir: float = 2.5
    on_metformin: bool = False

class T2DMModel:
    def __init__(self, insulin_resistance: float = 0.5):
        z = Z_RECEPTOR * (1 + insulin_resistance * 3)
        self.state = T2DMState(receptor_z=z)
        self.tick_count = 0

    def start_metformin(self):
        self.state.on_metformin = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if st.on_metformin:
            st.receptor_z = max(Z_RECEPTOR, st.receptor_z * 0.995)
        g2 = gamma_sq(st.receptor_z, Z_RECEPTOR)
        st.glucose = 80 + g2 * 250
        st.hba1c = 4.0 + st.glucose / 50
        st.homa_ir = 1.0 + g2 * 10
        return {"disease": "T2DM", "glucose": st.glucose, "hba1c": st.hba1c,
                "homa_ir": st.homa_ir, "gamma_sq": g2}

# ============================================================================
# 3. HYPERTHYROIDISM
# ============================================================================
@dataclass
class HyperthyroidState:
    thyroid_z: float = Z_THYROID
    ft4: float = 1.2       # ng/dL (normal 0.8–1.8)
    tsh: float = 2.0       # mIU/L  (normal 0.4–4.0)
    hr: float = 72.0

class HyperthyroidModel:
    def __init__(self, severity: float = 0.6):
        self.state = HyperthyroidState()
        self.severity = severity
        self.on_ati: bool = False
        self.tick_count = 0

    def start_antithyroid(self):
        self.on_ati = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        target_z = Z_THYROID * (1 - self.severity * 0.5)
        if self.on_ati:
            st.thyroid_z = min(Z_THYROID, st.thyroid_z + (Z_THYROID - st.thyroid_z) * 0.02)
        else:
            st.thyroid_z = target_z
        g2 = gamma_sq(st.thyroid_z, Z_THYROID)
        st.ft4 = 1.2 + g2 * 5  # Elevated
        st.tsh = max(0.01, 2.0 * (1 - g2 * 5))
        st.hr = 72 + g2 * 80
        return {"disease": "Hyperthyroidism", "ft4": st.ft4, "tsh": st.tsh,
                "hr": st.hr, "gamma_sq": g2}

# ============================================================================
# 4. HYPOTHYROIDISM
# ============================================================================
@dataclass
class HypothyroidState:
    thyroid_z: float = Z_THYROID
    ft4: float = 1.2
    tsh: float = 2.0
    metabolic_rate: float = 1.0

class HypothyroidModel:
    def __init__(self, severity: float = 0.6):
        self.state = HypothyroidState()
        self.severity = severity
        self.on_levothyroxine: bool = False
        self.tick_count = 0

    def start_levothyroxine(self):
        self.on_levothyroxine = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        target_z = Z_THYROID * (1 + self.severity * 3)
        if self.on_levothyroxine:
            st.thyroid_z = max(Z_THYROID, st.thyroid_z * 0.99)
        else:
            st.thyroid_z = target_z
        g2 = gamma_sq(st.thyroid_z, Z_THYROID)
        st.ft4 = max(0.1, 1.2 * (1 - g2))
        st.tsh = 2.0 + g2 * 50
        st.metabolic_rate = max(0.3, 1.0 - g2)
        return {"disease": "Hypothyroidism", "ft4": st.ft4, "tsh": st.tsh,
                "metabolic_rate": st.metabolic_rate, "gamma_sq": g2}

# ============================================================================
# 5. CUSHING SYNDROME
# ============================================================================
@dataclass
class CushingState:
    adrenal_z: float = Z_ADRENAL
    cortisol_24h: float = 50.0  # μg/24h (normal <50)
    acth: float = 20.0          # pg/mL
    glucose: float = 100.0

class CushingModel:
    def __init__(self, source: str = "pituitary", severity: float = 0.6):
        self.state = CushingState()
        self.source = source
        self.severity = severity
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.adrenal_z = Z_ADRENAL * (1 - self.severity * 0.5)
        g2 = gamma_sq(st.adrenal_z, Z_ADRENAL)
        st.cortisol_24h = 50 + g2 * 400
        st.acth = 20 * (1 + g2 * 3) if self.source == "pituitary" else 5
        st.glucose = 100 + g2 * 100
        return {"disease": "Cushing", "cortisol_24h": st.cortisol_24h,
                "acth": st.acth, "glucose": st.glucose, "gamma_sq": g2}

# ============================================================================
# 6. ADDISON DISEASE
# ============================================================================
@dataclass
class AddisonState:
    adrenal_z: float = Z_ADRENAL
    cortisol_am: float = 15.0  # μg/dL (normal 6–23)
    acth: float = 20.0
    na: float = 140.0
    k: float = 4.0

class AddisonModel:
    def __init__(self, destruction: float = 0.7):
        self.state = AddisonState()
        self.destruction = destruction
        self.on_replacement = False
        self.tick_count = 0

    def start_replacement(self):
        self.on_replacement = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_replacement:
            st.adrenal_z = max(Z_ADRENAL, st.adrenal_z * 0.99)
        else:
            st.adrenal_z = Z_ADRENAL * (1 + self.destruction * 5)
        g2 = gamma_sq(st.adrenal_z, Z_ADRENAL)
        st.cortisol_am = max(0.5, 15 * (1 - g2))
        st.acth = 20 + g2 * 180
        st.na = 140 - g2 * 15
        st.k = 4.0 + g2 * 2
        return {"disease": "Addison", "cortisol_am": st.cortisol_am,
                "acth": st.acth, "na": st.na, "k": st.k, "gamma_sq": g2}

# ============================================================================
# 7. PHEOCHROMOCYTOMA
# ============================================================================
@dataclass
class PheoState:
    adrenal_z: float = Z_ADRENAL
    vma_24h: float = 5.0      # mg/24h (normal 2–7)
    bp_systolic: float = 120.0
    hr: float = 72.0
    paroxysmal: bool = True

class PheoModel:
    def __init__(self, severity: float = 0.6):
        self.state = PheoState()
        self.severity = severity
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        # Paroxysmal catecholamine surge
        surge = 1.0 if self.tick_count % 5 == 0 else 0.3
        st.adrenal_z = Z_ADRENAL * (1 - self.severity * 0.4 * surge)
        g2 = gamma_sq(st.adrenal_z, Z_ADRENAL)
        st.vma_24h = 5 + g2 * 30
        st.bp_systolic = 120 + g2 * 100
        st.hr = 72 + g2 * 60
        return {"disease": "Pheo", "vma_24h": st.vma_24h, "bp": st.bp_systolic,
                "hr": st.hr, "gamma_sq": g2}

# ============================================================================
# 8. ACROMEGALY
# ============================================================================
@dataclass
class AcromegalyState:
    pituitary_z: float = Z_PITUITARY
    igf1: float = 200.0     # ng/mL (normal 100–300)
    gh: float = 2.0          # ng/mL (normal <5)
    ring_size_increase: float = 0.0

class AcromegalyModel:
    def __init__(self, severity: float = 0.5):
        self.state = AcromegalyState()
        self.severity = severity
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.pituitary_z = Z_PITUITARY * (1 - self.severity * 0.4)
        g2 = gamma_sq(st.pituitary_z, Z_PITUITARY)
        st.gh = 2 + g2 * 30
        st.igf1 = 200 + g2 * 600
        st.ring_size_increase = g2 * 4
        return {"disease": "Acromegaly", "igf1": st.igf1, "gh": st.gh,
                "gamma_sq": g2}

# ============================================================================
# 9. DKA
# ============================================================================
@dataclass
class DKAState:
    beta_z: float = Z_BETA
    glucose: float = 350.0
    ph: float = 7.20
    anion_gap: float = 18.0
    hco3: float = 12.0
    ketones: float = 5.0

class DKAModel:
    def __init__(self, severity: float = 0.7):
        self.state = DKAState()
        self.severity = severity
        self.on_insulin_drip = False
        self.tick_count = 0

    def start_insulin_drip(self):
        self.on_insulin_drip = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_insulin_drip:
            self.severity = max(0, self.severity - 0.02)
        st.beta_z = Z_BETA * (1 + self.severity * 8)
        g2 = gamma_sq(st.beta_z, Z_BETA)
        st.glucose = 80 + g2 * 500
        st.ketones = g2 * 10
        st.hco3 = max(5, 24 - st.ketones * 2)
        st.ph = 7.40 - 0.05 * (24 - st.hco3) / 3
        st.anion_gap = 12 + st.ketones * 1.5
        return {"disease": "DKA", "glucose": st.glucose, "ph": st.ph,
                "ag": st.anion_gap, "ketones": st.ketones, "gamma_sq": g2}

# ============================================================================
# 10. THYROID STORM
# ============================================================================
@dataclass
class ThyroidStormState:
    thyroid_z: float = Z_THYROID
    ft4: float = 5.0
    tsh: float = 0.01
    hr: float = 140.0
    temp_c: float = 39.5
    burch_wartofsky: int = 45  # ≥45 = storm

class ThyroidStormModel:
    def __init__(self, severity: float = 0.8):
        self.state = ThyroidStormState()
        self.severity = severity
        self.on_treatment = False
        self.tick_count = 0

    def start_treatment(self):
        self.on_treatment = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_treatment:
            self.severity = max(0, self.severity - 0.03)
        st.thyroid_z = Z_THYROID * (1 - self.severity * 0.6)
        g2 = gamma_sq(st.thyroid_z, Z_THYROID)
        st.ft4 = 1.2 + g2 * 8
        st.tsh = max(0.001, 2.0 * (1 - g2 * 10))
        st.hr = 72 + g2 * 120
        st.temp_c = 37 + g2 * 4
        bw = int(g2 * 80)
        st.burch_wartofsky = min(90, bw)
        return {"disease": "Thyroid Storm", "ft4": st.ft4, "hr": st.hr,
                "temp": st.temp_c, "BW_score": st.burch_wartofsky, "gamma_sq": g2}

# ============================================================================
# UNIFIED ENGINE
# ============================================================================
class ClinicalEndocrinologyEngine:
    DISEASE_CLASSES = {
        "t1dm": T1DMModel, "t2dm": T2DMModel,
        "hyperthyroid": HyperthyroidModel, "hypothyroid": HypothyroidModel,
        "cushing": CushingModel, "addison": AddisonModel,
        "pheo": PheoModel, "acromegaly": AcromegalyModel,
        "dka": DKAModel, "thyroid_storm": ThyroidStormModel,
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
        results["endocrine_reserve"] = max(0.0, 1.0 - total_g2)
        return results
