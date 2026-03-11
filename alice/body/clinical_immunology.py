# -*- coding: utf-8 -*-
"""clinical_immunology.py — Top-10 Immune Diseases as Impedance-Mismatch Patterns
=================================================================================

│ Disease                  │ Γ Physics Mapping                          │ Clinical Scale  │
│──────────────────────────│────────────────────────────────────────────│────────────────│
│ 1. SLE                   │ Self-tolerance Z breakdown (Z_self→0)      │ SLEDAI         │
│ 2. Rheumatoid Arthritis  │ Joint Z attack (Z_synovium drift)         │ DAS-28         │
│ 3. Anaphylaxis           │ IgE Z cascade → systemic Γ≈1             │ WAO Grade      │
│ 4. Allergic Rhinitis     │ Airway Z hypersensitivity                 │ ARIA Score     │
│ 5. HIV/AIDS              │ CD4⁺ Z progressive destruction            │ CD4 count      │
│ 6. Sepsis                │ Systemic Z storm → multi-organ            │ SOFA score     │
│ 7. Transplant Rejection  │ Donor Z mismatch → graft Γ               │ Banff Grade    │
│ 8. Sarcoidosis           │ Granuloma Z encapsulation                 │ Organ staging  │
│ 9. Vasculitis            │ Vascular Z inflammation                   │ BVAS           │
│ 10. Immunodeficiency     │ Immune channel Z attenuation              │ Ig levels      │

Immune = pattern recognition. Z_SELF (75Ω) vs Z_PATHOGEN (variable).
Disease = recognition Z failure → attack self or miss pathogen.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict

# Physical Constants (Ω)
Z_SELF: float = 75.0           # Normal host tissue
Z_PATHOGEN_MIN: float = 200.0  # Weakest pathogen
Z_SYNOVIUM: float = 70.0       # Joint synovial membrane
Z_VASCULAR: float = 60.0       # Vascular endothelium

from alice.body.clinical_common import gamma_sq, ClinicalEngineBase, make_template_disease, MetricSpec

# ============================================================================
# 1. SLE
# ============================================================================
@dataclass
class SLEState:
    self_z: float = Z_SELF
    anti_dsDNA: float = 10.0  # IU/mL (normal <30)
    complement_c3: float = 100.0  # mg/dL (normal 80–160)
    sledai: int = 0

class SLEModel:
    def __init__(self, severity: float = 0.5, flare: bool = False):
        self.state = SLEState()
        self.severity = severity
        self.flare = flare
        self.on_immunosuppressant = False
        self.tick_count = 0

    def trigger_flare(self):
        self.flare = True

    def start_treatment(self):
        self.on_immunosuppressant = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        sev = self.severity * (1.5 if self.flare else 1.0)
        if self.on_immunosuppressant:
            sev *= 0.5
            self.flare = False
        st.self_z = Z_SELF * (1 - sev * 0.3)
        g2 = gamma_sq(st.self_z, Z_SELF)
        st.anti_dsDNA = 10 + g2 * 300
        st.complement_c3 = max(20, 100 - g2 * 80)
        st.sledai = min(40, int(g2 * 50))
        return {"disease": "SLE", "sledai": st.sledai, "anti_dsDNA": st.anti_dsDNA,
                "c3": st.complement_c3, "gamma_sq": g2}

# ============================================================================
# 2. RHEUMATOID ARTHRITIS
# ============================================================================
RAModel = make_template_disease(
    "RA", Z_SYNOVIUM, z_coeff=2.0, default_severity=0.5,
    treatment_factor=0.6,
    metrics=(
        MetricSpec("das28", 1.0, 7),
        MetricSpec("rf", 10, 200),
        MetricSpec("crp", 0.5, 15),
    ),
    default_extra={"joints": 8},
)

# ============================================================================
# 3. ANAPHYLAXIS
# ============================================================================
@dataclass
class AnaphylaxisState:
    mast_cell_z: float = Z_SELF
    histamine: float = 0.0  # ng/mL
    bp_systolic: float = 120.0
    wao_grade: int = 0  # 1–5

class AnaphylaxisModel:
    def __init__(self, allergen_z: float = 300.0, sensitivity: float = 0.8):
        self.state = AnaphylaxisState()
        self.allergen_z = allergen_z
        self.sensitivity = sensitivity
        self.epinephrine_given = False
        self.tick_count = 0

    def give_epinephrine(self):
        self.epinephrine_given = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.epinephrine_given:
            st.mast_cell_z = Z_SELF
            self.sensitivity *= 0.3
            self.epinephrine_given = False  # Single dose
        g2 = gamma_sq(self.allergen_z, Z_SELF) * self.sensitivity
        st.histamine = g2 * 50
        st.bp_systolic = max(40, 120 - g2 * 100)
        if g2 < 0.1: st.wao_grade = 1
        elif g2 < 0.3: st.wao_grade = 2
        elif g2 < 0.5: st.wao_grade = 3
        elif g2 < 0.7: st.wao_grade = 4
        else: st.wao_grade = 5
        return {"disease": "Anaphylaxis", "wao": st.wao_grade, "bp": st.bp_systolic,
                "histamine": st.histamine, "gamma_sq": g2}

# ============================================================================
# 4. ALLERGIC RHINITIS
# ============================================================================
AllergicRhinitisModel = make_template_disease(
    "Allergic Rhinitis", 40.0, z_coeff=3.0, default_severity=0.4,
    treatment_factor=0.4,
    metrics=(
        MetricSpec("ige", 100, 500),
        MetricSpec("sneeze_freq", 0, 20),
    ),
)

# ============================================================================
# 5. HIV/AIDS
# ============================================================================
@dataclass
class HIVState:
    cd4_z: float = Z_SELF
    cd4_count: float = 800.0  # cells/μL (normal 500–1500)
    viral_load: float = 100000.0  # copies/mL
    on_art: bool = False

class HIVModel:
    def __init__(self, initial_cd4: float = 400.0):
        self.state = HIVState(cd4_count=initial_cd4)
        self.decline_rate = 0.003
        self.tick_count = 0

    def start_art(self):
        self.state.on_art = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if st.on_art:
            st.cd4_count = min(800, st.cd4_count + 2)
            st.viral_load = max(20, st.viral_load * 0.9)
        else:
            st.cd4_count = max(10, st.cd4_count - st.cd4_count * self.decline_rate)
            st.viral_load = min(1e7, st.viral_load * 1.01)
        st.cd4_z = Z_SELF * (st.cd4_count / 800)
        g2 = gamma_sq(st.cd4_z, Z_SELF)
        return {"disease": "HIV", "cd4": st.cd4_count, "viral_load": st.viral_load,
                "gamma_sq": g2}

# ============================================================================
# 6. SEPSIS
# ============================================================================
@dataclass
class SepsisState:
    systemic_z: float = Z_SELF
    sofa: int = 0
    lactate: float = 1.0      # mmol/L (normal <2)
    bp_mean: float = 75.0     # MAP mmHg
    wbc: float = 8.0          # ×10³/μL

class SepsisModel:
    def __init__(self, pathogen_z: float = 300.0, virulence: float = 0.7):
        self.state = SepsisState()
        self.pathogen_z = pathogen_z
        self.virulence = virulence
        self.on_antibiotics = False
        self.on_vasopressors = False
        self.tick_count = 0

    def start_antibiotics(self):
        self.on_antibiotics = True

    def start_vasopressors(self):
        self.on_vasopressors = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        v = self.virulence
        if self.on_antibiotics:
            v = max(0, v - 0.02)
            self.virulence = v
        g2 = gamma_sq(self.pathogen_z, Z_SELF) * v
        st.lactate = 1.0 + g2 * 12
        bp_base = 75 - g2 * 40
        st.bp_mean = bp_base + (15 if self.on_vasopressors else 0)
        st.wbc = 8 + g2 * 20 if g2 < 0.5 else max(1, 8 - g2 * 10)
        # SOFA
        sofa = 0
        if st.lactate > 2: sofa += 2
        if st.bp_mean < 65: sofa += 3
        if g2 > 0.3: sofa += 2
        if g2 > 0.5: sofa += 3
        st.sofa = min(24, sofa)
        return {"disease": "Sepsis", "sofa": st.sofa, "lactate": st.lactate,
                "map": st.bp_mean, "gamma_sq": g2}

# ============================================================================
# 7. TRANSPLANT REJECTION
# ============================================================================
@dataclass
class RejectionState:
    graft_z: float = Z_SELF
    creatinine: float = 1.2
    banff_grade: str = "borderline"
    on_calcineurin: bool = False

class TransplantRejectionModel:
    def __init__(self, hla_mismatch: int = 3, organ: str = "kidney"):
        self.state = RejectionState()
        self.hla_mismatch = hla_mismatch
        self.organ = organ
        self.tick_count = 0

    def start_immunosuppression(self):
        self.state.on_calcineurin = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        z_mismatch = Z_SELF * (1 + self.hla_mismatch * 0.15)
        if st.on_calcineurin:
            z_mismatch = Z_SELF + (z_mismatch - Z_SELF) * 0.3
        st.graft_z = z_mismatch
        g2 = gamma_sq(st.graft_z, Z_SELF)
        st.creatinine = 1.0 + g2 * 5
        if g2 < 0.05: st.banff_grade = "no rejection"
        elif g2 < 0.15: st.banff_grade = "borderline"
        elif g2 < 0.3: st.banff_grade = "1A"
        elif g2 < 0.5: st.banff_grade = "1B"
        else: st.banff_grade = "2"
        return {"disease": "Transplant Rejection", "banff": st.banff_grade,
                "creatinine": st.creatinine, "organ": self.organ, "gamma_sq": g2}

# ============================================================================
# 8. SARCOIDOSIS
# ============================================================================
@dataclass
class SarcoidosisState:
    granuloma_z: float = Z_SELF * 1.5  # Encapsulated granuloma Z
    ace_level: float = 30.0     # U/L (normal 8–52)
    lung_involved: bool = True
    stage: int = 2  # Scadding 0–4

class SarcoidosisModel:
    def __init__(self, severity: float = 0.4, organs_involved: int = 1):
        self.state = SarcoidosisState()
        self.severity = severity
        self.organs_involved = organs_involved
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.granuloma_z = Z_SELF * (1 + self.severity * 2)
        g2 = gamma_sq(st.granuloma_z, Z_SELF)
        st.ace_level = 30 + g2 * 100
        if g2 < 0.1: st.stage = 1
        elif g2 < 0.3: st.stage = 2
        elif g2 < 0.5: st.stage = 3
        else: st.stage = 4
        return {"disease": "Sarcoidosis", "stage": st.stage, "ace": st.ace_level,
                "organs": self.organs_involved, "gamma_sq": g2}

# ============================================================================
# 9. VASCULITIS
# ============================================================================
@dataclass
class VasculitisState:
    vascular_z: float = Z_VASCULAR
    esr: float = 10.0      # mm/h
    anca: bool = False
    bvas: int = 0           # Birmingham Vasculitis Activity Score

class VasculitisModel:
    def __init__(self, vessel_size: str = "small", severity: float = 0.5):
        self.state = VasculitisState()
        self.vessel_size = vessel_size
        self.severity = severity
        self.on_treatment = False
        self.tick_count = 0

    def start_treatment(self):
        self.on_treatment = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        sev = self.severity * (0.4 if self.on_treatment else 1.0)
        st.vascular_z = Z_VASCULAR * (1 + sev * 3)
        g2 = gamma_sq(st.vascular_z, Z_VASCULAR)
        st.esr = 10 + g2 * 100
        st.anca = g2 > 0.2 and self.vessel_size == "small"
        st.bvas = min(63, int(g2 * 80))
        return {"disease": "Vasculitis", "bvas": st.bvas, "esr": st.esr,
                "anca": st.anca, "vessel": self.vessel_size, "gamma_sq": g2}

# ============================================================================
# 10. PRIMARY IMMUNODEFICIENCY
# ============================================================================
@dataclass
class ImmunodeficiencyState:
    immune_z: float = Z_SELF
    igg: float = 1000.0     # mg/dL (normal 700–1600)
    iga: float = 200.0      # mg/dL (normal 70–400)
    infections_per_year: float = 2.0

class ImmunodeficiencyModel:
    def __init__(self, deficiency: str = "cvid", severity: float = 0.6):
        self.state = ImmunodeficiencyState()
        self.deficiency = deficiency
        self.severity = severity
        self.on_ivig = False
        self.tick_count = 0

    def start_ivig(self):
        self.on_ivig = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        sev = self.severity * (0.3 if self.on_ivig else 1.0)
        st.immune_z = Z_SELF * (1 + sev * 4)
        g2 = gamma_sq(st.immune_z, Z_SELF)
        st.igg = max(50, 1000 * (1 - g2))
        st.iga = max(5, 200 * (1 - g2))
        st.infections_per_year = 2 + g2 * 15
        return {"disease": f"Immunodeficiency-{self.deficiency}", "igg": st.igg,
                "iga": st.iga, "infections_yr": st.infections_per_year, "gamma_sq": g2}

# ============================================================================
# UNIFIED ENGINE
# ============================================================================
class ClinicalImmunologyEngine(ClinicalEngineBase):
    DISEASE_CLASSES = {
        "sle": SLEModel, "ra": RAModel, "anaphylaxis": AnaphylaxisModel,
        "allergic_rhinitis": AllergicRhinitisModel, "hiv": HIVModel,
        "sepsis": SepsisModel, "transplant_rejection": TransplantRejectionModel,
        "sarcoidosis": SarcoidosisModel, "vasculitis": VasculitisModel,
        "immunodeficiency": ImmunodeficiencyModel,
    }
    RESERVE_KEY = "immune_reserve"
