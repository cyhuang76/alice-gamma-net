# -*- coding: utf-8 -*-
"""clinical_dermatology.py — Top-10 Skin Diseases as Impedance-Mismatch Patterns
================================================================================

│ Disease                  │ Γ Physics Mapping                          │ Clinical Scale  │
│──────────────────────────│────────────────────────────────────────────│────────────────│
│ 1. Atopic Dermatitis     │ Barrier Z failure (filaggrin)             │ SCORAD         │
│ 2. Psoriasis             │ Turnover Z oscillation (keratinocyte)     │ PASI           │
│ 3. Urticaria             │ Histamine Z surge (mast cell)             │ UAS-7          │
│ 4. Herpes Zoster         │ Dermatome Z reactivation                  │ VAS / PHN risk │
│ 5. Melanoma              │ Melanocyte Z transformation               │ Breslow / TNM  │
│ 6. Contact Dermatitis    │ External allergen Z mismatch              │ Patch test     │
│ 7. Acne Vulgaris         │ Pilosebaceous Z obstruction               │ IGA scale      │
│ 8. Vitiligo              │ Melanocyte Z autoimmune destruction       │ VASI score     │
│ 9. Cellulitis            │ Subcutaneous Z infection spread           │ Eron class     │
│ 10. Burns                │ Thermal Z cascade failure                  │ TBSA% / Depth  │

Skin = first impedance interface between organism and environment.
Z_SKIN_BASE = 60 Ω. Disease = barrier Z mismatch → reflected/leaked signal.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict

# Physical Constants (Ω)
Z_SKIN: float = 60.0          # Normal skin impedance
Z_EPIDERMIS: float = 55.0     # Epidermal layer
Z_DERMIS: float = 65.0        # Dermal layer
Z_SUBCUTANEOUS: float = 70.0  # Subcutaneous fat
Z_MELANOCYTE: float = 50.0    # Melanocyte

from alice.body.clinical_common import gamma_sq, ClinicalEngineBase, make_template_disease, MetricSpec

# ============================================================================
# 1. ATOPIC DERMATITIS
# ============================================================================
@dataclass
class AtopicDermatitisState:
    barrier_z: float = Z_EPIDERMIS
    scorad: float = 0.0       # SCORAD 0–103
    ige: float = 100.0        # IU/mL
    tewl: float = 5.0         # Transepidermal water loss g/m²h

class AtopicDermatitisModel:
    def __init__(self, severity: float = 0.5, filaggrin_mutation: bool = False):
        self.state = AtopicDermatitisState()
        self.severity = severity
        self.filaggrin = filaggrin_mutation
        self.on_emollient = False
        self.on_steroid = False
        self.tick_count = 0

    def start_emollient(self):
        self.on_emollient = True

    def start_steroid(self):
        self.on_steroid = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        sev = self.severity
        if self.filaggrin: sev *= 1.5
        if self.on_emollient: sev *= 0.7
        if self.on_steroid: sev *= 0.4
        st.barrier_z = Z_EPIDERMIS * (1 - sev * 0.4)
        g2 = gamma_sq(st.barrier_z, Z_EPIDERMIS)
        st.scorad = min(103, g2 * 130)
        st.ige = 100 + g2 * 1000
        st.tewl = 5 + g2 * 30
        return {"disease": "Atopic Dermatitis", "scorad": st.scorad,
                "ige": st.ige, "tewl": st.tewl, "gamma_sq": g2}

# ============================================================================
# 2. PSORIASIS
# ============================================================================
PsoriasisModel = make_template_disease(
    "Psoriasis", Z_EPIDERMIS, z_coeff=3.0, default_severity=0.5,
    treatment_factor=0.2,
    metrics=(
        MetricSpec("pasi", 0, 90, max_val=72),
        MetricSpec("turnover", 28, -25.2, min_val=3),
    ),
    default_extra={"area": 15.0},
)

# ============================================================================
# 3. URTICARIA
# ============================================================================
UrticariaModel = make_template_disease(
    "Urticaria", Z_DERMIS, z_coeff=-0.3, default_severity=0.5,
    treatment_factor=0.3,
    metrics=(
        MetricSpec("uas7", 0, 55, max_val=42),
        MetricSpec("wheals", 0, 30, as_int=True),
    ),
)

# ============================================================================
# 4. HERPES ZOSTER
# ============================================================================
@dataclass
class HerpesZosterState:
    dermatome_z: float = Z_DERMIS
    pain_vas: float = 0.0     # VAS 0–10
    rash_severity: float = 0.0
    phn_risk: float = 0.0     # Post-herpetic neuralgia risk

class HerpesZosterModel:
    def __init__(self, age: float = 70, severity: float = 0.6):
        self.state = HerpesZosterState()
        self.severity = severity
        self.age = age
        self.on_antiviral = False
        self.tick_count = 0

    def start_antiviral(self):
        self.on_antiviral = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        sev = self.severity
        if self.on_antiviral:
            sev *= 0.5
        st.dermatome_z = Z_DERMIS * (1 + sev * 3)
        g2 = gamma_sq(st.dermatome_z, Z_DERMIS)
        st.pain_vas = min(10, g2 * 12)
        st.rash_severity = min(10, g2 * 10)
        # PHN risk increases with age and severity
        st.phn_risk = min(1.0, g2 * (self.age / 50))
        return {"disease": "Herpes Zoster", "pain": st.pain_vas,
                "rash": st.rash_severity, "phn_risk": st.phn_risk, "gamma_sq": g2}

# ============================================================================
# 5. MELANOMA
# ============================================================================
@dataclass
class MelanomaState:
    melanocyte_z: float = Z_MELANOCYTE
    breslow_mm: float = 0.5   # Thickness (mm)
    clark_level: int = 2      # I–V
    mitotic_rate: float = 1.0

class MelanomaModel:
    def __init__(self, breslow: float = 1.5, growth_rate: float = 0.01):
        self.state = MelanomaState(breslow_mm=breslow)
        self.growth_rate = growth_rate
        self.excised = False
        self.tick_count = 0

    def excision(self):
        self.excised = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.excised:
            pass  # No local progression
        else:
            st.breslow_mm = min(20, st.breslow_mm * (1 + self.growth_rate))
        z_change = st.breslow_mm / 1.0  # Thicker → more Z transformation
        st.melanocyte_z = Z_MELANOCYTE * (1 + z_change)
        g2 = gamma_sq(st.melanocyte_z, Z_MELANOCYTE)
        if st.breslow_mm <= 1: st.clark_level = 2
        elif st.breslow_mm <= 2: st.clark_level = 3
        elif st.breslow_mm <= 4: st.clark_level = 4
        else: st.clark_level = 5
        st.mitotic_rate = g2 * 10
        return {"disease": "Melanoma", "breslow": st.breslow_mm,
                "clark": st.clark_level, "mitotic": st.mitotic_rate, "gamma_sq": g2}

# ============================================================================
# 6. CONTACT DERMATITIS
# ============================================================================
@dataclass
class ContactDermatitisState:
    skin_z: float = Z_SKIN
    severity: float = 0.0     # 0–10
    area_percent: float = 0.0

class ContactDermatitisModel:
    def __init__(self, allergen_z: float = 200.0, exposure: float = 0.5):
        self.state = ContactDermatitisState()
        self.allergen_z = allergen_z
        self.exposure = exposure
        self.removed = False
        self.tick_count = 0

    def remove_allergen(self):
        self.removed = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        eff_exposure = 0.0 if self.removed else self.exposure
        g2 = gamma_sq(self.allergen_z, Z_SKIN) * eff_exposure
        st.severity = min(10, g2 * 12)
        st.area_percent = g2 * 20
        return {"disease": "Contact Dermatitis", "severity": st.severity,
                "area": st.area_percent, "gamma_sq": g2}

# ============================================================================
# 7. ACNE VULGARIS
# ============================================================================
AcneModel = make_template_disease(
    "Acne", Z_EPIDERMIS, z_coeff=2.0, default_severity=0.4,
    treatment_factor=0.4,
    metrics=(
        MetricSpec("iga", 0, 7, max_val=5, as_int=True),
        MetricSpec("lesions", 0, 50, as_int=True),
    ),
)

# ============================================================================
# 8. VITILIGO
# ============================================================================
@dataclass
class VitiligoState:
    melanocyte_z: float = Z_MELANOCYTE
    vasi: float = 0.0        # VASI 0–100
    depigmented_area: float = 0.0  # Fraction

class VitiligoModel:
    def __init__(self, severity: float = 0.4, progression: float = 0.005):
        self.state = VitiligoState()
        self.severity = severity
        self.progression = progression
        self.on_phototherapy = False
        self.tick_count = 0

    def start_phototherapy(self):
        self.on_phototherapy = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_phototherapy:
            self.severity = max(0, self.severity - 0.003)
        else:
            self.severity = min(1.0, self.severity + self.progression)
        st.melanocyte_z = Z_MELANOCYTE * (1 + self.severity * 5)
        g2 = gamma_sq(st.melanocyte_z, Z_MELANOCYTE)
        st.vasi = min(100, g2 * 130)
        st.depigmented_area = min(1.0, g2)
        return {"disease": "Vitiligo", "vasi": st.vasi,
                "area": st.depigmented_area, "gamma_sq": g2}

# ============================================================================
# 9. CELLULITIS
# ============================================================================
@dataclass
class CellulitisState:
    subcutaneous_z: float = Z_SUBCUTANEOUS
    eron_class: int = 1       # Eron I–IV
    temperature: float = 37.0
    wbc: float = 8.0

class CellulitisModel:
    def __init__(self, severity: float = 0.5):
        self.state = CellulitisState()
        self.severity = severity
        self.on_antibiotics = False
        self.tick_count = 0

    def start_antibiotics(self):
        self.on_antibiotics = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_antibiotics:
            self.severity = max(0, self.severity - 0.02)
        st.subcutaneous_z = Z_SUBCUTANEOUS * (1 + self.severity * 3)
        g2 = gamma_sq(st.subcutaneous_z, Z_SUBCUTANEOUS)
        st.temperature = 37.0 + g2 * 3
        st.wbc = 8.0 + g2 * 12
        if g2 < 0.1: st.eron_class = 1
        elif g2 < 0.3: st.eron_class = 2
        elif g2 < 0.5: st.eron_class = 3
        else: st.eron_class = 4
        return {"disease": "Cellulitis", "eron": st.eron_class,
                "temp": st.temperature, "wbc": st.wbc, "gamma_sq": g2}

# ============================================================================
# 10. BURNS
# ============================================================================
@dataclass
class BurnsState:
    skin_z: float = Z_SKIN
    tbsa: float = 0.0         # % Total Body Surface Area
    depth: str = "superficial" # superficial / partial / full
    fluid_requirement: float = 0.0  # Parkland formula mL

class BurnsModel:
    def __init__(self, tbsa: float = 20.0, depth: str = "partial"):
        self.state = BurnsState(tbsa=tbsa, depth=depth)
        self.resuscitation = False
        self.tick_count = 0

    def start_resuscitation(self):
        self.resuscitation = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        depth_factor = {"superficial": 0.3, "partial": 0.6, "full": 1.0}
        f = depth_factor.get(st.depth, 0.6)
        # Burn destroys skin Z
        st.skin_z = Z_SKIN * (1 + st.tbsa / 100 * f * 10)
        g2 = gamma_sq(st.skin_z, Z_SKIN)
        # Parkland: 4 mL × kg × %TBSA (assume 70 kg)
        st.fluid_requirement = 4 * 70 * st.tbsa
        if self.resuscitation:
            g2 *= 0.7  # Partial mitigation
        return {"disease": "Burns", "tbsa": st.tbsa, "depth": st.depth,
                "fluid_ml": st.fluid_requirement, "gamma_sq": g2}

# ============================================================================
# UNIFIED ENGINE
# ============================================================================
class ClinicalDermatologyEngine(ClinicalEngineBase):
    DISEASE_CLASSES = {
        "atopic_dermatitis": AtopicDermatitisModel, "psoriasis": PsoriasisModel,
        "urticaria": UrticariaModel, "herpes_zoster": HerpesZosterModel,
        "melanoma": MelanomaModel, "contact_dermatitis": ContactDermatitisModel,
        "acne": AcneModel, "vitiligo": VitiligoModel,
        "cellulitis": CellulitisModel, "burns": BurnsModel,
    }
    RESERVE_KEY = "skin_reserve"
