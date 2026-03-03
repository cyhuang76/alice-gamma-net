# -*- coding: utf-8 -*-
"""clinical_oncology.py — Top-10 Cancers as Impedance-Transformation Patterns
==============================================================================

│ Cancer                   │ Γ Physics Mapping                          │ Clinical Scale  │
│──────────────────────────│────────────────────────────────────────────│────────────────│
│ 1. Lung Cancer           │ Airway Z infiltration → obstruction       │ TNM / EGFR     │
│ 2. Breast Cancer         │ Ductal Z transformation                   │ TNM / ER/PR    │
│ 3. Colorectal Cancer     │ Mucosal Z growth → obstruction            │ TNM / CEA      │
│ 4. Hepatocellular        │ Hepatic Z transformation                  │ BCLC / AFP     │
│ 5. Pancreatic Cancer     │ Ductal Z stenosis + invasion              │ TNM / CA19-9   │
│ 6. Brain Tumor (GBM)     │ Neural Z infiltration + edema             │ KPS / WHO Gr   │
│ 7. Leukemia              │ Marrow Z takeover → pancytopenia          │ FAB / WBC      │
│ 8. Lymphoma              │ Lymphoid Z expansion                      │ Ann Arbor      │
│ 9. Renal Cell Carcinoma  │ Parenchymal Z invasion                    │ TNM / IMDC     │
│ 10. Metastasis           │ Impedance camouflage (Γ_tumor → Γ_host)  │ Sites / Burden │

CANCER AS IMPEDANCE CAMOUFLAGE:
Normal cell: Z_cell ≈ Z_tissue → Γ ≈ 0, immune invisible by design.
Cancer cell: Z_tumor drifts → Γ > 0, BUT tumor evolves camouflage:
  Z_tumor → Z_host (down-regulation of MHC, PD-L1 expression).
This is why cancer evades immunity: it minimizes its own Γ relative to self,
while maximizing its Γ relative to organ function.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List

# Physical Constants (Ω)
Z_HOST: float = 75.0          # Normal host tissue impedance
Z_IMMUNE: float = 75.0        # Immune self-recognition
Z_LUNG_TISSUE: float = 30.0   # Lung parenchyma
Z_BREAST_TISSUE: float = 55.0
Z_COLON_TISSUE: float = 65.0
Z_LIVER_TISSUE: float = 60.0
Z_PANCREAS_TISSUE: float = 55.0
Z_BRAIN_TISSUE: float = 50.0
Z_MARROW: float = 70.0
Z_LYMPH: float = 45.0
Z_RENAL: float = 50.0

def gamma_sq(z_l: float, z_0: float) -> float:
    g = (z_l - z_0) / (z_l + z_0)
    return g * g

# ============================================================================
# UNIVERSAL TUMOR MODEL (shared physics)
# ============================================================================
@dataclass
class TumorCore:
    """Base tumor state: all cancers share these impedance properties."""
    tumor_z: float = Z_HOST    # Current tumor impedance
    camouflage: float = 0.8    # Immune camouflage (0=visible, 1=invisible)
    size_cm: float = 2.0       # Primary tumor size
    growth_rate: float = 0.01  # Per-tick fractional growth
    on_treatment: bool = False
    metastatic: bool = False
    sites: int = 0

    def grow(self) -> float:
        """Grow tumor, return Γ² relative to organ."""
        if self.on_treatment:
            self.size_cm = max(0.1, self.size_cm * (1 - self.growth_rate * 0.5))
        else:
            self.size_cm = min(20, self.size_cm * (1 + self.growth_rate))
        return self.size_cm

# ============================================================================
# 1. LUNG CANCER
# ============================================================================
@dataclass
class LungCancerState:
    core: TumorCore = None
    histology: str = "adenocarcinoma"  # adeno/squamous/small-cell
    egfr_mutation: bool = False
    fev1_percent: float = 80.0
    stage: str = "II"

    def __post_init__(self):
        if self.core is None:
            self.core = TumorCore(tumor_z=Z_LUNG_TISSUE * 2)

class LungCancerModel:
    def __init__(self, size: float = 3.0, histology: str = "adenocarcinoma",
                 egfr: bool = False):
        self.state = LungCancerState(histology=histology, egfr_mutation=egfr)
        self.state.core.size_cm = size
        self.tick_count = 0

    def start_treatment(self):
        self.state.core.on_treatment = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.core.grow()
        z_ratio = st.core.size_cm / 3
        st.core.tumor_z = Z_LUNG_TISSUE * (1 + z_ratio * 2)
        g2 = gamma_sq(st.core.tumor_z, Z_LUNG_TISSUE)
        st.fev1_percent = max(20, 80 * (1 - g2))
        if g2 < 0.1: st.stage = "I"
        elif g2 < 0.25: st.stage = "II"
        elif g2 < 0.5: st.stage = "III"
        else: st.stage = "IV"
        return {"disease": "Lung Cancer", "stage": st.stage, "size": st.core.size_cm,
                "fev1": st.fev1_percent, "histology": st.histology, "gamma_sq": g2}

# ============================================================================
# 2. BREAST CANCER
# ============================================================================
@dataclass
class BreastCancerState:
    core: TumorCore = None
    er_positive: bool = True
    her2: bool = False
    stage: str = "II"

    def __post_init__(self):
        if self.core is None:
            self.core = TumorCore(tumor_z=Z_BREAST_TISSUE * 1.5)

class BreastCancerModel:
    def __init__(self, size: float = 2.5, er: bool = True, her2: bool = False):
        self.state = BreastCancerState(er_positive=er, her2=her2)
        self.state.core.size_cm = size
        self.tick_count = 0

    def start_treatment(self):
        self.state.core.on_treatment = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.core.grow()
        z_ratio = st.core.size_cm / 2
        st.core.tumor_z = Z_BREAST_TISSUE * (1 + z_ratio)
        g2 = gamma_sq(st.core.tumor_z, Z_BREAST_TISSUE)
        if g2 < 0.1: st.stage = "I"
        elif g2 < 0.25: st.stage = "II"
        elif g2 < 0.5: st.stage = "III"
        else: st.stage = "IV"
        return {"disease": "Breast Cancer", "stage": st.stage, "size": st.core.size_cm,
                "ER": st.er_positive, "HER2": st.her2, "gamma_sq": g2}

# ============================================================================
# 3. COLORECTAL CANCER
# ============================================================================
@dataclass
class CRCState:
    core: TumorCore = None
    cea: float = 3.0          # ng/mL (normal <5)
    stage: str = "II"
    obstruction: bool = False

    def __post_init__(self):
        if self.core is None:
            self.core = TumorCore(tumor_z=Z_COLON_TISSUE * 1.8)

class CRCModel:
    def __init__(self, size: float = 3.0):
        self.state = CRCState()
        self.state.core.size_cm = size
        self.tick_count = 0

    def start_treatment(self):
        self.state.core.on_treatment = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.core.grow()
        z_ratio = st.core.size_cm / 3
        st.core.tumor_z = Z_COLON_TISSUE * (1 + z_ratio * 2)
        g2 = gamma_sq(st.core.tumor_z, Z_COLON_TISSUE)
        st.cea = 3 + g2 * 100
        st.obstruction = g2 > 0.4
        if g2 < 0.1: st.stage = "I"
        elif g2 < 0.25: st.stage = "II"
        elif g2 < 0.5: st.stage = "III"
        else: st.stage = "IV"
        return {"disease": "CRC", "stage": st.stage, "cea": st.cea,
                "obstruction": st.obstruction, "gamma_sq": g2}

# ============================================================================
# 4. HEPATOCELLULAR CARCINOMA (HCC)
# ============================================================================
@dataclass
class HCCState:
    core: TumorCore = None
    afp: float = 10.0         # ng/mL (normal <10)
    bclc: str = "B"
    child_pugh: str = "A"

    def __post_init__(self):
        if self.core is None:
            self.core = TumorCore(tumor_z=Z_LIVER_TISSUE * 2)

class HCCModel:
    def __init__(self, size: float = 4.0, cirrhotic: bool = True):
        self.state = HCCState()
        self.state.core.size_cm = size
        self.cirrhotic = cirrhotic
        self.tick_count = 0

    def start_treatment(self):
        self.state.core.on_treatment = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.core.grow()
        z_ratio = st.core.size_cm / 3
        st.core.tumor_z = Z_LIVER_TISSUE * (1 + z_ratio * 2)
        g2 = gamma_sq(st.core.tumor_z, Z_LIVER_TISSUE)
        st.afp = 10 + g2 * 2000
        if g2 < 0.1: st.bclc = "0"
        elif g2 < 0.2: st.bclc = "A"
        elif g2 < 0.4: st.bclc = "B"
        elif g2 < 0.6: st.bclc = "C"
        else: st.bclc = "D"
        return {"disease": "HCC", "bclc": st.bclc, "afp": st.afp,
                "size": st.core.size_cm, "gamma_sq": g2}

# ============================================================================
# 5. PANCREATIC CANCER
# ============================================================================
@dataclass
class PancreaticCancerState:
    core: TumorCore = None
    ca19_9: float = 20.0      # U/mL (normal <37)
    stage: str = "II"
    biliary_obstruction: bool = False

    def __post_init__(self):
        if self.core is None:
            self.core = TumorCore(tumor_z=Z_PANCREAS_TISSUE * 2.5, growth_rate=0.015)

class PancreaticCancerModel:
    def __init__(self, size: float = 3.0):
        self.state = PancreaticCancerState()
        self.state.core.size_cm = size
        self.tick_count = 0

    def start_treatment(self):
        self.state.core.on_treatment = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.core.grow()
        z_ratio = st.core.size_cm / 2
        st.core.tumor_z = Z_PANCREAS_TISSUE * (1 + z_ratio * 3)
        g2 = gamma_sq(st.core.tumor_z, Z_PANCREAS_TISSUE)
        st.ca19_9 = 20 + g2 * 5000
        st.biliary_obstruction = g2 > 0.3
        if g2 < 0.15: st.stage = "I"
        elif g2 < 0.3: st.stage = "II"
        elif g2 < 0.5: st.stage = "III"
        else: st.stage = "IV"
        return {"disease": "Pancreatic Cancer", "stage": st.stage, "ca19_9": st.ca19_9,
                "biliary": st.biliary_obstruction, "gamma_sq": g2}

# ============================================================================
# 6. BRAIN TUMOR (GBM)
# ============================================================================
@dataclass
class GBMState:
    core: TumorCore = None
    kps: int = 80             # Karnofsky Performance Status 0–100
    who_grade: int = 4
    edema_volume_ml: float = 10.0
    midline_shift: bool = False

    def __post_init__(self):
        if self.core is None:
            self.core = TumorCore(tumor_z=Z_BRAIN_TISSUE * 3, growth_rate=0.012)

class GBMModel:
    def __init__(self, size: float = 4.0, who_grade: int = 4):
        self.state = GBMState(who_grade=who_grade)
        self.state.core.size_cm = size
        self.tick_count = 0

    def start_treatment(self):
        self.state.core.on_treatment = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.core.grow()
        z_ratio = st.core.size_cm / 3
        st.core.tumor_z = Z_BRAIN_TISSUE * (1 + z_ratio * 3)
        g2 = gamma_sq(st.core.tumor_z, Z_BRAIN_TISSUE)
        st.kps = max(0, int(100 * (1 - g2)))
        st.edema_volume_ml = g2 * 60
        st.midline_shift = g2 > 0.4
        return {"disease": "GBM", "kps": st.kps, "who": st.who_grade,
                "size": st.core.size_cm, "edema_ml": st.edema_volume_ml,
                "gamma_sq": g2}

# ============================================================================
# 7. LEUKEMIA
# ============================================================================
@dataclass
class LeukemiaState:
    marrow_z: float = Z_MARROW
    wbc: float = 50.0         # ×10³/μL
    hgb: float = 10.0         # g/dL
    platelets: float = 80.0   # ×10³/μL
    blast_percent: float = 40.0
    subtype: str = "AML"

class LeukemiaModel:
    def __init__(self, subtype: str = "AML", blast: float = 50.0):
        self.state = LeukemiaState(subtype=subtype, blast_percent=blast)
        self.on_chemo = False
        self.tick_count = 0

    def start_chemo(self):
        self.on_chemo = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_chemo:
            st.blast_percent = max(0, st.blast_percent * 0.95)
        # Blasts displace normal marrow → Z mismatch
        blast_frac = st.blast_percent / 100
        st.marrow_z = Z_MARROW * (1 + blast_frac * 5)
        g2 = gamma_sq(st.marrow_z, Z_MARROW)
        st.wbc = 8 + blast_frac * 100
        st.hgb = max(4, 14 * (1 - g2))
        st.platelets = max(10, 250 * (1 - g2))
        return {"disease": f"Leukemia-{st.subtype}", "blasts": st.blast_percent,
                "wbc": st.wbc, "hgb": st.hgb, "plt": st.platelets, "gamma_sq": g2}

# ============================================================================
# 8. LYMPHOMA
# ============================================================================
@dataclass
class LymphomaState:
    lymph_z: float = Z_LYMPH
    ldh: float = 200.0        # U/L (normal 140–280)
    ann_arbor: str = "II"
    b_symptoms: bool = False
    subtype: str = "DLBCL"

class LymphomaModel:
    def __init__(self, subtype: str = "DLBCL", severity: float = 0.4):
        self.state = LymphomaState(subtype=subtype)
        self.severity = severity
        self.on_rchop = False
        self.tick_count = 0

    def start_rchop(self):
        self.on_rchop = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        sev = self.severity * (0.3 if self.on_rchop else 1.0)
        st.lymph_z = Z_LYMPH * (1 + sev * 4)
        g2 = gamma_sq(st.lymph_z, Z_LYMPH)
        st.ldh = 200 + g2 * 500
        st.b_symptoms = g2 > 0.3
        if g2 < 0.1: st.ann_arbor = "I"
        elif g2 < 0.25: st.ann_arbor = "II"
        elif g2 < 0.5: st.ann_arbor = "III"
        else: st.ann_arbor = "IV"
        return {"disease": f"Lymphoma-{st.subtype}", "stage": st.ann_arbor,
                "ldh": st.ldh, "b_symptoms": st.b_symptoms, "gamma_sq": g2}

# ============================================================================
# 9. RENAL CELL CARCINOMA
# ============================================================================
@dataclass
class RCCState:
    core: TumorCore = None
    hgb: float = 14.0
    calcium: float = 9.5      # mg/dL
    stage: str = "II"
    imdc_risk: str = "intermediate"

    def __post_init__(self):
        if self.core is None:
            self.core = TumorCore(tumor_z=Z_RENAL * 2)

class RCCModel:
    def __init__(self, size: float = 5.0):
        self.state = RCCState()
        self.state.core.size_cm = size
        self.tick_count = 0

    def start_treatment(self):
        self.state.core.on_treatment = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.core.grow()
        z_ratio = st.core.size_cm / 4
        st.core.tumor_z = Z_RENAL * (1 + z_ratio * 2)
        g2 = gamma_sq(st.core.tumor_z, Z_RENAL)
        st.hgb = max(7, 14 * (1 - g2 * 0.5))
        st.calcium = 9.5 + g2 * 4  # Hypercalcemia
        if g2 < 0.1: st.stage = "I"
        elif g2 < 0.25: st.stage = "II"
        elif g2 < 0.5: st.stage = "III"
        else: st.stage = "IV"
        risk_points = (1 if st.hgb < 12 else 0) + (1 if st.calcium > 10 else 0)
        st.imdc_risk = ["favorable", "intermediate", "poor"][min(risk_points, 2)]
        return {"disease": "RCC", "stage": st.stage, "hgb": st.hgb,
                "calcium": st.calcium, "imdc": st.imdc_risk, "gamma_sq": g2}

# ============================================================================
# 10. METASTASIS (Universal Model)
# ============================================================================
@dataclass
class MetastasisState:
    primary_z: float = Z_HOST
    target_z: float = Z_HOST
    sites: List[str] = None
    burden: float = 0.0        # Total metastatic volume (cm³)
    camouflage: float = 0.9    # High camouflage enables metastasis

    def __post_init__(self):
        if self.sites is None:
            self.sites = []

class MetastasisModel:
    """Universal metastasis model:
    Cancer = impedance camouflage. The tumor that metastasizes has learned
    to match Z_target at every new site: Γ_tumor→host ≈ 0 (immune evasion),
    while Γ_tumor→organ ≫ 0 (organ dysfunction).
    """
    def __init__(self, primary: str = "lung", initial_sites: List[str] = None,
                 camouflage: float = 0.9):
        self.state = MetastasisState(camouflage=camouflage)
        self.primary = primary
        self.state.sites = initial_sites or ["liver"]
        self.on_immunotherapy = False
        self.tick_count = 0

    def start_immunotherapy(self):
        """Immunotherapy = break camouflage → restore Γ_tumor→immune."""
        self.on_immunotherapy = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_immunotherapy:
            st.camouflage = max(0.1, st.camouflage * 0.97)
        # Immune detection = Γ(tumor, immune) * (1 - camouflage)
        immune_visibility = (1 - st.camouflage)
        # Growth inversely proportional to immune visibility
        growth = 0.02 * st.camouflage
        st.burden = max(0.1, st.burden + growth * len(st.sites))
        # Organ damage at each site
        g2_per_site = gamma_sq(Z_HOST * (1 + st.burden / 10), Z_HOST)
        total_g2 = g2_per_site * len(st.sites)
        return {"disease": f"Metastasis-{self.primary}", "sites": st.sites,
                "burden_cm3": st.burden, "camouflage": st.camouflage,
                "immune_visibility": immune_visibility,
                "gamma_sq": min(1.0, total_g2)}

# ============================================================================
# UNIFIED ENGINE
# ============================================================================
class ClinicalOncologyEngine:
    DISEASE_CLASSES = {
        "lung_cancer": LungCancerModel, "breast_cancer": BreastCancerModel,
        "crc": CRCModel, "hcc": HCCModel,
        "pancreatic": PancreaticCancerModel, "gbm": GBMModel,
        "leukemia": LeukemiaModel, "lymphoma": LymphomaModel,
        "rcc": RCCModel, "metastasis": MetastasisModel,
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
        results["host_reserve"] = max(0.0, 1.0 - total_g2)
        return results
