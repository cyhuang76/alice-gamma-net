# -*- coding: utf-8 -*-
"""clinical_gastroenterology.py — Top-10 GI/Hepatobiliary Diseases as Impedance Patterns
======================================================================================

│ Disease                │ Γ Physics Mapping                              │ Clinical Scale │
│────────────────────────│────────────────────────────────────────────────│───────────────│
│ 1. GERD                │ LES Z failure → retrograde Γ                   │ LA Grade A–D  │
│ 2. Peptic Ulcer        │ Mucosal barrier Z breakdown                    │ Forrest I–III │
│ 3. IBD (Crohn's/UC)    │ Mucosal Z oscillation (inflammation)          │ CDAI / Mayo   │
│ 4. IBS                 │ Gut-brain axis Γ elevation                     │ Rome IV type  │
│ 5. Liver Cirrhosis     │ Progressive hepatic Z → portal HTN            │ Child-Pugh    │
│ 6. Cholelithiasis      │ Bile duct Z obstruction                        │ Murphy sign   │
│ 7. Pancreatitis        │ Ductal Z obstruction → autodigestion           │ Ranson score  │
│ 8. Bowel Obstruction   │ Transmission line Z discontinuity              │ SBO grade     │
│ 9. Colorectal Cancer   │ Mucosal Z transformation                       │ TNM / CEA     │
│ 10. Hepatitis B/C      │ Viral Z contamination of hepatocytes           │ ALT / Fibrosis│

The GI tract is a transmission line from mouth to anus:
    Z varies along its length (stomach acid = low Z, intestinal mucosa = matched Z).
    Every GI disease = Z mismatch at a specific segment.
"""

from __future__ import annotations
import math, random
from dataclasses import dataclass, field
from typing import Dict, List

# ============================================================================
# Physical Constants
# ============================================================================
Z_ESOPHAGEAL: float = 55.0
Z_GASTRIC: float = 40.0        # Acidic environment → lower Z
Z_MUCOSA: float = 65.0         # Intestinal mucosal Z
Z_HEPATOCYTE: float = 65.0     # Normal liver cell Z
Z_BILE_DUCT: float = 45.0      # Normal bile duct Z
Z_PANCREATIC: float = 60.0     # Normal pancreatic duct Z
Z_COLONIC: float = 55.0        # Normal colonic mucosal Z
Z_LES: float = 70.0            # Lower esophageal sphincter (closed = high Z barrier)
Z_LES_OPEN: float = 15.0       # LES open (swallowing)
Z_PORTAL: float = 35.0         # Normal portal venous Z
Z_OBSTRUCTED: float = 1e6      # Complete obstruction


def gamma(z_l: float, z_0: float) -> float:
    return (z_l - z_0) / (z_l + z_0)

def gamma_sq(z_l: float, z_0: float) -> float:
    g = gamma(z_l, z_0)
    return g * g


# ============================================================================
# 1. GERD
# ============================================================================
@dataclass
class GERDState:
    les_z: float = Z_LES
    reflux_episodes: int = 0
    la_grade: str = "A"

class GERDModel:
    """GERD: LES Z failure → gastric acid reflux into esophagus."""

    def __init__(self, les_weakness: float = 0.5):
        self.state = GERDState(les_z=Z_LES * (1 - les_weakness * 0.7))
        self.rng = random.Random(42)
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        # Probabilistic reflux when LES Z too low
        threshold = st.les_z / Z_LES
        if self.rng.random() > threshold:
            st.reflux_episodes += 1
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(Z_GASTRIC, st.les_z)  # Acid vs weakened barrier
        rate = st.reflux_episodes / max(self.tick_count, 1) * 100
        if rate < 5: st.la_grade = "A"
        elif rate < 15: st.la_grade = "B"
        elif rate < 30: st.la_grade = "C"
        else: st.la_grade = "D"
        return {"disease": "GERD", "les_z": st.les_z, "reflux_rate": rate,
                "la_grade": st.la_grade, "gamma_sq": g2}


# ============================================================================
# 2. PEPTIC ULCER
# ============================================================================
@dataclass
class PepticUlcerState:
    mucosal_z: float = Z_MUCOSA
    ulcer_depth: float = 0.0   # 0=healthy, 1=perforation
    bleeding: bool = False

class PepticUlcerModel:
    """Peptic Ulcer: mucosal barrier Z breakdown → acid-tissue Γ."""

    def __init__(self, h_pylori: bool = True, nsaid: bool = False):
        rate = 0.003
        if h_pylori: rate += 0.002
        if nsaid: rate += 0.002
        self.erosion_rate = rate
        self.state = PepticUlcerState()
        self.on_ppi = False
        self.tick_count = 0

    def start_ppi(self):
        self.on_ppi = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_ppi:
            st.ulcer_depth = max(0, st.ulcer_depth - 0.004)
        else:
            st.ulcer_depth = min(1.0, st.ulcer_depth + self.erosion_rate)
        st.mucosal_z = Z_MUCOSA * (1 - st.ulcer_depth * 0.8)
        st.bleeding = st.ulcer_depth > 0.5
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(Z_GASTRIC, st.mucosal_z)
        if st.ulcer_depth < 0.2: forrest = "III"
        elif st.ulcer_depth < 0.5: forrest = "II"
        else: forrest = "I"
        return {"disease": "Peptic Ulcer", "depth": st.ulcer_depth,
                "bleeding": st.bleeding, "forrest": forrest, "gamma_sq": g2}


# ============================================================================
# 3. IBD (Crohn's / UC)
# ============================================================================
@dataclass
class IBDState:
    subtype: str = "crohn"  # crohn / uc
    mucosal_z: float = Z_MUCOSA
    inflammation: float = 0.0  # 0–1
    flare: bool = False

class IBDModel:
    """IBD: chronic mucosal Z oscillation — flare/remission cycles."""

    def __init__(self, subtype: str = "crohn", severity: float = 0.5):
        self.state = IBDState(subtype=subtype)
        self.severity = severity
        self.on_biologic = False
        self.rng = random.Random(42)
        self.tick_count = 0

    def start_biologic(self):
        self.on_biologic = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        # Flare probability
        flare_prob = self.severity * (0.3 if self.on_biologic else 1.0) * 0.02
        if not st.flare and self.rng.random() < flare_prob:
            st.flare = True
        elif st.flare and self.rng.random() < 0.05:
            st.flare = False
        if st.flare:
            st.inflammation = min(1.0, st.inflammation + 0.02)
        else:
            st.inflammation = max(0, st.inflammation - 0.01)
        st.mucosal_z = Z_MUCOSA * (1 + st.inflammation * 3)
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.mucosal_z, Z_MUCOSA)
        if st.subtype == "crohn":
            cdai = st.inflammation * 450
            score_str = f"CDAI={cdai:.0f}"
        else:
            mayo = min(12, int(st.inflammation * 12))
            score_str = f"Mayo={mayo}"
        return {"disease": f"IBD-{st.subtype}", "inflammation": st.inflammation,
                "flare": st.flare, "score": score_str, "gamma_sq": g2}


# ============================================================================
# 4. IBS
# ============================================================================
@dataclass
class IBSState:
    subtype: str = "mixed"  # diarrhea / constipation / mixed
    gut_brain_gamma: float = 0.0
    visceral_sensitivity: float = 0.5

class IBSModel:
    """IBS: gut-brain axis Γ elevation without structural Z change."""

    def __init__(self, subtype: str = "mixed", sensitivity: float = 0.6):
        self.state = IBSState(subtype=subtype, visceral_sensitivity=sensitivity)
        self.rng = random.Random(42)
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        # IBS: structural Z is normal, but neural perception Γ is elevated
        stress = self.rng.gauss(0.3, 0.15)
        st.gut_brain_gamma = st.visceral_sensitivity * max(0, stress)
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = st.gut_brain_gamma ** 2
        return {"disease": "IBS", "subtype": st.subtype,
                "gut_brain_gamma": st.gut_brain_gamma, "gamma_sq": g2}


# ============================================================================
# 5. LIVER CIRRHOSIS
# ============================================================================
@dataclass
class CirrhosisState:
    hepatic_z: float = Z_HEPATOCYTE
    portal_z: float = Z_PORTAL
    child_pugh: str = "A"
    meld_score: float = 6.0
    fibrosis_stage: int = 0  # F0–F4

class CirrhosisModel:
    """Cirrhosis: progressive hepatic Z increase → portal hypertension."""

    def __init__(self, initial_fibrosis: int = 2, progression: float = 0.001):
        self.state = CirrhosisState(fibrosis_stage=initial_fibrosis)
        self.progression = progression
        self._apply_fibrosis()
        self.tick_count = 0

    def _apply_fibrosis(self):
        st = self.state
        st.hepatic_z = Z_HEPATOCYTE * (1 + st.fibrosis_stage * 0.5)
        st.portal_z = Z_PORTAL * (1 + st.fibrosis_stage * 0.3)

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.hepatic_z = min(Z_HEPATOCYTE * 5, st.hepatic_z * (1 + self.progression))
        st.portal_z = min(Z_PORTAL * 4, st.portal_z * (1 + self.progression * 0.5))
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.hepatic_z, Z_HEPATOCYTE)
        # Child-Pugh from Γ²
        if g2 < 0.15: st.child_pugh = "A"
        elif g2 < 0.35: st.child_pugh = "B"
        else: st.child_pugh = "C"
        st.meld_score = 6 + g2 * 34  # MELD 6–40
        return {"disease": "Cirrhosis", "hepatic_z": st.hepatic_z,
                "child_pugh": st.child_pugh, "meld": st.meld_score, "gamma_sq": g2}


# ============================================================================
# 6. CHOLELITHIASIS
# ============================================================================
@dataclass
class CholelithiasisState:
    bile_duct_z: float = Z_BILE_DUCT
    stone_size_mm: float = 0.0
    obstructed: bool = False

class CholelithiasisModel:
    """Gallstones: bile duct Z obstruction."""

    def __init__(self, stone_size: float = 8.0):
        self.state = CholelithiasisState(stone_size_mm=stone_size)
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.obstructed = st.stone_size_mm > 6
        if st.obstructed:
            obstruction_frac = min(1.0, (st.stone_size_mm - 6) / 10)
            st.bile_duct_z = Z_BILE_DUCT + (Z_OBSTRUCTED - Z_BILE_DUCT) * obstruction_frac
        else:
            st.bile_duct_z = Z_BILE_DUCT
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.bile_duct_z, Z_BILE_DUCT)
        return {"disease": "Cholelithiasis", "stone_mm": st.stone_size_mm,
                "obstructed": st.obstructed, "gamma_sq": g2}


# ============================================================================
# 7. PANCREATITIS
# ============================================================================
@dataclass
class PancreatitisState:
    ductal_z: float = Z_PANCREATIC
    severity: str = "mild"
    lipase: float = 30.0   # Normal <60 U/L
    necrosis_pct: float = 0.0

class PancreatitisModel:
    """Pancreatitis: ductal Z obstruction → enzyme autodigestion."""

    def __init__(self, cause: str = "gallstone", initial_severity: float = 0.3):
        self.state = PancreatitisState()
        self.cause = cause
        self.severity_param = initial_severity
        self.tick_count = 0
        self.on_treatment = False

    def treat(self):
        self.on_treatment = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_treatment:
            st.ductal_z = max(Z_PANCREATIC, st.ductal_z * 0.99)
        else:
            st.ductal_z = min(Z_PANCREATIC * 10,
                              st.ductal_z * (1 + 0.005 * self.severity_param))
        g2 = gamma_sq(st.ductal_z, Z_PANCREATIC)
        st.lipase = 30 + g2 * 2000  # Lipase rises with obstruction
        st.necrosis_pct = min(1.0, max(0, g2 - 0.3) * 2)
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.ductal_z, Z_PANCREATIC)
        if g2 < 0.15: st.severity = "mild"
        elif g2 < 0.4: st.severity = "moderate"
        else: st.severity = "severe"
        ranson = min(11, int(g2 * 12))
        return {"disease": "Pancreatitis", "severity": st.severity,
                "lipase": st.lipase, "ranson": ranson, "gamma_sq": g2}


# ============================================================================
# 8. BOWEL OBSTRUCTION
# ============================================================================
@dataclass
class BowelObstructionState:
    obstruction_z: float = Z_MUCOSA
    proximal_distension: float = 0.0
    complete: bool = False

class BowelObstructionModel:
    """Bowel obstruction: transmission line Z discontinuity."""

    def __init__(self, level: str = "small", complete: bool = False):
        self.state = BowelObstructionState(complete=complete)
        self.level = level
        self.operated = False
        self.tick_count = 0

    def operate(self):
        self.operated = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.operated:
            st.obstruction_z = max(Z_MUCOSA, st.obstruction_z * 0.9)
        elif st.complete:
            st.obstruction_z = Z_OBSTRUCTED
        else:
            st.obstruction_z = Z_MUCOSA * 5  # Partial
        st.proximal_distension = min(1.0, gamma_sq(st.obstruction_z, Z_MUCOSA))
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.obstruction_z, Z_MUCOSA)
        return {"disease": "Bowel Obstruction", "level": self.level,
                "complete": st.complete, "distension": st.proximal_distension, "gamma_sq": g2}


# ============================================================================
# 9. COLORECTAL CANCER
# ============================================================================
@dataclass
class CRCState:
    tumor_z: float = Z_COLONIC
    tumor_size_cm: float = 1.0
    cea: float = 2.0   # Normal <5 ng/mL
    stage: str = "I"

class CRCModel:
    """Colorectal Cancer: mucosal Z transformation → growing Γ island."""

    def __init__(self, initial_size: float = 1.5, growth_rate: float = 0.003):
        self.state = CRCState(tumor_size_cm=initial_size)
        self.growth_rate = growth_rate
        self.on_treatment = False
        self.tick_count = 0

    def start_treatment(self):
        self.on_treatment = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        r = self.growth_rate * (0.3 if self.on_treatment else 1.0)
        st.tumor_size_cm = min(12, st.tumor_size_cm * (1 + r))
        st.tumor_z = Z_COLONIC * (1 + st.tumor_size_cm * 0.8)
        st.cea = 2.0 + st.tumor_size_cm * 3
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.tumor_z, Z_COLONIC)
        if st.tumor_size_cm <= 2: st.stage = "I"
        elif st.tumor_size_cm <= 4: st.stage = "II"
        elif st.tumor_size_cm <= 7: st.stage = "III"
        else: st.stage = "IV"
        return {"disease": "CRC", "tumor_cm": st.tumor_size_cm,
                "stage": st.stage, "cea": st.cea, "gamma_sq": g2}


# ============================================================================
# 10. HEPATITIS B/C
# ============================================================================
@dataclass
class HepatitisState:
    viral_type: str = "B"
    viral_load: float = 0.0
    hepatocyte_z: float = Z_HEPATOCYTE
    alt: float = 20.0   # Normal <40 IU/L
    fibrosis: int = 0    # F0–F4

class HepatitisModel:
    """Hepatitis: viral Z contamination → hepatocyte impedance drift."""

    def __init__(self, viral_type: str = "B", initial_load: float = 1e6):
        self.state = HepatitisState(viral_type=viral_type, viral_load=initial_load)
        self.on_antiviral = False
        self.tick_count = 0

    def start_antiviral(self):
        self.on_antiviral = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_antiviral:
            st.viral_load = max(0, st.viral_load * 0.95)  # 5% reduction/tick
        else:
            st.viral_load = min(1e8, st.viral_load * 1.001)
        # Viral load → hepatocyte Z drift
        load_norm = math.log1p(st.viral_load) / math.log1p(1e8)
        st.hepatocyte_z = Z_HEPATOCYTE * (1 + load_norm * 1.5)
        st.alt = 20 + load_norm * 200
        # Fibrosis progresses slowly
        if load_norm > 0.5 and self.tick_count % 200 == 0:
            st.fibrosis = min(4, st.fibrosis + 1)
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.hepatocyte_z, Z_HEPATOCYTE)
        return {"disease": f"Hepatitis {st.viral_type}", "viral_load": st.viral_load,
                "alt": st.alt, "fibrosis": f"F{st.fibrosis}", "gamma_sq": g2}


# ============================================================================
# UNIFIED ENGINE
# ============================================================================
class ClinicalGastroenterologyEngine:
    """Unified engine for all 10 GI/Hepatobiliary diseases."""

    DISEASE_CLASSES = {
        "gerd": GERDModel,
        "peptic_ulcer": PepticUlcerModel,
        "ibd": IBDModel,
        "ibs": IBSModel,
        "cirrhosis": CirrhosisModel,
        "cholelithiasis": CholelithiasisModel,
        "pancreatitis": PancreatitisModel,
        "bowel_obstruction": BowelObstructionModel,
        "crc": CRCModel,
        "hepatitis": HepatitisModel,
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
        results["gi_reserve"] = max(0.0, 1.0 - total_g2)
        return results
