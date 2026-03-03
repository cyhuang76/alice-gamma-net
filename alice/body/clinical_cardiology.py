# -*- coding: utf-8 -*-
"""clinical_cardiology.py — Top-10 Cardiac Diseases as Impedance-Mismatch Patterns
================================================================================

All cardiac pathology = Γ mismatch at specific vascular/valvular/myocardial interfaces.

│ Disease                │ Γ Physics Mapping                              │ Clinical Scale │
│────────────────────────│────────────────────────────────────────────────│───────────────│
│ 1. Myocardial Infarct  │ Coronary occlusion → regional Γ → 1.0         │ KILLIP I–IV   │
│ 2. Heart Failure       │ Pump Z mismatch → CO × SVR → reduced output   │ NYHA I–IV     │
│ 3. Atrial Fibrillation │ SA-node Z oscillation → irregular RR           │ CHA₂DS₂-VASc │
│ 4. Hypertension        │ Arteriolar Z chronically ↑                     │ JNC Stage     │
│ 5. Aortic Stenosis     │ Valve Z obstruction → LV overload              │ Gradient mmHg │
│ 6. Cardiomyopathy      │ Myocardial Z structural change                 │ LVEF %        │
│ 7. Pericarditis        │ Pericardial Z change → constrictive             │ Pain VAS 0–10 │
│ 8. Pulmonary HTN       │ Pulmonary vascular Z ↑                         │ WHO Class     │
│ 9. Endocarditis        │ Biofilm Z contamination at valve                │ Duke Criteria │
│ 10. Aortic Dissection  │ Wall Z discontinuity → transmission line tear   │ Stanford A/B  │

Core: Γ = (Z_L − Z₀) / (Z_L + Z₀),  T = 1 − Γ²,  Severity ∝ Σ Γ²

References:
    Killip T, Kimball JT. (1967) Am J Cardiol 20:457.
    Yancy CW et al. (2013) JACC 62:e147 (ACCF/AHA HF Guidelines).
    Lip GYH et al. (2010) Chest 137:263 (CHA₂DS₂-VASc).
"""

from __future__ import annotations
import math, random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ============================================================================
# Physical Constants — Cardiovascular Impedance
# ============================================================================
Z_AORTA: float = 50.0          # Normal aortic characteristic impedance
Z_CORONARY: float = 55.0       # Normal coronary artery Z
Z_ARTERIOLAR: float = 60.0     # Normal arteriolar Z
Z_VENOUS: float = 30.0         # Normal venous Z
Z_MYOCARDIUM: float = 65.0     # Normal myocardial tissue Z
Z_VALVE_OPEN: float = 5.0      # Normal open valve Z (very low)
Z_PERICARDIUM: float = 40.0    # Normal pericardial Z
Z_PULM_ART: float = 20.0       # Normal pulmonary artery Z (low pressure)
Z_OCCLUDED: float = 1e6        # Occluded vessel (open circuit)

ALL_CARDIAC_CHANNELS = [
    "coronary_lad", "coronary_lcx", "coronary_rca",
    "lv_myocardium", "rv_myocardium",
    "aortic_valve", "mitral_valve", "tricuspid_valve", "pulmonic_valve",
    "aorta", "pulmonary_artery", "systemic_arterioles",
    "sa_node", "av_node", "pericardium",
]


def gamma(z_load: float, z_source: float) -> float:
    return (z_load - z_source) / (z_load + z_source)


def gamma_sq(z_load: float, z_source: float) -> float:
    g = gamma(z_load, z_source)
    return g * g


# ============================================================================
# 1. MYOCARDIAL INFARCTION (MI)
# ============================================================================
@dataclass
class MIState:
    territory: str = "LAD"       # LAD / LCx / RCA
    occlusion_pct: float = 0.0   # 0–1
    onset_tick: int = 0
    reperfused: bool = False
    reperfusion_tick: int = 0
    necrosis_pct: float = 0.0    # Irreversible damage fraction
    troponin: float = 0.0

TERRITORY_CHANNELS = {
    "LAD": ["coronary_lad", "lv_myocardium"],
    "LCx": ["coronary_lcx", "lv_myocardium"],
    "RCA": ["coronary_rca", "rv_myocardium"],
}

class MIModel:
    """Myocardial Infarction: coronary occlusion → regional Γ jump."""

    def __init__(self, territory: str = "LAD", occlusion: float = 0.95):
        self.state = MIState(territory=territory, occlusion_pct=occlusion)
        self.channels: Dict[str, float] = {ch: Z_CORONARY for ch in ALL_CARDIAC_CHANNELS}
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state

        # Ischemia progresses if not reperfused
        if not st.reperfused:
            isch_ticks = self.tick_count - st.onset_tick
            # Necrosis grows logarithmically (golden window ~6h)
            st.necrosis_pct = min(1.0, st.occlusion_pct * math.log1p(isch_ticks / 50) / 3)
        else:
            # Reperfusion saves penumbra but necrosis is fixed
            pass

        # Compute effective Z in territory
        for ch in TERRITORY_CHANNELS.get(st.territory, []):
            z_eff = Z_CORONARY + (Z_OCCLUDED - Z_CORONARY) * st.occlusion_pct
            self.channels[ch] = z_eff

        # Troponin release proportional to necrosis
        st.troponin = st.necrosis_pct * 10.0  # ng/mL

        return self.get_clinical_score()

    def reperfuse(self, tick: int):
        self.state.reperfused = True
        self.state.reperfusion_tick = tick
        self.state.occlusion_pct *= 0.1  # PCI opens 90%

    def get_clinical_score(self) -> Dict:
        affected = TERRITORY_CHANNELS.get(self.state.territory, [])
        total_gamma_sq = sum(gamma_sq(self.channels[ch], Z_CORONARY) for ch in affected)
        # KILLIP class: I (no HF), II (rales), III (pulm edema), IV (shock)
        if total_gamma_sq < 0.1:
            killip = 1
        elif total_gamma_sq < 0.4:
            killip = 2
        elif total_gamma_sq < 0.7:
            killip = 3
        else:
            killip = 4
        return {
            "disease": "MI",
            "territory": self.state.territory,
            "killip_class": killip,
            "total_gamma_sq": total_gamma_sq,
            "necrosis_pct": self.state.necrosis_pct,
            "troponin": self.state.troponin,
        }


# ============================================================================
# 2. HEART FAILURE (CHF)
# ============================================================================
@dataclass
class CHFState:
    ejection_fraction: float = 0.60  # Normal EF
    nyha_class: int = 1
    onset_tick: int = 0
    progression_rate: float = 0.0005
    on_treatment: bool = False

class CHFModel:
    """Heart Failure: pump Z mismatch → reduced cardiac output."""

    def __init__(self, initial_ef: float = 0.55, progression: float = 0.0005):
        self.state = CHFState(ejection_fraction=initial_ef, progression_rate=progression)
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        # EF declines over time (slower with treatment)
        rate = st.progression_rate * (0.5 if st.on_treatment else 1.0)
        st.ejection_fraction = max(0.05, st.ejection_fraction - rate)
        return self.get_clinical_score()

    def start_treatment(self):
        self.state.on_treatment = True

    def get_clinical_score(self) -> Dict:
        ef = self.state.ejection_fraction
        # Z mismatch: lower EF → higher Γ between LV and aorta
        z_lv_eff = Z_MYOCARDIUM / max(ef, 0.05)  # Impaired pump → higher Z
        g2 = gamma_sq(z_lv_eff, Z_AORTA)
        # NYHA classification
        if ef >= 0.50:
            nyha = 1
        elif ef >= 0.40:
            nyha = 2
        elif ef >= 0.25:
            nyha = 3
        else:
            nyha = 4
        self.state.nyha_class = nyha
        return {
            "disease": "CHF",
            "ef": ef,
            "nyha_class": nyha,
            "gamma_sq": g2,
            "z_lv_effective": z_lv_eff,
        }


# ============================================================================
# 3. ATRIAL FIBRILLATION (AF)
# ============================================================================
@dataclass
class AFState:
    base_rate: float = 75.0      # Normal sinus rate
    af_active: bool = False
    ventricular_rate: float = 75.0
    stroke_risk_score: int = 0   # CHA₂DS₂-VASc

class AFModel:
    """AF: SA-node Z oscillation → irregular rhythm → stroke risk."""

    def __init__(self, risk_factors: int = 2):
        self.state = AFState(stroke_risk_score=risk_factors)
        self.rng = random.Random(42)
        self.tick_count = 0

    def trigger_af(self):
        self.state.af_active = True

    def cardiovert(self):
        self.state.af_active = False
        self.state.ventricular_rate = self.state.base_rate

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if st.af_active:
            # Irregular ventricular rate: chaotic Z oscillation
            st.ventricular_rate = 80 + 100 * self.rng.random()  # 80–180 bpm
        else:
            st.ventricular_rate = st.base_rate
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        # Γ from rhythm irregularity: deviation from optimal 75 bpm
        z_rhythm = Z_AORTA * (st.ventricular_rate / 75.0)
        g2 = gamma_sq(z_rhythm, Z_AORTA)
        return {
            "disease": "AF",
            "af_active": st.af_active,
            "ventricular_rate": st.ventricular_rate,
            "gamma_sq": g2,
            "cha2ds2_vasc": st.stroke_risk_score,
        }


# ============================================================================
# 4. HYPERTENSION
# ============================================================================
@dataclass
class HTNState:
    systolic: float = 120.0
    diastolic: float = 80.0
    arteriolar_z: float = Z_ARTERIOLAR
    on_medication: bool = False

class HTNModel:
    """Hypertension: chronic arteriolar Z elevation."""

    def __init__(self, initial_sys: float = 150.0, initial_dia: float = 95.0):
        z_ratio = initial_sys / 120.0
        self.state = HTNState(
            systolic=initial_sys, diastolic=initial_dia,
            arteriolar_z=Z_ARTERIOLAR * z_ratio,
        )
        self.tick_count = 0

    def treat(self):
        self.state.on_medication = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if st.on_medication:
            # Medication reduces arteriolar Z by 0.5% per tick
            st.arteriolar_z = max(Z_ARTERIOLAR, st.arteriolar_z * 0.995)
        else:
            # Untreated: slow Z drift upward (vascular remodeling)
            st.arteriolar_z = min(Z_ARTERIOLAR * 3, st.arteriolar_z * 1.0003)
        # BP from Z
        st.systolic = 120.0 * (st.arteriolar_z / Z_ARTERIOLAR)
        st.diastolic = 80.0 * (st.arteriolar_z / Z_ARTERIOLAR)
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.arteriolar_z, Z_ARTERIOLAR)
        # JNC staging
        if st.systolic < 120:
            stage = "Normal"
        elif st.systolic < 130:
            stage = "Elevated"
        elif st.systolic < 140:
            stage = "Stage 1"
        elif st.systolic < 180:
            stage = "Stage 2"
        else:
            stage = "Crisis"
        return {
            "disease": "Hypertension",
            "systolic": st.systolic,
            "diastolic": st.diastolic,
            "jnc_stage": stage,
            "gamma_sq": g2,
        }


# ============================================================================
# 5. AORTIC STENOSIS
# ============================================================================
@dataclass
class AorticStenosisState:
    valve_z: float = Z_VALVE_OPEN
    gradient_mmhg: float = 0.0
    progression_rate: float = 0.002

class AorticStenosisModel:
    """Aortic Stenosis: progressive valve Z obstruction → LV overload."""

    def __init__(self, initial_valve_z: float = 20.0, rate: float = 0.002):
        self.state = AorticStenosisState(valve_z=initial_valve_z, progression_rate=rate)
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        # Valve Z increases over time (calcification)
        st.valve_z = min(500.0, st.valve_z * (1 + st.progression_rate))
        # Pressure gradient proportional to Γ²
        g2 = gamma_sq(st.valve_z, Z_VALVE_OPEN)
        st.gradient_mmhg = g2 * 100  # Scale to clinical range
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.valve_z, Z_VALVE_OPEN)
        if st.gradient_mmhg < 20:
            severity = "Mild"
        elif st.gradient_mmhg < 40:
            severity = "Moderate"
        else:
            severity = "Severe"
        return {
            "disease": "Aortic Stenosis",
            "valve_z": st.valve_z,
            "gradient_mmhg": st.gradient_mmhg,
            "severity": severity,
            "gamma_sq": g2,
        }


# ============================================================================
# 6. CARDIOMYOPATHY
# ============================================================================
@dataclass
class CardiomyopathyState:
    subtype: str = "dilated"  # dilated / hypertrophic / restrictive
    myocardial_z: float = Z_MYOCARDIUM
    lvef: float = 0.55

class CardiomyopathyModel:
    """Cardiomyopathy: structural myocardial Z change."""

    SUBTYPE_Z = {
        "dilated": Z_MYOCARDIUM * 0.6,     # Thinned, compliant
        "hypertrophic": Z_MYOCARDIUM * 2.0, # Thickened, stiff
        "restrictive": Z_MYOCARDIUM * 2.5,  # Fibrotic, very stiff
    }

    def __init__(self, subtype: str = "dilated"):
        target_z = self.SUBTYPE_Z.get(subtype, Z_MYOCARDIUM)
        self.state = CardiomyopathyState(subtype=subtype, myocardial_z=Z_MYOCARDIUM)
        self.target_z = target_z
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        # Z drifts toward disease target
        st.myocardial_z += (self.target_z - st.myocardial_z) * 0.005
        # LVEF inversely related to Γ²
        g2 = gamma_sq(st.myocardial_z, Z_MYOCARDIUM)
        st.lvef = max(0.05, 0.60 * (1 - g2))
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.myocardial_z, Z_MYOCARDIUM)
        return {
            "disease": "Cardiomyopathy",
            "subtype": st.subtype,
            "myocardial_z": st.myocardial_z,
            "lvef": st.lvef,
            "gamma_sq": g2,
        }


# ============================================================================
# 7. PERICARDITIS
# ============================================================================
@dataclass
class PericarditisState:
    pericardial_z: float = Z_PERICARDIUM
    effusion_volume: float = 0.0  # mL (normal <50)
    pain_vas: float = 0.0         # VAS 0–10
    constrictive: bool = False

class PericarditisModel:
    """Pericarditis: pericardial Z change → constrictive physiology."""

    def __init__(self, effusion: float = 0.0, constrictive: bool = False):
        self.state = PericarditisState(
            effusion_volume=effusion, constrictive=constrictive
        )
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        # Effusion increases pericardial Z (fluid between heart and pericardium)
        z_fluid = 10.0 + st.effusion_volume * 0.5
        st.pericardial_z = Z_PERICARDIUM + z_fluid
        if st.constrictive:
            st.pericardial_z *= 2.0  # Fibrotic thickening
        g2 = gamma_sq(st.pericardial_z, Z_PERICARDIUM)
        st.pain_vas = min(10.0, g2 * 12)
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.pericardial_z, Z_PERICARDIUM)
        tamponade_risk = st.effusion_volume > 200
        return {
            "disease": "Pericarditis",
            "pericardial_z": st.pericardial_z,
            "effusion_ml": st.effusion_volume,
            "pain_vas": st.pain_vas,
            "tamponade_risk": tamponade_risk,
            "gamma_sq": g2,
        }


# ============================================================================
# 8. PULMONARY HYPERTENSION
# ============================================================================
@dataclass
class PulmHTNState:
    pulm_art_z: float = Z_PULM_ART
    mean_pap: float = 15.0  # Normal mPAP < 20 mmHg
    who_class: int = 1

class PulmHTNModel:
    """Pulmonary Hypertension: pulmonary vascular Z elevation."""

    def __init__(self, initial_pap: float = 30.0, progression: float = 0.001):
        z_ratio = initial_pap / 15.0
        self.state = PulmHTNState(pulm_art_z=Z_PULM_ART * z_ratio, mean_pap=initial_pap)
        self.progression = progression
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.pulm_art_z = min(Z_PULM_ART * 8, st.pulm_art_z * (1 + self.progression))
        st.mean_pap = 15.0 * (st.pulm_art_z / Z_PULM_ART)
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.pulm_art_z, Z_PULM_ART)
        if st.mean_pap < 25:
            st.who_class = 1
        elif st.mean_pap < 35:
            st.who_class = 2
        elif st.mean_pap < 50:
            st.who_class = 3
        else:
            st.who_class = 4
        return {
            "disease": "Pulmonary HTN",
            "pulm_art_z": st.pulm_art_z,
            "mean_pap": st.mean_pap,
            "who_class": st.who_class,
            "gamma_sq": g2,
        }


# ============================================================================
# 9. ENDOCARDITIS
# ============================================================================
@dataclass
class EndocarditisState:
    affected_valve: str = "mitral"  # mitral / aortic / tricuspid
    vegetation_z: float = 0.0       # Biofilm impedance
    duke_major: int = 0
    duke_minor: int = 0

VALVE_Z_NORMAL = {
    "mitral": Z_VALVE_OPEN,
    "aortic": Z_VALVE_OPEN,
    "tricuspid": Z_VALVE_OPEN + 2,
    "pulmonic": Z_VALVE_OPEN + 3,
}

class EndocarditisModel:
    """Endocarditis: biofilm Z contamination at valve surface."""

    def __init__(self, valve: str = "mitral", virulence: float = 0.5):
        self.state = EndocarditisState(affected_valve=valve)
        self.virulence = virulence
        self.tick_count = 0
        self.on_antibiotics = False

    def start_antibiotics(self):
        self.on_antibiotics = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_antibiotics:
            st.vegetation_z = max(0, st.vegetation_z - 0.5)
        else:
            st.vegetation_z = min(200, st.vegetation_z + self.virulence)
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        base_z = VALVE_Z_NORMAL.get(st.affected_valve, Z_VALVE_OPEN)
        effective_valve_z = base_z + st.vegetation_z
        g2 = gamma_sq(effective_valve_z, base_z)
        # Duke criteria approximation
        st.duke_major = 1 if st.vegetation_z > 10 else 0
        st.duke_minor = min(3, int(st.vegetation_z / 30))
        definite = st.duke_major >= 2 or (st.duke_major >= 1 and st.duke_minor >= 3)
        return {
            "disease": "Endocarditis",
            "valve": st.affected_valve,
            "vegetation_z": st.vegetation_z,
            "gamma_sq": g2,
            "duke_definite": definite,
        }


# ============================================================================
# 10. AORTIC DISSECTION
# ============================================================================
@dataclass
class AorticDissectionState:
    stanford_type: str = "B"   # A (ascending) / B (descending)
    intimal_tear_z: float = Z_AORTA
    propagation: float = 0.0   # 0–1 extent of dissection
    malperfusion: List[str] = field(default_factory=list)

class AorticDissectionModel:
    """Aortic Dissection: wall Z discontinuity → transmission line tear."""

    def __init__(self, stanford: str = "B", initial_tear: float = 0.3):
        self.state = AorticDissectionState(
            stanford_type=stanford, propagation=initial_tear,
        )
        self.tick_count = 0
        self.operated = False

    def emergency_surgery(self):
        self.operated = True
        self.state.propagation *= 0.2

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if not self.operated:
            # Dissection propagates (faster in Type A)
            rate = 0.008 if st.stanford_type == "A" else 0.003
            st.propagation = min(1.0, st.propagation + rate)
        # Wall Z at tear site
        st.intimal_tear_z = Z_AORTA * (1 + st.propagation * 10)
        # Malperfusion of branch vessels
        st.malperfusion = []
        if st.propagation > 0.4:
            st.malperfusion.append("renal")
        if st.propagation > 0.6:
            st.malperfusion.append("mesenteric")
        if st.propagation > 0.8:
            st.malperfusion.append("limb")
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.intimal_tear_z, Z_AORTA)
        mortality_risk = st.propagation * (1.5 if st.stanford_type == "A" else 0.8)
        return {
            "disease": "Aortic Dissection",
            "stanford": st.stanford_type,
            "propagation": st.propagation,
            "malperfusion": st.malperfusion,
            "gamma_sq": g2,
            "mortality_risk": min(1.0, mortality_risk),
        }


# ============================================================================
# UNIFIED ENGINE
# ============================================================================
class ClinicalCardiologyEngine:
    """Unified engine managing all 10 cardiac disease models."""

    DISEASE_CLASSES = {
        "mi": MIModel,
        "chf": CHFModel,
        "af": AFModel,
        "hypertension": HTNModel,
        "aortic_stenosis": AorticStenosisModel,
        "cardiomyopathy": CardiomyopathyModel,
        "pericarditis": PericarditisModel,
        "pulmonary_htn": PulmHTNModel,
        "endocarditis": EndocarditisModel,
        "aortic_dissection": AorticDissectionModel,
    }

    def __init__(self):
        self.active_diseases: Dict[str, object] = {}
        self.tick_count = 0
        self.history: List[Dict] = []

    def add_disease(self, name: str, **kwargs):
        cls = self.DISEASE_CLASSES.get(name)
        if cls:
            self.active_diseases[name] = cls(**kwargs)

    def tick(self) -> Dict:
        self.tick_count += 1
        results = {}
        total_gamma_sq = 0.0
        for name, model in self.active_diseases.items():
            r = model.tick()
            results[name] = r
            total_gamma_sq += r.get("gamma_sq", 0.0)
        results["total_gamma_sq"] = total_gamma_sq
        results["cardiac_reserve"] = max(0.0, 1.0 - total_gamma_sq)
        self.history.append(results)
        return results
