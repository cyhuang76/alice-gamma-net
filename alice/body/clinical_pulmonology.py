# -*- coding: utf-8 -*-
"""clinical_pulmonology.py — Top-10 Pulmonary Diseases as Impedance-Mismatch Patterns
===================================================================================

│ Disease                 │ Γ Physics Mapping                           │ Clinical Scale  │
│─────────────────────────│─────────────────────────────────────────────│────────────────│
│ 1. Asthma               │ Bronchial Z oscillation (reactance)         │ FEV1/FVC %     │
│ 2. COPD                 │ Progressive airway Z increase               │ GOLD I–IV      │
│ 3. Pneumonia            │ Alveolar Z fill (fluid replaces air)        │ CURB-65 0–5    │
│ 4. Pulmonary Embolism   │ Vascular occlusion → Γ → 1.0               │ Wells Score    │
│ 5. Pneumothorax         │ Pleural Z discontinuity                     │ Size %         │
│ 6. Pulmonary Fibrosis   │ Progressive Z stiffening                    │ FVC % pred     │
│ 7. ARDS                 │ Surfactant loss → alveolar Z collapse       │ PaO2/FiO2      │
│ 8. Sleep Apnea (OSA)    │ Periodic airway Z occlusion                 │ AHI events/hr  │
│ 9. Lung Cancer          │ Focal Z infiltration                        │ TNM staging    │
│ 10. Cystic Fibrosis     │ Mucus Z elevation in airways                │ FEV1 % pred    │

Core: Γ = (Z_L − Z₀) / (Z_L + Z₀),  T = 1 − Γ²

The lung is an LC resonator (alice/body/lung.py):
    L = inertia of air column
    C = compliance of alveoli
    Every disease changes L, C, or both → shifts resonance → Γ rises.
"""

from __future__ import annotations
import math, random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ============================================================================
# Physical Constants — Pulmonary Impedance
# ============================================================================
Z_AIRWAY: float = 30.0         # Normal airway impedance
Z_ALVEOLAR: float = 20.0       # Normal alveolar impedance (air-filled)
Z_ALVEOLAR_FLUID: float = 200.0  # Fluid-filled alveolus
Z_PLEURAL: float = 5.0         # Normal pleural space (negative pressure, low Z)
Z_PLEURAL_AIR: float = 500.0   # Pneumothorax (air in pleural space)
Z_PULM_VASC: float = 15.0      # Normal pulmonary vasculature Z
Z_INTERSTITIAL: float = 25.0   # Normal interstitial tissue Z
Z_MUCUS: float = 150.0         # Mucus plug Z

FEV1_NORMAL: float = 1.0       # Normalised FEV1
FVC_NORMAL: float = 1.0        # Normalised FVC


from alice.body.clinical_common import gamma, gamma_sq, ClinicalEngineBase


# ============================================================================
# 1. ASTHMA
# ============================================================================
@dataclass
class AsthmaState:
    bronchial_z: float = Z_AIRWAY
    fev1_ratio: float = 0.80    # FEV1/FVC
    exacerbation: bool = False
    on_bronchodilator: bool = False

class AsthmaModel:
    """Asthma: bronchial reactance → Z oscillation → intermittent Γ spikes."""

    def __init__(self, baseline_z: float = Z_AIRWAY * 1.3, reactivity: float = 0.5):
        self.state = AsthmaState(bronchial_z=baseline_z)
        self.baseline_z = baseline_z
        self.reactivity = reactivity
        self.rng = random.Random(42)
        self.tick_count = 0

    def trigger_exacerbation(self):
        self.state.exacerbation = True

    def use_bronchodilator(self):
        self.state.on_bronchodilator = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if st.exacerbation:
            # Bronchospasm: Z spikes
            st.bronchial_z = self.baseline_z * (2.0 + self.reactivity * self.rng.random())
        elif st.on_bronchodilator:
            st.bronchial_z = max(Z_AIRWAY, st.bronchial_z * 0.95)
            if st.bronchial_z <= Z_AIRWAY * 1.05:
                st.on_bronchodilator = False
        else:
            # Baseline: mild Z fluctuation
            st.bronchial_z = self.baseline_z * (1.0 + 0.1 * self.rng.gauss(0, 1))
        g2 = gamma_sq(st.bronchial_z, Z_AIRWAY)
        st.fev1_ratio = max(0.2, 0.80 * (1 - g2))
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.bronchial_z, Z_AIRWAY)
        if st.fev1_ratio >= 0.70:
            severity = "Intermittent"
        elif st.fev1_ratio >= 0.60:
            severity = "Mild persistent"
        elif st.fev1_ratio >= 0.50:
            severity = "Moderate persistent"
        else:
            severity = "Severe persistent"
        return {"disease": "Asthma", "fev1_ratio": st.fev1_ratio,
                "bronchial_z": st.bronchial_z, "severity": severity, "gamma_sq": g2}


# ============================================================================
# 2. COPD
# ============================================================================
@dataclass
class COPDState:
    airway_z: float = Z_AIRWAY
    fev1_pct: float = 100.0  # % predicted
    gold_stage: int = 0

class COPDModel:
    """COPD: chronic irreversible airway Z increase → progressive Γ."""

    def __init__(self, initial_fev1: float = 65.0, decline_rate: float = 0.02):
        z_ratio = 100.0 / max(initial_fev1, 10)
        self.state = COPDState(airway_z=Z_AIRWAY * z_ratio, fev1_pct=initial_fev1)
        self.decline = decline_rate
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.fev1_pct = max(10, st.fev1_pct - self.decline)
        st.airway_z = Z_AIRWAY * (100.0 / max(st.fev1_pct, 10))
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.airway_z, Z_AIRWAY)
        if st.fev1_pct >= 80: st.gold_stage = 1
        elif st.fev1_pct >= 50: st.gold_stage = 2
        elif st.fev1_pct >= 30: st.gold_stage = 3
        else: st.gold_stage = 4
        return {"disease": "COPD", "fev1_pct": st.fev1_pct,
                "gold_stage": st.gold_stage, "gamma_sq": g2}


# ============================================================================
# 3. PNEUMONIA
# ============================================================================
@dataclass
class PneumoniaState:
    consolidation_pct: float = 0.0  # Fraction of lung consolidated
    alveolar_z: float = Z_ALVEOLAR
    curb65: int = 0

class PneumoniaModel:
    """Pneumonia: alveolar fluid fill → Z mismatch (air→fluid)."""

    def __init__(self, consolidation: float = 0.3, virulence: float = 0.5):
        self.state = PneumoniaState(consolidation_pct=consolidation)
        self.virulence = virulence
        self.on_antibiotics = False
        self.tick_count = 0

    def treat(self):
        self.on_antibiotics = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_antibiotics:
            st.consolidation_pct = max(0, st.consolidation_pct - 0.005)
        else:
            st.consolidation_pct = min(0.9, st.consolidation_pct + 0.003 * self.virulence)
        st.alveolar_z = Z_ALVEOLAR + (Z_ALVEOLAR_FLUID - Z_ALVEOLAR) * st.consolidation_pct
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.alveolar_z, Z_ALVEOLAR)
        st.curb65 = min(5, int(st.consolidation_pct * 6))
        return {"disease": "Pneumonia", "consolidation": st.consolidation_pct,
                "curb65": st.curb65, "gamma_sq": g2}


# ============================================================================
# 4. PULMONARY EMBOLISM (PE)
# ============================================================================
@dataclass
class PEState:
    clot_burden: float = 0.0  # 0–1
    pulm_vasc_z: float = Z_PULM_VASC
    hemodynamic_compromise: bool = False

class PEModel:
    """PE: acute vascular occlusion → Γ jump in pulmonary circuit."""

    def __init__(self, clot_burden: float = 0.5):
        self.state = PEState(clot_burden=clot_burden)
        self.on_anticoag = False
        self.tick_count = 0

    def anticoagulate(self):
        self.on_anticoag = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_anticoag:
            st.clot_burden = max(0, st.clot_burden - 0.002)
        st.pulm_vasc_z = Z_PULM_VASC / max(1 - st.clot_burden, 0.01)
        st.hemodynamic_compromise = st.clot_burden > 0.5
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.pulm_vasc_z, Z_PULM_VASC)
        if st.clot_burden < 0.2: wells = "Low"
        elif st.clot_burden < 0.5: wells = "Moderate"
        else: wells = "High"
        return {"disease": "PE", "clot_burden": st.clot_burden,
                "massive": st.hemodynamic_compromise, "wells": wells, "gamma_sq": g2}


# ============================================================================
# 5. PNEUMOTHORAX
# ============================================================================
@dataclass
class PneumothoraxState:
    size_pct: float = 0.0      # % of hemithorax
    pleural_z: float = Z_PLEURAL
    tension: bool = False

class PneumothoraxModel:
    """Pneumothorax: air in pleural space → Z discontinuity."""

    def __init__(self, size: float = 30.0, tension: bool = False):
        self.state = PneumothoraxState(size_pct=size, tension=tension)
        self.chest_tube = False
        self.tick_count = 0

    def insert_chest_tube(self):
        self.chest_tube = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.chest_tube:
            st.size_pct = max(0, st.size_pct - 2.0)
        elif st.tension:
            st.size_pct = min(100, st.size_pct + 1.5)
        st.pleural_z = Z_PLEURAL + (Z_PLEURAL_AIR - Z_PLEURAL) * (st.size_pct / 100)
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.pleural_z, Z_PLEURAL)
        return {"disease": "Pneumothorax", "size_pct": st.size_pct,
                "tension": st.tension, "gamma_sq": g2}


# ============================================================================
# 6. PULMONARY FIBROSIS (IPF)
# ============================================================================
@dataclass
class FibrosisState:
    interstitial_z: float = Z_INTERSTITIAL
    fvc_pct: float = 100.0

class FibrosisModel:
    """IPF: progressive interstitial Z stiffening."""

    def __init__(self, initial_fvc: float = 70.0, rate: float = 0.015):
        z_ratio = 100.0 / max(initial_fvc, 10)
        self.state = FibrosisState(interstitial_z=Z_INTERSTITIAL * z_ratio, fvc_pct=initial_fvc)
        self.rate = rate
        self.on_antifibrotic = False
        self.tick_count = 0

    def start_antifibrotic(self):
        self.on_antifibrotic = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        r = self.rate * (0.5 if self.on_antifibrotic else 1.0)
        st.fvc_pct = max(15, st.fvc_pct - r)
        st.interstitial_z = Z_INTERSTITIAL * (100 / max(st.fvc_pct, 10))
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.interstitial_z, Z_INTERSTITIAL)
        if st.fvc_pct >= 80: severity = "Mild"
        elif st.fvc_pct >= 50: severity = "Moderate"
        else: severity = "Severe"
        return {"disease": "IPF", "fvc_pct": st.fvc_pct,
                "severity": severity, "gamma_sq": g2}


# ============================================================================
# 7. ARDS
# ============================================================================
@dataclass
class ARDSState:
    surfactant_level: float = 1.0  # Normalised (1 = healthy)
    pao2_fio2: float = 400.0      # Normal >300
    alveolar_z: float = Z_ALVEOLAR

class ARDSModel:
    """ARDS: surfactant loss → alveolar Z collapse → bilateral infiltrates."""

    def __init__(self, initial_surfactant: float = 0.5):
        self.state = ARDSState(surfactant_level=initial_surfactant)
        self.on_ventilator = False
        self.tick_count = 0

    def intubate(self):
        self.on_ventilator = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_ventilator:
            st.surfactant_level = min(1.0, st.surfactant_level + 0.002)
        else:
            st.surfactant_level = max(0.01, st.surfactant_level - 0.005)
        # Without surfactant, alveolar surface tension Z increases dramatically
        st.alveolar_z = Z_ALVEOLAR / max(st.surfactant_level, 0.01)
        st.pao2_fio2 = 400 * st.surfactant_level
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.alveolar_z, Z_ALVEOLAR)
        if st.pao2_fio2 > 300: berlin = "None"
        elif st.pao2_fio2 > 200: berlin = "Mild"
        elif st.pao2_fio2 > 100: berlin = "Moderate"
        else: berlin = "Severe"
        return {"disease": "ARDS", "pao2_fio2": st.pao2_fio2,
                "berlin": berlin, "surfactant": st.surfactant_level, "gamma_sq": g2}


# ============================================================================
# 8. OBSTRUCTIVE SLEEP APNEA (OSA)
# ============================================================================
@dataclass
class OSAState:
    pharyngeal_z: float = Z_AIRWAY
    ahi: float = 0.0            # Apnea-Hypopnea Index
    desaturation_events: int = 0

class OSAModel:
    """OSA: periodic pharyngeal Z occlusion during sleep."""

    def __init__(self, collapsibility: float = 0.6):
        self.state = OSAState()
        self.collapsibility = collapsibility  # 0–1
        self.rng = random.Random(42)
        self.tick_count = 0
        self.on_cpap = False

    def start_cpap(self):
        self.on_cpap = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        # Each tick = 1 minute of sleep
        if self.on_cpap:
            st.pharyngeal_z = Z_AIRWAY * 1.05  # CPAP splints airway
        else:
            # Probabilistic collapse
            if self.rng.random() < self.collapsibility:
                st.pharyngeal_z = Z_AIRWAY * 10  # Collapsed
                st.desaturation_events += 1
            else:
                st.pharyngeal_z = Z_AIRWAY * 1.2  # Narrowed but open
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.pharyngeal_z, Z_AIRWAY)
        # AHI = events per hour (1 tick = 1 min)
        hours = max(self.tick_count / 60, 0.01)
        st.ahi = st.desaturation_events / hours
        if st.ahi < 5: severity = "Normal"
        elif st.ahi < 15: severity = "Mild"
        elif st.ahi < 30: severity = "Moderate"
        else: severity = "Severe"
        return {"disease": "OSA", "ahi": st.ahi, "pharyngeal_z": st.pharyngeal_z,
                "severity": severity, "gamma_sq": g2}


# ============================================================================
# 9. LUNG CANCER  (canonical model in clinical_oncology)
# ============================================================================
from alice.body.clinical_oncology import LungCancerModel  # noqa: E402


# ============================================================================
# 10. CYSTIC FIBROSIS (CF)
# ============================================================================
@dataclass
class CFState:
    mucus_z: float = Z_AIRWAY * 1.5
    fev1_pct: float = 80.0
    exacerbation: bool = False

class CFModel:
    """CF: CFTR mutation → thick mucus → chronic airway Z elevation."""

    def __init__(self, severity: float = 0.5):
        self.state = CFState()
        self.severity = severity  # 0–1
        self.on_modulator = False
        self.tick_count = 0

    def start_modulator(self):
        """CFTR modulator therapy (e.g. Trikafta)."""
        self.on_modulator = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_modulator:
            # Modulator reduces mucus Z
            st.mucus_z = max(Z_AIRWAY * 1.1, st.mucus_z * 0.998)
        else:
            st.mucus_z = min(Z_MUCUS, st.mucus_z * (1 + 0.001 * self.severity))
        g2 = gamma_sq(st.mucus_z, Z_AIRWAY)
        st.fev1_pct = max(15, 100 * (1 - g2))
        return self.get_clinical_score()

    def get_clinical_score(self) -> Dict:
        st = self.state
        g2 = gamma_sq(st.mucus_z, Z_AIRWAY)
        if st.fev1_pct >= 70: severity = "Mild"
        elif st.fev1_pct >= 40: severity = "Moderate"
        else: severity = "Severe"
        return {"disease": "CF", "fev1_pct": st.fev1_pct, "mucus_z": st.mucus_z,
                "severity": severity, "gamma_sq": g2}


# ============================================================================
# UNIFIED ENGINE
# ============================================================================
class ClinicalPulmonologyEngine(ClinicalEngineBase):
    """Unified engine managing all 10 pulmonary disease models."""

    DISEASE_CLASSES = {
        "asthma": AsthmaModel,
        "copd": COPDModel,
        "pneumonia": PneumoniaModel,
        "pe": PEModel,
        "pneumothorax": PneumothoraxModel,
        "ipf": FibrosisModel,
        "ards": ARDSModel,
        "osa": OSAModel,
        "lung_cancer": LungCancerModel,
        "cf": CFModel,
    }
    RESERVE_KEY = "pulmonary_reserve"
