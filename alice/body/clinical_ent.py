# -*- coding: utf-8 -*-
"""clinical_ent.py — Top-10 ENT Diseases as Impedance-Mismatch Patterns
=======================================================================

│ Disease                  │ Γ Physics Mapping                          │ Clinical Scale  │
│──────────────────────────│────────────────────────────────────────────│────────────────│
│ 1. SNHL                  │ Cochlear hair-cell Z degradation          │ PTA dB HL      │
│ 2. Conductive HL         │ Ossicular chain Z mismatch                │ Air-Bone Gap   │
│ 3. Ménière Disease       │ Endolymph Z oscillation                   │ AAO-HNS stage  │
│ 4. Tinnitus              │ Phantom Z signal (spontaneous Γ)          │ THI score      │
│ 5. Otitis Media          │ Middle ear Z fluid fill                   │ Tympanometry   │
│ 6. Vocal Cord Paralysis  │ Laryngeal Z open-circuit                  │ VHI / GRBAS    │
│ 7. Sinusitis             │ Sinus Z obstruction / mucosal edema       │ Lund-Mackay    │
│ 8. Anosmia               │ Olfactory Z disconnection                 │ UPSIT score    │
│ 9. SSHL                  │ Acute cochlear Z failure                  │ PTA recovery   │
│ 10. BPPV                 │ Otolith Z displacement                    │ Dix-Hallpike   │

The ear is a 3-stage impedance transformer:
Z_air (low) → tympanic membrane → ossicles → Z_cochlea (high).
Disease = impedance mismatch at any stage → reflected acoustic energy.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict

# Physical Constants (Ω)
Z_AIR: float = 15.0          # Air acoustic impedance (low)
Z_TYMPANUM: float = 30.0     # Tympanic membrane
Z_OSSICULAR: float = 45.0    # Ossicular chain
Z_COCHLEA: float = 55.0      # Cochlear fluid
Z_VESTIBULAR: float = 50.0   # Vestibular organ
Z_NASAL: float = 35.0        # Nasal passage
Z_LARYNX: float = 40.0       # Vocal fold
Z_SINUS: float = 30.0        # Paranasal sinus

def gamma_sq(z_l: float, z_0: float) -> float:
    g = (z_l - z_0) / (z_l + z_0)
    return g * g

# ============================================================================
# 1. SNHL (Sensorineural Hearing Loss)
# ============================================================================
@dataclass
class SNHLState:
    cochlear_z: float = Z_COCHLEA
    pta_db: float = 10.0     # Pure tone average (dB HL)
    hair_cell_fraction: float = 1.0
    speech_discrimination: float = 100.0  # % correct

class SNHLModel:
    def __init__(self, severity_db: float = 50.0, age_progression: float = 0.002):
        self.state = SNHLState()
        self.target_db = severity_db
        self.progression = age_progression
        self.with_hearing_aid = False
        self.tick_count = 0

    def apply_hearing_aid(self):
        self.with_hearing_aid = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.hair_cell_fraction = max(0.05, 1.0 - self.target_db / 120)
        st.cochlear_z = Z_COCHLEA / max(st.hair_cell_fraction, 0.05)
        g2 = gamma_sq(st.cochlear_z, Z_COCHLEA)
        st.pta_db = g2 * 120
        aided_gain = 0.5 if self.with_hearing_aid else 0
        st.speech_discrimination = max(10, 100 * (1 - g2) + aided_gain * 30)
        self.target_db = min(120, self.target_db + self.progression)
        return {"disease": "SNHL", "pta_db": st.pta_db,
                "speech": st.speech_discrimination, "gamma_sq": g2}

# ============================================================================
# 2. CONDUCTIVE HEARING LOSS
# ============================================================================
@dataclass
class CHLState:
    ossicular_z: float = Z_OSSICULAR
    air_bone_gap: float = 0.0  # dB
    pta_air: float = 10.0
    pta_bone: float = 10.0

class ConductiveHLModel:
    def __init__(self, cause: str = "otosclerosis", severity: float = 0.5):
        self.state = CHLState()
        self.cause = cause
        self.severity = severity
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.cause == "otosclerosis":
            st.ossicular_z = Z_OSSICULAR * (1 + self.severity * 4)
        elif self.cause == "ossicular_discontinuity":
            st.ossicular_z = Z_OSSICULAR * (1 + self.severity * 10)
        g2 = gamma_sq(st.ossicular_z, Z_OSSICULAR)
        st.pta_air = 10 + g2 * 60
        st.pta_bone = 10  # Normal bone conduction
        st.air_bone_gap = st.pta_air - st.pta_bone
        return {"disease": "CHL", "air_bone_gap": st.air_bone_gap,
                "pta_air": st.pta_air, "cause": self.cause, "gamma_sq": g2}

# ============================================================================
# 3. MÉNIÈRE DISEASE
# ============================================================================
@dataclass
class MeniereState:
    endolymph_z: float = Z_COCHLEA
    vertigo_severity: float = 0.0  # 0–10
    hearing_fluctuation: float = 0.0
    stage: int = 1  # AAO-HNS 1–4
    in_attack: bool = False

class MeniereModel:
    def __init__(self, severity: float = 0.5):
        self.state = MeniereState()
        self.severity = severity
        self.attack_cycle = 10  # ticks
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.in_attack = (self.tick_count % self.attack_cycle) < 3
        if st.in_attack:
            st.endolymph_z = Z_COCHLEA * (1 + self.severity * 3)
        else:
            st.endolymph_z = Z_COCHLEA * (1 + self.severity * 0.5)
        g2 = gamma_sq(st.endolymph_z, Z_COCHLEA)
        st.vertigo_severity = min(10, g2 * 15) if st.in_attack else 0
        st.hearing_fluctuation = g2 * 40  # dB fluctuation
        if g2 < 0.1: st.stage = 1
        elif g2 < 0.25: st.stage = 2
        elif g2 < 0.5: st.stage = 3
        else: st.stage = 4
        return {"disease": "Meniere", "stage": st.stage, "vertigo": st.vertigo_severity,
                "in_attack": st.in_attack, "gamma_sq": g2}

# ============================================================================
# 4. TINNITUS
# ============================================================================
@dataclass
class TinnitusState:
    phantom_z: float = Z_COCHLEA
    thi_score: float = 0.0     # Tinnitus Handicap Inventory 0–100
    loudness_db: float = 0.0   # Matched loudness dB
    frequency_hz: float = 4000.0

class TinnitusModel:
    def __init__(self, severity: float = 0.4, frequency: float = 4000.0):
        self.state = TinnitusState(frequency_hz=frequency)
        self.severity = severity
        self.on_trt = False  # Tinnitus Retraining Therapy
        self.tick_count = 0

    def start_trt(self):
        self.on_trt = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        sev = self.severity * (0.6 if self.on_trt else 1.0)
        st.phantom_z = Z_COCHLEA * (1 + sev * 2)
        g2 = gamma_sq(st.phantom_z, Z_COCHLEA)
        st.thi_score = min(100, g2 * 130)
        st.loudness_db = g2 * 20
        return {"disease": "Tinnitus", "thi": st.thi_score,
                "loudness_db": st.loudness_db, "freq_hz": st.frequency_hz, "gamma_sq": g2}

# ============================================================================
# 5. OTITIS MEDIA
# ============================================================================
@dataclass
class OtitisMediaState:
    middle_ear_z: float = Z_TYMPANUM
    effusion: bool = False
    tympanometry: str = "A"   # A=normal, B=flat, C=negative
    pta_db: float = 10.0

class OtitisMediaModel:
    def __init__(self, effusion: bool = True, acute: bool = True):
        self.state = OtitisMediaState(effusion=effusion)
        self.acute = acute
        self.on_antibiotics = False
        self.tick_count = 0

    def start_antibiotics(self):
        self.on_antibiotics = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_antibiotics and self.acute:
            st.effusion = self.tick_count < 5
        if st.effusion:
            st.middle_ear_z = Z_TYMPANUM * 3  # Fluid mass loading
            st.tympanometry = "B"
        else:
            st.middle_ear_z = Z_TYMPANUM
            st.tympanometry = "A"
        g2 = gamma_sq(st.middle_ear_z, Z_TYMPANUM)
        st.pta_db = 10 + g2 * 40
        return {"disease": "Otitis Media", "effusion": st.effusion,
                "tymp": st.tympanometry, "pta_db": st.pta_db, "gamma_sq": g2}

# ============================================================================
# 6. VOCAL CORD PARALYSIS
# ============================================================================
@dataclass
class VocalCordState:
    larynx_z: float = Z_LARYNX
    vhi: float = 0.0         # Voice Handicap Index 0–120
    voice_quality: float = 1.0
    aspiration_risk: float = 0.0

class VocalCordParalysisModel:
    def __init__(self, unilateral: bool = True, severity: float = 0.6):
        self.state = VocalCordState()
        self.unilateral = unilateral
        self.severity = severity
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        # Paralysis = open circuit (Z → ∞) for affected side
        factor = 3 if self.unilateral else 8
        st.larynx_z = Z_LARYNX * (1 + self.severity * factor)
        g2 = gamma_sq(st.larynx_z, Z_LARYNX)
        st.vhi = min(120, g2 * 150)
        st.voice_quality = max(0.1, 1.0 - g2)
        st.aspiration_risk = g2 * 0.5 if not self.unilateral else g2 * 0.2
        return {"disease": "Vocal Cord Paralysis", "vhi": st.vhi,
                "quality": st.voice_quality, "aspiration": st.aspiration_risk,
                "gamma_sq": g2}

# ============================================================================
# 7. SINUSITIS
# ============================================================================
@dataclass
class SinusitisState:
    sinus_z: float = Z_SINUS
    lund_mackay: int = 0     # CT score 0–24
    nasal_obstruction: float = 0.0
    facial_pain: float = 0.0

class SinusitisModel:
    def __init__(self, chronic: bool = False, severity: float = 0.5):
        self.state = SinusitisState()
        self.chronic = chronic
        self.severity = severity
        self.on_treatment = False
        self.tick_count = 0

    def start_treatment(self):
        self.on_treatment = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        sev = self.severity * (0.5 if self.on_treatment else 1.0)
        st.sinus_z = Z_SINUS * (1 + sev * 4)
        g2 = gamma_sq(st.sinus_z, Z_SINUS)
        st.lund_mackay = min(24, int(g2 * 30))
        st.nasal_obstruction = min(10, g2 * 12)
        st.facial_pain = min(10, g2 * 10)
        return {"disease": "Sinusitis", "lund_mackay": st.lund_mackay,
                "obstruction": st.nasal_obstruction, "pain": st.facial_pain,
                "gamma_sq": g2}

# ============================================================================
# 8. ANOSMIA
# ============================================================================
@dataclass
class AnosmiaState:
    olfactory_z: float = Z_NASAL
    upsit: float = 35.0     # University of Pennsylvania Smell ID (0–40)
    cause: str = "post-viral"

class AnosmiaModel:
    def __init__(self, severity: float = 0.7, cause: str = "post-viral"):
        self.state = AnosmiaState(cause=cause)
        self.severity = severity
        self.recovering = False
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.recovering:
            self.severity = max(0, self.severity - 0.005)
        st.olfactory_z = Z_NASAL * (1 + self.severity * 5)
        g2 = gamma_sq(st.olfactory_z, Z_NASAL)
        st.upsit = max(5, 35 * (1 - g2))
        return {"disease": "Anosmia", "upsit": st.upsit, "cause": st.cause,
                "gamma_sq": g2}

# ============================================================================
# 9. SSHL (Sudden Sensorineural Hearing Loss)
# ============================================================================
@dataclass
class SSHLState:
    cochlear_z: float = Z_COCHLEA
    pta_db: float = 10.0
    onset_hours: int = 0
    recovery: float = 0.0  # 0–1

class SSHLModel:
    def __init__(self, severity_db: float = 60.0):
        self.state = SSHLState()
        self.target_db = severity_db
        self.on_steroids = False
        self.tick_count = 0

    def start_steroids(self):
        self.on_steroids = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        st.onset_hours = self.tick_count
        if self.on_steroids:
            self.target_db = max(10, self.target_db * 0.97)
        st.cochlear_z = Z_COCHLEA * (1 + self.target_db / 30)
        g2 = gamma_sq(st.cochlear_z, Z_COCHLEA)
        st.pta_db = g2 * 120
        st.recovery = max(0, 1.0 - g2 / gamma_sq(Z_COCHLEA * 3, Z_COCHLEA))
        return {"disease": "SSHL", "pta_db": st.pta_db, "recovery": st.recovery,
                "gamma_sq": g2}

# ============================================================================
# 10. BPPV (Benign Paroxysmal Positional Vertigo)
# ============================================================================
@dataclass
class BPPVState:
    vestibular_z: float = Z_VESTIBULAR
    nystagmus: bool = False
    vertigo_severity: float = 0.0  # 0–10
    dix_hallpike: bool = True

class BPPVModel:
    def __init__(self, canal: str = "posterior", severity: float = 0.5):
        self.state = BPPVState()
        self.canal = canal
        self.severity = severity
        self.epley_done = False
        self.tick_count = 0

    def do_epley(self):
        self.epley_done = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.epley_done:
            self.severity = max(0, self.severity * 0.5)
            self.epley_done = False  # Single maneuver
        # Positional trigger
        triggered = (self.tick_count % 4) == 0
        if triggered:
            st.vestibular_z = Z_VESTIBULAR * (1 + self.severity * 3)
        else:
            st.vestibular_z = Z_VESTIBULAR * (1 + self.severity * 0.1)
        g2 = gamma_sq(st.vestibular_z, Z_VESTIBULAR)
        st.nystagmus = g2 > 0.1 and triggered
        st.vertigo_severity = min(10, g2 * 15) if triggered else 0
        st.dix_hallpike = g2 > 0.05
        return {"disease": "BPPV", "vertigo": st.vertigo_severity,
                "nystagmus": st.nystagmus, "canal": self.canal, "gamma_sq": g2}

# ============================================================================
# UNIFIED ENGINE
# ============================================================================
class ClinicalENTEngine:
    DISEASE_CLASSES = {
        "snhl": SNHLModel, "chl": ConductiveHLModel,
        "meniere": MeniereModel, "tinnitus": TinnitusModel,
        "otitis_media": OtitisMediaModel, "vocal_cord": VocalCordParalysisModel,
        "sinusitis": SinusitisModel, "anosmia": AnosmiaModel,
        "sshl": SSHLModel, "bppv": BPPVModel,
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
        results["sensory_reserve"] = max(0.0, 1.0 - total_g2)
        return results
