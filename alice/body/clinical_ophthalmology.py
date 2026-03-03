# -*- coding: utf-8 -*-
"""clinical_ophthalmology.py — Top-10 Eye Diseases as Impedance-Mismatch Patterns
=================================================================================

│ Disease                  │ Γ Physics Mapping                          │ Clinical Scale  │
│──────────────────────────│────────────────────────────────────────────│────────────────│
│ 1. Glaucoma              │ Optic nerve Z compression (IOP)           │ IOP / VF MD    │
│ 2. Cataract              │ Lens Z opacity (scatter)                  │ LOCS III       │
│ 3. Retinal Detachment    │ Photoreceptor Z discontinuity             │ Area / Macula  │
│ 4. AMD                   │ Central retinal Z degradation             │ AREDS stage    │
│ 5. Diabetic Retinopathy  │ Vascular Z leakage / neovascularization   │ ETDRS stage    │
│ 6. Refractive Error      │ Focal Z mismatch                          │ Diopters       │
│ 7. Dry Eye               │ Tear film Z disruption                    │ OSDI score     │
│ 8. Corneal Ulcer         │ Surface Z failure                         │ Size / Depth   │
│ 9. Optic Neuritis        │ Nerve Z demyelination                     │ VA / RAPD      │
│ 10. Strabismus           │ Binocular Z alignment error               │ Prism Δ        │

The eye is an optical impedance-matching system:
Z_cornea → Z_aqueous → Z_lens → Z_vitreous → Z_retina.
Disease = any Z mismatch along the optical axis → Γ² = lost signal.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict

# Physical Constants (Ω)
Z_CORNEA: float = 45.0
Z_AQUEOUS: float = 35.0
Z_LENS: float = 50.0
Z_VITREOUS: float = 35.0
Z_RETINA: float = 50.0
Z_OPTIC_NERVE: float = 55.0

def gamma_sq(z_l: float, z_0: float) -> float:
    g = (z_l - z_0) / (z_l + z_0)
    return g * g

# ============================================================================
# 1. GLAUCOMA
# ============================================================================
@dataclass
class GlaucomaState:
    iop: float = 16.0          # mmHg (normal 10–21)
    nerve_z: float = Z_OPTIC_NERVE
    vf_md: float = 0.0        # Visual field mean deviation dB
    cup_disc: float = 0.3     # Normal <0.5

class GlaucomaModel:
    def __init__(self, iop: float = 28.0, damage_rate: float = 0.005):
        self.state = GlaucomaState(iop=iop)
        self.damage_rate = damage_rate
        self.on_drops = False
        self.tick_count = 0

    def start_drops(self):
        self.on_drops = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_drops:
            st.iop = max(12, st.iop * 0.98)
        pressure_excess = max(0, st.iop - 21)
        st.nerve_z = min(Z_OPTIC_NERVE * 5,
                         st.nerve_z + pressure_excess * self.damage_rate)
        g2 = gamma_sq(st.nerve_z, Z_OPTIC_NERVE)
        st.vf_md = -g2 * 30  # dB loss (negative = worse)
        st.cup_disc = min(0.95, 0.3 + g2 * 0.6)
        return {"disease": "Glaucoma", "iop": st.iop, "vf_md": st.vf_md,
                "cup_disc": st.cup_disc, "gamma_sq": g2}

# ============================================================================
# 2. CATARACT
# ============================================================================
@dataclass
class CataractState:
    lens_z: float = Z_LENS
    va: float = 1.0           # Snellen decimal (1.0 = 20/20)
    locs_nuclear: int = 1     # LOCS III (1–6)
    locs_cortical: int = 1

class CataractModel:
    def __init__(self, opacity: float = 0.3, age_progression: float = 0.001):
        self.state = CataractState()
        self.opacity = opacity
        self.progression = age_progression
        self.operated = False
        self.tick_count = 0

    def surgery(self):
        self.operated = True
        self.opacity = 0.0
        self.state.lens_z = Z_LENS

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if not self.operated:
            self.opacity = min(0.9, self.opacity + self.progression)
        st.lens_z = Z_LENS * (1 + self.opacity * 4)
        g2 = gamma_sq(st.lens_z, Z_LENS)
        st.va = max(0.05, 1.0 - g2)
        st.locs_nuclear = min(6, 1 + int(self.opacity * 6))
        return {"disease": "Cataract", "va": st.va, "locs": st.locs_nuclear,
                "opacity": self.opacity, "gamma_sq": g2}

# ============================================================================
# 3. RETINAL DETACHMENT
# ============================================================================
@dataclass
class RetinalDetachmentState:
    retinal_z: float = Z_RETINA
    detached_area: float = 0.0  # Fraction 0–1
    macula_involved: bool = False
    va: float = 1.0

class RetinalDetachmentModel:
    def __init__(self, initial_area: float = 0.1, progression: float = 0.02):
        self.state = RetinalDetachmentState(detached_area=initial_area)
        self.progression = progression
        self.repaired = False
        self.tick_count = 0

    def surgery_repair(self):
        self.repaired = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.repaired:
            st.detached_area = max(0, st.detached_area * 0.95)
        else:
            st.detached_area = min(1.0, st.detached_area + self.progression)
        st.macula_involved = st.detached_area > 0.3
        st.retinal_z = Z_RETINA * (1 + st.detached_area * 10)
        g2 = gamma_sq(st.retinal_z, Z_RETINA)
        st.va = max(0.01, 1.0 - g2)
        return {"disease": "Retinal Detachment", "area": st.detached_area,
                "macula": st.macula_involved, "va": st.va, "gamma_sq": g2}

# ============================================================================
# 4. AMD (Age-related Macular Degeneration)
# ============================================================================
@dataclass
class AMDState:
    macular_z: float = Z_RETINA
    drusen_count: int = 0
    wet: bool = False      # Wet vs Dry
    va: float = 1.0
    areds_stage: int = 1   # 1–4

class AMDModel:
    def __init__(self, stage: int = 2, wet: bool = False):
        self.state = AMDState(areds_stage=stage, wet=wet)
        self.on_anti_vegf = False
        self.tick_count = 0

    def start_anti_vegf(self):
        self.on_anti_vegf = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        base_damage = st.areds_stage * 0.1
        if st.wet:
            base_damage *= 2
        if self.on_anti_vegf and st.wet:
            base_damage *= 0.4
        st.macular_z = Z_RETINA * (1 + base_damage * 3)
        g2 = gamma_sq(st.macular_z, Z_RETINA)
        st.va = max(0.02, 1.0 - g2)
        st.drusen_count = int(g2 * 20)
        return {"disease": "AMD", "stage": st.areds_stage, "wet": st.wet,
                "va": st.va, "gamma_sq": g2}

# ============================================================================
# 5. DIABETIC RETINOPATHY
# ============================================================================
@dataclass
class DRState:
    retinal_vascular_z: float = Z_RETINA
    hba1c: float = 8.0
    etdrs_stage: str = "mild NPDR"
    va: float = 1.0
    macular_edema: bool = False

class DiabeticRetinopathyModel:
    def __init__(self, hba1c: float = 9.0, duration: float = 10):
        self.state = DRState(hba1c=hba1c)
        self.damage = min(0.8, (hba1c - 5.5) * 0.02 * math.sqrt(duration))
        self.on_anti_vegf = False
        self.tick_count = 0

    def start_anti_vegf(self):
        self.on_anti_vegf = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        d = self.damage * (0.5 if self.on_anti_vegf else 1.0)
        st.retinal_vascular_z = Z_RETINA * (1 + d * 4)
        g2 = gamma_sq(st.retinal_vascular_z, Z_RETINA)
        st.macular_edema = g2 > 0.3
        if g2 < 0.1: st.etdrs_stage = "mild NPDR"
        elif g2 < 0.25: st.etdrs_stage = "moderate NPDR"
        elif g2 < 0.4: st.etdrs_stage = "severe NPDR"
        else: st.etdrs_stage = "PDR"
        st.va = max(0.05, 1.0 - g2)
        return {"disease": "DR", "stage": st.etdrs_stage, "va": st.va,
                "edema": st.macular_edema, "gamma_sq": g2}

# ============================================================================
# 6. REFRACTIVE ERROR
# ============================================================================
@dataclass
class RefractiveState:
    focal_z: float = Z_LENS
    diopters: float = 0.0     # Negative=myopia, Positive=hyperopia
    va_uncorrected: float = 1.0
    va_corrected: float = 1.0

class RefractiveModel:
    def __init__(self, diopters: float = -3.0):
        self.state = RefractiveState(diopters=diopters)
        self.corrected = False
        self.tick_count = 0

    def apply_correction(self):
        self.corrected = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        # Z mismatch proportional to |diopters|
        z_shift = abs(st.diopters) * 3
        st.focal_z = Z_LENS + z_shift * (1 if st.diopters > 0 else -1)
        g2 = gamma_sq(st.focal_z, Z_LENS)
        st.va_uncorrected = max(0.05, 1.0 - g2)
        st.va_corrected = 1.0 if self.corrected else st.va_uncorrected
        return {"disease": "Refractive", "diopters": st.diopters,
                "va_uc": st.va_uncorrected, "va_c": st.va_corrected, "gamma_sq": g2}

# ============================================================================
# 7. DRY EYE
# ============================================================================
@dataclass
class DryEyeState:
    tear_z: float = Z_CORNEA
    osdi: float = 0.0         # OSDI Score 0–100
    tbut: float = 10.0        # Tear break-up time (s)
    schirmer: float = 15.0    # mm/5min

class DryEyeModel:
    def __init__(self, severity: float = 0.4, evaporative: bool = True):
        self.state = DryEyeState()
        self.severity = severity
        self.evaporative = evaporative
        self.on_artificial_tears = False
        self.tick_count = 0

    def start_treatment(self):
        self.on_artificial_tears = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        sev = self.severity * (0.5 if self.on_artificial_tears else 1.0)
        st.tear_z = Z_CORNEA * (1 + sev * 2)
        g2 = gamma_sq(st.tear_z, Z_CORNEA)
        st.osdi = min(100, g2 * 130)
        st.tbut = max(1, 10 - g2 * 8)
        st.schirmer = max(1, 15 - g2 * 12)
        return {"disease": "Dry Eye", "osdi": st.osdi, "tbut": st.tbut,
                "schirmer": st.schirmer, "gamma_sq": g2}

# ============================================================================
# 8. CORNEAL ULCER
# ============================================================================
@dataclass
class CornealUlcerState:
    corneal_z: float = Z_CORNEA
    ulcer_size_mm: float = 2.0
    depth_fraction: float = 0.3  # 0–1 (1=perforation)
    va: float = 0.5

class CornealUlcerModel:
    def __init__(self, size: float = 3.0, depth: float = 0.3):
        self.state = CornealUlcerState(ulcer_size_mm=size, depth_fraction=depth)
        self.on_antibiotics = False
        self.tick_count = 0

    def start_antibiotics(self):
        self.on_antibiotics = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_antibiotics:
            st.ulcer_size_mm = max(0, st.ulcer_size_mm * 0.97)
            st.depth_fraction = max(0, st.depth_fraction * 0.97)
        else:
            st.ulcer_size_mm = min(10, st.ulcer_size_mm * 1.02)
            st.depth_fraction = min(1.0, st.depth_fraction * 1.01)
        severity = st.ulcer_size_mm / 10 * st.depth_fraction
        st.corneal_z = Z_CORNEA * (1 + severity * 5)
        g2 = gamma_sq(st.corneal_z, Z_CORNEA)
        st.va = max(0.01, 1.0 - g2)
        return {"disease": "Corneal Ulcer", "size_mm": st.ulcer_size_mm,
                "depth": st.depth_fraction, "va": st.va, "gamma_sq": g2}

# ============================================================================
# 9. OPTIC NEURITIS
# ============================================================================
@dataclass
class OpticNeuritisState:
    nerve_z: float = Z_OPTIC_NERVE
    va: float = 0.3
    rapd: bool = True          # Relative Afferent Pupillary Defect
    pain_with_movement: bool = True
    recovering: bool = False

class OpticNeuritisModel:
    def __init__(self, severity: float = 0.7, ms_associated: bool = False):
        self.state = OpticNeuritisState()
        self.severity = severity
        self.ms_associated = ms_associated
        self.on_steroids = False
        self.tick_count = 0

    def start_steroids(self):
        self.on_steroids = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_steroids:
            self.severity = max(0.05, self.severity * 0.97)
            st.recovering = True
        st.nerve_z = Z_OPTIC_NERVE * (1 + self.severity * 3)
        g2 = gamma_sq(st.nerve_z, Z_OPTIC_NERVE)
        st.va = max(0.02, 1.0 - g2)
        st.rapd = g2 > 0.15
        st.pain_with_movement = g2 > 0.2 and not st.recovering
        return {"disease": "Optic Neuritis", "va": st.va, "rapd": st.rapd,
                "pain": st.pain_with_movement, "gamma_sq": g2}

# ============================================================================
# 10. STRABISMUS
# ============================================================================
@dataclass
class StrabismusState:
    alignment_z_left: float = Z_RETINA
    alignment_z_right: float = Z_RETINA
    prism_diopters: float = 0.0  # Deviation in prism Δ
    stereoacuity: float = 40.0   # arc-sec (normal 40)

class StrabismusModel:
    def __init__(self, deviation: float = 20.0, direction: str = "esotropia"):
        self.state = StrabismusState(prism_diopters=deviation)
        self.direction = direction
        self.corrected = False
        self.tick_count = 0

    def surgery_or_prism(self):
        self.corrected = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        dev = st.prism_diopters * (0.1 if self.corrected else 1.0)
        z_shift = dev * 0.5
        st.alignment_z_left = Z_RETINA
        st.alignment_z_right = Z_RETINA + z_shift
        g2 = gamma_sq(st.alignment_z_right, st.alignment_z_left)
        st.stereoacuity = 40 + g2 * 500  # Worse stereopsis
        return {"disease": "Strabismus", "deviation": st.prism_diopters,
                "direction": self.direction, "stereoacuity": st.stereoacuity,
                "gamma_sq": g2}

# ============================================================================
# UNIFIED ENGINE
# ============================================================================
class ClinicalOphthalmologyEngine:
    DISEASE_CLASSES = {
        "glaucoma": GlaucomaModel, "cataract": CataractModel,
        "retinal_detachment": RetinalDetachmentModel, "amd": AMDModel,
        "diabetic_retinopathy": DiabeticRetinopathyModel,
        "refractive": RefractiveModel, "dry_eye": DryEyeModel,
        "corneal_ulcer": CornealUlcerModel, "optic_neuritis": OpticNeuritisModel,
        "strabismus": StrabismusModel,
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
        results["visual_reserve"] = max(0.0, 1.0 - total_g2)
        return results
