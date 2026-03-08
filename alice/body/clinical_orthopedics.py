# -*- coding: utf-8 -*-
"""clinical_orthopedics.py — Top-10 Musculoskeletal Diseases as Impedance-Mismatch Patterns
============================================================================================

│ Disease                  │ Γ Physics Mapping                          │ Clinical Scale  │
│──────────────────────────│────────────────────────────────────────────│────────────────│
│ 1. Fracture              │ Structural Z discontinuity (open circuit) │ AO/OTA class   │
│ 2. Osteoporosis          │ Bone Z degradation (T-score)              │ DXA T-score    │
│ 3. Disc Herniation       │ Spinal Z compression                     │ Oswestry ODI   │
│ 4. Osteoarthritis        │ Cartilage Z wear                         │ K-L Grade      │
│ 5. ACL Tear              │ Stabilization Z failure                   │ Lachman / IKDC │
│ 6. Tendinitis            │ Connective Z inflammation                 │ VAS / DASH     │
│ 7. Scoliosis             │ Structural Z misalignment                 │ Cobb angle     │
│ 8. Osteosarcoma          │ Bone Z transformation                     │ TNM / Ennneking│
│ 9. Gout                  │ Crystal Z deposition                      │ Serum urate    │
│ 10. Osteomyelitis        │ Bone Z infection                          │ Cierny-Mader   │

Bone/ligament/cartilage = structural impedance. Z_BONE ≈ highest biological Z.
Disease = structural Z failure → mechanical energy reflected → pain.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict

# Physical Constants (Ω)
Z_BONE: float = 120.0       # Cortical bone (high impedance)
Z_CARTILAGE: float = 80.0   # Articular cartilage
Z_TENDON: float = 90.0      # Tendon/ligament
Z_DISC: float = 70.0        # Intervertebral disc
Z_SYNOVIAL: float = 50.0    # Synovial fluid

def gamma_sq(z_l: float, z_0: float) -> float:
    g = (z_l - z_0) / (z_l + z_0)
    return g * g


def transmission(z_l: float, z_0: float) -> float:
    """T = 1 - Γ² at a single interface (C1)."""
    return 1.0 - gamma_sq(z_l, z_0)


# ============================================================================
# JOINT IMPEDANCE TRANSFORMER  (multi-layer graded transition)
# ============================================================================
# A healthy joint is NOT a single impedance discontinuity.
# It is a multi-layer transformer:  Bone → Cartilage → Synovial → Cartilage → Bone
# Each layer steps Z down/up gradually, keeping every per-interface Γ small.
# Total transmission = product of per-interface transmissions.
#
# Physics analogy: quarter-wave transformer in microwave engineering.
# When cartilage degrades (OA), the graded transition collapses and
# the bone-to-synovial interface becomes abrupt → Γ² jumps → pain.
# ============================================================================

def joint_transmission(cartilage_z: float = Z_CARTILAGE) -> dict:
    """
    Compute multi-layer transmission through a synovial joint.

    Healthy:  Bone(120) → Cartilage(80) → Synovial(50) → Cartilage(80) → Bone(120)
    OA:       Bone(120) → thin_cart → Synovial(50) → thin_cart → Bone(120)
    End-stage (bone-on-bone): Bone(120) → Synovial(50) → Bone(120)

    Returns dict with per-interface Γ², total Γ², and total T.
    """
    # 4 interfaces in the healthy layered stack
    interfaces = [
        ("bone→cart",     Z_BONE,       cartilage_z),
        ("cart→synovial", cartilage_z,  Z_SYNOVIAL),
        ("synovial→cart", Z_SYNOVIAL,   cartilage_z),
        ("cart→bone",     cartilage_z,  Z_BONE),
    ]

    T_total = 1.0
    detail = {}
    for name, z_source, z_load in interfaces:
        g2 = gamma_sq(z_load, z_source)
        T_total *= (1.0 - g2)
        detail[name] = g2

    detail["T_total"] = T_total
    detail["gamma_sq_total"] = 1.0 - T_total
    return detail


def joint_transmission_no_cartilage() -> dict:
    """Bone-on-bone (end-stage OA): only 2 interfaces, no graded transition."""
    interfaces = [
        ("bone→synovial", Z_BONE,     Z_SYNOVIAL),
        ("synovial→bone", Z_SYNOVIAL, Z_BONE),
    ]
    T_total = 1.0
    detail = {}
    for name, z_source, z_load in interfaces:
        g2 = gamma_sq(z_load, z_source)
        T_total *= (1.0 - g2)
        detail[name] = g2
    detail["T_total"] = T_total
    detail["gamma_sq_total"] = 1.0 - T_total
    return detail


# ============================================================================
# AGING PAIN IMMUNITY MODEL
# ============================================================================
# Slow Z drift does NOT cause pain because:
# 1. C2 Hebbian adaptation continuously re-matches: ΔZ = −η·Γ·x_pre·x_post
# 2. Reflection heat = Γ² × gain is tiny for small Γ
# 3. Cooling rate (0.035/tick) >> heating from slow drift
# 4. Pain threshold T > 0.7 is never reached
#
# Pain requires EITHER:
#   - Sudden Z change (fracture, amputation)    → dΓ/dt >> C2 adaptation rate
#   - η ≈ 0 (cartilage, avascular tissue)       → C2 cannot compensate
#   - Late-stage aging: δ·D(t) > η·ΣΓ²          → aging outpaces adaptation
# ============================================================================

def aging_pain_model(
    z_drift_rate: float,     # dZ/dt per tick (aging speed)
    eta: float,              # C2 adaptation rate
    cooling: float = 0.035,  # passive cooling per tick
    heat_gain: float = 0.10, # Γ² → temperature coefficient
    pain_threshold: float = 0.7,
    n_ticks: int = 1000,
) -> dict:
    """
    Simulate whether slow impedance drift produces pain.

    Returns trajectory of Z, Γ², temperature, and whether pain was triggered.
    """
    z_source = 75.0  # reference Z
    z_load = 75.0    # starts matched
    temp = 0.3       # resting temperature
    pain_triggered = False
    peak_gamma_sq = 0.0
    peak_temp = 0.0

    for t in range(n_ticks):
        # Aging: Z drifts up
        z_load += z_drift_rate

        # C2 Hebbian adaptation: tries to re-match
        gamma = (z_load - z_source) / (z_load + z_source)
        g2 = gamma * gamma
        delta_z = -eta * gamma  # simplified C2: ΔZ ∝ −η·Γ
        z_load += delta_z

        # Recalculate after adaptation
        gamma = (z_load - z_source) / (z_load + z_source)
        g2 = gamma * gamma
        peak_gamma_sq = max(peak_gamma_sq, g2)

        # Temperature dynamics
        heat = g2 * heat_gain
        temp = max(0.0, min(1.0, temp + heat - cooling))
        peak_temp = max(peak_temp, temp)

        if temp > pain_threshold:
            pain_triggered = True
            break

    return {
        "pain_triggered": pain_triggered,
        "peak_gamma_sq": peak_gamma_sq,
        "peak_temp": peak_temp,
        "final_z_load": z_load,
        "final_z_mismatch": abs(z_load - z_source),
        "ticks_run": t + 1 if pain_triggered else n_ticks,
    }


# ============================================================================
# 1. FRACTURE
# ============================================================================
@dataclass
class FractureState:
    bone_z: float = Z_BONE
    callus_z: float = 0.0      # Healing callus Z (starts at 0)
    healing_fraction: float = 0.0  # 0→1
    pain_vas: float = 8.0
    ao_class: str = "A1"

class FractureModel:
    def __init__(self, ao_class: str = "A2", displacement: float = 0.5):
        self.state = FractureState(ao_class=ao_class)
        self.displacement = displacement
        self.fixated = False
        self.tick_count = 0

    def fixation(self):
        self.fixated = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        heal_rate = 0.02 if self.fixated else 0.005
        st.healing_fraction = min(1.0, st.healing_fraction + heal_rate)
        st.callus_z = Z_BONE * st.healing_fraction
        g2 = gamma_sq(st.callus_z, Z_BONE) if st.healing_fraction < 1 \
            else 0.0
        st.pain_vas = max(0, 8 * (1 - st.healing_fraction))
        return {"disease": "Fracture", "ao": st.ao_class, "healing": st.healing_fraction,
                "pain": st.pain_vas, "gamma_sq": g2}

# ============================================================================
# 2. OSTEOPOROSIS
# ============================================================================
@dataclass
class OsteoporosisState:
    bone_z: float = Z_BONE
    t_score: float = 0.0      # DXA T-score (normal >-1)
    fracture_risk: float = 0.0  # FRAX 10-year %

class OsteoporosisModel:
    def __init__(self, t_score: float = -2.5, age: float = 70):
        self.state = OsteoporosisState(t_score=t_score)
        self.age = age
        self.on_bisphosphonate = False
        self.tick_count = 0

    def start_bisphosphonate(self):
        self.on_bisphosphonate = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_bisphosphonate:
            st.t_score = min(0, st.t_score + 0.002)
        else:
            st.t_score = max(-5, st.t_score - 0.001)
        # T-score → Z degradation
        z_deficit = max(0, -st.t_score) / 5
        st.bone_z = Z_BONE * (1 - z_deficit * 0.4)
        g2 = gamma_sq(st.bone_z, Z_BONE)
        st.fracture_risk = min(50, g2 * 60 * (self.age / 50))
        return {"disease": "Osteoporosis", "t_score": st.t_score,
                "frax": st.fracture_risk, "gamma_sq": g2}

# ============================================================================
# 3. DISC HERNIATION
# ============================================================================
@dataclass
class DiscHerniationState:
    disc_z: float = Z_DISC
    odi: float = 0.0           # Oswestry Disability Index 0–100
    pain_vas: float = 0.0
    radiculopathy: bool = False
    level: str = "L4-L5"

class DiscHerniationModel:
    def __init__(self, severity: float = 0.5, level: str = "L4-L5"):
        self.state = DiscHerniationState(level=level)
        self.severity = severity
        self.on_conservative = False
        self.tick_count = 0

    def start_conservative(self):
        self.on_conservative = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        sev = self.severity
        if self.on_conservative:
            sev = max(0, sev - 0.005)
            self.severity = sev
        st.disc_z = Z_DISC * (1 - sev * 0.4)
        g2 = gamma_sq(st.disc_z, Z_DISC)
        st.odi = min(100, g2 * 130)
        st.pain_vas = min(10, g2 * 12)
        st.radiculopathy = g2 > 0.2
        return {"disease": "Disc Herniation", "odi": st.odi, "pain": st.pain_vas,
                "radiculopathy": st.radiculopathy, "level": st.level, "gamma_sq": g2}

# ============================================================================
# 4. OSTEOARTHRITIS
# ============================================================================
@dataclass
class OAState:
    cartilage_z: float = Z_CARTILAGE
    kl_grade: int = 2          # Kellgren-Lawrence 0–4
    womac: float = 0.0         # WOMAC score
    joint_space: float = 4.0   # mm

class OsteoarthritisModel:
    def __init__(self, kl_grade: int = 2, joint: str = "knee"):
        self.state = OAState(kl_grade=kl_grade)
        self.joint = joint
        self.progression = 0.001
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        wear = st.kl_grade / 4  # 0–1 normalized
        st.cartilage_z = Z_CARTILAGE * (1 - wear * 0.6)
        g2 = gamma_sq(st.cartilage_z, Z_CARTILAGE)
        st.womac = min(96, g2 * 120)
        st.joint_space = max(0.5, 4.0 * (1 - g2))
        return {"disease": "OA", "kl": st.kl_grade, "womac": st.womac,
                "joint_space": st.joint_space, "joint": self.joint, "gamma_sq": g2}

# ============================================================================
# 5. ACL TEAR
# ============================================================================
@dataclass
class ACLState:
    ligament_z: float = Z_TENDON
    lachman: bool = True       # Positive = torn
    ikdc: float = 50.0         # IKDC score 0–100
    partial: bool = False

class ACLModel:
    def __init__(self, partial: bool = False):
        self.state = ACLState(partial=partial)
        self.reconstructed = False
        self.tick_count = 0

    def reconstruction(self):
        self.reconstructed = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.reconstructed:
            heal = min(1.0, self.tick_count * 0.01)
            st.ligament_z = Z_TENDON * heal
        else:
            tear_frac = 0.5 if st.partial else 0.05
            st.ligament_z = Z_TENDON * tear_frac
        g2 = gamma_sq(st.ligament_z, Z_TENDON)
        st.lachman = g2 > 0.1
        st.ikdc = max(0, 100 * (1 - g2))
        return {"disease": "ACL", "lachman": st.lachman, "ikdc": st.ikdc,
                "partial": st.partial, "gamma_sq": g2}

# ============================================================================
# 6. TENDINITIS
# ============================================================================
@dataclass
class TendinitisState:
    tendon_z: float = Z_TENDON
    pain_vas: float = 0.0
    dash: float = 0.0          # DASH score 0–100
    location: str = "achilles"

class TendinitisModel:
    def __init__(self, severity: float = 0.4, location: str = "achilles"):
        self.state = TendinitisState(location=location)
        self.severity = severity
        self.resting = False
        self.tick_count = 0

    def rest(self):
        self.resting = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.resting:
            self.severity = max(0, self.severity - 0.01)
        st.tendon_z = Z_TENDON * (1 + self.severity * 2)
        g2 = gamma_sq(st.tendon_z, Z_TENDON)
        st.pain_vas = min(10, g2 * 12)
        st.dash = min(100, g2 * 130)
        return {"disease": "Tendinitis", "pain": st.pain_vas, "dash": st.dash,
                "location": st.location, "gamma_sq": g2}

# ============================================================================
# 7. SCOLIOSIS
# ============================================================================
@dataclass
class ScoliosisState:
    spinal_z: float = Z_BONE
    cobb_angle: float = 15.0   # degrees
    risser: int = 3            # Risser sign 0–5

class ScoliosisModel:
    def __init__(self, cobb: float = 25.0, growing: bool = True):
        self.state = ScoliosisState(cobb_angle=cobb)
        self.growing = growing
        self.braced = False
        self.tick_count = 0

    def start_brace(self):
        self.braced = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.growing and not self.braced:
            st.cobb_angle = min(80, st.cobb_angle + 0.1)
        elif self.braced:
            st.cobb_angle = max(st.cobb_angle - 0.02, 10)
        z_misalign = st.cobb_angle / 90
        st.spinal_z = Z_BONE * (1 + z_misalign * 2)
        g2 = gamma_sq(st.spinal_z, Z_BONE)
        return {"disease": "Scoliosis", "cobb": st.cobb_angle,
                "risser": st.risser, "gamma_sq": g2}

# ============================================================================
# 8. OSTEOSARCOMA
# ============================================================================
@dataclass
class OsteosarcomaState:
    bone_z: float = Z_BONE
    tumor_cm: float = 5.0
    alp: float = 100.0        # Alkaline phosphatase U/L
    enneking: str = "IIB"

class OsteosarcomaModel:
    def __init__(self, size_cm: float = 6.0, grade: str = "high"):
        self.state = OsteosarcomaState(tumor_cm=size_cm)
        self.grade = grade
        self.on_chemo = False
        self.tick_count = 0

    def start_chemo(self):
        self.on_chemo = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_chemo:
            st.tumor_cm = max(0.5, st.tumor_cm * 0.98)
        else:
            st.tumor_cm = min(20, st.tumor_cm * 1.01)
        z_transform = st.tumor_cm / 5
        st.bone_z = Z_BONE * (1 + z_transform * 2)
        g2 = gamma_sq(st.bone_z, Z_BONE)
        st.alp = 100 + g2 * 500
        if g2 < 0.15: st.enneking = "IA"
        elif g2 < 0.3: st.enneking = "IB"
        elif g2 < 0.5: st.enneking = "IIB"
        else: st.enneking = "III"
        return {"disease": "Osteosarcoma", "size_cm": st.tumor_cm,
                "alp": st.alp, "enneking": st.enneking, "gamma_sq": g2}

# ============================================================================
# 9. GOUT
# ============================================================================
@dataclass
class GoutState:
    joint_z: float = Z_SYNOVIAL
    urate: float = 7.0         # mg/dL (target <6)
    pain_vas: float = 0.0
    flare: bool = False
    tophi: bool = False

class GoutModel:
    def __init__(self, urate: float = 9.0, chronic: bool = False):
        self.state = GoutState(urate=urate, tophi=chronic)
        self.on_allopurinol = False
        self.tick_count = 0

    def start_allopurinol(self):
        self.on_allopurinol = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_allopurinol:
            st.urate = max(4, st.urate * 0.99)
        # Crystal Z deposition when urate > 6.8
        crystal = max(0, st.urate - 6.8) / 5
        st.joint_z = Z_SYNOVIAL * (1 + crystal * 5)
        g2 = gamma_sq(st.joint_z, Z_SYNOVIAL)
        st.flare = g2 > 0.2
        st.pain_vas = min(10, g2 * 12) if st.flare else g2 * 3
        st.tophi = g2 > 0.4
        return {"disease": "Gout", "urate": st.urate, "pain": st.pain_vas,
                "flare": st.flare, "tophi": st.tophi, "gamma_sq": g2}

# ============================================================================
# 10. OSTEOMYELITIS
# ============================================================================
@dataclass
class OsteomyelitisState:
    bone_z: float = Z_BONE
    wbc: float = 8.0
    esr: float = 10.0
    cierny_mader: int = 1     # Stage 1–4

class OsteomyelitisModel:
    def __init__(self, acute: bool = True, severity: float = 0.5):
        self.state = OsteomyelitisState()
        self.acute = acute
        self.severity = severity
        self.on_antibiotics = False
        self.tick_count = 0

    def start_antibiotics(self):
        self.on_antibiotics = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_antibiotics:
            self.severity = max(0, self.severity - 0.015)
        st.bone_z = Z_BONE * (1 + self.severity * 3)
        g2 = gamma_sq(st.bone_z, Z_BONE)
        st.wbc = 8 + g2 * 15
        st.esr = 10 + g2 * 80
        if g2 < 0.1: st.cierny_mader = 1
        elif g2 < 0.25: st.cierny_mader = 2
        elif g2 < 0.5: st.cierny_mader = 3
        else: st.cierny_mader = 4
        return {"disease": "Osteomyelitis", "cierny": st.cierny_mader,
                "wbc": st.wbc, "esr": st.esr, "gamma_sq": g2}

# ============================================================================
# UNIFIED ENGINE
# ============================================================================
class ClinicalOrthopedicsEngine:
    DISEASE_CLASSES = {
        "fracture": FractureModel, "osteoporosis": OsteoporosisModel,
        "disc_herniation": DiscHerniationModel, "oa": OsteoarthritisModel,
        "acl": ACLModel, "tendinitis": TendinitisModel,
        "scoliosis": ScoliosisModel, "osteosarcoma": OsteosarcomaModel,
        "gout": GoutModel, "osteomyelitis": OsteomyelitisModel,
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
        results["structural_reserve"] = max(0.0, 1.0 - total_g2)
        return results
