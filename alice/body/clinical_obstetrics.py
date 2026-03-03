# -*- coding: utf-8 -*-
"""clinical_obstetrics.py — Top-10 OB/GYN Diseases as Impedance-Mismatch Patterns
=================================================================================

│ Disease                  │ Γ Physics Mapping                          │ Clinical Scale  │
│──────────────────────────│────────────────────────────────────────────│────────────────│
│ 1. Preeclampsia          │ Placental vascular Z failure              │ BP / Proteinuria│
│ 2. PCOS                  │ HPG oscillator Z lock-up                  │ Rotterdam / AMH│
│ 3. Endometriosis         │ Ectopic endometrial Z                     │ rASRM stage    │
│ 4. Uterine Fibroids      │ Structural Z distortion (leiomyoma)      │ FIGO / UFS-QOL │
│ 5. Preterm Birth         │ Cervical Z weakness                       │ GA weeks / CL  │
│ 6. Gestational DM        │ Insulin Z drift in pregnancy              │ OGTT / Glucose │
│ 7. Ovarian Cancer        │ Gonadal Z transformation                  │ CA-125 / FIGO  │
│ 8. Menopause             │ Estrogen Z depletion                      │ MRS / FSH      │
│ 9. Amniotic Fluid Embol  │ Foreign particulate Z surge               │ DIC / SOFA     │
│ 10. Postpartum Hemorrhage│ Uterine atony Z failure                   │ EBL mL / Shock │

Reproductive = oscillator system. Disease = HPG axis Z mismatch → rhythm failure.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict

# Physical Constants (Ω)
Z_UTERINE: float = 80.0       # Uterine smooth muscle
Z_PLACENTAL: float = 50.0     # Placental villous impedance
Z_OVARIAN: float = 75.0       # Ovarian follicular
Z_CERVICAL: float = 65.0      # Cervix
Z_HPG: float = 70.0           # HPG axis set-point

def gamma_sq(z_l: float, z_0: float) -> float:
    g = (z_l - z_0) / (z_l + z_0)
    return g * g

# ============================================================================
# 1. PREECLAMPSIA
# ============================================================================
@dataclass
class PreeclampsiaState:
    placental_z: float = Z_PLACENTAL
    bp_systolic: float = 120.0
    proteinuria: float = 0.0   # g/day
    platelets: float = 250.0   # ×10³/μL
    gestational_age: float = 30.0  # weeks
    severe: bool = False

class PreeclampsiaModel:
    def __init__(self, severity: float = 0.5, ga: float = 32.0):
        self.state = PreeclampsiaState(gestational_age=ga)
        self.severity = severity
        self.on_mgso4 = False
        self.delivered = False
        self.tick_count = 0

    def start_mgso4(self):
        self.on_mgso4 = True

    def deliver(self):
        self.delivered = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.delivered:
            self.severity = max(0, self.severity - 0.05)
        sev = self.severity
        if self.on_mgso4:
            sev *= 0.7
        st.placental_z = Z_PLACENTAL * (1 + sev * 4)
        g2 = gamma_sq(st.placental_z, Z_PLACENTAL)
        st.bp_systolic = 110 + g2 * 80
        st.proteinuria = g2 * 5
        st.platelets = max(50, 250 - g2 * 200)
        st.severe = st.bp_systolic >= 160 or st.proteinuria >= 3
        return {"disease": "Preeclampsia", "bp": st.bp_systolic,
                "proteinuria": st.proteinuria, "severe": st.severe, "gamma_sq": g2}

# ============================================================================
# 2. PCOS
# ============================================================================
@dataclass
class PCOSState:
    hpg_z: float = Z_HPG
    amh: float = 8.0          # ng/mL (normal 1–3.5)
    testosterone: float = 0.6  # ng/mL
    menstrual_cycle: float = 35.0  # days (normal 28)
    ovulation: bool = False

class PCOSModel:
    def __init__(self, severity: float = 0.5, insulin_resistant: bool = True):
        self.state = PCOSState()
        self.severity = severity
        self.insulin_resistant = insulin_resistant
        self.on_ocp = False
        self.tick_count = 0

    def start_ocp(self):
        self.on_ocp = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        sev = self.severity * (0.4 if self.on_ocp else 1.0)
        st.hpg_z = Z_HPG * (1 + sev * 2)
        g2 = gamma_sq(st.hpg_z, Z_HPG)
        st.amh = 3 + g2 * 15
        st.testosterone = 0.3 + g2 * 1.5
        st.menstrual_cycle = 28 + g2 * 60
        st.ovulation = g2 < 0.2
        return {"disease": "PCOS", "amh": st.amh, "testosterone": st.testosterone,
                "cycle_days": st.menstrual_cycle, "ovulating": st.ovulation,
                "gamma_sq": g2}

# ============================================================================
# 3. ENDOMETRIOSIS
# ============================================================================
@dataclass
class EndometriosisState:
    ectopic_z: float = Z_UTERINE
    rasrm_stage: int = 2     # rASRM I–IV
    pain_vas: float = 0.0
    ca125: float = 20.0      # U/mL (normal <35)

class EndometriosisModel:
    def __init__(self, severity: float = 0.5):
        self.state = EndometriosisState()
        self.severity = severity
        self.on_gnrh_agonist = False
        self.tick_count = 0

    def start_gnrh_agonist(self):
        self.on_gnrh_agonist = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        sev = self.severity * (0.3 if self.on_gnrh_agonist else 1.0)
        st.ectopic_z = Z_UTERINE * (1 + sev * 3)
        g2 = gamma_sq(st.ectopic_z, Z_UTERINE)
        st.pain_vas = min(10, g2 * 12)
        st.ca125 = 20 + g2 * 100
        if g2 < 0.1: st.rasrm_stage = 1
        elif g2 < 0.25: st.rasrm_stage = 2
        elif g2 < 0.5: st.rasrm_stage = 3
        else: st.rasrm_stage = 4
        return {"disease": "Endometriosis", "stage": st.rasrm_stage,
                "pain": st.pain_vas, "ca125": st.ca125, "gamma_sq": g2}

# ============================================================================
# 4. UTERINE FIBROIDS
# ============================================================================
@dataclass
class FibroidState:
    uterine_z: float = Z_UTERINE
    largest_cm: float = 3.0
    ufs_qol: float = 50.0      # UFS-QOL 0–100
    heavy_bleeding: bool = False

class FibroidModel:
    def __init__(self, size_cm: float = 5.0, count: int = 2):
        self.state = FibroidState(largest_cm=size_cm)
        self.count = count
        self.tick_count = 0

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        mass_effect = st.largest_cm / 10 * self.count
        st.uterine_z = Z_UTERINE * (1 + mass_effect)
        g2 = gamma_sq(st.uterine_z, Z_UTERINE)
        st.ufs_qol = max(0, 100 - g2 * 120)
        st.heavy_bleeding = g2 > 0.15
        return {"disease": "Fibroids", "size_cm": st.largest_cm,
                "count": self.count, "qol": st.ufs_qol, "gamma_sq": g2}

# ============================================================================
# 5. PRETERM BIRTH
# ============================================================================
@dataclass
class PretermState:
    cervical_z: float = Z_CERVICAL
    cervical_length: float = 35.0  # mm (normal >25)
    ga_weeks: float = 28.0
    contractions: bool = False

class PretermBirthModel:
    def __init__(self, cervical_length: float = 20.0, ga: float = 28.0):
        self.state = PretermState(cervical_length=cervical_length, ga_weeks=ga)
        self.on_progesterone = False
        self.tick_count = 0

    def start_progesterone(self):
        self.on_progesterone = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_progesterone:
            st.cervical_length = min(35, st.cervical_length + 0.1)
        else:
            st.cervical_length = max(0, st.cervical_length - 0.3)
        shortening = max(0, 1 - st.cervical_length / 35)
        st.cervical_z = Z_CERVICAL * (1 - shortening * 0.5)
        g2 = gamma_sq(st.cervical_z, Z_CERVICAL)
        st.contractions = g2 > 0.1
        return {"disease": "Preterm", "ga": st.ga_weeks, "cl_mm": st.cervical_length,
                "contractions": st.contractions, "gamma_sq": g2}

# ============================================================================
# 6. GESTATIONAL DM
# ============================================================================
@dataclass
class GDMState:
    receptor_z: float = Z_HPG
    fasting_glucose: float = 95.0  # mg/dL
    ogtt_2h: float = 140.0  # mg/dL
    hba1c: float = 5.5

class GDMModel:
    def __init__(self, severity: float = 0.4):
        self.state = GDMState()
        self.severity = severity
        self.on_insulin = False
        self.tick_count = 0

    def start_insulin(self):
        self.on_insulin = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        sev = self.severity * (0.5 if self.on_insulin else 1.0)
        st.receptor_z = Z_HPG * (1 + sev * 2)
        g2 = gamma_sq(st.receptor_z, Z_HPG)
        st.fasting_glucose = 85 + g2 * 50
        st.ogtt_2h = 120 + g2 * 80
        st.hba1c = 5.0 + g2 * 2
        return {"disease": "GDM", "fasting": st.fasting_glucose,
                "ogtt_2h": st.ogtt_2h, "hba1c": st.hba1c, "gamma_sq": g2}

# ============================================================================
# 7. OVARIAN CANCER
# ============================================================================
@dataclass
class OvarianCancerState:
    ovarian_z: float = Z_OVARIAN
    ca125: float = 20.0      # U/mL (normal <35)
    figo_stage: str = "I"
    tumor_cm: float = 3.0

class OvarianCancerModel:
    def __init__(self, stage: str = "II", growth_rate: float = 0.01):
        self.state = OvarianCancerState(figo_stage=stage)
        self.growth_rate = growth_rate
        self.on_chemo = False
        self.tick_count = 0

    def start_chemo(self):
        self.on_chemo = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        if self.on_chemo:
            st.tumor_cm = max(0.1, st.tumor_cm * 0.98)
        else:
            st.tumor_cm = min(20, st.tumor_cm * (1 + self.growth_rate))
        z_transform = st.tumor_cm / 3
        st.ovarian_z = Z_OVARIAN * (1 + z_transform)
        g2 = gamma_sq(st.ovarian_z, Z_OVARIAN)
        st.ca125 = 20 + g2 * 500
        if g2 < 0.1: st.figo_stage = "I"
        elif g2 < 0.3: st.figo_stage = "II"
        elif g2 < 0.5: st.figo_stage = "III"
        else: st.figo_stage = "IV"
        return {"disease": "Ovarian Cancer", "ca125": st.ca125,
                "stage": st.figo_stage, "size_cm": st.tumor_cm, "gamma_sq": g2}

# ============================================================================
# 8. MENOPAUSE
# ============================================================================
@dataclass
class MenopauseState:
    ovarian_z: float = Z_OVARIAN
    fsh: float = 8.0           # mIU/mL (pre: 3–10, post: 30–120)
    estradiol: float = 100.0   # pg/mL (pre: 30–400, post: <30)
    mrs: float = 0.0           # Menopause Rating Scale
    hot_flashes: int = 0       # per day

class MenopauseModel:
    def __init__(self, years_post: float = 2.0):
        self.state = MenopauseState()
        self.years_post = years_post
        self.on_hrt = False
        self.tick_count = 0

    def start_hrt(self):
        self.on_hrt = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        depletion = min(0.9, self.years_post * 0.1)
        if self.on_hrt:
            depletion *= 0.3
        st.ovarian_z = Z_OVARIAN * (1 + depletion * 5)
        g2 = gamma_sq(st.ovarian_z, Z_OVARIAN)
        st.fsh = 8 + g2 * 100
        st.estradiol = max(5, 100 * (1 - g2))
        st.mrs = min(44, g2 * 55)
        st.hot_flashes = int(g2 * 15)
        return {"disease": "Menopause", "fsh": st.fsh, "estradiol": st.estradiol,
                "mrs": st.mrs, "hot_flashes": st.hot_flashes, "gamma_sq": g2}

# ============================================================================
# 9. AMNIOTIC FLUID EMBOLISM (AFE)
# ============================================================================
@dataclass
class AFEState:
    systemic_z: float = Z_PLACENTAL
    bp_systolic: float = 120.0
    dic: bool = False
    sofa: int = 0
    mortality_risk: float = 0.0

class AFEModel:
    def __init__(self, severity: float = 0.8):
        self.state = AFEState()
        self.severity = severity
        self.resuscitation = False
        self.tick_count = 0

    def start_resuscitation(self):
        self.resuscitation = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        sev = self.severity * (0.5 if self.resuscitation else 1.0)
        # Foreign particulate matter enters maternal circulation
        st.systemic_z = Z_PLACENTAL * (1 + sev * 10)
        g2 = gamma_sq(st.systemic_z, Z_PLACENTAL)
        st.bp_systolic = max(40, 120 - g2 * 80)
        st.dic = g2 > 0.4
        st.sofa = min(24, int(g2 * 30))
        st.mortality_risk = min(1.0, g2 * 1.2)
        return {"disease": "AFE", "bp": st.bp_systolic, "dic": st.dic,
                "sofa": st.sofa, "mortality": st.mortality_risk, "gamma_sq": g2}

# ============================================================================
# 10. POSTPARTUM HEMORRHAGE
# ============================================================================
@dataclass
class PPHState:
    uterine_z: float = Z_UTERINE
    ebl_ml: float = 500.0     # Estimated blood loss
    hr: float = 80.0
    bp_systolic: float = 120.0
    shock_index: float = 0.67

class PPHModel:
    def __init__(self, cause: str = "atony", severity: float = 0.6):
        self.state = PPHState()
        self.cause = cause
        self.severity = severity
        self.uterotonics = False
        self.tick_count = 0

    def give_uterotonics(self):
        self.uterotonics = True

    def tick(self) -> Dict:
        self.tick_count += 1
        st = self.state
        sev = self.severity
        if self.uterotonics and self.cause == "atony":
            sev *= 0.3
        # Atony = muscle Z failure
        st.uterine_z = Z_UTERINE * (1 - sev * 0.5)
        g2 = gamma_sq(st.uterine_z, Z_UTERINE)
        st.ebl_ml = 300 + g2 * 2000
        st.hr = 80 + g2 * 60
        st.bp_systolic = max(60, 120 - g2 * 60)
        st.shock_index = st.hr / max(st.bp_systolic, 1)
        return {"disease": "PPH", "ebl_ml": st.ebl_ml, "cause": self.cause,
                "shock_index": st.shock_index, "gamma_sq": g2}

# ============================================================================
# UNIFIED ENGINE
# ============================================================================
class ClinicalObstetricsEngine:
    DISEASE_CLASSES = {
        "preeclampsia": PreeclampsiaModel, "pcos": PCOSModel,
        "endometriosis": EndometriosisModel, "fibroids": FibroidModel,
        "preterm": PretermBirthModel, "gdm": GDMModel,
        "ovarian_cancer": OvarianCancerModel, "menopause": MenopauseModel,
        "afe": AFEModel, "pph": PPHModel,
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
        results["reproductive_reserve"] = max(0.0, 1.0 - total_g2)
        return results
