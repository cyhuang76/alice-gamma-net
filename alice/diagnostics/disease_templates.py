# -*- coding: utf-8 -*-
"""disease_templates.py — 120 Disease Γ Signature Templates
============================================================

Each disease is a 12-dimensional Γ fingerprint representing the expected
organ-system impedance mismatch pattern.

Modelling principles:
    - Primary organs: Γ > 0.5 (severe mismatch)
    - Secondary organs: Γ = 0.1–0.3 (moderate effect)
    - Unaffected organs: Γ < 0.05 (normal)
    - Systemic diseases (SLE, Sepsis): multiple elevated Γ values
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from alice.diagnostics.lab_mapping import ORGAN_LIST


# ============================================================================
# 1. Disease Template Data Class
# ============================================================================

@dataclass
class DiseaseTemplate:
    """A single disease Γ signature template.

    Parameters
    ----------
    disease_id : str
        Unique identifier (e.g. "mi_acute").
    specialty : str
        Clinical specialty (e.g. "cardiology").
    display_name : str
        Bilingual display name (中文 English).
    gamma_signature : dict[str, float]
        Expected Γ value for each of the 12 organ systems.
    primary_organs : list[str]
        Organs primarily affected by this disease.
    suggested_tests : list[str]
        Recommended follow-up tests to confirm/exclude.
    key_labs : list[str]
        Most discriminating lab values.
    """

    disease_id: str
    specialty: str
    display_name: str
    gamma_signature: Dict[str, float] = field(default_factory=dict)
    primary_organs: List[str] = field(default_factory=list)
    suggested_tests: List[str] = field(default_factory=list)
    key_labs: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Ensure all 12 organs have a value (default 0.0)
        for organ in ORGAN_LIST:
            self.gamma_signature.setdefault(organ, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disease_id": self.disease_id,
            "specialty": self.specialty,
            "display_name": self.display_name,
            "gamma_signature": self.gamma_signature,
            "primary_organs": self.primary_organs,
            "suggested_tests": self.suggested_tests,
            "key_labs": self.key_labs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiseaseTemplate":
        return cls(
            disease_id=d["disease_id"],
            specialty=d["specialty"],
            display_name=d["display_name"],
            gamma_signature=d.get("gamma_signature", {}),
            primary_organs=d.get("primary_organs", []),
            suggested_tests=d.get("suggested_tests", []),
            key_labs=d.get("key_labs", []),
        )


# ============================================================================
# 2. Load / Save
# ============================================================================

_DEFAULT_JSON_PATH = os.path.join(os.path.dirname(__file__), "disease_templates.json")


def load_disease_templates(path: Optional[str] = None) -> List[DiseaseTemplate]:
    """Load disease templates from JSON file.

    Parameters
    ----------
    path : str or None
        Path to JSON file.  Defaults to the bundled disease_templates.json.

    Returns
    -------
    list[DiseaseTemplate]
    """
    path = path or _DEFAULT_JSON_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [DiseaseTemplate.from_dict(d) for d in data]


def save_disease_templates(templates: List[DiseaseTemplate], path: Optional[str] = None) -> None:
    """Save disease templates to JSON file."""
    path = path or _DEFAULT_JSON_PATH
    data = [t.to_dict() for t in templates]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================================
# 3. Built-in Templates (120 diseases, 12 specialties × 10 each)
# ============================================================================

def build_all_templates() -> List[DiseaseTemplate]:
    """Construct the full 120-disease template set programmatically.

    These Γ signatures are based on published clinical literature
    mapping disease patterns to laboratory derangement profiles.
    """
    templates: List[DiseaseTemplate] = []

    def _t(did: str, spec: str, name: str,
           sig: Dict[str, float], pri: List[str],
           tests: List[str], labs: List[str]) -> None:
        templates.append(DiseaseTemplate(
            disease_id=did, specialty=spec, display_name=name,
            gamma_signature=sig, primary_organs=pri,
            suggested_tests=tests, key_labs=labs,
        ))

    # ==== CARDIOLOGY (10) ====
    _t("mi_acute", "cardiology", "急性心肌梗塞 Acute MI",
       {"cardiac": 0.85, "vascular": 0.35, "immune": 0.20, "pulmonary": 0.15, "heme": 0.05},
       ["cardiac", "vascular"],
       ["ECG", "Serial Troponin", "Coronary angiography", "Echocardiogram"],
       ["Troponin", "CK_MB", "BNP"])

    _t("heart_failure", "cardiology", "心衰竭 Heart Failure",
       {"cardiac": 0.70, "pulmonary": 0.30, "renal": 0.25, "hepatic": 0.15, "vascular": 0.20},
       ["cardiac", "pulmonary"],
       ["Echocardiogram", "BNP", "Chest X-ray", "Cardiac MRI"],
       ["BNP", "Troponin", "Na", "Cr"])

    _t("atrial_fibrillation", "cardiology", "心房顫動 Atrial Fibrillation",
       {"cardiac": 0.55, "vascular": 0.15, "pulmonary": 0.10},
       ["cardiac"],
       ["ECG", "Holter monitor", "Echocardiogram", "TSH"],
       ["BNP", "TSH", "K"])

    _t("hypertension", "cardiology", "高血壓 Hypertension",
       {"cardiac": 0.40, "vascular": 0.45, "renal": 0.25, "neuro": 0.10},
       ["cardiac", "vascular"],
       ["24h ABPM", "Echocardiogram", "Renal US", "Fundoscopy"],
       ["Cr", "K", "Na", "LDL"])

    _t("aortic_stenosis", "cardiology", "主動脈瓣狹窄 Aortic Stenosis",
       {"cardiac": 0.65, "pulmonary": 0.10},
       ["cardiac"],
       ["Echocardiogram", "Cardiac catheterisation"],
       ["BNP", "Troponin"])

    _t("cardiomyopathy", "cardiology", "心肌病變 Cardiomyopathy",
       {"cardiac": 0.60, "pulmonary": 0.20, "hepatic": 0.10},
       ["cardiac"],
       ["Echocardiogram", "Cardiac MRI", "Genetic testing", "Endomyocardial biopsy"],
       ["BNP", "Troponin", "CK_MB"])

    _t("pericarditis", "cardiology", "心包膜炎 Pericarditis",
       {"cardiac": 0.50, "immune": 0.25, "pulmonary": 0.10},
       ["cardiac", "immune"],
       ["ECG", "Echocardiogram", "CRP", "CT chest"],
       ["CRP", "ESR", "Troponin"])

    _t("pulmonary_htn", "cardiology", "肺高壓 Pulmonary Hypertension",
       {"cardiac": 0.45, "pulmonary": 0.55, "vascular": 0.20},
       ["pulmonary", "cardiac"],
       ["Right heart catheterisation", "CT pulmonary angiogram", "V/Q scan"],
       ["BNP", "D_Dimer"])

    _t("endocarditis", "cardiology", "心內膜炎 Endocarditis",
       {"cardiac": 0.60, "immune": 0.50, "vascular": 0.15, "heme": 0.10},
       ["cardiac", "immune"],
       ["Blood cultures ×3", "Echocardiogram (TEE)", "Duke criteria"],
       ["CRP", "ESR", "WBC", "PCT"])

    _t("aortic_dissection", "cardiology", "主動脈剝離 Aortic Dissection",
       {"cardiac": 0.70, "vascular": 0.80, "renal": 0.20, "neuro": 0.15},
       ["cardiac", "vascular"],
       ["CT aorta with contrast", "TEE", "D-Dimer"],
       ["D_Dimer", "Troponin", "Lactate"])

    # ==== PULMONOLOGY (10) ====
    _t("pneumonia", "pulmonology", "肺炎 Pneumonia",
       {"pulmonary": 0.65, "immune": 0.40, "cardiac": 0.10},
       ["pulmonary", "immune"],
       ["Chest X-ray", "Sputum culture", "Blood cultures", "CT chest"],
       ["WBC", "CRP", "PCT"])

    _t("copd_exacerbation", "pulmonology", "COPD 急性惡化 COPD Exacerbation",
       {"pulmonary": 0.60, "cardiac": 0.20, "immune": 0.15},
       ["pulmonary"],
       ["Spirometry", "ABG", "Chest X-ray"],
       ["WBC", "CRP", "CO2"])

    _t("asthma_acute", "pulmonology", "急性氣喘 Acute Asthma",
       {"pulmonary": 0.55, "immune": 0.20},
       ["pulmonary"],
       ["Spirometry", "Peak flow", "ABG", "Chest X-ray"],
       ["WBC", "CRP"])

    _t("pulmonary_embolism", "pulmonology", "肺栓塞 Pulmonary Embolism",
       {"pulmonary": 0.70, "cardiac": 0.30, "vascular": 0.25, "heme": 0.10},
       ["pulmonary", "vascular"],
       ["CT pulmonary angiogram", "V/Q scan", "Lower limb US"],
       ["D_Dimer", "Troponin", "BNP"])

    _t("pleural_effusion", "pulmonology", "肋膜積液 Pleural Effusion",
       {"pulmonary": 0.50, "hepatic": 0.15, "cardiac": 0.15, "immune": 0.10},
       ["pulmonary"],
       ["Chest X-ray", "Thoracentesis (Light criteria)", "CT chest"],
       ["Albumin", "Total_Protein", "WBC"])

    _t("ards", "pulmonology", "急性呼吸窘迫症 ARDS",
       {"pulmonary": 0.85, "cardiac": 0.25, "immune": 0.35, "renal": 0.15},
       ["pulmonary", "immune"],
       ["ABG", "Chest X-ray/CT", "Echocardiogram"],
       ["WBC", "CRP", "PCT", "Lactate"])

    _t("lung_cancer", "pulmonology", "肺癌 Lung Cancer",
       {"pulmonary": 0.55, "immune": 0.20, "heme": 0.15},
       ["pulmonary"],
       ["CT chest", "PET-CT", "Bronchoscopy + biopsy", "Tumour markers"],
       ["Hb", "Plt", "ALP", "Ca"])

    _t("pneumothorax", "pulmonology", "氣胸 Pneumothorax",
       {"pulmonary": 0.60, "cardiac": 0.15},
       ["pulmonary"],
       ["Chest X-ray", "CT chest"],
       [])

    _t("tuberculosis", "pulmonology", "肺結核 Tuberculosis",
       {"pulmonary": 0.50, "immune": 0.40, "hepatic": 0.05},
       ["pulmonary", "immune"],
       ["Chest X-ray", "Sputum AFB ×3", "Quantiferon/PPD", "CT chest"],
       ["WBC", "ESR", "CRP", "Albumin"])

    _t("sleep_apnea", "pulmonology", "睡眠呼吸中止 Sleep Apnea",
       {"pulmonary": 0.35, "cardiac": 0.20, "endocrine": 0.15, "neuro": 0.10},
       ["pulmonary"],
       ["Polysomnography", "Epworth Sleepiness Scale"],
       ["Glucose", "HbA1c", "TSH"])

    # ==== GASTROENTEROLOGY (10) ====
    _t("hepatitis_acute", "gastroenterology", "急性肝炎 Acute Hepatitis",
       {"hepatic": 0.75, "immune": 0.25, "GI": 0.15, "heme": 0.10},
       ["hepatic", "immune"],
       ["Hepatitis A/B/C serology", "Liver US", "Ceruloplasmin", "Autoimmune markers"],
       ["AST", "ALT", "Bil_total", "Albumin", "INR"])

    _t("cirrhosis", "gastroenterology", "肝硬化 Cirrhosis",
       {"hepatic": 0.70, "GI": 0.30, "heme": 0.25, "renal": 0.20, "immune": 0.15},
       ["hepatic", "GI"],
       ["Liver US", "FibroScan", "Upper GI endoscopy", "AFP"],
       ["Albumin", "INR", "Bil_total", "Plt", "AST", "ALT"])

    _t("pancreatitis_acute", "gastroenterology", "急性胰臟炎 Acute Pancreatitis",
       {"GI": 0.75, "hepatic": 0.15, "immune": 0.20, "renal": 0.10},
       ["GI"],
       ["CT abdomen", "MRCP", "Ranson criteria"],
       ["Amylase", "Lipase", "CRP", "Ca", "WBC"])

    _t("gerd", "gastroenterology", "胃食道逆流 GERD",
       {"GI": 0.30},
       ["GI"],
       ["Upper GI endoscopy", "24h pH monitoring", "PPI trial"],
       [])

    _t("ibd_crohn", "gastroenterology", "克隆氏症 Crohn's Disease",
       {"GI": 0.55, "immune": 0.40, "heme": 0.20, "bone": 0.10},
       ["GI", "immune"],
       ["Colonoscopy + biopsy", "MR enterography", "Faecal calprotectin"],
       ["CRP", "ESR", "Albumin", "Hb", "Plt"])

    _t("ibd_uc", "gastroenterology", "潰瘍性大腸炎 Ulcerative Colitis",
       {"GI": 0.50, "immune": 0.35, "heme": 0.20},
       ["GI", "immune"],
       ["Colonoscopy + biopsy", "Faecal calprotectin", "Stool cultures"],
       ["CRP", "ESR", "Albumin", "Hb"])

    _t("gi_bleed_upper", "gastroenterology", "上消化道出血 Upper GI Bleed",
       {"GI": 0.60, "heme": 0.45, "cardiac": 0.15},
       ["GI", "heme"],
       ["Upper GI endoscopy", "Angiography if massive"],
       ["Hb", "Hct", "BUN", "INR", "Plt"])

    _t("cholecystitis", "gastroenterology", "膽囊炎 Cholecystitis",
       {"GI": 0.45, "hepatic": 0.30, "immune": 0.20},
       ["GI", "hepatic"],
       ["RUQ ultrasound", "HIDA scan", "MRCP"],
       ["WBC", "CRP", "ALP", "GGT", "Bil_total"])

    _t("hcc", "gastroenterology", "肝細胞癌 Hepatocellular Carcinoma",
       {"hepatic": 0.65, "GI": 0.20, "immune": 0.15, "heme": 0.10},
       ["hepatic"],
       ["Liver US", "CT/MRI triphasic", "AFP", "Liver biopsy"],
       ["AST", "ALT", "Albumin", "Bil_total", "Plt"])

    _t("celiac", "gastroenterology", "乳糜瀉 Celiac Disease",
       {"GI": 0.45, "immune": 0.30, "heme": 0.20, "bone": 0.15},
       ["GI", "immune"],
       ["tTG-IgA", "Duodenal biopsy", "HLA-DQ2/DQ8"],
       ["Hb", "Ferritin", "Vit_D", "Folate", "Albumin"])

    # ==== NEPHROLOGY (10) ====
    _t("aki", "nephrology", "急性腎損傷 AKI",
       {"renal": 0.80, "cardiac": 0.15, "hepatic": 0.10},
       ["renal"],
       ["Renal US", "Urinalysis", "FENa", "Renal biopsy if indicated"],
       ["Cr", "BUN", "K", "Na", "CO2"])

    _t("ckd", "nephrology", "慢性腎病 CKD",
       {"renal": 0.60, "vascular": 0.20, "heme": 0.25, "bone": 0.20, "endocrine": 0.10},
       ["renal"],
       ["eGFR trend", "UACR", "Renal US", "PTH/Ca/PO4"],
       ["Cr", "BUN", "K", "Ca", "Hb", "Albumin"])

    _t("nephrotic_syndrome", "nephrology", "腎病症候群 Nephrotic Syndrome",
       {"renal": 0.65, "vascular": 0.15, "immune": 0.20, "heme": 0.10},
       ["renal", "immune"],
       ["Urinalysis (proteinuria)", "Serum albumin", "Lipid panel", "Renal biopsy"],
       ["Albumin", "TC", "Cr", "Total_Protein"])

    _t("glomerulonephritis", "nephrology", "腎絲球腎炎 Glomerulonephritis",
       {"renal": 0.70, "immune": 0.35, "heme": 0.15},
       ["renal", "immune"],
       ["Urinalysis", "Complement C3/C4", "ANA/ANCA", "Renal biopsy"],
       ["Cr", "BUN", "Hb", "Albumin", "CRP"])

    _t("rta", "nephrology", "腎小管酸中毒 Renal Tubular Acidosis",
       {"renal": 0.50, "bone": 0.20, "neuro": 0.10},
       ["renal"],
       ["ABG", "Urine pH", "Urine anion gap", "Serum K/Ca"],
       ["K", "CO2", "Ca", "Cl", "Na"])

    _t("pyelonephritis", "nephrology", "腎盂腎炎 Pyelonephritis",
       {"renal": 0.55, "immune": 0.40},
       ["renal", "immune"],
       ["Urinalysis", "Urine culture", "Renal US/CT"],
       ["WBC", "CRP", "PCT", "Cr"])

    _t("kidney_stone", "nephrology", "腎結石 Nephrolithiasis",
       {"renal": 0.40, "bone": 0.10},
       ["renal"],
       ["CT KUB (non-contrast)", "Urinalysis", "24h urine collection"],
       ["Ca", "Uric_Acid", "Cr"])

    _t("hyponatremia", "nephrology", "低血鈉 Hyponatremia",
       {"renal": 0.45, "neuro": 0.35, "endocrine": 0.15},
       ["renal", "neuro"],
       ["Serum osmolality", "Urine osmolality", "Urine Na", "TSH", "Cortisol"],
       ["Na", "Glucose", "TSH", "Cr"])

    _t("hyperkalemia", "nephrology", "高血鉀 Hyperkalemia",
       {"renal": 0.50, "cardiac": 0.35},
       ["renal", "cardiac"],
       ["ECG", "ABG", "Urine K", "Renal function"],
       ["K", "Cr", "CO2"])

    _t("dialysis_need", "nephrology", "需透析 End-Stage Renal Disease",
       {"renal": 0.90, "heme": 0.35, "bone": 0.30, "vascular": 0.25, "cardiac": 0.20},
       ["renal"],
       ["Dialysis access planning", "PTH", "Iron studies", "Hepatitis panel"],
       ["Cr", "BUN", "K", "Ca", "Hb", "Ferritin"])

    # ==== ENDOCRINOLOGY (10) ====
    _t("dka", "endocrinology", "糖尿病酮酸中毒 DKA",
       {"endocrine": 0.80, "renal": 0.35, "neuro": 0.30, "vascular": 0.20, "cardiac": 0.15},
       ["endocrine", "renal", "neuro"],
       ["ABG", "Serum ketones", "Urine ketones", "Serum osmolality"],
       ["Glucose", "K", "Na", "CO2", "BUN", "Cr"])

    _t("dm_type2", "endocrinology", "第二型糖尿病 Type 2 DM",
       {"endocrine": 0.50, "vascular": 0.30, "renal": 0.20, "neuro": 0.15},
       ["endocrine", "vascular"],
       ["HbA1c", "Fasting glucose", "OGTT", "Lipid panel", "UACR"],
       ["HbA1c", "Glucose", "LDL", "Cr", "TC"])

    _t("hyperthyroid", "endocrinology", "甲狀腺亢進 Hyperthyroidism",
       {"endocrine": 0.65, "cardiac": 0.30, "bone": 0.15},
       ["endocrine", "cardiac"],
       ["TSH", "FT4", "FT3", "TSH-receptor Ab", "Thyroid US", "Uptake scan"],
       ["TSH", "FT4", "FT3"])

    _t("hypothyroid", "endocrinology", "甲狀腺低下 Hypothyroidism",
       {"endocrine": 0.55, "cardiac": 0.15, "heme": 0.15, "neuro": 0.10},
       ["endocrine"],
       ["TSH", "FT4", "Anti-TPO", "Lipid panel"],
       ["TSH", "FT4", "TC", "Hb"])

    _t("addison", "endocrinology", "愛迪生氏症 Adrenal Insufficiency",
       {"endocrine": 0.70, "renal": 0.25, "immune": 0.15, "neuro": 0.10},
       ["endocrine"],
       ["AM cortisol", "ACTH stimulation test", "ACTH level", "21-hydroxylase Ab"],
       ["Na", "K", "Glucose", "Ca"])

    _t("cushing", "endocrinology", "乙萬氏症候群 Cushing Syndrome",
       {"endocrine": 0.60, "vascular": 0.25, "bone": 0.20, "immune": 0.15},
       ["endocrine"],
       ["24h urine cortisol", "Dexamethasone suppression", "Midnight salivary cortisol"],
       ["Glucose", "K", "Na", "WBC"])

    _t("pheo", "endocrinology", "嗜鉻細胞瘤 Pheochromocytoma",
       {"endocrine": 0.55, "cardiac": 0.40, "vascular": 0.35},
       ["endocrine", "cardiac"],
       ["Plasma free metanephrines", "24h urine catecholamines", "CT/MRI adrenal"],
       ["Glucose", "Ca"])

    _t("hypoparathyroid", "endocrinology", "副甲狀腺低下 Hypoparathyroidism",
       {"endocrine": 0.45, "bone": 0.30, "neuro": 0.25},
       ["endocrine", "bone"],
       ["PTH", "Ca", "PO4", "Mg", "Vitamin D"],
       ["Ca", "Vit_D", "ALP"])

    _t("hypoglycemia", "endocrinology", "低血糖 Hypoglycemia",
       {"endocrine": 0.50, "neuro": 0.40, "cardiac": 0.15},
       ["endocrine", "neuro"],
       ["72h fast", "C-peptide", "Insulin level", "Cortisol"],
       ["Glucose", "K"])

    _t("metabolic_syndrome", "endocrinology", "代謝症候群 Metabolic Syndrome",
       {"endocrine": 0.40, "vascular": 0.40, "hepatic": 0.15, "renal": 0.10},
       ["endocrine", "vascular"],
       ["Fasting glucose", "Lipid panel", "Waist circumference", "BP"],
       ["HbA1c", "Glucose", "TG", "HDL", "LDL"])

    # ==== IMMUNOLOGY / RHEUMATOLOGY (10) ====
    _t("sepsis", "immunology", "敗血症 Sepsis",
       {"immune": 0.80, "cardiac": 0.30, "renal": 0.35, "pulmonary": 0.25, "hepatic": 0.20, "heme": 0.15},
       ["immune", "renal", "cardiac"],
       ["Blood cultures ×2", "Lactate", "PCT", "qSOFA/SOFA score"],
       ["WBC", "CRP", "PCT", "Lactate", "Cr", "Plt"])

    _t("sle", "immunology", "全身性紅斑狼瘡 SLE",
       {"immune": 0.65, "renal": 0.35, "heme": 0.30, "bone": 0.20, "vascular": 0.15, "neuro": 0.15},
       ["immune", "renal", "heme"],
       ["ANA", "Anti-dsDNA", "C3/C4", "Urinalysis", "CBC"],
       ["WBC", "Hb", "Plt", "Cr", "ESR", "CRP"])

    _t("rheumatoid_arthritis", "immunology", "類風濕性關節炎 RA",
       {"immune": 0.55, "bone": 0.45, "heme": 0.15},
       ["immune", "bone"],
       ["RF", "Anti-CCP", "X-ray hands/feet", "Joint US"],
       ["CRP", "ESR", "Hb", "Plt"])

    _t("anaphylaxis", "immunology", "過敏性休克 Anaphylaxis",
       {"immune": 0.85, "cardiac": 0.35, "pulmonary": 0.30},
       ["immune", "cardiac", "pulmonary"],
       ["Tryptase", "IgE", "Allergy testing (after recovery)"],
       ["WBC", "CRP"])

    _t("hiv_aids", "immunology", "HIV/AIDS",
       {"immune": 0.75, "heme": 0.20, "pulmonary": 0.15, "neuro": 0.15, "GI": 0.10},
       ["immune"],
       ["HIV Ag/Ab", "CD4 count", "Viral load", "Resistance testing"],
       ["WBC", "Lymphocytes", "Hb", "Plt"])

    _t("vasculitis", "immunology", "血管炎 Vasculitis",
       {"immune": 0.60, "vascular": 0.45, "renal": 0.30, "pulmonary": 0.15},
       ["immune", "vascular"],
       ["ANCA (c-ANCA/p-ANCA)", "Biopsy", "Angiography"],
       ["CRP", "ESR", "Cr", "Hb", "WBC"])

    _t("gout", "immunology", "痛風 Gout",
       {"immune": 0.40, "bone": 0.50, "renal": 0.20},
       ["bone", "immune"],
       ["Synovial fluid analysis (crystals)", "Joint X-ray", "Dual-energy CT"],
       ["Uric_Acid", "CRP", "Cr", "WBC"])

    _t("sjogren", "immunology", "乾燥症 Sjögren Syndrome",
       {"immune": 0.50, "GI": 0.15, "neuro": 0.10},
       ["immune"],
       ["Anti-SSA/SSB", "Schirmer test", "Salivary gland biopsy"],
       ["ESR", "CRP", "Total_Protein"])

    _t("sarcoidosis", "immunology", "類肉瘤病 Sarcoidosis",
       {"immune": 0.50, "pulmonary": 0.40, "bone": 0.15, "hepatic": 0.10, "endocrine": 0.10},
       ["immune", "pulmonary"],
       ["Chest X-ray/CT", "ACE level", "Biopsy", "Ca"],
       ["Ca", "ALP", "CRP", "ESR"])

    _t("drug_allergy", "immunology", "藥物過敏 Drug Allergy",
       {"immune": 0.55, "hepatic": 0.20, "heme": 0.15, "renal": 0.10},
       ["immune"],
       ["Drug provocation testing", "Patch testing", "IgE specific"],
       ["WBC", "CRP", "AST", "ALT", "Cr"])

    # ==== HEMATOLOGY (10) ====
    _t("iron_deficiency_anemia", "hematology", "缺鐵性貧血 Iron Deficiency Anemia",
       {"heme": 0.65, "GI": 0.15, "cardiac": 0.10},
       ["heme"],
       ["Iron studies", "Ferritin", "TIBC", "Stool OB", "Upper/lower GI scope"],
       ["Hb", "MCV", "Ferritin", "RBC"])

    _t("b12_folate_anemia", "hematology", "巨球性貧血 Megaloblastic Anemia",
       {"heme": 0.55, "neuro": 0.25, "GI": 0.10},
       ["heme", "neuro"],
       ["B12", "Folate", "MMA", "Homocysteine", "Peripheral smear"],
       ["Hb", "MCV", "Vit_B12", "Folate"])

    _t("hemolytic_anemia", "hematology", "溶血性貧血 Hemolytic Anemia",
       {"heme": 0.70, "hepatic": 0.20, "immune": 0.15},
       ["heme"],
       ["Reticulocyte count", "Haptoglobin", "LDH", "Direct Coombs", "Peripheral smear"],
       ["Hb", "Bil_total", "Bil_direct", "Hct"])

    _t("thrombocytopenia", "hematology", "血小板低下 Thrombocytopenia",
       {"heme": 0.55, "immune": 0.25, "hepatic": 0.10},
       ["heme", "immune"],
       ["Peripheral smear", "Bone marrow biopsy", "Anti-platelet Ab"],
       ["Plt", "WBC", "Hb"])

    _t("dic", "hematology", "瀰漫性血管內凝血 DIC",
       {"heme": 0.80, "hepatic": 0.25, "immune": 0.30, "vascular": 0.20},
       ["heme", "immune"],
       ["D-Dimer", "Fibrinogen", "PT/INR", "Peripheral smear"],
       ["Plt", "INR", "D_Dimer", "Hb", "Bil_total"])

    _t("leukemia_acute", "hematology", "急性白血病 Acute Leukemia",
       {"heme": 0.85, "immune": 0.45, "hepatic": 0.15, "bone": 0.10},
       ["heme", "immune"],
       ["Peripheral smear", "Bone marrow biopsy", "Flow cytometry", "Cytogenetics"],
       ["WBC", "Hb", "Plt", "Neutrophils", "Lymphocytes"])

    _t("lymphoma", "hematology", "淋巴瘤 Lymphoma",
       {"heme": 0.55, "immune": 0.45, "hepatic": 0.15},
       ["heme", "immune"],
       ["Lymph node biopsy", "PET-CT", "Bone marrow biopsy", "LDH"],
       ["WBC", "Hb", "ESR", "CRP", "Albumin"])

    _t("myelodysplastic", "hematology", "骨髓化生不良 MDS",
       {"heme": 0.60, "immune": 0.20},
       ["heme"],
       ["Peripheral smear", "Bone marrow biopsy", "Cytogenetics", "Flow cytometry"],
       ["Hb", "WBC", "Plt", "MCV"])

    _t("polycythemia_vera", "hematology", "真性紅血球增多 Polycythemia Vera",
       {"heme": 0.55, "vascular": 0.25, "cardiac": 0.10},
       ["heme", "vascular"],
       ["JAK2 V617F", "EPO level", "Bone marrow biopsy"],
       ["Hb", "Hct", "RBC", "Plt", "WBC"])

    _t("thalassemia", "hematology", "地中海型貧血 Thalassemia",
       {"heme": 0.50, "hepatic": 0.15, "bone": 0.10},
       ["heme"],
       ["Hb electrophoresis", "Genetic testing", "Iron studies", "Peripheral smear"],
       ["Hb", "MCV", "RBC", "Ferritin"])

    # ==== ORTHOPEDICS / BONE (10) ====
    _t("osteoporosis", "orthopedics", "骨質疏鬆 Osteoporosis",
       {"bone": 0.55, "endocrine": 0.15},
       ["bone"],
       ["DEXA scan", "FRAX score", "Spine X-ray"],
       ["Ca", "Vit_D", "ALP", "Cr"])

    _t("osteoarthritis", "orthopedics", "骨關節炎 Osteoarthritis",
       {"bone": 0.40, "immune": 0.10},
       ["bone"],
       ["X-ray affected joint", "MRI if needed"],
       ["CRP", "ESR", "Uric_Acid"])

    _t("fracture_pathologic", "orthopedics", "病理性骨折 Pathologic Fracture",
       {"bone": 0.65, "heme": 0.15, "immune": 0.10},
       ["bone"],
       ["X-ray", "CT/MRI", "Bone biopsy", "Bone scan"],
       ["Ca", "ALP", "Vit_D", "Total_Protein"])

    _t("osteomyelitis", "orthopedics", "骨髓炎 Osteomyelitis",
       {"bone": 0.60, "immune": 0.45, "heme": 0.10},
       ["bone", "immune"],
       ["MRI", "Blood cultures", "Bone biopsy", "Bone scan"],
       ["WBC", "CRP", "ESR", "PCT"])

    _t("gout_tophaceous", "orthopedics", "痛風石 Tophaceous Gout",
       {"bone": 0.55, "renal": 0.25, "immune": 0.20},
       ["bone", "renal"],
       ["Synovial fluid", "Dual-energy CT", "X-ray"],
       ["Uric_Acid", "Cr", "CRP"])

    _t("paget_disease", "orthopedics", "乳頭狀骨病 Paget Disease of Bone",
       {"bone": 0.50, "cardiac": 0.10},
       ["bone"],
       ["ALP", "Bone scan", "X-ray", "Bone biopsy"],
       ["ALP", "Ca"])

    _t("multiple_myeloma", "orthopedics", "多發性骨髓瘤 Multiple Myeloma",
       {"bone": 0.65, "heme": 0.40, "renal": 0.35, "immune": 0.25},
       ["bone", "heme", "renal"],
       ["SPEP/UPEP", "Bone marrow biopsy", "Skeletal survey", "Beta-2 microglobulin"],
       ["Ca", "Cr", "Total_Protein", "Hb", "Albumin"])

    _t("vitamin_d_deficiency", "orthopedics", "維生素D缺乏 Vitamin D Deficiency",
       {"bone": 0.40, "immune": 0.10, "endocrine": 0.10},
       ["bone"],
       ["25-OH Vitamin D", "PTH", "Ca", "PO4"],
       ["Vit_D", "Ca", "ALP"])

    _t("ankylosing_spondylitis", "orthopedics", "僵直性脊椎炎 Ankylosing Spondylitis",
       {"bone": 0.55, "immune": 0.35},
       ["bone", "immune"],
       ["HLA-B27", "SI joint X-ray/MRI", "CRP/ESR"],
       ["CRP", "ESR", "ALP"])

    _t("bone_metastasis", "orthopedics", "骨轉移 Bone Metastasis",
       {"bone": 0.70, "heme": 0.15, "hepatic": 0.10},
       ["bone"],
       ["Bone scan", "PET-CT", "Biopsy", "Tumour markers"],
       ["Ca", "ALP", "Hb"])

    # ==== NEUROLOGY (10) ====
    _t("stroke_ischemic", "neurology", "缺血性中風 Ischemic Stroke",
       {"neuro": 0.75, "vascular": 0.35, "cardiac": 0.20},
       ["neuro", "vascular"],
       ["CT head (non-contrast)", "CT angiogram", "MRI/DWI", "Carotid US"],
       ["Glucose", "Na", "Hb", "INR", "LDL"])

    _t("stroke_hemorrhagic", "neurology", "出血性中風 Hemorrhagic Stroke",
       {"neuro": 0.80, "vascular": 0.30, "cardiac": 0.15, "heme": 0.10},
       ["neuro", "vascular"],
       ["CT head (non-contrast)", "CT angiogram", "MRI", "Coagulation panel"],
       ["INR", "Plt", "Glucose", "Na"])

    _t("meningitis", "neurology", "腦膜炎 Meningitis",
       {"neuro": 0.70, "immune": 0.50},
       ["neuro", "immune"],
       ["LP (CSF analysis)", "Blood cultures", "CT head (before LP)", "PCR panel"],
       ["WBC", "CRP", "PCT", "Na", "Glucose"])

    _t("seizure_disorder", "neurology", "癲癇 Seizure Disorder",
       {"neuro": 0.55, "endocrine": 0.10},
       ["neuro"],
       ["EEG", "MRI brain", "Metabolic panel", "Drug levels"],
       ["Na", "Ca", "Glucose", "NH3"])

    _t("hepatic_encephalopathy", "neurology", "肝性腦病 Hepatic Encephalopathy",
       {"neuro": 0.60, "hepatic": 0.55, "GI": 0.15},
       ["neuro", "hepatic"],
       ["NH3 level", "EEG", "CT head (rule out other)", "Lactulose trial"],
       ["NH3", "Albumin", "INR", "Bil_total"])

    _t("parkinson", "neurology", "帕金森氏症 Parkinson Disease",
       {"neuro": 0.45},
       ["neuro"],
       ["Clinical exam (UPDRS)", "DaTscan", "MRI brain"],
       [])

    _t("ms", "neurology", "多發性硬化 Multiple Sclerosis",
       {"neuro": 0.55, "immune": 0.30},
       ["neuro", "immune"],
       ["MRI brain/spine", "LP (oligoclonal bands)", "VEP"],
       ["WBC", "CRP", "Vit_D"])

    _t("dementia", "neurology", "失智症 Dementia",
       {"neuro": 0.40, "vascular": 0.10},
       ["neuro"],
       ["MMSE/MoCA", "MRI brain", "FDG-PET", "CSF biomarkers"],
       ["TSH", "Vit_B12", "Folate", "Na"])

    _t("metabolic_encephalopathy", "neurology", "代謝性腦病 Metabolic Encephalopathy",
       {"neuro": 0.65, "renal": 0.30, "hepatic": 0.25, "endocrine": 0.20},
       ["neuro", "renal"],
       ["Metabolic panel", "ABG", "NH3", "CT head", "EEG"],
       ["Na", "Ca", "Glucose", "NH3", "BUN", "Lactate"])

    _t("guillain_barre", "neurology", "乖乖巴利症 Guillain-Barré Syndrome",
       {"neuro": 0.60, "immune": 0.35, "pulmonary": 0.15},
       ["neuro", "immune"],
       ["LP (albuminocytologic dissociation)", "NCS/EMG", "Respiratory monitoring"],
       ["WBC", "CRP", "Total_Protein"])

    # ==== ONCOLOGY (10) ====
    _t("tumor_lysis", "oncology", "腫瘤溶解症候群 Tumor Lysis Syndrome",
       {"renal": 0.60, "cardiac": 0.25, "bone": 0.20, "heme": 0.15},
       ["renal", "cardiac"],
       ["Uric acid q6h", "K/Ca/PO4 q6h", "LDH", "ECG"],
       ["Uric_Acid", "K", "Ca", "Cr", "Lactate"])

    _t("febrile_neutropenia", "oncology", "發燒性嗜中性球低下 Febrile Neutropenia",
       {"immune": 0.75, "heme": 0.50},
       ["immune", "heme"],
       ["Blood cultures ×2", "Urine culture", "Chest X-ray", "MASCC score"],
       ["WBC", "Neutrophils", "CRP", "PCT"])

    _t("paraneoplastic", "oncology", "副腫瘤症候群 Paraneoplastic Syndrome",
       {"immune": 0.40, "endocrine": 0.35, "neuro": 0.30, "heme": 0.15},
       ["immune", "endocrine", "neuro"],
       ["CT chest/abdomen", "Anti-neuronal Ab", "Tumour markers"],
       ["Ca", "Na", "Glucose", "Hb"])

    _t("hypercalcemia_malignancy", "oncology", "惡性高血鈣 Hypercalcemia of Malignancy",
       {"bone": 0.45, "renal": 0.30, "neuro": 0.25, "cardiac": 0.15},
       ["bone", "renal"],
       ["PTHrP", "PTH", "PET-CT", "Skeletal survey"],
       ["Ca", "Albumin", "Cr", "ALP"])

    _t("cancer_cachexia", "oncology", "癌症惡病質 Cancer Cachexia",
       {"GI": 0.35, "heme": 0.30, "immune": 0.25, "hepatic": 0.15},
       ["GI", "heme"],
       ["Nutritional assessment", "Prealbumin", "CRP", "Body composition"],
       ["Albumin", "Total_Protein", "Hb", "CRP"])

    _t("chemo_hepatotoxicity", "oncology", "化療肝毒性 Chemotherapy Hepatotoxicity",
       {"hepatic": 0.60, "immune": 0.15, "GI": 0.10},
       ["hepatic"],
       ["LFTs trend", "Drug level", "Liver US", "Liver biopsy if severe"],
       ["AST", "ALT", "Bil_total", "ALP", "INR"])

    _t("chemo_nephrotoxicity", "oncology", "化療腎毒性 Chemotherapy Nephrotoxicity",
       {"renal": 0.55, "heme": 0.10},
       ["renal"],
       ["Cr trend", "Urinalysis", "Renal US", "Drug levels"],
       ["Cr", "BUN", "K", "Ca"])

    _t("anemia_of_chronic_disease", "oncology", "慢性病貧血 Anemia of Chronic Disease",
       {"heme": 0.45, "immune": 0.25},
       ["heme", "immune"],
       ["Iron studies", "Reticulocyte count", "EPO level"],
       ["Hb", "Ferritin", "CRP", "MCV"])

    _t("dvt", "oncology", "深部靜脈栓塞 DVT",
       {"vascular": 0.55, "heme": 0.15, "pulmonary": 0.10},
       ["vascular"],
       ["Compression US", "D-Dimer", "CT venogram if needed"],
       ["D_Dimer", "Plt", "INR"])

    _t("siadh", "oncology", "SIADH 抗利尿激素異常分泌",
       {"renal": 0.40, "neuro": 0.35, "endocrine": 0.25},
       ["renal", "neuro"],
       ["Serum osmolality", "Urine osmolality", "Urine Na", "TSH/Cortisol", "CT chest"],
       ["Na", "Glucose", "Cr"])

    # ==== OBSTETRICS / REPRODUCTIVE (10) ====
    _t("preeclampsia", "obstetrics", "子癇前症 Preeclampsia",
       {"repro": 0.60, "renal": 0.40, "hepatic": 0.30, "vascular": 0.25, "heme": 0.15},
       ["repro", "renal"],
       ["24h urine protein", "Spot UPCR", "LFTs", "CBC with smear", "Uric acid"],
       ["Cr", "AST", "ALT", "Plt", "Uric_Acid"])

    _t("hellp", "obstetrics", "HELLP 症候群 HELLP Syndrome",
       {"repro": 0.55, "hepatic": 0.65, "heme": 0.55, "renal": 0.25},
       ["hepatic", "heme", "repro"],
       ["CBC with smear", "LFTs", "LDH", "Haptoglobin", "Coagulation panel"],
       ["AST", "ALT", "Plt", "Hb", "Bil_total", "INR"])

    _t("gestational_dm", "obstetrics", "妊娠糖尿病 Gestational DM",
       {"repro": 0.35, "endocrine": 0.45, "vascular": 0.15},
       ["endocrine", "repro"],
       ["OGTT 75g", "Fasting glucose", "HbA1c"],
       ["Glucose", "HbA1c"])

    _t("ectopic_pregnancy", "obstetrics", "子宮外孕 Ectopic Pregnancy",
       {"repro": 0.70, "heme": 0.25},
       ["repro"],
       ["Quantitative βhCG (serial)", "Transvaginal US", "Progesterone"],
       ["Hb", "Hct"])

    _t("hyperemesis", "obstetrics", "妊娠劇吐 Hyperemesis Gravidarum",
       {"repro": 0.40, "endocrine": 0.20, "renal": 0.25, "hepatic": 0.10},
       ["repro", "renal"],
       ["Metabolic panel", "Ketonuria", "TSH/FT4", "Liver US"],
       ["Na", "K", "Cl", "TSH", "AST"])

    _t("pcos", "obstetrics", "多囊性卵巢 PCOS",
       {"repro": 0.55, "endocrine": 0.35, "vascular": 0.10},
       ["repro", "endocrine"],
       ["LH/FSH ratio", "Testosterone", "DHEA-S", "Pelvic US", "OGTT"],
       ["FSH", "LH", "Testosterone", "Glucose", "HbA1c"])

    _t("ovarian_failure", "obstetrics", "卵巢早衰 Premature Ovarian Failure",
       {"repro": 0.65, "bone": 0.20, "endocrine": 0.15},
       ["repro"],
       ["FSH (×2)", "AMH", "Karyotype", "Anti-adrenal Ab", "DEXA"],
       ["FSH", "LH"])

    _t("menorrhagia", "obstetrics", "經血過多 Menorrhagia",
       {"repro": 0.40, "heme": 0.35},
       ["repro", "heme"],
       ["Pelvic US", "Endometrial biopsy", "Hysteroscopy", "Coagulation panel"],
       ["Hb", "Ferritin", "Plt", "INR"])

    _t("male_hypogonadism", "obstetrics", "男性性腺低下 Male Hypogonadism",
       {"repro": 0.55, "endocrine": 0.20, "bone": 0.15},
       ["repro", "endocrine"],
       ["Testosterone (AM ×2)", "LH/FSH", "Prolactin", "DEXA"],
       ["Testosterone", "FSH", "LH"])

    _t("infertility_eval", "obstetrics", "不孕評估 Infertility Evaluation",
       {"repro": 0.45, "endocrine": 0.20},
       ["repro"],
       ["FSH/LH/E2 day 3", "AMH", "Semen analysis", "HSG", "Pelvic US"],
       ["FSH", "LH", "Testosterone"])

    # ==== DERMATOLOGY (not in 12-organ Z, map to nearest) (5 + 5 from ENT/Ophtho) ====
    _t("psoriasis", "dermatology", "乾癬 Psoriasis",
       {"immune": 0.45, "bone": 0.20, "hepatic": 0.10},
       ["immune", "bone"],
       ["Skin biopsy", "CRP/ESR", "HLA typing", "Lipid panel"],
       ["CRP", "ESR", "Uric_Acid", "AST"])

    _t("pemphigus", "dermatology", "天疱瘡 Pemphigus",
       {"immune": 0.55, "heme": 0.10},
       ["immune"],
       ["Skin biopsy (DIF)", "Desmoglein Ab", "Anti-BP180/230"],
       ["WBC", "CRP", "Albumin"])

    _t("drug_reaction_severe", "dermatology", "嚴重藥物反應 SJS/TEN",
       {"immune": 0.70, "hepatic": 0.30, "renal": 0.25, "heme": 0.20},
       ["immune", "hepatic"],
       ["Skin biopsy", "Drug causality assessment (ALDEN)", "Renal/LFTs"],
       ["WBC", "CRP", "AST", "ALT", "Cr"])

    _t("urticaria_chronic", "dermatology", "慢性蕁麻疹 Chronic Urticaria",
       {"immune": 0.35},
       ["immune"],
       ["CBC", "TSH", "Anti-FcεRI", "IgE", "CRP"],
       ["WBC", "CRP", "TSH"])

    _t("dermatitis_atopic", "dermatology", "異位性皮膚炎 Atopic Dermatitis",
       {"immune": 0.30},
       ["immune"],
       ["IgE", "Allergy testing", "Skin biopsy if atypical"],
       ["WBC", "CRP"])

    # ==== ENT (5) ====
    _t("sinusitis_chronic", "ENT", "慢性鼻竇炎 Chronic Sinusitis",
       {"immune": 0.30, "pulmonary": 0.10},
       ["immune"],
       ["CT sinuses", "Nasal endoscopy", "Allergy testing"],
       ["WBC", "CRP", "ESR"])

    _t("sudden_hearing_loss", "ENT", "突發性聽力喪失 Sudden Sensorineural Hearing Loss",
       {"neuro": 0.40, "vascular": 0.15, "immune": 0.15},
       ["neuro"],
       ["Audiometry", "MRI IAC (rule out acoustic neuroma)", "CBC/BMP"],
       ["WBC", "CRP", "Glucose"])

    _t("peritonsillar_abscess", "ENT", "扁桃腺周圍膿瘍 Peritonsillar Abscess",
       {"immune": 0.45, "GI": 0.10},
       ["immune"],
       ["CT neck with contrast", "Needle aspiration", "Culture"],
       ["WBC", "CRP", "PCT"])

    _t("epistaxis_severe", "ENT", "嚴重鼻出血 Severe Epistaxis",
       {"heme": 0.30, "vascular": 0.15},
       ["heme"],
       ["CBC", "Coagulation panel", "Nasal endoscopy"],
       ["Hb", "Plt", "INR"])

    _t("laryngeal_cancer", "ENT", "喉癌 Laryngeal Cancer",
       {"immune": 0.20, "pulmonary": 0.15, "heme": 0.10},
       ["pulmonary"],
       ["Laryngoscopy + biopsy", "CT neck", "PET-CT"],
       ["Hb", "Albumin", "CRP"])

    # ==== OPHTHALMOLOGY (5) ====
    _t("diabetic_retinopathy", "ophthalmology", "糖尿病視網膜病變 Diabetic Retinopathy",
       {"vascular": 0.40, "endocrine": 0.35, "neuro": 0.10},
       ["vascular", "endocrine"],
       ["Fundoscopy", "OCT", "Fluorescein angiography", "HbA1c"],
       ["HbA1c", "Glucose", "Cr", "LDL"])

    _t("glaucoma", "ophthalmology", "青光眼 Glaucoma",
       {"neuro": 0.30, "vascular": 0.15},
       ["neuro"],
       ["IOP measurement", "Visual field", "OCT RNFL", "Gonioscopy"],
       [])

    _t("optic_neuritis", "ophthalmology", "視神經炎 Optic Neuritis",
       {"neuro": 0.50, "immune": 0.35},
       ["neuro", "immune"],
       ["MRI brain/orbits", "VEP", "AQP4-Ab", "MOG-Ab"],
       ["WBC", "CRP", "ESR"])

    _t("retinal_vein_occlusion", "ophthalmology", "視網膜靜脈阻塞 Retinal Vein Occlusion",
       {"vascular": 0.45, "heme": 0.15, "cardiac": 0.10},
       ["vascular"],
       ["Fundoscopy", "OCT", "FFA", "Hypertension/DM workup"],
       ["Glucose", "HbA1c", "LDL", "Hb"])

    _t("thyroid_eye_disease", "ophthalmology", "甲狀腺眼病 Thyroid Eye Disease",
       {"endocrine": 0.50, "immune": 0.30, "neuro": 0.15},
       ["endocrine", "immune"],
       ["TSH", "FT4", "TSH-receptor Ab", "CT/MRI orbits", "Visual field"],
       ["TSH", "FT4", "FT3", "CRP"])

    return templates

