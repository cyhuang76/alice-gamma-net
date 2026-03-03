# -*- coding: utf-8 -*-
"""alice.diagnostics — Lab-Γ Diagnostic Engine

Maps laboratory values to organ-system impedances and performs
disease-template matching via the minimum reflection action.

Physics:  Γ = (Z_patient − Z_normal) / (Z_patient + Z_normal)
Energy:   Γ² + T = 1  at every organ, every evaluation
"""

from alice.diagnostics.lab_mapping import LabItem, LabMapper, ORGAN_SYSTEMS
from alice.diagnostics.gamma_engine import GammaEngine, PatientGammaVector
from alice.diagnostics.disease_templates import DiseaseTemplate, load_disease_templates

__all__ = [
    "LabItem",
    "LabMapper",
    "ORGAN_SYSTEMS",
    "GammaEngine",
    "PatientGammaVector",
    "DiseaseTemplate",
    "load_disease_templates",
]
