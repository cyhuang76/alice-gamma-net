# -*- coding: utf-8 -*-
"""alice.diagnostics — Lab-Γ Diagnostic Engine

Maps laboratory values to organ-system impedances and performs
disease-template matching via the minimum reflection action.

Physics:  Γ = (Z_patient − Z_normal) / (Z_patient + Z_normal)
Energy:   Γ² + T = 1  at every organ, every evaluation
Feedback: ΔW = −η · Γ_error · x_pre · x_post  (C2 Hebbian closed loop)
API:      FastAPI REST + Streamlit Web UI (Phase 3)
"""

from alice.diagnostics.lab_mapping import LabItem, LabMapper, ORGAN_SYSTEMS
from alice.diagnostics.gamma_engine import GammaEngine, PatientGammaVector
from alice.diagnostics.disease_templates import DiseaseTemplate, load_disease_templates
from alice.diagnostics.feedback import (
    FeedbackEngine,
    FeedbackRecord,
    FeedbackType,
    HebbianUpdater,
)

__all__ = [
    "LabItem",
    "LabMapper",
    "ORGAN_SYSTEMS",
    "GammaEngine",
    "PatientGammaVector",
    "DiseaseTemplate",
    "load_disease_templates",
    "FeedbackEngine",
    "FeedbackRecord",
    "FeedbackType",
    "HebbianUpdater",
]
