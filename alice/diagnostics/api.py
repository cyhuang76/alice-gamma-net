# -*- coding: utf-8 -*-
"""api.py — Lab-Γ Diagnostic REST API
=======================================

FastAPI service exposing the Lab-Γ engine as a stateless REST API.

Endpoints:
    POST /diagnose          Lab values → differential diagnosis (top-N)
    POST /gamma             Lab values → raw 12-D Γ vector + Z breakdown
    POST /feedback          Physician confirms / rejects / corrects
    GET  /templates         List all 125 disease templates
    GET  /templates/{id}    Single template detail
    GET  /organs            Organ system impedance reference table
    GET  /labs              Supported laboratory items catalogue
    GET  /feedback/stats    Feedback statistics
    GET  /health            API health check

Physics:
    Γ = (Z_patient − Z_normal) / (Z_patient + Z_normal)
    C1: Γ² + T = 1  at every organ
    C2: ΔW = −η · Γ_error · x_pre · x_post  (feedback)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from alice.diagnostics.lab_mapping import (
    LAB_CATALOGUE,
    ORGAN_LIST,
    ORGAN_SYSTEMS,
    LabMapper,
)
from alice.diagnostics.gamma_engine import GammaEngine, PatientGammaVector
from alice.diagnostics.disease_templates import load_disease_templates
from alice.diagnostics.feedback import FeedbackEngine, HebbianUpdater


# ============================================================================
# Pydantic Models
# ============================================================================

class DiagnoseRequest(BaseModel):
    """Input: laboratory values as name→value dict."""
    lab_values: Dict[str, float] = Field(
        ..., description="Lab name → measured value (e.g. {'AST': 480, 'ALT': 520})"
    )
    top_n: int = Field(default=5, ge=1, le=20, description="Number of candidates")


class OrganGamma(BaseModel):
    organ: str
    z_patient: float
    z_normal: float
    gamma: float
    transmission: float  # T = 1 - Γ²
    c1_ok: bool          # Γ² + T == 1


class GammaResponse(BaseModel):
    """12-D organ Γ vector with full physics breakdown."""
    organs: List[OrganGamma]
    total_gamma_squared: float
    health_index: float
    timestamp: float


class CandidateResponse(BaseModel):
    rank: int
    disease_id: str
    display_name: str
    specialty: str
    confidence: float
    distance: float
    severity: str
    primary_deviations: List[str]
    suggested_tests: List[str]
    organ_matches: List[Dict[str, Any]]


class DiagnoseResponse(BaseModel):
    gamma: GammaResponse
    candidates: List[CandidateResponse]
    engine_weights: Dict[str, float]


class FeedbackRequest(BaseModel):
    """Physician feedback on a diagnosis."""
    feedback_type: str = Field(
        ..., description="confirm | reject | correct"
    )
    lab_values: Dict[str, float]
    disease_id: str = Field(..., description="Disease template ID")
    apply_now: bool = Field(default=True, description="Apply Hebbian update immediately")


class FeedbackResponse(BaseModel):
    feedback_type: str
    disease_id: str
    weight_deltas: Dict[str, float]
    applied: bool
    stats: Dict[str, int]


class TemplateResponse(BaseModel):
    disease_id: str
    specialty: str
    display_name: str
    gamma_signature: Dict[str, float]
    primary_organs: List[str]
    suggested_tests: List[str]
    key_labs: List[str]


class LabItemResponse(BaseModel):
    name: str
    unit: str
    ref_low: Optional[float]
    ref_high: Optional[float]
    organs: List[str]


# ============================================================================
# Application Factory
# ============================================================================

def create_app() -> FastAPI:
    """Create and configure the Lab-Γ API application."""

    app = FastAPI(
        title="Lab-Γ Diagnostic Engine API",
        description=(
            "Impedance-based differential diagnosis.\n\n"
            "Physics: Γ = (Z_patient − Z_normal) / (Z_patient + Z_normal)\n"
            "C1: Γ² + T = 1 (energy conservation)\n"
            "C2: ΔW = −η · Γ_error · x_pre · x_post (Hebbian feedback)"
        ),
        version="3.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Shared engine state
    templates = load_disease_templates()
    engine = GammaEngine(templates=templates)
    feedback_engine = FeedbackEngine(engine)

    # Store on app for test access
    app.state.engine = engine
    app.state.feedback_engine = feedback_engine

    # ── Helper ──────────────────────────────────────────────────────────

    def _build_gamma_response(
        gamma_vec: PatientGammaVector,
        z_patient: Dict[str, float],
    ) -> GammaResponse:
        organs = []
        mapper = engine.mapper
        for organ in ORGAN_LIST:
            g = gamma_vec[organ]
            z_n = mapper.organ_z_normal[organ]
            z_p = z_patient[organ]
            t = 1.0 - g ** 2
            organs.append(OrganGamma(
                organ=organ,
                z_patient=round(z_p, 2),
                z_normal=z_n,
                gamma=round(g, 6),
                transmission=round(t, 6),
                c1_ok=abs(g ** 2 + t - 1.0) < 1e-10,
            ))
        return GammaResponse(
            organs=organs,
            total_gamma_squared=round(gamma_vec.total_gamma_squared, 6),
            health_index=round(gamma_vec.health_index, 6),
            timestamp=time.time(),
        )

    # ── Endpoints ───────────────────────────────────────────────────────

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "engine": "Lab-Γ Diagnostic Engine v3.0",
            "templates": len(engine.templates),
            "organs": len(ORGAN_LIST),
            "labs": len(LAB_CATALOGUE),  # dict[name, LabItem]
            "feedback_records": len(feedback_engine.records),
        }

    @app.post("/diagnose", response_model=DiagnoseResponse)
    def diagnose(req: DiagnoseRequest):
        """Lab values → Γ vector → differential diagnosis."""
        gamma_vec, z_patient, candidates = engine.diagnose_detailed(
            req.lab_values, top_n=req.top_n
        )
        gamma_resp = _build_gamma_response(gamma_vec, z_patient)

        cands = []
        for c in candidates:
            cands.append(CandidateResponse(
                rank=c.rank,
                disease_id=c.disease_id,
                display_name=c.display_name,
                specialty=c.specialty,
                confidence=round(c.confidence, 6),
                distance=round(c.distance, 6),
                severity=c.severity,
                primary_deviations=c.primary_deviations,
                suggested_tests=c.suggested_tests,
                organ_matches=[
                    {"organ": o, "gamma_patient": round(gp, 4), "gamma_template": round(gt, 4)}
                    for o, gp, gt in c.organ_matches
                ],
            ))

        return DiagnoseResponse(
            gamma=gamma_resp,
            candidates=cands,
            engine_weights=dict(engine.organ_weights),
        )

    @app.post("/gamma", response_model=GammaResponse)
    def compute_gamma(req: DiagnoseRequest):
        """Lab values → raw 12-D Γ vector (no disease matching)."""
        gamma_vec, z_patient, _ = engine.lab_to_gamma_detailed(req.lab_values)
        return _build_gamma_response(gamma_vec, z_patient)

    @app.post("/feedback", response_model=FeedbackResponse)
    def submit_feedback(req: FeedbackRequest):
        """Physician confirms / rejects / corrects a diagnosis."""
        ft = req.feedback_type.lower()
        if ft == "confirm":
            rec = feedback_engine.record_confirm(req.lab_values, req.disease_id)
        elif ft == "reject":
            rec = feedback_engine.record_reject(req.lab_values, req.disease_id)
        elif ft == "correct":
            rec = feedback_engine.record_correct(req.lab_values, req.disease_id)
        else:
            raise HTTPException(400, f"Unknown feedback type: {req.feedback_type}")

        if req.apply_now:
            feedback_engine.apply_single(rec)

        return FeedbackResponse(
            feedback_type=rec.feedback_type,
            disease_id=rec.target_disease_id,
            weight_deltas=rec.weight_deltas,
            applied=rec.applied,
            stats=feedback_engine.stats(),
        )

    @app.get("/feedback/stats")
    def feedback_stats():
        return feedback_engine.stats()

    @app.get("/feedback/drift")
    def feedback_drift():
        return {
            "weight_offsets": feedback_engine.weight_offsets,
            "current_weights": dict(engine.organ_weights),
            "report": feedback_engine.weight_drift_report(),
        }

    @app.get("/templates", response_model=List[TemplateResponse])
    def list_templates():
        return [
            TemplateResponse(**t.to_dict())
            for t in engine.templates
        ]

    @app.get("/templates/{disease_id}", response_model=TemplateResponse)
    def get_template(disease_id: str):
        for t in engine.templates:
            if t.disease_id == disease_id:
                return TemplateResponse(**t.to_dict())
        raise HTTPException(404, f"Template not found: {disease_id}")

    @app.get("/organs")
    def list_organs():
        return {
            "organs": [
                {"name": organ, "z_normal": z, "index": i}
                for i, (organ, z) in enumerate(ORGAN_SYSTEMS.items())
            ]
        }

    @app.get("/labs")
    def list_labs():
        items = []
        for lab in LAB_CATALOGUE.values():
            items.append(LabItemResponse(
                name=lab.name,
                unit=lab.unit,
                ref_low=lab.ref_low,
                ref_high=lab.ref_high,
                organs=list(lab.organ_weights.keys()),
            ))
        return {"labs": items, "total": len(items)}

    return app


# ============================================================================
# Standalone Runner
# ============================================================================

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("alice.diagnostics.api:app", host="0.0.0.0", port=8420, reload=True)
