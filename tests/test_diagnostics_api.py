# -*- coding: utf-8 -*-
"""tests/test_diagnostics_api.py — Phase 3 REST API Tests
============================================================

Tests the Lab-Γ Diagnostic Engine REST API using httpx TestClient.
Covers all endpoints, physics verification, and feedback flow.
"""

import pytest
from fastapi.testclient import TestClient

from alice.diagnostics.api import create_app
from alice.diagnostics.lab_mapping import ORGAN_LIST, LAB_CATALOGUE


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def client():
    """Fresh API app per test (isolated state)."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def hepatitis_labs():
    return {"AST": 480, "ALT": 520, "Bil_total": 3.2, "Alb": 2.8, "INR": 1.8}


@pytest.fixture
def mi_labs():
    return {"Troponin_I": 15.0, "CK_MB": 80.0, "BNP": 1200.0, "WBC": 14.0}


# ============================================================================
# A. Health Check
# ============================================================================

class TestHealth:

    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["templates"] >= 100
        assert data["organs"] == 12
        assert data["labs"] == len(LAB_CATALOGUE)


# ============================================================================
# B. Diagnose Endpoint
# ============================================================================

class TestDiagnose:

    def test_diagnose_hepatitis(self, client, hepatitis_labs):
        r = client.post("/diagnose", json={"lab_values": hepatitis_labs})
        assert r.status_code == 200
        data = r.json()

        # Γ vector present
        assert len(data["gamma"]["organs"]) == 12
        assert data["gamma"]["health_index"] < 1.0
        assert data["gamma"]["total_gamma_squared"] > 0.0

        # Candidates present
        assert len(data["candidates"]) > 0
        top = data["candidates"][0]
        assert top["rank"] == 1
        assert top["confidence"] > 0.0
        assert top["specialty"] != ""
        assert top["severity"] in ["mild", "moderate", "severe", "critical"]

    def test_diagnose_mi(self, client, mi_labs):
        r = client.post("/diagnose", json={"lab_values": mi_labs, "top_n": 3})
        assert r.status_code == 200
        data = r.json()
        assert len(data["candidates"]) <= 3

    def test_diagnose_empty_labs(self, client):
        r = client.post("/diagnose", json={"lab_values": {}})
        assert r.status_code == 200
        data = r.json()
        # All Γ should be ~0
        for organ_data in data["gamma"]["organs"]:
            assert abs(organ_data["gamma"]) < 1e-6

    def test_c1_in_response(self, client, hepatitis_labs):
        """C1 energy conservation: Γ² + T = 1 for all organs."""
        r = client.post("/diagnose", json={"lab_values": hepatitis_labs})
        data = r.json()
        for org in data["gamma"]["organs"]:
            assert org["c1_ok"], f"C1 violated at {org['organ']}"
            assert abs(org["gamma"] ** 2 + org["transmission"] - 1.0) < 1e-4

    def test_engine_weights_in_response(self, client, hepatitis_labs):
        r = client.post("/diagnose", json={"lab_values": hepatitis_labs})
        data = r.json()
        assert "engine_weights" in data
        assert len(data["engine_weights"]) == 12

    def test_top_n_validation(self, client, hepatitis_labs):
        r = client.post("/diagnose", json={"lab_values": hepatitis_labs, "top_n": 0})
        assert r.status_code == 422  # validation error

    def test_diagnose_candidate_fields(self, client, hepatitis_labs):
        r = client.post("/diagnose", json={"lab_values": hepatitis_labs})
        data = r.json()
        c = data["candidates"][0]
        assert "disease_id" in c
        assert "display_name" in c
        assert "organ_matches" in c
        assert isinstance(c["suggested_tests"], list)


# ============================================================================
# C. Gamma Endpoint
# ============================================================================

class TestGamma:

    def test_gamma_returns_12_organs(self, client, hepatitis_labs):
        r = client.post("/gamma", json={"lab_values": hepatitis_labs})
        assert r.status_code == 200
        data = r.json()
        assert len(data["organs"]) == 12

    def test_gamma_c1(self, client, mi_labs):
        r = client.post("/gamma", json={"lab_values": mi_labs})
        data = r.json()
        for org in data["organs"]:
            assert org["c1_ok"]


# ============================================================================
# D. Feedback Endpoint
# ============================================================================

class TestFeedback:

    def test_confirm(self, client, hepatitis_labs):
        r = client.post("/feedback", json={
            "feedback_type": "confirm",
            "lab_values": hepatitis_labs,
            "disease_id": "hepatitis_acute",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["feedback_type"] == "confirm"
        assert data["applied"]
        assert data["stats"]["confirm"] == 1
        assert data["stats"]["applied"] == 1

    def test_reject(self, client, mi_labs):
        r = client.post("/feedback", json={
            "feedback_type": "reject",
            "lab_values": mi_labs,
            "disease_id": "pneumonia_bacterial",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["feedback_type"] == "reject"

    def test_correct(self, client, mi_labs):
        r = client.post("/feedback", json={
            "feedback_type": "correct",
            "lab_values": mi_labs,
            "disease_id": "mi_acute",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["feedback_type"] == "correct"

    def test_invalid_feedback_type(self, client, hepatitis_labs):
        r = client.post("/feedback", json={
            "feedback_type": "invalid",
            "lab_values": hepatitis_labs,
            "disease_id": "test",
        })
        assert r.status_code == 400

    def test_no_apply(self, client, hepatitis_labs):
        r = client.post("/feedback", json={
            "feedback_type": "confirm",
            "lab_values": hepatitis_labs,
            "disease_id": "hepatitis_acute",
            "apply_now": False,
        })
        data = r.json()
        assert not data["applied"]
        assert data["stats"]["pending"] == 1

    def test_weight_deltas_present(self, client, hepatitis_labs):
        r = client.post("/feedback", json={
            "feedback_type": "confirm",
            "lab_values": hepatitis_labs,
            "disease_id": "hepatitis_acute",
        })
        data = r.json()
        assert isinstance(data["weight_deltas"], dict)
        assert len(data["weight_deltas"]) == 12


class TestFeedbackStats:

    def test_stats_endpoint(self, client, hepatitis_labs):
        # First add some feedback
        client.post("/feedback", json={
            "feedback_type": "confirm",
            "lab_values": hepatitis_labs,
            "disease_id": "hepatitis_acute",
        })
        r = client.get("/feedback/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1

    def test_drift_endpoint(self, client, hepatitis_labs):
        client.post("/feedback", json={
            "feedback_type": "confirm",
            "lab_values": hepatitis_labs,
            "disease_id": "hepatitis_acute",
        })
        r = client.get("/feedback/drift")
        assert r.status_code == 200
        data = r.json()
        assert "weight_offsets" in data
        assert "current_weights" in data
        assert "report" in data


# ============================================================================
# E. Templates Endpoint
# ============================================================================

class TestTemplates:

    def test_list_templates(self, client):
        r = client.get("/templates")
        assert r.status_code == 200
        data = r.json()
        assert len(data) >= 100  # 125 templates
        t = data[0]
        assert "disease_id" in t
        assert "gamma_signature" in t
        assert len(t["gamma_signature"]) == 12

    def test_get_template_by_id(self, client):
        r = client.get("/templates/mi_acute")
        assert r.status_code == 200
        data = r.json()
        assert data["disease_id"] == "mi_acute"
        assert data["specialty"] == "cardiology"

    def test_template_not_found(self, client):
        r = client.get("/templates/nonexistent_xyz")
        assert r.status_code == 404


# ============================================================================
# F. Reference Endpoints
# ============================================================================

class TestReference:

    def test_organs(self, client):
        r = client.get("/organs")
        assert r.status_code == 200
        data = r.json()
        assert len(data["organs"]) == 12

    def test_labs(self, client):
        r = client.get("/labs")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == len(LAB_CATALOGUE)
        lab = data["labs"][0]
        assert "name" in lab
        assert "unit" in lab
        assert "ref_low" in lab


# ============================================================================
# G. Integration: Full Workflow
# ============================================================================

class TestFullWorkflow:

    def test_diagnose_then_confirm_changes_weights(self, client, hepatitis_labs):
        """Full pipeline: diagnose → confirm → weights change → re-diagnose."""
        # Step 1: Initial diagnosis
        r1 = client.post("/diagnose", json={"lab_values": hepatitis_labs})
        w_before = r1.json()["engine_weights"]

        # Step 2: Confirm top result
        top_id = r1.json()["candidates"][0]["disease_id"]
        client.post("/feedback", json={
            "feedback_type": "confirm",
            "lab_values": hepatitis_labs,
            "disease_id": top_id,
        })

        # Step 3: Re-diagnose — weights should have changed
        r2 = client.post("/diagnose", json={"lab_values": hepatitis_labs})
        w_after = r2.json()["engine_weights"]

        # At least one weight changed
        any_changed = any(
            abs(w_after[o] - w_before[o]) > 1e-12
            for o in ORGAN_LIST
        )
        assert any_changed, "Weights did not change after feedback"

    def test_c1_holds_throughout_workflow(self, client, hepatitis_labs, mi_labs):
        """C1 must hold after all operations."""
        # Diagnose
        client.post("/diagnose", json={"lab_values": hepatitis_labs})
        # Multiple feedbacks
        for _ in range(5):
            client.post("/feedback", json={
                "feedback_type": "confirm",
                "lab_values": hepatitis_labs,
                "disease_id": "hepatitis_acute",
            })
            client.post("/feedback", json={
                "feedback_type": "reject",
                "lab_values": mi_labs,
                "disease_id": "pneumonia_bacterial",
            })

        # Verify C1 still holds
        r = client.post("/diagnose", json={"lab_values": hepatitis_labs})
        for org in r.json()["gamma"]["organs"]:
            assert org["c1_ok"], f"C1 violated at {org['organ']}"
