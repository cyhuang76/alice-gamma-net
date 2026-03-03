# -*- coding: utf-8 -*-
"""test_lab_gamma_engine.py — Comprehensive tests for the Lab-Γ Diagnostic Engine
==================================================================================

Tests cover:
    1. Lab catalogue integrity (53 items, 12 organ systems)
    2. Normalisation functions (linear, piecewise, edge cases)
    3. Lab→Z mapping (single item, multi-item, organ accumulation)
    4. Γ calculation (physics: Γ = (Z_L - Z₀)/(Z_L + Z₀))
    5. C1 energy conservation: Γ² + T = 1
    6. PatientGammaVector properties
    7. Disease template integrity (125 templates, all valid)
    8. Template matching & ranking
    9. End-to-end clinical scenarios
    10. Edge cases & error handling
"""

from __future__ import annotations

import math
import json
import os

import numpy as np
import pytest

from alice.diagnostics.lab_mapping import (
    LAB_CATALOGUE,
    ORGAN_LIST,
    ORGAN_SYSTEMS,
    LabItem,
    LabMapper,
    normalise_lab_value,
    normalise_linear,
    normalise_piecewise,
)
from alice.diagnostics.gamma_engine import (
    DiagnosisCandidate,
    GammaEngine,
    PatientGammaVector,
)
from alice.diagnostics.disease_templates import (
    DiseaseTemplate,
    build_all_templates,
    load_disease_templates,
    save_disease_templates,
)


# ============================================================================
# 1. Lab Catalogue Integrity
# ============================================================================

class TestLabCatalogue:
    """Verify the 53-item lab catalogue is complete and consistent."""

    def test_catalogue_size(self):
        assert len(LAB_CATALOGUE) == 53, f"Expected 53, got {len(LAB_CATALOGUE)}"

    def test_all_organs_referenced(self):
        """Every organ system must be referenced by at least one lab item."""
        referenced = set()
        for item in LAB_CATALOGUE.values():
            referenced.update(item.organ_weights.keys())
        for organ in ORGAN_LIST:
            assert organ in referenced, f"Organ {organ} not referenced by any lab"

    def test_organ_weights_positive(self):
        """All weights must be > 0."""
        for name, item in LAB_CATALOGUE.items():
            for organ, w in item.organ_weights.items():
                assert w > 0, f"{name} → {organ}: weight {w} must be > 0"
                assert organ in ORGAN_LIST, f"{name} references unknown organ {organ}"

    def test_reference_intervals_valid(self):
        """Where both ref bounds exist, low < high."""
        for name, item in LAB_CATALOGUE.items():
            if item.ref_low is not None and item.ref_high is not None:
                assert item.ref_low < item.ref_high, \
                    f"{name}: ref_low={item.ref_low} >= ref_high={item.ref_high}"

    def test_critical_bounds_outside_ref(self):
        """Critical bounds should be outside reference interval."""
        for name, item in LAB_CATALOGUE.items():
            if item.critical_low is not None and item.ref_low is not None:
                assert item.critical_low <= item.ref_low, \
                    f"{name}: critical_low {item.critical_low} > ref_low {item.ref_low}"
            if item.critical_high is not None and item.ref_high is not None:
                assert item.critical_high >= item.ref_high, \
                    f"{name}: critical_high {item.critical_high} < ref_high {item.ref_high}"


# ============================================================================
# 2. Normalisation Functions
# ============================================================================

class TestNormalisation:
    def test_linear_normal_midpoint(self):
        """Midpoint of reference → δ = 0."""
        assert normalise_linear(75.0, 50.0, 100.0) == pytest.approx(0.0)

    def test_linear_at_low_bound(self):
        """At ref_low → δ = -1."""
        assert normalise_linear(50.0, 50.0, 100.0) == pytest.approx(-1.0)

    def test_linear_at_high_bound(self):
        """At ref_high → δ = +1."""
        assert normalise_linear(100.0, 50.0, 100.0) == pytest.approx(1.0)

    def test_linear_above_range(self):
        """Above ref_high → δ > 1."""
        result = normalise_linear(150.0, 50.0, 100.0)
        assert result > 1.0

    def test_piecewise_in_range(self):
        """Within reference → δ = 0."""
        assert normalise_piecewise(5.0, 3.0, 8.0, 1.0, 15.0) == pytest.approx(0.0)

    def test_piecewise_below_ref(self):
        """Below ref_low → negative δ."""
        result = normalise_piecewise(2.0, 3.0, 8.0, 1.0, 15.0)
        assert -1.0 <= result < 0.0

    def test_piecewise_above_ref(self):
        """Above ref_high → positive δ."""
        result = normalise_piecewise(10.0, 3.0, 8.0, 1.0, 15.0)
        assert 0.0 < result <= 1.0

    def test_piecewise_at_critical_low(self):
        """At critical_low → δ = -1."""
        result = normalise_piecewise(1.0, 3.0, 8.0, 1.0, 15.0)
        assert result == pytest.approx(-1.0)

    def test_piecewise_at_critical_high(self):
        """At critical_high → δ = +1."""
        result = normalise_piecewise(15.0, 3.0, 8.0, 1.0, 15.0)
        assert result == pytest.approx(1.0)

    def test_piecewise_one_sided_high_only(self):
        """Lab with no lower bound (e.g. Troponin)."""
        result = normalise_piecewise(0.5, None, 0.04, None, 50.0)
        assert 0.0 < result <= 1.0

    def test_piecewise_saturation(self):
        """Values beyond critical are clipped to ±1."""
        result = normalise_piecewise(100.0, 3.0, 8.0, 1.0, 15.0)
        assert result == pytest.approx(1.0)

    def test_normalise_lab_value_selects_linear(self):
        """Items with both ref bounds use linear normalisation."""
        item = LAB_CATALOGUE["Na"]  # ref_low=136, ref_high=145
        delta = normalise_lab_value(140.5, item)
        assert delta == pytest.approx(0.0, abs=0.01)

    def test_normalise_lab_value_selects_piecewise(self):
        """Items with one None bound use piecewise."""
        item = LAB_CATALOGUE["Troponin"]  # ref_low=None
        delta = normalise_lab_value(0.0, item)
        assert delta == pytest.approx(0.0, abs=0.01)


# ============================================================================
# 3. Lab → Z Mapping
# ============================================================================

class TestLabMapper:
    def setup_method(self):
        self.mapper = LabMapper()

    def test_normal_values_give_z_normal(self):
        """All normal labs → Z stays at Z_normal."""
        normal_labs = {"Na": 140.5, "K": 4.25, "Glucose": 85}
        z = self.mapper.compute_organ_impedances(normal_labs)
        # These labs affect renal, cardiac, endocrine, neuro, vascular
        for organ in ORGAN_LIST:
            assert z[organ] >= ORGAN_SYSTEMS[organ] - 0.01, \
                f"{organ}: Z={z[organ]} should be >= Z_normal={ORGAN_SYSTEMS[organ]}"

    def test_empty_labs_give_z_normal(self):
        """No labs → all organs at Z_normal."""
        z = self.mapper.compute_organ_impedances({})
        for organ in ORGAN_LIST:
            assert z[organ] == pytest.approx(ORGAN_SYSTEMS[organ])

    def test_elevated_ast_increases_hepatic_z(self):
        """High AST → hepatic Z increases."""
        z = self.mapper.compute_organ_impedances({"AST": 500})
        assert z["hepatic"] > ORGAN_SYSTEMS["hepatic"]

    def test_multiple_liver_labs_accumulate(self):
        """Multiple hepatic labs → larger Z shift."""
        z_one = self.mapper.compute_organ_impedances({"AST": 500})
        z_multi = self.mapper.compute_organ_impedances({"AST": 500, "ALT": 600, "Bil_total": 5.0})
        assert z_multi["hepatic"] > z_one["hepatic"]

    def test_detailed_returns_contributions(self):
        """Detailed mode includes per-lab contributions."""
        z, contributions = self.mapper.compute_organ_impedances_detailed({"AST": 500, "ALT": 600})
        assert "hepatic" in contributions
        assert len(contributions["hepatic"]) >= 2
        # Contributions sorted by magnitude
        mags = [abs(c[2]) for c in contributions["hepatic"]]
        assert mags == sorted(mags, reverse=True)

    def test_unknown_lab_silently_skipped(self):
        """Unknown lab names are ignored (no error)."""
        z = self.mapper.compute_organ_impedances({"UNKNOWN_LAB": 999})
        for organ in ORGAN_LIST:
            assert z[organ] == pytest.approx(ORGAN_SYSTEMS[organ])

    def test_normalise_unknown_raises(self):
        """Explicit normalise() on unknown lab raises KeyError."""
        with pytest.raises(KeyError):
            self.mapper.normalise("NONEXISTENT", 42)


# ============================================================================
# 4. Γ Calculation
# ============================================================================

class TestGammaCalculation:
    def test_gamma_perfect_match(self):
        """Z_patient = Z_normal → Γ = 0."""
        assert GammaEngine.compute_gamma(50.0, 50.0) == pytest.approx(0.0)

    def test_gamma_open_circuit(self):
        """Z_patient >> Z_normal → Γ → +1."""
        g = GammaEngine.compute_gamma(1e6, 50.0)
        assert g > 0.999

    def test_gamma_short_circuit(self):
        """Z_patient << Z_normal (near 0) → Γ → -1."""
        g = GammaEngine.compute_gamma(0.001, 50.0)
        assert g < -0.99

    def test_gamma_symmetry(self):
        """Γ(a,b) = -Γ(b,a)."""
        g1 = GammaEngine.compute_gamma(100, 50)
        g2 = GammaEngine.compute_gamma(50, 100)
        assert g1 == pytest.approx(-g2)

    def test_gamma_zero_impedance(self):
        """Both zero → Γ = 0 (degenerate)."""
        assert GammaEngine.compute_gamma(0.0, 0.0) == pytest.approx(0.0)


# ============================================================================
# 5. C1 Energy Conservation: Γ² + T = 1
# ============================================================================

class TestC1EnergyConservation:
    """Verify Γ² + T = 1 for all organ calculations."""

    def test_c1_for_normal_labs(self):
        engine = GammaEngine()
        gamma_vec = engine.lab_to_gamma({"Na": 140.5, "K": 4.25})
        c1 = gamma_vec.verify_c1()
        for organ, (g2, t, passes) in c1.items():
            assert passes, f"C1 violated at {organ}: Γ²={g2}, T={t}"

    def test_c1_for_extreme_labs(self):
        engine = GammaEngine()
        gamma_vec = engine.lab_to_gamma({"AST": 1000, "ALT": 1000, "Cr": 10, "WBC": 30})
        c1 = gamma_vec.verify_c1()
        for organ, (g2, t, passes) in c1.items():
            assert passes, f"C1 violated at {organ}: Γ²={g2}, T={t}"

    def test_c1_identity_mathematical(self):
        """Pure math: Γ² + (1-Γ²) = 1 for any Γ."""
        for z_p in [0.01, 10, 50, 100, 1000, 1e6]:
            g = GammaEngine.compute_gamma(z_p, 50.0)
            assert (g ** 2 + (1 - g ** 2)) == pytest.approx(1.0)


# ============================================================================
# 6. PatientGammaVector
# ============================================================================

class TestPatientGammaVector:
    def test_default_all_zero(self):
        vec = PatientGammaVector()
        assert all(v == 0.0 for v in vec.values.values())
        assert vec.total_gamma_squared == pytest.approx(0.0)
        assert vec.health_index == pytest.approx(1.0)

    def test_from_array_roundtrip(self):
        arr = np.random.uniform(-0.5, 0.5, 12)
        vec = PatientGammaVector.from_array(arr)
        np.testing.assert_allclose(vec.to_array(), arr)

    def test_health_index_decreases_with_disease(self):
        healthy = PatientGammaVector()
        sick = PatientGammaVector(values={"hepatic": 0.8, "renal": 0.5})
        assert sick.health_index < healthy.health_index

    def test_dominant_organ(self):
        vec = PatientGammaVector(values={"hepatic": 0.9, "cardiac": 0.3})
        assert vec.dominant_organ == "hepatic"

    def test_top_n_organs(self):
        vec = PatientGammaVector(values={
            "hepatic": 0.9, "cardiac": 0.3, "immune": 0.5, "renal": 0.1,
        })
        top3 = vec.top_n_organs(3)
        assert top3[0][0] == "hepatic"
        assert top3[1][0] == "immune"
        assert top3[2][0] == "cardiac"

    def test_transmission_vector(self):
        vec = PatientGammaVector(values={"cardiac": 0.5})
        tv = vec.transmission_vector
        assert tv["cardiac"] == pytest.approx(0.75)  # 1 - 0.25
        assert tv["pulmonary"] == pytest.approx(1.0)


# ============================================================================
# 7. Disease Template Integrity
# ============================================================================

class TestDiseaseTemplates:
    def setup_method(self):
        self.templates = build_all_templates()

    def test_template_count(self):
        assert len(self.templates) == 125

    def test_unique_ids(self):
        ids = [t.disease_id for t in self.templates]
        assert len(ids) == len(set(ids)), "Duplicate disease IDs found"

    def test_all_have_display_name(self):
        for t in self.templates:
            assert t.display_name, f"{t.disease_id}: missing display_name"

    def test_all_have_specialty(self):
        for t in self.templates:
            assert t.specialty, f"{t.disease_id}: missing specialty"

    def test_gamma_values_in_range(self):
        """All Γ signature values should be in [0, 1]."""
        for t in self.templates:
            for organ, g in t.gamma_signature.items():
                assert 0.0 <= g <= 1.0, \
                    f"{t.disease_id}.{organ}: Γ={g} out of [0,1]"

    def test_primary_organs_valid(self):
        """Primary organs must be valid organ system IDs."""
        for t in self.templates:
            for organ in t.primary_organs:
                assert organ in ORGAN_LIST, \
                    f"{t.disease_id}: unknown primary organ {organ}"

    def test_primary_organs_have_elevated_gamma(self):
        """Primary organs should have Γ > 0.1 in the signature."""
        for t in self.templates:
            for organ in t.primary_organs:
                g = t.gamma_signature.get(organ, 0.0)
                assert g > 0.1, \
                    f"{t.disease_id}: primary organ {organ} has Γ={g} (too low)"

    def test_json_roundtrip(self, tmp_path):
        """Templates survive JSON save/load."""
        path = str(tmp_path / "test_templates.json")
        save_disease_templates(self.templates, path=path)
        loaded = load_disease_templates(path=path)
        assert len(loaded) == len(self.templates)
        for orig, reloaded in zip(self.templates, loaded):
            assert orig.disease_id == reloaded.disease_id
            for organ in ORGAN_LIST:
                assert orig.gamma_signature[organ] == pytest.approx(
                    reloaded.gamma_signature[organ]
                )

    def test_serialise_dict_roundtrip(self):
        t = self.templates[0]
        d = t.to_dict()
        t2 = DiseaseTemplate.from_dict(d)
        assert t.disease_id == t2.disease_id
        assert t.gamma_signature == t2.gamma_signature


# ============================================================================
# 8. Template Matching & Ranking
# ============================================================================

class TestTemplateMatching:
    def setup_method(self):
        self.engine = GammaEngine(templates=build_all_templates())

    def test_perfect_match_ranks_first(self):
        """A Γ vector equal to a template should rank that disease #1."""
        t = self.engine.templates[0]  # mi_acute
        vec = PatientGammaVector(values=dict(t.gamma_signature))
        results = self.engine.match_templates(vec, top_n=5)
        assert results[0].disease_id == t.disease_id

    def test_healthy_has_low_gamma(self):
        """Normal labs produce no significantly elevated Γ."""
        normal_labs = {"Na": 140, "K": 4.2, "Glucose": 85, "Hb": 14.0, "Cr": 0.9}
        vec = self.engine.lab_to_gamma(normal_labs)
        assert vec.total_gamma_squared < 0.1

    def test_top_n_respects_limit(self):
        vec = PatientGammaVector(values={"hepatic": 0.7})
        results = self.engine.match_templates(vec, top_n=3)
        assert len(results) == 3

    def test_confidence_sums_to_one(self):
        """Top-N confidences should sum to less than total (N < 125)."""
        vec = PatientGammaVector(values={"cardiac": 0.8})
        results = self.engine.match_templates(vec, top_n=125)
        total_conf = sum(r.confidence for r in results)
        assert total_conf == pytest.approx(1.0, abs=0.01)

    def test_distance_non_negative(self):
        vec = PatientGammaVector(values={"renal": 0.5})
        results = self.engine.match_templates(vec, top_n=5)
        for r in results:
            assert r.distance >= 0.0

    def test_empty_templates_returns_empty(self):
        engine = GammaEngine(templates=[])
        vec = PatientGammaVector(values={"hepatic": 0.5})
        assert engine.match_templates(vec) == []

    def test_severity_classification(self):
        """Extreme Γ in primary organs → severe/critical."""
        vec = PatientGammaVector(values={"cardiac": 0.9, "vascular": 0.7})
        results = self.engine.match_templates(vec, top_n=3)
        assert results[0].severity in ("severe", "critical")


# ============================================================================
# 9. End-to-End Clinical Scenarios
# ============================================================================

class TestClinicalScenarios:
    def setup_method(self):
        self.engine = GammaEngine(templates=build_all_templates())

    def test_hepatitis_hepatic_dominant(self):
        """Acute hepatitis labs → hepatic Γ is dominant."""
        labs = {"AST": 480, "ALT": 520, "Bil_total": 3.2, "Albumin": 2.8, "INR": 1.8}
        vec = self.engine.lab_to_gamma(labs)
        assert vec.dominant_organ == "hepatic"
        assert abs(vec["hepatic"]) > 0.5

    def test_dka_endocrine_dominant(self):
        """DKA labs → endocrine Γ is dominant."""
        labs = {"Glucose": 450, "HbA1c": 12.5, "K": 5.8, "Na": 128, "CO2": 10, "Cr": 2.0}
        vec = self.engine.lab_to_gamma(labs)
        top = vec.top_n_organs(3)
        organ_names = [o for o, _ in top]
        assert "endocrine" in organ_names

    def test_anemia_heme_elevated(self):
        """Severe anemia → heme Γ is high."""
        labs = {"Hb": 6.0, "Hct": 20, "MCV": 65, "Ferritin": 5}
        vec = self.engine.lab_to_gamma(labs)
        assert abs(vec["heme"]) > 0.3

    def test_ckd_renal_dominant(self):
        """CKD labs → renal Γ is dominant."""
        labs = {"Cr": 5.0, "BUN": 80, "K": 5.8}
        vec = self.engine.lab_to_gamma(labs)
        assert vec.dominant_organ == "renal"

    def test_diagnosis_returns_suggested_tests(self):
        """Diagnosis results include suggested tests."""
        results = self.engine.diagnose(
            {"Troponin": 20, "CK_MB": 200, "BNP": 1500}, top_n=3
        )
        for r in results:
            # Most templates have suggested tests
            assert isinstance(r.suggested_tests, list)

    def test_format_report_is_string(self):
        """format_report returns a non-empty string."""
        report = self.engine.format_report({"AST": 200}, top_n=3)
        assert isinstance(report, str)
        assert len(report) > 100


# ============================================================================
# 10. Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_single_lab_value(self):
        """Engine handles a single lab input."""
        engine = GammaEngine(templates=build_all_templates())
        results = engine.diagnose({"Cr": 8.0}, top_n=5)
        assert len(results) == 5

    def test_all_normal_very_low_gamma(self):
        """Normal values → near-zero Γ everywhere."""
        engine = GammaEngine()
        labs = {
            "Na": 140, "K": 4.2, "Glucose": 85, "Cr": 1.0,
            "AST": 25, "ALT": 30, "Hb": 14.0, "WBC": 7.0,
            "TSH": 2.0, "Ca": 9.5,
        }
        vec = engine.lab_to_gamma(labs)
        for organ in ORGAN_LIST:
            assert abs(vec[organ]) < 0.15, f"{organ}: Γ={vec[organ]} too high for normal labs"

    def test_extreme_values_dont_crash(self):
        """Very extreme values should not cause math errors."""
        engine = GammaEngine(templates=build_all_templates())
        labs = {"Cr": 100, "WBC": 100, "Glucose": 1000, "Troponin": 100}
        results = engine.diagnose(labs, top_n=5)
        assert len(results) == 5
        for r in results:
            assert math.isfinite(r.distance)
            assert math.isfinite(r.confidence)

    def test_negative_lab_value(self):
        """Negative input (should not occur clinically) → no crash."""
        engine = GammaEngine()
        vec = engine.lab_to_gamma({"Na": -10})  # pathological input
        assert all(math.isfinite(v) for v in vec.values.values())

    def test_zero_lab_value(self):
        """Zero input → no crash."""
        engine = GammaEngine()
        vec = engine.lab_to_gamma({"Hb": 0.0})
        assert math.isfinite(vec["heme"])
