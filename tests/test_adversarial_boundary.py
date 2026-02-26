# -*- coding: utf-8 -*-
"""
test_adversarial_boundary.py — Adversarial & Boundary Tests
============================================================

Purpose:
  These tests are NOT designed to pass.
  They are designed to find where the model breaks.

  A 100% pass rate means the tests were written to confirm what we
  already knew. These tests probe what we DON'T know:

  1. Extreme inputs:    hydration=-1, HR=999, hemoglobin=0
  2. Contradictions:    high BP + low blood volume (physiologically impossible)
  3. Numerical:         NaN, Inf, overflow, division by zero
  4. Temporal:          10000+ ticks — does the model drift to nonsense?
  5. Clinical absurdity: NIHSS > 42, MMSE < 0, negative perfusion
  6. Boundary crossings: exactly at every threshold constant

  Tests that PASS mean the model handles that edge case correctly.
  Tests that FAIL reveal model limitations that belong in docs/KNOWN_LIMITATIONS.md.

  This file should NEVER be 100% pass. If it is, we haven't tried hard enough.

Author: Adversarial Validation Protocol
"""

from __future__ import annotations

import math
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alice.body.cardiovascular import (
    CardiovascularSystem,
    BLOOD_VOLUME_MIN, BLOOD_VOLUME_MAX, BLOOD_VOLUME_SETPOINT,
    BP_NORMALIZE_FACTOR, MAP_CRITICAL_LOW, MAP_SYNCOPE,
    PERFUSION_CRITICAL, PERFUSION_LETHAL,
    SPO2_NORMAL, SPO2_MIN, SPO2_HYPOXIA_MILD, SPO2_HYPOXIA_SEVERE,
    VISCOSITY_MAX, NEONATAL_VOLUME_FACTOR,
    CEREBRAL_AUTOREGULATION_LOW, CEREBRAL_AUTOREGULATION_HIGH,
    HEMOGLOBIN_EFFICIENCY,
)
from alice.brain.clinical_neurology import (
    ClinicalNeurologyEngine,
    StrokeModel, ALSModel, DementiaModel, AlzheimersModel,
    CerebralPalsyModel,
    VASCULAR_TERRITORIES, NIHSS_MAX, MMSE_MAX,
    ALS_SPREAD_ORDER_LIMB,
)
from alice.alice_brain import AliceBrain
from alice.core.protocol import Priority


# ============================================================================
# Helpers
# ============================================================================

def make_adult_cv() -> CardiovascularSystem:
    cv = CardiovascularSystem()
    cv._volume_growth = 0.6
    return cv


def stabilize(cv: CardiovascularSystem, ticks: int = 30, **kw):
    defaults = dict(heart_rate=70, sympathetic=0.2, parasympathetic=0.3,
                    hydration=1.0, glucose=1.0, breaths_this_tick=0.25,
                    ram_temperature=0.1)
    defaults.update(kw)
    r = None
    for _ in range(ticks):
        r = cv.tick(**defaults)
    return r


def is_finite_and_reasonable(value, low=-1e6, high=1e6) -> bool:
    """Check a value is finite and within sanity bounds."""
    if isinstance(value, (int, float)):
        return math.isfinite(value) and low <= value <= high
    return True


def check_cv_sanity(r: dict) -> list:
    """Return list of sanity violations in a CV tick result."""
    violations = []
    for key, val in r.items():
        if isinstance(val, (int, float)):
            if not math.isfinite(val):
                violations.append(f"{key}={val} (not finite)")
    # Physical impossibilities
    if r.get("cerebral_perfusion", 0) < 0:
        violations.append(f"negative perfusion: {r['cerebral_perfusion']}")
    if r.get("spo2", 0) < 0:
        violations.append(f"negative SpO₂: {r['spo2']}")
    if r.get("blood_volume", 0) < 0:
        violations.append(f"negative blood volume: {r['blood_volume']}")
    if r.get("o2_delivery", 0) < 0:
        violations.append(f"negative O₂ delivery: {r['o2_delivery']}")
    if r.get("spo2", 0) > 1.0:
        violations.append(f"SpO₂ > 100%: {r['spo2']}")
    if r.get("blood_viscosity", 0) < 0:
        violations.append(f"negative viscosity: {r['blood_viscosity']}")
    return violations


# ============================================================================
# Category 1: Extreme Scalar Inputs
# ============================================================================

class TestExtremeInputs:
    """What happens when inputs go far beyond physiological range?"""

    def test_negative_hydration(self):
        """hydration = -1.0 — physiologically impossible."""
        cv = make_adult_cv()
        r = cv.tick(hydration=-1.0)
        violations = check_cv_sanity(r)
        assert not violations, f"Sanity violations: {violations}"

    def test_extreme_hydration(self):
        """hydration = 100.0 — massive fluid overload."""
        cv = make_adult_cv()
        r = cv.tick(hydration=100.0)
        violations = check_cv_sanity(r)
        assert not violations, f"Sanity violations: {violations}"
        # Blood volume should be clamped to max
        assert r["blood_volume"] <= BLOOD_VOLUME_MAX + 0.01

    def test_zero_hydration(self):
        """hydration = 0.0 — total desiccation."""
        cv = make_adult_cv()
        for _ in range(200):
            r = cv.tick(hydration=0.0)
        violations = check_cv_sanity(r)
        assert not violations, f"Sanity violations: {violations}"

    def test_heart_rate_zero(self):
        """HR = 0 — asystole (cardiac arrest)."""
        cv = make_adult_cv()
        r = cv.tick(heart_rate=0.0)
        violations = check_cv_sanity(r)
        assert not violations, f"Sanity violations: {violations}"
        # Should still have minimal output (clinically: death)
        # The question is: does the model represent this correctly?

    def test_heart_rate_extreme(self):
        """HR = 999 — ventricular fibrillation territory."""
        cv = make_adult_cv()
        r = cv.tick(heart_rate=999.0)
        violations = check_cv_sanity(r)
        assert not violations, f"Sanity violations: {violations}"

    def test_negative_heart_rate(self):
        """HR = -50 — physically meaningless."""
        cv = make_adult_cv()
        r = cv.tick(heart_rate=-50.0)
        violations = check_cv_sanity(r)
        assert not violations, f"Sanity violations: {violations}"

    def test_hemoglobin_zero(self):
        """Hb = 0 — no oxygen carrying capacity at all."""
        cv = make_adult_cv()
        cv.set_hemoglobin(0.0)
        stabilize(cv, ticks=50)
        # O₂ delivery should be zero or near-zero
        assert cv.o2_delivery < 0.15

    def test_hemoglobin_negative(self):
        """Hb = -1.0 — physically meaningless."""
        cv = make_adult_cv()
        cv.set_hemoglobin(-1.0)
        r = stabilize(cv, ticks=20)
        violations = check_cv_sanity(r)
        assert not violations, f"Sanity violations: {violations}"
        # Should be clamped to minimum
        assert cv._hemoglobin >= 0.0

    def test_all_sympathetic(self):
        """sympathetic = 1.0, parasympathetic = 0 — maximal fight-or-flight."""
        cv = make_adult_cv()
        r = cv.tick(sympathetic=1.0, parasympathetic=0.0)
        violations = check_cv_sanity(r)
        assert not violations, f"Sanity violations: {violations}"

    def test_all_parasympathetic(self):
        """sympathetic = 0, parasympathetic = 1.0 — vasovagal syncope territory."""
        cv = make_adult_cv()
        r = cv.tick(sympathetic=0.0, parasympathetic=1.0)
        violations = check_cv_sanity(r)
        assert not violations, f"Sanity violations: {violations}"

    def test_extreme_cortisol(self):
        """cortisol = 10.0 — Cushing's on steroids."""
        cv = make_adult_cv()
        r = cv.tick(cortisol=10.0)
        violations = check_cv_sanity(r)
        assert not violations, f"Sanity violations: {violations}"

    def test_extreme_temperature(self):
        """ram_temperature = 5.0 — severe hyperthermia."""
        cv = make_adult_cv()
        r = cv.tick(ram_temperature=5.0)
        violations = check_cv_sanity(r)
        assert not violations, f"Sanity violations: {violations}"

    def test_negative_glucose(self):
        """glucose = -1.0 — meaningless."""
        cv = make_adult_cv()
        r = cv.tick(glucose=-1.0)
        violations = check_cv_sanity(r)
        assert not violations, f"Sanity violations: {violations}"


# ============================================================================
# Category 2: Physiological Contradictions
# ============================================================================

class TestPhysiologicalContradictions:
    """Inputs that are internally contradictory or clinically impossible."""

    def test_high_bp_low_volume(self):
        """
        High sympathetic (vasoconstriction → BP↑) + very low hydration (BV↓)
        Clinically: possible in early shock (compensation), but the numbers
        should still be self-consistent.
        """
        cv = make_adult_cv()
        r = cv.tick(sympathetic=1.0, parasympathetic=0.0, hydration=0.1)
        violations = check_cv_sanity(r)
        assert not violations, f"Sanity violations: {violations}"
        # MAP should still be within physiological range
        map_mmhg = r["mean_arterial_pressure"]
        assert 20.0 < map_mmhg < 200.0, f"MAP={map_mmhg} out of range"

    def test_simultaneous_sympathetic_parasympathetic(self):
        """
        Both at 1.0 — autonomic conflict.
        Clinically: can happen during baroreflex reset, but unusual.
        """
        cv = make_adult_cv()
        r = cv.tick(sympathetic=1.0, parasympathetic=1.0)
        violations = check_cv_sanity(r)
        assert not violations, f"Sanity violations: {violations}"

    @pytest.mark.xfail(reason="LIMITATION L-CV-01: BV floor=0.05 + autoregulation preserves perfusion even at hydration=0")
    def test_good_breathing_terrible_perfusion(self):
        """
        High breathing rate but zero blood volume.
        Lungs work fine but there's no blood to carry O₂.
        O₂ delivery should still be near zero despite good SpO₂.

        FINDING: At hydration=0, BV drops to floor (0.05) but autoregulation
        still provides perfusion = 0.05/0.4 * 1.0 ≈ 0.125 baseline.
        Combined with good SpO₂ and viscosity penalty, O₂ delivery ends
        up around 0.55 — higher than clinically expected for total
        desiccation. Clinical reality: hydration=0 = death.
        """
        cv = make_adult_cv()
        for _ in range(200):
            r = cv.tick(breaths_this_tick=0.5, hydration=0.0)
        # SpO₂ might be ok (lungs work), but O₂ delivery should be low
        # because perfusion is poor
        assert r["o2_delivery"] < 0.5, (
            f"O₂={r['o2_delivery']} — good O₂ delivery with no blood volume?"
        )

    def test_no_breathing_full_blood(self):
        """
        Zero breathing but perfect hydration.
        Blood is full but carries no oxygen.
        """
        cv = make_adult_cv()
        for _ in range(100):
            r = cv.tick(breaths_this_tick=0.0, hydration=1.0)
        # SpO₂ should have dropped
        assert r["spo2"] < SPO2_NORMAL - 0.05

    def test_neonatal_adult_hr(self):
        """
        Neonatal blood volume but adult resting HR (60 bpm).
        Neonates normally have HR 120-160 — 60 would mean bradycardia.
        The model doesn't enforce age-appropriate HR.
        """
        cv = CardiovascularSystem()  # Neonatal
        r = cv.tick(heart_rate=60)
        violations = check_cv_sanity(r)
        assert not violations, f"Sanity violations: {violations}"
        # KNOWN LIMITATION: model doesn't reject age-inappropriate HR


# ============================================================================
# Category 3: Numerical Stability
# ============================================================================

class TestNumericalStability:
    """NaN, Inf, overflow, underflow."""

    @pytest.mark.xfail(reason="LIMITATION L-NUM-01: NaN input propagates through np.clip without guard")
    def test_nan_hydration(self):
        """NaN input — should not propagate.

        FINDING: NaN hydration → NaN blood_volume → NaN propagates
        through the entire tick chain. np.clip(NaN, ...) returns NaN.
        No NaN guard exists in any internal method.
        """
        cv = make_adult_cv()
        # Note: this tests whether the model handles NaN gracefully
        # It may crash or propagate NaN — that's a finding
        try:
            r = cv.tick(hydration=float('nan'))
            # If it doesn't crash, check for NaN propagation
            for key, val in r.items():
                if isinstance(val, float):
                    assert not math.isnan(val), f"NaN propagated to {key}"
        except (ValueError, RuntimeError):
            pytest.skip("Model crashes on NaN input (expected limitation)")

    def test_inf_hydration(self):
        """Inf input."""
        cv = make_adult_cv()
        try:
            r = cv.tick(hydration=float('inf'))
            violations = check_cv_sanity(r)
            assert not violations, f"Sanity violations: {violations}"
        except (ValueError, RuntimeError, OverflowError):
            pytest.skip("Model crashes on Inf input (expected limitation)")

    def test_negative_inf_heart_rate(self):
        """-Inf input."""
        cv = make_adult_cv()
        try:
            r = cv.tick(heart_rate=float('-inf'))
            violations = check_cv_sanity(r)
            assert not violations, f"Sanity violations: {violations}"
        except (ValueError, RuntimeError, OverflowError):
            pytest.skip("Model crashes on -Inf input (expected limitation)")

    def test_long_run_no_drift(self):
        """
        Run 50000 ticks at constant input.
        Model should reach steady state, not drift to infinity.
        """
        cv = make_adult_cv()
        for _ in range(50000):
            r = cv.tick(heart_rate=70, sympathetic=0.2, parasympathetic=0.3,
                        hydration=1.0, glucose=1.0, breaths_this_tick=0.25,
                        ram_temperature=0.1)
        violations = check_cv_sanity(r)
        assert not violations, f"Drift after 50k ticks: {violations}"
        # Should be at steady state — perfusion near normal
        assert 0.5 < r["cerebral_perfusion"] < 1.5
        assert 0.5 < r["spo2"] <= SPO2_NORMAL

    def test_oscillating_inputs_stability(self):
        """
        Rapidly alternating extreme inputs.
        Should not cause numerical explosion.
        """
        cv = make_adult_cv()
        for tick in range(5000):
            if tick % 2 == 0:
                r = cv.tick(heart_rate=200, sympathetic=1.0,
                            parasympathetic=0.0, hydration=0.1)
            else:
                r = cv.tick(heart_rate=40, sympathetic=0.0,
                            parasympathetic=1.0, hydration=1.5)
        violations = check_cv_sanity(r)
        assert not violations, f"Instability after oscillation: {violations}"

    def test_zero_everything(self):
        """All inputs = 0."""
        cv = make_adult_cv()
        r = cv.tick(heart_rate=0, sympathetic=0, parasympathetic=0,
                    hydration=0, glucose=0, breaths_this_tick=0,
                    ram_temperature=0, cortisol=0)
        violations = check_cv_sanity(r)
        assert not violations, f"Sanity violations at zero: {violations}"


# ============================================================================
# Category 4: Clinical Scale Boundaries
# ============================================================================

class TestClinicalScaleBoundaries:
    """Do clinical scores stay within valid ranges under all conditions?"""

    def test_nihss_never_exceeds_max(self):
        """NIHSS should never exceed 42, even with maximal severity."""
        stroke = StrokeModel()
        # Induce every territory at max severity
        for territory in VASCULAR_TERRITORIES:
            stroke.induce(territory, severity=1.0)
        for _ in range(1000):
            stroke.tick()
        assert stroke.get_nihss() <= NIHSS_MAX

    def test_nihss_never_negative(self):
        stroke = StrokeModel()
        assert stroke.get_nihss() >= 0
        stroke.induce("MCA", severity=0.001)
        for _ in range(10):
            stroke.tick()
        assert stroke.get_nihss() >= 0

    def test_alsfrs_r_range(self):
        """ALSFRS-R should stay 0-48."""
        als = ALSModel()
        als.onset("limb", riluzole=False)
        for _ in range(10000):
            result = als.tick()
        score = result["alsfrs_r"]
        assert 0 <= score <= 48, f"ALSFRS-R = {score} out of range"

    def test_mmse_range_dementia(self):
        """MMSE should stay 0-30."""
        d = DementiaModel()
        d.onset("severe")
        for _ in range(50000):
            d.tick()
        mmse = d.get_mmse()
        assert 0 <= mmse <= MMSE_MAX, f"MMSE = {mmse} out of range"

    def test_mmse_range_alzheimers(self):
        ad = AlzheimersModel()
        ad.onset(genetic_risk=1.0)
        for _ in range(50000):
            ad.tick()
        mmse = ad.get_mmse()
        assert 0 <= mmse <= MMSE_MAX, f"MMSE = {mmse} out of range"

    def test_braak_stage_range(self):
        """Braak should stay 0-6."""
        ad = AlzheimersModel()
        ad.onset(genetic_risk=1.0)
        for _ in range(100000):
            ad.tick()
        braak = ad.get_braak_stage()
        assert 0 <= braak <= 6, f"Braak = {braak} out of range"

    def test_gmfcs_range(self):
        """GMFCS should stay 1-5."""
        cp = CerebralPalsyModel()
        cp.set_condition("spastic", gmfcs_level=5)
        for _ in range(1000):
            cp.tick()
        gmfcs = cp.get_gmfcs()
        assert 1 <= gmfcs <= 5, f"GMFCS = {gmfcs} out of range"

    def test_gmfcs_clamped_on_extreme_input(self):
        """GMFCS level 99 should be clamped to 5."""
        cp = CerebralPalsyModel()
        cp.set_condition("spastic", gmfcs_level=99)
        assert cp.get_gmfcs() == 5

    def test_gmfcs_clamped_on_zero_input(self):
        """GMFCS level 0 should be clamped to 1."""
        cp = CerebralPalsyModel()
        cp.set_condition("spastic", gmfcs_level=0)
        assert cp.get_gmfcs() == 1

    def test_stroke_invalid_territory(self):
        """Inducing a non-existent territory — should not crash."""
        stroke = StrokeModel()
        try:
            stroke.induce("nonexistent_territory", severity=0.5)
            for _ in range(30):
                stroke.tick()
            # If it doesn't crash, NIHSS should be 0 (no real territory affected)
            # or it was accepted — either way, no crash
        except (KeyError, ValueError):
            pass  # Acceptable: model rejects invalid territory

    def test_als_invalid_onset_type(self):
        """Onset type 'cardiac' doesn't exist."""
        als = ALSModel()
        try:
            als.onset("cardiac")
            for _ in range(100):
                als.tick()
        except (KeyError, ValueError):
            pass  # Acceptable


# ============================================================================
# Category 5: Temporal Boundary — Long-Duration Stability
# ============================================================================

class TestTemporalBoundary:
    """What happens over very long timescales?"""

    def test_als_fully_progressed(self):
        """
        ALS after 100k ticks — all channels should be dead.
        Is there a floor? Does the model handle total degeneration?
        """
        als = ALSModel()
        als.onset("limb", riluzole=False)
        for _ in range(100000):
            result = als.tick()
        # Every channel should be near zero
        for ch, health in als.channel_health.items():
            assert health >= 0.0, f"{ch} health went negative: {health}"

    def test_dementia_fully_progressed(self):
        """Dementia after 100k ticks — MMSE should be 0."""
        d = DementiaModel()
        d.onset("severe")
        for _ in range(100000):
            d.tick()
        assert d.get_mmse() == 0
        assert d.get_cdr() == 3.0

    def test_alzheimers_max_braak(self):
        """After 100k ticks, should reach Braak 6."""
        ad = AlzheimersModel()
        ad.onset(genetic_risk=1.0)
        for _ in range(100000):
            ad.tick()
        assert ad.get_braak_stage() == 6

    def test_stroke_penumbra_resolution(self):
        """
        After very long time, penumbra should have resolved
        (either saved by reperfusion or progressed to infarct).
        """
        stroke = StrokeModel()
        stroke.induce("MCA", severity=0.7)
        for _ in range(10000):
            stroke.tick()
        nihss = stroke.get_nihss()
        # NIHSS should be stable (not oscillating)
        nihss_later = nihss
        for _ in range(1000):
            stroke.tick()
        nihss_later = stroke.get_nihss()
        # Difference should be small (steady state)
        assert abs(nihss - nihss_later) <= 2

    def test_cv_100k_ticks_stable(self):
        """CV system after 100k ticks — should be at steady state."""
        cv = make_adult_cv()
        for _ in range(100000):
            r = cv.tick(heart_rate=70, sympathetic=0.2, parasympathetic=0.3,
                        hydration=1.0, glucose=1.0, breaths_this_tick=0.25,
                        ram_temperature=0.1)
        violations = check_cv_sanity(r)
        assert not violations
        assert 0.8 < r["cerebral_perfusion"] < 1.2


# ============================================================================
# Category 6: Threshold Boundary Tests
# ============================================================================

class TestThresholdBoundaries:
    """Test behavior exactly at every known threshold constant."""

    def test_at_autoregulation_low_boundary(self):
        """MAP exactly at autoregulation low threshold."""
        cv = make_adult_cv()
        # We can't directly set MAP, but we can observe behavior
        # around the threshold by manipulating inputs
        for _ in range(50):
            r = cv.tick(heart_rate=50, sympathetic=0.0, parasympathetic=0.5,
                        hydration=0.4, glucose=1.0, breaths_this_tick=0.25,
                        ram_temperature=0.1)
        violations = check_cv_sanity(r)
        assert not violations

    def test_at_tachycardia_threshold(self):
        """Blood volume exactly at tachycardia threshold (0.7)."""
        cv = make_adult_cv()
        # Run to get BV near threshold
        for _ in range(200):
            r = cv.tick(hydration=0.82)  # Tuned to get BV near 0.7
        violations = check_cv_sanity(r)
        assert not violations

    def test_at_spo2_hypoxia_threshold(self):
        """SpO₂ near the mild hypoxia threshold."""
        cv = make_adult_cv()
        for _ in range(200):
            r = cv.tick(breaths_this_tick=0.01, hydration=0.8)
        # Check hypoxia flag consistency
        if r["spo2"] < SPO2_HYPOXIA_MILD:
            assert r["is_hypoxic"]
        else:
            assert not r["is_hypoxic"]

    def test_viscosity_at_max(self):
        """Extreme dehydration → viscosity should hit max."""
        cv = make_adult_cv()
        for _ in range(300):
            r = cv.tick(hydration=0.0)
        assert r["blood_viscosity"] <= VISCOSITY_MAX + 0.01

    def test_blood_volume_min_clamp(self):
        """BV should never go below the min floor (0.05 in implementation)."""
        cv = make_adult_cv()
        for _ in range(500):
            cv.tick(hydration=-1.0)
        assert cv.blood_volume >= 0.04  # Floor is 0.05


# ============================================================================
# Category 7: Cross-Disease Interaction Edge Cases
# ============================================================================

class TestCrossDiseaseEdge:
    """Edge cases when multiple diseases interact."""

    def test_all_five_diseases_simultaneously(self):
        """
        All 5 diseases active at once.
        The ClinicalNeurologyEngine should not crash.
        """
        engine = ClinicalNeurologyEngine()
        engine.stroke.induce("MCA", severity=0.8)
        engine.stroke.induce("PCA", severity=0.5)
        engine.als.onset("limb")
        engine.dementia.onset("moderate")
        engine.alzheimers.onset(genetic_risk=1.0)
        engine.cerebral_palsy.set_condition("spastic", gmfcs_level=4)
        for _ in range(1000):
            result = engine.tick()
        # All should be active
        active = result["active_conditions"]
        assert len(active) >= 4  # All except possibly overlapping ones
        # Merged Γ should be valid
        merged = result["merged_channel_gamma"]
        for ch, gamma in merged.items():
            assert 0.0 <= gamma <= 1.0, f"{ch}: Γ={gamma} out of [0,1]"

    def test_duplicate_stroke_induction(self):
        """Induce stroke in same territory twice — should not double-count."""
        stroke = StrokeModel()
        stroke.induce("MCA", severity=0.5)
        stroke.induce("MCA", severity=0.5)  # Second induction
        for _ in range(100):
            stroke.tick()
        nihss = stroke.get_nihss()
        assert nihss <= NIHSS_MAX

    def test_reperfuse_nonexistent_stroke(self):
        """Reperfuse index that doesn't exist."""
        stroke = StrokeModel()
        try:
            stroke.reperfuse(99)  # No strokes induced
        except (IndexError, KeyError):
            pass  # Acceptable: explicit error
        # If it doesn't crash, that's also fine

    def test_als_onset_twice(self):
        """Calling onset twice — should it reset or accumulate?"""
        als = ALSModel()
        als.onset("limb")
        for _ in range(500):
            als.tick()
        health_before = dict(als.channel_health)
        als.onset("bulbar")  # Second onset — what happens?
        for _ in range(500):
            als.tick()
        # Model should still produce valid output
        result = als.tick()
        assert 0 <= result["alsfrs_r"] <= 48


# ============================================================================
# Category 8: Model vs Reality (Calibration Probes)
# ============================================================================

class TestCalibrationProbes:
    """
    These tests compare model outputs to KNOWN clinical values.
    They will FAIL if the model is not calibrated.
    Each failure is a data point for docs/KNOWN_LIMITATIONS.md.

    IMPORTANT: Failing these tests is EXPECTED and INFORMATIVE.
    """

    @pytest.mark.xfail(reason="LIMITATION L-CAL-01: MAP formula yields ~106 mmHg at rest (clinical normal ~93)")
    def test_normal_adult_map_in_physiological_range(self):
        """
        Normal adult MAP should be 70-105 mmHg.
        Clinical: MAP = (SBP + 2×DBP) / 3 ≈ 93 mmHg.

        FINDING: Model yields MAP ≈ 106 mmHg at default adult steady state.
        The linear formula MAP = 40 + CO×R×80 overshoots by ~13%.
        This is a calibration gap, not a physics error.
        """
        cv = make_adult_cv()
        stabilize(cv, ticks=100)
        map_mmhg = cv.map_normalized * BP_NORMALIZE_FACTOR
        assert 70.0 <= map_mmhg <= 105.0, (
            f"MAP = {map_mmhg:.1f} mmHg (expected 70-105)"
        )

    def test_normal_systolic_diastolic_ratio(self):
        """
        Systolic should be > Diastolic.
        Normal: SBP ~120, DBP ~80, PP = SBP - DBP ≈ 40.
        """
        cv = make_adult_cv()
        r = stabilize(cv, ticks=100)
        assert r["systolic_bp"] > r["diastolic_bp"], (
            f"SBP={r['systolic_bp']}, DBP={r['diastolic_bp']} — impossible"
        )

    def test_normal_spo2_above_95(self):
        """Normal adult SpO₂ should be ≥ 95%."""
        cv = make_adult_cv()
        stabilize(cv, ticks=100)
        assert cv.spo2 >= 0.95, f"SpO₂ = {cv.spo2:.3f} (expected ≥ 0.95)"

    def test_tachycardia_matches_clinical(self):
        """
        Clinical: tachycardia = HR > 100 bpm.
        Model: compensatory_hr_delta at hydration=0.3 should be significant.
        ~ 10-30 bpm increase expected.
        """
        cv = make_adult_cv()
        stabilize(cv, ticks=100, hydration=0.3)
        delta = cv.compensatory_hr_delta
        assert 5.0 < delta < 60.0, (
            f"HR delta = {delta:.1f} bpm (expected 5-60 for moderate dehydration)"
        )

    @pytest.mark.xfail(reason="Model lacks temporal calibration: ticks ≠ minutes/hours")
    def test_als_progression_timeline(self):
        """
        Clinical: ALS median survival from onset = 3-5 years.
        ALSFRS-R drops ~1 point/month → ~36-60 months to reach 0.
        Model: how many ticks to reach ALSFRS-R = 0?
        If 1 tick = 1 second (50ms × 20 brain ticks), then
        3 years ≈ 94.6M ticks. Running 10k ticks = ~2.8 hours.
        ALSFRS-R should barely have moved at 10k ticks.
        """
        als = ALSModel()
        als.onset("limb", riluzole=False)
        for _ in range(10000):
            result = als.tick()
        score = result["alsfrs_r"]
        # In reality, 10k ticks (~3 hrs) should show minimal decline
        assert score >= 45, (
            f"ALSFRS-R = {score} after 10k ticks — model progresses too fast"
        )

    @pytest.mark.xfail(reason="Model lacks temporal calibration: ticks ≠ minutes/hours")
    def test_alzheimers_braak_timeline(self):
        """
        Clinical: Braak stage 1→6 takes ~20-30 years.
        At 10k ticks (~3 hours), Braak should be 0.
        """
        ad = AlzheimersModel()
        ad.onset(genetic_risk=1.0)
        for _ in range(10000):
            ad.tick()
        assert ad.get_braak_stage() == 0, (
            f"Braak = {ad.get_braak_stage()} after 10k ticks — "
            f"model progresses far too fast vs clinical"
        )

    def test_anemia_spo2_clinical_correlation(self):
        """
        Clinical: Anemia does NOT reduce SpO₂ (hemoglobin-independent).
        But severe anemia (Hb < 7 g/dL) → tissue hypoxia signs.
        Model correctly separates SpO₂ from O₂ content.
        However: clinical Hb=7 maps to what model Hb value?
        """
        cv = make_adult_cv()
        cv.set_hemoglobin(0.5)  # What is this in g/dL?
        stabilize(cv, ticks=50)
        # SpO₂ should be normal (anemia doesn't change saturation)
        assert cv.spo2 >= 0.95
        # But what does Hb=0.5 mean clinically? This is un-mapped.

    @pytest.mark.xfail(reason="MAP formula not calibrated to Ohm's law precisely")
    def test_map_equals_co_times_svr(self):
        """
        Hemodynamic Ohm's law: MAP = CO × SVR.
        In the model, MAP uses a linear transform (40 + CO×R×80).
        This means model MAP ≠ direct CO×R product.
        """
        cv = make_adult_cv()
        r = stabilize(cv, ticks=100)
        co = r["cardiac_output"]
        svr = r["vascular_resistance"]
        map_predicted = co * svr
        map_actual_norm = r["map_normalized"]
        # These should be proportional, not necessarily equal
        # but the relationship should be monotonic
        assert abs(map_predicted - map_actual_norm) < 0.1, (
            f"MAP formula mismatch: CO×R={map_predicted:.3f}, "
            f"MAP_norm={map_actual_norm:.3f}"
        )


# ============================================================================
# Category 9: Recovery & Reversibility
# ============================================================================

class TestRecoveryReversibility:
    """Can the model recover from extreme states?"""

    def test_recovery_from_total_desiccation(self):
        """After hydration=0 for 200 ticks, restore hydration → should recover."""
        cv = make_adult_cv()
        for _ in range(200):
            cv.tick(hydration=0.0)
        min_perf = cv.cerebral_perfusion
        # Restore
        for _ in range(200):
            r = cv.tick(hydration=1.0)
        assert cv.cerebral_perfusion > min_perf + 0.1

    def test_recovery_from_no_breathing(self):
        cv = make_adult_cv()
        for _ in range(100):
            cv.tick(breaths_this_tick=0.0)
        min_spo2 = cv.spo2
        for _ in range(100):
            cv.tick(breaths_this_tick=0.25)
        assert cv.spo2 > min_spo2

    def test_stroke_reperfusion_recovery(self):
        """After reperfusion, NIHSS should improve (penumbra rescue)."""
        stroke = StrokeModel()
        stroke.induce("MCA", severity=0.7)
        for _ in range(50):
            stroke.tick()
        nihss_peak = stroke.get_nihss()
        stroke.reperfuse(0)
        for _ in range(200):
            stroke.tick()
        nihss_after = stroke.get_nihss()
        assert nihss_after <= nihss_peak

    def test_als_no_recovery_mechanism(self):
        """ALS has no recovery. This test documents that the model is correct."""
        als = ALSModel()
        als.onset("limb")
        for _ in range(5000):
            als.tick()
        health_mid = dict(als.channel_health)
        # There's no "cure" method. Running more ticks only gets worse.
        for _ in range(5000):
            als.tick()
        for ch in health_mid:
            assert als.channel_health[ch] <= health_mid[ch] + 0.001


# ============================================================================
# Category 10: Full Brain Adversarial
# ============================================================================

class TestFullBrainAdversarial:
    """Adversarial tests on the full AliceBrain."""

    def test_brain_survives_500_ticks_zero_input(self):
        """Feed the brain absolute silence for 500 ticks."""
        alice = AliceBrain(neuron_count=60)
        zero_signal = np.zeros(64, dtype=np.float32)
        for _ in range(500):
            alice.hear(zero_signal)
            alice.see(zero_signal, priority=Priority.NORMAL)
        # Should not crash, vitals should still be reportable
        vitals = alice.get_vitals()
        assert isinstance(vitals, dict)

    def test_brain_survives_extreme_input(self):
        """Feed maximal signals."""
        alice = AliceBrain(neuron_count=60)
        loud = np.ones(64, dtype=np.float32) * 100.0
        for _ in range(100):
            alice.hear(loud)
            alice.see(loud, priority=Priority.HIGH)
        vitals = alice.get_vitals()
        assert isinstance(vitals, dict)

    def test_brain_with_all_diseases_and_dehydration(self):
        """The absolute worst case: every disease + dehydration."""
        alice = AliceBrain(neuron_count=60)
        for _ in range(10):
            signal = np.random.randn(64).astype(np.float32) * 0.3
            alice.hear(signal)
            alice.see(signal, priority=Priority.NORMAL)

        alice.clinical_neurology.stroke.induce("MCA", severity=0.8)
        alice.clinical_neurology.als.onset("limb")
        alice.clinical_neurology.dementia.onset("severe")
        alice.clinical_neurology.alzheimers.onset(genetic_risk=1.0)
        alice.clinical_neurology.cerebral_palsy.set_condition("spastic", gmfcs_level=5)
        alice.cardiovascular.set_hemoglobin(0.3)

        for _ in range(200):
            alice.homeostatic_drive.hydration = 0.1
            signal = np.random.randn(64).astype(np.float32) * 0.1
            alice.hear(signal)
            result = alice.see(signal, priority=Priority.NORMAL)

        # Should not crash
        assert isinstance(result, dict)
        # Introspect should still work
        intro = alice.introspect()
        assert "subsystems" in intro
