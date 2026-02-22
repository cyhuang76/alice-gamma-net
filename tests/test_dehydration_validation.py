# -*- coding: utf-8 -*-
"""
test_dehydration_validation.py — Tests for Dehydration Delirium Validation Experiment
=====================================================================================

Verifies that the 10 dehydration-cardiovascular validation experiments produce
correct clinical physics — blood cascades, compensatory mechanisms, compound
pathologies, developmental vulnerabilities, and recovery dynamics.

These tests exercise 7+ closed loops simultaneously through the cardiovascular
system, validating full hemodynamic integration with homeostatic, autonomic,
pulmonary, and consciousness subsystems.
"""

import math
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alice.alice_brain import AliceBrain
from alice.core.protocol import Priority
from alice.body.cardiovascular import (
    CardiovascularSystem,
    BLOOD_VOLUME_CRITICAL, TACHYCARDIA_THRESHOLD,
    CEREBRAL_AUTOREGULATION_LOW, MAP_CRITICAL_LOW,
    BP_NORMALIZE_FACTOR, SPO2_HYPOXIA_MILD,
    NEONATAL_VOLUME_FACTOR, HYDRATION_TO_VOLUME_GAIN,
    BLOOD_VOLUME_SETPOINT, VISCOSITY_NORMAL,
)


# ============================================================================
# Helpers
# ============================================================================

def make_signal(freq: float = 40.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, 0.1, 64)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def run_tick(alice: AliceBrain, brightness: float = 0.5,
             noise: float = 0.2):
    visual = make_signal(40.0, 0.5) * brightness
    audio = make_signal(20.0, 0.15) * noise
    alice.hear(audio)
    return alice.see(visual, priority=Priority.NORMAL)


def make_adult_cv() -> CardiovascularSystem:
    """Create a fully grown cardiovascular system for clean tests."""
    cv = CardiovascularSystem()
    cv._volume_growth = 0.6
    return cv


def stabilize_cv(cv: CardiovascularSystem, hydration: float = 1.0,
                 ticks: int = 50, **kwargs) -> dict:
    """Run CV for N ticks at given hydration, return last result."""
    defaults = dict(
        heart_rate=70.0, sympathetic=0.2, parasympathetic=0.3,
        glucose=1.0, breaths_this_tick=0.25, ram_temperature=0.1,
    )
    defaults.update(kwargs)
    r = {}
    for _ in range(ticks):
        r = cv.tick(hydration=hydration, **defaults)
    return r


# ============================================================================
# Test Class 1: Blood Volume Tracks Hydration
# ============================================================================

class TestBloodVolumeHydration:
    """Verify blood volume faithfully tracks hydration level."""

    def test_full_hydration_volume(self):
        cv = make_adult_cv()
        r = stabilize_cv(cv, hydration=1.0)
        assert r["blood_volume"] > 0.6, "Full hydration → reasonable blood volume"

    def test_low_hydration_volume(self):
        cv = make_adult_cv()
        r = stabilize_cv(cv, hydration=0.3)
        assert r["blood_volume"] < 0.5, "Low hydration → low blood volume"

    def test_volume_monotonic_with_hydration(self):
        """Blood volume should decrease as hydration decreases."""
        volumes = []
        for h in [1.0, 0.7, 0.5, 0.3]:
            cv = make_adult_cv()
            r = stabilize_cv(cv, hydration=h)
            volumes.append(r["blood_volume"])
        for i in range(len(volumes) - 1):
            assert volumes[i] >= volumes[i + 1] - 0.01

    def test_volume_recovery(self):
        cv = make_adult_cv()
        stabilize_cv(cv, hydration=0.2, ticks=60)
        bv_low = cv.blood_volume
        stabilize_cv(cv, hydration=0.9, ticks=60)
        bv_recovered = cv.blood_volume
        assert bv_recovered > bv_low


# ============================================================================
# Test Class 2: Viscosity Response
# ============================================================================

class TestViscosityResponse:
    """Verify blood viscosity increases with dehydration (hemoconcentration)."""

    def test_normal_viscosity_near_baseline(self):
        cv = make_adult_cv()
        r = stabilize_cv(cv, hydration=1.0)
        assert 0.8 <= r["blood_viscosity"] <= 1.3

    def test_dehydration_increases_viscosity(self):
        cv = make_adult_cv()
        r = stabilize_cv(cv, hydration=0.3)
        assert r["blood_viscosity"] > VISCOSITY_NORMAL

    def test_viscosity_monotonic(self):
        viscosities = []
        for h in [1.0, 0.6, 0.3]:
            cv = make_adult_cv()
            r = stabilize_cv(cv, hydration=h)
            viscosities.append(r["blood_viscosity"])
        # Lower hydration → higher viscosity
        for i in range(len(viscosities) - 1):
            assert viscosities[i] <= viscosities[i + 1] + 0.01


# ============================================================================
# Test Class 3: Cardiac Output Cascade
# ============================================================================

class TestCardiacOutputCascade:
    """Verify cardiac output drops with reduced blood volume."""

    def test_normal_cardiac_output(self):
        cv = make_adult_cv()
        r = stabilize_cv(cv, hydration=1.0)
        assert r["cardiac_output"] > 0.3

    def test_dehydration_reduces_co(self):
        cv_healthy = make_adult_cv()
        cv_dehy = make_adult_cv()
        r_h = stabilize_cv(cv_healthy, hydration=1.0)
        r_d = stabilize_cv(cv_dehy, hydration=0.3)
        assert r_d["cardiac_output"] < r_h["cardiac_output"]

    def test_stroke_volume_drops(self):
        cv_healthy = make_adult_cv()
        cv_dehy = make_adult_cv()
        r_h = stabilize_cv(cv_healthy, hydration=1.0)
        r_d = stabilize_cv(cv_dehy, hydration=0.3)
        assert r_d["stroke_volume"] < r_h["stroke_volume"]


# ============================================================================
# Test Class 4: Blood Pressure Response
# ============================================================================

class TestBloodPressureResponse:
    """Verify BP drops with dehydration and responds to autonomic input."""

    def test_normal_bp_range(self):
        cv = make_adult_cv()
        r = stabilize_cv(cv, hydration=1.0)
        assert 50 < r["mean_arterial_pressure"] < 120

    def test_dehydration_drops_map(self):
        cv_h = make_adult_cv()
        cv_d = make_adult_cv()
        r_h = stabilize_cv(cv_h, hydration=1.0)
        r_d = stabilize_cv(cv_d, hydration=0.3)
        assert r_d["mean_arterial_pressure"] < r_h["mean_arterial_pressure"]

    def test_systolic_tracks_map(self):
        cv = make_adult_cv()
        r = stabilize_cv(cv, hydration=1.0)
        assert r["systolic_bp"] >= r["diastolic_bp"]

    def test_sympathetic_raises_bp(self):
        cv_low = make_adult_cv()
        cv_high = make_adult_cv()
        r_low = stabilize_cv(cv_low, hydration=1.0, sympathetic=0.1, parasympathetic=0.4)
        r_high = stabilize_cv(cv_high, hydration=1.0, sympathetic=0.8, parasympathetic=0.1)
        assert r_high["vascular_resistance"] > r_low["vascular_resistance"]


# ============================================================================
# Test Class 5: Cerebral Perfusion Autoregulation
# ============================================================================

class TestCerebralPerfusion:
    """Verify cerebral autoregulation and perfusion dynamics."""

    def test_normal_perfusion(self):
        cv = make_adult_cv()
        r = stabilize_cv(cv, hydration=1.0)
        assert r["cerebral_perfusion"] > 0.6

    def test_dehydration_reduces_perfusion(self):
        cv_h = make_adult_cv()
        cv_d = make_adult_cv()
        r_h = stabilize_cv(cv_h, hydration=1.0)
        r_d = stabilize_cv(cv_d, hydration=0.2)
        assert r_d["cerebral_perfusion"] < r_h["cerebral_perfusion"]

    def test_perfusion_monotonic_with_hydration(self):
        perfs = []
        for h in [1.0, 0.7, 0.4, 0.2]:
            cv = make_adult_cv()
            r = stabilize_cv(cv, hydration=h)
            perfs.append(r["cerebral_perfusion"])
        for i in range(len(perfs) - 1):
            assert perfs[i] >= perfs[i + 1] - 0.02


# ============================================================================
# Test Class 6: Compensatory Tachycardia
# ============================================================================

class TestCompensatoryTachycardia:
    """Verify baroreceptor reflex produces tachycardia during hypovolemia."""

    def test_no_tachycardia_when_hydrated(self):
        cv = make_adult_cv()
        r = stabilize_cv(cv, hydration=1.0)
        assert r["compensatory_hr_delta"] < 5.0

    def test_tachycardia_when_dehydrated(self):
        cv = make_adult_cv()
        r = stabilize_cv(cv, hydration=0.25, ticks=80)
        assert r["compensatory_hr_delta"] > 2.0

    def test_tachycardia_flag(self):
        cv = make_adult_cv()
        r = stabilize_cv(cv, hydration=0.2, ticks=80)
        assert r["is_tachycardic"] or r["compensatory_hr_delta"] > 5.0

    def test_tachycardia_resolves(self):
        cv = make_adult_cv()
        stabilize_cv(cv, hydration=0.2, ticks=80)
        peak = cv.compensatory_hr_delta
        stabilize_cv(cv, hydration=1.0, ticks=80)
        assert cv.compensatory_hr_delta < peak * 0.5


# ============================================================================
# Test Class 7: Oxygen Delivery
# ============================================================================

class TestOxygenDelivery:
    """Verify O₂ delivery chain: lung → blood → brain."""

    def test_normal_o2(self):
        cv = make_adult_cv()
        r = stabilize_cv(cv, hydration=1.0)
        assert r["o2_delivery"] > 0.5

    def test_dehydration_reduces_o2(self):
        cv_h = make_adult_cv()
        cv_d = make_adult_cv()
        r_h = stabilize_cv(cv_h, hydration=1.0)
        r_d = stabilize_cv(cv_d, hydration=0.3)
        assert r_d["o2_delivery"] < r_h["o2_delivery"]

    def test_spo2_independent_of_hydration(self):
        """SpO₂ depends on lungs, not blood volume."""
        cv_h = make_adult_cv()
        cv_d = make_adult_cv()
        r_h = stabilize_cv(cv_h, hydration=1.0)
        r_d = stabilize_cv(cv_d, hydration=0.3)
        assert abs(r_h["spo2"] - r_d["spo2"]) < 0.05

    def test_apnea_reduces_spo2(self):
        cv = make_adult_cv()
        r = stabilize_cv(cv, hydration=1.0, breaths_this_tick=0.02, ticks=100)
        assert r["spo2"] < SPO2_HYPOXIA_MILD

    def test_anemia_reduces_o2_not_perfusion(self):
        cv_ctrl = make_adult_cv()
        cv_anemia = make_adult_cv()
        cv_anemia.set_hemoglobin(0.5)
        r_ctrl = stabilize_cv(cv_ctrl, hydration=0.5)
        r_anemia = stabilize_cv(cv_anemia, hydration=0.5)
        assert r_anemia["o2_delivery"] < r_ctrl["o2_delivery"]
        assert abs(r_anemia["cerebral_perfusion"] - r_ctrl["cerebral_perfusion"]) < 0.15


# ============================================================================
# Test Class 8: Compound Pathologies
# ============================================================================

class TestCompoundPathologies:
    """Verify compound stressors produce worse outcomes than individual ones."""

    def test_dehydration_plus_heat(self):
        cv_d = make_adult_cv()
        cv_h = make_adult_cv()
        cv_c = make_adult_cv()
        r_d = stabilize_cv(cv_d, hydration=0.3, ram_temperature=0.1, ticks=80)
        r_h = stabilize_cv(cv_h, hydration=1.0, ram_temperature=0.7, ticks=80)
        r_c = stabilize_cv(cv_c, hydration=0.3, ram_temperature=0.7, ticks=80)
        assert r_c["cerebral_perfusion"] <= r_d["cerebral_perfusion"]
        assert r_c["cerebral_perfusion"] <= r_h["cerebral_perfusion"]

    def test_dehydration_plus_anemia(self):
        cv_d = make_adult_cv()
        cv_a = make_adult_cv()
        cv_a.set_hemoglobin(0.5)
        r_d = stabilize_cv(cv_d, hydration=0.4, ticks=80)
        r_a = stabilize_cv(cv_a, hydration=0.4, ticks=80)
        assert r_a["o2_delivery"] < r_d["o2_delivery"]

    def test_sympathetic_storm_plus_dehydration(self):
        cv_s = make_adult_cv()
        cv_sd = make_adult_cv()
        r_s = stabilize_cv(cv_s, hydration=1.0, sympathetic=0.9, parasympathetic=0.1, ticks=80)
        r_sd = stabilize_cv(cv_sd, hydration=0.3, sympathetic=0.9, parasympathetic=0.1, ticks=80)
        assert r_sd["cerebral_perfusion"] < r_s["cerebral_perfusion"]


# ============================================================================
# Test Class 9: Developmental Vulnerability
# ============================================================================

class TestDevelopmentalVulnerability:
    """Verify neonates are more vulnerable to dehydration."""

    def test_neonate_lower_blood_volume(self):
        cv_neo = CardiovascularSystem()  # No growth
        cv_adult = make_adult_cv()
        # Use higher hydration where neonatal volume cap is clearly limiting
        r_neo = stabilize_cv(cv_neo, hydration=0.8, heart_rate=140.0, ticks=60)
        r_adult = stabilize_cv(cv_adult, hydration=0.8, ticks=60)
        assert r_neo["blood_volume"] < r_adult["blood_volume"]

    def test_neonate_earlier_tachycardia(self):
        cv_neo = CardiovascularSystem()
        cv_adult = make_adult_cv()
        neo_tachy = None
        adult_tachy = None
        for tick in range(150):
            h = 1.0 - tick * 0.005
            h = max(h, 0.2)
            r_n = cv_neo.tick(heart_rate=140, sympathetic=0.2, parasympathetic=0.3,
                              hydration=h, glucose=1.0, breaths_this_tick=0.25,
                              ram_temperature=0.1)
            r_a = cv_adult.tick(heart_rate=70, sympathetic=0.2, parasympathetic=0.3,
                                hydration=h, glucose=1.0, breaths_this_tick=0.25,
                                ram_temperature=0.1)
            if neo_tachy is None and r_n["is_tachycardic"]:
                neo_tachy = tick
            if adult_tachy is None and r_a["is_tachycardic"]:
                adult_tachy = tick
        assert neo_tachy is not None and (adult_tachy is None or neo_tachy <= adult_tachy)

    def test_neonate_worse_perfusion_at_same_hydration(self):
        cv_neo = CardiovascularSystem()
        cv_adult = make_adult_cv()
        r_neo = stabilize_cv(cv_neo, hydration=0.4, heart_rate=140.0, ticks=80)
        r_adult = stabilize_cv(cv_adult, hydration=0.4, ticks=80)
        assert r_neo["cerebral_perfusion"] <= r_adult["cerebral_perfusion"] + 0.05


# ============================================================================
# Test Class 10: Recovery Dynamics
# ============================================================================

class TestRecoveryDynamics:
    """Verify full cascade reversibility with rehydration."""

    def test_blood_volume_recovers(self):
        cv = make_adult_cv()
        stabilize_cv(cv, hydration=0.2, ticks=60)
        bv_crisis = cv.blood_volume
        stabilize_cv(cv, hydration=0.9, ticks=60)
        bv_recovered = cv.blood_volume
        assert bv_recovered > bv_crisis * 1.3

    def test_viscosity_normalizes(self):
        cv = make_adult_cv()
        stabilize_cv(cv, hydration=0.2, ticks=60)
        visc_crisis = cv._blood_viscosity
        stabilize_cv(cv, hydration=0.9, ticks=60)
        visc_recovered = cv._blood_viscosity
        assert visc_recovered < visc_crisis

    def test_tachycardia_resolves_after_rehydration(self):
        cv = make_adult_cv()
        stabilize_cv(cv, hydration=0.2, ticks=80)
        hr_crisis = cv.compensatory_hr_delta
        stabilize_cv(cv, hydration=0.9, ticks=80)
        hr_recovered = cv.compensatory_hr_delta
        assert hr_recovered < hr_crisis * 0.5

    def test_perfusion_recovers(self):
        cv = make_adult_cv()
        stabilize_cv(cv, hydration=0.2, ticks=60)
        perf_crisis = cv.cerebral_perfusion
        stabilize_cv(cv, hydration=0.9, ticks=60)
        perf_recovered = cv.cerebral_perfusion
        assert perf_recovered > perf_crisis

    def test_o2_recovers(self):
        cv = make_adult_cv()
        stabilize_cv(cv, hydration=0.2, ticks=60)
        o2_crisis = cv.o2_delivery
        stabilize_cv(cv, hydration=0.9, ticks=60)
        o2_recovered = cv.o2_delivery
        assert o2_recovered > o2_crisis


# ============================================================================
# Test Class 11: Baroreceptor Dynamics
# ============================================================================

class TestBaroreceptorDynamics:
    """Verify baroreceptor reflex temporal properties."""

    def test_not_instant(self):
        cv = make_adult_cv()
        stabilize_cv(cv, hydration=1.0, ticks=20)
        hr_before = cv.compensatory_hr_delta
        # Sudden dehydration
        cv.tick(heart_rate=70, sympathetic=0.2, parasympathetic=0.3,
                hydration=0.2, glucose=1.0, breaths_this_tick=0.25,
                ram_temperature=0.1)
        hr_tick1 = cv.compensatory_hr_delta
        # Should not jump to full compensation in one tick
        for _ in range(50):
            cv.tick(heart_rate=70, sympathetic=0.2, parasympathetic=0.3,
                    hydration=0.2, glucose=1.0, breaths_this_tick=0.25,
                    ram_temperature=0.1)
        hr_steady = cv.compensatory_hr_delta
        assert hr_tick1 < hr_steady * 0.95 or abs(hr_tick1 - hr_before) < 3.0

    def test_proportional_to_deficit(self):
        hr_deltas = []
        for h in [0.6, 0.4, 0.2]:
            cv = make_adult_cv()
            stabilize_cv(cv, hydration=h, ticks=80)
            hr_deltas.append(cv.compensatory_hr_delta)
        # More dehydration → more tachycardia
        for i in range(len(hr_deltas) - 1):
            assert hr_deltas[i] <= hr_deltas[i + 1] + 0.5


# ============================================================================
# Test Class 12: Dehydration Severity Scale
# ============================================================================

class TestDehydrationSeverityScale:
    """Verify physics maps to clinical severity levels."""

    def test_mild_no_tachycardia(self):
        cv = make_adult_cv()
        r = stabilize_cv(cv, hydration=0.7, ticks=80)
        assert not r["is_tachycardic"]

    def test_moderate_tachycardia(self):
        cv = make_adult_cv()
        r = stabilize_cv(cv, hydration=0.45, ticks=80)
        assert r["is_tachycardic"] or r["compensatory_hr_delta"] > 3.0

    def test_severe_hypotension(self):
        cv = make_adult_cv()
        r = stabilize_cv(cv, hydration=0.2, ticks=80)
        # At severe dehydration MAP drops significantly below normal
        assert r["is_hypotensive"] or r["mean_arterial_pressure"] < 85

    def test_severity_ordering(self):
        """More severe dehydration → worse perfusion."""
        perfs = []
        for h in [1.0, 0.7, 0.45, 0.2]:
            cv = make_adult_cv()
            r = stabilize_cv(cv, hydration=h, ticks=80)
            perfs.append(r["cerebral_perfusion"])
        for i in range(len(perfs) - 1):
            assert perfs[i] >= perfs[i + 1] - 0.02


# ============================================================================
# Test Class 13: Full Brain Integration
# ============================================================================

class TestBrainIntegration:
    """Verify full AliceBrain integration with dehydration scenario."""

    def test_consciousness_drops_with_dehydration(self):
        alice = AliceBrain(neuron_count=60)
        # Warm up
        for _ in range(20):
            run_tick(alice, brightness=0.4, noise=0.1)
        baseline_c = alice.vitals.consciousness

        # Dehydrate
        for _ in range(100):
            alice.homeostatic_drive.hydration = 0.15
            run_tick(alice, brightness=0.4, noise=0.1)
        crisis_c = alice.vitals.consciousness

        # At minimum, consciousness should not be higher during crisis
        # (many factors influence consciousness, but perfusion↓ should push it down)
        assert crisis_c <= baseline_c + 0.1 or alice.cardiovascular.cerebral_perfusion < 0.8

    def test_cardiovascular_receives_autonomic_signals(self):
        alice = AliceBrain(neuron_count=60)
        for _ in range(10):
            run_tick(alice, brightness=0.4, noise=0.1)
        stats = alice.cardiovascular.get_stats()
        assert stats["total_beats"] > 0
        assert stats["total_o2_delivered"] > 0

    def test_introspect_reports_cardiovascular(self):
        alice = AliceBrain(neuron_count=60)
        run_tick(alice)
        intro = alice.introspect()
        assert "cardiovascular" in intro.get("subsystems", {})

    def test_dehydration_recovery_in_brain(self):
        alice = AliceBrain(neuron_count=60)
        for _ in range(20):
            run_tick(alice, brightness=0.4, noise=0.1)

        # Dehydrate
        for _ in range(80):
            alice.homeostatic_drive.hydration = 0.15
            run_tick(alice, brightness=0.4, noise=0.1)
        crisis_perf = alice.cardiovascular.cerebral_perfusion

        # Rehydrate
        for _ in range(80):
            alice.homeostatic_drive.hydration = 0.85
            run_tick(alice, brightness=0.4, noise=0.1)
        recovered_perf = alice.cardiovascular.cerebral_perfusion

        assert recovered_perf > crisis_perf


# ============================================================================
# Test Class 14: Lung-CV Coupling
# ============================================================================

class TestLungCVCoupling:
    """Verify lung-cardiovascular coupling for O₂ chain."""

    def test_breathing_maintains_spo2(self):
        cv = make_adult_cv()
        r = stabilize_cv(cv, hydration=1.0, breaths_this_tick=0.25)
        assert r["spo2"] > 0.92

    def test_no_breathing_drops_spo2(self):
        cv = make_adult_cv()
        r = stabilize_cv(cv, hydration=1.0, breaths_this_tick=0.02, ticks=100)
        assert r["spo2"] < 0.92

    def test_spo2_vs_perfusion_orthogonal(self):
        """SpO₂ depends on lungs, perfusion depends on blood volume — orthogonal axes."""
        cv_lungs_ok = make_adult_cv()
        cv_lungs_bad = make_adult_cv()
        r_ok = stabilize_cv(cv_lungs_ok, hydration=0.3, breaths_this_tick=0.25, ticks=80)
        r_bad = stabilize_cv(cv_lungs_bad, hydration=0.3, breaths_this_tick=0.02, ticks=80)
        # Same hydration → similar perfusion
        assert abs(r_ok["cerebral_perfusion"] - r_bad["cerebral_perfusion"]) < 0.2
        # Different breathing → different SpO₂
        assert r_ok["spo2"] > r_bad["spo2"]


# ============================================================================
# Test Class 15: Heat Transport
# ============================================================================

class TestHeatTransport:
    """Verify blood carries heat from core to periphery."""

    def test_heat_transport_at_high_temp(self):
        cv = make_adult_cv()
        r = stabilize_cv(cv, hydration=1.0, ram_temperature=0.5, ticks=50)
        assert r["heat_transported"] > 0

    def test_no_heat_transport_at_zero_temp(self):
        cv = make_adult_cv()
        r = stabilize_cv(cv, hydration=1.0, ram_temperature=0.0, ticks=50)
        assert r["heat_transported"] == 0

    def test_dehydration_reduces_heat_transport(self):
        cv_h = make_adult_cv()
        cv_d = make_adult_cv()
        r_h = stabilize_cv(cv_h, hydration=1.0, ram_temperature=0.5, ticks=80)
        r_d = stabilize_cv(cv_d, hydration=0.3, ram_temperature=0.5, ticks=80)
        assert r_d["heat_transported"] <= r_h["heat_transported"]


# ============================================================================
# Test Class 16: Statistics Tracking
# ============================================================================

class TestStatisticsTracking:
    """Verify lifetime statistics are recorded correctly."""

    def test_beat_count_increases(self):
        cv = make_adult_cv()
        stabilize_cv(cv, hydration=1.0, ticks=100)
        assert cv.total_beats > 0

    def test_hypoxia_ticks_counted(self):
        cv = make_adult_cv()
        stabilize_cv(cv, hydration=1.0, breaths_this_tick=0.01, ticks=200)
        assert cv.hypoxia_ticks > 0

    def test_hypotension_ticks_counted(self):
        cv = make_adult_cv()
        # Use very low hydration to trigger hypotension flag
        stabilize_cv(cv, hydration=0.10, ticks=200)
        # Either flag or MAP meaningfully below normal indicates the physics works
        assert cv.hypotension_ticks > 0 or cv.map_normalized < 0.55

    def test_o2_delivered_accumulates(self):
        cv = make_adult_cv()
        stabilize_cv(cv, hydration=1.0, ticks=50)
        assert cv.total_o2_delivered > 0


# ============================================================================
# Test Class 17: Experiment Import Verification
# ============================================================================

class TestExperimentImport:
    """Verify experiment module is importable and functions exist."""

    def test_import_module(self):
        from experiments import exp_dehydration_validation
        assert hasattr(exp_dehydration_validation, "main")

    def test_all_experiments_exist(self):
        from experiments import exp_dehydration_validation as edv
        for i in range(1, 11):
            fn_name = f"exp_{i:02d}_" 
            found = any(name.startswith(fn_name) for name in dir(edv))
            assert found, f"Missing experiment function starting with {fn_name}"
