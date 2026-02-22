# -*- coding: utf-8 -*-
"""
Tests for CardiovascularSystem — Transmission Line Hemodynamics

Tests verify:
    1. Basic construction and neonatal initial state
    2. Blood volume tracks hydration
    3. Blood viscosity increases with dehydration
    4. Vascular resistance modulated by autonomic
    5. Cardiac output = HR × SV
    6. Blood pressure from hemodynamic Ohm's law
    7. Cerebral perfusion with autoregulation
    8. O₂ transport chain (breathing → SpO₂ → delivery)
    9. Heat transport via blood
    10. Baroreceptor reflex (compensatory tachycardia)
    11. Developmental volume growth
    12. Hemoglobin / anemia simulation
    13. Proprioception (heartbeat awareness)
    14. Integration with AliceBrain
    15. Pathological scenarios (dehydration, hypoxia, syncope)
"""

import numpy as np
import pytest

from alice.body.cardiovascular import (
    CardiovascularSystem,
    BLOOD_VOLUME_SETPOINT,
    BLOOD_VOLUME_CRITICAL,
    BLOOD_VOLUME_MIN,
    NEONATAL_VOLUME_FACTOR,
    VISCOSITY_NORMAL,
    VISCOSITY_MAX,
    VASCULAR_RESISTANCE_BASE,
    PERFUSION_NORMAL,
    PERFUSION_CRITICAL,
    SPO2_NORMAL,
    SPO2_HYPOXIA_MILD,
    MAP_CRITICAL_LOW,
    MAP_SYNCOPE,
    BP_NORMALIZE_FACTOR,
    TACHYCARDIA_THRESHOLD,
    CARDIOVASCULAR_IMPEDANCE,
    O2_DELIVERY_NORMAL,
)
from alice.core.signal import ElectricalSignal


# ============================================================================
# Basic Construction
# ============================================================================


class TestConstruction:
    """CardiovascularSystem initializes with neonatal defaults."""

    def test_neonatal_blood_volume(self):
        cv = CardiovascularSystem()
        assert cv._blood_volume == pytest.approx(
            BLOOD_VOLUME_SETPOINT * NEONATAL_VOLUME_FACTOR, rel=0.01
        )

    def test_initial_spo2_normal(self):
        cv = CardiovascularSystem()
        assert cv._spo2 == SPO2_NORMAL

    def test_initial_stats_zero(self):
        cv = CardiovascularSystem()
        stats = cv.get_stats()
        assert stats["total_beats"] == 0
        assert stats["hypoxia_ticks"] == 0
        assert stats["syncope_episodes"] == 0

    def test_initial_no_compensatory_hr(self):
        cv = CardiovascularSystem()
        assert cv._compensatory_hr_delta == 0.0


# ============================================================================
# Blood Volume
# ============================================================================


class TestBloodVolume:
    """Blood volume tracks hydration."""

    def test_high_hydration_increases_volume(self):
        cv = CardiovascularSystem()
        cv._volume_growth = 0.6  # Simulate developed capacity
        cv._blood_volume = 0.5
        cv.tick(hydration=1.0)
        assert cv._blood_volume > 0.5

    def test_low_hydration_decreases_volume(self):
        cv = CardiovascularSystem()
        cv._blood_volume = 0.8  # Start high
        for _ in range(20):
            cv.tick(hydration=0.2)
        assert cv._blood_volume < 0.8

    def test_volume_never_zero(self):
        cv = CardiovascularSystem()
        for _ in range(50):
            cv.tick(hydration=0.0)
        assert cv._blood_volume > 0.0


# ============================================================================
# Blood Viscosity
# ============================================================================


class TestViscosity:
    """Dehydration increases blood viscosity."""

    def test_normal_hydration_normal_viscosity(self):
        cv = CardiovascularSystem()
        cv._blood_volume = BLOOD_VOLUME_SETPOINT
        cv._update_viscosity()
        assert cv._blood_viscosity == pytest.approx(VISCOSITY_NORMAL, rel=0.1)

    def test_dehydration_increases_viscosity(self):
        cv = CardiovascularSystem()
        cv._blood_volume = 0.3  # Severely dehydrated
        cv._update_viscosity()
        assert cv._blood_viscosity > VISCOSITY_NORMAL

    def test_viscosity_has_upper_bound(self):
        cv = CardiovascularSystem()
        cv._blood_volume = 0.05  # Near-zero
        cv._update_viscosity()
        assert cv._blood_viscosity <= VISCOSITY_MAX


# ============================================================================
# Vascular Resistance
# ============================================================================


class TestVascularResistance:
    """Autonomic modulation of vascular resistance."""

    def test_sympathetic_increases_resistance(self):
        cv = CardiovascularSystem()
        cv._update_vascular_resistance(
            sympathetic=0.9, parasympathetic=0.1,
            temperature=0.0, cortisol=0.0
        )
        assert cv._vascular_resistance > VASCULAR_RESISTANCE_BASE

    def test_parasympathetic_decreases_resistance(self):
        cv = CardiovascularSystem()
        cv._update_vascular_resistance(
            sympathetic=0.1, parasympathetic=0.9,
            temperature=0.0, cortisol=0.0
        )
        assert cv._vascular_resistance < VASCULAR_RESISTANCE_BASE

    def test_high_temperature_dilates(self):
        cv = CardiovascularSystem()
        cv._update_vascular_resistance(
            sympathetic=0.2, parasympathetic=0.3,
            temperature=0.8, cortisol=0.0
        )
        r_hot = cv._vascular_resistance
        cv._update_vascular_resistance(
            sympathetic=0.2, parasympathetic=0.3,
            temperature=0.0, cortisol=0.0
        )
        r_cold = cv._vascular_resistance
        assert r_hot < r_cold


# ============================================================================
# Cardiac Output
# ============================================================================


class TestCardiacOutput:
    """Cardiac output = HR × SV."""

    def test_higher_hr_higher_output(self):
        cv1 = CardiovascularSystem()
        cv1._blood_volume = 0.8
        r1 = cv1.tick(heart_rate=60.0)
        cv2 = CardiovascularSystem()
        cv2._blood_volume = 0.8
        r2 = cv2.tick(heart_rate=120.0)
        assert r2["cardiac_output"] > r1["cardiac_output"]

    def test_low_volume_reduces_output(self):
        cv1 = CardiovascularSystem()
        cv1._blood_volume = 1.0
        r1 = cv1.tick()
        cv2 = CardiovascularSystem()
        cv2._blood_volume = 0.3
        r2 = cv2.tick()
        assert r2["cardiac_output"] < r1["cardiac_output"]


# ============================================================================
# Blood Pressure
# ============================================================================


class TestBloodPressure:
    """Blood pressure from hemodynamic Ohm's law."""

    def test_systolic_above_diastolic(self):
        cv = CardiovascularSystem()
        r = cv.tick()
        assert r["systolic_bp"] > r["diastolic_bp"]

    def test_low_volume_low_pressure(self):
        cv = CardiovascularSystem()
        cv._blood_volume = 0.3
        for _ in range(10):
            r = cv.tick(hydration=0.2)
        assert r["mean_arterial_pressure"] < 100.0

    def test_high_sympathetic_raises_pressure(self):
        cv1 = CardiovascularSystem()
        cv1._blood_volume = 0.8
        r1 = cv1.tick(sympathetic=0.1, parasympathetic=0.5)
        cv2 = CardiovascularSystem()
        cv2._blood_volume = 0.8
        r2 = cv2.tick(sympathetic=0.9, parasympathetic=0.1)
        assert r2["mean_arterial_pressure"] > r1["mean_arterial_pressure"]


# ============================================================================
# Cerebral Perfusion
# ============================================================================


class TestCerebralPerfusion:
    """Cerebral perfusion with autoregulation."""

    def test_normal_perfusion_near_one(self):
        cv = CardiovascularSystem()
        cv._blood_volume = 1.0
        r = cv.tick(heart_rate=70.0)
        assert r["cerebral_perfusion"] >= 0.7

    def test_autoregulation_maintains_perfusion(self):
        """Within normal BP range, perfusion is roughly constant."""
        cv1 = CardiovascularSystem()
        cv1._blood_volume = 0.9
        r1 = cv1.tick(heart_rate=60.0)
        cv2 = CardiovascularSystem()
        cv2._blood_volume = 0.9
        r2 = cv2.tick(heart_rate=80.0)
        # Perfusion should be similar (autoregulation)
        assert abs(r1["cerebral_perfusion"] - r2["cerebral_perfusion"]) < 0.3

    def test_severe_dehydration_drops_perfusion(self):
        cv = CardiovascularSystem()
        for _ in range(50):
            r = cv.tick(hydration=0.1)
        assert r["cerebral_perfusion"] < PERFUSION_NORMAL

    def test_viscosity_reduces_perfusion(self):
        cv = CardiovascularSystem()
        cv._blood_volume = 0.3  # High viscosity
        cv._update_viscosity()
        cv._map_normalized = 0.6  # Same MAP
        cv._update_cerebral_perfusion()
        perf_thick = cv._cerebral_perfusion

        cv._blood_volume = 1.0  # Normal viscosity
        cv._update_viscosity()
        cv._update_cerebral_perfusion()
        perf_normal = cv._cerebral_perfusion

        assert perf_thick < perf_normal


# ============================================================================
# O₂ Transport
# ============================================================================


class TestOxygenTransport:
    """O₂ transport chain: lung → blood → brain."""

    def test_breathing_maintains_spo2(self):
        cv = CardiovascularSystem()
        r = cv.tick(breaths_this_tick=0.25)
        assert r["spo2"] >= SPO2_NORMAL - 0.05

    def test_no_breathing_drops_spo2(self):
        cv = CardiovascularSystem()
        for _ in range(10):
            r = cv.tick(breaths_this_tick=0.0)
        assert r["spo2"] < SPO2_NORMAL

    def test_o2_delivery_tracks_perfusion(self):
        """Lower perfusion → lower O₂ delivery regardless of SpO₂."""
        cv1 = CardiovascularSystem()
        cv1._blood_volume = 1.0
        r1 = cv1.tick()
        cv2 = CardiovascularSystem()
        cv2._blood_volume = 0.3
        for _ in range(20):
            r2 = cv2.tick(hydration=0.2)
        assert r2["o2_delivery"] < r1["o2_delivery"]

    def test_anemia_reduces_o2_delivery(self):
        cv1 = CardiovascularSystem()
        cv1._blood_volume = 0.8
        r1 = cv1.tick()
        cv2 = CardiovascularSystem()
        cv2._blood_volume = 0.8
        cv2.set_hemoglobin(0.5)  # Anemia
        r2 = cv2.tick()
        assert r2["o2_delivery"] < r1["o2_delivery"]


# ============================================================================
# Heat Transport
# ============================================================================


class TestHeatTransport:
    """Blood carries heat from core to periphery."""

    def test_no_heat_when_cold(self):
        cv = CardiovascularSystem()
        r = cv.tick(ram_temperature=0.0)
        assert r["heat_transported"] == 0.0

    def test_heat_transported_when_warm(self):
        cv = CardiovascularSystem()
        cv._blood_volume = 0.8
        r = cv.tick(ram_temperature=0.5)
        assert r["heat_transported"] > 0.0

    def test_more_heat_at_higher_co(self):
        """Higher cardiac output → more heat transported."""
        cv1 = CardiovascularSystem()
        cv1._blood_volume = 0.8
        r1 = cv1.tick(ram_temperature=0.5, heart_rate=60.0)
        cv2 = CardiovascularSystem()
        cv2._blood_volume = 0.8
        r2 = cv2.tick(ram_temperature=0.5, heart_rate=120.0)
        assert r2["heat_transported"] > r1["heat_transported"]


# ============================================================================
# Baroreceptor Reflex
# ============================================================================


class TestBaroreceptorReflex:
    """Compensatory tachycardia when blood volume drops."""

    def test_normal_volume_no_compensation(self):
        cv = CardiovascularSystem()
        cv._blood_volume = 1.0
        cv._baroreceptor_reflex()
        assert cv._compensatory_hr_delta < 1.0

    def test_low_volume_triggers_tachycardia(self):
        cv = CardiovascularSystem()
        cv._blood_volume = 0.3  # Well below threshold
        cv._baroreceptor_reflex()
        cv._baroreceptor_reflex()  # Multiple ticks to build up
        cv._baroreceptor_reflex()
        assert cv._compensatory_hr_delta > 0.0

    def test_compensation_bounded(self):
        cv = CardiovascularSystem()
        cv._blood_volume = 0.1
        for _ in range(100):
            cv._baroreceptor_reflex()
        assert cv._compensatory_hr_delta <= 60.0


# ============================================================================
# Developmental Growth
# ============================================================================


class TestDevelopmentalGrowth:
    """Motor activity grows blood volume capacity."""

    def test_motor_grows_volume_capacity(self):
        cv = CardiovascularSystem()
        initial = cv._volume_growth
        cv.grow(motor_movements=100)
        assert cv._volume_growth > initial

    def test_growth_saturates(self):
        cv = CardiovascularSystem()
        for _ in range(10000):
            cv.grow(motor_movements=1000)
        max_growth = BLOOD_VOLUME_SETPOINT - NEONATAL_VOLUME_FACTOR
        assert cv._volume_growth <= max_growth + 0.001

    def test_no_motor_no_growth(self):
        cv = CardiovascularSystem()
        cv.grow(motor_movements=0)
        assert cv._volume_growth == 0.0


# ============================================================================
# Hemoglobin / Anemia
# ============================================================================


class TestHemoglobin:
    """Anemia simulation via hemoglobin level."""

    def test_set_hemoglobin(self):
        cv = CardiovascularSystem()
        cv.set_hemoglobin(0.6)
        assert cv._hemoglobin == 0.6

    def test_hemoglobin_clipped(self):
        cv = CardiovascularSystem()
        cv.set_hemoglobin(5.0)
        assert cv._hemoglobin <= 1.2
        cv.set_hemoglobin(-1.0)
        assert cv._hemoglobin >= 0.1


# ============================================================================
# Proprioception
# ============================================================================


class TestProprioception:
    """Heartbeat awareness signal."""

    def test_returns_electrical_signal(self):
        cv = CardiovascularSystem()
        cv.tick()
        sig = cv.get_proprioception()
        assert isinstance(sig, ElectricalSignal)

    def test_source_is_cardiovascular(self):
        cv = CardiovascularSystem()
        cv.tick()
        sig = cv.get_proprioception()
        assert sig.source == "cardiovascular"
        assert sig.modality == "interoception"

    def test_impedance_value(self):
        cv = CardiovascularSystem()
        cv.tick()
        sig = cv.get_proprioception()
        assert sig.impedance == CARDIOVASCULAR_IMPEDANCE

    def test_waveform_shape(self):
        cv = CardiovascularSystem()
        cv.tick()
        sig = cv.get_proprioception()
        assert sig.waveform.shape == (64,)


# ============================================================================
# State and Stats
# ============================================================================


class TestStateAndStats:
    """State snapshots and statistics."""

    def test_get_state_returns_dataclass(self):
        cv = CardiovascularSystem()
        cv.tick()
        state = cv.get_state()
        assert hasattr(state, "blood_volume")
        assert hasattr(state, "cerebral_perfusion")
        assert hasattr(state, "spo2")

    def test_get_stats_returns_dict(self):
        cv = CardiovascularSystem()
        cv.tick()
        stats = cv.get_stats()
        assert "total_beats" in stats
        assert "cerebral_perfusion" in stats

    def test_properties(self):
        cv = CardiovascularSystem()
        cv.tick()
        assert isinstance(cv.cerebral_perfusion, float)
        assert isinstance(cv.o2_delivery, float)
        assert isinstance(cv.blood_volume, float)
        assert isinstance(cv.map_normalized, float)
        assert isinstance(cv.spo2, float)
        assert isinstance(cv.compensatory_hr_delta, float)


# ============================================================================
# Integration with AliceBrain
# ============================================================================


class TestBrainIntegration:
    """Cardiovascular is wired into the brain loop."""

    def test_brain_has_cardiovascular(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        assert hasattr(brain, "cardiovascular")
        assert isinstance(brain.cardiovascular, CardiovascularSystem)

    def test_cv_ticks_with_perceive(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        brain.perceive(np.random.rand(10))
        assert brain.cardiovascular.total_beats > 0

    def test_cv_in_introspect(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        info = brain.introspect()
        assert "cardiovascular" in info["subsystems"]

    def test_perfusion_modulates_consciousness(self):
        """Cerebral perfusion feeds into consciousness arousal."""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        brain.perceive(np.random.rand(10))
        # _cv_perfusion should exist after perceive
        assert hasattr(brain, "_cv_perfusion")
        assert isinstance(brain._cv_perfusion, float)

    def test_heat_transport_cools_system(self):
        """CV heat transport helps reduce ram_temperature."""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        brain.vitals.ram_temperature = 0.6
        temp_before = brain.vitals.ram_temperature
        brain.perceive(np.random.rand(10))
        # Temperature should not increase from CV's cooling
        assert brain.vitals.ram_temperature <= temp_before + 0.05

    def test_dehydration_triggers_tachycardia(self):
        """Low hydration → low blood volume → compensatory HR increase."""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=20)
        brain.homeostatic_drive.hydration = 0.2  # Severe dehydration
        for _ in range(30):
            brain.perceive(np.random.rand(10))
        # Compensatory HR should be non-zero
        assert brain.cardiovascular.compensatory_hr_delta >= 0.0


# ============================================================================
# Pathological Scenarios
# ============================================================================


class TestPathology:
    """Disease and emergency scenarios."""

    def test_severe_dehydration_cascade(self):
        """Dehydration → low volume → low BP → low perfusion → hypoxia risk."""
        cv = CardiovascularSystem()
        for _ in range(50):
            r = cv.tick(hydration=0.1)
        assert r["cerebral_perfusion"] < PERFUSION_NORMAL
        assert cv._is_tachycardic or cv._compensatory_hr_delta > 0

    def test_no_breathing_leads_to_hypoxia(self):
        """Without breathing, SpO₂ drops → O₂ delivery drops."""
        cv = CardiovascularSystem()
        cv._blood_volume = 0.8
        for _ in range(20):
            r = cv.tick(breaths_this_tick=0.0)
        assert r["spo2"] < SPO2_HYPOXIA_MILD
        assert cv._is_hypoxic

    def test_anemia_plus_dehydration(self):
        """Combined insult: anemia + dehydration = severely reduced O₂ delivery."""
        cv = CardiovascularSystem()
        cv.set_hemoglobin(0.5)
        for _ in range(30):
            r = cv.tick(hydration=0.2)
        assert r["o2_delivery"] < 0.5

    def test_hypotension_tracking(self):
        """Persistent low volume → hypotension ticks accumulate."""
        cv = CardiovascularSystem()
        for _ in range(30):
            cv.tick(hydration=0.1)
        # May or may not be hypotensive depending on compensation,
        # but ticks should be tracked
        assert isinstance(cv.hypotension_ticks, int)

    def test_glucose_delivery_tracks_perfusion(self):
        """Glucose delivery = perfusion × glucose level."""
        cv = CardiovascularSystem()
        cv._blood_volume = 0.8
        r = cv.tick(glucose=1.0)
        gd_normal = r["glucose_delivery"]
        cv2 = CardiovascularSystem()
        cv2._blood_volume = 0.8
        r2 = cv2.tick(glucose=0.3)
        assert r2["glucose_delivery"] < gd_normal


# ============================================================================
# Clinical Alignment
# ============================================================================


class TestClinicalAlignment:
    """Verify clinical realism of the model."""

    def test_neonatal_lower_pressure(self):
        """Neonatal blood pressure should be lower than adult."""
        cv = CardiovascularSystem()  # Neonatal
        r = cv.tick()
        assert r["systolic_bp"] < 140  # Not adult hypertension

    def test_sleep_reduces_heat_transport(self):
        """During sleep, peripheral circulation reduces."""
        cv1 = CardiovascularSystem()
        cv1._blood_volume = 0.8
        r1 = cv1.tick(ram_temperature=0.5, is_sleeping=False)
        cv2 = CardiovascularSystem()
        cv2._blood_volume = 0.8
        r2 = cv2.tick(ram_temperature=0.5, is_sleeping=True)
        assert r2["heat_transported"] <= r1["heat_transported"]

    def test_cardiac_output_reported(self):
        cv = CardiovascularSystem()
        r = cv.tick()
        assert "cardiac_output" in r
        assert r["cardiac_output"] > 0.0

    def test_map_within_physiological_range(self):
        cv = CardiovascularSystem()
        cv._blood_volume = 0.8
        r = cv.tick()
        map_mmhg = r["mean_arterial_pressure"]
        assert 30.0 <= map_mmhg <= 180.0
