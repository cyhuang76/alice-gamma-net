# -*- coding: utf-8 -*-
"""Tests for Phase 10: AutonomicNervousSystem"""
import numpy as np
import pytest
from alice.brain.autonomic import (
    AutonomicNervousSystem,
    RESTING_HEART_RATE, MAX_HEART_RATE, MIN_HEART_RATE,
    RESTING_BREATH_RATE, CORE_TEMPERATURE,
)
from alice.core.signal import ElectricalSignal


class TestAutonomicBasics:
    def test_init_defaults(self):
        ans = AutonomicNervousSystem()
        assert ans.sympathetic == pytest.approx(0.2)
        assert ans.parasympathetic == pytest.approx(0.3)
        assert ans.heart_rate == RESTING_HEART_RATE
        assert ans.energy == pytest.approx(1.0)

    def test_reset(self):
        ans = AutonomicNervousSystem()
        ans.sympathetic = 0.9
        ans.energy = 0.1
        ans.reset()
        assert ans.sympathetic == pytest.approx(0.2)
        assert ans.energy == pytest.approx(1.0)


class TestAutonomicTick:
    def test_tick_increments(self):
        ans = AutonomicNervousSystem()
        ans.tick()
        assert ans.total_ticks == 1

    def test_pain_raises_sympathetic(self):
        ans = AutonomicNervousSystem()
        for _ in range(20):
            ans.tick(pain_level=0.9)
        assert ans.sympathetic > 0.3  # Homeostasis pulls back, but pain should drive sympathetic up

    def test_calm_raises_parasympathetic(self):
        ans = AutonomicNervousSystem()
        for _ in range(20):
            ans.tick(pain_level=0.0, emotional_valence=0.8)
        assert ans.parasympathetic > 0.3

    def test_stress_raises_heart_rate(self):
        ans = AutonomicNervousSystem()
        for _ in range(30):
            ans.tick(pain_level=0.8, ram_temperature=0.8)
        assert ans.heart_rate > RESTING_HEART_RATE

    def test_sleep_raises_parasympathetic(self):
        ans = AutonomicNervousSystem()
        for _ in range(20):
            ans.tick(is_sleeping=True)
        assert ans.parasympathetic > 0.3

    def test_energy_depletes_under_stress(self):
        ans = AutonomicNervousSystem()
        initial = ans.energy
        for _ in range(50):
            ans.tick(pain_level=0.9, ram_temperature=0.9)
        assert ans.energy < initial

    def test_energy_recovers_at_rest(self):
        ans = AutonomicNervousSystem()
        ans.energy = 0.5
        for _ in range(50):
            ans.tick(pain_level=0.0, emotional_valence=0.5)
        assert ans.energy > 0.5

    def test_cortisol_accumulates(self):
        ans = AutonomicNervousSystem()
        for _ in range(30):
            ans.tick(pain_level=0.8)
        assert ans.cortisol > 0.03  # Cortisol has decay (CORTISOL_DECAY=0.98), steady-state value is lower

    def test_cortisol_decays(self):
        ans = AutonomicNervousSystem()
        ans.cortisol = 0.8
        for _ in range(50):
            ans.tick(pain_level=0.0)
        assert ans.cortisol < 0.8


class TestAutonomicBalance:
    def test_resting_balance(self):
        ans = AutonomicNervousSystem()
        balance = ans.get_autonomic_balance()
        # Initial: sym=0.2, para=0.3 → balance < 0
        assert balance < 0

    def test_stress_positive_balance(self):
        ans = AutonomicNervousSystem()
        for _ in range(30):
            ans.tick(pain_level=0.9, ram_temperature=0.9)
        assert ans.get_autonomic_balance() > 0

    def test_stress_level(self):
        ans = AutonomicNervousSystem()
        assert ans.get_stress_level() >= 0.0
        for _ in range(50):
            ans.tick(pain_level=0.9, ram_temperature=0.9)
        assert ans.get_stress_level() > 0.2


class TestAutonomicPupil:
    def test_pupil_range(self):
        ans = AutonomicNervousSystem()
        assert 0.0 <= ans.get_pupil_aperture() <= 1.0

    def test_stress_dilates_pupil(self):
        ans = AutonomicNervousSystem()
        rest_pupil = ans.get_pupil_aperture()
        for _ in range(30):
            ans.tick(pain_level=0.9)
        assert ans.get_pupil_aperture() > rest_pupil


class TestAutonomicSignal:
    def test_signal_type(self):
        ans = AutonomicNervousSystem()
        sig = ans.get_signal()
        assert isinstance(sig, ElectricalSignal)
        assert sig.source == "autonomic"
        assert sig.modality == "interoception"


class TestAutonomicStats:
    def test_vitals_keys(self):
        ans = AutonomicNervousSystem()
        ans.tick()
        v = ans.get_vitals()
        assert "sympathetic" in v
        assert "parasympathetic" in v
        assert "heart_rate" in v
        assert "energy" in v
        assert "cortisol" in v

    def test_waveforms(self):
        ans = AutonomicNervousSystem()
        for _ in range(5):
            ans.tick()
        wf = ans.get_waveforms(last_n=3)
        assert len(wf["sympathetic"]) == 3


class TestBreathing:
    def test_voluntary_slow_breath(self):
        ans = AutonomicNervousSystem()
        for _ in range(20):
            ans.tick(voluntary_breath=6.0)
        # Slow breathing → parasympathetic ↑
        assert ans.parasympathetic > 0.3
