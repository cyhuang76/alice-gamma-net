# -*- coding: utf-8 -*-
"""Tests for Phase 8: AliceEar"""
import numpy as np
import pytest
from alice.body.ear import (
    AliceEar, BRAINWAVE_FREQ_MIN, BRAINWAVE_FREQ_MAX,
    AUDITORY_NERVE_IMPEDANCE, AUDITORY_NERVE_SNR,
    COCHLEA_RESOLUTION,
)
from alice.core.signal import ElectricalSignal, BrainWaveBand


class TestEarBasics:
    def test_init_defaults(self):
        ear = AliceEar()
        assert ear.cochlea_resolution == COCHLEA_RESOLUTION
        assert ear.hearing_sensitivity == 1.0
        assert ear.total_listens == 0

    def test_init_custom(self):
        ear = AliceEar(cochlea_resolution=128, hearing_sensitivity=0.5)
        assert ear.cochlea_resolution == 128
        assert ear.hearing_sensitivity == 0.5

    def test_sensitivity_clamp(self):
        ear = AliceEar(hearing_sensitivity=10.0)
        assert ear.hearing_sensitivity == 2.0

    def test_adjust_sensitivity(self):
        ear = AliceEar()
        ear.adjust_sensitivity(0.3)
        assert ear.hearing_sensitivity == pytest.approx(0.3)


class TestHear:
    def test_returns_electrical_signal(self):
        ear = AliceEar()
        wave = np.sin(np.linspace(0, 2 * np.pi * 440, 1024))
        sig = ear.hear(wave)
        assert isinstance(sig, ElectricalSignal)

    def test_source_and_modality(self):
        ear = AliceEar()
        sig = ear.hear(np.random.randn(256))
        assert sig.source == "ear"
        assert sig.modality == "auditory"

    def test_frequency_in_brainwave_range(self):
        ear = AliceEar()
        sig = ear.hear(np.random.randn(512))
        assert BRAINWAVE_FREQ_MIN <= sig.frequency <= BRAINWAVE_FREQ_MAX

    def test_impedance_and_snr(self):
        ear = AliceEar()
        sig = ear.hear(np.random.randn(256))
        assert sig.impedance == AUDITORY_NERVE_IMPEDANCE
        assert sig.snr == AUDITORY_NERVE_SNR

    def test_waveform_resolution(self):
        ear = AliceEar(cochlea_resolution=64)
        sig = ear.hear(np.random.randn(256))
        assert len(sig.waveform) == 64

    def test_amplitude_positive(self):
        ear = AliceEar()
        sig = ear.hear(np.random.randn(256) * 5.0)
        assert sig.amplitude > 0.0

    def test_empty_input(self):
        ear = AliceEar()
        sig = ear.hear(np.array([]))
        assert isinstance(sig, ElectricalSignal)

    def test_counter_increments(self):
        ear = AliceEar()
        ear.hear(np.random.randn(64))
        ear.hear(np.random.randn(128))
        assert ear.total_listens == 2
        assert ear.total_samples == 192

    def test_sensitivity_affects_amplitude(self):
        ear_loud = AliceEar(hearing_sensitivity=2.0)
        ear_quiet = AliceEar(hearing_sensitivity=0.1)
        wave = np.sin(np.linspace(0, 2 * np.pi * 100, 256))
        amp_loud = ear_loud.hear(wave).amplitude
        amp_quiet = ear_quiet.hear(wave).amplitude
        assert amp_loud > amp_quiet


class TestAttendFrequencyBand:
    def test_returns_signal(self):
        ear = AliceEar()
        wave = np.random.randn(512)
        sig = ear.attend_frequency_band(wave, center_hz=1000.0, bandwidth_hz=200.0)
        assert isinstance(sig, ElectricalSignal)
        assert sig.source == "ear"

    def test_empty_input(self):
        ear = AliceEar()
        sig = ear.attend_frequency_band(np.array([]), 1000.0, 200.0)
        assert isinstance(sig, ElectricalSignal)


class TestCochleaSnapshot:
    def test_snapshot_keys(self):
        ear = AliceEar()
        snap = ear.get_cochlea_snapshot(np.random.randn(256))
        assert "resonated" in snap
        assert "displacement" in snap
        assert "spectrum" in snap
        assert "voltage" in snap
        assert "dominant_freq" in snap
        assert "band" in snap
        assert "amplitude" in snap


class TestEarStats:
    def test_stats_keys(self):
        ear = AliceEar()
        ear.hear(np.random.randn(64))
        stats = ear.get_stats()
        assert stats["total_listens"] == 1
        assert stats["cochlea_resolution"] == COCHLEA_RESOLUTION
        assert stats["impedance"] == AUDITORY_NERVE_IMPEDANCE
