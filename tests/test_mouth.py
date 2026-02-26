# -*- coding: utf-8 -*-
"""Tests for Phase 9: AliceMouth"""
import numpy as np
import pytest
from alice.body.mouth import (
    AliceMouth, VOCAL_FOLD_REST_FREQ, VOCAL_FOLD_MIN_FREQ, VOCAL_FOLD_MAX_FREQ,
    VOWEL_FORMANTS, MOUTH_IMPEDANCE, OUTPUT_SAMPLE_POINTS,
)
from alice.core.signal import ElectricalSignal


class TestMouthBasics:
    def test_init_defaults(self):
        m = AliceMouth()
        assert m.base_pitch == VOCAL_FOLD_REST_FREQ
        assert m.sample_points == OUTPUT_SAMPLE_POINTS
        assert m.total_utterances == 0

    def test_init_custom_pitch(self):
        m = AliceMouth(base_pitch=220.0)
        assert m.base_pitch == 220.0

    def test_pitch_clamped(self):
        m = AliceMouth(base_pitch=10000.0)
        assert m.base_pitch == VOCAL_FOLD_MAX_FREQ


class TestSpeak:
    def test_returns_dict(self):
        m = AliceMouth()
        result = m.speak(target_pitch=200.0, volume=0.5)
        assert isinstance(result, dict)
        assert "waveform" in result
        assert "final_pitch" in result
        assert "signal" in result

    def test_waveform_shape(self):
        m = AliceMouth(duration=0.01)  # 0.01s at 16kHz = 160 samples
        result = m.speak(target_pitch=150.0)
        assert len(result["waveform"]) == m._wave_samples

    def test_signal_is_electrical(self):
        m = AliceMouth()
        result = m.speak(target_pitch=150.0)
        sig = result["signal"]
        assert isinstance(sig, ElectricalSignal)
        assert sig.source == "mouth"
        assert sig.modality == "vocal"

    def test_pitch_converges(self):
        m = AliceMouth()
        result = m.speak(target_pitch=200.0, duration_steps=20)
        assert result["pitch_error"] < 50.0  # Should converge somewhat

    def test_volume_affects_output(self):
        m = AliceMouth()
        loud = m.speak(target_pitch=150.0, volume=1.0)
        m2 = AliceMouth()
        quiet = m2.speak(target_pitch=150.0, volume=0.1)
        assert loud["volume"] > quiet["volume"]

    def test_tremor_increases_with_temperature(self):
        m = AliceMouth()
        calm = m.speak(target_pitch=150.0, ram_temperature=0.0)
        m2 = AliceMouth()
        anxious = m2.speak(target_pitch=150.0, ram_temperature=0.9)
        assert anxious["tremor_intensity"] > calm["tremor_intensity"]

    def test_counter(self):
        m = AliceMouth()
        m.speak(target_pitch=150.0)
        m.speak(target_pitch=200.0)
        assert m.total_utterances == 2


class TestSayVowel:
    @pytest.mark.parametrize("vowel", ["a", "i", "u", "e", "o"])
    def test_all_vowels(self, vowel):
        m = AliceMouth()
        result = m.say_vowel(vowel)
        assert isinstance(result, dict)
        assert result["signal"].source == "mouth"

    def test_phoneme_counter(self):
        m = AliceMouth()
        m.say_vowel("a")
        m.say_vowel("i")
        assert m.total_phonemes == 2


class TestProprioception:
    def test_returns_signal(self):
        m = AliceMouth()
        sig = m.get_proprioception()
        assert isinstance(sig, ElectricalSignal)
        assert sig.source == "mouth"
        assert sig.modality == "proprioception"

    def test_impedance(self):
        m = AliceMouth()
        sig = m.get_proprioception()
        assert sig.impedance == MOUTH_IMPEDANCE


class TestMouthStats:
    def test_stats_keys(self):
        m = AliceMouth()
        m.speak(target_pitch=150.0)
        stats = m.get_stats()
        assert "total_utterances" in stats
        assert "base_pitch" in stats
        assert "current_pitch" in stats
        assert stats["total_utterances"] == 1
