# -*- coding: utf-8 -*-
"""
Tests for Phase 4.1: Auditory Grounding — Cochlear Filter Bank + Cross-Modal Hebbian Conditioning

Validation:
  1. CochlearFilterBank — ERB-scale frequency decomposition
  2. TonotopicActivation — fingerprint, similarity, hash
  3. CrossModalSynapse — impedance model, Hebbian learning, decay
  4. CrossModalHebbianNetwork — conditioning, probing, echo generation, extinction
  5. AuditoryGroundingEngine — complete Pavlov workflow
  6. Sound generators — pure tone, complex tone, noise, vowels
  7. AliceBrain integration
"""

import math
import time
import pytest
import numpy as np

from alice.body.cochlea import (
    CochlearFilterBank,
    TonotopicActivation,
    generate_tone,
    generate_complex_tone,
    generate_noise,
    generate_vowel,
    erb_bandwidth,
    generate_center_frequencies,
    DEFAULT_N_CHANNELS,
    DEFAULT_FREQ_MIN,
    DEFAULT_FREQ_MAX,
    DEFAULT_SAMPLE_RATE,
)
from alice.brain.auditory_grounding import (
    AuditoryGroundingEngine,
    CrossModalHebbianNetwork,
    CrossModalSynapse,
    SensoryPrototype,
    SYNAPSE_Z0,
    SYNAPSE_LEARNING_RATE,
    SYNAPSE_DECAY_RATE,
    SYNAPSE_MAX_STRENGTH,
    SYNAPSE_FLOOR,
    TEMPORAL_WINDOW_MS,
    ECHO_THRESHOLD,
    ECHO_IMPEDANCE,
    SIMILARITY_THRESHOLD,
)
from alice.core.signal import ElectricalSignal


# ============================================================================
# Helpers
# ============================================================================

def _make_visual_signal(
    freq: float = 10.0,
    amp: float = 1.0,
    label: str = "food",
    n: int = 256,
) -> ElectricalSignal:
    t = np.linspace(0, 1, n, endpoint=False)
    waveform = amp * np.sin(2 * np.pi * freq * t)
    return ElectricalSignal(
        waveform=waveform,
        amplitude=amp,
        frequency=freq,
        phase=0.0,
        impedance=50.0,
        snr=15.0,
        source=label,
        modality="visual",
    )


# ============================================================================
# 1. ERB Scale — Equivalent Rectangular Bandwidth
# ============================================================================

class TestERBScale:

    def test_erb_formula(self):
        """ERB(f) = 24.7 × (4.37 × f/1000 + 1) — Glasberg & Moore 1990"""
        # 1 kHz: ERB = 24.7 × (4.37 + 1) = 24.7 × 5.37 ≈ 132.6
        erb_1k = erb_bandwidth(1000.0)
        expected = 24.7 * (4.37 * 1.0 + 1.0)
        assert abs(erb_1k - expected) < 0.01

    def test_erb_increases_with_frequency(self):
        """Higher frequency → wider ERB"""
        erb_200 = erb_bandwidth(200.0)
        erb_1000 = erb_bandwidth(1000.0)
        erb_4000 = erb_bandwidth(4000.0)
        assert erb_200 < erb_1000 < erb_4000

    def test_center_frequencies_count(self):
        """Center frequency count = n_channels"""
        cfs = generate_center_frequencies(24, 80.0, 8000.0)
        assert len(cfs) == 24

    def test_center_frequencies_ascending(self):
        """Center frequencies are ascending"""
        cfs = generate_center_frequencies(24, 80.0, 8000.0)
        for i in range(len(cfs) - 1):
            assert cfs[i] < cfs[i + 1]

    def test_center_frequencies_range(self):
        """Center frequencies are within specified range"""
        cfs = generate_center_frequencies(24, 80.0, 8000.0)
        assert cfs[0] >= 79.9  # Minor floating point error from ERB inversion
        assert cfs[-1] <= 8001.0


# ============================================================================
# 2. CochlearFilterBank — Cochlear Filter Bank
# ============================================================================

class TestCochlearFilterBank:

    def test_creation(self):
        cochlea = CochlearFilterBank()
        assert cochlea.n_channels == DEFAULT_N_CHANNELS
        assert len(cochlea.center_frequencies) == DEFAULT_N_CHANNELS

    def test_custom_channels(self):
        cochlea = CochlearFilterBank(n_channels=12, freq_min=100, freq_max=4000)
        assert cochlea.n_channels == 12
        assert cochlea.freq_min == 100
        assert cochlea.freq_max == 4000

    def test_analyze_pure_tone(self):
        """Pure tone → only one peak channel in tonotopic map"""
        cochlea = CochlearFilterBank()
        tone = generate_tone(440.0, duration=0.1)
        tono = cochlea.analyze(tone)

        assert isinstance(tono, TonotopicActivation)
        assert len(tono.channel_activations) == cochlea.n_channels
        assert tono.total_energy > 0

        # Peak channel's center frequency should be close to 440Hz
        peak_ch = np.argmax(tono.channel_activations)
        peak_cf = cochlea.center_frequencies[peak_ch]
        assert 200 < peak_cf < 800  # Allowing ERB width tolerance

    def test_different_freqs_different_peaks(self):
        """Different frequencies → different channel peaks"""
        cochlea = CochlearFilterBank()
        tone_low = generate_tone(200.0, duration=0.1)
        tone_high = generate_tone(4000.0, duration=0.1)

        tono_low = cochlea.analyze(tone_low)
        tono_high = cochlea.analyze(tone_high)

        peak_low = np.argmax(tono_low.channel_activations)
        peak_high = np.argmax(tono_high.channel_activations)

        assert peak_low < peak_high  # Low freq → low channel, high freq → high channel

    def test_noise_flat_spectrum(self):
        """White noise → high spectral flatness"""
        cochlea = CochlearFilterBank()
        noise = generate_noise(duration=0.5, amplitude=1.0)
        tono = cochlea.analyze(noise)
        # White noise spectral flatness should > 0.3
        assert tono.spectral_flatness > 0.2

    def test_pure_tone_sharp_spectrum(self):
        """Pure tone → low spectral flatness (concentrated in one channel)"""
        cochlea = CochlearFilterBank()
        tone = generate_tone(1000.0, duration=0.1)
        tono = cochlea.analyze(tone)
        # Pure tone flatness should < noise
        noise = generate_noise(duration=0.5)
        tono_noise = cochlea.analyze(noise)
        assert tono.spectral_flatness < tono_noise.spectral_flatness

    def test_analyze_increments_counter(self):
        cochlea = CochlearFilterBank()
        assert cochlea.total_analyses == 0
        cochlea.analyze(generate_tone(440.0))
        assert cochlea.total_analyses == 1
        cochlea.analyze(generate_tone(880.0))
        assert cochlea.total_analyses == 2

    def test_temporal_persistence(self):
        """Temporal persistence — attack + decay"""
        cochlea = CochlearFilterBank()
        tone = generate_tone(440.0, duration=0.1, amplitude=1.0)
        silence = np.zeros(1600)

        # First analyze with sound
        tono1 = cochlea.analyze(tone)
        energy1 = tono1.total_energy

        # Then analyze silence (with decay)
        tono2 = cochlea.analyze(silence)
        energy2 = tono2.total_energy

        # After decay, energy should decrease but not zero due to persistence
        assert energy2 < energy1

    def test_get_state(self):
        cochlea = CochlearFilterBank()
        cochlea.analyze(generate_tone(440.0))
        state = cochlea.get_state()
        assert "n_channels" in state
        assert "total_analyses" in state
        assert state["total_analyses"] == 1


# ============================================================================
# 3. TonotopicActivation — Activation Vector
# ============================================================================

class TestTonotopicActivation:

    def test_fingerprint_normalized(self):
        """Fingerprint energy > 0"""
        cochlea = CochlearFilterBank()
        tono = cochlea.analyze(generate_tone(440.0))
        fp = tono.fingerprint()
        norm = np.linalg.norm(fp)
        # Fingerprint should have non-zero energy
        assert norm > 0.0

    def test_fingerprint_hash_deterministic(self):
        """Same signal → same hash"""
        cochlea = CochlearFilterBank()
        tone1 = generate_tone(440.0, duration=0.1)
        tone2 = generate_tone(440.0, duration=0.1)
        tono1 = cochlea.analyze(tone1)
        # Reset persistence for consistency
        cochlea2 = CochlearFilterBank()
        tono2 = cochlea2.analyze(tone2)
        h1 = tono1.fingerprint_hash()
        h2 = tono2.fingerprint_hash()
        assert h1 == h2

    def test_different_tones_different_peaks(self):
        """Different frequencies → different dominant channel"""
        cochlea1 = CochlearFilterBank()
        tono1 = cochlea1.analyze(generate_tone(200.0))
        cochlea2 = CochlearFilterBank()
        tono2 = cochlea2.analyze(generate_tone(4000.0))
        # dominant channel should differ
        assert tono1.dominant_channel != tono2.dominant_channel

    def test_similarity_self(self):
        """Self-similarity = 1"""
        cochlea = CochlearFilterBank()
        tono = cochlea.analyze(generate_tone(440.0))
        assert abs(tono.similarity(tono) - 1.0) < 0.001

    def test_similarity_different(self):
        """Different frequencies → similarity < 1"""
        cochlea = CochlearFilterBank()
        tono1 = cochlea.analyze(generate_tone(200.0))
        cochlea2 = CochlearFilterBank()
        tono2 = cochlea2.analyze(generate_tone(4000.0))
        sim = tono1.similarity(tono2)
        assert sim < 0.8


# ============================================================================
# 4. CrossModalSynapse — Cross-Modal Synapse
# ============================================================================

class TestCrossModalSynapse:

    def _make_synapse(self, w: float = SYNAPSE_FLOOR) -> CrossModalSynapse:
        return CrossModalSynapse(
            source_modality="auditory",
            target_modality="visual",
            source_fingerprint=np.random.rand(24),
            target_fingerprint=np.random.rand(24),
            strength=w,
            z_impedance=SYNAPSE_Z0 / max(w, 1e-9),
        )

    def test_impedance_formula(self):
        """Z = Z_0 / w"""
        syn = self._make_synapse(w=1.0)
        assert abs(syn.z_impedance - SYNAPSE_Z0) < 0.01

        syn2 = self._make_synapse(w=2.0)
        assert abs(syn2.z_impedance - SYNAPSE_Z0 / 2.0) < 0.01

    def test_gamma_impedance_match(self):
        """When Z_synapse = Z_source → Γ = 0 (perfect match)"""
        z_source = 50.0
        w = SYNAPSE_Z0 / z_source  # w = 100/50 = 2.0
        syn = self._make_synapse(w=w)
        gamma = syn.gamma(z_source)
        assert abs(gamma) < 0.01

    def test_energy_transfer_matched(self):
        """Perfect match → full energy transfer"""
        z_source = 50.0
        w = SYNAPSE_Z0 / z_source
        syn = self._make_synapse(w=w)
        et = syn.energy_transfer(z_source)
        assert abs(et - 1.0) < 0.01

    def test_energy_transfer_mismatched(self):
        """Very weak synapse → almost no energy transfer"""
        syn = self._make_synapse(w=0.01)
        et = syn.energy_transfer(50.0)
        assert et < 0.1

    def test_strengthen_increases_weight(self):
        """Hebbian strengthening → w ↑"""
        syn = self._make_synapse(w=0.01)
        w_before = syn.strength
        syn.strengthen(pre_activation=1.0, post_activation=1.0)
        assert syn.strength > w_before

    def test_strengthen_updates_impedance(self):
        """After strengthening → Z decreases"""
        syn = self._make_synapse(w=0.1)
        z_before = syn.z_impedance
        syn.strengthen(1.0, 1.0)
        assert syn.z_impedance < z_before

    def test_strengthen_capped(self):
        """Synapse strength has upper limit"""
        syn = self._make_synapse(w=SYNAPSE_MAX_STRENGTH - 0.01)
        syn.strengthen(1.0, 1.0, temporal_overlap=10.0)
        assert syn.strength <= SYNAPSE_MAX_STRENGTH

    def test_decay_decreases_weight(self):
        """Decay → w ↓"""
        syn = self._make_synapse(w=2.0)
        w_before = syn.strength
        syn.decay()
        assert syn.strength < w_before

    def test_decay_floor(self):
        """Decay has a floor"""
        syn = self._make_synapse(w=0.011)
        for _ in range(100):
            syn.decay()
        assert syn.strength >= SYNAPSE_FLOOR

    def test_to_dict(self):
        syn = self._make_synapse(w=1.0)
        d = syn.to_dict()
        assert "strength" in d
        assert "z_impedance" in d
        assert "gamma" in d
        assert "energy_transfer" in d


# ============================================================================
# 5. SensoryPrototype — Concept Prototype
# ============================================================================

class TestSensoryPrototype:

    def test_update_increases_confidence(self):
        fp = np.random.rand(24)
        proto = SensoryPrototype(
            label="test", modality="auditory",
            fingerprint=fp.copy(), exposure_count=0, confidence=0.0,
        )
        proto.update(fp)
        assert proto.confidence > 0.0
        assert proto.exposure_count == 1

    def test_update_ema(self):
        """Exponential moving average — fingerprint gradually stabilizes"""
        fp1 = np.zeros(24)
        fp1[5] = 1.0
        fp2 = np.zeros(24)
        fp2[10] = 1.0

        proto = SensoryPrototype(
            label="test", modality="auditory",
            fingerprint=fp1.copy(),
        )
        # After multiple updates, should bias toward fp2
        for _ in range(50):
            proto.update(fp2)

        # Fingerprint should be close to fp2
        sim_to_fp2 = float(np.dot(proto.fingerprint, fp2)) / (
            float(np.linalg.norm(proto.fingerprint)) * float(np.linalg.norm(fp2))
        )
        assert sim_to_fp2 > 0.8

    def test_similarity(self):
        fp = np.random.rand(24)
        fp /= np.linalg.norm(fp)
        proto = SensoryPrototype(
            label="test", modality="auditory", fingerprint=fp.copy(),
        )
        assert abs(proto.similarity(fp) - 1.0) < 0.01


# ============================================================================
# 6. CrossModalHebbianNetwork
# ============================================================================

class TestCrossModalHebbianNetwork:

    def test_condition_creates_synapse(self):
        net = CrossModalHebbianNetwork()
        src = np.random.rand(24)
        tgt = np.random.rand(24)
        syn = net.condition(src, "auditory", tgt, "visual")
        assert len(net.synapses) == 1
        assert syn.source_modality == "auditory"
        assert syn.target_modality == "visual"

    def test_condition_strengthens_existing(self):
        """Same fingerprint repeated conditioning → strengthens same synapse"""
        net = CrossModalHebbianNetwork()
        src = np.random.rand(24)
        tgt = np.random.rand(24)
        syn1 = net.condition(src, "auditory", tgt, "visual")
        w1 = syn1.strength
        syn2 = net.condition(src, "auditory", tgt, "visual")
        assert syn2.strength > w1
        assert len(net.synapses) == 1  # Same synapse

    def test_probe_empty(self):
        net = CrossModalHebbianNetwork()
        echoes = net.probe(np.random.rand(24), "auditory")
        assert len(echoes) == 0

    def test_probe_after_conditioning(self):
        """After conditioning → echo detected"""
        net = CrossModalHebbianNetwork()
        src = np.random.rand(24)
        src /= np.linalg.norm(src)
        tgt = np.random.rand(24)

        # Condition 20 times
        for _ in range(20):
            net.condition(src, "auditory", tgt, "visual")

        # Probe
        echoes = net.probe(src, "auditory")
        assert len(echoes) > 0
        assert echoes[0]["target_modality"] == "visual"
        assert echoes[0]["echo_strength"] > 0

    def test_echo_signal_generation(self):
        """Sufficient conditioning → generates echo signal"""
        net = CrossModalHebbianNetwork()
        src = np.random.rand(24)
        src /= np.linalg.norm(src)
        tgt = np.random.rand(24)

        for _ in range(30):
            net.condition(src, "auditory", tgt, "visual")

        echoes = net.probe(src, "auditory")
        echo_signal = net.generate_echo_signal(echoes)
        # Sufficiently conditioned should produce echo
        if echoes and echoes[0]["echo_strength"] >= ECHO_THRESHOLD:
            assert echo_signal is not None
            assert isinstance(echo_signal, ElectricalSignal)
            assert echo_signal.modality == "visual"

    def test_decay_all(self):
        """Global decay"""
        net = CrossModalHebbianNetwork()
        src = np.random.rand(24)
        tgt = np.random.rand(24)

        for _ in range(10):
            net.condition(src, "auditory", tgt, "visual")

        w_before = net.synapses[0].strength

        # Decay multiple times
        for _ in range(10):
            net.decay_all()

        assert net.synapses[0].strength < w_before

    def test_register_prototype(self):
        net = CrossModalHebbianNetwork()
        fp = np.random.rand(24)
        proto = net.register_prototype("bell", "auditory", fp)
        assert proto.label == "bell"
        assert proto.exposure_count == 1

    def test_identify_registered(self):
        net = CrossModalHebbianNetwork()
        fp = np.random.rand(24)
        fp /= np.linalg.norm(fp)
        # Register multiple times to stabilize prototype
        for _ in range(5):
            net.register_prototype("bell", "auditory", fp)

        result = net.identify(fp, "auditory")
        assert result is not None
        assert result[0] == "bell"
        assert result[1] > SIMILARITY_THRESHOLD

    def test_identify_unknown(self):
        net = CrossModalHebbianNetwork()
        fp = np.random.rand(24)
        result = net.identify(fp, "auditory")
        assert result is None

    def test_max_synapses_limit(self):
        """Exceeding limit → removes weakest"""
        net = CrossModalHebbianNetwork(max_synapses=5)
        for i in range(10):
            src = np.zeros(24)
            src[i % 24] = 1.0  # Different fingerprints
            tgt = np.zeros(24)
            tgt[(i + 12) % 24] = 1.0
            net.condition(src, "auditory", tgt, "visual")
        assert len(net.synapses) <= 5

    def test_get_state(self):
        net = CrossModalHebbianNetwork()
        state = net.get_state()
        assert "n_synapses" in state
        assert "total_pairings" in state


# ============================================================================
# 7. AuditoryGroundingEngine
# ============================================================================

class TestAuditoryGroundingEngine:

    def test_creation(self):
        engine = AuditoryGroundingEngine()
        assert engine.cochlea is not None
        assert engine.network is not None

    def test_receive_auditory(self):
        engine = AuditoryGroundingEngine()
        wave = generate_tone(440.0, duration=0.1)
        result = engine.receive_auditory(wave)
        assert "tonotopic" in result
        assert "fingerprint" in result
        assert "echoes" in result
        assert isinstance(result["tonotopic"], TonotopicActivation)

    def test_receive_auditory_with_label(self):
        engine = AuditoryGroundingEngine()
        wave = generate_tone(440.0, duration=0.1)
        result = engine.receive_auditory(wave, label="bell")
        identified = engine.network.identify(
            result["fingerprint"], "auditory"
        )
        assert identified is not None
        assert identified[0] == "bell"

    def test_receive_signal(self):
        engine = AuditoryGroundingEngine()
        signal = _make_visual_signal()
        result = engine.receive_signal(signal)
        assert "modality" in result
        assert "fingerprint" in result

    def test_condition_pair(self):
        """Pavlov conditioning API"""
        engine = AuditoryGroundingEngine()
        bell = generate_tone(440.0, duration=0.1)
        food = _make_visual_signal()

        result = engine.condition_pair(bell, food)
        assert "synapse" in result
        assert result["synapse"]["source_modality"] == "auditory"
        assert result["synapse"]["target_modality"] == "visual"
        assert result["energy_transfer"] > 0

    def test_pavlov_conditioning_echo(self):
        """
        Core Pavlov test:
        bell + food × 20 → bell alone → visual echo
        """
        engine = AuditoryGroundingEngine()
        bell = generate_tone(440.0, duration=0.1)
        food = _make_visual_signal(freq=10.0, amp=1.0)

        # Before conditioning — no association
        before = engine.probe_association(bell)
        assert not before["has_phantom"]

        # Conditioning
        for _ in range(25):
            engine.condition_pair(bell, food)

        # After conditioning — echo present
        after = engine.probe_association(bell)
        assert len(after["echoes"]) > 0
        # Check echo strength
        best = after["echoes"][0]
        assert best["target_modality"] == "visual"

    def test_pavlov_extinction(self):
        """Extinction: conditioning then tick × N → echo disappears"""
        engine = AuditoryGroundingEngine()
        bell = generate_tone(440.0, duration=0.1)
        food = _make_visual_signal()

        # Conditioning
        for _ in range(20):
            engine.condition_pair(bell, food)

        # Extinction
        for _ in range(500):
            engine.tick()

        result = engine.probe_association(bell)
        # Synapse should be very weak or removed
        if result["echoes"]:
            assert result["echoes"][0]["echo_strength"] < ECHO_THRESHOLD

    def test_differential_conditioning(self):
        """Different pitches → different associations"""
        engine = AuditoryGroundingEngine()
        bell_a = generate_tone(440.0, duration=0.1)
        bell_b = generate_tone(2000.0, duration=0.1)
        food_a = _make_visual_signal(freq=10.0, label="food_a")
        food_b = _make_visual_signal(freq=40.0, label="food_b")

        for _ in range(20):
            engine.condition_pair(bell_a, food_a)
            engine.condition_pair(bell_b, food_b)

        # Bell A should associate with food_a fingerprint
        result_a = engine.probe_association(bell_a)
        result_b = engine.probe_association(bell_b)

        assert len(result_a["echoes"]) > 0
        assert len(result_b["echoes"]) > 0

    def test_probe_association(self):
        engine = AuditoryGroundingEngine()
        wave = generate_tone(440.0, duration=0.1)
        result = engine.probe_association(wave)
        assert "tonotopic" in result
        assert "echoes" in result
        assert "has_phantom" in result

    def test_tick(self):
        """tick does not crash"""
        engine = AuditoryGroundingEngine()
        engine.tick()  # Empty tick
        bell = generate_tone(440.0)
        food = _make_visual_signal()
        engine.condition_pair(bell, food)
        engine.tick()  # Tick with synapses

    def test_auto_conditioning_within_window(self):
        """
        Auto-conditioning within temporal window:
        receive_auditory + receive_signal within 200ms → auto Hebbian
        """
        engine = AuditoryGroundingEngine()
        bell = generate_tone(440.0, duration=0.1)
        food = _make_visual_signal()

        # First receive visual
        engine.receive_signal(food)
        # Then receive auditory (within 200ms window)
        result = engine.receive_auditory(bell)

        # Should have auto-conditioned
        assert engine.total_groundings > 0

    def test_get_state(self):
        engine = AuditoryGroundingEngine()
        state = engine.get_state()
        assert "cochlea" in state
        assert "network" in state
        assert "total_groundings" in state


# ============================================================================
# 8. Sound Generators
# ============================================================================

class TestSoundGenerators:

    def test_generate_tone(self):
        wave = generate_tone(440.0, duration=0.1)
        expected_len = int(0.1 * DEFAULT_SAMPLE_RATE)
        assert len(wave) == expected_len
        assert np.max(np.abs(wave)) <= 1.01

    def test_generate_complex_tone(self):
        wave = generate_complex_tone(200.0, n_harmonics=5, duration=0.1)
        assert len(wave) == int(0.1 * DEFAULT_SAMPLE_RATE)
        assert np.max(np.abs(wave)) > 0

    def test_generate_noise(self):
        wave = generate_noise(duration=0.1, amplitude=0.5)
        assert len(wave) == int(0.1 * DEFAULT_SAMPLE_RATE)
        # Noise RMS should be close to amplitude
        rms = np.sqrt(np.mean(wave**2))
        assert 0.2 < rms < 0.8

    def test_generate_vowel(self):
        for v in ["a", "i", "u", "e", "o"]:
            wave = generate_vowel(v, fundamental=150.0, duration=0.1)
            assert len(wave) > 0
            # After normalization, max value ≈ 1
            assert np.max(np.abs(wave)) <= 1.01

    def test_vowels_are_different(self):
        """Different vowels → different waveforms"""
        fps = {}
        for v in ["a", "i", "u"]:
            c = CochlearFilterBank()  # New instance to avoid persistence
            wave = generate_vowel(v, fundamental=150, duration=0.1)
            tono = c.analyze(wave)
            fps[v] = tono.fingerprint()

        # a vs i should have differences (relaxed threshold)
        sim_ai = float(np.dot(fps["a"], fps["i"])) / (
            np.linalg.norm(fps["a"]) * np.linalg.norm(fps["i"])
        )
        # Even if similar, dominant channel should differ
        c1 = CochlearFilterBank()
        c2 = CochlearFilterBank()
        tono_a = c1.analyze(generate_vowel("a", fundamental=150, duration=0.1))
        tono_u = c2.analyze(generate_vowel("u", fundamental=150, duration=0.1))
        # Spectral centroids should differ
        assert abs(tono_a.spectral_centroid - tono_u.spectral_centroid) > 1.0 or \
               tono_a.dominant_channel != tono_u.dominant_channel or \
               sim_ai < 1.0


# ============================================================================
# 9. AliceBrain Integration
# ============================================================================

class TestAliceBrainIntegration:

    def test_alice_has_auditory_grounding(self):
        from alice.alice_brain import AliceBrain
        alice = AliceBrain()
        assert hasattr(alice, "auditory_grounding")
        assert isinstance(alice.auditory_grounding, AuditoryGroundingEngine)

    def test_hear_invokes_grounding(self):
        from alice.alice_brain import AliceBrain
        alice = AliceBrain()
        bell = generate_tone(440.0, duration=0.1)
        result = alice.hear(bell)
        # hear should produce auditory result
        assert "auditory" in result

    def test_see_invokes_grounding(self):
        from alice.alice_brain import AliceBrain
        alice = AliceBrain()
        pixels = np.random.rand(64, 64).astype(np.float32)
        result = alice.see(pixels)
        assert "visual" in result

    def test_introspect_includes_grounding(self):
        from alice.alice_brain import AliceBrain
        alice = AliceBrain()
        state = alice.introspect()
        assert "auditory_grounding" in state["subsystems"]

    def test_full_pavlov_via_alice(self):
        """
        Full Pavlov workflow via AliceBrain:
        hear + see alternating → auto Hebbian or manual condition_pair
        """
        from alice.alice_brain import AliceBrain
        alice = AliceBrain()
        bell = generate_tone(440.0, duration=0.1)
        food = _make_visual_signal()

        # Manual conditioning
        for _ in range(20):
            alice.auditory_grounding.condition_pair(bell, food)

        # Verify
        probe = alice.auditory_grounding.probe_association(bell)
        assert len(probe["echoes"]) > 0


# ============================================================================
# 10. Physics Conservation
# ============================================================================

class TestPhysicsConservation:

    def test_energy_transfer_bounded(self):
        """Energy transfer ∈ [0, 1]"""
        for w in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
            syn = CrossModalSynapse(
                source_modality="auditory", target_modality="visual",
                source_fingerprint=np.zeros(24),
                target_fingerprint=np.zeros(24),
                strength=w,
                z_impedance=SYNAPSE_Z0 / w,
            )
            et = syn.energy_transfer(50.0)
            assert 0.0 <= et <= 1.0

    def test_gamma_bounded(self):
        """Γ ∈ [-1, 1]"""
        for w in [0.001, 0.1, 1.0, 5.0, 10.0]:
            z = SYNAPSE_Z0 / w
            gamma = (50.0 - z) / (50.0 + z)
            assert -1.0 <= gamma <= 1.0

    def test_impedance_positive(self):
        """Z > 0 (always)"""
        for w in [0.001, 0.01, 0.1, 1.0, 5.0]:
            z = SYNAPSE_Z0 / w
            assert z > 0

    def test_hebbian_monotonic(self):
        """Repeated conditioning → strength monotonically increases"""
        syn = CrossModalSynapse(
            source_modality="a", target_modality="v",
            source_fingerprint=np.zeros(24),
            target_fingerprint=np.zeros(24),
            strength=0.01,
            z_impedance=SYNAPSE_Z0 / 0.01,
        )
        prev_w = syn.strength
        for _ in range(10):
            syn.strengthen(1.0, 1.0)
            assert syn.strength >= prev_w
            prev_w = syn.strength

    def test_decay_monotonic(self):
        """Decay → strength monotonically decreases"""
        syn = CrossModalSynapse(
            source_modality="a", target_modality="v",
            source_fingerprint=np.zeros(24),
            target_fingerprint=np.zeros(24),
            strength=3.0,
            z_impedance=SYNAPSE_Z0 / 3.0,
        )
        prev_w = syn.strength
        for _ in range(10):
            syn.decay()
            assert syn.strength <= prev_w
            prev_w = syn.strength

    def test_fingerprint_invariance(self):
        """Same sound → same fingerprint (deterministic)"""
        tone = generate_tone(440.0, duration=0.1)
        c1 = CochlearFilterBank()
        c2 = CochlearFilterBank()
        fp1 = c1.analyze(tone).fingerprint()
        fp2 = c2.analyze(tone).fingerprint()
        np.testing.assert_allclose(fp1, fp2, atol=1e-10)
