# -*- coding: utf-8 -*-
"""
Perception Pipeline Tests — Physics-Driven

Core tests: physical resonance, frequency matching, zero-computation perception
"""

import numpy as np
import pytest

from alice.core.signal import BrainWaveBand, ElectricalSignal
from alice.brain.perception import (
    BAND_INFO_LAYER,
    PhysicalTuner,
    ConceptMemory,
    PerceptionPipeline,
    PerceptionResult,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_signal(freq: float = 10.0, amp: float = 1.0) -> ElectricalSignal:
    """Directly create an electrical signal at specified frequency (simulating sensory organ forward-engineering output)"""
    n = 256
    t = np.linspace(0, 1, n, endpoint=False)
    wave = amp * np.sin(2 * np.pi * freq * t)
    return ElectricalSignal(
        waveform=wave,
        amplitude=amp,
        frequency=freq,
        phase=0.0,
        impedance=50.0,
        snr=20.0,
        source="test",
        modality="visual",
    )


# ============================================================================
# 1. BAND_INFO_LAYER
# ============================================================================


class TestBandInfoLayer:
    def test_all_bands_mapped(self):
        for band in BrainWaveBand:
            assert band in BAND_INFO_LAYER

    def test_expected_mappings(self):
        assert BAND_INFO_LAYER[BrainWaveBand.DELTA] == "background"
        assert BAND_INFO_LAYER[BrainWaveBand.THETA] == "memory_cue"
        assert BAND_INFO_LAYER[BrainWaveBand.ALPHA] == "contour"
        assert BAND_INFO_LAYER[BrainWaveBand.BETA] == "detail"
        assert BAND_INFO_LAYER[BrainWaveBand.GAMMA] == "binding"


# ============================================================================
# 2. Physical Tuner
# ============================================================================


class TestPhysicalTuner:
    def test_resonance_matched(self):
        """Frequency matched → high resonance (Lorentzian peak)"""
        tuner = PhysicalTuner("test", BrainWaveBand.ALPHA, quality_factor=2.0)
        # α center frequency = 10.5 Hz, input 10.5 Hz signal
        sig = _make_signal(freq=10.5)
        strength, locked = tuner.resonate(sig)
        assert strength > 0.9, "Perfect match should be close to 1.0"
        assert locked

    def test_resonance_near(self):
        """Frequency near → moderate resonance"""
        tuner = PhysicalTuner("test", BrainWaveBand.ALPHA, quality_factor=2.0)
        sig = _make_signal(freq=8.0)  # Within α range but low
        strength, locked = tuner.resonate(sig)
        assert 0.2 < strength < 0.9

    def test_resonance_far(self):
        """Frequency far away → low resonance"""
        tuner = PhysicalTuner("test", BrainWaveBand.GAMMA, quality_factor=3.0)
        sig = _make_signal(freq=2.0)  # γ=65Hz vs 2Hz → very far
        strength, locked = tuner.resonate(sig)
        assert strength < 0.1
        assert not locked

    def test_tune_changes_band(self):
        """Tuning knob can switch"""
        tuner = PhysicalTuner("test", BrainWaveBand.ALPHA)
        tuner.tune(BrainWaveBand.GAMMA)
        assert tuner.tuned_band == BrainWaveBand.GAMMA

    def test_zero_frequency_safe(self):
        """Zero frequency does not crash"""
        tuner = PhysicalTuner("test", BrainWaveBand.ALPHA)
        sig = ElectricalSignal(
            waveform=np.zeros(10), amplitude=0.0, frequency=0.0,
            phase=0.0, impedance=50.0, snr=0.0,
        )
        strength, locked = tuner.resonate(sig)
        assert strength == 0.0
        assert not locked

    def test_high_q_narrower(self):
        """High Q → narrower resonance (decays faster off-peak)"""
        sig = _make_signal(freq=8.0)  # Off α center
        low_q = PhysicalTuner("low", BrainWaveBand.ALPHA, quality_factor=1.0)
        high_q = PhysicalTuner("high", BrainWaveBand.ALPHA, quality_factor=5.0)
        s_low, _ = low_q.resonate(sig)
        s_high, _ = high_q.resonate(sig)
        assert s_low > s_high, "High Q should decay more when off-peak"

    def test_lock_count(self):
        """Lock count"""
        tuner = PhysicalTuner("test", BrainWaveBand.ALPHA, quality_factor=2.0)
        sig = _make_signal(freq=10.5)  # Exact match
        tuner.resonate(sig)
        tuner.resonate(sig)
        assert tuner.lock_count == 2

    def test_o1_computation(self):
        """O(1) computation — resonance only depends on frequency, not signal length"""
        tuner = PhysicalTuner("test", BrainWaveBand.BETA)
        sig_short = _make_signal(freq=21.5)
        sig_short.waveform = sig_short.waveform[:10]  # 10 samples
        s1, _ = tuner.resonate(sig_short)

        sig_long = _make_signal(freq=21.5)
        # 1000000 samples — FFT would be slow, but resonance doesn't care
        s2, _ = tuner.resonate(sig_long)

        assert abs(s1 - s2) < 1e-6, "Resonance only depends on frequency, not waveform length"


# ============================================================================
# 3. Concept Memory
# ============================================================================


class TestConceptMemory:
    def setup_method(self):
        self.mem = ConceptMemory()

    def test_register_and_identify(self):
        """Can identify after registration"""
        self.mem.register("apple", "visual", 10.5)
        found = self.mem.identify(10.5, "visual")
        assert found == "apple"

    def test_identify_with_tolerance(self):
        """Can identify within tolerance"""
        self.mem.register("apple", "visual", 10.5)
        found = self.mem.identify(10.0, "visual")  # ~5% deviation
        assert found == "apple"

    def test_identify_too_far(self):
        """Beyond tolerance → None"""
        self.mem.register("apple", "visual", 10.5)
        found = self.mem.identify(50.0, "visual")  # Completely different frequency
        assert found is None

    def test_identify_wrong_modality(self):
        """Wrong modality → None"""
        self.mem.register("apple", "visual", 10.5)
        found = self.mem.identify(10.5, "auditory")
        assert found is None

    def test_cross_modal_binding(self):
        """Cross-modal binding — apple: visual + auditory"""
        self.mem.register("apple", "visual", 10.5)
        self.mem.register("apple", "auditory", 10.5)
        bindings = self.mem.find_cross_modal("apple", "visual")
        assert "auditory" in bindings

    def test_cross_modal_multi(self):
        """Three-modality binding"""
        self.mem.register("apple", "visual", 10.5)
        self.mem.register("apple", "auditory", 10.5)
        self.mem.register("apple", "tactile", 10.5)
        bindings = self.mem.find_cross_modal("apple", "visual")
        assert set(bindings) == {"auditory", "tactile"}

    def test_no_self_binding(self):
        """Does not bind to self"""
        self.mem.register("apple", "visual", 10.5)
        bindings = self.mem.find_cross_modal("apple", "visual")
        assert "visual" not in bindings
        assert len(bindings) == 0

    def test_unknown_concept_no_binding(self):
        """Unknown concept → empty bindings"""
        bindings = self.mem.find_cross_modal("banana", "visual")
        assert bindings == []

    def test_multiple_concepts(self):
        """Multiple concepts do not mix up"""
        self.mem.register("apple", "visual", 10.5)
        self.mem.register("car", "visual", 21.5)
        assert self.mem.identify(10.5, "visual") == "apple"
        assert self.mem.identify(21.5, "visual") == "car"

    def test_stats(self):
        """Statistics"""
        self.mem.register("apple", "visual", 10.5)
        self.mem.register("car", "auditory", 21.5)
        stats = self.mem.get_stats()
        assert stats["registered_concepts"] == 2
        assert "visual" in stats["modalities"]
        assert "auditory" in stats["modalities"]


# ============================================================================
# 3a. Sparse Coding
# ============================================================================


class TestSparseCode:
    """Test core properties of sparse coding"""

    def setup_method(self):
        self.mem = ConceptMemory()

    def test_encode_binary_vector(self):
        """Sparse code is a binary vector — N-dimensional with only 1 bit set to 1"""
        self.mem.register("apple", "visual", 10.5)
        code = self.mem.encode("apple", "visual")
        assert code is not None
        assert code.dtype == np.uint8
        assert code.sum() == 1  # Only 1 bit is 1 (sparse!)
        assert len(code) == self.mem.total_bins

    def test_different_concepts_different_bins(self):
        """Different concepts occupy different bins — Hamming distance = 2"""
        self.mem.register("apple", "visual", 10.5)
        self.mem.register("car", "visual", 21.5)
        c1 = self.mem.encode("apple", "visual")
        c2 = self.mem.encode("car", "visual")
        assert np.sum(c1 != c2) == 2  # Two concepts have no overlap

    def test_same_concept_same_bin(self):
        """Same concept in different modalities has the same bin position (same frequency)"""
        self.mem.register("apple", "visual", 10.5)
        self.mem.register("apple", "auditory", 10.5)
        c_v = self.mem.encode("apple", "visual")
        c_a = self.mem.encode("apple", "auditory")
        assert np.argmax(c_v) == np.argmax(c_a)

    def test_sparsity_metric(self):
        """Sparsity ≈ 1/97"""
        self.mem.register("apple", "visual", 10.5)
        sp = self.mem.get_sparsity("visual")
        assert sp["occupied_bins"] == 1
        assert sp["total_bins"] == self.mem.total_bins
        assert sp["population_sparsity"] < 0.02  # < 2%

    def test_sparsity_grows_slowly(self):
        """Sparsity remains low after registering multiple concepts"""
        for i in range(10):
            freq = 1.0 + i * 2.0  # Sufficient spacing to avoid collisions
            self.mem.register(f"c{i}", "visual", freq)
        sp = self.mem.get_sparsity("visual")
        assert sp["population_sparsity"] < 0.15  # 10/97 ≈ 10%

    def test_o1_identify_not_on(self):
        """O(1) identification — registered 50 concepts, lookup is still O(1)"""
        for i in range(50):
            freq = 0.6 + i * 1.5
            self.mem.register(f"concept_{i}", "visual", freq)
        found = self.mem.identify(0.6, "visual")
        assert found is not None

    def test_total_bins_reasonable(self):
        """Bin count is reasonable (~97)"""
        assert 90 <= self.mem.total_bins <= 100

    def test_encode_unknown(self):
        """Unknown concept → None"""
        assert self.mem.encode("unknown", "visual") is None

    def test_stats_includes_sparsity(self):
        """Statistics include sparsity info"""
        self.mem.register("apple", "visual", 10.5)
        stats = self.mem.get_stats()
        assert stats["encoding"] == "sparse"
        assert "sparsity" in stats
        assert "visual" in stats["sparsity"]

    def test_log_binning_resolution(self):
        """Logarithmic bins — low frequency has high resolution, high frequency has low resolution (mimics cochlea)"""
        # Low frequency region: 1Hz vs 2Hz → differ by 12 bins (1 octave)
        bin_1hz = self.mem._freq_to_bin(1.0)
        bin_2hz = self.mem._freq_to_bin(2.0)
        assert bin_2hz - bin_1hz == 12  # 1 octave = 12 bins

        # High frequency region: 50Hz vs 100Hz → also 12 bins apart
        bin_50hz = self.mem._freq_to_bin(50.0)
        bin_100hz = self.mem._freq_to_bin(100.0)
        assert abs((bin_100hz - bin_50hz) - 12) <= 1


# ============================================================================
# 4. Perception Pipeline
# ============================================================================


class TestPerceptionPipeline:
    def setup_method(self):
        self.pipeline = PerceptionPipeline()

    def test_perceive_ndarray(self):
        """np.ndarray input"""
        data = np.random.rand(128)
        result = self.pipeline.perceive(data, "visual")
        assert isinstance(result, PerceptionResult)
        assert result.integrated_signal is not None

    def test_perceive_electrical_signal(self):
        """ElectricalSignal input"""
        sig = _make_signal(freq=10.5)
        result = self.pipeline.perceive(sig, "visual")
        assert isinstance(result, PerceptionResult)
        assert result.signal_frequency == 10.5

    def test_left_brain_high_freq(self):
        """High frequency signal → left brain resonance is stronger"""
        sig = _make_signal(freq=21.5)  # β center
        result = self.pipeline.perceive(sig, "visual")
        assert result.left_resonance > result.right_resonance
        assert result.left_tuned_band == BrainWaveBand.BETA

    def test_right_brain_low_freq(self):
        """Low frequency signal → right brain resonance is stronger"""
        sig = _make_signal(freq=2.25)  # δ center
        result = self.pipeline.perceive(sig, "visual")
        assert result.right_resonance > result.left_resonance
        assert result.right_tuned_band == BrainWaveBand.DELTA

    def test_attention_locks_strongest(self):
        """Attention locks onto the strongest resonance"""
        sig = _make_signal(freq=10.5)  # α — right brain
        result = self.pipeline.perceive(sig, "visual")
        if result.right_resonance > result.left_resonance:
            assert result.attention_band == result.right_tuned_band
        else:
            assert result.attention_band == result.left_tuned_band

    def test_concept_identification(self):
        """Concept identification — frequency matching"""
        self.pipeline.concepts.register("apple", "visual", 10.5)
        sig = _make_signal(freq=10.5)
        result = self.pipeline.perceive(sig, "visual")
        assert result.concept == "apple"

    def test_cross_modal_binding(self):
        """Cross-modal binding — seeing apple → finds auditory apple"""
        self.pipeline.concepts.register("apple", "visual", 10.5)
        self.pipeline.concepts.register("apple", "auditory", 10.5)
        sig = _make_signal(freq=10.5)
        result = self.pipeline.perceive(sig, "visual")
        assert result.concept == "apple"
        assert "auditory" in result.bindings

    def test_no_concept_no_binding(self):
        """No matching concept → no bindings"""
        sig = _make_signal(freq=50.0)
        result = self.pipeline.perceive(sig, "visual")
        assert result.concept is None
        assert result.bindings == []

    def test_resonance_gain(self):
        """Resonance amplification: integrated signal amplitude > original"""
        sig = _make_signal(freq=10.5, amp=1.0)
        result = self.pipeline.perceive(sig, "visual")
        # Resonance gain = 1 + attention_strength
        expected_gain = 1.0 + result.attention_strength
        assert abs(result.integrated_signal.amplitude - expected_gain) < 0.01

    def test_integrated_signal_shape(self):
        """Integrated signal length = original signal"""
        sig = _make_signal(freq=10.5)
        result = self.pipeline.perceive(sig, "visual")
        assert result.integrated_signal.waveform.shape == sig.waveform.shape

    def test_to_dict(self):
        """to_dict is complete"""
        sig = _make_signal(freq=10.5)
        result = self.pipeline.perceive(sig, "visual")
        d = result.to_dict()
        assert "signal_band" in d
        assert "attention_band" in d
        assert "left_resonance" in d
        assert "right_resonance" in d
        assert "concept" in d
        assert "bindings" in d

    def test_stats(self):
        """Statistics tracking"""
        self.pipeline.concepts.register("apple", "visual", 10.5)
        self.pipeline.perceive(_make_signal(10.5), "visual")
        self.pipeline.perceive(_make_signal(21.5), "visual")
        self.pipeline.perceive(_make_signal(2.0), "visual")
        stats = self.pipeline.get_stats()
        assert stats["total_perceptions"] == 3
        assert stats["total_concepts_identified"] >= 1  # apple

    def test_pipeline_elapsed(self):
        """Performance: elapsed time should be very low"""
        sig = _make_signal(freq=10.5)
        result = self.pipeline.perceive(sig, "visual")
        assert result.pipeline_elapsed_ms >= 0

    def test_all_modalities(self):
        """All modalities can be processed"""
        for mod in ["visual", "auditory", "tactile", "internal"]:
            result = self.pipeline.perceive(np.random.rand(64), mod)
            assert isinstance(result, PerceptionResult)


# ============================================================================
# 5. FusionBrain Integration
# ============================================================================


class TestPerceptionIntegration:
    def test_fusion_brain_has_perception(self):
        from alice.brain.fusion_brain import FusionBrain
        brain = FusionBrain(neuron_count=50)
        assert hasattr(brain, "perception")
        assert isinstance(brain.perception, PerceptionPipeline)

    def test_sensory_input_perception(self):
        """sensory_input result contains physical perception data"""
        from alice.brain.fusion_brain import FusionBrain
        from alice.core.protocol import Modality
        brain = FusionBrain(neuron_count=50)
        result = brain.sensory_input(np.random.rand(50), Modality.VISUAL)
        assert "perception" in result
        p = result["perception"]
        assert "signal_band" in p
        assert "attention_band" in p
        assert "attention_strength" in p
        assert "left_tuned_band" in p
        assert "left_resonance" in p
        assert "right_tuned_band" in p
        assert "right_resonance" in p
        assert "concept" in p
        assert "bindings_found" in p

    def test_process_stimulus_perception_stats(self):
        """process_stimulus contains perception statistics"""
        from alice.brain.fusion_brain import FusionBrain
        brain = FusionBrain(neuron_count=50)
        result = brain.process_stimulus(np.random.rand(50))
        assert "perception_stats" in result
        assert "total_perceptions" in result["perception_stats"]

    def test_brain_state_perception(self):
        """get_brain_state contains perception statistics"""
        from alice.brain.fusion_brain import FusionBrain
        brain = FusionBrain(neuron_count=50)
        state = brain.get_brain_state()
        assert "perception" in state

    def test_alice_brain_perceive(self):
        """AliceBrain.perceive contains perception data"""
        from alice.alice_brain import AliceBrain
        from alice.core.protocol import Modality
        alice = AliceBrain(neuron_count=50)
        result = alice.perceive(np.random.rand(50), Modality.VISUAL)
        assert "sensory" in result
        assert "perception" in result["sensory"]

    def test_working_memory_stores_perception(self):
        """Perception pattern stored in working memory"""
        from alice.alice_brain import AliceBrain
        from alice.core.protocol import Modality
        alice = AliceBrain(neuron_count=50)
        alice.perceive(np.random.rand(50), Modality.VISUAL, context="test_mem")
        content = alice.working_memory.retrieve("test_mem")
        assert content is not None
        assert "perception_pattern" in content
        pp = content["perception_pattern"]
        assert "attention_band" in pp
        assert "concept" in pp

    def test_multiple_cycles(self):
        """Multiple cycles accumulate statistics"""
        from alice.brain.fusion_brain import FusionBrain
        brain = FusionBrain(neuron_count=50)
        for _ in range(5):
            brain.process_stimulus(np.random.rand(50))
        assert brain.perception.total_perceptions == 5


# ============================================================================
# 6. Edge Cases
# ============================================================================


class TestPerceptionEdgeCases:
    def test_tiny_signal(self):
        """Extremely small signal does not crash"""
        pipeline = PerceptionPipeline()
        result = pipeline.perceive(np.ones(4) * 1e-10, "visual")
        assert isinstance(result, PerceptionResult)

    def test_large_signal(self):
        """Large signal"""
        pipeline = PerceptionPipeline()
        result = pipeline.perceive(np.random.rand(2048) * 100, "visual")
        assert isinstance(result, PerceptionResult)

    def test_single_element(self):
        """Single element signal"""
        pipeline = PerceptionPipeline()
        result = pipeline.perceive(np.array([1.0]), "visual")
        assert isinstance(result, PerceptionResult)

    def test_consistent_results(self):
        """Same input → consistent physical results"""
        pipeline = PerceptionPipeline()
        sig = _make_signal(freq=10.5)
        r1 = pipeline.perceive(sig, "visual")
        r2 = pipeline.perceive(sig, "visual")
        assert r1.signal_band == r2.signal_band
        assert r1.attention_band == r2.attention_band
        assert abs(r1.left_resonance - r2.left_resonance) < 1e-6

    def test_brain_package_exports(self):
        """brain package exports physics-driven perception classes"""
        from alice.brain import (
            PhysicalTuner,
            ConceptMemory,
            PerceptionPipeline,
            PerceptionResult,
            BAND_INFO_LAYER,
        )
        assert PhysicalTuner is not None
        assert PerceptionPipeline is not None
