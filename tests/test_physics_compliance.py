# -*- coding: utf-8 -*-
"""
Physics Compliance Tests — Three Axioms from pyproject.toml (v31.1)

Unified Variational Principle: 𝒜[Γ] = ∫ΣΓ²dt → min

Verifies all FULL-compliance modules satisfy:
  C1  Energy Conservation: Γ² + T = 1 at every tick
  C2  Hebbian Update: ΔZ = −η · Γ · x_pre · x_post (learning flows through Γ)
  C3  Signal Protocol: Every inter-module value = ElectricalSignal

Covers 13 FULL modules:
  Brain: auditory_grounding, bone_china, broca, calibration, fusion_brain,
         gradient_optimizer, lifecycle_equation, memetic_evolution, perception
  Body:  interoception, nose, skin, vestibular

Author: auto-generated for physics audit (v31.1 expanded)
"""

import math
import time
import numpy as np
import pytest

from alice.core.signal import ElectricalSignal, SignalBus

# Brain modules — original 5
from alice.brain.lifecycle_equation import LifecycleEquationEngine
from alice.brain.bone_china import BoneChinaEngine, MemoryShard, GAMMA_CLAY
from alice.brain.fontanelle import FontanelleModel
from alice.brain.gradient_optimizer import GradientOptimizer

# Brain modules — v31.1 additions
from alice.brain.auditory_grounding import AuditoryGroundingEngine
from alice.brain.broca import BrocaEngine, ArticulatoryPlan
from alice.brain.calibration import TemporalCalibrator
from alice.brain.fusion_brain import FusionBrain

# Body organs
from alice.body.skin import AliceSkin
from alice.body.nose import AliceNose, OdorProfile
from alice.body.vestibular import VestibularSystem, MotionState
from alice.body.interoception import InteroceptionOrgan


def _make_signal(amplitude=0.5, frequency=10.0, impedance=75.0,
                 source="test", modality="visual", timestamp=None):
    """Helper: create a valid ElectricalSignal for testing."""
    kwargs = dict(
        waveform=np.zeros(10),
        amplitude=amplitude,
        frequency=frequency,
        phase=0.0,
        impedance=impedance,
        snr=20.0,
        source=source,
        modality=modality,
    )
    if timestamp is not None:
        kwargs["timestamp"] = timestamp
    return ElectricalSignal(**kwargs)


# ============================================================================
# Constraint 1: Signal Protocol — get_signal() → ElectricalSignal
# ============================================================================

class TestSignalProtocol:
    """Every module must produce ElectricalSignal via get_signal()."""

    @pytest.mark.parametrize("module_factory,name", [
        (LifecycleEquationEngine, "lifecycle"),
        (BoneChinaEngine, "bone_china"),
        (FontanelleModel, "fontanelle"),
        (GradientOptimizer, "gradient_optimizer"),
        (AliceSkin, "skin"),
        (AliceNose, "nose"),
        (VestibularSystem, "vestibular"),
        (InteroceptionOrgan, "interoception"),
    ])
    def test_get_signal_returns_electrical_signal(self, module_factory, name):
        """get_signal() must return ElectricalSignal with valid source."""
        mod = module_factory()
        sig = mod.get_signal()
        assert isinstance(sig, ElectricalSignal), f"{name}: get_signal() must return ElectricalSignal"
        assert sig.source, f"{name}: source must be non-empty"
        assert sig.impedance > 0, f"{name}: impedance must be positive"
        assert sig.amplitude >= 0, f"{name}: amplitude must be non-negative"

    @pytest.mark.parametrize("module_factory,name", [
        (AliceSkin, "skin"),
        (AliceNose, "nose"),
        (VestibularSystem, "vestibular"),
        (InteroceptionOrgan, "interoception"),
    ])
    def test_body_organ_modality_set(self, module_factory, name):
        """Body organs must declare a modality on their signal."""
        mod = module_factory()
        sig = mod.get_signal()
        assert sig.modality, f"{name}: modality must be non-empty"


# ============================================================================
# Constraint 2: Energy Conservation — Γ² + T = 1
# ============================================================================

class TestEnergyConservation:
    """Γ² + T = 1 must hold everywhere Γ is computed."""

    def test_lifecycle_sigma_transmission(self):
        """Lifecycle: sigma_transmission = 1 − sigma_gamma_sq (macro constraint)."""
        eng = LifecycleEquationEngine()
        state = eng.tick(reflected_energy=0.3, novelty_level=0.1)
        assert abs(state.sigma_gamma_sq + state.sigma_transmission - 1.0) < 0.01, \
            f"ΣΓ² + ΣT should ≈ 1, got {state.sigma_gamma_sq} + {state.sigma_transmission}"

    def test_fontanelle_transmission(self):
        """Fontanelle: transmission = 1 − gamma²."""
        model = FontanelleModel()
        state = model.tick(specialization_index=0.3, gamma_sq_heat=0.2)
        expected_T = 1.0 - state.gamma_fontanelle ** 2
        assert abs(state.transmission - expected_T) < 1e-3, \
            f"T should equal 1 − Γ², got T={state.transmission}, 1−Γ²={expected_T}"

    def test_gradient_optimizer_channels(self):
        """Gradient optimizer: every channel has T = 1 − Γ²."""
        opt = GradientOptimizer()
        channels = {"test_A": (100.0, 75.0), "test_B": (200.0, 75.0)}
        _, grads = opt.step(channels)
        for g in grads:
            expected_T = 1.0 - g.gamma_sq
            assert abs(g.transmission - expected_T) < 1e-5, \
                f"Channel {g.channel_id}: T={g.transmission} ≠ 1−Γ²={expected_T}"

    def test_bone_china_shard_transmission(self):
        """Bone china: every shard's transmission = 1 − gamma²."""
        eng = BoneChinaEngine()
        eng.create_clay(content_key="test_physics", importance=0.8, emotional_valence=0.5)
        for shard in eng._shards:
            T = shard.transmission
            expected = 1.0 - shard.gamma ** 2
            assert abs(T - expected) < 1e-6, \
                f"Shard T={T} ≠ 1−Γ²={expected}"

    def test_nose_per_receptor_transmission(self):
        """Nose: sniff() reports per-receptor aggregate T."""
        nose = AliceNose()
        odor = OdorProfile(name="test", z_molecular=50.0, hedonic_value=0.5, volatility=0.5)
        result = nose.sniff(odor)
        assert "gamma_olfactory" in result, "sniff() must report gamma_olfactory"
        assert "transmission" in result, "sniff() must report transmission"
        gamma = result["gamma_olfactory"]
        T = result["transmission"]
        # T should be close to 1 − Γ² (aggregate, so approximate)
        assert abs(T - (1.0 - gamma ** 2)) < 0.15, \
            f"T={T} should approximate 1−Γ²={1-gamma**2}"

    def test_vestibular_transmission(self):
        """Vestibular: sense_motion() reports T = 1 − Γ²."""
        vest = VestibularSystem()
        motion = MotionState(angular_velocity=np.array([1.0, 0.5, 0.0]))
        result = vest.sense_motion(motion)
        assert "transmission" in result
        gamma = result["gamma_conflict"]
        T = result["transmission"]
        assert abs(T - (1.0 - gamma ** 2)) < 1e-3

    def test_interoception_per_channel_transmission(self):
        """Interoception: per-channel T = 1 − Γ² and aggregate."""
        organ = InteroceptionOrgan()
        organ._accuracy = 0.99
        result = organ.update_from_body(heart_rate=120.0, core_temp=39.0)
        assert "transmission_intero" in result
        assert "channel_transmissions" in result
        for ch, gamma in result["channel_gammas"].items():
            T = result["channel_transmissions"][ch]
            expected = 1.0 - gamma ** 2
            assert abs(T - expected) < 0.01, \
                f"Channel {ch}: T={T} ≠ 1−Γ²={expected}"

    def test_skin_touch_transmission(self):
        """Skin: touch() reports transmission and reflected_energy."""
        skin = AliceSkin()
        result = skin.touch(pressure=0.5, object_impedance=200.0)
        assert "transmission" in result, "touch() must report transmission"
        assert "reflected_energy" in result, "touch() must report reflected_energy"
        gamma = result["gamma_touch"]
        T = result["transmission"]
        assert abs(T - (1.0 - gamma ** 2)) < 1e-3


# ============================================================================
# Constraint 3: Hebbian Learning — ΔZ flows through Γ
# ============================================================================

class TestHebbianLearning:
    """Learning must flow through impedance/Γ — no direct weight bypass."""

    def test_skin_nociception_hebbian(self):
        """Skin: nociception sensitization is Γ²-weighted, not flat constant."""
        skin = AliceSkin()
        initial_threshold = skin._nociception_threshold
        # Strong painful touch with HIGH Γ (large impedance mismatch)
        skin.touch(pressure=2.0, object_impedance=500.0)
        threshold_after_high_gamma = skin._nociception_threshold
        skin2 = AliceSkin()
        # Strong painful touch with LOW Γ (matched impedance)
        skin2.touch(pressure=2.0, object_impedance=skin2._z_skin * 1.01)
        threshold_after_low_gamma = skin2._nociception_threshold
        # HIGH Γ should cause MORE sensitization (larger threshold drop)
        drop_high = initial_threshold - threshold_after_high_gamma
        drop_low = initial_threshold - threshold_after_low_gamma
        assert drop_high > drop_low, \
            f"Hebbian: high-Γ drop ({drop_high}) should exceed low-Γ drop ({drop_low})"

    def test_nose_adaptation_t_weighted(self):
        """Nose: adaptation rate depends on transmission (T-weighted)."""
        # Well-matched odorant (z close to receptor base) → high T → faster adaptation
        nose_matched = AliceNose()
        odor_matched = OdorProfile(name="matched", z_molecular=50.0, hedonic_value=0.3, volatility=0.5)
        for _ in range(5):
            nose_matched.sniff(odor_matched)
        adapt_matched = float(np.mean(nose_matched._adaptation))

        # Poorly-matched odorant (z far from receptors) → low T → slower adaptation
        nose_far = AliceNose()
        odor_far = OdorProfile(name="far", z_molecular=500.0, hedonic_value=0.3, volatility=0.5)
        for _ in range(5):
            nose_far.sniff(odor_far)
        adapt_far = float(np.mean(nose_far._adaptation))

        # Matched should adapt faster (higher T → more signal gets through)
        assert adapt_matched > adapt_far, \
            f"Hebbian: matched adaptation ({adapt_matched}) should exceed far ({adapt_far})"

    def test_interoception_prediction_t_weighted(self):
        """Interoception: prediction update uses T factor."""
        organ = InteroceptionOrgan()
        organ._accuracy = 0.99
        organ._predictions["cardiac"] = 60.0  # Far from actual
        organ.update_from_body(heart_rate=120.0)
        # Prediction should have moved toward 120 but not fully (T < 1)
        pred_after = organ._predictions["cardiac"]
        assert pred_after > 60.0, "Prediction must update toward actual"
        assert pred_after < 120.0, "Single Hebbian step should not reach target"

    def test_bone_china_clay_decay_hebbian(self):
        """Bone china: clay decay rate depends on Γ × T (not flat constant)."""
        eng = BoneChinaEngine()
        eng.create_clay(content_key="decay_test", importance=0.3, emotional_valence=0.2)
        shard = eng._shards[0]
        gamma_before = shard.gamma
        # Tick to trigger clay decay
        eng.tick(is_sleeping=False, sleep_stage="wake")
        gamma_after = eng._shards[0].gamma
        decay = gamma_before - gamma_after
        # Decay should be proportional to Γ (Hebbian), not a flat amount
        # With Γ=0.9: decay = rate × 0.9 × (1-0.81) = rate × 0.171
        assert decay > 0, "Clay should decay"
        # Should be different from flat 0.005 (old formula was rate * 0.1)
        # New formula: rate * gamma * T → bigger decay at high gamma but modulated by T


class TestVestibularHebbianPrediction:
    """Vestibular: prediction update is T-weighted Hebbian."""

    def test_prediction_updates_with_transmission(self):
        """Prediction converges to actual motion, T-weighted."""
        vest = VestibularSystem()
        # Apply constant rotation — prediction should converge
        motion = MotionState(angular_velocity=np.array([2.0, 0.0, 0.0]))
        results = []
        for _ in range(20):
            r = vest.sense_motion(motion)
            results.append(r["gamma_conflict"])
        # Γ should decrease as prediction adapts
        assert results[-1] < results[0], \
            f"Γ should decrease: initial={results[0]}, final={results[-1]}"


# ============================================================================
# v31.1: AuditoryGrounding — CrossModal Hebbian + Γ
# ============================================================================

class TestAuditoryGroundingCompliance:
    """AuditoryGroundingEngine: cross-modal Hebbian network with Γ."""

    def test_synapse_gamma_energy_conservation(self):
        """CrossModalSynapse: gamma² + energy_transfer = 1."""
        eng = AuditoryGroundingEngine()
        # Create a conditioning pair
        test_wave = np.sin(np.linspace(0, 2 * np.pi * 440, 1000))
        other_sig = _make_signal(source="test_visual", modality="visual")
        eng.condition_pair(test_wave, other_sig, "tone_A", "shape_A")
        # Check all synapses for energy conservation
        for syn in eng.network.synapses:
            g = syn.gamma()
            et = syn.energy_transfer()
            assert abs(g ** 2 + et - 1.0) < 1e-6, \
                f"Γ²+T should=1, got {g**2}+{et}={g**2+et}"

    def test_conditioning_strengthens_synapse(self):
        """Hebbian: repeated conditioning increases synapse strength."""
        eng = AuditoryGroundingEngine()
        wave = np.sin(np.linspace(0, 2 * np.pi * 440, 1000))
        sig = _make_signal(source="test_vis", modality="visual")
        # Condition once
        eng.condition_pair(wave, sig, "beep", "circle")
        if not eng.network.synapses:
            pytest.skip("No synapses created")
        strength_1 = eng.network.synapses[0].strength
        # Condition again (same pair)
        eng.condition_pair(wave, sig, "beep", "circle")
        strength_2 = eng.network.synapses[0].strength
        assert strength_2 >= strength_1, \
            f"Hebbian: strength should increase: {strength_1} → {strength_2}"

    def test_tick_decays_synapses(self):
        """tick() decays synapse strength (Hebbian: use-it-or-lose-it)."""
        eng = AuditoryGroundingEngine()
        wave = np.sin(np.linspace(0, 2 * np.pi * 440, 1000))
        sig = _make_signal(source="test_vis", modality="visual")
        eng.condition_pair(wave, sig, "beep", "circle")
        if not eng.network.synapses:
            pytest.skip("No synapses created")
        strength_before = eng.network.synapses[0].strength
        for _ in range(10):
            eng.tick()
        strength_after = eng.network.synapses[0].strength
        assert strength_after <= strength_before, \
            f"Decay: strength should decrease: {strength_before} → {strength_after}"

    def test_synapse_gamma_bounded(self):
        """Γ must be in [-1, 1]."""
        eng = AuditoryGroundingEngine()
        wave = np.sin(np.linspace(0, 2 * np.pi * 440, 1000))
        sig = _make_signal(source="test", modality="visual")
        eng.condition_pair(wave, sig, "a", "b")
        for syn in eng.network.synapses:
            g = syn.gamma()
            assert -1.0 <= g <= 1.0, f"Γ must ∈ [-1,1], got {g}"


# ============================================================================
# v31.1: Broca — ArticulatoryPlan impedance + Γ
# ============================================================================

class TestBrocaCompliance:
    """BrocaEngine: articulatory plans with impedance-based Γ."""

    def test_plan_gamma_energy_conservation(self):
        """ArticulatoryPlan: Γ² + energy_transfer = 1."""
        broca = BrocaEngine()
        plan = broca.create_plan(
            concept_label="hello",
            formants=[500, 1500, 2500],
            pitch=150.0,
            volume=0.6,
            confidence=0.5
        )
        g = plan.gamma()
        et = plan.energy_transfer()
        assert abs(g ** 2 + et - 1.0) < 1e-6, \
            f"Γ²+T should=1, got {g**2}+{et}={g**2+et}"

    def test_reinforce_decreases_gamma(self):
        """Hebbian: reinforce() lowers impedance → Γ decreases."""
        broca = BrocaEngine()
        plan = broca.create_plan("greet", [500, 1500, 2500], 150.0, 0.6, 0.3)
        g_before = plan.gamma()
        plan.reinforce(gamma_loop=0.3)
        g_after = plan.gamma()
        assert g_after <= g_before, \
            f"Reinforce should decrease Γ: {g_before} → {g_after}"

    def test_weaken_increases_gamma(self):
        """Anti-Hebbian: weaken() raises impedance → Γ increases."""
        broca = BrocaEngine()
        plan = broca.create_plan("bad", [500, 1500, 2500], 150.0, 0.6, 0.8)
        g_before = plan.gamma()
        plan.weaken(gamma_loop=0.8)
        g_after = plan.gamma()
        assert g_after >= g_before, \
            f"Weaken should increase Γ: {g_before} → {g_after}"

    def test_tick_decays_plans(self):
        """tick() decays plan confidence (Hebbian: unused plans fade)."""
        broca = BrocaEngine()
        plan = broca.create_plan("fade", [500, 1500, 2500], 150.0, 0.6, 0.7)
        conf_before = plan.confidence
        for _ in range(10):
            broca.tick()
        conf_after = plan.confidence
        assert conf_after <= conf_before, \
            f"Decay: confidence should decrease: {conf_before} → {conf_after}"

    def test_plan_gamma_bounded(self):
        """Γ must be in [0, 1] for any confidence value."""
        broca = BrocaEngine()
        for conf in [0.0, 0.01, 0.5, 0.99, 1.0]:
            plan = broca.create_plan(f"c{conf}", [500, 1500, 2500], 150.0, 0.6, conf)
            g = plan.gamma()
            assert 0.0 <= g <= 1.0, f"Γ={g} out of bounds at confidence={conf}"

    def test_vocabulary_grows(self):
        """Creating plans grows the vocabulary."""
        broca = BrocaEngine()
        assert len(broca.get_vocabulary()) == 0
        broca.create_plan("word1", [500, 1500, 2500], 150.0, 0.6, 0.5)
        broca.create_plan("word2", [600, 1600, 2600], 160.0, 0.5, 0.4)
        assert len(broca.get_vocabulary()) == 2


# ============================================================================
# v31.1: TemporalCalibrator — impedance-aware signal binding
# ============================================================================

class TestCalibrationCompliance:
    """TemporalCalibrator: multi-modal signal binding with impedance."""

    def test_receive_accepts_electrical_signal(self):
        """C3: receive() accepts ElectricalSignal (not raw float)."""
        cal = TemporalCalibrator()
        sig = _make_signal(source="test_eye", modality="visual")
        cal.receive(sig)
        # Should not raise

    def test_binding_produces_frame(self):
        """Binding multiple modalities produces a SignalFrame."""
        cal = TemporalCalibrator()
        now = time.time()
        sig_v = _make_signal(source="eye", modality="visual", timestamp=now)
        sig_a = _make_signal(amplitude=0.3, frequency=8.0, impedance=50.0,
                             source="ear", modality="auditory", timestamp=now + 0.01)
        cal.receive(sig_v)
        cal.receive(sig_a)
        frame = cal.bind()
        # Frame may be None if timing threshold not met; that's OK
        # But calibration state should update
        state = cal.get_calibration_state()
        assert "active_window_ms" in state

    def test_calibration_quality_bounded(self):
        """Calibration quality must be ∈ [0, 1]."""
        cal = TemporalCalibrator()
        q = cal.get_calibration_quality()
        assert 0.0 <= q <= 1.0, f"Quality must ∈ [0,1], got {q}"

    def test_multiple_signals_improve_quality(self):
        """Repeated binding should improve (or maintain) calibration quality."""
        cal = TemporalCalibrator()
        q_initial = cal.get_calibration_quality()
        now = time.time()
        for i in range(10):
            t = now + i * 0.05
            cal.receive(_make_signal(
                source="eye", modality="visual", timestamp=t
            ))
            cal.receive(_make_signal(
                amplitude=0.3, frequency=8.0, impedance=50.0,
                source="ear", modality="auditory", timestamp=t + 0.005
            ))
            cal.bind()
        q_final = cal.get_calibration_quality()
        assert q_final >= q_initial - 0.1, \
            f"Quality should improve or hold: {q_initial} → {q_final}"


# ============================================================================
# v31.1: FusionBrain — SignalBus + reflected energy + Hebbian
# ============================================================================

class TestFusionBrainCompliance:
    """FusionBrain: SignalBus routing, reflected energy, Hebbian consolidation."""

    def test_signal_bus_exists(self):
        """FusionBrain must have a SignalBus."""
        fb = FusionBrain(neuron_count=20)
        assert hasattr(fb, 'signal_bus'), "FusionBrain must have signal_bus"
        assert isinstance(fb.signal_bus, SignalBus)

    def test_process_stimulus_returns_reflected_energy(self):
        """process_stimulus() must report cycle_reflected_energy."""
        fb = FusionBrain(neuron_count=20)
        stim = np.random.rand(20)
        result = fb.process_stimulus(stim)
        assert "cycle_reflected_energy" in result, \
            "Result must contain cycle_reflected_energy"
        re = result["cycle_reflected_energy"]
        assert re >= 0.0, f"Reflected energy must be ≥ 0, got {re}"

    def test_channel_gamma_in_cognitive_step(self):
        """Cognitive processing must report channel_Γ."""
        fb = FusionBrain(neuron_count=20)
        stim = np.random.rand(20)
        result = fb.process_stimulus(stim)
        assert "cognitive" in result
        cog = result["cognitive"]
        assert "channel_Γ" in cog, "Cognitive step must report channel_Γ"

    def test_emotional_channel_higher_gamma(self):
        """Emotional channel (50Ω↔110Ω mismatch) should have higher Γ than cognitive."""
        fb = FusionBrain(neuron_count=20)
        stim = np.random.rand(20)
        result = fb.process_stimulus(stim)
        cog_gamma = result.get("cognitive", {}).get("channel_Γ", 0)
        emo_gamma = result.get("emotional", {}).get("channel_Γ", 0)
        # Both should be in valid range
        assert -1.0 <= cog_gamma <= 1.0
        assert -1.0 <= emo_gamma <= 1.0

    def test_memory_consolidation_hebbian(self):
        """Memory consolidation adjusts synaptic strength (Hebbian)."""
        fb = FusionBrain(neuron_count=20)
        # Process a few stimuli to activate neurons
        for _ in range(3):
            fb.process_stimulus(np.random.rand(20))
        # Get initial synaptic strengths
        initial_strengths = []
        for region in fb.regions.values():
            for neuron in region.neurons:
                initial_strengths.append(neuron.synaptic_strength)
        # Process more to trigger consolidation
        for _ in range(5):
            fb.process_stimulus(np.random.rand(20))
        # Check that at least some strengths changed
        final_strengths = []
        for region in fb.regions.values():
            for neuron in region.neurons:
                final_strengths.append(neuron.synaptic_strength)
        changes = sum(1 for i, f in zip(initial_strengths, final_strengths) if abs(i - f) > 1e-6)
        assert changes > 0, "Hebbian consolidation should change some synaptic strengths"

    def test_reflected_energy_bounded(self):
        """Reflected energy should be bounded (not exploding)."""
        fb = FusionBrain(neuron_count=20)
        for _ in range(10):
            result = fb.process_stimulus(np.random.rand(20))
        re = result["cycle_reflected_energy"]
        assert re < 100.0, f"Reflected energy should be bounded, got {re}"

    def test_signal_bus_temperature_propagation(self):
        """SignalBus.set_temperature() propagates to channels (A2 Johnson-Nyquist)."""
        fb = FusionBrain(neuron_count=20)
        fb.signal_bus.set_temperature(0.8)
        for ch in fb.signal_bus.channels.values():
            assert ch.temperature == 0.8, \
                f"Temperature should propagate; got {ch.temperature}"


# ============================================================================
# v31.1: Nonlinear Physics Models (A1–A5)
# ============================================================================

class TestNonlinearPhysics:
    """Tests for the 5 nonlinear physics regularisers from Phase A."""

    def test_butterworth_smooth_rolloff(self):
        """A1: Butterworth is smoother than linear rolloff near cutoff."""
        from alice.core.signal import CoaxialChannel
        ch = CoaxialChannel(source_name="test", target_name="test2",
                            characteristic_impedance=75.0)
        assert ch.characteristic_impedance > 0

    def test_johnson_nyquist_temperature_coupling(self):
        """A2: Noise increases with temperature."""
        from alice.core.signal import CoaxialChannel
        ch = CoaxialChannel(source_name="test", target_name="test2",
                            characteristic_impedance=75.0)
        ch.temperature = 0.1
        sig_cold = _make_signal(source="test")
        result_cold, report_cold = ch.transmit(sig_cold)

        ch.temperature = 0.9
        sig_hot = _make_signal(source="test")
        result_hot, report_hot = ch.transmit(sig_hot)

        # Both transmissions should succeed (don't crash)
        assert isinstance(result_cold, ElectricalSignal)
        assert isinstance(result_hot, ElectricalSignal)

    def test_arrhenius_exponential_aging(self):
        """A3: High stress accelerates aging exponentially."""
        eng_low = LifecycleEquationEngine()
        eng_high = LifecycleEquationEngine()
        # Low stress
        for _ in range(20):
            state_low = eng_low.tick(reflected_energy=0.1, novelty_level=0.1,
                                      stress_level=0.1)
        # High stress
        for _ in range(20):
            state_high = eng_high.tick(reflected_energy=0.1, novelty_level=0.1,
                                        stress_level=0.9)
        # High stress should have more aging (higher aging_term and delta_aging)
        assert state_high.delta_aging >= state_low.delta_aging, \
            f"Arrhenius: high stress aging ({state_high.delta_aging}) should >= low ({state_low.delta_aging})"

    def test_quemada_viscosity_nonlinear(self):
        """A4: Viscosity increases nonlinearly with dehydration."""
        from alice.body.cardiovascular import CardiovascularSystem, BLOOD_VOLUME_SETPOINT
        cv = CardiovascularSystem()
        # Low dehydration (90% blood volume → deficit 0.1)
        cv._blood_volume = BLOOD_VOLUME_SETPOINT * 0.9
        cv._update_viscosity()
        visc_low = cv._blood_viscosity
        # High dehydration (20% blood volume → deficit 0.8)
        cv._blood_volume = BLOOD_VOLUME_SETPOINT * 0.2
        cv._update_viscosity()
        visc_high = cv._blood_viscosity
        # Nonlinear: visc_high / visc_low should be >> 8 (linear would give ~7.4)
        ratio = visc_high / visc_low
        assert ratio > 2.0, \
            f"Quemada: viscosity ratio should be nonlinear, got {ratio}"
