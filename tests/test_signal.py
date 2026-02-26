# -*- coding: utf-8 -*-
"""
Unified Electrical Signal Framework Tests

Coverage:
1. ElectricalSignal creation & physical properties
2. BrainWaveBand frequency band classification
3. CoaxialChannel transmission & reflection
4. SignalBus bus topology
5. Protocol integration (MessagePacket ↔ ElectricalSignal)
6. FusionBrain coaxial channel integration
7. Pain loop integration (reflected energy → heat generation)
"""

import math
import numpy as np
import pytest

from alice.core.signal import (
    BrainWaveBand,
    ElectricalSignal,
    CoaxialChannel,
    TransmissionReport,
    SignalBus,
    REGION_IMPEDANCE,
)
from alice.core.protocol import MessagePacket, Modality, Priority
from alice.brain.fusion_brain import FusionBrain, BrainRegionType
from alice.alice_brain import SystemState, AliceBrain


# ============================================================================
# 1. BrainWaveBand
# ============================================================================


class TestBrainWaveBand:
    def test_frequency_ranges(self):
        """Each frequency band has the correct frequency range"""
        assert BrainWaveBand.DELTA.freq_range == (0.5, 4.0)
        assert BrainWaveBand.THETA.freq_range == (4.0, 8.0)
        assert BrainWaveBand.ALPHA.freq_range == (8.0, 13.0)
        assert BrainWaveBand.BETA.freq_range == (13.0, 30.0)
        assert BrainWaveBand.GAMMA.freq_range == (30.0, 100.0)

    def test_from_frequency(self):
        """Correctly determine band from frequency"""
        assert BrainWaveBand.from_frequency(2.0) == BrainWaveBand.DELTA
        assert BrainWaveBand.from_frequency(6.0) == BrainWaveBand.THETA
        assert BrainWaveBand.from_frequency(10.0) == BrainWaveBand.ALPHA
        assert BrainWaveBand.from_frequency(20.0) == BrainWaveBand.BETA
        assert BrainWaveBand.from_frequency(50.0) == BrainWaveBand.GAMMA

    def test_center_freq(self):
        """Center frequency calculation"""
        assert BrainWaveBand.ALPHA.center_freq == 10.5


# ============================================================================
# 2. ElectricalSignal
# ============================================================================


class TestElectricalSignal:
    def test_from_raw(self):
        """Create electrical signal from raw data"""
        data = np.random.rand(100)
        sig = ElectricalSignal.from_raw(data, source="test", modality="visual")

        assert sig.waveform.shape == (100,)
        assert sig.amplitude >= 0
        assert sig.frequency >= 0.5
        assert 0 <= sig.phase <= 2 * math.pi
        assert sig.impedance == 75.0  # Default value
        assert sig.source == "test"
        assert sig.modality == "visual"

    def test_from_raw_with_impedance(self):
        """Specify impedance"""
        sig = ElectricalSignal.from_raw(np.ones(10), impedance=110.0)
        assert sig.impedance == 110.0

    def test_from_neural_activity(self):
        """Create from neural activity"""
        acts = np.random.rand(50)
        sig = ElectricalSignal.from_neural_activity(acts, region_impedance=50.0)
        assert sig.source == "neural"
        assert sig.impedance == 50.0

    def test_band_property(self):
        """Band determination"""
        sig = ElectricalSignal(
            waveform=np.zeros(10),
            amplitude=1.0,
            frequency=10.0,  # Alpha
            phase=0.0,
            impedance=75.0,
            snr=20.0,
        )
        assert sig.band == BrainWaveBand.ALPHA

    def test_power_and_energy(self):
        """Power and energy calculation"""
        data = np.array([1.0, 2.0, 3.0])
        sig = ElectricalSignal.from_raw(data)
        assert sig.power == pytest.approx(np.mean(data**2), rel=1e-4)
        assert sig.energy == pytest.approx(np.sum(np.abs(data)), rel=1e-4)

    def test_rms(self):
        """RMS calculation"""
        data = np.array([1.0, -1.0, 1.0, -1.0])
        sig = ElectricalSignal.from_raw(data)
        assert sig.rms == pytest.approx(1.0, rel=1e-4)

    def test_attenuate(self):
        """Signal attenuation"""
        data = np.array([2.0, 4.0, 6.0])
        sig = ElectricalSignal.from_raw(data)
        att = sig.attenuate(0.5)

        np.testing.assert_allclose(att.waveform, data * 0.5)
        assert att.amplitude == sig.amplitude * 0.5

    def test_add_noise(self):
        """Add noise"""
        data = np.ones(1000)
        sig = ElectricalSignal.from_raw(data)
        noisy = sig.add_noise(0.1)

        # Waveform should differ
        assert not np.allclose(sig.waveform, noisy.waveform)
        # SNR should change
        assert noisy.snr != sig.snr

    def test_phase_shift(self):
        """Phase shift"""
        sig = ElectricalSignal.from_raw(np.ones(10))
        shifted = sig.phase_shift(math.pi)
        assert shifted.phase != sig.phase

    def test_resample(self):
        """Resample"""
        data = np.random.rand(100)
        sig = ElectricalSignal.from_raw(data)
        resampled = sig.resample(50)
        assert resampled.waveform.shape == (50,)

    def test_resample_same_size(self):
        """Resample to same size"""
        data = np.random.rand(100)
        sig = ElectricalSignal.from_raw(data)
        resampled = sig.resample(100)
        np.testing.assert_allclose(resampled.waveform, data.flatten())

    def test_to_dict(self):
        """Serialization"""
        sig = ElectricalSignal.from_raw(np.random.rand(10))
        d = sig.to_dict()
        assert "amplitude" in d
        assert "frequency" in d
        assert "band" in d
        assert "power" in d
        assert "rms" in d


# ============================================================================
# 3. CoaxialChannel
# ============================================================================


class TestCoaxialChannel:
    def test_reflection_coefficient_matched(self):
        """Impedance matched → Γ ≈ 0"""
        ch = CoaxialChannel("A", "B", characteristic_impedance=75.0)
        sig = ElectricalSignal.from_raw(np.ones(10), impedance=75.0)
        gamma = ch.reflection_coefficient(sig)
        assert gamma == pytest.approx(0.0, abs=1e-10)

    def test_reflection_coefficient_mismatched(self):
        """Impedance mismatched → Γ ≠ 0"""
        ch = CoaxialChannel("A", "B", characteristic_impedance=50.0)
        sig = ElectricalSignal.from_raw(np.ones(10), impedance=110.0)
        gamma = ch.reflection_coefficient(sig)
        # Γ = (110-50)/(110+50) = 60/160 = 0.375
        assert gamma == pytest.approx(0.375, rel=1e-4)

    def test_transmit_matched_low_loss(self):
        """Matched channel transmission → low loss"""
        ch = CoaxialChannel("A", "B", characteristic_impedance=75.0, length=1.0)
        data = np.ones(10) * 0.5
        sig = ElectricalSignal.from_raw(data, impedance=75.0)

        transmitted, report = ch.transmit(sig)
        assert report.impedance_matched  # Impedance matched
        assert report.reflected_power_ratio < 0.01  # Almost no reflection
        assert transmitted.power > 0  # Signal passes through

    def test_transmit_mismatched_reflection(self):
        """Mismatched channel → has reflection"""
        ch = CoaxialChannel("A", "B", characteristic_impedance=50.0, length=1.0)
        data = np.ones(10) * 0.5
        sig = ElectricalSignal.from_raw(data, impedance=110.0)

        transmitted, report = ch.transmit(sig)
        assert not report.impedance_matched
        assert report.reflected_power_ratio > 0.1  # Significant reflection
        assert report.reflected_energy > 0

    def test_attenuation_increases_with_length(self):
        """Attenuation increases with distance (verified via transmission report)"""
        sig = ElectricalSignal.from_raw(np.ones(10) * 5.0, impedance=75.0)

        ch_short = CoaxialChannel("A", "B", characteristic_impedance=75.0, length=0.5)
        ch_long = CoaxialChannel("A", "B", characteristic_impedance=75.0, length=5.0)

        _, report_short = ch_short.transmit(sig)
        _, report_long = ch_long.transmit(sig)

        # Longer channel has more attenuation
        assert report_long.attenuation_db > report_short.attenuation_db
        # Longer channel has lower transmission factor
        assert report_long.total_transmission_factor < report_short.total_transmission_factor

    def test_stats_tracking(self):
        """Channel statistics tracking"""
        ch = CoaxialChannel("A", "B", characteristic_impedance=75.0)
        sig = ElectricalSignal.from_raw(np.ones(10), impedance=75.0)

        ch.transmit(sig)
        ch.transmit(sig)
        ch.transmit(sig)

        stats = ch.get_stats()
        assert stats["transmissions"] == 3
        assert stats["channel"] == "A→B"


# ============================================================================
# 4. SignalBus
# ============================================================================


class TestSignalBus:
    def test_default_topology(self):
        """Default topology has 12 channels (6 bidirectional)"""
        bus = SignalBus.create_default_topology()
        assert len(bus.channels) == 12

    def test_send_existing_channel(self):
        """Send through existing channel"""
        bus = SignalBus.create_default_topology()
        sig = ElectricalSignal.from_raw(np.random.rand(10), impedance=50.0)

        transmitted, report = bus.send("somatosensory", "prefrontal", sig)
        assert transmitted is not None
        assert report is not None
        assert report.channel == "somatosensory→prefrontal"

    def test_send_nonexistent_channel(self):
        """Send through nonexistent channel → None"""
        bus = SignalBus()
        sig = ElectricalSignal.from_raw(np.random.rand(10))
        transmitted, report = bus.send("fake_a", "fake_b", sig)
        assert transmitted is None
        assert report is None

    def test_total_reflected_energy(self):
        """Bus accumulated reflected energy"""
        bus = SignalBus.create_default_topology()
        sig = ElectricalSignal.from_raw(np.ones(10) * 0.5, impedance=110.0)

        # Send to multiple channels (impedance mismatch)
        bus.send("somatosensory", "prefrontal", sig)
        bus.send("somatosensory", "limbic", sig)

        assert bus.get_total_reflected_energy() > 0

    def test_bus_summary(self):
        """Bus summary"""
        bus = SignalBus.create_default_topology()
        sig = ElectricalSignal.from_raw(np.ones(10) * 0.5, impedance=75.0)
        bus.send("somatosensory", "prefrontal", sig)

        summary = bus.get_bus_summary()
        assert summary["total_channels"] == 12
        assert summary["total_transmissions"] >= 1
        assert "bus_efficiency" in summary


# ============================================================================
# 5. Protocol Integration
# ============================================================================


class TestProtocolIntegration:
    def test_packet_from_electrical_signal(self):
        """MessagePacket can be created from ElectricalSignal"""
        esig = ElectricalSignal.from_raw(np.random.rand(50), impedance=75.0)
        packet = MessagePacket.from_signal(esig, Modality.VISUAL, Priority.NORMAL)

        assert packet.electrical_signal is not None
        assert packet.raw_data is not None
        assert packet.electrical_signal.impedance == 75.0
        assert packet.frequency_tag > 0

    def test_packet_from_ndarray_auto_converts(self):
        """np.ndarray is automatically converted to ElectricalSignal"""
        data = np.random.rand(50)
        packet = MessagePacket.from_signal(data, Modality.VISUAL, Priority.NORMAL)

        assert packet.electrical_signal is not None
        assert packet.raw_data is not None
        assert hasattr(packet.electrical_signal, "frequency")


# ============================================================================
# 6. FusionBrain Coaxial Integration
# ============================================================================


class TestFusionBrainCoaxial:
    def test_brain_has_signal_bus(self):
        """FusionBrain contains SignalBus"""
        fb = FusionBrain(neuron_count=20)
        assert fb.signal_bus is not None
        assert len(fb.signal_bus.channels) == 12

    def test_brain_regions_have_impedance(self):
        """Brain regions have correct characteristic impedance"""
        fb = FusionBrain(neuron_count=20)
        assert fb.regions[BrainRegionType.SOMATOSENSORY].impedance == 50.0
        assert fb.regions[BrainRegionType.PREFRONTAL].impedance == 75.0
        assert fb.regions[BrainRegionType.LIMBIC].impedance == 110.0
        assert fb.regions[BrainRegionType.MOTOR].impedance == 75.0

    def test_process_stimulus_returns_reflected_energy(self):
        """Processing stimulus returns reflected energy"""
        fb = FusionBrain(neuron_count=20)
        result = fb.process_stimulus(np.random.rand(20))

        assert "cycle_reflected_energy" in result
        assert "channel_reports" in result
        assert result["cycle_reflected_energy"] >= 0

    def test_cognitive_channel_has_gamma(self):
        """Cognitive channel reports reflection coefficient"""
        fb = FusionBrain(neuron_count=20)
        result = fb.process_stimulus(np.random.rand(20))

        assert "channel_Γ" in result["cognitive"]

    def test_emotional_channel_has_heat(self):
        """Emotional channel reports heat generation"""
        fb = FusionBrain(neuron_count=20)
        result = fb.process_stimulus(np.random.rand(20))

        assert "emotional_heat" in result["emotional"]

    def test_motor_has_dual_channels(self):
        """Motor execution has cognitive + emotional dual channels"""
        fb = FusionBrain(neuron_count=20)
        result = fb.process_stimulus(np.random.rand(20))

        assert "cognitive_channel_Γ" in result["motor"]
        assert "emotional_channel_Γ" in result["motor"]

    def test_signal_band_in_sensory(self):
        """Sensory input contains frequency band information"""
        fb = FusionBrain(neuron_count=20)
        result = fb.process_stimulus(np.random.rand(20))

        assert "signal_band" in result["sensory"]
        assert result["sensory"]["signal_band"] in ["delta", "theta", "alpha", "beta", "gamma"]

    def test_get_cycle_reflected_energy(self):
        """get_cycle_reflected_energy() returns last cycle's reflected energy"""
        fb = FusionBrain(neuron_count=20)
        fb.process_stimulus(np.random.rand(20))
        energy = fb.get_cycle_reflected_energy()
        assert energy >= 0

    def test_brain_report_includes_bus(self):
        """Report includes coaxial bus info"""
        fb = FusionBrain(neuron_count=20)
        fb.process_stimulus(np.random.rand(20))
        report = fb.generate_report()

        assert "Coaxial Bus" in report
        assert "Impedance mismatch rate" in report
        assert "Reflected energy" in report

    def test_brain_state_includes_impedance(self):
        """Brain state includes per-region impedance"""
        fb = FusionBrain(neuron_count=20)
        state = fb.get_brain_state()
        for region_info in state["regions"].values():
            assert "impedance" in region_info

    def test_accept_electrical_signal_input(self):
        """FusionBrain accepts ElectricalSignal as input"""
        fb = FusionBrain(neuron_count=20)
        esig = ElectricalSignal.from_raw(np.random.rand(20), impedance=50.0)
        result = fb.process_stimulus(esig)
        assert result["cycle"] == 1


# ============================================================================
# 7. Pain Loop Integration
# ============================================================================


class TestPainLoopIntegration:
    def test_reflected_energy_increases_temperature(self):
        """Reflected energy → temperature increase (cumulative reflections)"""
        state = SystemState()

        # Simulate sustained large reflected energy (combined with other stressors)
        for _ in range(5):
            state.tick(
                critical_queue_len=1,
                high_queue_len=0,
                total_queue_len=5,
                sensory_activity=0.3,
                emotional_valence=0.0,
                left_brain_activity=0.0,
                right_brain_activity=0.0,
                cycle_elapsed_ms=10.0,
                reflected_energy=1.0,
            )

        assert state.ram_temperature > 0

    def test_zero_reflection_minimal_heat(self):
        """With reflection vs without reflection → with reflection has higher temperature"""
        state1 = SystemState()
        state2 = SystemState()

        # Multiple ticks with base load to exceed cooling threshold
        for _ in range(5):
            state1.tick(1, 0, 5, 0.3, 0.0, 0.0, 0.0, 10.0, reflected_energy=0.0)
            state2.tick(1, 0, 5, 0.3, 0.0, 0.0, 0.0, 10.0, reflected_energy=0.5)

        assert state2.ram_temperature > state1.ram_temperature

    def test_alice_brain_full_loop(self):
        """AliceBrain full loop: signal → coaxial transmission → reflection → pain"""
        brain = AliceBrain(neuron_count=20)

        # Multiple perception stimuli
        for _ in range(5):
            result = brain.perceive(np.random.rand(20), Modality.VISUAL, Priority.NORMAL)

        vitals = brain.vitals.get_vitals()
        assert vitals["total_ticks"] == 5
        # Under normal stimulation, should stay healthy
        assert vitals["consciousness"] > 0.5

    def test_impedance_mismatch_causes_more_heat(self):
        """Severe impedance mismatch → more reflection → more heat"""
        state_matched = SystemState()
        state_mismatched = SystemState()

        # With base load to exceed cooling threshold
        for _ in range(10):
            state_matched.tick(1, 0, 5, 0.3, 0.0, 0.5, 0.5, 10.0, reflected_energy=0.001)

        for _ in range(10):
            state_mismatched.tick(1, 0, 5, 0.3, 0.0, 0.5, 0.5, 10.0, reflected_energy=0.5)

        assert state_mismatched.ram_temperature > state_matched.ram_temperature

        assert state_mismatched.ram_temperature > state_matched.ram_temperature


# ============================================================================
# 8. REGION_IMPEDANCE Reference Table
# ============================================================================


class TestRegionImpedance:
    def test_all_regions_defined(self):
        """All brain region impedances are defined"""
        for rt in BrainRegionType:
            assert rt.value in REGION_IMPEDANCE

    def test_impedance_values(self):
        """Impedance values match design"""
        assert REGION_IMPEDANCE["somatosensory"] == 50.0
        assert REGION_IMPEDANCE["prefrontal"] == 75.0
        assert REGION_IMPEDANCE["limbic"] == 110.0
        assert REGION_IMPEDANCE["motor"] == 75.0
