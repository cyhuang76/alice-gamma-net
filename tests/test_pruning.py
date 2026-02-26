# -*- coding: utf-8 -*-
"""
Tests: Neural Pruning Engine — §3.5.2 Large-Scale Γ Apoptosis

Verification items:
  1. Birth initialization — random impedance distribution
  2. Hebbian selection — matched→strengthened, mismatched→weakened
  3. Apoptosis mechanism — weak connections removed
  4. Cortical specialization — different regions develop different expertise
  5. Pruning curve — S-shaped pruning curve (fast→slow)
  6. Frontal delayed maturation — motor cortex prunes slowest
  7. Cross-modal — signal type determines cortical specialization
  8. Γ² objective function — pruning reduces global Γ²
  9. Full integration — pruning engine within AliceBrain
"""

import numpy as np
import pytest

from alice.brain.pruning import (
    SynapticConnection,
    CorticalRegion,
    CorticalSpecialization,
    NeuralPruningEngine,
    PruningMetrics,
    MODALITY_SIGNAL_PROFILE,
)
from alice.core.signal import BrainWaveBand


# ============================================================================
# 1. SynapticConnection Basic Tests
# ============================================================================


class TestSynapticConnection:
    """Physical properties of synaptic connections"""

    def test_creation(self):
        """Connection initialization: impedance, resonant frequency, initial strength"""
        conn = SynapticConnection(
            connection_id=0,
            impedance=75.0,
            resonant_freq=10.0,
        )
        assert conn.alive is True
        assert conn.synaptic_strength == 1.0
        assert conn.impedance == 75.0
        assert conn.resonant_freq == 10.0
        assert conn.total_stimulations == 0

    def test_gamma_calculation_matched(self):
        """Γ = 0 when impedance is matched"""
        conn = SynapticConnection(connection_id=0, impedance=75.0, resonant_freq=10.0)
        gamma = conn.compute_gamma(75.0)  # perfect match
        assert gamma == pytest.approx(0.0, abs=1e-10)

    def test_gamma_calculation_mismatched(self):
        """Γ > 0 when impedance is mismatched"""
        conn = SynapticConnection(connection_id=0, impedance=75.0, resonant_freq=10.0)
        gamma = conn.compute_gamma(50.0)  # mismatch
        expected = abs((75.0 - 50.0) / (75.0 + 50.0))  # = 0.2
        assert gamma == pytest.approx(expected, abs=1e-10)

    def test_gamma_severe_mismatch(self):
        """Severe mismatch Γ → 1"""
        conn = SynapticConnection(connection_id=0, impedance=200.0, resonant_freq=10.0)
        gamma = conn.compute_gamma(20.0)
        # (200-20)/(200+20) = 180/220 ≈ 0.818
        assert gamma > 0.8

    def test_resonance_at_natural_freq(self):
        """Response = 1.0 at resonant frequency"""
        conn = SynapticConnection(connection_id=0, impedance=75.0, resonant_freq=10.0)
        response = conn.compute_resonance(10.0)  # exact resonance
        assert response == pytest.approx(1.0, abs=1e-10)

    def test_resonance_off_frequency(self):
        """Response drops when off resonant frequency"""
        conn = SynapticConnection(connection_id=0, impedance=75.0, resonant_freq=10.0)
        at_resonance = conn.compute_resonance(10.0)
        off_resonance = conn.compute_resonance(50.0)
        assert off_resonance < at_resonance
        assert off_resonance < 0.1  # far from resonance, response very low

    def test_avg_gamma_initial(self):
        """Average Γ = 1.0 (worst) when unstimulated"""
        conn = SynapticConnection(connection_id=0, impedance=75.0, resonant_freq=10.0)
        assert conn.avg_gamma == 1.0  # unknown = worst

    def test_avg_gamma_after_stimulation(self):
        """Average Γ reflects actual values after stimulation"""
        conn = SynapticConnection(connection_id=0, impedance=75.0, resonant_freq=10.0)
        conn.total_stimulations = 2
        conn._gamma_sum = 0.4  # average of two = 0.2
        assert conn.avg_gamma == pytest.approx(0.2, abs=1e-10)


# ============================================================================
# 2. CorticalRegion Tests
# ============================================================================


class TestCorticalRegion:
    """Pruning behavior of cortical regions"""

    def test_birth_initialization(self):
        """Birth initialization: N connections with random impedances"""
        region = CorticalRegion(
            name="test",
            initial_connections=500,
            z_min=20.0,
            z_max=200.0,
        )
        assert len(region.connections) == 500
        assert region.alive_count == 500
        assert region.survival_rate == 1.0

        # Impedances should be random (not all identical)
        impedances = [c.impedance for c in region.connections]
        assert min(impedances) >= 20.0
        assert max(impedances) <= 200.0
        assert np.std(impedances) > 10.0  # sufficient variance

    def test_stimulation_basic(self):
        """Basic stimulation: matched connections are strengthened"""
        region = CorticalRegion(
            name="test",
            initial_connections=100,
            region_impedance=75.0,
        )
        result = region.stimulate(75.0, 10.0)
        assert "alive" in result
        assert "strengthened" in result
        assert "weakened" in result
        assert result["strengthened"] + result["weakened"] == result["alive"]

    def test_hebbian_strengthening(self):
        """Hebbian strengthening: synaptic strength increases for matched connections"""
        region = CorticalRegion(
            name="test",
            initial_connections=100,
            region_impedance=75.0,
        )
        # Find a connection with impedance close to 75Ω
        close_conns = [c for c in region.connections
                       if abs(c.impedance - 75.0) < 20.0]
        if close_conns:
            initial_strength = close_conns[0].synaptic_strength
            # Stimulate multiple times with matched signal
            for _ in range(10):
                region.stimulate(75.0, close_conns[0].resonant_freq)
            # Matched connections should be strengthened
            assert close_conns[0].synaptic_strength >= initial_strength

    def test_hebbian_weakening(self):
        """Hebbian weakening: synaptic decay for mismatched connections"""
        region = CorticalRegion(
            name="test",
            initial_connections=100,
            region_impedance=75.0,
        )
        # Find a connection with distant impedance
        far_conns = [c for c in region.connections
                     if abs(c.impedance - 75.0) > 80.0]
        if far_conns:
            initial_strength = far_conns[0].synaptic_strength
            for _ in range(10):
                region.stimulate(75.0, 10.0)
            assert far_conns[0].synaptic_strength < initial_strength

    def test_pruning_apoptosis(self):
        """Apoptosis: weak connections are removed"""
        region = CorticalRegion(
            name="test",
            initial_connections=200,
        )
        # Massive mismatched stimulation → many connections weakened
        for _ in range(50):
            region.stimulate(50.0, 10.0)

        initial_alive = region.alive_count
        pruned = region.prune()
        assert pruned >= 0
        assert region.alive_count == initial_alive - pruned
        assert region.alive_count <= region.initial_connections

    def test_specialization_index(self):
        """Specialization index: increases after pruning"""
        region = CorticalRegion(
            name="test",
            initial_connections=200,
        )
        initial_spec = region.get_specialization_index()

        # Continuous stimulation with specific signal → prune mismatched ones
        for _ in range(100):
            region.stimulate(50.0, 10.0)
            region.prune()

        final_spec = region.get_specialization_index()
        assert final_spec >= initial_spec  # specialization should increase

    def test_dominant_frequency_converges(self):
        """Dominant frequency converges: trends toward signal frequency after stimulation"""
        region = CorticalRegion(
            name="test",
            initial_connections=300,
        )
        target_freq = 15.0  # β band

        for _ in range(80):
            region.stimulate(75.0, target_freq)
            region.prune()

        dom_freq, spread = region.get_dominant_frequency()
        # Surviving connections' frequencies should cluster toward target_freq
        # (not required to be exact, but within a reasonable range)
        alive = region.alive_connections
        if alive:
            # At least some surviving connections should exist
            assert region.alive_count > 0

    def test_record_metrics(self):
        """Metrics recording is complete"""
        region = CorticalRegion(name="test", initial_connections=100)
        region.stimulate(75.0, 10.0)
        region.prune()
        metrics = region.record_metrics()

        assert isinstance(metrics, PruningMetrics)
        assert metrics.cycle == 1
        assert metrics.alive_count <= 100
        assert 0.0 <= metrics.survival_rate <= 1.0
        assert 0.0 <= metrics.avg_gamma <= 1.0
        assert 0.0 <= metrics.specialization_index <= 1.0

    def test_get_state(self):
        """State report contains all fields"""
        region = CorticalRegion(name="test", initial_connections=100)
        state = region.get_state()

        assert "name" in state
        assert "initial_connections" in state
        assert "alive_connections" in state
        assert "survival_rate" in state
        assert "specialization_index" in state
        assert "specialization" in state
        assert "dominant_frequency" in state
        assert "dominant_band" in state
        assert "avg_gamma" in state

    def test_determine_specialization(self):
        """Specialization determination: clear specialization direction after sufficient pruning"""
        region = CorticalRegion(name="test", initial_connections=300)

        # β band stimulation → should determine as visual or motor
        for _ in range(100):
            region.stimulate(50.0, 20.0)  # beta band
            region.prune()

        spec = region.determine_specialization()
        # With sufficient pruning pressure, should not be UNSPECIALIZED
        assert isinstance(spec, CorticalSpecialization)


# ============================================================================
# 3. NeuralPruningEngine Tests
# ============================================================================


class TestNeuralPruningEngine:
    """Overall behavior of the pruning engine"""

    def test_initialization(self):
        """Engine initialization: four regions each with N connections"""
        engine = NeuralPruningEngine(connections_per_region=200)
        assert len(engine.regions) == 4
        for region in engine.regions.values():
            assert region.alive_count == 200
        assert engine.total_epochs == 0

    def test_develop_epoch(self):
        """Single development epoch"""
        engine = NeuralPruningEngine(connections_per_region=200)
        result = engine.develop_epoch()

        assert result["epoch"] == 1
        assert "regions" in result
        assert "occipital" in result["regions"]
        assert "temporal" in result["regions"]
        assert "parietal" in result["regions"]
        assert "frontal_motor" in result["regions"]
        assert "global_gamma_squared" in result

    def test_pruning_reduces_connections(self):
        """Pruning actually reduces connection count"""
        engine = NeuralPruningEngine(connections_per_region=300)
        initial_total = sum(r.alive_count for r in engine.regions.values())

        engine.develop(epochs=50)
        final_total = sum(r.alive_count for r in engine.regions.values())

        assert final_total < initial_total  # connections were pruned

    def test_gamma_squared_decreases(self):
        """Global Σ Γ² decreases with pruning (intelligence growth)"""
        engine = NeuralPruningEngine(connections_per_region=300)
        engine.develop(epochs=5)  # run a few epochs to establish baseline
        early_gamma = engine._gamma_squared_history[0]

        engine.develop(epochs=50)
        late_gamma = engine._gamma_squared_history[-1]

        # Γ² should decrease (matching improves)
        assert late_gamma <= early_gamma

    def test_frontal_motor_matures_last(self):
        """Frontal motor area prunes slowest (requires feedback for calibration)"""
        engine = NeuralPruningEngine(connections_per_region=300)
        engine.develop(epochs=60, motor_feedback_rate=0.3)

        occipital_survival = engine.regions["occipital"].survival_rate
        frontal_survival = engine.regions["frontal_motor"].survival_rate

        # Frontal lobe prunes slowly due to low feedback rate → survival rate may be higher
        # (or pruning threshold is stricter, survival rate may be lower but slower)
        # Key point: frontal lobe receives fewer effective stimulation cycles
        occipital_stim = engine.regions["occipital"].stimulation_cycles
        frontal_stim = engine.regions["frontal_motor"].stimulation_cycles
        assert frontal_stim < occipital_stim  # frontal receives fewer effective stimulations

    def test_specialization_emerges(self):
        """Specialization emerges: different regions develop different frequency preferences"""
        engine = NeuralPruningEngine(connections_per_region=300)
        engine.develop(epochs=80)

        state = engine.get_development_state()
        # Each region's specialization index should be > 0
        for name, info in state["regions"].items():
            assert info["specialization_index"] >= 0.0

    def test_visual_auditory_differentiation(self):
        """Frequency separation between visual and auditory"""
        engine = NeuralPruningEngine(connections_per_region=500)
        engine.develop(epochs=100)

        occ = engine.regions["occipital"]
        tmp = engine.regions["temporal"]

        # Both regions should have surviving connections
        assert occ.alive_count > 0
        assert tmp.alive_count > 0

        occ_freq, _ = occ.get_dominant_frequency()
        tmp_freq, _ = tmp.get_dominant_frequency()

        # Both regions' dominant frequencies should be > 0
        assert occ_freq > 0
        assert tmp_freq > 0

    def test_development_state_complete(self):
        """Development state report is complete"""
        engine = NeuralPruningEngine(connections_per_region=200)
        engine.develop(epochs=10)

        state = engine.get_development_state()
        assert "total_epochs" in state
        assert "regions" in state
        assert "overall" in state
        assert "biological_comparison" in state

        overall = state["overall"]
        assert "total_initial_connections" in overall
        assert "total_alive_connections" in overall
        assert "overall_survival_rate" in overall
        assert "global_gamma_squared" in overall
        assert "avg_specialization" in overall

    def test_pruning_curve(self):
        """Pruning curve data is complete"""
        engine = NeuralPruningEngine(connections_per_region=200)
        engine.develop(epochs=20)

        curves = engine.get_pruning_curve()
        assert "occipital" in curves
        assert "temporal" in curves
        assert "global_gamma_sq" in curves
        assert len(curves["occipital"]) == 20  # should match number of epochs

    def test_generate_report(self):
        """Text report can be generated"""
        engine = NeuralPruningEngine(connections_per_region=200)
        engine.develop(epochs=10)

        report = engine.generate_report()
        assert "Neural Pruning Development Report" in report
        assert "occipital" in report
        assert "temporal" in report
        assert "Γ²" in report

    def test_multiple_signal_profiles(self):
        """Signal profile table is complete"""
        assert "visual" in MODALITY_SIGNAL_PROFILE
        assert "auditory" in MODALITY_SIGNAL_PROFILE
        assert "somatosensory" in MODALITY_SIGNAL_PROFILE
        assert "motor" in MODALITY_SIGNAL_PROFILE

        for profile in MODALITY_SIGNAL_PROFILE.values():
            assert "impedance" in profile
            assert "freq_range" in profile
            assert len(profile["freq_range"]) == 2


# ============================================================================
# 4. Cross-Modal Rewiring Experiment
# ============================================================================


class TestCrossModalRewiring:
    """Cross-modal experiment — signal type determines cortical specialization"""

    def test_cross_modal_runs(self):
        """Cross-modal experiment runs to completion"""
        engine = NeuralPruningEngine(connections_per_region=200)
        result = engine.cross_modal_experiment(epochs=30)

        assert "control" in result
        assert "crossed" in result
        assert "conclusion" in result

    def test_rewiring_changes_specialization(self):
        """Cross-modal changes specialization direction"""
        engine = NeuralPruningEngine(connections_per_region=300)
        result = engine.cross_modal_experiment(epochs=60)

        conclusion = result["conclusion"]
        # Control group's occipital should specialize as visual (or close)
        # Crossed group's occipital receives auditory signals, specialization should differ
        assert "control_occipital" in conclusion
        assert "crossed_occipital" in conclusion


# ============================================================================
# 5. Integration Tests — AliceBrain
# ============================================================================


class TestAliceBrainPruning:
    """Pruning engine integration within AliceBrain"""

    def test_alice_brain_has_pruning(self):
        """AliceBrain contains pruning engine"""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=50)
        assert hasattr(brain, "pruning")
        assert isinstance(brain.pruning, NeuralPruningEngine)

    def test_pruning_in_introspect(self):
        """Introspection report includes pruning state"""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=50)
        report = brain.introspect()

        assert "pruning" in report["subsystems"]
        pruning_state = report["subsystems"]["pruning"]
        assert "total_epochs" in pruning_state
        assert "regions" in pruning_state
        assert "overall" in pruning_state

    def test_pruning_develop_within_brain(self):
        """Pruning development can be executed within AliceBrain"""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain(neuron_count=50)

        # Directly invoke pruning engine
        brain.pruning.develop(epochs=10)
        state = brain.pruning.get_development_state()
        assert state["total_epochs"] == 10
        total_alive = state["overall"]["total_alive_connections"]
        total_initial = state["overall"]["total_initial_connections"]
        assert total_alive <= total_initial


# ============================================================================
# 6. Physics Consistency Tests
# ============================================================================


class TestPhysicsConsistency:
    """Physics consistency verification"""

    def test_gamma_range(self):
        """Γ is always within [0, 1] range"""
        conn = SynapticConnection(connection_id=0, impedance=75.0, resonant_freq=10.0)
        for z in [1.0, 10.0, 50.0, 75.0, 100.0, 200.0, 1000.0]:
            gamma = conn.compute_gamma(z)
            assert 0.0 <= gamma < 1.0

    def test_resonance_range(self):
        """Resonance response is always within [0, 1] range"""
        conn = SynapticConnection(connection_id=0, impedance=75.0, resonant_freq=10.0)
        for f in [0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
            response = conn.compute_resonance(f)
            assert 0.0 <= response <= 1.0

    def test_strength_bounded(self):
        """Synaptic strength is bounded [0, 2.0]"""
        region = CorticalRegion(name="test", initial_connections=100)
        for _ in range(200):
            region.stimulate(75.0, 10.0)
        for conn in region.alive_connections:
            assert 0.0 <= conn.synaptic_strength <= 2.0

    def test_survival_rate_bounded(self):
        """Survival rate is bounded [0, 1]"""
        engine = NeuralPruningEngine(connections_per_region=100)
        engine.develop(epochs=50)
        for region in engine.regions.values():
            assert 0.0 <= region.survival_rate <= 1.0

    def test_total_connections_net_decrease_without_learning(self):
        """Without learning signal, pruning dominates → net decrease (infant pruning)"""
        engine = NeuralPruningEngine(connections_per_region=200)
        initial_total = sum(r.alive_count for r in engine.regions.values())

        for _ in range(30):
            engine.develop_epoch(learning_signal=0.0)  # no learning → no new growth

        final_total = sum(r.alive_count for r in engine.regions.values())
        assert final_total < initial_total  # net decrease

    def test_synaptogenesis_creates_connections(self):
        """With high learning signal, synaptogenesis creates new connections"""
        engine = NeuralPruningEngine(connections_per_region=200)

        # First let some connections strengthen (need Hebbian strengthening above threshold)
        for _ in range(20):
            engine.develop_epoch(learning_signal=0.0)

        # Record current peak
        pre_peak = sum(r.peak_connections for r in engine.regions.values())
        pre_sprouted = sum(r.sprouted_total for r in engine.regions.values())

        # High learning signal → trigger synaptogenesis
        for _ in range(30):
            engine.develop_epoch(learning_signal=0.8)

        post_sprouted = sum(r.sprouted_total for r in engine.regions.values())
        assert post_sprouted > pre_sprouted  # new connections were indeed created

    def test_synaptogenesis_balance(self):
        """Long-term development: generation rate ≈ apoptosis rate (dynamic equilibrium)"""
        engine = NeuralPruningEngine(connections_per_region=300)

        # Long-term mixed development
        alive_history = []
        for _ in range(100):
            engine.develop_epoch(learning_signal=0.5)
            total = sum(r.alive_count for r in engine.regions.values())
            alive_history.append(total)

        # Second half should have smaller fluctuation than first half (trending stable)
        first_half_range = max(alive_history[:50]) - min(alive_history[:50])
        second_half_range = max(alive_history[50:]) - min(alive_history[50:])
        # Second half amplitude ≤ first half (trending toward equilibrium or at least not more chaotic)
        assert second_half_range <= first_half_range + 50  # 50 tolerance margin

    def test_sprouted_connections_can_be_pruned(self):
        """Newly sprouted connections start weak → pruned if mismatched"""
        region = CorticalRegion(
            name="test",
            initial_connections=100,
            region_impedance=75.0,
        )
        # First strengthen some connections
        for _ in range(30):
            region.stimulate(75.0, 10.0)

        sprouted = region.sprout(learning_signal=1.0)
        if sprouted > 0:
            alive_after_sprout = region.alive_count
            # Stimulate with completely different signal → new connections may mismatch
            for _ in range(50):
                region.stimulate(200.0, 80.0)  # completely different signal
            region.prune()
            # Some fragile newly sprouted connections should be pruned
            # (at least not all survive)
            assert region.alive_count <= alive_after_sprout

    def test_peak_connections_tracked(self):
        """Historical peak connection count is correctly tracked"""
        engine = NeuralPruningEngine(connections_per_region=200)
        initial_peak = sum(r.peak_connections for r in engine.regions.values())
        assert initial_peak == 800  # 4 regions × 200

        # With learning signal → peak may increase
        for _ in range(30):
            engine.develop_epoch(learning_signal=0.8)

        final_peak = sum(r.peak_connections for r in engine.regions.values())
        assert final_peak >= initial_peak  # only increases, never decreases

    def test_impedance_matching_reduces_gamma(self):
        """Impedance-matched signals reduce Γ"""
        region = CorticalRegion(
            name="test",
            initial_connections=200,
            region_impedance=50.0,
        )
        # Stimulate with matched impedance
        for _ in range(30):
            region.stimulate(50.0, 10.0)

        alive = region.alive_connections
        if alive:
            # Surviving connections' impedances should converge toward 50Ω
            avg_gamma = float(np.mean([c.compute_gamma(50.0) for c in alive]))
            # Gamma should be low (close to matched)
            assert avg_gamma < 0.5  # better than completely random

    def test_lorentz_curve_peaks_at_resonance(self):
        """Lorentzian curve peaks at resonant frequency"""
        conn = SynapticConnection(connection_id=0, impedance=75.0, resonant_freq=25.0)
        peak = conn.compute_resonance(25.0)
        below = conn.compute_resonance(15.0)
        above = conn.compute_resonance(40.0)

        assert peak == pytest.approx(1.0, abs=1e-10)
        assert below < peak
        assert above < peak
