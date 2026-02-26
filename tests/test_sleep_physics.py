# -*- coding: utf-8 -*-
"""
Sleep Physics Engine Tests
Tests for Sleep Physics Engine — Offline Impedance Renormalization
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from alice.brain.sleep_physics import (
    SleepPhysicsEngine,
    ImpedanceDebtTracker,
    SynapticEntropyTracker,
    SlowWaveOscillator,
    REMDreamDiagnostic,
    SleepQualityReport,
    METABOLIC_COST,
    RECOVERY_RATE,
    RECALIBRATION_RATE,
    DOWNSCALE_FACTOR,
)


# ============================================================
# Impedance Debt Tracker
# ============================================================

class TestImpedanceDebtTracker:
    """Physical behavior of impedance debt"""

    def test_initial_state(self):
        tracker = ImpedanceDebtTracker()
        assert tracker.debt == 0.0
        assert tracker.peak_debt == 0.0
        assert tracker.total_accumulated == 0.0
        assert tracker.total_repaired == 0.0

    def test_accumulate(self):
        tracker = ImpedanceDebtTracker()
        tracker.accumulate(0.1)  # reflected energy 0.1
        assert tracker.debt > 0.0
        assert tracker.total_accumulated > 0.0

    def test_accumulate_scales_with_reflection(self):
        t1 = ImpedanceDebtTracker()
        t2 = ImpedanceDebtTracker()
        t1.accumulate(0.1)
        t2.accumulate(0.5)
        assert t2.debt > t1.debt, "Larger reflected energy → more debt"

    def test_repair_reduces_debt(self):
        tracker = ImpedanceDebtTracker()
        tracker.accumulate(0.5)
        initial_debt = tracker.debt
        tracker.repair("n3")  # N3 recalibration is strongest
        assert tracker.debt < initial_debt

    def test_repair_n3_stronger_than_n1(self):
        t_n3 = ImpedanceDebtTracker()
        t_n1 = ImpedanceDebtTracker()
        t_n3.accumulate(0.5)
        t_n1.accumulate(0.5)
        debt_before = t_n3.debt
        t_n3.repair("n3")
        t_n1.repair("n1")
        assert t_n3.debt < t_n1.debt, "N3 repair strength should be greater than N1"

    def test_debt_bounded(self):
        tracker = ImpedanceDebtTracker()
        for _ in range(1000):
            tracker.accumulate(1.0)
        assert tracker.debt <= 1.0

    def test_peak_debt_tracked(self):
        tracker = ImpedanceDebtTracker()
        tracker.accumulate(0.3)
        tracker.accumulate(0.3)
        peak = tracker.debt
        tracker.repair("n3")
        assert tracker.peak_debt >= peak

    def test_get_state(self):
        tracker = ImpedanceDebtTracker()
        tracker.accumulate(0.2)
        state = tracker.get_state()
        assert "debt" in state
        assert "peak_debt" in state
        assert "repair_ratio" in state


# ============================================================
# Synaptic Entropy Tracker
# ============================================================

class TestSynapticEntropyTracker:
    """Shannon entropy of synaptic weights"""

    def test_uniform_distribution_max_entropy(self):
        """Uniform distribution → maximum entropy"""
        tracker = SynapticEntropyTracker()
        strengths = [1.0] * 100
        H = tracker.compute(strengths)
        max_H = math.log2(100)
        assert abs(H - max_H) < 0.01

    def test_single_element(self):
        tracker = SynapticEntropyTracker()
        H = tracker.compute([1.0])
        assert H == 0.0  # Single element entropy is 0

    def test_skewed_lower_entropy(self):
        """Skewed distribution → lower than max entropy"""
        tracker = SynapticEntropyTracker()
        # Few very strong, most very weak
        strengths = [0.1] * 90 + [10.0] * 10
        H_skew = tracker.compute(strengths)

        strengths_uniform = [1.0] * 100
        H_uniform = tracker.compute(strengths_uniform)

        assert H_skew < H_uniform

    def test_empty_list(self):
        tracker = SynapticEntropyTracker()
        H = tracker.compute([])
        assert H == 0.0

    def test_entropy_deficit_computed(self):
        tracker = SynapticEntropyTracker()
        tracker.compute([1.0] * 100)
        assert tracker.entropy_deficit >= 0.0

    def test_history_recorded(self):
        tracker = SynapticEntropyTracker()
        tracker.compute([1.0, 2.0, 3.0])
        tracker.record()
        assert len(tracker.history) == 1

    def test_get_state(self):
        tracker = SynapticEntropyTracker()
        tracker.compute([1.0, 2.0])
        state = tracker.get_state()
        assert "current_entropy" in state
        assert "optimal_entropy" in state


# ============================================================
# Slow Wave Oscillator
# ============================================================

class TestSlowWaveOscillator:
    """N3 deep sleep δ wave generator"""

    def test_initial_state(self):
        osc = SlowWaveOscillator()
        assert osc.phase == 0.0
        assert osc.cycle_count == 0

    def test_tick_produces_result(self):
        osc = SlowWaveOscillator()
        result = osc.tick()
        assert "is_up_state" in result
        assert "delta_amplitude" in result
        assert "should_replay" in result
        assert "replay_count" in result

    def test_amplitude_bounded(self):
        osc = SlowWaveOscillator()
        for _ in range(100):
            result = osc.tick()
            assert -1.0 <= result["delta_amplitude"] <= 1.0

    def test_up_down_alternation(self):
        """Slow waves should alternate between UP and DOWN states"""
        osc = SlowWaveOscillator()
        states = []
        for _ in range(200):
            result = osc.tick()
            states.append(result["is_up_state"])
        # Should have both UP and DOWN
        assert True in states
        assert False in states

    def test_replay_triggers_on_peak(self):
        osc = SlowWaveOscillator()
        replays = 0
        for _ in range(200):
            result = osc.tick()
            replays += result["replay_count"]
        assert replays > 0, "Should have replay triggers"

    def test_reset(self):
        osc = SlowWaveOscillator()
        for _ in range(50):
            osc.tick()
        osc.reset()
        assert osc.phase == 0.0


# ============================================================
# REM Dream Diagnostic
# ============================================================

class TestREMDreamDiagnostic:
    """Channel health probing during REM"""

    def test_probe_healthy_channels(self):
        dream = REMDreamDiagnostic(rng=np.random.default_rng(42))
        channels = [
            ("ch_good", 50.0, 52.0),  # Γ ≈ 0
        ]
        result = dream.probe_channels(channels)
        assert result["healthy"] > 0 or result["damaged"] > 0

    def test_probe_damaged_channels(self):
        dream = REMDreamDiagnostic(rng=np.random.default_rng(42))
        channels = [
            ("ch_bad", 50.0, 200.0),  # Γ >> 0
        ]
        result = dream.probe_channels(channels)
        assert result["damaged"] >= 1

    def test_repair_queue(self):
        dream = REMDreamDiagnostic(rng=np.random.default_rng(42))
        channels = [
            ("damaged_1", 50.0, 300.0),
            ("damaged_2", 75.0, 400.0),
        ]
        dream.probe_channels(channels)
        queue = dream.get_repair_queue()
        assert len(queue) > 0
        assert "channel" in queue[0]
        assert "gamma" in queue[0]

    def test_dream_fragments_limited(self):
        dream = REMDreamDiagnostic(rng=np.random.default_rng(42))
        channels = [(f"ch_{i}", 50.0, float(50 + i * 20)) for i in range(10)]
        for _ in range(20):
            dream.probe_channels(channels)
        assert len(dream.dream_fragments) <= 50

    def test_empty_channels(self):
        dream = REMDreamDiagnostic()
        result = dream.probe_channels([])
        assert result["probes"] == 0

    def test_get_state(self):
        dream = REMDreamDiagnostic(rng=np.random.default_rng(42))
        channels = [("ch_1", 50.0, 60.0)]
        dream.probe_channels(channels)
        state = dream.get_state()
        assert "probes_sent" in state
        assert "dream_health_ratio" in state


# ============================================================
# Sleep Quality Report
# ============================================================

class TestSleepQualityReport:
    """Sleep quality score calculation"""

    def test_empty_report(self):
        report = SleepQualityReport()
        assert report.quality_score >= 0.0

    def test_perfect_sleep(self):
        report = SleepQualityReport(
            total_sleep_ticks=100,
            n3_ticks=25,
            rem_ticks=20,
            n3_ratio=0.25,
            rem_ratio=0.20,
            interruptions=0,
            energy_restored=0.5,
            impedance_debt_repaired=0.3,
            dream_health_ratio=1.0,
        )
        assert report.quality_score > 0.8

    def test_poor_sleep(self):
        report = SleepQualityReport(
            total_sleep_ticks=30,
            n3_ticks=0,
            rem_ticks=0,
            n3_ratio=0.0,
            rem_ratio=0.0,
            interruptions=5,
            energy_restored=0.0,
            impedance_debt_repaired=0.0,
            dream_health_ratio=0.0,
        )
        assert report.quality_score < 0.3

    def test_quality_bounded(self):
        report = SleepQualityReport(
            total_sleep_ticks=200,
            n3_ticks=100,
            rem_ticks=100,
            n3_ratio=0.5,
            rem_ratio=0.5,
            energy_restored=1.0,
            impedance_debt_repaired=1.0,
            dream_health_ratio=1.0,
        )
        assert 0.0 <= report.quality_score <= 1.0

    def test_to_dict(self):
        report = SleepQualityReport()
        d = report.to_dict()
        assert "quality_score" in d
        assert "n3_ratio" in d


# ============================================================
# Main Engine: SleepPhysicsEngine
# ============================================================

class TestSleepPhysicsEngine:
    """Overall behavior of the sleep physics engine"""

    def test_initial_state(self):
        engine = SleepPhysicsEngine()
        assert engine.energy == 1.0
        assert engine.sleep_pressure == 0.0
        assert not engine.should_sleep()

    def test_awake_tick_consumes_energy(self):
        engine = SleepPhysicsEngine()
        engine.awake_tick(reflected_energy=0.1)
        assert engine.energy < 1.0

    def test_awake_tick_accumulates_debt(self):
        engine = SleepPhysicsEngine()
        engine.awake_tick(reflected_energy=0.1)
        assert engine.impedance_debt.debt > 0.0

    def test_awake_tick_increases_pressure(self):
        engine = SleepPhysicsEngine()
        for _ in range(50):
            engine.awake_tick(reflected_energy=0.1)
        assert engine.sleep_pressure > 0.0

    def test_should_sleep_after_exhaustion(self):
        engine = SleepPhysicsEngine()
        for _ in range(200):
            engine.awake_tick(reflected_energy=0.1)
        assert engine.should_sleep()

    def test_sleep_tick_recovers_energy(self):
        engine = SleepPhysicsEngine(energy=0.3)
        engine.impedance_debt.accumulate(0.5)
        engine.sleep_tick(stage="n3", synaptic_strengths=[1.0] * 50)
        assert engine.energy > 0.3

    def test_sleep_tick_repairs_debt(self):
        engine = SleepPhysicsEngine()
        engine.impedance_debt.accumulate(0.5)
        initial_debt = engine.impedance_debt.debt
        engine.sleep_tick(stage="n3")
        assert engine.impedance_debt.debt < initial_debt

    def test_n3_triggers_slow_wave(self):
        engine = SleepPhysicsEngine()
        result = engine.sleep_tick(stage="n3", synaptic_strengths=[1.0] * 50)
        assert result["slow_wave"] is not None

    def test_rem_triggers_dreams(self):
        engine = SleepPhysicsEngine()
        channels = [("ch_1", 50.0, 100.0), ("ch_2", 75.0, 80.0)]
        result = engine.sleep_tick(stage="rem", channel_impedances=channels)
        assert result["dream_result"] is not None

    def test_n3_downscaling(self):
        engine = SleepPhysicsEngine()
        strengths = [1.0, 1.5, 0.8]
        # Run enough ticks to hit a DOWN state
        downscaled = False
        for _ in range(20):
            result = engine.sleep_tick(stage="n3", synaptic_strengths=strengths)
            if result["downscaled"]:
                downscaled = True
                assert result["downscale_strengths"] is not None
                for orig, ds in zip(strengths, result["downscale_strengths"]):
                    assert ds <= orig
                break
        assert downscaled, "N3 should trigger downscaling"

    def test_begin_end_sleep(self):
        engine = SleepPhysicsEngine(energy=0.5)
        engine.impedance_debt.accumulate(0.3)
        engine.begin_sleep()
        for _ in range(30):
            engine.sleep_tick(stage="n3", synaptic_strengths=[1.0] * 50)
        report = engine.end_sleep()
        assert isinstance(report, SleepQualityReport)
        assert report.total_sleep_ticks == 30
        assert report.n3_ticks == 30

    def test_complete_day_night_cycle(self):
        engine = SleepPhysicsEngine()
        result = engine.simulate_day_night(
            awake_ticks=50,
            sleep_ticks=60,
            n_synapses=100,
        )
        assert "pre_sleep" in result
        assert "post_sleep" in result
        assert "sleep_report" in result
        # Post-sleep energy should be higher than pre-sleep
        assert result["post_sleep"]["energy"] > result["pre_sleep"]["energy"]
        # Post-sleep debt should be lower than pre-sleep
        assert result["post_sleep"]["impedance_debt"] < result["pre_sleep"]["impedance_debt"]

    def test_energy_never_negative(self):
        engine = SleepPhysicsEngine(energy=0.01)
        for _ in range(100):
            engine.awake_tick(reflected_energy=0.5)
        assert engine.energy >= 0.0

    def test_pressure_bounded(self):
        engine = SleepPhysicsEngine(energy=0.0)
        engine.impedance_debt.accumulate(10.0)
        engine._update_pressure()
        assert 0.0 <= engine.sleep_pressure <= 1.0

    def test_get_state(self):
        engine = SleepPhysicsEngine()
        engine.awake_tick(reflected_energy=0.1)
        state = engine.get_state()
        assert "energy" in state
        assert "sleep_pressure" in state
        assert "impedance_debt" in state
        assert "entropy" in state
        assert "statistics" in state

    def test_sleep_schedule_generation(self):
        schedule = SleepPhysicsEngine._generate_sleep_schedule(100)
        assert len(schedule) == 100
        assert all(s in ("n1", "n2", "n3", "rem") for s in schedule)
        # Should contain all stages
        stages = set(schedule)
        assert "n3" in stages
        assert "rem" in stages

    def test_static_downscaling(self):
        strengths = [1.0, 2.0, 0.5]
        result = SleepPhysicsEngine.apply_downscaling(strengths, factor=0.9)
        assert result[0] == pytest.approx(0.9, abs=0.01)
        assert result[1] == pytest.approx(1.8, abs=0.01)
        # Preserve relative differences
        assert result[1] / result[0] == pytest.approx(2.0, abs=0.01)

    def test_static_impedance_recalibration(self):
        channels = [("ch_1", 50.0, 100.0)]
        result = SleepPhysicsEngine.impedance_recalibration(channels, 0.1)
        name, z_src, z_load_new = result[0]
        # z_load should move toward z_src
        assert z_load_new < 100.0
        assert z_load_new > 50.0

    def test_downscaling_preserves_floor(self):
        strengths = [0.01, 0.001]
        result = SleepPhysicsEngine.apply_downscaling(strengths, factor=0.5, floor=0.05)
        assert all(s >= 0.05 for s in result)


# ============================================================
# AliceBrain Integration Tests
# ============================================================

class TestAliceBrainSleepPhysics:
    """AliceBrain integration with sleep physics engine"""

    def test_alice_brain_has_sleep_physics(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        assert hasattr(brain, "sleep_physics")
        assert isinstance(brain.sleep_physics, SleepPhysicsEngine)

    def test_sleep_physics_in_introspect(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        intro = brain.introspect()
        assert "sleep_physics" in intro["subsystems"]
        sp = intro["subsystems"]["sleep_physics"]
        assert "energy" in sp
        assert "sleep_pressure" in sp

    def test_sleep_physics_state_changes(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        state_before = brain.sleep_physics.get_state()
        # Simulate some awake ticks
        for _ in range(3):
            brain.sleep_physics.awake_tick(reflected_energy=0.05)
        state_after = brain.sleep_physics.get_state()
        assert state_after["energy"] < state_before["energy"]


# ============================================================
# Physics Consistency Tests
# ============================================================

class TestPhysicsConsistency:
    """Ensure all physical quantities obey conservation laws"""

    def test_energy_conservation_awake(self):
        """Energy only decreases during wakefulness"""
        engine = SleepPhysicsEngine(energy=0.5)
        for _ in range(10):
            engine.awake_tick(reflected_energy=0.05)
        assert engine.energy < 0.5

    def test_energy_conservation_sleep(self):
        """Energy increases during N3 sleep"""
        engine = SleepPhysicsEngine(energy=0.3)
        for _ in range(20):
            engine.sleep_tick(stage="n3")
        assert engine.energy > 0.3

    def test_debt_conservation(self):
        """Awake accumulation + sleep repair = conservation"""
        engine = SleepPhysicsEngine()
        for _ in range(50):
            engine.awake_tick(reflected_energy=0.1)
        accumulated = engine.impedance_debt.total_accumulated
        assert accumulated > 0

        for _ in range(50):
            engine.sleep_tick(stage="n3")
        repaired = engine.impedance_debt.total_repaired
        assert repaired > 0
        assert engine.impedance_debt.debt < accumulated

    def test_entropy_positive(self):
        tracker = SynapticEntropyTracker()
        H = tracker.compute([0.5, 1.0, 1.5, 2.0])
        assert H > 0.0

    def test_n3_recovery_greater_than_n1(self):
        """N3 recovery should be greater than N1"""
        assert RECOVERY_RATE["n3"] > RECOVERY_RATE["n1"]

    def test_n3_metabolic_less_than_wake(self):
        """N3 metabolic cost should be less than wakefulness"""
        assert METABOLIC_COST["n3"] < METABOLIC_COST["wake"]

    def test_n3_recalibration_strongest(self):
        """N3 recalibration strength should be the strongest"""
        assert RECALIBRATION_RATE["n3"] == max(RECALIBRATION_RATE.values())

    def test_downscale_less_than_one(self):
        """Downscale factor < 1.0"""
        assert DOWNSCALE_FACTOR["n3"] < 1.0

    def test_wake_no_recovery(self):
        """No recovery during wakefulness"""
        assert RECOVERY_RATE["wake"] == 0.0
        assert RECALIBRATION_RATE["wake"] == 0.0
        assert DOWNSCALE_FACTOR["wake"] == 1.0
