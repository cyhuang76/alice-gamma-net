# -*- coding: utf-8 -*-
"""Tests for Phase 11: SleepCycle"""
import numpy as np
import pytest
from alice.brain.sleep import (
    SleepCycle, SleepStage, STAGE_PARAMS, STAGE_DURATION,
    SLEEP_PRESSURE_THRESHOLD,
)


class TestSleepBasics:
    def test_init_awake(self):
        sc = SleepCycle()
        assert sc.stage == SleepStage.WAKE
        assert sc.is_sleeping() is False
        assert sc.sleep_pressure == 0.0

    def test_reset(self):
        sc = SleepCycle()
        sc.stage = SleepStage.N3
        sc.sleep_pressure = 0.9
        sc.reset()
        assert sc.stage == SleepStage.WAKE
        assert sc.sleep_pressure == 0.0


class TestSleepPressure:
    def test_pressure_accumulates_while_awake(self):
        sc = SleepCycle()
        for _ in range(100):
            sc.tick()
        assert sc.sleep_pressure > 0.0

    def test_pressure_releases_during_sleep(self):
        sc = SleepCycle()
        sc.sleep_pressure = 0.8
        sc.tick(force_sleep=True)
        initial = sc.sleep_pressure
        for _ in range(20):
            sc.tick()
        assert sc.sleep_pressure < initial

    def test_should_sleep_threshold(self):
        sc = SleepCycle()
        sc.sleep_pressure = SLEEP_PRESSURE_THRESHOLD - 0.01
        assert sc.should_sleep() is False
        sc.sleep_pressure = SLEEP_PRESSURE_THRESHOLD + 0.01
        assert sc.should_sleep() is True


class TestSleepTransitions:
    def test_force_sleep_enters_n1(self):
        sc = SleepCycle()
        sc.sleep_pressure = 0.9  # Ensure it doesn't wake up immediately
        sc.tick(force_sleep=True)
        assert sc.stage == SleepStage.N1

    def test_force_wake(self):
        sc = SleepCycle()
        sc.sleep_pressure = 0.9
        sc.tick(force_sleep=True)
        assert sc.is_sleeping()
        sc.tick(force_wake=True)
        assert sc.stage == SleepStage.WAKE

    def test_automatic_stage_progression(self):
        sc = SleepCycle()
        sc.sleep_pressure = 0.9
        sc.tick(force_sleep=True)
        assert sc.stage == SleepStage.N1

        # Advance N1 duration ticks
        for _ in range(STAGE_DURATION[SleepStage.N1] + 1):
            sc.tick()
        assert sc.stage != SleepStage.N1  # Should have transitioned

    def test_external_stimulus_wakes(self):
        sc = SleepCycle()
        sc.sleep_pressure = 0.9
        sc.tick(force_sleep=True)
        # N1 arousal_threshold = 0.2
        sc.tick(external_stimulus_strength=0.5)
        assert sc.stage == SleepStage.WAKE

    def test_deep_sleep_resists_weak_stimulus(self):
        sc = SleepCycle()
        sc.sleep_pressure = 0.9
        # Force to N3
        sc.stage = SleepStage.N3
        sc._stage_ticks = 0
        # N3 arousal_threshold = 0.7
        sc.tick(external_stimulus_strength=0.3)
        assert sc.stage == SleepStage.N3  # Not woken up


class TestSensoryGate:
    def test_wake_gate_full_open(self):
        sc = SleepCycle()
        assert sc.get_sensory_gate() == 1.0

    def test_deep_sleep_gate_mostly_closed(self):
        sc = SleepCycle()
        sc.stage = SleepStage.N3
        assert sc.get_sensory_gate() == pytest.approx(0.1)


class TestSleepStages:
    def test_is_deep_sleep(self):
        sc = SleepCycle()
        sc.stage = SleepStage.N3
        assert sc.is_deep_sleep() is True
        sc.stage = SleepStage.N2
        assert sc.is_deep_sleep() is False

    def test_is_dreaming(self):
        sc = SleepCycle()
        sc.stage = SleepStage.REM
        assert sc.is_dreaming() is True
        sc.stage = SleepStage.N3
        assert sc.is_dreaming() is False

    def test_consciousness_levels(self):
        sc = SleepCycle()
        assert sc.get_consciousness_level() == 1.0  # WAKE
        sc.stage = SleepStage.N3
        assert sc.get_consciousness_level() == 0.1


class TestSleepConsolidation:
    def test_n3_triggers_consolidation(self):
        sc = SleepCycle()
        sc.stage = SleepStage.N3
        result = sc.tick()
        assert result["should_consolidate"] is True
        assert result["consolidation_rate"] == 1.0

    def test_wake_no_consolidation(self):
        sc = SleepCycle()
        result = sc.tick()
        assert result["should_consolidate"] is False

    def test_rem_triggers_consolidation(self):
        sc = SleepCycle()
        sc.stage = SleepStage.REM
        sc._stage_ticks = 0
        sc.sleep_pressure = 0.9
        result = sc.tick()
        assert result["should_consolidate"] is True


class TestSleepTickResult:
    def test_result_keys(self):
        sc = SleepCycle()
        result = sc.tick()
        assert "stage" in result
        assert "sensory_gate" in result
        assert "consolidation_rate" in result
        assert "consciousness" in result
        assert "sleep_pressure" in result
        assert "energy_recovery" in result


class TestSleepStats:
    def test_stats_keys(self):
        sc = SleepCycle()
        sc.tick()
        stats = sc.get_stats()
        assert "stage" in stats
        assert "sleep_pressure" in stats
        assert "cycles_completed" in stats
        assert "is_sleeping" in stats

    def test_waveforms(self):
        sc = SleepCycle()
        for _ in range(5):
            sc.tick()
        wf = sc.get_waveforms(last_n=3)
        assert len(wf["pressure"]) == 3


class TestFullSleepCycle:
    def test_complete_one_cycle(self):
        sc = SleepCycle()
        sc.sleep_pressure = 0.9
        sc.tick(force_sleep=True)

        # Run through a full cycle
        for _ in range(200):
            sc.tick()

        # Should have completed at least a partial cycle
        assert sc._total_sleep_ticks > 0
