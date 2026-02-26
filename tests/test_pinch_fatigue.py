# -*- coding: utf-8 -*-
"""
test_pinch_fatigue.py — Lorentz Compression Fatigue Engine Unit Tests
Pollock-Barraclough (1905) Neural Aging Physics

Test strategy:
  1. Lorentz compression force ∈ I² (physical law correctness)
  2. Elastic/plastic separation (below yield = full recovery, above = permanent)
  3. Fatigue accumulation (Coffin-Manson + Miner's rule)
  4. Sleep repair (elastic recovery >> plastic recovery)
  5. Work hardening (cumulative use → increased resistance)
  6. Temperature acceleration (Arrhenius)
  7. Aging → impedance drift → Γ drift
  8. Multi-channel independent aging
  9. Life expectancy estimation
  10. Full system integration
"""

import pytest
import numpy as np

from alice.brain.pinch_fatigue import (
    PinchFatigueEngine,
    ConductorState,
    PinchEvent,
    AgingSignal,
    YIELD_STRAIN,
    PINCH_SENSITIVITY,
    PINCH_EXPONENT,
    MAX_FATIGUE_DAMAGE,
)


# ============================================================================
# Fixture
# ============================================================================

@pytest.fixture
def engine():
    return PinchFatigueEngine()


@pytest.fixture
def aged_engine():
    """Aged engine (multiple high-current shocks)."""
    eng = PinchFatigueEngine()
    # Current intensity must exceed yield threshold to produce plastic aging
    high_i = (YIELD_STRAIN / PINCH_SENSITIVITY) ** (1 / PINCH_EXPONENT) + 1.0
    for _ in range(500):
        eng.apply_current_pulse("visual", high_i, temperature=0.7)
    return eng


# ============================================================================
# 1. Lorentz Compression Force Physical Correctness
# ============================================================================

class TestPinchPhysics:
    """Lorentz compression force ∈ I² — Pollock-Barraclough law."""

    def test_strain_proportional_to_i_squared(self, engine):
        """ε ∈ I² — Lorentz compression force is proportional to current squared."""
        e1 = engine.apply_current_pulse("ch1", 0.5)
        engine.reset()
        e2 = engine.apply_current_pulse("ch1", 1.0)

        # I=1.0 should produce 4x the strain of I=0.5
        ratio = e2.pinch_strain / e1.pinch_strain
        expected = (1.0 / 0.5) ** PINCH_EXPONENT
        assert abs(ratio - expected) < 0.01

    def test_low_current_no_plastic(self, engine):
        """Low current → pure elastic, no plastic deformation."""
        event = engine.apply_current_pulse("ch1", 0.1)
        assert event.plastic_increment == 0.0
        assert not event.was_over_yield

    def test_high_current_produces_plastic(self, engine):
        """High current → exceeds yield → produces plastic deformation."""
        # Current intensity must be enough so that ε = sensitivity × I^exp > YIELD_STRAIN
        high_i = (YIELD_STRAIN / PINCH_SENSITIVITY) ** (1 / PINCH_EXPONENT) + 0.5
        event = engine.apply_current_pulse("ch1", high_i)
        assert event.was_over_yield
        assert event.plastic_increment > 0

    def test_zero_current_no_strain(self, engine):
        """Zero current → zero strain."""
        # tick with zero activity should be fine
        signal = engine.tick({"ch1": 0.01})
        assert signal.total_plastic_strain == 0.0


# ============================================================================
# 2. Elastic/Plastic Separation
# ============================================================================

class TestElasticPlasticSeparation:
    """Material mechanics: elastic is recoverable, plastic is irreversible."""

    def test_elastic_recovers_with_repair(self, engine):
        """Elastic strain decreases after repair tick."""
        engine.apply_current_pulse("ch1", 0.5)
        ch = engine._channels["ch1"]
        elastic_before = ch.elastic_strain
        engine.repair_tick(is_sleeping=True)
        assert ch.elastic_strain < elastic_before

    def test_plastic_barely_recovers(self, engine):
        """Plastic strain barely recovers."""
        high_i = (YIELD_STRAIN / PINCH_SENSITIVITY) ** (1 / PINCH_EXPONENT) + 1.0
        engine.apply_current_pulse("ch1", high_i)
        ch = engine._channels["ch1"]
        plastic_before = ch.plastic_strain
        assert plastic_before > 0

        # 100 repair ticks
        for _ in range(100):
            engine.repair_tick(is_sleeping=True)

        # Plastic strain should be mostly preserved
        ratio = ch.plastic_strain / plastic_before
        assert ratio > 0.5, "Plastic strain recovers too fast (violates aging physics)"

    def test_elastic_vs_plastic_recovery_ratio(self, engine):
        """Elastic recovery rate >> plastic recovery rate."""
        high_i = (YIELD_STRAIN / PINCH_SENSITIVITY) ** (1 / PINCH_EXPONENT) + 1.0
        engine.apply_current_pulse("ch1", high_i)
        ch = engine._channels["ch1"]

        elastic_0 = ch.elastic_strain
        plastic_0 = ch.plastic_strain

        for _ in range(50):
            engine.repair_tick(is_sleeping=True)

        elastic_recovery = 1.0 - (ch.elastic_strain / max(elastic_0, 1e-12))
        plastic_recovery = 1.0 - (ch.plastic_strain / max(plastic_0, 1e-12))

        assert elastic_recovery > plastic_recovery * 5, \
            "Elastic recovery should be > 5x faster than plastic recovery"


# ============================================================================
# 3. Coffin-Manson Fatigue Accumulation
# ============================================================================

class TestFatigueCumulation:
    """Coffin-Manson fatigue life law."""

    def test_fatigue_increases_with_cycles(self, engine):
        """Repeated over-yield cycles → fatigue accumulation."""
        high_i = (YIELD_STRAIN / PINCH_SENSITIVITY) ** (1 / PINCH_EXPONENT) + 1.0
        for _ in range(50):
            engine.apply_current_pulse("ch1", high_i)

        ch = engine._channels["ch1"]
        assert ch.fatigue_damage > 0
        assert ch.over_yield_cycles == 50

    def test_fatigue_bounded(self, engine):
        """Fatigue damage has an upper bound."""
        high_i = (YIELD_STRAIN / PINCH_SENSITIVITY) ** (1 / PINCH_EXPONENT) + 3.0
        for _ in range(10000):
            engine.apply_current_pulse("ch1", high_i, temperature=0.9)

        ch = engine._channels["ch1"]
        assert ch.fatigue_damage <= MAX_FATIGUE_DAMAGE

    def test_higher_strain_faster_fatigue(self, engine):
        """Higher strain → faster fatigue (Coffin-Manson nonlinearity)."""
        base_i = (YIELD_STRAIN / PINCH_SENSITIVITY) ** (1 / PINCH_EXPONENT) + 0.5
        high_i = base_i + 2.0

        eng1 = PinchFatigueEngine()
        eng2 = PinchFatigueEngine()
        for _ in range(100):
            eng1.apply_current_pulse("ch1", base_i)
            eng2.apply_current_pulse("ch1", high_i)

        f1 = eng1._channels["ch1"].fatigue_damage
        f2 = eng2._channels["ch1"].fatigue_damage
        assert f2 > f1 * 1.5, "High strain should produce significantly more fatigue"


# ============================================================================
# 4. Sleep Repair
# ============================================================================

class TestSleepRepair:
    """Sleep repair: elastic part repairable, plastic nearly irreversible."""

    def test_sleep_repairs_elastic_faster(self, engine):
        """Sleep repairs elastic strain faster than waking."""
        engine.apply_current_pulse("ch1", 0.5)
        ch = engine._channels["ch1"]
        e0 = ch.elastic_strain

        engine2 = PinchFatigueEngine()
        engine2.apply_current_pulse("ch1", 0.5)
        ch2 = engine2._channels["ch1"]

        # Sleep repair
        for _ in range(10):
            engine.repair_tick(is_sleeping=True)
        # Waking repair
        for _ in range(10):
            engine2.repair_tick(is_sleeping=False)

        assert ch.elastic_strain < ch2.elastic_strain

    def test_sleep_cannot_reverse_aging(self, aged_engine):
        """Sleep cannot reverse aging (core physics)."""
        ch = aged_engine._channels["visual"]
        plastic_before = ch.plastic_strain
        impedance_drift_before = ch.impedance_drift
        assert plastic_before > 0, "aged_engine should have plastic strain"
        assert impedance_drift_before > 0, "aged_engine should have impedance drift"

        # Moderate sleep (not infinite BDNF)
        for _ in range(200):
            aged_engine.repair_tick(is_sleeping=True, growth_factor=0.0)

        # Plastic strain mostly preserved (core evidence of irreversible aging)
        plastic_ratio = ch.plastic_strain / plastic_before
        assert plastic_ratio > 0.5, "Sleep should not significantly reverse plastic strain"


# ============================================================================
# 5. Work Hardening
# ============================================================================

class TestWorkHardening:
    """Work hardening: plastic deformation increases yield strength."""

    def test_yield_increases_after_plastic(self, engine):
        """Yield strength increases after plastic deformation."""
        high_i = (YIELD_STRAIN / PINCH_SENSITIVITY) ** (1 / PINCH_EXPONENT) + 1.0
        for _ in range(200):
            engine.apply_current_pulse("ch1", high_i)

        ch = engine._channels["ch1"]
        assert ch.effective_yield > YIELD_STRAIN
        assert ch.work_hardening > 0


# ============================================================================
# 6. Temperature Acceleration
# ============================================================================

class TestTemperatureAcceleration:
    """Arrhenius temperature acceleration: high stress (high temp) → accelerated aging."""

    def test_high_temp_more_fatigue(self):
        """High temperature (high stress) → more fatigue accumulation."""
        eng_cold = PinchFatigueEngine()
        eng_hot = PinchFatigueEngine()

        high_i = (YIELD_STRAIN / PINCH_SENSITIVITY) ** (1 / PINCH_EXPONENT) + 1.0
        for _ in range(200):
            eng_cold.apply_current_pulse("ch1", high_i, temperature=0.2)
            eng_hot.apply_current_pulse("ch1", high_i, temperature=0.9)

        f_cold = eng_cold._channels["ch1"].fatigue_damage
        f_hot = eng_hot._channels["ch1"].fatigue_damage
        assert f_hot > f_cold, "High temperature should accelerate aging"


# ============================================================================
# 7. Aging → Impedance Drift → Γ
# ============================================================================

class TestAgingImpedanceDrift:
    """Plastic strain → geometric deformation → impedance drift → Γ drift."""

    def test_aged_channel_has_impedance_drift(self, aged_engine):
        """Aged channel impedance has deviated from design value."""
        ch = aged_engine._channels["visual"]
        assert ch.impedance_drift > 0
        assert ch.current_impedance > ch.design_impedance

    def test_aged_channel_has_gamma_drift(self, aged_engine):
        """Aged channel produces nonzero Γ."""
        ch = aged_engine._channels["visual"]
        assert ch.gamma_drift > 0

    def test_new_channel_no_drift(self, engine):
        """New channel has no drift."""
        engine.apply_current_pulse("ch1", 0.1)
        assert engine.get_impedance_drift("ch1") == 0.0
        assert engine.get_gamma_aging("ch1") == 0.0


# ============================================================================
# 8. Multi-Channel Independent Aging
# ============================================================================

class TestMultiChannelAging:
    """Each channel ages independently."""

    def test_channels_age_independently(self, engine):
        """High-load channel ages faster than low-load channel."""
        high_i = (YIELD_STRAIN / PINCH_SENSITIVITY) ** (1 / PINCH_EXPONENT) + 1.0
        for _ in range(300):
            engine.apply_current_pulse("heavy", high_i)
            engine.apply_current_pulse("light", 0.1)

        age_heavy = engine.get_channel_age("heavy")
        age_light = engine.get_channel_age("light")
        assert age_heavy > age_light

    def test_nonexistent_channel_returns_zero(self, engine):
        """Nonexistent channel returns 0."""
        assert engine.get_channel_age("doesnt_exist") == 0.0
        assert engine.get_gamma_aging("doesnt_exist") == 0.0
        assert engine.get_life_expectancy("doesnt_exist") == 1.0


# ============================================================================
# 9. Life Expectancy Estimation
# ============================================================================

class TestLifeExpectancy:
    """Structural integrity / life expectancy estimation."""

    def test_new_channel_full_life(self, engine):
        """New channel life = 1.0."""
        engine.apply_current_pulse("ch1", 0.1)
        assert engine.get_life_expectancy("ch1") == 1.0

    def test_aged_channel_reduced_life(self, aged_engine):
        """Aged channel has reduced life expectancy."""
        life = aged_engine.get_life_expectancy("visual")
        assert life < 1.0


# ============================================================================
# 10. tick() Integration
# ============================================================================

class TestTickIntegration:
    """tick() full-channel integration."""

    def test_tick_returns_aging_signal(self, engine):
        """tick returns AgingSignal."""
        signal = engine.tick({"ch1": 0.5, "ch2": 0.3})
        assert isinstance(signal, AgingSignal)
        assert signal.total_channels == 2

    def test_tick_sleeping_no_new_damage(self, engine):
        """No new Lorentz compression stress applied during sleep."""
        # First create some damage
        engine.tick({"ch1": 0.9})
        ch = engine._channels["ch1"]
        plastic_before = ch.plastic_strain

        # Sleep tick
        engine.tick({"ch1": 0.9}, is_sleeping=True)
        # Sleep should not increase plasticity (activity is skipped)
        assert ch.plastic_strain <= plastic_before

    def test_tick_history_accumulates(self, engine):
        """History records accumulate."""
        for i in range(10):
            engine.tick({"ch1": 0.5})
        assert len(engine._age_history) >= 10

    def test_growth_factor_slows_aging(self):
        """BDNF (neurotrophic factor) slows aging."""
        eng_no_bdnf = PinchFatigueEngine()
        eng_bdnf = PinchFatigueEngine()

        high_i = (YIELD_STRAIN / PINCH_SENSITIVITY) ** (1 / PINCH_EXPONENT) + 1.0
        for _ in range(200):
            eng_no_bdnf.apply_current_pulse("ch1", high_i)
            eng_bdnf.apply_current_pulse("ch1", high_i)

        for _ in range(200):
            eng_no_bdnf.repair_tick(is_sleeping=True, growth_factor=0.0)
            eng_bdnf.repair_tick(is_sleeping=True, growth_factor=1.0)

        p_no = eng_no_bdnf._channels["ch1"].plastic_strain
        p_bdnf = eng_bdnf._channels["ch1"].plastic_strain
        assert p_bdnf < p_no, "BDNF should reduce plastic residue"


# ============================================================================
# 11. Stats and State Query
# ============================================================================

class TestStatsAndState:
    """Stats and state query interface."""

    def test_get_stats_empty(self, engine):
        """Stats of an empty engine."""
        stats = engine.get_stats()
        assert stats["total_channels"] == 0

    def test_get_stats_populated(self, aged_engine):
        """Stats with data."""
        stats = aged_engine.get_stats()
        assert stats["total_channels"] == 1
        assert stats["total_pinch_events"] > 0

    def test_get_state_includes_channels(self, aged_engine):
        """Full state includes channel details."""
        state = aged_engine.get_state()
        assert "channels" in state
        assert "visual" in state["channels"]

    def test_get_channel_state(self, aged_engine):
        """Query single channel state."""
        cs = aged_engine.get_channel_state("visual")
        assert cs is not None
        assert "fatigue_damage" in cs
        assert "structural_integrity" in cs

    def test_get_channel_state_none(self, engine):
        """Nonexistent channel returns None."""
        assert engine.get_channel_state("nope") is None

    def test_get_waveforms(self, engine):
        """Waveform data."""
        for _ in range(5):
            engine.tick({"ch1": 0.3})
        wf = engine.get_waveforms()
        assert "age_factor" in wf
        assert len(wf["age_factor"]) >= 5

    def test_reset(self, aged_engine):
        """Reset clears all state."""
        aged_engine.reset()
        assert len(aged_engine._channels) == 0
        assert aged_engine._total_plastic_strain == 0.0


# ============================================================================
# 12. ConductorState Properties
# ============================================================================

class TestConductorState:
    """ConductorState dataclass properties."""

    def test_age_factor_bounded(self):
        cs = ConductorState(channel_id="test", fatigue_damage=1.5)
        assert cs.age_factor == 1.0

    def test_structural_integrity_complement(self):
        cs = ConductorState(channel_id="test", fatigue_damage=0.3)
        assert abs(cs.structural_integrity - 0.7) < 0.001

    def test_is_degraded_threshold(self):
        assert not ConductorState(channel_id="a", fatigue_damage=0.2).is_degraded
        assert ConductorState(channel_id="b", fatigue_damage=0.4).is_degraded


# ============================================================================
# 13. AliceBrain Integration
# ============================================================================

class TestAliceBrainIntegration:
    """PinchFatigueEngine integration within AliceBrain."""

    def test_brain_has_pinch_fatigue(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        assert hasattr(brain, "pinch_fatigue")
        assert isinstance(brain.pinch_fatigue, PinchFatigueEngine)

    def test_perceive_includes_aging(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        stimulus = np.random.randn(64)
        result = brain.perceive(stimulus)
        assert "pinch_fatigue" in result
        pf = result["pinch_fatigue"]
        assert "mean_age" in pf
        assert "cognitive_impact" in pf

    def test_introspect_includes_pinch_fatigue(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        brain.perceive(np.random.randn(64))
        intro = brain.introspect()
        assert "pinch_fatigue" in intro["subsystems"]
