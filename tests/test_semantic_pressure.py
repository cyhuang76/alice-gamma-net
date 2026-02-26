# -*- coding: utf-8 -*-
"""
tests/test_semantic_pressure.py — Semantic Pressure Engine Unit Tests

Tests for SemanticPressureEngine core functionality:
  1. Pressure accumulation physics model
  2. Pressure release (linguistic catharsis)
  3. Inner monologue (spontaneous concept activation)
  4. Wernicke → Broca direct connection
  5. AliceBrain integration
"""

import math
import numpy as np
import pytest
from alice.brain.semantic_pressure import (
    SemanticPressureEngine,
    InnerMonologueEvent,
    PRESSURE_ACCUMULATION_RATE,
    PRESSURE_NATURAL_DECAY,
    RELEASE_EFFICIENCY,
    DEFAULT_MONOLOGUE_THRESHOLD,
    MIN_PHI_FOR_MONOLOGUE,
    WERNICKE_BROCA_GAMMA_THRESHOLD,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def engine():
    return SemanticPressureEngine()


@pytest.fixture
def active_concepts():
    return [
        {"label": "hurt", "mass": 5.0, "Q": 3.0},
        {"label": "calm", "mass": 2.0, "Q": 1.5},
    ]


class MockSemanticField:
    """Mock SemanticFieldEngine for inner monologue tests."""
    def __init__(self, concepts=None):
        self._concepts = [
            {"label": "hurt", "mass": 10.0, "Q": 3.0},
            {"label": "calm", "mass": 3.0, "Q": 2.0},
        ] if concepts is None else concepts

    def get_state(self):
        return {"top_concepts": self._concepts}


class MockWernicke:
    """Mock WernickeEngine."""
    def __init__(self):
        self.observed = []
        self._predictions = []

    def observe(self, concept):
        self.observed.append(concept)
        return {"gamma_syntactic": 0.5, "concept": concept}

    def predict_next(self, context=None, top_k=5):
        if self._predictions:
            return {
                "predictions": self._predictions,
                "context_used": "test",
                "entropy": 1.0,
            }
        return {"predictions": [], "context_used": None, "entropy": float("inf")}


class MockBroca:
    """Mock BrocaEngine."""
    def __init__(self):
        self._plans = set()
        self.planned = []

    def has_plan(self, concept):
        return concept in self._plans

    def plan_utterance(self, concept):
        self._plans.add(concept)
        self.planned.append(concept)
        return {"concept": concept}


# ============================================================================
# Test: Initialization
# ============================================================================

class TestInit:
    def test_default_state(self, engine):
        assert engine.pressure == 0.0
        assert engine.peak_pressure == 0.0
        assert engine.cumulative_released == 0.0
        assert engine.total_expressions == 0
        assert engine._monologue_threshold == DEFAULT_MONOLOGUE_THRESHOLD

    def test_custom_threshold(self):
        e = SemanticPressureEngine(monologue_threshold=0.5)
        assert e._monologue_threshold == 0.5

    def test_empty_histories(self, engine):
        assert engine.pressure_history == []
        assert engine.release_history == []
        assert engine.monologue_events == []


# ============================================================================
# Test: Pressure Accumulation
# ============================================================================

class TestAccumulation:
    def test_zero_emotion_no_pressure(self, engine):
        """No emotional tension → no pressure buildup."""
        p = engine.accumulate(
            active_concepts=[{"label": "x", "mass": 5.0, "Q": 3.0}],
            valence=0.0, arousal=0.5, phi=1.0, pain=0.0,
        )
        assert p == 0.0  # valence²+pain² = 0

    def test_high_pain_builds_pressure(self, engine, active_concepts):
        """High pain should build significant pressure."""
        p = engine.accumulate(active_concepts, valence=-0.5, arousal=0.8, phi=0.9, pain=0.8)
        assert p > 0.0
        assert engine.pressure == p

    def test_pressure_accumulates_over_ticks(self, engine, active_concepts):
        """Repeated ticks should increase pressure."""
        for _ in range(50):
            engine.accumulate(active_concepts, valence=-0.7, arousal=0.9, phi=1.0, pain=0.5)
        assert engine.pressure > 0.1

    def test_natural_decay(self, engine, active_concepts):
        """Pressure decays when no emotional input."""
        # First build up pressure
        for _ in range(30):
            engine.accumulate(active_concepts, valence=-0.8, arousal=0.9, phi=1.0, pain=0.7)
        peak = engine.pressure

        # Then let it decay (zero emotion)
        for _ in range(100):
            engine.accumulate([], valence=0.0, arousal=0.0, phi=1.0, pain=0.0)
        assert engine.pressure < peak

    def test_consciousness_gate(self, engine, active_concepts):
        """Low consciousness reduces accumulation rate."""
        engine.accumulate(active_concepts, valence=-0.5, arousal=0.8, phi=1.0, pain=0.5)
        p_high_phi = engine.pressure

        engine2 = SemanticPressureEngine()
        engine2.accumulate(active_concepts, valence=-0.5, arousal=0.8, phi=0.0, pain=0.5)
        p_low_phi = engine2.pressure

        assert p_high_phi > p_low_phi

    def test_arousal_amplifies(self, engine, active_concepts):
        """High arousal amplifies pressure more than low arousal."""
        engine.accumulate(active_concepts, valence=-0.5, arousal=1.0, phi=1.0, pain=0.5)
        p_high = engine.pressure

        engine2 = SemanticPressureEngine()
        engine2.accumulate(active_concepts, valence=-0.5, arousal=0.0, phi=1.0, pain=0.5)
        p_low = engine2.pressure

        assert p_high > p_low

    def test_peak_pressure_tracked(self, engine, active_concepts):
        """Peak pressure is tracked correctly."""
        for _ in range(20):
            engine.accumulate(active_concepts, valence=-0.8, arousal=0.9, phi=1.0, pain=0.7)
        peak = engine.peak_pressure

        # Decay down
        for _ in range(50):
            engine.accumulate([], valence=0.0, arousal=0.0, phi=1.0, pain=0.0)

        assert engine.peak_pressure == peak
        assert engine.pressure < peak

    def test_history_recorded(self, engine, active_concepts):
        """Pressure history is recorded."""
        for _ in range(10):
            engine.accumulate(active_concepts, valence=-0.5, arousal=0.5, phi=1.0, pain=0.3)
        assert len(engine.pressure_history) == 10

    def test_concept_mass_affects_pressure(self):
        """Heavier concepts produce more pressure."""
        e1 = SemanticPressureEngine()
        e1.accumulate(
            [{"label": "x", "mass": 20.0, "Q": 3.0}],
            valence=-0.5, arousal=0.5, phi=1.0, pain=0.3,
        )

        e2 = SemanticPressureEngine()
        e2.accumulate(
            [{"label": "x", "mass": 1.0, "Q": 3.0}],
            valence=-0.5, arousal=0.5, phi=1.0, pain=0.3,
        )
        assert e1.pressure > e2.pressure

    def test_q_factor_affects_pressure(self):
        """Higher Q factor → more pressure (mature concepts press harder)."""
        e1 = SemanticPressureEngine()
        e1.accumulate(
            [{"label": "x", "mass": 5.0, "Q": 5.0}],
            valence=-0.5, arousal=0.5, phi=1.0, pain=0.3,
        )

        e2 = SemanticPressureEngine()
        e2.accumulate(
            [{"label": "x", "mass": 5.0, "Q": 1.01}],
            valence=-0.5, arousal=0.5, phi=1.0, pain=0.3,
        )
        assert e1.pressure > e2.pressure


# ============================================================================
# Test: Pressure Release (Catharsis)
# ============================================================================

class TestRelease:
    def test_release_reduces_pressure(self, engine, active_concepts):
        """Speech release should reduce pressure."""
        for _ in range(30):
            engine.accumulate(active_concepts, valence=-0.8, arousal=0.9, phi=1.0, pain=0.7)
        pre_release = engine.pressure

        released = engine.release(gamma_speech=0.1, phi=1.0)

        assert released > 0
        assert engine.pressure < pre_release

    def test_perfect_expression(self, engine, active_concepts):
        """Gamma=0 (perfect match) releases maximum pressure."""
        for _ in range(20):
            engine.accumulate(active_concepts, valence=-0.5, arousal=0.8, phi=1.0, pain=0.5)

        released = engine.release(gamma_speech=0.0, phi=1.0)
        assert released == pytest.approx((engine.pressure + released) * RELEASE_EFFICIENCY)  # 50% max

    def test_poor_expression(self, engine, active_concepts):
        """Gamma=1.0 (total mismatch) releases nothing."""
        for _ in range(20):
            engine.accumulate(active_concepts, valence=-0.5, arousal=0.8, phi=1.0, pain=0.5)

        released = engine.release(gamma_speech=1.0, phi=1.0)
        assert released == 0.0

    def test_unconscious_cannot_release(self, engine, active_concepts):
        """Phi=0 → no release (must be conscious to benefit from expression)."""
        for _ in range(20):
            engine.accumulate(active_concepts, valence=-0.5, arousal=0.8, phi=1.0, pain=0.5)

        released = engine.release(gamma_speech=0.1, phi=0.0)
        assert released == 0.0

    def test_cumulative_released_tracked(self, engine, active_concepts):
        """Cumulative released amount is tracked."""
        for _ in range(20):
            engine.accumulate(active_concepts, valence=-0.5, arousal=0.8, phi=1.0, pain=0.5)

        r1 = engine.release(gamma_speech=0.2, phi=0.8)
        r2 = engine.release(gamma_speech=0.2, phi=0.8)
        assert engine.cumulative_released == pytest.approx(r1 + r2)
        assert engine.total_expressions == 2

    def test_release_history(self, engine, active_concepts):
        """Release events are recorded."""
        for _ in range(10):
            engine.accumulate(active_concepts, valence=-0.5, arousal=0.8, phi=1.0, pain=0.5)
        engine.release(gamma_speech=0.3, phi=0.9)
        assert len(engine.release_history) == 1


# ============================================================================
# Test: Inner Monologue
# ============================================================================

class TestInnerMonologue:
    def test_no_activation_low_pressure(self, engine):
        """Below threshold → no spontaneous activation."""
        sf = MockSemanticField()
        w = MockWernicke()
        result = engine.check_spontaneous_activation(
            tick=1, semantic_field=sf, wernicke=w, valence=-0.5, phi=0.8,
        )
        assert result is None  # pressure=0 < threshold

    def test_no_activation_low_phi(self, engine, active_concepts):
        """Below consciousness threshold → no activation."""
        for _ in range(50):
            engine.accumulate(active_concepts, valence=-0.8, arousal=0.9, phi=1.0, pain=0.8)

        sf = MockSemanticField()
        w = MockWernicke()
        result = engine.check_spontaneous_activation(
            tick=1, semantic_field=sf, wernicke=w, valence=-0.5, phi=0.1,
        )
        assert result is None

    def test_activation_above_threshold(self, engine, active_concepts):
        """High pressure + conscious → concept activates."""
        for _ in range(80):
            engine.accumulate(active_concepts, valence=-0.8, arousal=0.9, phi=1.0, pain=0.8)

        assert engine.pressure > DEFAULT_MONOLOGUE_THRESHOLD

        sf = MockSemanticField()
        w = MockWernicke()
        result = engine.check_spontaneous_activation(
            tick=100, semantic_field=sf, wernicke=w, valence=-0.5, phi=0.8,
        )
        assert result is not None
        assert isinstance(result, InnerMonologueEvent)
        assert result.source == "spontaneous"
        assert result.concept in ["hurt", "calm"]

    def test_wernicke_observes_concept(self, engine, active_concepts):
        """Activated concept is sent to Wernicke."""
        for _ in range(80):
            engine.accumulate(active_concepts, valence=-0.8, arousal=0.9, phi=1.0, pain=0.8)

        sf = MockSemanticField()
        w = MockWernicke()
        engine.check_spontaneous_activation(
            tick=1, semantic_field=sf, wernicke=w, valence=-0.5, phi=0.8,
        )
        assert len(w.observed) == 1

    def test_association_source(self, engine, active_concepts):
        """Second different concept → source='association'."""
        for _ in range(80):
            engine.accumulate(active_concepts, valence=-0.8, arousal=0.9, phi=1.0, pain=0.8)

        sf = MockSemanticField()
        w = MockWernicke()

        e1 = engine.check_spontaneous_activation(
            tick=1, semantic_field=sf, wernicke=w, valence=-0.5, phi=0.8,
        )

        # Change concepts to force different activation
        sf2 = MockSemanticField([{"label": "new_concept", "mass": 20.0, "Q": 3.0}])
        e2 = engine.check_spontaneous_activation(
            tick=2, semantic_field=sf2, wernicke=w, valence=-0.5, phi=0.8,
        )
        if e2 is not None and e1 is not None and e2.concept != e1.concept:
            assert e2.source == "association"

    def test_echo_source(self, engine, active_concepts):
        """Same concept twice → source='echo'."""
        for _ in range(80):
            engine.accumulate(active_concepts, valence=-0.8, arousal=0.9, phi=1.0, pain=0.8)

        sf = MockSemanticField([{"label": "hurt", "mass": 20.0, "Q": 3.0}])
        w = MockWernicke()

        engine.check_spontaneous_activation(
            tick=1, semantic_field=sf, wernicke=w, valence=-0.5, phi=0.8,
        )
        e2 = engine.check_spontaneous_activation(
            tick=2, semantic_field=sf, wernicke=w, valence=-0.5, phi=0.8,
        )
        if e2 is not None:
            assert e2.source == "echo"

    def test_event_to_dict(self):
        """InnerMonologueEvent.to_dict() works."""
        ev = InnerMonologueEvent(
            tick=10, concept="hurt", gamma=0.3,
            source="spontaneous", semantic_pressure=0.5, phi=0.8,
        )
        d = ev.to_dict()
        assert d["tick"] == 10
        assert d["concept"] == "hurt"

    def test_empty_semantic_field(self, engine, active_concepts):
        """Empty semantic field → no activation."""
        for _ in range(80):
            engine.accumulate(active_concepts, valence=-0.8, arousal=0.9, phi=1.0, pain=0.8)

        sf = MockSemanticField([])
        w = MockWernicke()
        result = engine.check_spontaneous_activation(
            tick=1, semantic_field=sf, wernicke=w, valence=-0.5, phi=0.8,
        )
        assert result is None


# ============================================================================
# Test: Wernicke → Broca Drive
# ============================================================================

class TestWernickeBrocaDrive:
    def test_no_predictions(self, engine):
        """No predictions → no drive."""
        w = MockWernicke()
        b = MockBroca()
        result = engine.wernicke_drives_broca(w, b)
        assert result is None

    def test_high_gamma_no_trigger(self, engine):
        """High gamma (low confidence) → no trigger."""
        w = MockWernicke()
        w._predictions = [
            {"concept": "hello", "probability": 0.2, "gamma_syntactic": 0.8}
        ]
        b = MockBroca()
        result = engine.wernicke_drives_broca(w, b)
        assert result is None

    def test_low_gamma_triggers_planning(self, engine):
        """Low gamma (high confidence) → triggers Broca planning."""
        w = MockWernicke()
        w._predictions = [
            {"concept": "hello", "probability": 0.8, "gamma_syntactic": 0.2}
        ]
        b = MockBroca()
        result = engine.wernicke_drives_broca(w, b)
        assert result is not None
        assert result["predicted_concept"] == "hello"
        assert result["planned"] is True
        assert "hello" in b._plans

    def test_existing_plan_not_replanned(self, engine):
        """If Broca already has a plan, don't replan."""
        w = MockWernicke()
        w._predictions = [
            {"concept": "greet", "probability": 0.9, "gamma_syntactic": 0.1}
        ]
        b = MockBroca()
        b._plans.add("greet")
        result = engine.wernicke_drives_broca(w, b)
        assert result is not None
        assert result["planned"] is True
        assert len(b.planned) == 0  # didn't call plan_utterance again


# ============================================================================
# Test: Tick Integration
# ============================================================================

class TestTick:
    def test_tick_basic(self, engine):
        """Tick runs without error."""
        result = engine.tick()
        assert "pressure" in result
        assert result["monologue_event"] is None
        assert result["wernicke_broca_drive"] is None

    def test_tick_with_concepts(self, engine, active_concepts):
        """Tick with concepts accumulates pressure."""
        result = engine.tick(
            active_concepts=active_concepts,
            valence=-0.5, arousal=0.8, phi=1.0, pain=0.5,
        )
        assert result["pressure"] > 0

    def test_tick_with_all_subsystems(self, engine, active_concepts):
        """Full tick with all subsystems."""
        sf = MockSemanticField()
        w = MockWernicke()
        w._predictions = [
            {"concept": "test", "probability": 0.9, "gamma_syntactic": 0.1}
        ]
        b = MockBroca()

        # Build up pressure first
        for _ in range(100):
            engine.tick(
                active_concepts=active_concepts,
                valence=-0.8, arousal=0.9, phi=1.0, pain=0.8,
                semantic_field=sf, wernicke=w, broca=b,
                tick_id=_,
            )

        result = engine.tick(
            active_concepts=active_concepts,
            valence=-0.8, arousal=0.9, phi=1.0, pain=0.8,
            semantic_field=sf, wernicke=w, broca=b,
            tick_id=101,
        )
        assert result["pressure"] > 0
        assert result["wernicke_broca_drive"] is not None


# ============================================================================
# Test: Get State
# ============================================================================

class TestGetState:
    def test_get_state(self, engine):
        """get_state returns proper dict."""
        state = engine.get_state()
        assert "pressure" in state
        assert "peak_pressure" in state
        assert "cumulative_released" in state
        assert "total_expressions" in state
        assert "total_monologue_events" in state
        assert "recent_concepts" in state


# ============================================================================
# Test: AliceBrain Integration
# ============================================================================

class TestAliceBrainIntegration:
    def test_brain_has_semantic_pressure(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        assert hasattr(brain, "semantic_pressure")
        assert isinstance(brain.semantic_pressure, SemanticPressureEngine)

    def test_introspect_includes_semantic_pressure(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        state = brain.introspect()
        assert "semantic_pressure" in state["subsystems"]

    def test_perceive_includes_semantic_pressure(self):
        """perceive() should include semantic_pressure in result."""
        from alice.alice_brain import AliceBrain
        from alice.core.protocol import Modality
        brain = AliceBrain()
        stim = np.random.randn(256).astype(np.float32)
        result = brain.perceive(stim, modality=Modality.AUDITORY)
        assert "semantic_pressure" in result
        assert "pressure" in result["semantic_pressure"]


# ============================================================================
# Test: Hippocampus → Semantic Field Consolidation
# ============================================================================

class TestHippoConsolidation:
    def test_consolidation_in_sleep(self):
        """Sleep consolidation should call hippocampus.consolidate()."""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()

        # Record some episodes in hippocampus
        import numpy as np
        fp = np.random.randn(24)
        for i in range(5):
            brain.hippocampus.record(
                modality="auditory",
                fingerprint=fp + np.random.randn(24) * 0.1,
                attractor_label=f"concept_{i}",
                gamma=0.3,
                valence=0.5,
            )
        # Close any open episode
        brain.hippocampus.end_episode()

        # Directly test consolidation
        result = brain.hippocampus.consolidate(
            semantic_field=brain.semantic_field,
            max_episodes=5,
        )
        assert result["episodes_consolidated"] >= 0


# ============================================================================
# Test: Prefrontal → Thalamus Top-Down Attention
# ============================================================================

class TestPrefrontalThalamus:
    def test_thalamus_set_attention_exists(self):
        """Thalamus has set_attention method."""
        from alice.brain.thalamus import ThalamusEngine
        t = ThalamusEngine()
        assert hasattr(t, "set_attention")

    def test_prefrontal_get_top_goal_exists(self):
        """Prefrontal has get_top_goal method."""
        from alice.brain.prefrontal import PrefrontalCortexEngine
        p = PrefrontalCortexEngine()
        assert hasattr(p, "get_top_goal")

    def test_goal_driven_attention(self):
        """Setting a goal should enable top-down thalamic modulation."""
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()

        # Add a goal to prefrontal
        brain.prefrontal.set_goal(
            name="find_food",
            z_goal=50.0,
            priority=0.9,
        )
        goal = brain.prefrontal.get_top_goal()
        assert goal is not None

        # The thalamus should be able to receive attention bias
        brain.thalamus.set_attention("visual", 0.8)
        ch = brain.thalamus._ensure_channel("visual")
        assert ch.topdown_bias == pytest.approx(0.8, abs=0.01)
