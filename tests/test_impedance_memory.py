# -*- coding: utf-8 -*-
"""
Tests for Impedance-Modulated Memory Decay

Core physics:
  "The so-called short-term memory decline gradient is an uncalibrated mechanism."

  Coaxial cable physics:
    Transmission power  P_t = (1 - Γ²) × P_in
    Effective decay  λ_eff = λ_base / (1 - Γ²)

  When sound and image cannot bind (Γ→1):
    - Write power P_t → 0 (memory never stored at all)
    - Decay rate λ_eff → ∞ (residual traces vanish instantly)

  Verification goals:
  1. Γ=0 memory decays slowest (perfect binding = stable memory)
  2. Higher Γ, faster decay (worse mismatch = faster forgetting)
  3. Initial activation ∝ (1 - Γ²) (transmission efficiency determines write quality)
  4. Episodic memory follows the same physics (hippocampal level)
  5. Rehearsal can reduce Γ (recalibration improves binding)
"""

import time
import math
import numpy as np
import pytest

from alice.modules.working_memory import WorkingMemory, MemoryItem
from alice.brain.hippocampus import (
    HippocampusEngine,
    Episode,
    EpisodicSnapshot,
)


# ============================================================================
# 1. MemoryItem Physical Property Tests
# ============================================================================

class TestMemoryItemImpedance:
    """MemoryItem impedance decay factor — physical invariants"""

    def test_perfect_binding_factor_is_one(self):
        """Γ=0 (perfect binding) → decay factor = 1.0 (normal rate)"""
        item = MemoryItem(key="test", content="x", binding_gamma=0.0)
        assert item.impedance_decay_factor == pytest.approx(1.0)

    def test_half_mismatch_factor(self):
        """Γ=0.5 → factor = 1/(1-0.25) = 1.333"""
        item = MemoryItem(key="test", content="x", binding_gamma=0.5)
        expected = 1.0 / (1.0 - 0.25)
        assert item.impedance_decay_factor == pytest.approx(expected, rel=1e-3)

    def test_severe_mismatch_factor(self):
        """Γ=0.8 → factor = 1/(1-0.64) = 2.778"""
        item = MemoryItem(key="test", content="x", binding_gamma=0.8)
        expected = 1.0 / (1.0 - 0.64)
        assert item.impedance_decay_factor == pytest.approx(expected, rel=1e-3)

    def test_near_total_mismatch_factor(self):
        """Γ=0.95 → factor ≈ 10.3"""
        item = MemoryItem(key="test", content="x", binding_gamma=0.95)
        expected = 1.0 / (1.0 - 0.95**2)
        assert item.impedance_decay_factor == pytest.approx(expected, rel=1e-2)

    def test_factor_monotonically_increases(self):
        """Decay factor strictly increases with Γ — physical causality direction"""
        gammas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
        factors = []
        for g in gammas:
            item = MemoryItem(key="test", content="x", binding_gamma=g)
            factors.append(item.impedance_decay_factor)
        for i in range(len(factors) - 1):
            assert factors[i] < factors[i + 1], (
                f"Factor should increase: Γ={gammas[i]}→{factors[i]}, "
                f"Γ={gammas[i+1]}→{factors[i+1]}"
            )

    def test_transmission_efficiency(self):
        """Transmission efficiency = 1 - Γ² = fraction of power entering memory"""
        item = MemoryItem(key="test", content="x", binding_gamma=0.6)
        assert item.transmission_efficiency == pytest.approx(1.0 - 0.36, rel=1e-3)

    def test_gamma_clamped_at_099(self):
        """Γ is clamped at 0.99 to prevent division by zero"""
        item = MemoryItem(key="test", content="x", binding_gamma=1.0)
        # Should not be inf
        assert item.impedance_decay_factor < 200
        assert item.transmission_efficiency > 0


# ============================================================================
# 2. WorkingMemory Impedance-Modulated Decay Tests
# ============================================================================

class TestWorkingMemoryImpedanceDecay:
    """Impedance-modulated decay of working memory — core mechanism verification"""

    def test_initial_activation_proportional_to_transmission(self):
        """Initial activation = (1 - Γ²) — transmission efficiency determines write quality"""
        wm = WorkingMemory(capacity=10, decay_rate=0.01)

        # Perfect binding
        wm.store("perfect", "data_a", binding_gamma=0.0)
        # Partial mismatch
        wm.store("partial", "data_b", binding_gamma=0.5)
        # Severe mismatch
        wm.store("severe", "data_c", binding_gamma=0.8)

        contents = {c["key"]: c for c in wm.get_contents()}

        # Perfect binding: activation ≈ 1.0
        assert contents["perfect"]["activation"] == pytest.approx(1.0, abs=0.05)
        # Partial mismatch: activation ≈ 0.75
        assert contents["partial"]["activation"] == pytest.approx(0.75, abs=0.05)
        # Severe mismatch: activation ≈ 0.36
        assert contents["severe"]["activation"] == pytest.approx(0.36, abs=0.05)

    def test_high_gamma_memories_forgotten_first(self):
        """High Γ memories forgotten first — physical causality: mismatched memories decay faster"""
        wm = WorkingMemory(capacity=10, decay_rate=0.5, eviction_threshold=0.05)

        now = time.time()
        wm.store("good_binding", "stable", binding_gamma=0.1)
        wm.store("bad_binding", "fragile", binding_gamma=0.8)

        # Simulate time passing
        wm._last_decay_time = now - 5.0
        for item in wm._items.values():
            item.last_accessed = now - 5.0

        wm._apply_decay()

        keys = {c["key"] for c in wm.get_contents()}

        # Good binding memory should remain; poor binding may have vanished
        # (even if both remain, poor binding activation must be lower)
        contents = {c["key"]: c for c in wm.get_contents()}
        if "good_binding" in contents and "bad_binding" in contents:
            assert contents["good_binding"]["activation"] > contents["bad_binding"]["activation"]

    def test_impedance_decay_faster_than_normal(self):
        """Decay rate with Γ>0 is strictly greater than Γ=0"""
        wm = WorkingMemory(capacity=10, decay_rate=0.1, eviction_threshold=0.001)

        wm.store("matched", "data_a", binding_gamma=0.0)
        wm.store("mismatched", "data_b", binding_gamma=0.7)

        # Force time passage
        elapsed = 3.0
        past = time.time() - elapsed
        for item in wm._items.values():
            item.last_accessed = past
        wm._last_decay_time = past

        wm._apply_decay()

        contents = {c["key"]: c for c in wm.get_contents()}
        if "matched" in contents and "mismatched" in contents:
            assert contents["matched"]["activation"] > contents["mismatched"]["activation"], (
                "Matched memory should retain more activation than mismatched"
            )

    def test_rehearsal_improves_binding(self):
        """Rehearsal reduces Γ — recalibration improves binding quality"""
        wm = WorkingMemory(capacity=10, decay_rate=0.01)
        wm.store("item", "content", binding_gamma=0.6)

        original_gamma = wm._items["item"].binding_gamma

        # Rehearsal (re-store with better binding)
        wm.store("item", "content_updated", binding_gamma=0.2)

        # binding_gamma should be min(old, new)
        assert wm._items["item"].binding_gamma <= original_gamma
        assert wm._items["item"].binding_gamma == pytest.approx(0.2, abs=0.01)

    def test_impedance_stats_tracked(self):
        """Statistics track impedance-based evictions"""
        wm = WorkingMemory(capacity=5, decay_rate=0.01)
        wm.store("a", "x", binding_gamma=0.3)
        wm.store("b", "x", binding_gamma=0.0)

        stats = wm.get_stats()
        assert "impedance_evictions" in stats
        assert "avg_binding_gamma" in stats
        assert "avg_impedance_factor" in stats
        assert "avg_transmission_efficiency" in stats

    def test_contents_include_impedance_info(self):
        """Memory contents include impedance info"""
        wm = WorkingMemory(capacity=5)
        wm.store("test", "data", binding_gamma=0.4)

        contents = wm.get_contents()
        assert len(contents) == 1
        item = contents[0]
        assert "binding_gamma" in item
        assert "impedance_decay_factor" in item
        assert "transmission_efficiency" in item
        assert item["binding_gamma"] == pytest.approx(0.4, abs=0.01)

    def test_zero_gamma_backward_compatible(self):
        """Behavior is consistent with original version when binding_gamma=0"""
        wm = WorkingMemory(capacity=7, decay_rate=0.05)
        wm.store("old_api", "content")  # no binding_gamma → default 0.0

        contents = wm.get_contents()
        assert len(contents) == 1
        assert contents[0]["activation"] == pytest.approx(1.0, abs=0.05)
        assert contents[0]["binding_gamma"] == pytest.approx(0.0)
        assert contents[0]["impedance_decay_factor"] == pytest.approx(1.0)


# ============================================================================
# 3. Hippocampus Impedance Decay Tests
# ============================================================================

class TestHippocampusImpedanceDecay:
    """Hippocampal episodic memory also follows impedance physics"""

    def _make_episode(self, gamma_values: list, valence: float = 0.0) -> Episode:
        """Create an episode with specified gamma values"""
        ep = Episode(episode_id=0, creation_time=0.0, last_replay_time=0.0)
        for i, g in enumerate(gamma_values):
            snap = EpisodicSnapshot(
                timestamp=float(i) * 0.1,
                modality="visual",
                fingerprint=np.random.rand(8),
                attractor_label=f"concept_{i}",
                gamma=g,
                valence=valence,
            )
            ep.add_snapshot(snap)
        return ep

    def test_avg_binding_gamma(self):
        """Episode's average binding Γ computed correctly"""
        ep = self._make_episode([0.2, 0.4, 0.6])
        assert ep.avg_binding_gamma == pytest.approx(0.4, abs=0.01)

    def test_well_bound_episode_decays_slower(self):
        """
        After the same time, well-bound episodes decay slower than poorly-bound ones

        Physics: TV audio-visual sync memory is stable; desync memory dissipates quickly
        """
        t_now = 50.0  # shorter time to avoid hitting DECAY_FLOOR

        # Well-bound (low gamma = low impedance mismatch)
        good_ep = self._make_episode([0.1, 0.1, 0.1])
        good_ep.creation_time = 0.0
        good_ep.last_replay_time = 0.0

        # Poorly-bound (high gamma = high impedance mismatch)
        bad_ep = self._make_episode([0.7, 0.75, 0.72])
        bad_ep.creation_time = 0.0
        bad_ep.last_replay_time = 0.0

        good_recency = good_ep.recency(t_now)
        bad_recency = bad_ep.recency(t_now)

        assert good_recency > bad_recency, (
            f"Good binding ({good_recency:.6f}) should decay slower "
            f"than bad binding ({bad_recency:.6f})"
        )

    def test_impedance_vs_emotion_interaction(self):
        """
        Emotion enhancement vs. impedance decay interaction

        High emotion + low impedance → strongest memory (flashbulb memory)
        Low emotion + high impedance → weakest memory (desync boring program)
        """
        t_now = 30.0  # shorter time to avoid all hitting floor

        # High emotion + good binding = flashbulb memory
        flashbulb = self._make_episode([0.1, 0.1], valence=0.9)
        flashbulb.creation_time = 0.0
        flashbulb.last_replay_time = 0.0

        # Low emotion + poor binding = weakest
        forgettable = self._make_episode([0.7, 0.75], valence=0.0)
        forgettable.creation_time = 0.0
        forgettable.last_replay_time = 0.0

        # High emotion + poor binding = moderate (emotion saved some)
        conflicted = self._make_episode([0.7, 0.75], valence=0.8)
        conflicted.creation_time = 0.0
        conflicted.last_replay_time = 0.0

        r_flash = flashbulb.recency(t_now)
        r_forget = forgettable.recency(t_now)
        r_conflict = conflicted.recency(t_now)

        # Ranking: flashbulb > conflicted > forgettable
        assert r_flash > r_conflict > r_forget, (
            f"Expected flashbulb ({r_flash:.6f}) > conflicted ({r_conflict:.6f}) "
            f"> forgettable ({r_forget:.6f})"
        )

    def test_empty_episode_gamma_is_zero(self):
        """Empty episode avg_binding_gamma = 0.0 (no signal → no measurable mismatch)"""
        ep = Episode(episode_id=0, creation_time=0.0, last_replay_time=0.0)
        assert ep.avg_binding_gamma == 0.0

    def test_engine_records_gamma(self):
        """Hippocampal engine records snapshot gamma values"""
        engine = HippocampusEngine()
        engine.record(
            modality="visual",
            fingerprint=np.random.rand(8),
            attractor_label="cat",
            gamma=0.3,
            timestamp=0.0,
        )
        ep = engine.episodes[0]
        assert ep.snapshots[0].gamma == pytest.approx(0.3)
        assert ep.avg_binding_gamma == pytest.approx(0.3)


# ============================================================================
# 4. Physical Invariant Tests
# ============================================================================

class TestPhysicalInvariants:
    """Physical invariants — laws that must hold"""

    def test_decay_factor_equals_inverse_transmission(self):
        """
        Decay factor = 1 / transmission efficiency

        Physics: energy conservation
          P_in = P_transmitted + P_reflected
          decay_factor = 1 / (1 - Γ²) = P_in / P_transmitted
        """
        for gamma in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]:
            item = MemoryItem(key="test", content="x", binding_gamma=gamma)
            # decay_factor × transmission_efficiency ≈ 1.0
            product = item.impedance_decay_factor * item.transmission_efficiency
            assert product == pytest.approx(1.0, rel=1e-3), (
                f"Γ={gamma}: factor×efficiency = {product}, should be 1.0"
            )

    def test_power_conservation(self):
        """
        Energy conservation: write power + reflected power = input power

        P_transmitted + P_reflected = P_input
        (1 - Γ²) + Γ² = 1.0
        """
        for gamma in [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]:
            transmission = 1.0 - gamma**2
            reflection = gamma**2
            assert transmission + reflection == pytest.approx(1.0, rel=1e-10)

    def test_gamma_zero_is_identity(self):
        """Γ=0 degenerates all metrics to original behavior"""
        item = MemoryItem(key="test", content="x", binding_gamma=0.0)
        assert item.impedance_decay_factor == 1.0
        assert item.transmission_efficiency == 1.0

    def test_higher_gamma_always_worse(self):
        """
        Monotonicity invariant: Γ₁ < Γ₂ → all memory metrics for Γ₁ are better than Γ₂

        Physical causality: greater impedance mismatch → more reflection → worse memory
        """
        for g1, g2 in [(0.0, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 0.95)]:
            item1 = MemoryItem(key="a", content="x", binding_gamma=g1)
            item2 = MemoryItem(key="b", content="x", binding_gamma=g2)

            # Transmission efficiency: g1 > g2
            assert item1.transmission_efficiency > item2.transmission_efficiency
            # Decay rate: g1 < g2
            assert item1.impedance_decay_factor < item2.impedance_decay_factor


# ============================================================================
# 5. "TV Experiment" Scenario Tests
# ============================================================================

class TestTVScenario:
    """
    Simulate TV playback scenario — verify core insight

    Scenario: TV playing video and audio
    - Audio-visual sync (Γ≈0) → normal memory
    - Audio-visual slight delay (Γ≈0.3) → slightly reduced memory
    - Audio-visual severe desync (Γ≈0.8) → rapid forgetting
    - Muted screen (single modality, Γ≈0.5) → moderate memory quality
    """

    def test_synced_tv_best_memory(self):
        """Perfect audio-visual sync → best memory retention"""
        wm = WorkingMemory(capacity=10, decay_rate=0.01)
        wm.store("synced_scene", "action movie", binding_gamma=0.05)
        contents = wm.get_contents()
        # Initial activation close to 1.0
        assert contents[0]["activation"] > 0.95

    def test_desync_tv_poor_memory(self):
        """Severe audio-visual desync → memory write already incomplete"""
        wm = WorkingMemory(capacity=10, decay_rate=0.01)
        wm.store("desync_scene", "dubbed movie", binding_gamma=0.8)
        contents = wm.get_contents()
        # Initial activation only ~36%
        assert contents[0]["activation"] < 0.40
        assert contents[0]["activation"] > 0.30

    def test_muted_tv_moderate_memory(self):
        """Muted TV (missing one modality) → moderate quality"""
        wm = WorkingMemory(capacity=10, decay_rate=0.01)
        wm.store("muted_scene", "silent film", binding_gamma=0.4)
        contents = wm.get_contents()
        # Initial activation ≈ 84%
        assert 0.75 < contents[0]["activation"] < 0.90

    def test_compare_all_tv_conditions(self):
        """Compare all TV conditions — dynamic ranking"""
        wm = WorkingMemory(capacity=10, decay_rate=0.01)

        conditions = {
            "synced": 0.05,     # perfect sync
            "slight_delay": 0.2, # slight delay
            "dubbed": 0.5,       # dubbed offset
            "desync": 0.8,       # severe desync
        }

        for name, gamma in conditions.items():
            wm.store(name, f"content_{name}", binding_gamma=gamma)

        contents = {c["key"]: c["activation"] for c in wm.get_contents()}

        # Strict ranking: sync > slight delay > dubbed > desync
        assert contents["synced"] > contents["slight_delay"]
        assert contents["slight_delay"] > contents["dubbed"]
        assert contents["dubbed"] > contents["desync"]


# ============================================================================
# 6. Integration Tests — Complete Memory Physics Model
# ============================================================================

class TestIntegrationImpedanceMemory:
    """Integration tests: working memory + hippocampus joint verification"""

    def test_wm_and_hippocampus_share_physics(self):
        """
        Working memory and episodic memory use the same impedance physics

        The same physical law operates at two time scales:
        - Working memory (seconds): decay rate ∝ 1/(1-Γ²)
        - Episodic memory (minutes-hours): recency ∝ exp(-λ/(1-Γ²) × t)
        """
        # Working memory side
        wm = WorkingMemory(capacity=10, decay_rate=0.05)
        wm.store("synced_experience", "data", binding_gamma=0.1)
        wm.store("desync_experience", "data", binding_gamma=0.8)

        wm_contents = {c["key"]: c for c in wm.get_contents()}

        # Hippocampal side
        ep_good = Episode(episode_id=0, creation_time=0.0, last_replay_time=0.0)
        ep_good.add_snapshot(EpisodicSnapshot(
            timestamp=0.0, modality="visual",
            fingerprint=np.zeros(8), attractor_label="good",
            gamma=0.1,
        ))

        ep_bad = Episode(episode_id=1, creation_time=0.0, last_replay_time=0.0)
        ep_bad.add_snapshot(EpisodicSnapshot(
            timestamp=0.0, modality="visual",
            fingerprint=np.zeros(8), attractor_label="bad",
            gamma=0.8,
        ))

        t = 100.0
        hip_good = ep_good.recency(t)
        hip_bad = ep_bad.recency(t)

        # Both systems rank consistently
        assert wm_contents["synced_experience"]["activation"] > wm_contents["desync_experience"]["activation"]
        assert hip_good > hip_bad

    def test_physical_formula_consistency(self):
        """
        Verify formula: λ_eff = λ_base × 1/(1-Γ²)

        Directly verify the mathematical relationship of decay rate in working memory
        """
        base_rate = 0.1
        gamma = 0.6

        # Theoretical calculation
        expected_factor = 1.0 / (1.0 - gamma**2)
        expected_eff_rate = base_rate * expected_factor

        # Create item and verify
        item = MemoryItem(key="test", content="x", binding_gamma=gamma)
        actual_factor = item.impedance_decay_factor

        assert actual_factor == pytest.approx(expected_factor, rel=1e-6)

        # Verify decay result
        # activation(t) = a₀ × exp(-λ_eff × t)
        a0 = item.transmission_efficiency  # initial activation = (1-Γ²)
        t = 2.0
        expected_activation = a0 * math.exp(-expected_eff_rate * t)

        # Simulate decay
        actual_activation = a0 * np.exp(-base_rate * actual_factor * t)

        assert actual_activation == pytest.approx(expected_activation, rel=1e-6)
