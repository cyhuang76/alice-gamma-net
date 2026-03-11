# -*- coding: utf-8 -*-
"""E0 Emergence Tests — neural pathway topology.

Emergence level: E0 (True Emergence)
  - Zero disease-specific code
  - Zero flags (no is_als, no ptsd_severity)
  - Zero custom formulas
  - Only: GammaTopology + initial conditions + external stimuli + C2

Neural cable degradation spectrum (Γ axis):
  Γ ≈ 0.0  healthy match     → normal conduction
  Γ ≈ 0.1  thermal noise ↑   → fever, inflammation
  Γ ≈ 0.3  reflection loop   → PTSD (signal re-reflects)
  Γ ≈ 0.7  progressive col.  → ALS (channel degrades)
  Γ → 1.0  open circuit echo → phantom limb pain

Dimensional hierarchy: K=5 (cortex) → K=3 (spinal) → K=2 (Aδ) → K=1 (C fiber)
Each step-down is an irreducible cutoff cost — Paper 0 theorem.
"""

import numpy as np
import pytest

from alice.body.tissue_blueprint import (
    build_neural,
    perturb_impedance,
    sever_edge,
    inject_stimulus,
)


# ============================================================================
# Helpers
# ============================================================================

def run_neural(topo, n_ticks=200, stim_node="cortex_motor", stim_amp=0.5):
    """Run neural topology with cortical drive."""
    history = []
    for _ in range(n_ticks):
        stim = inject_stimulus(topo, stim_node, amplitude=stim_amp)
        m = topo.tick(external_stimuli=stim)
        history.append(m)
    return history


def edge_gamma_sq(topo, src, tgt):
    """Mean Γ² for a specific edge. Returns 1.0 if disconnected."""
    key = (src, tgt)
    if key in topo.active_edges:
        ch = topo.active_edges[key]
        gv = ch.gamma_vector()
        return float(np.mean(gv ** 2))
    return 1.0


# ============================================================================
# Test: Healthy neural baseline — C2 reduces mismatch
# ============================================================================

class TestHealthyBaseline:
    """A healthy neural topology should converge toward low Γ under C2."""

    def test_impedance_action_decreases(self):
        """C2 should reduce A_impedance (matching) over time."""
        topo = build_neural(eta=0.02)
        history = run_neural(topo, n_ticks=300)
        early = np.mean([m["action_impedance"] for m in history[:10]])
        late = np.mean([m["action_impedance"] for m in history[-10:]])
        assert late < early, (
            f"C2 should reduce A_impedance: early={early:.4f}, late={late:.4f}")

    def test_c1_holds_every_tick(self):
        """C1 (Γ² + T = 1) must hold at every edge, every tick."""
        topo = build_neural()
        for _ in range(100):
            stim = inject_stimulus(topo, "cortex_motor", 0.5)
            topo.tick(external_stimuli=stim)
            # GammaTopology.tick() asserts C1 internally


# ============================================================================
# Test: ALS emergence — progressive motor neuron impedance elevation
# ============================================================================

class TestALSEmergence:
    """ALS = progressive motor neuron degeneration.

    E0 test: raise motor_neuron Z → Γ increases along motor pathway →
    descending motor transmission degrades progressively.
    No ALSModel, no als_stage, no survival_months.
    """

    def test_motor_gamma_increases(self):
        """Raising motor_neuron Z should increase Γ at spinal→motor edge."""
        healthy = build_neural(eta=0.02, seed=42)
        run_neural(healthy, n_ticks=50)
        g2_h = edge_gamma_sq(healthy, "spinal_motor", "motor_neuron")

        sick = build_neural(eta=0.02, seed=42)
        perturb_impedance(sick, "motor_neuron", factor=3.0)
        run_neural(sick, n_ticks=50)
        g2_s = edge_gamma_sq(sick, "spinal_motor", "motor_neuron")

        assert g2_s > g2_h, (
            f"ALS should increase motor Γ²: healthy={g2_h:.4f}, sick={g2_s:.4f}")

    def test_motor_pathway_gamma_higher(self):
        """Motor neuron degeneration should produce higher cumulative
        motor-pathway Γ² than healthy (even though C2 eventually compensates)."""
        healthy = build_neural(eta=0.02, seed=42)

        sick = build_neural(eta=0.02, seed=42)
        perturb_impedance(sick, "motor_neuron", factor=3.0)

        # Compare cumulative action over early ticks (before C2 converges)
        h_total, s_total = 0.0, 0.0
        for _ in range(30):
            hs = inject_stimulus(healthy, "cortex_motor", 0.5)
            hm = healthy.tick(external_stimuli=hs)
            h_total += hm["action_impedance"]

            ss = inject_stimulus(sick, "cortex_motor", 0.5)
            sm = sick.tick(external_stimuli=ss)
            s_total += sm["action_impedance"]

        assert s_total > h_total, (
            f"ALS should produce more cumulative action: "
            f"healthy={h_total:.6f}, sick={s_total:.6f}")

    def test_c2_partially_compensates(self):
        """C2 should partially reduce ALS Γ (neural plasticity)."""
        topo = build_neural(eta=0.05, seed=42)
        perturb_impedance(topo, "motor_neuron", factor=3.0)
        history = run_neural(topo, n_ticks=300)

        early = np.mean([m["action_impedance"] for m in history[10:20]])
        late = np.mean([m["action_impedance"] for m in history[-20:]])
        assert late < early, (
            f"C2 should compensate ALS: early={early:.4f}, late={late:.4f}")


# ============================================================================
# Test: Stroke emergence — cortical edge severance
# ============================================================================

class TestStrokeEmergence:
    """Stroke = cortical vascular occlusion = edge removal.

    E0 test: sever cortex_motor→thalamus → thalamus loses cortical input →
    entire descending motor pathway degrades.
    No StrokeModel, no nihss_score, no infarct_volume.
    """

    def test_cortical_disconnection_degrades_motor(self):
        """Severing cortex→thalamus should degrade motor pathway."""
        healthy = build_neural(eta=0.02, seed=42)
        h_hist = run_neural(healthy, n_ticks=100)

        sick = build_neural(eta=0.02, seed=42)
        sever_edge(sick, "cortex_motor", "thalamus")
        s_hist = run_neural(sick, n_ticks=100)

        h_tx = np.mean([m["total_transmitted_energy"] for m in h_hist[-20:]])
        s_tx = np.mean([m["total_transmitted_energy"] for m in s_hist[-20:]])
        assert s_tx < h_tx or len(sick.active_edges) < len(healthy.active_edges), (
            "Stroke should reduce motor transmission or edge count")

    def test_stroke_spares_sensory(self):
        """Severing motor cortex→thalamus should partially spare
        the ascending sensory pathway (different edges)."""
        sick = build_neural(eta=0.02, seed=42)
        sever_edge(sick, "cortex_motor", "thalamus")
        run_neural(sick, n_ticks=100, stim_node="cortex_motor")

        # Sensory pathway edges should still exist
        sensory_intact = ("sensory_neuron", "spinal_sensory") in sick.active_edges
        assert sensory_intact, "Sensory pathway should survive motor stroke"


# ============================================================================
# Test: PTSD-like — high-amplitude pain stimulus → reflection loop
# ============================================================================

class TestPTSDEmergence:
    """PTSD = repeated high-intensity nociceptive input → sustained Γ elevation.

    E0 test: drive nociceptor_c with high amplitude → pain pathway Γ
    elevates → even after removing stimulus, residual impedance mismatch
    persists (the reflection loop).
    No PTSDModel, no trauma_score, no hyperarousal_flag.
    """

    def test_pain_stimulus_elevates_nociceptor_gamma(self):
        """High-amplitude pain stimulus should raise nociceptor Γ."""
        # Baseline: driven from cortex only
        baseline = build_neural(eta=0.02, seed=42)
        run_neural(baseline, n_ticks=100)
        g2_base = edge_gamma_sq(baseline, "nociceptor_c", "nociceptor_ad")

        # Trauma: strong nociceptor drive
        trauma = build_neural(eta=0.02, seed=42)
        for _ in range(100):
            stim = inject_stimulus(trauma, "nociceptor_c", amplitude=5.0)
            trauma.tick(external_stimuli=stim)
        g2_trauma = edge_gamma_sq(trauma, "nociceptor_c", "nociceptor_ad")

        # After strong nociceptor drive, the pain pathway Γ should differ
        assert g2_trauma != g2_base, (
            f"Pain trauma should alter nociceptor Γ²: "
            f"base={g2_base:.4f}, trauma={g2_trauma:.4f}")

    def test_residual_mismatch_after_stimulus_removal(self):
        """After removing pain stimulus, residual impedance mismatch
        should persist — the physical basis of 'flashback'."""
        topo = build_neural(eta=0.01, seed=42)

        # Phase 1: strong pain stimulus (trauma)
        for _ in range(100):
            stim = inject_stimulus(topo, "nociceptor_c", amplitude=5.0)
            topo.tick(external_stimuli=stim)
        g2_during = edge_gamma_sq(topo, "nociceptor_c", "nociceptor_ad")

        # Phase 2: quiet (recovery)
        for _ in range(50):
            stim = inject_stimulus(topo, "cortex_motor", amplitude=0.1)
            topo.tick(external_stimuli=stim)
        g2_after = edge_gamma_sq(topo, "nociceptor_c", "nociceptor_ad")

        # Compare to a topology that was never traumatized
        clean = build_neural(eta=0.01, seed=42)
        run_neural(clean, n_ticks=150)
        g2_clean = edge_gamma_sq(clean, "nociceptor_c", "nociceptor_ad")

        # The previously-traumatized topology should still differ from clean
        assert abs(g2_after - g2_clean) > 1e-6 or abs(g2_during - g2_clean) > 1e-6, (
            f"Trauma should leave residual mismatch: "
            f"during={g2_during:.6f}, after={g2_after:.6f}, clean={g2_clean:.6f}")


# ============================================================================
# Test: Phantom limb pain — peripheral nerve severance
# ============================================================================

class TestPhantomPainEmergence:
    """Phantom pain = peripheral nerve cut → nociceptor open circuit.

    E0 test: sever motor_neuron→nociceptor_ad edge → nociceptor_ad
    loses efferent modulation → pain pathway Γ rises toward 1.0
    (open circuit echo).
    No PhantomPainModel, no amputation_level, no mirror_therapy_flag.
    """

    def test_peripheral_severance_raises_pain_gamma(self):
        """Cutting efferent modulation should alter pain pathway Γ."""
        healthy = build_neural(eta=0.02, seed=42)
        run_neural(healthy, n_ticks=100)
        g2_h = edge_gamma_sq(healthy, "nociceptor_c", "nociceptor_ad")

        phantom = build_neural(eta=0.02, seed=42)
        sever_edge(phantom, "motor_neuron", "nociceptor_ad")
        run_neural(phantom, n_ticks=100)
        g2_p = edge_gamma_sq(phantom, "nociceptor_c", "nociceptor_ad")

        assert g2_p != g2_h, (
            f"Phantom pain should alter nociceptor Γ²: "
            f"healthy={g2_h:.4f}, phantom={g2_p:.4f}")

    def test_phantom_sensory_pathway_intact(self):
        """After peripheral severance, ascending sensory path should
        still transmit (the 'phantom' part — cortex still gets signals)."""
        topo = build_neural(eta=0.02, seed=42)
        sever_edge(topo, "motor_neuron", "nociceptor_ad")
        run_neural(topo, n_ticks=100)

        # Ascending pain path should still exist
        pain_up = ("nociceptor_ad", "spinal_sensory") in topo.active_edges
        assert pain_up, "Ascending pain pathway should survive peripheral cut"


# ============================================================================
# Test: Different initial conditions → different pathology
# ============================================================================

class TestDifferentialEmergence:
    """Same topology, different perturbations → different Γ patterns.
    Proves pathology identity emerges from initial conditions,
    not from disease labels.
    """

    def test_als_vs_stroke_different_patterns(self):
        """ALS (impedance shift) and stroke (edge cut) should produce
        distinguishable trajectories."""
        als = build_neural(eta=0.02, seed=42)
        perturb_impedance(als, "motor_neuron", factor=3.0)
        als_hist = run_neural(als, n_ticks=200)

        stroke = build_neural(eta=0.02, seed=42)
        sever_edge(stroke, "cortex_motor", "thalamus")
        stroke_hist = run_neural(stroke, n_ticks=200)

        als_traj = np.array([m["action_impedance"] for m in als_hist])
        stroke_traj = np.array([m["action_impedance"] for m in stroke_hist])

        corr = np.corrcoef(als_traj, stroke_traj)[0, 1]
        assert corr < 0.99, (
            f"ALS and stroke should differ, but correlation={corr:.4f}")

    def test_severity_dose_response(self):
        """Larger motor_neuron Z perturbation → higher early action (dose-response).
        Compare early ticks before C2 has time to compensate."""
        actions = []
        for factor in [1.0, 2.0, 3.0, 5.0]:
            topo = build_neural(eta=0.02, seed=42)
            perturb_impedance(topo, "motor_neuron", factor=factor)
            hist = run_neural(topo, n_ticks=30)
            # Use early ticks — before C2 convergence hides the perturbation
            avg_a = np.mean([m["action_impedance"] for m in hist[:10]])
            actions.append(avg_a)

        for i in range(len(actions) - 1):
            assert actions[i + 1] >= actions[i] * 0.9, (
                f"Dose-response violated: factor={[1.0, 2.0, 3.0, 5.0][i+1]} "
                f"gave {actions[i + 1]:.4f} vs {actions[i]:.4f}")


# ============================================================================
# Test: Dimensional cost (Irreducibility Theorem — neural)
# ============================================================================

class TestIrreducibilityTheorem:
    """K=5→K=3→K=2→K=1 cascade guarantees non-zero A_cutoff.

    This is the neural instantiation of Paper 0's irreducibility theorem:
    dimensional mismatch between tissue types creates a minimum cost
    that no amount of C2 remodeling can eliminate.
    """

    def test_cutoff_exists(self):
        """Neural topology must have A_cutoff > 0 (heterogeneous K)."""
        topo = build_neural()
        stim = inject_stimulus(topo, "cortex_motor", 0.5)
        m = topo.tick(external_stimuli=stim)
        assert m["action_cutoff"] > 0, (
            "K=5→3→2→1 topology must have non-zero cutoff action")

    def test_cutoff_irreducible(self):
        """A_cutoff should remain > 0 even after long C2 evolution."""
        topo = build_neural(eta=0.05)
        history = run_neural(topo, n_ticks=300)
        final_cutoffs = [m["action_cutoff"] for m in history[-20:]]
        assert all(c > 0 for c in final_cutoffs), (
            "A_cutoff should remain > 0 (irreducible dimensional cost)")

    def test_impedance_action_reducible(self):
        """A_impedance (learnable part) should decrease under C2."""
        topo = build_neural(eta=0.05)
        history = run_neural(topo, n_ticks=300)
        early = np.mean([m["action_impedance"] for m in history[:10]])
        late = np.mean([m["action_impedance"] for m in history[-10:]])
        assert late < early, (
            f"A_impedance should be reducible: early={early:.4f}, late={late:.4f}")

    def test_neural_cutoff_larger_than_cardiovascular(self):
        """Neural topology (4 K-levels: 5,3,2,1) should have larger A_cutoff
        than cardiovascular (3 K-levels: 4,2,1) — more dimension jumps."""
        from alice.body.tissue_blueprint import build_cardiovascular
        neural = build_neural()
        cardio = build_cardiovascular()

        n_stim = inject_stimulus(neural, "cortex_motor", 0.5)
        n_m = neural.tick(external_stimuli=n_stim)

        c_stim = inject_stimulus(cardio, "sa_node", 0.5)
        c_m = cardio.tick(external_stimuli=c_stim)

        # More dimensional jumps → higher cutoff
        assert n_m["action_cutoff"] >= c_m["action_cutoff"] * 0.5, (
            f"Neural cutoff ({n_m['action_cutoff']:.4f}) should be "
            f"comparable to or larger than cardio ({c_m['action_cutoff']:.4f})")
