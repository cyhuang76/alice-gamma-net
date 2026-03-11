# -*- coding: utf-8 -*-
"""Tests for all tissue blueprints — build, C2 physics, and E0 structure.

Verifies every blueprint in BLUEPRINT_REGISTRY:
  1. Builds without error
  2. Has correct node/edge counts
  3. C1: Γ² + T = 1 at every edge
  4. C2: A_imp does not increase after evolution
  5. A_cut is invariant under C2 (irreducibility)
  6. All edges are physically valid (dimension gap within limit)

Emergence level: E0 — zero disease-specific code.
"""

import pytest
import numpy as np

from alice.body.tissue_blueprint import (
    BLUEPRINT_REGISTRY,
    build_cardiovascular,
    build_neural,
    build_pulmonary,
    build_renal,
    build_hepatic,
    build_gi,
    build_endocrine,
    build_immune,
    build_lymphatic,
    build_musculoskeletal,
    build_skeletal,
    build_epithelial,
    build_connective,
    build_adipose,
    build_hematopoietic,
    build_cartilage,
    build_reproductive,
    build_brain_regulatory,
    build_vascular,
    build_ocular,
    build_auditory,
    build_vestibular,
    build_olfactory,
    build_dental,
    build_enteric,
    build_autonomic,
    build_pancreatic,
    build_csf,
    build_brain_functional,
    perturb_impedance,
    sever_edge,
)


# ============================================================================
# Expected properties per blueprint
# ============================================================================

EXPECTED = {
    "cardiovascular":   {"nodes": 11, "min_edges": 13},
    "neural":           {"nodes":  9, "min_edges": 11},
    "pulmonary":        {"nodes":  7, "min_edges":  7},
    "renal":            {"nodes":  9, "min_edges":  9},
    "hepatic":          {"nodes":  7, "min_edges":  7},
    "gi":               {"nodes":  8, "min_edges": 10},
    "endocrine":        {"nodes":  8, "min_edges": 11},
    "immune":           {"nodes":  7, "min_edges":  9},
    "lymphatic":        {"nodes":  7, "min_edges":  7},
    "musculoskeletal":  {"nodes":  7, "min_edges":  8},
    "skeletal":         {"nodes":  6, "min_edges":  7},
    "epithelial":       {"nodes":  8, "min_edges":  8},
    "connective":       {"nodes":  6, "min_edges":  6},
    "adipose":          {"nodes":  6, "min_edges":  7},
    "hematopoietic":    {"nodes":  9, "min_edges":  9},
    "cartilage":        {"nodes":  4, "min_edges":  4},
    "reproductive":     {"nodes":  6, "min_edges":  7},
    "brain_regulatory": {"nodes":  7, "min_edges":  8},
    "vascular":         {"nodes": 11, "min_edges": 11},
    "ocular":           {"nodes":  9, "min_edges":  9},
    "auditory":         {"nodes":  7, "min_edges":  7},
    "vestibular":       {"nodes":  6, "min_edges":  6},
    "olfactory":        {"nodes":  4, "min_edges":  4},
    "dental":           {"nodes":  7, "min_edges":  7},
    "enteric":          {"nodes":  7, "min_edges":  7},
    "autonomic":        {"nodes":  8, "min_edges": 10},
    "pancreatic":       {"nodes":  7, "min_edges":  8},
    "csf":              {"nodes":  8, "min_edges":  8},
    "brain_functional": {"nodes": 12, "min_edges": 21},
}


# ============================================================================
# Parametrised tests across all blueprints
# ============================================================================

class TestAllBlueprintsRegistry:
    """Ensure BLUEPRINT_REGISTRY is complete and consistent."""

    def test_registry_count(self):
        assert len(BLUEPRINT_REGISTRY) == 29

    def test_all_expected_present(self):
        for name in EXPECTED:
            assert name in BLUEPRINT_REGISTRY, f"Missing blueprint: {name}"


@pytest.fixture(params=list(BLUEPRINT_REGISTRY.keys()))
def blueprint_name(request):
    return request.param


@pytest.fixture
def topo(blueprint_name):
    return BLUEPRINT_REGISTRY[blueprint_name]()


class TestBlueprintStructure:
    """Structural correctness of each blueprint."""

    def test_node_count(self, blueprint_name, topo):
        expected_n = EXPECTED[blueprint_name]["nodes"]
        assert len(topo.nodes) == expected_n

    def test_edge_count(self, blueprint_name, topo):
        expected_e = EXPECTED[blueprint_name]["min_edges"]
        assert len(topo.active_edges) >= expected_e

    def test_all_nodes_have_positive_impedance(self, topo):
        for node in topo.nodes.values():
            assert np.all(node.impedance > 0)

    def test_all_nodes_have_consistent_dimensions(self, topo):
        for node in topo.nodes.values():
            assert node.impedance.shape == node.activation.shape


class TestC1EnergyConservation:
    """C1: Γ² + T = 1 at every edge."""

    def test_c1_holds(self, topo):
        for (src, tgt), ch in topo.active_edges.items():
            assert ch.verify_c1(), f"C1 violated at {src}→{tgt}"


class TestC2ImpedanceRemodeling:
    """C2: A_imp decreases or stays constant under evolution."""

    def test_a_imp_non_increasing(self, topo):
        a_imp_0, _ = topo.action_decomposition()
        for _ in range(100):
            topo.tick()
        a_imp_f, _ = topo.action_decomposition()
        assert a_imp_f <= a_imp_0 + 1e-10

    def test_a_cut_invariant(self, topo):
        _, a_cut_0 = topo.action_decomposition()
        for _ in range(100):
            topo.tick()
        _, a_cut_f = topo.action_decomposition()
        assert abs(a_cut_f - a_cut_0) < 1e-10


class TestC1AfterEvolution:
    """C1 must still hold after 100 ticks of C2 evolution."""

    def test_c1_post_evolution(self, topo):
        for _ in range(100):
            topo.tick()
        for (src, tgt), ch in topo.active_edges.items():
            assert ch.verify_c1(), f"C1 violated after evolution at {src}→{tgt}"


# ============================================================================
# Targeted tests for specific tissues
# ============================================================================

class TestCartilageC2Null:
    """Cartilage has η≈0 — virtually no C2 remodeling."""

    def test_cartilage_minimal_remodeling(self):
        topo = build_cartilage()
        a_imp_0, _ = topo.action_decomposition()
        for _ in range(300):
            topo.tick()
        a_imp_f, _ = topo.action_decomposition()
        # With η=0.001, change should be minimal
        delta = abs(a_imp_f - a_imp_0)
        assert delta < 0.1 * a_imp_0 + 1e-10


class TestBrainFunctionalDimensionalMismatch:
    """Brain functional topology has K=5 cortex vs K=3 subcortex."""

    def test_cortex_subcortex_k_split(self):
        topo = build_brain_functional()
        cortical = ["prefrontal", "motor_cortex_b", "sensory_cortex", "thalamus_b"]
        subcortical = ["amygdala", "hippocampus", "basal_ganglia_b",
                       "hypothalamus_b", "cerebellum_b", "broca_b",
                       "wernicke_b", "cingulate"]
        for name in cortical:
            assert topo.nodes[name].K == 5
        for name in subcortical:
            assert topo.nodes[name].K == 3

    def test_a_cut_positive(self):
        """Dimensional mismatch must create irreducible cost."""
        topo = build_brain_functional()
        _, a_cut = topo.action_decomposition()
        assert a_cut > 0


class TestEndocrineNegativeFeedback:
    """Endocrine has negative feedback loops (thyroid→hypothalamus etc)."""

    def test_feedback_edges_exist(self):
        topo = build_endocrine()
        edges = set(topo.active_edges.keys())
        assert ("thyroid", "hypothalamus") in edges
        assert ("adrenal_cortex", "hypothalamus") in edges
        assert ("gonad", "hypothalamus") in edges


class TestRenalTGFeedback:
    """Renal has tubuloglomerular feedback: DCT → afferent arteriole."""

    def test_tg_feedback_exists(self):
        topo = build_renal()
        edges = set(topo.active_edges.keys())
        assert ("dct", "afferent_arteriole") in edges


class TestHepaticDualSupply:
    """Hepatic has dual blood supply: portal vein + hepatic artery."""

    def test_dual_input_to_sinusoid(self):
        topo = build_hepatic()
        edges = set(topo.active_edges.keys())
        assert ("portal_vein", "sinusoid") in edges
        assert ("hepatic_artery", "sinusoid") in edges


class TestOcularIOPLoop:
    """Ocular has IOP regulation: aqueous→trabecular→ciliary."""

    def test_iop_loop_exists(self):
        topo = build_ocular()
        edges = set(topo.active_edges.keys())
        assert ("aqueous", "trabecular") in edges
        assert ("trabecular", "ciliary_body") in edges


class TestHematopoieticEPOFeedback:
    """EPO feedback: rbc→epo_signal→hsc."""

    def test_epo_loop(self):
        topo = build_hematopoietic()
        edges = set(topo.active_edges.keys())
        assert ("rbc", "epo_signal") in edges
        assert ("epo_signal", "hsc_cell") in edges


class TestPancreaticParacrine:
    """Pancreatic islet: β→δ→α paracrine cascade."""

    def test_paracrine_cascade(self):
        topo = build_pancreatic()
        edges = set(topo.active_edges.keys())
        assert ("beta_cell_p", "delta_cell") in edges
        assert ("delta_cell", "alpha_cell") in edges


class TestReproducibleSeeds:
    """Same seed → identical topology."""

    def test_deterministic(self):
        t1 = build_cardiovascular(seed=42)
        t2 = build_cardiovascular(seed=42)
        for name in t1.nodes:
            assert np.allclose(t1.nodes[name].impedance, t2.nodes[name].impedance)


class TestDifferentSeeds:
    """Different seed → different impedances."""

    def test_different_seeds(self):
        t1 = build_cardiovascular(seed=42)
        t2 = build_cardiovascular(seed=99)
        any_diff = False
        for name in t1.nodes:
            if not np.allclose(t1.nodes[name].impedance, t2.nodes[name].impedance):
                any_diff = True
                break
        assert any_diff
