# -*- coding: utf-8 -*-
"""Tissue blueprints — wiring organ systems onto GammaTopology.

This module bridges the gap between:
  - alice/core/gamma_topology.py  (physics engine: GammaNode, GammaTopology, C1/C2/C3)
  - alice/body/clinical_*.py      (disease models: currently ad-hoc Z updates)

Each organ system is defined as a tissue blueprint:
  1. A set of GammaNodes with biologically correct TissueTypes
  2. A wiring diagram (which nodes connect to which)
  3. A factory function that returns a ready-to-run GammaTopology

Diseases are NOT coded here. Disease = perturbation of initial conditions
(impedance shift, edge removal, external stimulus pattern) applied to a
healthy blueprint, then evolved via GammaTopology.tick().

Emergence level: E0 — zero disease-specific code in the topology layer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from alice.core.gamma_topology import (
    GammaNode,
    GammaTopology,
    TissueType,
    # Biological presets
    CARDIAC_PURKINJE,
    AUTONOMIC_PREGANGLIONIC,
    MOTOR_ALPHA,
    SENSORY_AB,
    PAIN_AD_FIBER,
    PAIN_C_FIBER,
    CORTICAL_PYRAMIDAL,
    ENTERIC_NEURON,
)


# ============================================================================
# Cardiovascular tissue types (extending core presets)
# ============================================================================

VASCULAR_SMOOTH_MUSCLE = TissueType(
    "vascular_smooth_muscle", n_modes=2, z_mean=60.0, z_std=12.0,
    diameter_um=5.0, myelinated=False,
    description="Vascular smooth muscle — arteriolar tone control")

ENDOTHELIAL = TissueType(
    "endothelial", n_modes=1, z_mean=45.0, z_std=8.0,
    diameter_um=1.0, myelinated=False,
    description="Endothelial lining — capillary exchange interface")


# ============================================================================
# Blueprint: a named set of nodes + wiring rules
# ============================================================================

def _make_node(name: str, tissue: TissueType,
               z_override: Optional[np.ndarray] = None,
               seed: int = 0) -> GammaNode:
    """Create a GammaNode from a TissueType with reproducible impedance."""
    rng = np.random.default_rng(seed)
    if z_override is not None:
        z = np.asarray(z_override, dtype=np.float64)
    else:
        z = np.abs(rng.normal(tissue.z_mean, tissue.z_std, size=tissue.n_modes))
        z = np.clip(z, 1.0, 500.0)
    activation = np.zeros(tissue.n_modes, dtype=np.float64)
    return GammaNode(name=name, impedance=z, activation=activation)


# ============================================================================
# Cardiovascular Blueprint
# ============================================================================
#
# Topology (simplified but physically grounded):
#
#   sa_node (K=4) ──→ av_node (K=4) ──→ lv_myocardium (K=4)
#       │                                     │
#       │                              aortic_valve (K=4)
#       │                                     │
#       └──→ rv_myocardium (K=4)         aorta (K=4)
#                   │                         │
#            pulm_valve (K=4)         ┌───────┴───────┐
#                   │            coronary (K=4)  arterioles (K=2)
#            pulm_artery (K=2)                       │
#                                             capillaries (K=1)
#
# Note: arterioles (K=2) < aorta (K=4) → dimensional cutoff exists.
#       capillaries (K=1) < arterioles (K=2) → further cutoff.
#       This is physically correct: large elastic arteries carry more
#       modes than thin resistance vessels.
#

# Wiring diagram: list of (source, target) pairs
CARDIOVASCULAR_WIRING: List[Tuple[str, str]] = [
    # Conduction system
    ("sa_node", "av_node"),
    ("av_node", "lv_myocardium"),
    ("av_node", "rv_myocardium"),
    # Left heart → systemic
    ("lv_myocardium", "aortic_valve"),
    ("aortic_valve", "aorta"),
    ("aorta", "coronary"),
    ("aorta", "arterioles"),
    ("arterioles", "capillaries"),
    # Right heart → pulmonary
    ("rv_myocardium", "pulm_valve"),
    ("pulm_valve", "pulm_artery"),
    # Return paths (bidirectional physiology)
    ("coronary", "rv_myocardium"),
    ("capillaries", "rv_myocardium"),
    ("pulm_artery", "lv_myocardium"),
]

# Node definitions: (name, TissueType, seed)
CARDIOVASCULAR_NODES: List[Tuple[str, TissueType, int]] = [
    ("sa_node",        CARDIAC_PURKINJE,          100),
    ("av_node",        CARDIAC_PURKINJE,          101),
    ("lv_myocardium",  CARDIAC_PURKINJE,          102),
    ("rv_myocardium",  CARDIAC_PURKINJE,          103),
    ("aortic_valve",   CARDIAC_PURKINJE,          104),
    ("pulm_valve",     CARDIAC_PURKINJE,          105),
    ("aorta",          CARDIAC_PURKINJE,          106),
    ("coronary",       CARDIAC_PURKINJE,          107),
    ("arterioles",     VASCULAR_SMOOTH_MUSCLE,    108),  # K=2 (dimensional step-down)
    ("capillaries",    ENDOTHELIAL,               109),  # K=1 (minimal)
    ("pulm_artery",    AUTONOMIC_PREGANGLIONIC,   110),  # K=2
]


def build_cardiovascular(
    eta: float = 0.01,
    max_dimension_gap: int = 3,
    seed: Optional[int] = 42,
) -> GammaTopology:
    """Build a healthy cardiovascular GammaTopology.

    Returns a topology with 11 nodes and ~14 edges representing
    the major cardiovascular transmission line network.

    Disease induction: perturb the returned topology's impedances
    or edges, then call topology.tick() to let C2 evolve the system.
    """
    nodes = [_make_node(name, tissue, seed=s)
             for name, tissue, s in CARDIOVASCULAR_NODES]

    topo = GammaTopology(
        nodes=nodes,
        eta=eta,
        max_dimension_gap=max_dimension_gap,
        gamma_threshold=0.3,
    )

    # Wire according to blueprint
    for src, tgt in CARDIOVASCULAR_WIRING:
        topo.activate_edge(src, tgt)

    return topo


# ============================================================================
# Neural Blueprint
# ============================================================================
#
# The neural cable model: cortex → spinal cord → periphery
# Dimensional hierarchy: K=5 → K=3 → K=2 → K=1
#
# Topology:
#
#   cortex_motor (K=5) ──→ cortex_sensory (K=5)
#       │                        │
#   thalamus (K=5)          thalamus (relay)
#       │                        │
#   spinal_motor (K=3) ←── spinal_sensory (K=3)
#       │                        ↑
#   motor_neuron (K=3)     sensory_neuron (K=3)
#       │                        ↑
#   nociceptor_ad (K=2)    nociceptor_ad (K=2)
#       │                        ↑
#   nociceptor_c (K=1)     nociceptor_c (K=1)
#
# The K=5→K=3→K=2→K=1 cascade means:
#   - cortex→spinal: 2 modes cut off (dimensional cost)
#   - spinal→nociceptor_ad: 1 mode cut off
#   - nociceptor_ad→nociceptor_c: 1 mode cut off
#   → Paper 0 irreducibility theorem in action
#
# The neural cable degradation spectrum (Γ axis):
#   Γ ≈ 0.0  healthy match     → normal conduction
#   Γ ≈ 0.1  thermal noise ↑   → fever, inflammation
#   Γ ≈ 0.3  reflection loop   → PTSD (signal re-reflects)
#   Γ ≈ 0.7  progressive col.  → ALS (channel degrades)
#   Γ → 1.0  open circuit echo → phantom limb pain
#

NEURAL_NODES: List[Tuple[str, TissueType, int]] = [
    ("cortex_motor",    CORTICAL_PYRAMIDAL,       200),
    ("cortex_sensory",  CORTICAL_PYRAMIDAL,       201),
    ("thalamus",        CORTICAL_PYRAMIDAL,       202),  # relay hub, K=5
    ("spinal_motor",    MOTOR_ALPHA,              203),  # K=3 step-down
    ("spinal_sensory",  SENSORY_AB,               204),  # K=3
    ("motor_neuron",    MOTOR_ALPHA,              205),  # K=3 peripheral
    ("sensory_neuron",  SENSORY_AB,               206),  # K=3
    ("nociceptor_ad",   PAIN_AD_FIBER,            207),  # K=2 thin myelinated
    ("nociceptor_c",    PAIN_C_FIBER,             208),  # K=1 unmyelinated
]

NEURAL_WIRING: List[Tuple[str, str]] = [
    # Descending motor pathway: cortex → spinal → periphery
    ("cortex_motor", "thalamus"),
    ("thalamus", "spinal_motor"),
    ("spinal_motor", "motor_neuron"),
    # Ascending sensory pathway: periphery → spinal → cortex
    ("sensory_neuron", "spinal_sensory"),
    ("spinal_sensory", "thalamus"),
    ("thalamus", "cortex_sensory"),
    # Pain pathway (thin fibers)
    ("nociceptor_c", "nociceptor_ad"),
    ("nociceptor_ad", "spinal_sensory"),
    # Cross-connections (sensorimotor integration)
    ("cortex_motor", "cortex_sensory"),
    ("spinal_motor", "spinal_sensory"),
    # Motor → nociceptor (efferent modulation)
    ("motor_neuron", "nociceptor_ad"),
]


def build_neural(
    eta: float = 0.01,
    max_dimension_gap: int = 2,
    seed: Optional[int] = 42,
) -> GammaTopology:
    """Build a healthy neural pathway GammaTopology.

    Returns a topology with 9 nodes (K=1,2,3,5) and ~12 edges
    representing a cortex-to-periphery sensorimotor + pain pathway.

    The dimensional hierarchy (K=5→3→2→1) means irreducible cutoff
    costs exist at every level transition — this is the neural cable
    degradation model from the Emergence Standard.
    """
    nodes = [_make_node(name, tissue, seed=s)
             for name, tissue, s in NEURAL_NODES]

    topo = GammaTopology(
        nodes=nodes,
        eta=eta,
        max_dimension_gap=max_dimension_gap,
        gamma_threshold=0.3,
    )

    for src, tgt in NEURAL_WIRING:
        topo.activate_edge(src, tgt)

    return topo


# ============================================================================
# Perturbation helpers (E0-compatible: modify initial conditions only)
# ============================================================================

def perturb_impedance(topo: GammaTopology, node_name: str,
                      factor: float) -> None:
    """Scale a node's impedance by a factor (e.g. 1.5 = +50%).

    This represents a pathological initial condition, not a disease model.
    After perturbation, run topology.tick() and observe emergent Γ patterns.
    """
    node = topo.nodes[node_name]
    node.impedance = np.clip(node.impedance * factor, 1.0, 500.0)


def sever_edge(topo: GammaTopology, src: str, tgt: str) -> None:
    """Remove an edge (e.g. coronary occlusion).

    Downstream nodes lose input → their impedance drifts → Γ rises.
    """
    topo.deactivate_edge(src, tgt)


def inject_stimulus(topo: GammaTopology, node_name: str,
                    amplitude: float = 1.0) -> Dict[str, np.ndarray]:
    """Create a stimulus dict for a single node.

    Returns a dict suitable for topology.tick(external_stimuli=...).
    """
    node = topo.nodes[node_name]
    stim = np.full(node.K, amplitude, dtype=np.float64)
    return {node_name: stim}
