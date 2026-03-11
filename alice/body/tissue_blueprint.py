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
    # Non-neural presets
    ALVEOLAR_EPITHELIUM,
    AIRWAY_SMOOTH_MUSCLE,
    GLOMERULAR,
    TUBULAR_EPITHELIUM,
    HEPATOCYTE,
    SINUSOIDAL,
    GI_MUCOSAL,
    GI_SMOOTH_MUSCLE,
    ENDOCRINE_GLAND,
    HYPOTHALAMIC,
    LYMPHOCYTE,
    MAST_INNATE,
    LYMPHATIC_VESSEL,
    SKELETAL_MUSCLE,
    TENDON,
    BONE_CELL,
    KERATINOCYTE,
    DERMAL_FIBROBLAST,
    FIBROBLAST_CT,
    ADIPOCYTE,
    HSC,
    ERYTHROID,
    CHONDROCYTE,
    GONADAL,
    RETINAL,
    LENS,
    COCHLEAR_HAIR,
    VESTIBULAR_CELL,
    OLFACTORY_RECEPTOR,
    DENTAL,
    BETA_CELL,
    EPENDYMAL,
    BAROREFLEX,
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
    offset = seed if seed is not None else 0
    nodes = [_make_node(name, tissue, seed=s + offset)
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


# ============================================================================
# Pulmonary Blueprint
# ============================================================================
#
#   trachea (K=2) → main_bronchus (K=2) → bronchiole (K=1)
#       │                                       │
#   mucociliary (K=1)                    alveolus (K=2)
#                                               │
#                                      pulm_capillary (K=2)
#                                               │
#                                      pulm_vein (K=2)
#
# Dimensional step-down: bronchiole (K=1) < bronchus (K=2)
# represents the airway narrowing physics.

PULMONARY_NODES: List[Tuple[str, TissueType, int]] = [
    ("trachea",         AIRWAY_SMOOTH_MUSCLE,    300),   # K=1
    ("main_bronchus",   AIRWAY_SMOOTH_MUSCLE,    301),   # K=1
    ("bronchiole",      AIRWAY_SMOOTH_MUSCLE,    302),   # K=1
    ("alveolus",        ALVEOLAR_EPITHELIUM,     303),   # K=2
    ("pulm_capillary",  ALVEOLAR_EPITHELIUM,     304),   # K=2
    ("pulm_vein",       AUTONOMIC_PREGANGLIONIC, 305),   # K=2
    ("mucociliary",     AIRWAY_SMOOTH_MUSCLE,    306),   # K=1
]

PULMONARY_WIRING: List[Tuple[str, str]] = [
    ("trachea", "main_bronchus"),
    ("main_bronchus", "bronchiole"),
    ("bronchiole", "alveolus"),
    ("alveolus", "pulm_capillary"),
    ("pulm_capillary", "pulm_vein"),
    ("trachea", "mucociliary"),
    # Gas exchange feedback
    ("pulm_capillary", "alveolus"),
]


def build_pulmonary(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy pulmonary GammaTopology (7 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in PULMONARY_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=2, gamma_threshold=0.3)
    for src, tgt in PULMONARY_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Renal Blueprint
# ============================================================================
#
#   afferent_arteriole (K=3) → glomerulus (K=3) → efferent_arteriole (K=3)
#                                   │
#                              bowman (K=2) → PCT (K=2)
#                                               │
#                              collecting_duct (K=1) ← DCT (K=2) ← loop_henle (K=2)
#                                   │
#                              renal_vein (K=2)

RENAL_NODES: List[Tuple[str, TissueType, int]] = [
    ("afferent_arteriole", GLOMERULAR,         400),   # K=3
    ("glomerulus",         GLOMERULAR,         401),   # K=3
    ("efferent_arteriole", GLOMERULAR,         402),   # K=3
    ("bowman",             TUBULAR_EPITHELIUM, 403),   # K=2
    ("pct",                TUBULAR_EPITHELIUM, 404),   # K=2 proximal convoluted tubule
    ("loop_henle",         TUBULAR_EPITHELIUM, 405),   # K=2
    ("dct",                TUBULAR_EPITHELIUM, 406),   # K=2 distal convoluted tubule
    ("collecting_duct",    TUBULAR_EPITHELIUM, 407),   # K=2
    ("renal_vein",         AUTONOMIC_PREGANGLIONIC, 408), # K=2
]

RENAL_WIRING: List[Tuple[str, str]] = [
    ("afferent_arteriole", "glomerulus"),
    ("glomerulus", "efferent_arteriole"),
    ("glomerulus", "bowman"),
    ("bowman", "pct"),
    ("pct", "loop_henle"),
    ("loop_henle", "dct"),
    ("dct", "collecting_duct"),
    ("collecting_duct", "renal_vein"),
    # Tubuloglomerular feedback
    ("dct", "afferent_arteriole"),
]


def build_renal(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy renal GammaTopology (9 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in RENAL_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=2, gamma_threshold=0.3)
    for src, tgt in RENAL_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Hepatic Blueprint
# ============================================================================
#
#   portal_vein (K=3) → sinusoid (K=1) → hepatocyte (K=3) → central_vein (K=1)
#                                ↑                                │
#   hepatic_artery (K=3) ───────┘                          hepatic_vein (K=1)
#                                                                 │
#   bile_duct (K=1) ←── hepatocyte                          ivc (K=1)
#
# Dual blood supply (portal + arterial) is physically critical.

HEPATIC_NODES: List[Tuple[str, TissueType, int]] = [
    ("portal_vein",    HEPATOCYTE,  500),   # K=3
    ("hepatic_artery", HEPATOCYTE,  501),   # K=3
    ("sinusoid",       SINUSOIDAL,  502),   # K=1  (fenestrated)
    ("hepatocyte",     HEPATOCYTE,  503),   # K=3
    ("central_vein",   SINUSOIDAL,  504),   # K=1
    ("hepatic_vein",   SINUSOIDAL,  505),   # K=1
    ("bile_duct",      SINUSOIDAL,  506),   # K=1
]

HEPATIC_WIRING: List[Tuple[str, str]] = [
    ("portal_vein", "sinusoid"),
    ("hepatic_artery", "sinusoid"),
    ("sinusoid", "hepatocyte"),
    ("hepatocyte", "central_vein"),
    ("central_vein", "hepatic_vein"),
    ("hepatocyte", "bile_duct"),
    # Regeneration feedback
    ("hepatocyte", "sinusoid"),
]


def build_hepatic(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy hepatic GammaTopology (7 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in HEPATIC_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=3, gamma_threshold=0.3)
    for src, tgt in HEPATIC_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Gastrointestinal Blueprint
# ============================================================================
#
#   esophagus (K=1) → stomach (K=2) → duodenum (K=2) → jejunum (K=2)
#                                                           │
#   colon (K=1) ← ileum (K=2) ←────────────────────────────┘
#       │
#   rectum (K=1)
#
#   + enteric_plexus (K=1) modulates all smooth muscle layers

GI_NODES: List[Tuple[str, TissueType, int]] = [
    ("esophagus",       GI_SMOOTH_MUSCLE, 600),   # K=1
    ("stomach",         GI_MUCOSAL,       601),   # K=2
    ("duodenum",        GI_MUCOSAL,       602),   # K=2
    ("jejunum",         GI_MUCOSAL,       603),   # K=2
    ("ileum",           GI_MUCOSAL,       604),   # K=2
    ("colon",           GI_SMOOTH_MUSCLE, 605),   # K=1
    ("rectum",          GI_SMOOTH_MUSCLE, 606),   # K=1
    ("enteric_plexus",  ENTERIC_NEURON,   607),   # K=1
]

GI_WIRING: List[Tuple[str, str]] = [
    ("esophagus", "stomach"),
    ("stomach", "duodenum"),
    ("duodenum", "jejunum"),
    ("jejunum", "ileum"),
    ("ileum", "colon"),
    ("colon", "rectum"),
    # Enteric nervous system modulation
    ("enteric_plexus", "stomach"),
    ("enteric_plexus", "duodenum"),
    ("enteric_plexus", "colon"),
    # Feedback: mucosal sensing → enteric plexus
    ("duodenum", "enteric_plexus"),
]


def build_gi(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy gastrointestinal GammaTopology (8 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in GI_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=2, gamma_threshold=0.3)
    for src, tgt in GI_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Endocrine Blueprint (HPT + HPA + HPG axes)
# ============================================================================
#
#   hypothalamus (K=3) → ant_pituitary (K=2) → thyroid (K=2)
#                 │               │                 │
#                 │               └→ adrenal (K=2)  └→ target_tissue (K=1)
#                 │               │
#                 └→ post_pituitary (K=2) → kidney_adr (K=1)
#                                 │
#                                 └→ gonad (K=2) → target_tissue
#
# Negative feedback loops are the core physics.

ENDOCRINE_NODES: List[Tuple[str, TissueType, int]] = [
    ("hypothalamus",     HYPOTHALAMIC,     700),   # K=3
    ("ant_pituitary",    ENDOCRINE_GLAND,  701),   # K=2
    ("post_pituitary",   ENDOCRINE_GLAND,  702),   # K=2
    ("thyroid",          ENDOCRINE_GLAND,  703),   # K=2
    ("adrenal_cortex",   ENDOCRINE_GLAND,  704),   # K=2
    ("adrenal_medulla",  ENDOCRINE_GLAND,  705),   # K=2
    ("gonad",            GONADAL,          706),   # K=2
    ("target_tissue",    ENDOCRINE_GLAND,  707),   # K=2
]

ENDOCRINE_WIRING: List[Tuple[str, str]] = [
    # HPT axis
    ("hypothalamus", "ant_pituitary"),
    ("ant_pituitary", "thyroid"),
    ("thyroid", "target_tissue"),
    # HPA axis
    ("ant_pituitary", "adrenal_cortex"),
    ("hypothalamus", "adrenal_medulla"),
    # HPG axis
    ("ant_pituitary", "gonad"),
    # Posterior pituitary (ADH, oxytocin)
    ("hypothalamus", "post_pituitary"),
    # Negative feedback (critical for homeostasis)
    ("thyroid", "hypothalamus"),
    ("adrenal_cortex", "hypothalamus"),
    ("gonad", "hypothalamus"),
    ("target_tissue", "hypothalamus"),
]


def build_endocrine(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy endocrine GammaTopology (8 nodes, 3 axes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in ENDOCRINE_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=2, gamma_threshold=0.3)
    for src, tgt in ENDOCRINE_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Immune Blueprint
# ============================================================================
#
#   bone_marrow (K=3) → thymus (K=2) → lymph_node (K=2)
#         │                                  │
#         └→ spleen (K=2) ←─────────────────┘
#                │                           │
#   effector_site (K=1) ← ── ── ── ── ── ── ┘
#         │
#   inflammatory_site (K=1)
#
# bone_marrow (K=3) also → B-cells → lymph_node

IMMUNE_NODES: List[Tuple[str, TissueType, int]] = [
    ("bone_marrow",        HSC,         800),   # K=3 (stem cells)
    ("thymus",             LYMPHOCYTE,  801),   # K=2
    ("lymph_node",         LYMPHOCYTE,  802),   # K=2
    ("spleen",             LYMPHOCYTE,  803),   # K=2
    ("effector_site",      MAST_INNATE, 804),   # K=1
    ("inflammatory_site",  MAST_INNATE, 805),   # K=1
    ("mucosal_immune",     MAST_INNATE, 806),   # K=1  (MALT/GALT)
]

IMMUNE_WIRING: List[Tuple[str, str]] = [
    # Lymphocyte maturation path
    ("bone_marrow", "thymus"),
    ("thymus", "lymph_node"),
    ("bone_marrow", "lymph_node"),   # B-cell direct
    ("lymph_node", "spleen"),
    ("lymph_node", "effector_site"),
    ("effector_site", "inflammatory_site"),
    # Mucosal immunity
    ("bone_marrow", "mucosal_immune"),
    ("lymph_node", "mucosal_immune"),
    # Feedback: antigen presentation → lymph node
    ("inflammatory_site", "lymph_node"),
]


def build_immune(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy immune GammaTopology (7 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in IMMUNE_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=3, gamma_threshold=0.3)
    for src, tgt in IMMUNE_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Lymphatic Blueprint
# ============================================================================
#
#   tissue_fluid (K=1) → initial_lymphatic (K=1) → collecting_vessel (K=1)
#                                                          │
#   thoracic_duct (K=1) ← efferent_lymph (K=1) ← lymph_node_lym (K=2)
#         │
#   venous_return (K=1)

LYMPHATIC_NODES: List[Tuple[str, TissueType, int]] = [
    ("tissue_fluid",       LYMPHATIC_VESSEL, 900),    # K=1
    ("initial_lymphatic",  LYMPHATIC_VESSEL, 901),    # K=1
    ("collecting_vessel",  LYMPHATIC_VESSEL, 902),    # K=1
    ("lymph_node_lym",     LYMPHOCYTE,       903),    # K=2 (filtering)
    ("efferent_lymph",     LYMPHATIC_VESSEL, 904),    # K=1
    ("thoracic_duct",      LYMPHATIC_VESSEL, 905),    # K=1
    ("venous_return",      LYMPHATIC_VESSEL, 906),    # K=1
]

LYMPHATIC_WIRING: List[Tuple[str, str]] = [
    ("tissue_fluid", "initial_lymphatic"),
    ("initial_lymphatic", "collecting_vessel"),
    ("collecting_vessel", "lymph_node_lym"),
    ("lymph_node_lym", "efferent_lymph"),
    ("efferent_lymph", "thoracic_duct"),
    ("thoracic_duct", "venous_return"),
    # Edema feedback: high tissue_fluid → increased drainage
    ("tissue_fluid", "collecting_vessel"),
]


def build_lymphatic(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy lymphatic GammaTopology (7 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in LYMPHATIC_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=2, gamma_threshold=0.3)
    for src, tgt in LYMPHATIC_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Musculoskeletal Blueprint (Muscular)
# ============================================================================
#
#   motor_cortex_m (K=5) → spinal_mn (K=3) → nmj (K=2)
#                                                  │
#   golgi_tendon (K=1) ← tendon_m (K=1) ← muscle_fiber (K=3)
#         │                                        ↑
#   spinal_mn ← ─ ─ ─ ─ (stretch reflex) ← muscle_spindle (K=2)

MUSCULOSKELETAL_NODES: List[Tuple[str, TissueType, int]] = [
    ("motor_cortex_m",  CORTICAL_PYRAMIDAL, 1000),  # K=5
    ("spinal_mn",       MOTOR_ALPHA,        1001),  # K=3
    ("nmj",             AUTONOMIC_PREGANGLIONIC, 1002),  # K=2 (neuromuscular junction)
    ("muscle_fiber",    SKELETAL_MUSCLE,    1003),  # K=3
    ("muscle_spindle",  SENSORY_AB,         1004),  # K=3 (proprioception)
    ("tendon_m",        TENDON,             1005),  # K=1
    ("golgi_tendon",    TENDON,             1006),  # K=1 (force sensor)
]

MUSCULOSKELETAL_WIRING: List[Tuple[str, str]] = [
    ("motor_cortex_m", "spinal_mn"),
    ("spinal_mn", "nmj"),
    ("nmj", "muscle_fiber"),
    ("muscle_fiber", "tendon_m"),
    ("tendon_m", "golgi_tendon"),
    # Stretch reflex (proprioceptive feedback)
    ("muscle_spindle", "spinal_mn"),
    ("muscle_fiber", "muscle_spindle"),
    # Golgi tendon organ → inhibitory feedback
    ("golgi_tendon", "spinal_mn"),
]


def build_musculoskeletal(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy musculoskeletal GammaTopology (7 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in MUSCULOSKELETAL_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=3, gamma_threshold=0.3)
    for src, tgt in MUSCULOSKELETAL_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Skeletal (Bone) Blueprint
# ============================================================================
#
#   mechanical_load (K=2) → osteocyte (K=2) → osteoblast (K=2)
#                                │                  │
#                        osteoclast (K=2) ←─────────┘
#                                │
#                         bone_matrix (K=1)
#                                │
#                        periosteum (K=1)
#
# Wolff's law: bone remodels along lines of mechanical stress.
# C2 naturally models this: load → Γ → remodeling.

SKELETAL_NODES: List[Tuple[str, TissueType, int]] = [
    ("mechanical_load",  BONE_CELL, 1100),   # K=2
    ("osteocyte",        BONE_CELL, 1101),   # K=2 (mechanosensor)
    ("osteoblast",       BONE_CELL, 1102),   # K=2 (bone formation)
    ("osteoclast",       BONE_CELL, 1103),   # K=2 (bone resorption)
    ("bone_matrix",      TENDON,    1104),   # K=1 (mineralized)
    ("periosteum",       TENDON,    1105),   # K=1
]

SKELETAL_WIRING: List[Tuple[str, str]] = [
    ("mechanical_load", "osteocyte"),
    ("osteocyte", "osteoblast"),
    ("osteocyte", "osteoclast"),
    ("osteoblast", "bone_matrix"),
    ("osteoclast", "bone_matrix"),
    ("bone_matrix", "periosteum"),
    # Feedback: matrix strain → osteocyte
    ("bone_matrix", "osteocyte"),
]


def build_skeletal(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy skeletal (bone) GammaTopology (6 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in SKELETAL_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=2, gamma_threshold=0.3)
    for src, tgt in SKELETAL_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Epithelial (Skin/Dermal) Blueprint
# ============================================================================
#
#   basal_layer (K=2) → spinosum (K=2) → granulosum (K=2) → corneum (K=1)
#        ↑                                                        │
#   dermis (K=1) ← melanocyte (K=1)                          surface (K=1)
#        │
#   subcutaneous (K=1)

EPITHELIAL_NODES: List[Tuple[str, TissueType, int]] = [
    ("basal_layer",   KERATINOCYTE,     1200),   # K=2
    ("spinosum",      KERATINOCYTE,     1201),   # K=2
    ("granulosum",    KERATINOCYTE,     1202),   # K=2
    ("corneum",       DERMAL_FIBROBLAST, 1203),  # K=1 (dead cells, barrier)
    ("surface",       DERMAL_FIBROBLAST, 1204),  # K=1
    ("melanocyte",    DERMAL_FIBROBLAST, 1205),  # K=1
    ("dermis",        DERMAL_FIBROBLAST, 1206),  # K=1
    ("subcutaneous",  DERMAL_FIBROBLAST, 1207),  # K=1
]

EPITHELIAL_WIRING: List[Tuple[str, str]] = [
    ("basal_layer", "spinosum"),
    ("spinosum", "granulosum"),
    ("granulosum", "corneum"),
    ("corneum", "surface"),
    ("melanocyte", "basal_layer"),
    ("dermis", "basal_layer"),
    ("subcutaneous", "dermis"),
    # Wound healing feedback
    ("surface", "basal_layer"),
]


def build_epithelial(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy epithelial/skin GammaTopology (8 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in EPITHELIAL_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=2, gamma_threshold=0.3)
    for src, tgt in EPITHELIAL_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Connective Tissue Blueprint
# ============================================================================
#
#   fibroblast_c (K=1) → collagen_matrix (K=1) → ecm (K=1)
#        ↑                       │                    │
#   growth_factor (K=1)    ground_substance (K=1)  tensile_load (K=1)
#
# Low-K system — connective tissue is mechanically simple but
# critical for structural integrity.

CONNECTIVE_NODES: List[Tuple[str, TissueType, int]] = [
    ("fibroblast_c",      FIBROBLAST_CT, 1300),  # K=1
    ("collagen_matrix",   FIBROBLAST_CT, 1301),  # K=1
    ("ecm",               FIBROBLAST_CT, 1302),  # K=1
    ("ground_substance",  FIBROBLAST_CT, 1303),  # K=1
    ("growth_factor",     FIBROBLAST_CT, 1304),  # K=1
    ("tensile_load",      FIBROBLAST_CT, 1305),  # K=1
]

CONNECTIVE_WIRING: List[Tuple[str, str]] = [
    ("fibroblast_c", "collagen_matrix"),
    ("collagen_matrix", "ecm"),
    ("collagen_matrix", "ground_substance"),
    ("ecm", "tensile_load"),
    ("growth_factor", "fibroblast_c"),
    # Mechano-feedback: load → fibroblast activation
    ("tensile_load", "fibroblast_c"),
]


def build_connective(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy connective tissue GammaTopology (6 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in CONNECTIVE_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=1, gamma_threshold=0.3)
    for src, tgt in CONNECTIVE_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Adipose Blueprint
# ============================================================================
#
#   pre_adipocyte (K=1) → adipocyte_a (K=1) → leptin_signal (K=1)
#        ↑                       │                   │
#   insulin_signal (K=1)  lipid_droplet (K=1)  hypothalamus_a (K=1)
#
# Simple K=1 system. Leptin → hypothalamus feedback is the key loop.

ADIPOSE_NODES: List[Tuple[str, TissueType, int]] = [
    ("pre_adipocyte",   ADIPOCYTE, 1400),   # K=1
    ("adipocyte_a",     ADIPOCYTE, 1401),   # K=1
    ("lipid_droplet",   ADIPOCYTE, 1402),   # K=1
    ("leptin_signal",   ADIPOCYTE, 1403),   # K=1
    ("insulin_signal",  ADIPOCYTE, 1404),   # K=1
    ("hypothalamus_a",  ADIPOCYTE, 1405),   # K=1
]

ADIPOSE_WIRING: List[Tuple[str, str]] = [
    ("pre_adipocyte", "adipocyte_a"),
    ("adipocyte_a", "lipid_droplet"),
    ("adipocyte_a", "leptin_signal"),
    ("insulin_signal", "pre_adipocyte"),
    ("insulin_signal", "adipocyte_a"),
    ("leptin_signal", "hypothalamus_a"),
    # Negative feedback
    ("hypothalamus_a", "adipocyte_a"),
]


def build_adipose(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy adipose GammaTopology (6 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in ADIPOSE_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=1, gamma_threshold=0.3)
    for src, tgt in ADIPOSE_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Hematopoietic Blueprint
# ============================================================================
#
#   hsc_cell (K=3) → myeloid_progenitor (K=2) → granulocyte (K=1)
#        │                    │
#        │             erythroid_prog (K=1) → rbc (K=1)
#        │
#        └→ lymphoid_progenitor (K=2) → lymphocyte_h (K=1)
#                                              │
#        epo_signal (K=1) → hsc_cell     platelet (K=1)
#
# K=3→2→1 cascade models the commitment and differentiation.

HEMATOPOIETIC_NODES: List[Tuple[str, TissueType, int]] = [
    ("hsc_cell",             HSC,       1500),   # K=3 (stem cell)
    ("myeloid_progenitor",   LYMPHOCYTE, 1501),  # K=2
    ("lymphoid_progenitor",  LYMPHOCYTE, 1502),  # K=2
    ("erythroid_prog",       ERYTHROID, 1503),   # K=1
    ("granulocyte",          ERYTHROID, 1504),   # K=1
    ("rbc",                  ERYTHROID, 1505),   # K=1
    ("lymphocyte_h",         ERYTHROID, 1506),   # K=1
    ("platelet",             ERYTHROID, 1507),   # K=1
    ("epo_signal",           ERYTHROID, 1508),   # K=1
]

HEMATOPOIETIC_WIRING: List[Tuple[str, str]] = [
    ("hsc_cell", "myeloid_progenitor"),
    ("hsc_cell", "lymphoid_progenitor"),
    ("myeloid_progenitor", "erythroid_prog"),
    ("myeloid_progenitor", "granulocyte"),
    ("erythroid_prog", "rbc"),
    ("lymphoid_progenitor", "lymphocyte_h"),
    ("myeloid_progenitor", "platelet"),
    # EPO feedback
    ("epo_signal", "hsc_cell"),
    ("rbc", "epo_signal"),
]


def build_hematopoietic(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy hematopoietic GammaTopology (9 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in HEMATOPOIETIC_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=3, gamma_threshold=0.3)
    for src, tgt in HEMATOPOIETIC_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Cartilage Blueprint (C2-null: η ≈ 0)
# ============================================================================
#
#   perichondrium (K=1) → chondrocyte_c (K=1) → cartilage_matrix (K=1)
#                                │
#        mechanical_load_c (K=1) ┘
#
# Special case: virtually no C2 remodeling (avascular, low turnover).
# η is set very small. This is the physical basis of OA irreversibility.

CARTILAGE_NODES: List[Tuple[str, TissueType, int]] = [
    ("perichondrium",       CHONDROCYTE, 1600),  # K=1
    ("chondrocyte_c",       CHONDROCYTE, 1601),  # K=1
    ("cartilage_matrix",    CHONDROCYTE, 1602),  # K=1
    ("mechanical_load_c",   CHONDROCYTE, 1603),  # K=1
]

CARTILAGE_WIRING: List[Tuple[str, str]] = [
    ("perichondrium", "chondrocyte_c"),
    ("chondrocyte_c", "cartilage_matrix"),
    ("mechanical_load_c", "chondrocyte_c"),
    ("cartilage_matrix", "mechanical_load_c"),
]


def build_cartilage(eta: float = 0.001, seed: Optional[int] = 42) -> GammaTopology:
    """Build a cartilage GammaTopology (4 nodes, η≈0 C2-null)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in CARTILAGE_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=1, gamma_threshold=0.3)
    for src, tgt in CARTILAGE_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Reproductive Blueprint
# ============================================================================
#
#   hypothalamus_r (K=3) → pituitary_r (K=2) → gonad_r (K=2) → gamete (K=1)
#         ↑                       │
#         └── feedback ──── gonad_r
#                                 │
#         uterus (K=2) ←─────────┘   (female path)
#                │
#         placenta_r (K=1)  (gestational)

REPRODUCTIVE_NODES: List[Tuple[str, TissueType, int]] = [
    ("hypothalamus_r", HYPOTHALAMIC,    1700),    # K=3
    ("pituitary_r",    ENDOCRINE_GLAND, 1701),    # K=2
    ("gonad_r",        GONADAL,         1702),    # K=2
    ("gamete",         GONADAL,         1703),    # K=2
    ("uterus",         GONADAL,         1704),    # K=2
    ("placenta_r",     ENDOCRINE_GLAND, 1705),    # K=2
]

REPRODUCTIVE_WIRING: List[Tuple[str, str]] = [
    ("hypothalamus_r", "pituitary_r"),
    ("pituitary_r", "gonad_r"),
    ("gonad_r", "gamete"),
    ("gonad_r", "uterus"),
    ("uterus", "placenta_r"),
    # HPG negative feedback
    ("gonad_r", "hypothalamus_r"),
    ("placenta_r", "hypothalamus_r"),
]


def build_reproductive(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy reproductive GammaTopology (6 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in REPRODUCTIVE_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=2, gamma_threshold=0.3)
    for src, tgt in REPRODUCTIVE_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Brain (Regulatory / Baroreflex) Blueprint — 19th tissue
# ============================================================================
#
#   carotid_baroreceptor (K=3) → nts (K=3) → vagal_motor (K=2)
#                                  │                │
#   aortic_baroreceptor (K=3) → nts         sympathetic_out (K=2)
#                                  │                │
#                          rvlm (K=2) → sympathetic_out
#                                  │
#                          cvlm (K=2) → vagal_motor
#
# NTS = nucleus tractus solitarius; RVLM/CVLM = rostral/caudal
# ventrolateral medulla. This is the brainstem cardiovascular reflex arc.

BRAIN_REG_NODES: List[Tuple[str, TissueType, int]] = [
    ("carotid_baroreceptor", BAROREFLEX,  1800),   # K=3
    ("aortic_baroreceptor",  BAROREFLEX,  1801),   # K=3
    ("nts",                  BAROREFLEX,  1802),   # K=3
    ("rvlm",                 AUTONOMIC_PREGANGLIONIC, 1803),  # K=2
    ("cvlm",                 AUTONOMIC_PREGANGLIONIC, 1804),  # K=2
    ("vagal_motor",          AUTONOMIC_PREGANGLIONIC, 1805),  # K=2
    ("sympathetic_out",      AUTONOMIC_PREGANGLIONIC, 1806),  # K=2
]

BRAIN_REG_WIRING: List[Tuple[str, str]] = [
    ("carotid_baroreceptor", "nts"),
    ("aortic_baroreceptor", "nts"),
    ("nts", "rvlm"),
    ("nts", "cvlm"),
    ("rvlm", "sympathetic_out"),
    ("cvlm", "vagal_motor"),
    ("nts", "vagal_motor"),
    # Feedback: sympathetic tone → baroreceptor sensitivity
    ("sympathetic_out", "carotid_baroreceptor"),
]


def build_brain_regulatory(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a brainstem baroreflex GammaTopology (7 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in BRAIN_REG_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=2, gamma_threshold=0.3)
    for src, tgt in BRAIN_REG_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Vascular (standalone arterial tree) Blueprint
# ============================================================================
#
#   aortic_root (K=4) → asc_aorta (K=4) → aortic_arch (K=4)
#                                               │
#           desc_aorta (K=4) ←──────────────────┘
#                │
#        renal_artery (K=2) → renal_bed (K=2)
#                │
#        iliac (K=2) → femoral (K=2) → arteriole_v (K=1)
#                                           │
#                                      capillary_v (K=1) → venule (K=1)

VASCULAR_NODES: List[Tuple[str, TissueType, int]] = [
    ("aortic_root",   CARDIAC_PURKINJE,          1900),  # K=4
    ("asc_aorta",     CARDIAC_PURKINJE,          1901),  # K=4
    ("aortic_arch",   CARDIAC_PURKINJE,          1902),  # K=4
    ("desc_aorta",    CARDIAC_PURKINJE,          1903),  # K=4
    ("renal_artery",  VASCULAR_SMOOTH_MUSCLE,    1904),  # K=2
    ("renal_bed",     VASCULAR_SMOOTH_MUSCLE,    1905),  # K=2
    ("iliac",         VASCULAR_SMOOTH_MUSCLE,    1906),  # K=2
    ("femoral",       VASCULAR_SMOOTH_MUSCLE,    1907),  # K=2
    ("arteriole_v",   ENDOTHELIAL,               1908),  # K=1
    ("capillary_v",   ENDOTHELIAL,               1909),  # K=1
    ("venule",        ENDOTHELIAL,               1910),  # K=1
]

VASCULAR_WIRING: List[Tuple[str, str]] = [
    ("aortic_root", "asc_aorta"),
    ("asc_aorta", "aortic_arch"),
    ("aortic_arch", "desc_aorta"),
    ("desc_aorta", "renal_artery"),
    ("renal_artery", "renal_bed"),
    ("desc_aorta", "iliac"),
    ("iliac", "femoral"),
    ("femoral", "arteriole_v"),
    ("arteriole_v", "capillary_v"),
    ("capillary_v", "venule"),
    # Venous return → aortic root
    ("venule", "aortic_root"),
]


def build_vascular(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a standalone vascular tree GammaTopology (11 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in VASCULAR_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=3, gamma_threshold=0.3)
    for src, tgt in VASCULAR_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
#                         TIER 2: NEW TISSUE SYSTEMS
# ============================================================================


# ============================================================================
# Ocular Blueprint
# ============================================================================
#
#   cornea (K=1) → lens_o (K=1) → vitreous (K=1) → retina (K=3)
#                                                       │
#                                              optic_nerve (K=3)
#                                                       │
#                                              visual_cortex (K=5)
#       ciliary_body (K=2) → lens_o  (accommodation)
#       aqueous (K=1) → trabecular (K=1) (IOP regulation)

OCULAR_NODES: List[Tuple[str, TissueType, int]] = [
    ("cornea",          LENS,               2000),  # K=1 (avascular)
    ("aqueous",         LENS,               2001),  # K=1
    ("ciliary_body",    ENDOCRINE_GLAND,    2002),  # K=2
    ("lens_o",          LENS,               2003),  # K=1
    ("vitreous",        LENS,               2004),  # K=1
    ("retina",          RETINAL,            2005),  # K=3
    ("optic_nerve",     RETINAL,            2006),  # K=3
    ("trabecular",      LENS,               2007),  # K=1
    ("visual_cortex_o", CORTICAL_PYRAMIDAL, 2008),  # K=5
]

OCULAR_WIRING: List[Tuple[str, str]] = [
    ("cornea", "aqueous"),
    ("aqueous", "lens_o"),
    ("ciliary_body", "lens_o"),
    ("lens_o", "vitreous"),
    ("vitreous", "retina"),
    ("retina", "optic_nerve"),
    ("optic_nerve", "visual_cortex_o"),
    ("aqueous", "trabecular"),
    # IOP feedback
    ("trabecular", "ciliary_body"),
]


def build_ocular(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy ocular GammaTopology (9 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in OCULAR_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=3, gamma_threshold=0.3)
    for src, tgt in OCULAR_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Auditory Blueprint
# ============================================================================
#
#   ear_canal (K=1) → tympanic (K=1) → ossicles (K=2) → oval_window (K=2)
#                                                              │
#   auditory_cortex_a (K=5) ← auditory_nerve (K=3) ← cochlea_a (K=2)

AUDITORY_NODES: List[Tuple[str, TissueType, int]] = [
    ("ear_canal",         AIRWAY_SMOOTH_MUSCLE, 2100),  # K=1
    ("tympanic",          TENDON,               2101),  # K=1 (membrane)
    ("ossicles",          COCHLEAR_HAIR,        2102),  # K=2
    ("oval_window",       COCHLEAR_HAIR,        2103),  # K=2
    ("cochlea_a",         COCHLEAR_HAIR,        2104),  # K=2
    ("auditory_nerve",    MOTOR_ALPHA,          2105),  # K=3
    ("auditory_cortex_a", CORTICAL_PYRAMIDAL,   2106),  # K=5
]

AUDITORY_WIRING: List[Tuple[str, str]] = [
    ("ear_canal", "tympanic"),
    ("tympanic", "ossicles"),
    ("ossicles", "oval_window"),
    ("oval_window", "cochlea_a"),
    ("cochlea_a", "auditory_nerve"),
    ("auditory_nerve", "auditory_cortex_a"),
    # Efferent modulation (stapedius reflex)
    ("auditory_cortex_a", "cochlea_a"),
]


def build_auditory(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy auditory GammaTopology (7 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in AUDITORY_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=3, gamma_threshold=0.3)
    for src, tgt in AUDITORY_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Vestibular Blueprint
# ============================================================================
#
#   semicircular (K=2) → vestibular_n (K=2) → vestibular_nucleus (K=3)
#                                                     │
#   otolith (K=2) → vestibular_n              cerebellum_v (K=3)
#                                                     │
#                                              oculomotor (K=2)  (VOR)

VESTIBULAR_NODES: List[Tuple[str, TissueType, int]] = [
    ("semicircular",       VESTIBULAR_CELL,  2200),  # K=2
    ("otolith",            VESTIBULAR_CELL,  2201),  # K=2
    ("vestibular_n",       VESTIBULAR_CELL,  2202),  # K=2
    ("vestibular_nucleus", MOTOR_ALPHA,      2203),  # K=3
    ("cerebellum_v",       MOTOR_ALPHA,      2204),  # K=3
    ("oculomotor",         AUTONOMIC_PREGANGLIONIC, 2205),  # K=2
]

VESTIBULAR_WIRING: List[Tuple[str, str]] = [
    ("semicircular", "vestibular_n"),
    ("otolith", "vestibular_n"),
    ("vestibular_n", "vestibular_nucleus"),
    ("vestibular_nucleus", "cerebellum_v"),
    ("vestibular_nucleus", "oculomotor"),
    # VOR feedback
    ("oculomotor", "vestibular_nucleus"),
]


def build_vestibular(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy vestibular GammaTopology (6 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in VESTIBULAR_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=2, gamma_threshold=0.3)
    for src, tgt in VESTIBULAR_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Olfactory Blueprint
# ============================================================================
#
#   olfactory_epithelium (K=1) → olfactory_bulb (K=2) → piriform_cortex (K=3)
#                                       │
#                               olfactory_nerve (K=1)
#
# Unique: olfactory receptor neurons undergo lifelong neurogenesis.

OLFACTORY_NODES: List[Tuple[str, TissueType, int]] = [
    ("olfactory_epithelium", OLFACTORY_RECEPTOR, 2300),  # K=1
    ("olfactory_nerve",      OLFACTORY_RECEPTOR, 2301),  # K=1
    ("olfactory_bulb",       AUTONOMIC_PREGANGLIONIC, 2302),  # K=2
    ("piriform_cortex",      MOTOR_ALPHA,        2303),  # K=3
]

OLFACTORY_WIRING: List[Tuple[str, str]] = [
    ("olfactory_epithelium", "olfactory_nerve"),
    ("olfactory_nerve", "olfactory_bulb"),
    ("olfactory_bulb", "piriform_cortex"),
    # Top-down modulation
    ("piriform_cortex", "olfactory_bulb"),
]


def build_olfactory(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy olfactory GammaTopology (4 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in OLFACTORY_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=2, gamma_threshold=0.3)
    for src, tgt in OLFACTORY_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Oral/Dental Blueprint
# ============================================================================
#
#   enamel (K=1) → dentin (K=1) → pulp (K=1) → periapical (K=1)
#                                    │
#   periodontal_lig (K=1) ← root (K=1)
#        │
#   alveolar_bone (K=2)
#
# Highest Z tissue in the body (mineralized enamel).

DENTAL_NODES: List[Tuple[str, TissueType, int]] = [
    ("enamel",            DENTAL,    2400),   # K=1 (highest Z)
    ("dentin",            DENTAL,    2401),   # K=1
    ("pulp",              DENTAL,    2402),   # K=1
    ("root",              DENTAL,    2403),   # K=1
    ("periapical",        DENTAL,    2404),   # K=1
    ("periodontal_lig",   TENDON,    2405),   # K=1
    ("alveolar_bone",     BONE_CELL, 2406),   # K=2
]

DENTAL_WIRING: List[Tuple[str, str]] = [
    ("enamel", "dentin"),
    ("dentin", "pulp"),
    ("pulp", "root"),
    ("root", "periapical"),
    ("root", "periodontal_lig"),
    ("periodontal_lig", "alveolar_bone"),
    # Pain feedback
    ("pulp", "dentin"),
]


def build_dental(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy oral/dental GammaTopology (7 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in DENTAL_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=2, gamma_threshold=0.3)
    for src, tgt in DENTAL_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Enteric Nervous System Blueprint
# ============================================================================
#
#   vagus_input (K=2) → myenteric_plexus (K=1) → longitudinal_muscle (K=1)
#                                │
#   sympathetic_input (K=2)  submucosal_plexus (K=1) → circular_muscle (K=1)
#                                │                            │
#                         mucosal_sensor (K=1) → submucosal_plexus
#
# The "second brain" — 500 million neurons, operates independently.

ENTERIC_NODES: List[Tuple[str, TissueType, int]] = [
    ("vagus_input",          AUTONOMIC_PREGANGLIONIC, 2500),  # K=2
    ("sympathetic_input",    AUTONOMIC_PREGANGLIONIC, 2501),  # K=2
    ("myenteric_plexus",     ENTERIC_NEURON,          2502),  # K=1
    ("submucosal_plexus",    ENTERIC_NEURON,          2503),  # K=1
    ("longitudinal_muscle",  GI_SMOOTH_MUSCLE,        2504),  # K=1
    ("circular_muscle",      GI_SMOOTH_MUSCLE,        2505),  # K=1
    ("mucosal_sensor",       ENTERIC_NEURON,          2506),  # K=1
]

ENTERIC_WIRING: List[Tuple[str, str]] = [
    ("vagus_input", "myenteric_plexus"),
    ("sympathetic_input", "myenteric_plexus"),
    ("myenteric_plexus", "longitudinal_muscle"),
    ("myenteric_plexus", "submucosal_plexus"),
    ("submucosal_plexus", "circular_muscle"),
    ("mucosal_sensor", "submucosal_plexus"),
    # Intrinsic reflexes
    ("circular_muscle", "mucosal_sensor"),
]


def build_enteric(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy enteric nervous system GammaTopology (7 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in ENTERIC_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=2, gamma_threshold=0.3)
    for src, tgt in ENTERIC_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Autonomic Nervous System Blueprint
# ============================================================================
#
#   hypothalamus_ans (K=3) → sympathetic_chain (K=2) → target_organ_s (K=1)
#           │                                               │
#           └→ vagal_nucleus (K=3) → vagus_nerve (K=2) → target_organ_ps (K=1)
#                                                               │
#   adrenal_medulla_ans (K=2) ← sympathetic_chain       heart_node (K=2)
#
# Sympathetic vs parasympathetic = impedance balance.

AUTONOMIC_NODES: List[Tuple[str, TissueType, int]] = [
    ("hypothalamus_ans",    HYPOTHALAMIC,             2600),  # K=3
    ("sympathetic_chain",   AUTONOMIC_PREGANGLIONIC,  2601),  # K=2
    ("vagal_nucleus",       BAROREFLEX,               2602),  # K=3
    ("vagus_nerve",         AUTONOMIC_PREGANGLIONIC,  2603),  # K=2
    ("target_organ_s",      ENTERIC_NEURON,           2604),  # K=1
    ("target_organ_ps",     ENTERIC_NEURON,           2605),  # K=1
    ("adrenal_medulla_ans", ENDOCRINE_GLAND,          2606),  # K=2
    ("heart_node",          AUTONOMIC_PREGANGLIONIC,  2607),  # K=2
]

AUTONOMIC_WIRING: List[Tuple[str, str]] = [
    ("hypothalamus_ans", "sympathetic_chain"),
    ("hypothalamus_ans", "vagal_nucleus"),
    ("sympathetic_chain", "target_organ_s"),
    ("sympathetic_chain", "adrenal_medulla_ans"),
    ("sympathetic_chain", "heart_node"),
    ("vagal_nucleus", "vagus_nerve"),
    ("vagus_nerve", "target_organ_ps"),
    ("vagus_nerve", "heart_node"),
    # Feedback
    ("heart_node", "vagal_nucleus"),
    ("target_organ_s", "hypothalamus_ans"),
]


def build_autonomic(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy autonomic NS GammaTopology (8 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in AUTONOMIC_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=3, gamma_threshold=0.3)
    for src, tgt in AUTONOMIC_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Pancreatic Blueprint
# ============================================================================
#
#   glucose_sensor (K=2) → beta_cell_p (K=2) → insulin_out (K=1)
#                                │                    │
#   alpha_cell (K=2) → glucagon_out (K=1)     peripheral_tissue (K=1)
#         │                                          │
#   delta_cell (K=1)                           glucose_sensor (feedback)
#
# Islet microanatomy: β→δ→α paracrine cascade.

PANCREATIC_NODES: List[Tuple[str, TissueType, int]] = [
    ("glucose_sensor",     BETA_CELL,     2700),  # K=2
    ("beta_cell_p",        BETA_CELL,     2701),  # K=2
    ("alpha_cell",         BETA_CELL,     2702),  # K=2
    ("delta_cell",         ENTERIC_NEURON, 2703), # K=1 (somatostatin)
    ("insulin_out",        ENTERIC_NEURON, 2704), # K=1
    ("glucagon_out",       ENTERIC_NEURON, 2705), # K=1
    ("peripheral_tissue",  ENTERIC_NEURON, 2706), # K=1
]

PANCREATIC_WIRING: List[Tuple[str, str]] = [
    ("glucose_sensor", "beta_cell_p"),
    ("glucose_sensor", "alpha_cell"),
    ("beta_cell_p", "insulin_out"),
    ("beta_cell_p", "delta_cell"),
    ("alpha_cell", "glucagon_out"),
    ("delta_cell", "alpha_cell"),
    ("insulin_out", "peripheral_tissue"),
    # Glucose homeostasis feedback
    ("peripheral_tissue", "glucose_sensor"),
]


def build_pancreatic(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy pancreatic islet GammaTopology (7 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in PANCREATIC_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=2, gamma_threshold=0.3)
    for src, tgt in PANCREATIC_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# CSF / Glymphatic Blueprint
# ============================================================================
#
#   choroid_plexus (K=1) → ventricle (K=1) → subarachnoid (K=1)
#                                                   │
#   venous_drainage (K=1) ← arachnoid_villi (K=1) ←┘
#                                                   │
#   glymphatic_channel (K=1) → perivascular (K=1) → interstitial (K=1)
#
# Waste clearance operates primarily during sleep (N3).

CSF_NODES: List[Tuple[str, TissueType, int]] = [
    ("choroid_plexus",      EPENDYMAL, 2800),   # K=1
    ("ventricle",           EPENDYMAL, 2801),   # K=1
    ("subarachnoid",        EPENDYMAL, 2802),   # K=1
    ("arachnoid_villi",     EPENDYMAL, 2803),   # K=1
    ("venous_drainage",     EPENDYMAL, 2804),   # K=1
    ("glymphatic_channel",  EPENDYMAL, 2805),   # K=1
    ("perivascular",        EPENDYMAL, 2806),   # K=1
    ("interstitial",        EPENDYMAL, 2807),   # K=1
]

CSF_WIRING: List[Tuple[str, str]] = [
    ("choroid_plexus", "ventricle"),
    ("ventricle", "subarachnoid"),
    ("subarachnoid", "arachnoid_villi"),
    ("arachnoid_villi", "venous_drainage"),
    ("subarachnoid", "glymphatic_channel"),
    ("glymphatic_channel", "perivascular"),
    ("perivascular", "interstitial"),
    # Recirculation
    ("interstitial", "venous_drainage"),
]


def build_csf(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a healthy CSF/glymphatic GammaTopology (8 nodes)."""
    nodes = [_make_node(n, t, seed=s) for n, t, s in CSF_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=1, gamma_threshold=0.3)
    for src, tgt in CSF_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Brain Functional Topology
# ============================================================================
#
# This is NOT the neural cable (build_neural). This is the cortical-
# subcortical functional connectivity graph.
#
#   prefrontal (K=5) ↔ amygdala (K=3) ↔ hippocampus (K=3)
#        │                  │                    │
#   basal_ganglia (K=3)  thalamus_b (K=5) ← hypothalamus_b (K=3)
#        │                  │
#   motor_cortex_b (K=5) ←─┘
#        │
#   cerebellum_b (K=3)
#        │
#   broca_b (K=3) ↔ wernicke_b (K=3)
#
# K=5 for cortical association areas, K=3 for subcortical structures.
# Dimensional mismatch (5→3) creates natural communication barriers
# between cortex and subcortex — this IS the physics of top-down control.

BRAIN_FUNCTIONAL_NODES: List[Tuple[str, TissueType, int]] = [
    ("prefrontal",      CORTICAL_PYRAMIDAL, 2900),  # K=5
    ("motor_cortex_b",  CORTICAL_PYRAMIDAL, 2901),  # K=5
    ("sensory_cortex",  CORTICAL_PYRAMIDAL, 2902),  # K=5
    ("thalamus_b",      CORTICAL_PYRAMIDAL, 2903),  # K=5 (relay hub)
    ("amygdala",        MOTOR_ALPHA,        2904),  # K=3
    ("hippocampus",     MOTOR_ALPHA,        2905),  # K=3
    ("basal_ganglia_b", MOTOR_ALPHA,        2906),  # K=3
    ("hypothalamus_b",  MOTOR_ALPHA,        2907),  # K=3
    ("cerebellum_b",    MOTOR_ALPHA,        2908),  # K=3
    ("broca_b",         MOTOR_ALPHA,        2909),  # K=3
    ("wernicke_b",      MOTOR_ALPHA,        2910),  # K=3
    ("cingulate",       MOTOR_ALPHA,        2911),  # K=3 (salience)
]

BRAIN_FUNCTIONAL_WIRING: List[Tuple[str, str]] = [
    # Cortical ↔ thalamic (thalamocortical loops)
    ("thalamus_b", "prefrontal"),
    ("thalamus_b", "motor_cortex_b"),
    ("thalamus_b", "sensory_cortex"),
    ("prefrontal", "thalamus_b"),
    ("sensory_cortex", "thalamus_b"),
    # Prefrontal executive control
    ("prefrontal", "basal_ganglia_b"),
    ("prefrontal", "amygdala"),           # top-down emotional regulation
    ("prefrontal", "motor_cortex_b"),
    # Limbic circuit
    ("amygdala", "hippocampus"),
    ("hippocampus", "prefrontal"),
    ("amygdala", "hypothalamus_b"),
    ("cingulate", "amygdala"),
    ("cingulate", "prefrontal"),
    # Basal ganglia motor loop
    ("basal_ganglia_b", "thalamus_b"),
    ("motor_cortex_b", "basal_ganglia_b"),
    # Cerebellar loop
    ("motor_cortex_b", "cerebellum_b"),
    ("cerebellum_b", "thalamus_b"),
    # Language circuit
    ("wernicke_b", "broca_b"),            # arcuate fasciculus
    ("broca_b", "motor_cortex_b"),
    ("sensory_cortex", "wernicke_b"),
    # Hippocampal memory → cortex
    ("hippocampus", "sensory_cortex"),
]


def build_brain_functional(eta: float = 0.01, seed: Optional[int] = 42) -> GammaTopology:
    """Build a brain functional connectivity GammaTopology (12 nodes).

    Covers major cortical-subcortical circuits: executive, limbic,
    motor, language, memory. K=5 cortex vs K=3 subcortex creates
    natural dimensional mismatch — the physics of top-down control.
    """
    nodes = [_make_node(n, t, seed=s) for n, t, s in BRAIN_FUNCTIONAL_NODES]
    topo = GammaTopology(nodes=nodes, eta=eta, max_dimension_gap=3, gamma_threshold=0.3)
    for src, tgt in BRAIN_FUNCTIONAL_WIRING:
        topo.activate_edge(src, tgt)
    return topo


# ============================================================================
# Registry: all available blueprints
# ============================================================================

BLUEPRINT_REGISTRY = {
    # Tier 1 — original 19 tissues
    "cardiovascular":    build_cardiovascular,
    "neural":            build_neural,
    "pulmonary":         build_pulmonary,
    "renal":             build_renal,
    "hepatic":           build_hepatic,
    "gi":                build_gi,
    "endocrine":         build_endocrine,
    "immune":            build_immune,
    "lymphatic":         build_lymphatic,
    "musculoskeletal":   build_musculoskeletal,
    "skeletal":          build_skeletal,
    "epithelial":        build_epithelial,
    "connective":        build_connective,
    "adipose":           build_adipose,
    "hematopoietic":     build_hematopoietic,
    "cartilage":         build_cartilage,
    "reproductive":      build_reproductive,
    "brain_regulatory":  build_brain_regulatory,
    "vascular":          build_vascular,
    # Tier 2 — new tissue systems
    "ocular":            build_ocular,
    "auditory":          build_auditory,
    "vestibular":        build_vestibular,
    "olfactory":         build_olfactory,
    "dental":            build_dental,
    "enteric":           build_enteric,
    "autonomic":         build_autonomic,
    "pancreatic":        build_pancreatic,
    "csf":               build_csf,
    "brain_functional":  build_brain_functional,
}


# ============================================================================
#                  WHOLE-BODY INTER-ORGAN ARCHITECTURE
# ============================================================================
#
# Three Chains → One Body:
#
#   Chain ① — 29 organ GammaTopologies (intra-organ physics)
#   Chain ② — Dual inter-organ bus:
#             Vascular bus (K=4→2→1): aorta → arteriolar branches → capillary beds
#             Neural bus (K=3→2→1): autonomic preganglionic → postganglionic → target
#   Chain ③ — Brain command center: brain_functional (K=5/3)
#
# Connection principle:
#   Each organ declares two interface nodes:
#     • vascular_port — the node receiving arterial blood supply
#     • neural_port  — the node receiving autonomic innervation
#
#   The whole-body topology:
#     1. Instantiate all 29 organ blueprints (nodes prefixed: "cv.", "pulm.", etc.)
#     2. Create vascular hub nodes (one per organ): "vbus.cv", "vbus.pulm", ...
#     3. Create neural hub nodes (autonomic branches): "nbus.cv", "nbus.pulm", ...
#     4. Wire: vbus hub → organ vascular_port (blood supply)
#     5. Wire: nbus hub → organ neural_port (innervation)
#     6. Wire: brain.hypothalamus_b → nbus hubs (descending command)
#     7. Wire: organ → vbus → brain (ascending feedback: interoception)
#
# Physics:
#   • Dimensional cascade: brain (K=5) → autonomic (K=3→2) → organ (K=1~3)
#     → $A_\text{cut}$ costs emerge naturally at each level transition
#   • C2 impedance remodeling runs on ALL edges (intra + inter)
#     → systemic homeostasis emerges from physics alone (E0)
#   • Disease cascade: Γ↑ at vascular hub → all downstream organs Γ↑
#     → zero disease-specific code needed
#

# ── Interface node definitions ────────────────────────────────────────
#
# For each organ: (prefix, vascular_port_node, neural_port_node)
# vascular_port = the node that receives blood supply
# neural_port   = the node that receives autonomic innervation
# Some organs share both roles in one node; that's fine.

ORGAN_INTERFACES: Dict[str, Dict[str, str]] = {
    # Organ           vascular port              neural port
    "cardiovascular": {"vascular": "coronary",         "neural": "sa_node"},
    "neural":         {"vascular": "cortex_motor",     "neural": "cortex_motor"},
    "pulmonary":      {"vascular": "pulm_capillary",   "neural": "trachea"},
    "renal":          {"vascular": "afferent_arteriole","neural": "afferent_arteriole"},
    "hepatic":        {"vascular": "portal_vein",      "neural": "hepatocyte"},
    "gi":             {"vascular": "stomach",          "neural": "enteric_plexus"},
    "endocrine":      {"vascular": "hypothalamus",     "neural": "hypothalamus"},
    "immune":         {"vascular": "bone_marrow",      "neural": "spleen"},
    "lymphatic":      {"vascular": "venous_return",    "neural": "thoracic_duct"},
    "musculoskeletal":{"vascular": "muscle_fiber",     "neural": "spinal_mn"},
    "skeletal":       {"vascular": "periosteum",       "neural": "osteocyte"},
    "epithelial":     {"vascular": "dermis",           "neural": "basal_layer"},
    "connective":     {"vascular": "fibroblast_c",     "neural": "fibroblast_c"},
    "adipose":        {"vascular": "adipocyte_a",      "neural": "adipocyte_a"},
    "hematopoietic":  {"vascular": "hsc_cell",          "neural": "hsc_cell"},
    "cartilage":      {"vascular": "perichondrium",    "neural": "perichondrium"},
    "reproductive":   {"vascular": "gonad_r",          "neural": "hypothalamus_r"},
    "brain_regulatory":{"vascular": "nts",             "neural": "nts"},
    "vascular":       {"vascular": "aortic_root",      "neural": "aortic_root"},
    "ocular":         {"vascular": "ciliary_body",     "neural": "optic_nerve"},
    "auditory":       {"vascular": "cochlea_a",        "neural": "auditory_nerve"},
    "vestibular":     {"vascular": "vestibular_n",     "neural": "vestibular_nucleus"},
    "olfactory":      {"vascular": "olfactory_bulb",   "neural": "piriform_cortex"},
    "dental":         {"vascular": "pulp",             "neural": "pulp"},
    "enteric":        {"vascular": "myenteric_plexus", "neural": "vagus_input"},
    "autonomic":      {"vascular": "adrenal_medulla_ans","neural": "hypothalamus_ans"},
    "pancreatic":     {"vascular": "beta_cell_p",      "neural": "glucose_sensor"},
    "csf":            {"vascular": "choroid_plexus",   "neural": "choroid_plexus"},
    "brain_functional":{"vascular": "thalamus_b",      "neural": "hypothalamus_b"},
}


# ── Vascular bus tissue types ─────────────────────────────────────────

VASCULAR_BUS_TRUNK = TissueType(
    "vascular_bus_trunk", n_modes=4, z_mean=40.0, z_std=8.0,
    diameter_um=25.0, myelinated=False,
    description="Aortic trunk — main vascular bus backbone (K=4)")

VASCULAR_BUS_BRANCH = TissueType(
    "vascular_bus_branch", n_modes=2, z_mean=55.0, z_std=12.0,
    diameter_um=5.0, myelinated=False,
    description="Arteriolar branch — organ-specific vascular feed (K=2)")

NEURAL_BUS_TRUNK = TissueType(
    "neural_bus_trunk", n_modes=3, z_mean=70.0, z_std=15.0,
    diameter_um=5.0, myelinated=True,
    description="Autonomic trunk — main neural bus backbone (K=3)")

NEURAL_BUS_BRANCH = TissueType(
    "neural_bus_branch", n_modes=2, z_mean=90.0, z_std=20.0,
    diameter_um=3.0, myelinated=True,
    description="Autonomic branch — organ-specific neural feed (K=2)")


def build_whole_body(
    eta: float = 0.01,
    max_dimension_gap: int = 3,
    seed: int = 42,
) -> GammaTopology:
    """Build a unified whole-body GammaTopology.

    Architecture (three chains unified):
      Chain ① — 29 organ sub-topologies, nodes prefixed by organ name
      Chain ② — Dual bus: vascular (K=4→2) + neural (K=3→2) hubs
      Chain ③ — Brain (brain_functional) as central command

    Dimensional cascade:
      Brain K=5 → Neural bus K=3 → Organ K=1~3: A_cut emerges naturally
      Aorta K=4 → Branch K=2 → Organ capillary K=1: vascular A_cut

    Returns
    -------
    GammaTopology with ~260 nodes and ~350 edges,
    all satisfying C1 (energy conservation) at every edge.

    Emergence level: E0 — zero disease-specific code.
    """
    all_nodes: List[GammaNode] = []
    all_wiring: List[Tuple[str, str]] = []

    # Track organ interface nodes (prefixed names)
    vascular_ports: Dict[str, str] = {}   # organ → prefixed vascular port name
    neural_ports: Dict[str, str] = {}     # organ → prefixed neural port name

    rng = np.random.default_rng(seed)

    # ── Step 1: Instantiate all 29 organ sub-topologies ───────────────
    #
    # Each organ's nodes get prefixed: "cv.sa_node", "pulm.alveolus", etc.
    # Each organ's edges are recorded with prefixed names.

    for organ_name, build_fn in BLUEPRINT_REGISTRY.items():
        prefix = _organ_prefix(organ_name)
        organ_seed = seed + hash(organ_name) % 10000

        # Build the standalone organ topology
        organ_topo = build_fn(seed=organ_seed)

        # Extract nodes with prefixed names
        for node_name, node in organ_topo.nodes.items():
            prefixed_name = f"{prefix}.{node_name}"
            prefixed_node = GammaNode(
                name=prefixed_name,
                impedance=node.impedance.copy(),
                activation=node.activation.copy(),
            )
            all_nodes.append(prefixed_node)

        # Extract edges with prefixed names
        for (src, tgt) in organ_topo.active_edges.keys():
            all_wiring.append((f"{prefix}.{src}", f"{prefix}.{tgt}"))

        # Record interface nodes (prefixed)
        iface = ORGAN_INTERFACES[organ_name]
        vascular_ports[organ_name] = f"{prefix}.{iface['vascular']}"
        neural_ports[organ_name] = f"{prefix}.{iface['neural']}"

    # ── Step 2: Create vascular bus (aorta → branches → organs) ───────
    #
    # Trunk: single aortic hub node (K=4)
    # Branches: one per organ (K=2), connecting trunk → organ vascular port

    vbus_trunk = _make_node("vbus.aorta", VASCULAR_BUS_TRUNK, seed=seed + 9000)
    all_nodes.append(vbus_trunk)

    for organ_name in BLUEPRINT_REGISTRY:
        branch_name = f"vbus.{_organ_prefix(organ_name)}"
        branch_node = _make_node(
            branch_name, VASCULAR_BUS_BRANCH,
            seed=seed + 9100 + hash(organ_name) % 1000,
        )
        all_nodes.append(branch_node)

        # Trunk → branch (K=4 → K=2: cutoff cost = 2)
        all_wiring.append(("vbus.aorta", branch_name))

        # Branch → organ vascular port (K=2 → organ K: variable cutoff)
        all_wiring.append((branch_name, vascular_ports[organ_name]))

        # Return path: organ vascular port → branch (venous return)
        all_wiring.append((vascular_ports[organ_name], branch_name))

    # ── Step 3: Create neural bus (brain → autonomic → organs) ────────
    #
    # Trunk: single autonomic hub (K=3) — connected to brain
    # Branches: one per organ (K=2), connecting trunk → organ neural port

    nbus_trunk = _make_node("nbus.autonomic", NEURAL_BUS_TRUNK, seed=seed + 9500)
    all_nodes.append(nbus_trunk)

    for organ_name in BLUEPRINT_REGISTRY:
        branch_name = f"nbus.{_organ_prefix(organ_name)}"
        branch_node = _make_node(
            branch_name, NEURAL_BUS_BRANCH,
            seed=seed + 9600 + hash(organ_name) % 1000,
        )
        all_nodes.append(branch_node)

        # Trunk → branch (K=3 → K=2: cutoff cost = 1)
        all_wiring.append(("nbus.autonomic", branch_name))

        # Branch → organ neural port (K=2 → organ K: variable cutoff)
        all_wiring.append((branch_name, neural_ports[organ_name]))

        # Afferent path: organ neural port → branch (ascending signal)
        all_wiring.append((neural_ports[organ_name], branch_name))

    # ── Step 4: Connect brain to buses ────────────────────────────────
    #
    # Brain command hierarchy:
    #   brain_functional.hypothalamus_b (K=3) → nbus.autonomic (K=3)
    #   brain_functional.thalamus_b (K=5) ← nbus.autonomic (K=3) (ascending)
    #   brain_functional.hypothalamus_b (K=3) ← vbus.aorta (K=4) (interoception)

    brain_prefix = _organ_prefix("brain_functional")

    # Descending: brain → neural bus
    all_wiring.append((f"{brain_prefix}.hypothalamus_b", "nbus.autonomic"))
    # Ascending: neural bus → brain thalamus (sensory relay)
    all_wiring.append(("nbus.autonomic", f"{brain_prefix}.thalamus_b"))

    # Vascular interoception: vascular bus → brain
    all_wiring.append(("vbus.aorta", f"{brain_prefix}.hypothalamus_b"))
    # Brain → vascular regulation (autonomic tone)
    all_wiring.append((f"{brain_prefix}.hypothalamus_b", "vbus.aorta"))

    # ── Step 5: Cross-bus link (vascular ↔ neural) ────────────────────
    #
    # The two buses are not independent — blood pressure affects neural
    # function and vice versa. Link the trunks bidirectionally.

    all_wiring.append(("vbus.aorta", "nbus.autonomic"))
    all_wiring.append(("nbus.autonomic", "vbus.aorta"))

    # ── Step 6: Assemble the whole-body GammaTopology ─────────────────

    topo = GammaTopology(
        nodes=all_nodes,
        eta=eta,
        max_dimension_gap=max_dimension_gap,
        gamma_threshold=0.3,
    )

    # Activate all edges
    for src, tgt in all_wiring:
        topo.activate_edge(src, tgt)

    return topo


def _organ_prefix(organ_name: str) -> str:
    """Short prefix for organ node names in whole-body topology."""
    _PREFIX_MAP = {
        "cardiovascular": "cv",
        "neural": "neur",
        "pulmonary": "pulm",
        "renal": "ren",
        "hepatic": "hep",
        "gi": "gi",
        "endocrine": "endo",
        "immune": "imm",
        "lymphatic": "lymp",
        "musculoskeletal": "msk",
        "skeletal": "skel",
        "epithelial": "epi",
        "connective": "conn",
        "adipose": "adip",
        "hematopoietic": "hema",
        "cartilage": "cart",
        "reproductive": "repr",
        "brain_regulatory": "breg",
        "vascular": "vasc",
        "ocular": "ocul",
        "auditory": "aud",
        "vestibular": "vest",
        "olfactory": "olfa",
        "dental": "dent",
        "enteric": "ent",
        "autonomic": "auto",
        "pancreatic": "panc",
        "csf": "csf",
        "brain_functional": "bfun",
    }
    return _PREFIX_MAP[organ_name]
