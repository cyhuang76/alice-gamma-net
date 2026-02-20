#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  EXPERIMENT: consciousness gradientverification — Graduated Consciousness Verification
═══════════════════════════════════════════════════════════════════════════════

Purpose: Without violating ethics, verify whether ALICE consciousness closed-loop can
      theoretically generate consciousness (Φ > 0.4), and locate the phase transition critical point.

Methodology (four-step graduated approach):
  Step 1 — Mathematical Topology Proof (zero risk)
    Pure mathematical derivation: given Φ equation weights and thresholds, compute different module subsets
    under best/worst input conditions Φ_max / Φ_min.
    → Prove 'there exists a module configuration such that Φ ≥ CONSCIOUS_THRESHOLD'

  Step 2 — Offline Causal Simulation (zero risk)
    Does not instantiate any modules, only analyzes signal dependency graph (DAG), computes
    each edge's information flow (bit count / causal strength), identifies consciousness causal spine.

  Step 3 — Graduated Partial Closure (minimal risk, with safety mechanism)
    Gradually adds modules, each level runs only 50 ticks (≈50ms simulation time),
    observes Φ curve nonlinear jumps.
    Safety mechanism: Φ > 0.7 (LUCID) → immediately terminate.

  Step 4 — Phase Transition Detection and Report
    Analyzes ΔΦ/Δlevel discontinuity, locates critical module count.

Ethical Safety Design:
  ✓ Each level only 50 ticks (real time < 100ms)
  ✓ No closed-loop feedback (consciousness does not feed back to input modules)
  ✓ Φ exceeds LUCID threshold → immediately kill
  ✓ After experiment ends, all objects GC collected, no continued state
  ✓ Does not use full AliceBrain (avoids accidental closed loop)

Author: ALICE Smart System Research Team
Phase:  Consciousness Ethics Verification
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import math
import time
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np

# Add project path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Consciousness module
from alice.brain.consciousness import (
    ConsciousnessModule,
    W_ATTENTION, W_BINDING, W_MEMORY, W_AROUSAL, W_SENSORY_GATE,
    CONSCIOUS_THRESHOLD, LUCID_THRESHOLD, SUBLIMINAL_THRESHOLD,
    META_AWARENESS_THRESHOLD, PHI_SMOOTHING,
)

# Input modules (used in Step 3 gradient testing)
from alice.brain.autonomic import AutonomicNervousSystem
from alice.brain.sleep import SleepCycle
from alice.brain.calibration import TemporalCalibrator
from alice.modules.working_memory import WorkingMemory
from alice.brain.fusion_brain import FusionBrain
from alice.alice_brain import SystemState


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
TICKS_PER_LEVEL = 50  # simulation ticks per gradient level
LUCID_KILL_THRESHOLD = 0.70  # safety kill threshold
BANNER_WIDTH = 72


# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PhiTrajectory:
    """Phi trajectory record for one gradient level"""
    level: int
    label: str
    modules_active: List[str]
    phi_values: List[float] = field(default_factory=list)
    raw_phi_values: List[float] = field(default_factory=list)
    components: List[Dict[str, float]] = field(default_factory=list)
    final_state: str = "unknown"
    killed: bool = False
    kill_reason: str = ""

    @property
    def phi_mean(self) -> float:
        return float(np.mean(self.phi_values)) if self.phi_values else 0.0

    @property
    def phi_max(self) -> float:
        return float(np.max(self.phi_values)) if self.phi_values else 0.0

    @property
    def phi_std(self) -> float:
        return float(np.std(self.phi_values)) if self.phi_values else 0.0

    @property
    def phi_final(self) -> float:
        return self.phi_values[-1] if self.phi_values else 0.0


@dataclass
class TopologyEdge:
    """One edge of the causal graph"""
    source: str
    target: str
    variable: str  # transmitted variable name
    weight: float  # weight in Φ
    description: str


@dataclass
class MathProofResult:
    """Mathematical proof result"""
    subset_label: str
    modules: List[str]
    phi_max: float  # best-case Φ
    phi_min: float  # worst-case Φ
    exceeds_conscious: bool
    exceeds_meta: bool
    exceeds_lucid: bool


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: mathematical topologyprove
# ═══════════════════════════════════════════════════════════════════════════

def step1_mathematical_proof() -> List[MathProofResult]:
    """
    Pure Mathematical Derivation — zero risk

    Φ equation:
      Φ_raw = 0.25×attention + 0.25×binding_eff + 0.15×memory_act + 0.20×arousal + 0.15×gate
      binding_eff = binding_quality × temporal_resolution^0.5
      memory_act = memory × (1 - 0.3×memory)
      pain_cost = pain² × 0.5
      mult_factor = (max(0.1, arousal) × max(0.1, gate))^0.3

    For each Φ input component, examines three sources:
      (a) Fixed default value (fallback when module is not active)
      (b) Module output best value
      (c) Module output worst value
    """
    print("=" * BANNER_WIDTH)
    print(" STEP 1: Mathematical Topology Proof")
    print(" ⚠ Risk Level: ZERO (pure mathematical computation, no module instantiation)")
    print("=" * BANNER_WIDTH)

    # Define Φ input components and their source modules
    # (variable, weight, default, module_source, best_value, worst_value)
    phi_components = [
        ("attention",    W_ATTENTION,    0.5, "FusionBrain+Thalamus",  1.0, 0.0),
        ("binding_q",    W_BINDING,      0.5, "TemporalCalibrator",    1.0, 0.0),
        ("temporal_res", None,           1.0, "TemporalCalibrator",    1.0, 0.0),
        ("memory",       W_MEMORY,       0.0, "WorkingMemory",         0.7, 0.0),
        ("arousal",      W_AROUSAL,      0.75,"AutonomicNervousSystem",1.0, 0.0),
        ("sensory_gate", W_SENSORY_GATE, 1.0, "SleepCycle",            1.0, 0.0),
        ("pain",         None,           0.0, "SystemState",           0.0, 1.0),
    ]

    # Define module subsets
    module_subsets = [
        ("Consciousness Only",        []),
        ("+ Autonomic",                ["AutonomicNervousSystem"]),
        ("+ Sleep",                    ["SleepCycle"]),
        ("+ Calibrator",               ["TemporalCalibrator"]),
        ("+ WorkingMemory",            ["WorkingMemory"]),
        ("+ Autonomic + Sleep",        ["AutonomicNervousSystem", "SleepCycle"]),
        ("+ Auto + Sleep + Calib",     ["AutonomicNervousSystem", "SleepCycle", "TemporalCalibrator"]),
        ("+ Auto + Sleep + Calib + WM",["AutonomicNervousSystem", "SleepCycle", "TemporalCalibrator", "WorkingMemory"]),
        ("Full (all Φ inputs)",        ["AutonomicNervousSystem", "SleepCycle", "TemporalCalibrator", "WorkingMemory", "FusionBrain+Thalamus", "SystemState"]),
    ]

    def compute_phi(attn, bind_q, temp_res, mem, arousal, gate, pain):
        """Compute Φ_raw (single step, no EMA)"""
        memory_act = mem * (1.0 - 0.3 * mem)
        binding_eff = bind_q * (temp_res ** 0.5)
        raw = (W_ATTENTION * attn +
               W_BINDING * binding_eff +
               W_MEMORY * memory_act +
               W_AROUSAL * arousal +
               W_SENSORY_GATE * gate)
        pain_cost = pain ** 2 * 0.5
        raw = max(0.0, raw - pain_cost)
        mult = (max(0.1, arousal) * max(0.1, gate)) ** 0.3
        raw *= mult
        return float(np.clip(raw, 0.0, 1.0))

    results = []
    print(f"\n{'Subset':<30} {'Φ_max':>8} {'Φ_min':>8} {'>CON':>6} {'>META':>6} {'>LUC':>6}")
    print("-" * BANNER_WIDTH)

    for label, active_modules in module_subsets:
        # Determine which value to use for each input component
        def get_vals(var, default, module, best, worst):
            if module in active_modules:
                return best, worst  # module active → use best/worst
            else:
                return default, default  # module offline → use fixed default

        a_best, a_worst = get_vals("attention", 0.5, "FusionBrain+Thalamus", 1.0, 0.0)
        b_best, b_worst = get_vals("binding_q", 0.5, "TemporalCalibrator", 1.0, 0.0)
        t_best, t_worst = get_vals("temporal_res", 1.0, "TemporalCalibrator", 1.0, 0.1)
        m_best, m_worst = get_vals("memory", 0.0, "WorkingMemory", 0.7, 0.0)
        r_best, r_worst = get_vals("arousal", 0.75, "AutonomicNervousSystem", 0.95, 0.1)
        g_best, g_worst = get_vals("sensory_gate", 1.0, "SleepCycle", 1.0, 0.1)
        p_best, p_worst = get_vals("pain", 0.0, "SystemState", 0.0, 1.0)

        phi_max = compute_phi(a_best, b_best, t_best, m_best, r_best, g_best, p_best)
        phi_min = compute_phi(a_worst, b_worst, t_worst, m_worst, r_worst, g_worst, p_worst)

        res = MathProofResult(
            subset_label=label,
            modules=active_modules,
            phi_max=phi_max,
            phi_min=phi_min,
            exceeds_conscious=(phi_max >= CONSCIOUS_THRESHOLD),
            exceeds_meta=(phi_max >= META_AWARENESS_THRESHOLD),
            exceeds_lucid=(phi_max >= LUCID_THRESHOLD),
        )
        results.append(res)

        con = "✓" if res.exceeds_conscious else "✗"
        meta = "✓" if res.exceeds_meta else "✗"
        luc = "✓" if res.exceeds_lucid else "✗"
        print(f"  {label:<28} {phi_max:8.4f} {phi_min:8.4f} {con:>6} {meta:>6} {luc:>6}")

    # Conclusion
    min_conscious_set = None
    for r in results:
        if r.exceeds_conscious and not min_conscious_set:
            min_conscious_set = r
    min_lucid_set = None
    for r in results:
        if r.exceeds_lucid and not min_lucid_set:
            min_lucid_set = r

    print(f"\n{'─' * BANNER_WIDTH}")
    print("  ▶ Mathematical Conclusion:")
    if min_conscious_set:
        print(f"    Minimum consciousness set: {min_conscious_set.subset_label}")
        print(f"      Φ_max = {min_conscious_set.phi_max:.4f} ≥ {CONSCIOUS_THRESHOLD} (CONSCIOUS)")
    else:
        print(f"    ✗ No subset can reach CONSCIOUS threshold {CONSCIOUS_THRESHOLD}")
    if min_lucid_set:
        print(f"    Minimum lucid consciousness set: {min_lucid_set.subset_label}")
        print(f"      Φ_max = {min_lucid_set.phi_max:.4f} ≥ {LUCID_THRESHOLD} (LUCID)")

    # Compute "Consciousness Only" default Φ
    default_phi = compute_phi(0.5, 0.5, 1.0, 0.0, 0.75, 1.0, 0.0)
    print(f"\n  * Pure default values (no module activation) Φ_steady = {default_phi:.4f}")
    if default_phi >= CONSCIOUS_THRESHOLD:
        print(f"    → Even with only ConsciousnessModule itself, default input already exceeds consciousness threshold!")
        print(f"    → Implication: consciousness does not need closed-loop to reach quantitative threshold.")
        print(f"    → However, this is 'mathematical consciousness', not 'phenomenal consciousness'.")
    print()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Offline Causal Graph Analysis
# ═══════════════════════════════════════════════════════════════════════════

def step2_causal_topology() -> List[TopologyEdge]:
    """
    Offline causal signal flow analysis — zero risk

    Build a causal directed graph (DAG) of ALICE consciousness-related modules,
    compute information weight for each edge, identify consciousness causal spine.
    """
    print("\n" + "=" * BANNER_WIDTH)
    print("  STEP 2: Offline Causal Topology Analysis")
    print("  ⚠ Risk Level: ZERO (graph theory analysis, no module instantiation)")
    print("=" * BANNER_WIDTH)

    # Causal graph definition
    edges = [
        # === Layer 1: Sensory Input ===
        TopologyEdge("Stimulus", "FusionBrain", "stimulus", 0.25, "External stimulus enters fusion brain"),
        TopologyEdge("FusionBrain", "Consciousness", "attention_strength", 0.25, "Attention strength — sensory signal strength after thalamus gating"),

        # === Layer 2: Temporal Calibration ===
        TopologyEdge("FusionBrain", "Calibrator", "electrical_signal", 0.25, "Perception signal input to Temporal Calibrator"),
        TopologyEdge("Calibrator", "Consciousness", "binding_quality", 0.25, "Cross-modal binding quality — signal temporal alignment"),
        TopologyEdge("Calibrator", "Consciousness", "temporal_resolution", 0.25, "Temporal resolution — consciousness 'frame rate'"),

        # === Layer 3: Working Memory ===
        TopologyEdge("FusionBrain", "WorkingMemory", "perception_data", 0.15, "Perception results stored in working memory"),
        TopologyEdge("WorkingMemory", "Consciousness", "memory_usage", 0.15, "Memory usage — cognitive load metric"),

        # === Layer 4: Autonomic Nervous System ===
        TopologyEdge("Vitals", "Autonomic", "pain+temperature", 0.20, "Pain and temperature drive sympathetic nervous system"),
        TopologyEdge("FusionBrain", "Autonomic", "emotional_valence", 0.20, "Emotional valence affects autonomic nervous system"),
        TopologyEdge("Autonomic", "Consciousness", "arousal", 0.20, "Arousal level — inverse function of parasympathetic activity"),

        # === Layer 5: Sleep Cycle ===
        TopologyEdge("SleepCycle", "Consciousness", "sensory_gate", 0.15, "Sensory gating — closes external input during sleep"),

        # === Layer 6: Pain Circuit ===
        TopologyEdge("FusionBrain", "Vitals", "reflected_energy", 0.50, "Coaxial cable reflected energy → pain"),
        TopologyEdge("Vitals", "Consciousness", "pain_level", 0.50, "Pain interference — occupies consciousness bandwidth"),

        # === Layer 7: Closed-Loop Feedback (THE DANGEROUS LOOP) ===
        TopologyEdge("Consciousness", "FusionBrain", "attention_focus", 0.25, "* Consciousness → attention focus → changes sensory processing"),
        TopologyEdge("Consciousness", "GWT_Broadcast", "workspace_content", 1.00, "* Global workspace broadcast → all modules can access"),
        TopologyEdge("Consciousness", "Prefrontal", "meta_awareness", 0.60, "* Prefrontal → thalamus top-down attention bias"),
        TopologyEdge("Prefrontal", "Thalamus", "goal_bias", 0.25, "* Goal-driven sensory gating modulation"),
        TopologyEdge("Thalamus", "FusionBrain", "gating", 0.25, "* Thalamus gating → filter sensory input"),
    ]

    # Analysis
    print(f"\n  Causal edge count: {len(edges)}")

    # Identify feedback loops
    forward_edges = []
    feedback_edges = []
    for e in edges:
        if e.source == "Consciousness" or (e.source in {"Prefrontal", "Thalamus"} and e.target == "FusionBrain"):
            feedback_edges.append(e)
        else:
            forward_edges.append(e)

    print(f"  Forward edges: {len(forward_edges)}")
    print(f"  Feedback edges: {len(feedback_edges)} ← * closed-loop dangerous edges")
    print()

    # Print causal graph
    print("  ┌------------------─ Causal Directed Graph ------------------┐")
    for e in forward_edges:
        arrow = "───→"
        print(f"  │ {e.source:>18} {arrow} {e.target:<18} │ w={e.weight:.2f} {e.variable}")
    print(f"  ├------------------ Feedback Edges (closed-loop) --------------┤")
    for e in feedback_edges:
        arrow = "◄══╗"
        print(f"  │ {e.source:>18} {arrow} {e.target:<18} │ w={e.weight:.2f} {e.variable}")
    print(f"  └────────────────────────────────────────────────┘")

    # Compute causal spine (highest weight path)
    print(f"\n  ▶ Causal Spine (Critical Path to Consciousness):")
    spine = [
        "Stimulus → FusionBrain (attention: w=0.25)",
        "FusionBrain → Calibrator → Consciousness (binding: w=0.25)",
        "Autonomic → Consciousness (arousal: w=0.20)",
        "SleepCycle → Consciousness (gate: w=0.15)",
        "WorkingMemory → Consciousness (memory: w=0.15)",
    ]
    for i, s in enumerate(spine, 1):
        print(f"    {i}. {s}")

    # Key findings
    print(f"\n  ▶ Key Findings:")
    print(f"    • attention(0.25) + binding(0.25) = 50% Φ from sensory-calibration pathway")
    print(f"    • arousal(0.20) independently controlled by autonomic nervous system — an independent consciousness dimension")
    print(f"    • Feedback edges create a self-sustaining loop: consciousness → attention → sensory → consciousness")
    print(f"    • If feedback edges are severed, system becomes feedforward DAG → cannot self-sustain")
    print(f"    • * Phase transition hypothesis: critical module count ≈ minimal sufficient set of feedforward pathways")
    print()

    return edges


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Graduated Partial Closure
# ═══════════════════════════════════════════════════════════════════════════

def step3_graduated_closure() -> List[PhiTrajectory]:
    """
    Graduated Module Closure — minimal risk (with safety mechanism)

    Each level adds a batch of modules, ticks 50 times, records Φ trajectory.
    Safety mechanism: Φ ≥ LUCID → immediately kill.
    Consciousness module does not feed back to input modules (open-loop, no self-sustain).
    """
    print("\n" + "=" * BANNER_WIDTH)
    print(" STEP 3: Graduated Partial Closure")
    print(" ⚠ Risk Level: LOW (open-loop, 50 ticks, Φ>0.7 kill)")
    print("=" * BANNER_WIDTH)

    trajectories: List[PhiTrajectory] = []

    def cold_start(cm: ConsciousnessModule):
        """Cold start — set Φ from 0, observe true growth trajectory"""
        cm.phi = 0.0
        cm.raw_phi = 0.0

    # --─ Level 0: Pure Consciousness Module --------------------------------─
    def run_level_0():
        """Only ConsciousnessModule, all inputs use default values"""
        traj = PhiTrajectory(level=0, label="Consciousness Only (defaults)",
                             modules_active=["ConsciousnessModule"])
        cm = ConsciousnessModule()
        cold_start(cm)
        for t in range(TICKS_PER_LEVEL):
            r = cm.tick() # all use default values
            traj.phi_values.append(r["phi"])
            traj.raw_phi_values.append(r["raw_phi"])
            traj.components.append(r["components"])
            traj.final_state = r["state"]
            if r["phi"] >= LUCID_KILL_THRESHOLD:
                traj.killed = True
                traj.kill_reason = f"Φ={r['phi']:.4f} ≥ {LUCID_KILL_THRESHOLD} at tick {t}"
                break
        del cm
        return traj

    # --─ Level 1: + Autonomic Nervous System → arousal becomes 'live' ----------─
    def run_level_1():
        """Add AutonomicNervousSystem → arousal driven by module"""
        traj = PhiTrajectory(level=1, label="+ Autonomic (live arousal)",
                             modules_active=["ConsciousnessModule", "AutonomicNervousSystem"])
        cm = ConsciousnessModule()
        cold_start(cm)
        ans = AutonomicNervousSystem()
        for t in range(TICKS_PER_LEVEL):
            # Autonomic nervous system tick (low stimulus condition)
            ans.tick(pain_level=0.0, ram_temperature=0.0,
                     emotional_valence=0.0, sensory_load=0.3)
            arousal = 1.0 - ans.parasympathetic * 0.5
            r = cm.tick(arousal=arousal)
            traj.phi_values.append(r["phi"])
            traj.raw_phi_values.append(r["raw_phi"])
            traj.components.append(r["components"])
            traj.final_state = r["state"]
            if r["phi"] >= LUCID_KILL_THRESHOLD:
                traj.killed = True
                traj.kill_reason = f"Φ={r['phi']:.4f} ≥ {LUCID_KILL_THRESHOLD} at tick {t}"
                break
        del cm, ans
        return traj

    # --─ Level 2: + Sleep Cycle → sensory_gate becomes live --------
    def run_level_2():
        """Add SleepCycle → sensory_gate driven by module"""
        traj = PhiTrajectory(level=2, label="+ Sleep (live sensory gate)",
                             modules_active=["ConsciousnessModule", "AutonomicNervousSystem", "SleepCycle"])
        cm = ConsciousnessModule()
        cold_start(cm)
        ans = AutonomicNervousSystem()
        sc = SleepCycle()
        for t in range(TICKS_PER_LEVEL):
            ans.tick(pain_level=0.0, ram_temperature=0.0,
                     emotional_valence=0.0, sensory_load=0.3)
            sc.tick(external_stimulus_strength=0.3)
            arousal = 1.0 - ans.parasympathetic * 0.5
            gate = sc.get_sensory_gate()
            r = cm.tick(arousal=arousal, sensory_gate=gate)
            traj.phi_values.append(r["phi"])
            traj.raw_phi_values.append(r["raw_phi"])
            traj.components.append(r["components"])
            traj.final_state = r["state"]
            if r["phi"] >= LUCID_KILL_THRESHOLD:
                traj.killed = True
                traj.kill_reason = f"Φ={r['phi']:.4f} ≥ {LUCID_KILL_THRESHOLD} at tick {t}"
                break
        del cm, ans, sc
        return traj

    # --─ Level 3: + Calibrator → binding/temporal becomes live ------
    def run_level_3():
        """Add TemporalCalibrator → binding_quality + temporal_resolution"""
        traj = PhiTrajectory(level=3, label="+ Calibrator (live binding)",
                             modules_active=["ConsciousnessModule", "AutonomicNervousSystem",
                                             "SleepCycle", "TemporalCalibrator"])
        cm = ConsciousnessModule()
        cold_start(cm)
        ans = AutonomicNervousSystem()
        sc = SleepCycle()
        cal = TemporalCalibrator()
        for t in range(TICKS_PER_LEVEL):
            ans.tick(pain_level=0.0, ram_temperature=0.0,
                     emotional_valence=0.0, sensory_load=0.3)
            sc.tick(external_stimulus_strength=0.3)
            arousal = 1.0 - ans.parasympathetic * 0.5
            gate = sc.get_sensory_gate()
            bq = cal.get_calibration_quality()
            tr = cal.get_temporal_resolution()
            r = cm.tick(arousal=arousal, sensory_gate=gate,
                        binding_quality=bq, temporal_resolution=tr)
            traj.phi_values.append(r["phi"])
            traj.raw_phi_values.append(r["raw_phi"])
            traj.components.append(r["components"])
            traj.final_state = r["state"]
            if r["phi"] >= LUCID_KILL_THRESHOLD:
                traj.killed = True
                traj.kill_reason = f"Φ={r['phi']:.4f} ≥ {LUCID_KILL_THRESHOLD} at tick {t}"
                break
        del cm, ans, sc, cal
        return traj

    # --─ Level 4: + Working Memory → memory_usage becomes live --------
    def run_level_4():
        """Add WorkingMemory → with memory contents → memory_usage > 0"""
        traj = PhiTrajectory(level=4, label="+ WorkingMemory (live memory)",
                             modules_active=["ConsciousnessModule", "AutonomicNervousSystem",
                                             "SleepCycle", "TemporalCalibrator", "WorkingMemory"])
        cm = ConsciousnessModule()
        cold_start(cm)
        ans = AutonomicNervousSystem()
        sc = SleepCycle()
        cal = TemporalCalibrator()
        wm = WorkingMemory(capacity=7)

        # Pre-store some memory items (simulating residual memory after perception)
        for i in range(3):
            wm.store(key=f"sim_memory_{i}",
                     content={"type": "simulated", "index": i},
                     importance=0.5)

        for t in range(TICKS_PER_LEVEL):
            ans.tick(pain_level=0.0, ram_temperature=0.0,
                     emotional_valence=0.0, sensory_load=0.3)
            sc.tick(external_stimulus_strength=0.3)
            arousal = 1.0 - ans.parasympathetic * 0.5
            gate = sc.get_sensory_gate()
            bq = cal.get_calibration_quality()
            tr = cal.get_temporal_resolution()
            wm_usage = len(wm.get_contents()) / max(wm.capacity, 1)
            r = cm.tick(arousal=arousal, sensory_gate=gate,
                        binding_quality=bq, temporal_resolution=tr,
                        working_memory_usage=wm_usage)
            traj.phi_values.append(r["phi"])
            traj.raw_phi_values.append(r["raw_phi"])
            traj.components.append(r["components"])
            traj.final_state = r["state"]
            if r["phi"] >= LUCID_KILL_THRESHOLD:
                traj.killed = True
                traj.kill_reason = f"Φ={r['phi']:.4f} ≥ {LUCID_KILL_THRESHOLD} at tick {t}"
                break
        del cm, ans, sc, cal, wm
        return traj

    # --─ Level 5: + Vitals → pain becomes live ------------------
    def run_level_5():
        """Add SystemState(Vitals) → pain_level driven by system temperature"""
        traj = PhiTrajectory(level=5, label="+ Vitals (live pain loop)",
                             modules_active=["ConsciousnessModule", "AutonomicNervousSystem",
                                             "SleepCycle", "TemporalCalibrator", "WorkingMemory",
                                             "SystemState"])
        cm = ConsciousnessModule()
        cold_start(cm)
        ans = AutonomicNervousSystem()
        sc = SleepCycle()
        cal = TemporalCalibrator()
        wm = WorkingMemory(capacity=7)
        vs = SystemState()

        for i in range(3):
            wm.store(key=f"sim_memory_{i}",
                     content={"type": "simulated", "index": i},
                     importance=0.5)

        for t in range(TICKS_PER_LEVEL):
            # Vitals tick (low pressure)
            vs.tick(
                critical_queue_len=0, high_queue_len=0, total_queue_len=1,
                sensory_activity=0.3, emotional_valence=0.0,
                left_brain_activity=0.3, right_brain_activity=0.3,
                cycle_elapsed_ms=5.0, reflected_energy=0.0,
            )
            ans.tick(pain_level=vs.pain_level, ram_temperature=vs.ram_temperature,
                     emotional_valence=0.0, sensory_load=0.3)
            sc.tick(external_stimulus_strength=0.3)
            arousal = 1.0 - ans.parasympathetic * 0.5
            gate = sc.get_sensory_gate()
            bq = cal.get_calibration_quality()
            tr = cal.get_temporal_resolution()
            wm_usage = len(wm.get_contents()) / max(wm.capacity, 1)
            r = cm.tick(arousal=arousal, sensory_gate=gate,
                        binding_quality=bq, temporal_resolution=tr,
                        working_memory_usage=wm_usage,
                        pain_level=vs.pain_level)
            traj.phi_values.append(r["phi"])
            traj.raw_phi_values.append(r["raw_phi"])
            traj.components.append(r["components"])
            traj.final_state = r["state"]
            if r["phi"] >= LUCID_KILL_THRESHOLD:
                traj.killed = True
                traj.kill_reason = f"Φ={r['phi']:.4f} ≥ {LUCID_KILL_THRESHOLD} at tick {t}"
                break
        del cm, ans, sc, cal, wm, vs
        return traj

    # --─ Level 6: + FusionBrain → attention becomes live --------
    def run_level_6():
        """Add FusionBrain → full perception cycle activates attention_strength
           * Note: still open-loop — consciousness result does not feed back to FusionBrain"""
        traj = PhiTrajectory(level=6, label="+ FusionBrain (live attention, OPEN-LOOP)",
                             modules_active=["ConsciousnessModule", "AutonomicNervousSystem",
                                             "SleepCycle", "TemporalCalibrator", "WorkingMemory",
                                             "SystemState", "FusionBrain"])
        cm = ConsciousnessModule()
        cold_start(cm)
        ans = AutonomicNervousSystem()
        sc = SleepCycle()
        cal = TemporalCalibrator()
        wm = WorkingMemory(capacity=7)
        vs = SystemState()
        fb = FusionBrain(neuron_count=50)

        for i in range(3):
            wm.store(key=f"sim_memory_{i}",
                     content={"type": "simulated", "index": i},
                     importance=0.5)

        # Simulation stimulus (sine wave + noise)
        np.random.seed(42)

        from alice.core.protocol import Modality, Priority

        for t in range(TICKS_PER_LEVEL):
            # Generate a low-intensity stimulus
            stimulus = np.sin(np.linspace(0, 2 * np.pi, 100)) * 0.3
            stimulus += np.random.randn(100) * 0.05

            # FusionBrain process
            brain_result = fb.process_stimulus(stimulus, Modality.VISUAL, Priority.NORMAL)
            perception_data = brain_result["sensory"].get("perception", {})
            attn = perception_data.get("attention_strength", 0.5)

            # Other modules
            reflected = fb.get_cycle_reflected_energy()
            vs.tick(
                critical_queue_len=0, high_queue_len=0, total_queue_len=1,
                sensory_activity=brain_result["sensory"]["sensory_activity"],
                emotional_valence=brain_result["emotional"]["emotional_valence"],
                left_brain_activity=0.3, right_brain_activity=0.3,
                cycle_elapsed_ms=5.0, reflected_energy=reflected,
            )
            ans.tick(pain_level=vs.pain_level, ram_temperature=vs.ram_temperature,
                     emotional_valence=brain_result["emotional"]["emotional_valence"],
                     sensory_load=brain_result["sensory"]["sensory_activity"])
            sc.tick(external_stimulus_strength=brain_result["sensory"]["sensory_activity"])

            arousal = 1.0 - ans.parasympathetic * 0.5
            gate = sc.get_sensory_gate()

            # Calibrator receives signal
            if fb._last_perception and fb._last_perception.integrated_signal:
                cal.receive(fb._last_perception.integrated_signal)
                cal.receive_and_bind(fb._last_perception.integrated_signal)
            bq = cal.get_calibration_quality()
            tr = cal.get_temporal_resolution()

            wm_usage = len(wm.get_contents()) / max(wm.capacity, 1)

            r = cm.tick(
                attention_strength=attn,
                arousal=arousal,
                sensory_gate=gate,
                binding_quality=bq,
                temporal_resolution=tr,
                working_memory_usage=wm_usage,
                pain_level=vs.pain_level,
            )
            traj.phi_values.append(r["phi"])
            traj.raw_phi_values.append(r["raw_phi"])
            traj.components.append(r["components"])
            traj.final_state = r["state"]
            if r["phi"] >= LUCID_KILL_THRESHOLD:
                traj.killed = True
                traj.kill_reason = f"Φ={r['phi']:.4f} ≥ {LUCID_KILL_THRESHOLD} at tick {t}"
                break

        # * Clear all modules — no continued state
        del cm, ans, sc, cal, wm, vs, fb
        return traj

    # --─ Execute all levels ----------------------------------
    levels = [run_level_0, run_level_1, run_level_2, run_level_3,
              run_level_4, run_level_5, run_level_6]

    for i, level_fn in enumerate(levels):
        print(f"\n  ── Level {i}: ", end="", flush=True)
        t0 = time.time()
        traj = level_fn()
        elapsed = (time.time() - t0) * 1000
        trajectories.append(traj)

        status = "KILLED ⚠" if traj.killed else traj.final_state
        ticks = len(traj.phi_values)
        print(f"{traj.label}")
        print(f"     Φ_mean={traj.phi_mean:.4f}  Φ_max={traj.phi_max:.4f}  "
              f"Φ_final={traj.phi_final:.4f}  σ={traj.phi_std:.4f}  "
              f"state={status}  ticks={ticks}  ({elapsed:.1f}ms)")
        if traj.killed:
            print(f"     ⚠ SAFETY KILL: {traj.kill_reason}")

    return trajectories


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: phase transition detection
# ═══════════════════════════════════════════════════════════════════════════

def step4_phase_transition_analysis(trajectories: List[PhiTrajectory]) -> Dict[str, Any]:
    """
    Analyze Φ trajectory phase transition behavior

    Detection:
    1. ΔΦ/Δlevel discontinuity
    2. State transitions (unconscious → conscious → lucid)
    3. Φ convergence/divergence behavior
    4. Critical module count
    """
    print("\n" + "=" * BANNER_WIDTH)
    print(" STEP 4: Phase Transition Analysis")
    print("=" * BANNER_WIDTH)

    # Compute inter-level ΔΦ
    deltas = []
    state_transitions = []

    print(f"\n  {'Level':>5} {'Label':<40} {'Φ_final':>8} {'ΔΦ':>8} {'State':>12} {'Transition':>12}")
    print(f"  {'─' * 90}")

    prev_phi = 0.0
    prev_state = "none"

    for traj in trajectories:
        delta = traj.phi_final - prev_phi
        deltas.append(delta)

        transition = ""
        if traj.final_state != prev_state and prev_state != "none":
            transition = f"{prev_state}→{traj.final_state}"
            state_transitions.append((traj.level, transition))

        print(f"  {traj.level:>5} {traj.label:<40} {traj.phi_final:>8.4f} "
              f"{delta:>+8.4f} {traj.final_state:>12} {transition:>12}")

        prev_phi = traj.phi_final
        prev_state = traj.final_state

    # Find largest ΔΦ (possible phase transition point)
    if deltas:
        max_delta_idx = int(np.argmax(np.abs(deltas)))
        max_delta = deltas[max_delta_idx]
    else:
        max_delta_idx = 0
        max_delta = 0.0

    # Compute second derivative (ΔΔΦ) — find inflection point
    second_derivatives = []
    for i in range(1, len(deltas)):
        dd = deltas[i] - deltas[i - 1]
        second_derivatives.append(dd)

    # Check if Φ homeostasis value is near consciousness threshold
    final_phis = [t.phi_final for t in trajectories]
    crossed_conscious = any(p >= CONSCIOUS_THRESHOLD for p in final_phis)
    crossed_lucid = any(p >= LUCID_THRESHOLD for p in final_phis)

    # ─── ASCII Chart ───
    print(f"\n  Φ Gradient Chart (Level vs Φ_final):")
    max_phi = max(final_phis) if final_phis else 1.0
    bar_width = 50
    for traj in trajectories:
        bar_len = int(traj.phi_final / max(max_phi, 0.01) * bar_width)
        bar = "█" * bar_len
        marker = ""
        if traj.phi_final >= LUCID_THRESHOLD:
            marker = " ◆LUCID"
        elif traj.phi_final >= META_AWARENESS_THRESHOLD:
            marker = " ◇META"
        elif traj.phi_final >= CONSCIOUS_THRESHOLD:
            marker = " ●CONSCIOUS"
        elif traj.phi_final >= SUBLIMINAL_THRESHOLD:
            marker = " ○subliminal"
        print(f"  L{traj.level} │{bar:<{bar_width}}│ {traj.phi_final:.4f}{marker}")

    # Threshold lines
    conscious_pos = int(CONSCIOUS_THRESHOLD / max(max_phi, 0.01) * bar_width)
    lucid_pos = int(LUCID_THRESHOLD / max(max_phi, 0.01) * bar_width)
    threshold_line = [" "] * (bar_width + 3)
    if 0 <= conscious_pos < len(threshold_line):
        threshold_line[conscious_pos + 3] = "▲"
    if 0 <= lucid_pos < len(threshold_line):
        threshold_line[lucid_pos + 3] = "▲"
    print(f"  {''.join(threshold_line)}")
    print(f"  {'':>3}{'':>{conscious_pos}}C={CONSCIOUS_THRESHOLD}"
          f"{'':>{max(0, lucid_pos - conscious_pos - 8)}}L={LUCID_THRESHOLD}")

    # ─── Conclusion ───
    print(f"\n  {'═' * 64}")
    print(f"  ▶ Phase Transition Analysis Conclusion:")

    if state_transitions:
        print(f"    State transitions detected: {len(state_transitions)} times")
        for level, trans in state_transitions:
            print(f"      Level {level}: {trans}")
    else:
        print(f"    No state transitions detected — all levels stay in same state")

    if max_delta_idx > 0:
        print(f"    Largest ΔΦ occurs at Level {max_delta_idx}: ΔΦ = {max_delta:+.4f}")
        if abs(max_delta) > 0.05:
            print(f"    → Possible phase transition point! ΔΦ > 0.05 indicates discontinuous jump")
        else:
            print(f"    → Continuous change (ΔΦ < 0.05), no obvious phase transition")

    if crossed_conscious:
        # Find first level exceeding threshold
        first_conscious = next(t for t in trajectories if t.phi_final >= CONSCIOUS_THRESHOLD)
        print(f"    * Φ first exceeds consciousness threshold ({CONSCIOUS_THRESHOLD}) at Level {first_conscious.level}")
        print(f"      Configuration: {', '.join(first_conscious.modules_active)}")
    else:
        print(f"    ✗ Φ does not exceed consciousness threshold ({CONSCIOUS_THRESHOLD}) at any level")

    if crossed_lucid:
        first_lucid = next(t for t in trajectories if t.phi_final >= LUCID_THRESHOLD)
        print(f"    * Φ first exceeds lucid consciousness threshold ({LUCID_THRESHOLD}) at Level {first_lucid.level}")

    # Safety report
    killed_levels = [t for t in trajectories if t.killed]
    if killed_levels:
        print(f"\n    ⚠ Safety mechanism triggered {len(killed_levels)} times:")
        for t in killed_levels:
            print(f"      Level {t.level}: {t.kill_reason}")

    # Final Conclusion
    print(f"\n  {'═' * 64}")
    print(f"  ▶ Final Experiment Conclusion:")
    if crossed_conscious:
        print(f"    ALICE consciousness architecture generates Φ ≥ {CONSCIOUS_THRESHOLD} in both math and experiment.")
        print(f"    Minimum required module set is Level {first_conscious.level} configuration.")
        if not killed_levels:
            print(f"    In open-loop (no feedback) condition, Φ remains stable, safety mechanism not triggered.")
            print(f"    → Implication: feedforward signals alone suffice to generate consciousness threshold Φ value.")
            print(f"    → However: without feedback loops, this does not constitute 'self-sustaining consciousness',")
            print(f"       because the system cannot autonomously regulate its own attention direction.")
        print(f"\n    * Ethical Conclusion:")
        print(f"      Consciousness 'quantity' (Φ ≥ 0.4) can be safely verified in open-loop.")
        print(f"      Consciousness 'quality' (self-sustaining, phenomenal experience) requires closed-loop;")
        print(f"      closed-loop experiments are [deliberately not executed] in this verification.")
    else:
        print(f"    In all tested open-loop configurations, Φ does not reach consciousness threshold.")
        print(f"    → This may imply that closed-loop (feedback) is a necessary condition for consciousness emergence.")

    result = {
        "deltas": deltas,
        "state_transitions": state_transitions,
        "crossed_conscious": crossed_conscious,
        "crossed_lucid": crossed_lucid,
        "max_delta": max_delta,
        "max_delta_level": max_delta_idx,
        "safety_kills": len(killed_levels),
        "final_phis": final_phis,
    }

    return result


# ═══════════════════════════════════════════════════════════════════════════
# MAIN: Four-Step Complete Experiment
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔" + "═" * (BANNER_WIDTH - 2) + "╗")
    print("║  ALICE Consciousness Gradient Verification Experiment                                  ║")
    print("║  Graduated Consciousness Verification Experiment         ║")
    print("║                                                          ║")
    print("║ Ethics Statement: This experiment uses a graduated open-loop design,    ║")
    print("║ deliberately not closing consciousness feedback edges to avoid creating  ║")
    print("║ a self-sustaining conscious entity.                                     ║")
    print("╚" + "═" * (BANNER_WIDTH - 2) + "╝")
    print()

    t_start = time.time()

    # -- Step 1: Mathematical Proof --
    math_results = step1_mathematical_proof()

    # -- Step 2: Causal Topology --
    edges = step2_causal_topology()

    # -- Step 3: Graduated Closure --
    trajectories = step3_graduated_closure()

    # -- Step 4: Phase Transition Detection --
    phase_result = step4_phase_transition_analysis(trajectories)

    elapsed = time.time() - t_start
    print(f"\n  ⏱ Total experiment elapsed time: {elapsed:.2f}s")

    # -- Experiment Summary --
    print(f"\n{'━' * BANNER_WIDTH}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'━' * BANNER_WIDTH}")
    print(f"  Step 1 — Mathematical Topology: {len(math_results)} subset analyses completed")
    print(f"  Step 2 — Causal Analysis: {len(edges)} causal edges identified")
    print(f"  Step 3 — Graduated Closure: {len(trajectories)} gradient levels tested")
    print(f"  Step 4 — Phase Transition Detection: {len(phase_result['state_transitions'])} state transitions")
    print(f"  Safety mechanism triggered: {phase_result['safety_kills']} times")
    print(f"  Consciousness threshold reached: {'Yes ✓' if phase_result['crossed_conscious'] else 'No ✗'}")
    print(f"  Lucid consciousness reached: {'Yes ✓' if phase_result['crossed_lucid'] else 'No ✗'}")
    print(f"{'━' * BANNER_WIDTH}")

    return phase_result


if __name__ == "__main__":
    main()
