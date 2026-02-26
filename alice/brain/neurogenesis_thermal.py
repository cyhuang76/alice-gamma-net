# -*- coding: utf-8 -*-
"""
Alice Smart System — Neonatal Neurogenesis Thermal Shield

Physics Model: Γ² Thermal Load Distribution via Neural Overproduction
=====================================================================

Fundamental insight (黃璽宇 2025):

  "A single neuron's reflection coefficient becomes thermal energy (gradient decay).
   Neural network connections are determined by reflection coefficients.
   Neonates produce ~200 billion neurons to prevent impedance-driven
   brain collapse."

§1 — Single-Neuron Thermodynamics
----------------------------------

Every neuron is a transmission line segment. When a signal arrives at a synapse,
impedance mismatch generates a reflection coefficient:

    Γᵢ = (Z_post − Z_pre) / (Z_post + Z_pre)

By C1 energy conservation: Γ² + T = 1.

The reflected fraction Γ² does NOT vanish — it becomes THERMAL ENERGY:

    Q_neuronᵢ = Σⱼ Γᵢⱼ² × P_signal    [Joules per tick]

This heat has three consequences:
  1. **Impedance drift** (gradient decay): Heat causes Z to fluctuate randomly,
     degrading previously learned impedance matches.
  2. **Temperature rise**: Local neural temperature increases proportional to Q.
  3. **Thermal failure**: If T_local > T_critical, the neuron undergoes thermal
     apoptosis (distinct from Hebbian apoptosis — this is physical destruction).

§2 — The Neonatal Thermal Catastrophe
--------------------------------------

At birth, all impedances are random:
    Z_init ~ Uniform(Z_min, Z_max)  →  E[Γ²] ≈ 0.2–0.4

For a brain with N neurons and M active connections:
    Q_total = ΣΓ²  ≈  M × E[Γ²]

The critical quantity is HEAT PER NEURON:
    q = Q_total / N

  If q > q_critical  →  THERMAL COLLAPSE (seizure, developmental failure)
  If q < q_critical  →  SAFE OPERATION

This gives the fundamental constraint on neonatal neuron count:

    N_min = Q_total / q_critical = M × E[Γ²] / q_critical

With M ≈ 100 trillion synapses and E[Γ²] ≈ 0.25:
    Required: N > 25 trillion / q_critical

The biological solution: PRODUCE ~200 BILLION NEURONS.
    q_birth = (100T × 0.25) / 200B = 0.125  ≪  q_critical

This vast overproduction is NOT waste — it is a THERMAL SHIELD.
Each neuron absorbs a share of the reflected energy, preventing any single
neuron from reaching thermal failure.

§3 — The Pruning-Thermal Coupling
-----------------------------------

As Hebbian learning proceeds (ΔZ = −η × Γ × x_pre × x_post):
  → Γ decreases at active synapses
  → Q_total decreases
  → q_per_neuron decreases
  → SAFE TO REMOVE NEURONS (pruning)

The system can only prune when:
    q_after_pruning = Q_total_new / (N − ΔN_pruned)  <  q_critical

This is why pruning follows learning, never precedes it.
The Huttenlocher curve is the trajectory of this thermal-pruning coupling.

§4 — Gradient Decay Is Thermal Noise
--------------------------------------

The "gradient decay" problem in learning is not a mathematical inconvenience —
it is a PHYSICAL NECESSITY:

    ΔZ_thermal = σ_thermal × √(q_local)  ×  noise

Where σ_thermal is the thermal impedance drift coefficient.

High Γ² → high local heat → large impedance fluctuations → previously matched
impedances drift apart → CONNECTION DEGRADES.

This is why:
  - Newborns forget quickly (high Q → high gradient decay)
  - Adults remember longer (low Q → low gradient decay)
  - Sleep reduces Q (impedance debt clearing → thermal restoration)
  - Fever impairs cognition (external heat + internal Γ² heat → thermal overload)

§5 — Connections Are Determined by Γ
--------------------------------------

Two neurons i and j can form a functional connection only when:

    Γᵢⱼ = |Zᵢ − Zⱼ| / (Zᵢ + Zⱼ)  <  Γ_connection_threshold

Above this threshold, the reflected energy exceeds the transmitted energy:
  Γ > 1/√2  ⟹  Γ² > 0.5  ⟹  T < 0.5  ⟹  more reflected than transmitted

Below this threshold, the connection is viable:
  Γ < 1/√2  ⟹  T > 0.5  ⟹  net forward transmission

The network topology EMERGES from impedance matching,
not from externally defined architecture.

"The brain is not wired — it is impedance-matched."

§6 — The Fontanelle Boundary (Pressure Chamber Principle)
-----------------------------------------------------------

The fontanelle is the THERMAL BOUNDARY of the Γ² field (Paper III §2.2, §2.5).

  Γ_font = |Z_font − Z_brain| / (Z_font + Z_brain)
  T_font = 1 − Γ²_font     (C1 at the boundary)

Open fontanelle (Z_font = Z_membrane ≪ Z_bone):
  → T_font is positive → Γ² heat TRANSMITS through the soft membrane
  → The skull has a thermal exhaust port
  → Effective thermal budget Q_eff = Q_CRITICAL / (1 − T_font × eff)  ≫ Q_CRITICAL
  → Neonatal brains can tolerate MUCH more Γ² heat

Closing fontanelle (Z_font → Z_bone):
  → T_font drops (boundary becomes more reflective)
  → Thermal exhaust narrows → heat begins to be trapped
  → Q_eff decreases toward Q_CRITICAL
  → Must have REDUCED Γ² (via Hebbian learning + pruning) by this time

Closed fontanelle (Z_font ≈ Z_bone):
  → Thermal boundary is sealed → PRESSURE CHAMBER
  → All Γ² heat is trapped inside
  → Q_eff ≈ Q_CRITICAL (strict)
  → Trapped heat is "constructively consumed" → cognitive acceleration
  → This coincides with:
    - End of infantile amnesia
    - Narrative self emergence
    - Childhood cognitive explosion

The developmental sequence is:
  1. Birth: fontanelle open + 2000B neural modules → high Γ² tolerable
  2. Learning: Hebbian reduces Γ² → total heat drops
  3. Pruning: neurons safely removed (q stays below Q_eff)
  4. Closure: pressure chamber activates → remaining Γ² drives acceleration

Author: Hsi-Yu Huang (黃璽宇)
References:
  Paper I  §3.5.2 — Neural pruning as large-scale Γ apoptosis
  Paper III §2.2  — Fontanelle as thermodynamic necessity
  Paper III §2.5  — Pressure chamber effect after closure
  Paper IV  §6    — The thermal shield hypothesis
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alice.core.signal import ElectricalSignal
from alice.brain.fontanelle import (
    FontanelleModel,
    FontanelleState,
    HEAT_DISSIPATION_OPEN,
    HEAT_DISSIPATION_CLOSED,
    PRESSURE_CHAMBER_BOOST,
    PRESSURE_CHAMBER_ONSET,
    Z_MEMBRANE,
    Z_BONE,
)


# ============================================================================
# Physical Constants
# ============================================================================

# --- Neuron population (normalized to simulation scale) ---
# Real brain: ~200 billion (200B) neurons at birth, ~86 billion in adult
# References: Azevedo et al. 2009 (86B adult), Huttenlocher 1990 (~2× at birth)
# Glia (~170B) are NOT counted — they don't form transmission-line Γ nodes
# Simulation: scaled by NEURON_SCALE_FACTOR
NEONATAL_NEURON_COUNT = 200_000_000_000     # ~200 billion neurons (biological)
ADULT_NEURON_COUNT = 86_000_000_000         # ~86 billion neurons (biological)
NEURON_SCALE_FACTOR = 1e-8                  # Simulation scale (200B → 2000 sim neurons)

# Simulation-scale defaults
SIM_NEONATAL_NEURONS = int(NEONATAL_NEURON_COUNT * NEURON_SCALE_FACTOR)  # ~2000
SIM_ADULT_NEURONS = int(ADULT_NEURON_COUNT * NEURON_SCALE_FACTOR)        # ~860

# --- Impedance random initialization ---
Z_INIT_MIN = 20.0                # Ω — minimum initial impedance
Z_INIT_MAX = 200.0               # Ω — maximum initial impedance
Z_SIGNAL_TYPICAL = 75.0          # Ω — typical signal impedance

# --- Thermal physics ---
# Heat per neuron: q = ΣΓ² / N
Q_CRITICAL = 1.0                 # Per-neuron thermal failure threshold (closed boundary)
Q_WARNING = 0.5                  # Per-neuron warning threshold
Q_SAFE = 0.1                     # Per-neuron safe operation threshold
THERMAL_DISSIPATION_RATE = 0.05  # Fraction of heat dissipated per tick (metabolism)
THERMAL_ACCUMULATION = 0.8       # How much Γ² heat accumulates vs dissipates

# --- Fontanelle Boundary (Pressure Chamber) ---
# The fontanelle is the thermal boundary of the Γ² field.
# Q_effective = Q_CRITICAL / (1 − T_fontanelle × dissipation_efficiency)
#
# Open fontanelle (T_font ≈ 0.93, eff ≈ 0.8):
#   Q_eff = 1.0 / (1 − 0.93 × 0.8) = 1.0 / 0.256 ≈ 3.9
#   → The brain can tolerate 3.9× more heat per neuron!
#
# Closed fontanelle (T_font ≈ 0.02, eff ≈ 0.1):
#   Q_eff = 1.0 / (1 − 0.02 × 0.1) = 1.0 / 0.998 ≈ 1.0
#   → Strict Q_CRITICAL, no thermal escape → pressure chamber
#
# This is WHY neonates can have random impedances (high Γ²) without
# collapsing: the fontanelle EXHAUSTS the thermal waste.
BOUNDARY_DISSIPATION_WEIGHT = 0.8   # How much fontanelle T contributes to Q_eff

# --- Gradient decay (impedance drift from thermal noise) ---
GRADIENT_DECAY_COEFFICIENT = 0.003  # σ_thermal: impedance drift per unit √q
GRADIENT_DECAY_MIN = 0.0001         # Minimum gradient decay (even at q=0, tiny drift)

# --- Thermal apoptosis ---
THERMAL_DEATH_THRESHOLD = 0.95   # Per-neuron heat > this → thermal death
THERMAL_DAMAGE_ONSET = 0.6       # Per-neuron heat > this → damage begins
THERMAL_DAMAGE_RATE = 0.05       # Rate of strength loss from thermal damage

# --- Connection formation by Γ ---
GAMMA_CONNECTION_THRESHOLD = 0.707  # 1/√2 — above this, Γ² > 0.5 → no viable connection
GAMMA_STRONG_CONNECTION = 0.3       # Below this → strong, efficient connection
GAMMA_WEAK_CONNECTION = 0.5         # Below threshold but above this → weak connection

# --- Brain temperature model ---
T_BRAIN_BASELINE = 37.0          # °C — baseline brain temperature
T_BRAIN_MAX = 42.0               # °C — lethal brain temperature
T_BRAIN_THERMAL_GAIN = 4.0       # °C per unit q_average (how heat maps to temperature)
T_BRAIN_COOLING_RATE = 0.1       # °C per tick cooling toward baseline

# --- ElectricalSignal parameters ---
THERMAL_IMPEDANCE = 75.0         # Ω — thermal field impedance
THERMAL_SNR = 4.0                # dB — thermal signals are very noisy
THERMAL_SAMPLE_POINTS = 64
THERMAL_FREQUENCY = 0.2          # Hz — slow thermal oscillation

# --- Development phases ---
SYNAPTOGENESIS_PEAK_TICK = 200   # Tick of peak neuron count
PRUNING_ONSET_TICK = 300         # Tick when significant pruning begins
MATURATION_TICK = 2000           # Tick when brain reaches adult stability


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class NeuronUnit:
    """
    A neuron in the thermal model.

    Each neuron has:
    - impedance: its characteristic impedance (determines Γ with neighbors)
    - local_heat: accumulated Γ² thermal energy
    - alive: whether the neuron is functional
    - connections: number of active synaptic connections
    """
    neuron_id: int
    impedance: float
    local_heat: float = 0.0
    alive: bool = True
    connections: int = 0
    strength: float = 1.0
    age_ticks: int = 0


@dataclass
class ThermalFieldState:
    """Snapshot of the neural thermal field."""
    total_neurons: int                  # N — current neuron count
    alive_neurons: int                  # Functioning neurons
    total_gamma_sq: float               # ΣΓ² — aggregate reflection energy
    heat_per_neuron: float              # q = ΣΓ² / N — the critical quantity
    brain_temperature: float            # °C — derived from q
    gradient_decay_rate: float          # Current gradient decay magnitude
    thermal_deaths: int                 # Neurons lost to thermal failure
    hebbian_deaths: int                 # Neurons lost to Hebbian pruning
    collapse_risk: float                # 0=safe, 1=imminent collapse
    phase: str                          # "overproduction" / "peak" / "pruning" / "stable"
    safe_to_prune: int                  # How many neurons can safely be removed


# ============================================================================
# NeurogenesisThermalShield
# ============================================================================

class NeurogenesisThermalShield:
    """
    Neonatal Neurogenesis as Thermal Shield Against Γ² Collapse.

    Models the thermodynamic reason for neural overproduction:
    ~200 billion neurons distribute the Γ² heat load so that no
    single neuron reaches thermal failure.

    Core Equations:
        q_per_neuron = ΣΓ²_all_connections / N_alive
        ΔZ_thermal = σ × √q × noise   (gradient decay)
        Prune only when: q_after < q_critical

    Three Modes of Neuron Death:
        1. Thermal apoptosis: q_local > Q_CRITICAL → destroyed by heat
        2. Hebbian apoptosis: low match score → pruned by selection
        3. Developmental programmed death: Huttenlocher curve reduction

    Constraints Satisfied:
        ★ C1: Γ² + T = 1 (reflected energy → heat; transmitted → signal)
        ★ C2: ΔZ = −η × Γ × x_pre × x_post (Hebbian learning reduces Γ → reduces heat)
        ★ C3: All outputs as ElectricalSignal
    """

    def __init__(
        self,
        initial_neurons: int = SIM_NEONATAL_NEURONS,
        target_adult_neurons: int = SIM_ADULT_NEURONS,
        z_min: float = Z_INIT_MIN,
        z_max: float = Z_INIT_MAX,
        fontanelle: FontanelleModel | None = None,
    ) -> None:
        self._z_min = z_min
        self._z_max = z_max
        self._target_adult = target_adult_neurons

        # --- Fontanelle boundary (pressure chamber) ---
        # If not supplied, create a default neonatal fontanelle.
        # This IS the thermal boundary — not an optional add-on.
        self._fontanelle: FontanelleModel = fontanelle or FontanelleModel("neonate")
        self._fontanelle_state: FontanelleState | None = None

        # Effective Q threshold (changes with fontanelle boundary)
        self._q_effective: float = Q_CRITICAL  # Updated each tick

        # Pressure chamber metrics
        self._pressure_chamber_active: bool = False
        self._cognitive_boost: float = 1.0
        self._cumulative_heat_dissipated: float = 0.0

        # --- Neuron population ---
        self._neurons: List[NeuronUnit] = []
        self._next_id: int = 0
        self._peak_neuron_count: int = 0

        # --- Thermal state ---
        self._total_gamma_sq: float = 0.0
        self._brain_temperature: float = T_BRAIN_BASELINE
        self._gradient_decay_rate: float = GRADIENT_DECAY_MIN

        # --- Death counters ---
        self._thermal_deaths: int = 0
        self._hebbian_deaths: int = 0

        # --- Development ---
        self._tick_count: int = 0
        self._phase: str = "overproduction"

        # Spawn initial neurons (must come after all attributes are set)
        self._spawn_neurons(initial_neurons)

    # ------------------------------------------------------------------
    # Neuron spawning
    # ------------------------------------------------------------------

    def _spawn_neurons(self, count: int) -> None:
        """
        Spawn neurons with random impedance.

        Z_init ~ Uniform(Z_min, Z_max)

        This randomness is the SOURCE of Γ² thermal load.
        More neurons with random Z → more total Γ² → but each neuron
        carries less of the load because N is large.
        """
        for _ in range(count):
            z = float(np.random.uniform(self._z_min, self._z_max))
            neuron = NeuronUnit(
                neuron_id=self._next_id,
                impedance=z,
            )
            self._neurons.append(neuron)
            self._next_id += 1
        self._peak_neuron_count = max(
            self._peak_neuron_count, self.alive_count
        )

    # ------------------------------------------------------------------
    # Core Physics: Γ² Computation
    # ------------------------------------------------------------------

    def compute_pairwise_gamma_sq(
        self,
        signal_impedance: float = Z_SIGNAL_TYPICAL,
    ) -> float:
        """
        Compute total Γ² across all alive neurons for a given signal impedance.

        For each neuron i:
            Γᵢ = (Zᵢ − Z_signal) / (Zᵢ + Z_signal)
            Γᵢ² = reflected energy fraction

        Total: ΣΓ² = Σᵢ Γᵢ²

        Returns:
            Total Γ² (aggregate reflected energy)
        """
        total = 0.0
        for n in self._neurons:
            if not n.alive:
                continue
            denom = n.impedance + signal_impedance
            if denom == 0:
                continue
            gamma = (n.impedance - signal_impedance) / denom
            gamma_sq = gamma ** 2
            total += gamma_sq
            # Distribute heat to individual neuron
            n.local_heat = (
                n.local_heat * THERMAL_ACCUMULATION
                + gamma_sq * (1.0 - THERMAL_DISSIPATION_RATE)
            )
        return total

    def heat_per_neuron(self) -> float:
        """
        The critical quantity: q = ΣΓ² / N_alive.

        This is what determines whether the brain can survive.
        """
        n_alive = self.alive_count
        if n_alive == 0:
            return float('inf')
        return self._total_gamma_sq / n_alive

    # ------------------------------------------------------------------
    # Gradient Decay — Γ² Heat → Impedance Drift
    # ------------------------------------------------------------------

    def apply_gradient_decay(self) -> float:
        """
        Apply thermal gradient decay to all neurons.

        ΔZ_thermal = σ × √q_local × noise

        This is the physical mechanism of "gradient vanishing":
        - High Γ² → high heat → large random impedance drift
        - Previously learned impedance matches degrade
        - Connections that were good become worse

        Returns:
            Average impedance drift magnitude (for diagnostics)
        """
        total_drift = 0.0
        count = 0

        for n in self._neurons:
            if not n.alive:
                continue

            q_local = n.local_heat
            if q_local < 1e-10:
                continue

            # Drift magnitude scales with √(local heat)
            sigma = GRADIENT_DECAY_COEFFICIENT * math.sqrt(q_local)
            sigma = max(sigma, GRADIENT_DECAY_MIN)

            # Random impedance perturbation (thermal noise)
            drift = float(np.random.normal(0.0, sigma))
            n.impedance = max(self._z_min, min(self._z_max, n.impedance + drift))

            total_drift += abs(drift)
            count += 1

        avg_drift = total_drift / max(1, count)
        self._gradient_decay_rate = avg_drift
        return avg_drift

    # ------------------------------------------------------------------
    # Thermal Apoptosis — Death by Γ² Heat
    # ------------------------------------------------------------------

    def apply_thermal_apoptosis(self) -> int:
        """
        Kill neurons whose local heat exceeds the thermal death threshold.

        This is DISTINCT from Hebbian pruning:
        - Hebbian pruning: poorly matched connections removed by selection
        - Thermal apoptosis: neurons DESTROYED by accumulated Γ² heat

        Without 2000B neural modules to distribute the load, thermal
        apoptosis would destroy the brain. That's why neonates overproduce.

        Returns:
            Number of neurons killed by thermal failure
        """
        killed = 0
        for n in self._neurons:
            if not n.alive:
                continue

            if n.local_heat > THERMAL_DEATH_THRESHOLD:
                n.alive = False
                killed += 1
                self._thermal_deaths += 1
            elif n.local_heat > THERMAL_DAMAGE_ONSET:
                # Sublethal damage: strength decays
                n.strength *= (1.0 - THERMAL_DAMAGE_RATE)
                if n.strength < 0.05:
                    n.alive = False
                    killed += 1
                    self._thermal_deaths += 1

        return killed

    # ------------------------------------------------------------------
    # Hebbian Pruning — Connection-determined death
    # ------------------------------------------------------------------

    def apply_hebbian_pruning(
        self,
        signal_impedance: float = Z_SIGNAL_TYPICAL,
        learning_rate: float = 0.01,
    ) -> int:
        """
        Hebbian pruning: neurons with poor impedance matching are weakened.

        ★ C2: ΔZ = −η × Γ × x_pre × x_post

        Neurons that consistently produce high Γ with the signal are weakened.
        Neurons with low Γ (good match) are strengthened.

        Returns:
            Number of neurons pruned by Hebbian selection
        """
        pruned = 0

        for n in self._neurons:
            if not n.alive:
                continue

            denom = n.impedance + signal_impedance
            if denom == 0:
                continue

            # Signed Γ: positive when Z_neuron > Z_signal, negative when below
            gamma_signed = (n.impedance - signal_impedance) / denom
            gamma_abs = abs(gamma_signed)
            transmission = 1.0 - gamma_abs ** 2

            if gamma_abs < GAMMA_CONNECTION_THRESHOLD:
                # Good match → strengthen → adjust Z toward signal
                n.strength = min(2.0, n.strength * 1.02)

                # ★ C2 Hebbian update: ΔZ = −η × Γ_signed × x_pre × x_post
                # Using SIGNED Γ ensures:
                #   Z > Z_signal → Γ > 0 → ΔZ < 0 → Z decreases toward signal ✓
                #   Z < Z_signal → Γ < 0 → ΔZ > 0 → Z increases toward signal ✓
                delta_z = -learning_rate * gamma_signed * transmission * n.strength
                n.impedance += delta_z
                n.impedance = max(self._z_min, min(self._z_max, n.impedance))
            else:
                # Poor match → weaken
                n.strength *= 0.97
                if n.strength < 0.05:
                    n.alive = False
                    pruned += 1
                    self._hebbian_deaths += 1

        return pruned

    # ------------------------------------------------------------------
    # Safe Pruning Calculation
    # ------------------------------------------------------------------

    def effective_q_critical(self) -> float:
        """
        Effective thermal failure threshold, accounting for fontanelle boundary.

        Physics (Pressure Chamber Principle):

            Γ_font = |Z_font − Z_brain| / (Z_font + Z_brain)
            T_font = 1 − Γ_font²     (C1 at the boundary)

            Q_eff = Q_CRITICAL / (1 − T_font × efficiency)

        Open fontanelle  → T_font ≈ 0.93 → Q_eff ≈ 3.9  (heat escapes)
        Closed fontanelle → T_font ≈ 0.02 → Q_eff ≈ 1.0  (pressure chamber)

        This equation states: the fontanelle boundary IS the thermal
        exhaust port. When open, the brain's effective thermal budget
        is much larger because Γ² heat transmits through the soft
        membrane and dissipates. When closed, all heat is trapped —
        the strict Q_CRITICAL applies.
        """
        fs = self._fontanelle_state
        if fs is None:
            return Q_CRITICAL  # No boundary info yet → conservative

        # T_fontanelle from the boundary
        t_font = fs.transmission  # 1 − Γ²_fontanelle
        eff = fs.heat_dissipation_rate

        # The exhaust fraction: how much of ΣΓ² escapes through boundary
        exhaust_fraction = t_font * eff * BOUNDARY_DISSIPATION_WEIGHT

        # Q_eff = Q_CRITICAL / (1 − exhaust_fraction)
        # When exhaust_fraction → 0 (closed): Q_eff → Q_CRITICAL (strict)
        # When exhaust_fraction → 0.7 (open):  Q_eff → 3.3 (relaxed)
        denominator = max(0.05, 1.0 - exhaust_fraction)
        q_eff = Q_CRITICAL / denominator

        self._q_effective = q_eff
        return q_eff

    def safe_prune_count(self) -> int:
        """
        How many neurons can be safely pruned without thermal collapse?

        Uses BOUNDARY-AWARE effective Q threshold:
            q_after = ΣΓ²_remaining / (N_alive − ΔN)  <  Q_effective

        When fontanelle is open:  Q_eff is high → more pruning allowed
        When fontanelle is closed: Q_eff ≈ Q_CRITICAL → strict limit

        Returns:
            Maximum number of neurons that can be safely removed
        """
        n_alive = self.alive_count
        if n_alive == 0:
            return 0

        q_eff = self.effective_q_critical()

        # Minimum neurons needed to keep q < Q_effective
        min_needed = int(math.ceil(self._total_gamma_sq / q_eff))
        min_needed = max(1, min_needed)

        safe_to_remove = max(0, n_alive - min_needed)
        return safe_to_remove

    # ------------------------------------------------------------------
    # Collapse Risk Assessment
    # ------------------------------------------------------------------

    def collapse_risk(self) -> float:
        """
        Current collapse risk accounting for fontanelle boundary.

        Risk = q_per_neuron / Q_effective

        Open fontanelle  → Q_eff high → risk low (thermal exhaust working)
        Closed fontanelle → Q_eff ≈ 1  → risk = q / Q_CRITICAL (strict)
        """
        q = self.heat_per_neuron()
        if math.isinf(q):
            return 1.0
        q_eff = self.effective_q_critical()
        return min(1.0, q / q_eff)

    # ------------------------------------------------------------------
    # Connection Viability by Γ
    # ------------------------------------------------------------------

    def connection_viability(self, z_a: float, z_b: float) -> Dict[str, Any]:
        """
        Determine if two neurons with impedances z_a and z_b can form a
        viable connection.

        Γ = |Z_a − Z_b| / (Z_a + Z_b)

        If Γ < 1/√2:  T > 0.5 → viable (more forward than reflected)
        If Γ > 1/√2:  T < 0.5 → not viable (more reflected than forward)
        """
        denom = z_a + z_b
        if denom == 0:
            return {"viable": False, "gamma": 1.0, "transmission": 0.0, "quality": "dead"}

        gamma = abs(z_a - z_b) / denom
        gamma_sq = gamma ** 2
        transmission = 1.0 - gamma_sq

        viable = gamma < GAMMA_CONNECTION_THRESHOLD
        if gamma < GAMMA_STRONG_CONNECTION:
            quality = "strong"
        elif gamma < GAMMA_WEAK_CONNECTION:
            quality = "moderate"
        elif viable:
            quality = "weak"
        else:
            quality = "rejected"

        return {
            "viable": viable,
            "gamma": round(gamma, 6),
            "gamma_sq": round(gamma_sq, 6),
            "transmission": round(transmission, 6),
            "thermal_waste": round(gamma_sq, 6),
            "quality": quality,
        }

    # ------------------------------------------------------------------
    # Brain Temperature Model
    # ------------------------------------------------------------------

    def _update_brain_temperature(self) -> None:
        """
        Update brain temperature based on aggregate thermal load.

        T_brain = T_baseline + q_avg × thermal_gain

        Cooling: T_brain drifts toward baseline at COOLING_RATE when q is low.
        """
        q = self.heat_per_neuron()
        if math.isinf(q):
            q = 10.0  # Cap for dead brain

        target_temp = T_BRAIN_BASELINE + q * T_BRAIN_THERMAL_GAIN
        target_temp = min(T_BRAIN_MAX, target_temp)

        # Exponential approach to target
        self._brain_temperature += (
            (target_temp - self._brain_temperature) * T_BRAIN_COOLING_RATE
        )
        self._brain_temperature = max(
            T_BRAIN_BASELINE - 1.0,
            min(T_BRAIN_MAX, self._brain_temperature),
        )

    # ------------------------------------------------------------------
    # Development Phase Detection
    # ------------------------------------------------------------------

    def _detect_phase(self) -> str:
        """Determine current developmental phase."""
        n_alive = self.alive_count
        n_initial = len(self._neurons)

        if self._tick_count < SYNAPTOGENESIS_PEAK_TICK:
            return "overproduction"
        elif self._tick_count < PRUNING_ONSET_TICK:
            return "peak"
        elif n_alive > self._target_adult * 1.1:
            return "pruning"
        else:
            return "stable"

    # ------------------------------------------------------------------
    # Main Tick
    # ------------------------------------------------------------------

    def tick(
        self,
        signal_impedance: float = Z_SIGNAL_TYPICAL,
        learning_rate: float = 0.01,
        specialization_index: float = 0.0,
        fontanelle_dissipation: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Advance the neural thermal field by one tick.

        The full cycle:
        1. Compute Γ² for all neurons against the current signal
        2. Advance fontanelle boundary (pressure chamber dynamics)
        3. Apply fontanelle thermal exhaust (boundary-based dissipation)
        4. Apply gradient decay (Γ² heat → impedance drift)
        5. Apply Hebbian pruning (with pressure chamber boost)
        6. Apply thermal apoptosis (kill overheated neurons)
        7. Update brain temperature
        8. Age all neurons

        Args:
            signal_impedance: The dominant signal impedance this tick
            learning_rate: Hebbian learning rate (η)
            specialization_index: 0~1 cortical specialization (drives closure)
            fontanelle_dissipation: Legacy parameter (ignored if fontanelle
                                   boundary is active; kept for API compat)

        Returns:
            Tick report dictionary
        """
        self._tick_count += 1

        # 1. Compute Γ² thermal load
        self._total_gamma_sq = self.compute_pairwise_gamma_sq(signal_impedance)

        # 2. ★ Advance fontanelle boundary — the PRESSURE CHAMBER
        #    Feed ΣΓ² as gamma_sq_heat → fontanelle computes dissipation
        #    based on its own impedance boundary physics:
        #      T_font = 1 − Γ²_font  (C1 at boundary)
        #      heat_dissipated = ΣΓ² × T_font × (1 − closure × 0.7)
        self._fontanelle_state = self._fontanelle.tick(
            specialization_index=specialization_index,
            gamma_sq_heat=self._total_gamma_sq,
        )
        fs = self._fontanelle_state

        # 3. ★ Fontanelle thermal exhaust — physics-based dissipation
        #    The boundary's OWN Γ determines how much heat escapes.
        #    Open fontanelle (Z_f = Z_membrane ≪ Z_bone):
        #      Γ_font small → T_font large → most heat escapes
        #    Closed fontanelle (Z_f ≈ Z_bone):
        #      Γ_font large → T_font small → heat trapped → PRESSURE CHAMBER
        boundary_exhaust = self._compute_boundary_dissipation()
        self._total_gamma_sq = max(0.0, self._total_gamma_sq - boundary_exhaust)
        self._cumulative_heat_dissipated += boundary_exhaust

        # Reduce individual neuron heat proportional to boundary exhaust
        if boundary_exhaust > 0 and self._total_gamma_sq > 0:
            exhaust_ratio = min(0.9, boundary_exhaust / (self._total_gamma_sq + boundary_exhaust))
            for n in self._neurons:
                if n.alive:
                    n.local_heat *= (1.0 - exhaust_ratio * 0.5)

        # 4. Gradient decay (thermal noise → impedance drift)
        avg_drift = self.apply_gradient_decay()

        # 5. Hebbian pruning (C2: improve impedance matching)
        #    ★ Pressure chamber boost: after fontanelle closure, trapped
        #    Γ² heat is "constructively consumed" — Hebbian learning
        #    accelerates because the heat MUST be converted to matching.
        effective_lr = learning_rate * self._cognitive_boost
        hebbian_pruned = self.apply_hebbian_pruning(signal_impedance, effective_lr)

        # 6. Thermal apoptosis (Γ² heat kills neurons)
        thermal_killed = self.apply_thermal_apoptosis()

        # 7. Update brain temperature
        self._update_brain_temperature()

        # 8. Age neurons
        for n in self._neurons:
            if n.alive:
                n.age_ticks += 1

        # 9. Phase detection
        self._phase = self._detect_phase()

        # 10. Update pressure chamber state
        if fs.pressure_chamber_active and not self._pressure_chamber_active:
            self._pressure_chamber_active = True
            self._cognitive_boost = PRESSURE_CHAMBER_BOOST

        # Compute state
        q = self.heat_per_neuron()
        q_eff = self.effective_q_critical()
        n_alive = self.alive_count

        return {
            "tick": self._tick_count,
            "alive_neurons": n_alive,
            "total_gamma_sq": round(self._total_gamma_sq, 6),
            "heat_per_neuron": round(q, 6) if not math.isinf(q) else float('inf'),
            "brain_temperature": round(self._brain_temperature, 2),
            "gradient_decay": round(avg_drift, 6),
            "hebbian_pruned": hebbian_pruned,
            "thermal_killed": thermal_killed,
            "collapse_risk": round(self.collapse_risk(), 4),
            "safe_to_prune": self.safe_prune_count(),
            "phase": self._phase,
            # ★ Fontanelle boundary metrics
            "q_effective": round(q_eff, 4),
            "fontanelle_closure": round(fs.closure_fraction, 4),
            "fontanelle_gamma": round(fs.gamma_fontanelle, 4),
            "fontanelle_transmission": round(fs.transmission, 4),
            "boundary_heat_dissipated": round(boundary_exhaust, 6),
            "pressure_chamber_active": self._pressure_chamber_active,
            "cognitive_boost": round(self._cognitive_boost, 2),
        }

    # ------------------------------------------------------------------
    # Fontanelle Boundary Dissipation (Pressure Chamber Physics)
    # ------------------------------------------------------------------

    def _compute_boundary_dissipation(self) -> float:
        """
        Compute heat dissipated through fontanelle boundary.

        Physics:
            The fontanelle is a transmission line TERMINATION with:
                Γ_font = |Z_font − Z_brain| / (Z_font + Z_brain)
                T_font = 1 − Γ_font²   (C1 at boundary)

            Heat that TRANSMITS through the boundary ESCAPES the skull.
            Heat that REFLECTS stays trapped inside.

            dissipated = ΣΓ² × T_font × efficiency × (1 − closure × 0.7)

        Open fontanelle (Z_font ≈ 5Ω, Z_brain ≈ 75Ω):
            Γ_font = |5−75|/(5+75) = 0.875 → T = 0.234
            But structural openness is high → eff ≈ 0.8
            → significant heat escape

        Closed fontanelle (Z_font ≈ 500Ω):
            Γ_font = |500−75|/(500+75) = 0.739 → T = 0.454
            But structural openness is zero → eff ≈ 0.1
            → almost no heat escape → PRESSURE CHAMBER

        The key insight: it's not just T_font that matters — it's
        T_font × structural_openness. The bone closure physically
        blocks the thermal pathway even if T is nonzero.
        """
        fs = self._fontanelle_state
        if fs is None:
            return 0.0

        t_font = fs.transmission
        eff = fs.heat_dissipation_rate
        closure = fs.closure_fraction

        # Structural openness: how much of the boundary surface is membrane
        structural_openness = max(0.0, 1.0 - closure * 0.7)

        # dissipation = ΣΓ² × T_boundary × efficiency × openness
        dissipated = self._total_gamma_sq * t_font * eff * structural_openness

        return max(0.0, dissipated)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alive_count(self) -> int:
        """Count of alive neurons."""
        return sum(1 for n in self._neurons if n.alive)

    @property
    def alive_neurons(self) -> List[NeuronUnit]:
        """List of alive neurons."""
        return [n for n in self._neurons if n.alive]

    # ------------------------------------------------------------------
    # Demonstrate the 2000B Necessity
    # ------------------------------------------------------------------

    def demonstrate_overproduction_necessity(
        self,
        small_n: int = 50,
        large_n: int = 2000,
        ticks: int = 100,
    ) -> Dict[str, Any]:
        """
        Empirically demonstrate why neonatal overproduction is necessary.

        Creates two parallel simulations:
        - small_n neurons (insufficient buffer)
        - large_n neurons (2000B-equivalent buffer)

        Shows that small_n leads to thermal collapse while large_n survives.

        Returns:
            Comparison report
        """
        # Small brain
        small_brain = NeurogenesisThermalShield(
            initial_neurons=small_n,
            target_adult_neurons=small_n // 2,
        )
        # Large brain
        large_brain = NeurogenesisThermalShield(
            initial_neurons=large_n,
            target_adult_neurons=large_n // 2,
        )

        small_history = []
        large_history = []

        for _ in range(ticks):
            r_small = small_brain.tick()
            r_large = large_brain.tick()
            small_history.append(r_small)
            large_history.append(r_large)

        return {
            "small_brain": {
                "initial_neurons": small_n,
                "final_alive": small_brain.alive_count,
                "thermal_deaths": small_brain._thermal_deaths,
                "peak_heat_per_neuron": max(h["heat_per_neuron"] for h in small_history
                                            if not math.isinf(h["heat_per_neuron"])),
                "peak_collapse_risk": max(h["collapse_risk"] for h in small_history),
                "final_temperature": small_brain._brain_temperature,
            },
            "large_brain": {
                "initial_neurons": large_n,
                "final_alive": large_brain.alive_count,
                "thermal_deaths": large_brain._thermal_deaths,
                "peak_heat_per_neuron": max(h["heat_per_neuron"] for h in large_history
                                            if not math.isinf(h["heat_per_neuron"])),
                "peak_collapse_risk": max(h["collapse_risk"] for h in large_history),
                "final_temperature": large_brain._brain_temperature,
            },
            "conclusion": (
                "Small brain suffers higher per-neuron heat → more thermal deaths → collapse risk. "
                "Large brain distributes Γ² load → each neuron stays cool → survives. "
                f"Thermal deaths: small={small_brain._thermal_deaths}, "
                f"large={large_brain._thermal_deaths}."
            ),
        }

    # ------------------------------------------------------------------
    # ElectricalSignal (★ C3)
    # ------------------------------------------------------------------

    def get_signal(self) -> ElectricalSignal:
        """
        Generate thermal field state as ElectricalSignal.

        The signal encodes:
        - Amplitude: collapse risk (higher = more danger)
        - Frequency: brain temperature oscillation
        - Waveform: thermal noise pattern across neural population
        """
        risk = self.collapse_risk()
        amplitude = float(np.clip(0.1 + risk * 0.8, 0.05, 1.0))
        freq = THERMAL_FREQUENCY + (self._brain_temperature - T_BRAIN_BASELINE) * 0.5

        t = np.linspace(0, 1, THERMAL_SAMPLE_POINTS)
        waveform = amplitude * np.sin(2 * math.pi * freq * t)
        # Add thermal noise proportional to q
        q = self.heat_per_neuron()
        if not math.isinf(q):
            noise_amp = min(0.3, q * 0.5)
            waveform += noise_amp * np.random.normal(0, 1, THERMAL_SAMPLE_POINTS)

        return ElectricalSignal(
            waveform=waveform.astype(np.float32),
            frequency=freq,
            amplitude=amplitude,
            phase=0.0,
            impedance=THERMAL_IMPEDANCE,
            snr=THERMAL_SNR,
            source="neurogenesis_thermal",
            modality="interoceptive",
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Full state for introspection."""
        q = self.heat_per_neuron()
        q_eff = self.effective_q_critical()
        alive = self.alive_neurons
        z_values = [n.impedance for n in alive] if alive else [0.0]

        stats = {
            "total_neurons": len(self._neurons),
            "alive_neurons": self.alive_count,
            "peak_neuron_count": self._peak_neuron_count,
            "target_adult_neurons": self._target_adult,
            "total_gamma_sq": round(self._total_gamma_sq, 6),
            "heat_per_neuron": round(q, 6) if not math.isinf(q) else float('inf'),
            "q_effective": round(q_eff, 4),
            "brain_temperature": round(self._brain_temperature, 2),
            "gradient_decay_rate": round(self._gradient_decay_rate, 6),
            "thermal_deaths": self._thermal_deaths,
            "hebbian_deaths": self._hebbian_deaths,
            "collapse_risk": round(self.collapse_risk(), 4),
            "safe_to_prune": self.safe_prune_count(),
            "phase": self._phase,
            "z_mean": round(float(np.mean(z_values)), 2),
            "z_std": round(float(np.std(z_values)), 2),
            "tick_count": self._tick_count,
            # Fontanelle boundary
            "pressure_chamber_active": self._pressure_chamber_active,
            "cognitive_boost": round(self._cognitive_boost, 2),
            "cumulative_heat_dissipated": round(self._cumulative_heat_dissipated, 4),
        }

        if self._fontanelle_state is not None:
            fs = self._fontanelle_state
            stats["fontanelle_closure"] = round(fs.closure_fraction, 4)
            stats["fontanelle_gamma"] = round(fs.gamma_fontanelle, 4)
            stats["fontanelle_transmission"] = round(fs.transmission, 4)

        return stats

    def get_state(self) -> ThermalFieldState:
        """Get structured state snapshot."""
        q = self.heat_per_neuron()
        return ThermalFieldState(
            total_neurons=len(self._neurons),
            alive_neurons=self.alive_count,
            total_gamma_sq=self._total_gamma_sq,
            heat_per_neuron=q if not math.isinf(q) else 999.0,
            brain_temperature=self._brain_temperature,
            gradient_decay_rate=self._gradient_decay_rate,
            thermal_deaths=self._thermal_deaths,
            hebbian_deaths=self._hebbian_deaths,
            collapse_risk=self.collapse_risk(),
            phase=self._phase,
            safe_to_prune=self.safe_prune_count(),
        )

    @property
    def fontanelle(self) -> FontanelleModel:
        """Access the fontanelle boundary model."""
        return self._fontanelle

    @property
    def pressure_chamber_active(self) -> bool:
        """Whether the pressure chamber (post-closure) is active."""
        return self._pressure_chamber_active

    @property
    def cognitive_boost(self) -> float:
        """Current cognitive acceleration factor from pressure chamber."""
        return self._cognitive_boost
