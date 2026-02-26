# -*- coding: utf-8 -*-
"""
Attention Plasticity Engine
Attention Plasticity: Training the Brain's Filter

Physics:
  "A pro gamer sees an opponent move and reacts in just 70ms.
    An average person needs 250ms.
    The difference is not genetic — it's synaptic."

  Attention is not a software setting; it is a property of a physical circuit.
  Training changes not 'willpower' — training changes:

  1. Thalamic gate RC time constant
     - More experience → smaller gate capacitance → faster switching
     - gate_time_constant = C × R → exponential decay
     - Pro gamer: gate response ~20ms
     - Untrained person: gate response ~80ms

  2. Perceptual tuner quality factor Q
     - High Q = narrowband filter = precisely locked to a specific frequency
     - Low Q = wideband filter = hears everything but imprecisely
     - Repeated exposure to same stimulus → synaptic strengthening → L/C ratio increases → Q rises
     - Q = sqrt(L/C) × (1/R)
     - Wine sommelier: taste Q >> average person

  3. Cortical inhibition efficiency
     - Prefrontal cortex inhibits unwanted impulses (NoGo)
     - Each successful inhibition → inhibitory synapse strengthening → future inhibition costs less energy
     - Athletes: low inhibition energy cost → energy lasts longer → longer focus

  4. Response pathway myelination
     - Frequently used neural pathways → thicker myelin → faster conduction velocity
     - Sensory→Thalamus→Cortex→Basal ganglia→Motor: delay in each segment can be shortened
     - Total delay = Σ(segment length / conduction velocity) + Σ(synaptic delay)

  "Practice doesn't make you smarter.
    Practice makes your circuits faster, more precise, and more energy-efficient."
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ============================================================================
# Physical Constants
# ============================================================================

# --- Gate time constant plasticity ---
GATE_TAU_INITIAL = 0.3          # Initial gate EMA coefficient (slow)
GATE_TAU_MIN = 0.05             # Fastest gate response after training
GATE_TAU_LEARNING_RATE = 0.002  # Improvement per successful attention event
GATE_TAU_DECAY = 0.0001         # Degradation rate when unused

# --- Tuner Q plasticity ---
Q_INITIAL = 2.0                 # Initial quality factor (wideband)
Q_MAX = 12.0                    # Maximum quality factor (extreme narrowband) — expert level
Q_LEARNING_RATE = 0.01          # Q growth per successful resonance
Q_DECAY_RATE = 0.0005           # Q decay when unused
Q_CROSS_MODAL_TRANSFER = 0.3    # Cross-modal transfer rate (visual training helps auditory)

# --- Inhibition efficiency plasticity ---
INHIBITION_EFFICIENCY_INITIAL = 1.0    # Initial inhibition cost multiplier (1.0 = standard)
INHIBITION_EFFICIENCY_MIN = 0.3        # Minimum cost after training (0.3 = expert)
INHIBITION_LEARNING_RATE = 0.005       # Improvement per successful inhibition
INHIBITION_DECAY_RATE = 0.0002         # Degradation when unused

# --- PFC capacity plasticity ---
PFC_CAPACITY_INITIAL = 1.0      # Initial PFC energy upper limit
PFC_CAPACITY_MAX = 2.5          # Maximum after training (meditation master)
PFC_CAPACITY_GROWTH = 0.001     # Capacity growth per successful sustained attention
PFC_CAPACITY_DECAY = 0.00005    # Capacity atrophy when unused

# --- Reaction delay plasticity ---
CONDUCTION_DELAY_INITIAL = 0.050   # Initial conduction delay (per segment) in seconds
CONDUCTION_DELAY_MIN = 0.010       # Minimum delay after myelination
MYELINATION_RATE = 0.001           # Myelination rate (per use)
MYELINATION_DECAY = 0.000003       # Myelin degradation when inactive (far slower than learning)

# Pathway segments
PATHWAY_SEGMENTS = {
    "sensory_thalamic":  0.030,    # Sensory organ → Thalamus (30ms)
    "thalamic_cortical": 0.020,    # Thalamus → Cortex (20ms)
    "cortical_bg":       0.015,    # Cortex → Basal ganglia (15ms)
    "bg_motor":          0.010,    # Basal ganglia → Motor (10ms)
    "amygdala_shortcut": 0.012,    # Amygdala low road (12ms) — fastest pathway
}

# --- Attention bandwidth plasticity ---
ATTENTION_SLOTS_INITIAL = 3     # Initial simultaneous attention slots
ATTENTION_SLOTS_MAX = 5         # Maximum after training (pilot level)
SLOT_TRAINING_THRESHOLD = 100   # Required successful multi-focus attention events

# --- Training statistics ---
EXPOSURE_MEMORY_WINDOW = 1000   # Track last N exposures


# ============================================================================
# Modality Training Record
# ============================================================================


@dataclass
class ModalityTrainingRecord:
    """
    Training record for a single modality

    Tracks the history of attention, successful identification, and successful inhibition
    for this modality → used to compute plasticity adjustments
    """
    modality: str

    # Exposure statistics
    total_exposures: int = 0           # Total exposure count
    successful_locks: int = 0          # Successful resonance lock count
    successful_identifications: int = 0  # Successful concept identification count
    successful_inhibitions: int = 0    # Successful inhibition count
    multi_focus_successes: int = 0     # Multi-focus attention success count

    # Current plasticity parameters (actual values)
    gate_tau: float = GATE_TAU_INITIAL
    tuner_q: float = Q_INITIAL
    inhibition_efficiency: float = INHIBITION_EFFICIENCY_INITIAL
    conduction_delays: Dict[str, float] = field(
        default_factory=lambda: dict(PATHWAY_SEGMENTS)
    )

    # Timestamps
    last_exposure_time: float = 0.0
    last_successful_lock_time: float = 0.0

    @property
    def lock_rate(self) -> float:
        """Lock success rate"""
        return self.successful_locks / max(1, self.total_exposures)

    @property
    def total_reaction_delay(self) -> float:
        """Total reaction delay (seconds) = sum of conduction delays across all pathway segments"""
        return sum(self.conduction_delays.values())

    @property
    def training_level(self) -> str:
        """Training level"""
        exp = self.total_exposures
        if exp < 50:
            return "novice"
        elif exp < 500:
            return "intermediate"
        elif exp < 5000:
            return "advanced"
        elif exp < 50000:
            return "expert"
        else:
            return "master"


# ============================================================================
# Global Training Record
# ============================================================================


@dataclass
class GlobalTrainingState:
    """
    Global training state

    Training parameters shared across modalities
    """
    pfc_capacity: float = PFC_CAPACITY_INITIAL
    attention_slots: int = ATTENTION_SLOTS_INITIAL
    total_training_ticks: int = 0
    total_successful_attentions: int = 0
    total_successful_multi_focus: int = 0


# ============================================================================
# Main Engine
# ============================================================================


class AttentionPlasticityEngine:
    """
    Attention Plasticity Engine

    Physics:
      Each attention task (perception→identification→response) forms a complete circuit.
      Successfully completing the circuit → synaptic strengthening (LTP)
      Failure or disuse → synaptic weakening (LTD)

      This is not a reward function. This is Hebbian learning:
      "Neurons that fire together, wire together."

    Training loop:
      1. Sensory stimulus arrives    → on_exposure()
      2. Successful resonance lock   → on_successful_lock()
      3. Successful concept ID       → on_successful_identification()
      4. Successful impulse inhibit  → on_successful_inhibition()
      5. Per-tick decay              → decay_tick()

    Read current parameters:
      - get_gate_tau(modality)        → Gate time constant
      - get_tuner_q(modality)         → Tuner quality factor
      - get_inhibition_cost(modality) → Inhibition energy cost multiplier
      - get_reaction_delay(modality)  → Total reaction delay
      - get_pfc_capacity()            → PFC energy upper limit
      - get_attention_slots()         → Simultaneous attention count
    """

    def __init__(self):
        self._modality_records: Dict[str, ModalityTrainingRecord] = {}
        self._global = GlobalTrainingState()

    # ------------------------------------------------------------------
    # Record Management
    # ------------------------------------------------------------------

    def _ensure_record(self, modality: str) -> ModalityTrainingRecord:
        """Ensure a training record exists for the given modality"""
        if modality not in self._modality_records:
            self._modality_records[modality] = ModalityTrainingRecord(
                modality=modality
            )
        return self._modality_records[modality]

    # ------------------------------------------------------------------
    # Training Events (LTP — Long-Term Potentiation)
    # ------------------------------------------------------------------

    def on_exposure(self, modality: str):
        """
        Sensory stimulus exposure — basic statistics tracking

        Called each time a signal passes through the thalamic gate.
        Exposure alone does not improve anything — successful circuit closure is required.
        """
        rec = self._ensure_record(modality)
        rec.total_exposures += 1
        rec.last_exposure_time = time.time()

    def on_successful_lock(self, modality: str):
        """
        Successful resonance lock — improves gate speed + tuner Q

        Physics:
          Repeated resonance in sensory cortex → LTP → synaptic efficiency increase
          → Gate RC time constant decreases (capacitance decreases)
          → Tuner Q increases (L/C ratio increases)

        Esports analogy:
          "Repeatedly seeing a flash signal → recognizing it as an enemy's move"
          Each success makes the next recognition 2ms faster.
        """
        rec = self._ensure_record(modality)
        rec.successful_locks += 1
        rec.last_successful_lock_time = time.time()

        # === Gate τ improvement ===
        # τ(n+1) = τ(n) × (1 - learning_rate)
        # Exponential decay: more training = faster, but with a physical lower bound
        rec.gate_tau = max(
            GATE_TAU_MIN,
            rec.gate_tau * (1.0 - GATE_TAU_LEARNING_RATE)
        )

        # === Tuner Q improvement ===
        # Q(n+1) = Q(n) + rate × (Q_max - Q(n)) / Q_max
        # Asymptotic growth: closer to ceiling = slower (biological saturation)
        headroom = (Q_MAX - rec.tuner_q) / Q_MAX
        rec.tuner_q = min(
            Q_MAX,
            rec.tuner_q + Q_LEARNING_RATE * headroom
        )

        # === Conduction delay improvement (myelination) ===
        # Most frequently used pathways are myelinated first
        for segment in ["sensory_thalamic", "thalamic_cortical"]:
            rec.conduction_delays[segment] = max(
                CONDUCTION_DELAY_MIN,
                rec.conduction_delays[segment] * (1.0 - MYELINATION_RATE)
            )

        # === Cross-modal transfer ===
        self._cross_modal_transfer(modality, "tuner_q", Q_CROSS_MODAL_TRANSFER)

    def on_successful_identification(self, modality: str):
        """
        Successful concept identification — improves tuner Q (stronger effect) + cortical pathway myelination

        A higher-level training signal: not just "sensed it," but "knows what it is."
        """
        rec = self._ensure_record(modality)
        rec.successful_identifications += 1

        # Q improvement (identification gives a larger LTP signal than locking)
        headroom = (Q_MAX - rec.tuner_q) / Q_MAX
        rec.tuner_q = min(
            Q_MAX,
            rec.tuner_q + Q_LEARNING_RATE * 2.0 * headroom
        )

        # Cortex→Basal ganglia pathway myelination (action selection follows concept ID)
        for segment in ["cortical_bg", "bg_motor"]:
            rec.conduction_delays[segment] = max(
                CONDUCTION_DELAY_MIN,
                rec.conduction_delays[segment] * (1.0 - MYELINATION_RATE)
            )

        # PFC capacity micro-increase (each successful cognition slightly strengthens PFC)
        self._global.pfc_capacity = min(
            PFC_CAPACITY_MAX,
            self._global.pfc_capacity + PFC_CAPACITY_GROWTH
        )

        self._global.total_successful_attentions += 1

    def on_successful_inhibition(self, modality: str):
        """
        Successful impulse inhibition — improves inhibition efficiency

        Physics:
          PFC → striatum inhibition pathway (NoGo) synapse strengthening
          → Future inhibition of the same impulse requires less PFC energy

        Meditation analogy:
          Repeatedly practicing "observe without reacting" → inhibition circuit efficiency improves
          → Experts can remain undistracted for extended periods
        """
        rec = self._ensure_record(modality)
        rec.successful_inhibitions += 1

        # Inhibition efficiency improvement
        rec.inhibition_efficiency = max(
            INHIBITION_EFFICIENCY_MIN,
            rec.inhibition_efficiency * (1.0 - INHIBITION_LEARNING_RATE)
        )

        # PFC capacity growth (successful inhibition = meaningful PFC training)
        self._global.pfc_capacity = min(
            PFC_CAPACITY_MAX,
            self._global.pfc_capacity + PFC_CAPACITY_GROWTH * 2.0
        )

    def on_multi_focus_success(self, modalities: List[str]):
        """
        Multi-focus attention success — expands attention bandwidth

        Physics:
          Simultaneously maintaining multiple thalamic gates open with successful identification
          → Thalamic TRN (reticular nucleus) inhibition efficiency improves
          → Can sustain more channels simultaneously

        Pilot analogy:
          Simultaneously monitoring speedometer, altimeter, radar, comms → 4-channel attention
        """
        for mod in modalities:
            rec = self._ensure_record(mod)
            rec.multi_focus_successes += 1

        self._global.total_successful_multi_focus += 1

        # Reaching threshold → unlock a new attention slot
        if (self._global.total_successful_multi_focus > 0 and
                self._global.total_successful_multi_focus % SLOT_TRAINING_THRESHOLD == 0):
            self._global.attention_slots = min(
                ATTENTION_SLOTS_MAX,
                self._global.attention_slots + 1
            )

    # ------------------------------------------------------------------
    # Forgetting and Decay (LTD — Long-Term Depression)
    # ------------------------------------------------------------------

    def decay_tick(self):
        """
        Natural decay per tick

        Physics:
          Unused synapses → synaptic pruning → ability degradation
          "Use it or lose it" — this is a physical law, not a punishment.

        But decay is far slower than learning (biological asymmetry):
          Building a habit takes 1000 repetitions
          Forgetting a habit takes 10000 ticks of disuse
        """
        self._global.total_training_ticks += 1

        for rec in self._modality_records.values():
            # Gate τ degradation (becomes slower)
            rec.gate_tau = min(
                GATE_TAU_INITIAL,
                rec.gate_tau + GATE_TAU_DECAY
            )

            # Q degradation (becomes duller)
            rec.tuner_q = max(
                Q_INITIAL,
                rec.tuner_q - Q_DECAY_RATE
            )

            # Inhibition efficiency degradation (becomes more energy-costly)
            rec.inhibition_efficiency = min(
                INHIBITION_EFFICIENCY_INITIAL,
                rec.inhibition_efficiency + INHIBITION_DECAY_RATE
            )

            # Myelin degradation (conduction becomes slower)
            for seg, default_delay in PATHWAY_SEGMENTS.items():
                current = rec.conduction_delays.get(seg, default_delay)
                rec.conduction_delays[seg] = min(
                    default_delay,
                    current + MYELINATION_DECAY
                )

        # PFC capacity degradation
        self._global.pfc_capacity = max(
            PFC_CAPACITY_INITIAL,
            self._global.pfc_capacity - PFC_CAPACITY_DECAY
        )

    # ------------------------------------------------------------------
    # Cross-Modal Transfer
    # ------------------------------------------------------------------

    def _cross_modal_transfer(
        self, source_modality: str, param: str, transfer_rate: float
    ):
        """
        Cross-modal transfer — training one modality benefits others

        Physics:
          Higher-order attention cortex (e.g., FEF — frontal eye fields) is cross-modal.
          Training visual attention also improves auditory reaction speed.
          But the transfer effect is only 30% of direct training.

        Esports analogy:
          "After training visual reactions, auditory reaction to enemy footsteps also gets faster."
        """
        source_rec = self._ensure_record(source_modality)

        for mod, rec in self._modality_records.items():
            if mod == source_modality:
                continue

            if param == "tuner_q":
                improvement = source_rec.tuner_q - rec.tuner_q
                if improvement > 0:
                    rec.tuner_q += improvement * transfer_rate * Q_LEARNING_RATE

            elif param == "gate_tau":
                improvement = rec.gate_tau - source_rec.gate_tau
                if improvement > 0:
                    rec.gate_tau -= improvement * transfer_rate * GATE_TAU_LEARNING_RATE

    # ------------------------------------------------------------------
    # Read Current Training Parameters (for use by other engines)
    # ------------------------------------------------------------------

    def get_gate_tau(self, modality: str) -> float:
        """Get the gate time constant for a modality"""
        if modality in self._modality_records:
            return self._modality_records[modality].gate_tau
        return GATE_TAU_INITIAL

    def get_tuner_q(self, modality: str) -> float:
        """Get the tuner quality factor for a modality"""
        if modality in self._modality_records:
            return self._modality_records[modality].tuner_q
        return Q_INITIAL

    def get_inhibition_cost_multiplier(self, modality: str) -> float:
        """Get the inhibition cost multiplier for a modality"""
        if modality in self._modality_records:
            return self._modality_records[modality].inhibition_efficiency
        return INHIBITION_EFFICIENCY_INITIAL

    def get_reaction_delay(self, modality: str) -> float:
        """
        Get the total reaction delay for a modality (seconds)

        = Sum of all pathway segment delays
        Initial ≈ 87ms, fastest after training ≈ 50ms (including synaptic delays)
        """
        if modality in self._modality_records:
            return self._modality_records[modality].total_reaction_delay
        return sum(PATHWAY_SEGMENTS.values())

    def get_pfc_capacity(self) -> float:
        """Get PFC energy capacity upper limit"""
        return self._global.pfc_capacity

    def get_attention_slots(self) -> int:
        """Get the number of simultaneous attention slots"""
        return self._global.attention_slots

    def get_training_record(self, modality: str) -> Optional[ModalityTrainingRecord]:
        """Get the complete training record for a modality"""
        return self._modality_records.get(modality)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Complete training state snapshot"""
        modality_states = {}
        for mod, rec in self._modality_records.items():
            modality_states[mod] = {
                "training_level": rec.training_level,
                "total_exposures": rec.total_exposures,
                "lock_rate": round(rec.lock_rate, 4),
                "gate_tau": round(rec.gate_tau, 6),
                "tuner_q": round(rec.tuner_q, 4),
                "inhibition_efficiency": round(rec.inhibition_efficiency, 4),
                "total_reaction_delay_ms": round(
                    rec.total_reaction_delay * 1000, 2
                ),
                "conduction_delays_ms": {
                    seg: round(d * 1000, 3)
                    for seg, d in rec.conduction_delays.items()
                },
            }

        return {
            "global": {
                "pfc_capacity": round(self._global.pfc_capacity, 4),
                "attention_slots": self._global.attention_slots,
                "total_training_ticks": self._global.total_training_ticks,
                "total_successful_attentions":
                    self._global.total_successful_attentions,
                "total_multi_focus":
                    self._global.total_successful_multi_focus,
            },
            "modalities": modality_states,
        }
