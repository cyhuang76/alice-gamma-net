# -*- coding: utf-8 -*-
"""
Alice's Pinch Fatigue Engine — Neural Aging Physics via Pollock-Barraclough
Pinch Fatigue Engine — Neural Aging via Electromagnetic Self-Compression

===============================================================================
Historical Background (1905):
  Pollock & Barraclough studied a hollow copper lightning rod twisted by a lightning strike,
  discovering that the conductor was not melted by the lightning's heat, but physically squeezed and deformed by its powerful magnetic field force.

  This is the "Pinch Effect" —
  When current I flows through a conductor, the conductor's self-induced magnetic field B = μ₀I/(2πr) produces an inward Lorentz force:
    F_pinch = J × B (current density × magnetic field = inward pressure)

  For hollow conductors (such as lightning rods, inner conductors of coaxial cables),
  this force can cause the conductor to radially contract, changing its cross-sectional geometry → changing impedance.
===============================================================================

Physical core:
  Alice's neural pathways = coaxial cables.
  Each high-intensity neural activity = large current pulse.
  Large current → pinch force → microscopic conductor strain.

  Strain is divided into two types (fundamental materials mechanics):
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Elastic deformation: fully recovers after force is removed
    → This is the existing ImpedanceDebtTracker
    → Sleep repair = stress unloading → deformation rebound
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Plastic deformation: exceeds yield strength → permanent deformation → irreversible
    → This is "aging" ★
    → Accumulated microscopic plastic strain = residual that cannot be repaired by sleep
    → Coffin-Manson fatigue law: N_f = C / (Δε_plastic)^β
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Geometric deformation → impedance change:
    Z_coax = (Z₀/2π) × ln(r_outer/r_inner)
    Pinch causes r_inner ↓ → ln ratio ↑ → Z ↑ → Γ drift
    Accumulated plastic deformation → Z permanently deviates from design value → physical mechanism of aging

  Neuroscience correspondence:
    ・Axon = coaxial cable inner conductor
    ・High-frequency firing = large current
    ・Pinch force → axonal membrane physical stress → ion channel redistribution
    ・Repeated high stress → membrane microcracks → myelin degradation → permanent impedance shift
    ・Increased axon diameter variability in aging brains = evidence of accumulated geometric deformation
    ・This is why the elderly have decreased SNR — not "rusting", but "crushing"

  Wöhler/S-N curve (fatigue life):
    Metal under alternating stress does not fracture immediately,
    but fails after N cycles due to microcrack propagation.
    ・Low stress → essentially infinite life (below fatigue limit)
    ・High stress → failure in few cycles
    ・This explains why "occasional excitement" does not cause aging,
      but "sustained high stress" accelerates aging.

References:
  [40] Pollock, J. A., & Barraclough, S. H. (1905).
       "On the mechanical effects of electric currents in conductors."
       Proceedings of the Royal Society of New South Wales, 39, 131-158.

  [41] Coffin, L. F. (1954). A study of the effects of cyclic thermal
       stresses on a ductile metal. Trans. ASME, 76, 931-950.
       (Coffin-Manson low-cycle fatigue law)

  [42] Wöhler, A. (1870). Über die Festigkeitsversuche mit Eisen und Stahl.
       Zeitschrift für Bauwesen, 20.
       (original S-N fatigue curve literature)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ============================================================================
# Physical Constants — Pinch Fatigue
# ============================================================================

# --- Pinch Force ---
MU_0 = 4 * math.pi * 1e-7          # Vacuum permeability (H/m) — used for normalized analogy
PINCH_SENSITIVITY = 0.02            # Current → pinch strain sensitivity
PINCH_EXPONENT = 2.0                # Pinch force ∝ I² (physical law)

# --- Material Mechanics ---
YIELD_STRAIN = 0.3                  # Yield strain (exceeding → plastic deformation)
ELASTIC_MODULUS = 5.0               # Elastic modulus E (scaling factor for recovery speed)
WORK_HARDENING_RATE = 0.001         # Work hardening (each plastic strain → slight yield strength increase)
MAX_WORK_HARDENING = 0.2            # Maximum work hardening amount

# --- Fatigue Life (Coffin-Manson) ---
FATIGUE_EXPONENT = 2.0              # β in N_f = C / (Δε)^β
FATIGUE_COEFFICIENT = 100.0         # C in Coffin-Manson
FATIGUE_DAMAGE_PER_CYCLE = 0.0001   # Base damage increment per over-limit cycle
MAX_FATIGUE_DAMAGE = 1.0            # Damage reaching this value = conductor failure

# --- Aging → Impedance Drift ---
AGING_IMPEDANCE_SENSITIVITY = 0.15  # Plastic strain → Z drift conversion
MAX_IMPEDANCE_DRIFT = 0.5           # Maximum permanent Z drift (relative to design value)

# --- Self-Repair ---
NEURAL_REPAIR_RATE = 0.0002         # Per-tick splice repair rate (very slow)
SLEEP_REPAIR_BOOST = 3.0            # Sleep repair acceleration multiplier
GROWTH_FACTOR_REPAIR = 0.001        # Neurotrophic factor-driven repair (simulating BDNF)

# --- Temperature Effect (Arrhenius Acceleration) ---
ARRHENIUS_ACTIVATION = 0.02         # Temperature-accelerated fatigue coefficient
REFERENCE_TEMP = 0.5                # Reference temperature (normalized)

# --- History ---
MAX_HISTORY = 1000


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class ConductorState:
    """
    Conductor state of a single neural pathway

    Physics: each axon/neural pathway is modeled as the inner conductor of a coaxial cable,
    subjected to the pinch force of current, experiencing elastic and plastic deformation.
    """
    channel_id: str                 # Channel ID

    # --- Geometric State ---
    design_impedance: float = 75.0  # Design impedance (Ω)
    current_impedance: float = 75.0 # Current impedance (including aging drift)

    # --- Strain State ---
    elastic_strain: float = 0.0     # Current elastic strain (recoverable)
    plastic_strain: float = 0.0     # Cumulative plastic strain (irreversible = aging)
    peak_strain: float = 0.0        # Historical peak strain

    # --- Fatigue ---
    fatigue_damage: float = 0.0     # Miner's rule damage accumulation [0, 1]
    cycle_count: int = 0            # Total stress cycle count
    over_yield_cycles: int = 0      # Over-limit cycle count

    # --- Work Hardening ---
    work_hardening: float = 0.0     # Cumulative work hardening
    effective_yield: float = YIELD_STRAIN  # Effective yield strength

    # --- Derived Quantities ---
    impedance_drift: float = 0.0    # Permanent impedance drift
    gamma_drift: float = 0.0        # Γ drift due to aging

    @property
    def age_factor(self) -> float:
        """Aging factor [0, 1]: 0=brand new, 1=end of life"""
        return min(self.fatigue_damage, 1.0)

    @property
    def structural_integrity(self) -> float:
        """Structural integrity [0, 1]: 1=intact, 0=failed"""
        return max(0.0, 1.0 - self.fatigue_damage)

    @property
    def is_degraded(self) -> bool:
        """Whether significantly aged (fatigue >30%)"""
        return self.fatigue_damage > 0.3


@dataclass
class PinchEvent:
    """A single pinch stress event"""
    channel_id: str
    current_intensity: float     # Normalized current intensity
    pinch_strain: float          # Pinch strain ε
    plastic_increment: float     # Plastic increment for this event
    fatigue_increment: float     # Fatigue increment for this event
    was_over_yield: bool         # Whether yield was exceeded
    temperature: float           # Temperature at event time


@dataclass
class AgingSignal:
    """Output signal of the aging engine"""
    mean_age_factor: float       # Mean aging factor
    max_age_factor: float        # Max aging factor
    total_plastic_strain: float  # Global plastic strain
    mean_impedance_drift: float  # Mean impedance drift
    degraded_channels: int       # Number of significantly aged channels
    total_channels: int          # Total channel count
    cognitive_impact: float      # Cognitive impact of aging [0, 1]
    repair_activity: float       # Repair activity level


# ============================================================================
# Main Engine
# ============================================================================


class PinchFatigueEngine:
    """
    Pinch Fatigue Engine — Neural aging physics model based on Pollock-Barraclough (1905)

    Core mechanism:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. High current pulse → pinch force F ∝ I² → radial strain ε
    2. ε < ε_yield → elastic (full recovery) = sleep-repairable "fatigue"
    3. ε > ε_yield → plastic (permanent) = aging ★
    4. Plastic accumulation → geometric deformation → Z drift → Γ_aging ≠ 0
    5. Coffin-Manson fatigue law → life prediction
    6. Work hardening → yield strength slightly increases with experience (younger is more fragile?)
    7. Temperature acceleration → Arrhenius (stress/anxiety = high temp → accelerated aging)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Relationship with existing system:
    ・ImpedanceDebtTracker = elastic strain portion (sleep-repairable)
    ・PinchFatigueEngine  = plastic strain portion (permanent aging) ★ NEW
    ・The two are complementary: debt = short-term fatigue, aging = long-term degradation
    """

    def __init__(self):
        # --- Channel State ---
        self._channels: Dict[str, ConductorState] = {}

        # --- Global Statistics ---
        self._tick_count: int = 0
        self._total_pinch_events: int = 0
        self._total_plastic_strain: float = 0.0
        self._total_repair: float = 0.0

        # --- History ---
        self._age_history: List[float] = []
        self._strain_history: List[float] = []

    # ------------------------------------------------------------------
    # Channel Management
    # ------------------------------------------------------------------

    def _get_or_create_channel(self, channel_id: str,
                                 design_z: float = 75.0) -> ConductorState:
        """Get or create channel"""
        if channel_id not in self._channels:
            self._channels[channel_id] = ConductorState(
                channel_id=channel_id,
                design_impedance=design_z,
                current_impedance=design_z,
            )
        return self._channels[channel_id]

    # ------------------------------------------------------------------
    # Core: Pinch Stress Calculation
    # ------------------------------------------------------------------

    def apply_current_pulse(
        self,
        channel_id: str,
        current_intensity: float,
        temperature: float = 0.5,
        design_z: float = 75.0,
    ) -> PinchEvent:
        """
        Apply current pulse to a channel → compute pinch strain → update aging

        Physics:
          F_pinch ∝ μ₀ × I²  (pinch force proportional to current squared)
          ε = F / (E × A)    (strain = force / area × modulus)

          If ε < ε_yield → purely elastic (recoverable)
          If ε ≥ ε_yield → plastic portion = ε - ε_yield (permanent)

        Arrhenius temperature correction:
          High temp (high stress) → yield strength decreases → easier to enter plasticity
          Low temp (relaxed) → yield strength increases → harder to age

        Args:
            channel_id: Channel ID
            current_intensity: Normalized current intensity [0, 1+]
            temperature: Normalized temperature [0, 1] (= ram_temperature / anxiety)
            design_z: Design impedance

        Returns:
            PinchEvent recording the physical quantities of this event
        """
        ch = self._get_or_create_channel(channel_id, design_z)
        self._tick_count += 1
        ch.cycle_count += 1

        # 1. Pinch strain ε = sensitivity × I^exponent
        pinch_strain = PINCH_SENSITIVITY * (current_intensity ** PINCH_EXPONENT)

        # 2. Arrhenius temperature correction — high temp lowers yield strength
        temp_factor = 1.0 + ARRHENIUS_ACTIVATION * (temperature - REFERENCE_TEMP)
        effective_yield = ch.effective_yield / max(temp_factor, 0.5)

        # 3. Elastic/plastic separation
        plastic_increment = 0.0
        was_over_yield = False

        if pinch_strain > effective_yield:
            # Exceeds yield → plastic deformation
            was_over_yield = True
            ch.over_yield_cycles += 1
            plastic_increment = pinch_strain - effective_yield

            # Accumulate plastic strain
            ch.plastic_strain += plastic_increment
            self._total_plastic_strain += plastic_increment

            # Work hardening — plastic deformation makes material "harder"
            # Physics: dislocation density increases → harder to deform subsequently
            hardening = min(
                plastic_increment * WORK_HARDENING_RATE,
                MAX_WORK_HARDENING - ch.work_hardening
            )
            ch.work_hardening += max(0, hardening)
            ch.effective_yield = YIELD_STRAIN + ch.work_hardening

        # 4. Elastic strain (always present, recoverable)
        ch.elastic_strain = min(pinch_strain, effective_yield)
        ch.peak_strain = max(ch.peak_strain, pinch_strain)

        # 5. Coffin-Manson fatigue damage accumulation
        fatigue_increment = 0.0
        if was_over_yield and plastic_increment > 0:
            # D_increment = (Δε_p / C)^β
            fatigue_increment = FATIGUE_DAMAGE_PER_CYCLE * (
                (plastic_increment / (FATIGUE_COEFFICIENT * 0.01)) ** FATIGUE_EXPONENT
            )
            # Temperature acceleration
            fatigue_increment *= temp_factor
            ch.fatigue_damage += fatigue_increment
            ch.fatigue_damage = min(ch.fatigue_damage, MAX_FATIGUE_DAMAGE)

        # 6. Plastic strain → impedance drift
        # Z_coax ∝ ln(r_outer/r_inner)
        # Pinch: r_inner ↓ → ln ↑ → Z ↑
        ch.impedance_drift = min(
            ch.plastic_strain * AGING_IMPEDANCE_SENSITIVITY,
            MAX_IMPEDANCE_DRIFT
        )
        ch.current_impedance = ch.design_impedance * (1.0 + ch.impedance_drift)

        # 7. Compute Γ drift due to aging
        z_load = ch.current_impedance
        z_design = ch.design_impedance
        ch.gamma_drift = abs(z_load - z_design) / (z_load + z_design)

        self._total_pinch_events += 1

        return PinchEvent(
            channel_id=channel_id,
            current_intensity=current_intensity,
            pinch_strain=round(pinch_strain, 6),
            plastic_increment=round(plastic_increment, 6),
            fatigue_increment=round(fatigue_increment, 8),
            was_over_yield=was_over_yield,
            temperature=round(temperature, 3),
        )

    # ------------------------------------------------------------------
    # Repair Mechanism
    # ------------------------------------------------------------------

    def repair_tick(self, is_sleeping: bool = False,
                     growth_factor: float = 0.0):
        """
        Per-tick self-repair

        Physical correspondence:
        ・Elastic strain → natural rebound (per tick)
        ・Plastic strain → extremely slow biological repair (neurotrophic factor BDNF)
        ・Sleep → accelerated repair (only accelerates, doesn't change physical limits)
        ・Fatigue damage → nearly irreversible (true aging)

        This explains why:
        ・Sleep "eliminates fatigue" = elastic rebound (ImpedanceDebt)
        ・But sleep "cannot reverse aging" = plastic irreversibility
        ・BDNF from exercise/socializing can "weakly" slow plastic accumulation
        """
        sleep_factor = SLEEP_REPAIR_BOOST if is_sleeping else 1.0
        bdnf_repair = growth_factor * GROWTH_FACTOR_REPAIR

        for ch in self._channels.values():
            # 1. Elastic strain rebound (fast)
            ch.elastic_strain *= (1.0 - 0.1 * sleep_factor)

            # 2. Plastic strain micro-repair (extremely slow — this is why aging can only be "slowed" not "reversed")
            repair_amount = NEURAL_REPAIR_RATE * sleep_factor + bdnf_repair
            actual_repair = min(repair_amount, ch.plastic_strain * 0.01)
            ch.plastic_strain = max(0, ch.plastic_strain - actual_repair)
            self._total_repair += actual_repair

            # 3. Update impedance drift (reflect latest plastic state)
            ch.impedance_drift = min(
                ch.plastic_strain * AGING_IMPEDANCE_SENSITIVITY,
                MAX_IMPEDANCE_DRIFT
            )
            ch.current_impedance = ch.design_impedance * (1.0 + ch.impedance_drift)
            z_load = ch.current_impedance
            z_design = ch.design_impedance
            ch.gamma_drift = abs(z_load - z_design) / (z_load + z_design)

            # 4. Fatigue damage — almost no repair (core aging mechanism)
            # Physics: once microcracks form, they don't disappear on their own
            # Only extremely minimal biological repair
            ch.fatigue_damage = max(
                0, ch.fatigue_damage - bdnf_repair * 0.01
            )

    # ------------------------------------------------------------------
    # Global Tick (integrated into perceive)
    # ------------------------------------------------------------------

    def tick(
        self,
        channel_activities: Optional[Dict[str, float]] = None,
        temperature: float = 0.5,
        is_sleeping: bool = False,
        growth_factor: float = 0.0,
    ) -> AgingSignal:
        """
        Called per tick — process pinch + repair for all channels

        Args:
            channel_activities: {channel_name: current_intensity} dictionary
            temperature: System temperature (ram_temperature)
            is_sleeping: Whether sleeping
            growth_factor: Neurotrophic factor (BDNF) level [0,1]

        Returns:
            AgingSignal
        """
        self._tick_count += 1

        # 1. Apply pinch stress to each active channel
        if channel_activities and not is_sleeping:
            for channel_id, intensity in channel_activities.items():
                if intensity > 0.05:  # Ignore weak current
                    self.apply_current_pulse(
                        channel_id, intensity, temperature
                    )

        # 2. Repair
        self.repair_tick(is_sleeping, growth_factor)

        # 3. Compute global aging signal
        signal = self._compute_aging_signal()

        # 4. Record history
        self._age_history.append(signal.mean_age_factor)
        self._strain_history.append(signal.total_plastic_strain)
        if len(self._age_history) > MAX_HISTORY:
            self._age_history = self._age_history[-MAX_HISTORY:]
            self._strain_history = self._strain_history[-MAX_HISTORY:]

        return signal

    def _compute_aging_signal(self) -> AgingSignal:
        """Compute global aging signal"""
        channels = list(self._channels.values())
        if not channels:
            return AgingSignal(
                mean_age_factor=0.0, max_age_factor=0.0,
                total_plastic_strain=0.0, mean_impedance_drift=0.0,
                degraded_channels=0, total_channels=0,
                cognitive_impact=0.0, repair_activity=0.0,
            )

        ages = [ch.age_factor for ch in channels]
        drifts = [ch.impedance_drift for ch in channels]
        degraded = sum(1 for ch in channels if ch.is_degraded)

        mean_age = float(np.mean(ages))
        max_age = float(np.max(ages))

        # Cognitive impact = nonlinear function (small aging has little effect, but accelerates sharply with accumulation)
        cognitive_impact = mean_age ** 1.5  # Superlinear

        return AgingSignal(
            mean_age_factor=round(mean_age, 6),
            max_age_factor=round(max_age, 6),
            total_plastic_strain=round(self._total_plastic_strain, 6),
            mean_impedance_drift=round(float(np.mean(drifts)), 6),
            degraded_channels=degraded,
            total_channels=len(channels),
            cognitive_impact=round(cognitive_impact, 6),
            repair_activity=round(self._total_repair, 6),
        )

    # ------------------------------------------------------------------
    # Query Interface
    # ------------------------------------------------------------------

    def get_channel_age(self, channel_id: str) -> float:
        """Get channel aging factor"""
        if channel_id in self._channels:
            return self._channels[channel_id].age_factor
        return 0.0

    def get_gamma_aging(self, channel_id: str) -> float:
        """Get channel Γ drift due to aging"""
        if channel_id in self._channels:
            return self._channels[channel_id].gamma_drift
        return 0.0

    def get_impedance_drift(self, channel_id: str) -> float:
        """Get channel permanent impedance drift"""
        if channel_id in self._channels:
            return self._channels[channel_id].impedance_drift
        return 0.0

    def get_life_expectancy(self, channel_id: str) -> float:
        """
        Estimate channel remaining life ratio [0, 1]

        Based on Miner's rule: failure when Σ(n_i/N_fi) = 1
        """
        if channel_id in self._channels:
            return self._channels[channel_id].structural_integrity
        return 1.0

    def get_channel_state(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Get complete channel state"""
        if channel_id not in self._channels:
            return None
        ch = self._channels[channel_id]
        return {
            "channel_id": ch.channel_id,
            "design_impedance": ch.design_impedance,
            "current_impedance": round(ch.current_impedance, 4),
            "impedance_drift": round(ch.impedance_drift, 6),
            "gamma_drift": round(ch.gamma_drift, 6),
            "elastic_strain": round(ch.elastic_strain, 6),
            "plastic_strain": round(ch.plastic_strain, 6),
            "fatigue_damage": round(ch.fatigue_damage, 6),
            "age_factor": round(ch.age_factor, 6),
            "structural_integrity": round(ch.structural_integrity, 6),
            "cycle_count": ch.cycle_count,
            "over_yield_cycles": ch.over_yield_cycles,
            "work_hardening": round(ch.work_hardening, 6),
            "effective_yield": round(ch.effective_yield, 6),
            "is_degraded": ch.is_degraded,
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Global statistics"""
        channels = list(self._channels.values())
        if not channels:
            return {
                "total_channels": 0, "total_pinch_events": 0,
                "mean_age_factor": 0.0, "total_plastic_strain": 0.0,
                "degraded_channels": 0, "total_repair": 0.0,
            }

        return {
            "total_channels": len(channels),
            "total_pinch_events": self._total_pinch_events,
            "mean_age_factor": round(float(np.mean(
                [ch.age_factor for ch in channels]
            )), 6),
            "max_age_factor": round(float(np.max(
                [ch.age_factor for ch in channels]
            )), 6),
            "total_plastic_strain": round(self._total_plastic_strain, 6),
            "mean_impedance_drift": round(float(np.mean(
                [ch.impedance_drift for ch in channels]
            )), 6),
            "mean_gamma_drift": round(float(np.mean(
                [ch.gamma_drift for ch in channels]
            )), 6),
            "degraded_channels": sum(
                1 for ch in channels if ch.is_degraded
            ),
            "total_repair": round(self._total_repair, 6),
            "tick_count": self._tick_count,
        }

    def get_state(self) -> Dict[str, Any]:
        """Complete engine state (with history)"""
        return {
            **self.get_stats(),
            "age_history": self._age_history[-100:],
            "strain_history": self._strain_history[-100:],
            "channels": {
                ch_id: {
                    "age": round(ch.age_factor, 6),
                    "Z_drift": round(ch.impedance_drift, 6),
                    "plastic": round(ch.plastic_strain, 6),
                    "fatigue": round(ch.fatigue_damage, 6),
                    "integrity": round(ch.structural_integrity, 6),
                }
                for ch_id, ch in self._channels.items()
            },
        }

    def get_waveforms(self) -> Dict[str, List[float]]:
        """Waveform visualization data"""
        return {
            "age_factor": self._age_history[-200:],
            "plastic_strain": self._strain_history[-200:],
        }

    def reset(self):
        """Reset (for testing)"""
        self._channels.clear()
        self._tick_count = 0
        self._total_pinch_events = 0
        self._total_plastic_strain = 0.0
        self._total_repair = 0.0
        self._age_history.clear()
        self._strain_history.clear()
