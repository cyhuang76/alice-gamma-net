# -*- coding: utf-8 -*-
"""
Fontanelle Model — Thermodynamic Window of Neonatal Γ-Field Organization

Paper I §3.5, Eq. 21: The Fontanelle Equation:
  Z_fontanelle = Z_membrane ≪ Z_bone
  Γ_fontanelle < 1

Paper III §2.2: The Fontanelle as Thermodynamic Necessity

Physics:
  If the skull were rigid at birth, expanding neural tissue would encounter
  infinite mechanical impedance (Z_skull → ∞), producing:
    - Total reflection (Γ → 1)
    - Intracranial pressure surge
    - Topological deadlock (craniosynostosis pathophysiology)

  The unfused fontanelle replaces this with an elastic boundary:
    Z_fontanelle = Z_membrane ≪ Z_bone → Γ_fontanelle < 1

  The fontanelle provides:
    1. Degrees of freedom for Γ-field self-organization
    2. Thermal exhaust port for Γ² waste heat during synaptogenesis
    3. Mechanical compliance for brain volume expansion

  In transmission line terms: a matched termination for the brain's
  mechanical expansion waves.

Developmental Timeline:
  Birth          → fontanelle wide open (maximum compliance)
  6 months       → anterior fontanelle narrowing
  12-18 months   → anterior fontanelle closing
  18-36 months   → fontanelle fully closed → infantile amnesia ends → ego born

  Closure coincides with:
    - End of infantile amnesia
    - Specialization index crossing critical threshold
    - Topological freeze
    - Narrative self emergence

The Pressure Chamber Effect (Paper III §2.5):
  After closure, Γ² waste heat previously dissipated through the fontanelle
  is trapped → constructive consumption → childhood cognitive explosion.

Clinical correspondence:
  - Craniosynostosis: premature fontanelle closure → Z → ∞ → Γ → 1 → intracranial pressure
  - Microcephaly: insufficient expansion → developmental limit
  - Hydrocephalus: CSF accumulation → Z_fontanelle pathological increase

"The soft spot is the thermodynamic window through which the mind first finds its shape."

Author: Hsi-Yu Huang (黃璽宇)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from alice.core.signal import ElectricalSignal


# ============================================================================
# Physical Constants
# ============================================================================

# --- Impedance values ---
Z_MEMBRANE = 5.0                # Fontanelle membrane impedance (very low, allows expansion)
Z_BONE = 500.0                  # Mature skull bone impedance (very high, rigid)
Z_SKULL_INFANT = 10.0           # Infant skull impedance (partially ossified)
Z_SKULL_TODDLER = 100.0         # Toddler skull impedance (mostly ossified)
Z_SKULL_CHILD = 400.0           # Child skull impedance (nearly adult)

# --- Closure dynamics ---
CLOSURE_RATE_BASE = 0.001       # Base closure rate per tick
CLOSURE_ACCELERATION = 0.0001   # Closure rate increases as specialization improves
CLOSURE_THRESHOLD = 0.95        # Z_ratio at which fontanelle is considered closed

# --- Thermal dissipation ---
HEAT_DISSIPATION_OPEN = 0.8     # Fraction of Γ² heat dissipated when fontanelle is open
HEAT_DISSIPATION_CLOSED = 0.1   # Much less dissipation when closed
HEAT_DISSIPATION_TRANSITION = 0.4  # During closing

# --- Specialization threshold for closure ---
SPECIALIZATION_CLOSURE_THRESHOLD = 0.6  # Specialization index above this → closure begins
SPECIALIZATION_LOCK_THRESHOLD = 0.85    # Above this → full closure (topological freeze)

# --- Pressure chamber effect ---
PRESSURE_CHAMBER_BOOST = 1.5    # Cognitive acceleration factor after closure
PRESSURE_CHAMBER_ONSET = 0.8    # Closure fraction above which pressure chamber activates

# --- Stage mapping ---
STAGE_Z_MAP = {
    "neonate": Z_MEMBRANE,        # Wide open
    "infant": Z_SKULL_INFANT,     # Narrowing
    "toddler": Z_SKULL_TODDLER,  # Nearly closed
    "child": Z_SKULL_CHILD,       # Closed
}


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class FontanelleState:
    """Snapshot of fontanelle state."""
    z_fontanelle: float             # Current fontanelle impedance
    z_bone: float                   # Reference bone impedance
    gamma_fontanelle: float         # Γ = |Z_f - Z_brain| / (Z_f + Z_brain)
    transmission: float             # T = 1 − Γ² (energy conservation)
    closure_fraction: float         # 0=wide open, 1=fully closed
    is_closed: bool                 # Whether considered closed
    heat_dissipation_rate: float    # Current thermal dissipation efficiency
    pressure_chamber_active: bool   # Whether pressure chamber effect is active
    topological_frozen: bool        # Whether topology is locked


# ============================================================================
# FontanelleModel
# ============================================================================

class FontanelleModel:
    """
    Fontanelle Model — the thermodynamic window of neonatal Γ-field self-organization.

    Implements:
    - Eq. 21: Z_fontanelle = Z_membrane ≪ Z_bone → Γ_fontanelle < 1
    - Developmental closure dynamics tied to specialization index
    - Thermal exhaust: Γ² waste heat dissipation through fontanelle
    - Pressure chamber effect after closure
    - Topological freeze detection
    """

    def __init__(self, developmental_stage: str = "neonate") -> None:
        # Current fontanelle impedance (starts at Z_membrane)
        self._z_fontanelle: float = STAGE_Z_MAP.get(developmental_stage, Z_MEMBRANE)

        # Target bone impedance
        self._z_bone: float = Z_BONE

        # Closure state
        self._closure_fraction: float = 0.0  # 0=open, 1=closed
        if developmental_stage == "child":
            self._closure_fraction = 1.0
        elif developmental_stage == "toddler":
            self._closure_fraction = 0.7
        elif developmental_stage == "infant":
            self._closure_fraction = 0.3

        # Thermal exhaust
        self._heat_dissipation_rate: float = HEAT_DISSIPATION_OPEN
        self._cumulative_heat_dissipated: float = 0.0

        # Pressure chamber
        self._pressure_chamber_active: bool = False
        self._cognitive_boost: float = 1.0

        # Topology
        self._topological_frozen: bool = developmental_stage in ("child",)

        # Statistics
        self._tick_count: int = 0

    # ------------------------------------------------------------------
    # Core Equation: Γ_fontanelle = |Z_fontanelle - Z_brain| / (Z_fontanelle + Z_brain)
    # ------------------------------------------------------------------

    def compute_gamma(self, z_brain: float = 75.0) -> float:
        """
        Compute fontanelle reflection coefficient.

        Γ_fontanelle = |Z_fontanelle - Z_brain| / (Z_fontanelle + Z_brain)

        When fontanelle is open (Z_f ≈ Z_membrane ≪ Z_bone):
          Γ is low → allows expansion and reorganization.

        When fontanelle is closed (Z_f ≈ Z_bone):
          Γ is high → no further large-scale reorganization.
        """
        z_f = self._z_fontanelle
        denom = z_f + z_brain
        if denom == 0:
            return 0.0
        return abs(z_f - z_brain) / denom

    # ------------------------------------------------------------------
    # Tick — Advance fontanelle dynamics
    # ------------------------------------------------------------------

    def tick(
        self,
        specialization_index: float = 0.0,
        gamma_sq_heat: float = 0.0,
    ) -> FontanelleState:
        """
        Advance fontanelle dynamics by one tick.

        Args:
            specialization_index: 0~1 from NeuralPruningEngine (cortical specialization)
            gamma_sq_heat: Current Γ² waste heat to be dissipated

        Returns:
            FontanelleState snapshot
        """
        self._tick_count += 1

        # ==========================================
        # Closure dynamics
        # ==========================================
        if not self._topological_frozen:
            if specialization_index > SPECIALIZATION_CLOSURE_THRESHOLD:
                # Begin closure: rate proportional to specialization above threshold
                closure_rate = CLOSURE_RATE_BASE + CLOSURE_ACCELERATION * (
                    specialization_index - SPECIALIZATION_CLOSURE_THRESHOLD
                )
                self._closure_fraction = min(1.0, self._closure_fraction + closure_rate)

            if specialization_index > SPECIALIZATION_LOCK_THRESHOLD:
                self._closure_fraction = min(1.0, self._closure_fraction + CLOSURE_RATE_BASE * 5)

            # Topological freeze at full closure
            if self._closure_fraction >= CLOSURE_THRESHOLD:
                self._topological_frozen = True

        # Update Z_fontanelle based on closure fraction
        # Linear interpolation: Z_membrane → Z_bone
        self._z_fontanelle = Z_MEMBRANE + self._closure_fraction * (Z_BONE - Z_MEMBRANE)

        # ★ Thermal dissipation via Γ² + T = 1 physics
        # The fontanelle is a thermal exhaust port: heat that TRANSMITS through
        # the fontanelle boundary escapes. T_fontanelle determines dissipation.
        gamma = self.compute_gamma()
        transmission = 1.0 - gamma ** 2  # T = 1 − Γ²

        if self._closure_fraction < 0.3:
            self._heat_dissipation_rate = HEAT_DISSIPATION_OPEN
        elif self._closure_fraction < 0.8:
            self._heat_dissipation_rate = HEAT_DISSIPATION_TRANSITION
        else:
            self._heat_dissipation_rate = HEAT_DISSIPATION_CLOSED

        # Physics-based: dissipation = heat × T × structural openness
        heat_dissipated = gamma_sq_heat * transmission * (1.0 - self._closure_fraction * 0.7)
        self._cumulative_heat_dissipated += heat_dissipated

        # ==========================================
        # Pressure chamber effect
        # ==========================================
        if self._closure_fraction > PRESSURE_CHAMBER_ONSET and not self._pressure_chamber_active:
            self._pressure_chamber_active = True
            self._cognitive_boost = PRESSURE_CHAMBER_BOOST

        # Return state
        return FontanelleState(
            z_fontanelle=round(self._z_fontanelle, 2),
            z_bone=self._z_bone,
            gamma_fontanelle=round(gamma, 4),
            transmission=round(transmission, 4),
            closure_fraction=round(self._closure_fraction, 4),
            is_closed=self._closure_fraction >= CLOSURE_THRESHOLD,
            heat_dissipation_rate=round(self._heat_dissipation_rate, 4),
            pressure_chamber_active=self._pressure_chamber_active,
            topological_frozen=self._topological_frozen,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_heat_dissipation(self, gamma_sq_heat: float) -> float:
        """How much Γ² heat can the fontanelle dissipate right now?"""
        return gamma_sq_heat * self._heat_dissipation_rate

    def get_cognitive_boost(self) -> float:
        """Pressure chamber cognitive acceleration factor."""
        return self._cognitive_boost if self._pressure_chamber_active else 1.0

    def is_closed(self) -> bool:
        """Whether the fontanelle is considered closed."""
        return self._closure_fraction >= CLOSURE_THRESHOLD

    def get_state(self) -> Dict[str, Any]:
        """Full state for introspection."""
        gamma = self.compute_gamma()
        return {
            "z_fontanelle": round(self._z_fontanelle, 2),
            "gamma_fontanelle": round(gamma, 4),
            "closure_fraction": round(self._closure_fraction, 4),
            "is_closed": self.is_closed(),
            "heat_dissipation_rate": round(self._heat_dissipation_rate, 4),
            "pressure_chamber_active": self._pressure_chamber_active,
            "cognitive_boost": round(self._cognitive_boost, 2),
            "topological_frozen": self._topological_frozen,
            "cumulative_heat_dissipated": round(self._cumulative_heat_dissipated, 4),
            "tick_count": self._tick_count,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Alias for get_state."""
        return self.get_state()

    # ------------------------------------------------------------------
    # Signal Protocol
    # ------------------------------------------------------------------

    def get_signal(self) -> ElectricalSignal:
        """Generate ElectricalSignal encoding fontanelle state."""
        gamma = self.compute_gamma()
        amplitude = float(np.clip(1.0 - self._closure_fraction, 0.01, 1.0))
        freq = 1.0 + gamma * 10.0  # Low freq (developmental)
        t = np.linspace(0, 1, 64)
        waveform = amplitude * np.sin(2 * np.pi * freq * t)
        return ElectricalSignal(
            waveform=waveform.astype(np.float32),
            amplitude=amplitude,
            frequency=freq,
            phase=0.0,
            impedance=self._z_fontanelle,
            snr=10.0,
            source="fontanelle",
            modality="internal",
        )
