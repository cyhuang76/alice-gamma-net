# -*- coding: utf-8 -*-
"""
Gradient Optimizer — Explicit Variational Gradient for MRP

Paper I Eq. 11: The Gradient Equation
  ∇W = −∂(ΣΓ²)/∂Z

  "The arrow of growth always points toward less reflection."

Physics:
  The Minimum Reflection Principle (MRP) states ΣΓ²→min.
  The Gradient Equation makes this explicit: at every impedance junction,
  the optimal direction of impedance adjustment is the negative gradient
  of the total reflected energy with respect to each channel's impedance.

  For a channel with impedance Z_i and load impedance Z_L,i:
    Γ_i = (Z_L,i - Z_i) / (Z_L,i + Z_i)
    Γ_i² = ((Z_L,i - Z_i) / (Z_L,i + Z_i))²
    
    ∂(Γ_i²)/∂Z_i = -4·Z_L,i·(Z_L,i - Z_i) / (Z_L,i + Z_i)³

  The gradient descent rule:
    ΔZ_i = -η · ∂(ΣΓ_i²)/∂Z_i
    
  This drives Z_i → Z_L,i (impedance matching → Γ → 0).

  The existing system uses Hebbian learning, PID control, and various
  local optimization rules that *implicitly* achieve gradient descent on ΣΓ².
  This module makes the variational optimization *explicit*.

Implementation:
  The GradientOptimizer collects (Z_channel, Z_load) pairs from all
  active channels, computes the analytic gradient, and produces
  impedance adjustment suggestions that any module can apply.

"Not just minimizing Γ² — knowing exactly which direction to adjust."

Author: Hsi-Yu Huang (黃璽宇)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alice.core.signal import ElectricalSignal


# ============================================================================
# Physical Constants
# ============================================================================

# --- Gradient descent parameters ---
ETA_GRADIENT = 0.01             # Base learning rate for gradient descent
ETA_MAX = 0.05                  # Maximum learning rate (aggressive matching)
ETA_MIN = 0.001                 # Minimum learning rate (fine-tuning)
MOMENTUM = 0.9                  # Momentum coefficient for smoother convergence
GRADIENT_CLIP = 10.0            # Maximum gradient magnitude (prevents explosion)

# --- Regularization ---
L2_REGULARIZATION = 0.001       # Small L2 penalty to prevent impedance drift to extremes
Z_MIN = 1.0                    # Minimum allowed impedance
Z_MAX = 1000.0                 # Maximum allowed impedance

# --- History ---
MAX_HISTORY = 500


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ChannelGradient:
    """Gradient information for a single channel."""
    channel_id: str
    z_channel: float            # Channel impedance Z_i
    z_load: float               # Load impedance Z_L,i
    gamma: float                # Current Γ_i
    gamma_sq: float             # Current Γ_i²
    transmission: float         # T_i = 1 − Γ_i² (energy conservation)
    d_gamma_sq_dz: float        # ∂(Γ²)/∂Z_i — the gradient
    delta_z: float              # Recommended ΔZ_i (negative gradient direction)


@dataclass
class GradientStep:
    """Result of one gradient optimization step."""
    sigma_gamma_sq: float       # Total ΣΓ²
    gradient_norm: float        # ||∇(ΣΓ²)||
    mean_delta_z: float         # Mean recommended impedance adjustment
    channels_updated: int       # Number of channels with non-zero gradient
    converged: bool             # Whether gradient norm is below threshold


# ============================================================================
# GradientOptimizer
# ============================================================================

class GradientOptimizer:
    """
    Explicit variational gradient optimizer for the Minimum Reflection Principle.

    Computes ∇W = −∂(ΣΓ²)/∂Z for all channels and produces impedance
    adjustment recommendations.

    The gradient for channel i:
      ∂(Γ_i²)/∂Z_i = -4·Z_L,i·(Z_L,i - Z_i) / (Z_L,i + Z_i)³

    This drives impedance matching: Z_i → Z_L,i → Γ_i → 0.
    """

    def __init__(self, eta: float = ETA_GRADIENT) -> None:
        self._eta: float = eta
        self._momentum_buffer: Dict[str, float] = {}  # Channel → velocity
        self._tick_count: int = 0

        # Statistics
        self._total_steps: int = 0
        self._sigma_history: List[float] = []
        self._gradient_norm_history: List[float] = []

    # ------------------------------------------------------------------
    # Core: Analytic gradient computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_gamma(z_channel: float, z_load: float) -> float:
        """Γ = (Z_L - Z) / (Z_L + Z)"""
        denom = z_load + z_channel
        if denom == 0:
            return 0.0
        return (z_load - z_channel) / denom

    @staticmethod
    def compute_gamma_sq_gradient(z_channel: float, z_load: float) -> float:
        """
        Analytic gradient: ∂(Γ²)/∂Z_i

        Derivation:
          Γ = (Z_L - Z) / (Z_L + Z)
          Γ² = (Z_L - Z)² / (Z_L + Z)²
          
          Let u = Z_L - Z, v = Z_L + Z
          Γ² = u²/v²
          
          d(Γ²)/dZ = d/dZ [(Z_L - Z)² / (Z_L + Z)²]
                    = [2(Z_L - Z)(-1)(Z_L + Z)² - (Z_L - Z)²·2(Z_L + Z)] / (Z_L + Z)⁴
                    = -2(Z_L - Z)(Z_L + Z)[Z_L + Z + Z_L - Z] / (Z_L + Z)⁴
                    = -2(Z_L - Z)·2Z_L / (Z_L + Z)³
                    = -4·Z_L·(Z_L - Z) / (Z_L + Z)³
        """
        denom = z_load + z_channel
        if denom == 0:
            return 0.0
        return -4.0 * z_load * (z_load - z_channel) / (denom ** 3)

    # ------------------------------------------------------------------
    # Gradient step for all channels
    # ------------------------------------------------------------------

    def step(
        self,
        channels: Dict[str, Tuple[float, float]],
        eta: Optional[float] = None,
    ) -> Tuple[GradientStep, List[ChannelGradient]]:
        """
        Perform one gradient descent step on all channels.

        Args:
            channels: dict of channel_id → (Z_channel, Z_load)
            eta: Optional override for learning rate

        Returns:
            (GradientStep summary, list of per-channel gradients)
        """
        self._tick_count += 1
        self._total_steps += 1
        lr = eta if eta is not None else self._eta

        gradients: List[ChannelGradient] = []
        sigma_gamma_sq = 0.0
        gradient_norm_sq = 0.0

        for ch_id, (z_ch, z_ld) in channels.items():
            gamma = self.compute_gamma(z_ch, z_ld)
            gamma_sq = gamma ** 2
            transmission = 1.0 - gamma_sq  # ★ Energy conservation: T = 1 − Γ²
            sigma_gamma_sq += gamma_sq

            # Analytic gradient: ∂(Γ²)/∂Z = −4·Z_L·(Z_L−Z)/(Z_L+Z)³
            # This contains Γ implicitly: ∂(Γ²)/∂Z = 2Γ·∂Γ/∂Z (Hebbian-compatible)
            d_gamma_sq_dz = self.compute_gamma_sq_gradient(z_ch, z_ld)

            # Clip gradient
            d_gamma_sq_dz = float(np.clip(d_gamma_sq_dz, -GRADIENT_CLIP, GRADIENT_CLIP))

            # Add L2 regularization: pulls Z toward neutral range
            z_neutral = 75.0  # Standard characteristic impedance
            l2_grad = L2_REGULARIZATION * (z_ch - z_neutral)
            d_gamma_sq_dz += l2_grad

            # Momentum
            velocity = self._momentum_buffer.get(ch_id, 0.0)
            velocity = MOMENTUM * velocity + (1 - MOMENTUM) * d_gamma_sq_dz
            self._momentum_buffer[ch_id] = velocity

            # ΔZ weighted by T: only transmitted energy drives impedance adjustment
            delta_z = -lr * velocity * max(0.1, transmission)
            delta_z = float(np.clip(delta_z, -50.0, 50.0))  # Limit step size

            gradient_norm_sq += d_gamma_sq_dz ** 2

            gradients.append(ChannelGradient(
                channel_id=ch_id,
                z_channel=round(z_ch, 2),
                z_load=round(z_ld, 2),
                gamma=round(gamma, 6),
                gamma_sq=round(gamma_sq, 6),
                transmission=round(transmission, 6),
                d_gamma_sq_dz=round(d_gamma_sq_dz, 8),
                delta_z=round(delta_z, 4),
            ))

        gradient_norm = math.sqrt(gradient_norm_sq)
        mean_delta_z = np.mean([g.delta_z for g in gradients]) if gradients else 0.0

        # History
        self._sigma_history.append(sigma_gamma_sq)
        self._gradient_norm_history.append(gradient_norm)
        if len(self._sigma_history) > MAX_HISTORY:
            self._sigma_history = self._sigma_history[-MAX_HISTORY:]
            self._gradient_norm_history = self._gradient_norm_history[-MAX_HISTORY:]

        step_result = GradientStep(
            sigma_gamma_sq=round(sigma_gamma_sq, 6),
            gradient_norm=round(gradient_norm, 8),
            mean_delta_z=round(float(mean_delta_z), 6),
            channels_updated=len([g for g in gradients if abs(g.delta_z) > 0.001]),
            converged=gradient_norm < 0.01,
        )

        return step_result, gradients

    # ------------------------------------------------------------------
    # Convenience: compute ΣΓ² without stepping
    # ------------------------------------------------------------------

    def compute_sigma_gamma_sq(self, channels: Dict[str, Tuple[float, float]]) -> float:
        """Compute total ΣΓ² for given channels."""
        total = 0.0
        for z_ch, z_ld in channels.values():
            gamma = self.compute_gamma(z_ch, z_ld)
            total += gamma ** 2
        return total

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Full state for introspection."""
        return {
            "eta": self._eta,
            "total_steps": self._total_steps,
            "tick_count": self._tick_count,
            "momentum_channels": len(self._momentum_buffer),
            "last_sigma": self._sigma_history[-1] if self._sigma_history else 0.0,
            "last_gradient_norm": self._gradient_norm_history[-1] if self._gradient_norm_history else 0.0,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Alias for get_state."""
        return self.get_state()

    # ------------------------------------------------------------------
    # Signal Protocol
    # ------------------------------------------------------------------

    def get_signal(self) -> ElectricalSignal:
        """Generate ElectricalSignal encoding the gradient landscape."""
        sigma = self._sigma_history[-1] if self._sigma_history else 0.5
        amplitude = float(np.clip(sigma, 0.01, 1.0))
        freq = 2.0 + sigma * 8.0
        t = np.linspace(0, 1, 64)
        waveform = amplitude * np.sin(2 * np.pi * freq * t)
        return ElectricalSignal(
            waveform=waveform.astype(np.float32),
            amplitude=amplitude,
            frequency=freq,
            phase=0.0,
            impedance=75.0,
            snr=12.0,
            source="gradient_optimizer",
            modality="internal",
        )

    def get_convergence_curve(self, last_n: int = 200) -> Dict[str, List[float]]:
        """Get convergence data for plotting."""
        return {
            "sigma_gamma_sq": self._sigma_history[-last_n:],
            "gradient_norm": self._gradient_norm_history[-last_n:],
        }
