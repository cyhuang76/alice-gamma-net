# -*- coding: utf-8 -*-
"""
Thalamus — Sensory Gate & Attentional Router (Phase 5.3)

Physics:
  "The thalamus is the brain's telephone switchboard.
   All sensory signals (except olfaction) must pass through the thalamus before reaching cortex.
   The thalamus is not just a relay—it is an active gate controller."

  Each sensory channel has a gate gain G ∈ [0, 1]:
    - G = 1.0: fully passed (attention focused)
    - G = 0.0: fully blocked (attention ignored / sleep gating)
    - 0 < G < 1: partially attenuated (background channel)

  Gate gain is determined by three factors:
    1. Attentional focus (top-down): selective attention from prefrontal cortex / consciousness module
    2. Salience (bottom-up): signal's own prominence (high Γ, high amplitude)
    3. Arousal: sleep→wakefulness is global modulation

  G_total = G_arousal × (α × G_topdown + (1-α) × G_bottomup)
  α = balance parameter between top-down vs bottom-up

Circuit analogy:
  Thalamus = multiplexer (MUX) + variable gain amplifier (VGA)
  Each channel has independent gain control
  Sleep = global enable signal (EN) pulled low
  Startle = low-level interrupt (IRQ) unconditional pass-through

Core concepts:
  - Thalamic reticular nucleus (TRN): inhibitory gate, increases inter-channel competition
  - Gate oscillation: α waves (8-13 Hz) = "idle" gate pulses, β/γ waves = "open" gate
  - Attention bottleneck: at most ≤3 channels can be fully open simultaneously
  - Thalamic burst: burst firing at low arousal → discontinuous signaling
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from alice.brain.attention_plasticity import AttentionPlasticityEngine


# ============================================================================
# Physical constants
# ============================================================================

# Gate gain bounds
GATE_MIN = 0.05           # Even fully closed has slight leakage (biological realism)
GATE_MAX = 1.0            # Maximum gain

# Top-down vs Bottom-up balance
ALPHA_DEFAULT = 0.6       # Default biased toward top-down (adult brain)
ALPHA_INFANT = 0.2        # Infant biased toward bottom-up (stimulus-driven)

# Salience thresholds
SALIENCE_GAMMA_WEIGHT = 0.4     # Γ contribution to salience
SALIENCE_AMPLITUDE_WEIGHT = 0.3  # Amplitude contribution to salience
SALIENCE_NOVELTY_WEIGHT = 0.3    # Novelty contribution to salience

# Attention bottleneck
MAX_FOCUSED_CHANNELS = 3   # Upper limit of simultaneously fully focused channels

# Thalamic burst threshold
BURST_AROUSAL_THRESHOLD = 0.3   # Arousal below this → thalamus enters burst mode
BURST_PROBABILITY = 0.3         # Probability of signal passing in burst mode

# Adaptation (habituation)
HABITUATION_RATE = 0.02         # Sustained same stimulus → gate gradually closes
HABITUATION_RECOVERY = 0.01    # Recovery rate after stimulus stops

# Gate smoothing
GATE_MOMENTUM = 0.3             # EMA smoothing coefficient (gates don't instantly open/close)

# Thalamic impedance
THALAMIC_IMPEDANCE = 50.0      # Ω (standard matching impedance)

# Startle circuit — unconditional pass-through
STARTLE_AMPLITUDE_THRESHOLD = 0.8  # Signals above this amplitude unconditionally pass through
STARTLE_GAMMA_THRESHOLD = 0.85     # Signals above this Γ also unconditionally pass through


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class SensoryChannel:
    """
    Gate state for a single sensory channel.

    Each sensory channel (visual, auditory, tactile...) has an independent gate in the thalamus.
    """
    modality: str                          # "visual", "auditory", "tactile", ...
    gate_gain: float = 0.5                 # Current gate gain G ∈ [GATE_MIN, GATE_MAX]
    topdown_bias: float = 0.5             # Top-down attention bias
    habituation: float = 0.0              # Habituation level (0=fresh, 1=fully habituated)
    last_salience: float = 0.0            # Previous salience value
    last_signal_hash: int = 0             # Coarse hash of previous signal (for novelty)
    signal_count: int = 0                 # Count of passed signals
    blocked_count: int = 0                # Count of blocked signals
    total_count: int = 0                  # Total signal count
    _last_fingerprint: Optional[np.ndarray] = None  # Previous signal fingerprint


@dataclass
class ThalamicGateResult:
    """
    Result of thalamic gate processing.

    Tells downstream modules: whether the signal passed, the gain applied, and whether startle was triggered.
    """
    modality: str
    passed: bool                  # Whether the signal passed the gate
    gate_gain: float              # Actual applied gain
    salience: float               # Signal salience score
    is_startle: bool              # Whether startle circuit was triggered
    is_burst: bool                # Whether in thalamic burst mode
    gated_fingerprint: Optional[np.ndarray] = None  # Gain-modulated fingerprint
    reason: str = ""              # Reason for gate decision


# ============================================================================
# Main engine
# ============================================================================


class ThalamusEngine:
    """
    Thalamus Engine — Sensory gate + attentional router.

    All sensory signals must pass through the thalamus before entering the brain:
    1. Compute signal salience (bottom-up)
    2. Receive top-down attention bias
    3. Combine with arousal to compute gate gain
    4. Startle circuit unconditional pass-through
    5. Habituation (sustained same stimulus → automatic ignoring)
    6. Thalamic burst (random gating at low arousal)
    """

    def __init__(self, alpha: float = ALPHA_DEFAULT):
        # Top-down vs Bottom-up balance
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

        # Channels for each modality
        self._channels: Dict[str, SensoryChannel] = {}

        # Global arousal modulation
        self._arousal: float = 0.8  # Default: awake

        # Attention bottleneck tracking
        self._focused_modalities: List[str] = []

        # Attention plasticity engine (optional external injection)
        self._plasticity: Optional["AttentionPlasticityEngine"] = None

        # Statistics
        self._total_gated: int = 0
        self._total_passed: int = 0
        self._total_blocked: int = 0
        self._total_startles: int = 0
        self._total_bursts: int = 0

    def set_plasticity_engine(self, engine: "AttentionPlasticityEngine"):
        """Inject attention plasticity engine — makes gate speed trainable."""
        self._plasticity = engine

    # ------------------------------------------------------------------
    # Channel management
    # ------------------------------------------------------------------

    def _ensure_channel(self, modality: str) -> SensoryChannel:
        """Ensure the modality channel exists; auto-create if not."""
        if modality not in self._channels:
            self._channels[modality] = SensoryChannel(modality=modality)
        return self._channels[modality]

    # ------------------------------------------------------------------
    # Core gate operation
    # ------------------------------------------------------------------

    def gate(
        self,
        modality: str,
        fingerprint: Optional[np.ndarray] = None,
        amplitude: float = 0.5,
        gamma: float = 0.5,
        arousal: Optional[float] = None,
    ) -> ThalamicGateResult:
        """
        Thalamic gate — determines whether a signal passes and at what gain.

        Physical process:
        1. Compute bottom-up salience S_bottom = f(Γ, amplitude, novelty)
        2. Combine with top-down bias → G_combined
        3. Global arousal modulation → G_total
        4. Startle circuit check (high amplitude/high Γ unconditional pass-through)
        5. Thalamic burst mode (random gating at low arousal)
        6. Habituation (repeated stimulus attenuation)

        Args:
            modality: Sensory modality name
            fingerprint: Sensory fingerprint vector (optional)
            amplitude: Signal amplitude (0~1)
            gamma: Signal impedance mismatch Γ (0~1)
            arousal: Arousal override (None=use global value)

        Returns:
            ThalamicGateResult
        """
        ch = self._ensure_channel(modality)
        ch.total_count += 1
        self._total_gated += 1

        current_arousal = arousal if arousal is not None else self._arousal

        # --- 1. Startle circuit: unconditional pass-through ---
        is_startle = (
            amplitude >= STARTLE_AMPLITUDE_THRESHOLD or
            gamma >= STARTLE_GAMMA_THRESHOLD
        )
        if is_startle:
            self._total_startles += 1
            ch.signal_count += 1
            self._total_passed += 1
            ch.habituation = max(0.0, ch.habituation - 0.1)  # Startle breaks habituation
            gated_fp = fingerprint  # No attenuation
            return ThalamicGateResult(
                modality=modality,
                passed=True,
                gate_gain=GATE_MAX,
                salience=1.0,
                is_startle=True,
                is_burst=False,
                gated_fingerprint=gated_fp,
                reason="startle_bypass",
            )

        # --- 2. Compute bottom-up salience ---
        # Novelty: difference from previous fingerprint
        novelty = 1.0
        if fingerprint is not None and ch._last_fingerprint is not None:
            if fingerprint.shape == ch._last_fingerprint.shape:
                fp_norm = np.linalg.norm(fingerprint)
                prev_norm = np.linalg.norm(ch._last_fingerprint)
                if fp_norm > 1e-10 and prev_norm > 1e-10:
                    cos_sim = float(np.dot(fingerprint, ch._last_fingerprint) / (fp_norm * prev_norm))
                    novelty = 1.0 - max(0.0, cos_sim)

        salience_bottom = (
            SALIENCE_GAMMA_WEIGHT * gamma +
            SALIENCE_AMPLITUDE_WEIGHT * amplitude +
            SALIENCE_NOVELTY_WEIGHT * novelty
        )
        salience_bottom = float(np.clip(salience_bottom, 0.0, 1.0))

        # --- 3. Top-down + Bottom-up synthesis ---
        g_topdown = ch.topdown_bias
        g_combined = self.alpha * g_topdown + (1.0 - self.alpha) * salience_bottom

        # --- 4. Habituation attenuation ---
        # Sustained same stimulus → gate gradually decreases
        if novelty < 0.2:
            ch.habituation = min(1.0, ch.habituation + HABITUATION_RATE)
        else:
            ch.habituation = max(0.0, ch.habituation - HABITUATION_RECOVERY)

        habituation_factor = 1.0 - ch.habituation * 0.6  # Max 60% attenuation

        # --- 5. Global arousal modulation ---
        g_total = current_arousal * g_combined * habituation_factor

        # --- 6. Thalamic burst mode ---
        is_burst = False
        if current_arousal < BURST_AROUSAL_THRESHOLD:
            is_burst = True
            self._total_bursts += 1
            # Low arousal → signal randomly passes or blocks
            if np.random.random() > BURST_PROBABILITY:
                # Blocked in burst mode
                ch.blocked_count += 1
                self._total_blocked += 1
                return ThalamicGateResult(
                    modality=modality,
                    passed=False,
                    gate_gain=0.0,
                    salience=salience_bottom,
                    is_startle=False,
                    is_burst=True,
                    reason="burst_mode_blocked",
                )

        # --- 7. Gate gain smoothing (EMA) ---
        # gate_momentum can be dynamically adjusted by attention plasticity engine
        gate_momentum = GATE_MOMENTUM
        if self._plasticity is not None:
            gate_momentum = self._plasticity.get_gate_tau(modality)
        target_gain = float(np.clip(g_total, GATE_MIN, GATE_MAX))
        ch.gate_gain += (target_gain - ch.gate_gain) * gate_momentum
        ch.gate_gain = float(np.clip(ch.gate_gain, GATE_MIN, GATE_MAX))

        # --- 8. Apply gain to fingerprint ---
        gated_fp = None
        if fingerprint is not None:
            gated_fp = fingerprint * ch.gate_gain
            ch._last_fingerprint = fingerprint.copy()

        # --- 9. Determine pass/block ---
        passed = ch.gate_gain > GATE_MIN * 1.5  # Slightly above minimum leakage
        ch.last_salience = salience_bottom

        if passed:
            ch.signal_count += 1
            self._total_passed += 1
            reason = "gate_open"
        else:
            ch.blocked_count += 1
            self._total_blocked += 1
            reason = "gate_closed"

        return ThalamicGateResult(
            modality=modality,
            passed=passed,
            gate_gain=round(ch.gate_gain, 4),
            salience=round(salience_bottom, 4),
            is_startle=False,
            is_burst=is_burst,
            gated_fingerprint=gated_fp,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Top-down attention control
    # ------------------------------------------------------------------

    def set_attention(self, modality: str, bias: float):
        """
        Set top-down attention bias.

        Prefrontal cortex / consciousness module tells the thalamus: "pay attention to this channel."

        Args:
            modality: Modality to focus on
            bias: Bias strength (0=ignore, 1=fully focused)
        """
        ch = self._ensure_channel(modality)
        ch.topdown_bias = float(np.clip(bias, 0.0, 1.0))

        # Attention bottleneck: max N channels focused simultaneously (can be expanded via training)
        max_channels = MAX_FOCUSED_CHANNELS
        if self._plasticity is not None:
            max_channels = self._plasticity.get_attention_slots()
        if bias > 0.5:
            if modality not in self._focused_modalities:
                self._focused_modalities.append(modality)
            # Exceeds bottleneck → evict the earliest focused
            while len(self._focused_modalities) > max_channels:
                evicted = self._focused_modalities.pop(0)
                if evicted in self._channels:
                    self._channels[evicted].topdown_bias *= 0.5
        else:
            if modality in self._focused_modalities:
                self._focused_modalities.remove(modality)

    def set_arousal(self, arousal: float):
        """
        Set global arousal level.

        Arousal affects gate gain for all channels:
        - arousal = 1.0: fully awake, gates responsive
        - arousal = 0.0: deep sleep, gates nearly all closed
        """
        self._arousal = float(np.clip(arousal, 0.0, 1.0))

    # ------------------------------------------------------------------
    # TRN competitive inhibition
    # ------------------------------------------------------------------

    def apply_trn_inhibition(self):
        """
        Thalamic reticular nucleus (TRN) competitive inhibition.

        TRN makes channels compete with each other: focused channels gain boost,
        unfocused channels gain suppressed.
        This is the physical basis of "attention exclusivity".
        """
        if not self._focused_modalities:
            return

        for modality, ch in self._channels.items():
            if modality in self._focused_modalities:
                # Focused channel: slight boost
                ch.topdown_bias = min(1.0, ch.topdown_bias + 0.05)
            else:
                # Unfocused channel: TRN inhibition
                ch.topdown_bias = max(0.0, ch.topdown_bias - 0.03)

    # ------------------------------------------------------------------
    # Batch gating
    # ------------------------------------------------------------------

    def gate_all(
        self,
        signals: Dict[str, Dict[str, Any]],
        arousal: Optional[float] = None,
    ) -> Dict[str, ThalamicGateResult]:
        """
        Execute gating on all sensory channels simultaneously.

        Args:
            signals: {modality: {"fingerprint": ..., "amplitude": ..., "gamma": ...}}
            arousal: Arousal override

        Returns:
            {modality: ThalamicGateResult}
        """
        # Execute TRN competition first
        self.apply_trn_inhibition()

        results = {}
        for modality, sig_info in signals.items():
            results[modality] = self.gate(
                modality=modality,
                fingerprint=sig_info.get("fingerprint"),
                amplitude=sig_info.get("amplitude", 0.5),
                gamma=sig_info.get("gamma", 0.5),
                arousal=arousal,
            )
        return results

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_channel_states(self) -> Dict[str, Dict[str, Any]]:
        """Get gate states for all channels."""
        return {
            modality: {
                "gate_gain": round(ch.gate_gain, 4),
                "topdown_bias": round(ch.topdown_bias, 4),
                "habituation": round(ch.habituation, 4),
                "last_salience": round(ch.last_salience, 4),
                "signal_count": ch.signal_count,
                "blocked_count": ch.blocked_count,
                "pass_rate": round(ch.signal_count / max(1, ch.total_count), 4),
            }
            for modality, ch in self._channels.items()
        }

    def get_focused_modalities(self) -> List[str]:
        """Get the list of currently focused modalities."""
        return list(self._focused_modalities)

    def get_state(self) -> Dict[str, Any]:
        """Get complete thalamus state."""
        return {
            "arousal": round(self._arousal, 4),
            "alpha": round(self.alpha, 4),
            "focused": list(self._focused_modalities),
            "channels": self.get_channel_states(),
            "stats": {
                "total_gated": self._total_gated,
                "total_passed": self._total_passed,
                "total_blocked": self._total_blocked,
                "total_startles": self._total_startles,
                "total_bursts": self._total_bursts,
                "pass_rate": round(
                    self._total_passed / max(1, self._total_gated), 4
                ),
            },
        }

    def reset(self):
        """Reset thalamus state."""
        self._channels.clear()
        self._focused_modalities.clear()
        self._arousal = 0.8
        self._total_gated = 0
        self._total_passed = 0
        self._total_blocked = 0
        self._total_startles = 0
        self._total_bursts = 0
