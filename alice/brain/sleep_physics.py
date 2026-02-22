# -*- coding: utf-8 -*-
"""
Sleep Physics Engine — Offline Impedance Renormalization & Energy Conservation

Core equations:

  Awake  = External impedance matching: Minimize Γ_ext (perception→action loop)
  Sleep  = Internal impedance matching: Minimize Γ_int (eliminate local minima)

  These two processes are complementary:
    - While awake, focus on external signals → local impedance matching → but accumulate global "impedance debt"
    - While sleeping, cut off external senses → global impedance renormalization → eliminate fragmentation → reduce Σ Γ²_int

Physics (three conservation laws):

  1. Energy conservation:
     dE/dt = -P_metabolic(stage) + P_recovery(stage)
     Awake consumption > recovery → E decreases → sleep pressure rises
     N3 deep sleep has maximum recovery → E recharges

  2. Impedance debt:
     Each signal transmission while awake → accumulates Γ² residue (thermal line fatigue)
     D_imp += Σ Γ²_cycle × α_fatigue
     During N3, impedance recalibration → D_imp *= (1 - recalibration_rate)

  3. Synaptic entropy:
     Awake learning → Hebbian +5% only enhances locally → strength distribution skews → entropy increases
     N3 downscaling → all synapses × 0.95 → preserves relative differences → entropy decreases
     REM random activation → tests pathway robustness → prunes weak connections

  Physical drivers of sleep pressure:
     P_sleep = f(E_deficit, D_impedance, H_entropy)
     = w₁(1 - E) + w₂·D + w₃·H
     No longer fixed accumulation, but driven by weighted sum of three physical quantities

Slow-Wave Oscillation:
  N3 deep sleep produces δ waves (~0.75 Hz)
  UP state (~500ms) → synaptic activation, memory replay
  DOWN state (~500ms) → global silence, downscaling

  δ waves = "mini-awake" alternations:
  Each UP state is a "micro cognitive cycle"
  → replay daytime memories → Hebbian consolidation
  → DOWN state → global downscaling (prevents saturation)

Dreams (REM):
  REM = diagnostic mode
  Randomly generate internal ElectricalSignal → traverse all channels
  If Γ ≈ 0 → channel healthy → skip
  If Γ >> 0 → mark for next-day repair
  Dream content = the channel patterns being randomly tested
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Physical constants
# ============================================================================

# --- Energy conservation ---
METABOLIC_COST = {
    "wake":  0.008,    # Awake metabolic cost/tick
    "n1":    0.003,    # N1 reduced cost
    "n2":    0.002,    # N2 even lower
    "n3":    0.001,    # N3 lowest power consumption
    "rem":   0.004,    # REM slightly higher (brain active)
}

RECOVERY_RATE = {
    "wake":  0.000,    # No recovery while awake
    "n1":    0.004,    # Light recovery
    "n2":    0.008,    # Spindle wave recovery
    "n3":    0.015,    # Deep sleep maximum recovery!
    "rem":   0.005,    # REM moderate recovery
}

# --- Impedance debt ---
IMPEDANCE_FATIGUE_RATE = 0.05       # Γ² → debt conversion rate
RECALIBRATION_RATE = {
    "wake":  0.000,
    "n1":    0.01,
    "n2":    0.03,
    "n3":    0.08,     # N3 strongest recalibration
    "rem":   0.02,
}

# --- Synaptic entropy ---
DOWNSCALE_FACTOR = {
    "wake":  1.000,    # No downscaling
    "n1":    1.000,
    "n2":    0.998,    # Minimal downscaling
    "n3":    0.990,    # N3 downscaling -1%
    "rem":   1.000,    # REM no downscaling (enhanced during dreaming)
}

# --- Slow-wave oscillation ---
SLOW_WAVE_FREQ = 0.75       # Hz, N3 δ waves
UP_STATE_DUTY = 0.5          # UP state duty cycle
REPLAY_PER_UP_STATE = 3      # Replay 3 memories per UP state

# --- REM diagnostic ---
REM_PROBE_COUNT = 5          # Base probe count per tick
REM_DREAM_NOISE_AMP = 0.3   # Base dream signal amplitude

# --- Fatigue-modulated dreaming (託夢物理) ---
# Physics: PGO wave amplitude scales with impedance debt.
#   A_pgo = A_base × (1 + α·D_debt)
#   More fatigue → louder PGO → clearer dream reflections.
# Safety: amplitude capped at 2× base (human REM rebound ceiling).
#   Extra energy cost = β·(A_actual - A_base)² ensures net recovery ≥ 0.
FATIGUE_DREAM_AMP_ALPHA = 1.0       # Debt→amplitude coupling strength
FATIGUE_DREAM_AMP_MAX = 2.0         # Maximum amplitude multiplier (safety cap)
FATIGUE_DREAM_PROBE_BONUS = 5       # Max extra probes from fatigue
FATIGUE_DREAM_ENERGY_BETA = 0.008   # Extra energy cost per unit (amp-base)²

# --- Sleep pressure weights ---
PRESSURE_WEIGHT_ENERGY = 0.4
PRESSURE_WEIGHT_IMPEDANCE = 0.35
PRESSURE_WEIGHT_ENTROPY = 0.25

# --- Sleep quality ---
OPTIMAL_N3_RATIO = 0.25      # Ideal N3 is 25% of total sleep
OPTIMAL_REM_RATIO = 0.20     # Ideal REM is 20%


# ============================================================================
# Impedance debt tracker
# ============================================================================


class ImpedanceDebtTracker:
    """
    Impedance Debt — "line fatigue" during wakefulness

    Physical model:
      Each signal transmission causes micro impedance shifts in channels due to electromagnetic thermal effects.
      Shift amount ∝ Γ² (higher reflection → more heat → more fatigue)

      D(t) = D(t-1) + α·Σ Γ²_cycle        (awake accumulation)
      D(t) = D(t-1) × (1 - β_stage)        (sleep repair)

    Biological correspondence:
      Adenosine accumulation → metabolic byproduct of neural activity during wakefulness
      Caffeine = blocks adenosine receptors = temporarily ignoring debt (but not eliminating it)
    """

    def __init__(self):
        self.debt: float = 0.0
        self.peak_debt: float = 0.0
        self.total_accumulated: float = 0.0
        self.total_repaired: float = 0.0
        self.history: List[float] = []
        self._max_history: int = 500

    def accumulate(self, reflected_energy_sum: float):
        """Awake: accumulate impedance debt"""
        increment = reflected_energy_sum * IMPEDANCE_FATIGUE_RATE
        self.debt += increment
        self.debt = min(self.debt, 1.0)
        self.total_accumulated += increment
        self.peak_debt = max(self.peak_debt, self.debt)

    def repair(self, stage: str):
        """Sleep: repair impedance debt"""
        rate = RECALIBRATION_RATE.get(stage, 0.0)
        repaired = self.debt * rate
        self.debt *= (1.0 - rate)
        self.total_repaired += repaired

    def record(self):
        self.history.append(self.debt)
        if len(self.history) > self._max_history:
            del self.history[:-self._max_history]

    def get_state(self) -> Dict[str, Any]:
        return {
            "debt": round(self.debt, 6),
            "peak_debt": round(self.peak_debt, 6),
            "total_accumulated": round(self.total_accumulated, 6),
            "total_repaired": round(self.total_repaired, 6),
            "repair_ratio": round(
                self.total_repaired / max(1e-9, self.total_accumulated), 4
            ),
        }


# ============================================================================
# Synaptic entropy tracker
# ============================================================================


class SynapticEntropyTracker:
    """
    Synaptic Entropy — Shannon entropy of synaptic strength distribution

    Physical model:
      H = -Σ p_i · log(p_i)
      where p_i = strength_i / Σ strengths

      Awake learning → local enhancement → distribution skews → entropy decreases (over-specialization)
      But frequent learning of different tasks → strengths disperse → entropy increases (fragmentation)
      N3 downscaling → global proportional scaling → preserves relative differences → entropy unchanged
      REM random activation → redistributes → entropy tends toward optimal

    Optimal state:
      Entropy too low = over-specialization (only remembers one thing)
      Entropy too high = fragmentation (can't remember anything clearly)
      Ideal = medium entropy (structured but flexible)
    """

    def __init__(self):
        self.current_entropy: float = 0.0
        self.optimal_entropy: float = 0.0  # Will be set after first computation
        self.entropy_deficit: float = 0.0  # |current - optimal|
        self.history: List[float] = []
        self._max_history: int = 500
        self._initialized: bool = False

    def compute(self, synaptic_strengths: List[float]) -> float:
        """
        Compute Shannon entropy of synaptic strength distribution

        Args:
            synaptic_strengths: List of all synaptic strengths

        Returns:
            Shannon entropy value (bits)
        """
        if not synaptic_strengths:
            return 0.0

        strengths = np.array(synaptic_strengths, dtype=np.float64)
        strengths = np.clip(strengths, 1e-10, None)

        # Normalize to probability distribution
        total = np.sum(strengths)
        if total < 1e-10:
            return 0.0

        p = strengths / total
        p = p[p > 1e-10]

        # Shannon entropy
        H = -float(np.sum(p * np.log2(p)))

        self.current_entropy = H

        # Optimal entropy = 70% of uniform distribution entropy (max entropy)
        max_entropy = math.log2(max(1, len(strengths)))
        if not self._initialized:
            self.optimal_entropy = max_entropy * 0.70
            self._initialized = True

        self.entropy_deficit = abs(H - self.optimal_entropy)
        # Normalize to 0~1
        self.entropy_deficit = min(1.0, self.entropy_deficit / max(1.0, max_entropy))

        return H

    def record(self):
        self.history.append(self.current_entropy)
        if len(self.history) > self._max_history:
            del self.history[:-self._max_history]

    def get_state(self) -> Dict[str, Any]:
        return {
            "current_entropy": round(self.current_entropy, 4),
            "optimal_entropy": round(self.optimal_entropy, 4),
            "entropy_deficit": round(self.entropy_deficit, 4),
        }


# ============================================================================
# Slow-wave oscillator
# ============================================================================


class SlowWaveOscillator:
    """
    Slow-Wave Oscillator — δ wave generator during N3 deep sleep

    Physical model:
      f_δ = 0.75 Hz → period T = 1.333 seconds
      UP state (~667ms): synaptic activation, memory replay window
      DOWN state (~667ms): global silence, synaptic downscaling

      δ(t) = sin(2π · f_δ · t)
      UP state  = δ(t) > 0
      DOWN state = δ(t) ≤ 0

    Biological correspondence:
      N3 deep sleep slow-wave oscillation drives hippocampus→cortex memory transfer
      Each UP state triggers a sharp-wave ripple → memory replay
    """

    def __init__(self):
        self.phase: float = 0.0
        self.frequency: float = SLOW_WAVE_FREQ
        self.cycle_count: int = 0
        self.up_state_count: int = 0
        self.down_state_count: int = 0
        self.replays_triggered: int = 0

    def tick(self) -> Dict[str, Any]:
        """
        Advance slow-wave oscillation by one tick.

        Returns:
            {
                "is_up_state": bool,
                "delta_amplitude": float (-1~1),
                "should_replay": bool (UP state and in replay window),
                "replay_count": int (number of memories to replay this time),
            }
        """
        # δ wave
        delta = math.sin(2 * math.pi * self.phase)
        self.phase += self.frequency / 10.0  # Assuming 10 ticks/second
        if self.phase >= 1.0:
            self.phase -= 1.0
            self.cycle_count += 1

        is_up = delta > 0
        if is_up:
            self.up_state_count += 1
        else:
            self.down_state_count += 1

        # Trigger replay at UP state peak moment
        should_replay = is_up and delta > 0.7
        replay_count = 0
        if should_replay:
            replay_count = REPLAY_PER_UP_STATE
            self.replays_triggered += replay_count

        return {
            "is_up_state": is_up,
            "delta_amplitude": round(delta, 4),
            "should_replay": should_replay,
            "replay_count": replay_count,
        }

    def reset(self):
        self.phase = 0.0

    def get_state(self) -> Dict[str, Any]:
        return {
            "phase": round(self.phase, 4),
            "cycle_count": self.cycle_count,
            "up_state_count": self.up_state_count,
            "down_state_count": self.down_state_count,
            "replays_triggered": self.replays_triggered,
        }


# ============================================================================
# REM dream diagnostic
# ============================================================================


class REMDreamDiagnostic:
    """
    REM Dreams — Channel Health Diagnostic

    Physical model:
      During REM, the brain produces random internal signals (PGO waves)
      These signals traverse neural pathways → test impedance matching status

      If Γ_channel ≈ 0 → channel healthy → skip
      If Γ_channel >> 0 → marked as "needs repair" → affects next-day calibration

    Physical interpretation of dreams:
      Dream content = concepts corresponding to randomly tested channels
      Dream absurdity = random combination of concepts from different modalities (visual+auditory+motor cross)
      Nightmares = testing high-Γ channels → high reflected energy → mild pain trigger
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()
        self.probes_sent: int = 0
        self.healthy_channels: int = 0
        self.damaged_channels: int = 0
        self.repair_queue: List[Dict[str, Any]] = []
        self.dream_fragments: List[Dict[str, Any]] = []
        self.total_dream_reflection: float = 0.0

    def probe_channels(
        self,
        channel_impedances: List[Tuple[str, float, float]],
        fatigue_factor: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Send random probe signals to test channel health.

        Fatigue modulation (託夢物理):
          When fatigue_factor > 0, PGO waves are stronger and more channels
          are scanned — mirroring human REM rebound after sleep deprivation.

          A_pgo = A_base × (1 + α·fatigue)  capped at FATIGUE_DREAM_AMP_MAX × A_base
          n_probe = base + floor(fatigue × FATIGUE_DREAM_PROBE_BONUS)

        Args:
            channel_impedances: [(channel_name, source_Z, load_Z), ...]
            fatigue_factor: Impedance debt level (0.0~1.0), drives dream vividness

        Returns:
            Probe results including fatigue-modulated metrics
        """
        if not channel_impedances:
            return {"probes": 0, "healthy": 0, "damaged": 0,
                    "dream_intensity": 0.0, "fatigue_factor": 0.0,
                    "amp_multiplier": 1.0}

        # --- Fatigue-modulated amplitude ---
        fatigue_factor = float(np.clip(fatigue_factor, 0.0, 1.0))
        amp_multiplier = min(
            FATIGUE_DREAM_AMP_MAX,
            1.0 + FATIGUE_DREAM_AMP_ALPHA * fatigue_factor,
        )
        probe_amp = REM_DREAM_NOISE_AMP * amp_multiplier

        # --- Fatigue-modulated probe count ---
        bonus_probes = int(fatigue_factor * FATIGUE_DREAM_PROBE_BONUS)
        n_probe = min(REM_PROBE_COUNT + bonus_probes, len(channel_impedances))
        indices = self.rng.choice(
            len(channel_impedances), size=n_probe, replace=False
        )

        healthy = 0
        damaged = 0
        fragments = []

        for idx in indices:
            name, z_src, z_load = channel_impedances[idx]
            self.probes_sent += 1

            # Compute reflection coefficient
            gamma = abs((z_load - z_src) / (z_load + z_src)) if (z_load + z_src) > 0 else 1.0

            # Random probe signal frequency
            probe_freq = float(self.rng.uniform(1, 100))

            reflected_energy = probe_amp ** 2 * gamma ** 2
            self.total_dream_reflection += reflected_energy

            if gamma < 0.3:
                healthy += 1
                self.healthy_channels += 1
            else:
                damaged += 1
                self.damaged_channels += 1
                self.repair_queue.append({
                    "channel": name,
                    "gamma": round(gamma, 4),
                    "z_src": round(z_src, 2),
                    "z_load": round(z_load, 2),
                })

            fragments.append({
                "channel": name,
                "gamma": round(gamma, 4),
                "probe_freq": round(probe_freq, 1),
                "probe_amp": round(probe_amp, 4),
                "is_healthy": gamma < 0.3,
            })

        self.dream_fragments.extend(fragments)
        # Keep the most recent 50 entries
        if len(self.dream_fragments) > 50:
            self.dream_fragments = self.dream_fragments[-50:]

        return {
            "probes": n_probe,
            "healthy": healthy,
            "damaged": damaged,
            "dream_intensity": round(
                sum(f["gamma"] for f in fragments) / max(1, len(fragments)), 4
            ),
            "fatigue_factor": round(fatigue_factor, 4),
            "amp_multiplier": round(amp_multiplier, 4),
        }

    def get_repair_queue(self) -> List[Dict[str, Any]]:
        """Get the list of channels needing repair"""
        queue = self.repair_queue.copy()
        self.repair_queue.clear()
        return queue

    def get_state(self) -> Dict[str, Any]:
        return {
            "probes_sent": self.probes_sent,
            "healthy_channels": self.healthy_channels,
            "damaged_channels": self.damaged_channels,
            "total_dream_reflection": round(self.total_dream_reflection, 6),
            "dream_health_ratio": round(
                self.healthy_channels / max(1, self.probes_sent), 4
            ),
            "recent_dreams": len(self.dream_fragments),
        }


# ============================================================================
# Sleep quality assessment
# ============================================================================


@dataclass
class SleepQualityReport:
    """Sleep quality report"""

    total_sleep_ticks: int = 0
    n3_ticks: int = 0
    rem_ticks: int = 0
    n3_ratio: float = 0.0
    rem_ratio: float = 0.0
    interruptions: int = 0
    energy_restored: float = 0.0
    impedance_debt_repaired: float = 0.0
    entropy_change: float = 0.0
    memories_consolidated: int = 0
    slow_wave_cycles: int = 0
    dream_health_ratio: float = 0.0

    @property
    def quality_score(self) -> float:
        """
        Sleep quality score (0~1)

        = 0.25·(N3 sufficiency) + 0.20·(REM sufficiency)
          + 0.20·(debt repair rate) + 0.15·(energy recovery)
          + 0.10·(no interruptions) + 0.10·(channel health)
        """
        n3_score = min(1.0, self.n3_ratio / OPTIMAL_N3_RATIO)
        rem_score = min(1.0, self.rem_ratio / OPTIMAL_REM_RATIO)
        debt_score = min(1.0, self.impedance_debt_repaired / max(0.01, 0.3))
        energy_score = min(1.0, self.energy_restored / 0.5)
        no_interrupt = max(0.0, 1.0 - self.interruptions * 0.2)
        health = self.dream_health_ratio

        return float(np.clip(
            0.25 * n3_score + 0.20 * rem_score
            + 0.20 * debt_score + 0.15 * energy_score
            + 0.10 * no_interrupt + 0.10 * health,
            0.0, 1.0,
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_sleep_ticks": self.total_sleep_ticks,
            "n3_ticks": self.n3_ticks,
            "rem_ticks": self.rem_ticks,
            "n3_ratio": round(self.n3_ratio, 3),
            "rem_ratio": round(self.rem_ratio, 3),
            "interruptions": self.interruptions,
            "energy_restored": round(self.energy_restored, 4),
            "impedance_debt_repaired": round(self.impedance_debt_repaired, 4),
            "entropy_change": round(self.entropy_change, 4),
            "memories_consolidated": self.memories_consolidated,
            "slow_wave_cycles": self.slow_wave_cycles,
            "dream_health_ratio": round(self.dream_health_ratio, 4),
            "quality_score": round(self.quality_score, 3),
        }


# ============================================================================
# Main engine: Sleep Physics Engine
# ============================================================================


class SleepPhysicsEngine:
    """
    Sleep Physics Engine — integrates energy conservation, impedance recalibration, synaptic downscaling

    Core equations:

      Energy:   E(t+1) = E(t) - P_metabolic + P_recovery
      Debt:     D(t+1) = D(t) + α·ΣΓ² (awake) | D(t)·(1-β) (sleep)
      Entropy:  H(t)   = -Σ p_i·log₂(p_i)
      Pressure: P_sleep = w₁(1-E) + w₂·D + w₃·H_deficit

    Usage:
      engine = SleepPhysicsEngine()

      # Awake tick
      engine.awake_tick(reflected_energy=0.05, synaptic_strengths=[...])

      # Sleep tick
      result = engine.sleep_tick(
          stage="n3",
          recent_memories=[...],
          channel_impedances=[...],
          synaptic_strengths=[...],
      )
    """

    def __init__(self, energy: float = 1.0):
        # Three major physical quantities
        self.energy: float = energy
        self.impedance_debt = ImpedanceDebtTracker()
        self.entropy_tracker = SynapticEntropyTracker()

        # Subsystems
        self.slow_wave = SlowWaveOscillator()
        self.dream_diagnostic = REMDreamDiagnostic()

        # Sleep pressure (driven by physical quantities)
        self.sleep_pressure: float = 0.0

        # Statistical tracking
        self.awake_ticks: int = 0
        self.sleep_ticks: int = 0
        self.n3_ticks: int = 0
        self.rem_ticks: int = 0
        self.total_replays: int = 0
        self.total_downscales: int = 0
        self.interruptions: int = 0

        # Current sleep tracking
        self._current_sleep_report: Optional[SleepQualityReport] = None
        self._pre_sleep_energy: float = 0.0
        self._pre_sleep_debt: float = 0.0
        self._pre_sleep_entropy: float = 0.0

        # History
        self.energy_history: List[float] = []
        self.pressure_history: List[float] = []
        self._max_history: int = 500

    # ------------------------------------------------------------------
    # Awake tick
    # ------------------------------------------------------------------

    def awake_tick(
        self,
        reflected_energy: float = 0.0,
        synaptic_strengths: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Called each cognitive cycle while awake

        Args:
            reflected_energy: Total reflected energy across all channels this cycle
            synaptic_strengths: List of all synaptic strengths

        Returns:
            Physical state
        """
        self.awake_ticks += 1

        # Energy consumption
        self.energy -= METABOLIC_COST["wake"]
        self.energy = float(np.clip(self.energy, 0.0, 1.0))

        # Impedance debt accumulation
        self.impedance_debt.accumulate(reflected_energy)
        self.impedance_debt.record()

        # Synaptic entropy computation
        if synaptic_strengths:
            self.entropy_tracker.compute(synaptic_strengths)
        self.entropy_tracker.record()

        # Update sleep pressure
        self._update_pressure()

        # History recording
        self.energy_history.append(self.energy)
        self.pressure_history.append(self.sleep_pressure)
        for hist in (self.energy_history, self.pressure_history):
            if len(hist) > self._max_history:
                del hist[:-self._max_history]

        return {
            "energy": round(self.energy, 4),
            "impedance_debt": round(self.impedance_debt.debt, 4),
            "entropy": round(self.entropy_tracker.current_entropy, 4),
            "sleep_pressure": round(self.sleep_pressure, 4),
            "should_sleep": self.sleep_pressure > 0.7,
        }

    # ------------------------------------------------------------------
    # Sleep tick
    # ------------------------------------------------------------------

    def sleep_tick(
        self,
        stage: str,
        recent_memories: Optional[List[Any]] = None,
        channel_impedances: Optional[List[Tuple[str, float, float]]] = None,
        synaptic_strengths: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Called each tick while sleeping

        Args:
            stage: Sleep stage ("n1"/"n2"/"n3"/"rem")
            recent_memories: Recent memories available for replay
            channel_impedances: [(name, z_src, z_load), ...]
            synaptic_strengths: All synaptic strengths

        Returns:
            {
                "energy": float,
                "impedance_debt": float,
                "entropy": float,
                "replayed": int,
                "downscaled": bool,
                "dream_result": dict or None,
                "slow_wave": dict or None,
            }
        """
        self.sleep_ticks += 1

        # Energy recovery
        cost = METABOLIC_COST.get(stage, 0.002)
        recovery = RECOVERY_RATE.get(stage, 0.0)
        self.energy += (recovery - cost)
        self.energy = float(np.clip(self.energy, 0.0, 1.0))

        # Impedance debt repair
        pre_debt = self.impedance_debt.debt
        self.impedance_debt.repair(stage)
        self.impedance_debt.record()

        replayed = 0
        downscaled = False
        slow_wave_result = None
        dream_result = None
        downscale_strengths = None

        if stage == "n3":
            self.n3_ticks += 1

            # === Slow-wave oscillation ===
            slow_wave_result = self.slow_wave.tick()

            # UP state → memory replay
            if slow_wave_result["should_replay"] and recent_memories:
                n = min(slow_wave_result["replay_count"], len(recent_memories))
                replayed = n
                self.total_replays += n

            # DOWN state → synaptic downscaling
            if not slow_wave_result["is_up_state"] and synaptic_strengths:
                factor = DOWNSCALE_FACTOR["n3"]
                downscale_strengths = [s * factor for s in synaptic_strengths]
                downscaled = True
                self.total_downscales += 1

        elif stage == "n2":
            # N2 spindle waves — light replay
            if recent_memories:
                replayed = min(1, len(recent_memories))
                self.total_replays += replayed

        elif stage == "rem":
            self.rem_ticks += 1

            # === REM dream diagnostic (fatigue-modulated) ===
            # Physics: impedance debt drives PGO wave amplitude.
            # More fatigue → louder probes → more vivid dreams.
            # Safety: extra energy cost clamped so net recovery ≥ 0.
            fatigue = self.impedance_debt.debt  # 0.0 ~ 1.0
            if channel_impedances:
                dream_result = self.dream_diagnostic.probe_channels(
                    channel_impedances,
                    fatigue_factor=fatigue,
                )

                # Extra energy cost for vivid dreaming
                amp_mult = dream_result.get("amp_multiplier", 1.0)
                extra_cost = FATIGUE_DREAM_ENERGY_BETA * (amp_mult - 1.0) ** 2
                # Safety clamp: never let REM become net energy negative
                max_extra = max(0.0, recovery - cost - 0.0005)  # keep ≥ +0.0005 net
                extra_cost = min(extra_cost, max_extra)
                self.energy -= extra_cost
                self.energy = float(np.clip(self.energy, 0.0, 1.0))

        # Synaptic entropy update
        target_strengths = downscale_strengths or synaptic_strengths
        if target_strengths:
            self.entropy_tracker.compute(target_strengths)
        self.entropy_tracker.record()

        # Update pressure (pressure decreases during sleep)
        self._update_pressure()

        # History
        self.energy_history.append(self.energy)
        self.pressure_history.append(self.sleep_pressure)
        for hist in (self.energy_history, self.pressure_history):
            if len(hist) > self._max_history:
                del hist[:-self._max_history]

        # Update sleep report
        if self._current_sleep_report:
            self._current_sleep_report.total_sleep_ticks += 1
            if stage == "n3":
                self._current_sleep_report.n3_ticks += 1
            elif stage == "rem":
                self._current_sleep_report.rem_ticks += 1
            self._current_sleep_report.memories_consolidated += replayed

        return {
            "energy": round(self.energy, 4),
            "impedance_debt": round(self.impedance_debt.debt, 4),
            "entropy": round(self.entropy_tracker.current_entropy, 4),
            "sleep_pressure": round(self.sleep_pressure, 4),
            "replayed": replayed,
            "downscaled": downscaled,
            "downscale_strengths": downscale_strengths,
            "slow_wave": slow_wave_result,
            "dream_result": dream_result,
        }

    # ------------------------------------------------------------------
    # Sleep start / end
    # ------------------------------------------------------------------

    def begin_sleep(self):
        """Mark sleep onset, initialize quality tracking"""
        self._pre_sleep_energy = self.energy
        self._pre_sleep_debt = self.impedance_debt.debt
        self._pre_sleep_entropy = self.entropy_tracker.current_entropy
        self._current_sleep_report = SleepQualityReport()
        self.slow_wave.reset()

    def end_sleep(self) -> SleepQualityReport:
        """End sleep, produce quality report"""
        if self._current_sleep_report is None:
            return SleepQualityReport()

        report = self._current_sleep_report
        total = max(1, report.total_sleep_ticks)
        report.n3_ratio = report.n3_ticks / total
        report.rem_ratio = report.rem_ticks / total
        report.energy_restored = self.energy - self._pre_sleep_energy
        report.impedance_debt_repaired = self._pre_sleep_debt - self.impedance_debt.debt
        report.entropy_change = (
            self.entropy_tracker.current_entropy - self._pre_sleep_entropy
        )
        report.slow_wave_cycles = self.slow_wave.cycle_count
        report.dream_health_ratio = self.dream_diagnostic.get_state()[
            "dream_health_ratio"
        ]
        report.interruptions = self.interruptions

        self._current_sleep_report = None
        return report

    # ------------------------------------------------------------------
    # Sleep pressure
    # ------------------------------------------------------------------

    def _update_pressure(self):
        """
        Physically driven sleep pressure:

        P = w₁·(1 - E) + w₂·D + w₃·H_deficit

        Lower energy → more pressure
        Higher debt → more pressure
        Greater entropy deviation → more pressure
        """
        energy_deficit = 1.0 - self.energy
        debt = self.impedance_debt.debt
        entropy_deficit = self.entropy_tracker.entropy_deficit

        self.sleep_pressure = float(np.clip(
            PRESSURE_WEIGHT_ENERGY * energy_deficit
            + PRESSURE_WEIGHT_IMPEDANCE * debt
            + PRESSURE_WEIGHT_ENTROPY * entropy_deficit,
            0.0, 1.0,
        ))

    def should_sleep(self) -> bool:
        return self.sleep_pressure > 0.7

    # ------------------------------------------------------------------
    # Global synaptic downscaling
    # ------------------------------------------------------------------

    @staticmethod
    def apply_downscaling(
        synaptic_strengths: List[float],
        factor: float = 0.990,
        floor: float = 0.05,
    ) -> List[float]:
        """
        Synaptic downscaling — Tononi's Synaptic Homeostasis Hypothesis

        All synapses proportionally scaled: s_i := s_i × factor
        Preserves relative differences (who's strong/weak unchanged)
        But overall activity level decreases → prevents saturation → frees capacity for next-day learning

        Args:
            synaptic_strengths: List of synaptic strengths
            factor: Scaling factor (0.99 = reduce by 1%)
            floor: Minimum value (won't drop to 0)

        Returns:
            List of downscaled strengths
        """
        return [max(floor, s * factor) for s in synaptic_strengths]

    # ------------------------------------------------------------------
    # Impedance recalibration
    # ------------------------------------------------------------------

    @staticmethod
    def impedance_recalibration(
        channel_impedances: List[Tuple[str, float, float]],
        recalibration_strength: float = 0.05,
    ) -> List[Tuple[str, float, float]]:
        """
        Impedance recalibration during sleep — fine-tune channel impedance so Γ→0

        Physical model:
          For each channel, fine-tune Z_load to approach Z_source:
          Z_load_new = Z_load + recal_strength × (Z_source - Z_load)

        Args:
            channel_impedances: [(name, z_src, z_load), ...]
            recalibration_strength: Recalibration strength (0~1)

        Returns:
            Recalibrated [(name, z_src, z_load_new), ...]
        """
        result = []
        for name, z_src, z_load in channel_impedances:
            z_load_new = z_load + recalibration_strength * (z_src - z_load)
            result.append((name, z_src, z_load_new))
        return result

    # ------------------------------------------------------------------
    # Full day-night cycle simulation
    # ------------------------------------------------------------------

    def simulate_day_night(
        self,
        awake_ticks: int = 100,
        sleep_ticks: int = 110,
        reflected_energy_per_tick: float = 0.05,
        n_synapses: int = 200,
        n_channels: int = 6,
        learning_events: int = 10,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """
        Simulate a full day-night cycle

        Args:
            awake_ticks: Number of awake ticks
            sleep_ticks: Number of sleep ticks
            reflected_energy_per_tick: Average reflected energy per tick
            n_synapses: Total number of synapses
            n_channels: Number of channels
            learning_events: Number of daytime learning events
            rng: Random number generator

        Returns:
            Complete day-night report
        """
        rng = rng or np.random.default_rng(42)

        # Initial synaptic strengths (random)
        synaptic_strengths = list(rng.uniform(0.5, 1.5, n_synapses))

        # Channel impedances
        channel_impedances = [
            (f"ch_{i}", float(rng.uniform(50, 110)),
             float(rng.uniform(50, 110)))
            for i in range(n_channels)
        ]

        # Recent memories (simulated learning products)
        recent_memories = [f"memory_{i}" for i in range(20)]

        # === Daytime ===
        day_log = []
        for t in range(awake_ticks):
            # Simulate learning: some synapses strengthen
            if t % (awake_ticks // max(1, learning_events)) == 0:
                boost_idx = rng.integers(0, n_synapses, size=10)
                for idx in boost_idx:
                    synaptic_strengths[idx] = min(
                        2.0, synaptic_strengths[idx] * 1.05
                    )

            # Reflected energy (random fluctuation)
            re = reflected_energy_per_tick * float(rng.uniform(0.5, 1.5))

            result = self.awake_tick(
                reflected_energy=re,
                synaptic_strengths=synaptic_strengths,
            )
            if t % 10 == 0:
                day_log.append({"tick": t, **result})

        pre_sleep_state = {
            "energy": self.energy,
            "impedance_debt": self.impedance_debt.debt,
            "entropy": self.entropy_tracker.current_entropy,
            "sleep_pressure": self.sleep_pressure,
        }

        # === Nighttime ===
        self.begin_sleep()

        # Sleep stage allocation (following typical proportions)
        stage_schedule = self._generate_sleep_schedule(sleep_ticks)

        night_log = []
        for t, stage in enumerate(stage_schedule):
            result = self.sleep_tick(
                stage=stage,
                recent_memories=recent_memories if recent_memories else None,
                channel_impedances=channel_impedances,
                synaptic_strengths=synaptic_strengths,
            )

            # Apply downscaling
            if result.get("downscale_strengths"):
                synaptic_strengths = result["downscale_strengths"]

            # Consume replayed memories
            if result["replayed"] > 0 and recent_memories:
                recent_memories = recent_memories[result["replayed"]:]

            if t % 10 == 0:
                night_log.append({"tick": t, "stage": stage, **result})

        sleep_report = self.end_sleep()

        post_sleep_state = {
            "energy": self.energy,
            "impedance_debt": self.impedance_debt.debt,
            "entropy": self.entropy_tracker.current_entropy,
            "sleep_pressure": self.sleep_pressure,
        }

        return {
            "pre_sleep": pre_sleep_state,
            "post_sleep": post_sleep_state,
            "sleep_report": sleep_report.to_dict(),
            "day_log": day_log,
            "night_log": night_log,
            "synaptic_strengths_final": synaptic_strengths,
            "channel_impedances_final": channel_impedances,
        }

    # ------------------------------------------------------------------
    # Sleep stage schedule
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_sleep_schedule(total_ticks: int) -> List[str]:
        """
        Generate sleep stage schedule (simulates 90-minute cycles)

        A typical night = 4-5 cycles:
          N1(5%) → N2(45%) → N3(25%) → N2(5%) → REM(20%)
        """
        cycle = [
            ("n1", 0.05),
            ("n2", 0.22),
            ("n3", 0.25),
            ("n2", 0.05),
            ("rem", 0.18),
            ("n2", 0.15),
            ("n3", 0.10),
        ]

        schedule = []
        while len(schedule) < total_ticks:
            for stage, ratio in cycle:
                n = max(1, int(total_ticks * ratio / 4))  # 4 cycles
                schedule.extend([stage] * n)
                if len(schedule) >= total_ticks:
                    break

        return schedule[:total_ticks]

    # ------------------------------------------------------------------
    # State query
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        return {
            "energy": round(self.energy, 4),
            "sleep_pressure": round(self.sleep_pressure, 4),
            "should_sleep": self.should_sleep(),
            "impedance_debt": self.impedance_debt.get_state(),
            "entropy": self.entropy_tracker.get_state(),
            "slow_wave": self.slow_wave.get_state(),
            "dream_diagnostic": self.dream_diagnostic.get_state(),
            "statistics": {
                "awake_ticks": self.awake_ticks,
                "sleep_ticks": self.sleep_ticks,
                "n3_ticks": self.n3_ticks,
                "rem_ticks": self.rem_ticks,
                "total_replays": self.total_replays,
                "total_downscales": self.total_downscales,
            },
        }
