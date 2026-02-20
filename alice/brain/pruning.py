# -*- coding: utf-8 -*-
"""
Neural Pruning Engine — Large-Scale Γ Apoptosis

§3.5.2 Core physics:

  Birth = random impedance initialization
    Z_init^(i) ~ Uniform(Z_min, Z_max),  i = 1, 2, ..., N
    Random impedance means Γ is almost everywhere non-zero → reflected energy is enormous
    → System temperature is high → newborns cry easily, collapse easily

  Neural pruning = large-scale impedance screening
    if Γ_ij → 0  ⇒  Hebbian strengthening (+5%)  ⇒  survives
    if Γ_ij >> 0 ⇒  Hebbian weakening (-5%)  ⇒  apoptosis

  What's pruned isn't "bad" neurons—it's impedance-mismatched connections.
  What survives are pathways that happen to resonate with the frequencies flowing through them.

  Physical inevitability of cortical functionalization:
    Occipital → vision (spatial frequency) → α/β/γ bandpass → visual cortex
    Temporal → audition (temporal frequency) → θ/α tonotopic → auditory cortex
    Parietal → somatosensory (pressure/temperature) → broadband → somatosensory cortex
    Frontal → motor (PID error loop) → requires feedback to tune → matures latest

  Physical objective function of intelligence:
    Σ Γ_i² → min

Coaxial cable correspondence:
  Each synaptic connection = one segment of coaxial cable
  Impedance matching = whether the signal can pass
  Resonant frequency = which frequency the connection is naturally most sensitive to
  Pruning = unplugging cables that can't transmit signals
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alice.core.signal import BrainWaveBand


# ============================================================================
# 1. Synaptic connection — single coaxial cable
# ============================================================================


@dataclass
class SynapticConnection:
    """
    Synaptic connection — a prunable segment of coaxial cable.

    Physical properties:
    - impedance        : Connection's characteristic impedance (Ω), random at birth
    - resonant_freq    : Natural resonant frequency (Hz), derived from impedance
    - synaptic_strength: Synaptic strength (0→dead, 2.0→strongest)
    - alive            : Whether alive (strength < threshold → apoptosis)

    At birth: impedance ~ Uniform(Z_min, Z_max)
    After pruning: only impedance-matched connections survive
    """

    connection_id: int
    impedance: float                    # Connection characteristic impedance (Ω)
    resonant_freq: float                # Natural resonant frequency (Hz)
    synaptic_strength: float = 1.0      # Synaptic strength
    alive: bool = True                  # Whether alive
    total_stimulations: int = 0         # Cumulative stimulation count
    _gamma_sum: float = 0.0            # Cumulative |Γ| sum
    _match_sum: float = 0.0            # Cumulative match score sum

    @property
    def avg_gamma(self) -> float:
        """Average reflection coefficient |Γ| — lower means better matched."""
        if self.total_stimulations == 0:
            return 1.0  # Unstimulated = unknown
        return self._gamma_sum / self.total_stimulations

    @property
    def avg_match(self) -> float:
        """Average match score — higher means better adapted."""
        if self.total_stimulations == 0:
            return 0.0
        return self._match_sum / self.total_stimulations

    def compute_gamma(self, signal_impedance: float) -> float:
        """
        Compute reflection coefficient Γ.

        Γ = (Z_connection - Z_signal) / (Z_connection + Z_signal)

        |Γ| → 0 : Perfect match, signal passes without loss
        |Γ| → 1 : Total reflection, signal bounces back
        """
        denom = self.impedance + signal_impedance
        if denom == 0:
            return 0.0
        return abs((self.impedance - signal_impedance) / denom)

    def compute_resonance(self, signal_freq: float, Q: float = 5.0) -> float:
        """
        Compute resonance response — Lorentzian curve.

        L(f) = 1 / (1 + Q² × ((f/f₀) - (f₀/f))²)

        f₀ = Connection's natural resonant frequency
        Q  = Quality factor (higher = narrower band)
        """
        f0 = self.resonant_freq
        if f0 <= 0 or signal_freq <= 0:
            return 0.0
        ratio = signal_freq / f0
        inv_ratio = f0 / signal_freq
        return 1.0 / (1.0 + Q ** 2 * (ratio - inv_ratio) ** 2)


# ============================================================================
# 2. Cortical region — prunable connection group
# ============================================================================


class CorticalSpecialization(Enum):
    """Cortical specialization direction."""
    UNSPECIALIZED = "unspecialized"    # Undifferentiated (newborn)
    VISUAL = "visual"                  # Visual cortex (α/β/γ)
    AUDITORY = "auditory"              # Auditory cortex (θ/α)
    SOMATOSENSORY = "somatosensory"    # Somatosensory cortex (broadband)
    MOTOR = "motor"                    # Motor cortex (β, requires feedback)


@dataclass
class PruningMetrics:
    """Per-pruning-cycle statistics."""
    cycle: int
    alive_count: int
    pruned_this_cycle: int
    sprouted_this_cycle: int       # ★ Newly sprouted connections this cycle
    total_pruned: int
    total_sprouted: int             # ★ Cumulative sprouted connections
    survival_rate: float
    avg_gamma: float
    avg_strength: float
    specialization_index: float
    dominant_freq: float
    freq_spread: float
    peak_connections: int           # ★ Historical peak connection count


class CorticalRegion:
    """
    Cortical region — contains a large number of prunable synaptic connections.

    Physical model:
    - At birth: N connections with randomly distributed impedance
    - Each signal received: compute each connection's Γ and resonance response
    - Hebbian rule: match → strengthen, mismatch → weaken
    - Weakened below threshold → apoptosis
    - Final surviving connections = the region's "specialization"
    """

    def __init__(
        self,
        name: str,
        initial_connections: int = 1000,
        region_impedance: float = 75.0,
        z_min: float = 20.0,
        z_max: float = 200.0,
        freq_min: float = 0.5,
        freq_max: float = 100.0,
        hebbian_strengthen: float = 1.05,    # +5%
        hebbian_weaken: float = 0.97,        # -3%
        death_threshold: float = 0.10,       # Apoptosis threshold
        match_threshold: float = 0.25,        # Match score threshold
        Q: float = 2.0,                      # Resonance quality factor
    ):
        self.name = name
        self.initial_connections = initial_connections
        self.region_impedance = region_impedance
        self.z_min = z_min
        self.z_max = z_max
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.hebbian_strengthen = hebbian_strengthen
        self.hebbian_weaken = hebbian_weaken
        self.death_threshold = death_threshold
        self.match_threshold = match_threshold
        self.Q = Q

        # Initialize random connections (birth)
        self.connections: List[SynapticConnection] = []
        self._birth_initialize(initial_connections)

        # Development tracking
        self.stimulation_cycles: int = 0
        self.pruned_total: int = 0
        self.sprouted_total: int = 0                    # ★ Cumulative sprouted
        self.peak_connections: int = initial_connections  # ★ Historical peak connection count
        self.specialization: CorticalSpecialization = CorticalSpecialization.UNSPECIALIZED
        self._metrics_history: List[PruningMetrics] = []

        # Synaptogenesis parameters
        # Physical meaning: strong connections carry high current → EM field diffuses → induces neighboring axon sprouting
        self._sprout_strength_threshold: float = 1.4   # Strength > this to sprout
        self._sprout_noise_z: float = 0.15             # New connection impedance perturbation (±15%)
        self._sprout_noise_f: float = 0.15             # New connection frequency perturbation (±15%)
        self._sprout_initial_strength: float = 0.30    # New connection initial strength (fragile)
        self._sprout_max_per_cycle: int = max(5, initial_connections // 50)  # Per-cycle cap
        self._sprout_cooldown: int = 0                 # Sprouting cooldown counter
        self._next_connection_id: int = initial_connections  # Auto-increment ID

        # Impedance adaptation tracking
        self._impedance_adaptation_rate: float = 0.01  # Rate of impedance moving toward match direction

    # ------------------------------------------------------------------
    def _birth_initialize(self, n: int):
        """
        Birth initialization — Z_init ~ Uniform(Z_min, Z_max)

        Each connection's natural resonant frequency is derived from impedance:
        f_natural = f_min + (Z - Z_min) / (Z_max - Z_min) × (f_max - f_min)

        This simulates how cables with different impedances have different
        frequency response characteristics.
        """
        for i in range(n):
            z = np.random.uniform(self.z_min, self.z_max)
            # Impedance mapped to resonant frequency (log space is more physical)
            z_norm = (z - self.z_min) / max(1e-6, self.z_max - self.z_min)
            f_natural = self.freq_min * (self.freq_max / self.freq_min) ** z_norm

            self.connections.append(SynapticConnection(
                connection_id=i,
                impedance=z,
                resonant_freq=f_natural,
            ))

    # ------------------------------------------------------------------
    def stimulate(
        self,
        signal_impedance: float,
        signal_frequency: float,
    ) -> Dict[str, Any]:
        """
        Stimulate the region — signal flows through all alive connections.

        Physical process:
        1. Compute each connection's Γ (impedance match)
        2. Compute each connection's resonance response (frequency match)
        3. Combined match score = (1 - |Γ|) × resonance response
        4. Hebbian rule:
           Match score > threshold → synaptic strengthening +5%, impedance adapts toward signal
           Match score < threshold → synaptic weakening -5%

        Returns:
            Statistical summary of this stimulation
        """
        self.stimulation_cycles += 1

        strengthened = 0
        weakened = 0
        total_gamma = 0.0
        total_match = 0.0
        alive_count = 0

        for conn in self.connections:
            if not conn.alive:
                continue

            alive_count += 1

            # 1. Impedance match
            gamma = conn.compute_gamma(signal_impedance)

            # 2. Frequency resonance response
            resonance = conn.compute_resonance(signal_frequency, self.Q)

            # 3. Combined match score
            match_score = (1.0 - gamma) * resonance

            # Cumulative statistics
            conn.total_stimulations += 1
            conn._gamma_sum += gamma
            conn._match_sum += match_score
            total_gamma += gamma
            total_match += match_score

            # 4. Hebbian rule
            if match_score > self.match_threshold:
                # Match → strengthen → survive
                conn.synaptic_strength = min(
                    2.0, conn.synaptic_strength * self.hebbian_strengthen
                )
                # Impedance adaptation: move toward signal impedance (learning)
                conn.impedance += (
                    (signal_impedance - conn.impedance) * self._impedance_adaptation_rate
                )
                # Resonant frequency also adapts
                conn.resonant_freq += (
                    (signal_frequency - conn.resonant_freq) * self._impedance_adaptation_rate
                )
                strengthened += 1
            else:
                # Mismatch → weaken → may undergo apoptosis
                conn.synaptic_strength = max(
                    0.0, conn.synaptic_strength * self.hebbian_weaken
                )
                weakened += 1

        return {
            "alive": alive_count,
            "strengthened": strengthened,
            "weakened": weakened,
            "avg_gamma": total_gamma / max(1, alive_count),
            "avg_match": total_match / max(1, alive_count),
        }

    # ------------------------------------------------------------------
    def prune(self) -> int:
        """
        Execute apoptosis — remove connections with synaptic_strength < death_threshold.

        Physical meaning:
          Repeated Hebbian weakening drops synaptic strength below threshold
          → Synapse disappears (apoptosis)
          → Freed resources allocated to surviving connections
          → Survivors become stronger
        """
        pruned = 0
        for conn in self.connections:
            if conn.alive and conn.synaptic_strength < self.death_threshold:
                conn.alive = False
                pruned += 1
                self.pruned_total += 1
        return pruned

    # ------------------------------------------------------------------
    def sprout(self, learning_signal: float = 0.0) -> int:
        """
        Synaptogenesis — experience-driven connection sprouting.

        Physical model:
          When a coaxial cable carries high power (synaptic_strength > threshold),
          surrounding EM field induces neighboring axon myelin depolarization → sprouts new connections.
          New connections inherit approximate impedance from parent (±noise) but start with low strength (0.30),
          must prove their match in subsequent Hebbian selection or be pruned.

        Biological correspondence:
          - Hippocampal DG generates ~700 new neurons/day (Spalding 2013)
          - Skill learning causes experience-dependent synaptogenesis
          - LTP (Long-Term Potentiation) → synapse splitting → new connections

        Args:
            learning_signal: External learning signal strength (0~1), from Hebbian / curiosity / reward
                            Higher → greater sprouting probability

        Returns:
            Number of newly sprouted connections
        """
        if self._sprout_cooldown > 0:
            self._sprout_cooldown -= 1
            return 0

        # Find "seed connections" — alive connections with strength above threshold
        alive = self.alive_connections
        if not alive:
            return 0

        # Learning signal modulates sprouting threshold: high signal → lower threshold → easier sprouting
        effective_threshold = self._sprout_strength_threshold * (1.0 - 0.3 * min(1.0, learning_signal))

        seeds = [c for c in alive if c.synaptic_strength > effective_threshold]
        if not seeds:
            return 0

        # Sprout count: min(seed_count, per_cycle_cap) × (1 + learning_signal)
        max_sprouts = min(len(seeds), self._sprout_max_per_cycle)
        n_sprout = max(1, int(max_sprouts * (0.5 + 0.5 * min(1.0, learning_signal))))

        sprouted = 0
        # Start sprouting from strongest seeds
        seeds.sort(key=lambda c: c.synaptic_strength, reverse=True)

        for seed in seeds[:n_sprout]:
            # New connection impedance = parent ± perturbation (exploring neighboring impedance space)
            z_noise = seed.impedance * self._sprout_noise_z
            new_z = seed.impedance + np.random.uniform(-z_noise, z_noise)
            new_z = max(self.z_min, min(self.z_max, new_z))

            # New connection frequency = parent ± perturbation
            f_noise = seed.resonant_freq * self._sprout_noise_f
            new_f = seed.resonant_freq + np.random.uniform(-f_noise, f_noise)
            new_f = max(self.freq_min, min(self.freq_max, new_f))

            new_conn = SynapticConnection(
                connection_id=self._next_connection_id,
                impedance=new_z,
                resonant_freq=new_f,
                synaptic_strength=self._sprout_initial_strength,
                alive=True,
            )
            self.connections.append(new_conn)
            self._next_connection_id += 1
            sprouted += 1

        self.sprouted_total += sprouted
        self.peak_connections = max(self.peak_connections, self.alive_count)

        # Brief cooldown after sprouting (prevent explosive growth)
        if sprouted > 0:
            self._sprout_cooldown = 2

        return sprouted

    # ------------------------------------------------------------------
    @property
    def alive_connections(self) -> List[SynapticConnection]:
        """Alive connections."""
        return [c for c in self.connections if c.alive]

    @property
    def alive_count(self) -> int:
        """Number of alive connections."""
        return sum(1 for c in self.connections if c.alive)

    @property
    def survival_rate(self) -> float:
        """Survival rate (relative to historical peak)."""
        return self.alive_count / max(1, self.peak_connections)

    # ------------------------------------------------------------------
    def get_specialization_index(self) -> float:
        """
        Specialization index — how "specialized" this region is.

        Computation:
        1. Average |Γ| of alive connections → lower = better matched
        2. Frequency concentration of alive connections → more concentrated = more specialized
        3. Combined: (1 - avg_Γ) × freq_concentration

        0.0 = Completely random (newborn)
        1.0 = Fully specialized (adult cortex)
        """
        alive = self.alive_connections
        if not alive:
            return 0.0

        # Impedance match
        avg_gamma = float(np.mean([c.avg_gamma for c in alive]))
        impedance_match = 1.0 - min(1.0, avg_gamma)

        # Frequency concentration — inverse of coefficient of variation
        freqs = np.array([c.resonant_freq for c in alive])
        if len(freqs) > 1 and np.mean(freqs) > 0:
            cv = float(np.std(freqs) / np.mean(freqs))  # Coefficient of variation
            freq_concentration = 1.0 / (1.0 + cv)
        else:
            freq_concentration = 1.0

        # Survival rate compression (more pruning = stricter selection = more specialized)
        pruning_pressure = 1.0 - self.survival_rate

        # Combined specialization index
        return impedance_match * 0.4 + freq_concentration * 0.3 + pruning_pressure * 0.3

    # ------------------------------------------------------------------
    def get_dominant_frequency(self) -> Tuple[float, float]:
        """
        Get the dominant frequency and spread of alive connections.

        Returns: (dominant_freq, freq_spread)
        """
        alive = self.alive_connections
        if not alive:
            return (0.0, 0.0)

        freqs = [c.resonant_freq for c in alive]
        # Weighted average (synaptic strength weighted)
        strengths = [c.synaptic_strength for c in alive]
        total_w = sum(strengths)
        if total_w > 0:
            weighted_freq = sum(f * s for f, s in zip(freqs, strengths)) / total_w
        else:
            weighted_freq = float(np.mean(freqs))

        spread = float(np.std(freqs))
        return (weighted_freq, spread)

    # ------------------------------------------------------------------
    def get_dominant_band(self) -> BrainWaveBand:
        """Get the dominant brainwave band of alive connections."""
        dom_freq, _ = self.get_dominant_frequency()
        return BrainWaveBand.from_frequency(dom_freq)

    # ------------------------------------------------------------------
    def determine_specialization(self) -> CorticalSpecialization:
        """
        Determine specialization direction based on frequency distribution of alive connections.

        Visual cortex: Dominated by α/β/γ (8-100 Hz)
        Auditory cortex: Dominated by θ/α (4-13 Hz)
        Somatosensory cortex: Broadband (no clear preference)
        Motor cortex: Dominated by β (13-30 Hz), but requires heavy stimulation
        """
        alive = self.alive_connections
        if not alive or self.get_specialization_index() < 0.3:
            self.specialization = CorticalSpecialization.UNSPECIALIZED
            return self.specialization

        dom_freq, spread = self.get_dominant_frequency()
        band = BrainWaveBand.from_frequency(dom_freq)

        # Determine specialization direction
        if band in (BrainWaveBand.BETA, BrainWaveBand.GAMMA):
            if spread < 15.0:
                self.specialization = CorticalSpecialization.VISUAL
            else:
                self.specialization = CorticalSpecialization.MOTOR
        elif band in (BrainWaveBand.THETA, BrainWaveBand.ALPHA):
            self.specialization = CorticalSpecialization.AUDITORY
        elif band == BrainWaveBand.DELTA:
            self.specialization = CorticalSpecialization.SOMATOSENSORY
        else:
            # Broadband → somatosensory
            if spread > 20.0:
                self.specialization = CorticalSpecialization.SOMATOSENSORY
            else:
                self.specialization = CorticalSpecialization.VISUAL

        return self.specialization

    # ------------------------------------------------------------------
    def record_metrics(self) -> PruningMetrics:
        """Record a metrics snapshot of the current state."""
        alive = self.alive_connections
        dom_freq, freq_spread = self.get_dominant_frequency()

        metrics = PruningMetrics(
            cycle=self.stimulation_cycles,
            alive_count=self.alive_count,
            pruned_this_cycle=0,  # Must update after prune()
            sprouted_this_cycle=0,  # Must update after sprout()
            total_pruned=self.pruned_total,
            total_sprouted=self.sprouted_total,
            survival_rate=self.survival_rate,
            avg_gamma=float(np.mean([c.avg_gamma for c in alive])) if alive else 1.0,
            avg_strength=float(np.mean([c.synaptic_strength for c in alive])) if alive else 0.0,
            specialization_index=self.get_specialization_index(),
            dominant_freq=dom_freq,
            freq_spread=freq_spread,
            peak_connections=self.peak_connections,
        )
        self._metrics_history.append(metrics)
        return metrics

    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        """Get the complete state of the region."""
        alive = self.alive_connections
        dom_freq, freq_spread = self.get_dominant_frequency()

        return {
            "name": self.name,
            "initial_connections": self.initial_connections,
            "alive_connections": self.alive_count,
            "pruned_total": self.pruned_total,
            "survival_rate": round(self.survival_rate, 4),
            "stimulation_cycles": self.stimulation_cycles,
            "specialization_index": round(self.get_specialization_index(), 4),
            "specialization": self.determine_specialization().value,
            "dominant_frequency": round(dom_freq, 2),
            "frequency_spread": round(freq_spread, 2),
            "dominant_band": self.get_dominant_band().value,
            "avg_gamma": round(
                float(np.mean([c.avg_gamma for c in alive])) if alive else 1.0, 4
            ),
            "avg_synaptic_strength": round(
                float(np.mean([c.synaptic_strength for c in alive])) if alive else 0.0, 4
            ),
            "region_impedance": self.region_impedance,
            "sprouted_total": self.sprouted_total,
            "peak_connections": self.peak_connections,
            "net_change": self.sprouted_total - self.pruned_total,
        }


# ============================================================================
# 3. Neural pruning engine — whole-brain large-scale Γ apoptosis coordinator
# ============================================================================


# Sensory modality signal characteristics (for automatic development)
MODALITY_SIGNAL_PROFILE: Dict[str, Dict[str, Any]] = {
    "visual": {
        "impedance": 50.0,
        "freq_range": (8.0, 80.0),     # α/β/γ — visual spatial frequency
        "description": "Visual input — spatial frequency mapped to α/β/γ bands",
    },
    "auditory": {
        "impedance": 75.0,
        "freq_range": (4.0, 13.0),     # θ/α — auditory temporal frequency
        "description": "Auditory input — temporal frequency mapped to θ/α bands",
    },
    "somatosensory": {
        "impedance": 50.0,
        "freq_range": (0.5, 50.0),     # Broadband — somatosensory
        "description": "Somatosensory input — broadband pressure/temperature signals",
    },
    "motor": {
        "impedance": 75.0,
        "freq_range": (13.0, 30.0),    # β — motor control
        "description": "Motor feedback — PID error loop β band",
    },
}


class NeuralPruningEngine:
    """
    Neural pruning engine — self-organizing process from random connections to specialized cortex.

    Simulates the neural pruning described in §3.5.2:
    - Birth: ~200 billion synapses, each with random impedance
    - Experience-driven pruning: sensory stimuli → Hebbian selection → apoptosis
    - Result: each cortical region specializes for the signal type it receives

    Intelligence = lim(t→∞) Σ Γ_i² → min

    Usage:
        engine = NeuralPruningEngine(connections_per_region=2000)
        # Simulate development:
        for epoch in range(100):
            result = engine.develop_epoch(
                sensory_diet={"occipital": "visual", "temporal": "auditory", ...}
            )
        # View results:
        state = engine.get_development_state()
    """

    def __init__(
        self,
        connections_per_region: int = 1000,
        z_min: float = 20.0,
        z_max: float = 200.0,
        stimuli_per_epoch: int = 10,
    ):
        """
        Initialize the pruning engine.

        Args:
            connections_per_region: Initial number of connections per cortical region
            z_min: Minimum random impedance
            z_max: Maximum random impedance
            stimuli_per_epoch: Number of stimuli per development epoch
        """
        self.connections_per_region = connections_per_region
        self.stimuli_per_epoch = stimuli_per_epoch

        # Four cortical regions (corresponding to the four major cortices in §3.5.2)
        self.regions: Dict[str, CorticalRegion] = {
            "occipital": CorticalRegion(
                name="occipital",
                initial_connections=connections_per_region,
                region_impedance=50.0,    # Low impedance, open reception
                z_min=z_min, z_max=z_max,
            ),
            "temporal": CorticalRegion(
                name="temporal",
                initial_connections=connections_per_region,
                region_impedance=75.0,    # Standard impedance
                z_min=z_min, z_max=z_max,
            ),
            "parietal": CorticalRegion(
                name="parietal",
                initial_connections=connections_per_region,
                region_impedance=50.0,    # Low impedance, broadband reception
                z_min=z_min, z_max=z_max,
            ),
            "frontal_motor": CorticalRegion(
                name="frontal_motor",
                initial_connections=connections_per_region,
                region_impedance=75.0,    # Standard impedance
                z_min=z_min, z_max=z_max,
                # Motor cortex: harder to prune (requires feedback for calibration)
                match_threshold=0.3,      # Stricter match threshold
            ),
        }

        # Default sensory diet (which region receives which signal type)
        self.default_sensory_diet: Dict[str, str] = {
            "occipital": "visual",
            "temporal": "auditory",
            "parietal": "somatosensory",
            "frontal_motor": "motor",
        }

        # Development tracking
        self.total_epochs: int = 0
        self._epoch_history: List[Dict[str, Any]] = []

        # Whole-brain Γ² sum (physical objective function of intelligence)
        self._gamma_squared_history: List[float] = []

    # ------------------------------------------------------------------
    def _generate_stimuli(
        self,
        modality: str,
        count: int,
    ) -> List[Tuple[float, float]]:
        """
        Generate a list of stimuli signals for a given modality.

        Returns: [(impedance, frequency), ...]
        """
        profile = MODALITY_SIGNAL_PROFILE.get(modality, MODALITY_SIGNAL_PROFILE["visual"])
        z_base = profile["impedance"]
        f_min, f_max = profile["freq_range"]

        stimuli = []
        for _ in range(count):
            # Impedance has slight variation (natural signals are never perfectly uniform)
            z = z_base + np.random.normal(0, z_base * 0.1)
            z = max(10.0, z)
            # Frequency randomized within the modality range (representing different stimuli)
            f = np.random.uniform(f_min, f_max)
            stimuli.append((z, f))

        return stimuli

    # ------------------------------------------------------------------
    def develop_epoch(
        self,
        sensory_diet: Optional[Dict[str, str]] = None,
        motor_feedback_rate: float = 0.3,
        learning_signal: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Execute one development epoch.

        One Epoch represents a period of sensory experience:
        1. Apply modality-corresponding stimuli to each region (Hebbian selection)
        2. Synaptogenesis — strong connections sprout new neighbors
        3. Execute apoptosis (remove overly weak connections)
        4. Record metrics

        Biological trajectory:
          Infancy:  pruning >> synaptogenesis → net decrease (cortical functionalization)
          Learning: synaptogenesis ≈ pruning   → dynamic equilibrium
          Aging:    pruning >> synaptogenesis → degeneration

        Args:
            sensory_diet: Sensory allocation {"region_name": "modality_name"}
            motor_feedback_rate: Motor feedback rate (0~1), simulates "requires feedback for calibration"
                                 → Motor cortex only receives effective stimuli when feedback is present
            learning_signal: Learning signal intensity (0~1), from curiosity / Hebbian / reward
                            Higher → more active synaptogenesis

        Returns:
            Development report for this Epoch
        """
        self.total_epochs += 1
        diet = sensory_diet or self.default_sensory_diet

        epoch_result: Dict[str, Any] = {
            "epoch": self.total_epochs,
            "regions": {},
        }

        total_pruned_this_epoch = 0
        total_sprouted_this_epoch = 0

        for region_name, modality in diet.items():
            if region_name not in self.regions:
                continue

            region = self.regions[region_name]

            # Special handling for motor cortex:
            # Requires feedback for calibration → effective stimuli count is discounted
            # This explains why the frontal motor area develops the slowest
            effective_count = self.stimuli_per_epoch
            if region_name == "frontal_motor":
                effective_count = max(1, int(self.stimuli_per_epoch * motor_feedback_rate))

            # 1. Generate and apply stimuli (Hebbian selection)
            stimuli = self._generate_stimuli(modality, effective_count)
            for z, f in stimuli:
                region.stimulate(z, f)

            # 2. ★ Synaptogenesis — strong connections sprout new neighbors
            #    Before pruning! Let new connections have a chance to grow first
            sprouted = region.sprout(learning_signal=learning_signal)
            total_sprouted_this_epoch += sprouted

            # 3. Execute apoptosis (mismatched ones die, including fragile new connections)
            pruned = region.prune()
            total_pruned_this_epoch += pruned

            # Determine specialization direction
            region.determine_specialization()

            # Record metrics
            metrics = region.record_metrics()
            metrics.pruned_this_cycle = pruned
            metrics.sprouted_this_cycle = sprouted

            epoch_result["regions"][region_name] = {
                "modality_fed": modality,
                "stimuli_count": effective_count,
                "alive": region.alive_count,
                "sprouted_this_epoch": sprouted,
                "pruned_this_epoch": pruned,
                "total_pruned": region.pruned_total,
                "total_sprouted": region.sprouted_total,
                "peak_connections": region.peak_connections,
                "survival_rate": round(region.survival_rate, 4),
                "specialization_index": round(region.get_specialization_index(), 4),
                "specialization": region.specialization.value,
                "dominant_freq": round(metrics.dominant_freq, 2),
                "dominant_band": region.get_dominant_band().value,
                "avg_gamma": round(metrics.avg_gamma, 4),
            }

        # Compute whole-brain Γ² sum
        gamma_sq_sum = self._compute_global_gamma_squared()
        self._gamma_squared_history.append(gamma_sq_sum)

        epoch_result["total_pruned_this_epoch"] = total_pruned_this_epoch
        epoch_result["total_sprouted_this_epoch"] = total_sprouted_this_epoch
        epoch_result["net_change_this_epoch"] = total_sprouted_this_epoch - total_pruned_this_epoch
        epoch_result["global_gamma_squared"] = round(gamma_sq_sum, 6)

        self._epoch_history.append(epoch_result)
        return epoch_result

    # ------------------------------------------------------------------
    def develop(
        self,
        epochs: int = 100,
        sensory_diet: Optional[Dict[str, str]] = None,
        motor_feedback_rate: float = 0.3,
        learning_signal: float = 0.0,
        progress_callback: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple development epochs consecutively.

        Args:
            epochs: Number of development epochs
            sensory_diet: Sensory allocation
            motor_feedback_rate: Motor feedback rate
            learning_signal: Learning signal intensity (0~1)
            progress_callback: Progress callback (receives epoch_result)

        Returns:
            List of results from all Epochs
        """
        results = []
        for i in range(epochs):
            result = self.develop_epoch(sensory_diet, motor_feedback_rate, learning_signal)
            results.append(result)
            if progress_callback:
                progress_callback(result)
        return results

    # ------------------------------------------------------------------
    def _compute_global_gamma_squared(self) -> float:
        """
        Compute whole-brain Σ Γ_i² — physical objective function of intelligence.

        Intelligence = lim(t→∞) Σ Γ_i² → min
        """
        total = 0.0
        count = 0
        for region in self.regions.values():
            for conn in region.alive_connections:
                total += conn.avg_gamma ** 2
                count += 1
        return total / max(1, count)  # Normalize

    # ------------------------------------------------------------------
    def get_development_state(self) -> Dict[str, Any]:
        """
        Get the complete development state report.

        Includes: connection count, survival rate, specialization index, dominant frequency, etc. for each region.
        """
        state: Dict[str, Any] = {
            "total_epochs": self.total_epochs,
            "regions": {},
        }

        total_initial = 0
        total_alive = 0
        total_sprouted = 0
        total_peak = 0

        for name, region in self.regions.items():
            state["regions"][name] = region.get_state()
            total_initial += region.initial_connections
            total_alive += region.alive_count
            total_sprouted += region.sprouted_total
            total_peak += region.peak_connections

        # Whole-brain statistics
        state["overall"] = {
            "total_initial_connections": total_initial,
            "total_alive_connections": total_alive,
            "total_sprouted": total_sprouted,
            "total_peak_connections": total_peak,
            "total_pruned": total_initial + total_sprouted - total_alive,
            "net_change": total_sprouted - (total_initial + total_sprouted - total_alive),
            "overall_survival_rate": round(total_alive / max(1, total_peak), 4),
            "overall_pruning_ratio": round(
                (total_peak - total_alive) / max(1, total_peak), 4
            ),
            "avg_specialization": round(
                float(np.mean([r.get_specialization_index() for r in self.regions.values()])), 4
            ),
            "global_gamma_squared": round(self._compute_global_gamma_squared(), 6),
            "gamma_squared_history": [round(g, 6) for g in self._gamma_squared_history[-50:]],
        }

        # Biological comparison
        # Birth: ~200 billion, after pruning: ~86 billion → ~57% survival
        # But learning involves synaptogenesis → peak can exceed initial
        state["biological_comparison"] = {
            "birth_synapses_bio": "~200 billion",
            "adult_synapses_bio": "~86 billion (at age ~25)",
            "learning_synaptogenesis_bio": "~700 neurons/day in hippocampus (Spalding 2013)",
            "survival_rate_bio": "~43% (birth→adult)",
            "simulated_birth": total_initial,
            "simulated_peak": total_peak,
            "simulated_alive": total_alive,
            "simulated_sprouted": total_sprouted,
            "simulated_survival": round(total_alive / max(1, total_peak), 4),
        }

        return state

    # ------------------------------------------------------------------
    def get_pruning_curve(self) -> Dict[str, List[float]]:
        """
        Get pruning curve data (for plotting).

        Returns:
            History of alive counts per region over epochs
        """
        curves: Dict[str, List[float]] = {name: [] for name in self.regions}
        curves["global_gamma_sq"] = list(self._gamma_squared_history)

        for epoch_result in self._epoch_history:
            for name in self.regions:
                region_data = epoch_result.get("regions", {}).get(name, {})
                curves[name].append(region_data.get("alive", 0))

        return curves

    # ------------------------------------------------------------------
    def cross_modal_experiment(
        self,
        epochs: int = 50,
    ) -> Dict[str, Any]:
        """
        Cross-modal experiment — verify "signal type determines cortical specialization".

        Design:
        1. Control group: Normal allocation (occipital→visual, temporal→auditory)
        2. Experimental group: Cross allocation (occipital→auditory, temporal→visual)

        Prediction: Under cross allocation, the occipital lobe will specialize as auditory cortex.
        This corresponds to the phenomenon of occipital rewiring in congenitally blind individuals.
        """
        # Control group: normal development
        control = NeuralPruningEngine(
            connections_per_region=self.connections_per_region,
        )
        control.develop(epochs, sensory_diet={
            "occipital": "visual",
            "temporal": "auditory",
            "parietal": "somatosensory",
            "frontal_motor": "motor",
        })

        # Experimental group: cross-modal
        crossed = NeuralPruningEngine(
            connections_per_region=self.connections_per_region,
        )
        crossed.develop(epochs, sensory_diet={
            "occipital": "auditory",     # ← visual deprivation, rewired to auditory
            "temporal": "visual",        # ← reversed
            "parietal": "somatosensory",
            "frontal_motor": "motor",
        })

        return {
            "control": control.get_development_state(),
            "crossed": crossed.get_development_state(),
            "conclusion": {
                "control_occipital": control.regions["occipital"].specialization.value,
                "crossed_occipital": crossed.regions["occipital"].specialization.value,
                "control_temporal": control.regions["temporal"].specialization.value,
                "crossed_temporal": crossed.regions["temporal"].specialization.value,
                "rewiring_demonstrated": (
                    crossed.regions["occipital"].specialization != control.regions["occipital"].specialization
                ),
            },
        }

    # ------------------------------------------------------------------
    def generate_report(self, title: str = "Neural Pruning Development Report") -> str:
        """Generate a text report."""
        state = self.get_development_state()
        overall = state["overall"]

        lines = [
            "=" * 70,
            f"  {title}",
            "=" * 70,
            f"  Total development epochs: {state['total_epochs']}",
            f"  Whole-brain Σ Γ² : {overall['global_gamma_squared']:.6f}",
            f"  Initial connections: {overall['total_initial_connections']}",
            f"  Cumulative sprouted: {overall['total_sprouted']}  (synaptogenesis)",
            f"  Historical peak: {overall['total_peak_connections']}",
            f"  Alive connections: {overall['total_alive_connections']}",
            f"  Pruned count: {overall['total_pruned']}",
            f"  Net change   : {overall['net_change']:+d}",
            f"  Survival rate: {overall['overall_survival_rate']:.1%}",
            f"  Avg specialization: {overall['avg_specialization']:.4f}",
            "",
            "  ── Cortical Regions ──",
        ]

        for name, info in state["regions"].items():
            # Survival rate progress bar
            bar_len = 25
            filled = int(info["survival_rate"] * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)

            lines.extend([
                f"",
                f"  [{name}]",
                f"    Alive: {bar} {info['survival_rate']:.1%}"
                f"  ({info['alive_connections']}/{info['peak_connections']})",
                f"    Sprouted: {info['sprouted_total']:+d}  Net change: {info['net_change']:+d}",
                f"    Specialization: {info['specialization']} (index: {info['specialization_index']:.4f})",
                f"    Dominant freq: {info['dominant_frequency']:.1f} Hz ({info['dominant_band']})",
                f"    Avg Γ: {info['avg_gamma']:.4f}  Avg synaptic: {info['avg_synaptic_strength']:.4f}",
            ])

        # Biological comparison
        bio = state["biological_comparison"]
        lines.extend([
            "",
            "  ── Biological Comparison ──",
            f"  Bio birth synapses   : {bio['birth_synapses_bio']}",
            f"  Bio adult synapses   : {bio['adult_synapses_bio']}",
            f"  Hippocampal daily new: {bio['learning_synaptogenesis_bio']}",
            f"  Bio survival rate    : {bio['survival_rate_bio']}",
            f"  Simulated survival   : {bio['simulated_survival']:.1%}",
            "",
            "  ── Physical Conclusion ──",
            f"  Intelligence objective function Σ Γ² = {overall['global_gamma_squared']:.6f}",
            f"  (Closer to 0 = better whole-brain impedance matching = more 'intelligent')",
            "=" * 70,
        ])

        return "\n".join(lines)
