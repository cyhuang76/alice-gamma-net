# -*- coding: utf-8 -*-
"""clinical_neurology.py — Unified Physical Model for Five Major Clinical Neurological Diseases
================================================================

Coaxial cable physics → clinical neuropathology mapping:

│ Disease        │ Physical Mapping                         │ Clinical Scale  │
│───────────────│──────────────────────────────────────────│────────────────│
│ Stroke         │ Acute vascular occlusion → regional Γ jumps to 1.0 │ NIHSS 0-42    │
│ ALS            │ Motor neurons die one by one → Γ progresses to 1.0 │ ALSFRS-R 0-48 │
│ Dementia       │ Diffuse cognitive channel Γ drift                   │ MMSE 0-30     │
│ Alzheimer AD   │ Amyloid plaques = dielectric contamination → Braak staging │ MMSE + Braak │
│ Cerebral Palsy │ Developmental calibration failure → Γ_baseline > 0  │ GMFCS I-V     │

Core Formulas:
    Γ = (Z_L - Z₀) / (Z_L + Z₀)
    Transmission efficiency T = 1 - Γ²
    Clinical score ∝ Σ(affected_channels × Γ²)

The essence of all neurological diseases: different impedance mismatch patterns in communication channels, but the same physical laws.

References:
    [46] Brott, T. et al. (1989). Measurements of acute cerebral infarction:
         a clinical examination scale. *Stroke*, 20(7), 864-870.
    [47] Cedarbaum, J. M. et al. (1999). The ALSFRS-R: a revised ALS
         functional rating scale. *J Neurol Sci*, 169, 13-21.
    [48] Folstein, M. F. et al. (1975). Mini-Mental State: a practical method
         for grading cognitive state. *J Psychiatr Res*, 12(3), 189-198.
    [49] Braak, H. & Braak, E. (1991). Neuropathological stageing of
         Alzheimer-related changes. *Acta Neuropathol*, 82, 239-259.
    [50] Palisano, R. et al. (1997). Development and reliability of a system
         to classify gross motor function in children with cerebral palsy.
         *Dev Med Child Neurol*, 39, 214-223.
    [51] Hardy, J. & Higgins, G. (1992). Alzheimer's disease: the amyloid
         cascade hypothesis. *Science*, 256, 184-185.
    [52] Lance, J. W. (1980). Symposium synopsis. In: Spasticity: Disordered
         Motor Control. Year Book Medical Publishers, pp. 485-494.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ============================================================================
# Physical Constants — Coaxial Cable Clinical Mapping
# ============================================================================

Z_NORMAL: float = 50.0       # Normal channel impedance (Ω)
Z_ISCHEMIC: float = 1e6      # Infarcted tissue impedance (open circuit)

# ---------- Stroke: Vascular Territory → ALICE Module Mapping ----------
VASCULAR_TERRITORIES: Dict[str, List[str]] = {
    "MCA": ["broca", "wernicke", "hand", "perception", "attention", "motor_face"],
    "ACA": ["prefrontal", "motor_gross", "consciousness"],
    "PCA": ["hippocampus", "perception", "thalamus"],
    "basilar": ["autonomic", "calibration", "mouth", "consciousness"],
}

# NIHSS items → ALICE channels + max scores
NIHSS_MAPPING: Dict[str, Dict[str, Any]] = {
    "loc":          {"max": 3, "channels": ["consciousness"]},
    "loc_questions": {"max": 2, "channels": ["consciousness"]},
    "loc_commands": {"max": 2, "channels": ["prefrontal"]},
    "gaze":         {"max": 2, "channels": ["perception"]},
    "visual":       {"max": 3, "channels": ["perception"]},
    "facial":       {"max": 3, "channels": ["motor_face"]},
    "motor_arm":    {"max": 4, "channels": ["hand"]},
    "motor_leg":    {"max": 4, "channels": ["motor_gross"]},
    "ataxia":       {"max": 2, "channels": ["calibration"]},
    "sensory":      {"max": 2, "channels": ["perception"]},
    "language":     {"max": 3, "channels": ["broca", "wernicke"]},
    "dysarthria":   {"max": 2, "channels": ["mouth"]},
    "extinction":   {"max": 2, "channels": ["attention"]},
}
NIHSS_MAX = 42

# ---------- ALS: Progression Regions ----------
ALS_SPREAD_ORDER_LIMB = [
    "hand", "motor_gross", "calibration", "mouth", "broca", "respiratory",
]
ALS_SPREAD_ORDER_BULBAR = [
    "mouth", "broca", "hand", "motor_gross", "calibration", "respiratory",
]
ALS_PROGRESSION_RATE = 0.003        # Health decay per tick
ALS_RILUZOLE_FACTOR = 0.70          # Riluzole slows by 30%
ALS_SPREAD_THRESHOLD = 0.50         # Current region < this value to spread to next region
ALS_FASCICULATION_RANGE = (0.3, 0.8)  # Fasciculations appear in this health range

# ---------- Dementia: MMSE Mapping ----------
DEMENTIA_DOMAINS: Dict[str, Dict[str, Any]] = {
    "orientation":   {"max": 10, "channels": ["calibration"]},
    "registration":  {"max": 3,  "channels": ["hippocampus"]},
    "attention":     {"max": 5,  "channels": ["attention"]},
    "recall":        {"max": 3,  "channels": ["hippocampus"]},
    "language":      {"max": 8,  "channels": ["broca", "wernicke"]},
    "visuospatial":  {"max": 1,  "channels": ["perception"]},
}
MMSE_MAX = 30

DEMENTIA_DRIFT_RATES = {
    "mild":     0.0005,
    "moderate": 0.0010,
    "severe":   0.0020,
}

# CDR (Clinical Dementia Rating) derived from MMSE
CDR_THRESHOLDS = [(26, 0.0), (21, 0.5), (16, 1.0), (11, 2.0), (0, 3.0)]

# ---------- Alzheimer's: Braak Staging → Regional Susceptibility ----------
BRAAK_STAGE_PROFILES: Dict[int, Dict[str, float]] = {
    0: {},
    1: {"hippocampus": 0.15},
    2: {"hippocampus": 0.30, "amygdala": 0.10},
    3: {"hippocampus": 0.50, "amygdala": 0.25, "thalamus": 0.10},
    4: {"hippocampus": 0.70, "amygdala": 0.45, "thalamus": 0.25,
        "wernicke": 0.15},
    5: {"hippocampus": 0.85, "amygdala": 0.60, "thalamus": 0.40,
        "wernicke": 0.35, "prefrontal": 0.30, "broca": 0.20,
        "perception": 0.15},
    6: {"hippocampus": 0.95, "amygdala": 0.80, "thalamus": 0.60,
        "wernicke": 0.55, "prefrontal": 0.50, "broca": 0.40,
        "perception": 0.35, "consciousness": 0.25},
}
AMYLOID_ACCUMULATION_RATE = 0.0008   # Amyloid protein accumulation rate per tick
TAU_PROPAGATION_RATE = 0.0005        # Tau protein propagation rate
TAU_PROPAGATION_LAMBDA = 2.0         # Distance decay constant

# ---------- Cerebral Palsy: GMFCS → Baseline Γ ----------
GMFCS_BASELINE_GAMMA: Dict[int, float] = {
    1: 0.10,   # Can walk, minor limitations
    2: 0.25,   # Can walk, with limitations
    3: 0.45,   # Walks with assistive devices
    4: 0.65,   # Severe self-mobility limitations
    5: 0.85,   # Requires wheelchair transport
}

CP_SPASTICITY_GAIN = 0.8     # Velocity-dependent Γ gain (Lance 1980)
CP_DYSKINETIC_NOISE = 0.15   # Involuntary movement noise amplitude
CP_ATAXIC_PRECISION_GAIN = 0.5  # Precision demand → intention tremor gain

# Channel list shared by all models
ALL_CHANNELS = [
    "consciousness", "perception", "hand", "mouth",
    "broca", "wernicke", "hippocampus", "thalamus",
    "amygdala", "prefrontal", "basal_ganglia",
    "attention", "calibration", "autonomic",
    "motor_gross", "motor_face", "respiratory",
]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class StrokeEvent:
    """A stroke event"""
    territory: str
    severity: float
    onset_tick: int
    core_channels: List[str] = field(default_factory=list)
    penumbra_channels: List[str] = field(default_factory=list)
    reperfused: bool = False
    reperfusion_tick: int = 0


@dataclass
class ALSState:
    """ALS state"""
    onset_type: str            # "limb" | "bulbar"
    onset_tick: int
    progression_rate: float
    riluzole: bool = False
    active_channels: List[str] = field(default_factory=list)


@dataclass
class DementiaState:
    """Dementia state"""
    onset_tick: int
    drift_rate: float
    noise_amplitude: float = 0.02


@dataclass
class AlzheimersState:
    """Alzheimer's disease state"""
    onset_tick: int
    amyloid_rate: float
    tau_rate: float
    genetic_risk: float = 1.0


@dataclass
class CerebralPalsyState:
    """Cerebral palsy state"""
    cp_type: str              # "spastic" | "dyskinetic" | "ataxic"
    gmfcs_level: int          # 1-5
    onset_tick: int = 0


# ============================================================================
# Stroke Model
# ============================================================================

class StrokeModel:
    """
    Acute stroke — vascular occlusion → regional Γ discontinuity

    Physics: vascular occlusion → regional tissue Z → ∞ → Γ → 1.0
    Core (irreversible) vs penumbra (salvageable)
    Clinical scale: NIHSS 0-42 (Brott 1989)

    Penumbra model:
        Core:     Γ = severity × 0.95 (nearly complete open circuit)
        Penumbra: Γ = severity × [0.3~0.7] (partial ischemia)
        Reperfusion (within 4.5h) → accelerated penumbra recovery
    """

    PENUMBRA_FRACTION = 0.40
    CORE_GAMMA_FACTOR = 0.95
    PENUMBRA_GAMMA_RANGE = (0.30, 0.70)
    REPERFUSION_WINDOW = 270       # 4.5h at 1 tick/min
    NATURAL_RECOVERY_RATE = 0.001
    REPERFUSION_RECOVERY_RATE = 0.01
    DIASCHISIS_GAMMA = 0.08        # Diaschisis (remote inhibition effect)

    def __init__(self) -> None:
        self.strokes: List[StrokeEvent] = []
        self.channel_gamma: Dict[str, float] = {}
        self._tick = 0

    # ------------------------------------------------------------------
    def induce(self, territory: str = "MCA", severity: float = 0.8) -> StrokeEvent:
        """Induce stroke: specify vascular territory and severity"""
        severity = max(0.0, min(1.0, severity))
        channels = list(VASCULAR_TERRITORIES.get(territory, []))
        if not channels:
            raise ValueError(f"Unknown territory: {territory}")

        n_core = max(1, int(len(channels) * (1 - self.PENUMBRA_FRACTION)))
        core = channels[:n_core]
        penumbra = channels[n_core:]

        event = StrokeEvent(
            territory=territory,
            severity=severity,
            onset_tick=self._tick,
            core_channels=core,
            penumbra_channels=penumbra,
        )
        self.strokes.append(event)

        # Core zone Γ
        for ch in core:
            self.channel_gamma[ch] = max(
                self.channel_gamma.get(ch, 0),
                self.CORE_GAMMA_FACTOR * severity,
            )
        # Penumbra zone Γ
        for ch in penumbra:
            lo, hi = self.PENUMBRA_GAMMA_RANGE
            gamma = lo + (hi - lo) * severity
            self.channel_gamma[ch] = max(self.channel_gamma.get(ch, 0), gamma)

        # Remote inhibition (Diaschisis)
        for ch in ALL_CHANNELS:
            if ch not in self.channel_gamma:
                self.channel_gamma[ch] = self.DIASCHISIS_GAMMA * severity

        return event

    # ------------------------------------------------------------------
    def reperfuse(self, stroke_idx: int = 0) -> bool:
        """Thrombolysis / mechanical thrombectomy — only effective for penumbra within time window"""
        if stroke_idx >= len(self.strokes):
            return False
        stroke = self.strokes[stroke_idx]
        elapsed = self._tick - stroke.onset_tick
        if elapsed <= self.REPERFUSION_WINDOW and not stroke.reperfused:
            stroke.reperfused = True
            stroke.reperfusion_tick = self._tick
            return True
        return False

    # ------------------------------------------------------------------
    def tick(self, brain_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Advance one timestep"""
        self._tick += 1
        for stroke in self.strokes:
            rate = (self.REPERFUSION_RECOVERY_RATE
                    if stroke.reperfused else self.NATURAL_RECOVERY_RATE)
            # Penumbra recovery
            for ch in stroke.penumbra_channels:
                if ch in self.channel_gamma:
                    self.channel_gamma[ch] = max(
                        0.0, self.channel_gamma[ch] - rate
                    )
            # Diaschisis slow recovery
            for ch in ALL_CHANNELS:
                if (ch not in stroke.core_channels
                        and ch not in stroke.penumbra_channels
                        and ch in self.channel_gamma):
                    self.channel_gamma[ch] = max(
                        0.0, self.channel_gamma[ch] - rate * 0.5
                    )
            # Core zone does not recover

        return {
            "channel_gamma": dict(self.channel_gamma),
            "nihss": self.get_nihss(),
            "active_strokes": len(self.strokes),
        }

    # ------------------------------------------------------------------
    def get_nihss(self) -> int:
        """Compute NIH Stroke Scale (0-42)"""
        score = 0.0
        for _item, info in NIHSS_MAPPING.items():
            channels = info["channels"]
            if not channels:
                continue
            mean_gamma = sum(
                self.channel_gamma.get(ch, 0) for ch in channels
            ) / len(channels)
            score += info["max"] * mean_gamma
        return min(NIHSS_MAX, round(score))

    # ------------------------------------------------------------------
    def introspect(self) -> Dict[str, Any]:
        return {
            "condition": "stroke",
            "active": len(self.strokes) > 0,
            "stroke_count": len(self.strokes),
            "nihss": self.get_nihss(),
            "channel_gamma": dict(self.channel_gamma),
            "events": [
                {"territory": s.territory, "severity": s.severity,
                 "reperfused": s.reperfused, "tick": s.onset_tick}
                for s in self.strokes
            ],
        }


# ============================================================================
# ALS Model
# ============================================================================

class ALSModel:
    """
    Amyotrophic Lateral Sclerosis (ALS)

    Physics: motor neurons die one by one → channel Γ progresses to 1.0
    El Escorial criteria: diagnosis requires multi-regional upper + lower motor neuron involvement
    Clinical scale: ALSFRS-R 0-48 (Cedarbaum 1999)

    Channel health h(t) = h₀ × exp(-k × (t - t_onset))
    - k = 0.003 (untreated)
    - k = 0.003 × 0.70 (Riluzole, Bensimon 1994)
    When h < spread_threshold → next region begins to degenerate
    """

    def __init__(self) -> None:
        self.state: Optional[ALSState] = None
        self.channel_health: Dict[str, float] = {}
        self._fasciculations: List[Dict[str, Any]] = []
        self._tick = 0
        self._spread_index = 0

    # ------------------------------------------------------------------
    def onset(self, onset_type: str = "limb",
              riluzole: bool = False) -> ALSState:
        """Disease onset"""
        rate = ALS_PROGRESSION_RATE
        if riluzole:
            rate *= ALS_RILUZOLE_FACTOR
        spread = (ALS_SPREAD_ORDER_LIMB if onset_type == "limb"
                  else ALS_SPREAD_ORDER_BULBAR)

        self.state = ALSState(
            onset_type=onset_type,
            onset_tick=self._tick,
            progression_rate=rate,
            riluzole=riluzole,
            active_channels=[spread[0]],
        )
        self.channel_health = {ch: 1.0 for ch in spread}
        self._spread_index = 0
        return self.state

    # ------------------------------------------------------------------
    def tick(self, brain_state: Optional[Dict] = None) -> Dict[str, Any]:
        self._tick += 1
        if self.state is None:
            return {"active": False}

        spread = (ALS_SPREAD_ORDER_LIMB if self.state.onset_type == "limb"
                  else ALS_SPREAD_ORDER_BULBAR)
        rate = self.state.progression_rate

        # Activated channels degenerate
        for ch in self.state.active_channels:
            if ch in self.channel_health:
                self.channel_health[ch] *= math.exp(-rate)
                self.channel_health[ch] = max(0.0, self.channel_health[ch])

        # Spread check
        if self._spread_index < len(spread) - 1:
            current_ch = spread[self._spread_index]
            if self.channel_health.get(current_ch, 1.0) < ALS_SPREAD_THRESHOLD:
                self._spread_index += 1
                next_ch = spread[self._spread_index]
                if next_ch not in self.state.active_channels:
                    self.state.active_channels.append(next_ch)

        # Fasciculations
        self._fasciculations.clear()
        for ch in self.state.active_channels:
            h = self.channel_health.get(ch, 1.0)
            lo, hi = ALS_FASCICULATION_RANGE
            if lo < h < hi and random.random() < 0.1:
                self._fasciculations.append({
                    "channel": ch, "tick": self._tick, "health": h
                })

        return {
            "active": True,
            "channel_health": dict(self.channel_health),
            "channel_gamma": self.get_channel_gamma(),
            "alsfrs_r": self.get_alsfrs_r(),
            "fasciculations": list(self._fasciculations),
            "respiratory_failure": self.channel_health.get("respiratory", 1.0) < 0.1,
        }

    # ------------------------------------------------------------------
    def get_channel_gamma(self) -> Dict[str, float]:
        """Health → Γ: h=1.0 → Γ=0, h=0.0 → Γ=1.0"""
        return {ch: 1.0 - h for ch, h in self.channel_health.items()}

    # ------------------------------------------------------------------
    def get_alsfrs_r(self) -> int:
        """
        ALSFRS-R (0-48, 48=healthy)
        4 domains × 3 items × 4 points each
        """
        if self.state is None:
            return 48

        domains = {
            "bulbar":      ["mouth", "broca"],
            "fine_motor":  ["hand"],
            "gross_motor": ["motor_gross", "calibration"],
            "respiratory": ["respiratory"],
        }

        score = 0.0
        for _domain, channels in domains.items():
            mean_health = sum(
                self.channel_health.get(ch, 1.0) for ch in channels
            ) / max(1, len(channels))
            score += 12.0 * mean_health  # 12 points per domain

        return min(48, round(score))

    # ------------------------------------------------------------------
    def introspect(self) -> Dict[str, Any]:
        return {
            "condition": "als",
            "active": self.state is not None,
            "onset_type": self.state.onset_type if self.state else None,
            "riluzole": self.state.riluzole if self.state else False,
            "alsfrs_r": self.get_alsfrs_r(),
            "channel_health": dict(self.channel_health),
            "spread_index": self._spread_index,
        }


# ============================================================================
# Dementia Model
# ============================================================================

class DementiaModel:
    """
    Dementia — diffuse cognitive channel impedance drift

    Physics: multi-domain cognitive channel Γ rises slowly and simultaneously
    Γ_domain(t) = min(1.0, drift_rate × (t - onset) + noise)
    Clinical scale: MMSE 0-30 (Folstein 1975), CDR 0-3 (Morris 1993)

    Domain degradation order (typical): memory → executive → language → visuospatial → social
    """

    DOMAIN_ONSET_DELAYS = {
        "hippocampus": 0,        # Memory first
        "prefrontal":  200,      # Executive function
        "attention":   300,
        "broca":       400,
        "wernicke":    400,
        "perception":  500,
        "calibration": 600,
    }

    def __init__(self) -> None:
        self.state: Optional[DementiaState] = None
        self.domain_gamma: Dict[str, float] = {}
        self._tick = 0

    # ------------------------------------------------------------------
    def onset(self, severity: str = "mild") -> DementiaState:
        drift_rate = DEMENTIA_DRIFT_RATES.get(severity, 0.0005)
        self.state = DementiaState(
            onset_tick=self._tick,
            drift_rate=drift_rate,
        )
        self.domain_gamma = {ch: 0.0 for ch in self.DOMAIN_ONSET_DELAYS}
        return self.state

    # ------------------------------------------------------------------
    def tick(self, brain_state: Optional[Dict] = None) -> Dict[str, Any]:
        self._tick += 1
        if self.state is None:
            return {"active": False}

        elapsed = self._tick - self.state.onset_tick
        for ch, delay in self.DOMAIN_ONSET_DELAYS.items():
            if elapsed > delay:
                effective_t = elapsed - delay
                drift = self.state.drift_rate * effective_t
                noise = random.gauss(0, self.state.noise_amplitude)
                self.domain_gamma[ch] = max(
                    0.0, min(1.0, drift + noise)
                )

        return {
            "active": True,
            "domain_gamma": dict(self.domain_gamma),
            "mmse": self.get_mmse(),
            "cdr": self.get_cdr(),
        }

    # ------------------------------------------------------------------
    def get_mmse(self) -> int:
        """Mini-Mental State Examination (0-30, 30=healthy)"""
        if self.state is None:
            return MMSE_MAX

        score = 0.0
        for _domain, info in DEMENTIA_DOMAINS.items():
            channels = info["channels"]
            mean_gamma = sum(
                self.domain_gamma.get(ch, 0) for ch in channels
            ) / max(1, len(channels))
            transmission = 1.0 - mean_gamma ** 2
            score += info["max"] * transmission

        return max(0, min(MMSE_MAX, round(score)))

    # ------------------------------------------------------------------
    def get_cdr(self) -> float:
        """Clinical Dementia Rating (0, 0.5, 1, 2, 3)"""
        mmse = self.get_mmse()
        for threshold, cdr in CDR_THRESHOLDS:
            if mmse >= threshold:
                return cdr
        return 3.0

    # ------------------------------------------------------------------
    def introspect(self) -> Dict[str, Any]:
        return {
            "condition": "dementia",
            "active": self.state is not None,
            "mmse": self.get_mmse(),
            "cdr": self.get_cdr(),
            "domain_gamma": dict(self.domain_gamma),
        }


# ============================================================================
# Alzheimer's Model
# ============================================================================

class AlzheimersModel:
    """
    Alzheimer's Disease — amyloid plaques = dielectric contamination

    Physics: Amyloid-β accumulates in coaxial cable insulation → ε↑ → Z↑ → Γ↑
        Z_coax = (1/2π) × √(μ/ε) × ln(r_outer/r_inner)
        Plaque: ε_eff = ε₀ × (1 + amyloid_load)
        → Z ↓ actually BUT protein plaques create impedance discontinuities
        → Net effect: Γ ↑ due to scattering at plaque boundaries

    Braak staging: hippocampus → temporal lobe → frontal lobe → whole brain
    "Amyloid is the match, Tau is the fire" — Hardy & Higgins 1992

    Clinical scale: MMSE + Braak Stage
    """

    # Inter-regional "connection distance" (for Tau propagation)
    REGION_DISTANCES: Dict[str, Dict[str, float]] = {
        "hippocampus": {"amygdala": 1.0, "thalamus": 1.5, "wernicke": 2.0,
                        "prefrontal": 2.5, "broca": 3.0, "perception": 3.5,
                        "consciousness": 4.0},
    }

    def __init__(self) -> None:
        self.state: Optional[AlzheimersState] = None
        self.amyloid_load: Dict[str, float] = {}
        self.tau_load: Dict[str, float] = {}
        self.channel_gamma: Dict[str, float] = {}
        self._tick = 0

    # ------------------------------------------------------------------
    def onset(self, genetic_risk: float = 1.0) -> AlzheimersState:
        """Disease onset. genetic_risk > 1.0 indicates carrying APOE ε4 or other risk genes"""
        self.state = AlzheimersState(
            onset_tick=self._tick,
            amyloid_rate=AMYLOID_ACCUMULATION_RATE * genetic_risk,
            tau_rate=TAU_PROPAGATION_RATE * genetic_risk,
            genetic_risk=genetic_risk,
        )
        # Amyloid begins depositing from hippocampus
        self.amyloid_load = {"hippocampus": 0.01}
        self.tau_load = {"hippocampus": 0.01}
        return self.state

    # ------------------------------------------------------------------
    def tick(self, brain_state: Optional[Dict] = None) -> Dict[str, Any]:
        self._tick += 1
        if self.state is None:
            return {"active": False}

        # 1. Amyloid accumulation (all deposited regions)
        for region in list(self.amyloid_load.keys()):
            self.amyloid_load[region] = min(
                1.0,
                self.amyloid_load[region] + self.state.amyloid_rate
            )

        # 2. Tau propagation (prion-like spreading from hippocampus)
        hippo_tau = self.tau_load.get("hippocampus", 0)
        if hippo_tau > 0.1:  # Tau needs to accumulate to a certain level before spreading
            distances = self.REGION_DISTANCES.get("hippocampus", {})
            for region, dist in distances.items():
                propagation = (hippo_tau
                               * math.exp(-dist / TAU_PROPAGATION_LAMBDA)
                               * self.state.tau_rate)
                current = self.tau_load.get(region, 0)
                self.tau_load[region] = min(1.0, current + propagation)
                # Tau arrives → triggers amyloid deposition in that region
                if region not in self.amyloid_load and self.tau_load[region] > 0.05:
                    self.amyloid_load[region] = 0.01

        # 3. Tau self-accumulation
        for region in list(self.tau_load.keys()):
            self.tau_load[region] = min(
                1.0,
                self.tau_load[region] + self.state.tau_rate * 0.5
            )

        # 4. Compute channel Γ (amyloid + tau joint damage)
        for region in set(list(self.amyloid_load.keys()) +
                          list(self.tau_load.keys())):
            amyloid = self.amyloid_load.get(region, 0)
            tau = self.tau_load.get(region, 0)
            # Γ = amyloid scattering + Tau neurofibrillary tangles
            combined = min(1.0, amyloid * 0.4 + tau * 0.6)
            self.channel_gamma[region] = combined

        return {
            "active": True,
            "braak_stage": self.get_braak_stage(),
            "mmse": self.get_mmse(),
            "amyloid_load": dict(self.amyloid_load),
            "tau_load": dict(self.tau_load),
            "channel_gamma": dict(self.channel_gamma),
        }

    # ------------------------------------------------------------------
    def get_braak_stage(self) -> int:
        """Determine Braak staging based on damage pattern (0-6)"""
        hippo_g = self.channel_gamma.get("hippocampus", 0)
        amyg_g = self.channel_gamma.get("amygdala", 0)
        pfc_g = self.channel_gamma.get("prefrontal", 0)
        occ_g = self.channel_gamma.get("perception", 0)

        if hippo_g < 0.05:
            return 0
        if hippo_g < 0.20:
            return 1
        if hippo_g < 0.40 and amyg_g < 0.15:
            return 2
        if hippo_g < 0.55:
            return 3
        if pfc_g < 0.20:
            return 4
        if occ_g < 0.25:
            return 5
        return 6

    # ------------------------------------------------------------------
    def get_mmse(self) -> int:
        """Compute MMSE from channel Γ"""
        score = 0.0
        for _domain, info in DEMENTIA_DOMAINS.items():
            channels = info["channels"]
            mean_gamma = sum(
                self.channel_gamma.get(ch, 0) for ch in channels
            ) / max(1, len(channels))
            transmission = 1.0 - mean_gamma ** 2
            score += info["max"] * transmission
        return max(0, min(MMSE_MAX, round(score)))

    # ------------------------------------------------------------------
    def introspect(self) -> Dict[str, Any]:
        return {
            "condition": "alzheimers",
            "active": self.state is not None,
            "braak_stage": self.get_braak_stage(),
            "mmse": self.get_mmse(),
            "amyloid_load": dict(self.amyloid_load),
            "tau_load": dict(self.tau_load),
            "channel_gamma": dict(self.channel_gamma),
        }


# ============================================================================
# Cerebral Palsy Model
# ============================================================================

class CerebralPalsyModel:
    """
    Cerebral Palsy (CP) — developmental impedance calibration failure

    Physics: brain injury at birth → impedance can never be correctly calibrated → Γ_baseline > 0 (permanent)
    Static brain lesion: injury doesn't progress, but manifestation changes with development

    Three types with different impedance patterns:
    1. Spastic (70-80%): velocity-dependent Γ increase (Lance 1980)
       → Γ(v) = Γ_baseline + spasticity_gain × |v|
    2. Dyskinetic (10-20%): random Γ fluctuation (involuntary movement)
       → Γ(t) = Γ_baseline + noise(t)
    3. Ataxic (5-10%): calibration error amplification (intention tremor)
       → Γ(precision) = Γ_baseline × (1 + precision_gain × demand)

    Clinical scale: GMFCS I-V (Palisano 1997)
    """

    # Motor channels primarily affected by CP
    MOTOR_CHANNELS = ["hand", "motor_gross", "calibration", "mouth"]
    COGNITIVE_CHANNELS = ["attention", "perception"]  # Mild cognitive impact

    def __init__(self) -> None:
        self.state: Optional[CerebralPalsyState] = None
        self.baseline_gamma: Dict[str, float] = {}
        self.current_gamma: Dict[str, float] = {}
        self._tick = 0

    # ------------------------------------------------------------------
    def set_condition(self, cp_type: str = "spastic",
                      gmfcs_level: int = 3) -> CerebralPalsyState:
        """Set cerebral palsy type and severity"""
        gmfcs_level = max(1, min(5, gmfcs_level))
        base_g = GMFCS_BASELINE_GAMMA[gmfcs_level]

        self.state = CerebralPalsyState(
            cp_type=cp_type,
            gmfcs_level=gmfcs_level,
            onset_tick=self._tick,
        )

        # Motor channels: high baseline Γ
        for ch in self.MOTOR_CHANNELS:
            self.baseline_gamma[ch] = base_g

        # Cognitive channels: lower baseline (CP primarily affects motor)
        for ch in self.COGNITIVE_CHANNELS:
            self.baseline_gamma[ch] = base_g * 0.3

        self.current_gamma = dict(self.baseline_gamma)
        return self.state

    # ------------------------------------------------------------------
    def tick(self, brain_state: Optional[Dict] = None,
             motor_velocity: float = 0.0,
             precision_demand: float = 0.0) -> Dict[str, Any]:
        """Advance one timestep"""
        self._tick += 1
        if self.state is None:
            return {"active": False}

        cp_type = self.state.cp_type

        for ch in list(self.baseline_gamma.keys()):
            base = self.baseline_gamma[ch]

            if cp_type == "spastic":
                # Lance (1980): velocity-dependent spasticity
                velocity_component = CP_SPASTICITY_GAIN * abs(motor_velocity)
                self.current_gamma[ch] = min(
                    1.0, base + velocity_component
                )

            elif cp_type == "dyskinetic":
                # Involuntary movement = random Γ fluctuation
                noise = random.gauss(0, CP_DYSKINETIC_NOISE)
                self.current_gamma[ch] = max(
                    0.0, min(1.0, base + noise)
                )

            elif cp_type == "ataxic":
                # Intention tremor = higher precision demand → higher Γ
                precision_component = (CP_ATAXIC_PRECISION_GAIN
                                       * precision_demand)
                self.current_gamma[ch] = min(
                    1.0, base + precision_component
                )

        return {
            "active": True,
            "cp_type": cp_type,
            "gmfcs": self.state.gmfcs_level,
            "channel_gamma": dict(self.current_gamma),
            "spasticity_index": self._compute_spasticity_index(),
        }

    # ------------------------------------------------------------------
    def _compute_spasticity_index(self) -> float:
        """Modified Ashworth Scale approximation (0-4)"""
        if self.state is None:
            return 0.0
        gammas = [self.current_gamma.get(ch, 0)
                  for ch in self.MOTOR_CHANNELS]
        mean_g = sum(gammas) / max(1, len(gammas))
        return round(mean_g * 4.0, 1)  # 0-4 scale

    # ------------------------------------------------------------------
    def get_gmfcs(self) -> int:
        """GMFCS Level (1-5)"""
        if self.state is None:
            return 0
        return self.state.gmfcs_level

    # ------------------------------------------------------------------
    def introspect(self) -> Dict[str, Any]:
        return {
            "condition": "cerebral_palsy",
            "active": self.state is not None,
            "cp_type": self.state.cp_type if self.state else None,
            "gmfcs": self.get_gmfcs(),
            "baseline_gamma": dict(self.baseline_gamma),
            "current_gamma": dict(self.current_gamma),
            "spasticity_index": self._compute_spasticity_index(),
        }


# ============================================================================
# Unified Clinical Neurology Engine
# ============================================================================

class ClinicalNeurologyEngine:
    """
    Unified physical model for five major clinical neurological diseases

    All neurological diseases are different patterns of coaxial cable impedance matching failure:
    - Stroke: acute, regional, salvageable (penumbra)
    - ALS: progressive, motor system, irreversible
    - Dementia: diffuse, cognitive system, slow
    - Alzheimer's: protein deposition, specific staging pattern
    - Cerebral palsy: developmental, static, primarily motor

    Unified output: channel_gamma dict for main loop queries
    """

    def __init__(self) -> None:
        self.stroke = StrokeModel()
        self.als = ALSModel()
        self.dementia = DementiaModel()
        self.alzheimers = AlzheimersModel()
        self.cerebral_palsy = CerebralPalsyModel()
        self._tick = 0

    # ------------------------------------------------------------------
    def tick(self, brain_state: Optional[Dict] = None,
             motor_velocity: float = 0.0,
             precision_demand: float = 0.0) -> Dict[str, Any]:
        """Advance all active disease models"""
        self._tick += 1

        results: Dict[str, Any] = {"tick": self._tick}

        # Tick all models
        if self.stroke.strokes:
            results["stroke"] = self.stroke.tick(brain_state)
        if self.als.state is not None:
            results["als"] = self.als.tick(brain_state)
        if self.dementia.state is not None:
            results["dementia"] = self.dementia.tick(brain_state)
        if self.alzheimers.state is not None:
            results["alzheimers"] = self.alzheimers.tick(brain_state)
        if self.cerebral_palsy.state is not None:
            results["cerebral_palsy"] = self.cerebral_palsy.tick(
                brain_state, motor_velocity, precision_demand
            )

        # Merge all channel Γ (take max — worst severity for comorbidities)
        results["merged_channel_gamma"] = self.get_merged_channel_gamma()
        results["active_conditions"] = self.get_active_conditions()

        return results

    # ------------------------------------------------------------------
    def get_merged_channel_gamma(self) -> Dict[str, float]:
        """Merge all active diseases' channel Γ (take max — most severe takes priority)"""
        merged: Dict[str, float] = {}

        sources = []
        if self.stroke.strokes:
            sources.append(self.stroke.channel_gamma)
        if self.als.state is not None:
            sources.append(self.als.get_channel_gamma())
        if self.dementia.state is not None:
            sources.append(self.dementia.domain_gamma)
        if self.alzheimers.state is not None:
            sources.append(self.alzheimers.channel_gamma)
        if self.cerebral_palsy.state is not None:
            sources.append(self.cerebral_palsy.current_gamma)

        for source in sources:
            for ch, gamma in source.items():
                merged[ch] = max(merged.get(ch, 0), gamma)

        return merged

    # ------------------------------------------------------------------
    def get_active_conditions(self) -> List[str]:
        """List all active diseases"""
        active = []
        if self.stroke.strokes:
            active.append("stroke")
        if self.als.state is not None:
            active.append("als")
        if self.dementia.state is not None:
            active.append("dementia")
        if self.alzheimers.state is not None:
            active.append("alzheimers")
        if self.cerebral_palsy.state is not None:
            active.append("cerebral_palsy")
        return active

    # ------------------------------------------------------------------
    def get_clinical_summary(self) -> Dict[str, Any]:
        """Overview of all clinical scales"""
        summary: Dict[str, Any] = {}
        if self.stroke.strokes:
            summary["nihss"] = self.stroke.get_nihss()
        if self.als.state is not None:
            summary["alsfrs_r"] = self.als.get_alsfrs_r()
        if self.dementia.state is not None:
            summary["mmse_dementia"] = self.dementia.get_mmse()
            summary["cdr"] = self.dementia.get_cdr()
        if self.alzheimers.state is not None:
            summary["mmse_ad"] = self.alzheimers.get_mmse()
            summary["braak_stage"] = self.alzheimers.get_braak_stage()
        if self.cerebral_palsy.state is not None:
            summary["gmfcs"] = self.cerebral_palsy.get_gmfcs()
        return summary

    # ------------------------------------------------------------------
    def introspect(self) -> Dict[str, Any]:
        return {
            "active_conditions": self.get_active_conditions(),
            "clinical_summary": self.get_clinical_summary(),
            "stroke": self.stroke.introspect(),
            "als": self.als.introspect(),
            "dementia": self.dementia.introspect(),
            "alzheimers": self.alzheimers.introspect(),
            "cerebral_palsy": self.cerebral_palsy.introspect(),
        }

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "tick": self._tick,
            "active_conditions": self.get_active_conditions(),
            "merged_channel_gamma": self.get_merged_channel_gamma(),
            **self.get_clinical_summary(),
        }
