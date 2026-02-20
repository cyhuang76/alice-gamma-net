# -*- coding: utf-8 -*-
"""pharmacology.py — Computational Pharmacology Engine + Four Neurological Disease Extensions
================================================================

Unified framework: "every drug = an impedance modification factor α_drug"

    Z_eff = Z_tissue × (1 + α_drug)

    α < 0 → lower impedance (therapeutic)
    α > 0 → raise impedance (side effect / toxicity)
    α = 0 → placebo

All four new diseases are based on different failure modes of coaxial cable physics:

│ Disease        │ Physical Mapping                              │ Clinical Scale     │
│───────────────│───────────────────────────────────────────────│───────────────────│
│ MS             │ Demyelination = insulation stripping → leakage along path │ EDSS 0-10         │
│ Parkinson PD   │ Substantia nigra DA depletion → basal ganglia circuit failure → tremor/rigidity │ UPDRS 0-199 │
│ Epilepsy       │ Excitation/inhibition imbalance → positive feedback → Γ oscillation amplification │ Seizure freq + severity │
│ Depression MDD │ Chronic low-level Γ elevation → sustained low emotional transmission │ HAM-D 0-52 │

References:
    [53] Kurtzke, J. F. (1983). Rating neurological impairment in multiple
         sclerosis: an expanded disability status scale (EDSS). *Neurology*,
         33(11), 1444-1452.
    [54] Fahn, S. & Elton, R. L. (1987). Unified Parkinson's Disease Rating
         Scale. In: *Recent Developments in PD*, Vol. 2, pp. 153-163.
    [55] Fisher, R. S. et al. (2017). ILAE classification of the epilepsies.
         *Epilepsia*, 58(4), 512-521.
    [56] Hamilton, M. (1960). A rating scale for depression. *J Neurol
         Neurosurg Psychiatry*, 23(1), 56-62.
    [57] Compston, A. & Coles, A. (2008). Multiple sclerosis. *The Lancet*,
         372(9648), 1502-1517.
    [58] Bensimon, G. et al. (1994). A controlled trial of riluzole in ALS.
         *NEJM*, 330(9), 585-591.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# Physical Constants — Pharmacology + New Disease Mapping
# ============================================================================

Z_NORMAL: float = 50.0         # Normal channel impedance (Ω)

# ---------- Unified Drug Channel List ----------
ALL_CHANNELS = [
    "consciousness", "perception", "hand", "mouth",
    "broca", "wernicke", "hippocampus", "thalamus",
    "amygdala", "prefrontal", "basal_ganglia",
    "attention", "calibration", "autonomic",
    "motor_gross", "motor_face", "respiratory",
]

# ---------- MS: White Matter Tracts → Channel Mapping ----------
MS_TRACT_CHANNELS: Dict[str, List[str]] = {
    "optic_nerve":       ["perception"],
    "corticospinal":     ["hand", "motor_gross", "motor_face"],
    "cerebellar":        ["calibration"],
    "brainstem":         ["autonomic", "mouth", "respiratory"],
    "corpus_callosum":   ["consciousness", "attention", "prefrontal"],
}

# EDSS scoring items → functional systems
EDSS_FUNCTIONAL_SYSTEMS: Dict[str, Dict[str, Any]] = {
    "visual":     {"max": 6, "channels": ["perception"]},
    "brainstem":  {"max": 5, "channels": ["autonomic", "mouth"]},
    "pyramidal":  {"max": 6, "channels": ["hand", "motor_gross"]},
    "cerebellar": {"max": 5, "channels": ["calibration"]},
    "sensory":    {"max": 6, "channels": ["perception", "thalamus"]},
    "bowel":      {"max": 5, "channels": ["autonomic"]},
    "cerebral":   {"max": 5, "channels": ["prefrontal", "consciousness"]},
}
EDSS_MAX = 10.0
MS_RELAPSE_RATE = 0.002         # Relapse probability per tick (RRMS)
MS_DEMYELINATION_RATE = 0.001   # Persistent demyelination (progressive)
MS_REMYELINATION_RATE = 0.0003  # Remyelination during remission
MS_LESION_GAMMA_RANGE = (0.2, 0.6)  # Single lesion Γ range

# ---------- PD: Basal Ganglia Dopamine ----------
PD_DOPAMINE_DEPLETION_RATE = 0.0005    # Dopamine decay per tick
PD_TREMOR_FREQUENCY = 5.0               # 4-6 Hz resting tremor
PD_RIGIDITY_GAIN = 0.7                  # Rigidity Γ gain
PD_BRADYKINESIA_LATENCY_FACTOR = 2.0    # Response delay multiplier
PD_MOTOR_CHANNELS = ["hand", "motor_gross", "motor_face", "mouth"]
PD_COGNITIVE_CHANNELS = ["prefrontal", "attention", "hippocampus"]

# UPDRS (simplified): 4 sections, 0~50 each
UPDRS_SECTIONS: Dict[str, Dict[str, Any]] = {
    "mentation":  {"max": 16, "channels": ["prefrontal", "hippocampus"]},
    "adl":        {"max": 52, "channels": ["hand", "motor_gross", "mouth"]},
    "motor":      {"max": 108, "channels": ["hand", "motor_gross", "motor_face",
                                             "calibration"]},
    "complications": {"max": 23, "channels": ["basal_ganglia"]},
}
UPDRS_MAX = 199

# L-DOPA Pharmacology
LDOPA_ALPHA = -0.35                # Impedance modification factor (therapeutic)
LDOPA_WEARING_OFF_RATE = 0.0008    # Drug effect decay
LDOPA_DYSKINESIA_THRESHOLD = 2000  # Long-term use exceeding this ticks → dyskinesia

# ---------- Epilepsy: Excitation/Inhibition Balance ----------
SEIZURE_THRESHOLD = 0.80       # Excitation exceeds this → seizure
KINDLING_INCREMENT = 0.005     # Each seizure lowers threshold (kindling effect)
SEIZURE_DURATION_RANGE = (10, 60)   # Seizure duration ticks
POSTICTAL_DEPRESSION = 0.15    # Post-ictal depression depth
POSTICTAL_DURATION = 100       # Post-ictal recovery period ticks
EXCITATION_NOISE = 0.03        # Background excitation noise
INHIBITION_DECAY = 0.002       # Inhibition natural decay

EPILEPSY_FOCAL_CHANNELS: Dict[str, List[str]] = {
    "temporal":  ["hippocampus", "amygdala", "wernicke"],
    "frontal":   ["prefrontal", "motor_gross", "broca"],
    "parietal":  ["perception", "attention"],
    "occipital": ["perception"],
}

# Anti-epileptic drugs
VALPROATE_ALPHA = -0.25        # Effect of raising seizure threshold
CARBAMAZEPINE_ALPHA = -0.20

# ---------- Depression: Monoamine Hypothesis ----------
SEROTONIN_BASELINE = 1.0
SEROTONIN_DEPLETION_RATE = 0.0003   # Chronic depletion
NOREPINEPHRINE_BASELINE = 1.0
NE_DEPLETION_RATE = 0.0002
ANHEDONIA_REWARD_PENALTY = 0.6      # Reward pathway efficiency reduced by 40%
COGNITIVE_DISTORTION_RATE = 0.001   # Beck's cognitive distortion

# HAM-D items → channels
HAMD_ITEMS: Dict[str, Dict[str, Any]] = {
    "depressed_mood":    {"max": 4, "channels": ["amygdala"]},
    "guilt":             {"max": 4, "channels": ["prefrontal"]},
    "suicide":           {"max": 4, "channels": ["prefrontal", "consciousness"]},
    "insomnia_early":    {"max": 2, "channels": ["autonomic"]},
    "insomnia_middle":   {"max": 2, "channels": ["autonomic"]},
    "insomnia_late":     {"max": 2, "channels": ["autonomic"]},
    "work_interest":     {"max": 4, "channels": ["basal_ganglia", "prefrontal"]},
    "retardation":       {"max": 4, "channels": ["motor_gross", "hand"]},
    "agitation":         {"max": 4, "channels": ["amygdala", "autonomic"]},
    "anxiety_psychic":   {"max": 4, "channels": ["amygdala"]},
    "anxiety_somatic":   {"max": 4, "channels": ["autonomic"]},
    "somatic_gi":        {"max": 2, "channels": ["autonomic"]},
    "somatic_general":   {"max": 2, "channels": ["motor_gross"]},
    "genital":           {"max": 2, "channels": ["autonomic"]},
    "hypochondriasis":   {"max": 4, "channels": ["prefrontal"]},
    "weight_loss":       {"max": 2, "channels": ["autonomic"]},
    "insight":           {"max": 2, "channels": ["consciousness"]},
}
HAMD_MAX = 52

# SSRI Pharmacology
SSRI_ALPHA = -0.20                 # Impedance modification (primarily affects emotional channels)
SSRI_ONSET_DELAY = 300             # 2-4 weeks (~300 ticks = 5 days at 1tick/min × scaling)
SSRI_TARGET_CHANNELS = ["amygdala", "prefrontal", "hippocampus",
                        "basal_ganglia", "consciousness"]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DrugProfile:
    """Drug impedance modification profile"""
    name: str
    alpha: float                       # Impedance modification factor
    target_channels: List[str]         # Affected channels
    onset_delay: int = 0               # Onset delay (ticks)
    half_life: int = 500               # Half-life (ticks)
    side_effect_channels: List[str] = field(default_factory=list)
    side_effect_alpha: float = 0.0     # Side effect impedance modification
    administered_tick: int = 0         # Administration tick


@dataclass
class MSLesion:
    """MS lesion"""
    tract: str
    channels: List[str]
    onset_tick: int
    severity: float            # 0-1 demyelination degree
    active: bool = True        # Active inflammation


@dataclass
class MSState:
    """Multiple sclerosis state"""
    ms_type: str               # "RRMS" | "PPMS" | "SPMS"
    onset_tick: int = 0
    lesions: List[MSLesion] = field(default_factory=list)
    in_relapse: bool = False
    relapse_count: int = 0


@dataclass
class PDState:
    """Parkinson's state"""
    onset_tick: int = 0
    dopamine_level: float = 1.0       # 0-1, substantia nigra dopamine
    on_ldopa: bool = False
    ldopa_start_tick: int = 0
    total_ldopa_ticks: int = 0


@dataclass
class EpilepsyState:
    """Epilepsy state"""
    focus: str = "temporal"            # Seizure focus
    seizure_threshold: float = SEIZURE_THRESHOLD
    onset_tick: int = 0
    total_seizures: int = 0
    in_seizure: bool = False
    seizure_remaining: int = 0
    postictal_remaining: int = 0


@dataclass
class DepressionState:
    """Major depressive state"""
    onset_tick: int = 0
    serotonin: float = SEROTONIN_BASELINE
    norepinephrine: float = NOREPINEPHRINE_BASELINE
    cognitive_distortion: float = 0.0   # 0-1 Beck's cognitive distortion
    on_ssri: bool = False
    ssri_start_tick: int = 0


# ============================================================================
# Unified Pharmacology Engine — PharmacologyEngine
# ============================================================================

class PharmacologyEngine:
    """
    Unified Computational Pharmacology Engine

    Core Formulas:
        Z_eff(ch, t) = Z_normal × (1 + Σ α_drug,i(t))
        Γ_drug(ch) = (Z_eff - Z_normal) / (Z_eff + Z_normal)

    Pharmacokinetics:
        α_effective(t) = α₀ × onset_factor(t) × decay_factor(t)
        onset_factor = min(1, (t - t_admin) / onset_delay)
        decay_factor = 0.5^((t - t_admin - onset_delay) / half_life)

    Drug Interactions:
        Multi-drug α nonlinear superposition: α_total = Σα_i + Σ_{i<j} α_i × α_j × interaction_coeff
    """

    def __init__(self) -> None:
        self.active_drugs: List[DrugProfile] = []
        self.channel_alpha: Dict[str, float] = {ch: 0.0 for ch in ALL_CHANNELS}
        self._tick = 0
        self._drug_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def administer(self, drug: DrugProfile) -> None:
        """Administer drug"""
        drug.administered_tick = self._tick
        self.active_drugs.append(drug)
        self._drug_history.append({
            "name": drug.name,
            "alpha": drug.alpha,
            "tick": self._tick,
            "targets": drug.target_channels,
        })

    # ------------------------------------------------------------------
    def _compute_effective_alpha(self, drug: DrugProfile) -> float:
        """Compute drug's effective α at current tick"""
        elapsed = self._tick - drug.administered_tick
        if elapsed < 0:
            return 0.0

        # Onset curve (sigmoid-like ramp)
        if drug.onset_delay > 0 and elapsed < drug.onset_delay:
            onset_factor = elapsed / drug.onset_delay
        else:
            onset_factor = 1.0

        # Decay curve (exponential decay after onset)
        post_onset = max(0, elapsed - drug.onset_delay)
        if drug.half_life > 0:
            decay_factor = 0.5 ** (post_onset / drug.half_life)
        else:
            decay_factor = 1.0

        return drug.alpha * onset_factor * decay_factor

    # ------------------------------------------------------------------
    def tick(self, brain_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Advance one timestep, update drug impedance modifications for all channels"""
        self._tick += 1

        # Reset channel α
        self.channel_alpha = {ch: 0.0 for ch in ALL_CHANNELS}

        # Accumulate all active drugs
        active_drugs = []
        for drug in self.active_drugs:
            eff_alpha = self._compute_effective_alpha(drug)
            if abs(eff_alpha) < 1e-6:
                continue  # Drug effect has vanished
            active_drugs.append(drug)

            # Therapeutic channels
            for ch in drug.target_channels:
                if ch in self.channel_alpha:
                    self.channel_alpha[ch] += eff_alpha

            # Side effect channels
            side_alpha = drug.side_effect_alpha * (
                self._compute_effective_alpha(drug) / drug.alpha
                if drug.alpha != 0 else 0
            )
            for ch in drug.side_effect_channels:
                if ch in self.channel_alpha:
                    self.channel_alpha[ch] += side_alpha

        self.active_drugs = active_drugs

        # Compute Γ_drug for each channel
        channel_gamma = {}
        for ch, alpha in self.channel_alpha.items():
            z_eff = Z_NORMAL * (1 + alpha)
            z_eff = max(0.01, z_eff)  # Avoid division by zero
            gamma = (z_eff - Z_NORMAL) / (z_eff + Z_NORMAL)
            channel_gamma[ch] = max(-1.0, min(1.0, gamma))

        return {
            "active_drugs": len(self.active_drugs),
            "channel_alpha": dict(self.channel_alpha),
            "channel_gamma": channel_gamma,
        }

    # ------------------------------------------------------------------
    def get_channel_gamma(self) -> Dict[str, float]:
        """Get drug-induced channel Γ modifications"""
        result = {}
        for ch, alpha in self.channel_alpha.items():
            z_eff = Z_NORMAL * (1 + alpha)
            z_eff = max(0.01, z_eff)
            gamma = (z_eff - Z_NORMAL) / (z_eff + Z_NORMAL)
            result[ch] = max(-1.0, min(1.0, gamma))
        return result

    # ------------------------------------------------------------------
    def introspect(self) -> Dict[str, Any]:
        return {
            "engine": "pharmacology",
            "active_drugs": [
                {"name": d.name, "alpha": d.alpha,
                 "effective_alpha": self._compute_effective_alpha(d)}
                for d in self.active_drugs
            ],
            "channel_alpha": dict(self.channel_alpha),
            "drug_history_count": len(self._drug_history),
        }


# ============================================================================
# Multiple Sclerosis Model
# ============================================================================

class MSModel:
    """
    Multiple Sclerosis — demyelination = coaxial cable insulation stripping

    Physics:
        Z_coax = (1/2π) √(μ/ε) ln(r_outer/r_inner)
        Demyelination → ε changes + effective r_outer shrinks
        → Z₀ offset → each damaged segment (lesion) produces reflection
        → Signal "leaks along the way" rather than "terminal open circuit"

    Distinction from stroke/ALS:
        Stroke = regional sudden occlusion
        ALS  = one-by-one disconnection
        MS   = distributed insulation degradation

    Clinical scale: EDSS 0-10 (Kurtzke 1983)
    Subtypes:
        RRMS (Relapsing-Remitting): relapsing-remitting
        PPMS (Primary Progressive): primary progressive
        SPMS (Secondary Progressive): RRMS → secondary progressive
    """

    def __init__(self) -> None:
        self.state: Optional[MSState] = None
        self.channel_gamma: Dict[str, float] = {}
        self._tick = 0

    # ------------------------------------------------------------------
    def onset(self, ms_type: str = "RRMS") -> MSState:
        """Disease onset"""
        self.state = MSState(ms_type=ms_type, onset_tick=self._tick)
        self.channel_gamma = {ch: 0.0 for ch in ALL_CHANNELS}
        # Initial lesion
        self._create_lesion()
        return self.state

    # ------------------------------------------------------------------
    def _create_lesion(self) -> None:
        """Generate new demyelination lesion"""
        if self.state is None:
            return
        tract = random.choice(list(MS_TRACT_CHANNELS.keys()))
        channels = MS_TRACT_CHANNELS[tract]
        lo, hi = MS_LESION_GAMMA_RANGE
        severity = lo + (hi - lo) * random.random()
        lesion = MSLesion(
            tract=tract,
            channels=list(channels),
            onset_tick=self._tick,
            severity=severity,
            active=True,
        )
        self.state.lesions.append(lesion)

    # ------------------------------------------------------------------
    def tick(self, brain_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Advance one timestep"""
        self._tick += 1
        if self.state is None:
            return {"active": False}

        ms_type = self.state.ms_type

        # RRMS: random relapse
        if ms_type == "RRMS" and not self.state.in_relapse:
            if random.random() < MS_RELAPSE_RATE:
                self.state.in_relapse = True
                self.state.relapse_count += 1
                self._create_lesion()

        # Relapse ends (random 50-150 ticks)
        if self.state.in_relapse:
            if random.random() < 0.01:
                self.state.in_relapse = False

        # Progressive: continuously generate lesions
        if ms_type in ("PPMS", "SPMS"):
            if random.random() < MS_DEMYELINATION_RATE:
                self._create_lesion()

        # SPMS conversion: RRMS converts to SPMS after 3+ relapses
        if ms_type == "RRMS" and self.state.relapse_count >= 3:
            if random.random() < 0.001:
                self.state.ms_type = "SPMS"

        # Update channel Γ — take max of all lesion effects
        self.channel_gamma = {ch: 0.0 for ch in ALL_CHANNELS}
        for lesion in self.state.lesions:
            # Active lesions worsen, inactive lesions may partially remyelinate
            if lesion.active:
                lesion.severity = min(1.0, lesion.severity + MS_DEMYELINATION_RATE)
            else:
                lesion.severity = max(0.0, lesion.severity - MS_REMYELINATION_RATE)

            # Active lesions convert to remission during non-relapse period
            if lesion.active and not self.state.in_relapse:
                age = self._tick - lesion.onset_tick
                if age > 50:
                    lesion.active = False

            # Map to channels
            for ch in lesion.channels:
                self.channel_gamma[ch] = max(
                    self.channel_gamma.get(ch, 0),
                    lesion.severity,
                )

        return {
            "active": True,
            "ms_type": self.state.ms_type,
            "lesion_count": len(self.state.lesions),
            "relapse_count": self.state.relapse_count,
            "in_relapse": self.state.in_relapse,
            "edss": self.get_edss(),
            "channel_gamma": dict(self.channel_gamma),
        }

    # ------------------------------------------------------------------
    def get_edss(self) -> float:
        """Expanded Disability Status Scale (0-10)"""
        if self.state is None:
            return 0.0

        # Compute functional system scores
        fs_scores = []
        for _sys, info in EDSS_FUNCTIONAL_SYSTEMS.items():
            channels = info["channels"]
            mean_gamma = sum(
                self.channel_gamma.get(ch, 0) for ch in channels
            ) / max(1, len(channels))
            fs_score = info["max"] * mean_gamma
            fs_scores.append(fs_score)

        # Simplified EDSS calculation: weighted average based on functional system scores
        if not fs_scores:
            return 0.0
        max_fs = max(fs_scores)
        n_affected = sum(1 for s in fs_scores if s > 1.0)

        # EDSS grading logic
        if max_fs < 1.0:
            return 0.0
        elif max_fs < 2.0:
            return min(3.5, 1.0 + n_affected * 0.5)
        elif max_fs < 3.5:
            return min(5.5, 3.0 + n_affected * 0.5)
        else:
            return min(EDSS_MAX, 4.0 + max_fs * 0.8)

    # ------------------------------------------------------------------
    def introspect(self) -> Dict[str, Any]:
        return {
            "condition": "multiple_sclerosis",
            "active": self.state is not None,
            "edss": self.get_edss(),
            "lesion_count": len(self.state.lesions) if self.state else 0,
            "channel_gamma": dict(self.channel_gamma),
        }


# ============================================================================
# Parkinson's Disease Model
# ============================================================================

class ParkinsonModel:
    """
    Parkinson's Disease — substantia nigra dopamine depletion → basal ganglia circuit failure

    Physics:
        Dopamine = basal ganglia channel's "impedance matching lubricant"
        DA↓ → Z_basal_ganglia ↑ → Γ ↑ → motor output reflected back
        → Tremor (residual signals oscillate in circuit at 4-6 Hz)
        → Rigidity (Γ persistently high → resistance doesn't vary with speed, distinct from spasticity)
        → Bradykinesia (transmission efficiency T = 1-Γ² → delay multiplied)

    Treatment:
        L-DOPA → exogenous dopamine → Z_bg ↓ → Γ ↓
        But long-term use → receptor downregulation → wearing-off + dyskinesia

    Clinical scale: UPDRS 0-199 (Fahn & Elton 1987)
    """

    def __init__(self) -> None:
        self.state: Optional[PDState] = None
        self.channel_gamma: Dict[str, float] = {}
        self._tick = 0
        self._tremor_phase: float = 0.0

    # ------------------------------------------------------------------
    def onset(self) -> PDState:
        """Parkinson's disease onset"""
        self.state = PDState(onset_tick=self._tick, dopamine_level=0.8)
        self.channel_gamma = {ch: 0.0 for ch in ALL_CHANNELS}
        return self.state

    # ------------------------------------------------------------------
    def start_ldopa(self) -> None:
        """Start L-DOPA treatment"""
        if self.state:
            self.state.on_ldopa = True
            self.state.ldopa_start_tick = self._tick

    # ------------------------------------------------------------------
    def tick(self, brain_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Advance one timestep"""
        self._tick += 1
        if self.state is None:
            return {"active": False}

        # Dopamine continuous depletion
        self.state.dopamine_level = max(
            0.0,
            self.state.dopamine_level - PD_DOPAMINE_DEPLETION_RATE,
        )

        # L-DOPA effect
        effective_da = self.state.dopamine_level
        if self.state.on_ldopa:
            self.state.total_ldopa_ticks += 1
            # L-DOPA supplementation effect fades over time (wearing off)
            ldopa_elapsed = self._tick - self.state.ldopa_start_tick
            ldopa_boost = abs(LDOPA_ALPHA) * (
                0.5 ** (ldopa_elapsed * LDOPA_WEARING_OFF_RATE)
            )
            effective_da = min(1.0, self.state.dopamine_level + ldopa_boost)

        # DA deficit → Γ rises
        da_deficit = 1.0 - effective_da

        # Motor channel Γ
        for ch in PD_MOTOR_CHANNELS:
            base_gamma = da_deficit * PD_RIGIDITY_GAIN
            # Tremor: 4-6 Hz oscillation superposition
            self._tremor_phase += 2 * math.pi * PD_TREMOR_FREQUENCY / 60.0
            tremor = 0.1 * da_deficit * abs(math.sin(self._tremor_phase))
            self.channel_gamma[ch] = min(1.0, base_gamma + tremor)

        # Cognitive channels (milder impact)
        for ch in PD_COGNITIVE_CHANNELS:
            self.channel_gamma[ch] = min(1.0, da_deficit * 0.3)

        # Dyskinesia (long-term L-DOPA side effect)
        dyskinesia = 0.0
        if (self.state.on_ldopa and
                self.state.total_ldopa_ticks > LDOPA_DYSKINESIA_THRESHOLD):
            excess = self.state.total_ldopa_ticks - LDOPA_DYSKINESIA_THRESHOLD
            dyskinesia = min(0.3, excess * 0.0001)
            for ch in PD_MOTOR_CHANNELS:
                self.channel_gamma[ch] = min(
                    1.0, self.channel_gamma[ch] + dyskinesia
                )

        return {
            "active": True,
            "dopamine_level": round(self.state.dopamine_level, 4),
            "effective_dopamine": round(effective_da, 4),
            "updrs": self.get_updrs(),
            "tremor_amplitude": round(
                0.1 * da_deficit * abs(math.sin(self._tremor_phase)), 4),
            "dyskinesia": round(dyskinesia, 4),
            "channel_gamma": dict(self.channel_gamma),
        }

    # ------------------------------------------------------------------
    def get_updrs(self) -> int:
        """Unified Parkinson's Disease Rating Scale (0-199)"""
        if self.state is None:
            return 0
        score = 0.0
        for _section, info in UPDRS_SECTIONS.items():
            channels = info["channels"]
            mean_gamma = sum(
                self.channel_gamma.get(ch, 0) for ch in channels
            ) / max(1, len(channels))
            score += info["max"] * mean_gamma
        return min(UPDRS_MAX, round(score))

    # ------------------------------------------------------------------
    def introspect(self) -> Dict[str, Any]:
        return {
            "condition": "parkinson",
            "active": self.state is not None,
            "dopamine_level": self.state.dopamine_level if self.state else 1.0,
            "updrs": self.get_updrs(),
            "channel_gamma": dict(self.channel_gamma),
        }


# ============================================================================
# Epilepsy Model
# ============================================================================

class EpilepsyModel:
    """
    Epilepsy — excitation/inhibition imbalance → positive feedback runaway

    Physics:
        Normal brain = damped oscillation (amplitude gradually decays)
        Epileptic brain = positive feedback (amplitude gradually amplifies → Γ exceeds threshold → synchronous discharge)

        Mathematics:
            excitation(t+1) = excitation(t) + noise - inhibition(t)
            if excitation > threshold: SEIZURE
            inhibition(t+1) = inhibition(t) - decay + excitation(t) × coupling

        Kindling effect: each seizure lowers threshold → increasingly easier to seize

    Clinical scale: seizure frequency + severity
    Subtypes:
        Focal: starts from a specific seizure focus
        Generalized: bilateral synchronous
    """

    def __init__(self) -> None:
        self.state: Optional[EpilepsyState] = None
        self.channel_gamma: Dict[str, float] = {}
        self.excitation: float = 0.0
        self.inhibition: float = 0.5     # GABAergic inhibition
        self._tick = 0
        self._seizure_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def onset(self, focus: str = "temporal") -> EpilepsyState:
        """Seizure focus setting"""
        self.state = EpilepsyState(
            focus=focus,
            onset_tick=self._tick,
        )
        self.channel_gamma = {ch: 0.0 for ch in ALL_CHANNELS}
        return self.state

    # ------------------------------------------------------------------
    def tick(self, brain_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Advance one timestep"""
        self._tick += 1
        if self.state is None:
            return {"active": False}

        in_seizure = self.state.in_seizure
        seizure_event = False

        # Post-ictal suppression period
        if self.state.postictal_remaining > 0:
            self.state.postictal_remaining -= 1
            # Post-ictal all channel Γ decrease (exhaustive suppression)
            for ch in ALL_CHANNELS:
                self.channel_gamma[ch] = max(
                    0.0, self.channel_gamma.get(ch, 0) - 0.02
                )
            return {
                "active": True,
                "phase": "postictal",
                "excitation": round(self.excitation, 4),
                "inhibition": round(self.inhibition, 4),
                "seizure_count": self.state.total_seizures,
                "channel_gamma": dict(self.channel_gamma),
            }

        if in_seizure:
            # During seizure: all related channel Γ → 1.0 (synchronous discharge)
            focus_channels = EPILEPSY_FOCAL_CHANNELS.get(
                self.state.focus, ["consciousness"])
            for ch in focus_channels:
                self.channel_gamma[ch] = 1.0
            # Generalization to other channels
            for ch in ALL_CHANNELS:
                if ch not in focus_channels:
                    self.channel_gamma[ch] = min(
                        1.0, self.channel_gamma.get(ch, 0) + 0.05
                    )

            self.state.seizure_remaining -= 1
            if self.state.seizure_remaining <= 0:
                self.state.in_seizure = False
                self.state.postictal_remaining = POSTICTAL_DURATION
                # Post-ictal Γ plummets (exhaustion)
                self.excitation = 0.0
                self.inhibition = 0.5 + POSTICTAL_DEPRESSION
        else:
            # Interictal: excitation/inhibition dynamics
            noise = random.gauss(0, EXCITATION_NOISE)
            self.excitation += noise - (self.inhibition - 0.5) * 0.1
            self.excitation = max(0.0, min(1.0, self.excitation))

            self.inhibition -= INHIBITION_DECAY
            self.inhibition = max(0.0, min(1.0, self.inhibition))

            # Seizure determination
            if self.excitation > self.state.seizure_threshold:
                self.state.in_seizure = True
                self.state.total_seizures += 1
                duration = random.randint(*SEIZURE_DURATION_RANGE)
                self.state.seizure_remaining = duration
                seizure_event = True

                # Kindling effect: lower threshold
                self.state.seizure_threshold = max(
                    0.3, self.state.seizure_threshold - KINDLING_INCREMENT
                )

                self._seizure_log.append({
                    "tick": self._tick,
                    "duration": duration,
                    "threshold": self.state.seizure_threshold,
                })

            # Interictal channel Γ — low-level abnormality
            focus_channels = EPILEPSY_FOCAL_CHANNELS.get(
                self.state.focus, ["consciousness"])
            for ch in ALL_CHANNELS:
                if ch in focus_channels:
                    self.channel_gamma[ch] = min(
                        1.0, self.excitation * 0.3
                    )
                else:
                    self.channel_gamma[ch] = max(
                        0.0, self.channel_gamma.get(ch, 0) - 0.01
                    )

        return {
            "active": True,
            "phase": "seizure" if in_seizure else "interictal",
            "seizure_event": seizure_event,
            "excitation": round(self.excitation, 4),
            "inhibition": round(self.inhibition, 4),
            "threshold": round(self.state.seizure_threshold, 4),
            "seizure_count": self.state.total_seizures,
            "in_seizure": self.state.in_seizure,
            "channel_gamma": dict(self.channel_gamma),
        }

    # ------------------------------------------------------------------
    def force_seizure(self) -> None:
        """Force trigger a seizure (experimental)"""
        if self.state:
            self.excitation = self.state.seizure_threshold + 0.1

    # ------------------------------------------------------------------
    def introspect(self) -> Dict[str, Any]:
        return {
            "condition": "epilepsy",
            "active": self.state is not None,
            "seizure_count": self.state.total_seizures if self.state else 0,
            "threshold": self.state.seizure_threshold if self.state else SEIZURE_THRESHOLD,
            "excitation": round(self.excitation, 4),
            "channel_gamma": dict(self.channel_gamma),
        }


# ============================================================================
# Major Depressive Disorder Model
# ============================================================================

class DepressionModel:
    """
    Major Depressive Disorder — chronic low-level Γ elevation

    Physics:
        Serotonin (5-HT) + norepinephrine (NE) = "impedance matching lubricant" for emotional channels
        5-HT/NE ↓ → Z_emotional ↑ → Γ ↑ → sustained low emotional signal transmission efficiency

        The brain's "leaky pipe": not a sudden burst (stroke), nor a disconnection (ALS),
        but pipe walls slowly corroding → systemic low-efficiency transmission → no motivation for anything

    Clinical features → physical mapping:
        - Depressed mood → amygdala Γ ↑ (emotional processing efficiency ↓)
        - Anhedonia → basal_ganglia reward pathway T ↓
        - Cognitive distortion → prefrontal Γ ↑ (executive control efficiency ↓)
        - Insomnia → autonomic Γ ↑ (autonomic dysregulation)
        - Psychomotor retardation → motor Γ ↑

    Treatment:
        SSRI → blocks 5-HT reuptake → synaptic cleft 5-HT ↑ → Z_emotional ↓ → Γ ↓
        But requires 2-4 weeks to take effect (receptor down/up-regulation takes time)

    Clinical scale: HAM-D 0-52 (Hamilton 1960)
    """

    def __init__(self) -> None:
        self.state: Optional[DepressionState] = None
        self.channel_gamma: Dict[str, float] = {}
        self._tick = 0

    # ------------------------------------------------------------------
    def onset(self, severity: str = "moderate") -> DepressionState:
        """Disease onset"""
        self.state = DepressionState(onset_tick=self._tick)

        # Initial monoamine levels depend on severity
        if severity == "mild":
            self.state.serotonin = 0.75
            self.state.norepinephrine = 0.80
        elif severity == "moderate":
            self.state.serotonin = 0.55
            self.state.norepinephrine = 0.60
        elif severity == "severe":
            self.state.serotonin = 0.35
            self.state.norepinephrine = 0.40
        else:
            self.state.serotonin = 0.55
            self.state.norepinephrine = 0.60

        self.channel_gamma = {ch: 0.0 for ch in ALL_CHANNELS}
        return self.state

    # ------------------------------------------------------------------
    def start_ssri(self) -> None:
        """Start SSRI treatment"""
        if self.state:
            self.state.on_ssri = True
            self.state.ssri_start_tick = self._tick

    # ------------------------------------------------------------------
    def tick(self, brain_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Advance one timestep"""
        self._tick += 1
        if self.state is None:
            return {"active": False}

        # Natural depletion (SSRI slows reuptake rate)
        depletion_factor = 0.3 if self.state.on_ssri else 1.0
        self.state.serotonin = max(
            0.1, self.state.serotonin - SEROTONIN_DEPLETION_RATE * depletion_factor
        )
        self.state.norepinephrine = max(
            0.1, self.state.norepinephrine - NE_DEPLETION_RATE * depletion_factor
        )

        # SSRI effect (delayed onset) — blocks reuptake → synaptic cleft concentration ↑
        effective_5ht = self.state.serotonin
        if self.state.on_ssri:
            ssri_elapsed = self._tick - self.state.ssri_start_tick
            if ssri_elapsed >= SSRI_ONSET_DELAY:
                # Full effect: multiplicative boost (SSRIs raise synaptic 5-HT 2-3x)
                ssri_boost = abs(SSRI_ALPHA) * 1.5
            elif ssri_elapsed > 0:
                # Gradual onset (sigmoid ramp)
                ssri_boost = abs(SSRI_ALPHA) * 1.5 * (
                    ssri_elapsed / SSRI_ONSET_DELAY
                )
            else:
                ssri_boost = 0.0
            effective_5ht = min(1.0, self.state.serotonin + ssri_boost)

        # Cognitive distortion (Beck's model)
        self.state.cognitive_distortion = min(
            1.0,
            self.state.cognitive_distortion + COGNITIVE_DISTORTION_RATE * (
                1.0 - effective_5ht
            ),
        )
        # SSRI can reverse cognitive distortion (effective 5-HT above 0.45 begins to alleviate)
        if self.state.on_ssri and effective_5ht > 0.45:
            cd_reversal = 0.002 * (effective_5ht - 0.3)
            self.state.cognitive_distortion = max(
                0.0, self.state.cognitive_distortion - cd_reversal
            )

        # Monoamine deficit → channel Γ
        monoamine_deficit = 1.0 - (effective_5ht * 0.6 +
                                    self.state.norepinephrine * 0.4)

        # Emotional channels (primary impact)
        self.channel_gamma["amygdala"] = min(1.0, monoamine_deficit * 0.8)
        self.channel_gamma["prefrontal"] = min(
            1.0, monoamine_deficit * 0.5 +
            self.state.cognitive_distortion * 0.3
        )

        # Reward pathway (anhedonia)
        self.channel_gamma["basal_ganglia"] = min(
            1.0, monoamine_deficit * ANHEDONIA_REWARD_PENALTY
        )

        # Autonomic (insomnia, orthostatic symptoms)
        self.channel_gamma["autonomic"] = min(
            1.0, monoamine_deficit * 0.4
        )

        # Motor channels (psychomotor retardation)
        for ch in ["hand", "motor_gross"]:
            self.channel_gamma[ch] = min(1.0, monoamine_deficit * 0.3)

        # Cognitive channels
        self.channel_gamma["attention"] = min(1.0, monoamine_deficit * 0.4)
        self.channel_gamma["hippocampus"] = min(1.0, monoamine_deficit * 0.35)
        self.channel_gamma["consciousness"] = min(
            1.0, monoamine_deficit * 0.2
        )

        return {
            "active": True,
            "serotonin": round(self.state.serotonin, 4),
            "effective_serotonin": round(effective_5ht, 4),
            "norepinephrine": round(self.state.norepinephrine, 4),
            "cognitive_distortion": round(self.state.cognitive_distortion, 4),
            "hamd": self.get_hamd(),
            "channel_gamma": dict(self.channel_gamma),
        }

    # ------------------------------------------------------------------
    def get_hamd(self) -> int:
        """Hamilton Depression Rating Scale (0-52)"""
        if self.state is None:
            return 0
        score = 0.0
        for _item, info in HAMD_ITEMS.items():
            channels = info["channels"]
            mean_gamma = sum(
                self.channel_gamma.get(ch, 0) for ch in channels
            ) / max(1, len(channels))
            score += info["max"] * mean_gamma
        return min(HAMD_MAX, round(score))

    # ------------------------------------------------------------------
    def introspect(self) -> Dict[str, Any]:
        return {
            "condition": "depression",
            "active": self.state is not None,
            "serotonin": self.state.serotonin if self.state else 1.0,
            "hamd": self.get_hamd(),
            "channel_gamma": dict(self.channel_gamma),
        }


# ============================================================================
# ClinicalPharmacologyEngine — Unified Clinical Pharmacology Engine
# ============================================================================

class ClinicalPharmacologyEngine:
    """
    Unified integration: pharmacology engine + four neurological diseases

    Architecture:
        PharmacologyEngine (unified drug system)
         ├── MSModel (multiple sclerosis)
         ├── ParkinsonModel (Parkinson's disease)
         ├── EpilepsyModel (epilepsy)
         └── DepressionModel (depression)

    tick() → advance all active models + pharmacokinetics
    get_merged_channel_gamma() → take max Γ across all conditions
    """

    def __init__(self) -> None:
        self.pharmacology = PharmacologyEngine()
        self.ms = MSModel()
        self.parkinson = ParkinsonModel()
        self.epilepsy = EpilepsyModel()
        self.depression = DepressionModel()
        self._tick = 0

    # ------------------------------------------------------------------
    def tick(self, brain_state: Optional[Dict] = None) -> Dict[str, Any]:
        self._tick += 1
        results: Dict[str, Any] = {}

        # Advance pharmacokinetics
        pharma_result = self.pharmacology.tick(brain_state)
        results["pharmacology"] = pharma_result

        # Advance each disease model
        if self.ms.state is not None:
            results["ms"] = self.ms.tick(brain_state)
        if self.parkinson.state is not None:
            results["parkinson"] = self.parkinson.tick(brain_state)
        if self.epilepsy.state is not None:
            results["epilepsy"] = self.epilepsy.tick(brain_state)
        if self.depression.state is not None:
            results["depression"] = self.depression.tick(brain_state)

        # Merge channel Γ
        results["merged_gamma"] = self.get_merged_channel_gamma()
        results["clinical_summary"] = self.get_clinical_summary()

        return results

    # ------------------------------------------------------------------
    def get_merged_channel_gamma(self) -> Dict[str, float]:
        """Merge all disease + drug channel Γ (take max + drug modification)"""
        merged = {ch: 0.0 for ch in ALL_CHANNELS}

        # Disease Γ — take max
        for model in [self.ms, self.parkinson, self.epilepsy, self.depression]:
            for ch, gamma in model.channel_gamma.items():
                merged[ch] = max(merged.get(ch, 0), gamma)

        # Drug modification — additive (can offset disease Γ)
        pharma_gamma = self.pharmacology.get_channel_gamma()
        for ch in ALL_CHANNELS:
            merged[ch] = max(0.0, min(1.0,
                merged[ch] + pharma_gamma.get(ch, 0)))

        return merged

    # ------------------------------------------------------------------
    def get_clinical_summary(self) -> Dict[str, Any]:
        """Get clinical summary of all active conditions"""
        summary: Dict[str, Any] = {
            "active_conditions": [],
            "active_drugs": len(self.pharmacology.active_drugs),
        }
        if self.ms.state is not None:
            summary["active_conditions"].append("MS")
            summary["edss"] = self.ms.get_edss()
        if self.parkinson.state is not None:
            summary["active_conditions"].append("PD")
            summary["updrs"] = self.parkinson.get_updrs()
        if self.epilepsy.state is not None:
            summary["active_conditions"].append("Epilepsy")
            summary["seizure_count"] = (
                self.epilepsy.state.total_seizures)
        if self.depression.state is not None:
            summary["active_conditions"].append("MDD")
            summary["hamd"] = self.depression.get_hamd()
        return summary

    # ------------------------------------------------------------------
    def introspect(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "engine": "clinical_pharmacology",
            "pharmacology": self.pharmacology.introspect(),
        }
        if self.ms.state is not None:
            result["ms"] = self.ms.introspect()
        if self.parkinson.state is not None:
            result["parkinson"] = self.parkinson.introspect()
        if self.epilepsy.state is not None:
            result["epilepsy"] = self.epilepsy.introspect()
        if self.depression.state is not None:
            result["depression"] = self.depression.introspect()
        return result
