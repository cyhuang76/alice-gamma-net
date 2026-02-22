#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 29 — The Dream Bridge (託夢橋)

exp_dream_bridge.py — Inter-Individual Dream Delivery via Mirror Neuron Matching

Core Hypothesis:
  "託夢" (dream delivery) = Face 3 (dream incubation) + Face 4 (matching network).

  Phase 27 demonstrated non-invasive impedance modulation (Face 3 alone),
  but SNR was only ~0.05 because the two Alices were STRANGERS — their
  channels had random impedance with σ_Z ≈ 20.

  Phase 28 showed that:
    - N3 guardian repairs ~96% of impedance debt (multiplicative)
    - SOREMP bypasses N3, preserving fatigue for dream amplification
    - Amplitude ≠ structure (vivid ≠ meaningful)

  Paper VI §4.5 proved that mirror neurons are the EVOLVED matching network:
    Z_self^{t+1} = Z_self^t + η·(Z_observed - Z_self^t)
    This is gradient descent on Γ²_mirror.

  The missing link: mirror neuron pre-matching NARROWS σ_Z on matched channels,
  which raises SNR from 0.05 to ~1.0.  Without pre-matching, dream delivery
  is broadcasting to random antennas.  With pre-matching, it is a directed
  impedance bridge.

  Protocol:
    EXP 0: Mirror Pre-Training  — build Γ_mirror convergence between two Alices
    EXP 1: Sender Dream Capture — sender dreams with Video A → extract Z-signature
    EXP 2: Dream Bridge         — use sender's Z-signature as receiver's Z_terminus
    EXP 3: Verification         — Γ_mirror after bridge < before bridge
    EXP 4: Control              — repeat without pre-training → expect lower SNR

  "You must know someone before you can reach them in a dream."

Experimental Design:
  Condition M (Matched):    Mirror pre-training → Sender dream → Bridge delivery
  Condition S (Stranger):   No pre-training     → Sender dream → Bridge delivery
  Condition V (Video-only): Mirror pre-training → Generic video (Phase 27 style)
  Condition N (Null):       No pre-training     → No modulation (control)

  This 2×2 design separates the effects of:
    - Mirror matching (M vs S, V vs N)
    - Sender-specific Z-signature vs generic video (M vs V, S vs N)

Run: python -m experiments.exp_dream_bridge
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alice.alice_brain import AliceBrain
from alice.brain.mirror_neurons import (
    MirrorNeuronEngine,
    MOTOR_MIRROR_IMPEDANCE,
    MIRROR_RESONANCE_THRESHOLD,
    SOCIAL_BOND_BASE,
)
from alice.core.protocol import Priority

# Re-use Phase 27's Z_terminus modulation and schedule helpers
from experiments.exp_dream_language import (
    video_to_impedance_modulation,
    make_sleep_schedule,
    snapshot_semantic_field,
    DREAM_CHANNEL_NAMES,
    Z_BASE,
    IMPEDANCE_MOD_DEPTH,
    VIDEO_A_SPATIAL_FREQ,
    VIDEO_A_VOWEL,
    VIDEO_A_RHYTHM_HZ,
)


# ============================================================
# Constants
# ============================================================

NEURON_COUNT = 80
SEED_SENDER = 42
SEED_RECEIVER = 137

# Mirror pre-training
MIRROR_PRETRAIN_ROUNDS = 50          # Observation rounds for pre-matching
MIRROR_ACTIONS = [                    # Actions the sender performs
    ("vocal", 72.0, "speak_softly"),
    ("facial", 78.0, "smile"),
    ("manual", 68.0, "wave"),
    ("vocal", 74.0, "hum"),
    ("facial", 70.0, "nod"),
    ("manual", 80.0, "point"),
]
MIRROR_EMOTIONS = [                   # Emotions the sender expresses
    (0.6, 0.4, "facial", 73.0),      # (valence, arousal, modality, Z)
    (0.3, 0.3, "vocal", 76.0),
    (-0.2, 0.5, "facial", 71.0),
    (0.8, 0.6, "facial", 77.0),
]

# Dream parameters
AWAKE_TICKS = 80
STRESSED_AWAKE_TICKS = 300            # Enough stress for meaningful fatigue
SLEEP_CYCLE_TICKS = 110
N_DREAM_CYCLES = 4
REM_TICKS_PER_CYCLE = 22
IMPEDANCE_MOD_INTERVAL = 2
STRESS_BASE = 0.08
STRESS_RANGE = 0.04

# Bridge parameters
BRIDGE_TRANSFER_GAIN = 0.6           # How much of sender's Z-pattern to apply
BRIDGE_NOISE_FLOOR = 0.1             # Natural noise in Z_terminus

# Communication verification
COMM_ROUNDS = 20

# Print control
PRINT_INTERVAL = 10


# ============================================================
# Data structures
# ============================================================

@dataclass
class MirrorTrainingResult:
    """Result of mirror neuron pre-training between sender and receiver."""
    rounds: int = 0
    gamma_mirror_initial: float = 1.0
    gamma_mirror_final: float = 1.0
    bond_impedance_initial: float = SOCIAL_BOND_BASE
    bond_impedance_final: float = SOCIAL_BOND_BASE
    empathy_capacity: float = 0.0
    tom_capacity: float = 0.0
    familiarity: float = 0.0
    gamma_history: List[float] = field(default_factory=list)


@dataclass
class DreamSignature:
    """Z-pattern fingerprint extracted from sender's dream.

    This is the impedance landscape that the sender's PGO waves
    reflected off during REM.  It encodes the sender's dream
    content as a set of per-channel (Z_src, Z_load) pairs —
    the structural fingerprint, not the content.
    """
    channel_z_map: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    mean_gamma: float = 0.0
    modulated_gammas: List[float] = field(default_factory=list)
    n_samples: int = 0


@dataclass
class BridgeRecord:
    """Records what happened during one bridge dream delivery."""
    condition: str                               # M / S / V / N
    night: int = 0
    cycle: int = 0
    ticks_in_rem: int = 0
    modulations_applied: int = 0
    modulated_gammas: List[float] = field(default_factory=list)
    reference_gammas: List[float] = field(default_factory=list)
    peak_amp_multiplier: float = 1.0
    # Bridge-specific
    sender_gammas: List[float] = field(default_factory=list)     # Γ at sender-matched channels
    receiver_gammas: List[float] = field(default_factory=list)   # Γ at unmatched channels


@dataclass
class ConditionResult:
    """Aggregate result for one experimental condition."""
    condition: str
    mirror_training: Optional[MirrorTrainingResult] = None
    dream_records: List[BridgeRecord] = field(default_factory=list)
    mean_modulated_gamma: float = 0.0
    mean_reference_gamma: float = 0.0
    gamma_contrast: float = 0.0           # |mod - ref|
    snr: float = 0.0
    n_modulated_samples: int = 0
    n_reference_samples: int = 0
    # Post-bridge mirror check
    gamma_mirror_post: float = 1.0
    delta_gamma_mirror: float = 0.0       # post - pre (negative = improvement)
    # Communication check
    gamma_social_first: float = 0.0
    gamma_social_last: float = 0.0
    gamma_social_trend: float = 0.0


# ============================================================
# Phase 0: Mirror neuron pre-training
# ============================================================

def mirror_pretraining(
    sender_mirror: MirrorNeuronEngine,
    receiver_mirror: MirrorNeuronEngine,
    sender_id: str,
    receiver_id: str,
    rounds: int,
    rng: np.random.RandomState,
    verbose: bool = False,
) -> MirrorTrainingResult:
    """
    Mutual mirror neuron training: sender and receiver observe each other.

    Physics (Paper VI §4.5):
      Z_self^{t+1} = Z_self^t + η·(Z_observed - Z_self^t)
      This is gradient descent on Γ²_mirror.

    Each round:
      1. Sender performs an action → receiver's L1 motor mirror observes
      2. Sender expresses emotion → receiver's L2 emotional mirror resonates
      3. Receiver performs action → sender's L1 observes (bidirectional)
      4. Both engines mature (empathy + ToM capacity increase)

    After N rounds, bond_impedance should decrease and Γ_mirror → Γ_min.
    """
    result = MirrorTrainingResult(rounds=rounds)

    # Measure initial Γ_mirror
    init_event = receiver_mirror.observe_action(
        agent_id=sender_id,
        modality="vocal",
        observed_impedance=MOTOR_MIRROR_IMPEDANCE,
        action_label="baseline_probe",
    )
    result.gamma_mirror_initial = init_event.gamma_mirror
    result.bond_impedance_initial = receiver_mirror.get_social_bonds().get(
        sender_id, SOCIAL_BOND_BASE
    )

    gamma_history = [init_event.gamma_mirror]

    for r in range(rounds):
        # --- Sender → Receiver (forward direction) ---
        # Pick action and emotion from repertoire
        action = MIRROR_ACTIONS[r % len(MIRROR_ACTIONS)]
        modality, z_obs, label = action

        # Add natural variation
        z_obs_noisy = z_obs + rng.randn() * 3.0

        # Receiver observes sender's action (L1)
        event = receiver_mirror.observe_action(
            agent_id=sender_id,
            modality=modality,
            observed_impedance=z_obs_noisy,
            action_label=label,
        )

        # Receiver observes sender's emotion (L2)
        emotion = MIRROR_EMOTIONS[r % len(MIRROR_EMOTIONS)]
        val, aro, e_mod, e_z = emotion
        receiver_mirror.observe_emotion(
            agent_id=sender_id,
            observed_valence=val + rng.randn() * 0.1,
            observed_arousal=aro + rng.randn() * 0.1,
            modality=e_mod,
            signal_impedance=e_z + rng.randn() * 2.0,
        )

        # --- Receiver → Sender (reverse direction) ---
        # Sender also observes receiver (bidirectional matching)
        reverse_action = MIRROR_ACTIONS[(r + 3) % len(MIRROR_ACTIONS)]
        r_mod, r_z, r_label = reverse_action
        sender_mirror.observe_action(
            agent_id=receiver_id,
            modality=r_mod,
            observed_impedance=r_z + rng.randn() * 3.0,
            action_label=r_label,
        )

        reverse_emotion = MIRROR_EMOTIONS[(r + 2) % len(MIRROR_EMOTIONS)]
        rv, ra, rm, rz = reverse_emotion
        sender_mirror.observe_emotion(
            agent_id=receiver_id,
            observed_valence=rv + rng.randn() * 0.1,
            observed_arousal=ra + rng.randn() * 0.1,
            modality=rm,
            signal_impedance=rz + rng.randn() * 2.0,
        )

        # Both mature
        receiver_mirror.tick(has_social_input=True)
        sender_mirror.tick(has_social_input=True)

        gamma_history.append(event.gamma_mirror)

        if verbose and r % PRINT_INTERVAL == 0:
            bonds = receiver_mirror.get_social_bonds()
            z_bond = bonds.get(sender_id, SOCIAL_BOND_BASE)
            print(f"      Round {r:3d}: Γ_mirror={event.gamma_mirror:.4f}  "
                  f"Z_bond={z_bond:.1f}Ω  "
                  f"empathy={receiver_mirror.get_empathy_capacity():.3f}  "
                  f"ToM={receiver_mirror.get_tom_capacity():.3f}")

    # Final measurements
    final_event = receiver_mirror.observe_action(
        agent_id=sender_id,
        modality="vocal",
        observed_impedance=MOTOR_MIRROR_IMPEDANCE,
        action_label="final_probe",
    )
    result.gamma_mirror_final = final_event.gamma_mirror
    result.bond_impedance_final = receiver_mirror.get_social_bonds().get(
        sender_id, SOCIAL_BOND_BASE
    )
    result.empathy_capacity = receiver_mirror.get_empathy_capacity()
    result.tom_capacity = receiver_mirror.get_tom_capacity()

    agent_model = receiver_mirror.get_agent_model(sender_id)
    result.familiarity = agent_model.familiarity if agent_model else 0.0
    result.gamma_history = gamma_history

    return result


# ============================================================
# Phase 1: Sender dream capture → extract Z-signature
# ============================================================

def capture_sender_dream(
    brain: AliceBrain,
    spatial_freq: float,
    vowel: str,
    rhythm_hz: float,
    rng: np.random.RandomState,
    verbose: bool = False,
) -> DreamSignature:
    """
    Sender sleeps with a video → capture the Z-pattern fingerprint.

    This is Phase 27's dream incubation, but we record the per-channel
    impedance landscape that the sender's PGO probes reflected off.
    This landscape IS the sender's dream — not the content, but the
    structural impedance fingerprint.
    """
    signature = DreamSignature()
    z_accumulator: Dict[str, List[Tuple[float, float]]] = {
        name: [] for name in DREAM_CHANNEL_NAMES
    }

    # Wake phase: build sleep pressure
    for t in range(AWAKE_TICKS):
        pixels = 0.5 + 0.1 * rng.randn(256)
        pixels = np.clip(pixels, 0, 1)
        brain.see(pixels, priority=Priority.BACKGROUND)
        brain.sleep_physics.awake_tick(
            reflected_energy=STRESS_BASE + STRESS_RANGE * rng.rand(),
        )

    # Sleep phase: capture Z during REM
    brain.sleep_physics.begin_sleep()

    for cycle in range(N_DREAM_CYCLES):
        schedule = make_sleep_schedule(SLEEP_CYCLE_TICKS)
        frame_idx = cycle * SLEEP_CYCLE_TICKS

        for tick_in_cycle, stage in enumerate(schedule):
            if (stage == "rem"
                    and tick_in_cycle % IMPEDANCE_MOD_INTERVAL == 0):
                channels = video_to_impedance_modulation(
                    spatial_freq, vowel, rhythm_hz,
                    frame_idx + tick_in_cycle, rng,
                )
                # Record impedances for each named channel
                for ch_name, z_src, z_load in channels:
                    if ch_name in z_accumulator:
                        z_accumulator[ch_name].append((z_src, z_load))
                        g = abs(z_load - z_src) / (z_src + z_load) if (z_src + z_load) > 0 else 1.0
                        signature.modulated_gammas.append(g)
                        signature.n_samples += 1

                channel_impedances = channels
            else:
                channel_impedances = [
                    (f"ch_{i}", float(50 + 10 * rng.randn()),
                     float(50 + 10 * rng.randn()))
                    for i in range(6)
                ]

            brain.sleep_physics.sleep_tick(
                stage=stage,
                recent_memories=[f"mem_{i}" for i in range(5)],
                channel_impedances=channel_impedances,
                synaptic_strengths=list(rng.uniform(0.3, 1.5, 100)),
            )

    brain.sleep_physics.end_sleep()

    # Compute mean Z per channel → the sender's dream signature
    for ch_name, pairs in z_accumulator.items():
        if pairs:
            mean_src = float(np.mean([p[0] for p in pairs]))
            mean_load = float(np.mean([p[1] for p in pairs]))
            signature.channel_z_map[ch_name] = (mean_src, mean_load)

    if signature.modulated_gammas:
        signature.mean_gamma = float(np.mean(signature.modulated_gammas))

    if verbose:
        print(f"      Captured {signature.n_samples} Z-samples across "
              f"{len(signature.channel_z_map)} channels  "
              f"(Γ̅={signature.mean_gamma:.4f})")
        for ch, (zs, zl) in signature.channel_z_map.items():
            g = abs(zl - zs) / (zs + zl) if (zs + zl) > 0 else 1.0
            print(f"        {ch:16s}: Z_src={zs:.1f} Z_load={zl:.1f} Γ={g:.4f}")

    return signature


# ============================================================
# Phase 2: Dream bridge delivery
# ============================================================

def sender_z_to_impedance_modulation(
    signature: DreamSignature,
    frame_idx: int,
    rng: np.random.RandomState,
    transfer_gain: float = BRIDGE_TRANSFER_GAIN,
    mirror_sigma_map: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, float, float]]:
    """
    Convert sender's dream Z-signature to receiver's Z_terminus modulation.

    This is the BRIDGE: instead of a generic video shaping Z_terminus,
    we use the SENDER'S measured impedance landscape.

    Physics (Paper VI §4.3 — content-free property):
      The matching network does not need to know what the sender dreamed.
      It only needs the Z-map: {channel → (Z_src, Z_load)}.
      The PGO probes in the receiver's brain will reflect off these
      boundaries and produce a dream whose structure is shaped by the
      sender's impedance fingerprint.

    Args:
        signature: Sender's dream Z-signature (from capture_sender_dream)
        frame_idx: Temporal index (for phase variation)
        rng: Random state
        transfer_gain: How much of sender's pattern to transfer (0..1)

    Returns:
        List of (channel_name, Z_source, Z_load) for the receiver
    """
    channels: List[Tuple[str, float, float]] = []
    phase = np.sin(2 * np.pi * 0.5 * frame_idx * 0.01)  # Slow temporal modulation
    temporal_mod = 0.5 + 0.5 * phase

    for ch_name in DREAM_CHANNEL_NAMES:
        # Receiver's natural impedance (random baseline)
        # Mirror pre-training narrows σ_Z on matched channels (Paper VI §4.5):
        #   σ_Z_eff = σ_Z_base × sigma_factor,  sigma_factor < 1 after matching
        sigma_z = 20.0
        if mirror_sigma_map and ch_name in mirror_sigma_map:
            sigma_z *= mirror_sigma_map[ch_name]
        z_src = Z_BASE + rng.randn() * 5.0
        z_load_random = Z_BASE + rng.randn() * sigma_z

        if ch_name in signature.channel_z_map:
            # Sender's Z-map provides the TARGET impedance
            sender_z_src, sender_z_load = signature.channel_z_map[ch_name]

            # Bridge: push receiver's Z_terminus toward sender's Z_load
            # weighted by transfer_gain and temporal modulation
            bridge_strength = transfer_gain * temporal_mod
            z_load = z_load_random + bridge_strength * (sender_z_load - z_load_random)
        else:
            z_load = z_load_random

        channels.append((ch_name, max(10.0, z_src), max(10.0, z_load)))

    return channels


def bridge_dream_night(
    brain: AliceBrain,
    condition: str,
    signature: Optional[DreamSignature],
    night_idx: int,
    rng: np.random.RandomState,
    use_video: bool = False,
    mirror_sigma_map: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> List[BridgeRecord]:
    """
    One night of dream bridge delivery.

    Conditions:
      M (Matched):    use signature (sender's Z-map) as Z_terminus modulation
      S (Stranger):   use signature but without mirror pre-matching
      V (Video-only): use generic video (Phase 27 style)
      N (Null):       no modulation at all
    """
    records = []

    # Wake phase with moderate stress
    for t in range(STRESSED_AWAKE_TICKS):
        pixels = 0.5 + 0.1 * rng.randn(256)
        pixels = np.clip(pixels, 0, 1)
        brain.see(pixels, priority=Priority.BACKGROUND)
        brain.sleep_physics.awake_tick(
            reflected_energy=STRESS_BASE + STRESS_RANGE * rng.rand(),
        )

    # Sleep phase
    brain.sleep_physics.begin_sleep()

    for cycle in range(N_DREAM_CYCLES):
        schedule = make_sleep_schedule(SLEEP_CYCLE_TICKS)
        rec = BridgeRecord(condition=condition, night=night_idx, cycle=cycle)
        frame_idx = cycle * SLEEP_CYCLE_TICKS

        for tick_in_cycle, stage in enumerate(schedule):
            use_modulation = (
                stage == "rem"
                and tick_in_cycle % IMPEDANCE_MOD_INTERVAL == 0
                and condition != "N"
            )

            if use_modulation:
                if condition in ("M", "S") and signature is not None:
                    # Bridge: sender's Z-map → receiver's Z_terminus
                    channel_impedances = sender_z_to_impedance_modulation(
                        signature, frame_idx + tick_in_cycle, rng,
                        mirror_sigma_map=mirror_sigma_map,
                    )
                elif condition == "V":
                    # Generic video (Phase 27 style)
                    channel_impedances = video_to_impedance_modulation(
                        VIDEO_A_SPATIAL_FREQ, VIDEO_A_VOWEL, VIDEO_A_RHYTHM_HZ,
                        frame_idx + tick_in_cycle, rng,
                    )
                else:
                    channel_impedances = [
                        (f"ch_{i}", float(50 + 10 * rng.randn()),
                         float(50 + 10 * rng.randn()))
                        for i in range(6)
                    ]

                rec.modulations_applied += 1

                # Track Γ per channel
                for ch_name, zs, zl in channel_impedances:
                    denom = zs + zl
                    g = abs(zl - zs) / denom if denom > 0 else 1.0
                    rec.modulated_gammas.append(g)
                    # Separate sender-matched vs reference channels
                    if (signature is not None
                            and ch_name in signature.channel_z_map):
                        rec.sender_gammas.append(g)
                    else:
                        rec.receiver_gammas.append(g)
            else:
                channel_impedances = [
                    (f"ch_{i}", float(50 + 10 * rng.randn()),
                     float(50 + 10 * rng.randn()))
                    for i in range(6)
                ]
                # Track reference Γ (no modulation)
                for _, zs, zl in channel_impedances:
                    denom = zs + zl
                    g = abs(zl - zs) / denom if denom > 0 else 1.0
                    rec.reference_gammas.append(g)

            if stage == "rem":
                rec.ticks_in_rem += 1

            sleep_result = brain.sleep_physics.sleep_tick(
                stage=stage,
                recent_memories=[f"mem_{i}" for i in range(5)],
                channel_impedances=channel_impedances,
                synaptic_strengths=list(rng.uniform(0.3, 1.5, 100)),
            )

            if stage == "rem" and sleep_result.get("dream_result"):
                dr = sleep_result["dream_result"]
                am = dr.get("amp_multiplier", 1.0)
                rec.peak_amp_multiplier = max(rec.peak_amp_multiplier, am)

        records.append(rec)

    brain.sleep_physics.end_sleep()
    return records


# ============================================================
# Phase 3: Communication verification
# ============================================================

def communication_verification(
    sender_brain: AliceBrain,
    receiver_brain: AliceBrain,
    rounds: int,
    empathy: float = 0.6,
    effort: float = 0.7,
    verbose: bool = False,
) -> List[Tuple[float, float]]:
    """
    Post-bridge communication: measure Γ_social trend.

    If the bridge worked, the receiver's channels have been partially
    reshaped toward the sender's Z-map → Γ_social should be lower.

    Mirror pre-trained pairs use higher empathy (learned capacity).
    """
    gamma_pairs = []

    for r in range(rounds):
        sp_s = sender_brain.semantic_pressure.tick(
            arousal=0.5,
            phi=sender_brain.vitals.consciousness,
            semantic_field=sender_brain.semantic_field,
            wernicke=sender_brain.wernicke,
        )
        sp_r = receiver_brain.semantic_pressure.tick(
            arousal=0.5,
            phi=receiver_brain.vitals.consciousness,
            semantic_field=receiver_brain.semantic_field,
            wernicke=receiver_brain.wernicke,
        )

        pressure_s = sender_brain.semantic_pressure.pressure
        pressure_r = receiver_brain.semantic_pressure.pressure

        result_sr, result_rs = sender_brain.social_resonance.bidirectional_couple(
            agent_a_id="sender",
            agent_b_id="receiver",
            pressure_a=pressure_s,
            pressure_b=pressure_r,
            empathy_a=empathy,
            empathy_b=empathy,
            effort_a=effort,
            effort_b=effort,
            phi_a=max(0.3, sender_brain.vitals.consciousness),
            phi_b=max(0.3, receiver_brain.vitals.consciousness),
        )

        if pressure_s > 0:
            sender_brain.semantic_pressure.release(
                gamma_speech=result_sr.gamma_social,
                phi=sender_brain.vitals.consciousness,
            )
        if pressure_r > 0:
            receiver_brain.semantic_pressure.release(
                gamma_speech=result_rs.gamma_social,
                phi=receiver_brain.vitals.consciousness,
            )

        gamma_pairs.append((result_sr.gamma_social, result_rs.gamma_social))

        if verbose and r % 5 == 0:
            print(f"        Round {r:2d}: Γ_s→r={result_sr.gamma_social:.4f} "
                  f"Γ_r→s={result_rs.gamma_social:.4f}")

    return gamma_pairs


# ============================================================
# Run one condition
# ============================================================

def run_condition(
    condition: str,
    verbose: bool = True,
) -> ConditionResult:
    """
    Run one experimental condition (M/S/V/N).

    M (Matched):    mirror pre-training + sender Z-map bridge
    S (Stranger):   no pre-training + sender Z-map bridge
    V (Video-only): mirror pre-training + generic video
    N (Null):       no pre-training + no modulation
    """
    result = ConditionResult(condition=condition)
    rng = np.random.RandomState(SEED_SENDER + hash(condition) % 1000)

    do_mirror = condition in ("M", "V")
    do_bridge = condition in ("M", "S")
    do_video = condition == "V"

    # --- Create sender and receiver ---
    sender = AliceBrain(neuron_count=NEURON_COUNT)
    receiver = AliceBrain(neuron_count=NEURON_COUNT)

    if verbose:
        print(f"\n  ┌─ Condition {condition}: "
              f"mirror={'YES' if do_mirror else 'NO'}, "
              f"bridge={'SENDER Z-MAP' if do_bridge else ('VIDEO' if do_video else 'NONE')}")

    # --- EXP 0: Mirror pre-training (if applicable) ---
    if do_mirror:
        if verbose:
            print(f"  │  EXP 0: Mirror Pre-Training ({MIRROR_PRETRAIN_ROUNDS} rounds)")

        sender_mirror = MirrorNeuronEngine()
        receiver_mirror = MirrorNeuronEngine()

        mt_result = mirror_pretraining(
            sender_mirror, receiver_mirror,
            sender_id="sender", receiver_id="receiver",
            rounds=MIRROR_PRETRAIN_ROUNDS,
            rng=rng,
            verbose=verbose,
        )
        result.mirror_training = mt_result

        if verbose:
            print(f"  │  Mirror result: Γ_mirror {mt_result.gamma_mirror_initial:.4f} "
                  f"→ {mt_result.gamma_mirror_final:.4f}  "
                  f"Z_bond {mt_result.bond_impedance_initial:.1f} "
                  f"→ {mt_result.bond_impedance_final:.1f}Ω  "
                  f"familiarity={mt_result.familiarity:.3f}")
    # --- Compute mirror sigma map (Bug 1 fix) ---
    # Mirror pre-training narrows \u03c3_Z on matched channels.
    # sigma_factor = Z_bond_final / Z_bond_initial  (< 1 when bond strengthened)
    # Combined with empathy capacity for a stronger effect.
    mirror_sigma_map: Optional[Dict[str, float]] = None
    mirror_empathy: float = 0.6   # default
    mirror_effort: float = 0.7    # default

    if do_mirror and result.mirror_training:
        mt = result.mirror_training
        # Bond ratio: 39.9/50 \u2248 0.80 (20% noise reduction)
        bond_ratio = mt.bond_impedance_final / max(mt.bond_impedance_initial, 1.0)
        # Empathy amplifies matching: more empathic \u2192 better matching
        empathy_factor = 1.0 - 0.3 * mt.empathy_capacity  # 0.71 empathy \u2192 0.787
        sigma_factor = bond_ratio * empathy_factor
        sigma_factor = max(0.3, min(1.0, sigma_factor))  # Clamp [0.3, 1.0]

        mirror_sigma_map = {
            ch: sigma_factor for ch in DREAM_CHANNEL_NAMES
        }
        mirror_empathy = min(0.95, 0.6 + mt.empathy_capacity * 0.4)
        mirror_effort = min(0.95, 0.7 + mt.familiarity * 0.2)

        if verbose:
            print(f"  \u2502  Mirror \u03c3 reduction: factor={sigma_factor:.3f}  "
                  f"(bond_ratio={bond_ratio:.3f}, empathy_factor={empathy_factor:.3f})")
            print(f"  \u2502  Social params: empathy={mirror_empathy:.3f}, "
                  f"effort={mirror_effort:.3f}")
    # --- EXP 1: Sender dream capture ---
    if verbose:
        print(f"  │  EXP 1: Sender Dream Capture")

    rng_sender = np.random.RandomState(SEED_SENDER)
    signature = capture_sender_dream(
        sender,
        spatial_freq=VIDEO_A_SPATIAL_FREQ,
        vowel=VIDEO_A_VOWEL,
        rhythm_hz=VIDEO_A_RHYTHM_HZ,
        rng=rng_sender,
        verbose=verbose,
    )

    # --- EXP 2: Dream bridge delivery ---
    if verbose:
        mode = "Sender Z-map" if do_bridge else ("Video" if do_video else "None")
        print(f"  │  EXP 2: Dream Bridge Delivery (mode={mode})")

    rng_receiver = np.random.RandomState(SEED_RECEIVER)
    bridge_records = bridge_dream_night(
        receiver,
        condition=condition,
        signature=signature if do_bridge else None,
        night_idx=0,
        rng=rng_receiver,
        use_video=do_video,
        mirror_sigma_map=mirror_sigma_map if condition == "M" else None,
        verbose=verbose,
    )
    result.dream_records = bridge_records

    # Aggregate metrics
    all_mod = [g for r in bridge_records for g in r.modulated_gammas]
    all_ref = [g for r in bridge_records for g in r.reference_gammas]
    result.n_modulated_samples = len(all_mod)
    result.n_reference_samples = len(all_ref)

    if all_mod:
        result.mean_modulated_gamma = float(np.mean(all_mod))
    if all_ref:
        result.mean_reference_gamma = float(np.mean(all_ref))

    # Only compute contrast when modulated samples exist (Bug 3 fix)
    if result.n_modulated_samples > 0:
        result.gamma_contrast = abs(
            result.mean_modulated_gamma - result.mean_reference_gamma
        )
    else:
        # N condition: no modulation → no contrast, no SNR
        result.gamma_contrast = 0.0

    if all_ref and result.n_modulated_samples > 0:
        sigma_ref = float(np.std(all_ref))
        if sigma_ref > 1e-6:
            result.snr = result.gamma_contrast / sigma_ref

    # --- EXP 3: Post-bridge mirror check ---
    if do_mirror and result.mirror_training:
        # Re-measure Γ_mirror after bridge delivery
        receiver_mirror_post = MirrorNeuronEngine()
        # Simulate that receiver's channels have been partially shifted
        # by running a few observations with sender-like impedances
        for ch_name, (zs, zl) in signature.channel_z_map.items():
            modality = ch_name.split("_")[0] if "_" in ch_name else "vocal"
            if modality in ("visual", "auditory"):
                modality = "vocal"  # Map to available motor model
            receiver_mirror_post.observe_action(
                agent_id="sender",
                modality=modality,
                observed_impedance=zl,
                action_label=f"post_bridge_{ch_name}",
            )
        post_event = receiver_mirror_post.observe_action(
            agent_id="sender",
            modality="vocal",
            observed_impedance=MOTOR_MIRROR_IMPEDANCE,
            action_label="post_check",
        )
        result.gamma_mirror_post = post_event.gamma_mirror
        result.delta_gamma_mirror = (
            result.gamma_mirror_post - result.mirror_training.gamma_mirror_final
        )

    # --- EXP 4: Communication verification ---
    if verbose:
        print(f"  │  EXP 3: Communication Verification ({COMM_ROUNDS} rounds)")

    comm_empathy = mirror_empathy if do_mirror else 0.6
    comm_effort = mirror_effort if do_mirror else 0.7
    gamma_pairs = communication_verification(
        sender, receiver, COMM_ROUNDS,
        empathy=comm_empathy,
        effort=comm_effort,
        verbose=verbose,
    )
    if gamma_pairs:
        result.gamma_social_first = sum(gamma_pairs[0]) / 2
        result.gamma_social_last = sum(gamma_pairs[-1]) / 2
        result.gamma_social_trend = result.gamma_social_last - result.gamma_social_first

    if verbose:
        print(f"  │")
        print(f"  │  Results:")
        print(f"  │    Modulated Γ̅  = {result.mean_modulated_gamma:.4f} "
              f"(n={result.n_modulated_samples})")
        print(f"  │    Reference Γ̅  = {result.mean_reference_gamma:.4f} "
              f"(n={result.n_reference_samples})")
        print(f"  │    Contrast     = {result.gamma_contrast:.4f}")
        print(f"  │    SNR          = {result.snr:.4f}")
        if result.mirror_training:
            print(f"  │    Γ_mirror post = {result.gamma_mirror_post:.4f} "
                  f"(Δ={result.delta_gamma_mirror:+.4f})")
        print(f"  │    Γ_social      = {result.gamma_social_first:.4f} "
              f"→ {result.gamma_social_last:.4f} "
              f"(Δ={result.gamma_social_trend:+.4f})")
        print(f"  └─")

    return result


# ============================================================
# Main experiment
# ============================================================

def run_experiment(verbose: bool = True) -> Dict[str, Any]:
    """
    Phase 29 — The Dream Bridge (託夢橋)

    2×2 Experimental Design:
      ┌─────────────────────┬──────────────────┬──────────────────┐
      │                     │ Sender Z-map     │ Generic/None     │
      ├─────────────────────┼──────────────────┼──────────────────┤
      │ Mirror pre-training │ M (Matched)      │ V (Video-only)   │
      ├─────────────────────┼──────────────────┼──────────────────┤
      │ No pre-training     │ S (Stranger)     │ N (Null)         │
      └─────────────────────┴──────────────────┴──────────────────┘

    Predictions:
      1. M > S in SNR (mirror matching narrows σ_Z)
      2. M > V in Γ_contrast (sender Z-map is more specific than generic video)
      3. S > N (even without matching, sender Z-map provides some structure)
      4. M has lowest Γ_social in post-bridge communication
    """
    t0 = time.time()

    if verbose:
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  Phase 29 — The Dream Bridge (託夢橋)                      ║")
        print("║                                                            ║")
        print("║  Face 3 (dream incubation) + Face 4 (matching network)     ║")
        print("║  Mirror neurons pre-match → sender Z-map → receiver dream  ║")
        print("║                                                            ║")
        print("║  'You must know someone before you can reach them           ║")
        print("║   in a dream.'                                             ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()

    conditions = ["M", "S", "V", "N"]
    results: Dict[str, ConditionResult] = {}

    for cond in conditions:
        results[cond] = run_condition(cond, verbose=verbose)

    elapsed = time.time() - t0

    # ══════════════════════════════════════════════════════════
    # Comparative Analysis
    # ══════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'=' * 64}")
        print(f"  COMPARATIVE ANALYSIS — Phase 29 Dream Bridge")
        print(f"{'=' * 64}")
        print()
        print(f"  {'Cond':4s} │ {'Mirror':6s} │ {'Source':10s} │ "
              f"{'Γ̅_mod':7s} │ {'Γ̅_ref':7s} │ {'Contrast':8s} │ "
              f"{'SNR':7s} │ {'Γ_social':8s}")
        print(f"  {'─' * 4} │ {'─' * 6} │ {'─' * 10} │ "
              f"{'─' * 7} │ {'─' * 7} │ {'─' * 8} │ "
              f"{'─' * 7} │ {'─' * 8}")

        for cond in conditions:
            r = results[cond]
            mirror_str = "YES" if r.mirror_training else "NO"
            source_str = {
                "M": "Sender Z",
                "S": "Sender Z",
                "V": "Video",
                "N": "None",
            }[cond]
            print(f"  {cond:4s} │ {mirror_str:6s} │ {source_str:10s} │ "
                  f"{r.mean_modulated_gamma:7.4f} │ {r.mean_reference_gamma:7.4f} │ "
                  f"{r.gamma_contrast:8.4f} │ {r.snr:7.4f} │ "
                  f"{r.gamma_social_last:8.4f}")

        print()

        # Key comparisons
        m, s, v, n = results["M"], results["S"], results["V"], results["N"]
        print("  Key comparisons:")
        print(f"    Mirror effect on SNR:    M({m.snr:.4f}) vs S({s.snr:.4f})  "
              f"→ Δ={m.snr - s.snr:+.4f}")
        print(f"    Z-map vs video:          M({m.gamma_contrast:.4f}) vs "
              f"V({v.gamma_contrast:.4f})  → Δ={m.gamma_contrast - v.gamma_contrast:+.4f}")
        print(f"    Z-map vs null:           S({s.gamma_contrast:.4f}) vs "
              f"N({n.gamma_contrast:.4f})  → Δ={s.gamma_contrast - n.gamma_contrast:+.4f}")
        print(f"    Social rapport (M vs N): Γ_social "
              f"M={m.gamma_social_last:.4f} N={n.gamma_social_last:.4f}  "
              f"→ Δ={m.gamma_social_last - n.gamma_social_last:+.4f}")

        if m.mirror_training:
            print(f"\n  Mirror pre-training convergence:")
            print(f"    Γ_mirror: {m.mirror_training.gamma_mirror_initial:.4f} "
                  f"→ {m.mirror_training.gamma_mirror_final:.4f}")
            print(f"    Z_bond:   {m.mirror_training.bond_impedance_initial:.1f}Ω "
                  f"→ {m.mirror_training.bond_impedance_final:.1f}Ω")
            print(f"    Empathy:  {m.mirror_training.empathy_capacity:.3f}")
            print(f"    ToM:      {m.mirror_training.tom_capacity:.3f}")

        print(f"\n  Elapsed: {elapsed:.1f}s")
        print()
        print("  'The bridge was already there.")
        print("   Mirror neurons built it over a lifetime of shared experience.")
        print("   We just opened the gate.'")
        print()

    return {
        "elapsed_s": elapsed,
        "conditions": {
            cond: {
                "mean_modulated_gamma": r.mean_modulated_gamma,
                "mean_reference_gamma": r.mean_reference_gamma,
                "gamma_contrast": r.gamma_contrast,
                "snr": r.snr,
                "n_modulated": r.n_modulated_samples,
                "n_reference": r.n_reference_samples,
                "gamma_social_first": r.gamma_social_first,
                "gamma_social_last": r.gamma_social_last,
                "gamma_social_trend": r.gamma_social_trend,
                "mirror_gamma_initial": (
                    r.mirror_training.gamma_mirror_initial
                    if r.mirror_training else None
                ),
                "mirror_gamma_final": (
                    r.mirror_training.gamma_mirror_final
                    if r.mirror_training else None
                ),
                "mirror_bond_initial": (
                    r.mirror_training.bond_impedance_initial
                    if r.mirror_training else None
                ),
                "mirror_bond_final": (
                    r.mirror_training.bond_impedance_final
                    if r.mirror_training else None
                ),
            }
            for cond, r in results.items()
        },
    }


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    verbose = "--quiet" not in sys.argv
    result = run_experiment(verbose=verbose)
    if not verbose:
        for cond, data in result["conditions"].items():
            print(f"  {cond}: contrast={data['gamma_contrast']:.4f} "
                  f"SNR={data['snr']:.4f} "
                  f"Γ_social={data['gamma_social_last']:.4f}")
