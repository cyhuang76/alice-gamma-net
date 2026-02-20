# -*- coding: utf-8 -*-
"""
Experiment: Temporal Calibration — Cross-Modal Signal Binding (Action Model)

Experiment contents:
  1. Synchronous vs asynchronous: two signals arriving simultaneously vs with delay
  2. Drift accumulation and calibration: continuous drift → calibrator tracking → auto-compensation
  3. Complete action loop: perceive → reach → feedback → temporal binding
  4. Cognitive dissonance: behavior when calibration quality collapses

'In a coaxial cable, sound and images belong to the same frame
  because they arrive at the same moment.
  This is the action model.'
"""

import math
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from alice.brain.calibration import (
    TemporalCalibrator,
    _FeatureMatcher,
    TEMPORAL_WINDOW_MS,
    EXTENDED_WINDOW_MS,
)
from alice.core.signal import ElectricalSignal


def _make_signal(
    freq=30.0, amp=1.0, phase=0.0,
    source="test", modality="visual", timestamp=None, n=64,
):
    t = np.linspace(0, 1, n, endpoint=False)
    waveform = amp * np.sin(2 * np.pi * freq * t + phase)
    return ElectricalSignal(
        waveform=waveform, amplitude=amp, frequency=freq, phase=phase,
        impedance=50.0, snr=15.0,
        timestamp=timestamp if timestamp is not None else time.time(),
        source=source, modality=modality,
    )


def separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ============================================================================
# Experiment 1: Synchronize vs Desynchronize
# ============================================================================

def exp1_sync_vs_desync():
    separator("Experiment 1: Synchronous vs Asynchronous — Time window binding")

    delays_ms = [0, 5, 10, 20, 30, 40, 50, 80, 100, 150, 200, 300]

    print(f" Time window: {TEMPORAL_WINDOW_MS}ms (primary) / {EXTENDED_WINDOW_MS}ms (extended)")
    print(f" Sending visual + proprioception with different delays\n")
    print(f" {'delay(ms)':>8s} {'bound?':>5s} {'modalities':>6s} {'bindings':>6s} {'score':>6s} note")
    print(f"  {'-'*8}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*20}")

    for delay_ms in delays_ms:
        cal = TemporalCalibrator()
        now = time.time()

        sig_v = _make_signal(modality="visual", source="eye", timestamp=now,
                             freq=30.0, amp=1.0)
        sig_p = _make_signal(modality="proprioception", source="hand",
                             timestamp=now + delay_ms / 1000.0,
                             freq=31.0, amp=1.05) # Slightly different features

        cal.receive(sig_v)
        cal.receive(sig_p)
        frame = cal.bind()

        n_modalities = len(frame.modalities) if frame else 0
        n_bindings = len(frame.bindings) if frame else 0
        score = ""
        if frame and frame.binding_scores:
            score = f"{list(frame.binding_scores.values())[0]:.3f}"

        bound = n_modalities >= 2

        if delay_ms <= TEMPORAL_WINDOW_MS:
            note = "Within primary window → temporal binding"
        elif delay_ms <= EXTENDED_WINDOW_MS:
            note = "Extended window → feature binding?"
        else:
            note = "Beyond all windows"

        print(f"  {delay_ms:8d}  {'  ✓  ' if bound else '  ✗  '}  "
              f"{n_modalities:6d}  {n_bindings:6d}  {score:>6s}  {note}")

    print(f"\n Conclusion: the closer in time, the easier to bind; beyond the window requires feature matching")


# ============================================================================
# Experiment 2: Drift Tracking and Calibration
# ============================================================================

def exp2_drift_tracking():
    separator("Experiment 2: Drift Tracking — How the calibrator tracks temporal offset")

    cal = TemporalCalibrator()
    fixed_drift_ms = 15.0 # Proprioception has fixed 15ms delay

    print(f" Simulation scenario: proprioception channel has fixed delay of {fixed_drift_ms}ms")
    print(f" Calibrator tracks drift via EMA and auto-compensates\n")
    print(f" {'cycle':>4s} {'cal quality':>8s} {'drift est(ms)':>12s} {'bound?':>8s} quality bar")
    print(f"  {'-'*4}  {'-'*8}  {'-'*12}  {'-'*8}  {'-'*30}")

    for i in range(30):
        now = time.time()
        sig_v = _make_signal(modality="visual", timestamp=now, freq=30.0)
        sig_p = _make_signal(modality="proprioception",
                             timestamp=now + fixed_drift_ms / 1000.0,
                             freq=31.0, amp=1.05)

        cal.receive(sig_v)
        cal.receive(sig_p)
        frame = cal.bind()

        state = cal.get_calibration_state()
        quality = state["quality"]
        drifts = state["drifts_ms"]
        prop_drift = drifts.get("proprioception", 0.0)
        bound = frame.is_complete if frame else False

        bar_len = int(quality * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)

        if (i + 1) % 3 == 0 or i < 3:
            print(f"  {i+1:4d}  {quality:8.4f}  {prop_drift:12.4f}  "
                  f"{'  ✓   ' if bound else '  ✗   '}  {bar}")

    final_state = cal.get_calibration_state()
    print(f"\n Final drift estimate: {final_state['drifts_ms']}")
    print(f" Final calibration quality: {final_state['quality']:.4f}")
    print(f" Conclusion: calibrator tracked proprioception {fixed_drift_ms}ms drift")


# ============================================================================
# Experiment 3: Complete Action Loop
# ============================================================================

def exp3_action_loop():
    separator("Experiment 3: Complete Action Loop — perceive → reach → bind")

    from alice.alice_brain import AliceBrain
    from alice.body.hand import AliceHand

    brain = AliceBrain(neuron_count=20)
    brain.hand = AliceHand(initial_pos=(960.0, 540.0))

    targets = [
        (300.0, 200.0, "left target"),
        (1500.0, 800.0, "right target"),
        (960.0, 540.0, "back to center"),
    ]

    print(f" Action loop: eyes see target → brain processes → hand reaches out → temporal binding\n")

    for tx, ty, name in targets:
        # 1. Perceive (eyes see the target)
        stimulus = np.random.randn(64) * 0.5
        p_result = brain.perceive(stimulus)
        cal_p = p_result.get("calibration", {})

        # 2. Reach (hand moves toward target)
        r_result = brain.reach_for(tx, ty)
        tb = r_result.get("temporal_binding", {})

        print(f"  → {name} ({tx}, {ty})")
        print(f" Perception: calibration quality={cal_p.get('quality', 'N/A')}")
        print(f" Reach: reached={r_result['reach']['reached']}, "
              f"steps={r_result['reach']['steps']}, "
              f"dopamine={r_result['dopamine']}")
        print(f" Temporal binding: frame_id={tb.get('frame_id', 'N/A')}, "
              f"modalities={tb.get('bound_modalities', [])}, "
              f"bindings={tb.get('bindings', 0)}")
        if tb.get("binding_scores"):
            for pair, score in tb["binding_scores"].items():
                print(f"      {pair}: {score:.4f}")
        print()

    # Final statistics
    stats = brain.calibrator.get_stats()
    print(f" === Calibrator Statistics ===")
    print(f" Total signals: {stats['total_signals']}")
    print(f" Total frames: {stats['total_frames']}")
    print(f" Temporal bindings: {stats['temporal_bindings']}")
    print(f" Feature bindings: {stats['feature_bindings']}")
    print(f" Calibration quality: {stats['calibration_quality']:.4f}")


# ============================================================================
# Experiment 4: Feature Matching Heatmap
# ============================================================================

def exp4_feature_heatmap():
    separator("Experiment 4: Feature Matching — Match scores for different feature combinations")

    frequencies = [5, 15, 30, 50, 80]
    amplitudes = [0.1, 0.5, 1.0, 2.0, 5.0]

    # Reference signal
    ref = _make_signal(freq=30.0, amp=1.0, phase=0.0)

    print(f" Reference signal: freq=30Hz, amp=1.0, phase=0.0")
    print(f" Comparing different frequency × amplitude combinations for match scores\n")

    header = f"  {'freq\\amp':>8s}"
    for a in amplitudes:
        header += f"  {a:>5.1f}"
    print(header)
    print(f"  {'-'*8}" + "  -----" * len(amplitudes))

    for f in frequencies:
        row = f"  {f:>6d}Hz"
        for a in amplitudes:
            sig = _make_signal(freq=float(f), amp=a, phase=0.0)
            score = _FeatureMatcher.match(ref, sig)
            # Use color scale display
            if score > 0.8:
                ch = "██"
            elif score > 0.6:
                ch = "▓▓"
            elif score > 0.4:
                ch = "▒▒"
            elif score > 0.2:
                ch = "░░"
            else:
                ch = "  "
            row += f"  {ch}{score:.1f}"
        print(row)

    print(f"\n Legend: ██>0.8 ▓▓>0.6 ▒▒>0.4 ░░>0.2 empty<0.2")
    print(f" Conclusion: the closer frequency and amplitude are to the reference signal, the higher the match score")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("  Alice Smart System — Temporal Calibration Experiment")
    print(" Cross-Modal Signal Binding — Action Model")
    print(" 'In a coaxial cable, signals arriving at the same moment belong to the same event'")
    print("█" * 70)

    exp1_sync_vs_desync()
    exp2_drift_tracking()
    exp3_action_loop()
    exp4_feature_heatmap()

    separator("ALL EXPERIMENTS COMPLETED ✓")
    print(" 'Signal matching is temporal correlation.")
    print(" Most of the time, sound and images arrive at the same moment.")
    print(" Sometimes there are errors, but other signals are used to find correspondences and calibrate errors.")
    print(" This is the action model.'\n")
