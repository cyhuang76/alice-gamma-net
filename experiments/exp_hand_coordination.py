# -*- coding: utf-8 -*-
"""
Experiment: Alice Hand — PID Control + Anxiety Tremor + Hand-Eye Coordination

Experiment contents:
  1. Basic reaching: hand moves from center to four corners — verify PID convergence
  2. Anxiety ladder: same target, ram_temperature 0→1 — see how tremor spirals out of control
  3. Multi-target continuous tracking: hand-eye coordination + dopamine accumulation curve
  4. Trajectory visualization: use ASCII art to draw hand trajectory

'This is not moveTo(x,y) in a program. This is a physics system where PID controls real motors,
  with inertia, friction, tremor — like your hand.'
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from alice.body.hand import AliceHand, REACH_THRESHOLD, DOPAMINE_ON_REACH


def separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ============================================================================
# Experiment 1: Basic Reaching — PID Convergence
# ============================================================================

def exp1_basic_reach():
    separator("Experiment 1: Basic Reaching — PID Convergence")

    hand = AliceHand(workspace_size=(1920, 1080), initial_pos=(960, 540))

    targets = [
        (100, 100, "top-left corner"),
        (1800, 100, "top-right corner"),
        (1800, 980, "bottom-right corner"),
        (100, 980, "bottom-left corner"),
        (960, 540, "back to center"),
    ]

    print(f" Starting position: ({hand.x}, {hand.y})")
    print(f" Reaching threshold: {REACH_THRESHOLD} pixels\n")

    all_reached = True
    for tx, ty, name in targets:
        result = hand.reach(tx, ty, ram_temperature=0.0)
        status = "✓" if result["reached"] else "✗"
        if not result["reached"]:
            all_reached = False
        print(f"  {status} → {name} ({tx}, {ty})")
        print(f" Steps: {result['steps']:3d} | "
              f"Final error: {result['final_error']:.2f}px | "
              f"Peak tension: {result['peak_tension']:.2f} | "
              f"Dopamine: {result['dopamine']:.1f}")

    print(f"\n Conclusion: {'ALL REACHED — PID controller converged normally' if all_reached else 'SOME MISSED'}")
    assert all_reached, "PID controller failed to converge to all targets"


# ============================================================================
# Experiment 2: Anxiety Ladder — How Tremor Spirals Out of Control
# ============================================================================

def exp2_anxiety_ladder():
    separator("Experiment 2: Anxiety Ladder — ram_temperature vs tremor intensity")

    target = (500.0, 300.0)
    temperatures = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print(f" Target: {target}")
    print(f" Start: (100, 100)")
    print(f" {'Temperature':>6s} {'Tremor':>8s} {'Steps':>4s} {'Reached?':>5s} {'Final err':>8s} {'Peak tension':>8s} Tremor bar")
    print(f"  {'-'*6}  {'-'*8}  {'-'*4}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*30}")

    tremors = []
    for temp in temperatures:
        hand = AliceHand(initial_pos=(100.0, 100.0))
        result = hand.reach(target[0], target[1], ram_temperature=temp)

        bar_len = int(result["tremor_intensity"] * 5)
        bar = "█" * min(bar_len, 30) + "░" * max(0, 30 - bar_len)

        print(f"  {temp:6.1f}  {result['tremor_intensity']:8.3f}  "
              f"{result['steps']:4d}  {'  ✓  ' if result['reached'] else '  ✗  '}  "
              f"{result['final_error']:8.3f}  {result['peak_tension']:8.3f}  {bar}")

        tremors.append(result["tremor_intensity"])

    # Verify: tremor increases with anxiety
    assert tremors[-1] > tremors[0], "Highest anxiety tremor should be larger than calm state"
    print(f"\n Conclusion: Tremor from {tremors[0]:.3f} → {tremors[-1]:.3f}")
    print(f" {'Nonlinear runaway ✓' if tremors[-1] > tremors[0] * 5 else 'Linear growth'}")


# ============================================================================
# Experiment 3: Multi-Target Continuous Tracking — Dopamine Accumulation
# ============================================================================

def exp3_dopamine_accumulation():
    separator("Experiment 3: Multi-Target Continuous Tracking — Dopamine Accumulation Curve")

    hand = AliceHand(initial_pos=(960.0, 540.0))
    rng = np.random.RandomState(12345)

    n_targets = 20
    targets = [
        (rng.uniform(100, 1800), rng.uniform(100, 980))
        for _ in range(n_targets)
    ]

    dopamine_curve = []
    success_count = 0

    print(f" Continuously reaching {n_targets} random targets\n")

    for i, (tx, ty) in enumerate(targets):
        result = hand.reach(tx, ty, ram_temperature=0.1)
        dopamine_curve.append(hand.total_dopamine)
        if result["reached"]:
            success_count += 1

        if (i + 1) % 5 == 0:
            print(f"  [{i+1:2d}/{n_targets}]  "
                  f"Accumulated dopamine: {hand.total_dopamine:5.1f} | "
                  f"Success rate: {success_count/(i+1)*100:5.1f}% | "
                  f"Last position: ({hand.x:.0f}, {hand.y:.0f})")

    stats = hand.get_stats()
    print(f"\n === Final Statistics ===")
    print(f" Total movements: {stats['total_movements']}")
    print(f" Successful reaches: {stats['total_reaches']}")
    print(f" Success rate: {stats['reach_rate']*100:.1f}%")
    print(f" Accumulated dopamine: {stats['total_dopamine']:.1f}")

    # Dopamine curve (ASCII)
    print(f"\n Dopamine accumulation curve:")
    max_d = max(dopamine_curve) if dopamine_curve else 1
    for i, d in enumerate(dopamine_curve):
        bar_len = int(d / max_d * 40) if max_d > 0 else 0
        print(f"  {i+1:2d} |{'█' * bar_len}")

    assert stats["total_reaches"] > 0, "Should reach at least some targets"


# ============================================================================
# Experiment 4: Trajectory Visualization — ASCII Art
# ============================================================================

def exp4_trajectory_visualization():
    separator("Experiment 4: Trajectory Visualization — from (100,100) to (700,500)")

    hand = AliceHand(workspace_size=(800, 600), initial_pos=(100.0, 100.0))
    result = hand.reach(700.0, 500.0, ram_temperature=0.0)
    trajectory = result["trajectory"]

    # ASCII canvas (40 x 20)
    W, H = 40, 20
    canvas = [[" " for _ in range(W)] for _ in range(H)]

    # Draw trajectory
    for i, (px, py) in enumerate(trajectory):
        cx = int(px / 800 * (W - 1))
        cy = int(py / 600 * (H - 1))
        cx = max(0, min(W - 1, cx))
        cy = max(0, min(H - 1, cy))
        canvas[cy][cx] = "·"

    # Mark start and end points
    sx, sy = int(100 / 800 * (W-1)), int(100 / 600 * (H-1))
    ex, ey = int(700 / 800 * (W-1)), int(500 / 600 * (H-1))
    canvas[sy][sx] = "S"
    canvas[ey][ex] = "E"

    # Render
    print(f"  ┌{'─' * W}┐")
    for row in canvas:
        print(f"  │{''.join(row)}│")
    print(f"  └{'─' * W}┘")
    print(f" S=start(100,100) E=end(700,500)")
    print(f" Steps: {result['steps']} Error: {result['final_error']:.2f}px")

    # Anxiety version
    print(f"\n --- Same path, ram_temperature=0.7 ---\n")

    hand2 = AliceHand(workspace_size=(800, 600), initial_pos=(100.0, 100.0))
    result2 = hand2.reach(700.0, 500.0, ram_temperature=0.7)
    trajectory2 = result2["trajectory"]

    canvas2 = [[" " for _ in range(W)] for _ in range(H)]
    for px, py in trajectory2:
        cx = int(px / 800 * (W - 1))
        cy = int(py / 600 * (H - 1))
        cx = max(0, min(W - 1, cx))
        cy = max(0, min(H - 1, cy))
        canvas2[cy][cx] = "·"
    canvas2[sy][sx] = "S"
    canvas2[ey][ex] = "E"

    print(f"  ┌{'─' * W}┐")
    for row in canvas2:
        print(f"  │{''.join(row)}│")
    print(f"  └{'─' * W}┘")
    print(f" Steps: {result2['steps']} Error: {result2['final_error']:.2f}px Tremor: {result2['tremor_intensity']:.3f}")

    assert result["reached"], "Should reach target when calm"


# ============================================================================
# Experiment 5: Proprioception — Resting vs Motor
# ============================================================================

def exp5_proprioception():
    separator("Experiment 5: Proprioception Signal — Still vs Moving")

    hand_still = AliceHand(initial_pos=(400.0, 300.0))
    hand_moving = AliceHand(initial_pos=(100.0, 100.0))

    # Let hand_moving start moving
    hand_moving.set_target(700.0, 500.0)
    for _ in range(10):
        hand_moving.tick(ram_temperature=0.0)

    sig_still = hand_still.get_proprioception()
    sig_moving = hand_moving.get_proprioception()

    print(f" Still hand:")
    print(f" Position: ({hand_still.x:.0f}, {hand_still.y:.0f})")
    print(f" Frequency: {sig_still.frequency:.1f} Hz (low freq = still)")
    print(f" Amplitude: {sig_still.amplitude:.3f}")
    print(f" Impedance: {sig_still.impedance} Ω")

    print(f"\n Moving hand:")
    print(f" Position: ({hand_moving.x:.0f}, {hand_moving.y:.0f})")
    print(f" Velocity: ({hand_moving.vx:.2f}, {hand_moving.vy:.2f})")
    print(f" Frequency: {sig_moving.frequency:.1f} Hz (high freq = fast movement)")
    print(f" Amplitude: {sig_moving.amplitude:.3f}")
    print(f" Impedance: {sig_moving.impedance} Ω")

    assert sig_moving.frequency > sig_still.frequency, "Moving hand should have higher frequency"
    print(f"\n Conclusion: Moving frequency ({sig_moving.frequency:.1f} Hz) > Still ({sig_still.frequency:.1f} Hz) ✓")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print(" Alice Smart System — Hand Motor Experiment")
    print(" PID Controller + Muscle Physics + Anxiety Tremor + Hand-Eye Coordination")
    print("█" * 70)

    exp1_basic_reach()
    exp2_anxiety_ladder()
    exp3_dopamine_accumulation()
    exp4_trajectory_visualization()
    exp5_proprioception()

    separator("ALL EXPERIMENTS PASSED ✓")
    print(" 'Her hand is not moveTo(x,y).")
    print(" Her hand is PID-controlled, with inertia, and trembles due to anxiety.")
    print(" That is why she is alive.'\n")
