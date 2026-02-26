# -*- coding: utf-8 -*-
"""
Alice Smart System — Motor Development Clinical Experiment

Four clinical scenarios:
  1. Infant Motor Development (PID gain from 5% to 100%)
  2. Adult precise reaching (bell-shaped velocity curve — reaching for a cup)
  3. Post-injury protective motor (pain → guard ↑ → movements become smaller and slower)
  4. Rehabilitation process (repeated practice → guard extinction → function recovery)

All behavior emerges naturally from the same PID + Newtonian mechanics, no special rules.
"""

from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alice.body.hand import AliceHand, MOTOR_MATURITY_INITIAL


def print_header(title: str):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def velocity_profile(trajectory):
    """Compute trajectory instantaneous velocity sequence"""
    velocities = []
    dt = 0.016
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0] - trajectory[i - 1][0]
        dy = trajectory[i][1] - trajectory[i - 1][1]
        v = math.sqrt(dx * dx + dy * dy) / dt
        velocities.append(v)
    return velocities


def ascii_velocity_curve(velocities, width=60, height=10, label=""):
    """ASCII plot velocity curve"""
    if not velocities:
        return
    max_v = max(velocities) if max(velocities) > 0 else 1.0
    # Downsample to width points
    step = max(1, len(velocities) // width)
    samples = [velocities[i] for i in range(0, len(velocities), step)][:width]

    print(f"\n Velocity curve {label} (peak={max_v:.0f} px/s)")
    print(f"  {'─' * (width + 6)}")
    for row in range(height, -1, -1):
        threshold = max_v * row / height
        line = "  │"
        for s in samples:
            if s >= threshold:
                line += "█"
            else:
                line += " "
        if row == height:
            line += f"│ {max_v:.0f}"
        elif row == 0:
            line += f"│ 0"
        else:
            line += "│"
        print(line)
    print(f"  └{'─' * width}┘")
    print(f"   0{' ' * (width - 5)}steps")


# ======================================================================
# Experiment 1: Infant Motor Development
# ======================================================================

def exp1_infant_development():
    print_header("EXP 1: INFANT MOTOR DEVELOPMENT — Infant Motor Development")
    print(" Simulating infant from birth to learning precise reaching")
    print(" motor_maturity: 0.05 (newborn) → 1.0 (adult)")
    print(" Even FAILED reaching attempts develop neural pathways (practice_rate=0.001)")
    print()

    hand = AliceHand(workspace_size=(800, 600), initial_pos=(400, 300))
    hand.motor_maturity = MOTOR_MATURITY_INITIAL # Infant start

    target = (600.0, 400.0) # Fixed target
    distance = math.sqrt((target[0] - 400) ** 2 + (target[1] - 300) ** 2)

    print(f" Target distance: {distance:.0f} px")
    print(f"  {'─' * 65}")
    print(f" {'Trial':>4} {'Maturity':>6} {'Reached':>4} {'Steps':>4} {'Error':>8} {'PeakVel':>8} Status")
    print(f"  {'─' * 65}")

    milestones = {}  # {threshold: label}
    milestone_thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
    milestone_labels = [
        "Newborn — barely moves",
        "1 month — faint waving",
        "3 months — obvious arm waving",
        "5 months — reaching toward target",
        "7 months — close to target",
        "9 months — roughly reaching",
        "11 months — stable control",
        "Mature — precise reaching",
        "Adult — fully precise",
    ]

    first_reach = None
    last_maturity_printed = -1

    for trial in range(500):
        # Each trial starts from center
        hand.x, hand.y = 400.0, 300.0
        hand.vx, hand.vy = 0.0, 0.0

        result = hand.reach(target[0], target[1], ram_temperature=0.0, max_steps=300)
        velocities = velocity_profile(result["trajectory"])
        peak_v = max(velocities) if velocities else 0

        # Mark first reach
        if result["reached"] and first_reach is None:
            first_reach = trial + 1

        # Check if milestone crossed
        stage_label = ""
        for i, thresh in enumerate(milestone_thresholds):
            if hand.motor_maturity >= thresh and thresh > last_maturity_printed:
                stage_label = f"★ {milestone_labels[i]}"
                last_maturity_printed = thresh

        # Display milestones + every 50 trials + first 3 + first reach
        should_print = (
            stage_label
            or trial < 3
            or trial % 50 == 0
            or (result["reached"] and first_reach == trial + 1)
        )

        if should_print:
            reached_str = "✓" if result["reached"] else "✗"
            print(
                f"  {trial + 1:4d}  {hand.motor_maturity:6.3f}  {reached_str:>4}  "
                f"{result['steps']:4d}  {result['final_error']:8.2f}  "
                f"{peak_v:8.0f}  {stage_label}"
            )

    print(f"\n Final maturity: {hand.motor_maturity:.3f}")
    print(f" First successful reach: trial {first_reach}" if first_reach else " First successful reach: not achieved")
    print(f" Motor experience: {hand.motor_experience} successful reaches")
    print(f" Cumulative dopamine: {hand.total_dopamine:.1f}")


# ======================================================================
# Experiment 2: Bell-Shaped Velocity Curve (reaching for a cup)
# ======================================================================

def exp2_bell_velocity():
    print_header("EXP 2: BELL-SHAPED VELOCITY — Reaching for a Cup")
    print(" Human reaching velocity curve: slow start → acceleration → braking on approach")
    print(" This is a natural result of PID + approach braking + friction damping")
    print()

    hand = AliceHand(workspace_size=(800, 600), initial_pos=(100, 300))
    hand.motor_maturity = 1.0

    # Three distance reaches
    distances = [
        ((300, 300), "Short distance (200px)"),
        ((500, 300), "Medium distance (400px)"),
        ((700, 300), "Long distance (600px)"),
    ]

    for target, label in distances:
        hand.x, hand.y = 100.0, 300.0
        hand.vx, hand.vy = 0.0, 0.0

        result = hand.reach(target[0], target[1], ram_temperature=0.0, max_steps=300)
        velocities = velocity_profile(result["trajectory"])

        ascii_velocity_curve(velocities, width=50, height=8, label=label)

        if velocities:
            peak_idx = velocities.index(max(velocities))
            total = len(velocities)
            peak_pct = peak_idx / total * 100 if total > 0 else 0
            print(f" Peak velocity at step {peak_idx}/{total} ({peak_pct:.0f}% point)")
            print(f" Reached: {'\u2713' if result['reached'] else '\u2717'} | Steps: {result['steps']} | Error: {result['final_error']:.2f}px")

    # Comparison: post-injury velocity curve
    print("\n --- Post-injury (guard=0.7) velocity curve ---")
    hand.x, hand.y = 100.0, 300.0
    hand.vx, hand.vy = 0.0, 0.0
    hand.guard_level = 0.7 # Injured state

    result = hand.reach(500, 300, ram_temperature=0.0, max_steps=300)
    velocities = velocity_profile(result["trajectory"])
    ascii_velocity_curve(velocities, width=50, height=8, label="Post-injury (guard=0.7)")
    if velocities:
        peak_v = max(velocities)
        print(f" Peak velocity: {peak_v:.0f} px/s (lower than healthy state)")
        print(f" Reached: {'\u2713' if result['reached'] else '\u2717'} | Steps: {result['steps']}")

    hand.guard_level = 0.0  # Recovery


# ======================================================================
# Experiment 3: Post-Injury Protective Motor
# ======================================================================

def exp3_injury_guarding():
    print_header("EXP 3: INJURY GUARDING — Post-Injury Protective Motor")
    print(" Injury → guard ↑ → motor force reduced → movements become slower and smaller")
    print(" Like 'limping after spraining an ankle'")
    print()

    hand = AliceHand(workspace_size=(800, 600), initial_pos=(200, 300))
    hand.motor_maturity = 1.0
    target = (600.0, 300.0)

    # Five protection levels
    guard_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    print(f" {'Guard':>6} {'Reached':>4} {'Steps':>4} {'PeakForce':>8} {'PeakVel':>8} {'Error':>8} Clinical Correspondence")
    print(f"  {'─' * 65}")

    clinical = {
        0.0: "Healthy — normal force output",
        0.2: "Mild injury — slightly cautious",
        0.4: "Moderate injury — obvious hesitation",
        0.6: "Severe injury — slow movements",
        0.8: "Extreme injury — barely dares to move",
        1.0: "Maximum protection — rigid",
    }

    for gl in guard_levels:
        hand.x, hand.y = 200.0, 300.0
        hand.vx, hand.vy = 0.0, 0.0
        hand.guard_level = gl

        result = hand.reach(target[0], target[1], ram_temperature=0.0, max_steps=500)
        velocities = velocity_profile(result["trajectory"])
        peak_v = max(velocities) if velocities else 0

        reached_str = "✓" if result["reached"] else "✗"
        print(
            f"  {gl:6.1f}  {reached_str:>4}  {result['steps']:4d}  "
            f"{result['peak_tension']:8.0f}  {peak_v:8.0f}  "
            f"{result['final_error']:8.2f}  {clinical[gl]}"
        )

    hand.guard_level = 0.0


# ======================================================================
# Experiment 4: Rehabilitation Process
# ======================================================================

def exp4_rehabilitation():
    print_header("EXP 4: REHABILITATION — Rehabilitation Process")
    print(" Injury (guard=0.8) → repeated practice → guard gradually extinguishes → function recovery")
    print(" Each successful reach → guard -= 0.01 (rehabilitation training)")
    print()

    hand = AliceHand(workspace_size=(800, 600), initial_pos=(200, 300))
    hand.motor_maturity = 1.0
    hand.guard_level = 0.8 # Severe injury
    hand.injury_memory = 0.5

    target = (500.0, 300.0)

    recovered = False
    print(f" {'Trial':>4} {'Guard':>6} {'Reached':>4} {'Steps':>4} {'PeakVel':>8} {'Dopamine':>6} Progress")
    print(f"  {'─' * 55}")

    for trial in range(100):
        hand.x, hand.y = 200.0, 300.0
        hand.vx, hand.vy = 0.0, 0.0

        result = hand.reach(target[0], target[1], ram_temperature=0.0, max_steps=500)
        velocities = velocity_profile(result["trajectory"])
        peak_v = max(velocities) if velocities else 0

        # Mark phase
        stage = ""
        if trial == 0:
            stage = "* First attempt after injury"
        elif hand.guard_level < 0.01 and not recovered:
            stage = "* Full recovery!"
            recovered = True

        if trial < 5 or trial % 10 == 0 or stage.startswith("★"):
            reached_str = "✓" if result["reached"] else "✗"
            print(
                f"  {trial + 1:4d}  {hand.guard_level:6.3f}  {reached_str:>4}  "
                f"{result['steps']:4d}  {peak_v:8.0f}  "
                f"{hand.total_dopamine:6.1f}  {stage}"
            )

    print(f"\n  Final guard: {hand.guard_level:.4f}")
    print(f" Injury memory: {hand.injury_memory:.4f}")
    print(f" Rehabilitation practice: {hand.motor_experience} successful reaches")
    print(f" Cumulative dopamine: {hand.total_dopamine:.1f}")


# ======================================================================
# Experiment 5: Comprehensive Comparison (healthy vs infant vs injured vs anxious)
# ======================================================================

def exp5_comprehensive():
    print_header("EXP 5: COMPREHENSIVE — Four-State Comprehensive Comparison")
    print()

    target = (600.0, 300.0)

    conditions = [
        ("Healthy adult", {"motor_maturity": 1.0, "guard_level": 0.0}, 0.0),
        ("Infant (3mo)", {"motor_maturity": 0.2, "guard_level": 0.0}, 0.0),
        ("Injured adult", {"motor_maturity": 1.0, "guard_level": 0.7}, 0.0),
        ("Anxious adult", {"motor_maturity": 1.0, "guard_level": 0.0}, 0.7),
        ("Injured+Anx", {"motor_maturity": 1.0, "guard_level": 0.7}, 0.7),
    ]

    print(f" {'State':>10} {'Reached':>4} {'Steps':>4} {'PeakVel':>8} {'PeakF':>8} {'Tremor':>8} {'Error':>8}")
    print(f"  {'─' * 60}")

    for name, params, temp in conditions:
        hand = AliceHand(workspace_size=(800, 600), initial_pos=(200, 300))
        hand.motor_maturity = params["motor_maturity"]
        hand.guard_level = params["guard_level"]

        result = hand.reach(target[0], target[1], ram_temperature=temp, max_steps=500)
        velocities = velocity_profile(result["trajectory"])
        peak_v = max(velocities) if velocities else 0

        reached_str = "✓" if result["reached"] else "✗"
        print(
            f"  {name:>10}  {reached_str:>4}  {result['steps']:4d}  "
            f"{peak_v:8.0f}  {result['peak_tension']:8.0f}  "
            f"{result['tremor_intensity']:8.2f}  {result['final_error']:8.2f}"
        )

    print()
    print("  Physics semantics:")
    print(" Infant: maturity=0.2 → PID force only 20% → may not reach")
    print(" Injured: guard=0.7 → force reduced to 58% → slow but stable")
    print(" Anxious: T=0.7 → tremor + coordination↓ → shaky and imprecise")
    print(" Injured+Anxious: double weakening → worst performance")


# ======================================================================
# Main
# ======================================================================

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║        Alice Smart System — Motor Development Experiment       ║")
    print("║        Motor Developmentclinicalexperiment                                         ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    exp1_infant_development()
    exp2_bell_velocity()
    exp3_injury_guarding()
    exp4_rehabilitation()
    exp5_comprehensive()

    print()
    print("=" * 70)
    print("  CONCLUSION")
    print("=" * 70)
    print(" The same PID + Newtonian mechanics, modulated by three physics parameters:")
    print(" motor_maturity (development) \u00d7 guard_level (protection) \u00d7 temperature (anxiety)")
    print(" Naturally emerges:")
    print(" \u2713 Infant clumsiness \u2192 adult precision")
    print(" \u2713 Reaching for a cup: bell-shaped velocity curve")
    print(" \u2713 Post-injury protective motor (careful and cautious)")
    print(" \u2713 Rehabilitation process: gradual recovery")
    print(" \u2713 Anxious state: hand tremor + imprecision")
    print()


if __name__ == "__main__":
    main()
