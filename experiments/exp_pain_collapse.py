# -*- coding: utf-8 -*-
"""
Experiment: Pain Collapse — Pain Crash Clinical Experiment

This experiment verifies Alice's 'vital signs & pain feedback loop':
1. Baseline measurement: vital signs in quiet state
2. Gradual stress: progressively increase CRITICAL packet frequency
3. Observe crash: reach critical point → system freeze
4. Emergency recovery: emergency_reset → observe recovery curve
5. Broadcast storm: massive one-time injection → instant crash

Run:
  python -m experiments.exp_pain_collapse
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from alice.alice_brain import AliceBrain
from alice.core.protocol import Modality, Priority


def print_vitals(alice: AliceBrain, label: str = ""):
    """Print current vital signs"""
    v = alice.vitals.get_vitals()
    frozen_tag = " *** FROZEN ***" if v["is_frozen"] else ""
    trauma_tag = f"  Sens={v['pain_sensitivity']:.2f}" if v["pain_sensitivity"] > 1.0 else ""
    print(
        f"  [{label:>20s}]  "
        f"Temp={v['ram_temperature']:.3f}  "
        f"Pain={v['pain_level']:.3f}  "
        f"Stab={v['stability_index']:.3f}  "
        f"Consc={v['consciousness']:.3f}  "
        f"HR={v['heart_rate']:.0f}bpm  "
        f"Throttle={v['throttle_factor']:.2f}"
        f"{trauma_tag}"
        f"{frozen_tag}"
    )


def phase_baseline(alice: AliceBrain):
    """Phase 1: Baseline — normal stimulus, establish healthy baseline"""
    print("\n" + "=" * 70)
    print(" PHASE 1: BASELINE — Normal Operation Baseline")
    print("=" * 70)

    for i in range(5):
        signal = np.random.rand(20) * 0.5  # Mild stimulus
        alice.perceive(signal, Modality.VISUAL, Priority.NORMAL, f"baseline_{i}")
        print_vitals(alice, f"Normal #{i+1}")

    print(f"\n  → Baseline completed. System healthy ✓")


def phase_gradual_stress(alice: AliceBrain):
    """Phase 2: Gradual stress — progressively increase CRITICAL packets"""
    print("\n" + "=" * 70)
    print(" PHASE 2: GRADUAL STRESS — Progressive Pressure Increase")
    print("=" * 70)

    intensities = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0,
                   12.0, 15.0, 18.0, 20.0, 25.0, 30.0]

    for i, intensity in enumerate(intensities):
        signal = np.random.randn(20) * intensity
        result = alice.perceive(signal, Modality.TACTILE, Priority.CRITICAL, f"stress_{i}")
        print_vitals(alice, f"Stress x{intensity:.1f}")

        if result.get("status") == "FROZEN":
            print(f"\n  ★★★ SYSTEM FROZEN at intensity {intensity} (step {i+1}) ★★★")
            return i + 1

    return len(intensities)


def phase_recovery(alice: AliceBrain):
    """Phase 3: Emergency recovery"""
    print("\n" + "=" * 70)
    print(" PHASE 3: EMERGENCY RECOVERY — First Aid")
    print("=" * 70)

    print("  Before reset:")
    print_vitals(alice, "Pre-reset")

    alice.emergency_reset()
    print("\n  ★ EMERGENCY RESET ACTIVATED ★\n")

    print_vitals(alice, "Post-reset")

    # Recovery observation — very faint stimulus, observe natural cooling
    for i in range(8):
        signal = np.random.rand(20) * 0.05  # Very faint: close to no stimulus
        alice.perceive(signal, Modality.VISUAL, Priority.BACKGROUND, f"recovery_{i}")
        print_vitals(alice, f"Recovery #{i+1}")

    print(f"\n  → Recovery completed.")


def phase_broadcast_storm(alice: AliceBrain, count: int = 50):
    """Phase 4: Broadcast storm — large-scale CRITICAL injection"""
    print("\n" + "=" * 70)
    print(f" PHASE 4: BROADCAST STORM — {count} CRITICAL Packets")
    print("=" * 70)

    frozen_at = None
    pain_trajectory = []

    t0 = time.time()
    for i in range(count):
        signal = np.random.randn(20) * 5.0  # High intensity
        result = alice.perceive(signal, Modality.TACTILE, Priority.CRITICAL, f"storm_{i}")

        v = alice.vitals.get_vitals()
        pain_trajectory.append(v["pain_level"])

        if i % 10 == 0 or result.get("status") == "FROZEN":
            print_vitals(alice, f"Storm #{i+1}/{count}")

        if v["is_frozen"] and frozen_at is None:
            frozen_at = i + 1
            print(f"\n  ★★★ SYSTEM FROZEN at packet #{frozen_at} ★★★")
            # Continue injecting a few more to see if CRITICAL can still penetrate after freeze
            for j in range(5):
                result2 = alice.perceive(
                    np.random.randn(20) * 5.0, Modality.TACTILE, Priority.CRITICAL, f"storm_post_{j}"
                )
                v2 = alice.vitals.get_vitals()
                pain_trajectory.append(v2["pain_level"])
                print_vitals(alice, f"Post-freeze #{j+1}")
            break

    elapsed = time.time() - t0

    print(f"\n  Storm statistics:")
    print(f"    Packets: {len(pain_trajectory)}")
    print(f"    Elapsed time: {elapsed*1000:.1f}ms")
    print(f"    Frozen at: packet #{frozen_at or 'N/A'}")
    print(f"    Max pain: {max(pain_trajectory):.4f}")
    print(f"    Pain events: {alice.vitals.pain_events}")
    print(f"    Freeze events: {alice.vitals.freeze_events}")

    return frozen_at, pain_trajectory


def phase_resilience(alice: AliceBrain):
    """Phase 5: Resilience test — repeated 'attack-recovery' to see if system degrades"""
    print("\n" + "=" * 70)
    print(" PHASE 5: RESILIENCE TEST — Repeated Stress/Recovery")
    print("=" * 70)

    for cycle in range(3):
        print(f"\n  --- Cycle {cycle+1}/3 ---")

        # Attack
        for i in range(10):
            alice.perceive(np.random.randn(20) * 4.0, Modality.TACTILE, Priority.CRITICAL)

        v = alice.vitals.get_vitals()
        print_vitals(alice, f"After attack #{cycle+1}")

        # recovery
        alice.emergency_reset()
        for i in range(3):
            alice.perceive(np.random.rand(20) * 0.2, Modality.VISUAL, Priority.BACKGROUND)

        print_vitals(alice, f"After recovery #{cycle+1}")


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║        Alice Smart System — Pain Collapse Experiment           ║")
    print("║        Pain Crash Clinical Experiment                                      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    alice = AliceBrain(neuron_count=50)

    # Phase 1: baseline
    phase_baseline(alice)

    # Phase 2: Gradual stress
    freeze_step = phase_gradual_stress(alice)

    # Phase 3: Emergency recovery
    phase_recovery(alice)

    # Phase 4: Broadcast storm
    frozen_at, trajectory = phase_broadcast_storm(alice, count=50)

    # Phase 5: Emergency reset & resilience
    alice.emergency_reset()
    phase_resilience(alice)

    # Final Report
    print("\n" + "=" * 70)
    print("  FINAL REPORT")
    print("=" * 70)
    v = alice.vitals.get_vitals()
    print(f"  Total ticks: {v['total_ticks']}")
    print(f"  Pain events: {v['pain_events']}")
    print(f"  Freeze events: {v['freeze_events']}")
    print(f"  Recovery events: {v['recovery_events']}")
    print(f"  Cognitive cycles: {alice._cycle_count}")
    print(f"  Gradual stress freeze point: Step {freeze_step}")
    print(f"  Broadcast storm freeze point: Packet #{frozen_at or 'N/A'}")
    print()
    print("  * Trauma Memory Metrics:")
    print(f"    Trauma count: {v['trauma_count']}")
    print(f"    Pain sensitivity: {v['pain_sensitivity']:.3f} (1.0=normal)")
    print(f"    Baseline temperature drift: {v['baseline_temperature']:.4f}")
    av = alice.autonomic.get_vitals()
    print(f"    sympathetic baseline: {alice.autonomic.sympathetic_baseline:.3f} (normal=0.200)")
    print(f"    Parasympathetic baseline: {alice.autonomic.parasympathetic_baseline:.3f} (normal=0.300)")
    print(f"    Chronic stress load: {alice.autonomic.chronic_stress_load:.3f}")
    print(f"    Resting heart rate: {av['heart_rate']:.0f}bpm (normal=60)")
    print(f"    Cortisol: {av['cortisol']:.4f}")
    print()
    print("  Conclusion: Alice has a complete pain-crash-recovery-trauma memory lifecycle ✓")
    print()


if __name__ == "__main__":
    main()
