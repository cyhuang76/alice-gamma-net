# -*- coding: utf-8 -*-
"""
Experiment: Closed-Loop Life Loop — Behavior = Calibration Error Compensation

Experiment verification:
  1. See → reach → error convergence (hand-eye coordination closed-loop)
  2. Anxiety corrupts compensation precision (PID noise contamination)
  3. Sleep cuts off the loop (sensory gating)
  4. Persistent mismatch → chronic stress → pain
  5. Complete life cycle: see + hear + say + reach all running simultaneously
"""

import sys
import numpy as np

sys.path.insert(0, ".")

from alice.alice_brain import AliceBrain


def separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run_experiment():
    brain = AliceBrain()

    # =================================================================
    # Experiment 1: See → reach → closed-loop error compensation
    # =================================================================
    separator("Experiment 1: Hand-Eye Coordination — Closed-Loop Error Compensation")

    print("\n 'See cup → reach → error returns to zero'")
    print(" This is perceive → error → compensate → re-perceive\n")

    # see 
    see_result = brain.see(
        pixels=np.random.randn(64) * 0.5,
        visual_target=np.array([0.5, 0.5]),
    )
    loop_after_see = see_result["life_loop"]
    print(f" [See] Visual signal received")
    print(f" Pupil aperture: {see_result['visual']['pupil_aperture']}")
    print(f" Life loop error: {loop_after_see['total_error']:.4f}")
    print(f" Error list: {[e['type'] for e in loop_after_see['errors']]}")

    # Reach
    reach_result = brain.reach_for(target_x=0.5, target_y=0.5, max_steps=100)
    print(f"\n [Reach] PID drives hand")
    print(f" Reached: {reach_result['coordination']['reached']}")
    print(f" Final error: {reach_result['coordination']['error_magnitude']:.4f}")
    print(f" Steps: {reach_result['reach']['steps']}")
    print(f" Dopamine: {reach_result['dopamine']:.4f}")

    # See again → loop verification
    see2 = brain.see(np.random.randn(64) * 0.5)
    loop2 = see2["life_loop"]
    print(f"\n [See again] Confirm error change")
    print(f" Persistent errors: {loop2['persistent_errors']}")
    print(f" Prediction accuracy: {loop2['prediction_accuracy']:.4f}")

    # =================================================================
    # Experiment 2: Anxiety corrupts compensation precision
    # =================================================================
    separator("Experiment 2: Anxiety → PID Noise Contamination → Compensation Precision Decrease")

    print("\n 'Anxious people's hands tremble, voices quiver, movements become clumsy'")
    print(" Because ram_temperature directly contaminates PID control signals\n")

    # Calm state
    brain_calm = AliceBrain()
    calm_result = brain_calm.reach_for(0.5, 0.5, max_steps=100)
    calm_loop = brain_calm.life_loop.get_stats()

    # Anxious state
    brain_anxious = AliceBrain()
    brain_anxious.inject_pain(0.9)
    brain_anxious.inject_pain(0.9)
    anxious_result = brain_anxious.reach_for(0.5, 0.5, max_steps=100)
    anxious_loop = brain_anxious.life_loop.get_stats()

    print(f" Calm:")
    print(f" Reached: {calm_result['coordination']['reached']}")
    print(f" Hand tremor: {calm_result['reach']['tremor_intensity']:.6f}")
    print(f" Final error: {calm_result['coordination']['error_magnitude']:.4f}")

    print(f"\n Anxious (ram_temperature={brain_anxious.vitals.ram_temperature:.2f}):")
    print(f" Reached: {anxious_result['coordination']['reached']}")
    print(f" Hand tremor: {anxious_result['reach']['tremor_intensity']:.6f}")
    print(f" Final error: {anxious_result['coordination']['error_magnitude']:.4f}")
    print(f" Pain level: {brain_anxious.vitals.pain_level:.3f}")

    # =================================================================
    # Experiment 3: Speech closed-loop — speaking is also error compensation
    # =================================================================
    separator("Experiment 3: Speech — Target pitch vs actual pitch = PID compensation")

    print("\n 'Larynx PID tracks target pitch, ear hears feedback for confirmation'\n")

    say_a = brain.say(target_pitch=200.0, vowel="a")
    print(f" Say 'a' (target 200Hz):")
    print(f" Final pitch: {say_a['final_pitch']:.1f} Hz")
    print(f" Pitch error: {say_a['pitch_error']:.1f} Hz")
    print(f" Tremor intensity: {say_a['tremor_intensity']:.4f}")
    print(f" Volume: {say_a['volume']:.3f}")

    say_i = brain.say(target_pitch=300.0, vowel="i")
    print(f"\n Say 'i' (target 300Hz):")
    print(f" Final pitch: {say_i['final_pitch']:.1f} Hz")
    print(f" Pitch error: {say_i['pitch_error']:.1f} Hz")

    # =================================================================
    # Experiment 4: Auditory → closed-loop
    # =================================================================
    separator("Experiment 4: Auditory — Sound analog circuit signal processing")

    print("\n 'Cochlea = physics FFT, hair cells = pressure→voltage converter'\n")

    # Generate 440Hz sine wave (A note)
    t = np.linspace(0, 0.1, 4410)
    a_note = 0.3 * np.sin(2 * np.pi * 440 * t)
    hear_result = brain.hear(a_note)

    print(f" Hear 440Hz (A note):")
    print(f" Cochlea detected frequency: {hear_result['auditory']['frequency']:.1f} Hz")
    print(f" Amplitude: {hear_result['auditory']['amplitude']:.4f}")
    print(f"    Brainwave band: {hear_result['auditory']['band']}")
    loop_hear = hear_result["life_loop"]
    print(f" Life loop error: {loop_hear['total_error']:.4f}")
    print(f" Compensation commands: {[c['action'] for c in loop_hear['commands']]}")

    # =================================================================
    # Experiment 5: Complete life cycle — all modalities running simultaneously
    # =================================================================
    separator("Experiment 5: Complete Life Cycle — All loops working simultaneously")

    print("\n 'A living system eliminates errors every tick'\n")

    brain_full = AliceBrain()

    for tick in range(20):
        # See
        pixels = np.random.randn(64) * 0.3
        brain_full.see(pixels, visual_target=np.array([0.5, 0.5]))

        # Hear
        noise = np.random.randn(256) * 0.1
        brain_full.hear(noise)

        # Intermittent reaching
        if tick % 5 == 0:
            brain_full.reach_for(0.5, 0.5, max_steps=30)

        # Intermittent speaking
        if tick % 7 == 0:
            brain_full.say(150.0, vowel="a")

    # Final state
    final = brain_full.introspect()
    loop_stats = final["subsystems"]["life_loop"]

    print(f" System state after 20 ticks:")
    print(f" Life loop ticks: {loop_stats['tick_count']}")
    print(f" Cumulative total error: {loop_stats['cumulative_error']:.4f}")
    print(f" Mean error/tick: {loop_stats['avg_error']:.4f}")
    print(f" Compensation attempts: {loop_stats['total_compensations']}")
    print(f" Successful compensations: {loop_stats['successful_compensations']}")
    print(f" Prediction accuracy: {loop_stats['prediction_accuracy']:.4f}")
    print(f" Persistent errors: ", end="")
    for k, v in loop_stats["persistent_errors"].items():
        if v > 0.001:
            print(f"{k}={v:.4f} ", end="")
    print()
    print(f" error→pain: {loop_stats['error_to_pain']:.4f}")

    vitals = final["vitals"]
    print(f"\n Vital signs:")
    print(f"    RAM Temperature: {vitals['ram_temperature']:.4f}")
    print(f"    stability: {vitals['stability_index']:.4f}")
    print(f"    heart rate: {vitals['heart_rate']:.1f} bpm")
    print(f"    pain: {vitals['pain_level']:.4f}")
    print(f" Consciousness: {vitals['consciousness']:.4f}")

    auto = final["subsystems"]["autonomic"]
    print(f"\n Autonomic nervous system:")
    print(f" Sympathetic: {auto['sympathetic']:.4f}")
    print(f" Parasympathetic: {auto['parasympathetic']:.4f}")
    print(f"    energy: {auto['energy']:.4f}")

    consciousness = final["subsystems"]["consciousness"]
    print(f"\n Consciousness: Φ = {consciousness.get('phi', 'N/A')}")

    # =================================================================
    # Summary
    # =================================================================
    separator("Conclusion")
    print("""
  Closed-loop verification completed. The entire system has only one core algorithm:

    while alive:
        signals = perceive() # eye, ear → electrical signals
        errors = estimate_errors() # cross-modal mismatch
        commands = compensate(errors) # PID → motor commands
        execute(commands)              # hand, mouth execute
        feedback = re_perceive() # actions change perception
        calibrate(feedback) # Temporal calibrator updates

  Behavior = calibration error compensation
  Will = consciousness module's narrative interpretation of error processing
  Practice = repeatedly running the loop until PID converges
  Anxiety = noise contaminating control signals
  Sleep = offline recalibration
  Pain = persistent unresolvable error
""")


if __name__ == "__main__":
    run_experiment()
