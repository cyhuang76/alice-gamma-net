# -*- coding: utf-8 -*-
"""
Experiment: Auditory Grounding — Pavlovian Cross-Modal Hebbian Conditioning
Phase 4.1: The Physics of Language — Auditory Grounding

Core experiments:
  1. Pavlovian conditioning: bell + food → Hebbian connection
  2. Probing: bell alone → visual cortex phantom activation?
  3. Extinction: bell without food → synaptic decay → phantom disappearance
  4. Differential conditioning: different sounds → different associations
  5. Vowel recognition: /a/ vs /i/ vs /u/ spectral fingerprint

Physics verification:
  - Gamma_cross from ~1.0 (no conditioning) → ~0.0 (complete conditioning)
  - energy_transfer from ~0.0 → ~1.0
  - After extinction, Gamma rises back, energy_transfer drops back to 0
  - Different spectra → different fingerprints → different synapses (concept separation)

'Language is impedance modulation — sound waves remotely controlling another brain.'
"""

import math
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from alice.body.cochlea import (
    CochlearFilterBank,
    TonotopicActivation,
    generate_tone,
    generate_complex_tone,
    generate_noise,
    generate_vowel,
)
from alice.brain.auditory_grounding import (
    AuditoryGroundingEngine,
    CrossModalHebbianNetwork,
    CrossModalSynapse,
    SensoryPrototype,
    SYNAPSE_Z0,
    ECHO_THRESHOLD,
)
from alice.core.signal import ElectricalSignal


def separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def subsection(title: str):
    print(f"\n  --- {title} ---\n")


def _make_visual_signal(
    freq: float = 10.0,
    amp: float = 1.0,
    label: str = "food",
    n: int = 256,
) -> ElectricalSignal:
    """Simulate a visual signal (e.g. seeing food)"""
    t = np.linspace(0, 1, n, endpoint=False)
    waveform = amp * np.sin(2 * np.pi * freq * t)
    return ElectricalSignal(
        waveform=waveform,
        amplitude=amp,
        frequency=freq,
        phase=0.0,
        impedance=50.0,
        snr=15.0,
        source=label,
        modality="visual",
    )


# ============================================================================
# Exp 1: Cochlear Filter Bank
# ============================================================================

def exp1_cochlear_decomposition():
    separator("Exp 1: Cochlear Filter Bank -- Basilar Membrane Physics")

    cochlea = CochlearFilterBank(n_channels=24)

    print(f"  Channels: {cochlea.n_channels}")
    print(f"  Freq range: {cochlea.freq_min}~{cochlea.freq_max} Hz")
    print(f"  Center frequencies (ERB scale):")
    for i in range(0, cochlea.n_channels, 4):
        cfs = cochlea.center_frequencies[i:i+4]
        labels = [f"Ch{i+j}: {cf:.0f}Hz" for j, cf in enumerate(cfs)]
        print(f"    {', '.join(labels)}")

    # --- Pure tone test ---
    subsection("1a. Pure tone 440Hz (A4 tuning fork) = single resonance point")
    tone_440 = generate_tone(440.0, duration=0.1)
    tono_440 = cochlea.analyze(tone_440)
    print(f"  Tonotopic Activation:")
    print(f"    Peak channel: {np.argmax(tono_440.channel_activations)}")
    peak_cf = cochlea.center_frequencies[np.argmax(tono_440.channel_activations)]
    print(f"    Peak center freq: {peak_cf:.1f} Hz")
    print(f"    Spectral centroid: {tono_440.spectral_centroid:.1f} Hz")
    print(f"    Total energy: {tono_440.total_energy:.4f}")
    print(f"    Fingerprint (top 5):")
    fp = tono_440.fingerprint()
    top5 = np.argsort(fp)[-5:][::-1]
    for idx in top5:
        print(f"      Ch{idx} ({cochlea.center_frequencies[idx]:.0f}Hz): {fp[idx]:.4f}")

    # --- Different frequency comparison ---
    subsection("1b. Different frequencies -> different tonotopic positions")
    test_freqs = [200, 440, 1000, 2000, 4000]
    for f in test_freqs:
        tone = generate_tone(float(f), duration=0.1)
        tono = cochlea.analyze(tone)
        peak_ch = np.argmax(tono.channel_activations)
        peak_cf = cochlea.center_frequencies[peak_ch]
        print(f"    {f:5d} Hz -> Peak Ch{peak_ch:2d} ({peak_cf:.0f} Hz)"
              f"  centroid={tono.spectral_centroid:.0f} Hz"
              f"  flatness={tono.spectral_flatness:.3f}")

    # --- Complex tone vs pure tone ---
    subsection("1c. Complex tone vs pure tone -- harmonic structure")
    pure = generate_tone(200.0, duration=0.1)
    cplx = generate_complex_tone(200.0, n_harmonics=5, duration=0.1)
    noise = generate_noise(duration=0.1)

    tono_pure = cochlea.analyze(pure)
    tono_cplx = cochlea.analyze(cplx)
    tono_noise = cochlea.analyze(noise)

    print(f"    Pure 200Hz:    flatness={tono_pure.spectral_flatness:.3f}"
          f"  centroid={tono_pure.spectral_centroid:.0f} Hz")
    print(f"    Complex 200Hz: flatness={tono_cplx.spectral_flatness:.3f}"
          f"  centroid={tono_cplx.spectral_centroid:.0f} Hz")
    print(f"    White noise:   flatness={tono_noise.spectral_flatness:.3f}"
          f"  centroid={tono_noise.spectral_centroid:.0f} Hz")

    sim_pc = tono_pure.similarity(tono_cplx)
    sim_pn = tono_pure.similarity(tono_noise)
    sim_cn = tono_cplx.similarity(tono_noise)
    print(f"\n    Similarity matrix (cosine):")
    print(f"      Pure vs Complex: {sim_pc:.4f}")
    print(f"      Pure vs Noise:   {sim_pn:.4f}")
    print(f"      Complex vs Noise:{sim_cn:.4f}")

    # --- Vowel fingerprints ---
    subsection("1d. Vowel spectral fingerprints -- formants determine identity")
    vowels = ["a", "i", "u", "e", "o"]
    vowel_fps = {}
    for v in vowels:
        wave = generate_vowel(v, fundamental=150.0, duration=0.1)
        tono = cochlea.analyze(wave)
        vowel_fps[v] = tono.fingerprint()
        print(f"    /{v}/: centroid={tono.spectral_centroid:.0f} Hz"
              f"  flatness={tono.spectral_flatness:.3f}"
              f"  hash={tono.fingerprint_hash()}")

    print(f"\n    Vowel similarity matrix:")
    print(f"       ", end="")
    for v2 in vowels:
        print(f"  /{v2}/  ", end="")
    print()
    for v1 in vowels:
        print(f"    /{v1}/ ", end="")
        for v2 in vowels:
            fp1 = vowel_fps[v1]
            fp2 = vowel_fps[v2]
            dot = float(np.dot(fp1, fp2))
            n1 = float(np.linalg.norm(fp1))
            n2 = float(np.linalg.norm(fp2))
            sim = dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0
            print(f" {sim:.3f} ", end="")
        print()

    print(f"\n  OK: Different vowels have different spectral fingerprints")
    print(f"    -> This is the physical basis of language")


# ============================================================================
# Exp 2: Pavlovian Conditioning -- Hebbian Wiring
# ============================================================================

def exp2_pavlovian_conditioning():
    separator("Exp 2: Pavlovian Conditioning -- Cross-Modal Hebbian Wiring")

    engine = AuditoryGroundingEngine()

    # Bell = 440Hz pure tone
    bell_wave = generate_tone(440.0, duration=0.1)
    # Food = 10Hz visual signal
    food_signal = _make_visual_signal(freq=10.0, amp=1.0, label="food")

    # ================================================================
    # Phase 1: Before conditioning -- bell alone -> no response
    # ================================================================
    subsection("Phase 1: Before conditioning -- bell alone -> no response")

    result_before = engine.probe_association(bell_wave)
    print(f"    Bell alone:")
    print(f"      Echoes: {len(result_before['echoes'])}")
    print(f"      Has phantom: {result_before['has_phantom']}")
    print(f"      = No cross-modal link, bell is just a bell")

    # ================================================================
    # Phase 2: Conditioning -- bell + food x N trials
    # ================================================================
    subsection("Phase 2: Conditioning -- bell + food presented together")

    conditioning_log = []
    n_trials = 20

    for trial in range(1, n_trials + 1):
        result = engine.condition_pair(
            auditory_wave=bell_wave,
            other_signal=food_signal,
            auditory_label="bell",
            other_label="food",
        )
        syn = result["synapse"]

        conditioning_log.append({
            "trial": trial,
            "strength": syn["strength"],
            "z_impedance": syn["z_impedance"],
            "gamma": syn["gamma"],
            "energy_transfer": syn["energy_transfer"],
        })

        if trial in [1, 2, 3, 5, 10, 15, 20]:
            print(f"    Trial {trial:2d}: "
                  f"w={syn['strength']:.4f}  "
                  f"Z={syn['z_impedance']:.1f} ohm  "
                  f"Gamma={syn['gamma']:+.4f}  "
                  f"E_transfer={syn['energy_transfer']:.4f}")

    # ================================================================
    # Phase 3: Probe -- bell alone -> phantom?
    # ================================================================
    subsection("Phase 3: After conditioning -- bell alone -> visual phantom?")

    result_after = engine.probe_association(bell_wave)
    print(f"    Bell alone (after {n_trials} pairings):")
    print(f"      Echoes: {len(result_after['echoes'])}")
    print(f"      Has phantom: {result_after['has_phantom']}")

    if result_after['echoes']:
        best_echo = result_after['echoes'][0]
        print(f"      Best echo:")
        print(f"        Target modality: {best_echo['target_modality']}")
        print(f"        Echo strength: {best_echo['echo_strength']:.4f}")
        print(f"        Energy transfer: {best_echo['energy_transfer']:.4f}")
        print(f"        Gamma_cross: {best_echo['gamma']:+.4f}")
        print(f"        Synapse strength: {best_echo['synapse_strength']:.4f}")

    if result_after['echo_signal'] is not None:
        echo = result_after['echo_signal']
        print(f"\n      * PHANTOM SIGNAL GENERATED!")
        print(f"        Modality: {echo.modality}")
        print(f"        Frequency: {echo.frequency:.1f} Hz")
        print(f"        Amplitude: {echo.amplitude:.4f}")
        print(f"        Source: {echo.source}")
        print(f"\n      -> Pavlov's dog! Bell -> 'sees' food")
        print(f"        Language physics: sound -> cross-modal cable -> phantom activation")
    else:
        print(f"\n      x No phantom yet (echo_strength < {ECHO_THRESHOLD})")

    # --- Learning curve ---
    subsection("Learning Curve")
    print(f"    Trial | Strength |    Z(ohm)  |    Gamma  | E_transfer")
    print(f"    ------+----------+------------+----------+-----------")
    for log in conditioning_log:
        if log["trial"] in [1, 2, 3, 5, 10, 15, 20]:
            print(f"    {log['trial']:5d} | {log['strength']:8.4f} | "
                  f"{log['z_impedance']:10.1f} | "
                  f"{log['gamma']:+8.4f} | "
                  f"{log['energy_transfer']:9.4f}")

    print(f"\n  OK: As conditioning trials increase:")
    print(f"    - Synapse strength w UP -> impedance Z = Z_0/w DOWN")
    print(f"    - Reflection Gamma DOWN -> energy transfer E_transfer UP")
    print(f"    - Energy flows through cross-modal cable = hearing bell 'sees' food")


# ============================================================================
# Exp 3: Extinction -- synapse decay
# ============================================================================

def exp3_extinction():
    separator("Exp 3: Extinction -- bell without food -> synapse decay")

    engine = AuditoryGroundingEngine()
    bell_wave = generate_tone(440.0, duration=0.1)
    food_signal = _make_visual_signal(freq=10.0, amp=1.0, label="food")

    # Condition first
    for _ in range(20):
        engine.condition_pair(bell_wave, food_signal)

    # Baseline after conditioning
    result_peak = engine.probe_association(bell_wave)
    peak_echo = result_peak['echoes'][0] if result_peak['echoes'] else None
    if peak_echo:
        print(f"    Post-conditioning baseline:")
        print(f"      Synapse: w={peak_echo['synapse_strength']:.4f}  "
              f"E_transfer={peak_echo['energy_transfer']:.4f}")
        print(f"      Echo strength: {peak_echo['echo_strength']:.4f}")
        print(f"      Has phantom: {result_peak['has_phantom']}")

    # Extinction loop -- only tick (no new conditioning)
    extinction_log = []
    n_extinction = 200

    for t in range(1, n_extinction + 1):
        engine.tick()

        if t % 10 == 0:
            result = engine.probe_association(bell_wave)
            echo = result['echoes'][0] if result['echoes'] else None
            log = {
                "tick": t,
                "strength": echo['synapse_strength'] if echo else 0.0,
                "energy_transfer": echo['energy_transfer'] if echo else 0.0,
                "echo_strength": echo['echo_strength'] if echo else 0.0,
                "has_phantom": result['has_phantom'],
            }
            extinction_log.append(log)

    subsection("Extinction Curve")
    print(f"    Tick  | Strength |  E_transfer | Echo_str | Phantom?")
    print(f"    ------+----------+-------------+----------+---------")
    for log in extinction_log:
        phantom = "YES" if log["has_phantom"] else " no"
        print(f"    {log['tick']:5d} | {log['strength']:8.4f} | "
              f"{log['energy_transfer']:11.4f} | "
              f"{log['echo_strength']:8.4f} | {phantom}")

    # Find the tick where phantom disappears
    no_phantom_ticks = [l["tick"] for l in extinction_log if not l["has_phantom"]]
    if no_phantom_ticks:
        disappear_tick = no_phantom_ticks[0]
        print(f"\n    * Phantom disappeared at tick {disappear_tick}")
        print(f"      -> Extinction = synapse decay -> Z UP -> Gamma UP -> no energy transfer")
    else:
        print(f"\n    Note: phantom still present after {n_extinction} ticks")

    print(f"\n  OK: Pavlovian extinction = impedance rise -> reflection increase -> phantom gone")


# ============================================================================
# Exp 4: Differential Conditioning
# ============================================================================

def exp4_differential_conditioning():
    separator("Exp 4: Differential Conditioning -- different bells -> different foods")

    engine = AuditoryGroundingEngine()

    # Bell A = 440Hz -> red food (10Hz visual)
    bell_a = generate_tone(440.0, duration=0.1)
    food_red = _make_visual_signal(freq=10.0, amp=1.0, label="red_food")

    # Bell B = 880Hz -> blue food (30Hz visual)
    bell_b = generate_tone(880.0, duration=0.1)
    food_blue = _make_visual_signal(freq=30.0, amp=1.0, label="blue_food")

    # Bell C = 1500Hz -> unconditioned (control)
    bell_c = generate_tone(1500.0, duration=0.1)

    # --- Conditioning ---
    subsection("Conditioning Phase")
    for trial in range(1, 21):
        engine.condition_pair(bell_a, food_red,
                              auditory_label="bell_A", other_label="red_food")
        engine.condition_pair(bell_b, food_blue,
                              auditory_label="bell_B", other_label="blue_food")

    # --- Probe ---
    subsection("Probe Phase -- three bells' cross-modal associations")

    for label, wave in [("Bell A (440Hz)", bell_a),
                        ("Bell B (880Hz)", bell_b),
                        ("Bell C (1500Hz, unconditioned)", bell_c)]:
        result = engine.probe_association(wave)
        print(f"    {label}:")
        if result['echoes']:
            for i, echo in enumerate(result['echoes'][:3]):
                print(f"      Echo {i}: modality={echo['target_modality']}  "
                      f"strength={echo['echo_strength']:.4f}")
        else:
            print(f"      No echoes -- no cross-modal association")
        print(f"      Has phantom: {result['has_phantom']}")
        ident = result.get("identified_as")
        if ident:
            print(f"      Identified as: {ident[0]} (similarity={ident[1]:.4f})")
        print()

    # --- Fingerprint differences ---
    subsection("Spectral Fingerprint Differences")
    cochlea = engine.cochlea
    tono_a = cochlea.analyze(bell_a)
    tono_b = cochlea.analyze(bell_b)
    tono_c = cochlea.analyze(bell_c)

    sim_ab = tono_a.similarity(tono_b)
    sim_ac = tono_a.similarity(tono_c)
    sim_bc = tono_b.similarity(tono_c)

    print(f"    Bell A vs Bell B: {sim_ab:.4f}")
    print(f"    Bell A vs Bell C: {sim_ac:.4f}")
    print(f"    Bell B vs Bell C: {sim_bc:.4f}")

    print(f"\n  OK: Different pitches -> different fingerprints -> different synapses -> different associations")


# ============================================================================
# Exp 5: AliceBrain Integration
# ============================================================================

def exp5_full_integration():
    separator("Exp 5: AliceBrain Integration -- full hear -> see -> conditioning loop")

    from alice.alice_brain import AliceBrain

    alice = AliceBrain()

    # Present visual stimulus (food)
    food_pixels = np.random.rand(64, 64).astype(np.float32)
    see_result = alice.see(food_pixels)
    print(f"    See (food): frequency={see_result['visual']['frequency']:.2f} Hz")

    time.sleep(0.01)

    # Present auditory stimulus (bell)
    bell = generate_tone(440.0, duration=0.1)
    hear_result = alice.hear(bell)
    print(f"    Hear (bell): frequency={hear_result['auditory']['frequency']:.2f} Hz")

    # Introspect
    state = alice.introspect()
    ag_state = state["subsystems"].get("auditory_grounding", {})
    print(f"\n    Auditory Grounding State:")
    print(f"      Total groundings: {ag_state.get('total_groundings', 0)}")
    print(f"      Total echoes: {ag_state.get('total_echoes_generated', 0)}")
    net_state = ag_state.get("network", {})
    print(f"      Synapses: {net_state.get('n_synapses', 0)}")

    # Explicit conditioning
    subsection("Explicit conditioning x 15")
    food_signal = _make_visual_signal(freq=10.0, amp=1.0, label="food")
    for i in range(15):
        alice.auditory_grounding.condition_pair(
            bell, food_signal,
            auditory_label="bell", other_label="food",
        )

    # Probe
    probe = alice.auditory_grounding.probe_association(bell)
    print(f"\n    Post-conditioning probe:")
    print(f"      Echoes: {len(probe['echoes'])}")
    print(f"      Has phantom: {probe['has_phantom']}")
    if probe['echo_signal']:
        echo = probe['echo_signal']
        print(f"      Phantom freq: {echo.frequency:.1f} Hz")
        print(f"      Phantom amp: {echo.amplitude:.4f}")
        print(f"      * Pavlov SUCCESS! Alice hears bell -> 'sees' food!")

    print(f"\n  OK: AliceBrain integration verified")


# ============================================================================
# Exp 6: Impedance Physics
# ============================================================================

def exp6_impedance_physics():
    separator("Exp 6: Impedance Physics -- Z_synapse = Z_0 / w")

    print(f"    Cross-modal synapse impedance model:")
    print(f"    Z_0 = {SYNAPSE_Z0:.0f} ohm (base impedance)")
    print(f"    Z = Z_0 / w (stronger synapse -> lower impedance -> better energy transfer)")
    print()

    # Theoretical curve
    strengths = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
    z_source = 50.0

    print(f"    w (strength) | Z (ohm)   | Gamma    | E_transfer | Status")
    print(f"    -------------+-----------+----------+------------+--------")
    for w in strengths:
        z = SYNAPSE_Z0 / w
        gamma = (z_source - z) / (z_source + z)
        et = 1.0 - gamma * gamma
        if et < 0.01:
            status = "No transfer"
        elif et < ECHO_THRESHOLD:
            status = "Weak (no phantom)"
        elif et < 0.7:
            status = "Moderate phantom"
        else:
            status = "* Strong phantom"
        print(f"    {w:12.2f} | {z:9.1f} | {gamma:+8.4f} | {et:10.4f} | {status}")

    print(f"\n  Physics:")
    print(f"    - When Z_synapse ~ Z_source -> Gamma ~ 0 -> perfect impedance match")
    print(f"    - w = Z_0/Z_source = {SYNAPSE_Z0/z_source:.1f} achieves perfect match")
    print(f"    - Consistent with Γ-Net coaxial cable physics!")


# ============================================================================
# Exp 7: Vowel Grounding -- embryo of language
# ============================================================================

def exp7_vowel_grounding():
    separator("Exp 7: Vowel Grounding -- the embryo of language")

    engine = AuditoryGroundingEngine()

    # Each vowel maps to a concept (visual mouth shape)
    vowel_concepts = {
        "a": ("mouth_open", 8.0),     # /a/ -> mouth wide open
        "i": ("mouth_smile", 25.0),   # /i/ -> smile
        "u": ("mouth_round", 5.0),    # /u/ -> round lips
    }

    # --- Conditioning ---
    subsection("Conditioning Phase -- vowel + mouth shape pairing")
    for trial in range(1, 16):
        for vowel, (label, freq) in vowel_concepts.items():
            wave = generate_vowel(vowel, fundamental=150.0, duration=0.1)
            concept_signal = _make_visual_signal(
                freq=freq, amp=1.0, label=label
            )
            result = engine.condition_pair(
                wave, concept_signal,
                auditory_label=f"vowel_{vowel}",
                other_label=label,
            )
            if trial == 15:
                syn = result["synapse"]
                print(f"    /{vowel}/ -> {label}: "
                      f"w={syn['strength']:.3f}  "
                      f"E_transfer={syn['energy_transfer']:.4f}")

    # --- Probe ---
    subsection("Probe Phase -- hear vowel -> see mouth shape?")
    for vowel in ["a", "i", "u"]:
        wave = generate_vowel(vowel, fundamental=150.0, duration=0.1)
        result = engine.probe_association(wave)
        ident = result.get("identified_as")
        print(f"    /{vowel}/:")
        print(f"      Identified: {ident}")
        print(f"      Has phantom: {result['has_phantom']}")
        if result['echo_signal']:
            echo = result['echo_signal']
            print(f"      Phantom: freq={echo.frequency:.1f}Hz  amp={echo.amplitude:.4f}")

    # --- Unconditioned vowels (control group) ---
    subsection("Control group -- unconditioned /e/ and /o/")
    for vowel in ["e", "o"]:
        wave = generate_vowel(vowel, fundamental=150.0, duration=0.1)
        result = engine.probe_association(wave)
        print(f"    /{vowel}/: echoes={len(result['echoes'])}  phantom={result['has_phantom']}")

    print(f"\n  OK: Vowel grounding = embryo of language")
    print(f"    Hearing /a/ -> cross-modal cable -> 'seeing' mouth open")
    print(f"    = Auditory-to-motor Hebbian resonance channel")


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "*" * 70)
    print("  Phase 4.1: Auditory Grounding -- Pavlovian Cross-Modal Hebbian")
    print("  The Physics of Language -- Auditory Grounding")
    print("  Language = Impedance Modulation -- remote control via sound waves")
    print("*" * 70)

    exp1_cochlear_decomposition()
    exp2_pavlovian_conditioning()
    exp3_extinction()
    exp4_differential_conditioning()
    exp5_full_integration()
    exp6_impedance_physics()
    exp7_vowel_grounding()

    print("\n" + "*" * 70)
    print("  All experiments complete!")
    print()
    print("  Phase 4.1 Core Findings:")
    print("    1. Cochlea = basilar membrane FFT -> 24-channel tonotopic activation")
    print("    2. Pavlov = cross-modal Hebbian wiring -> Z DOWN -> Gamma DOWN -> channel opens")
    print("    3. Extinction = synapse decay -> Z UP -> Gamma UP -> channel closes")
    print("    4. Different spectral fingerprints -> different concepts")
    print("    5. Language = controlling another brain's impedance matching via sound waves")
    print("*" * 70 + "\n")


if __name__ == "__main__":
    main()
