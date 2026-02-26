# -*- coding: utf-8 -*-
"""
Experiment: Eye × Oscilloscope
Experiment: Eye × Oscilloscope

Three experiments:
  1. Different images → different visual electrical signals (frequency mapping)
  2. Eye → FusionBrain → Oscilloscope (end-to-end channel observation)
  3. Standing wave visualization (impedance mismatch → reflection → standing wave)

'The lens is a Fourier transformer — this is physics, not computation.'
"""

import numpy as np
from alice.body.eye import AliceEye
from alice.alice_brain import AliceBrain
from alice.core.signal import BrainWaveBand
from alice.core.protocol import Modality, Priority


def banner(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ================================================================
# Experiment 1: Different images → different frequencies
# ================================================================
def experiment_1_visual_frequency_mapping():
    banner("Experiment 1: Visual Frequency Mapping (Spatial → Brainwave)")
    print("  Lens FFT maps spatial frequency to brainwave frequency")
    print("  Low spatial frequency → δ/θ (contours)")
    print("  High spatial frequency → β/γ (details)\n")

    eye = AliceEye(retina_resolution=256)

    images = {
        "Uniform brightness (DC)": np.full(256, 0.5),
        "Low-freq sine (f=2)": 0.5 + 0.5 * np.sin(2*np.pi*2*np.linspace(0,1,256)),
        "Mid-freq sine (f=20)": 0.5 + 0.5 * np.sin(2*np.pi*20*np.linspace(0,1,256)),
        "High-freq sine (f=80)": 0.5 + 0.5 * np.sin(2*np.pi*80*np.linspace(0,1,256)),
        "Step edge": np.concatenate([np.zeros(128), np.ones(128)]),
        "Random noise": np.random.RandomState(42).rand(256),
    }

    results = []
    for name, img in images.items():
        sig = eye.see(img)
        band = BrainWaveBand.from_frequency(sig.frequency)
        results.append((name, sig.frequency, sig.amplitude, band.value))
        print(f"  {name:20s} → {sig.frequency:7.2f} Hz ({band.value:5s}) amp={sig.amplitude:.4f}")

    # verification
    assert all(r[1] > 0 for r in results), "All frequencies must > 0"
    print(f"\n  ✓ All {len(results)} image types successfully converted to electrical signals")
    print(f"  ✓ Frequency range: {min(r[1] for r in results):.2f} ~ {max(r[1] for r in results):.2f} Hz")
    print(f"  ✓ Eye statistics: {eye.total_frames} frames seen")


# ================================================================
# Experiment 2: Eye → Brain → Oscilloscope
# ================================================================
def experiment_2_eye_to_oscilloscope():
    banner("Experiment 2: Eye → Brain → Oscilloscope (End-to-End)")
    print("  Eye sees image → electrical signal → coaxial cable transmission → oscilloscope capture\n")

    eye = AliceEye()
    alice = AliceBrain()

    # View a sinusoidal grating
    img = 0.5 + 0.5 * np.sin(2*np.pi*15*np.linspace(0, 1, 256))
    visual_signal = eye.see(img)

    print(f"  [Eye] Visual nerve output:")
    print(f"    Frequency = {visual_signal.frequency:.2f} Hz")
    print(f"    Amplitude = {visual_signal.amplitude:.4f}")
    print(f"    Impedance = {visual_signal.impedance:.0f} Ω")
    print(f"    Frequency band = {visual_signal.band.value}")

    # Input to brain
    result = alice.perceive(
        visual_signal.waveform,
        Modality.VISUAL,
        Priority.NORMAL,
        context="eye_experiment"
    )

    print(f"\n  [Brain] Processing result:")
    print(f"    Cycle  = {result['cycle']}")
    print(f"    Elapsed time   = {result['elapsed_ms']:.2f} ms")

    # Get oscilloscope data
    scope = alice.get_oscilloscope_data()

    print(f"\n  [Oscilloscope] Capture:")
    print(f"    Input waveform = {len(scope['input_waveform'])} points")
    print(f"    Input frequency = {scope['input_freq']:.2f} Hz")
    print(f"    Input frequency band = {scope['input_band']}")

    channels = scope.get("channels", {})
    print(f"    Channel count = {len(channels)}")
    for ch_name, ch_data in channels.items():
        wave_len = len(ch_data.get("waveform", []))
        gamma = ch_data.get("gamma", 0)
        refl  = ch_data.get("reflected_ratio", 0)
        match = ch_data.get("matched", True)
        status = "✓ matched" if match else "✗ MISMATCH"
        print(f"      {ch_name:25s}: {wave_len} pts | Γ={gamma:+.4f} | refl={refl*100:.1f}% | {status}")

    perc = scope.get("perception", {})
    if perc:
        print(f"\n  [Perception] Resonance result:")
        print(f"    Left brain resonance = {perc.get('left_resonance', 0):.4f} ({perc.get('left_band', '?')})")
        print(f"    Right brain resonance = {perc.get('right_resonance', 0):.4f} ({perc.get('right_band', '?')})")
        print(f"    Attention band = {perc.get('attention_band', '?')} (strength={perc.get('attention_strength', 0):.4f})")
        print(f"    Concept = {perc.get('concept', '?')}")

    # verification
    assert len(scope["input_waveform"]) > 0
    assert len(channels) == 4
    assert scope["perception"] is not None
    print(f"\n  ✓ End-to-end pipeline complete: Eye → Brain → Oscilloscope")


# ================================================================
# Experiment 3: Standing wave visualization
# ================================================================
def experiment_3_standing_waves():
    banner("Experiment 3: Standing Wave Visualization")
    print("  Impedance mismatch \u2192 reflection \u2192 incident wave + reflected wave = standing wave")
    print("  standing[i] = wave[i] + Γ × wave[N-1-i]\n")

    eye = AliceEye()
    alice = AliceBrain()

    # strong stimulus
    img = np.random.RandomState(99).rand(256)
    sig = eye.see(img)
    alice.perceive(sig.waveform, Modality.TACTILE, Priority.HIGH)

    scope = alice.get_oscilloscope_data()
    channels = scope.get("channels", {})

    for ch_name, ch_data in channels.items():
        wave = ch_data.get("waveform", [])
        gamma = ch_data.get("gamma", 0)

        if len(wave) < 4:
            continue

        # Compute standing wave
        n = len(wave)
        standing = [wave[i] + gamma * wave[n-1-i] for i in range(n)]

        # Standing wave nodes and antinodes
        rms_incident = np.sqrt(np.mean(np.array(wave)**2))
        rms_standing = np.sqrt(np.mean(np.array(standing)**2))
        vswr = (1 + abs(gamma)) / (1 - abs(gamma)) if abs(gamma) < 1 else float('inf')

        print(f"  {ch_name:25s}:")
        print(f"    Γ = {gamma:+.4f}")
        print(f"    VSWR = {vswr:.2f}")
        print(f"    Incident RMS = {rms_incident:.4f}")
        print(f"    Standing wave RMS = {rms_standing:.4f}")
        print(f"    Standing wave ratio = {rms_standing/max(rms_incident,1e-9):.2f}x")

    # Check which channels have impedance mismatch
    mismatched = [(n, ch) for n, ch in channels.items() if abs(ch.get("gamma", 0)) > 0.01]
    if mismatched:
        best_name, best_ch = max(mismatched, key=lambda x: abs(x[1]["gamma"]))
        print(f"\n  * Largest impedance mismatch channel: {best_name}")
        print(f"    Γ = {best_ch['gamma']:+.4f}")
        print(f"    Reflected energy → heating → pain — this is the signal physics cost")
    else:
        print(f"\n  * All channels well matched in impedance (Γ ≈ 0)")
        print(f"    sensory→prefrontal channel Γ = {channels.get('sensory→prefrontal', {}).get('gamma', 'N/A')}")

    print(f"\n  ✓ Standing wave computation completed — frontend oscilloscope will render CH2 from this")


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  Alice Smart System — Eye × Oscilloscope Experiment        ║")
    print("║  'The lens is a Fourier transformer — physics, not computation.' ║")
    print("╚═══════════════════════════════════════════════════════════╝")

    experiment_1_visual_frequency_mapping()
    experiment_2_eye_to_oscilloscope()
    experiment_3_standing_waves()

    print(f"\n{'='*60}")
    print("  All experiments completed ✓")
    print(f"{'='*60}")
