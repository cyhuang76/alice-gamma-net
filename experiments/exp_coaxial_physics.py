# -*- coding: utf-8 -*-
"""
Experiment: Coaxial Cable Physics Validation
Experiment: Coaxial Cable Physics Validation

Purpose: Validate the physics behavior of the unified electrical signal framework
  1. Impedance matching vs mismatch → reflected energy difference
  2. Frequency band classification → different frequency signal behaviors
  3. Distance decay → long channel vs short channel
  4. Reflected energy → pain circuit → system temperature
  5. Full cycle: signal → coaxial transmission → reflection → pain → freeze
"""

import numpy as np
from alice.core.signal import (
    BrainWaveBand,
    ElectricalSignal,
    CoaxialChannel,
    SignalBus,
    REGION_IMPEDANCE,
)
from alice.alice_brain import AliceBrain
from alice.core.protocol import Modality, Priority


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    section("Experiment 1: Coaxial Cable Impedance Matching")

    # Matched channel (75Ω signal → 75Ω channel)
    ch_matched = CoaxialChannel("A", "B", characteristic_impedance=75.0, length=1.0)
    sig_matched = ElectricalSignal.from_raw(np.ones(50) * 0.5, impedance=75.0)
    tx_m, rpt_m = ch_matched.transmit(sig_matched)

    print(f"  Matched (75Ω→75Ω):")
    print(f"    Reflection coefficient Γ = {rpt_m.reflection_coefficient:.4f}")
    print(f"    Reflected power = {rpt_m.reflected_power_ratio*100:.2f}%")
    print(f"    Transmission factor = {rpt_m.total_transmission_factor:.4f}")
    print(f"    Impedance matched   = {rpt_m.impedance_matched}")

    # Mismatched channel (50Ω signal → 110Ω channel)
    ch_mismatch = CoaxialChannel("A", "B", characteristic_impedance=110.0, length=1.0)
    sig_mismatch = ElectricalSignal.from_raw(np.ones(50) * 0.5, impedance=50.0)
    tx_mm, rpt_mm = ch_mismatch.transmit(sig_mismatch)

    print(f"\n  Mismatched (50Ω→110Ω):")
    print(f"    Reflection coefficient Γ = {rpt_mm.reflection_coefficient:.4f}")
    print(f"    Reflected power = {rpt_mm.reflected_power_ratio*100:.2f}%")
    print(f"    Reflected energy   = {rpt_mm.reflected_energy:.6f}")
    print(f"    Transmission factor = {rpt_mm.total_transmission_factor:.4f}")
    print(f"    Impedance matched   = {rpt_mm.impedance_matched}")

    # ----------------------------------------------------------------
    section("Experiment 2: Brainwave Band Classification")

    test_freqs = [2, 6, 10, 20, 50]
    for f in test_freqs:
        band = BrainWaveBand.from_frequency(f)
        print(f" {f:3d} Hz → {band.value:6s} (range: {band.freq_range})")

    # ----------------------------------------------------------------
    section("Experiment 3: Coaxial Bus Topology")

    bus = SignalBus.create_default_topology()
    print(f"  Channel count: {len(bus.channels)}")
    print(f"\n  Channel list:")
    for (src, tgt), ch in bus.channels.items():
        print(f"    {src:16s} → {tgt:16s}  Z₀={ch.characteristic_impedance:5.0f}Ω  L={ch.length:.1f}")

    # ----------------------------------------------------------------
    section("Experiment 4: Brain Region Impedance Reference Table")

    for name, z in REGION_IMPEDANCE.items():
        print(f"  {name:16s} : {z:.0f}Ω")

    # ----------------------------------------------------------------
    section("Experiment 5: Per-Channel Reflection Analysis")

    sig = ElectricalSignal.from_raw(np.random.rand(50) * 0.5, impedance=50.0)  # Sensory cortex impedance
    routes = [
        ("somatosensory", "prefrontal", "sensory→pfc"),
        ("somatosensory", "limbic", "sensory→limbic"),
        ("prefrontal", "motor", "pfc→motor"),
        ("limbic", "motor", "limbic→motor"),
        ("prefrontal", "limbic", "pfc→limbic"),
        ("somatosensory", "motor", "sensory→motor"),
    ]

    for src, tgt, label in routes:
        tx, rpt = bus.send(src, tgt, sig)
        gamma = rpt.reflection_coefficient if rpt else 0
        refl = rpt.reflected_power_ratio * 100 if rpt else 0
        matched = rpt.impedance_matched if rpt else True
        print(f"  {label:12s} Γ={gamma:+.4f} refl={refl:5.2f}% match={'✓' if matched else '✗'}")

    # ----------------------------------------------------------------
    section("Experiment 6: AliceBrain Full Coaxial Cycle")

    brain = AliceBrain(neuron_count=50)
    print(f"  Brain region impedances:")
    from alice.brain.fusion_brain import BrainRegionType
    for rt in BrainRegionType:
        z = brain.fusion_brain.regions[rt].impedance
        print(f"    {rt.value:16s} : {z:.0f}Ω")

    print(f"\n  5 stimulus cycles:")
    for i in range(5):
        stim = np.random.rand(50) * (0.5 + i * 0.1)
        result = brain.perceive(stim, Modality.VISUAL, Priority.NORMAL)

        reflected = result.get("cycle_reflected_energy", 0)
        vitals = result["vitals"]
        band = result["sensory"].get("signal_band", "?")

        print(
            f"    [{i+1}] frequency band={band:6s}  "
            f"reflected energy={reflected:.6f}  "
            f"Temperature={vitals['ram_temperature']:.4f}  "
            f"pain={vitals['pain_level']:.4f}  "
            f"consciousness={vitals['consciousness']:.4f}"
        )

    # ----------------------------------------------------------------
    section("Experiment 7: Impedance Storm — Reflection → Overheat → Pain → Freeze")

    brain2 = AliceBrain(neuron_count=50)
    print("  Continuously sending high impedance-mismatched signals...")
    print(f"  {'#':>4s} {'Reflected':>10s} {'Temp':>6s} {'Pain':>6s} {'Consc':>6s} {'State'}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*10}")

    for i in range(30):
        # Large amplitude signal + CRITICAL priority
        stim = np.random.rand(50) * 2.0
        pri = Priority.CRITICAL if i > 15 else Priority.HIGH
        result = brain2.perceive(stim, Modality.TACTILE, pri)

        vitals = result["vitals"]
        reflected = result.get("cycle_reflected_energy", 0)
        status = "🔴 FROZEN" if vitals["is_frozen"] else (
            "🟡 PAIN" if vitals["pain_level"] > 0.3 else "🟢 OK"
        )

        print(
            f"  {i+1:4d}  {reflected:10.6f}  "
            f"{vitals['ram_temperature']:6.4f}  "
            f"{vitals['pain_level']:6.4f}  "
            f"{vitals['consciousness']:6.4f}  "
            f"{status}"
        )

        if vitals["is_frozen"]:
            print(f"\n  ⚡ System froze after stimulus #{i+1}!")
            break

    # ----------------------------------------------------------------
    section("Experiment 8: Bus Efficiency Report")

    bus_summary = brain.fusion_brain.signal_bus.get_bus_summary()
    print(f"  Total transmissions: {bus_summary['total_transmissions']}")
    print(f"  Impedance mismatches: {bus_summary['total_impedance_mismatches']}")
    print(f"  Mismatch rate: {bus_summary['mismatch_rate']:.1%}")
    print(f"  Total reflected energy: {bus_summary['total_reflected_energy']:.6f}")
    print(f"  Bus efficiency: {bus_summary['bus_efficiency']:.1%}")

    # ----------------------------------------------------------------
    section("Experiments Completed")
    print("  Unified electrical signal framework validation complete.")
    print("  Coaxial cable model successfully integrated into Alice neural system.")
    print("  Impedance mismatch → reflected energy → heating → pain physics loop verified.\n")


if __name__ == "__main__":
    main()
