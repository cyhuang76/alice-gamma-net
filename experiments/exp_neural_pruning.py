# -*- coding: utf-8 -*-
"""
Experiment: Neural Pruning — From Random Connections to Specialized Cortex

§3.5.2 Large-scale Gamma Apoptosis

Five clinical scenarios:
  1. Normal development curve — whole-brain pruning progresses with epoch
  2. Cortical specialization divergence — different regions develop different frequency preferences
  3. Frontal delayed maturation — why motor cortex matures last
  4. Cross-modal rewiring — how congenital blindness redirects occipital lobe to auditory
  5. Gamma-squared intelligence curve — convergence of whole-brain Sigma Gamma^2 -> min
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from alice.brain.pruning import (
    NeuralPruningEngine,
    CorticalRegion,
    SynapticConnection,
    MODALITY_SIGNAL_PROFILE,
)
from alice.core.signal import BrainWaveBand


def print_header(title: str):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def ascii_bar(value: float, width: int = 30, label: str = "") -> str:
    """ASCII progress bar."""
    filled = int(value * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"  {bar} {value:.1%}  {label}"


def ascii_spark_line(data: list, width: int = 50) -> str:
    """ASCII sparkline chart."""
    if not data:
        return ""
    mn, mx = min(data), max(data)
    if mx - mn < 1e-10:
        return "▁" * min(len(data), width)

    # Downsample to 'width' points
    if len(data) > width:
        step = len(data) / width
        sampled = [data[int(i * step)] for i in range(width)]
    else:
        sampled = data

    blocks = "▁▂▃▄▅▆▇█"
    chars = []
    for v in sampled:
        idx = int((v - mn) / (mx - mn) * (len(blocks) - 1))
        chars.append(blocks[idx])
    return "".join(chars)


# ============================================================================
# Experiment 1: Normal development curve
# ============================================================================


def exp1_normal_development():
    """
    Normal development curve — pruning trajectory from birth to maturation.

    Simulates 200 epochs of whole-brain development:
    - Occipital <- visual (alpha/beta/gamma)
    - Temporal <- auditory (theta/alpha)
    - Parietal <- somatosensory (broadband)
    - Frontal <- motor (beta, 30% feedback rate)

    Expected: S-shaped pruning curve (rapid early, stable late)
    """
    print_header("Exp 1: Normal Development Curve — From Birth (Random Gamma) to Maturity (Specialized Cortex)")

    CONN = 2000
    EPOCHS = 200

    print(f"\n  Birth parameters:")
    print(f"    Connections per region: {CONN}")
    print(f"    Total initial connections: {CONN * 4}")
    print(f"    Impedance distribution: Z ~ Uniform(20 Ohm, 200 Ohm)")
    print(f"    Development period: {EPOCHS} epochs")
    print(f"    Sensory assignment: Occipital->Visual, Temporal->Auditory, Parietal->Somatosensory, Frontal->Motor")
    print(f"    Motor feedback rate: 30% (simulates 'feedback required for calibration')")

    engine = NeuralPruningEngine(connections_per_region=CONN)

    # Record key milestones
    milestones = [1, 5, 10, 20, 40, 60, 80, 100, 150, 200]
    milestone_data = []

    print(f"\n  │ {'Epoch':>5} │ {'Occipital':>8} │ {'Temporal':>8} │ {'Parietal':>8} │ {'Frontal':>8} │"
          f" {'Survival':>10} │ {'Sigma Gamma^2':>14} │")
    print(f"  │{'─' * 5:─>5}──│{'─' * 8:─>8}──│{'─' * 8:─>8}──│{'─' * 8:─>8}──│"
          f"{'─' * 8:─>8}──│{'─' * 10:─>10}──│{'─' * 10:─>10}──│")

    for epoch in range(1, EPOCHS + 1):
        engine.develop_epoch(motor_feedback_rate=0.3)

        if epoch in milestones:
            state = engine.get_development_state()
            overall = state["overall"]
            regions = state["regions"]

            occ = regions["occipital"]["alive_connections"]
            tmp = regions["temporal"]["alive_connections"]
            par = regions["parietal"]["alive_connections"]
            frt = regions["frontal_motor"]["alive_connections"]
            surv = overall["overall_survival_rate"]
            gamma_sq = overall["global_gamma_squared"]

            print(f"  │ {epoch:>5} │ {occ:>8} │ {tmp:>8} │ {par:>8} │ {frt:>8} │"
                  f" {surv:>9.1%} │ {gamma_sq:>10.6f} │")

            milestone_data.append({
                "epoch": epoch,
                "occipital": occ,
                "temporal": tmp,
                "parietal": par,
                "frontal": frt,
                "survival": surv,
                "gamma_sq": gamma_sq,
            })

    # Final report
    print(f"\n{engine.generate_report('Final Development Report')}")

    # Pruning curves
    curves = engine.get_pruning_curve()
    print(f"\n  Pruning curves (surviving connections vs Epoch):")
    for name in ["occipital", "temporal", "parietal", "frontal_motor"]:
        spark = ascii_spark_line(curves[name], width=50)
        final = curves[name][-1] if curves[name] else 0
        print(f"    [{name:14s}] {spark}  → {final}")

    print(f"\n  Sigma Gamma^2 curve (intelligence objective function):")
    spark = ascii_spark_line(curves["global_gamma_sq"], width=50)
    initial = curves["global_gamma_sq"][0] if curves["global_gamma_sq"] else 0
    final_g = curves["global_gamma_sq"][-1] if curves["global_gamma_sq"] else 0
    print(f"    [global_Γ²    ] {spark}  {initial:.6f} → {final_g:.6f}")

    return engine


# ============================================================================
# Experiment 2: Cortical specialization divergence
# ============================================================================


def exp2_cortical_specialization():
    """
    Cortical specialization divergence — different regions develop different frequency preferences.

    Verifies core predictions of §3.5.2:
      Occipital -> alpha/beta/gamma (8-80 Hz) — visual
      Temporal -> theta/alpha (4-13 Hz) — auditory
      Parietal -> broadband (0.5-50 Hz) — somatosensory
      Frontal -> beta (13-30 Hz) — motor (slowest)
    """
    print_header("Exp 2: Cortical Specialization — 'Why the occipital lobe becomes visual cortex...'")

    CONN = 2000
    engine = NeuralPruningEngine(connections_per_region=CONN)
    engine.develop(epochs=150, motor_feedback_rate=0.3)

    state = engine.get_development_state()

    print(f"\n  ┌{'─' * 68}┐")
    print(f"  │ {'Region':^10} │ {'Spec. Dir.':^12} │ {'Peak(Hz)':^10} │ {'Band':^8} │"
          f" {'Spec. Idx':^10} │ {'Survival':^8} │")
    print(f"  ├{'─' * 68}┤")

    for name in ["occipital", "temporal", "parietal", "frontal_motor"]:
        info = state["regions"][name]
        print(f"  │ {name:^10} │ {info['specialization']:^12} │"
              f" {info['dominant_frequency']:^10.1f} │ {info['dominant_band']:^8} │"
              f" {info['specialization_index']:^10.4f} │ {info['survival_rate']:^8.1%} │")

    print(f"  └{'─' * 68}┘")

    # Frequency distribution analysis
    print(f"\n  Frequency distribution analysis:")
    for name in ["occipital", "temporal", "parietal", "frontal_motor"]:
        region = engine.regions[name]
        alive = region.alive_connections
        if not alive:
            continue
        freqs = [c.resonant_freq for c in alive]

        # Brainwave band distribution
        band_counts = {"delta": 0, "theta": 0, "alpha": 0, "beta": 0, "gamma": 0}
        for f in freqs:
            band = BrainWaveBand.from_frequency(f)
            band_counts[band.value] += 1

        total = len(freqs)
        print(f"\n    [{name}] ({total} surviving connections)")
        for band_name, count in band_counts.items():
            pct = count / max(1, total)
            bar = "█" * int(pct * 30)
            print(f"      {band_name:6s} {bar} {pct:.0%} ({count})")

    # Theory comparison
    print(f"\n  ── Theoretical Prediction vs Actual Result ──")
    print(f"  Occipital predicted: alpha/beta/gamma (8-80 Hz) -> "
          f"actual: {state['regions']['occipital']['dominant_band']} "
          f"({state['regions']['occipital']['dominant_frequency']:.1f} Hz)")
    print(f"  Temporal predicted: theta/alpha (4-13 Hz)   -> "
          f"actual: {state['regions']['temporal']['dominant_band']} "
          f"({state['regions']['temporal']['dominant_frequency']:.1f} Hz)")
    print(f"  Parietal predicted: broadband            -> "
          f"actual: {state['regions']['parietal']['dominant_band']} "
          f"({state['regions']['parietal']['dominant_frequency']:.1f} Hz)")
    print(f"  Frontal predicted: beta (13-30 Hz)    -> "
          f"actual: {state['regions']['frontal_motor']['dominant_band']} "
          f"({state['regions']['frontal_motor']['dominant_frequency']:.1f} Hz)")

    return engine


# ============================================================================
# Experiment 3: Frontal delayed maturation
# ============================================================================


def exp3_frontal_delayed_maturation():
    """
    Frontal delayed maturation — why does motor cortex develop last?

    Physical explanation: PID closed-loop requires feedback signals to calibrate impedance.
    Infants must wave their hands ~335 times before motor pathway Gamma drops enough for precise reaching.

    Experimental design:
    - Control: all regions 100% feedback rate
    - Experimental: frontal 30% feedback rate (simulates reality)
    - Extreme: frontal 10% feedback rate (simulates severe lack of motor experience)
    """
    print_header("Exp 3: Frontal Delayed Maturation — 'Feedback Required for Impedance Calibration'")

    CONN = 1500
    EPOCHS = 100

    conditions = [
        ("100% feedback (control)", 1.0),
        ("30% feedback (normal infant)", 0.3),
        ("10% feedback (motor deprived)", 0.1),
    ]

    results = []

    for label, feedback_rate in conditions:
        engine = NeuralPruningEngine(connections_per_region=CONN)
        engine.develop(epochs=EPOCHS, motor_feedback_rate=feedback_rate)

        state = engine.get_development_state()
        frontal = state["regions"]["frontal_motor"]
        occipital = state["regions"]["occipital"]

        results.append({
            "label": label,
            "feedback": feedback_rate,
            "frontal_survival": frontal["survival_rate"],
            "frontal_spec": frontal["specialization_index"],
            "frontal_stim": frontal["stimulation_cycles"],
            "occipital_survival": occipital["survival_rate"],
            "occipital_spec": occipital["specialization_index"],
            "occipital_stim": occipital["stimulation_cycles"],
            "gamma_sq": state["overall"]["global_gamma_squared"],
        })

    print(f"\n  ┌{'─' * 90}┐")
    print(f"  │ {'Condition':^18} │ {'Frt.Surv.':^10} │ {'Frt.Spec.':^10} │ {'Frt.Stim':^8} │"
          f" {'Occ.Surv.':^10} │ {'Occ.Spec.':^10} │ {'Occ.Stim':^8} │")
    print(f"  ├{'─' * 90}┤")

    for r in results:
        print(f"  │ {r['label']:^18} │ {r['frontal_survival']:^10.1%} │"
              f" {r['frontal_spec']:^10.4f} │ {r['frontal_stim']:^8} │"
              f" {r['occipital_survival']:^10.1%} │ {r['occipital_spec']:^10.4f} │"
              f" {r['occipital_stim']:^8} │")

    print(f"  └{'─' * 90}┘")

    print(f"\n  ── Physical Explanation ──")
    print(f"  Why frontal motor cortex develops slowest:")
    print(f"    PID closed-loop requires feedback signals for impedance calibration")
    print(f"    -> Infants must wave hands hundreds of times for motor pathway Gamma to drop enough for precise reaching")
    print(f"    -> Lower feedback rate -> fewer effective stimuli -> slower pruning -> lower specialization")
    print(f"    -> This is not 'frontal lobe is dumber', but 'frontal lobe needs more feedback for impedance calibration'")

    return results


# ============================================================================
# Experiment 4: Cross-modal rewiring
# ============================================================================


def exp4_cross_modal_rewiring():
    """
    Cross-modal rewiring — how congenital blindness redirects occipital lobe to auditory.

    §8.3 Testable Prediction #6:
    'The occipital lobe of congenitally blind individuals should exhibit impedance
     characteristics matching auditory/tactile frequencies
     (because the signal dimensions flowing through changed -> different pathways selected for survival).'

    Experimental design:
    - Control: occipital<-visual, temporal<-auditory (normal development)
    - Experimental: occipital<-auditory, temporal<-visual (simulating congenital blindness)
    """
    print_header("Exp 4: Cross-Modal Rewiring — 'Signal Type Determines Cortical Specialization'")

    CONN = 2000
    EPOCHS = 150

    print(f"\n  Control: occipital <- visual, temporal <- auditory (normal development)")
    print(f"  Experimental: occipital <- auditory, temporal <- visual (congenital blindness simulation)")
    print(f"  Parameters: {CONN} connections/region, {EPOCHS} epochs")

    # Control
    control = NeuralPruningEngine(connections_per_region=CONN)
    control.develop(EPOCHS, sensory_diet={
        "occipital": "visual",
        "temporal": "auditory",
        "parietal": "somatosensory",
        "frontal_motor": "motor",
    })

    # Experimental
    crossed = NeuralPruningEngine(connections_per_region=CONN)
    crossed.develop(EPOCHS, sensory_diet={
        "occipital": "auditory",      # <- Occipital receives auditory signals
        "temporal": "visual",          # <- Temporal receives visual signals
        "parietal": "somatosensory",
        "frontal_motor": "motor",
    })

    ctrl_state = control.get_development_state()
    xmod_state = crossed.get_development_state()

    print(f"\n  ── Control Group Results ──")
    for name in ["occipital", "temporal"]:
        info = ctrl_state["regions"][name]
        print(f"    [{name}] Spec.: {info['specialization']:12s}  "
              f"Peak freq: {info['dominant_frequency']:6.1f} Hz ({info['dominant_band']})")

    print(f"\n  ── Experimental Group Results (Cross-Modal) ──")
    for name in ["occipital", "temporal"]:
        info = xmod_state["regions"][name]
        print(f"    [{name}] Spec.: {info['specialization']:12s}  "
              f"Peak freq: {info['dominant_frequency']:6.1f} Hz ({info['dominant_band']})")

    # Key comparison
    print(f"\n  ── Key Comparison ──")
    ctrl_occ_freq = ctrl_state["regions"]["occipital"]["dominant_frequency"]
    xmod_occ_freq = xmod_state["regions"]["occipital"]["dominant_frequency"]
    ctrl_tmp_freq = ctrl_state["regions"]["temporal"]["dominant_frequency"]
    xmod_tmp_freq = xmod_state["regions"]["temporal"]["dominant_frequency"]

    print(f"    Control-occipital peak freq: {ctrl_occ_freq:.1f} Hz  <-  Crossed-occipital peak freq: {xmod_occ_freq:.1f} Hz")
    print(f"    Control-temporal peak freq: {ctrl_tmp_freq:.1f} Hz  <-  Crossed-temporal peak freq: {xmod_tmp_freq:.1f} Hz")

    ctrl_occ_spec = ctrl_state["regions"]["occipital"]["specialization"]
    xmod_occ_spec = xmod_state["regions"]["occipital"]["specialization"]

    rewired = ctrl_occ_spec != xmod_occ_spec
    print(f"\n    Occipital rewiring: {'\u2713 confirmed' if rewired else 'partial'} "
          f"(control={ctrl_occ_spec}, crossed={xmod_occ_spec})")

    print(f"\n  ── Physical Conclusion ──")
    print(f"    What is pruned is not 'bad' neurons — it is impedance-mismatched connections.")
    print(f"    The occipital lobe becomes visual cortex not because it is innately 'visual',")
    print(f"    but because it happens to receive visual signals ->")
    print(f"    connections matching visual frequencies survive -> others apoptose -> becomes visual cortex.")
    print(f"    If rewired to auditory signals -> connections matching auditory frequencies survive -> rewiring.")


# ============================================================================
# Experiment 5: Gamma-squared intelligence curve
# ============================================================================


def exp5_gamma_intelligence_curve():
    """
    Gamma-squared intelligence curve — convergence of whole-brain Sigma Gamma^2 -> min.

    Intelligence = lim(t->inf) Sigma Gamma_i^2 -> min

    Verification: As development progresses, the whole-brain average Gamma^2 steadily decreases.
    i.e., the more connections correctly matched, the more 'intelligent' the system.
    """
    print_header("Exp 5: Gamma^2 Intelligence Curve — 'Intelligence = Sigma Gamma^2 -> min'")

    CONN = 2000
    EPOCHS = 200

    engine = NeuralPruningEngine(connections_per_region=CONN)
    engine.develop(EPOCHS, motor_feedback_rate=0.3)

    gamma_history = engine._gamma_squared_history
    curves = engine.get_pruning_curve()

    # Gamma^2 curve key points
    print(f"\n  Gamma^2 curve key milestones:")
    print(f"  ┌{'─' * 50}┐")
    print(f"  │ {'Epoch':>6} │ {'Sigma Gamma^2':>12} │ {'Reduction':>10} │ {'Status':^14} │")
    print(f"  ├{'─' * 50}┤")

    checkpoints = [0, 4, 9, 19, 49, 99, 149, 199]
    initial_gamma = gamma_history[0] if gamma_history else 0

    for i in checkpoints:
        if i < len(gamma_history):
            g = gamma_history[i]
            reduction = (1.0 - g / max(1e-10, initial_gamma)) * 100
            if i == 0:
                status = "Birth(chaos)"
            elif reduction < 20:
                status = "Developing"
            elif reduction < 50:
                status = "Rapid pruning"
            elif reduction < 80:
                status = "Specializing"
            else:
                status = "Mature stable"
            print(f"  │ {i + 1:>6} │ {g:>12.6f} │ {reduction:>9.1f}% │ {status:^14} │")

    print(f"  └{'─' * 50}┘")

    # Whole-brain Gamma^2 sparkline
    print(f"\n  Whole-brain Sigma Gamma^2 convergence curve:")
    spark = ascii_spark_line(gamma_history, width=60)
    print(f"    {spark}")
    print(f"    \u2191Birth(Gamma^2={gamma_history[0]:.4f}){'':>30}Mature(Gamma^2={gamma_history[-1]:.4f})\u2191")

    # Per-region surviving connection curves
    print(f"\n  Per-region pruning curves (connections vs Epoch):")
    for name in ["occipital", "temporal", "parietal", "frontal_motor"]:
        data = curves[name]
        spark = ascii_spark_line(data, width=60)
        start = data[0] if data else 0
        end = data[-1] if data else 0
        print(f"    [{name:14s}] {spark}  {start}→{end}")

    # Final intelligence quantification
    state = engine.get_development_state()
    overall = state["overall"]

    print(f"\n  \u2500\u2500 Intelligence Quantification \u2500\u2500")
    print(f"    Initial Sigma Gamma^2 = {gamma_history[0]:.6f}  (birth: whole-brain impedance chaos)")
    print(f"    Final Sigma Gamma^2   = {gamma_history[-1]:.6f}  (maturity: impedance matching complete)")
    print(f"    Improvement           = {(1.0 - gamma_history[-1] / max(1e-10, gamma_history[0])) * 100:.1f}%")
    print(f"    Avg specialization    = {overall['avg_specialization']:.4f}")
    print(f"    Whole-brain survival  = {overall['overall_survival_rate']:.1%}")
    print(f"    Total pruned          = {overall['total_pruned']}")

    print(f"\n  \u2500\u2500 Physical Conclusion \u2500\u2500")
    print(f"    Intelligence is not 'acquiring knowledge', but 'minimizing signal reflection across the whole brain'.")
    print(f"    Intelligence = lim(t->inf) Sigma Gamma_i^2 -> min")
    print(f"    From birth's {gamma_history[0]:.6f} to maturity's {gamma_history[-1]:.6f},")
    print(f"    represents life's physical process from 'random chaos' to 'ordered resonance'.")


# ============================================================================
# Main
# ============================================================================


def main():
    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + "  Gamma-Net ALICE Neural Pruning — §3.5.2 Large-Scale Gamma Apoptosis".center(70) + "║")
    print("║" + "  Self-Organization from Random Connections to Specialized Cortex".center(70) + "║")
    print("║" + "  Intelligence = Sigma Gamma^2 -> min".center(70) + "║")
    print("╚" + "═" * 70 + "╝")

    # Experiment 1: Normal development curve
    engine = exp1_normal_development()

    # Experiment 2: Cortical specialization divergence
    exp2_cortical_specialization()

    # Experiment 3: Frontal delayed maturation
    exp3_frontal_delayed_maturation()

    # Experiment 4: Cross-modal rewiring
    exp4_cross_modal_rewiring()

    # Experiment 5: Gamma^2 intelligence curve
    exp5_gamma_intelligence_curve()

    print()
    print("=" * 72)
    print("  All 5 experiments completed")
    print("  Core conclusions:")
    print("    1. Random impedance (birth) auto-forms cortical specialization via Hebbian selection")
    print("    2. Different sensory inputs determine specialization direction per region")
    print("    3. Frontal motor cortex matures last due to feedback requirement")
    print("    4. Cross-modal signals can rewire cortical function (congenital blindness)")
    print("    5. Whole-brain Sigma Gamma^2 steady decline = physical process of 'getting smarter'")
    print("=" * 72)


if __name__ == "__main__":
    main()
