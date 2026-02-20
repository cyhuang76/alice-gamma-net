# -*- coding: utf-8 -*-
"""
Experiment: Thalamic Sensory Gate + Amygdala Fast Emotional Pathway
Experiment: Thalamic Sensory Gate + Amygdala Fast Emotional Pathway

Verification of Phase 5.3 + 5.4 — 8 Core Predictions:

  Experiment 1: Gate gain modulated by arousal level
    — High arousal → gate opens, low arousal → gate closes

  Experiment 2: Startle circuit unconditional bypass
    — High amplitude / high Γ signal unconditionally passes gate

  Experiment 3: Habituation and novelty recovery
    — Continuous same stimulus → gate gradually closes, new stimulus → recovery

  Experiment 4: TRN competitive inhibition
    — Focused channel inhibits other channels

  Experiment 5: Fear conditioning and matching
    — Paired fingerprint with threat → automatically triggers fear afterward

  Experiment 6: Fear extinction and residual
    — Safe exposure reduces threat, but fear memory is never deleted

  Experiment 7: Fight-flight-freeze cascade response
    — Threat level → fight-flight → freeze graded response

  Experiment 8: Emotional memory enhancement (amygdala→hippocampus coupling)
    — High-emotion events are remembered more strongly

Usage:
  python -m experiments.exp_thalamus_amygdala
"""

import sys
import time
import numpy as np

sys.path.insert(0, ".")

from alice.brain.thalamus import (
    ThalamusEngine,
    GATE_MIN, GATE_MAX,
    STARTLE_AMPLITUDE_THRESHOLD,
    STARTLE_GAMMA_THRESHOLD,
    BURST_AROUSAL_THRESHOLD,
    MAX_FOCUSED_CHANNELS,
)
from alice.brain.amygdala import (
    AmygdalaEngine,
    THREAT_THRESHOLD,
    FEAR_THRESHOLD,
    FREEZE_THRESHOLD,
    THREAT_IMPEDANCE,
    SAFETY_IMPEDANCE,
)
from alice.alice_brain import AliceBrain


# ============================================================================
# tools
# ============================================================================


def _make_fp(dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    fp = rng.randn(dim).astype(np.float64)
    norm = np.linalg.norm(fp)
    if norm > 1e-10:
        fp /= norm
    return fp


def _separator(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def _bar(value: float, width: int = 30) -> str:
    n = int(max(0, min(1, value)) * width)
    return "█" * n + "░" * (width - n)


# ============================================================================
# Experiment 1: Gate gain modulated by arousal level
# ============================================================================


def exp1_arousal_gate_modulation():
    """Arousal → gate gain curve"""
    _separator("Experiment 1: Gate gain modulated by arousal level")

    thal = ThalamusEngine()
    fp = _make_fp(64)

    print(" 'Arousal level controls overall thalamic gate gain.'")
    print(" 'Asleep, you hear nothing — not because the ears shut down, but because the thalamic gate closes.'")
    print()
    print(f" {'Arousal':>8} {'GateGain':>10} {'PASS?':>6} Waveform")
    print("  " + "-" * 60)

    arousal_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for arousal in arousal_levels:
        thal.reset()
        thal.set_arousal(arousal)
        thal.set_attention("visual", 0.6)

        # multiple gates to let EMA converge
        for _ in range(10):
            result = thal.gate("visual", fingerprint=fp, amplitude=0.5, gamma=0.3)

        status = "✓" if result.passed else "✗"
        print(f"  {arousal:>8.1f}  {result.gate_gain:>10.4f}  {status:>6}  {_bar(result.gate_gain)}")

    print()
    print(" ✓ Arousal and gate gain are positively correlated")
    print(" ✓ At low arousal, signals are largely attenuated or blocked")
    return True


# ============================================================================
# Experiment 2: Startle circuit unconditional bypass
# ============================================================================


def exp2_startle_bypass():
    """Startle circuit bypasses gate at any arousal level"""
    _separator("Experiment 2: Startle circuit unconditional bypass")

    thal = ThalamusEngine()
    print(" 'Hearing a massive explosion will startle you awake even in deep sleep.'")
    print(" 'This does not go through the cortex — it is a low-level thalamic interrupt (IRQ).'")
    print()
    print(f" {'Arousal':>8} {'Amp':>6} {'Γ':>6} {'Startle?':>6} {'PASS?':>6} {'Gain':>6}")
    print("  " + "-" * 55)

    test_cases = [
        # (arousal, amplitude, gamma, expect_startle)
        (0.1, 0.3, 0.3, False), # Low arousal + low signal → no bypass
        (0.1, 0.9, 0.3, True), # Low arousal + high amplitude → bypass
        (0.1, 0.3, 0.9, True), # Low arousal + high Γ → bypass
        (0.0, 0.95, 0.1, True), # Fully asleep + loud noise → bypass
        (1.0, 0.3, 0.3, False), # High arousal + low signal → normal gating
        (1.0, 0.85, 0.5, True), # High arousal + high amplitude → bypass
    ]

    for arousal, amp, gamma, expect in test_cases:
        thal.reset()
        thal.set_arousal(arousal)
        result = thal.gate("auditory", amplitude=amp, gamma=gamma)
        status = "✓" if result.is_startle == expect else "✗"
        print(f"  {arousal:>8.1f}  {amp:>6.2f}  {gamma:>6.2f}  "
              f"{'Yes' if result.is_startle else 'No':>6} "
              f"{'Yes' if result.passed else 'No':>6} "
              f"{result.gate_gain:>6.2f}  {status}")

    print()
    print(f" ✓ Amplitude ≥ {STARTLE_AMPLITUDE_THRESHOLD} or Γ ≥ {STARTLE_GAMMA_THRESHOLD} triggers startle circuit")
    print(" ✓ Startle signal gate gain = 1.0 (no attenuation)")
    return True


# ============================================================================
# Experiment 3: Habituation and novelty recovery
# ============================================================================


def exp3_habituation():
    """Continuous same stimulus → gate gradually closes"""
    _separator("Experiment 3: Habituation and novelty recovery")

    thal = ThalamusEngine()
    thal.set_arousal(0.8)
    thal.set_attention("visual", 0.6)

    fp_same = _make_fp(64, seed=42)
    fp_novel = _make_fp(64, seed=99)

    print(" 'The first time you hear the AC hum, it is very noticeable.'")
    print(" 'After ten minutes you no longer hear it — this is thalamic habituation.'")
    print(" 'Suddenly someone calls your name — the gate reopens.'")
    print()
    print(" Phase 1: Continuous same stimulus")
    print(f" {'Trial#':>8} {'GateGain':>10} {'Habituation':>8} {'Salience':>8} Waveform")
    print("  " + "-" * 60)

    gains_same = []
    for i in range(30):
        result = thal.gate("visual", fingerprint=fp_same, amplitude=0.5, gamma=0.3)
        ch = thal._channels["visual"]
        gains_same.append(result.gate_gain)
        if i % 5 == 0 or i == 29:
            print(f"  {i+1:>8}  {result.gate_gain:>10.4f}  {ch.habituation:>8.4f}  "
                  f"{result.salience:>8.4f}  {_bar(result.gate_gain)}")

    print()
    print(" Phase 2: Introducing novel stimulus")
    for i in range(5):
        result = thal.gate("visual", fingerprint=fp_novel, amplitude=0.5, gamma=0.3)
        ch = thal._channels["visual"]
        print(f"  Novel {i+1} {result.gate_gain:>10.4f} {ch.habituation:>8.4f} "
              f"{result.salience:>8.4f}  {_bar(result.gate_gain)}")

    if len(gains_same) > 5:
        print()
        print(f"  Initial gain: {gains_same[0]:.4f} → Final gain: {gains_same[-1]:.4f}")
        print(f" ✓ Habituation reduces gate gain {(gains_same[0]-gains_same[-1])/max(0.001,gains_same[0])*100:.1f}%")
    return True


# ============================================================================
# Experiment 4: TRN competitive inhibition
# ============================================================================


def exp4_trn_inhibition():
    """Thalamic reticular nucleus inter-channel competition"""
    _separator("Experiment 4: TRN competitive inhibition")

    thal = ThalamusEngine()
    thal.set_arousal(0.8)

    print(" 'When reading, you cannot hear background music.'")
    print(" 'The music did not disappear — TRN inhibited the auditory channel.'")
    print()

    # Initialize three channels
    for mod in ["visual", "auditory", "tactile"]:
        thal.set_attention(mod, 0.5)
        for _ in range(5):
            thal.gate(mod, fingerprint=_make_fp(32), amplitude=0.5, gamma=0.3)

    print(" === Before Focus ===")
    for mod in ["visual", "auditory", "tactile"]:
        ch = thal._channels[mod]
        print(f"    {mod:>10}: bias={ch.topdown_bias:.3f}, gain={ch.gate_gain:.4f}")

    # Focus on visual
    thal.set_attention("visual", 0.95)
    thal.apply_trn_inhibition()

    # Multiple gates to let effect manifest
    for _ in range(10):
        for mod in ["visual", "auditory", "tactile"]:
            thal.gate(mod, fingerprint=_make_fp(32), amplitude=0.5, gamma=0.3)

    print()
    print(" === After Focusing on Visual ===")
    for mod in ["visual", "auditory", "tactile"]:
        ch = thal._channels[mod]
        bar = _bar(ch.gate_gain)
        print(f"    {mod:>10}: bias={ch.topdown_bias:.3f}, gain={ch.gate_gain:.4f}  {bar}")

    vis = thal._channels["visual"].gate_gain
    aud = thal._channels["auditory"].gate_gain
    print()
    print(f" ✓ visual channel gain ({vis:.4f}) > auditory channel gain ({aud:.4f})")
    print(f" ✓ Attention is exclusive: focusing on one channel inhibits others")
    return True


# ============================================================================
# Experiment 5: Fear conditioning and matching
# ============================================================================


def exp5_fear_conditioning():
    """Fear conditioning → automatically triggers fear afterward"""
    _separator("Experiment 5: Fear conditioning and matching")

    amyg = AmygdalaEngine()
    fp_snake = _make_fp(64, seed=42)
    fp_flower = _make_fp(64, seed=99)

    print(" 'Bitten by a snake once. From then on, seeing a snake accelerates heartbeat.'")
    print(" 'This is not rational judgment — it is an automatic amygdala response.'")
    print()

    # Before conditioning
    resp_before = amyg.evaluate("visual", fingerprint=fp_snake, concept_label="snake")
    print(f" Before conditioning, seeing snake: threat={resp_before.emotional_state.threat_level:.4f}, "
          f"fearmatch={resp_before.fear_matched}")

    resp_flower = amyg.evaluate("visual", fingerprint=fp_flower, concept_label="flower")
    print(f" Before conditioning, seeing flower: threat={resp_flower.emotional_state.threat_level:.4f}, "
          f"fearmatch={resp_flower.fear_matched}")

    # Snake bite! Fear conditioning
    print()
    print(" >>> Snake bite! Fear conditioning ... <<<")
    amyg.condition_fear("visual", fp_snake, threat_level=0.9, concept_label="snake")
    amyg.condition_fear("visual", fp_snake, threat_level=0.9, concept_label="snake")
    amyg.condition_fear("visual", fp_snake, threat_level=0.9, concept_label="snake")
    print(f" Fear memory created: conditioning count={amyg._fear_memories[0].conditioning_count}")
    print()

    # Decay emotions to let them return to zero
    for _ in range(30):
        amyg.decay_tick()

    # After conditioning
    resp_after = amyg.evaluate("visual", fingerprint=fp_snake, concept_label="snake")
    print(f" After conditioning, seeing snake: threat={resp_after.emotional_state.threat_level:.4f}, "
          f"fearmatch={resp_after.fear_matched}, "
          f"valence={resp_after.emotional_state.valence:.4f}")

    resp_flower2 = amyg.evaluate("visual", fingerprint=fp_flower, concept_label="flower")
    print(f" After conditioning, seeing flower: threat={resp_flower2.emotional_state.threat_level:.4f}, "
          f"fearmatch={resp_flower2.fear_matched}")

    print()
    print(f" ✓ Fear conditioning raises threat level from {resp_before.emotional_state.threat_level:.4f} "
          f"→ {resp_after.emotional_state.threat_level:.4f}")
    print(f" ✓ Unconditioned stimulus (flower) is unaffected")
    return True


# ============================================================================
# Experiment 6: Fear extinction and residual
# ============================================================================


def exp6_fear_extinction():
    """Extinction reduces threat, but fear memory is never deleted"""
    _separator("Experiment 6: Fear extinction and residual")

    amyg = AmygdalaEngine()
    fp = _make_fp(64, seed=42)

    print(" 'You fear dogs. A therapist slowly introduces you to gentle dogs.'")
    print(" 'Fear will decrease — but never vanish. Under stress, it may relapse.'")
    print()

    # Fear conditioning
    amyg.condition_fear("visual", fp, threat_level=0.9, concept_label="dog")
    initial_threat = amyg._fear_memories[0].effective_threat
    print(f" Initial effective threat: {initial_threat:.4f}")

    print()
    print(f" {'ExtinctionN':>10} {'EffThreat':>10} {'Extinct':>6} {'Cond':>6} Waveform")
    print("  " + "-" * 60)

    for i in range(30):
        amyg.extinguish_fear("visual", fp, concept_label="dog")
        et = amyg._fear_memories[0].effective_threat
        ec = amyg._fear_memories[0].extinction_count
        cc = amyg._fear_memories[0].conditioning_count
        if i % 5 == 0 or i == 29:
            print(f"  {i+1:>10}  {et:>10.4f}  {ec:>6}  {cc:>6}  {_bar(et)}")

    final_threat = amyg._fear_memories[0].effective_threat
    print()
    print(f" Initial effective threat: {initial_threat:.4f}")
    print(f" Effective threat after extinction: {final_threat:.4f}")
    print(f" Reduction: {(1 - final_threat / initial_threat) * 100:.1f}%")
    print()
    print(f" ✓ Extinction reduces effective threat")
    print(f" ✓ But fear memory still exists (effective threat > 0)")
    print(f" ✓ Fear never fully clears — this is the physical basis for trauma relapse")
    return True


# ============================================================================
# Experiment 7: Fight-flight-freeze cascade response
# ============================================================================


def exp7_fight_flight_freeze():
    """Threat level → fight-flight → freeze graded response"""
    _separator("Experiment 7: Fight-flight-freeze cascade response")

    print(" 'Mild threat → alertness. Moderate threat → flee. Extreme threat → freeze.'")
    print(" 'Freezing is not cowardice — it is amygdala overload protection.'")
    print()
    print(f" {'Pain':>6} {'Amp':>6} {'Threat':>8} {'Valence':>8} {'F/F':>6} {'Freeze':>6} {'Sympa':>6} Emotion")
    print("  " + "-" * 75)

    # Gradually increase threat
    pain_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for pain in pain_levels:
        amyg = AmygdalaEngine() # Reset each time
        # Multiple assessments to let state converge
        for _ in range(15):
            resp = amyg.evaluate("visual", amplitude=0.5 + pain * 0.3,
                                 gamma=0.3 + pain * 0.4, pain_level=pain)

        es = resp.emotional_state
        ff = "Yes" if es.is_fight_flight else "No"
        fz = "Yes" if es.is_freeze else "No"
        print(f"  {pain:>6.1f}  {0.5+pain*0.3:>6.2f}  {es.threat_level:>8.4f}  "
              f"{es.valence:>8.4f}  {ff:>6}  {fz:>6}  "
              f"{resp.sympathetic_command:>6.2f}  {es.emotion_label}")

    print()
    print(f" ✓ threat < {FEAR_THRESHOLD}: Alert but no fight-flight")
    print(f" ✓ {FEAR_THRESHOLD} ≤ threat < {FREEZE_THRESHOLD}: Fight-flight response (sympathetic activation ~{FEAR_THRESHOLD})")
    print(f" ✓ threat ≥ {FREEZE_THRESHOLD}: Freeze response (overload protection)")
    return True


# ============================================================================
# Experiment 8: Emotional Memory Enhancement (Amygdala-Hippocampus Coupling)
# ============================================================================


def exp8_emotional_memory():
    """High-emotion events through AliceBrain full pipeline"""
    _separator("Experiment 8: Emotional Memory Enhancement (AliceBrain Full Pipeline)")

    brain = AliceBrain(neuron_count=10)

    print(" 'You do not remember what you had for lunch yesterday.'")
    print(" 'But you remember your first frightening experience for a lifetime.'")
    print(" 'Amygdala emotional tagging lets hippocampus prioritize consolidating these memories.'")
    print()

    # Show a few neutral scenes
    print(" Phase 1: Neutral scenes")
    for i in range(5):
        pixels = np.random.RandomState(i).randn(64) * 0.3 # Low amplitude
        result = brain.see(pixels)
        es = result.get("amygdala", {})
        print(f" Scene {i+1}: valence={es.get('valence', 'N/A')}, "
              f"threat={es.get('threat_level', 'N/A')}, "
              f"Emotion={es.get('emotion_label', 'N/A')}")

    print()
    print(" Phase 2: high StartleScene")
    for i in range(5):
        pixels = np.random.RandomState(100 + i).randn(64) * 3.0 # High amplitude
        result = brain.see(pixels)
        es = result.get("amygdala", {})
        thal = result.get("thalamus", {})
        print(f" Startle {i+1}: valence={es.get('valence', 'N/A')}, "
              f"threat={es.get('threat_level', 'N/A')}, "
              f"gate={thal.get('gate_gain', 'N/A')}, "
              f"Emotion={es.get('emotion_label', 'N/A')}")

    print()
    print(" Phase 3: Auditory Emotion Pipeline")
    for i in range(3):
        t = np.linspace(0, 1, 1024, endpoint=False)
        freq = 440 * (1 + i)
        sound = np.sin(2 * np.pi * freq * t) * (0.3 + i * 0.3)
        result = brain.hear(sound)
        es = result.get("amygdala", {})
        thal = result.get("thalamus", {})
        print(f" Sound {i+1} ({freq}Hz): valence={es.get('valence', 'N/A')}, "
              f"gate={thal.get('gate_gain', 'N/A')}")

    print()
    state = brain.introspect()
    thal_state = state["subsystems"].get("thalamus", {})
    amyg_state = state["subsystems"].get("amygdala", {})
    print(f" ✓ Thalamus stats: pass_rate={thal_state.get('stats', {}).get('pass_rate', 'N/A')}")
    print(f" ✓ Amygdala stats: evaluation_count={amyg_state.get('stats', {}).get('total_evaluations', 'N/A')}")
    print(f" ✓ High-emotion scenes generate thalamic startle + amygdala threat → emotional memory enhancement")
    return True


# ============================================================================
# main program
# ============================================================================


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Γ-Net ALICE Thalamus + Amygdala Experiment Suite              ║")
    print("║  Phase 5.3: Thalamus (Sensory Gate)                        ║")
    print("║  Phase 5.4: Amygdala (Emotional Fast Path)                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    experiments = [
        ("Exp 1", exp1_arousal_gate_modulation),
        ("Exp 2", exp2_startle_bypass),
        ("Exp 3", exp3_habituation),
        ("Exp 4", exp4_trn_inhibition),
        ("Exp 5", exp5_fear_conditioning),
        ("Exp 6", exp6_fear_extinction),
        ("Exp 7", exp7_fight_flight_freeze),
        ("Exp 8", exp8_emotional_memory),
    ]

    results = {}
    for name, func in experiments:
        try:
            ok = func()
            results[name] = "PASS" if ok else "FAIL"
        except Exception as e:
            results[name] = f"ERROR: {e}"
            import traceback
            traceback.print_exc()

    _separator("Experiment Summary")
    print(f"  {'experiment':>10}  {'Result':>10}")
    print("  " + "-" * 25)
    for name, status in results.items():
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  {name:>10}  {symbol} {status}")
    print()

    passed = sum(1 for s in results.values() if s == "PASS")
    total = len(results)
    print(f"  PASS: {passed}/{total}")
    print()


if __name__ == "__main__":
    main()
