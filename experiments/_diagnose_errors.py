# -*- coding: utf-8 -*-
"""Root cause diagnosis for persistent errors."""
import numpy as np
import sys
sys.path.insert(0, ".")

from alice.body.ear import AliceEar
from alice.body.hand import AliceHand

print("=== 1. Hand Reach Diagnosis ===")
hand = AliceHand()
print(f"  Workspace: {hand.workspace_w} x {hand.workspace_h}")
print(f"  Init pos: ({hand.x:.1f}, {hand.y:.1f})")

# Test 1: reach(0.5, 0.5) - what experiments use
r1 = hand.reach(0.5, 0.5, max_steps=100)
print(f"  reach(0.5, 0.5):  reached={r1['reached']}, error={r1['final_error']:.2f}, pos={r1['final_pos']}")

# Test 2: reach to center (same as init)
hand2 = AliceHand()
r2 = hand2.reach(960.0, 540.0, max_steps=10)
print(f"  reach(960, 540):  reached={r2['reached']}, error={r2['final_error']:.2f}, steps={r2['steps']}")

# Test 3: reach to a nearby target
hand3 = AliceHand()
r3 = hand3.reach(970.0, 550.0, max_steps=500)
print(f"  reach(970, 550):  reached={r3['reached']}, error={r3['final_error']:.2f}, steps={r3['steps']}")

# Test 4: reach far target with many steps
hand4 = AliceHand()
r4 = hand4.reach(500.0, 300.0, max_steps=2000)
print(f"  reach(500, 300):  reached={r4['reached']}, error={r4['final_error']:.2f}, steps={r4['steps']}")

# Root cause: exp_life_loop uses reach_for(0.5, 0.5)
# But workspace is 1920x1080 pixels! 0.5 is almost the origin corner.
print(f"\n  ROOT CAUSE: reach_for(0.5, 0.5) treats 0.5 as PIXEL coordinate")
print(f"  But the hand starts at center (960, 540)")
print(f"  Distance to target: {np.sqrt((960-0.5)**2 + (540-0.5)**2):.1f} pixels")
print(f"  This is nearly the full diagonal - PID can't converge in 100 steps")

print("\n=== 2. Ear Frequency Detection Diagnosis ===")
ear = AliceEar()
print(f"  cochlea_resolution: {ear.cochlea_resolution}")

# The issue: ear.hear() does FFT but maps to brainwave bands (0.5-100Hz)
# 440Hz real-world audio -> FFT -> power spectrum -> mapped to brainwave freq
print(f"  ear maps audio to brainwave range: 0.5~100 Hz")

# 440Hz sine wave
t = np.linspace(0, 0.1, 4410)
a_note = 0.3 * np.sin(2 * np.pi * 440 * t)
signal = ear.hear(a_note)
print(f"\n  Input: 440Hz sine, {len(a_note)} samples")
print(f"  Output freq: {signal.frequency:.4f} Hz")
print(f"  Output amp:  {signal.amplitude:.4f}")
print(f"  Band: {signal.band.value}")

# The issue: ear.hear() does FFT but maps to brainwave bands (0.5-100Hz)
# 440Hz real-world audio -> FFT -> power spectrum -> mapped to brainwave freq
# This is BY DESIGN: the ear converts audio to neural signals in brainwave range
print(f"\n  ANALYSIS: ear.hear() does NOT output audio Hz")
print(f"  It converts sound -> cochlear processing -> brainwave-range signal")
print(f"  440Hz audio -> mapped to delta/theta/alpha/beta/gamma (0.5-100Hz)")
print(f"  This is biologically correct: auditory cortex fires at brainwave rates")

# Short signal test
short = 0.3 * np.sin(2 * np.pi * 10 * np.linspace(0, 1, 256))
sig2 = ear.hear(short)
print(f"\n  Input: 10Hz sine, 256 samples")
print(f"  Output freq: {sig2.frequency:.4f} Hz")
print(f"  Band: {sig2.band.value}")

print("\n=== 3. Gamma_loop Analysis ===")
from alice.brain.broca import BrocaEngine, SUCCESS_GAMMA_THRESHOLD
from alice.brain.semantic_field import SemanticFieldEngine
from alice.body.cochlea import CochlearFilterBank, generate_vowel
from alice.body.mouth import AliceMouth

cochlea = CochlearFilterBank()
broca = BrocaEngine(cochlea=cochlea)
mouth = AliceMouth()
engine = SemanticFieldEngine()

vowel = "a"
wave = generate_vowel(vowel, duration=0.3, sample_rate=16000)
tono = cochlea.analyze(wave, apply_persistence=False)
fp = tono.fingerprint()
engine.field.register_concept(f"vowel_{vowel}", fp, modality="auditory")

# Build mass
for _ in range(20):
    w = generate_vowel(vowel, duration=0.3, sample_rate=16000)
    t2 = cochlea.analyze(w, apply_persistence=False)
    engine.field.absorb(f"vowel_{vowel}", t2.fingerprint(), modality="auditory")

broca.create_vowel_plan(vowel)

# Diagnose: what does the feedback fingerprint look like vs the target?
plan = broca.plans[f"vowel_{vowel}"]
exec_result = broca.execute_plan(plan, mouth, ram_temperature=0.0)
fb_fp = exec_result["feedback_fingerprint"]
target_fp = engine.field.attractors[f"vowel_{vowel}"].modality_centroids["auditory"]

from alice.brain.semantic_field import cosine_similarity, gamma_semantic
attractor = engine.field.attractors[f"vowel_{vowel}"]
mass = attractor.total_mass
cos_sim = cosine_similarity(fb_fp, target_fp)
gamma = gamma_semantic(cos_sim, mass)
print(f"  Feedback FP shape: {fb_fp.shape}")
print(f"  Target FP shape:   {target_fp.shape}")
print(f"  Cosine similarity: {cos_sim:.6f}")
print(f"  Gamma semantic:    {gamma:.6f}")
print(f"  Concept mass:      {mass:.2f}")
print(f"  SUCCESS_THRESHOLD: {SUCCESS_GAMMA_THRESHOLD}")
print(f"\n  ANALYSIS: gamma={gamma:.4f} vs threshold={SUCCESS_GAMMA_THRESHOLD}")
print(f"  Because mouth.speak() produces a simplified waveform")
print(f"  whose cochlear fingerprint differs from generate_vowel()")
print(f"  The two waveform generators use different synthesis methods")

# Compare waveforms
spoke = mouth.speak(target_pitch=150, formants=(730, 1090, 2440), volume=0.5)
spoke_tono = cochlea.analyze(spoke["waveform"], apply_persistence=False)
spoke_fp = spoke_tono.fingerprint()

gen_tono = cochlea.analyze(wave[:len(spoke["waveform"])], apply_persistence=False)
gen_fp = gen_tono.fingerprint()

cos2 = cosine_similarity(spoke_fp, gen_fp)
gamma2 = gamma_semantic(cos2, 1.0)
print(f"\n  mouth.speak() FP vs generate_vowel() FP:")
print(f"    cosine similarity: {cos2:.6f}")
print(f"    gamma semantic:    {gamma2:.6f}")
print(f"\n  Top 5 channels of each:")
top_spoke = np.argsort(spoke_fp)[-5:][::-1]
top_gen = np.argsort(gen_fp)[-5:][::-1]
print(f"    mouth.speak() peaks at channels: {top_spoke}")
print(f"    generate_vowel() peaks at channels: {top_gen}")
