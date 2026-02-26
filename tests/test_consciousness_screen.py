# -*- coding: utf-8 -*-
"""
Tests for ConsciousnessScreen — Terminal Rendering Surface

Verifies:
  - C1: Γ² + T = 1 at every pixel, every frame
  - Pixel physics: terminal Γ from synaptic_strength
  - Dead pixels (stroke): permanent Γ=1
  - Burn-in (PTSD): trauma afterimage persists
  - Pain overload: screen saturation
  - Dream rendering: camera off, screen on
  - Stereo depth: ΔΓ differential signaling
  - C3: get_signal() returns ElectricalSignal
"""

import unittest
import numpy as np

from alice.brain.neural_display import (
    NeuralActivityDisplay as ConsciousnessScreen,
    ScreenFrame,
    StereoFrame,
    compute_stereo_depth,
    BURN_IN_RATE,
    BURN_IN_DECAY,
    BURN_IN_RESIDUAL,
    PAIN_OVERLOAD_THRESHOLD,
    PAIN_SATURATION_GAIN,
)
from alice.core.signal import ElectricalSignal


# ============================================================================
# C1: Energy Conservation — Γ² + T = 1
# ============================================================================

class TestC1EnergyConservation(unittest.TestCase):
    """C1: Γ_eff² + T_pixel = 1 at every pixel, every frame."""

    def test_c1_holds_for_random_neurons(self):
        """Γ² + T = 1 for every pixel with random neuron strengths."""
        screen = ConsciousnessScreen(resolution=50, screen_id="test")
        strengths = np.random.uniform(0.01, 2.0, 50)
        values = np.random.uniform(0, 1, 50)
        frame = screen.render(values, strengths, channel_gamma=0.3)

        for i in range(50):
            T_i = frame.clarity[i]
            G_i = frame.gamma_map[i]
            self.assertAlmostEqual(
                G_i ** 2 + T_i, 1.0, places=10,
                msg=f"Pixel {i}: Γ²+T = {G_i**2 + T_i:.15f} ≠ 1.0"
            )

    def test_c1_holds_with_dead_pixels(self):
        """C1 still holds when dead pixels are present."""
        screen = ConsciousnessScreen(resolution=20)
        screen.kill_pixels([3, 7, 15])
        strengths = np.ones(20) * 0.8
        values = np.ones(20) * 0.5
        frame = screen.render(values, strengths, channel_gamma=0.2)

        for i in range(20):
            T_i = frame.clarity[i]
            G_i = frame.gamma_map[i]
            self.assertAlmostEqual(
                G_i ** 2 + T_i, 1.0, places=10,
                msg=f"Pixel {i}: Γ²+T ≠ 1.0 (dead={i in [3,7,15]})"
            )

    def test_c1_holds_with_channel_loss(self):
        """C1 holds for various channel gamma values."""
        screen = ConsciousnessScreen(resolution=10)
        strengths = np.linspace(0.1, 1.5, 10)
        values = np.ones(10) * 0.5

        for channel_gamma in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            frame = screen.render(values, strengths, channel_gamma=channel_gamma)
            for i in range(10):
                T_i = frame.clarity[i]
                G_i = frame.gamma_map[i]
                self.assertAlmostEqual(
                    G_i ** 2 + T_i, 1.0, places=10,
                    msg=f"Γ_ch={channel_gamma}, pixel {i}: Γ²+T ≠ 1"
                )


# ============================================================================
# Pixel Physics
# ============================================================================

class TestPixelPhysics(unittest.TestCase):
    """Terminal Γ from synaptic_strength + channel Γ → pixel clarity."""

    def test_perfect_match_full_clarity(self):
        """s=1.0, Γ_ch=0 → T=1 (perfect match, clear image)."""
        screen = ConsciousnessScreen(resolution=5)
        strengths = np.ones(5)
        values = np.ones(5) * 0.5
        frame = screen.render(values, strengths, channel_gamma=0.0)

        for i in range(5):
            self.assertAlmostEqual(frame.clarity[i], 1.0, places=6)
            self.assertAlmostEqual(frame.gamma_map[i], 0.0, places=6)

    def test_zero_strength_snow(self):
        """s≈0 → T≈0 (dead neuron, snow pixel)."""
        screen = ConsciousnessScreen(resolution=5)
        strengths = np.ones(5) * 1e-6
        values = np.ones(5) * 0.5
        frame = screen.render(values, strengths, channel_gamma=0.0)

        for i in range(5):
            self.assertAlmostEqual(frame.clarity[i], 0.0, places=3)
            self.assertAlmostEqual(frame.gamma_map[i], 1.0, places=3)

    def test_half_strength_known_gamma(self):
        """s=0.5 → Γ_terminal = 1/3, T_terminal = 8/9."""
        screen = ConsciousnessScreen(resolution=1)
        strengths = np.array([0.5])
        values = np.array([1.0])
        frame = screen.render(values, strengths, channel_gamma=0.0)

        expected_gamma = 1.0 / 3.0
        expected_T = 1.0 - expected_gamma ** 2  # = 8/9
        self.assertAlmostEqual(frame.gamma_map[0], expected_gamma, places=6)
        self.assertAlmostEqual(frame.clarity[0], expected_T, places=6)

    def test_super_matched_neuron(self):
        """s>1 (enhanced neuron) → small Γ_terminal, high T."""
        screen = ConsciousnessScreen(resolution=1)
        # s=2: Γ_terminal = |（1-2)/(1+2)| = 1/3
        strengths = np.array([2.0])
        values = np.array([1.0])
        frame = screen.render(values, strengths, channel_gamma=0.0)

        expected_gamma = 1.0 / 3.0
        expected_T = 1.0 - expected_gamma ** 2
        self.assertAlmostEqual(frame.gamma_map[0], expected_gamma, places=6)
        self.assertAlmostEqual(frame.clarity[0], expected_T, places=6)

    def test_symmetry_s_and_inverse(self):
        """s=0.5 and s=2.0 have identical |Γ| (impedance symmetry)."""
        screen = ConsciousnessScreen(resolution=2)
        strengths = np.array([0.5, 2.0])
        values = np.array([1.0, 1.0])
        frame = screen.render(values, strengths, channel_gamma=0.0)

        self.assertAlmostEqual(
            frame.gamma_map[0], frame.gamma_map[1], places=6,
            msg="s=0.5 and s=2.0 should have identical |Γ|"
        )
        self.assertAlmostEqual(
            frame.clarity[0], frame.clarity[1], places=6,
            msg="s=0.5 and s=2.0 should have identical T"
        )

    def test_channel_gamma_reduces_clarity(self):
        """Higher channel Γ → lower T for all pixels."""
        screen = ConsciousnessScreen(resolution=5)
        strengths = np.ones(5) * 0.8
        values = np.ones(5) * 0.5

        frame_clear = screen.render(values, strengths, channel_gamma=0.0)
        frame_lossy = screen.render(values, strengths, channel_gamma=0.5)

        for i in range(5):
            self.assertGreater(
                frame_clear.clarity[i], frame_lossy.clarity[i],
                msg=f"Pixel {i}: Γ_ch=0 should be clearer than Γ_ch=0.5"
            )

    def test_combined_transmission_formula(self):
        """T_pixel = T_channel × T_terminal (composition law)."""
        screen = ConsciousnessScreen(resolution=1)
        s = 0.7
        gamma_ch = 0.4

        strengths = np.array([s])
        values = np.array([1.0])
        frame = screen.render(values, strengths, channel_gamma=gamma_ch)

        # Manual computation
        gamma_term = abs((1 - s) / (1 + s))
        T_channel = 1 - gamma_ch ** 2
        T_terminal = 1 - gamma_term ** 2
        T_expected = T_channel * T_terminal

        self.assertAlmostEqual(frame.clarity[0], T_expected, places=10)

    def test_content_is_signal_times_T(self):
        """Rendered content = signal × T (no burn-in, no noise, no pain)."""
        screen = ConsciousnessScreen(resolution=5)
        screen.set_temperature(0.0)  # No noise
        strengths = np.array([1.0, 0.5, 0.8, 1.2, 0.3])
        values = np.array([0.5, 1.0, 0.3, 0.7, 0.9])
        frame = screen.render(values, strengths, channel_gamma=0.0)

        for i in range(5):
            expected = values[i] * frame.clarity[i]
            self.assertAlmostEqual(
                frame.content[i], expected, places=6,
                msg=f"Pixel {i}: content should be signal × T"
            )


# ============================================================================
# Screen Rendering
# ============================================================================

class TestScreenRendering(unittest.TestCase):
    """Frame rendering mechanics."""

    def test_render_returns_screen_frame(self):
        """render() returns proper ScreenFrame."""
        screen = ConsciousnessScreen(resolution=10, screen_id="test_visual")
        frame = screen.render(np.ones(10), np.ones(10))
        self.assertIsInstance(frame, ScreenFrame)
        self.assertEqual(frame.screen_id, "test_visual")
        self.assertEqual(frame.resolution, 10)

    def test_brightness_is_mean_transmission(self):
        """Φ_screen = mean(T_i) across active pixels."""
        screen = ConsciousnessScreen(resolution=4)
        # s=1→T=1, s=0.5→T=8/9, s=0.5→T=8/9, s=1→T=1
        strengths = np.array([1.0, 0.5, 0.5, 1.0])
        values = np.ones(4)
        frame = screen.render(values, strengths, channel_gamma=0.0)

        expected = np.mean(frame.clarity[:4])
        self.assertAlmostEqual(frame.brightness, expected, places=6)

    def test_contrast_varies_with_mixed_strengths(self):
        """Mixed neuron health → non-zero contrast."""
        screen = ConsciousnessScreen(resolution=10)
        strengths = np.linspace(0.1, 1.0, 10)
        values = np.ones(10)
        frame = screen.render(values, strengths)

        self.assertGreater(frame.contrast, 0.0,
                           msg="Mixed strengths should produce non-zero contrast")

    def test_uniform_strengths_zero_contrast(self):
        """Uniform neuron health → zero contrast."""
        screen = ConsciousnessScreen(resolution=10)
        strengths = np.ones(10) * 0.8
        values = np.ones(10)
        frame = screen.render(values, strengths)

        self.assertAlmostEqual(frame.contrast, 0.0, places=6,
                               msg="Uniform strengths should have zero contrast")

    def test_frame_id_increments(self):
        """Sequential renders increment frame_id."""
        screen = ConsciousnessScreen(resolution=5)
        strengths = np.ones(5)
        values = np.ones(5)

        f1 = screen.render(values, strengths)
        f2 = screen.render(values, strengths)
        f3 = screen.render(values, strengths)

        self.assertEqual(f1.frame_id, 1)
        self.assertEqual(f2.frame_id, 2)
        self.assertEqual(f3.frame_id, 3)

    def test_total_renders_counter(self):
        """total_renders tracks number of render calls."""
        screen = ConsciousnessScreen(resolution=5)
        s, v = np.ones(5), np.ones(5)
        screen.render(v, s)
        screen.render(v, s)
        screen.render(v, s)
        self.assertEqual(screen.total_renders, 3)


# ============================================================================
# Dead Pixels (Stroke)
# ============================================================================

class TestDeadPixels(unittest.TestCase):
    """Stroke = permanent dead pixels (Γ=1, T=0)."""

    def test_dead_pixel_gamma_one(self):
        """Killed pixel has Γ=1.0."""
        screen = ConsciousnessScreen(resolution=10)
        screen.kill_pixels([3])
        frame = screen.render(np.ones(10), np.ones(10))
        self.assertEqual(frame.gamma_map[3], 1.0)

    def test_dead_pixel_zero_clarity(self):
        """Killed pixel has T=0."""
        screen = ConsciousnessScreen(resolution=10)
        screen.kill_pixels([5])
        frame = screen.render(np.ones(10), np.ones(10))
        self.assertEqual(frame.clarity[5], 0.0)

    def test_dead_pixel_zero_content(self):
        """Killed pixel renders zero content (signal × 0 = 0)."""
        screen = ConsciousnessScreen(resolution=10)
        screen.set_temperature(0.0)
        screen.kill_pixels([2])
        frame = screen.render(np.ones(10) * 0.8, np.ones(10))
        self.assertAlmostEqual(frame.content[2], 0.0, places=6)

    def test_dead_pixel_survives_render(self):
        """Dead pixel persists across multiple renders."""
        screen = ConsciousnessScreen(resolution=10)
        screen.kill_pixels([4])

        for _ in range(10):
            frame = screen.render(np.ones(10), np.ones(10))
            self.assertEqual(frame.gamma_map[4], 1.0)
            self.assertEqual(frame.clarity[4], 0.0)

    def test_dead_pixel_count(self):
        """dead_pixels field counts correctly."""
        screen = ConsciousnessScreen(resolution=20)
        screen.kill_pixels([1, 5, 10, 15])
        frame = screen.render(np.ones(20), np.ones(20))
        self.assertEqual(frame.dead_pixels, 4)
        self.assertEqual(frame.active_pixels, 16)

    def test_alive_pixels_unaffected(self):
        """Alive pixels render normally when some are dead."""
        screen = ConsciousnessScreen(resolution=10)
        screen.set_temperature(0.0)
        screen.kill_pixels([3, 7])
        frame = screen.render(np.ones(10) * 0.5, np.ones(10))

        # Alive pixels: s=1, Γ_ch=0 → T=1 → content = 0.5
        for i in [0, 1, 2, 4, 5, 6, 8, 9]:
            self.assertAlmostEqual(frame.content[i], 0.5, places=6)

    def test_kill_out_of_range_safe(self):
        """Killing pixel out of range does nothing."""
        screen = ConsciousnessScreen(resolution=5)
        screen.kill_pixels([-1, 5, 100])
        # No crash, no dead pixels
        frame = screen.render(np.ones(5), np.ones(5))
        self.assertEqual(frame.dead_pixels, 0)

    def test_total_dead_counter(self):
        """total_dead tracks unique kills."""
        screen = ConsciousnessScreen(resolution=10)
        screen.kill_pixels([1, 3])
        screen.kill_pixels([3, 5])  # 3 already dead
        self.assertEqual(screen.total_dead, 3)  # 1, 3, 5


# ============================================================================
# Burn-in (PTSD)
# ============================================================================

class TestBurnIn(unittest.TestCase):
    """PTSD afterimage — trauma burns into screen."""

    def test_burn_creates_afterimage(self):
        """burn() increases burn_in_total."""
        screen = ConsciousnessScreen(resolution=10)
        screen.burn(np.ones(10) * 1.0, intensity=1.0)
        frame = screen.render(np.zeros(10), np.ones(10))
        self.assertGreater(frame.burn_in_total, 0.0)

    def test_burn_in_visible_without_signal(self):
        """Afterimage visible even with zero input signal."""
        screen = ConsciousnessScreen(resolution=5)
        screen.set_temperature(0.0)
        screen.burn(np.ones(5) * 1.0, intensity=1.0)

        # Render with zero signal — burn-in should still show
        frame = screen.render(np.zeros(5), np.ones(5))
        total_content = np.sum(np.abs(frame.content))
        self.assertGreater(total_content, 0.0,
                           msg="Burn-in should be visible with zero signal")

    def test_burn_in_decays_over_ticks(self):
        """Afterimage decreases over time."""
        screen = ConsciousnessScreen(resolution=5)
        screen.burn(np.ones(5) * 1.0, intensity=1.0)

        initial_burn = float(np.sum(screen._burn_in))

        for _ in range(100):
            screen.tick()

        final_burn = float(np.sum(screen._burn_in))
        self.assertLess(final_burn, initial_burn,
                        msg="Burn-in should decay over ticks")

    def test_burn_in_never_reaches_zero(self):
        """Minimum residual persists (scar never fully heals)."""
        screen = ConsciousnessScreen(resolution=5)
        screen.burn(np.ones(5) * 1.0, intensity=1.0)

        # Many ticks
        for _ in range(10000):
            screen.tick()

        # Each burned pixel should retain at least BURN_IN_RESIDUAL
        for i in range(5):
            self.assertGreaterEqual(
                screen._burn_in[i], BURN_IN_RESIDUAL * 0.99,
                msg=f"Pixel {i}: burn-in should never reach zero"
            )

    def test_multiple_burns_accumulate(self):
        """Sequential burns stack up."""
        screen = ConsciousnessScreen(resolution=5)
        screen.burn(np.ones(5) * 0.5, intensity=1.0)
        first = float(np.sum(screen._burn_in))

        screen.burn(np.ones(5) * 0.5, intensity=1.0)
        second = float(np.sum(screen._burn_in))

        self.assertGreater(second, first,
                           msg="Second burn should add to first")


# ============================================================================
# Pain Overload
# ============================================================================

class TestPainOverload(unittest.TestCase):
    """Pain → screen saturation."""

    def test_no_overload_below_threshold(self):
        """Pain below threshold → normal rendering."""
        screen = ConsciousnessScreen(resolution=5)
        screen.set_temperature(0.0)
        strengths = np.ones(5)
        values = np.ones(5) * 0.5

        frame = screen.render(values, strengths, pain_level=0.3)
        self.assertFalse(frame.is_overloaded)
        # Content = signal × T = 0.5 × 1.0 = 0.5
        for i in range(5):
            self.assertAlmostEqual(frame.content[i], 0.5, places=6)

    def test_overload_above_threshold(self):
        """Pain above threshold → overload flag."""
        screen = ConsciousnessScreen(resolution=5)
        frame = screen.render(np.ones(5), np.ones(5), pain_level=0.9)
        self.assertTrue(frame.is_overloaded)

    def test_overload_amplifies_content(self):
        """Pain overload amplifies pixel values."""
        screen = ConsciousnessScreen(resolution=5)
        screen.set_temperature(0.0)
        strengths = np.ones(5)
        values = np.ones(5) * 0.3

        frame_normal = screen.render(values, strengths, pain_level=0.0)
        frame_pain = screen.render(values, strengths, pain_level=0.9)

        # Pain content should be larger than normal
        for i in range(5):
            self.assertGreater(
                abs(frame_pain.content[i]), abs(frame_normal.content[i]),
                msg=f"Pixel {i}: pain should amplify content"
            )


# ============================================================================
# Dream Rendering
# ============================================================================

class TestDreamRendering(unittest.TestCase):
    """Sleep = camera off, screen still on."""

    def test_dream_marked(self):
        """Dream frame flagged as dreaming."""
        screen = ConsciousnessScreen(resolution=5)
        frame = screen.render_dream(
            np.ones(5) * 0.5, np.ones(5), dream_vividness=0.7
        )
        self.assertTrue(frame.is_dreaming)

    def test_dream_less_vivid_than_waking(self):
        """Dream brightness < waking brightness (sensory cable disconnected)."""
        screen = ConsciousnessScreen(resolution=10)

        # Waking: direct injection, full clarity
        wake_frame = screen.render(np.ones(10), np.ones(10), channel_gamma=0.0)

        # Dream: high channel loss
        dream_frame = screen.render_dream(np.ones(10), np.ones(10), dream_vividness=0.5)

        self.assertGreater(
            wake_frame.brightness, dream_frame.brightness,
            msg="Waking should be brighter than dreaming"
        )

    def test_dream_has_content(self):
        """Dream still renders content (screen is on even though camera is off)."""
        screen = ConsciousnessScreen(resolution=5)
        frame = screen.render_dream(np.ones(5) * 0.5, np.ones(5), dream_vividness=0.8)
        self.assertGreater(frame.brightness, 0.0)


# ============================================================================
# Stereo Depth (Dual Screen)
# ============================================================================

class TestStereoDepth(unittest.TestCase):
    """Binocular depth perception from differential Γ."""

    def test_identical_frames_zero_depth(self):
        """Same Γ on left and right → depth = 0 (object at infinity)."""
        screen_l = ConsciousnessScreen(resolution=10, screen_id="left")
        screen_r = ConsciousnessScreen(resolution=10, screen_id="right")

        fl = screen_l.render(np.ones(10), np.ones(10))
        fr = screen_r.render(np.ones(10), np.ones(10))

        stereo = compute_stereo_depth(fl, fr)
        self.assertAlmostEqual(stereo.depth_range, 0.0, places=6)
        self.assertAlmostEqual(stereo.convergence, 0.0, places=6)

    def test_different_screens_nonzero_depth(self):
        """Different Γ patterns → nonzero depth (parallax)."""
        screen_l = ConsciousnessScreen(resolution=10, screen_id="left")
        screen_r = ConsciousnessScreen(resolution=10, screen_id="right")

        # Left screen: healthy neurons
        fl = screen_l.render(np.ones(10), np.ones(10) * 1.0)
        # Right screen: some degraded neurons → different Γ pattern
        fr = screen_r.render(np.ones(10), np.ones(10) * 0.5)

        stereo = compute_stereo_depth(fl, fr)
        self.assertGreater(stereo.depth_range, 0.0,
                           msg="Different screens should produce depth")

    def test_depth_map_signed(self):
        """ΔΓ is signed — positive means closer to left."""
        screen_l = ConsciousnessScreen(resolution=5, screen_id="left")
        screen_r = ConsciousnessScreen(resolution=5, screen_id="right")

        # Left: degraded (Γ higher) → ΔΓ > 0
        fl = screen_l.render(np.ones(5), np.ones(5) * 0.3)
        fr = screen_r.render(np.ones(5), np.ones(5) * 1.0)

        stereo = compute_stereo_depth(fl, fr)
        for i in range(5):
            self.assertGreater(stereo.depth_map[i], 0.0)

    def test_fusion_quality_identical_is_one(self):
        """Identical screens → perfect fusion (quality=1)."""
        screen_l = ConsciousnessScreen(resolution=10, screen_id="left")
        screen_r = ConsciousnessScreen(resolution=10, screen_id="right")

        fl = screen_l.render(np.ones(10), np.ones(10) * 0.7)
        fr = screen_r.render(np.ones(10), np.ones(10) * 0.7)

        stereo = compute_stereo_depth(fl, fr)
        self.assertAlmostEqual(stereo.fusion_quality, 1.0, places=6)

    def test_stereo_frame_type(self):
        """compute_stereo_depth returns StereoFrame."""
        screen_l = ConsciousnessScreen(resolution=5, screen_id="left")
        screen_r = ConsciousnessScreen(resolution=5, screen_id="right")
        fl = screen_l.render(np.ones(5), np.ones(5))
        fr = screen_r.render(np.ones(5), np.ones(5))
        stereo = compute_stereo_depth(fl, fr)
        self.assertIsInstance(stereo, StereoFrame)


# ============================================================================
# C3: Signal Protocol
# ============================================================================

class TestSignalProtocol(unittest.TestCase):
    """C3: get_signal() returns ElectricalSignal with Z metadata."""

    def test_get_signal_returns_electrical_signal(self):
        """get_signal() returns valid ElectricalSignal."""
        screen = ConsciousnessScreen(resolution=10)
        screen.render(np.ones(10), np.ones(10))
        sig = screen.get_signal()
        self.assertIsInstance(sig, ElectricalSignal)

    def test_signal_carries_impedance(self):
        """Screen signal has impedance metadata."""
        screen = ConsciousnessScreen(resolution=10)
        screen.render(np.ones(10), np.ones(10))
        sig = screen.get_signal()
        self.assertGreater(sig.impedance, 0.0)

    def test_signal_waveform_length(self):
        """Signal waveform matches screen resolution."""
        screen = ConsciousnessScreen(resolution=20)
        screen.render(np.ones(20), np.ones(20))
        sig = screen.get_signal()
        self.assertEqual(len(sig.waveform), 20)

    def test_healthy_screen_lower_impedance(self):
        """Healthy screen (low Γ) → lower impedance than degraded screen."""
        screen_good = ConsciousnessScreen(resolution=10, screen_id="good")
        screen_bad = ConsciousnessScreen(resolution=10, screen_id="bad")

        screen_good.render(np.ones(10), np.ones(10) * 1.0)   # All healthy
        screen_bad.render(np.ones(10), np.ones(10) * 0.1)    # All degraded

        z_good = screen_good.get_signal().impedance
        z_bad = screen_bad.get_signal().impedance

        self.assertLess(z_good, z_bad,
                        msg="Healthy screen should have lower impedance")


# ============================================================================
# State & Introspection
# ============================================================================

class TestStateIntrospection(unittest.TestCase):
    """get_state() and get_stats() return complete information."""

    def test_get_state_keys(self):
        """State dict contains all expected keys."""
        screen = ConsciousnessScreen(resolution=5)
        screen.render(np.ones(5), np.ones(5))
        state = screen.get_state()

        expected_keys = {
            "screen_id", "resolution", "brightness", "mean_gamma",
            "dead_pixels", "active_pixels", "burn_in_total",
            "temperature", "is_dreaming", "total_renders",
            "total_burns", "frame_id",
        }
        self.assertTrue(
            expected_keys.issubset(set(state.keys())),
            msg=f"Missing keys: {expected_keys - set(state.keys())}"
        )

    def test_get_stats_alias(self):
        """get_stats() is alias for get_state()."""
        screen = ConsciousnessScreen(resolution=5)
        screen.render(np.ones(5), np.ones(5))
        self.assertEqual(screen.get_state(), screen.get_stats())

    def test_temperature_setter(self):
        """set_temperature updates internal temperature."""
        screen = ConsciousnessScreen(resolution=5)
        screen.set_temperature(37.0)
        state = screen.get_state()
        self.assertAlmostEqual(state["temperature"], 37.0, places=1)

    def test_temperature_clamps_negative(self):
        """Negative temperature clamped to 0."""
        screen = ConsciousnessScreen(resolution=5)
        screen.set_temperature(-10.0)
        self.assertEqual(screen._temperature, 0.0)


# ============================================================================
# Noise
# ============================================================================

class TestNoise(unittest.TestCase):
    """Johnson-Nyquist thermal noise at pixel terminals."""

    def test_zero_temp_zero_noise(self):
        """Temperature = 0 → no noise."""
        screen = ConsciousnessScreen(resolution=5)
        screen.set_temperature(0.0)
        frame = screen.render(np.zeros(5), np.ones(5))
        self.assertAlmostEqual(frame.noise_floor, 0.0, places=10)

    def test_nonzero_temp_produces_noise(self):
        """Temperature > 0 → noise present."""
        screen = ConsciousnessScreen(resolution=50)
        screen.set_temperature(37.0)
        frame = screen.render(np.zeros(50), np.ones(50))
        self.assertGreater(frame.noise_floor, 0.0)

    def test_higher_temp_more_noise(self):
        """Higher temperature → more noise."""
        screen_cool = ConsciousnessScreen(resolution=100)
        screen_hot = ConsciousnessScreen(resolution=100)

        screen_cool.set_temperature(20.0)
        screen_hot.set_temperature(80.0)

        frame_cool = screen_cool.render(np.zeros(100), np.ones(100))
        frame_hot = screen_hot.render(np.zeros(100), np.ones(100))

        # With enough pixels, the noise floor should be detectably different
        self.assertGreater(frame_hot.noise_floor, frame_cool.noise_floor * 0.5,
                           msg="Hotter screen should be noisier")


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases(unittest.TestCase):
    """Edge cases and robustness."""

    def test_empty_screen_renders(self):
        """Resolution=1 works."""
        screen = ConsciousnessScreen(resolution=1)
        frame = screen.render(np.array([0.5]), np.array([1.0]))
        self.assertEqual(frame.resolution, 1)

    def test_mismatched_shorter_input(self):
        """Shorter input than resolution → only first N pixels updated."""
        screen = ConsciousnessScreen(resolution=10)
        frame = screen.render(np.ones(5), np.ones(5))
        # First 5 should have signal, rest should be snow (T=0)
        self.assertGreater(frame.clarity[0], 0.0)
        self.assertEqual(frame.clarity[9], 0.0)

    def test_channel_gamma_one_kills_all(self):
        """Γ_ch = 1.0 → T_channel = 0 → no signal through (cable cut)."""
        screen = ConsciousnessScreen(resolution=5)
        screen.set_temperature(0.0)
        frame = screen.render(np.ones(5), np.ones(5), channel_gamma=1.0)
        for i in range(5):
            self.assertAlmostEqual(frame.clarity[i], 0.0, places=6)

    def test_get_brightness_before_render(self):
        """get_brightness() returns 0 before any render."""
        screen = ConsciousnessScreen(resolution=5)
        self.assertEqual(screen.get_brightness(), 0.0)

    def test_get_frame_before_render(self):
        """get_frame() returns None before any render."""
        screen = ConsciousnessScreen(resolution=5)
        self.assertIsNone(screen.get_frame())


if __name__ == "__main__":
    unittest.main()
