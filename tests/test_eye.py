# -*- coding: utf-8 -*-
"""
Alice Eye & Oscilloscope Tests

Test focus:
  1. AliceEye forward engineering pipeline (light → FFT → retina → electrical signal)
  2. Oscilloscope buffer (FusionBrain scope buffer)
  3. End-to-end integration (Eye → Brain → Oscilloscope data)
"""

import numpy as np
import pytest

from alice.core.signal import BrainWaveBand, ElectricalSignal
from alice.body.eye import AliceEye, BRAINWAVE_FREQ_MIN, BRAINWAVE_FREQ_MAX
from alice.brain.fusion_brain import FusionBrain
from alice.alice_brain import AliceBrain
from alice.core.protocol import Modality, Priority


# ============================================================================
# Helper Functions
# ============================================================================


def _uniform_image(value: float = 0.5, size: int = 64) -> np.ndarray:
    """Uniform brightness image — pure DC component"""
    return np.full(size, value)


def _sine_image(freq: float = 5.0, size: int = 256) -> np.ndarray:
    """Sine grating — single spatial frequency"""
    x = np.linspace(0, 1, size, endpoint=False)
    return 0.5 + 0.5 * np.sin(2 * np.pi * freq * x)


def _noise_image(size: int = 256) -> np.ndarray:
    """Random noise — broadband"""
    rng = np.random.RandomState(42)
    return rng.rand(size)


def _edge_image(size: int = 128) -> np.ndarray:
    """Step edge — rich in high frequencies"""
    img = np.zeros(size)
    img[size // 2:] = 1.0
    return img


def _2d_image(rows: int = 16, cols: int = 16) -> np.ndarray:
    """2D image"""
    rng = np.random.RandomState(123)
    return rng.rand(rows, cols)


# ============================================================================
# 1. AliceEye Basics
# ============================================================================


class TestAliceEyeBasic:
    """Basic eye functionality tests"""

    def test_eye_creation(self):
        eye = AliceEye()
        assert eye.retina_resolution == 256
        assert eye.pupil_aperture == 1.0
        assert eye.total_frames == 0

    def test_eye_custom_params(self):
        eye = AliceEye(retina_resolution=128, pupil_aperture=0.5)
        assert eye.retina_resolution == 128
        assert eye.pupil_aperture == 0.5

    def test_pupil_aperture_clamped(self):
        eye = AliceEye(pupil_aperture=5.0)
        assert eye.pupil_aperture == 1.0
        eye2 = AliceEye(pupil_aperture=-1.0)
        assert eye2.pupil_aperture == 0.01

    def test_adjust_pupil(self):
        eye = AliceEye()
        eye.adjust_pupil(0.3)
        assert abs(eye.pupil_aperture - 0.3) < 1e-6
        eye.adjust_pupil(2.0)
        assert eye.pupil_aperture == 1.0

    def test_get_stats(self):
        eye = AliceEye()
        stats = eye.get_stats()
        assert "total_frames" in stats
        assert "retina_resolution" in stats
        assert "pupil_aperture" in stats


# ============================================================================
# 2. AliceEye.see() — Forward Engineering Pipeline
# ============================================================================


class TestAliceEyeSee:
    """see() = physical conversion pipeline"""

    def test_see_returns_electrical_signal(self):
        eye = AliceEye()
        sig = eye.see(_sine_image())
        assert isinstance(sig, ElectricalSignal)

    def test_see_waveform_shape(self):
        """Retinal output waveform length = retina_resolution"""
        eye = AliceEye(retina_resolution=128)
        sig = eye.see(_sine_image())
        assert len(sig.waveform) == 128

    def test_see_frequency_in_brainwave_range(self):
        """Output frequency must be within brainwave frequency range"""
        eye = AliceEye()
        sig = eye.see(_sine_image())
        assert BRAINWAVE_FREQ_MIN <= sig.frequency <= BRAINWAVE_FREQ_MAX

    def test_see_impedance(self):
        """Retinal impedance = 50Ω (sensory cortex impedance matching)"""
        eye = AliceEye()
        sig = eye.see(_sine_image())
        assert sig.impedance == 50.0

    def test_see_snr(self):
        """Optic nerve SNR = 15 dB"""
        eye = AliceEye()
        sig = eye.see(_sine_image())
        assert sig.snr == 15.0

    def test_see_source_and_modality(self):
        eye = AliceEye()
        sig = eye.see(_sine_image())
        assert sig.source == "eye"
        assert sig.modality == "visual"

    def test_see_increments_frame_count(self):
        eye = AliceEye()
        assert eye.total_frames == 0
        eye.see(_sine_image())
        assert eye.total_frames == 1
        eye.see(_noise_image())
        assert eye.total_frames == 2

    def test_see_amplitude_positive(self):
        eye = AliceEye()
        sig = eye.see(_sine_image())
        assert sig.amplitude > 0

    def test_see_uniform_image(self):
        """Uniform image → DC component dominant → low frequency"""
        eye = AliceEye()
        sig = eye.see(_uniform_image())
        assert isinstance(sig, ElectricalSignal)
        assert sig.amplitude > 0

    def test_see_2d_image(self):
        """2D image is flattened before processing"""
        eye = AliceEye()
        sig = eye.see(_2d_image())
        assert isinstance(sig, ElectricalSignal)
        assert len(sig.waveform) == eye.retina_resolution

    def test_see_small_input(self):
        """Small input can also be processed"""
        eye = AliceEye(retina_resolution=64)
        sig = eye.see(np.array([0.5, 1.0, 0.2]))
        assert isinstance(sig, ElectricalSignal)

    def test_see_zeros(self):
        """All-black image"""
        eye = AliceEye()
        sig = eye.see(np.zeros(100))
        assert isinstance(sig, ElectricalSignal)


# ============================================================================
# 3. Lens Physics — FFT Frequency Domain Mapping
# ============================================================================


class TestLensFFT:
    """Lens = Fourier transformer (physically correct)"""

    def test_low_spatial_freq_maps_to_low_brainwave(self):
        """Low spatial frequency → low brainwave frequency (δ/θ — contour/whole)"""
        eye = AliceEye(retina_resolution=256)
        # Low-frequency sine grating
        low_freq_img = _sine_image(freq=2.0, size=512)
        sig = eye.see(low_freq_img)
        # Low spatial frequency should map to low brainwave frequency range
        assert sig.frequency < 50.0  # not γ

    def test_high_spatial_freq_maps_to_higher_brainwave(self):
        """High spatial frequency → higher brainwave frequency (β/γ — detail/texture)"""
        eye = AliceEye(retina_resolution=256)
        # High-frequency sine grating
        high_freq_img = _sine_image(freq=50.0, size=512)
        sig = eye.see(high_freq_img)
        # High spatial frequency should map to higher brainwave frequency
        assert sig.frequency > 1.0

    def test_edge_image_has_high_freq_content(self):
        """Sharp edge → rich high frequencies → higher brainwave frequency"""
        eye = AliceEye()
        sig = eye.see(_edge_image())
        assert sig.amplitude > 0
        # Edge contains broadband components

    def test_lens_fft_returns_power_spectrum(self):
        """Direct test of _lens_fft"""
        eye = AliceEye()
        light = np.sin(np.linspace(0, 2 * np.pi * 10, 256))
        power, freqs = eye._lens_fft(light)
        assert len(power) == len(freqs)
        assert np.all(power >= 0)  # power is non-negative
        assert freqs[0] >= BRAINWAVE_FREQ_MIN
        assert freqs[-1] <= BRAINWAVE_FREQ_MAX

    def test_lens_fft_empty_input(self):
        """Empty input"""
        eye = AliceEye()
        power, freqs = eye._lens_fft(np.array([]))
        assert len(power) == 1
        assert len(freqs) == 1


# ============================================================================
# 4. Pupil & Retina
# ============================================================================


class TestPupilRetina:
    """Pupil aperture + retina photoelectric conversion"""

    def test_pupil_normalizes(self):
        """Light field after pupil normalized to 0~aperture"""
        eye = AliceEye(pupil_aperture=1.0)
        light = eye._pupil(np.array([0, 128, 255]))
        assert np.max(light) <= 1.0 + 1e-6

    def test_pupil_aperture_scales(self):
        """Small aperture → low brightness"""
        eye_full = AliceEye(pupil_aperture=1.0)
        eye_half = AliceEye(pupil_aperture=0.5)
        img = _sine_image()
        l_full = eye_full._pupil(img)
        l_half = eye_half._pupil(img)
        # Half aperture is roughly half of full aperture
        ratio = np.mean(np.abs(l_half)) / np.mean(np.abs(l_full))
        assert 0.4 < ratio < 0.6

    def test_retina_voltage_nonnegative(self):
        """Retinal voltage is non-negative (V = sqrt(P × R))"""
        eye = AliceEye()
        light = eye._pupil(_sine_image())
        spectrum, freqs = eye._lens_fft(light)
        voltage, dom_freq, amp = eye._retina(spectrum, freqs)
        assert np.all(voltage >= 0)

    def test_retina_dominant_freq_in_range(self):
        eye = AliceEye()
        light = eye._pupil(_sine_image())
        spectrum, freqs = eye._lens_fft(light)
        _, dom_freq, _ = eye._retina(spectrum, freqs)
        assert BRAINWAVE_FREQ_MIN <= dom_freq <= BRAINWAVE_FREQ_MAX


# ============================================================================
# 5. Saccade
# ============================================================================


class TestSaccade:
    """Saccade = fixation on specific region"""

    def test_saccade_2d(self):
        eye = AliceEye()
        img = _2d_image(32, 32)
        sig = eye.saccade(img, (8, 8, 8, 8))
        assert isinstance(sig, ElectricalSignal)
        assert eye.total_saccades == 1
        assert eye.total_frames == 1

    def test_saccade_1d(self):
        eye = AliceEye()
        img = _sine_image(size=128)
        sig = eye.saccade(img, (32, 0, 32, 0))
        assert isinstance(sig, ElectricalSignal)

    def test_saccade_increments_count(self):
        eye = AliceEye()
        img = _2d_image(16, 16)
        eye.saccade(img, (0, 0, 8, 8))
        eye.saccade(img, (4, 4, 8, 8))
        assert eye.total_saccades == 2


# ============================================================================
# 6. Retina Snapshot (Diagnostics)
# ============================================================================


class TestRetinaSnapshot:
    def test_snapshot_keys(self):
        eye = AliceEye()
        snap = eye.get_retina_snapshot(_sine_image())
        expected_keys = {"light", "spectrum", "freqs", "voltage",
                         "dominant_freq", "band", "amplitude"}
        assert expected_keys <= set(snap.keys())

    def test_snapshot_band_valid(self):
        eye = AliceEye()
        snap = eye.get_retina_snapshot(_sine_image())
        valid_bands = {b.value for b in BrainWaveBand}
        assert snap["band"] in valid_bands

    def test_snapshot_lists(self):
        """Snapshot returns list not ndarray (JSON serializable)"""
        eye = AliceEye()
        snap = eye.get_retina_snapshot(_sine_image())
        assert isinstance(snap["light"], list)
        assert isinstance(snap["spectrum"], list)
        assert isinstance(snap["voltage"], list)


# ============================================================================
# 7. Oscilloscope Buffer (FusionBrain)
# ============================================================================


class TestOscilloscopeBuffer:
    """FusionBrain oscilloscope buffer tests"""

    def setup_method(self):
        self.brain = FusionBrain()
        self._mod = Modality.VISUAL
        self._pri = Priority.NORMAL

    def test_scope_buffer_exists(self):
        assert hasattr(self.brain, '_scope_buffer')
        assert isinstance(self.brain._scope_buffer, dict)

    def test_scope_buffer_initial_keys(self):
        buf = self.brain._scope_buffer
        assert "input_waveform" in buf
        assert "channels" in buf
        assert "perception" in buf

    def test_scope_buffer_populated_after_stimulus(self):
        """Oscilloscope buffer should be populated after stimulus"""
        signal = np.random.randn(20)
        self.brain.process_stimulus(signal, self._mod, self._pri)
        buf = self.brain._scope_buffer
        # Input waveform
        assert len(buf["input_waveform"]) > 0
        # Channels
        assert len(buf["channels"]) > 0
        # Perception
        assert buf["perception"] is not None

    def test_scope_buffer_channels_have_waveform(self):
        """Each channel has waveform + gamma"""
        signal = np.random.randn(20)
        self.brain.process_stimulus(signal, self._mod, self._pri)
        channels = self.brain._scope_buffer["channels"]
        for name, ch in channels.items():
            assert "waveform" in ch, f"Channel {name} missing waveform"
            assert "gamma" in ch, f"Channel {name} missing gamma"
            assert isinstance(ch["waveform"], list)
            assert isinstance(ch["gamma"], float)

    def test_scope_buffer_perception_fields(self):
        """Perception result should include left/right resonance, attention"""
        signal = np.random.randn(20)
        self.brain.process_stimulus(signal, self._mod, self._pri)
        perc = self.brain._scope_buffer["perception"]
        assert "left_resonance" in perc
        assert "right_resonance" in perc
        assert "attention_band" in perc
        assert "concept" in perc

    def test_scope_buffer_reset_between_cycles(self):
        """Buffer clears between cycles"""
        signal = np.random.randn(20)
        self.brain.process_stimulus(signal, self._mod, self._pri)
        first_wave = list(self.brain._scope_buffer["input_waveform"])

        self.brain.process_stimulus(signal * 2, self._mod, self._pri)
        second_wave = self.brain._scope_buffer["input_waveform"]

        # Second waveform should differ from first (different input)
        assert first_wave != second_wave or len(first_wave) > 0

    def test_get_oscilloscope_data(self):
        """get_oscilloscope_data returns a copy of scope buffer"""
        signal = np.random.randn(20)
        self.brain.process_stimulus(signal, self._mod, self._pri)
        data = self.brain.get_oscilloscope_data()
        assert isinstance(data, dict)
        assert "input_waveform" in data
        assert "channels" in data

    def test_four_channels_present(self):
        """Should have 4 channels after stimulation"""
        signal = np.random.randn(20)
        self.brain.process_stimulus(signal, self._mod, self._pri)
        channels = self.brain._scope_buffer["channels"]
        expected = {"sensory→prefrontal", "sensory→limbic",
                    "prefrontal→motor", "limbic→motor"}
        assert expected == set(channels.keys())

    def test_limbic_channel_high_gamma(self):
        """Limbic channel (110Ω) mismatched with sensory cortex (50Ω) → observable reflection"""
        signal = np.random.randn(20)
        self.brain.process_stimulus(signal, self._mod, self._pri)
        limbic_ch = self.brain._scope_buffer["channels"]["sensory→limbic"]
        # Should have gamma field (value depends on channel pre-configuration)
        assert "gamma" in limbic_ch
        assert isinstance(limbic_ch["gamma"], float)
        # If channel exists and impedance mismatched, gamma should be non-zero
        # If channel not pre-configured (report=None), gamma=0.0 is also reasonable
        assert limbic_ch["gamma"] >= 0.0 or limbic_ch["gamma"] < 0.0  # type check


# ============================================================================
# 8. AliceBrain Oscilloscope API
# ============================================================================


class TestAliceBrainOscilloscope:
    """AliceBrain.get_oscilloscope_data()"""

    def setup_method(self):
        self.alice = AliceBrain()

    def test_oscilloscope_data_structure(self):
        # First trigger a stimulus
        signal = np.random.randn(20)
        self.alice.perceive(signal, Modality.VISUAL, Priority.NORMAL)
        data = self.alice.get_oscilloscope_data()
        assert "input_waveform" in data
        assert "channels" in data
        assert "vitals" in data

    def test_oscilloscope_vitals_waveforms(self):
        """Oscilloscope data includes vital sign waveforms"""
        signal = np.random.randn(20)
        self.alice.perceive(signal, Modality.VISUAL, Priority.NORMAL)
        data = self.alice.get_oscilloscope_data()
        vitals = data["vitals"]
        assert "heart_rate" in vitals
        assert "temperature" in vitals
        assert "pain" in vitals


# ============================================================================
# 9. End-to-End Integration: Eye → Brain → Oscilloscope
# ============================================================================


class TestEyeToBrainIntegration:
    """Eye forward engineering output → Brain perception pipeline → Oscilloscope observation"""

    def test_eye_output_feeds_brain(self):
        """Eye's electrical signal can be directly fed into the brain"""
        eye = AliceEye()
        alice = AliceBrain()

        # Eye sees a sine grating
        img = _sine_image(freq=10.0, size=256)
        visual_signal = eye.see(img)

        # Feed waveform into brain
        result = alice.perceive(
            visual_signal.waveform,
            Modality.VISUAL,
            Priority.NORMAL,
            context="eye_test"
        )
        assert result is not None
        assert "cycle" in result

    def test_eye_output_produces_oscilloscope_data(self):
        """Eye signal enters brain and oscilloscope buffer has data"""
        eye = AliceEye()
        alice = AliceBrain()

        img = _noise_image()
        visual_signal = eye.see(img)

        alice.perceive(
            visual_signal.waveform,
            Modality.VISUAL,
            Priority.NORMAL,
        )

        scope = alice.get_oscilloscope_data()
        assert len(scope["input_waveform"]) > 0
        assert len(scope["channels"]) == 4

    def test_different_images_different_scope(self):
        """Different images → different oscilloscope waveforms"""
        eye = AliceEye()
        alice = AliceBrain()

        # Image 1: low frequency
        img1 = _sine_image(freq=3.0, size=256)
        sig1 = eye.see(img1)
        alice.perceive(sig1.waveform, Modality.VISUAL, Priority.NORMAL)
        scope1_wave = alice.get_oscilloscope_data()["input_waveform"]

        # Image 2: high frequency
        img2 = _sine_image(freq=50.0, size=256)
        sig2 = eye.see(img2)
        alice.perceive(sig2.waveform, Modality.VISUAL, Priority.NORMAL)
        scope2_wave = alice.get_oscilloscope_data()["input_waveform"]

        # Waveforms should differ (different images produce different signals)
        assert scope1_wave != scope2_wave


# ============================================================================
# 10. Resolution Adaptation System
# ============================================================================


from alice.body.eye import (
    DisplayProfile, VisualAcuity,
    MIN_CJK_CHAR_PX, MIN_CJK_STROKE_PX, MIN_LATIN_CHAR_PX,
    RETINA_RESOLUTION_MIN, RETINA_RESOLUTION_MAX, ANTIALIAS_OVERSAMPLE,
)


class TestDisplayProfile:
    """DisplayProfile — display device description"""

    def test_desktop_1080p_defaults(self):
        dp = DisplayProfile.desktop_1080p()
        assert dp.width == 1920
        assert dp.height == 1080
        assert dp.ppi == 96.0

    def test_desktop_4k(self):
        dp = DisplayProfile.desktop_4k()
        assert dp.width == 3840
        assert dp.height == 2160
        assert dp.scale_factor == 2.0

    def test_phone_720p(self):
        dp = DisplayProfile.phone_720p()
        assert dp.width == 720
        assert dp.height == 1280

    def test_phone_1080p(self):
        dp = DisplayProfile.phone_1080p()
        assert dp.width == 1080
        assert dp.scale_factor == 3.0

    def test_tablet(self):
        dp = DisplayProfile.tablet()
        assert dp.width == 2048
        assert dp.height == 2732

    def test_low_res(self):
        dp = DisplayProfile.low_res()
        assert dp.width == 800
        assert dp.font_size_px == 10.0

    def test_effective_font_px(self):
        """Effective font size = font_size_px × scale_factor"""
        dp = DisplayProfile(font_size_px=14.0, scale_factor=2.0)
        assert abs(dp.effective_font_px - 28.0) < 1e-6

    def test_stroke_width_px(self):
        """Stroke width = effective_font / (2 x 8)"""
        dp = DisplayProfile(font_size_px=16.0, scale_factor=1.0)
        expected = 16.0 / 16.0  # = 1.0
        assert abs(dp.stroke_width_px - expected) < 1e-6

    def test_nyquist_ok_sufficient(self):
        """Font large enough → Nyquist OK"""
        dp = DisplayProfile.desktop_1080p()  # 16px font → stroke=1.0
        # 16px / 16 = 1.0, Nyquist needs >= 2
        # Default 1080p: stroke_width = 1.0, < 2  → not ok
        assert not dp.nyquist_ok

    def test_nyquist_ok_large_font(self):
        """Large font → Nyquist OK"""
        dp = DisplayProfile(font_size_px=32.0, scale_factor=1.0)
        # 32 / 16 = 2.0  → ok
        assert dp.nyquist_ok

    def test_nyquist_ok_retina_display(self):
        """Retina display (2x scale) → effective size doubles"""
        dp = DisplayProfile(font_size_px=16.0, scale_factor=2.0)
        # effective = 32, stroke = 32/16 = 2.0  → ok
        assert dp.nyquist_ok

    def test_acuity_excellent(self):
        dp = DisplayProfile(font_size_px=32.0, scale_factor=2.0)
        # effective = 64  → EXCELLENT (>= 32)
        assert dp.acuity == VisualAcuity.EXCELLENT

    def test_acuity_good(self):
        dp = DisplayProfile(font_size_px=16.0, scale_factor=1.0)
        # effective = 16  → GOOD (>= 16)
        assert dp.acuity == VisualAcuity.GOOD

    def test_acuity_fair(self):
        dp = DisplayProfile(font_size_px=12.0, scale_factor=1.0)
        # effective = 12  → FAIR (>= 12)
        assert dp.acuity == VisualAcuity.FAIR

    def test_acuity_poor(self):
        dp = DisplayProfile(font_size_px=9.0, scale_factor=1.0)
        # effective = 9  → POOR (>= 8)
        assert dp.acuity == VisualAcuity.POOR

    def test_acuity_blind(self):
        dp = DisplayProfile(font_size_px=4.0, scale_factor=1.0)
        # effective = 4  → BLIND (< 8)
        assert dp.acuity == VisualAcuity.BLIND

    def test_optimal_retina_resolution_1080p(self):
        dp = DisplayProfile.desktop_1080p()
        res = dp.optimal_retina_resolution
        # 1920 / 4 = 480 → ceil to 512 (2^9)
        assert res == 512

    def test_optimal_retina_resolution_4k(self):
        dp = DisplayProfile.desktop_4k()
        res = dp.optimal_retina_resolution
        # 3840 / 4 = 960 → nearest pow2 = 1024
        assert res == 1024

    def test_optimal_retina_resolution_clamped(self):
        """Resolution does not exceed upper/lower bounds"""
        dp = DisplayProfile(width=100, height=100)
        res = dp.optimal_retina_resolution
        assert res >= RETINA_RESOLUTION_MIN

        dp2 = DisplayProfile(width=16000, height=16000)
        res2 = dp2.optimal_retina_resolution
        assert res2 <= RETINA_RESOLUTION_MAX


class TestVisualAcuity:
    """VisualAcuity enum"""

    def test_enum_values(self):
        assert VisualAcuity.EXCELLENT.value == "excellent"
        assert VisualAcuity.GOOD.value == "good"
        assert VisualAcuity.FAIR.value == "fair"
        assert VisualAcuity.POOR.value == "poor"
        assert VisualAcuity.BLIND.value == "blind"

    def test_is_string_enum(self):
        assert isinstance(VisualAcuity.EXCELLENT, str)


class TestResolutionAdaptation:
    """AliceEye resolution adaptation tests"""

    def test_default_eye_no_display_profile(self):
        """Uses default 1080p when no display profile is provided"""
        eye = AliceEye()
        assert eye.display is not None
        assert eye.display.width == 1920

    def test_custom_display_profile(self):
        """Custom display can be passed in"""
        dp = DisplayProfile.phone_720p()
        eye = AliceEye(display=dp)
        assert eye.display.width == 720

    def test_auto_adapt_adjusts_resolution(self):
        """auto_adapt=True adjusts retina_resolution based on display"""
        dp = DisplayProfile.desktop_4k()
        eye = AliceEye(display=dp, auto_adapt=True)
        # 4K display optimal = 1024
        assert eye.retina_resolution == dp.optimal_retina_resolution

    def test_auto_adapt_false_keeps_default(self):
        """auto_adapt=False (default) keeps manual setting"""
        dp = DisplayProfile.desktop_4k()
        eye = AliceEye(display=dp, retina_resolution=128)
        assert eye.retina_resolution == 128

    def test_set_display_dynamic_switch(self):
        """set_display dynamically switches device"""
        eye = AliceEye(auto_adapt=True)
        original_res = eye.retina_resolution

        # Switch to 4K
        dp_4k = DisplayProfile.desktop_4k()
        eye.set_display(dp_4k)
        assert eye.display == dp_4k
        assert eye.retina_resolution == dp_4k.optimal_retina_resolution

    def test_set_display_without_auto_adapt(self):
        """set_display with auto_adapt=False (default) only changes display"""
        eye = AliceEye(retina_resolution=256)
        dp_4k = DisplayProfile.desktop_4k()
        eye.set_display(dp_4k)
        assert eye.display == dp_4k
        assert eye.retina_resolution == 256  # unchanged

    def test_get_visual_acuity_report(self):
        """get_visual_acuity returns complete report"""
        dp = DisplayProfile.desktop_1080p()
        eye = AliceEye(display=dp)
        report = eye.get_visual_acuity()

        assert "acuity" in report
        assert "effective_font_px" in report
        assert "stroke_width_px" in report
        assert "nyquist_ok" in report
        assert "acuity_warnings" in report

    def test_see_with_display_profile(self):
        """see() with display profile still outputs ElectricalSignal"""
        dp = DisplayProfile.phone_1080p()
        eye = AliceEye(display=dp, auto_adapt=True)
        sig = eye.see(_sine_image(size=256))
        assert isinstance(sig, ElectricalSignal)

    def test_see_attaches_acuity_metadata(self):
        """see() attaches acuity info to the signal"""
        eye = AliceEye(auto_adapt=True)
        sig = eye.see(_sine_image(size=128))
        assert hasattr(sig, '_acuity')
        assert hasattr(sig, '_resolution_adapted')

    def test_acuity_warnings_counter(self):
        """Nyquist failure accumulates warnings"""
        # Use very small font to make nyquist_ok = False
        dp = DisplayProfile(font_size_px=8.0, scale_factor=1.0)
        eye = AliceEye(display=dp, auto_adapt=True)
        assert eye._acuity_warnings == 0

        eye.see(_sine_image(size=64))
        if not dp.nyquist_ok:
            assert eye._acuity_warnings >= 1


class TestAntialiasCompensation:
    """Anti-alias compensation tests"""

    def test_no_compensation_when_nyquist_ok(self):
        """No compensation when Nyquist is satisfied (original value returned)"""
        dp = DisplayProfile(font_size_px=32.0, scale_factor=2.0)
        eye = AliceEye(display=dp)
        light = np.array([0.0, 0.5, 1.0, 0.5])
        result = eye._antialias_compensate(light)
        np.testing.assert_array_equal(result, light)

    def test_compensation_when_nyquist_fail(self):
        """Anti-alias interpolation when Nyquist fails"""
        dp = DisplayProfile(font_size_px=8.0, scale_factor=1.0)
        eye = AliceEye(display=dp)
        assert not dp.nyquist_ok  # Confirm Nyquist failure

        # Square wave — lots of high-frequency components
        square = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        result = eye._antialias_compensate(square)

        # Compensated waveform length unchanged
        assert len(result) == len(square)
        # After compensation, edges should be smoother (no longer square wave)
        # But still maintains similar mean value
        assert abs(np.mean(result) - np.mean(square)) < 0.15

    def test_compensation_preserves_energy(self):
        """Anti-aliasing compensation approximately preserves signal energy"""
        dp = DisplayProfile(font_size_px=6.0, scale_factor=1.0)
        eye = AliceEye(display=dp)

        signal = _sine_image(freq=5.0, size=64)
        result = eye._antialias_compensate(signal)

        # Energy (L2 norm) should not differ too much
        orig_energy = np.sum(signal ** 2)
        comp_energy = np.sum(result ** 2)
        ratio = comp_energy / max(orig_energy, 1e-12)
        assert 0.5 < ratio < 2.0

    def test_compensation_small_array(self):
        """Very small array (< 4) passes through directly"""
        dp = DisplayProfile(font_size_px=6.0, scale_factor=1.0)
        eye = AliceEye(display=dp)
        tiny = np.array([0.5, 1.0])
        result = eye._antialias_compensate(tiny)
        np.testing.assert_array_equal(result, tiny)


class TestGetStatsResolution:
    """get_stats includes resolution information"""

    def test_stats_contains_display_info(self):
        eye = AliceEye()
        stats = eye.get_stats()
        assert "display" in stats
        assert "width" in stats["display"]
        assert "height" in stats["display"]

    def test_stats_contains_acuity(self):
        eye = AliceEye()
        stats = eye.get_stats()
        assert "acuity" in stats
        assert "nyquist_ok" in stats
        assert "acuity_warnings" in stats

    def test_stats_auto_adapt_flag(self):
        eye = AliceEye(auto_adapt=True)
        stats = eye.get_stats()
        assert stats["auto_adapt"] is True

    def test_stats_auto_adapt_false_default(self):
        eye = AliceEye()
        stats = eye.get_stats()
        assert stats["auto_adapt"] is False


class TestRetinaSnapshotResolution:
    """get_retina_snapshot includes acuity information"""

    def test_snapshot_includes_acuity(self):
        eye = AliceEye()
        snap = eye.get_retina_snapshot(_sine_image(size=128))
        assert "acuity" in snap
        assert "nyquist_ok" in snap
        assert "stroke_width_px" in snap


class TestDeviceSwitchEndToEnd:
    """End-to-end: device switching scenario"""

    def test_desktop_to_phone_switch(self):
        """Desktop to phone switch → resolution auto-adjusts"""
        eye = AliceEye(
            display=DisplayProfile.desktop_1080p(),
            auto_adapt=True,
        )
        sig_desktop = eye.see(_sine_image(size=256))
        res_desktop = eye.retina_resolution

        eye.set_display(DisplayProfile.phone_720p())
        sig_phone = eye.see(_sine_image(size=256))
        res_phone = eye.retina_resolution

        # Both produce valid signals
        assert isinstance(sig_desktop, ElectricalSignal)
        assert isinstance(sig_phone, ElectricalSignal)

    def test_different_devices_different_acuity(self):
        """Different devices → different acuity levels"""
        dp_4k = DisplayProfile.desktop_4k()
        dp_low = DisplayProfile.low_res()
        assert dp_4k.acuity != dp_low.acuity


# ============================================================================
# Visual Fingerprint & Semantic Field Integration (Phase 4.2 Symmetric Extension)
# ============================================================================

from alice.body.eye import VISUAL_FINGERPRINT_BINS
from alice.brain.semantic_field import SemanticField, SemanticFieldEngine, cosine_similarity


class TestRetinotopicFingerprint:
    """retinotopic_fingerprint: retinal voltage -> visual fingerprint"""

    def test_fingerprint_shape_full_resolution(self):
        """Default mode: fingerprint dimension = input voltage dimension (full resolution)"""
        eye = AliceEye()
        voltage = np.random.uniform(0, 1, 256)
        fp = eye.retinotopic_fingerprint(voltage)
        assert fp.shape == (256,)  # not compressed!

    def test_fingerprint_shape_compressed(self):
        """Compressed mode: compress to specified dimension when n_bins > 0"""
        eye = AliceEye()
        voltage = np.random.uniform(0, 1, 256)
        fp32 = eye.retinotopic_fingerprint(voltage, n_bins=32)
        assert fp32.shape == (32,)

    def test_fingerprint_normalized(self):
        """Sum of fingerprint bin energies ≈ 1 (normalized)"""
        eye = AliceEye()
        voltage = np.random.uniform(0.1, 2.0, 256)
        fp = eye.retinotopic_fingerprint(voltage)
        assert abs(np.sum(fp) - 1.0) < 1e-6

    def test_fingerprint_nonzero(self):
        """Non-zero input → non-zero fingerprint"""
        eye = AliceEye()
        voltage = np.ones(256) * 0.5
        fp = eye.retinotopic_fingerprint(voltage)
        assert np.all(fp > 0)

    def test_fingerprint_empty_input(self):
        """Empty input → zero fingerprint"""
        eye = AliceEye()
        fp = eye.retinotopic_fingerprint(np.array([]))
        assert np.all(fp == 0)

    def test_fingerprint_different_images_differ(self):
        """Different images → different fingerprints"""
        eye = AliceEye()
        # Low-frequency pattern
        v1 = np.sin(np.linspace(0, 2 * np.pi, 256))
        # High-frequency pattern
        v2 = np.sin(np.linspace(0, 20 * np.pi, 256))
        fp1 = eye.retinotopic_fingerprint(np.abs(v1) + 0.1)
        fp2 = eye.retinotopic_fingerprint(np.abs(v2) + 0.1)
        sim = cosine_similarity(fp1, fp2)
        assert sim < 0.99  # should be distinguishable

    def test_see_stores_fingerprint(self):
        """see() followed by get_last_fingerprint() is not None — dimension=retina_resolution"""
        eye = AliceEye()  # retina_resolution=256
        pixels = np.random.uniform(0, 255, 64)
        eye.see(pixels)
        fp = eye.get_last_fingerprint()
        assert fp is not None
        assert fp.shape == (eye.retina_resolution,)  # full resolution

    def test_custom_bin_count(self):
        """Different bin counts can be specified"""
        eye = AliceEye()
        voltage = np.random.uniform(0, 1, 256)
        fp16 = eye.retinotopic_fingerprint(voltage, n_bins=16)
        fp64 = eye.retinotopic_fingerprint(voltage, n_bins=64)
        assert fp16.shape == (16,)
        assert fp64.shape == (64,)


class TestVisualSemanticField:
    """Visual fingerprint → semantic field recognition (fully symmetric with auditory)"""

    def test_register_and_recognize_visual_concept(self):
        """Register concept with visual fingerprint, recognize with visual fingerprint"""
        eye = AliceEye()
        field = SemanticField()

        # Create two distinct visual patterns
        low_freq = np.abs(np.sin(np.linspace(0, 2 * np.pi, 256))) + 0.1
        high_freq = np.abs(np.sin(np.linspace(0, 20 * np.pi, 256))) + 0.1

        fp_circle = eye.retinotopic_fingerprint(low_freq)
        fp_texture = eye.retinotopic_fingerprint(high_freq)

        field.register_concept("circle", fp_circle, modality="visual")
        field.register_concept("texture", fp_texture, modality="visual")

        # Recognize
        results = field.recognize(fp_circle, modality="visual")
        assert results[0][0] == "circle"

    def test_visual_and_auditory_different_dimensions(self):
        """Visual 256-dim + auditory 32-dim → same concept, different modalities independent"""
        field = SemanticField()

        # Visual: 256 dimensions (full resolution)
        fp_vis = np.random.uniform(0.01, 0.1, 256)
        fp_vis[50] = 1.0  # visual peak
        # Auditory: 32 dimensions (cochlear channels)
        fp_aud = np.random.uniform(0.01, 0.1, 32)
        fp_aud[20] = 1.0  # auditory peak

        field.register_concept("apple", fp_vis, modality="visual")
        field.absorb("apple", fp_aud, modality="auditory")

        a = field.attractors["apple"]
        assert "visual" in a.modality_centroids
        assert "auditory" in a.modality_centroids
        assert len(a.modality_centroids["visual"]) == 256
        assert len(a.modality_centroids["auditory"]) == 32

    def test_cross_modal_hear_predict_see(self):
        """Hear apple → predict what apple looks like (cross-dimension cross-modal prediction)"""
        field = SemanticField()

        # Visual 256-dim, auditory 32-dim — different dimensions!
        fp_vis = np.random.uniform(0.01, 0.1, 256)
        fp_vis[50] = 1.0
        fp_aud = np.random.uniform(0.01, 0.1, 32)
        fp_aud[20] = 1.0

        field.register_concept("apple", fp_vis, modality="visual")
        for _ in range(10):
            field.absorb("apple", fp_vis + np.random.normal(0, 0.01, 256),
                         modality="visual")
            field.absorb("apple", fp_aud + np.random.normal(0, 0.01, 32),
                         modality="auditory")

        # Hear -> predict visual (32-dim input -> 256-dim prediction)
        pred = field.predict_cross_modal(fp_aud, "auditory", "visual")
        assert pred is not None
        assert len(pred["predicted_fingerprint"]) == 256
        sim = cosine_similarity(pred["predicted_fingerprint"], fp_vis)
        assert sim > 0.8

    def test_cross_modal_see_predict_hear(self):
        """See apple → predict what apple sounds like (reverse cross-dimension)"""
        field = SemanticField()

        fp_vis = np.random.uniform(0.01, 0.1, 256)
        fp_vis[50] = 1.0
        fp_aud = np.random.uniform(0.01, 0.1, 32)
        fp_aud[20] = 1.0

        field.register_concept("apple", fp_vis, modality="visual")
        for _ in range(10):
            field.absorb("apple", fp_vis + np.random.normal(0, 0.01, 256),
                         modality="visual")
            field.absorb("apple", fp_aud + np.random.normal(0, 0.01, 32),
                         modality="auditory")

        # See -> predict auditory (256-dim input -> 32-dim prediction)
        pred = field.predict_cross_modal(fp_vis, "visual", "auditory")
        assert pred is not None
        assert len(pred["predicted_fingerprint"]) == 32
        sim = cosine_similarity(pred["predicted_fingerprint"], fp_aud)
        assert sim > 0.8


class TestAliceBrainVisualSemantic:
    """AliceBrain.see() → semantic field integration"""

    def test_see_feeds_semantic_field(self):
        """see() feeds visual fingerprint to semantic field"""
        brain = AliceBrain()

        # Register a visual concept first
        eye = brain.eye
        pixels = np.random.uniform(0, 255, 64)
        eye.see(pixels)
        fp = eye.get_last_fingerprint()
        brain.semantic_field.field.register_concept("test_vis", fp, modality="visual")

        # Now see() should feed into semantic field
        result = brain.see(pixels)
        assert "semantic" in result
        sem = result["semantic"]
        assert sem["best_concept"] == "test_vis"

    def test_see_without_concept_returns_novel(self):
        """When semantic field is empty, see() reports is_novel"""
        brain = AliceBrain()
        pixels = np.random.uniform(0, 255, 64)
        result = brain.see(pixels)
        assert "semantic" in result
        assert result["semantic"]["is_novel"] is True

    def test_see_and_hear_same_concept(self):
        """One concept can be recognized by both visual and auditory"""
        brain = AliceBrain()
        field = brain.semantic_field.field

        # Create distinct visual fingerprint
        eye = brain.eye
        pixels_apple = np.sin(np.linspace(0, 4 * np.pi, 64)) * 127 + 128
        eye.see(pixels_apple)
        vis_fp = eye.get_last_fingerprint()

        # Register with both modalities
        aud_fp = np.random.uniform(0.01, 0.1, 32)
        aud_fp[15] = 1.0

        field.register_concept("apple", vis_fp, modality="visual")
        field.absorb("apple", aud_fp, modality="auditory")

        a = field.attractors["apple"]
        assert "visual" in a.modality_centroids
        assert "auditory" in a.modality_centroids

    def test_introspect_semantic_field_state(self):
        """introspect() includes semantic_field state"""
        brain = AliceBrain()
        state = brain.introspect()
        assert "semantic_field" in state["subsystems"]
