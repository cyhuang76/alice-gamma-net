# -*- coding: utf-8 -*-
"""
Alice's Eye — Forward Engineering + Resolution Adaptation

Physics:
  "The eye is essentially forward engineering."
  Photons → Retina photoelectric conversion → Optic nerve electrical signal

  Key insight: A lens is a Fourier transformer.
  In Fourier optics, the focal plane of a convex lens contains the spatial
  frequency spectrum of the input image.
  So the eye "using FFT" is physically correct — this is the physical property
  of the lens, not the brain computing.

Pipeline:
  1. Pupil (aperture)        — Regulate light intake     → Normalization
  2. Lens (convex lens)      — Physical Fourier transform → Spatial spectrum
  3. Retina (photosensitive) — Photoelectric conversion   → Voltage waveform
  4. Optic nerve             — Coaxial cable             → ElectricalSignal

Spatial frequency → Brainwave frequency mapping (tonotopic-like):
  Low spatial freq (large color blocks, overall contour) → δ/θ   (0.5-8 Hz)
  Mid spatial freq (edges, object separation)            → α/β   (8-30 Hz)
  High spatial freq (texture, details)                   → γ     (30-100 Hz)

Resolution adaptation (new):
  "Insufficient resolution = poor vision = high-frequency detail loss"

  Physical analogy:
  - Nyquist frequency: at least 2 pixels needed to resolve one stroke
  - CJK character horizontal stroke width = 1px, character height = 12~48px
  - If resolution insufficient → FFT loses high-freq components → thin strokes vanish
  - "未" and "末" differ by one stroke; insufficient resolution cannot distinguish them

  Solution:
  1. Dynamically adjust retina sampling points based on input resolution
  2. Nyquist check: calculate minimum resolvable feature size
  3. When resolution insufficient, compensate with anti-aliasing interpolation
     (simulating eye's "coarse view" mode)
  4. Provide "visual acuity" metric = minimum stroke width distinguishable
     at current resolution

This is why:
  - Right brain (δ/θ/α tuner) sees "overall contour"
  - Left brain (β/γ tuner) sees "texture details"
  — The brain doesn't choose what to see; physics frequency determines what goes where.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alice.core.signal import BrainWaveBand, ElectricalSignal


# ============================================================================
# Physical constants of the eye
# ============================================================================

# Spatial frequency → Brainwave frequency mapping range
BRAINWAVE_FREQ_MIN = 0.5    # Hz (δ lower bound)
BRAINWAVE_FREQ_MAX = 100.0  # Hz (γ upper bound)

# Retina properties
RETINA_IMPEDANCE = 50.0     # Ω (optic nerve impedance = sensory cortex impedance)
RETINA_GAIN = 1.0           # Photoelectric conversion gain
OPTIC_NERVE_SNR = 15.0      # dB (optic nerve SNR)

# Visual fingerprint (retinotopic fingerprint)
# Do NOT compress to 32! Visual information density is far higher than auditory.
# Cochlea 32 channels = biological constraint (ERB critical bands),
# retina should preserve full spatial spectrum resolution.
# Cross-modal comparison uses attractor labels, no dimension alignment needed.
VISUAL_FINGERPRINT_BINS = 0   # 0 = use full resolution (= retina_resolution)


# ============================================================================
# Resolution adaptation system
# ============================================================================

# Minimum readable pixels for CJK characters (complex chars like 龜, 鑰 need more)
MIN_CJK_CHAR_PX = 16       # px — minimum readable size for CJK characters
MIN_CJK_STROKE_PX = 2      # px — minimum 2px per stroke to resolve (Nyquist)
MIN_LATIN_CHAR_PX = 8       # px — minimum readable size for Latin characters

# Retina resolution range
RETINA_RESOLUTION_MIN = 64      # Minimum sampling points (blurry vision)
RETINA_RESOLUTION_MAX = 2048    # Maximum sampling points (ultra-high resolution)
RETINA_RESOLUTION_DEFAULT = 256 # Default

# Anti-aliasing interpolation oversampling factor
ANTIALIAS_OVERSAMPLE = 4     # 4x oversample then downsample (simulating eye's "careful look")


class VisualAcuity(str, Enum):
    """
    Visual acuity level — Based on Nyquist criterion

    Physics: resolution determines how fine you can see.
    Like a real visual acuity test: 20/20, 20/40, etc.
    """
    EXCELLENT = "excellent"   # Every stroke is clear
    GOOD = "good"             # Most strokes discernible
    FAIR = "fair"             # Thin strokes may be lost
    POOR = "poor"             # Only contours visible
    BLIND = "blind"           # Almost nothing visible


@dataclass
class DisplayProfile:
    """
    Display device profile — The eye adjusts its retina based on this.

    Physical analogy: these are the physical parameters of the "screen seen
    by the eye". The eye automatically adjusts retina resolution based on
    screen characteristics.
    """
    width: int = 1920             # Pixel width
    height: int = 1080            # Pixel height
    ppi: float = 96.0             # Pixels per inch
    font_size_px: float = 16.0    # Text rendering size (px)
    scale_factor: float = 1.0     # OS scaling factor (e.g., 1.5x, 2x)

    @property
    def effective_font_px(self) -> float:
        """Effective rendering size (accounting for scaling)"""
        return self.font_size_px * self.scale_factor

    @property
    def stroke_width_px(self) -> float:
        """
        Single stroke width estimate

        CJK characters typically have 6~12 strokes distributed within character height.
        Stroke width ≈ font_height / (2 × average stroke count)
        """
        avg_strokes = 8  # Average stroke count for CJK characters
        return self.effective_font_px / (2.0 * avg_strokes)

    @property
    def nyquist_ok(self) -> bool:
        """
        Nyquist check: whether stroke width is sufficient for legibility

        Nyquist theorem: at least 2 pixels needed to resolve 1 stroke
        """
        return self.stroke_width_px >= MIN_CJK_STROKE_PX

    @property
    def acuity(self) -> VisualAcuity:
        """Visual acuity level"""
        eff = self.effective_font_px
        if eff >= MIN_CJK_CHAR_PX * 2:
            return VisualAcuity.EXCELLENT  # >= 32px — very clear
        elif eff >= MIN_CJK_CHAR_PX:
            return VisualAcuity.GOOD       # >= 16px — readable
        elif eff >= MIN_CJK_CHAR_PX * 0.75:
            return VisualAcuity.FAIR       # >= 12px — barely readable
        elif eff >= MIN_LATIN_CHAR_PX:
            return VisualAcuity.POOR       # >= 8px — Latin only
        else:
            return VisualAcuity.BLIND      # < 8px — almost nothing visible

    @property
    def optimal_retina_resolution(self) -> int:
        """
        Calculate optimal retina resolution based on display device

        Physics: retina cone cell density should match the spatial frequency
        content of the input image.
        Higher resolution → more cone cells needed → more FFT points.

        Formula:
          retina_res = max_dim / 4  (4x downsampling, balancing precision and performance)
          but not less than 64 and not more than 2048.
        """
        max_dim = max(self.width, self.height)
        # Round to nearest power of 2 (FFT performance)
        raw = max_dim / 4
        power_of_2 = int(2 ** round(math.log2(max(raw, RETINA_RESOLUTION_MIN))))
        return max(RETINA_RESOLUTION_MIN, min(RETINA_RESOLUTION_MAX, power_of_2))

    # --- Common device presets ---

    @classmethod
    def desktop_1080p(cls) -> 'DisplayProfile':
        """1080p desktop monitor"""
        return cls(width=1920, height=1080, ppi=96.0, font_size_px=16.0, scale_factor=1.0)

    @classmethod
    def desktop_4k(cls) -> 'DisplayProfile':
        """4K desktop monitor (2x scaling)"""
        return cls(width=3840, height=2160, ppi=163.0, font_size_px=16.0, scale_factor=2.0)

    @classmethod
    def phone_720p(cls) -> 'DisplayProfile':
        """720p phone"""
        return cls(width=720, height=1280, ppi=326.0, font_size_px=14.0, scale_factor=2.0)

    @classmethod
    def phone_1080p(cls) -> 'DisplayProfile':
        """1080p phone"""
        return cls(width=1080, height=1920, ppi=401.0, font_size_px=14.0, scale_factor=3.0)

    @classmethod
    def tablet(cls) -> 'DisplayProfile':
        """iPad-like tablet"""
        return cls(width=2048, height=2732, ppi=264.0, font_size_px=17.0, scale_factor=2.0)

    @classmethod
    def low_res(cls) -> 'DisplayProfile':
        """Low resolution (old monitor or very small font)"""
        return cls(width=800, height=600, ppi=72.0, font_size_px=10.0, scale_factor=1.0)



# ============================================================================
# Main class
# ============================================================================


class AliceEye:
    """
    Alice's Eye — Forward engineering sensory organ + Resolution adaptation

    Light → Lens (FFT) → Retina (photoelectric conversion) → Optic nerve (ElectricalSignal)

    Does not perform any "understanding" or "recognition".
    Only performs physical conversion — the rest is left to the brain's LC resonance.

    New: Resolution adaptation
      - Connect DisplayProfile → Auto-adjust retina resolution
      - Nyquist check → Warn about stroke loss risk
      - Anti-aliasing interpolation → Low resolution compensation
      - Visual acuity metric → Current capability assessment
    """

    def __init__(
        self,
        retina_resolution: int = RETINA_RESOLUTION_DEFAULT,
        pupil_aperture: float = 1.0,
        display: Optional[DisplayProfile] = None,
        auto_adapt: bool = False,
    ):
        """
        Args:
            retina_resolution: Retina resolution (FFT point count)
            pupil_aperture:    Pupil aperture 0.0~1.0 (regulate light intake)
            display:           Display device profile (None = default 1080p)
            auto_adapt:        Whether to auto-adjust resolution based on display
        """
        self.display = display or DisplayProfile.desktop_1080p()
        self.auto_adapt = auto_adapt

        if auto_adapt and display is not None:
            self.retina_resolution = self.display.optimal_retina_resolution
        else:
            self.retina_resolution = retina_resolution

        self.pupil_aperture = float(np.clip(pupil_aperture, 0.01, 1.0))

        # Statistics
        self.total_frames: int = 0
        self.total_saccades: int = 0
        self._acuity_warnings: int = 0  # Stroke loss warning count

        # Visual fingerprint cache (last see() retinotopic fingerprint)
        self._last_fingerprint: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retinotopic_fingerprint(
        self,
        retinal_voltage: np.ndarray,
        n_bins: int = VISUAL_FINGERPRINT_BINS,
    ) -> np.ndarray:
        """
        Convert retinal voltage to visual fingerprint.

        Physically symmetric with cochlear fingerprint, but **different resolution**:
          - Cochlea  = 32 ERB critical bands (biological constraint)
          - Retina   = N spatial frequency bands (N = retina_resolution)

        Cochlea compresses to 32 because human cochlea has only ~32 critical bands,
        but retina spatial spectrum resolution can be arbitrarily high.
        Compressing to 32 = losing fine texture = cannot read CJK strokes.

        n_bins = 0 (default): preserve full resolution (dimension = len(retinal_voltage))
        n_bins > 0: compress to specified dimension (for low-precision fast matching)

        Cross-modal comparison doesn't need dimension alignment — semantic field
        bridges through attractor labels, not direct comparison of different
        modality fingerprints.

        Args:
            retinal_voltage: Voltage vector after retina photoelectric conversion
            n_bins: 0=full resolution, >0=compress to n_bins dimensions

        Returns:
            np.ndarray: Normalized energy fingerprint
        """
        n = len(retinal_voltage)
        if n == 0:
            return np.zeros(max(n_bins, 1))

        if n_bins <= 0 or n_bins >= n:
            # === Full resolution mode ===
            # Each spatial frequency bin retains independent RMS energy
            fp = np.abs(retinal_voltage).copy()
            total = np.sum(fp)
            if total > 0:
                fp = fp / total
            return fp

        # === Compression mode ===
        # Divide retinal_voltage evenly into n_bins segments, RMS each
        bin_size = max(1, n // n_bins)
        fp = np.zeros(n_bins)
        for i in range(n_bins):
            start = i * bin_size
            end = min(start + bin_size, n)
            if start < n:
                segment = retinal_voltage[start:end]
                fp[i] = np.sqrt(np.mean(segment ** 2))

        # Normalize
        total = np.sum(fp)
        if total > 0:
            fp = fp / total

        return fp

    # ------------------------------------------------------------------

    def get_last_fingerprint(self) -> Optional[np.ndarray]:
        """Get the visual fingerprint from the most recent see() call."""
        return self._last_fingerprint

    def see(self, pixels: np.ndarray) -> ElectricalSignal:
        """
        See = Physical conversion (not computation)

        Steps:
          1. Resolution adaptation: dynamically adjust retina based on input size
          2. Pupil: light adjustment (amplitude scaling)
          3. Lens: spatial spectrum decomposition (FFT — this is the physics of the lens!)
          4. Anti-aliasing compensation: interpolation to preserve details at low resolution
          5. Retina: photoelectric conversion (spectrum → voltage waveform)
          6. Optic nerve: package as ElectricalSignal

        Args:
            pixels: Spatial distribution of light (1D or 2D ndarray)
                    Can be grayscale image, a row of pixels, or any photosensitive signal

        Returns:
            ElectricalSignal: Optic nerve electrical signal
        """
        self.total_frames += 1

        # === 0. Resolution adaptation ===
        effective_resolution = self._adapt_resolution(pixels)

        # === 1. Pupil — Aperture adjustment ===
        light = self._pupil(pixels)

        # === 2. Anti-aliasing compensation (oversample to preserve details at low resolution) ===
        light = self._antialias_compensate(light)

        # === 3. Lens — Fourier transform (physics!) ===
        spectrum, freqs = self._lens_fft(light)

        # === 4. Retina — Photoelectric conversion ===
        retinal_voltage, dominant_freq, amplitude = self._retina(
            spectrum, freqs
        )

        # === 5. Optic nerve — Package electrical signal ===
        signal = self._optic_nerve(retinal_voltage, dominant_freq, amplitude)

        # === 6. Visual fingerprint — retinotopic fingerprint ===
        self._last_fingerprint = self.retinotopic_fingerprint(retinal_voltage)

        # Attach visual acuity info to signal metadata
        signal._acuity = self.get_visual_acuity()
        signal._resolution_adapted = effective_resolution

        return signal

    # ------------------------------------------------------------------
    def adjust_pupil(self, aperture: float):
        """Pupil adjustment (0.0=nearly closed, 1.0=fully open)"""
        self.pupil_aperture = float(np.clip(aperture, 0.01, 1.0))

    # ------------------------------------------------------------------
    def set_display(self, display: DisplayProfile):
        """
        Switch display device — Dynamic resolution adjustment

        Called when Alice switches from desktop to phone, or font size changes.
        The eye auto-adjusts retina resolution to match the new device.

        Physical analogy: like when you switch from looking at a distant object
        to looking at a nearby phone screen, the eye's focal length (lens shape) changes.
        """
        self.display = display
        if self.auto_adapt:
            self.retina_resolution = display.optimal_retina_resolution

    # ------------------------------------------------------------------
    def get_visual_acuity(self) -> Dict[str, Any]:
        """
        Current visual acuity report

        Returns:
            {
                "acuity": visual acuity level,
                "nyquist_ok": whether strokes are discernible,
                "effective_font_px": effective font size,
                "stroke_width_px": stroke width,
                "retina_resolution": retina resolution,
                "display": device name,
                "warnings": list of warning messages,
            }
        """
        d = self.display
        warnings = []

        if not d.nyquist_ok:
            warnings.append(
                f"⚠ Nyquist violation: stroke width {d.stroke_width_px:.1f}px < "
                f"minimum {MIN_CJK_STROKE_PX}px → CJK characters may lose strokes"
            )

        if d.effective_font_px < MIN_CJK_CHAR_PX:
            warnings.append(
                f"⚠ Font too small: {d.effective_font_px:.0f}px < "
                f"minimum {MIN_CJK_CHAR_PX}px → CJK characters may be garbled"
            )

        acuity = d.acuity

        if acuity in (VisualAcuity.POOR, VisualAcuity.BLIND):
            warnings.append(
                f"⚠ Visual acuity level '{acuity.value}': recommend increasing font size or resolution"
            )

        return {
            "acuity": acuity.value,
            "nyquist_ok": d.nyquist_ok,
            "effective_font_px": round(d.effective_font_px, 1),
            "stroke_width_px": round(d.stroke_width_px, 2),
            "retina_resolution": self.retina_resolution,
            "display_resolution": f"{d.width}x{d.height}",
            "ppi": d.ppi,
            "scale_factor": d.scale_factor,
            "acuity_warnings": warnings,
            "total_acuity_warnings": self._acuity_warnings,
        }

    # ------------------------------------------------------------------
    def saccade(self, pixels: np.ndarray, region: Tuple[int, int, int, int]) -> ElectricalSignal:
        """
        Saccade — Fixation on a specific region

        Args:
            pixels: Full visual field
            region: (row_start, col_start, height, width) fixation region

        Returns:
            ElectricalSignal: Optic nerve electrical signal of the fixated region
        """
        self.total_saccades += 1
        r, c, h, w = region
        if pixels.ndim == 2:
            patch = pixels[r:r+h, c:c+w]
        else:
            patch = pixels[r:r+h]
        return self.see(patch)

    # ------------------------------------------------------------------
    # Internal physics pipeline
    # ------------------------------------------------------------------

    def _adapt_resolution(self, pixels: np.ndarray) -> int:
        """
        Resolution adaptation — Dynamically adjust retina based on input size

        Physical analogy:
        - Looking at large screen → retina needs more cone cells to cover
        - Looking at small phone → fewer cone cells suffice
        - But reading fine text → high density needed regardless of screen size

        Strategy:
        1. Calculate effective resolution of the input
        2. Check Nyquist condition
        3. Increase retina_resolution when insufficient
        """
        input_size = pixels.size

        if self.auto_adapt:
            # Dynamically adjust based on input size
            # But do not exceed the upper limit suggested by display profile
            optimal = self.display.optimal_retina_resolution

            # If input very small (low-res text), increase to at least recognizable
            if input_size < MIN_CJK_CHAR_PX ** 2:
                # Input too small, may be a single character → need maximum effort to see
                self.retina_resolution = max(optimal, RETINA_RESOLUTION_MIN * 2)
            else:
                self.retina_resolution = optimal

            # Nyquist warning
            if not self.display.nyquist_ok:
                self._acuity_warnings += 1

        return self.retina_resolution

    # ------------------------------------------------------------------
    def _antialias_compensate(self, light: np.ndarray) -> np.ndarray:
        """
        Anti-aliasing compensation — Detail preservation at low resolution

        Physical analogy:
        - Real eyes "squint" when unable to see clearly → reduce aperture but increase depth of field
        - Digital equivalent: oversampling + low-pass filter → preserve stroke edges

        When Nyquist condition is not met (stroke width < 2px):
        1. 4x oversampling (Sinc interpolation)
        2. Low-pass filter (cutoff frequency = Nyquist)
        3. Downsample back to original size

        This avoids:
        - Text losing a stroke (high frequency completely discarded)
        - Garbled fonts (aliasing = frequency folding)
        """
        if self.display.nyquist_ok:
            return light  # Resolution sufficient, no compensation needed

        n = len(light)
        if n < 4:
            return light

        # === Oversampling ===
        n_up = n * ANTIALIAS_OVERSAMPLE
        x_orig = np.linspace(0, 1, n)
        x_up = np.linspace(0, 1, n_up)
        upsampled = np.interp(x_up, x_orig, light)

        # === Low-pass filter (preserve frequencies below Nyquist) ===
        fft = np.fft.rfft(upsampled)
        freqs = np.fft.rfftfreq(n_up)

        # Cutoff frequency: original Nyquist = 0.5 / oversample
        cutoff = 0.5 / ANTIALIAS_OVERSAMPLE
        # Butterworth-style soft cutoff (avoid Gibbs ringing)
        rolloff = 1.0 / (1.0 + (freqs / max(cutoff, 1e-8)) ** 6)
        fft_filtered = fft * rolloff

        # === Downsample back to original size ===
        filtered = np.fft.irfft(fft_filtered, n=n_up)
        x_down = np.linspace(0, 1, n)
        x_filt = np.linspace(0, 1, n_up)
        result = np.interp(x_down, x_filt, filtered)

        return result

    # ------------------------------------------------------------------

    def _pupil(self, pixels: np.ndarray) -> np.ndarray:
        """
        Pupil — Aperture

        - Regulate light intake (multiply by aperture coefficient)
        - Flatten to 1D (retina is a 1D photosensitive surface unfolded)
        - Normalize to 0~1 (photon count → normalized light intensity)
        """
        flat = pixels.flatten().astype(np.float64)
        # Normalize
        max_val = np.max(np.abs(flat))
        if max_val > 0:
            flat = flat / max_val
        # Aperture
        return flat * self.pupil_aperture

    # ------------------------------------------------------------------
    def _lens_fft(
        self, light: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lens — Fourier Transform

        In Fourier optics:
          - The focal plane of a convex lens = spatial frequency spectrum of input light field
          - Each position corresponds to a spatial frequency
          - Brightness = energy at that frequency

        This is not the brain doing FFT — it is the physical optics property of the lens.

        Returns:
            (power_spectrum, frequency_bins)
            power_spectrum: power at each frequency
            frequency_bins: corresponding spatial frequencies (Hz mapped)
        """
        n = len(light)
        if n == 0:
            return np.zeros(1), np.array([0.5])

        # FFT = Lens imaging
        fft_result = np.fft.rfft(light)
        power = np.abs(fft_result) ** 2  # Power spectrum

        # Spatial frequency bin → map to brainwave frequency range
        n_bins = len(power)
        # Log mapping: low spatial freq → low brainwave freq, high spatial freq → high brainwave freq
        freq_bins = np.logspace(
            math.log10(BRAINWAVE_FREQ_MIN),
            math.log10(BRAINWAVE_FREQ_MAX),
            n_bins,
        )

        return power, freq_bins

    # ------------------------------------------------------------------
    def _retina(
        self,
        spectrum: np.ndarray,
        freqs: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Retina — Photoelectric conversion

        - Each photoreceptor corresponds to a spatial frequency (tonotopic arrangement)
        - Power spectrum → voltage waveform (square root = power→voltage)
        - Extract dominant frequency (brightest photoreceptor)
        - Total light intensity → amplitude

        Returns:
            (retinal_voltage, dominant_freq, amplitude)
        """
        # Power → Voltage (V = sqrt(P × R), R = retina impedance)
        voltage = np.sqrt(spectrum * RETINA_IMPEDANCE) * RETINA_GAIN

        # Uniformly resample to retina resolution
        if len(voltage) != self.retina_resolution:
            x_old = np.linspace(0, 1, len(voltage))
            x_new = np.linspace(0, 1, self.retina_resolution)
            voltage = np.interp(x_new, x_old, voltage)
            freqs = np.interp(x_new, x_old, freqs)

        # Dominant frequency = brightest photoreceptor
        if len(voltage) > 1:
            # Skip DC component (index 0)
            peak_idx = np.argmax(voltage[1:]) + 1
            dominant_freq = float(freqs[peak_idx])
        else:
            dominant_freq = 10.0  # Default alpha

        # Amplitude = total light intensity
        amplitude = float(np.sqrt(np.mean(voltage ** 2)))  # RMS

        return voltage, dominant_freq, amplitude

    # ------------------------------------------------------------------
    def _optic_nerve(
        self,
        retinal_voltage: np.ndarray,
        dominant_freq: float,
        amplitude: float,
    ) -> ElectricalSignal:
        """
        Optic nerve — Package electrical signal

        Retinal voltage → ElectricalSignal
        Ready for transmission along coaxial cable to sensory cortex
        """
        # Map to brainwave range
        freq = float(np.clip(dominant_freq, BRAINWAVE_FREQ_MIN, BRAINWAVE_FREQ_MAX))

        return ElectricalSignal(
            waveform=retinal_voltage,
            amplitude=amplitude,
            frequency=freq,
            phase=0.0,  # Visual signal initial phase = 0
            impedance=RETINA_IMPEDANCE,
            snr=OPTIC_NERVE_SNR,
            source="eye",
            modality="visual",
        )

    # ------------------------------------------------------------------
    # Statistics and diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        acuity_info = self.get_visual_acuity()
        stats = {
            "total_frames": self.total_frames,
            "total_saccades": self.total_saccades,
            "retina_resolution": self.retina_resolution,
            "pupil_aperture": self.pupil_aperture,
            "impedance": RETINA_IMPEDANCE,
            # Resolution adaptation info
            "display": {
                "width": self.display.width,
                "height": self.display.height,
                "ppi": self.display.ppi,
                "font_size_px": self.display.font_size_px,
                "effective_font_px": self.display.effective_font_px,
                "stroke_width_px": self.display.stroke_width_px,
            },
            "acuity": acuity_info["acuity"],
            "nyquist_ok": acuity_info["nyquist_ok"],
            "acuity_warnings": self._acuity_warnings,
            "auto_adapt": self.auto_adapt,
        }
        return stats

    def get_retina_snapshot(self, pixels: np.ndarray) -> Dict[str, Any]:
        """
        Diagnostic: get snapshot of each retina layer

        Returns:
            {
                "light": light field after pupil,
                "spectrum": lens Fourier spectrum,
                "freqs": frequency bins,
                "voltage": retinal voltage,
                "dominant_freq": dominant frequency,
                "band": corresponding brainwave band,
                "acuity": visual acuity,
            }
        """
        light = self._pupil(pixels)
        spectrum, freqs = self._lens_fft(light)
        voltage, dom_freq, amp = self._retina(spectrum, freqs)
        acuity_info = self.get_visual_acuity()

        return {
            "light": light.tolist(),
            "spectrum": spectrum.tolist(),
            "freqs": freqs.tolist(),
            "voltage": voltage.tolist(),
            "dominant_freq": dom_freq,
            "band": BrainWaveBand.from_frequency(dom_freq).value,
            "amplitude": amp,
            "acuity": acuity_info["acuity"],
            "nyquist_ok": acuity_info["nyquist_ok"],
            "stroke_width_px": acuity_info["stroke_width_px"],
        }
