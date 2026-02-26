# -*- coding: utf-8 -*-
"""
NeuralActivityDisplay — Terminal Rendering Surface of the Coaxial Cable System

Isomorphism (Signal monitoring system ↔ Neural system):
  Signal source   ↔  External input (perceive API)    — signals from outside
  Input connector ↔  Body organs (Eye/Ear/Skin/Nose)  — format converter (like HDMI port)
  Coax cable      ↔  CoaxialChannel (Γ physics)       — signal transmission with reflection
  Display         ↔  NeuralActivityDisplay              — electrical signal → rendered neural activity

The body organs (AliceEye, AliceEar, etc.) are NOT cameras or microphones.
They are INPUT CONNECTORS — they convert external signals into ElectricalSignal
format and feed them into the coaxial cable. Like an HDMI port on a monitor:
it doesn't generate the data, it receives it.

The signal source is the EXTERNAL WORLD (whatever you feed via perceive API).
What that source is — real sensor, typed text, simulated data — is irrelevant
to the cable and display. They only care about the electrical signal.

The display is PASSIVE — it does not compute, decide, or interpret.
It renders whatever electrical signal arrives at the cable terminal.
Neural activity is what appears on the display.

Pixel Physics:
  Each pixel = terminal load of one neural channel (one neuron).
  Terminal impedance: Z_i = Z_cable / s_i   (s_i = synaptic_strength)
  Terminal Γ:  Γ_terminal,i = |(1 - s_i) / (1 + s_i)|

  Combined with channel Γ (cable quality):
    T_channel  = 1 - Γ_channel²
    T_terminal = 1 - Γ_terminal²  = 4s / (1+s)²
    T_pixel    = T_channel × T_terminal
    Γ_eff      = √(1 - T_pixel)

  s = 1.0 → Γ_terminal = 0   → clear pixel   (healthy neuron, matched)
  s = 0   → Γ_terminal = 1   → snow pixel     (dead neuron, total reflection)
  s > 1   → Γ_terminal < 0   → super-matched  (enhanced neuron)

  C1 always holds: Γ_eff² + T_pixel = 1

Screen Properties:
  Brightness  = mean(T_i)   → backward-compatible with scalar Φ (screen model)
  Contrast    = std(T_i)    → dynamic range of the rendered image
  Dead pixel  = stroke      → Γ = 1.0 permanently (channel destroyed)
  Burn-in     = PTSD        → trauma afterimage persists (like CRT phosphor burn)
  Snow        = noise       → Johnson-Nyquist thermal noise at pixel terminals
  White flash = pain        → signal overload → all pixels saturate
  Black screen= coma        → all Γ = 1 → no signal → brightness = 0

Dual Screen (Binocular Vision):
  Left eye  → Cable L → Screen L  (left visual cortex)
  Right eye → Cable R → Screen R  (right visual cortex)
  Depth: ΔΓ_i = Γ_L,i - Γ_R,i   (differential signaling)
    |ΔΓ| large → near object (strong parallax)
    |ΔΓ| ≈ 0  → far object  (no parallax)
  This is identical to differential signaling in electronics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alice.core.signal import ElectricalSignal


# ============================================================================
# Physical Constants
# ============================================================================

# Burn-in (PTSD afterimage)
BURN_IN_RATE = 0.05          # Fraction of trauma signal burned into pixels per event
BURN_IN_DECAY = 0.003        # Multiplicative decay per tick (0.3%) — very slow healing
BURN_IN_RESIDUAL = 0.01      # Minimum scar that never fully heals

# Pain overload
PAIN_OVERLOAD_THRESHOLD = 0.7  # Pain level that starts saturating the screen
PAIN_SATURATION_GAIN = 3.0    # Amplification factor during overload

# Johnson-Nyquist noise
NOISE_SCALE = 0.005           # Noise amplitude per unit temperature

# Dream rendering
DREAM_CLARITY = 0.6           # Dreams are 60% as vivid as waking
DREAM_NOISE_BOOST = 2.0       # Double noise during dreams

# Screen impedance (for get_signal C3 compliance)
SCREEN_Z_BASE = 75.0          # Base characteristic impedance (Ω)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class ScreenFrame:
    """
    One rendered frame — snapshot of consciousness content.

    Physics: content_i = signal_i × T_i + burn_in_i + noise_i
    where T_i = 1 - Γ_eff,i²  (channel × terminal transmission)
    """
    content: np.ndarray           # Rendered pixel values
    clarity: np.ndarray           # T_i per pixel (0=snow, 1=clear)
    gamma_map: np.ndarray         # Γ_eff,i per pixel (0=clear, 1=snow)
    brightness: float             # Φ_screen = mean(T_i)
    contrast: float               # std(T_i) — dynamic range
    peak: float                   # max |content| (overload detection)
    noise_floor: float            # mean |noise|
    burn_in_total: float          # Σ burn_in (PTSD severity)
    dead_pixels: int              # Count of stroke-damaged pixels
    active_pixels: int            # Count of functioning pixels
    resolution: int               # Total pixel count
    is_overloaded: bool           # Pain saturation active
    is_dreaming: bool             # Dream mode active
    frame_id: int                 # Tick number
    screen_id: str                # Identifier ("visual", "auditory", ...)


@dataclass
class StereoFrame:
    """
    Dual screen frame with depth perception.

    Physics: Depth from differential signaling.
      ΔΓ_i = Γ_L,i - Γ_R,i
      |ΔΓ| large → near object  (strong parallax)
      |ΔΓ| ≈ 0  → far object   (no parallax)
    """
    left: ScreenFrame
    right: ScreenFrame
    depth_map: np.ndarray         # ΔΓ per pixel pair (signed)
    depth_range: float            # max |ΔΓ| (stereoscopic range)
    convergence: float            # mean |ΔΓ| (focal distance indicator)
    fusion_quality: float         # 0=diplopia (double vision), 1=perfect fusion


# ============================================================================
# NeuralActivityDisplay — Single Rendering Surface
# ============================================================================


class NeuralActivityDisplay:
    """
    Neural Activity Display — terminal rendering surface of the coaxial cable system.

    Each pixel is the terminal load of one neural channel.
    The display passively renders whatever arrives — it does not compute or decide.

    Per-pixel physics:
      Terminal Γ:   Γ_i = |(1 - s_i) / (1 + s_i)|     (from synaptic_strength)
      Channel T:    T_ch = 1 - Γ_ch²                    (cable quality)
      Terminal T:   T_term,i = 1 - Γ_i²  = 4s / (1+s)²
      Pixel T:      T_pixel,i = T_ch × T_term,i
      Effective Γ:  Γ_eff,i = √(1 - T_pixel,i)

      Content:      pixel_i = signal_i × T_pixel,i + burn_in_i + noise_i
    """

    def __init__(self, resolution: int = 100, screen_id: str = "primary"):
        """
        Args:
            resolution: Number of pixels (= number of neural channel terminals)
            screen_id:  Identifier for this screen ("visual", "auditory", etc.)
        """
        self.resolution = resolution
        self.screen_id = screen_id

        # Per-pixel state
        self._gammas = np.ones(resolution)              # Effective Γ (init: snow)
        self._values = np.zeros(resolution)             # Last signal values
        self._burn_in = np.zeros(resolution)            # PTSD afterimage
        self._dead = np.zeros(resolution, dtype=bool)   # Dead pixels (stroke)
        self._noise = np.zeros(resolution)              # Current noise
        self._frequencies = np.zeros(resolution)        # Per-pixel frequency

        # Screen state
        self._temperature = 0.0         # Johnson-Nyquist temperature
        self._frame_count = 0           # Frame counter
        self._last_frame: Optional[ScreenFrame] = None
        self._is_dreaming = False       # Dream mode flag
        self._rng = np.random.RandomState(42)  # Reproducible noise

        # Statistics
        self.total_renders = 0
        self.total_burns = 0
        self.total_dead = 0

    # ------------------------------------------------------------------
    # Core rendering
    # ------------------------------------------------------------------

    def render(
        self,
        signal_values: np.ndarray,
        neuron_strengths: np.ndarray,
        channel_gamma: float = 0.0,
        signal_frequency: float = 10.0,
        pain_level: float = 0.0,
    ) -> ScreenFrame:
        """
        Render one frame from incoming signals.

        Physics pipeline:
          1. Compute per-pixel terminal Γ from neuron synaptic_strength
          2. Combine with channel Γ → pixel transmission T_i
          3. Render: content_i = signal_i × T_i + burn_in_i + noise_i
          4. Apply pain overload if pain > threshold

        Args:
            signal_values:    Per-pixel signal amplitude (neuron activations)
            neuron_strengths: Per-pixel synaptic_strength (0~2, 1.0 = healthy)
            channel_gamma:    Channel Γ from CoaxialChannel (0 = direct injection)
            signal_frequency: Carrier frequency → "color" of rendered content
            pain_level:       System pain level (0~1)

        Returns:
            ScreenFrame with rendered content and all metrics
        """
        self.total_renders += 1
        self._frame_count += 1
        self._is_dreaming = False

        n = min(self.resolution, len(signal_values), len(neuron_strengths))
        channel_gamma = float(np.clip(channel_gamma, 0.0, 1.0))
        pain_level = float(np.clip(pain_level, 0.0, 1.0))

        # Full-resolution arrays (pixels beyond n stay as snow)
        full_gamma = np.ones(self.resolution)
        full_T = np.zeros(self.resolution)
        full_content = np.zeros(self.resolution)

        # === 1. Per-pixel terminal Γ from neuron strength ===
        #   Γ_terminal = |(1 - s) / (1 + s)|
        #   s = synaptic_strength: 1.0 = healthy, 0 = dead, >1 = enhanced
        s = np.clip(neuron_strengths[:n], 1e-6, 10.0)
        gamma_terminal = np.abs((1.0 - s) / (1.0 + s))

        # === 2. Combined transmission ===
        #   T_channel  = 1 - Γ_ch²
        #   T_terminal = 1 - Γ_term²
        #   T_pixel    = T_channel × T_terminal
        T_channel = 1.0 - channel_gamma ** 2
        T_terminal = 1.0 - gamma_terminal ** 2   # = 4s/(1+s)²
        T_pixel = T_channel * T_terminal

        # Effective per-pixel Γ: Γ_eff = √(1 - T_pixel)
        gamma_eff = np.sqrt(np.clip(1.0 - T_pixel, 0.0, 1.0))

        full_gamma[:n] = gamma_eff
        full_T[:n] = T_pixel

        # === 3. Dead pixels override: Γ = 1.0 always ===
        full_gamma[self._dead] = 1.0
        full_T[self._dead] = 0.0

        # Store effective gamma
        self._gammas = full_gamma.copy()

        # === 4. Render content ===
        #   content = signal × T + burn_in + noise
        self._values[:n] = signal_values[:n]
        full_content[:n] = self._values[:n] * full_T[:n]

        # Add burn-in (PTSD afterimage — present even without signal)
        full_content += self._burn_in

        # Add Johnson-Nyquist thermal noise
        if self._temperature > 0:
            self._noise = self._rng.normal(
                0, NOISE_SCALE * self._temperature, self.resolution
            )
            full_content += self._noise
        else:
            self._noise = np.zeros(self.resolution)

        # Update pixel frequencies
        self._frequencies[:n] = signal_frequency

        # === 5. Pain overload ===
        is_overloaded = False
        if pain_level > PAIN_OVERLOAD_THRESHOLD:
            overload = 1.0 + (pain_level - PAIN_OVERLOAD_THRESHOLD) * PAIN_SATURATION_GAIN
            full_content *= overload
            is_overloaded = True

        # Clamp content
        full_content = np.clip(full_content, -3.0, 3.0)

        # === 6. Build frame ===
        brightness = float(np.mean(full_T[:n])) if n > 0 else 0.0
        contrast = float(np.std(full_T[:n])) if n > 1 else 0.0

        frame = ScreenFrame(
            content=full_content.copy(),
            clarity=full_T.copy(),
            gamma_map=full_gamma.copy(),
            brightness=brightness,
            contrast=contrast,
            peak=float(np.max(np.abs(full_content))),
            noise_floor=float(np.mean(np.abs(self._noise))),
            burn_in_total=float(np.sum(self._burn_in)),
            dead_pixels=int(np.sum(self._dead)),
            active_pixels=self.resolution - int(np.sum(self._dead)),
            resolution=self.resolution,
            is_overloaded=is_overloaded,
            is_dreaming=False,
            frame_id=self._frame_count,
            screen_id=self.screen_id,
        )
        self._last_frame = frame
        return frame

    # ------------------------------------------------------------------
    def render_dream(
        self,
        memory_values: np.ndarray,
        neuron_strengths: np.ndarray,
        dream_vividness: float = 0.5,
    ) -> ScreenFrame:
        """
        Dream rendering — input connector disconnected, screen still on.

        During sleep, sensory input connectors disconnect (high channel Γ).
        Internal signals from hippocampal replay are rendered.
        Dreams are less vivid and noisier than waking perception.

        Physics:
          Input disconnected → channel Γ ≈ 1 (sensory cable disconnected)
          But internal replay signal bypasses the cable → dream channel Γ < 1
          Dream vividness modulates effective channel quality

        Args:
            memory_values:    Replay signal from memory consolidation
            neuron_strengths: Per-pixel synaptic strength (neuron health)
            dream_vividness:  0=faint, 1=vivid (modulates clarity)

        Returns:
            ScreenFrame marked as dreaming
        """
        # Dream channel Γ: less vivid = more cable loss
        dream_channel_gamma = 1.0 - DREAM_CLARITY * dream_vividness

        # Boost noise temperature for noisier dreams
        original_temp = self._temperature
        self._temperature = max(self._temperature * DREAM_NOISE_BOOST, 0.01)

        frame = self.render(
            signal_values=memory_values,
            neuron_strengths=neuron_strengths,
            channel_gamma=dream_channel_gamma,
            signal_frequency=5.0,   # Dreams tend θ/δ band (low frequency)
            pain_level=0.0,         # No external pain during dreams
        )

        # Restore temperature
        self._temperature = original_temp

        # Mark as dream
        frame.is_dreaming = True
        self._is_dreaming = True

        return frame

    # ------------------------------------------------------------------
    # PTSD burn-in
    # ------------------------------------------------------------------

    def burn(self, trauma_signal: np.ndarray, intensity: float = 1.0):
        """
        PTSD burn-in — trauma signal permanently marks the screen.

        Physics:
          Extreme signal (pain, fear) leaves a persistent afterimage on the
          rendering surface, like CRT phosphor burn-in. The image remains
          visible even after the signal source is removed.

          The afterimage decays very slowly over time but a minimum residual
          scar remains permanently — this is the physics of PTSD.

        Args:
            trauma_signal: The signal that caused the trauma
            intensity:     Trauma intensity multiplier (0~1)
        """
        self.total_burns += 1
        n = min(self.resolution, len(trauma_signal))
        intensity = float(np.clip(intensity, 0.0, 1.0))

        burn_amount = np.abs(trauma_signal[:n]) * BURN_IN_RATE * intensity
        self._burn_in[:n] += burn_amount

    # ------------------------------------------------------------------
    # Stroke — dead pixels
    # ------------------------------------------------------------------

    def kill_pixels(self, indices: List[int]):
        """
        Stroke — permanently destroy specific pixels.

        Physics:
          Stroke destroys neurons → channel terminal open circuit →
          Z_load = ∞ → Γ = 1.0 → total reflection → no signal reaches pixel.
          This is permanent and irreversible.

        Args:
            indices: Pixel indices to kill
        """
        for idx in indices:
            if 0 <= idx < self.resolution:
                if not self._dead[idx]:
                    self._dead[idx] = True
                    self._gammas[idx] = 1.0
                    self.total_dead += 1

    # ------------------------------------------------------------------
    # Per-tick update
    # ------------------------------------------------------------------

    def tick(self):
        """
        Per-cycle maintenance.

        - Burn-in decays slowly (multiplicative, 0.3% per tick)
        - Burns above residual threshold decay; below threshold they persist
        - Dead pixels remain dead (irreversible)
        """
        # Decay burn-in (only above residual threshold)
        above = self._burn_in > BURN_IN_RESIDUAL
        if np.any(above):
            self._burn_in[above] *= (1.0 - BURN_IN_DECAY)

        # Dead pixels stay dead
        self._gammas[self._dead] = 1.0

    # ------------------------------------------------------------------
    # Temperature coupling
    # ------------------------------------------------------------------

    def set_temperature(self, temperature: float):
        """
        Set Johnson-Nyquist temperature for noise coupling.

        Physics:
          Higher system temperature → more thermal noise at pixel terminals.
          Noise power ∝ k_B × T × Δf  (Johnson-Nyquist)
          This couples the pain loop (ram_temperature) to screen noise.
        """
        self._temperature = max(0.0, float(temperature))

    # ------------------------------------------------------------------
    # Readout
    # ------------------------------------------------------------------

    def get_brightness(self) -> float:
        """
        Overall screen brightness = Φ_screen.

        Physics: mean transmission across all active pixels.
        Backward-compatible as a scalar consciousness measure.
        """
        if self._last_frame is not None:
            return self._last_frame.brightness
        return 0.0

    def get_frame(self) -> Optional[ScreenFrame]:
        """Get the last rendered frame."""
        return self._last_frame

    # ------------------------------------------------------------------
    # C3 Signal Protocol compliance
    # ------------------------------------------------------------------

    def get_signal(self) -> ElectricalSignal:
        """
        Emit screen state as ElectricalSignal (C3 compliance).

        The screen's output signal represents the conscious experience:
          waveform  = rendered content (pixel values)
          amplitude = brightness (Φ)
          frequency = dominant frequency on screen
          impedance = derived from mean Γ:  Z = Z₀ × (1+Γ)/(1-Γ)
        """
        if self._last_frame is not None:
            content = self._last_frame.content
        else:
            content = np.zeros(self.resolution)

        # Dominant frequency
        active_freq = self._frequencies[self._frequencies > 0]
        dom_freq = float(np.mean(active_freq)) if len(active_freq) > 0 else 10.0

        # Screen impedance from mean Γ
        mean_gamma = float(np.mean(self._gammas))
        if mean_gamma < 0.999:
            screen_z = SCREEN_Z_BASE * (1.0 + mean_gamma) / (1.0 - mean_gamma)
        else:
            screen_z = 10000.0  # Nearly open circuit

        return ElectricalSignal.from_raw(
            content,
            source=f"screen_{self.screen_id}",
            modality="consciousness",
            impedance=min(screen_z, 10000.0),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Full screen state for introspection."""
        return {
            "screen_id": self.screen_id,
            "resolution": self.resolution,
            "brightness": round(self.get_brightness(), 4),
            "mean_gamma": round(float(np.mean(self._gammas)), 4),
            "dead_pixels": int(np.sum(self._dead)),
            "active_pixels": self.resolution - int(np.sum(self._dead)),
            "burn_in_total": round(float(np.sum(self._burn_in)), 4),
            "temperature": round(self._temperature, 4),
            "is_dreaming": self._is_dreaming,
            "total_renders": self.total_renders,
            "total_burns": self.total_burns,
            "frame_id": self._frame_count,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Alias for get_state."""
        return self.get_state()


# ============================================================================
# Stereo Depth Computation
# ============================================================================


def compute_stereo_depth(
    left_frame: ScreenFrame,
    right_frame: ScreenFrame,
) -> StereoFrame:
    """
    Compute stereoscopic depth from dual screen frames.

    Physics (differential signaling):
      Two cables carry the same scene from slightly different viewpoints.
      Depth = difference in impedance matching at corresponding terminals.

      ΔΓ_i = Γ_L,i - Γ_R,i

      Near object: left and right screens see DIFFERENT impedance patterns
        → |ΔΓ| large (strong parallax)
      Far object: left and right screens see NEARLY IDENTICAL patterns
        → |ΔΓ| ≈ 0  (no parallax)

      This is identical to differential signaling in electronics:
      two cables carry slightly different signals → receiver takes the
      difference → extracts depth, rejects common-mode noise.

    Args:
        left_frame:  Last frame from left screen
        right_frame: Last frame from right screen

    Returns:
        StereoFrame with depth map and fusion metrics
    """
    n = min(len(left_frame.gamma_map), len(right_frame.gamma_map))

    # ΔΓ per pixel (signed: positive = closer to left eye)
    depth_map = left_frame.gamma_map[:n] - right_frame.gamma_map[:n]

    # Depth range = maximum stereoscopic disparity
    abs_depth = np.abs(depth_map)
    depth_range = float(np.max(abs_depth)) if n > 0 else 0.0

    # Convergence = mean disparity → focal distance indicator
    convergence = float(np.mean(abs_depth)) if n > 0 else 0.0

    # Fusion quality = how well left and right agree
    #   Perfect fusion: all ΔΓ = 0 → quality = 1
    #   Diplopia (double vision): large variance in ΔΓ → quality → 0
    if n > 1:
        fusion_quality = float(np.clip(1.0 - np.std(depth_map) * 2.0, 0.0, 1.0))
    else:
        fusion_quality = 1.0

    return StereoFrame(
        left=left_frame,
        right=right_frame,
        depth_map=depth_map.copy(),
        depth_range=depth_range,
        convergence=convergence,
        fusion_quality=fusion_quality,
    )


# Backward compatibility alias
ConsciousnessScreen = NeuralActivityDisplay
