# -*- coding: utf-8 -*-
"""Shared impedance physics for all clinical modules.

Isomorphic to alice/core/gamma_topology.py:
  C1: Γ² + T = 1             (algebraic identity)
  C2: ΔZ = −η · Γ · x_in · x_out  (only legal Z mutation)
  C3: All values carry impedance     (ImpedanceChannel)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict


# ============================================================================
# Standalone helpers (legacy — prefer ImpedanceChannel for new code)
# ============================================================================

def gamma(z_load: float, z_source: float) -> float:
    """Reflection coefficient: Γ = (Z_load − Z_source) / (Z_load + Z_source)."""
    denom = z_load + z_source
    if abs(denom) < 1e-12:
        return 0.0
    return (z_load - z_source) / denom


def gamma_sq(z_load: float, z_source: float) -> float:
    """Squared reflection coefficient Γ²."""
    g = gamma(z_load, z_source)
    return g * g


# ============================================================================
# ImpedanceChannel — the atomic physics unit (isomorphic to GammaEdge)
# ============================================================================

class ImpedanceChannel:
    """Single impedance channel with built-in C1/C2 enforcement.

    Isomorphism with gamma_topology.py:
        GammaNode.impedance   ←→  ImpedanceChannel.z
        GammaEdge.gamma_vector ←→  ImpedanceChannel.gamma
        C2 ΔZ = −η·Γ·gate    ←→  ImpedanceChannel.remodel()
        C1 Γ²+T=1            ←→  ImpedanceChannel.transmission

    Z is read-only — can ONLY change through remodel() (C2).
    """
    __slots__ = ('_z', 'z_ref', 'z_min', 'z_max')

    def __init__(self, z_ref: float, z_init: float | None = None,
                 z_min: float = 1.0, z_max: float = 500.0):
        self.z_ref = z_ref
        self._z = z_init if z_init is not None else z_ref
        self.z_min = z_min
        self.z_max = z_max

    @property
    def z(self) -> float:
        """Current impedance (read-only — mutate via remodel only)."""
        return self._z

    @property
    def gamma(self) -> float:
        """Γ = (Z − Z_ref) / (Z + Z_ref)."""
        denom = self._z + self.z_ref
        if abs(denom) < 1e-12:
            return 0.0
        return (self._z - self.z_ref) / denom

    @property
    def gamma_sq(self) -> float:
        """Γ²."""
        g = self.gamma
        return g * g

    @property
    def transmission(self) -> float:
        """C1: T = 1 − Γ²."""
        return 1.0 - self.gamma_sq

    def remodel(self, x_in: float, x_out: float, eta: float,
                z_target: float | None = None) -> None:
        """C2: ΔZ = −η · Γ(Z, Z_target) · x_in · x_out.

        z_target: impedance to remodel toward (default: z_ref = healthy).
        Positive eta drives Z toward z_target (healing / matching).
        """
        target = z_target if z_target is not None else self.z_ref
        denom = self._z + target
        if abs(denom) < 1e-12:
            return
        g = (self._z - target) / denom
        self._z = max(self.z_min, min(self.z_max,
                                       self._z - eta * g * x_in * x_out))

    def verify_c1(self, tol: float = 1e-10) -> bool:
        """Verify Γ² + T = 1."""
        return abs(self.gamma_sq + self.transmission - 1.0) < tol


class ClinicalEngineBase:
    """Base class for all 12 clinical specialty engines.

    Subclasses must define:
        DISEASE_CLASSES: dict mapping disease name → model class
        RESERVE_KEY: str name for the reserve metric (e.g. "cardiac_reserve")
    """

    DISEASE_CLASSES: Dict[str, type] = {}
    RESERVE_KEY: str = "reserve"

    def __init__(self):
        self.active_diseases: Dict[str, object] = {}
        self.tick_count: int = 0

    def add_disease(self, name: str, **kwargs):
        cls = self.DISEASE_CLASSES.get(name)
        if cls:
            self.active_diseases[name] = cls(**kwargs)

    def tick(self) -> Dict:
        self.tick_count += 1
        results = {}
        total_g2 = 0.0
        for name, model in self.active_diseases.items():
            r = model.tick()
            results[name] = r
            total_g2 += r.get("gamma_sq", 0.0)
        # C1 hard constraint: Γ² + T = 1 → total Γ² ≤ 1.0
        if total_g2 > 1.0:
            warnings.warn(
                f"C1 violated: total Γ² = {total_g2:.4f} > 1.0, "
                f"clamping to 1.0 (tick {self.tick_count})"
            )
            total_g2 = 1.0
        results["total_gamma_sq"] = total_g2
        results[self.RESERVE_KEY] = 1.0 - total_g2
        return results


# ============================================================================
# Declarative Disease Model Factories
# ============================================================================

@dataclass(frozen=True)
class MetricSpec:
    """Clinical metric: value = offset + Γ² × scale, then clamp."""
    name: str
    offset: float = 0.0
    scale: float = 1.0
    min_val: float | None = None
    max_val: float | None = None
    as_int: bool = False


def make_template_disease(
    disease_name: str,
    z_base: float,
    z_coeff: float,
    default_severity: float = 0.5,
    treatment_factor: float = 0.5,
    metrics: tuple[MetricSpec, ...] = (),
    severity_decay: float = 0.0,
    severity_growth: float = 0.0,
    default_extra: dict | None = None,
    eta: float = 1.0,
) -> type:
    """Factory: severity → Z_target → C2 remodel → Γ² → metrics.

    Z changes ONLY through ImpedanceChannel.remodel() (C2).
    C1 (Γ²+T=1) guaranteed by ImpedanceChannel.
    """
    _defaults = default_extra or {}

    class _Model:
        def __init__(self, severity: float = default_severity, **extra):
            self.severity = severity
            z_init = z_base * (1 + severity * z_coeff)
            self.channel = ImpedanceChannel(z_ref=z_base, z_init=z_init)
            self.on_treatment = False
            self.tick_count = 0
            self._extra = {**_defaults, **extra}

        def start_treatment(self):
            self.on_treatment = True

        def tick(self) -> Dict:
            self.tick_count += 1
            if severity_decay and self.on_treatment:
                self.severity = max(0.0, self.severity - severity_decay)
            elif severity_growth and not self.on_treatment:
                self.severity = min(1.0, self.severity + severity_growth)
            tf = treatment_factor if self.on_treatment else 1.0
            sev = self.severity * tf
            # C2: remodel toward disease target
            z_target = z_base * (1 + sev * z_coeff)
            self.channel.remodel(x_in=1.0, x_out=1.0, eta=eta,
                                 z_target=z_target)
            g2 = self.channel.gamma_sq
            result = {"disease": disease_name, "gamma_sq": g2}
            for m in metrics:
                val = m.offset + g2 * m.scale
                if m.min_val is not None:
                    val = max(m.min_val, val)
                if m.max_val is not None:
                    val = min(m.max_val, val)
                if m.as_int:
                    val = int(val)
                result[m.name] = val
            result.update(self._extra)
            return result

    _Model.__name__ = disease_name.replace(" ", "") + "Model"
    _Model.__qualname__ = _Model.__name__
    return _Model


# ============================================================================
# Shared Tumor Physics
# ============================================================================

@dataclass
class TumorCore:
    """Universal tumor state: impedance camouflage + growth."""
    tumor_z: float = 75.0
    camouflage: float = 0.8
    size_cm: float = 2.0
    growth_rate: float = 0.01
    on_treatment: bool = False
    metastatic: bool = False
    sites: int = 0

    def grow(self) -> float:
        if self.on_treatment:
            self.size_cm = max(0.1, self.size_cm * (1 - self.growth_rate * 0.5))
        else:
            self.size_cm = min(20, self.size_cm * (1 + self.growth_rate))
        return self.size_cm


@dataclass(frozen=True)
class StageSpec:
    """Stage threshold: if Γ² < threshold, assign this stage label."""
    threshold: float
    stage: str


def make_tumor_model(
    disease_name: str,
    tissue_z: float,
    initial_z_mult: float = 1.5,
    size_denom: float = 2.0,
    z_mult: float = 1.0,
    default_size: float = 2.0,
    growth_rate: float = 0.01,
    stages: tuple[StageSpec, ...] = (),
    default_stage: str = "IV",
    markers: tuple[MetricSpec, ...] = (),
    bool_markers: tuple[tuple[str, float], ...] = (),
    default_extra: dict | None = None,
    eta: float = 1.0,
) -> type:
    """Factory: TumorCore.grow() → Z_target → C2 remodel → Γ² → stage.

    Z changes ONLY through ImpedanceChannel.remodel() (C2).
    """
    _defaults = default_extra or {}

    class _Model:
        def __init__(self, size: float = default_size, **extra):
            z_init = tissue_z * (1 + size / size_denom * z_mult)
            self.core = TumorCore(
                tumor_z=z_init,
                size_cm=size,
                growth_rate=growth_rate,
            )
            self.channel = ImpedanceChannel(z_ref=tissue_z, z_init=z_init)
            self.tick_count = 0
            self._extra = {**_defaults, **extra}

        def start_treatment(self):
            self.core.on_treatment = True

        def tick(self) -> Dict:
            self.tick_count += 1
            self.core.grow()
            # C2: remodel toward size-derived target
            z_target = tissue_z * (1 + self.core.size_cm / size_denom * z_mult)
            self.channel.remodel(x_in=1.0, x_out=1.0, eta=eta,
                                 z_target=z_target)
            g2 = self.channel.gamma_sq
            stage = default_stage
            for s in stages:
                if g2 < s.threshold:
                    stage = s.stage
                    break
            result = {"disease": disease_name, "stage": stage,
                      "size": self.core.size_cm, "gamma_sq": g2}
            for m in markers:
                val = m.offset + g2 * m.scale
                if m.min_val is not None:
                    val = max(m.min_val, val)
                if m.max_val is not None:
                    val = min(m.max_val, val)
                if m.as_int:
                    val = int(val)
                result[m.name] = val
            for name, threshold in bool_markers:
                result[name] = g2 > threshold
            result.update(self._extra)
            return result

    _Model.__name__ = disease_name.replace(" ", "") + "Model"
    _Model.__qualname__ = _Model.__name__
    return _Model
