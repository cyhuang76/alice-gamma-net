# -*- coding: utf-8 -*-
"""Shared impedance physics utilities for all clinical modules.

Centralises the Γ and Γ² calculations so every clinical engine uses
exactly the same formula with the same zero-guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


def gamma(z_load: float, z_source: float) -> float:
    """Reflection coefficient: Γ = (Z_load − Z_source) / (Z_load + Z_source).

    Returns 0.0 when the denominator is degenerate (both impedances zero).
    """
    denom = z_load + z_source
    if abs(denom) < 1e-12:
        return 0.0
    return (z_load - z_source) / denom


def gamma_sq(z_load: float, z_source: float) -> float:
    """Squared reflection coefficient Γ²."""
    g = gamma(z_load, z_source)
    return g * g


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
        results["total_gamma_sq"] = total_g2
        results[self.RESERVE_KEY] = max(0.0, 1.0 - total_g2)
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
) -> type:
    """Factory: severity → Z = z_base × (1 + sev × z_coeff) → Γ² → metrics."""
    _defaults = default_extra or {}

    class _Model:
        def __init__(self, severity: float = default_severity, **extra):
            self.severity = severity
            self.z_field = z_base
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
            self.z_field = z_base * (1 + sev * z_coeff)
            g2 = gamma_sq(self.z_field, z_base)
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
) -> type:
    """Factory: TumorCore.grow() → Z ratio → Γ² → stage + markers."""
    _defaults = default_extra or {}

    class _Model:
        def __init__(self, size: float = default_size, **extra):
            self.core = TumorCore(
                tumor_z=tissue_z * initial_z_mult,
                size_cm=size,
                growth_rate=growth_rate,
            )
            self.tick_count = 0
            self._extra = {**_defaults, **extra}

        def start_treatment(self):
            self.core.on_treatment = True

        def tick(self) -> Dict:
            self.tick_count += 1
            self.core.grow()
            z_ratio = self.core.size_cm / size_denom
            self.core.tumor_z = tissue_z * (1 + z_ratio * z_mult)
            g2 = gamma_sq(self.core.tumor_z, tissue_z)
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
