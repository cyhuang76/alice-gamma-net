# -*- coding: utf-8 -*-
"""lab_mapping.py — Laboratory Value → Organ Impedance Mapping Engine
=======================================================================

Physics:
    Each laboratory value that deviates from normal creates an impedance
    shift in the corresponding organ system.  Multiple deviations sum to
    produce the organ's total impedance:

        Z_organ = Z_normal × (1 + Σ_j  w_j · |δ_j|)

    where δ_j is the normalised deviation of lab item j and w_j is its
    weight contribution to that organ.

    This impedance is then used to calculate the reflection coefficient:
        Γ = (Z_patient − Z_normal) / (Z_patient + Z_normal)

Constraints:
    C1: Γ² + T = 1  (energy conservation at every organ)
    C3: All inter-module values carry Z metadata (ElectricalSignal protocol)

53 laboratory items → 12 organ systems.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# 1. Organ System Definitions
# ============================================================================

ORGAN_SYSTEMS: Dict[str, float] = {
    "cardiac":    50.0,   # Cardiovascular
    "pulmonary":  60.0,   # Pulmonary
    "hepatic":    65.0,   # Hepatic
    "renal":      70.0,   # Renal
    "endocrine":  75.0,   # Endocrine
    "immune":     75.0,   # Immune
    "heme":       55.0,   # Hematologic
    "GI":         65.0,   # Gastrointestinal
    "vascular":   45.0,   # Vascular
    "bone":      120.0,   # Musculoskeletal
    "neuro":      80.0,   # Neurological
    "repro":      95.0,   # Reproductive
}

ORGAN_LIST: List[str] = list(ORGAN_SYSTEMS.keys())


# ============================================================================
# 2. Lab Item Definition
# ============================================================================

@dataclass
class LabItem:
    """Definition of a single laboratory test item.

    Parameters
    ----------
    name : str
        Canonical name (e.g. "AST", "WBC").
    unit : str
        SI or conventional unit string.
    ref_low : float or None
        Lower bound of reference interval (None = no lower bound).
    ref_high : float or None
        Upper bound of reference interval (None = no upper bound).
    critical_low : float or None
        Critical (panic) low value.
    critical_high : float or None
        Critical (panic) high value.
    organ_weights : dict[str, float]
        Mapping of organ_id → weight.  Weights should sum ≤ 1.0 for
        each lab item across all organs, though this is not enforced.
    """

    name: str
    unit: str
    ref_low: Optional[float]
    ref_high: Optional[float]
    critical_low: Optional[float]
    critical_high: Optional[float]
    organ_weights: Dict[str, float] = field(default_factory=dict)

    # convenience properties
    @property
    def ref_mid(self) -> float:
        """Midpoint of reference interval."""
        low = self.ref_low if self.ref_low is not None else 0.0
        high = self.ref_high if self.ref_high is not None else low * 2 if low > 0 else 100.0
        return (low + high) / 2.0

    @property
    def ref_range(self) -> float:
        """Width of reference interval (used for σ estimation)."""
        low = self.ref_low if self.ref_low is not None else 0.0
        high = self.ref_high if self.ref_high is not None else low * 2 if low > 0 else 100.0
        return max(high - low, 1e-12)


# ============================================================================
# 3. Master Lab Catalogue (53 items)
# ============================================================================

def _build_lab_catalogue() -> Dict[str, LabItem]:
    """Build the canonical 53-item lab catalogue.

    Reference intervals are for general adult population.
    Sex-specific items use male defaults (caller may override).
    """
    items: Dict[str, LabItem] = {}

    def _add(name: str, unit: str,
             ref_low: Optional[float], ref_high: Optional[float],
             crit_low: Optional[float], crit_high: Optional[float],
             weights: Dict[str, float]) -> None:
        items[name] = LabItem(
            name=name, unit=unit,
            ref_low=ref_low, ref_high=ref_high,
            critical_low=crit_low, critical_high=crit_high,
            organ_weights=weights,
        )

    # ---------- CBC ----------
    _add("WBC",       "10³/μL", 4.0,  11.0,  1.0,   30.0,  {"immune": 0.40, "heme": 0.20})
    _add("RBC",       "10⁶/μL", 4.5,   5.5,  2.0,    7.0,  {"heme": 0.35, "cardiac": 0.10})
    _add("Hb",        "g/dL",  13.5,  17.5,  6.0,   20.0,  {"heme": 0.40, "cardiac": 0.15, "neuro": 0.05})
    _add("Hct",       "%",     38.0,  50.0, 20.0,   60.0,  {"heme": 0.30})
    _add("MCV",       "fL",    80.0, 100.0, 60.0,  120.0,  {"heme": 0.25})
    _add("Plt",       "10³/μL",150.0,400.0, 20.0, 1000.0,  {"heme": 0.30, "hepatic": 0.10, "immune": 0.10})
    _add("Neutrophils","%",    40.0,  70.0,  5.0,   90.0,  {"immune": 0.30})
    _add("Lymphocytes","%",    20.0,  40.0,  5.0,   80.0,  {"immune": 0.30})

    # ---------- BMP ----------
    _add("Na",  "mEq/L", 136.0, 145.0, 120.0, 160.0, {"renal": 0.30, "neuro": 0.25})
    _add("K",   "mEq/L",   3.5,   5.0,   2.5,   6.5, {"renal": 0.30, "cardiac": 0.25})
    _add("Cl",  "mEq/L",  98.0, 106.0,  80.0, 120.0, {"renal": 0.20})
    _add("CO2", "mEq/L",  22.0,  29.0,  10.0,  40.0, {"renal": 0.20, "pulmonary": 0.15})
    _add("BUN", "mg/dL",   7.0,  20.0,   2.0, 100.0, {"renal": 0.35, "hepatic": 0.10})
    _add("Cr",  "mg/dL",   0.7,   1.3,   0.2,  10.0, {"renal": 0.45})
    _add("Glucose", "mg/dL", 70.0, 100.0, 40.0, 500.0,
         {"endocrine": 0.40, "neuro": 0.15, "vascular": 0.10})
    _add("Ca",  "mg/dL",   8.5,  10.5,   6.0,  14.0, {"bone": 0.25, "neuro": 0.15, "endocrine": 0.10})

    # ---------- LFT ----------
    _add("AST",    "U/L",  10.0,  40.0, None, 1000.0, {"hepatic": 0.35, "cardiac": 0.10})
    _add("ALT",    "U/L",   7.0,  56.0, None, 1000.0, {"hepatic": 0.40})
    _add("ALP",    "U/L",  44.0, 147.0, None, 1000.0, {"hepatic": 0.20, "bone": 0.20})
    _add("GGT",    "U/L",   8.0,  61.0, None,  500.0, {"hepatic": 0.25})
    _add("Bil_total","mg/dL",0.1,   1.2, None,   20.0, {"hepatic": 0.30, "heme": 0.15})
    _add("Bil_direct","mg/dL",0.0,  0.3, None,   10.0, {"hepatic": 0.25})
    _add("Albumin","g/dL",  3.5,   5.0,  1.5, None,   {"hepatic": 0.25, "renal": 0.10, "GI": 0.15})
    _add("Total_Protein","g/dL",6.0,8.3, 3.0,   12.0, {"hepatic": 0.15, "immune": 0.15})
    _add("INR",    "ratio", 0.8,   1.2, None,    5.0,  {"hepatic": 0.25, "heme": 0.15})

    # ---------- TFT ----------
    _add("TSH",     "mIU/L",  0.4,  4.0,  0.01,  50.0,  {"endocrine": 0.45})
    _add("FT4",     "ng/dL",  0.8,  1.8,  0.1,    5.0,  {"endocrine": 0.35})
    _add("FT3",     "pg/mL",  2.3,  4.2,  0.5,   10.0,  {"endocrine": 0.25})

    # ---------- Lipid Panel ----------
    _add("TC",       "mg/dL", None, 200.0, None,  400.0, {"vascular": 0.25})
    _add("LDL",      "mg/dL", None, 130.0, None,  300.0, {"vascular": 0.35, "cardiac": 0.10})
    _add("HDL",      "mg/dL", 40.0, None,   15.0, None,  {"vascular": 0.25})
    _add("TG",       "mg/dL", None, 150.0, None, 1000.0, {"vascular": 0.20, "endocrine": 0.10})

    # ---------- Inflammatory / Cardiac Markers ----------
    _add("CRP",      "mg/L",  None,  3.0,  None, 200.0,  {"immune": 0.30, "vascular": 0.10, "bone": 0.05})
    _add("ESR",      "mm/hr", None, 15.0,  None, 120.0,  {"immune": 0.25})
    _add("PCT",      "ng/mL", None,  0.05, None, 100.0,  {"immune": 0.35})
    _add("Troponin", "ng/mL", None,  0.04, None,  50.0,  {"cardiac": 0.50})
    _add("BNP",      "pg/mL", None, 100.0, None, 5000.0, {"cardiac": 0.40})
    _add("CK_MB",    "U/L",   None,  25.0, None, 300.0,  {"cardiac": 0.30})
    _add("D_Dimer",  "ng/mL", None, 500.0, None,10000.0, {"vascular": 0.20, "pulmonary": 0.15})

    # ---------- Metabolic / Other ----------
    _add("HbA1c",  "%",       4.0,   5.6, None,   14.0, {"endocrine": 0.40, "vascular": 0.15})
    _add("Uric_Acid","mg/dL", 3.5,   7.2, None,   15.0, {"bone": 0.20, "renal": 0.15})
    _add("Ferritin","ng/mL", 20.0, 500.0,  5.0, 2000.0, {"heme": 0.30, "hepatic": 0.10})
    _add("Vit_D",  "ng/mL",  30.0, 100.0,  5.0, None,   {"bone": 0.25, "immune": 0.10})
    _add("Vit_B12","pg/mL", 200.0, 900.0,100.0, None,   {"neuro": 0.20, "heme": 0.15})
    _add("Folate", "ng/mL",   3.0, None,   1.0, None,   {"heme": 0.15})
    _add("Amylase","U/L",    28.0, 100.0, None, 1000.0, {"GI": 0.35})
    _add("Lipase", "U/L",     0.0, 160.0, None, 1000.0, {"GI": 0.40})
    _add("NH3",    "μg/dL",  15.0,  45.0, None,  200.0, {"hepatic": 0.20, "neuro": 0.20})
    _add("Lactate","mmol/L",  0.5,   2.2, None,   10.0, {"neuro": 0.15, "cardiac": 0.10})
    _add("Homocysteine","μmol/L",5.0,15.0,None,   50.0, {"vascular": 0.20, "neuro": 0.10})

    # ---------- Reproductive (optional) ----------
    _add("FSH",        "mIU/mL", 3.0, 10.0, None, 100.0, {"repro": 0.35})
    _add("LH",         "mIU/mL", 2.0, 15.0, None, 100.0, {"repro": 0.30})
    _add("Testosterone","ng/dL",300.0,1000.0,50.0,1500.0, {"repro": 0.30, "endocrine": 0.10})

    return items


# Module-level singleton
LAB_CATALOGUE: Dict[str, LabItem] = _build_lab_catalogue()


# ============================================================================
# 4. Normalisation Functions
# ============================================================================

def normalise_linear(value: float, ref_low: float, ref_high: float) -> float:
    """Normalise value to deviation δ using linear z-score.

    δ = 0  when value is at midpoint of reference interval.
    |δ| = 1  when value is one reference-range-width away from midpoint.

    Returns
    -------
    float : normalised deviation (-∞, +∞)
    """
    mid = (ref_low + ref_high) / 2.0
    half_range = (ref_high - ref_low) / 2.0
    if half_range < 1e-12:
        return 0.0
    return (value - mid) / half_range


def normalise_piecewise(
    value: float,
    ref_low: Optional[float],
    ref_high: Optional[float],
    critical_low: Optional[float],
    critical_high: Optional[float],
) -> float:
    """Piecewise normalisation with saturation.

    Returns δ ∈ [-1, +1] where:
        δ = 0  when within reference interval
        δ = -1 when at or below critical_low
        δ = +1 when at or above critical_high

    Used for lab items where only one bound is clinically relevant
    (e.g. Troponin has no meaningful low, only high).
    """
    # Defaults for missing bounds
    rl = ref_low if ref_low is not None else -1e12
    rh = ref_high if ref_high is not None else 1e12
    cl = critical_low if critical_low is not None else rl - (rh - rl) * 2
    ch = critical_high if critical_high is not None else rh + (rh - rl) * 2

    if rl <= value <= rh:
        return 0.0
    elif value < rl:
        denom = rl - cl
        if abs(denom) < 1e-12:
            return -1.0
        return max(-1.0, (value - rl) / denom)
    else:  # value > rh
        denom = ch - rh
        if abs(denom) < 1e-12:
            return 1.0
        return min(1.0, (value - rh) / denom)


def normalise_lab_value(value: float, item: LabItem) -> float:
    """Choose the appropriate normalisation strategy for a lab item.

    Strategy selection:
    - If both ref_low and ref_high exist → linear
    - If one bound is None → piecewise (saturating)
    """
    if item.ref_low is not None and item.ref_high is not None:
        return normalise_linear(value, item.ref_low, item.ref_high)
    else:
        return normalise_piecewise(
            value, item.ref_low, item.ref_high,
            item.critical_low, item.critical_high,
        )


# ============================================================================
# 5. Lab Mapper — Lab Values → 12-D Organ Impedance Vector
# ============================================================================

class LabMapper:
    """Maps a set of laboratory values to 12-dimensional organ impedances.

    Usage
    -----
    >>> mapper = LabMapper()
    >>> z_vector = mapper.compute_organ_impedances({"AST": 480, "ALT": 520, "Bil_total": 3.2})
    >>> z_vector["hepatic"]   # >> 65.0 (shifted upward)
    """

    def __init__(
        self,
        catalogue: Optional[Dict[str, LabItem]] = None,
        organ_z_normal: Optional[Dict[str, float]] = None,
    ):
        self.catalogue = catalogue or LAB_CATALOGUE
        self.organ_z_normal = organ_z_normal or dict(ORGAN_SYSTEMS)

    def normalise(self, lab_name: str, value: float) -> float:
        """Normalise a single lab value to deviation δ."""
        if lab_name not in self.catalogue:
            raise KeyError(f"Unknown lab item: {lab_name!r}. "
                           f"Available: {sorted(self.catalogue.keys())}")
        item = self.catalogue[lab_name]
        return normalise_lab_value(value, item)

    def compute_organ_impedances(
        self,
        lab_values: Dict[str, float],
    ) -> Dict[str, float]:
        """Convert lab values to 12-D organ impedance vector.

        Parameters
        ----------
        lab_values : dict
            Mapping of lab_name → measured value.

        Returns
        -------
        dict[str, float]
            Mapping of organ_id → Z_patient.
            Organs with no relevant lab values retain Z_normal.

        Physics
        -------
        Z_organ = Z_normal × (1 + Σ_j w_j · |δ_j|)
        """
        # Accumulate weighted absolute deviations per organ
        organ_shift: Dict[str, float] = {organ: 0.0 for organ in ORGAN_LIST}

        for lab_name, value in lab_values.items():
            if lab_name not in self.catalogue:
                continue  # silently skip unknown labs
            item = self.catalogue[lab_name]
            delta = normalise_lab_value(value, item)

            for organ, weight in item.organ_weights.items():
                organ_shift[organ] += weight * abs(delta)

        # Convert shifts to impedances
        z_patient: Dict[str, float] = {}
        for organ in ORGAN_LIST:
            z_normal = self.organ_z_normal[organ]
            z_patient[organ] = z_normal * (1.0 + organ_shift[organ])

        return z_patient

    def compute_organ_impedances_detailed(
        self,
        lab_values: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, List[Tuple[str, float, float]]]]:
        """Like compute_organ_impedances but also returns per-lab contributions.

        Returns
        -------
        (z_patient, contributions)
        contributions : dict[organ_id, list[(lab_name, delta, weighted_delta)]]
        """
        organ_shift: Dict[str, float] = {organ: 0.0 for organ in ORGAN_LIST}
        contributions: Dict[str, List[Tuple[str, float, float]]] = {
            organ: [] for organ in ORGAN_LIST
        }

        for lab_name, value in lab_values.items():
            if lab_name not in self.catalogue:
                continue
            item = self.catalogue[lab_name]
            delta = normalise_lab_value(value, item)

            for organ, weight in item.organ_weights.items():
                wd = weight * abs(delta)
                organ_shift[organ] += wd
                contributions[organ].append((lab_name, delta, wd))

        z_patient: Dict[str, float] = {}
        for organ in ORGAN_LIST:
            z_normal = self.organ_z_normal[organ]
            z_patient[organ] = z_normal * (1.0 + organ_shift[organ])
            # Sort contributions by magnitude (descending)
            contributions[organ].sort(key=lambda x: abs(x[2]), reverse=True)

        return z_patient, contributions

    @property
    def available_labs(self) -> List[str]:
        """List of all recognised lab item names."""
        return sorted(self.catalogue.keys())
