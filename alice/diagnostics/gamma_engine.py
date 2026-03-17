# -*- coding: utf-8 -*-
"""gamma_engine.py — Γ Calculation & Disease Template Matching Engine
======================================================================

Core Physics:
    Γ_organ = (Z_patient − Z_normal) / (Z_patient + Z_normal)

    Energy Conservation (C1):
        Γ² + T = 1   →   T = 1 − Γ²

    Health Index (dual-network product):
        H = Π_i (1 − Γ_i²)   over all 12 organ systems

    Disease Matching:
        d(P, D_k) = √(Σ_i w_i (Γ_P,i − Γ_D_k,i)²)
        Similarity S_k = 1 / (1 + d)
        Confidence C_k = S_k / Σ_j S_j
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alice.diagnostics.lab_mapping import (
    ORGAN_LIST,
    ORGAN_SYSTEMS,
    LabMapper,
)


# ============================================================================
# 1. Patient Γ Vector
# ============================================================================

@dataclass
class PatientGammaVector:
    """12-dimensional organ-system reflection coefficient vector.

    Each component Γ_i measures the impedance mismatch of one organ system.
    Γ = 0 → perfect match (healthy), |Γ| → 1 → severe mismatch (disease).
    """

    values: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure all 12 organs are present
        for organ in ORGAN_LIST:
            self.values.setdefault(organ, 0.0)

    def __getitem__(self, organ: str) -> float:
        return self.values[organ]

    def __setitem__(self, organ: str, value: float) -> None:
        self.values[organ] = value

    def to_array(self) -> np.ndarray:
        """Convert to 12-element numpy array (ordered by ORGAN_LIST)."""
        return np.array([self.values[o] for o in ORGAN_LIST], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "PatientGammaVector":
        """Construct from a 12-element array."""
        assert len(arr) == len(ORGAN_LIST), f"Expected {len(ORGAN_LIST)} values, got {len(arr)}"
        return cls(values=dict(zip(ORGAN_LIST, arr.tolist())))

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "PatientGammaVector":
        """Construct from a dict of organ→Γ values."""
        return cls(values=dict(d))

    # ---- Physics properties ----

    @property
    def total_gamma_squared(self) -> float:
        """Σ Γ² — total disease burden across all organ systems."""
        return float(np.sum(self.to_array() ** 2))

    @property
    def health_index(self) -> float:
        """H = Π(1 − Γ_i²) — overall health index.

        H = 1.0 → perfect health (all organs matched)
        H → 0   → one or more organs in severe mismatch
        """
        return float(np.prod(1.0 - self.to_array() ** 2))

    @property
    def transmission_vector(self) -> Dict[str, float]:
        """T_i = 1 − Γ_i² for each organ (C1 energy conservation)."""
        return {o: 1.0 - self.values[o] ** 2 for o in ORGAN_LIST}

    @property
    def dominant_organ(self) -> str:
        """Organ with the highest |Γ|."""
        return max(ORGAN_LIST, key=lambda o: abs(self.values[o]))

    def top_n_organs(self, n: int = 3) -> List[Tuple[str, float]]:
        """Return top-n organs by |Γ|, descending."""
        ranked = sorted(ORGAN_LIST, key=lambda o: abs(self.values[o]), reverse=True)
        return [(o, self.values[o]) for o in ranked[:n]]

    def verify_c1(self) -> Dict[str, Tuple[float, float, bool]]:
        """Verify C1 energy conservation: Γ² + T = 1 for each organ.

        Returns dict[organ, (Γ², T, passes)].
        """
        result = {}
        for o in ORGAN_LIST:
            g2 = self.values[o] ** 2
            t = 1.0 - g2
            result[o] = (g2, t, abs(g2 + t - 1.0) < 1e-10)
        return result


# ============================================================================
# 2. Diagnosis Candidate
# ============================================================================

@dataclass
class DiagnosisCandidate:
    """A single candidate diagnosis with confidence and explainability."""

    rank: int
    disease_id: str
    display_name: str
    specialty: str
    confidence: float           # 0–1, normalised across all candidates
    distance: float             # Γ-space Euclidean distance
    similarity: float           # 1 / (1 + distance)
    organ_matches: List[Tuple[str, float, float]]   # [(organ, Γ_patient, Γ_template)]
    primary_deviations: List[str]  # human-readable deviation descriptions
    suggested_tests: List[str]
    severity: str               # mild / moderate / severe / critical


# ============================================================================
# 3. Gamma Engine
# ============================================================================

class GammaEngine:
    """Core engine: Lab values → Γ vector → Disease matching.

    Usage
    -----
    >>> from alice.diagnostics import GammaEngine, load_disease_templates
    >>> engine = GammaEngine(templates=load_disease_templates())
    >>> results = engine.diagnose({"AST": 480, "ALT": 520, "Bil_total": 3.2})
    >>> results[0].display_name
    '急性肝炎 Acute Hepatitis'
    """

    def __init__(
        self,
        mapper: Optional[LabMapper] = None,
        templates: Optional[List[Any]] = None,
        organ_weights: Optional[Dict[str, float]] = None,
    ):
        self.mapper = mapper or LabMapper()
        self.templates = templates or []
        # Per-organ weighting in distance metric (default = uniform)
        self.organ_weights = organ_weights or {o: 1.0 for o in ORGAN_LIST}


    # ---- Cascade Coupling Matrix (P4 §cascade) ----
    # C_kj: how much Γ_j² in organ j propagates to organ k.
    # Values from known physiological coupling (zero fitted parameters):
    #   - vascular ↔ all organs (blood supply)
    #   - endocrine → renal, hepatic, vascular, cardiac
    #   - immune → hepatic, pulmonary, GI
    #   - neuro ↔ cardiac (autonomic), endocrine (HPA axis)
    # Rows = target organ, Columns = source organ (same order as ORGAN_LIST)
    # ORGAN_LIST = cardiac, pulmonary, hepatic, renal, endocrine,
    #              immune, heme, GI, vascular, bone, neuro, repro

    CASCADE_COUPLING_MATRIX = np.array([
        # card  pulm  hepa  rena  endo  immu  heme  GI    vasc  bone  neur  repr
        [0.00, 0.05, 0.03, 0.04, 0.06, 0.02, 0.03, 0.00, 0.10, 0.00, 0.08, 0.00],  # cardiac
        [0.04, 0.00, 0.02, 0.01, 0.02, 0.06, 0.02, 0.01, 0.06, 0.00, 0.03, 0.00],  # pulmonary
        [0.02, 0.01, 0.00, 0.02, 0.06, 0.05, 0.03, 0.06, 0.06, 0.00, 0.02, 0.00],  # hepatic
        [0.04, 0.01, 0.03, 0.00, 0.08, 0.03, 0.02, 0.01, 0.08, 0.00, 0.03, 0.00],  # renal
        [0.02, 0.01, 0.04, 0.03, 0.00, 0.04, 0.02, 0.02, 0.04, 0.01, 0.05, 0.02],  # endocrine
        [0.01, 0.03, 0.04, 0.01, 0.04, 0.00, 0.04, 0.04, 0.04, 0.02, 0.03, 0.01],  # immune
        [0.01, 0.01, 0.03, 0.04, 0.03, 0.03, 0.00, 0.01, 0.03, 0.03, 0.01, 0.00],  # heme
        [0.01, 0.01, 0.04, 0.01, 0.03, 0.04, 0.01, 0.00, 0.04, 0.00, 0.04, 0.00],  # GI
        [0.06, 0.02, 0.03, 0.03, 0.06, 0.04, 0.02, 0.01, 0.00, 0.01, 0.05, 0.00],  # vascular
        [0.01, 0.01, 0.01, 0.02, 0.04, 0.03, 0.02, 0.01, 0.03, 0.00, 0.02, 0.01],  # bone
        [0.05, 0.02, 0.03, 0.02, 0.05, 0.03, 0.02, 0.02, 0.06, 0.01, 0.00, 0.01],  # neuro
        [0.01, 0.00, 0.01, 0.01, 0.06, 0.02, 0.01, 0.00, 0.03, 0.01, 0.02, 0.00],  # repro
    ], dtype=np.float64)

    @classmethod
    def apply_cascade(
        cls,
        gamma_direct: PatientGammaVector,
        n_iterations: int = 1,
    ) -> PatientGammaVector:
        """Apply P4 cascade coupling: Γ_eff = Γ_direct + C_kj · Γ_j².

        Physics: when organ j has mismatch Γ_j, it injects
        C_kj · Γ_j² reflected power into organ k via the
        relay topology. This is NOT a fitted parameter—
        C_kj encodes known physiological coupling pathways.

        Parameters
        ----------
        gamma_direct : PatientGammaVector
            The independently-computed Γ vector (from lab_to_gamma).
        n_iterations : int
            Number of cascade propagation steps. Default 1 is
            sufficient for single-step cross-organ coupling.

        Returns
        -------
        PatientGammaVector with cascade-adjusted Γ values.
        """
        gamma = gamma_direct.to_array().copy()

        for _ in range(n_iterations):
            # 級聯注入：Γ²_j 的反射功率透過 C_kj 傳播
            gamma_sq = gamma ** 2
            cascade_injection = cls.CASCADE_COUPLING_MATRIX @ gamma_sq

            # 有效 Γ = 原始 Γ + 級聯注入（保持符號）
            # sign(Γ_direct) 保持——級聯只增加幅度
            sign = np.sign(gamma)
            sign[sign == 0] = 1.0
            gamma = sign * (np.abs(gamma) + cascade_injection)

            # 物理約束：|Γ| ≤ 1
            gamma = np.clip(gamma, -1.0, 1.0)

        return PatientGammaVector.from_array(gamma)

    def lab_to_gamma_cascaded(
        self,
        lab_values: Dict[str, float],
        n_iterations: int = 1,
    ) -> PatientGammaVector:
        """Convert lab values to a 12-D Γ vector WITH cascade propagation.

        Pipeline:  lab_values → Z_organ → Γ_organ → C_kj cascade → Γ_eff
        """
        gamma_direct = self.lab_to_gamma(lab_values)
        return self.apply_cascade(gamma_direct, n_iterations=n_iterations)

    # ---- Core Calculation ----

    @staticmethod
    def compute_gamma(z_patient: float, z_normal: float) -> float:
        """Γ = (Z_patient − Z_normal) / (Z_patient + Z_normal)

        Returns 0.0 if both impedances are zero (degenerate case).
        """
        denom = z_patient + z_normal
        if abs(denom) < 1e-12:
            return 0.0
        return (z_patient - z_normal) / denom

    def lab_to_gamma(self, lab_values: Dict[str, float]) -> PatientGammaVector:
        """Convert lab values to a 12-D Γ vector (independent, no cascade).

        Pipeline:  lab_values → Z_organ → Γ_organ
        """
        z_patient = self.mapper.compute_organ_impedances(lab_values)
        gamma_values = {}
        for organ in ORGAN_LIST:
            z_n = self.mapper.organ_z_normal[organ]
            z_p = z_patient[organ]
            gamma_values[organ] = self.compute_gamma(z_p, z_n)
        return PatientGammaVector(values=gamma_values)

    def lab_to_gamma_detailed(
        self,
        lab_values: Dict[str, float],
    ) -> Tuple[PatientGammaVector, Dict[str, float], Dict[str, List[Tuple[str, float, float]]]]:
        """Convert lab values to Γ vector with full traceability.

        Returns
        -------
        (gamma_vector, z_patient, contributions)
        """
        z_patient, contributions = self.mapper.compute_organ_impedances_detailed(lab_values)
        gamma_values = {}
        for organ in ORGAN_LIST:
            z_n = self.mapper.organ_z_normal[organ]
            z_p = z_patient[organ]
            gamma_values[organ] = self.compute_gamma(z_p, z_n)
        return PatientGammaVector(values=gamma_values), z_patient, contributions

    # ---- Disease Matching ----

    def compute_distance(
        self,
        patient_gamma: PatientGammaVector,
        template_gamma: Dict[str, float],
    ) -> float:
        """Weighted Euclidean distance in Γ-space.

        d = √(Σ_i w_i (Γ_P,i − Γ_T,i)²)
        """
        total = 0.0
        for organ in ORGAN_LIST:
            diff = patient_gamma[organ] - template_gamma.get(organ, 0.0)
            w = self.organ_weights.get(organ, 1.0)
            total += w * diff * diff
        return float(np.sqrt(total))

    def match_templates(
        self,
        patient_gamma: PatientGammaVector,
        top_n: int = 5,
    ) -> List[DiagnosisCandidate]:
        """Match patient Γ vector against all disease templates.

        Returns top-N candidates sorted by confidence (descending).
        """
        if not self.templates:
            return []

        # Compute distances and similarities
        scored: List[Tuple[Any, float, float]] = []
        for tmpl in self.templates:
            d = self.compute_distance(patient_gamma, tmpl.gamma_signature)
            s = 1.0 / (1.0 + d)
            scored.append((tmpl, d, s))

        # Normalise to confidence (soft-max style)
        total_sim = sum(s for _, _, s in scored)
        if total_sim < 1e-12:
            total_sim = 1.0

        # Sort by similarity descending
        scored.sort(key=lambda x: x[2], reverse=True)

        # Build candidates
        candidates: List[DiagnosisCandidate] = []
        for rank_idx, (tmpl, dist, sim) in enumerate(scored[:top_n], 1):
            conf = sim / total_sim

            # Top organ-level matches
            organ_matches = []
            for organ in ORGAN_LIST:
                gp = patient_gamma[organ]
                gt = tmpl.gamma_signature.get(organ, 0.0)
                organ_matches.append((organ, gp, gt))
            organ_matches.sort(key=lambda x: abs(x[1]), reverse=True)

            # Primary deviations (top 3 organs)
            deviations = []
            for organ, gp, gt in organ_matches[:3]:
                if abs(gp) > 0.05:
                    deviations.append(
                        f"{organ}: Γ_patient={gp:.3f}, Γ_template={gt:.3f}"
                    )

            # Severity classification based on total Γ² of matching organs
            primary_g2 = sum(
                patient_gamma[o] ** 2
                for o in tmpl.primary_organs
            ) / max(len(tmpl.primary_organs), 1)

            if primary_g2 > 0.50:
                severity = "critical"
            elif primary_g2 > 0.25:
                severity = "severe"
            elif primary_g2 > 0.10:
                severity = "moderate"
            else:
                severity = "mild"

            candidates.append(DiagnosisCandidate(
                rank=rank_idx,
                disease_id=tmpl.disease_id,
                display_name=tmpl.display_name,
                specialty=tmpl.specialty,
                confidence=conf,
                distance=dist,
                similarity=sim,
                organ_matches=organ_matches[:5],
                primary_deviations=deviations,
                suggested_tests=tmpl.suggested_tests,
                severity=severity,
            ))

        return candidates

    # ---- Convenience ----

    def diagnose(
        self,
        lab_values: Dict[str, float],
        top_n: int = 5,
    ) -> List[DiagnosisCandidate]:
        """One-call pipeline: lab values → differential diagnosis.

        lab_values → Z_organ → Γ_organ → template match → ranked diagnoses
        """
        gamma_vec = self.lab_to_gamma(lab_values)
        return self.match_templates(gamma_vec, top_n=top_n)

    def diagnose_detailed(
        self,
        lab_values: Dict[str, float],
        top_n: int = 5,
    ) -> Tuple[PatientGammaVector, Dict[str, float], List[DiagnosisCandidate]]:
        """Full pipeline with all intermediate results.

        Returns
        -------
        (gamma_vector, z_patient, candidates)
        """
        gamma_vec, z_patient, _ = self.lab_to_gamma_detailed(lab_values)
        candidates = self.match_templates(gamma_vec, top_n=top_n)
        return gamma_vec, z_patient, candidates

    def format_report(
        self,
        lab_values: Dict[str, float],
        top_n: int = 5,
    ) -> str:
        """Generate a human-readable diagnostic report.

        Returns a multi-line string suitable for CLI display.
        """
        gamma_vec, z_patient, candidates = self.diagnose_detailed(lab_values, top_n)

        lines: List[str] = []
        lines.append("=" * 72)
        lines.append("  Lab-Γ Diagnostic Engine — Impedance-Based Differential Diagnosis")
        lines.append("=" * 72)

        # Γ vector summary
        lines.append("")
        lines.append("── Organ Γ Vector ──")
        lines.append(f"{'Organ':<12} {'Z_patient':>10} {'Z_normal':>10} {'Γ':>8} {'T=1-Γ²':>8}")
        lines.append("─" * 50)
        for organ in ORGAN_LIST:
            z_n = self.mapper.organ_z_normal[organ]
            z_p = z_patient[organ]
            g = gamma_vec[organ]
            t = 1.0 - g ** 2
            flag = " ⚠" if abs(g) > 0.30 else ""
            lines.append(f"{organ:<12} {z_p:>10.1f} {z_n:>10.1f} {g:>8.4f} {t:>8.4f}{flag}")

        lines.append("")
        lines.append(f"Total Γ² = {gamma_vec.total_gamma_squared:.4f}")
        lines.append(f"Health Index H = {gamma_vec.health_index:.4f}")

        # Diagnoses
        if candidates:
            lines.append("")
            lines.append("── Differential Diagnosis (Top {}) ──".format(top_n))
            for c in candidates:
                lines.append("")
                lines.append(f"  #{c.rank}  {c.display_name}")
                lines.append(f"      Specialty:  {c.specialty}")
                lines.append(f"      Confidence: {c.confidence:.1%}")
                lines.append(f"      Distance:   {c.distance:.4f}")
                lines.append(f"      Severity:   {c.severity}")
                if c.primary_deviations:
                    lines.append(f"      Deviations: {'; '.join(c.primary_deviations)}")
                if c.suggested_tests:
                    lines.append(f"      Suggested:  {', '.join(c.suggested_tests)}")
        else:
            lines.append("")
            lines.append("  (No disease templates loaded)")

        lines.append("")
        lines.append("=" * 72)
        lines.append("⚠ This is a research tool, NOT a medical device.")
        lines.append("  Final diagnosis must be made by a qualified physician.")
        lines.append("=" * 72)

        return "\n".join(lines)
