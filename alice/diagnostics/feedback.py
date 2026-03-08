# -*- coding: utf-8 -*-
"""feedback.py — Closed-Loop Impedance-Remodeling Feedback for Lab-Γ Engine
================================================================

Physics:
    When a physician confirms or rejects a diagnosis, the system
    updates its internal impedance mapping weights via C2:

        ΔW_organ = −η · Γ_error · x_in · x_out

    where:
        Γ_error  =  Γ_predicted − Γ_target
        x_in    =  normalised lab deviation (input signal)
        x_out   =  organ activation (|Γ_predicted|)
        η        =  learning rate (default 0.01)

    This is the impedance-remodeling rule of Paper 1 applied
    to the diagnostic mapping weights rather than membrane impedances.

    Energy conservation (C1) is preserved because weight updates only
    redistribute how lab deviations map to organ impedances — they do
    not create or destroy signal energy.

Feedback types:
    CONFIRM  — Doctor confirms the top diagnosis → reinforce mapping
    REJECT   — Doctor rejects a diagnosis → push mapping away
    CORRECT  — Doctor provides the true diagnosis → pull toward it

Usage:
    >>> from alice.diagnostics.feedback import FeedbackEngine, FeedbackRecord
    >>> fb = FeedbackEngine(engine)
    >>> fb.record_confirm(lab_values, confirmed_disease_id="mi_acute")
    >>> fb.record_reject(lab_values, rejected_disease_id="pneumonia")
    >>> fb.record_correct(lab_values, true_disease_id="pe_acute")
    >>> fb.apply_pending()          # batch-apply all pending updates
    >>> fb.save("feedback_log.json")
"""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alice.diagnostics.lab_mapping import ORGAN_LIST, LabMapper


# ============================================================================
# 1. Feedback Record
# ============================================================================

class FeedbackType(str, Enum):
    """Type of physician feedback."""
    CONFIRM = "confirm"     # top diagnosis is correct
    REJECT = "reject"       # a specific diagnosis is wrong
    CORRECT = "correct"     # doctor provides the true diagnosis


@dataclass
class FeedbackRecord:
    """A single feedback event from a physician.

    Stores all context needed to replay the impedance remodeling.
    """
    timestamp: float                     # Unix timestamp
    feedback_type: str                   # FeedbackType value
    lab_values: Dict[str, float]         # original input
    predicted_gamma: Dict[str, float]    # Γ vector at time of prediction
    target_disease_id: str               # disease being confirmed/rejected/corrected-to
    target_gamma: Dict[str, float]       # Γ signature of the target disease
    weight_deltas: Dict[str, float]      # computed ΔW per organ (for auditability)
    applied: bool = False                # whether this update has been applied

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to JSON-safe dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeedbackRecord":
        """Reconstruct from dict."""
        return cls(**d)


# ============================================================================
# 2. impedance-remodeling weight Updater
# ============================================================================

class ImpedanceUpdater:
    """Computes C2-compliant weight updates for the Lab→Z mapping.

    The impedance-remodeling rule:
        ΔW_organ = −η · Γ_error · Σ_j |δ_j| · w_j,organ

    where the sum runs over lab items contributing to that organ.

    For CONFIRM:  Γ_error = Γ_predicted − Γ_target  (small → small update)
    For REJECT:   Γ_error = −(Γ_predicted − Γ_target) (push away)
    For CORRECT:  Γ_error = Γ_predicted − Γ_true_disease (pull toward truth)
    """

    def __init__(self, eta: float = 0.01, max_delta: float = 0.05):
        """
        Parameters
        ----------
        eta : float
            Learning rate.  Default 0.01 is conservative.
        max_delta : float
            Maximum absolute weight change per update (gradient clipping).
        """
        self.eta = eta
        self.max_delta = max_delta

    def compute_deltas(
        self,
        predicted_gamma: Dict[str, float],
        target_gamma: Dict[str, float],
        feedback_type: FeedbackType,
    ) -> Dict[str, float]:
        """Compute per-organ weight deltas.

        Parameters
        ----------
        predicted_gamma : dict[str, float]
            The patient's Γ vector at prediction time.
        target_gamma : dict[str, float]
            The Γ signature of the target disease.
        feedback_type : FeedbackType
            CONFIRM / REJECT / CORRECT.

        Returns
        -------
        dict[str, float]
            ΔW per organ system.
        """
        deltas: Dict[str, float] = {}

        for organ in ORGAN_LIST:
            g_pred = predicted_gamma.get(organ, 0.0)
            g_target = target_gamma.get(organ, 0.0)

            # Γ error = predicted − target
            gamma_error = g_pred - g_target

            # x_in ≈ |Γ_predicted| (how active this organ is)
            x_in = abs(g_pred)

            # x_out ≈ |Γ_target| (how much this organ matters for the disease)
            x_out = abs(g_target)

            # C2 impedance-remodeling: ΔW = −η · Γ_error · x_in · x_out
            dw = -self.eta * gamma_error * x_in * x_out

            # Sign flip for REJECT: push away instead of pull toward
            if feedback_type == FeedbackType.REJECT:
                dw = -dw

            # Gradient clipping
            dw = np.clip(dw, -self.max_delta, self.max_delta)

            deltas[organ] = float(dw)

        return deltas


# ============================================================================
# 3. Feedback Engine — Orchestrator
# ============================================================================

class FeedbackEngine:
    """Closed-loop feedback controller for the Lab-Γ diagnostic engine.

    Manages the feedback → impedance remodeling → weight persistence cycle.

    Usage
    -----
    >>> from alice.diagnostics import GammaEngine, load_disease_templates
    >>> engine = GammaEngine(templates=load_disease_templates())
    >>> fb = FeedbackEngine(engine)
    >>> fb.record_confirm(lab_values, "mi_acute")
    >>> fb.apply_pending()
    >>> fb.stats()
    {'total': 1, 'confirm': 1, 'reject': 0, 'correct': 0, 'applied': 1}
    """

    def __init__(
        self,
        engine: Any,  # GammaEngine — Any to avoid circular import
        updater: Optional[ImpedanceUpdater] = None,
    ):
        self.engine = engine
        self.updater = updater or ImpedanceUpdater()
        self.records: List[FeedbackRecord] = []

        # Cumulative weight offsets (additive to GammaEngine.organ_weights)
        self._weight_offsets: Dict[str, float] = {o: 0.0 for o in ORGAN_LIST}

    # ---- Recording ----

    def _find_template(self, disease_id: str) -> Optional[Dict[str, float]]:
        """Look up the Γ signature of a disease by ID."""
        for tmpl in self.engine.templates:
            if tmpl.disease_id == disease_id:
                return dict(tmpl.gamma_signature)
        return None

    def _make_record(
        self,
        lab_values: Dict[str, float],
        disease_id: str,
        feedback_type: FeedbackType,
    ) -> FeedbackRecord:
        """Create a FeedbackRecord for the given feedback event."""
        # Compute current patient Γ
        gamma_vec = self.engine.lab_to_gamma(lab_values)
        predicted_gamma = dict(gamma_vec.values)

        # Find target disease Γ
        target_gamma = self._find_template(disease_id)
        if target_gamma is None:
            # Unknown disease — use zero template (maximises error signal)
            target_gamma = {o: 0.0 for o in ORGAN_LIST}

        # Compute impedance-remodeling deltas
        deltas = self.updater.compute_deltas(
            predicted_gamma, target_gamma, feedback_type
        )

        return FeedbackRecord(
            timestamp=time.time(),
            feedback_type=feedback_type.value,
            lab_values=dict(lab_values),
            predicted_gamma=predicted_gamma,
            target_disease_id=disease_id,
            target_gamma=target_gamma,
            weight_deltas=deltas,
            applied=False,
        )

    def record_confirm(
        self,
        lab_values: Dict[str, float],
        confirmed_disease_id: str,
    ) -> FeedbackRecord:
        """Record that the physician confirmed a diagnosis.

        This reinforces the current mapping: ΔW pulls Γ closer to template.
        """
        rec = self._make_record(lab_values, confirmed_disease_id, FeedbackType.CONFIRM)
        self.records.append(rec)
        return rec

    def record_reject(
        self,
        lab_values: Dict[str, float],
        rejected_disease_id: str,
    ) -> FeedbackRecord:
        """Record that the physician rejected a diagnosis.

        This pushes the mapping away: ΔW moves Γ away from template.
        """
        rec = self._make_record(lab_values, rejected_disease_id, FeedbackType.REJECT)
        self.records.append(rec)
        return rec

    def record_correct(
        self,
        lab_values: Dict[str, float],
        true_disease_id: str,
    ) -> FeedbackRecord:
        """Record the physician's corrected (true) diagnosis.

        This pulls toward truth: ΔW moves Γ toward the correct template.
        """
        rec = self._make_record(lab_values, true_disease_id, FeedbackType.CORRECT)
        self.records.append(rec)
        return rec

    # ---- Applying Updates ----

    def apply_pending(self) -> int:
        """Apply all pending (unapplied) impedance remodelings to engine weights.

        Returns the number of updates applied.

        Physics:
            W_organ(t+1) = W_organ(t) + ΔW_organ
            where ΔW is computed by C2 impedance-remodeling rule.
            Weights are clamped to [0.1, 10.0] to prevent degenerate solutions.
        """
        count = 0
        for rec in self.records:
            if not rec.applied:
                for organ in ORGAN_LIST:
                    dw = rec.weight_deltas.get(organ, 0.0)
                    self._weight_offsets[organ] += dw
                    # Apply to engine
                    new_w = self.engine.organ_weights.get(organ, 1.0) + dw
                    # Clamp to prevent degenerate zero or negative weights
                    new_w = max(0.1, min(10.0, new_w))
                    self.engine.organ_weights[organ] = new_w
                rec.applied = True
                count += 1
        return count

    def apply_single(self, record: FeedbackRecord) -> None:
        """Apply a single feedback record's weight deltas."""
        if record.applied:
            return
        for organ in ORGAN_LIST:
            dw = record.weight_deltas.get(organ, 0.0)
            self._weight_offsets[organ] += dw
            new_w = self.engine.organ_weights.get(organ, 1.0) + dw
            new_w = max(0.1, min(10.0, new_w))
            self.engine.organ_weights[organ] = new_w
        record.applied = True

    # ---- Persistence ----

    def save(self, filepath: str) -> None:
        """Save all feedback records to a JSON file.

        The file includes both the records and the cumulative weight offsets,
        enabling full replay and auditability.
        """
        data = {
            "version": "2.0",
            "total_records": len(self.records),
            "weight_offsets": dict(self._weight_offsets),
            "records": [r.to_dict() for r in self.records],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, filepath: str) -> int:
        """Load feedback records from a JSON file.

        Returns the number of records loaded.
        Records marked as applied will not be re-applied.
        """
        if not os.path.exists(filepath):
            return 0
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded = 0
        for rec_dict in data.get("records", []):
            rec = FeedbackRecord.from_dict(rec_dict)
            self.records.append(rec)
            loaded += 1

        # Restore cumulative offsets
        offsets = data.get("weight_offsets", {})
        for organ in ORGAN_LIST:
            self._weight_offsets[organ] = offsets.get(organ, 0.0)

        return loaded

    # ---- Replay ----

    def replay_all(self) -> int:
        """Replay all records from scratch (reset weights first).

        Useful after loading from file to reconstruct learned state.
        Returns the number of records replayed.
        """
        # Reset engine weights to default
        from alice.diagnostics.lab_mapping import ORGAN_SYSTEMS
        for organ in ORGAN_LIST:
            self.engine.organ_weights[organ] = 1.0
        self._weight_offsets = {o: 0.0 for o in ORGAN_LIST}

        # Mark all as unapplied
        for rec in self.records:
            rec.applied = False

        # Apply all in order
        return self.apply_pending()

    # ---- Statistics ----

    def stats(self) -> Dict[str, int]:
        """Return summary statistics of feedback history."""
        result = {
            "total": len(self.records),
            "confirm": 0,
            "reject": 0,
            "correct": 0,
            "applied": 0,
            "pending": 0,
        }
        for rec in self.records:
            result[rec.feedback_type] = result.get(rec.feedback_type, 0) + 1
            if rec.applied:
                result["applied"] += 1
            else:
                result["pending"] += 1
        return result

    @property
    def weight_offsets(self) -> Dict[str, float]:
        """Cumulative weight offsets from all applied feedback."""
        return dict(self._weight_offsets)

    @property
    def current_weights(self) -> Dict[str, float]:
        """Current engine organ weights (base + offsets)."""
        return dict(self.engine.organ_weights)

    def weight_drift_report(self) -> str:
        """Human-readable report of how weights have drifted from defaults."""
        lines = ["── Weight Drift Report ──"]
        lines.append(f"{'Organ':<12} {'Default':>8} {'Current':>8} {'Offset':>8}")
        lines.append("─" * 40)
        for organ in ORGAN_LIST:
            default = 1.0
            current = self.engine.organ_weights.get(organ, 1.0)
            offset = self._weight_offsets.get(organ, 0.0)
            flag = " ⚠" if abs(offset) > 0.1 else ""
            lines.append(
                f"{organ:<12} {default:>8.3f} {current:>8.3f} {offset:>+8.4f}{flag}"
            )
        return "\n".join(lines)
