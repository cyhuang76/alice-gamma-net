# -*- coding: utf-8 -*-
"""
Tests for KidneySystem — Electrolyte Impedance Filter

Tests verify:
    1. Basic construction
    2. GFR depends on BP and maturity
    3. ADH response to dehydration
    4. Electrolyte homeostasis
    5. Waste (urea) clearance
    6. Urine concentration
    7. ElectricalSignal generation
    8. Γ² + T = 1 (C1)
    9. Developmental maturation
"""

import pytest
import numpy as np

from alice.body.kidney import (
    KidneySystem,
    GFR_NEONATAL,
    ADH_BASE,
    ELECTROLYTE_SETPOINTS,
    NEONATAL_KIDNEY_MATURITY,
    Z_BLOOD,
)
from alice.core.signal import ElectricalSignal


class TestKidneyConstruction:
    """KidneySystem initializes correctly."""

    def test_initial_gfr(self):
        kidney = KidneySystem()
        assert kidney._gfr == GFR_NEONATAL

    def test_initial_electrolytes(self):
        kidney = KidneySystem()
        for ion in ELECTROLYTE_SETPOINTS:
            assert ion in kidney._electrolytes

    def test_initial_maturity(self):
        kidney = KidneySystem()
        assert kidney._maturity == NEONATAL_KIDNEY_MATURITY


class TestGFR:
    """Glomerular filtration rate dynamics."""

    def test_high_bp_higher_gfr(self):
        kidney = KidneySystem()
        result_high = kidney.tick(blood_pressure_norm=0.8)
        gfr_high = result_high["gfr"]

        kidney2 = KidneySystem()
        result_low = kidney2.tick(blood_pressure_norm=0.3)
        gfr_low = result_low["gfr"]

        assert gfr_high > gfr_low


class TestADH:
    """ADH response to dehydration."""

    def test_dehydration_raises_adh(self):
        kidney = KidneySystem()
        result = kidney.tick(hydration=0.1)
        assert result["adh"] > ADH_BASE

    def test_hydrated_low_adh(self):
        kidney = KidneySystem()
        result = kidney.tick(hydration=0.9)
        assert result["adh"] <= ADH_BASE + 0.2


class TestElectrolytes:
    """Electrolyte homeostasis."""

    def test_electrolytes_stay_near_setpoint(self):
        kidney = KidneySystem()
        for _ in range(50):
            kidney.tick()
        for ion, setpoint in ELECTROLYTE_SETPOINTS.items():
            assert abs(kidney._electrolytes[ion] - setpoint) < 0.3


class TestWasteClearance:
    """Urea clearance."""

    def test_urea_cleared(self):
        kidney = KidneySystem()
        kidney._urea = 0.5
        for _ in range(30):
            kidney.tick(blood_pressure_norm=0.7)
        assert kidney._urea < 0.5


class TestUrineConcentration:
    """Urine concentration depends on ADH."""

    def test_dehydrated_concentrated_urine(self):
        kidney = KidneySystem()
        result = kidney.tick(hydration=0.1)
        conc_dehydrated = result["urine_concentration"]

        kidney2 = KidneySystem()
        result2 = kidney2.tick(hydration=0.9)
        conc_hydrated = result2["urine_concentration"]

        assert conc_dehydrated >= conc_hydrated


class TestKidneyEnergyConservation:
    """C1: Γ² + T = 1."""

    def test_gamma_transmission_sum(self):
        kidney = KidneySystem()
        kidney.tick()
        gamma = kidney._gamma_renal
        trans = kidney._transmission_renal
        assert abs(gamma ** 2 + trans - 1.0) < 1e-6


class TestKidneySignal:
    """ElectricalSignal generation."""

    def test_signal_type(self):
        kidney = KidneySystem()
        signal = kidney.get_signal()
        assert isinstance(signal, ElectricalSignal)
        assert signal.source == "kidney"


class TestKidneyStats:
    """Statistics."""

    def test_stats_keys(self):
        kidney = KidneySystem()
        stats = kidney.get_stats()
        assert "gfr" in stats
        assert "electrolytes" in stats
        assert "urea" in stats
        assert "adh" in stats
