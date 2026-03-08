# -*- coding: utf-8 -*-
"""web_ui.py — Lab-Γ Diagnostic Engine Streamlit Dashboard
============================================================

Launch:
    streamlit run alice/diagnostics/web_ui.py

Features:
    - Lab value input (sidebar with 53 sliders or manual entry)
    - 12-D Γ radar chart (patient vs healthy vs top disease template)
    - Differential diagnosis table with confidence bars
    - C1 verification panel (Γ² + T = 1)
    - Physician feedback buttons (confirm / reject / correct)
    - Weight drift monitor
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from alice.diagnostics.lab_mapping import (
    LAB_CATALOGUE,
    ORGAN_LIST,
    ORGAN_SYSTEMS,
    LabMapper,
)
from alice.diagnostics.gamma_engine import GammaEngine, PatientGammaVector
from alice.diagnostics.disease_templates import load_disease_templates
from alice.diagnostics.feedback import FeedbackEngine, ImpedanceUpdater


# ============================================================================
# Page Config
# ============================================================================

st.set_page_config(
    page_title="Lab-Γ Diagnostic Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Session State — persistent engine
# ============================================================================

if "engine" not in st.session_state:
    templates = load_disease_templates()
    st.session_state.engine = GammaEngine(templates=templates)
    st.session_state.feedback = FeedbackEngine(st.session_state.engine)

engine: GammaEngine = st.session_state.engine
fb: FeedbackEngine = st.session_state.feedback


# ============================================================================
# Sidebar — Lab Value Input
# ============================================================================

st.sidebar.title("🔬 Laboratory Values")
st.sidebar.caption("Enter measured values. Leave at 0 to skip.")

# Preset profiles
PRESETS = {
    "(Custom)": {},
    "Acute Hepatitis": {"AST": 480, "ALT": 520, "Bil_total": 3.2, "Alb": 2.8, "INR": 1.8, "WBC": 12.5, "CRP": 45.0},
    "Acute MI": {"Troponin_I": 15.0, "CK_MB": 80.0, "BNP": 1200.0, "WBC": 14.0, "Glucose": 180.0, "CRP": 35.0},
    "DKA": {"Glucose": 450.0, "HbA1c": 12.5, "BUN": 35.0, "Cr": 1.8, "Na": 128.0, "K": 5.8, "WBC": 15.0},
    "Sepsis": {"WBC": 22.0, "CRP": 180.0, "Cr": 2.5, "Bil_total": 3.0, "Lactate": 5.0, "INR": 1.6, "Plt": 80.0},
    "Iron-deficiency Anaemia": {"Hb": 7.5, "MCV": 68.0, "Fe": 20.0, "Ferritin": 5.0, "TIBC": 450.0},
    "CKD Stage 4": {"Cr": 4.5, "BUN": 65.0, "GFR": 18.0, "K": 5.5, "Ca": 7.8, "Phosphate": 6.5, "Hb": 9.0},
    "Hyperthyroidism": {"TSH": 0.01, "FT4": 5.5, "FT3": 12.0, "HR_proxy": 120.0},
    "Pneumonia": {"WBC": 18.0, "CRP": 120.0, "SpO2": 88.0, "Lactate": 3.0},
}

preset = st.sidebar.selectbox("📋 Preset Profile", list(PRESETS.keys()))

# Group labs by category
lab_groups = {}
for lab in LAB_CATALOGUE.values():
    cat = list(lab.organ_weights.keys())[0] if lab.organ_weights else "other"
    lab_groups.setdefault(cat, []).append(lab)

lab_values: dict[str, float] = {}

if preset != "(Custom)":
    lab_values = dict(PRESETS[preset])
    st.sidebar.info(f"Loaded preset: **{preset}**")
    st.sidebar.json(lab_values)
else:
    # Manual input mode
    input_mode = st.sidebar.radio("Input mode", ["Quick entry (text)", "Sliders"], horizontal=True)

    if input_mode == "Quick entry (text)":
        st.sidebar.markdown("Enter as `LabName: value` per line:")
        raw = st.sidebar.text_area(
            "Lab values",
            value="AST: 480\nALT: 520\nBil_total: 3.2",
            height=200,
        )
        for line in raw.strip().split("\n"):
            line = line.strip()
            if ":" in line:
                parts = line.split(":", 1)
                name = parts[0].strip()
                try:
                    val = float(parts[1].strip())
                    if val != 0:
                        lab_values[name] = val
                except ValueError:
                    pass
    else:
        # Slider mode — show common labs
        common_labs = [
            "AST", "ALT", "Bil_total", "Alb", "INR",
            "Cr", "BUN", "GFR", "Na", "K",
            "Hb", "WBC", "Plt", "CRP",
            "Troponin_I", "BNP", "Glucose", "HbA1c",
            "TSH", "FT4", "SpO2",
        ]
        for lab in LAB_CATALOGUE.values():
            if lab.name in common_labs:
                mid = (lab.ref_low + lab.ref_high) / 2 if lab.ref_low is not None and lab.ref_high is not None else 50.0
                max_val = (lab.ref_high or 100.0) * 5
                val = st.sidebar.slider(
                    f"{lab.name} ({lab.unit})",
                    min_value=0.0,
                    max_value=max_val,
                    value=0.0,
                    step=max_val / 200,
                    key=f"slider_{lab.name}",
                )
                if val > 0:
                    lab_values[lab.name] = val

# Filter zero values
lab_values = {k: v for k, v in lab_values.items() if v != 0}


# ============================================================================
# Main Display
# ============================================================================

st.title("⚡ Lab-Γ Diagnostic Engine")
st.caption("Impedance-based differential diagnosis • Γ = (Z_patient − Z_normal) / (Z_patient + Z_normal)")

if not lab_values:
    st.info("👈 Enter laboratory values in the sidebar to begin diagnosis.")
    st.stop()

# ── Run diagnosis ──────────────────────────────────────────────────────────

gamma_vec, z_patient, candidates = engine.diagnose_detailed(lab_values, top_n=10)

# ── Layout: 2 columns ─────────────────────────────────────────────────────

col_radar, col_table = st.columns([1, 1])

# ============================================================================
# Γ Radar Chart
# ============================================================================

with col_radar:
    st.subheader("🎯 Organ Γ Radar")

    # Patient Γ values
    patient_g = [gamma_vec[o] for o in ORGAN_LIST]
    patient_g.append(patient_g[0])  # close polygon

    labels = ORGAN_LIST + [ORGAN_LIST[0]]

    fig = go.Figure()

    # Patient trace
    fig.add_trace(go.Scatterpolar(
        r=[abs(g) for g in patient_g],
        theta=labels,
        fill="toself",
        name="Patient |Γ|",
        fillcolor="rgba(255, 65, 54, 0.25)",
        line=dict(color="rgba(255, 65, 54, 0.9)", width=2),
    ))

    # Top disease template overlay (if any)
    if candidates:
        top = candidates[0]
        tmpl_g = []
        for o in ORGAN_LIST:
            for om in top.organ_matches:
                if om[0] == o:
                    tmpl_g.append(abs(om[2]))
                    break
            else:
                tmpl_g.append(0.0)
        tmpl_g.append(tmpl_g[0])

        fig.add_trace(go.Scatterpolar(
            r=tmpl_g,
            theta=labels,
            fill="toself",
            name=f"Template: {top.display_name}",
            fillcolor="rgba(44, 160, 101, 0.15)",
            line=dict(color="rgba(44, 160, 101, 0.8)", width=2, dash="dot"),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
        ),
        showlegend=True,
        height=450,
        margin=dict(l=60, r=60, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Health summary
    h = gamma_vec.health_index
    g2 = gamma_vec.total_gamma_squared
    col_h1, col_h2 = st.columns(2)
    col_h1.metric("Health Index H", f"{h:.4f}", delta=f"{h - 1.0:+.4f}" if h < 1.0 else None)
    col_h2.metric("Total Γ²", f"{g2:.4f}")


# ============================================================================
# Differential Diagnosis Table
# ============================================================================

with col_table:
    st.subheader("📋 Differential Diagnosis")

    if not candidates:
        st.warning("No matching disease templates found.")
    else:
        for c in candidates[:5]:
            severity_colors = {
                "mild": "🟢", "moderate": "🟡",
                "severe": "🟠", "critical": "🔴",
            }
            icon = severity_colors.get(c.severity, "⚪")

            with st.expander(
                f"{icon} #{c.rank} {c.display_name}  —  {c.confidence:.1%}",
                expanded=(c.rank == 1),
            ):
                cols = st.columns(4)
                cols[0].metric("Confidence", f"{c.confidence:.1%}")
                cols[1].metric("Distance", f"{c.distance:.4f}")
                cols[2].metric("Severity", c.severity.upper())
                cols[3].metric("Specialty", c.specialty)

                if c.primary_deviations:
                    st.markdown("**Primary deviations:**")
                    for d in c.primary_deviations:
                        st.markdown(f"- `{d}`")

                if c.suggested_tests:
                    st.markdown(f"**Suggested tests:** {', '.join(c.suggested_tests)}")

                # Feedback buttons
                fb_cols = st.columns(3)
                if fb_cols[0].button(f"✅ Confirm", key=f"confirm_{c.disease_id}"):
                    rec = fb.record_confirm(lab_values, c.disease_id)
                    fb.apply_single(rec)
                    st.success(f"Confirmed: {c.display_name} — weights updated")
                    st.rerun()

                if fb_cols[1].button(f"❌ Reject", key=f"reject_{c.disease_id}"):
                    rec = fb.record_reject(lab_values, c.disease_id)
                    fb.apply_single(rec)
                    st.warning(f"Rejected: {c.display_name} — weights updated")
                    st.rerun()

                if fb_cols[2].button(f"🎯 Correct to this", key=f"correct_{c.disease_id}"):
                    rec = fb.record_correct(lab_values, c.disease_id)
                    fb.apply_single(rec)
                    st.info(f"Corrected to: {c.display_name} — weights updated")
                    st.rerun()


# ============================================================================
# C1 Verification Panel
# ============================================================================

st.markdown("---")
st.subheader("🔬 Physics Verification")

col_c1, col_z = st.columns(2)

with col_c1:
    st.markdown("**C1 Energy Conservation: Γ² + T = 1**")
    c1 = gamma_vec.verify_c1()
    c1_data = []
    all_ok = True
    for organ in ORGAN_LIST:
        g2, t, ok = c1[organ]
        c1_data.append({
            "Organ": organ,
            "Γ²": f"{g2:.6f}",
            "T": f"{t:.6f}",
            "Γ²+T": f"{g2 + t:.10f}",
            "C1": "✅" if ok else "❌",
        })
        if not ok:
            all_ok = False
    if all_ok:
        st.success("C1 holds for all 12 organs ✅")
    else:
        st.error("C1 violation detected!")
    st.dataframe(c1_data, use_container_width=True, hide_index=True)

with col_z:
    st.markdown("**Organ Impedance Breakdown**")
    z_data = []
    mapper = engine.mapper
    for organ in ORGAN_LIST:
        z_n = mapper.organ_z_normal[organ]
        z_p = z_patient[organ]
        g = gamma_vec[organ]
        z_data.append({
            "Organ": organ,
            "Z_normal": f"{z_n:.1f}",
            "Z_patient": f"{z_p:.1f}",
            "Γ": f"{g:.4f}",
            "Status": "⚠ HIGH" if abs(g) > 0.3 else "OK",
        })
    st.dataframe(z_data, use_container_width=True, hide_index=True)


# ============================================================================
# Feedback & Weight Drift
# ============================================================================

st.markdown("---")
col_fb, col_drift = st.columns(2)

with col_fb:
    st.subheader("📊 Feedback Statistics")
    stats = fb.stats()
    st.json(stats)

with col_drift:
    st.subheader("⚖️ Weight Drift")
    offsets = fb.weight_offsets
    if any(abs(v) > 1e-6 for v in offsets.values()):
        drift_data = []
        for organ in ORGAN_LIST:
            drift_data.append({
                "Organ": organ,
                "Weight": f"{engine.organ_weights.get(organ, 1.0):.4f}",
                "Offset": f"{offsets.get(organ, 0.0):+.6f}",
            })
        st.dataframe(drift_data, use_container_width=True, hide_index=True)
    else:
        st.info("No weight drift yet — all weights at default 1.0")


# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.caption(
    "⚠ **This is a research tool, NOT a medical device.** "
    "Final diagnosis must be made by a qualified physician. "
    "| Physics: A[Γ] = ∫ Σ Γ²(t) dt → min "
    "| © 2026 Alice Smart System (AGPL-3.0)"
)
