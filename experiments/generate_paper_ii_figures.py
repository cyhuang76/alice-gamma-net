# -*- coding: utf-8 -*-
"""
Generate Paper II Figures — Vascular Impedance Network
======================================================

Figure 1: paper_ii_murray.pdf — Murray's Law from Vascular Action Principle
Figure 2: paper_ii_cascade.pdf — Positive-feedback cascade
Figure 3: paper_ii_organs.pdf — Organ vascular impedance landscape
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from alice.body.vascular_impedance import (
    verify_murray_law_from_mrp,
    simulate_dual_network_cascade,
    VascularImpedanceNetwork,
    ORGAN_VASCULAR_Z,
)


def figure_murray(save_prefix="figures/paper_ii_murray"):
    """
    Figure 1: Murray's Law from Vascular Action Principle.
    Left: Action A_v(r_d) vs daughter radius, showing Murray minimum.
    Right: Agreement for n=2,3,4.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    r_parent = 5.0
    r_candidates = np.linspace(0.5, 7.0, 2000)
    
    # --- Left: Cost curves for n=2 ---
    n = 2
    lam = 2.0 / (r_parent ** 6)
    r_murray = r_parent / (n ** (1.0/3.0))
    r_impedance = r_parent / (n ** (1.0/4.0))
    
    Z_parent = 1.0 / (r_parent ** 4)
    
    costs_total = []
    costs_dissip = []
    costs_metab = []
    gammas = []
    
    for r_d in r_candidates:
        dissip = 1.0 / (n * r_d ** 4)
        metab = lam * n * r_d ** 2
        costs_dissip.append(dissip)
        costs_metab.append(metab)
        costs_total.append(dissip + metab)
        
        Z_d = 1.0 / (r_d ** 4)
        Z_par = Z_d / n
        g = (Z_par - Z_parent) / (Z_par + Z_parent + 1e-12)
        gammas.append(g ** 2)
    
    costs_total = np.array(costs_total)
    costs_dissip = np.array(costs_dissip)
    costs_metab = np.array(costs_metab)
    gammas = np.array(gammas)
    
    # Normalize for display
    costs_norm = costs_total / costs_total.max()
    
    ax1.plot(r_candidates, costs_norm, 'b-', linewidth=2.5, label=r'$A_v = Q^2/r^4 + \lambda r^2$ (total)')
    ax1.plot(r_candidates, costs_dissip / costs_total.max(), 'r--', linewidth=1.2, alpha=0.7, label=r'Dissipation $Q^2/r^4$')
    ax1.plot(r_candidates, costs_metab / costs_total.max(), 'g--', linewidth=1.2, alpha=0.7, label=r'Metabolic $\lambda r^2$')
    
    # Mark Murray minimum
    idx_murray = np.argmin(np.abs(r_candidates - r_murray))
    ax1.plot(r_murray, costs_norm[idx_murray], 'r*', markersize=18, zorder=5,
             label=f'Murray minimum ($r_p/n^{{1/3}}$={r_murray:.2f})')
    ax1.axvline(r_murray, color='r', linestyle=':', alpha=0.5)
    
    # Mark pure impedance matching
    idx_imp = np.argmin(np.abs(r_candidates - r_impedance))
    ax1.plot(r_impedance, costs_norm[idx_imp], 'kD', markersize=8, zorder=5,
             label=f'$\\Gamma=0$ ($r_p/n^{{1/4}}$={r_impedance:.2f})')
    ax1.axvline(r_impedance, color='k', linestyle=':', alpha=0.3)
    
    ax1.set_xlabel('Daughter radius $r_d$ (mm)', fontsize=12)
    ax1.set_ylabel('Normalized vascular action $A_v$', fontsize=12)
    ax1.set_title(f'Vascular Action ($n={n}$ branches)', fontsize=13)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_xlim(1, 7)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # --- Right: Agreement bar chart for n=2,3,4 ---
    ns = [2, 3, 4]
    r_murrays = []
    r_mrps = []
    r_gammas = []
    
    for n_d in ns:
        result = verify_murray_law_from_mrp(r_parent=5.0, n_daughters=n_d, n_trials=10000)
        r_murrays.append(result['r_murray_predicted'])
        r_mrps.append(result['r_action_optimal'])
        r_gammas.append(result['r_impedance_optimal'])
    
    x = np.arange(len(ns))
    width = 0.25
    
    bars1 = ax2.bar(x - width, r_murrays, width, label="Murray's Law", color='#2196F3', edgecolor='black')
    bars2 = ax2.bar(x, r_mrps, width, label='Vascular MRP', color='#F44336', edgecolor='black')
    bars3 = ax2.bar(x + width, r_gammas, width, label=r'Pure $\Gamma=0$', color='#4CAF50', edgecolor='black')
    
    ax2.set_xlabel('Number of daughter branches $n$', fontsize=12)
    ax2.set_ylabel('Optimal daughter radius (mm)', fontsize=12)
    ax2.set_title('Murray vs MRP vs $\\Gamma=0$', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'n={n}' for n in ns])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add agreement percentages
    for i, (rm, rp) in enumerate(zip(r_murrays, r_mrps)):
        agree = 100. * (1 - abs(rp - rm) / rm)
        ax2.annotate(f'{agree:.1f}%', xy=(x[i] - 0.05, max(rm, rp) + 0.1),
                     fontsize=9, ha='center', fontweight='bold', color='#2196F3')
    
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{save_prefix}.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_prefix}.png/.pdf")


def figure_cascade(save_prefix="figures/paper_ii_cascade"):
    """
    Figure 2: Positive-feedback cascade after vascular insult.
    """
    cascade = simulate_dual_network_cascade(
        organ="brain", n_ticks=500, stenosis_at=100, stenosis_fraction=0.6
    )
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    t = np.arange(500)
    
    # Top: Gamma traces
    ax1.plot(t, cascade['gamma_v_trace'], 'r-', linewidth=2, label=r'$\Gamma_v^2$ (vascular)')
    ax1.plot(t, cascade['gamma_n_trace'], 'b-', linewidth=2, label=r'$\Gamma_n^2$ (neural)')
    ax1.axvline(100, color='gray', linestyle='--', linewidth=1.5, label='Stenosis applied')
    ax1.set_ylabel(r'$\Gamma^2$', fontsize=13)
    ax1.set_title('Positive-Feedback Cascade: $\\Gamma_v\\uparrow \\to \\rho\\downarrow \\to \\Gamma_n\\uparrow \\to \\Gamma_v\\uparrow\\uparrow$', fontsize=12)
    ax1.legend(fontsize=10, loc='center left')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # Annotate stages
    ax1.annotate('Pre-stenosis\n(stable)', xy=(50, 0.97), fontsize=8, ha='center',
                 color='green', fontweight='bold')
    ax1.annotate('Cascade\n(dual failure)', xy=(350, 0.5), fontsize=8, ha='center',
                 color='red', fontweight='bold')
    
    # Bottom: Health and rho
    ax2.plot(t, cascade['health_trace'], 'g-', linewidth=2, label='Organ health $H$')
    ax2.plot(t, cascade['rho_trace'], color='orange', linestyle='--', linewidth=1.5,
             label=r'Material delivery $\rho$')
    ax2.axvline(100, color='gray', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Tick', fontsize=13)
    ax2.set_ylabel('Normalized value', fontsize=13)
    ax2.legend(fontsize=10, loc='center right')
    ax2.set_ylim(-0.01, 0.15)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{save_prefix}.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_prefix}.png/.pdf")


def figure_organs(save_prefix="figures/paper_ii_organs"):
    """
    Figure 3: Vascular impedance landscape across 10 organs.
    """
    organs = []
    gamma_v_sq = []
    rho_delivery = []
    
    for organ in ORGAN_VASCULAR_Z.keys():
        net = VascularImpedanceNetwork(organ)
        state = net.tick(cardiac_output=1.0, blood_pressure=0.85, gamma_neural=0.1)
        organs.append(organ.capitalize())
        gamma_v_sq.append(state.gamma_v_sq)
        rho_delivery.append(state.rho_delivery)
    
    # Sort by gamma_v_sq
    idx = np.argsort(gamma_v_sq)
    organs = [organs[i] for i in idx]
    gamma_v_sq = [gamma_v_sq[i] for i in idx]
    rho_delivery = [rho_delivery[i] for i in idx]
    
    fig, ax1 = plt.subplots(figsize=(9, 5))
    
    x = np.arange(len(organs))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(organs)))
    
    bars = ax1.bar(x, gamma_v_sq, color=colors, edgecolor='black', linewidth=0.8)
    
    # Add rho values as text
    for i, (gv, rho) in enumerate(zip(gamma_v_sq, rho_delivery)):
        ax1.text(i, gv + 0.003, f'ρ={rho:.3f}', ha='center', fontsize=7, 
                 fontweight='bold', color='#1565C0')
    
    ax1.set_xlabel('Organ', fontsize=12)
    ax1.set_ylabel(r'Vascular $\Gamma_v^2$', fontsize=13)
    ax1.set_title('Vascular Impedance Landscape Across 10 Organs', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(organs, rotation=35, ha='right', fontsize=10)
    ax1.set_ylim(0.85, 1.01)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line for average
    avg_gv = np.mean(gamma_v_sq)
    ax1.axhline(avg_gv, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    ax1.text(len(organs)-1, avg_gv + 0.002, f'avg={avg_gv:.3f}', fontsize=8, color='blue')
    
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{save_prefix}.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_prefix}.png/.pdf")


if __name__ == "__main__":
    print("Generating Paper II figures...")
    figure_murray()
    figure_cascade()
    figure_organs()
    print("Done.")
