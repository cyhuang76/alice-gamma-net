#!/usr/bin/env python3
"""Generate figures for P1 and P2 papers."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

# ===== Figure 1: Cognitive Standing Wave (P1) =====
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

x = np.linspace(0, 4*np.pi, 500)
incident = np.sin(x)

# Panel A: Novel stimulus (Gamma ~ 1)
gamma_high = 0.9
reflected_a = gamma_high * np.sin(-x + 0.3)
standing_a = incident + reflected_a
axes[0].plot(x, incident, 'b-', alpha=0.3, label=r'Incident $P_{\mathrm{in}}$')
axes[0].plot(x, reflected_a, 'r-', alpha=0.3, label=r'Reflected $\Gamma P_{\mathrm{in}}$')
axes[0].plot(x, standing_a, 'k-', lw=2, label='Standing wave')
axes[0].set_title('(a) Novel stimulus\n' + r'$\Gamma \approx 1$: "What is this?"', fontsize=10)
axes[0].set_ylabel('Amplitude')
axes[0].legend(fontsize=7, loc='upper right')
axes[0].set_xticks([])
axes[0].set_ylim(-2.2, 2.2)

# Panel B: Learned memory (epsilon > 0)
gamma_mid = 0.15
reflected_b = gamma_mid * np.sin(-x + 0.3)
standing_b = incident + reflected_b
axes[1].plot(x, incident, 'b-', alpha=0.3)
axes[1].plot(x, reflected_b, 'r-', alpha=0.3)
axes[1].plot(x, standing_b, 'k-', lw=2)
nodes = [np.pi*0.85, np.pi*1.85, np.pi*2.85, np.pi*3.85]
for n in nodes:
    axes[1].axvline(n, color='green', ls=':', alpha=0.5)
axes[1].set_title('(b) Learned memory\n' + r'$\varepsilon = 0.15$: "Apple!" (resonance)', fontsize=10)
axes[1].set_xticks([])
axes[1].set_ylim(-2.2, 2.2)
axes[1].annotate('nodes', xy=(np.pi*0.85, -1.9), fontsize=8, color='green', ha='center')

# Panel C: Muscle memory (Gamma = 0)
axes[2].plot(x, incident, 'b-', lw=2, label=r'Pure transmission ($\Gamma = 0$)')
axes[2].axhline(0, color='gray', ls='-', alpha=0.2)
axes[2].set_title('(c) Muscle memory\n' + r'$\Gamma \to 0$: automatic, unconscious', fontsize=10)
axes[2].set_xticks([])
axes[2].set_ylim(-2.2, 2.2)
axes[2].legend(fontsize=7, loc='upper right')

for ax in axes:
    ax.set_xlabel('Neural relay chain position')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/fig_p1_standing_wave.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig_p1_standing_wave.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig 1 (standing wave) saved.')


# ===== Figure 2: Brain Architecture Causal Chain (P1) =====
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.axis('off')

steps = [
    ('Impedance\nDisparity', r'$Z$ spans orders' + '\nof magnitude', '#e74c3c'),
    ('Relay\nNecessity', 'Multi-stage\nmatching', '#e67e22'),
    ('Bandwidth\nIncrease', 'Chebyshev\ntransformer', '#f1c40f'),
    ('Thermal\nCost', r'$\Gamma^2 P_{\mathrm{in}}$' + '\n= waste heat', '#3498db'),
    ('Sleep &\nAging', r'$D_Z$ clearance' + '\n+ plastic drift', '#9b59b6'),
]

for i, (title, desc, color) in enumerate(steps):
    cx = 0.1 + i * 0.19
    # Box
    rect = plt.Rectangle((cx-0.07, 0.3), 0.14, 0.45, 
                          facecolor=color, alpha=0.15, edgecolor=color, lw=2,
                          transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(cx, 0.62, title, transform=ax.transAxes, ha='center', va='center',
            fontsize=11, fontweight='bold', color=color)
    ax.text(cx, 0.42, desc, transform=ax.transAxes, ha='center', va='center',
            fontsize=8, color='#333333')
    # Arrow
    if i < len(steps) - 1:
        ax.annotate('', xy=(cx+0.10, 0.525), xytext=(cx+0.14, 0.525),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', lw=2, color='#555555'))

# Labels
ax.text(0.03, 0.85, 'P2', transform=ax.transAxes, fontsize=9, color='#e74c3c',
        fontweight='bold')
ax.text(0.22, 0.85, 'P1', transform=ax.transAxes, fontsize=9, color='#e67e22',
        fontweight='bold')
ax.text(0.41, 0.85, 'P1', transform=ax.transAxes, fontsize=9, color='#f1c40f',
        fontweight='bold')
ax.text(0.60, 0.85, 'P1', transform=ax.transAxes, fontsize=9, color='#3498db',
        fontweight='bold')
ax.text(0.79, 0.85, 'P3', transform=ax.transAxes, fontsize=9, color='#9b59b6',
        fontweight='bold')

# Bottom summary
ax.text(0.5, 0.08, 
        r'Intelligence $\leftarrow$ deep relay hierarchy $\leftarrow$ impedance physics'
        r'    |    Price: waste heat, sleep, aging',
        transform=ax.transAxes, ha='center', va='center',
        fontsize=10, style='italic', color='#555555',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='#cccccc'))

plt.tight_layout()
plt.savefig('figures/fig_p1_causal_chain.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig_p1_causal_chain.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig 2 (causal chain) saved.')


# ===== Figure 3: Blood as Crosstalk Isolator (P2) =====
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: Microwave analogy
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('(a) Microwave circuit', fontsize=11, fontweight='bold')

# Channel A
ax.add_patch(plt.Rectangle((1, 5.5), 8, 1.2, facecolor='#3498db', alpha=0.2, edgecolor='#3498db', lw=2))
ax.text(5, 6.1, 'Channel A', ha='center', fontsize=10, color='#2c3e50')
# Absorber
ax.add_patch(plt.Rectangle((1, 3.8), 8, 1.5, facecolor='#e74c3c', alpha=0.15, edgecolor='#e74c3c', lw=2, ls='--'))
ax.text(5, 4.55, 'Ferrite absorber\n(crosstalk isolation)', ha='center', fontsize=9, color='#c0392b')
# Channel B
ax.add_patch(plt.Rectangle((1, 2.4), 8, 1.2, facecolor='#3498db', alpha=0.2, edgecolor='#3498db', lw=2))
ax.text(5, 3.0, 'Channel B', ha='center', fontsize=10, color='#2c3e50')

# Stray pulse arrow (blocked)
ax.annotate('', xy=(5, 4.0), xytext=(5, 5.5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='red', ls='--'))
ax.text(5.8, 4.7, r'Stray $\Gamma$ pulse', fontsize=8, color='red', style='italic')
ax.text(6.5, 4.2, '(absorbed)', fontsize=8, color='#c0392b', style='italic')

# Right: Biological analogy
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('(b) Biological tissue', fontsize=11, fontweight='bold')

# Neural interface A
ax.add_patch(plt.Rectangle((1, 5.5), 8, 1.2, facecolor='#2ecc71', alpha=0.2, edgecolor='#27ae60', lw=2))
ax.text(5, 6.1, 'Neural interface A', ha='center', fontsize=10, color='#27ae60')
# Blood
ax.add_patch(plt.Rectangle((1, 3.8), 8, 1.5, facecolor='#e74c3c', alpha=0.15, edgecolor='#e74c3c', lw=2, ls='--'))
ax.text(5, 4.55, r'Blood ($Z_{\mathrm{blood}}$)' + '\ncrosstalk isolation + heat removal', ha='center', fontsize=9, color='#c0392b')
# Neural interface B
ax.add_patch(plt.Rectangle((1, 2.4), 8, 1.2, facecolor='#2ecc71', alpha=0.2, edgecolor='#27ae60', lw=2))
ax.text(5, 3.0, 'Neural interface B', ha='center', fontsize=10, color='#27ae60')

# Stray pulse arrow (blocked)
ax.annotate('', xy=(5, 4.0), xytext=(5, 5.5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='red', ls='--'))
ax.text(5.8, 4.7, r'Stray $\Gamma^2 P_{\mathrm{in}}$', fontsize=8, color='red', style='italic')
ax.text(6.5, 4.2, r'$\to$ heat $\to$ flow', fontsize=8, color='#c0392b', style='italic')

plt.tight_layout()
plt.savefig('figures/fig_p2_blood_isolation.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig_p2_blood_isolation.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig 3 (blood isolation) saved.')

print('All figures generated.')
