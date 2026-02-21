"""Generate Paper III Figure 1: PTSD frozen-state phase portrait."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(9, 7))

# ── Phase space: x = queue pressure, y = arousal (Θ) ──

# Healthy attractor region (lower-left)
healthy_x, healthy_y = 0.15, 0.20

# Pathological fixed point (upper-right)
patho_x, patho_y = 0.90, 0.95

# ── Vector field (background flow) ──
X, Y = np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 16))

# Two-attractor dynamics:
# Near healthy: flow toward (0.15, 0.20)
# Near pathological: flow toward (0.90, 0.95)
# Separatrix roughly diagonal

def flow(x, y):
    # Distance to each attractor
    dh = np.sqrt((x - healthy_x)**2 + (y - healthy_y)**2)
    dp = np.sqrt((x - patho_x)**2 + (y - patho_y)**2)
    
    # Blend: if closer to healthy, flow toward healthy; else toward pathological
    w = np.where(dh < dp, 1.0, 0.0)
    # Smooth transition
    sigma = 0.3
    w = 1.0 / (1.0 + np.exp(-(dp - dh) / sigma))
    
    ux = w * (healthy_x - x) * 0.3 + (1 - w) * (patho_x - x) * 0.3
    uy = w * (healthy_y - y) * 0.3 + (1 - w) * (patho_y - y) * 0.3
    return ux, uy

U, V = flow(X, Y)
mag = np.sqrt(U**2 + V**2)
ax.quiver(X, Y, U, V, mag, alpha=0.25, cmap="coolwarm", scale=3, width=0.004, zorder=1)

# ── Traumatic trajectory ──
# Path: healthy → build-up → trauma spike → pathological lock
t = np.linspace(0, 1, 300)

# Smooth parametric path
traj_x = np.where(t < 0.3,
    healthy_x + t / 0.3 * 0.15,                      # slow drift (learning)
    np.where(t < 0.5,
        0.30 + (t - 0.3) / 0.2 * 0.35,              # trauma escalation
        np.where(t < 0.7,
            0.65 + (t - 0.5) / 0.2 * 0.25,           # approaching lock
            patho_x                                     # locked
        )))

traj_y = np.where(t < 0.3,
    healthy_y + t / 0.3 * 0.10,                       # slow arousal
    np.where(t < 0.5,
        0.30 + (t - 0.3) / 0.2 * 0.40,              # arousal spike
        np.where(t < 0.7,
            0.70 + (t - 0.5) / 0.2 * 0.25,           # approaching lock
            patho_y                                     # locked
        )))

# Add slight curvature
traj_x += 0.05 * np.sin(t * np.pi * 2) * (1 - t)
traj_y += 0.03 * np.cos(t * np.pi * 1.5) * (1 - t)

ax.plot(traj_x, traj_y, "k-", linewidth=2.5, zorder=4, label="Traumatic trajectory")

# Direction arrows along trajectory
for i in [60, 120, 180, 240]:
    ax.annotate("", xy=(traj_x[i+5], traj_y[i+5]), xytext=(traj_x[i], traj_y[i]),
                arrowprops=dict(arrowstyle="->", color="black", lw=2), zorder=5)

# ── Attractors ──
# Healthy attractor
ax.plot(healthy_x, healthy_y, "o", color="#2E86C1", markersize=14, zorder=6)
ax.plot(healthy_x, healthy_y, "o", color="#AED6F1", markersize=8, zorder=7)
ax.annotate("Healthy\nattractor", xy=(healthy_x, healthy_y),
            xytext=(healthy_x + 0.12, healthy_y - 0.12),
            fontsize=10, fontweight="bold", color="#2E86C1",
            arrowprops=dict(arrowstyle="->", color="#2E86C1", lw=1.5),
            zorder=8)

# Pathological fixed point
ax.plot(patho_x, patho_y, "s", color="#C0392B", markersize=14, zorder=6)
ax.plot(patho_x, patho_y, "s", color="#F5B7B1", markersize=8, zorder=7)
ax.annotate("Pathological\nfixed point\n(frozen state)", xy=(patho_x, patho_y),
            xytext=(patho_x - 0.28, patho_y - 0.15),
            fontsize=10, fontweight="bold", color="#C0392B",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.5),
            zorder=8)

# ── Event annotations ──
# Act III → Act IV transition
trauma_idx = 90
ax.annotate("Act IV: Trauma\n(extreme pain + stress)",
            xy=(traj_x[trauma_idx], traj_y[trauma_idx]),
            xytext=(0.10, 0.70),
            fontsize=9, ha="center", color="#8B0000",
            arrowprops=dict(arrowstyle="->", color="#8B0000", lw=1.2,
                           connectionstyle="arc3,rad=0.2"),
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3F3", ec="#C0392B"),
            zorder=8)

# Act V: Safe environment but still locked
ax.annotate("Act V: Safe environment\n(system remains locked)",
            xy=(0.88, 0.92),
            xytext=(0.50, 0.55),
            fontsize=9, ha="center", color="#8B0000",
            arrowprops=dict(arrowstyle="->", color="#8B0000", lw=1.2,
                           connectionstyle="arc3,rad=-0.2"),
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3F3", ec="#C0392B"),
            zorder=8)

# ── Thermodynamic trap annotation ──
ax.text(0.72, 0.68,
        "Thermodynamic trap:\ncooling requires queue flush,\nbut impedance lock blocks access",
        fontsize=8, ha="center", va="center", style="italic",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFFFF0", ec="#DAA520", lw=1.2),
        zorder=8)

# ── Separatrix (dashed) ──
sep_x = np.linspace(0, 1, 100)
sep_y = 0.1 + 0.8 * sep_x + 0.1 * np.sin(sep_x * np.pi)
sep_y = np.clip(sep_y, 0, 1)
# Only draw portion that is visible
mask = (sep_y > 0.05) & (sep_y < 0.95) & (sep_x > 0.05) & (sep_x < 0.95)
ax.plot(sep_x[mask], sep_y[mask], "--", color="gray", linewidth=1.5, alpha=0.5,
        label="Basin boundary (approx.)", zorder=2)

# ── Axes ──
ax.set_xlabel("Processing Queue Pressure", fontsize=12, fontweight="bold")
ax.set_ylabel(r"System Arousal ($\Theta$)", fontsize=12, fontweight="bold")
ax.set_title("Figure 1.  PTSD Frozen-State Phase Portrait",
             fontsize=13, fontweight="bold", pad=15)

ax.set_xlim(-0.02, 1.05)
ax.set_ylim(-0.02, 1.05)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0"])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"])

ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("fig1_ptsd_phase_portrait.png", dpi=300, bbox_inches="tight")
plt.savefig("fig1_ptsd_phase_portrait.pdf", dpi=300, bbox_inches="tight")
print("Saved: fig1_ptsd_phase_portrait.png + .pdf")
