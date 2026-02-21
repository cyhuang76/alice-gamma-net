"""Generate Figure 1: Coaxial cable vs. myelinated axon cross-section analogy."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Arc, Wedge
from matplotlib.collections import PatchCollection

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# ── Colour palette ──
c_inner   = "#4A90D9"   # conductor / axoplasm
c_dielec  = "#FFD966"   # dielectric / myelin
c_outer   = "#888888"   # outer shield / extracellular
c_bg      = "#FFFFFF"

# ────────────────────────────────────────
# LEFT: Coaxial Cable Cross-Section
# ────────────────────────────────────────
ax = axes[0]
ax.set_aspect("equal")
ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
ax.set_title("Coaxial Cable", fontsize=14, fontweight="bold", pad=12)

# Outer shield (gray ring)
outer_shield = plt.Circle((0, 0), 2.8, fc=c_outer, ec="black", lw=1.5, zorder=1)
ax.add_patch(outer_shield)

# Dielectric (yellow)
dielectric = plt.Circle((0, 0), 2.3, fc=c_dielec, ec="black", lw=1.0, zorder=2)
ax.add_patch(dielectric)

# Inner conductor (blue)
inner_cond = plt.Circle((0, 0), 0.8, fc=c_inner, ec="black", lw=1.5, zorder=3)
ax.add_patch(inner_cond)

# Background fill outside shield
bg_ring = plt.Circle((0, 0), 3.1, fc="#F0F0F0", ec="none", zorder=0)
ax.add_patch(bg_ring)
# re-add shield on top
outer_shield2 = plt.Circle((0, 0), 2.8, fc=c_outer, ec="black", lw=1.5, zorder=1)
ax.add_patch(outer_shield2)
dielectric2 = plt.Circle((0, 0), 2.3, fc=c_dielec, ec="black", lw=1.0, zorder=2)
ax.add_patch(dielectric2)
inner_cond2 = plt.Circle((0, 0), 0.8, fc=c_inner, ec="black", lw=1.5, zorder=3)
ax.add_patch(inner_cond2)

# Labels with arrows
label_style = dict(fontsize=9, ha="center", va="center",
                   bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9))

ax.annotate("Inner\nconductor", xy=(0, 0), xytext=(-2.4, -2.7),
            fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9))

ax.annotate("Dielectric", xy=(1.5, 0), xytext=(2.8, -2.5),
            fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9))

ax.annotate("Outer shield", xy=(2.55, 0), xytext=(2.8, 2.5),
            fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9))

# Dimension arrows for r_inner and r_outer
ax.annotate("", xy=(0.8, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
ax.text(0.4, 0.25, r"$r_{inner}$", fontsize=10, color="red", ha="center", fontweight="bold")

ax.annotate("", xy=(2.3, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="<->", color="darkred", lw=1.5))
ax.text(1.15, -0.35, r"$r_{outer}$", fontsize=10, color="darkred", ha="center", fontweight="bold")

ax.axis("off")

# ────────────────────────────────────────
# RIGHT: Myelinated Axon Cross-Section
# ────────────────────────────────────────
ax = axes[1]
ax.set_aspect("equal")
ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
ax.set_title("Myelinated Axon", fontsize=14, fontweight="bold", pad=12)

# Extracellular fluid (light background)
ecf = plt.Circle((0, 0), 3.1, fc="#D5E8D4", ec="none", zorder=0)
ax.add_patch(ecf)

# Myelin sheath (concentric rings — layered look)
myelin_outer = plt.Circle((0, 0), 2.5, fc="#F5DEB3", ec="black", lw=1.5, zorder=1)
ax.add_patch(myelin_outer)

# Add wrapping rings to suggest myelin layers
for r in [2.3, 2.1, 1.9, 1.7]:
    ring = plt.Circle((0, 0), r, fc="none", ec="#C4A96A", lw=0.5, ls="--", zorder=2)
    ax.add_patch(ring)

# Inner myelin boundary
inner_myelin = plt.Circle((0, 0), 1.5, fc="#D5E8D4", ec="black", lw=0.8, zorder=3)
ax.add_patch(inner_myelin)

# Axoplasm (core)
axoplasm = plt.Circle((0, 0), 1.0, fc="#7EB6E8", ec="black", lw=1.5, zorder=4)
ax.add_patch(axoplasm)

# Axon membrane (tight line at axoplasm boundary)
ax_memb = plt.Circle((0, 0), 1.0, fc="none", ec="#2060A0", lw=2.0, zorder=5)
ax.add_patch(ax_memb)

# Labels
ax.annotate("Axoplasm", xy=(0, 0), xytext=(-2.4, -2.7),
            fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9))

ax.annotate("Myelin sheath\n(dielectric)", xy=(2.0, 0), xytext=(2.8, -2.5),
            fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9))

ax.annotate("Extracellular\nfluid (shield)", xy=(2.8, 0.5), xytext=(2.8, 2.5),
            fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9))

# Dimension lines
ax.annotate("", xy=(1.0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
ax.text(0.5, 0.25, r"$r_{inner}$", fontsize=10, color="red", ha="center", fontweight="bold")

ax.annotate("", xy=(2.5, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="<->", color="darkred", lw=1.5))
ax.text(1.25, -0.35, r"$r_{outer}$", fontsize=10, color="darkred", ha="center", fontweight="bold")

ax.axis("off")

# ── Shared elements ──
fig.suptitle(r"Figure 1.  Coaxial Cable $\longleftrightarrow$ Myelinated Axon Analogy",
             fontsize=13, fontweight="bold", y=0.02)

# Γ equation centred below
fig.text(0.5, 0.07,
         r"$\Gamma = \frac{Z_L - Z_0}{Z_L + Z_0}$   governs signal integrity in both systems",
         fontsize=11, ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.4", fc="#FFFFF0", ec="#999", lw=1))

plt.tight_layout(rect=[0, 0.12, 1, 0.98])
plt.savefig("fig1_coaxial_axon.png", dpi=300, bbox_inches="tight")
plt.savefig("fig1_coaxial_axon.pdf", dpi=300, bbox_inches="tight")
print("Saved: fig1_coaxial_axon.png + .pdf")
