#!/usr/bin/env python3
"""
generate_paper1_figures.py
==========================
Paper 1 figures: K-space phase separation and television inequality.

Outputs:
    figures/fig_p1_k_clustering.pdf      -- K-space clustering (3 snapshots)
    figures/fig_p1_television_space.pdf   -- Consciousness parameter space
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ============================================================
#  Figure 1: K-space phase separation under C2
# ============================================================
def generate_k_clustering_figure():
    """
    Simulate a toy network of N=40 nodes with K in {1..5}.
    Under C2 (Delta Z = -eta * Gamma * x_in * x_out),
    same-K nodes cluster and different-K nodes separate.
    Show 3 snapshots: t=0 (random), t=50, t=200.
    """
    np.random.seed(42)
    N = 40
    K = np.random.choice([1, 2, 3, 4, 5], size=N)

    # Initial random positions (for spring-layout visualization)
    pos = np.random.randn(N, 2) * 2.0

    # Adjacency: start with random connections
    n_edges = 80
    edges = set()
    while len(edges) < n_edges:
        i, j = np.random.randint(0, N, 2)
        if i != j:
            edges.add((min(i, j), max(i, j)))
    edges = list(edges)

    # Weight matrix (transmission T_ij)
    W = np.zeros((N, N))
    for (i, j) in edges:
        # Gamma = (K_j - K_i) / (K_j + K_i), T = 1 - Gamma^2
        gamma = (K[j] - K[i]) / (K[j] + K[i] + 1e-8)
        T = 1 - gamma**2
        W[i, j] = W[j, i] = T

    # Simulate C2 evolution: strengthen same-K, weaken different-K
    snapshots = {}
    W_history = {}
    snapshot_times = [0, 50, 200]

    W_current = W.copy()
    eta = 0.02
    for t in range(201):
        if t in snapshot_times:
            W_history[t] = W_current.copy()

        # C2 update
        for (i, j) in edges:
            gamma = (K[j] - K[i]) / (K[j] + K[i] + 1e-8)
            dw = eta * (1 - gamma**2) * W_current[i, j]
            if abs(K[i] - K[j]) == 0:
                W_current[i, j] += dw * 0.5
            else:
                W_current[i, j] -= dw * 0.3
            W_current[i, j] = np.clip(W_current[i, j], 0.01, 1.0)
            W_current[j, i] = W_current[i, j]

    # Spring layout based on weights
    def spring_layout(W_mat, pos_init, n_iter=100, k_attract=0.05, k_repel=0.5):
        pos = pos_init.copy()
        for _ in range(n_iter):
            force = np.zeros_like(pos)
            for i in range(N):
                for j in range(i + 1, N):
                    diff = pos[j] - pos[i]
                    dist = np.linalg.norm(diff) + 1e-6
                    # Repulsion
                    f_repel = -k_repel / (dist**2) * diff / dist
                    force[i] += f_repel
                    force[j] -= f_repel
                    # Attraction (weighted)
                    if W_mat[i, j] > 0.1:
                        f_attract = k_attract * W_mat[i, j] * diff
                        force[i] += f_attract
                        force[j] -= f_attract
            pos += force * 0.1
            # Center
            pos -= pos.mean(axis=0)
        return pos

    K_colors = {1: "#BBDEFB", 2: "#64B5F6", 3: "#1976D2",
                4: "#0D47A1", 5: "#311B92"}
    K_cmap = [K_colors[k] for k in range(1, 6)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=300)
    titles = ["t = 0 (Random)", "t = 50 (Clustering)", "t = 200 (Segregated)"]

    for ax_idx, (t_snap, title) in enumerate(zip(snapshot_times, titles)):
        ax = axes[ax_idx]
        W_snap = W_history[t_snap]

        # Compute layout
        p = spring_layout(W_snap, pos, n_iter=80 + t_snap)

        # Draw edges
        for (i, j) in edges:
            w = W_snap[i, j]
            if w > 0.1:
                alpha = min(w, 0.8)
                lw = 0.3 + w * 2
                ax.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]],
                        "-", color="gray", alpha=alpha * 0.5, linewidth=lw)

        # Draw nodes
        for i in range(N):
            color = K_colors[K[i]]
            ax.scatter(p[i, 0], p[i, 1], c=color, s=120, zorder=5,
                       edgecolors="black", linewidth=0.8)

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.axis("off")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=K_colors[k], markersize=10,
                              label=f"K = {k}", markeredgecolor="black")
                       for k in range(1, 6)]
    axes[2].legend(handles=legend_elements, loc="lower right", fontsize=8,
                   title="Mode count K", title_fontsize=9)

    fig.suptitle(
        r"K-Space Phase Separation Under C2: $\Delta Z = -\eta\,\Gamma\,"
        r"x_{\rm in}\,x_{\rm out}$"
        "\nSame-K nodes cluster; different-K nodes decouple",
        fontsize=11, fontweight="bold", y=1.02
    )

    fig.tight_layout()
    out = FIGURES_DIR / "fig_p1_k_clustering.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ============================================================
#  Figure 2: Television inequality parameter space
# ============================================================
def generate_television_space_figure():
    """
    Consciousness exists iff: G_arousal * sum(G_i) * Q_PFC > theta

    2D heatmap: G_arousal vs sum(G_i), with Q_PFC fixed,
    marking clinical states.
    """
    G_arousal = np.linspace(0, 1, 200)
    sum_Gi = np.linspace(0, 5, 200)

    GA, SG = np.meshgrid(G_arousal, sum_Gi)
    Q_PFC = 0.7  # fixed prefrontal capacity

    # Television product
    TV = GA * SG * Q_PFC
    theta = 0.5  # consciousness threshold

    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=300)

    # Heatmap
    im = ax.pcolormesh(G_arousal, sum_Gi, TV, cmap="RdYlGn",
                       shading="auto", vmin=0, vmax=2.0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(r"$G_{\rm arousal} \cdot \sum \Gamma_i \cdot Q_{\rm PFC}$",
                   fontsize=10)

    # Consciousness boundary
    boundary_sg = theta / (G_arousal * Q_PFC + 1e-10)
    boundary_sg = np.clip(boundary_sg, 0, 5)
    ax.plot(G_arousal, boundary_sg, "k-", linewidth=2.5,
            label=r"$\theta = 0.5$ (consciousness boundary)")

    # Fill unconscious region
    ax.fill_between(G_arousal, 0, np.minimum(boundary_sg, 5),
                    alpha=0.15, color="black")

    # Mark clinical states
    states = [
        (0.85, 3.5, "Awake\n(alert)", "#2E7D32"),
        (0.60, 2.0, "Drowsy", "#F9A825"),
        (0.20, 4.0, "Deep sleep\n(low arousal)", "#1565C0"),
        (0.10, 1.0, "General\nanesthesia", "#B71C1C"),
        (0.80, 0.5, "Occipital\ntrauma", "#4A148C"),
        (0.40, 3.5, "Light\nsedation", "#FF6F00"),
    ]

    for ga, sg, label, color in states:
        tv = ga * sg * Q_PFC
        marker = "o" if tv > theta else "x"
        size = 120 if tv > theta else 100
        ax.scatter(ga, sg, c=color, s=size, marker=marker,
                   zorder=5, edgecolors="white", linewidth=1.5)
        ax.annotate(label, (ga, sg), textcoords="offset points",
                    xytext=(12, 8), fontsize=7.5, color=color,
                    fontweight="bold")

    ax.set_xlabel(r"$G_{\rm arousal}$ (reticular activation)", fontsize=11)
    ax.set_ylabel(r"$\sum \Gamma_i$ (total sensory input)", fontsize=11)
    ax.set_title(
        r"Television Inequality: Consciousness $\Leftrightarrow$ "
        r"$G_{\rm arousal} \cdot \sum \Gamma_i \cdot Q_{\rm PFC} > \theta$"
        f"\n(fixed $Q_{{\\rm PFC}} = {Q_PFC}$)",
        fontsize=10
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.text(0.15, 0.8, "UNCONSCIOUS", ha="center", fontsize=12,
            color="white", fontweight="bold", alpha=0.7,
            transform=ax.transAxes)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_p1_television_space.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Paper 1 Figure Generation")
    print("=" * 60)
    generate_k_clustering_figure()
    generate_television_space_figure()
    print("Done.")
