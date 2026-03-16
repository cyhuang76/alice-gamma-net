#!/usr/bin/env python3
"""
generate_paper0_figures.py
==========================
P0 推導鏈流程圖：從熱力學第一定律到 C1/C2/C3，再延展至各分支論文。

Output: figures/fig_p0_deductive_chain.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def draw_box(ax, cx, cy, w, h, text, color="#2196F3", fontsize=9,
             fontweight="bold", textcolor="white", alpha=1.0, linestyle="-"):
    """繪製圓角方框節點。"""
    box = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="black", linewidth=1.2,
        alpha=alpha, linestyle=linestyle,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=textcolor,
            wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, color="black", lw=1.5, style="->"):
    """繪製帶箭頭連線。"""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(8.5, 10), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_aspect("equal")

    # ── Layer 0: 第一定律 ──
    draw_box(ax, 5, 11.2, 4.5, 0.7,
             "First Law of Thermodynamics\n(Energy Conservation at Interfaces)",
             color="#1565C0", fontsize=10)

    # ── Γ 定義 ──
    draw_arrow(ax, 5, 10.85, 5, 10.35)
    draw_box(ax, 5, 10.0, 4.0, 0.55,
             r"$\Gamma = (Z_2 - Z_1)\,/\,(Z_2 + Z_1)$",
             color="#0D47A1", fontsize=11)

    # ── MRP ──
    draw_arrow(ax, 5, 9.72, 5, 9.22)
    draw_box(ax, 5, 8.9, 4.2, 0.55,
             r"Minimum Reflection Principle: $\mathcal{A}[\Gamma]\to\min$",
             color="#1976D2", fontsize=9)

    # ── 三大約束 ──
    draw_arrow(ax, 3.5, 8.60, 2.0, 8.0)
    draw_arrow(ax, 5.0, 8.60, 5.0, 8.0)
    draw_arrow(ax, 6.5, 8.60, 8.0, 8.0)

    # C1
    draw_box(ax, 2.0, 7.7, 2.5, 0.50,
             r"C1: $\Gamma^2 + T = 1$" + "\nEnergy Conservation",
             color="#E53935", fontsize=8)
    # C2
    draw_box(ax, 5.0, 7.7, 2.8, 0.50,
             r"C2: $\Delta Z = -\eta\,\Gamma\,x_{\rm in}\,x_{\rm out}$"
             + "\nImpedance Remodeling",
             color="#43A047", fontsize=8)
    # C3
    draw_box(ax, 8.0, 7.7, 2.5, 0.50,
             "C3: Impedance-Tagged\nTransport",
             color="#FB8C00", fontsize=8)

    # ── 分隔線 ──
    ax.plot([0.5, 9.5], [7.15, 7.15], "k--", alpha=0.3, linewidth=0.8)
    ax.text(0.6, 7.25, "Derived from First Law (Paper 0)",
            fontsize=7, color="gray", style="italic")
    ax.text(0.6, 6.95, "Consequences (Papers 1–5)",
            fontsize=7, color="gray", style="italic")

    # ── Paper 1: 存在與拓撲 ──
    draw_arrow(ax, 5.0, 7.45, 2.0, 6.55)
    draw_box(ax, 2.0, 6.2, 3.0, 0.65,
             "Paper 1 — Topology & Mind\n"
             r"$A = A_{\rm imp} + A_{\rm cut}$" + "\n"
             "Brain, Memory, Consciousness",
             color="#7B1FA2", fontsize=7.5, textcolor="white")

    # ── Paper 2: 幾何量化 ──
    draw_arrow(ax, 5.0, 7.45, 5.0, 6.55)
    draw_box(ax, 5.0, 6.2, 3.0, 0.65,
             "Paper 2 — Dual Networks\n"
             r"Murray: $r_d = r_p / n^{1/3}$" + "\n"
             r"Kleiber: $B \propto M^{3/4}$",
             color="#00838F", fontsize=7.5, textcolor="white")

    # ── Paper 3: 時間動態 ──
    draw_arrow(ax, 5.0, 7.45, 8.0, 6.55)
    draw_box(ax, 8.0, 6.2, 3.0, 0.65,
             "Paper 3 — Temporal\n"
             r"$D_Z = \int\Gamma^2 P_{\rm in}\,dt$" + "\n"
             "29 C2 Laws, Sleep, Lifecycle",
             color="#C62828", fontsize=7.5, textcolor="white")

    # ── Paper 4: 病理學 ──
    draw_arrow(ax, 2.0, 5.85, 3.5, 5.15)
    draw_arrow(ax, 5.0, 5.85, 5.0, 5.15)
    draw_arrow(ax, 8.0, 5.85, 6.5, 5.15)
    draw_box(ax, 5.0, 4.8, 4.5, 0.65,
             "Paper 4 — Topological Pathology\n"
             r"Disease $=$ Impedance Failure, $\Gamma_k^2 > \theta$" + "\n"
             "No disease-specific code (E0 Emergence)",
             color="#4E342E", fontsize=7.5, textcolor="white")

    # ── Paper 5: 驗證 ──
    draw_arrow(ax, 5.0, 4.45, 5.0, 3.75)
    draw_box(ax, 5.0, 3.4, 5.0, 0.65,
             "Paper 5 — Full Verification\n"
             "NHANES $n = 49{,}774$, AUC $= 0.705$, zero fitted params\n"
             "29 Tissue Blueprints, 275 E0 tests, 19+ predictions",
             color="#263238", fontsize=7.5, textcolor="white")

    # ── 核心原則標注 ──
    draw_box(ax, 5.0, 2.4, 6.0, 0.50,
             "All results derived from C1/C2/C3 + initial conditions alone\n"
             "No layer imports from outer layers → zero circularity",
             color="#ECEFF1", fontsize=8, fontweight="normal",
             textcolor="#37474F", alpha=0.9, linestyle="--")

    fig.suptitle(r"$\Gamma$-Net: The Deductive Chain",
                 fontsize=14, fontweight="bold", y=0.98)

    out = FIGURES_DIR / "fig_p0_deductive_chain.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
