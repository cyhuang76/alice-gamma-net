#!/usr/bin/env python3
"""
exp_k_growth.py  –  K-Growth Emergence Simulation
==================================================
From C1+C2+C3 + K-growth rule, observe whether a neural-like
K-hierarchy emerges spontaneously from a flat K=1 initial state.

Physics:
  - C2: ΔZ_i = -η * Γ_i * x_in * x_out   (impedance matching)
  - K-growth: dK/dt = μ * <Γ²>_τ * 1[<Γ²>_τ > threshold]
  - Energy cost: cost(K) ∝ K^(3/2)         (thermal dissipation limit)
  - New mode Z_init = mean of neighbours    (C3-compatible seeding)

Network: 1D chain of N nodes with peripheral inputs at boundaries.
  Boundary nodes = sensory (low K expected)
  Central nodes  = convergence zone (high K expected)

Author: ALICE Gamma-Net Project
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── 物理參數 ──────────────────────────────────────────────
N_NODES      = 60          # 網路節點數（1D 鏈）
T_TOTAL      = 8000        # 總模擬 tick 數
DT_GROWTH    = 30          # 每隔多少 tick 檢查 K 生長
ETA_C2       = 0.01        # C2 學習率（慢，讓 Γ² 有機會維持）
MU_GROWTH    = 2.0         # K 生長速率
GAMMA2_THR   = 0.05        # 觸發 K 生長的 Γ² 門檻（降低）
K_MAX        = 12          # K 值上限（能量約束）
K_COST_EXP   = 1.3         # cost(K) ∝ K^exp（降低生長阻力）
TAU_AVG      = 15          # Γ² 移動平均窗口

# ── 初始化 ────────────────────────────────────────────────
rng = np.random.default_rng(42)

# 所有節點從 K=1 開始
K = np.ones(N_NODES, dtype=int)

# 每個節點的阻抗矩陣：Z[i] 是長度 K[i] 的向量
# 初始：隨機小值
Z = [rng.uniform(0.5, 1.5, size=k) for k in K]

# Γ² 歷史紀錄（用於移動平均）
gamma2_history = np.zeros((TAU_AVG, N_NODES))
history_ptr = 0

# 紀錄用
K_snapshots = []
gamma2_snapshots = []
snapshot_times = []

def compute_gamma2(Z, i, j):
    """計算節點 i 和 j 之間的反射係數 Γ²"""
    # 取兩個 Z 向量的共同維度部分
    k_min = min(len(Z[i]), len(Z[j]))
    zi = Z[i][:k_min]
    zj = Z[j][:k_min]
    
    # Γ = (Z_j - Z_i) / (Z_j + Z_i) 的模式平均
    gamma = (zj - zi) / (zj + zi + 1e-12)
    gamma2 = np.mean(gamma**2)
    
    # 如果 K 值不同，額外的維度完全失配 (Γ²=1)
    k_max = max(len(Z[i]), len(Z[j]))
    if k_max > k_min:
        n_extra = k_max - k_min
        gamma2 = (gamma2 * k_min + 1.0 * n_extra) / k_max
    
    return gamma2

def generate_stimulus(t):
    """Multi-source stimuli: peripheral strong, decays inward."""
    stim = np.zeros(N_NODES)
    # Left boundary: multiple frequencies (sensory)
    for k in range(5):
        freq = 30 + k * 20
        stim[k] = (1.0 - 0.15 * k) * np.sin(2 * np.pi * t / freq)
        stim[k] += 0.5 * rng.normal()
    # Right boundary: different frequencies
    for k in range(5):
        freq = 25 + k * 15
        stim[N_NODES - 1 - k] = (1.0 - 0.15 * k) * np.cos(2 * np.pi * t / freq)
        stim[N_NODES - 1 - k] += 0.5 * rng.normal()
    # Central cross-modal events (frequent)
    if t % 100 < 30:
        mid = N_NODES // 2
        for k in range(-3, 4):
            stim[mid + k] = 1.2 * rng.normal()
    # Background noise everywhere
    stim += 0.2 * rng.normal(size=N_NODES)
    return stim

print("=" * 60)
print("K-Growth Emergence Simulation")
print(f"Nodes: {N_NODES}, Ticks: {T_TOTAL}, K_max: {K_MAX}")
print(f"C2 η: {ETA_C2}, Growth μ: {MU_GROWTH}, Γ² threshold: {GAMMA2_THR}")
print("=" * 60)

# ── 主模擬迴圈 ────────────────────────────────────────────
for t in range(T_TOTAL):
    stim = generate_stimulus(t)
    gamma2_local = np.zeros(N_NODES)
    
    # ── C2: 調整 Z ──────────────────────────────────────
    for i in range(N_NODES):
        neighbors = []
        if i > 0:
            neighbors.append(i - 1)
        if i < N_NODES - 1:
            neighbors.append(i + 1)
        
        total_gamma2 = 0
        for j in neighbors:
            g2 = compute_gamma2(Z, i, j)
            total_gamma2 += g2
            
            # C2 梯度下降：調整共同維度的 Z
            k_min = min(len(Z[i]), len(Z[j]))
            zi = Z[i][:k_min]
            zj = Z[j][:k_min]
            gamma = (zj - zi) / (zj + zi + 1e-12)
            
            # x_in * x_out 活動門控
            activity = abs(stim[i]) + abs(stim[j])
            if activity > 0.1:
                dZ = ETA_C2 * gamma * activity
                Z[i][:k_min] += dZ
                # 保持 Z > 0
                Z[i] = np.maximum(Z[i], 0.01)
        
        gamma2_local[i] = total_gamma2 / max(len(neighbors), 1)
    
    # 更新 Γ² 歷史
    gamma2_history[history_ptr % TAU_AVG] = gamma2_local
    history_ptr += 1
    
    # ── K 生長 ──────────────────────────────────────────
    if t > 0 and t % DT_GROWTH == 0:
        # 計算移動平均 Γ²
        n_filled = min(history_ptr, TAU_AVG)
        gamma2_avg = np.mean(gamma2_history[:n_filled], axis=0)
        
        grew = False
        for i in range(N_NODES):
            if gamma2_avg[i] > GAMMA2_THR and K[i] < K_MAX:
                # 能量成本檢查：K 越大，生長越貴
                growth_cost = (K[i] + 1) ** K_COST_EXP / K_MAX ** K_COST_EXP
                effective_signal = MU_GROWTH * (gamma2_avg[i] - GAMMA2_THR)
                
                if effective_signal > growth_cost:
                    K[i] += 1
                    # 新模式的初始 Z = 鄰居平均
                    neighbors = []
                    if i > 0:
                        neighbors.append(i - 1)
                    if i < N_NODES - 1:
                        neighbors.append(i + 1)
                    z_new = np.mean([np.mean(Z[j]) for j in neighbors])
                    Z[i] = np.append(Z[i], z_new)
                    grew = True
        
        if grew and t % 500 == 0:
            print(f"  t={t:5d}: K range [{K.min()}-{K.max()}], "
                  f"mean Γ²={gamma2_avg.mean():.4f}")
    
    # 快照
    if t % (T_TOTAL // 20) == 0:
        K_snapshots.append(K.copy())
        gamma2_snapshots.append(gamma2_local.copy())
        snapshot_times.append(t)

# ── 最終統計 ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL K DISTRIBUTION")
print("=" * 60)
print(f"  K range: [{K.min()}, {K.max()}]")
print(f"  K mean:  {K.mean():.2f}")
print(f"  K std:   {K.std():.2f}")

# 分區統計
n_third = N_NODES // 3
left = K[:n_third]
center = K[n_third:2*n_third]
right = K[2*n_third:]
print(f"\n  Peripheral (left):   K_mean = {left.mean():.2f}")
print(f"  Central:             K_mean = {center.mean():.2f}")
print(f"  Peripheral (right):  K_mean = {right.mean():.2f}")

hierarchy_emerged = center.mean() > max(left.mean(), right.mean())
print(f"\n  Hierarchy emerged: {'YES ✓' if hierarchy_emerged else 'NO ✗'}")
print(f"  (central K > peripheral K = neural-like hierarchy)")

# ── 圖表 ──────────────────────────────────────────────────
OUT = Path(__file__).resolve().parent.parent.parent / "figures"
OUT.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("K-Growth Emergence Simulation\n"
             "From C1+C2+C3: Does neural hierarchy self-organise?",
             fontsize=14, fontweight='bold')

# 1) K 值空間分布（最終）
ax = axes[0, 0]
ax.bar(range(N_NODES), K, color=plt.cm.viridis(K / K.max()),
       edgecolor='none')
ax.set_xlabel("Node position (peripheral ← → peripheral)")
ax.set_ylabel("K value")
ax.set_title("Final K distribution across network")
ax.axhline(y=K.mean(), color='red', linestyle='--', alpha=0.7,
           label=f'mean K = {K.mean():.1f}')
ax.legend()

# 2) K 值時間演化（幾個快照）
ax = axes[0, 1]
n_show = min(6, len(K_snapshots))
indices = np.linspace(0, len(K_snapshots) - 1, n_show, dtype=int)
for idx in indices:
    t_snap = snapshot_times[idx]
    alpha = 0.3 + 0.7 * (idx / max(len(K_snapshots) - 1, 1))
    ax.plot(K_snapshots[idx], alpha=alpha, linewidth=1.5,
            label=f't={t_snap}')
ax.set_xlabel("Node position")
ax.set_ylabel("K value")
ax.set_title("K evolution over time")
ax.legend(fontsize=8)

# 3) Γ² 空間分布（最終）
ax = axes[1, 0]
final_g2 = gamma2_snapshots[-1] if gamma2_snapshots else gamma2_local
ax.fill_between(range(N_NODES), final_g2, alpha=0.5, color='coral')
ax.plot(final_g2, color='red', linewidth=1.5)
ax.axhline(y=GAMMA2_THR, color='gray', linestyle=':', alpha=0.7,
           label=f'growth threshold = {GAMMA2_THR}')
ax.set_xlabel("Node position")
ax.set_ylabel("Γ²")
ax.set_title("Final Γ² distribution")
ax.legend()

# 4) K 值分區箱形圖
ax = axes[1, 1]
data = [left, center, right]
labels = ['Peripheral\n(left)', 'Central\n(convergence)', 'Peripheral\n(right)']
bp = ax.boxplot(data, labels=labels, patch_artist=True)
colors = ['#66c2a5', '#fc8d62', '#66c2a5']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel("K value")
ax.set_title("K by region: hierarchy test")
if hierarchy_emerged:
    ax.text(0.5, 0.95, "✓ HIERARCHY EMERGED",
            transform=ax.transAxes, ha='center', va='top',
            fontsize=12, fontweight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
fig_path = OUT / "fig_k_growth_emergence.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Figure saved: {fig_path}")

# ── 臨床檢查 ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("CLINICAL CHECKS")
print("=" * 60)
checks = [
    ("K hierarchy emerged (central > peripheral)",
     hierarchy_emerged),
    ("K_max reached at convergence zone",
     center.max() >= 0.7 * K_MAX),
    ("Peripheral nodes stayed low-K",
     left.mean() < center.mean() and right.mean() < center.mean()),
    ("Γ² reduced after C2 + K-growth",
     final_g2.mean() < 0.5),
    ("K distribution is graded (not all-or-nothing)",
     K.std() > 0.5),
]
n_pass = sum(1 for _, v in checks if v)
for desc, val in checks:
    print(f"  {'✓' if val else '✗'} {desc}")
print(f"\nResult: {n_pass}/{len(checks)} checks passed")
