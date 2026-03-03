# -*- coding: utf-8 -*-
"""
Experiment: Bone China D_Z Heat Generation
══════════════════════════════════════════

Paper II Validation: Memory consolidation as the primary source of D_Z.

Physics hypothesis:
  Bone China Phase 3 (Bisque firing during N3 sleep) generates massive
  Γ² waste heat.
  Infants (high volume of novel learning) require more frequent and
  intense N3 firing than adults, resulting in much higher D_Z accumulation
  per day. This explains why infants sleep so much and why they possess
  a fontanelle (thermal exhaust port) that closes later in life.

This experiment simulates a diurnal sleep/wake cycle using the BoneChinaEngine
for two profiles: "Infant" (high learning, high sleep) and "Adult" (maintenance).
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Fix path to import alice modules
sys.path.append(str(Path(__file__).parent.parent))
from alice.brain.bone_china import BoneChinaEngine

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# 模擬參數
TICKS_PER_HOUR = 10
DAYS = 3

def simulate_profile(profile_name: str, wake_hours_per_cycle: float, sleep_hours_per_cycle: float, 
                     n_cycles_per_day: int, learning_rate: float, importance_mean: float):
    engine = BoneChinaEngine()
    
    wake_ticks = int(wake_hours_per_cycle * TICKS_PER_HOUR)
    sleep_ticks = int(sleep_hours_per_cycle * TICKS_PER_HOUR)
    
    # Sleep stage proportion per cycle
    n1_len = int(sleep_ticks * 0.1)
    n2_len = int(sleep_ticks * 0.4)
    n3_len = int(sleep_ticks * 0.3)
    rem_len = sleep_ticks - n1_len - n2_len - n3_len

    sleep_stages = (
        ["n1"] * n1_len + 
        ["n2"] * n2_len + 
        ["n3"] * n3_len + 
        ["rem"] * rem_len
    )
    
    time_h = []
    cumulative_heat = []
    daily_heat = []
    current_heat = 0.0
    item_counter = 0

    for day in range(DAYS):
        day_start_heat = current_heat
        
        for cycle in range(n_cycles_per_day):
            cycle_start_hour = (day * 24) + (cycle * (wake_hours_per_cycle + sleep_hours_per_cycle))
            
            # ─── WAKE PHASE ───
            for t in range(wake_ticks):
                if np.random.rand() < learning_rate:
                    imp = np.clip(np.random.normal(importance_mean, 0.2), 0.1, 1.0)
                    engine.create_clay(f"item_{item_counter}", importance=imp)
                    item_counter += 1
                
                res = engine.tick(is_sleeping=False)
                current_heat += res["heat_generated"]
                
                time_h.append(cycle_start_hour + t / TICKS_PER_HOUR)
                cumulative_heat.append(current_heat)

            # ─── SLEEP PHASE ───
            for t, stage in enumerate(sleep_stages):
                res = engine.tick(is_sleeping=True, sleep_stage=stage)
                current_heat += res["heat_generated"]
                
                time_h.append(cycle_start_hour + (wake_ticks + t) / TICKS_PER_HOUR)
                cumulative_heat.append(current_heat)
                
        daily_heat.append(current_heat - day_start_heat)

    # 最終狀態
    stats = engine.get_stats()
    return {
        "time": np.array(time_h),
        "heat": np.array(cumulative_heat),
        "daily_heat": daily_heat,
        "porcelain": stats["total_porcelain_ever"],
        "shattered": stats["total_shattered"]
    }

def main():
    print("Simulating Infant profile...")
    # 嬰兒: 12小時清醒, 12小時睡眠 -> 分 4 次 (每次 3h wake, 3h sleep)
    res_infant = simulate_profile("Infant", wake_hours_per_cycle=3, sleep_hours_per_cycle=3, 
                                  n_cycles_per_day=4, learning_rate=0.8, importance_mean=0.7)
    
    print("Simulating Adult profile...")
    # 成人: 16小時清醒, 8小時睡眠 -> 1 次 (16h wake, 8h sleep)
    res_adult = simulate_profile("Adult", wake_hours_per_cycle=16, sleep_hours_per_cycle=8, 
                                 n_cycles_per_day=1, learning_rate=0.3, importance_mean=0.4)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Cumulative Heat
    ax = axes[0]
    ax.plot(res_infant["time"], res_infant["heat"], label="Infant (High Learning)", 
            color="#E91E63", linewidth=2.5)
    ax.plot(res_adult["time"], res_adult["heat"], label="Adult (Maintenance)", 
            color="#2196F3", linewidth=2.5)
    
    # 標示睡眠週期 (以 Infant 為主畫陰影)
    for day in range(DAYS):
        ax.axvspan(day*24 + 12, day*24 + 24, color="#E91E63", alpha=0.1, lw=0)
        
    ax.set_title(r"(a) Cumulative Impedance Debt ($D_Z$)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (Hours)", fontsize=11)
    ax.set_ylabel(r"Cumulative Waste Heat ($\Sigma \Gamma^2$)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (b) Daily Daily Heat Bar Chart
    ax = axes[1]
    x = np.arange(DAYS)
    width = 0.35
    ax.bar(x - width/2, res_infant["daily_heat"], width, label="Infant", color="#E91E63", alpha=0.8)
    ax.bar(x + width/2, res_adult["daily_heat"], width, label="Adult", color="#2196F3", alpha=0.8)
    
    ax.set_title(r"(b) Daily $D_Z$ Heat Generation", fontsize=13, fontweight="bold")
    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel(r"Daily Heat ($\Gamma^2$/day)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Day {i+1}" for i in range(DAYS)])
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    fig.suptitle("Ceramic Memory Consolidation as Primary $D_Z$ Source\n(Bone China Engine Simulation)", 
                 fontsize=14, fontweight="bold", y=1.05)
    fig.tight_layout()

    out_png = OUTPUT_DIR / "fig_bone_china_heat.png"
    out_pdf = OUTPUT_DIR / "fig_bone_china_heat.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    
    print("\nResults:")
    print(f"  Infant: {res_infant['porcelain']} items learned, {res_infant['heat'][-1]:.2f} total heat")
    print(f"  Adult:  {res_adult['porcelain']} items learned,  {res_adult['heat'][-1]:.2f} total heat")
    print(f"  Infant generates {res_infant['heat'][-1] / res_adult['heat'][-1]:.1f}x more D_Z heat than adult.")
    print(f"\nSaved plots to:\n  {out_png}\n  {out_pdf}")

if __name__ == "__main__":
    main()
