import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path


# ── Color mapping by rating band ───────────────────────────────────────────────
def get_color(rating):
    if rating < 0.15:  return "#888780"   # Not rated  (bin centre ≈ 0.05)
    if rating < 2.5:   return "#c0392b"   # Poor
    if rating < 3.5:   return "#d97706"   # Average
    if rating < 4.0:   return "#5a8a1a"   # Good
    if rating < 4.5:   return "#0e7c55"   # Very Good
    return                     "#064e38"  # Excellent

def histo(df):
    # ── Build histogram bins ────────────────────────────────────────────────────────
    # Custom edges: isolate 0 as its own bin, then 0.2-wide bins from 1.8 → 5.0
    bin_edges = [0, 0.1] + list(np.arange(1.8, 5.01, 0.2))
    counts, edges = np.histogram(df["Aggregate rating"], bins=bin_edges)
    
    bin_centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))]
    bin_widths  = [edges[i+1] - edges[i] for i in range(len(counts))]
    bar_colors  = [get_color(c) for c in bin_centers]
    
    # X-tick labels: "0" for first bin, then rounded bin starts from 1.8 onward
    tick_labels = ["0"] + [f"{edges[i+1]:.1f}" for i in range(1, len(counts))]
    
    # ── Plot ────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#f5f3ef")
    ax.set_facecolor("#f5f3ef")
    
    bars = ax.bar(
        range(len(counts)),
        counts,
        width=0.75,
        color=bar_colors,
        edgecolor="none",
        zorder=3,
    )
    
    # ── Axes styling ────────────────────────────────────────────────────────────────
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(tick_labels, fontsize=10, color="#6b6860")
    ax.set_xlim(-0.6, len(counts) - 0.4)
    
    ax.set_ylim(0, 2500)
    ax.set_yticks([0, 500, 1000, 1500, 2000, 2500])
    ax.set_yticklabels(["0", "500", "1,000", "1,500", "2,000", "2,500"], fontsize=10, color="#6b6860")
    
    ax.yaxis.grid(True, color=(0, 0, 0, 0.08), linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)
    
    # ── Title ───────────────────────────────────────────────────────────────────────
    ax.set_title(
        f"Aggregate rating histogram (all {len(df):,} entries)",
        fontsize=13, fontweight="600", color="#1a1917",
        loc="left", pad=14
    )
    
    # ── Legend ──────────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color="#888780", label="Not rated (0)"),
        mpatches.Patch(color="#c0392b", label="Poor"),
        mpatches.Patch(color="#d97706", label="Average"),
        mpatches.Patch(color="#5a8a1a", label="Good"),
        mpatches.Patch(color="#0e7c55", label="Very Good"),
        mpatches.Patch(color="#064e38", label="Excellent"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper right",
        frameon=False,
        fontsize=10,
        ncol=6,
        bbox_to_anchor=(1, -0.08),
        handlelength=1.2,
        handleheight=0.9,
        labelcolor="#6b6860",
    )
    
    plt.tight_layout()
    base_path = Path.cwd().parents[1]  # go up two levels
    file_path = base_path / "Notebooks" / "reports" / "rating_histogram.png"
    plt.savefig(file_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
