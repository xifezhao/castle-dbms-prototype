#!/usr/bin/env python3
"""Generate clean, colorful, non-overlapping vector architecture diagram (LNCS-compliant)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/figures"))
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(12.8, 5.6), dpi=300)
ax.set_xlim(0, 18.0)
ax.set_ylim(0, 7.8)
ax.axis("off")


def box(x, y, w, h, title, body="", fc="#FFFFFF", ec="#2B4C7E", lw=1.2, ls="-",
        title_fs=9.5, body_fs=8.0, title_color="#0F2537"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.14",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2
    )
    ax.add_patch(p)
    if body:
        ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top",
                fontsize=title_fs, fontweight="bold", color=title_color, zorder=3)
        ax.text(x + w / 2, y + (h - 0.34) / 2, body, ha="center", va="center",
                fontsize=body_fs, color="#1E293B", linespacing=1.28, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=title_color, zorder=3)


def arrow(x1, y1, x2, y2, label="", lx=None, ly=None, lw=1.4, color="#334155"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.35",
                        color=color, lw=lw, shrinkA=0, shrinkB=0),
        zorder=4
    )
    if label:
        tx = (x1 + x2) / 2 if lx is None else lx
        ty = (y1 + y2) / 2 if ly is None else ly
        ax.text(tx, ty, label, ha="center", va="center", fontsize=8.0, fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.14", fc="#FFFFFF", ec=color, lw=0.7),
                zorder=5)


# ==============================================================================
# Panel (a): Physical State Bifurcation & Cache-Line Isolation
# ==============================================================================
ax.add_patch(FancyBboxPatch(
    (0.15, 0.15), 8.45, 7.50,
    boxstyle="round,pad=0.04,rounding_size=0.18",
    fc="#F8FAFC", ec="#64748B", lw=1.1, ls="--", zorder=1
))
ax.text(4.37, 7.45, "(a) Physical State Bifurcation & Cache-Line Isolation",
        ha="center", va="top", fontsize=10.5, fontweight="bold", color="#0F172A", zorder=3)

# Upper Range Directory
box(0.45, 6.00, 4.10, 0.98,
    "Upper Range Directory",
    "O(1) Key -> LeafCtrl* Anchor",
    fc="#E0F2FE", ec="#0284C7", lw=1.2, title_color="#0369A1")
arrow(2.50, 6.00, 2.50, 5.58, color="#0284C7")

# LeafCtrl container
ax.add_patch(FancyBboxPatch(
    (0.45, 0.40), 4.10, 5.15,
    boxstyle="round,pad=0.04,rounding_size=0.14",
    fc="#FFFFFF", ec="#1D4ED8", lw=1.5, zorder=2
))
ax.text(2.50, 5.38, "LeafCtrl (Mutable, align(64))",
        ha="center", va="top", fontsize=9.6, fontweight="bold", color="#1E3A8A", zorder=3)

box(0.65, 4.02, 3.70, 0.96,
    "CL 0: 64-bit state_word",
    "[63..2: PTR | 1: REB | 0: BUF]",
    fc="#DBEAFE", ec="#2563EB", lw=1.0, title_color="#1E40AF")

box(0.65, 2.84, 3.70, 0.96,
    "CL 1: inflight_writers[2]",
    "Phase-indexed SeqCst counters",
    fc="#EFF6FF", ec="#3B82F6", lw=1.0, title_color="#1D4ED8")

box(0.65, 0.58, 1.75, 2.02,
    "buffers[0]",
    "Primary Delta\nLazy OnceLock\n256 slots (K, V)\nready + tail",
    fc="#F0F9FF", ec="#0EA5E9", lw=1.0, title_fs=9.0, body_fs=7.8, title_color="#0369A1")

box(2.60, 0.58, 1.75, 2.02,
    "buffers[1]",
    "Redirect Delta\nLazy OnceLock\n256 slots (K, V)\nready + tail",
    fc="#E0F2FE", ec="#0284C7", lw=1.0, title_fs=9.0, body_fs=7.8, title_color="#0369A1")

# LeafSnap container
ax.add_patch(FancyBboxPatch(
    (5.25, 0.40), 3.10, 6.58,
    boxstyle="round,pad=0.04,rounding_size=0.14",
    fc="#FFFFFF", ec="#0D9488", lw=1.5, zorder=2
))
ax.text(6.80, 6.82, "LeafSnap (Immutable)",
        ha="center", va="top", fontsize=9.6, fontweight="bold", color="#115E59", zorder=3)
ax.text(6.80, 6.42, "!Send Thread-Local EBR",
        ha="center", va="top", fontsize=8.2, style="italic", color="#0F766E", zorder=3)

box(5.45, 4.92, 2.70, 1.18,
    "boundaries: Box<[K]>",
    "Monotonic segment keys",
    fc="#CCFBF1", ec="#14B8A6", lw=1.0, title_fs=9.0, body_fs=7.8, title_color="#115E59")

box(5.45, 2.86, 2.70, 1.82,
    "models: Box<[PLM]>",
    "64-bit slope a, intercept b\n-C target-feature=-fma\np_hat = round(a*k + b)",
    fc="#D1FAE5", ec="#10B981", lw=1.0, title_fs=9.0, body_fs=7.8, title_color="#065F46")

box(5.45, 0.58, 2.70, 2.04,
    "sorted_data: Box<[(K,V)]>",
    "Contiguous sorted base run\nBounded search [p +/- d_max]\nSequential range scan",
    fc="#ECFDF5", ec="#059669", lw=1.0, title_fs=9.0, body_fs=7.8, title_color="#065F46")

arrow(4.35, 4.50, 5.25, 4.50, "Acquire", lx=4.80, ly=4.90, lw=1.5, color="#0F766E")

# ==============================================================================
# Panel (b): Pure CoW Consolidation & Deterministic WAL Recovery
# ==============================================================================
ax.add_patch(FancyBboxPatch(
    (8.90, 0.15), 8.95, 7.50,
    boxstyle="round,pad=0.04,rounding_size=0.18",
    fc="#F8FAFC", ec="#64748B", lw=1.1, ls="--", zorder=1
))
ax.text(13.37, 7.45, "(b) Pure CoW Consolidation & Deterministic WAL Recovery",
        ha="center", va="top", fontsize=10.5, fontweight="bold", color="#0F172A", zorder=3)

box(9.18, 5.42, 3.85, 1.56,
    "1. Freeze Primary Buffer",
    "SeqCst CAS TAG_REBUILD;\nDrain inflight_writers[b];\nRedirect writes -> buffers[1-b]",
    fc="#FEF3C7", ec="#D97706", lw=1.1, title_fs=9.3, body_fs=8.0, title_color="#92400E")

box(13.70, 5.42, 3.85, 1.56,
    "2. Two-Way Merge Sort",
    "Merge frozen buffers[b]\nwith old sorted_data;\nPoll CancelToken / 1,024 recs",
    fc="#FFEDD5", ec="#EA580C", lw=1.1, title_fs=9.3, body_fs=8.0, title_color="#9A3412")

arrow(13.03, 6.20, 13.70, 6.20, color="#B45309")

box(9.18, 3.34, 3.85, 1.56,
    "3. Rigid Binary Reduction",
    "Fixed B=256 LSN chunks;\nLeft-to-right rayon::join;\nFMA disabled (-fma)",
    fc="#EDE9FE", ec="#7C3AED", lw=1.1, title_fs=9.3, body_fs=8.0, title_color="#5B21B6")

box(13.70, 3.34, 3.85, 1.56,
    "4. Single-Word CAS Swap",
    "CAS state_word -> (ptr1|1-b|0);\nFlip buffer & install PLM;\nRetire old snap via EBR",
    fc="#FCE7F3", ec="#DB2777", lw=1.1, title_fs=9.3, body_fs=8.0, title_color="#9D174D")

# Diagonal / stepped arrow from step 2 bottom to step 3 top
ax.annotate(
    "", xy=(11.10, 4.90), xytext=(15.62, 5.42),
    arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.35",
                    color="#6D28D9", lw=1.3, connectionstyle="arc3,rad=-0.12"),
    zorder=4
)
arrow(13.03, 4.12, 13.70, 4.12, color="#9D174D")

box(9.18, 0.42, 8.37, 2.38,
    "LSN-Driven Logical WAL & Two-Phase Redo-Only Recovery",
    "Logical Log (24 B): <lsn, key, payload> | MSB 1-bit LSN tag marks REBUILD_COMMIT\n"
    "Phase 1 (I/O Redo): Sequential checksummed WAL scan + leaf dispatch (Zero FP)\n"
    "Phase 2 (Reconstruct): Parallel rigid binary-tree PLM fit; >C overflow folded safely\n"
    "Invariant: 100/100 bit-exact SHA-256 reproduction across 1, 2, 4, 8, and 16 threads",
    fc="#EEF2FF", ec="#4F46E5", lw=1.3, title_fs=9.6, body_fs=8.0, title_color="#3730A3")

arrow(15.62, 3.34, 15.62, 2.80, "24 B", lx=16.25, ly=3.07, lw=1.4, color="#4F46E5")

fig.tight_layout(pad=0.15)
fig.savefig(f"{FIG_DIR}/fig_architecture.pdf", bbox_inches="tight")
plt.close(fig)
print("[OK] Generated colorful non-overlapping figures/fig_architecture.pdf")
