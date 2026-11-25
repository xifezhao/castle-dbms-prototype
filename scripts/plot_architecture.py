#!/usr/bin/env python3
"""Generate clean LNCS/TikZ-style vector architecture diagram (inspired by arXiv:2603.07107)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

FIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../figures"))
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Restrained academic Visio/TikZ palette (matching arXiv:2603.07107 Fig. 1 & Fig. 2)
C_BLUE   = "#C9DAF8"   # Soft academic blue (Upper Dir / state_word)
C_BLUE_L = "#E8F0FE"   # Very light ice blue (inflight_writers / buffers)
C_GOLD   = "#FFE599"   # Soft wheat gold (LeafSnap / Freeze)
C_APRIC  = "#F9CB9C"   # Soft apricot (Merge)
C_GREEN  = "#D9EAD3"   # Soft sage green (models / sorted_data)
C_LAV    = "#D9D2E9"   # Soft muted lavender (Rigid Binary Reduction / WAL)
C_GRAY   = "#F3F3F3"   # Light neutral gray (subtle sub-containers)
C_WHITE  = "#FFFFFF"
C_LINE   = "#1F1F1F"   # Crisp dark charcoal/black strokes & text

fig, ax = plt.subplots(figsize=(12.6, 5.25), dpi=300)
ax.set_xlim(0, 18.0)
ax.set_ylim(0, 7.5)
ax.axis("off")


def node(x, y, w, h, title, body="", fc=C_WHITE, ec=C_LINE, lw=0.95, ls="-",
         title_fs=9.2, body_fs=7.9, r=0.06):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2
    )
    ax.add_patch(p)
    if body:
        ax.text(x + w / 2, y + h - 0.14, title, ha="center", va="top",
                fontsize=title_fs, fontweight="bold", color=C_LINE, zorder=3)
        ax.text(x + w / 2, y + (h - 0.34) / 2, body, ha="center", va="center",
                fontsize=body_fs, color="#222222", linespacing=1.26, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=C_LINE, zorder=3)


def arrow(x1, y1, x2, y2, label="", lx=None, ly=None, lw=1.05, rad=0.0):
    conn = f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0"
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>,head_width=0.18,head_length=0.30",
                        color=C_LINE, lw=lw, shrinkA=0, shrinkB=0,
                        connectionstyle=conn),
        zorder=4
    )
    if label:
        tx = (x1 + x2) / 2 if lx is None else lx
        ty = (y1 + y2) / 2 if ly is None else ly
        ax.text(tx, ty, label, ha="center", va="center", fontsize=8.0, style="italic",
                color=C_LINE,
                bbox=dict(boxstyle="square,pad=0.08", fc=C_WHITE, ec="none"),
                zorder=5)


# ==============================================================================
# Clean Outer Solid Frame & Vertical Panel Divider (arXiv:2603.07107 Fig. 2 style)
# ==============================================================================
ax.add_patch(Rectangle((0.10, 0.10), 17.80, 7.30, facecolor=C_WHITE, edgecolor=C_LINE, lw=1.1, zorder=0))
ax.plot([8.78, 8.78], [0.10, 7.40], color=C_LINE, lw=0.95, zorder=1)

# ==============================================================================
# Panel (a): Physical State Bifurcation & Cache-Line Isolation
# ==============================================================================
ax.text(0.32, 7.22, "(a) Physical State Bifurcation & Cache-Line Isolation",
        ha="left", va="top", fontsize=9.8, fontweight="bold", style="italic", color=C_LINE, zorder=3)

# Upper Range Directory
node(0.38, 5.88, 4.15, 0.88,
     "Upper Range Directory",
     "O(1) key routing -> LeafCtrl* anchor",
     fc=C_BLUE, lw=0.95)
arrow(2.455, 5.88, 2.455, 5.46)

# LeafCtrl container (crisp solid black border, white background)
ax.add_patch(Rectangle((0.38, 0.36), 4.15, 5.10, facecolor=C_WHITE, edgecolor=C_LINE, lw=1.05, zorder=1))
ax.text(2.455, 5.32, "LeafCtrl (Mutable, align(64))",
        ha="center", va="top", fontsize=9.1, fontweight="bold", color=C_LINE, zorder=3)

node(0.56, 3.96, 3.79, 0.96,
     "CL 0: 64-bit state_word",
     "[63..2: PTR | 1: REB | 0: BUF]",
     fc=C_BLUE, lw=0.85)

node(0.56, 2.82, 3.79, 0.94,
     "CL 1: inflight_writers[2]",
     "Phase-indexed SeqCst counters",
     fc=C_BLUE_L, lw=0.85)

node(0.56, 0.56, 1.80, 2.04,
     "buffers[0]",
     "Primary Delta\nLazy OnceLock\n256 slots (K, V)\nready + tail",
     fc=C_GRAY, lw=0.85, title_fs=8.8, body_fs=7.6)

node(2.55, 0.56, 1.80, 2.04,
     "buffers[1]",
     "Redirect Delta\nLazy OnceLock\n256 slots (K, V)\nready + tail",
     fc=C_GRAY, lw=0.85, title_fs=8.8, body_fs=7.6)

# LeafSnap container (crisp solid black border, white background)
ax.add_patch(Rectangle((5.32, 0.36), 3.18, 6.40, facecolor=C_WHITE, edgecolor=C_LINE, lw=1.05, zorder=1))
ax.text(6.91, 6.62, "LeafSnap (Immutable)",
        ha="center", va="top", fontsize=9.1, fontweight="bold", color=C_LINE, zorder=3)
ax.text(6.91, 6.26, "!Send Thread-Local EBR",
        ha="center", va="top", fontsize=7.9, style="italic", color="#333333", zorder=3)

node(5.50, 4.82, 2.82, 1.14,
     "boundaries: Box<[K]>",
     "Monotonic segment keys",
     fc=C_GOLD, lw=0.85, title_fs=8.8, body_fs=7.6)

node(5.50, 2.80, 2.82, 1.80,
     "models: Box<[PLM]>",
     "64-bit slope a, intercept b\n-C target-feature=-fma\np_hat = round(a*k + b)",
     fc=C_GREEN, lw=0.85, title_fs=8.8, body_fs=7.6)

node(5.50, 0.56, 2.82, 2.02,
     "sorted_data: Box<[(K,V)]>",
     "Contiguous sorted base run\nBounded search [p +/- d_max]\nSequential range scan",
     fc=C_GREEN, lw=0.85, title_fs=8.8, body_fs=7.6)

arrow(4.35, 4.44, 5.32, 4.44, "Acquire", lx=4.83, ly=4.74, lw=1.1)

# ==============================================================================
# Panel (b): Pure CoW Consolidation & Deterministic WAL Recovery
# ==============================================================================
ax.text(9.00, 7.22, "(b) Pure CoW Consolidation & Deterministic WAL Recovery",
        ha="left", va="top", fontsize=9.8, fontweight="bold", style="italic", color=C_LINE, zorder=3)

# Subtle dashed grouping line separating CoW pipeline from WAL Recovery (matching Fig. 1 divider)
ax.plot([9.00, 17.68], [2.98, 2.98], color="#666666", lw=0.8, ls="--", zorder=1)

node(9.10, 5.26, 3.92, 1.50,
     "1. Freeze Primary Buffer",
     "SeqCst CAS TAG_REBUILD;\nDrain inflight_writers[b];\nRedirect writes -> buffers[1-b]",
     fc=C_GOLD, lw=0.9, title_fs=9.0, body_fs=7.8)

node(13.66, 5.26, 3.92, 1.50,
     "2. Two-Way Merge Sort",
     "Merge frozen buffers[b]\nwith old sorted_data;\nPoll CancelToken / 1,024 recs",
     fc=C_APRIC, lw=0.9, title_fs=9.0, body_fs=7.8)

arrow(13.02, 6.01, 13.66, 6.01)

node(9.10, 3.26, 3.92, 1.50,
     "3. Rigid Binary Reduction",
     "Fixed B=256 LSN chunks;\nLeft-to-right rayon::join;\nFMA disabled (-fma)",
     fc=C_LAV, lw=0.9, title_fs=9.0, body_fs=7.8)

node(13.66, 3.26, 3.92, 1.50,
     "4. Single-Word CAS Swap",
     "CAS state_word -> (ptr1|1-b|0);\nFlip buffer & install PLM;\nRetire old snap via EBR",
     fc=C_BLUE, lw=0.9, title_fs=9.0, body_fs=7.8)

# Clean stepped orthogonal connector from Step 2 bottom to Step 3 top
ax.plot([15.62, 15.62, 11.06], [5.26, 5.01, 5.01], color=C_LINE, lw=1.0, zorder=4)
arrow(11.06, 5.01, 11.06, 4.76, lw=1.0)
arrow(13.02, 4.01, 13.66, 4.01)

node(9.10, 0.36, 8.48, 2.32,
     "LSN-Driven Logical WAL & Two-Phase Redo-Only Recovery",
     "24 B Log: <lsn, key, payload> | MSB 1-bit tag: REBUILD_COMMIT (payload = freeze_lsn)\n"
     "Phase 1 (I/O Redo): Sequential checksummed WAL scan + leaf dispatch (Zero FP)\n"
     "Phase 2 (Reconstruct): Parallel rigid binary-tree PLM fit; >C overflow folded safely\n"
     "Invariant: 100/100 bit-exact SHA-256 reproduction across 1, 2, 4, 8, and 16 threads",
     fc=C_LAV, lw=0.95, title_fs=9.2, body_fs=7.6)

arrow(15.62, 3.26, 15.62, 2.68, "24 B", lx=16.05, ly=2.97, lw=1.05)

fig.tight_layout(pad=0.08)
fig.savefig(f"{FIG_DIR}/fig_architecture.pdf", bbox_inches="tight")
plt.close(fig)
print("[OK] Generated arXiv:2603.07107-style figures/fig_architecture.pdf")
