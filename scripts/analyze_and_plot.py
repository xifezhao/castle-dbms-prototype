#!/usr/bin/env python3
"""Aggregate Round-2 multi-seed CASTLE Colab results and generate LNCS-compliant PDF figures."""

import csv
import os
import statistics

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../results/raw_ecs32"))
FIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../figures"))
os.makedirs(FIG_DIR, exist_ok=True)

INDEXES = ["CASTLE", "BTreeOLC", "DeltaLearned", "ALEX_PART", "ALEX", "LIPP"]
LABELS = {
    "CASTLE": "CASTLE",
    "CASTLE_ARC": "CASTLE (Arc)",
    "BTreeOLC": "B+-Tree OLC",
    "DeltaLearned": "Delta-Learned",
    "ALEX_PART": "ALEX-Part",
    "ALEX": "ALEX (Global)",
    "LIPP": "LIPP (Global)",
}
SHORT_LABELS = {
    "CASTLE": "CASTLE",
    "BTreeOLC": "B+-Tree\nOLC",
    "DeltaLearned": "Delta-\nLearned",
    "ALEX_PART": "ALEX-\nPart",
    "ALEX": "ALEX\n(Global)",
    "LIPP": "LIPP\n(Global)",
}
THREADS = [1, 2, 4, 8, 16, 32]
SEEDS = [42, 1337, 2027, 7777, 9999]


def read_point(path):
    with open(path, "r") as f:
        row = next(csv.DictReader(f))
    return {
        "mops": float(row["throughput_ops"]) / 1e6,
        "p50_us": float(row["p50_ns"]) / 1e3,
        "p99_us": float(row["p99_ns"]) / 1e3,
        "p999_us": float(row["p999_ns"]) / 1e3,
        "build_ms": float(row["build_ms"]),
        "mem_mb": float(row["memory_bytes"]) / (1024.0 * 1024.0),
    }


def mean_std(vals):
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def summarize_grid(workload, indexes):
    print(f"\n=== {workload.upper()} (500K keys, mean ± std across 3 seeds) ===")
    data = {}
    for idx in indexes:
        data[idx] = {}
        row_str = f"{idx:13s}: "
        for t in THREADS:
            pts = [read_point(f"{RAW_DIR}/{workload}_{idx}_t{t}_s{s}.csv") for s in SEEDS]
            m_ops, s_ops = mean_std([p["mops"] for p in pts])
            m_p99, _ = mean_std([p["p99_us"] for p in pts])
            m_build, _ = mean_std([p["build_ms"] for p in pts])
            m_mem, _ = mean_std([p["mem_mb"] for p in pts])
            data[idx][t] = {
                "mops": m_ops, "mops_std": s_ops, "p99": m_p99,
                "build_ms": m_build, "mem_mb": m_mem,
            }
            row_str += f"t={t}: {m_ops:.2f}±{s_ops:.2f} (P99={m_p99:.1f}us) | "
        print(row_str)
    return data


ycsb_a = summarize_grid("ycsb_a", INDEXES)
ycsb_c = summarize_grid("ycsb_c", INDEXES + ["CASTLE_ARC"])

print("\n=== YCSB-E & TPC-C STRESS (500K keys, 16 workers, 5 seeds) ===")
wl_bar = {}
for wl in ["ycsb_e", "tpcc_stress"]:
    wl_bar[wl] = {}
    print(f"-- {wl} --")
    for idx in INDEXES:
        pts = [read_point(f"{RAW_DIR}/{wl}_{idx}_t16_s{s}.csv") for s in SEEDS]
        m_ops, s_ops = mean_std([p["mops"] for p in pts])
        m_p99, _ = mean_std([p["p99_us"] for p in pts])
        m_p999, _ = mean_std([p["p999_us"] for p in pts])
        wl_bar[wl][idx] = {"mops": m_ops, "std": s_ops, "p99": m_p99, "p999": m_p999}
        print(f"  {idx:13s}: {m_ops:.3f}±{s_ops:.3f} Mops/s | P99={m_p99:.2f}us | P99.9={m_p999:.2f}us")

print("\n=== OUT-OF-L3 SCALE-UP (10M keys, 16 workers, 5 seeds) ===")
for wl in ["ycsb_a", "tpcc_stress"]:
    print(f"-- scale10m {wl} --")
    for idx in ["CASTLE", "BTreeOLC", "DeltaLearned", "ALEX_PART"]:
        pts = [read_point(f"{RAW_DIR}/scale10m_{wl}_{idx}_t16_s{s}.csv") for s in SEEDS]
        m_ops, s_ops = mean_std([p["mops"] for p in pts])
        m_p50, _ = mean_std([p["p50_us"] for p in pts])
        m_p99, _ = mean_std([p["p99_us"] for p in pts])
        m_p999, _ = mean_std([p["p999_us"] for p in pts])
        m_build, _ = mean_std([p["build_ms"] for p in pts])
        m_mem, _ = mean_std([p["mem_mb"] for p in pts])
        print(f"  {idx:13s}: {m_ops:.3f}±{s_ops:.3f} Mops/s | P50={m_p50:.2f}us P99={m_p99:.2f}us P99.9={m_p999:.1f}us | build={m_build:.1f}ms | mem={m_mem:.2f}MB")

print("\n=== SKEW BURST (20s time-series, burst in [5, 15)) ===")
skew_data = {}
for idx in INDEXES:
    with open(f"{RAW_DIR}/skew_burst_{idx}.csv") as f:
        rows = list(csv.DictReader(f))
    ts = [int(r["timestamp_sec"]) for r in rows]
    p99 = [float(r["read_p99_us"]) for r in rows]
    skew_data[idx] = (ts, p99)
    print(f"  {idx:13s}: pre={statistics.mean(p99[:5]):.2f}us | burst_med={statistics.median(p99[5:15]):.2f}us | burst_peak={max(p99[5:15]):.2f}us | post={statistics.mean(p99[16:]):.2f}us")

print("\n=== RECOVERY ECS32 (4-Row Complete) ===")
with open(f"{RAW_DIR}/recovery_ecs32.csv") as f:
    for r in csv.DictReader(f):
        print(" ", r)

print("\n=== MEMORY & BUILD (500K keys, t=1) ===")
for idx in INDEXES:
    d = ycsb_a[idx][1]
    print(f"  {idx:13s}: build={d['build_ms']:.1f}ms | mem={d['mem_mb']:.2f}MB")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7.2,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

STYLES = {
    "CASTLE":       {"color": "#111111", "ls": "-",  "marker": "o", "lw": 1.8, "ms": 4.5, "hatch": ""},
    "BTreeOLC":     {"color": "#383838", "ls": "--", "marker": "s", "lw": 1.3, "ms": 4.0, "hatch": "//"},
    "DeltaLearned": {"color": "#5E5E5E", "ls": "-.", "marker": "^", "lw": 1.3, "ms": 4.2, "hatch": "\\\\"},
    "ALEX_PART":    {"color": "#808080", "ls": (0, (5, 1)), "marker": "P", "lw": 1.3, "ms": 4.2, "hatch": "++"},
    "ALEX":         {"color": "#9E9E9E", "ls": ":",  "marker": "D", "lw": 1.3, "ms": 3.7, "hatch": "xx"},
    "LIPP":         {"color": "#BCBCBC", "ls": (0, (3, 1, 1, 1)), "marker": "v", "lw": 1.3, "ms": 3.8, "hatch": ".."},
}

# Figure 1: YCSB-A & YCSB-C (evenly spaced categorical thread axis to prevent 1/2 tick crowding)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.65), dpi=300)
x_t = list(range(len(THREADS)))
for ax, dataset, title in [
    (ax1, ycsb_a, "(a) YCSB-A (50% Read / 50% Write, Zipf 0.99)"),
    (ax2, ycsb_c, "(b) YCSB-C (100% Read, Zipf 0.99)"),
]:
    for idx in INDEXES:
        st = STYLES[idx]
        ys = [dataset[idx][t]["mops"] for t in THREADS]
        errs = [dataset[idx][t]["mops_std"] for t in THREADS]
        ax.errorbar(
            x_t, ys, yerr=errs, label=LABELS[idx],
            color=st["color"], linestyle=st["ls"], marker=st["marker"],
            linewidth=st["lw"], markersize=st["ms"], capsize=2.0, elinewidth=0.8
        )
    ax.set_title(title, pad=5)
    ax.set_xlabel("Concurrent Worker Tasks")
    ax.set_ylabel("Throughput (Mops/s)")
    ax.set_xticks(x_t)
    ax.set_xticklabels([str(t) for t in THREADS])
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
fig.tight_layout(rect=[0, 0, 1, 0.87])
fig.savefig(f"{FIG_DIR}/fig1_throughput_scalability.pdf", bbox_inches="tight")
plt.close(fig)

# Figure 2: TPC-C & YCSB-E with crisp 2-line non-overlapping x-tick labels & error-bar-aware value labels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.55), dpi=300)
x_pos = range(len(INDEXES))
for ax, wl, title in [
    (ax1, "tpcc_stress", "(a) Index-Stressed TPC-C (16 Workers)"),
    (ax2, "ycsb_e", "(b) YCSB-E Short Range Scan (16 Workers)"),
]:
    vals = [wl_bar[wl][i]["mops"] for i in INDEXES]
    errs = [wl_bar[wl][i]["std"] for i in INDEXES]
    max_top = max(v + e for v, e in zip(vals, errs))
    bars = ax.bar(
        x_pos, vals, yerr=errs, width=0.55, capsize=2.2,
        color=[STYLES[i]["color"] for i in INDEXES],
        edgecolor="black", linewidth=0.7, error_kw={"elinewidth": 0.8}
    )
    for b, idx, err in zip(bars, INDEXES, errs):
        b.set_hatch(STYLES[idx]["hatch"])
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2.0, h + err + max_top * 0.03, f"{h:.2f}",
                ha="center", va="bottom", fontsize=6.8)
    ax.set_title(title, pad=6)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels([SHORT_LABELS[i] for i in INDEXES], fontsize=6.6)
    ax.set_ylabel("Throughput (Mops/s)")
    ax.set_ylim(0, max_top * 1.22)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout(w_pad=1.6)
fig.savefig(f"{FIG_DIR}/fig3_tpcc_rangescan.pdf", bbox_inches="tight")
plt.close(fig)

# Figure 3: Skew Burst P99 Read Latency
fig, ax = plt.subplots(figsize=(6.3, 2.6), dpi=300)
ax.axvspan(5, 15, color="#E6E6E6", alpha=0.7, label="1% Key-Range Skew Burst")
for idx in INDEXES:
    st = STYLES[idx]
    ts, p99 = skew_data[idx]
    ax.plot(
        ts, p99, label=LABELS[idx],
        color=st["color"], linestyle=st["ls"], marker=st["marker"],
        linewidth=st["lw"], markersize=st["ms"] - 0.5, markevery=2
    )
ax.set_yscale("log")
ax.set_xlabel("Elapsed Time (s)")
ax.set_ylabel("P99 Read Latency ($\\mu$s, log)")
ax.set_xlim(0, 19)
ax.grid(True, which="both", linestyle=":", linewidth=0.45, alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", ncol=4, frameon=True, facecolor="white", framealpha=0.9, bbox_to_anchor=(0.5, 1.27))
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig2_skew_burst_p99_latency.pdf", bbox_inches="tight")
plt.close(fig)

print(f"\n[OK] Regenerated all Round-2 grayscale-printable LNCS PDF figures in {FIG_DIR}")
