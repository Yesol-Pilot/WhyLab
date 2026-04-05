#!/usr/bin/env python3
"""Generate SWE-bench phase diagram: mean oscillation vs max_attempts, by temperature and audit."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Load data ──────────────────────────────────────────────────────
data_path = Path(__file__).resolve().parent.parent / "results" / "e9_swebench_phase_checkpoint.json"
with open(data_path) as f:
    episodes = json.load(f)

# ── Aggregate: mean oscillation per (temperature, max_attempts, audit) ──
from collections import defaultdict
agg = defaultdict(list)
for ep in episodes:
    key = (ep["temperature"], ep["max_attempts"], ep["audit"])
    agg[key].append(ep["oscillation_count"])

mean_osc = {k: np.mean(v) for k, v in agg.items()}

temps = [0.3, 0.7, 1.0]
attempts = [3, 5, 7]
audits = ["none", "C2"]

# ── Plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)

style = {
    "none": dict(color="#d62728", marker="o", linestyle="--", linewidth=1.8, label="Baseline (none)"),
    "C2":   dict(color="#1f77b4", marker="s", linestyle="-",  linewidth=1.8, label="C2 audit"),
}

for ax, T in zip(axes, temps):
    for audit in audits:
        ys = [mean_osc.get((T, att, audit), 0.0) for att in attempts]
        ax.plot(attempts, ys, **style[audit], markersize=7)
    ax.set_title(f"$T = {T}$", fontsize=13)
    ax.set_xlabel("max_attempts", fontsize=11)
    ax.set_xticks(attempts)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Mean oscillation count", fontsize=11)
axes[0].legend(fontsize=9, loc="upper left")

# Annotate the att>=5 region where audit helps
for ax in axes:
    ymin, ymax = ax.get_ylim()
    ax.axvspan(4.5, 7.5, alpha=0.06, color="blue")
    ax.annotate("audit reduces\noscillation", xy=(6, ymax * 0.55),
                fontsize=7.5, color="#1f77b4", ha="center", style="italic")

fig.suptitle("SWE-bench Phase Diagram: Oscillation vs. Retry Budget", fontsize=14, y=1.02)
fig.tight_layout()

out_dir = Path(__file__).resolve().parent
fig.savefig(out_dir / "e9_swebench_phase.pdf", bbox_inches="tight", dpi=300)
fig.savefig(out_dir / "e9_swebench_phase.png", bbox_inches="tight", dpi=300)
print("Saved e9_swebench_phase.pdf and .png")

# Print summary table
print("\nSummary (mean oscillation):")
print(f"{'T':>5} {'att':>5} {'none':>8} {'C2':>8} {'delta':>8}")
for T in temps:
    for att in attempts:
        n = mean_osc.get((T, att, "none"), 0)
        c = mean_osc.get((T, att, "C2"), 0)
        print(f"{T:5.1f} {att:5d} {n:8.3f} {c:8.3f} {c-n:+8.3f}")
