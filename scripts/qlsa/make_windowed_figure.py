"""Figure for the periodic re-linearization experiment.

    python scripts/qlsa/make_windowed_figure.py results/qlsa/windowed_iso17.json results/qlsa/windowed_md17.json

(a, b) relative deviation of the windowed linear propagation from the AM1 SCF density, per frame,
for several window lengths W; (c) mean deviation and number of SCF evaluations against W.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--dt", type=float, default=0.25, help="time step in fs")
ap.add_argument("--show-W", type=int, nargs="+", default=[5, 10, 20, 100])
ap.add_argument("--out", default="figures/qlsa/fig_windowed")
args = ap.parse_args()

names = {"iso17": "ISO17 isomer", "md17": "Aspirin (MD17)"}
colors = ["#1b6ca8", "#e08a1e", "#3a9b5c", "#b03a48", "#6b4c9a", "#7f7f7f", "#a6761d"]
data = []
for f in args.files:
    label = os.path.basename(f).replace("windowed_", "").replace(".json", "")
    data.append((label, json.load(open(f))["rows"]))

fig, axes = plt.subplots(1, len(data) + 1, figsize=(4.0 * (len(data) + 1), 3.2))
for ax, (label, rows) in zip(axes, data):
    # largest window first, so the small-W curves are drawn on top
    for c, W in reversed(list(zip(colors, args.show_W))):
        r = next((r for r in rows if r["W"] == W), None)
        if r is None:
            continue
        e = 100 * np.asarray(r["error_per_frame"])
        t = args.dt * np.arange(len(e))
        ax.plot(t, e, color=c, lw=0.6, label=f"W = {W}")
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("deviation from SCF density (%)")
    ax.set_title(names.get(label, label), fontsize=10)
    ax.set_ylim(0, 30)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], fontsize=8, frameon=False, loc="upper center", ncol=4,
              handlelength=1.2, columnspacing=1.0)

ax = axes[-1]
ax2 = ax.twinx()
for (label, rows), mk, c in zip(data, ["o", "s"], ["#1b6ca8", "#e08a1e"]):
    W = np.array([r["W"] for r in rows])
    ax.plot(W, [100 * r["rel_frobenius_all"] for r in rows], marker=mk, color=c,
            ms=4, lw=1, label=names.get(label, label))
ax2.plot(W, [r["n_scf"] for r in data[0][1]], ls="--", color="#7f7f7f", lw=1,
         label="SCF evaluations")
ax.set_xscale("log")
ax2.set_yscale("log")
ax.set_xlabel("window length W (steps)")
ax.set_ylabel("overall deviation (%)")
ax2.set_ylabel("SCF evaluations per 1000 steps (dashed)", color="#7f7f7f")
ax.set_ylim(bottom=0)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="upper left")
ax.set_ylim(0, 25)
for a, letter in zip(axes, "abc"):
    a.text(-0.16, 1.04, f"({letter})", transform=a.transAxes, fontsize=10, fontweight="bold")
fig.tight_layout()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
print("wrote", args.out + ".pdf")
