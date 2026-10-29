#!/usr/bin/env python3
"""Figures from results/observables/*.npz (written by extract_trajectory.py).

For each system label found:
    fig_rmsd_<label>, fig_rmsf_<label>, fig_energy_<label>, fig_energy_surface_<label>
as PDF and PNG in figures/md/.

    python scripts/analysis/make_md_figures.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EV_TO_KCAL = 23.060548867
LABEL = "XL-BOMD"
COLOR = "#16A085"
ELEMENT_COLORS = {"C": "#2C3E50", "O": "#C0392B", "H": "#27AE60", "N": "#2471A3"}
TITLES = {
    "water": r"H$_2$O",
    "iso17": r"ISO17 (C$_7$O$_2$H$_{10}$)",
    "md17": r"MD17 aspirin (C$_9$H$_8$O$_4$)",
}

p = argparse.ArgumentParser()
p.add_argument("--data", default="results/observables")
p.add_argument("--out", default="figures/md")
args = p.parse_args()
DATA, OUT = ROOT / args.data, ROOT / args.out
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 11,
    "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linestyle": "--",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def load(kind, label):
    f = DATA / f"{kind}_{label}.npz"
    return np.load(f, allow_pickle=False) if f.exists() else None


def settings(d):
    disc = int(d["discard"]) if "discard" in d.files else 0
    ref = str(d["reference"]) if "reference" in d.files else "frame 0"
    eq = f"first {disc} steps discarded" if disc else "all steps"
    return f"{eq}; reference = {ref}"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("  saved", name)


def fig_rmsd(label, title):
    d = load("rmsd", label)
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(d["axis"], d[LABEL], color=COLOR, lw=1.6, label=f"{LABEL} (AM1)")
    ax.set(xlabel="MD step", ylabel="RMSD (Å)", title=f"RMSD – {title}, NVE",
           xlim=(d["axis"][0], d["axis"][-1]))
    ax.set_ylim(bottom=0)
    ax.text(0.01, 0.02, settings(d), transform=ax.transAxes, fontsize=7, color="0.45", style="italic")
    ax.legend(loc="upper left")
    save(fig, f"fig_rmsd_{label}")


def fig_rmsf(label, title):
    d = load("rmsf", label)
    if d is None:
        return
    idx, y = d["axis"], d[LABEL]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(idx, y, color=COLOR, lw=1.6, marker="o", ms=3, label=f"{LABEL} (AM1)")
    ax.set_xlim(0.5, len(idx) + 0.5)
    ax.set_ylim(0, float(y.max()) * 1.25)
    if "symbols" in d.files:                      # shade runs of the same element
        sym = [str(s) for s in d["symbols"]]
        ymax, start = ax.get_ylim()[1], 0
        for i in range(1, len(sym) + 1):
            if i == len(sym) or sym[i] != sym[start]:
                col = ELEMENT_COLORS.get(sym[start], "0.5")
                ax.axvspan(start + 0.5, i + 0.5, alpha=0.07, color=col, zorder=0)
                ax.text((start + i + 1) / 2, ymax * 0.95, sym[start], fontsize=11,
                        color=col, ha="center", va="top")
                start = i
    ax.set(xlabel="Atom index", ylabel="RMSF (Å)", title=f"Per-atom fluctuation – {title}")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.text(0.01, 0.02, settings(d), transform=ax.transAxes, fontsize=7, color="0.45", style="italic")
    ax.legend(loc="upper right")
    save(fig, f"fig_rmsf_{label}")


def fig_energy(label, title):
    d = load("energy", label)
    if d is None:
        return
    t, y = d["axis"], d[LABEL] * EV_TO_KCAL
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True,
                                  gridspec_kw={"hspace": 0.15})
    ax.plot(t, y, color=COLOR, lw=1.2, label=f"{LABEL} (AM1)")
    ax.axhline(0, color="0.55", lw=0.7, ls=(0, (3, 3)))
    ax.set(ylabel=r"$\Delta E_\mathrm{tot}$ (kcal mol$^{-1}$)", title=f"Energy conservation – {title}")
    ax.legend(loc="lower left")
    win = max(5, len(t) // 20)            # 5% of the trajectory, as in the original figures
    cs = np.cumsum(np.insert(y, 0, 0.0))
    ys = (cs[win:] - cs[:-win]) / win
    ax2.plot(t[win // 2: win // 2 + len(ys)], ys, color=COLOR, lw=1.6)
    ax2.axhline(0, color="0.55", lw=0.7, ls=(0, (3, 3)))
    ax2.set(xlabel="MD step", ylabel=r"running mean of $\Delta E_\mathrm{tot}$")
    ax2.text(0.99, 0.05, f"window = {win} steps; {settings(d).split(';')[0]}",
             transform=ax2.transAxes, ha="right", fontsize=7, color="0.45", style="italic")
    save(fig, f"fig_energy_{label}")


def fig_surface(label, title):
    d = load("pmf", label)
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.plot(d["axis"], d[LABEL] * EV_TO_KCAL, color=COLOR, lw=1.8, label=f"{LABEL} (AM1)")
    ax.set(xlabel="RMSD from reference (Å)", ylabel=r"Energy (kcal mol$^{-1}$)",
           title=f"Potential energy vs RMSD (5th percentile per bin) – {title}",
           xlim=(d["axis"][0], d["axis"][-1]))
    ax.set_ylim(bottom=-0.02)
    ax.legend(loc="upper left")
    save(fig, f"fig_energy_surface_{label}")


labels = sorted({f.stem.split("_", 1)[1] for f in DATA.glob("*.npz")})
if not labels:
    raise SystemExit(f"no observables in {DATA}; run extract_trajectory.py first")
for label in labels:
    print(f"[{label}]")
    title = TITLES.get(label, label)
    for fn in (fig_rmsd, fig_rmsf, fig_energy, fig_surface):
        fn(label, title)
