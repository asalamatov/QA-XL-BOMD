#!/usr/bin/env python3
"""Figures from run_qlsa_experiment.py output.

    python scripts/qlsa/make_qlsa_figures.py results/qlsa/qlsa_iso17.json [more.json ...]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("files", nargs="+")
p.add_argument("--out", default="figures/qlsa")
p.add_argument("--gamma", type=float, default=0.5)
args = p.parse_args()
ROOT = Path(__file__).resolve().parents[2]
args.out = str(ROOT / args.out)
args.files = [f if os.path.isabs(f) else str(ROOT / f) for f in args.files]
os.makedirs(args.out, exist_ok=True)

rows, sources = [], set()
for f in args.files:
    z = json.load(open(f))
    rows += z["rows"]
    sources.add(z["meta"]["source"])


def key(r):
    return f"paper, α={r['alpha']}" if r["scheme"] == "paper" else f"PySEQM XL-BOMD, k={r['k']}"


series = defaultdict(list)
for r in rows:
    if abs(r["gamma"] - args.gamma) < 1e-9:
        series[key(r)].append(r)

has_sur = any("surrogate_vs_scf_rel" in r for r in rows)
fig, axes = plt.subplots(1, 4 if has_sur else 3, figsize=(17 if has_sur else 13, 3.8))
for name, rs in sorted(series.items()):
    rs = sorted({r["T"]: r for r in rs}.values(), key=lambda r: r["T"])
    T = [r["T"] for r in rs]
    axes[0].loglog(T, [r["kappa_eff"] for r in rs], "o-", label=name)
    axes[1].loglog(T, [r["degree"] for r in rs], "o-", label=name)
    ok = [r for r in rs if r["feasible"]]
    axes[2].loglog([r["T"] for r in ok], [r["P_rel_frobenius"] for r in ok], "o-", label=name)
    if has_sur:
        sur = [r for r in ok if "surrogate_vs_scf_rel" in r and r["surrogate_vs_scf_frames"] >= r["T"] - 8]
        axes[3].semilogx([r["T"] for r in sur], [100 * r["surrogate_vs_scf_rel"] for r in sur], "o-", label=name)
if has_sur:
    axes[3].set_ylabel("surrogate vs SCF density (% rel. Frobenius)")
axes[0].set_ylabel(r"$\kappa_\mathrm{eff} = \alpha_\mathrm{BE}/\sigma_\mathrm{min}$")
axes[1].set_ylabel("QLSA degree (block-encoding queries)")
axes[2].set_ylabel(r"$\|P_q-P_c\|_F / \|P_c\|_F$")
Ts = sorted({r["T"] for r in rows})
for ax in axes:
    ax.set_xticks(Ts)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([str(t) for t in Ts])
    ax.set_xlabel("window length T (steps)")
    ax.grid(alpha=0.3, which="both")
axes[0].legend(fontsize=8)
fig.suptitle(f"QLSA linear solve of the linearized XL-BOMD system, γ = {args.gamma}\n"
             f"input: {'; '.join(sorted(sources))}", fontsize=9)
fig.tight_layout()
tag = os.path.splitext(os.path.basename(args.files[0]))[0].replace("qlsa_", "")
out = os.path.join(args.out, f"fig_qlsa_{tag}_gamma{args.gamma}.pdf")
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200)
print("wrote", out)
