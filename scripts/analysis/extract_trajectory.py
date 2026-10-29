#!/usr/bin/env python3
"""Observables from a PySEQM XL-BOMD run.

Inputs:  results/md/<label>_run.0.xyz   trajectory (Angstrom)
         results/md/<label>_run.0.h5    /data/thermo/{Ek,Ep,T}, /data/steps
Outputs: results/observables/{rmsd,rmsf,energy,pmf}_<label>.npz

Settings used for the paper's figures:
    python scripts/analysis/extract_trajectory.py --label iso17
    python scripts/analysis/extract_trajectory.py --label md17 --discard 2000 --mean-ref

--discard N drops the first N frames (equilibration). For MD17 aspirin the
first 0.5 ps relaxes from the DFT-optimized start geometry.
--mean-ref uses the mean post-equilibration geometry as the RMSD/RMSF
reference; otherwise frame --ref (default 0) is used.
Each .npz stores the settings under the keys "discard" and "reference".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[2]
LABEL = "XL-BOMD"

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True, help="iso17, md17, water, ...")
parser.add_argument("--prefix", help="run prefix (default: results/md/<label>_run)")
parser.add_argument("--discard", type=int, default=0, help="drop first N frames")
parser.add_argument("--mean-ref", action="store_true", help="mean geometry as reference")
parser.add_argument("--ref", type=int, default=0, help="reference frame if not --mean-ref")
parser.add_argument("--out", default="results/observables")
args = parser.parse_args()

prefix = Path(args.prefix) if args.prefix else ROOT / "results/md" / f"{args.label}_run"
out = ROOT / args.out
out.mkdir(parents=True, exist_ok=True)


def read_xyz(path):
    """Return (symbols, coords (n_frames, n_atoms, 3))."""
    frames, symbols = [], None
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].strip())
        i += 2
        syms, coords = [], []
        for k in range(n):
            parts = lines[i + k].split()
            syms.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
        i += n
        frames.append(coords)
        if symbols is None:
            symbols = syms
    return symbols, np.array(frames)


def kabsch(P, Q):
    """Rotate P onto Q (both (n_atoms, 3))."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return (R @ Pc.T).T + Q.mean(axis=0)


symbols, coords = read_xyz(f"{prefix}.0.xyz")
print(f"xyz: {coords.shape[0]} frames, {coords.shape[1]} atoms")
with h5py.File(f"{prefix}.0.h5", "r") as f:
    Ek = f["data/thermo/Ek"][:].reshape(-1)
    Ep = f["data/thermo/Ep"][:].reshape(-1)
    steps = f["data/steps"][:].reshape(-1)

n = min(len(Ek), len(Ep), len(steps), len(coords))
coords, Ek, Ep, steps = coords[:n], Ek[:n], Ep[:n], steps[:n]
d = args.discard
if d >= n - 10:
    raise SystemExit(f"--discard {d} leaves too few frames (n = {n})")
coords, Ek, Ep, steps = coords[d:], Ek[d:], Ep[d:], steps[d:]
print(f"using frames {d}..{n - 1} ({len(coords)} frames)")

R_ref = coords.mean(axis=0) if args.mean_ref else coords[args.ref]
reference = "mean geometry" if args.mean_ref else f"frame {args.ref}"
meta = dict(discard=d, reference=reference)

# RMSD
rmsd = np.array([np.sqrt(np.mean(np.sum((kabsch(c, R_ref) - R_ref) ** 2, axis=1))) for c in coords])
np.savez(out / f"rmsd_{args.label}.npz", axis=steps, **{LABEL: rmsd}, **meta)
print(f"rmsd: {rmsd.min():.3f}-{rmsd.max():.3f} A (reference: {reference})")

# RMSF
aligned = np.stack([kabsch(c, R_ref) for c in coords])
rmsf = np.sqrt(np.mean(np.sum((aligned - aligned.mean(axis=0)) ** 2, axis=2), axis=0))
np.savez(out / f"rmsf_{args.label}.npz", axis=np.arange(1, coords.shape[1] + 1),
         **{LABEL: rmsf}, symbols=np.array(symbols), **meta)
print(f"rmsf: {rmsf.min():.3f}-{rmsf.max():.3f} A")

# total-energy drift (eV)
E_tot = Ek + Ep
drift = E_tot - E_tot[0]
np.savez(out / f"energy_{args.label}.npz", axis=steps, **{LABEL: drift}, **meta)
print(f"energy: final drift {drift[-1]:.3e} eV")

# potential energy vs RMSD: 5th percentile per RMSD bin, smoothed (eV)
nbins = 60
bins = np.linspace(rmsd.min(), rmsd.max(), nbins + 1)
centers = 0.5 * (bins[:-1] + bins[1:])
which = np.digitize(rmsd, bins) - 1   # the single maximum lands past the last bin, as in the paper's run
E_rel = Ep - Ep.min()
surf = np.full(nbins, np.nan)
for k in range(nbins):
    m = which == k
    if m.sum() >= 5:
        surf[k] = np.percentile(E_rel[m], 5)
ok = ~np.isnan(surf)
if ok.sum() >= 2:
    surf = np.interp(np.arange(nbins), np.where(ok)[0], surf[ok])
    surf = gaussian_filter1d(surf, sigma=2)
    surf -= surf.min()
    np.savez(out / f"pmf_{args.label}.npz", axis=centers, **{LABEL: surf}, **meta)
    print(f"energy surface: max {surf.max():.3f} eV")
else:
    print("energy surface too sparse; skipped")
