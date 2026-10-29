#!/usr/bin/env python3
"""Windowed re-linearization of the XL-BOMD surrogate.

A single surrogate D(P) = (1-gamma) D* + gamma P with one fixed D* drifts away
from the true SCF densities as the nuclei move. Here D* is refreshed every W
steps: at the start of each window it is set to the SCF density at the current
geometry (one classical SCF per window), and the linearized recurrence is then
propagated across the window, which is one linear system of size (W+1) d^2.
The propagated P carries over from window to window.

This is an offline test along the reference PySEQM trajectory: the geometries
and the reference SCF densities D_n come from export_densities.py, and the
nuclei are not re-propagated with surrogate forces.

For every window size W the script reports
  - the relative Frobenius distance between the windowed surrogate trajectory
    and the SCF densities, over all frames, and its maximum per frame;
  - the number of classical SCF calls (T/W) and quantum solves (T/W);
  - kappa_eff and the QSVT polynomial degree for one window of length W;
  - the worst deviation between the emulated QSVT solve and the classical
    solve of the same window.

    python scripts/qlsa/run_windowed_experiment.py --densities results/qlsa/densities_iso17.npz --label iso17
    python scripts/qlsa/run_windowed_experiment.py --synthetic 6 --label synthetic   # code test only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qlsa.qsvt import qsvt_solve  # noqa: E402
from qlsa.recurrences import (characteristic_roots, coefficients,  # noqa: E402
                              rhs_time_basis, time_matrix)

p = argparse.ArgumentParser()
src = p.add_mutually_exclusive_group(required=True)
src.add_argument("--densities")
src.add_argument("--synthetic", type=int, metavar="D")
p.add_argument("--label", required=True)
p.add_argument("--W", type=int, nargs="+", default=[5, 10, 20, 50, 100, 200, 500])
p.add_argument("--gamma", type=float, nargs="+", default=[0.5])
p.add_argument("--scheme", choices=["pyseqm", "paper"], default="pyseqm")
p.add_argument("--k", type=int, default=6)
p.add_argument("--alpha", type=float, default=0.5,
               help="dissipation of the two-step update (Niklasson Eq. 24), used with --scheme paper")
p.add_argument("--eps", type=float, default=1e-8)
p.add_argument("--max-qsvt-windows", type=int, default=50,
               help="emulate QSVT on at most this many windows per run (the rest are solved classically)")
p.add_argument("--out", default="results/qlsa")
args = p.parse_args()
out_dir = ROOT / args.out
out_dir.mkdir(parents=True, exist_ok=True)

if args.densities:
    path = Path(args.densities)
    z = np.load(path if path.is_absolute() else ROOT / path)
    D = z["D_frames"].astype(float)
    method = str(z["method"]) if "method" in z.files else "PySEQM AM1"
    source = f"{method} densities, {len(D)} frames ({path.name})"
else:
    rng = np.random.default_rng(0)
    d, n = args.synthetic, 400
    A = rng.normal(size=(d, d)); A = (A + A.T) / 2
    B = rng.normal(size=(d, d)); B = (B + B.T) / 2
    t = np.arange(n)[:, None, None]
    D = A + 0.3 * np.sin(2 * np.pi * t / 300) * B           # slowly drifting "SCF density"
    source = f"SYNTHETIC drifting matrices, d={d}, {n} frames (code test only)"
T = len(D) - 1
print(f"input: {source}")


def window_matrix(a, W):
    """System for one window: m history rows followed by W new steps."""
    return time_matrix(a, len(a) + W - 1)


def window_basis(m, W):
    return rhs_time_basis(m, m + W - 1)


def run(gamma, W):
    kw = dict(gamma=gamma, k=args.k) if args.scheme == "pyseqm" else dict(gamma=gamma, alpha=args.alpha)
    a, d_coef = coefficients(args.scheme, **kw)
    m = len(a)
    M = window_matrix(a, W)
    B = window_basis(m, W)
    P = [D[i].copy() for i in range(m)]           # initial history from SCF densities
    n_scf = 0
    qsvt_dev = []
    windows = 0
    s = m - 1                                     # last index with a known P
    while s < T:
        Dstar = D[s]                              # one SCF at the current geometry
        n_scf += 1
        W_eff = min(W, T - s)                     # new steps in this window
        Mw = M if W_eff == W else window_matrix(a, W_eff)
        Bw = B if W_eff == W else window_basis(m, W_eff)
        hist = np.stack(P[s - m + 1: s + 1])      # P_{s-m+1..s}
        blocks = np.concatenate([hist.reshape(m, -1), (d_coef * Dstar).reshape(1, -1)])
        rhs = Bw.T @ blocks
        x = np.linalg.solve(Mw, rhs)              # classical solve of this window
        if windows < args.max_qsvt_windows:
            r = qsvt_solve(Mw, rhs, args.eps)
            if r.feasible:
                xq = r.x_state * np.linalg.norm(x)
                qsvt_dev.append(float(np.linalg.norm(xq - x) / np.linalg.norm(x)))
        new = x.reshape(m + W_eff, *D.shape[1:])[m:]   # the W_eff new steps
        P.extend(list(new))
        s += len(new)
        windows += 1
    P = np.stack(P[:T + 1])
    err_frame = np.linalg.norm((P - D).reshape(T + 1, -1), axis=1) / np.linalg.norm(D.reshape(T + 1, -1), axis=1)
    rel_total = float(np.linalg.norm(P - D) / np.linalg.norm(D))
    r1 = qsvt_solve(M, B.T @ np.ones((m + 1, 1)), args.eps)   # cost of one full window
    return dict(scheme=args.scheme, k=args.k if args.scheme == "pyseqm" else None,
                alpha=args.alpha if args.scheme == "paper" else None, gamma=gamma, W=W,
                T=T, max_root=float(np.abs(characteristic_roots(a)).max()),
                rel_frobenius_all=rel_total, max_frame_error=float(err_frame.max()),
                final_frame_error=float(err_frame[-1]), n_windows=windows, n_scf=n_scf,
                kappa_eff_window=r1.kappa_eff, qsvt_degree_window=r1.degree,
                total_queries=int(r1.degree * windows),
                qsvt_vs_classical_max=max(qsvt_dev) if qsvt_dev else None,
                qsvt_windows_checked=len(qsvt_dev), error_per_frame=err_frame.tolist())


rows = []
for g in args.gamma:
    for W in args.W:
        if W > T:
            continue
        r = run(g, W)
        if r is None:
            continue
        rows.append(r)
        print(f"gamma={g:.2f} W={W:4d}  rel.err(all)={100*r['rel_frobenius_all']:6.2f}%  "
              f"max frame={100*r['max_frame_error']:6.2f}%  SCFs={r['n_scf']:4d}  "
              f"kappa_win={r['kappa_eff_window']:8.1f}  queries/win={r['qsvt_degree_window']:7d}  "
              f"QLSA vs classical<={r['qsvt_vs_classical_max']:.1e}")

meta = dict(source=source, scheme=args.scheme, k=args.k, eps=args.eps,
            note="offline test along the reference trajectory; D* refreshed with the SCF density "
                 "at each window start; QLSA emulated at matrix level (see qlsa/qsvt.py)")
out = out_dir / f"windowed_{args.label}.json"
json.dump(dict(meta=meta, rows=rows), open(out, "w"), indent=1)
print(f"wrote {out}")
