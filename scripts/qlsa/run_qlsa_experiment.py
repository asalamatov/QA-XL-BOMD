#!/usr/bin/env python3
"""QLSA experiment for the linearized XL-BOMD time-grid system.

For each recurrence scheme, surrogate parameter gamma and window length T:
  1. build M_time and the right-hand side from P_0..P_{m-1} and D*,
  2. propagate classically, step by step (the reference),
  3. emulate a QLSA linear solve of the whole window at tolerance eps,
  4. report kappa_eff, polynomial degree (block-encoding queries) and the
     deviation of the quantum output from the classical reference.

Inputs: --densities densities_<label>.npz written by export_densities.py
(real AM1 densities from PySEQM).  Without it, --synthetic d runs a clearly
labelled synthetic test (random symmetric matrices); synthetic results must
not be reported as molecular results.

    python scripts/qlsa/run_qlsa_experiment.py --densities results/qlsa/densities_iso17.npz --label iso17
    python scripts/qlsa/run_qlsa_experiment.py --synthetic 10 --label synthetic
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qlsa.qsvt import qsvt_solve
from qlsa.recurrences import (characteristic_roots, classical_sequential,
                              coefficients, rhs_time_basis, time_matrix)

p = argparse.ArgumentParser()
src = p.add_mutually_exclusive_group(required=True)
src.add_argument("--densities")
src.add_argument("--synthetic", type=int, metavar="D")
p.add_argument("--label", required=True)
p.add_argument("--T", type=int, nargs="+", default=[25, 50, 100, 200, 400])
p.add_argument("--gamma", type=float, nargs="+", default=[0.25, 0.5, 0.75])
p.add_argument("--paper-alpha", type=float, nargs="+", default=[0.0, 0.5])
p.add_argument("--k", type=int, default=6, help="PySEQM XL-BOMD history length")
p.add_argument("--eps", type=float, default=1e-8)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="results/qlsa")
args = p.parse_args()
args.out = str(ROOT / args.out)
os.makedirs(args.out, exist_ok=True)

# ── inputs ──────────────────────────────────────────────────────────────
if args.densities:
    z = np.load(ROOT / args.densities if not os.path.isabs(args.densities) else args.densities)
    Dstar = z["Dstar"]
    frames = z["D_frames"]            # SCF densities along the MD trajectory
    method = str(z["method"]) if "method" in z.files else "PySEQM AM1"
    source = f"{method} densities from {os.path.basename(args.densities)}"
else:
    rng = np.random.default_rng(args.seed)
    d = args.synthetic
    S = rng.normal(size=(d, d)); Dstar = (S + S.T) / 2
    frames = np.stack([Dstar + 1e-2 * n * (lambda X: (X + X.T) / 2)(rng.normal(size=(d, d)))
                       for n in range(16)])
    source = f"SYNTHETIC random symmetric matrices, d={d} (not molecular data)"
d = Dstar.shape[0]
print(f"input: {source}; d = {d}")


def history(m):
    if len(frames) < m:
        raise SystemExit(f"need {m} initial densities, file has {len(frames)}")
    return frames[:m]


def run_case(scheme, gamma, T, alpha=None):
    kw = dict(gamma=gamma)
    if scheme == "paper":
        kw["alpha"] = alpha
    else:
        kw["k"] = args.k
    a, d_coef = coefficients(scheme, **kw)
    m = len(a)
    roots = np.abs(characteristic_roots(a))
    M = time_matrix(a, T)
    hist = history(m)
    # classical step-by-step reference
    t0 = time.time()
    P_c = classical_sequential(a, d_coef, hist, Dstar, T)
    t_classical = time.time() - t0
    # right-hand side as (T+1, d^2): time profile (x) matrix block
    B = rhs_time_basis(m, T)
    blocks = np.concatenate([hist.reshape(m, -1), (d_coef * Dstar).reshape(1, -1)])
    rhs = B.T @ blocks
    # consistency: direct solve equals sequential propagation
    x_direct = np.linalg.solve(M, rhs)
    direct_vs_seq = np.linalg.norm(x_direct - P_c.reshape(T + 1, -1)) / np.linalg.norm(P_c)
    # emulated QSVT solve
    t0 = time.time()
    r = qsvt_solve(M, rhs, args.eps)
    t_qsvt = time.time() - t0
    row = dict(scheme=scheme, alpha=alpha, k=args.k if scheme == "pyseqm" else None,
               gamma=gamma, T=T, d=d, max_root=float(roots.max()),
               cond_M=float(np.linalg.cond(M)), kappa_eff=r.kappa_eff,
               block_encoding_alpha=r.alpha, degree=r.degree, feasible=r.feasible,
               eps=args.eps, direct_vs_sequential=float(direct_vs_seq),
               t_classical_s=t_classical, t_emulation_s=t_qsvt)
    if r.feasible:
        x_c = P_c.reshape(T + 1, -1)
        state_c = x_c / np.linalg.norm(x_c)
        # output state vs exact normalized solution
        row["state_error"] = float(np.linalg.norm(r.x_state - state_c))
        # deviation in P units, assuming the solution norm is known exactly
        P_q = (r.x_state * np.linalg.norm(x_c)).reshape(T + 1, d, d)
        row["P_rel_frobenius"] = float(np.linalg.norm(P_q - P_c) / np.linalg.norm(P_c))
        row["P_max_abs"] = float(np.abs(P_q - P_c).max())
        # surrogate trajectory vs SCF densities along the real trajectory
        n = min(len(frames), T + 1)
        if args.densities and n > m:
            row["surrogate_vs_scf_rel"] = float(
                np.linalg.norm(P_c[m:n] - frames[m:n]) / np.linalg.norm(frames[m:n]))
            row["surrogate_vs_scf_frames"] = int(n - m)
    return row


rows = []
for T in args.T:
    for g in args.gamma:
        for al in args.paper_alpha:
            rows.append(run_case("paper", g, T, alpha=al))
        rows.append(run_case("pyseqm", g, T))
        for r in rows[-len(args.paper_alpha) - 1:]:
            tag = f"{r['scheme']}" + (f"(alpha={r['alpha']})" if r["scheme"] == "paper" else f"(k={r['k']})")
            msg = (f"T={T:4d} gamma={g:.2f} {tag:18s} |root|max={r['max_root']:.4f} "
                   f"kappa_eff={r['kappa_eff']:.3e} degree={r['degree']}")
            msg += (f" P_rel_err={r['P_rel_frobenius']:.2e}" if r["feasible"] else " INFEASIBLE (degree too large)")
            print(msg)

meta = dict(source=source, eps=args.eps, seed=args.seed, k=args.k,
            numpy=np.__version__, note="Matrix-level QSVT emulation (see qlsa/qsvt.py); "
            "gate noise, phase finding, success probability and readout sampling are not modelled.")
out = os.path.join(args.out, f"qlsa_{args.label}.json")
with open(out, "w") as f:
    json.dump(dict(meta=meta, rows=rows), f, indent=1)
print(f"\nwrote {out}")
