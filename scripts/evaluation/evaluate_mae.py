#!/usr/bin/env python3
"""Energy and force MAEs of AM1 (PySEQM) or GFN2-xTB (tblite) against the DFT labels of ISO17 and MD17.

This evaluates the same electronic-structure model the XL-BOMD runs use
(AM1, PySEQM, same SCF settings as run_iso17.py / run_md17.py) on the
benchmark test sets the ML baselines are scored on, and compares with the
datasets' DFT (PBE+TS) energies and forces.

Energy offset: AM1 total energies and PBE total energies have different
zero points, so the energy MAE is reported after subtracting one constant
offset per dataset (all molecules in a dataset have the same composition).
The offset is the mean difference on a separate calibration sample that is
disjoint from the test sample, so it is not fitted on the test data.
Forces need no offset.

Usage:
    python scripts/evaluation/evaluate_mae.py iso17 --n 1000      # AM1 (PySEQM), data/iso17/*.db
    python scripts/evaluation/evaluate_mae.py md17  --n 1000      # AM1 (PySEQM), data/md17/md17_*.npz
    python scripts/evaluation/evaluate_mae.py md17  --method GFN2-xTB   # GFN2-xTB (pip install tblite)

Results are printed and written to results/mae/<dataset>.json (AM1) or
results/mae/<dataset>_gfn2.json (GFN2-xTB). The test and calibration samples are drawn with the
same seed for both methods, so the two are scored on identical structures.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

EV_TO_KCAL = 23.060548867  # kcal/mol per eV
HARTREE_TO_KCAL = 627.509474063
BOHR_TO_ANGSTROM = 0.529177210903

p = argparse.ArgumentParser()
p.add_argument("dataset", choices=["iso17", "md17"])
p.add_argument("--db-dir", default="data/iso17", help="folder with the ISO17 .db files")
p.add_argument("--npz", nargs="+", default=None, help="sGDML MD17 .npz files (default: all in data/md17)")
p.add_argument("--n", type=int, default=1000, help="test structures per split / molecule")
p.add_argument("--n-calib", type=int, default=200, help="structures for the energy offset")
p.add_argument("--batch", type=int, default=100)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--method", default="AM1",
               help="AM1, PM3 or MNDO (PySEQM), or GFN2-xTB / GFN1-xTB (tblite)")
p.add_argument("--xtb-accuracy", type=float, default=1.0, help="tblite accuracy setting")
p.add_argument("--out", default="results/mae")
args = p.parse_args()
ROOT = Path(__file__).resolve().parents[2]
args.out = str(ROOT / args.out)
args.db_dir = str(ROOT / args.db_dir) if not os.path.isabs(args.db_dir) else args.db_dir
if args.npz is None:
    args.npz = sorted(str(f) for f in (ROOT / "data/md17").glob("md17_*.npz"))
os.makedirs(args.out, exist_ok=True)

XTB = args.method.upper().endswith("XTB")
if XTB:
    from tblite.interface import Calculator
else:
    import torch
    from seqm.ElectronicStructure import Electronic_Structure
    from seqm.Molecule import Molecule
    from seqm.seqm_functions.constants import Constants
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    const = Constants().to(device)
rng = np.random.default_rng(args.seed)


def xtb_energy_forces(Z, R):
    """GFN2-xTB (tblite) single points. Z (n_atoms,), R (n_struct, n_atoms, 3) in Angstrom.
    Returns energies (kcal/mol), forces (kcal/mol/A) and a convergence mask."""
    E, F, ok = [], [], []
    res = None
    calc = None
    for Ri in R:
        pos = np.asarray(Ri, dtype=float) / BOHR_TO_ANGSTROM
        try:
            if calc is None:
                calc = Calculator(args.method, Z, pos)
                calc.set("accuracy", args.xtb_accuracy)
                calc.set("verbosity", 0)
            else:
                calc.update(pos)
            res = calc.singlepoint(res) if res is not None else calc.singlepoint()
            E.append(float(res.get("energy")) * HARTREE_TO_KCAL)
            F.append(-np.asarray(res.get("gradient")) * HARTREE_TO_KCAL / BOHR_TO_ANGSTROM)
            ok.append(True)
        except Exception as exc:  # SCC failure: record and continue with a fresh guess
            print("  tblite failed:", exc)
            E.append(np.nan); F.append(np.full((len(Z), 3), np.nan)); ok.append(False)
            res = None
    return np.array(E), np.array(F), np.array(ok)


def am1_energy_forces(Z, R):
    if XTB:
        return xtb_energy_forces(Z, R)
    """Z (n_atoms,), R (n_struct, n_atoms, 3) in Angstrom.
    Returns energies (kcal/mol), forces (kcal/mol/A) in the input atom order,
    and a mask of structures whose SCF converged."""
    order = np.argsort(-Z, kind="stable")           # PySEQM needs descending Z
    inv = np.argsort(order)
    Zs = Z[order]
    params = {
        "method": args.method,
        "scf_eps": 1e-6,                             # same as run_iso17.py / run_md17.py
        "scf_converger": [0, 0.1],
        "elements": sorted(set(int(z) for z in Zs)),
    }
    es = Electronic_Structure(params).to(device)
    E, F, ok = [], [], []
    for s in range(0, len(R), args.batch):
        Rb = torch.as_tensor(R[s:s + args.batch][:, order], device=device)
        Zb = torch.as_tensor(np.repeat(Zs[None], len(Rb), 0), dtype=torch.int64, device=device)
        mol = Molecule(const, params, Rb, Zb).to(device)
        es(mol)
        E.append(mol.Etot.detach().cpu().numpy() * EV_TO_KCAL)
        F.append(mol.force.detach().cpu().numpy()[:, inv] * EV_TO_KCAL)
        nc = getattr(es, "notconverged", None)
        ok.append(np.ones(len(Rb), bool) if nc is None
                  else ~np.asarray(nc.detach().cpu().numpy(), bool).reshape(-1))
    return np.concatenate(E), np.concatenate(F), np.concatenate(ok)


def evaluate(name, Z, R_cal, E_cal, R_test, E_test, F_test):
    t0 = time.time()
    Ec, _, okc = am1_energy_forces(Z, R_cal)
    offset = float(np.mean((E_cal - Ec)[okc]))
    Et, Ft, okt = am1_energy_forces(Z, R_test)
    e_err = (Et + offset - E_test)[okt]
    f_err = (Ft - F_test)[okt]
    res = dict(
        name=name, n_test=int(len(R_test)), n_converged=int(okt.sum()),
        n_calib=int(len(R_cal)), energy_offset_kcal=offset,
        energy_mae_kcal=float(np.mean(np.abs(e_err))),
        energy_rmse_kcal=float(np.sqrt(np.mean(e_err ** 2))),
        force_mae_kcal_A=float(np.mean(np.abs(f_err))),
        force_rmse_kcal_A=float(np.sqrt(np.mean(f_err ** 2))),
        seconds=time.time() - t0,
    )
    print(f"{name:28s} energy MAE {res['energy_mae_kcal']:8.3f} kcal/mol   "
          f"force MAE {res['force_mae_kcal_A']:8.3f} kcal/mol/A   "
          f"({res['n_converged']}/{res['n_test']} converged)")
    return res


results = dict(method=args.method, seed=args.seed, n=args.n, n_calib=args.n_calib,
               offset_note="constant energy offset fitted on a disjoint calibration sample",
               rows=[])

if args.dataset == "iso17":
    from ase.db import connect

    def sample_db(path, n, Z=None):
        with connect(path) as db:
            ids = [r.id for r in db.select()]
            pick = rng.choice(ids, size=min(n, len(ids)), replace=False)
            rows = [db.get(id=int(i)) for i in pick]
        if Z is None:
            Z = rows[0].toatoms().get_atomic_numbers()
        # every ISO17 molecule is C7O2H10, but atom order differs between isomers:
        # reorder each structure to the element order Z (taken from the calibration set)
        R, E, F = [], [], []
        for r in rows:
            a = r.toatoms()
            o = np.argsort(a.get_atomic_numbers(), kind="stable")
            o = o[np.argsort(np.argsort(Z, kind="stable"), kind="stable")]
            assert (a.get_atomic_numbers()[o] == Z).all()
            R.append(a.get_positions()[o])
            E.append(r["total_energy"] * EV_TO_KCAL)
            F.append(np.asarray(r.data["atomic_forces"])[o] * EV_TO_KCAL)
        return Z, np.array(R), np.array(E), np.array(F)

    Z, Rc, Ec, _ = sample_db(os.path.join(args.db_dir, "reference.db"), args.n_calib)
    for split, label in [("test_within.db", "ISO17 known mol / unk conf"),
                         ("test_other.db", "ISO17 unknown mol / unk conf")]:
        _, Rt, Et, Ft = sample_db(os.path.join(args.db_dir, split), args.n, Z=Z)
        results["rows"].append(evaluate(label, Z, Rc, Ec, Rt, Et, Ft))
else:
    for path in args.npz:
        z = np.load(path, allow_pickle=True)
        Z = z["z"].astype(int)
        R, E, F = z["R"], z["E"].reshape(-1), z["F"]     # sGDML: kcal/mol, kcal/mol/A
        idx = rng.permutation(len(R))
        cal, test = idx[:args.n_calib], idx[args.n_calib:args.n_calib + args.n]
        name = str(z["name"].item().decode() if isinstance(z["name"].item(), bytes) else z["name"].item())
        results["rows"].append(evaluate(f"MD17 {name}", Z, R[cal], E[cal], R[test], E[test], F[test]))

suffix = "" if args.method.upper() == "AM1" else "_" + args.method.lower().replace("-xtb", "").replace("-", "")
out = os.path.join(args.out, f"{args.dataset}{suffix}.json")
with open(out, "w") as f:
    json.dump(results, f, indent=1)
print("wrote", out)
