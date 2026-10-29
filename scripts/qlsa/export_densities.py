#!/usr/bin/env python3
"""Export AM1 density matrices from PySEQM for the QLSA experiment.

D*        : converged SCF density at the first frame of the XL-BOMD trajectory
D_frames  : converged SCF densities at the first N frames of that trajectory
            (used as the initial history P_0..P_{m-1} and as the SCF reference
            for the surrogate-error estimate)

    python scripts/qlsa/export_densities.py --xyz results/md/iso17_run.0.xyz --label iso17 --frames 200
    python scripts/qlsa/export_densities.py --xyz results/md/md17_run.0.xyz  --label md17  --frames 200
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants

p = argparse.ArgumentParser()
p.add_argument("--xyz", required=True, help="PySEQM trajectory, e.g. results/md/iso17_run.0.xyz")
p.add_argument("--label", required=True)
p.add_argument("--frames", type=int, default=200)
p.add_argument("--batch", type=int, default=50)
p.add_argument("--out", default="results/qlsa")
args = p.parse_args()
ROOT = Path(__file__).resolve().parents[2]
args.out = str(ROOT / args.out)
if not os.path.isabs(args.xyz):
    args.xyz = str(ROOT / args.xyz)
os.makedirs(args.out, exist_ok=True)

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

frames = read(args.xyz, index=f":{args.frames}")
Z = np.array(frames[0].get_atomic_numbers())
order = np.argsort(-Z, kind="stable")          # PySEQM needs descending Z
Z = Z[order]
R = np.stack([f.get_positions()[order] for f in frames])

seqm_parameters = {
    "method": "AM1",              # same settings as run_iso17.py / run_md17.py
    "scf_eps": 1e-6,
    "scf_converger": [0, 0.1],
    "elements": sorted(set(Z.tolist()) | {0})[1:],
}
const = Constants().to(device)
esdriver = Electronic_Structure(seqm_parameters).to(device)

dms = []
for s in range(0, len(R), args.batch):
    Rb = torch.as_tensor(R[s:s + args.batch], device=device)
    Zb = torch.as_tensor(np.repeat(Z[None], len(Rb), 0), dtype=torch.int64, device=device)
    mol = Molecule(const, seqm_parameters, Rb, Zb).to(device)
    esdriver(mol)
    if getattr(esdriver, "notconverged", None) is not None and bool(torch.any(esdriver.notconverged)):
        print(f"warning: SCF not converged in batch starting at frame {s}")
    dm = mol.dm.detach().cpu().numpy()
    if dm.ndim != 3:
        raise SystemExit(f"unexpected density shape {dm.shape} (open-shell?)")
    dms.append(dm)
D_frames = np.concatenate(dms)
out = os.path.join(args.out, f"densities_{args.label}.npz")
np.savez_compressed(out, Dstar=D_frames[0], D_frames=D_frames, Z=Z,
                    source=os.path.basename(args.xyz), method="PySEQM AM1")
print(f"wrote {out}: {D_frames.shape[0]} frames, d = {D_frames.shape[1]}")
