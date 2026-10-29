#!/usr/bin/env python3
"""XL-BOMD (AM1, PySEQM) runs for the ISO17 isomer, MD17 aspirin, and a water test.

The settings are those used for the trajectories in results/md/:
AM1, SCF tolerance 1e-6, Niklasson XL-BOMD with k = 6, 500 K initial
temperature, 0.25 fs time step, NVE, one frame written per step.

    python scripts/md/run_xlbomd.py iso17          # 4000 steps (1 ps)
    python scripts/md/run_xlbomd.py md17           # 10000 steps (2.5 ps)
    python scripts/md/run_xlbomd.py water          # 500-step installation test

Output: results/md/<system>_run.0.xyz (trajectory) and .0.h5 (energies,
temperature).  --seed fixes the random initial velocities.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from seqm.MolecularDynamics import XL_BOMD
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants

ROOT = Path(__file__).resolve().parents[2]

SYSTEMS = {
    "iso17": dict(xyz="inputs/iso17_mol.xyz", steps=4000, temp=500.0, dt=0.25, scf_eps=1e-6, thermo=10),
    "md17": dict(xyz="inputs/md17_mol.xyz", steps=10000, temp=500.0, dt=0.25, scf_eps=1e-6, thermo=10),
    "water": dict(xyz="inputs/water.xyz", steps=500, temp=300.0, dt=0.5, scf_eps=1e-8, thermo=50),
}

p = argparse.ArgumentParser()
p.add_argument("system", choices=SYSTEMS)
p.add_argument("--steps", type=int, help="override the number of MD steps")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="results/md")
args = p.parse_args()
cfg = SYSTEMS[args.system]
steps = args.steps or cfg["steps"]

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

atoms = read(ROOT / cfg["xyz"])
Z = np.array(atoms.get_atomic_numbers())
order = np.argsort(-Z, kind="stable")                 # PySEQM needs descending Z
Z, R = Z[order], np.array(atoms.get_positions())[order]
species = torch.as_tensor(Z[None], dtype=torch.int64, device=device)
coordinates = torch.as_tensor(R[None], dtype=torch.float64, device=device)

seqm_parameters = {
    "method": "AM1",
    "scf_eps": cfg["scf_eps"],
    "scf_converger": [0, 0.1],
    "elements": sorted(set(Z.tolist())),
}
const = Constants().to(device)
molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)

out = ROOT / args.out
out.mkdir(parents=True, exist_ok=True)
md = XL_BOMD(
    xl_bomd_params={"k": 6},
    seqm_parameters=seqm_parameters,
    Temp=cfg["temp"],
    timestep=cfg["dt"],
    output={"molid": [0], "thermo": cfg["thermo"], "dump": 1,
            "prefix": str(out / f"{args.system}_run")},
).to(device)

md.run(molecule, steps, seed=args.seed)
print(f"done: {out / (args.system + '_run.0.xyz')}")
