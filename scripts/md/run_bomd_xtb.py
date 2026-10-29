#!/usr/bin/env python3
"""Born-Oppenheimer MD with GFN2-xTB (tblite), saving the SCF density matrix at every step.

This is the GFN2-xTB counterpart of the AM1 runs. It uses the same starting geometries and the
same settings (NVE, 500 K initial temperature, 0.25 fs step, fixed seed), and it writes the
density matrices in the format the QLSA scripts read, so

    python scripts/md/run_bomd_xtb.py iso17 --steps 1000
    python scripts/qlsa/run_qlsa_experiment.py --densities results/qlsa/densities_iso17_gfn2.npz --label iso17_gfn2
    python scripts/qlsa/run_windowed_experiment.py --densities results/qlsa/densities_iso17_gfn2.npz --label iso17_gfn2

repeats the QLSA and re-linearization experiments with GFN2-xTB densities.

Each step is a fully converged SCC calculation (tblite has no XL-BOMD integrator), restarted
from the previous step's wavefunction. The QLSA experiments only need the SCF densities along a
trajectory, so this is what they consume.

Outputs:
    results/md/<label>_gfn2.xyz             trajectory (Angstrom), every --save-every steps
    results/md/<label>_gfn2_energies.npz    time (fs), potential, kinetic, total energy (eV), temperature (K)
    results/qlsa/densities_<label>_gfn2.npz D_frames (n_steps+1, norb, norb), Dstar = D_frames[0], Z, method

Requires: pip install tblite   (wheels for Linux and macOS; GFN2-xTB needs no parameter files)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
HARTREE_TO_EV = 27.211386245988
FORCE_AU_TO_EV_A = HARTREE_TO_EV * ANGSTROM_TO_BOHR      # Hartree/Bohr -> eV/Angstrom
ACC = 9.648533212e-3            # (eV/Angstrom)/amu -> Angstrom/fs^2
KE_UNIT = 103.6426965           # amu Angstrom^2/fs^2 -> eV
KB_EV = 8.617333262e-5          # eV/K
MASSES = {1: 1.00794, 6: 12.0107, 7: 14.0067, 8: 15.9994}
SYMBOL_TO_Z = {"H": 1, "C": 6, "N": 7, "O": 8}
Z_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_Z.items()}
INPUTS = {"iso17": "inputs/iso17_mol.xyz", "md17": "inputs/md17_mol.xyz", "water": "inputs/water.xyz"}

p = argparse.ArgumentParser()
p.add_argument("system", choices=sorted(INPUTS))
p.add_argument("--steps", type=int, default=1000, help="MD steps (1000 x 0.25 fs = 250 fs)")
p.add_argument("--dt", type=float, default=0.25, help="time step in fs")
p.add_argument("--temperature", type=float, default=500.0, help="initial temperature in K")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--method", default="GFN2-xTB")
p.add_argument("--accuracy", type=float, default=0.01,
               help="tblite accuracy (smaller is tighter; 0.01 converges densities to ~1e-7)")
p.add_argument("--save-every", type=int, default=1, help="write a trajectory frame every n steps")
args = p.parse_args()

from tblite.interface import Calculator  # noqa: E402  (import after argparse so -h works without tblite)


def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0])
    rows = [l.split() for l in lines[2:2 + n]]
    return np.array([SYMBOL_TO_Z[r[0]] for r in rows]), np.array([[float(v) for v in r[1:4]] for r in rows])


Z, R = read_xyz(ROOT / INPUTS[args.system])
m = np.array([MASSES[int(z)] for z in Z])
n_atoms = len(Z)

calc = Calculator(args.method, Z, R * ANGSTROM_TO_BOHR)
calc.set("accuracy", args.accuracy)
calc.set("verbosity", 0)
res = None


def evaluate(R):
    """Energy (eV), forces (eV/Angstrom) and density matrix at positions R (Angstrom)."""
    global calc, res
    try:
        calc.update(R * ANGSTROM_TO_BOHR)
    except AttributeError:      # older tblite: rebuild the calculator
        calc = Calculator(args.method, Z, R * ANGSTROM_TO_BOHR)
        calc.set("accuracy", args.accuracy)
        calc.set("verbosity", 0)
        res = None
    res = calc.singlepoint(res) if res is not None else calc.singlepoint()
    E = float(res.get("energy")) * HARTREE_TO_EV
    F = -np.asarray(res.get("gradient")) * FORCE_AU_TO_EV_A
    P = np.asarray(res.get("density-matrix"), dtype=float)
    if P.ndim == 3:             # spin-resolved: total density
        P = P.sum(axis=0)
    return E, F, P


# Maxwell-Boltzmann velocities at the target temperature, no net momentum, rescaled exactly
rng = np.random.default_rng(args.seed)
V = rng.normal(size=(n_atoms, 3)) * np.sqrt(KB_EV * args.temperature / (m[:, None] * KE_UNIT))
V -= (m[:, None] * V).sum(0) / m.sum()
dof = 3 * n_atoms - 3


def kinetic(V):
    return 0.5 * KE_UNIT * float((m[:, None] * V ** 2).sum())


V *= np.sqrt(args.temperature / (2 * kinetic(V) / (dof * KB_EV)))

label = f"{args.system}_gfn2"
(ROOT / "results/md").mkdir(parents=True, exist_ok=True)
(ROOT / "results/qlsa").mkdir(parents=True, exist_ok=True)
xyz = open(ROOT / f"results/md/{label}.xyz", "w")


def write_frame(step, R, E):
    xyz.write(f"{n_atoms}\nstep={step} t_fs={step * args.dt:.3f} E_pot_eV={E:.8f}\n")
    for z, r in zip(Z, R):
        xyz.write(f"{Z_TO_SYMBOL[int(z)]:2s} {r[0]:14.8f} {r[1]:14.8f} {r[2]:14.8f}\n")


E, F, P = evaluate(R)
dens = [P]
log = [(0.0, E, kinetic(V))]
write_frame(0, R, E)
t0 = time.time()
for step in range(1, args.steps + 1):          # velocity Verlet
    V += 0.5 * args.dt * ACC * F / m[:, None]
    R = R + args.dt * V
    E, F, P = evaluate(R)
    V += 0.5 * args.dt * ACC * F / m[:, None]
    dens.append(P)
    log.append((step * args.dt, E, kinetic(V)))
    if step % args.save_every == 0:
        write_frame(step, R, E)
    if step % max(1, args.steps // 10) == 0:
        Etot = np.array([l[1] + l[2] for l in log])
        print(f"step {step:6d}  E_tot drift {(Etot[-1] - Etot[0]) * 23.0605:+.4f} kcal/mol  "
              f"T {2 * log[-1][2] / (dof * KB_EV):7.1f} K  ({time.time() - t0:.0f} s)")
xyz.close()

t, Epot, Ekin = map(np.array, zip(*log))
np.savez(ROOT / f"results/md/{label}_energies.npz", t_fs=t, E_pot_eV=Epot, E_kin_eV=Ekin,
         E_tot_eV=Epot + Ekin, T_K=2 * Ekin / (dof * KB_EV), Z=Z, method=args.method,
         dt_fs=args.dt, seed=args.seed, accuracy=args.accuracy)
D = np.stack(dens)
np.savez_compressed(ROOT / f"results/qlsa/densities_{label}.npz", Dstar=D[0], D_frames=D, Z=Z,
                    frames=np.arange(len(D)), method=f"{args.method} (tblite)")
print(f"wrote results/md/{label}.xyz, results/md/{label}_energies.npz, "
      f"results/qlsa/densities_{label}.npz  (norb = {D.shape[1]}, {len(D)} frames)")
