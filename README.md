# QA-XL-BOMD

Code and data for the paper *Quantum-Accelerated Extended Lagrangian Molecular Dynamics*.

The repository has three parts. The first is a pair of extended Lagrangian Born–Oppenheimer MD
(XL-BOMD) trajectories, one for an ISO17 isomer (C₇O₂H₁₀) and one for MD17 aspirin (C₉H₈O₄),
computed with the AM1 Hamiltonian in [PySEQM](https://github.com/lanl/PYSEQM), along with the
analysis that produces the RMSD, RMSF, energy-conservation and energy-surface figures. The second
is the QLSA experiment: the linearized XL-BOMD electronic update is written as one linear system
over all time steps and solved twice, step by step and with an emulated QSVT quantum linear
solver, using AM1 densities taken from those trajectories. The third compares AM1 energies and
forces with the DFT labels of ISO17 and MD17.

Precomputed results are included, so the figures and the notebook work as soon as the repository
is cloned.

## Quick start

```bash
git clone https://github.com/asalamatov/QA-XL-BOMD.git
cd QA-XL-BOMD
./setup.sh                    # virtual env, PyTorch, packages, PySEQM (lanl/PYSEQM @ 9163fad)
source .venv/bin/activate
jupyter notebook notebooks/visualize_results.ipynb
```

`setup.sh` installs the default PyTorch build. For CUDA, pass the matching index URL, for example
`TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 ./setup.sh`. The notebook only needs
NumPy and Matplotlib.

## Reproducing the results

```bash
./run_all.sh check            # 500-step water run to check the PySEQM install
./run_all.sh md               # rerun both XL-BOMD trajectories (overwrites results/md)
./run_all.sh analysis         # observables and figures from results/md
./run_all.sh qlsa             # AM1 densities, QLSA experiment, figures
./run_all.sh qlsa-long        # windows up to T = 4000, synthetic input
./run_all.sh windowed         # periodic re-linearization (1000-frame density export)
./run_all.sh windowed-eq24    # optional: the same with the dissipative two-step update (after windowed)
./download_data.sh            # ISO17 (800 MB) and seven MD17 molecules into data/
./run_all.sh mae              # AM1 vs DFT errors; N_MAE=100 for a quick test
./run_all.sh xtb              # GFN2-xTB: MD with densities, QLSA, windowed, MAE (needs tblite)
./run_all.sh all              # check, md, analysis, qlsa, qlsa-long, windowed, mae
```

Without arguments, `./run_all.sh` runs `analysis` and `qlsa` on the included trajectories.

| stage | needs | time |
|---|---|---|
| analysis, notebook | NumPy, SciPy, h5py, Matplotlib | seconds |
| md | PySEQM | minutes on a GPU, longer on a CPU |
| qlsa | PySEQM for the density export, then NumPy and SciPy | a few minutes |
| qlsa-long | NumPy and SciPy (synthetic input) | about 10 minutes |
| windowed | PySEQM for the 1000-frame export, then NumPy and SciPy | 10-20 minutes |
| xtb | tblite (`pip install tblite`) and, for the MAE part, the downloaded datasets | about an hour on a CPU |
| mae | PySEQM and the downloaded datasets | minutes per 1000 structures |

## Results in brief

All numbers below are computed from the files in `results/`; the notebook prints them.

| quantity | ISO17 isomer | MD17 aspirin |
|---|---|---|
| RMSD (Å) | 0.14–0.37 | 0.32–0.76 |
| RMSF, heavy atoms / H, max (Å) | 0.16 / 0.29 | 0.64 / 1.23 |
| total-energy deviation (kcal/mol) | −0.070 to +0.013 | −0.075 to +0.041 |
| QSVT vs classical solve (rel. Frobenius) | 1.1e-13 to 1.6e-11 | 1.1e-13 to 1.6e-11 |
| drift of one fixed linearization, 25 / 200 steps | 3.2 % / 10.6 % | 4.1 % / 13.9 % |
| re-linearized every 5 / 10 steps, mean deviation over 250 fs | 0.9 % / 1.9 % | 0.9 % / 1.9 % |

Effective condition number of the time-grid system at γ = 0.5 (it depends only on the recurrence):

| recurrence | T = 25 | T = 400 | T = 4000 |
|---|---|---|---|
| two-step, α = 0 (no dissipation) | 41 | 592 | 5885 |
| two-step, α = 0.5 (Niklasson Eq. 24) | 25 | 29 | 29 |
| PySEQM k = 6 (α = 0.0055) | 69 | 719 | 1246 |

AM1 against the DFT labels (1000 test structures per set): energy MAE 13.5 and 15.2 kcal/mol on
the two ISO17 test splits and 1.6–4.7 kcal/mol on MD17; force MAE 7.0–16.8 kcal/mol/Å. That is
15–135 times the error of the best published machine-learned potentials on the same sets.

## Where each figure comes from

| figure | data | scripts |
|---|---|---|
| `figures/md/fig_rmsd_{iso17,md17}` | `results/observables/rmsd_*.npz` | `extract_trajectory.py`, `make_md_figures.py` |
| `figures/md/fig_rmsf_{iso17,md17}` | `results/observables/rmsf_*.npz` | same |
| `figures/md/fig_energy_{iso17,md17}` | `results/observables/energy_*.npz` | same |
| `figures/md/fig_energy_surface_{iso17,md17}` | `results/observables/pmf_*.npz` | same |
| `figures/qlsa/fig_qlsa_{iso17,md17}_gamma0.5` | `results/qlsa/qlsa_*.json` | `run_qlsa_experiment.py`, `make_qlsa_figures.py` |
| `figures/qlsa/fig_qlsa_long_windows_synthetic_gamma0.5` | `results/qlsa/qlsa_long_windows_synthetic.json` | same, `qlsa-long` stage |
| `figures/qlsa/fig_windowed` | `results/qlsa/windowed_{iso17,md17}.json` | `run_windowed_experiment.py`, `make_windowed_figure.py` |

## Methods

### XL-BOMD runs

`scripts/md/run_xlbomd.py` runs PySEQM's `XL_BOMD` (Niklasson's scheme with k = 6) with AM1, an
SCF tolerance of 10⁻⁶, a 500 K initial temperature, a 0.25 fs time step and no thermostat. The
ISO17 run is 4000 steps (1 ps) and the aspirin run 10000 steps (2.5 ps). Starting geometries are
in `inputs/`.

### Observables

`scripts/analysis/extract_trajectory.py` computes Kabsch-aligned RMSD, per-atom RMSF, total-energy
drift, and potential energy against RMSD (the 5th percentile in each RMSD bin, smoothed). For
ISO17 it uses every frame, with frame 0 as the reference. For aspirin it discards the first 2000
steps (0.5 ps) and uses the mean geometry. These settings are saved in each `.npz` file. Energies
are stored in eV and plotted in kcal/mol.

### QLSA experiment

With the surrogate D(P) = (1−γ)D* + γP, the XL-BOMD update becomes a constant-coefficient
recurrence, and a window of T steps becomes a lower-triangular system M x = b with
M = M_time ⊗ I. The experiment (`qlsa/`, `scripts/qlsa/`) solves this system step by step and
with an emulated QSVT solve (the Childs–Kothari–Somma polynomial at ε = 10⁻⁸). It does this for
the manuscript's two-step update (Niklasson et al. 2009, Eq. 24,
P_{n+1} = 2P_n − P_{n−1} + (2−α)(D_n−P_n) − α(D_{n−1}−P_{n−1})) at α = 0 and α = 0.5 and for PySEQM's k = 6 scheme, at
γ = 0.25, 0.5 and 0.75, and records the condition number, the polynomial degree (the number of
block-encoding queries), the deviation from the step-by-step solution, and how far the surrogate
trajectory drifts from the AM1 SCF densities along the real trajectory.

The emulation applies the QSVT polynomial to the singular values of the block-encoded matrix. It
does not model gate noise, the success probability, or readout sampling.

### MAE evaluation

`scripts/evaluation/evaluate_mae.py` runs AM1 single points on the ISO17 test splits and on MD17
frames and compares them with the datasets' PBE+TS energies and forces. AM1 and PBE total
energies have different zero points, so one constant energy offset per dataset is fitted on a
calibration sample that does not overlap the test sample. With 1000 test structures per split,
AM1 gives energy MAEs of 13–15 kcal/mol on ISO17 and 1.6–4.7 kcal/mol on MD17, and force MAEs
of 7–17 kcal/mol/Å (`results/mae/`).

### GFN2-xTB runs

The QLSA formulation only needs SCF density matrices along a trajectory, so it is not tied to
AM1. `scripts/md/run_bomd_xtb.py` runs Born–Oppenheimer MD with GFN2-xTB through
[tblite](https://github.com/tblite/tblite) from the same starting geometries and with the same
settings (NVE, 500 K, 0.25 fs, seed 0). Each step is a converged SCC calculation restarted from
the previous step, because tblite has no XL-BOMD integrator. The script writes the trajectory, the
energies and the density matrix at every step (`results/qlsa/densities_<system>_gfn2.npz`), and
`./run_all.sh xtb` then runs the QLSA and re-linearization experiments on those densities and the
MAE evaluation with `--method GFN2-xTB`. MAE results go to `results/mae/<dataset>_gfn2.json`,
next to the AM1 files, and both methods are scored on the same test structures.

## Repository layout

```
inputs/                 starting geometries (ISO17 isomer, MD17 aspirin, water)
qlsa/                   linear recurrences and the QSVT emulator
scripts/md/             run_xlbomd.py (AM1, PySEQM), run_bomd_xtb.py (GFN2-xTB, tblite)
scripts/analysis/       extract_trajectory.py, make_md_figures.py
scripts/qlsa/           export_densities.py, run_qlsa_experiment.py, run_windowed_experiment.py,
                        make_qlsa_figures.py, make_windowed_figure.py
scripts/evaluation/     evaluate_mae.py
results/md/             trajectories (.xyz) and energies (.h5)
results/observables/    RMSD, RMSF, energy and energy-surface data (.npz)
results/qlsa/           QLSA results (.json); density exports are written here
results/mae/            MAE results (iso17.json, md17.json)
figures/                PDF and PNG figures
notebooks/              visualize_results.ipynb
data/                   downloaded datasets (not tracked)
requirements.txt        packages; requirements-lock.txt lists the environment of the original runs
```

## Reproducibility notes

PySEQM draws random initial velocities. `run_xlbomd.py` now fixes the seed (default 0), but the
included trajectories were generated before that change, so `./run_all.sh md` produces
statistically equivalent trajectories rather than identical ones.

The ISO17 RMSF data were recomputed from the included trajectory with the settings above. They
differ from an earlier analysis run by at most 0.007 Å.

The single-window QLSA experiment used 200 exported frames per system, so its surrogate errors
are only available for windows up to T = 200. The re-linearization experiment uses 1000 frames.

ISO17 is available from <https://quantum-machine.org/datasets/> and MD17 from
<https://www.sgdml.org/#datasets>.

## Citation

If you use this code, please cite the paper (reference to follow on publication), PySEQM, and the
ISO17 and MD17 dataset papers.
