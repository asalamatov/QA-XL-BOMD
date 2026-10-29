#!/usr/bin/env bash
# Run the pipeline. Stages run in the order given.
#
#   ./run_all.sh                 # analysis + qlsa (uses the trajectories in results/md)
#   ./run_all.sh check           # 500-step water run: checks the PySEQM install
#   ./run_all.sh md              # rerun the XL-BOMD trajectories (minutes on GPU, longer on CPU)
#   ./run_all.sh analysis        # observables and figures from results/md
#   ./run_all.sh qlsa            # AM1 densities + QSVT linear-solver experiment + figures
#   ./run_all.sh windowed        # 1000-frame densities + periodic re-linearization experiment
#   ./run_all.sh mae             # AM1 vs DFT MAEs (run ./download_data.sh first)
#   ./run_all.sh windowed-eq24   # optional: windowed run with the dissipative two-step update
#   ./run_all.sh xtb             # GFN2-xTB: MD with densities, QLSA + windowed experiments, MAE (needs tblite)
#   ./run_all.sh all             # check md analysis qlsa qlsa-long windowed mae
#
# Set N_MAE to change the number of test structures per split (default 1000).
set -euo pipefail
cd "$(dirname "$0")"
if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

stages=("$@")
[[ ${#stages[@]} -eq 0 ]] && stages=(analysis qlsa)
[[ " ${stages[*]} " == *" all "* ]] && stages=(check md analysis qlsa qlsa-long windowed mae)

for stage in "${stages[@]}"; do
  echo "================ $stage ================"
  case "$stage" in
    check)
      python scripts/md/run_xlbomd.py water --out results/md/check
      ;;
    md)
      echo "Note: this overwrites results/md/{iso17,md17}_run.0.* (seed 0)."
      python scripts/md/run_xlbomd.py iso17
      python scripts/md/run_xlbomd.py md17
      ;;
    analysis)
      python scripts/analysis/extract_trajectory.py --label iso17
      python scripts/analysis/extract_trajectory.py --label md17 --discard 2000 --mean-ref
      python scripts/analysis/make_md_figures.py
      ;;
    qlsa)
      python scripts/qlsa/export_densities.py --xyz results/md/iso17_run.0.xyz --label iso17 --frames 200
      python scripts/qlsa/export_densities.py --xyz results/md/md17_run.0.xyz --label md17 --frames 200
      python scripts/qlsa/run_qlsa_experiment.py --densities results/qlsa/densities_iso17.npz --label iso17
      python scripts/qlsa/run_qlsa_experiment.py --densities results/qlsa/densities_md17.npz --label md17
      python scripts/qlsa/make_qlsa_figures.py results/qlsa/qlsa_iso17.json
      python scripts/qlsa/make_qlsa_figures.py results/qlsa/qlsa_md17.json
      ;;
    windowed)
      python scripts/qlsa/export_densities.py --xyz results/md/iso17_run.0.xyz --label iso17 --frames 1000
      python scripts/qlsa/export_densities.py --xyz results/md/md17_run.0.xyz --label md17 --frames 1000
      python scripts/qlsa/run_windowed_experiment.py --densities results/qlsa/densities_iso17.npz --label iso17
      python scripts/qlsa/run_windowed_experiment.py --densities results/qlsa/densities_md17.npz --label md17
      python scripts/qlsa/make_windowed_figure.py results/qlsa/windowed_iso17.json results/qlsa/windowed_md17.json
      ;;
    windowed-eq24)
      # uses the 1000-frame exports written by the windowed stage
      python scripts/qlsa/run_windowed_experiment.py --densities results/qlsa/densities_iso17.npz --label iso17_eq24 --scheme paper --alpha 0.5
      python scripts/qlsa/run_windowed_experiment.py --densities results/qlsa/densities_md17.npz --label md17_eq24 --scheme paper --alpha 0.5
      ;;
    xtb)
      # GFN2-xTB (tblite): 1000-step BOMD per system with densities, then the same experiments
      for sys in iso17 md17; do
        python scripts/md/run_bomd_xtb.py "$sys" --steps 1000
        python scripts/qlsa/run_qlsa_experiment.py --densities "results/qlsa/densities_${sys}_gfn2.npz" --label "${sys}_gfn2"
        python scripts/qlsa/run_windowed_experiment.py --densities "results/qlsa/densities_${sys}_gfn2.npz" --label "${sys}_gfn2"
        python scripts/qlsa/make_qlsa_figures.py "results/qlsa/qlsa_${sys}_gfn2.json"
      done
      python scripts/qlsa/make_windowed_figure.py results/qlsa/windowed_iso17_gfn2.json results/qlsa/windowed_md17_gfn2.json \
        --out figures/qlsa/fig_windowed_gfn2
      python scripts/evaluation/evaluate_mae.py iso17 --method GFN2-xTB --n "${N_MAE:-1000}"
      python scripts/evaluation/evaluate_mae.py md17 --method GFN2-xTB --n "${N_MAE:-1000}"
      ;;
    qlsa-long)
      python scripts/qlsa/run_qlsa_experiment.py --synthetic 4 --label long_windows_synthetic \
        --T 1000 2000 4000 --gamma 0.5 --paper-alpha 0.0 0.5
      python scripts/qlsa/make_qlsa_figures.py results/qlsa/qlsa_long_windows_synthetic.json
      ;;
    mae)
      python scripts/evaluation/evaluate_mae.py iso17 --n "${N_MAE:-1000}"
      python scripts/evaluation/evaluate_mae.py md17 --n "${N_MAE:-1000}"
      ;;
    *)
      echo "unknown stage: $stage"; exit 1
      ;;
  esac
done
echo "Done. Figures: figures/md, figures/qlsa. Notebook: notebooks/visualize_results.ipynb"
