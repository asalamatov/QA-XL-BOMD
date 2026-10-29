#!/usr/bin/env bash
# Download the benchmark datasets into data/ (about 1 GB in total).
# They are only needed for the MAE evaluation; the MD runs start from the
# geometries in inputs/.
#
#   ./download_data.sh            # ISO17 and the seven MD17 molecules
#   ./download_data.sh iso17      # ISO17 only
#   ./download_data.sh md17       # MD17 only
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/md17

fetch() {  # fetch <url> <destination>
  if [[ -s "$2" ]]; then echo "    exists: $2"; return; fi
  echo "    downloading $(basename "$2")"
  curl -fL --retry 3 --retry-delay 5 -o "$2.part" "$1"
  mv "$2.part" "$2"
}

targets=("$@")

if [[ ${#targets[@]} -eq 0 || " ${targets[*]} " == *" iso17 "* ]]; then
  echo "==> ISO17 (quantum-machine.org, 800 MB)"
  fetch https://quantum-machine.org/datasets/iso17.tar.gz data/iso17.tar.gz
  if [[ ! -f data/iso17/test_other.db ]]; then
    tar -xzf data/iso17.tar.gz -C data
  fi
  ls data/iso17/*.db >/dev/null
fi

if [[ ${#targets[@]} -eq 0 || " ${targets[*]} " == *" md17 "* ]]; then
  echo "==> MD17 (sgdml.org)"
  for mol in aspirin ethanol malonaldehyde naphthalene salicylic toluene uracil; do
    fetch "https://sgdml.org/secure_proxy.php?file=data/npz/md17_${mol}.npz" "data/md17/md17_${mol}.npz"
  done
fi
echo "==> Done"
