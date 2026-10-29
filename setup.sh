#!/usr/bin/env bash
# Create a virtual environment and install everything needed, including
# PySEQM from https://github.com/lanl/PYSEQM at a fixed commit.
#
#   ./setup.sh                      # CPU or default PyTorch build
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 ./setup.sh   # CUDA 12.4 build
#
# Environment variables: PYTHON (default python3), VENV (default .venv).
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
VENV=${VENV:-.venv}
PYSEQM_URL=https://github.com/lanl/PYSEQM.git
PYSEQM_COMMIT=9163fad70c25b4d5a56be5f530fd9b83c696f4b6

echo "==> Python: $($PYTHON --version)"
"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip

echo "==> PyTorch"
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  pip install torch --index-url "$TORCH_INDEX_URL"
else
  pip install torch
fi

echo "==> Python packages"
pip install -r requirements.txt

echo "==> tblite (GFN2-xTB; optional, used by ./run_all.sh xtb)"
pip install tblite || echo "WARNING: tblite did not install; the xtb stage will not run (AM1 stages are unaffected)"

echo "==> PySEQM @ ${PYSEQM_COMMIT:0:7}"
if [[ ! -d PYSEQM/.git ]]; then
  git clone "$PYSEQM_URL" PYSEQM
fi
git -C PYSEQM fetch --quiet origin
git -C PYSEQM checkout --quiet "$PYSEQM_COMMIT"
pip install ./PYSEQM

python - <<'EOF'
import torch, seqm
print(f"OK: PySEQM imported; torch {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
EOF
echo "==> Done. Activate with: source $VENV/bin/activate"
