#!/usr/bin/env bash
# Environment for stage 2 (SAM3, synth/segment.py) and stage 3 (MoGe-2, synth/calibrate.py).
# Needs: uv, git, an NVIDIA GPU with a CUDA 12.8 capable driver. Runs on Ubuntu and on Windows (Git Bash).
# Usage: bash setup.sh   then, once: <venv python dir>/hf auth login   (account with access to facebook/sam3)
set -euo pipefail
cd "$(dirname "$0")"

SAM3_COMMIT=2345a4ad109ac29c569da749c91d84f10dc08c40
MOGE_COMMIT=74fbce054ebed49800de42d0ad0e83495065719a

clone() {  # clone <url> <dir> <commit>
  [ -d "$2" ] || git clone -q "$1" "$2"
  git -C "$2" checkout -q "$3"
}
clone https://github.com/facebookresearch/sam3.git vendor/sam3 "$SAM3_COMMIT"
clone https://github.com/microsoft/MoGe.git vendor/MoGe "$MOGE_COMMIT"

[ -d .venv ] || uv venv --python 3.12 .venv
PY=.venv/bin/python
[ -x "$PY" ] || PY=.venv/Scripts/python.exe  # Windows

uv pip install --python "$PY" torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" --override overrides.txt -e vendor/sam3 -e vendor/MoGe -r requirements.txt

"$PY" -c "
import torch
from sam3.model_builder import build_sam3_image_model
from moge.model.v2 import MoGeModel
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
echo "Done. If not logged in yet: $(dirname "$PY")/hf auth login"
echo "Run stage 2: $PY -m synth.segment    stage 3: $PY -m synth.calibrate"
