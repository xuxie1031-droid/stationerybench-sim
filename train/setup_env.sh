#!/bin/bash
# Create the training venv ($TRAIN_VENV, default $GYRE_WORK/venv-train) on top of the pinned MolmoAct2 checkout.
# Exact package versions of the venv used for every run: requirements/train-freeze.txt (see environment.md).
#   bash train/setup_env.sh
set -ex
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"
E=$MOLMOACT2_DIR/experiments
uv venv --python 3.12 "$TRAIN_VENV"
source "$TRAIN_VENV/bin/activate"
uv pip install --no-config --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match --no-deps -e $E/lerobot
uv pip install --no-config --index-url https://pypi.org/simple "datasets>=4.0.0,<5.0.0" "diffusers>=0.27.2,<0.36.0" "huggingface-hub>=1.0.0,<2.0.0" "accelerate>=1.10.0,<2.0.0" "numpy>=2.0.0,<2.3.0" "setuptools>=71.0.0,<81.0.0" "einops>=0.8.0,<0.9.0" "opencv-python-headless>=4.9.0,<4.13.0" "av>=15.0.0,<16.0.0" "jsonlines>=4.0.0,<5.0.0" "pyserial>=3.5,<4.0" "wandb>=0.24.0,<0.25.0" "draccus==0.10.0" "gymnasium>=1.1.1,<2.0.0" "rerun-sdk>=0.24.0,<0.27.0" "deepdiff>=7.0.1,<9.0.0" "imageio[ffmpeg]>=2.34.0,<3.0.0" "termcolor>=2.4.0,<4.0.0" "transformers>=5.3.0,<6.0.0"
uv pip install --no-config --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps torch==2.10.0 torchvision torchaudio
uv pip install --no-config --index-url https://pypi.org/simple --force-reinstall --no-deps torchcodec==0.10.0
uv pip install --no-config --index-url https://pypi.org/simple -e $E --no-deps
uv pip install --no-config --index-url https://pypi.org/simple wandb safetensors torchmetrics msgspec scikit-learn bitsandbytes peft
python -c "import torch, transformers, lerobot, peft, torchvision; print(torch.__version__, transformers.__version__, peft.__version__, torchvision.__version__)"
echo SETUP_DONE
