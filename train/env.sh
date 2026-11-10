# source me: training env for MolmoAct2 fine-tuning (transformers 5.x, torch 2.10 / cu128) -- a separate venv from the
# server/sim venv (transformers 4.57).  Create it with train/setup_env.sh.  Sources scripts/env.sh for the repo paths.
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../scripts/env.sh"
source "$TRAIN_VENV/bin/activate"
export E=$MOLMOACT2_DIR/experiments                 # MolmoAct2 training code (olmo + launch_scripts), unmodified
export YF=$GYRE_ROOT/train/yam_ft                   # this repo's fine-tuning wrapper + launchers
# optional: FFmpeg with SVT-AV1 for the converter / torchcodec, if the system one lacks it
[ -n "${FFMPEG_ROOT:-}" ] && export LD_LIBRARY_PATH=$FFMPEG_ROOT/lib:${LD_LIBRARY_PATH:-} PATH=$FFMPEG_ROOT/bin:$PATH
export PYTHONPATH=$E:$E/lerobot/src:$PYTHONPATH
export LEROBOT_VIDEO_BACKEND=torchcodec
export HF_HOME=${HF_HOME:-$GYRE_WORK/hf_home}
export HF_LEROBOT_HOME=$LEROBOT_DATA_ROOT
export WANDB_MODE=${WANDB_MODE:-disabled}
export TOKENIZERS_PARALLELISM=false
export MOLMO_DATA_DIR=${MOLMO_DATA_DIR:-$GYRE_TRAIN_WORK/molmo_data}
export WANDB_PROJECT=${WANDB_PROJECT:-yam_ft} WANDB_ENTITY=${WANDB_ENTITY:-local}
mkdir -p "$GYRE_TRAIN_WORK/logs" "$GYRE_TRAIN_WORK/runs"
