# source me:  source scripts/env.sh
# Paths for this repository.  Override any of them before sourcing (see gyre_paths.py for the meaning of each).
export GYRE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export GYRE_WORK="${GYRE_WORK:-$GYRE_ROOT/work}"
export GYRE_DATA="${GYRE_DATA:-$GYRE_WORK/sim_data}"
export GYRE_EVAL_DIR="${GYRE_EVAL_DIR:-$GYRE_WORK/eval}"
export GYRE_SCRATCH="${GYRE_SCRATCH:-$GYRE_WORK/scratch}"
export GYRE_TRAIN_WORK="${GYRE_TRAIN_WORK:-$GYRE_WORK/train}"
export LEROBOT_DATA_ROOT="${LEROBOT_DATA_ROOT:-$GYRE_TRAIN_WORK/lerobot_data}"
export MOLMOACT2_DIR="${MOLMOACT2_DIR:-$GYRE_ROOT/third_party/molmoact2}"
export SIM_PY="${SIM_PY:-$MOLMOACT2_DIR/.venv/bin/python}"
export TRAIN_VENV="${TRAIN_VENV:-$GYRE_WORK/venv-train}"
export MOLMOACT2_BASE_DIR="${MOLMOACT2_BASE_DIR:-$GYRE_WORK/models/MolmoAct2-BimanualYAM}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-$GYRE_WORK/maniskill_data}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# repo root (gyre_paths), task envs, visual module and MolmoAct2 (sim_eval) importable from anywhere
export PYTHONPATH="$GYRE_ROOT:$GYRE_ROOT/envs/eraser_from_box:$GYRE_ROOT/envs/ruler_handover:$GYRE_ROOT/envs/visual:$MOLMOACT2_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$GYRE_WORK"
