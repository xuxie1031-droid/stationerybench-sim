"""Repository and working-directory paths, all overridable by environment variables.

`source scripts/env.sh` sets them (and PYTHONPATH) once per shell.  Every script in this repository
imports this module instead of hard-coding paths, so the code runs from any checkout location.

  GYRE_WORK            root for everything the pipeline writes (default <repo>/work)
  GYRE_DATA            synthetic demonstration data     (default $GYRE_WORK/sim_data)
  GYRE_EVAL_DIR        closed-loop eval runs + results.json           (default $GYRE_WORK/eval)
  GYRE_SCRATCH         host-local scratch (videos, temp files)         (default $GYRE_WORK/scratch)
  GYRE_TRAIN_WORK      training runs/, logs/                           (default $GYRE_WORK/train)
  LEROBOT_DATA_ROOT    converted LeRobot datasets (sim/<name>, real/<name>)  (default $GYRE_TRAIN_WORK/lerobot_data)
  MOLMOACT2_DIR        checkout of allenai/molmoact2 at the pinned commit (default <repo>/third_party/molmoact2)
  SIM_PY               python of the MolmoAct2 server/sim venv         (default $MOLMOACT2_DIR/.venv/bin/python)
  MOLMOACT2_BASE_DIR   local snapshot of allenai/MolmoAct2-BimanualYAM (default $GYRE_WORK/models/MolmoAct2-BimanualYAM)
  MS_ASSET_DIR         ManiSkill asset dir                              (default $GYRE_WORK/maniskill_data)
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("GYRE_WORK") or os.path.join(ROOT, "work")
DATA = os.environ.get("GYRE_DATA") or os.path.join(WORK, "sim_data")
AB = os.environ.get("GYRE_EVAL_DIR") or os.path.join(WORK, "eval")
SCRATCH = os.environ.get("GYRE_SCRATCH") or os.path.join(WORK, "scratch")
TRAIN_WORK = os.environ.get("GYRE_TRAIN_WORK") or os.path.join(WORK, "train")
LEROBOT_DATA = os.environ.get("LEROBOT_DATA_ROOT") or os.path.join(TRAIN_WORK, "lerobot_data")
MOLMOACT2 = os.environ.get("MOLMOACT2_DIR") or os.path.join(ROOT, "third_party", "molmoact2")
SIM_PY = os.environ.get("SIM_PY") or os.path.join(MOLMOACT2, ".venv", "bin", "python")
BASE_MODEL_DIR = os.environ.get("MOLMOACT2_BASE_DIR") or os.path.join(WORK, "models", "MolmoAct2-BimanualYAM")
MS_ASSET_DIR = os.environ.get("MS_ASSET_DIR") or os.path.join(WORK, "maniskill_data")

# code locations
ENV_EFB = os.path.join(ROOT, "envs", "eraser_from_box")
ENV_RH = os.path.join(ROOT, "envs", "ruler_handover")
VISUAL = os.path.join(ROOT, "envs", "visual")
EVAL = os.path.join(ROOT, "eval")
YAM_FT = os.path.join(ROOT, "train", "yam_ft")

# task env modules (eraser_task, ruler_handover_env, visual) and MolmoAct2's sim_eval package.  Appended, so a
# script's own directory (sys.path[0]) still wins for sibling modules with common names (e.g. a local helper).
for _p in (ENV_EFB, ENV_RH, VISUAL, MOLMOACT2):
    if _p not in sys.path:
        sys.path.append(_p)
