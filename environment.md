# Environment

Two Python environments, exactly as used for every number in this repository (Linux x86-64, CUDA 12.8 driver,
NVIDIA L40S 48 GB):

| venv | used for | key versions | full pin list |
|---|---|---|---|
| **sim / server** (`$MOLMOACT2_DIR/.venv`, created by `uv sync` in the MolmoAct2 checkout) | simulation, rendering, evaluation, policy server | Python 3.12.3, torch 2.11.0+cu128, transformers 4.57.6, mani-skill 3.0.1, sapien 3.0.3, numpy 2.5.4, gymnasium 1.4.0 | [requirements/sim-freeze.txt](requirements/sim-freeze.txt) |
| **train** (`$TRAIN_VENV`, created by [train/setup_env.sh](train/setup_env.sh)) | fine-tuning, checkpoint export | Python 3.12.3, torch 2.10.0+cu128, transformers 5.22.1, peft 0.18.1, accelerate 1.15.0, torchcodec 0.10.0, av 15.1.0, numpy 2.2.6, MolmoAct2 `experiments` + its bundled `lerobot` (editable) | [requirements/train-freeze.txt](requirements/train-freeze.txt) |

The two venvs differ on purpose: the released checkpoint's remote code and the stock server expect transformers 4.57,
while MolmoAct2's training code needs transformers 5.x.

## MolmoAct2 (dependency, not vendored)

```bash
git clone https://github.com/allenai/molmoact2 third_party/molmoact2
git -C third_party/molmoact2 checkout 6070080a20321b4f498ab30f28e1d09ac465edb7   # 2026-10-01, "Merge pull request #34"
git -C third_party/molmoact2 apply ../../patches/molmoact2/0001-host-server-accept-local-snapshot-dir.patch
```

Our changes to MolmoAct2, complete list:

1. `patches/molmoact2/0001-host-server-accept-local-snapshot-dir.patch` — `examples/yam/host_server_yam.py` accepts a
   local snapshot directory as `--repo-id` (upstream always calls `snapshot_download`). One line.
2. Runtime wrappers, no source edits:
   * `eval/host_server_yam_noclip.py` imports the stock server and, after the model is loaded, removes the single
     `np.clip(arr, -1.0, 1.0)` line from the remote-code `_FeatureNormalizer.unnormalize` (asserted to occur exactly
     once). Required for our fine-tuned weights, whose sim action targets were trained without that clip.
   * `eval/run_eval_efb.py`, `eval/run_eval_rh.py` wrap `sim_eval.run_eval` (register the env, add the verbatim
     instruction, log the rubric stage per step).
   * `train/yam_ft/train_yam_ft.py` wraps `experiments/launch_scripts/train_lerobot.py` (mixture registration,
     per-dataset clip rules, EMA, WSD schedule, gradient-sync-once-per-step for multi-node without RDMA).
3. The robot assets of the `bimanual_yam` agent come from MolmoAct2's `sim_eval/scripts/download_assets.py`.

## Setup

```bash
source scripts/env.sh                                  # paths; everything is written under $GYRE_WORK (default ./work)
cd "$MOLMOACT2_DIR" && uv sync && cd -                 # sim/server venv (or: uv pip install -r requirements/sim-freeze.txt)
"$SIM_PY" "$MOLMOACT2_DIR/sim_eval/scripts/download_assets.py"
bash train/setup_env.sh                                # only needed for training / conversion
```

CPU-only simulation needs a Vulkan device for SAPIEN's loaders: install Mesa's lavapipe (`mesa-vulkan-drivers`).
