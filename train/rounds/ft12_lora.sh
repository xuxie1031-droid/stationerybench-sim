#!/usr/bin/env bash
# The final round as run (ft12_lora), expressed with this repo's launcher.  See docs/TRAINING.md for every round.
#   train/rounds/ft12_lora.sh INIT_HF_DIR [HOSTS="local"] [MASTER_IP=127.0.0.1]
#     INIT_HF_DIR  HF export of the previous round's EMA weights with LoRA merged (as run: ft11_lora ema/step4000-merged)
# As run: 8 single-GPU nodes (global batch 16 = 8 ranks x 2 micro-batches), vision LoRA r64 + LLM LoRA r64 + action
# expert, VLM LR 2.5e-5 / AE LR 1e-5, warmup 200, WSD (linear decay over the last 20 %), EMA 0.999, 4000 steps,
# sim/real configured 1:1 (the effective share differs, see TRAINING.md), sim looks nominal2 0.6 / dr_train 0.4.
# As run the task weights were efb 2 / rh 1 / third task 2; the third task's data are not released, so mkmix.py
# renormalises over the two released tasks.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
IN=${1:?INIT_HF_DIR}
export LORA_HOSTS="${HOSTS:-local}" LORA_IP="${MASTER_IP:-127.0.0.1}" FT3_RUN=${FT3_RUN:-ft12_lora} FT3_STEPS=4000 FT3_NO_SWITCH=1 \
       FT3_VIS=custom FT3_VIS_CUSTOM="n2=0.6,dr=0.4" FT3_TASKS="efb=2,rh=1" FT3_INIT="$IN" \
       FT3_SRC_SHARE="default:rec=0.4,v2=0.4,v3=0.2;rh:rec=0.4,v2=0.5,v3=0.1;efb:v4=0.3,rec3=0.05,rec4=0.05,rec5=0.1,rec6=0.2,rec7=0.3"
DRY=1 bash "$YF/ft2_plan.sh" ft3_lora      # prints the mixture and the trainer arguments first
bash "$YF/ft2_plan.sh" ft3_lora
