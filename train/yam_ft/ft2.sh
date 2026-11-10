#!/bin/bash
# ft2 launcher (parameterised, NOT auto-started).  Example (dry run prints the command):
#   MODE=lora INIT=$GYRE_TRAIN_WORK/runs/<prev>/ema/step4000-merged-hf STEPS=4000 DRY=1 bash train/yam_ft/ft2.sh ft12_lora
# knobs (env):
#   MODE ae|lora   INIT  HF dir to start from (ft1 export or released ckpt)   STEPS  SAVE  GBS(16)
#   TASKS "efb=1,rh=1"  VARIANTS "v2=1,vnom=1,vdr=1,v3=1"  SIM_SHARE 0.5 (2/3 real -> 0.3333; 1 = sim-only arm)
#   AE_LR (2e-5)  AE_WARMUP (500)  VLM_LR (5e-5, LoRA)  HOSTS "local h1 .." + MASTER_IP (multi-node, replicate-only + nosync; first = rank 0)
#   DATA_ROOT=$GYRE_SCRATCH/lerobot_data (host-local copy made by stage_data.sh) and NW=8 recommended for multi-node
#   CLIP rules default "real/*:clip,ref/*:clip,sim/*:noclip"
# Final "real 2/3" phase: run a second short ft2.sh from the stage-1 export with SIM_SHARE=0.3333 (fresh warmup).
#   EMA=<decay> (e.g. 0.999; off by default) -> EMA weights saved as <run>/ema/step<N>; serve with EMA=1 serve_ckpt.sh
#   PHASE_W="grasp:3,place:2,default:1"  PHASE_TRANS="15:2"  (sim datasets only; real stays uniform)
# WiSE: train/yam_ft/wise.py on the final export (explored, not used for the released weights).
set -e
NAME=$1; source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"; cd "$GYRE_TRAIN_WORK"
MIXSTR=$(python $YF/mkmix.py --tasks "${TASKS:-efb=1,rh=1}" --variants "${VARIANTS:-v2=1,vnom=1,vdr=1,v3=1}" --sim-share ${SIM_SHARE:-0.5} ${V3_SHARE:+--v3-share $V3_SHARE} ${SRC_SHARE:+--src-share "$SRC_SHARE"} ${VIS:+--vis $VIS})
COMMON="--device_batch_size=1 --global_batch_size=${GBS:-16} --max_duration=${STEPS:?} --log_interval=20 --save_interval=${SAVE:?}"
if [ "$MODE" = lora ]; then
  ARGS="$COMMON --ft_vlm=true --ft_embedding=lm_head --lora_enable=true --lora_rank=64 --llm_learning_rate=${VLM_LR:-5e-5} --vit_learning_rate=${VLM_LR:-5e-5} --connector_learning_rate=${VLM_LR:-5e-5} --action_expert_learning_rate=${AE_LR:-2e-5} --scheduler.action_expert_t_warmup=${AE_WARMUP:-500}"
else
  ARGS="$COMMON --action_expert_learning_rate=${AE_LR:-2e-5} --scheduler.action_expert_t_warmup=${AE_WARMUP:-500}"
fi
export CLIP=${CLIP:-"real/*:clip,ref/*:clip,sim/*:noclip"}
[ -n "$EMA" ] && export YAM_FT_EMA=$EMA
[ -n "$PHASE_W" ] && export YAM_FT_PHASE_W=$PHASE_W
[ -n "$PHASE_TRANS" ] && export YAM_FT_PHASE_TRANS=$PHASE_TRANS
echo "mixture: $MIXSTR"; echo "args: $ARGS"; echo "init: ${INIT:?}"; echo "sched: ${YAM_FT_SCHED:-stock cosine (warmup per group, alpha_f 0.1)} ${YAM_FT_SCHED:+CONT_END_FRAC=$CONT_END_FRAC WSD_DECAY_FRAC=$WSD_DECAY_FRAC WSD_END_FRAC=$WSD_END_FRAC WSD_SHAPE=$WSD_SHAPE}  ema_init: ${YAM_FT_EMA_INIT:-none}  mix_switch: ${YAM_FT_MIX_SWITCH:-none}"; echo "ema: ${YAM_FT_EMA:-off}  phase_w: ${YAM_FT_PHASE_W:-off}  phase_trans: ${YAM_FT_PHASE_TRANS:-off}"
if [ -n "$YAM_FT_PHASE_W$YAM_FT_PHASE_TRANS" ]; then python $YF/phase_share.py "$MIXSTR"; fi
[ -n "$DRY" ] && exit 0
if [ -n "$HOSTS" ]; then
  # multi-node (HOSTS="local host2 ..." MASTER_IP=<rank0 private ip>): repo, checkpoints + INIT must be on a shared FS
  YAM_FT_ACCUM_NOSYNC=1 RUNS_DIR=${RUNS_DIR:-$GYRE_TRAIN_WORK/runs} INIT_CKPT=$INIT bash $YF/ddpN.sh $NAME "$MIXSTR" $ARGS
else
  RUNS_DIR=${RUNS_DIR:-$GYRE_TRAIN_WORK/runs} INIT_CKPT=$INIT nohup bash $YF/run_ft.sh $NAME "$MIXSTR" $ARGS > /dev/null 2>&1 &
  echo "started $NAME pid $!"
fi
