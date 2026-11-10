#!/bin/bash
# usage:  train/yam_ft/run_ft.sh NAME REPOS [train_lerobot args...]   (normally called by ft2.sh / ddpN.sh)
#   REPOS: comma list relative to $LEROBOT_DATA_ROOT with optional :weight (e.g. sim/efb_v4n2_p0:0.1,real/*:0.5)
#   multi-node: NNODES=2 NODE_RANK=0|1 MASTER=<ip of rank0> yam_ft/run_ft.sh ...   (run on both hosts)
# Default recipe = action-expert-only (fits one 48 GB L40S). Logs: logs/NAME.log, ckpts: runs/NAME/stepN
set -e
export OLMO_SHARED_FS=1 NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
# DATA_ROOT=$GYRE_SCRATCH/lerobot_data: read datasets from the host-local copy made by stage_data.sh
if [ -n "$DATA_ROOT" ]; then export LEROBOT_DATA_ROOT=$DATA_ROOT HF_LEROBOT_HOME=$DATA_ROOT; fi
cd "$GYRE_TRAIN_WORK"
NAME=$1; REPOS=$2; shift 2
RUNS=${RUNS_DIR:-$GYRE_TRAIN_WORK/runs}   # multi-node runs must use a shared FS here
mkdir -p logs runs $RUNS/$NAME
# CLIP="real/*:clip,ref/*:clip,sim/*:noclip" (per-dataset clip of normalised targets, see train_yam_ft.py); NOCLIP=1 == "*:noclip"
# any noclip/bound rule -> marker runs/<NAME>/NOCLIP, read by eval/serve_ckpt.sh to serve with the no-clip server variant
[ "${NOCLIP:-0}" = 1 ] && CLIP="*:noclip"
unset YAM_FT_NOCLIP
if [ -n "$CLIP" ]; then export YAM_FT_CLIP="$CLIP"; else unset YAM_FT_CLIP; fi
if echo "$CLIP" | grep -q -E "noclip|bound="; then echo "$CLIP" > $RUNS/$NAME/NOCLIP; else rm -f $RUNS/$NAME/NOCLIP; fi
NP=${NPROC:-1}   # GPUs per node
if [ "${NNODES:-1}" = 1 ]; then L="--standalone --nproc-per-node=$NP"; else L="--nnodes=$NNODES --nproc-per-node=$NP --node_rank=$NODE_RANK --master_addr=$MASTER --master_port=${PORT:-29415}"; fi
YAM_FT_REPOS=$REPOS exec torchrun $L $YF/train_yam_ft.py ${INIT_CKPT:-$MOLMOACT2_BASE_DIR} yam_ft \
  --packing=false --dynamic_seq_len=true --num_workers=${NW:-4} --pin_memory=true --log_interval=${LOGI:-10} \
  --save_folder=$RUNS/$NAME --save_num_checkpoints_to_keep=1 \
  --ft_vlm=false --ft_action_expert=true --ft_embedding=none --lora_enable=false --add_depth_tokens=false \
  "$@" > logs/$NAME${NODE_RANK:+.rank$NODE_RANK}.log 2>&1
