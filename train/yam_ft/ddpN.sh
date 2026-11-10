#!/bin/bash
# N-node replicate-only (DDP-like HSDP, shard degree 1, 1 GPU per node) launcher.  Run from the rank-0 machine:
#   HOSTS="local host2 host3 host4" MASTER_IP=<rank-0 private IP> train/yam_ft/ddpN.sh NAME REPOS [train args...]
#   first host = rank 0 ("local" = this machine), others reached by ssh.  The repo checkout, $GYRE_TRAIN_WORK (checkpoints)
#   and INIT must be on a filesystem shared by all hosts at the same path.  Without RDMA, keep YAM_FT_ACCUM_NOSYNC=1
#   (gradient all-reduce once per optimizer step instead of per micro-batch).
# env passed through: CLIP RUNS_DIR YAM_FT_PARAM_HASH YAM_FT_ACCUM_NOSYNC INIT_CKPT YAM_FT_EMA* YAM_FT_PHASE_* PORT
NAME=$1; REPOS=$2; shift 2
ARGS=$(printf "%q " "$@")
read -ra HS <<< "${HOSTS:?set HOSTS}"; N=${#HS[@]}; P=${PORT:-29533}; MIP=${MASTER_IP:?set MASTER_IP}
on(){ h=$1; shift; if [ "$h" = local ]; then bash -c "$*"; else ssh $h "$*"; fi; }
for ((r=N-1; r>=0; r--)); do h=${HS[$r]}
  on $h "cd '$GYRE_TRAIN_WORK'; GYRE_WORK='$GYRE_WORK' GYRE_TRAIN_WORK='$GYRE_TRAIN_WORK' LEROBOT_DATA_ROOT='$LEROBOT_DATA_ROOT' MOLMOACT2_DIR='$MOLMOACT2_DIR' MOLMOACT2_BASE_DIR='$MOLMOACT2_BASE_DIR' TRAIN_VENV='$TRAIN_VENV' CLIP='$CLIP' DATA_ROOT='$DATA_ROOT' NW='$NW' RUNS_DIR='${RUNS_DIR:-$GYRE_TRAIN_WORK/runs}' YAM_FT_PARAM_HASH='$YAM_FT_PARAM_HASH' YAM_FT_ACCUM_NOSYNC='${YAM_FT_ACCUM_NOSYNC:-1}' INIT_CKPT='$INIT_CKPT' YAM_FT_EMA='$YAM_FT_EMA' YAM_FT_EMA_DEVICE='$YAM_FT_EMA_DEVICE' YAM_FT_EMA_TRACE='$YAM_FT_EMA_TRACE' YAM_FT_PHASE_W='$YAM_FT_PHASE_W' YAM_FT_PHASE_TRANS='$YAM_FT_PHASE_TRANS' YAM_FT_SCHED='$YAM_FT_SCHED' CONT_END_FRAC='$CONT_END_FRAC' WSD_DECAY_FRAC='$WSD_DECAY_FRAC' WSD_END_FRAC='$WSD_END_FRAC' WSD_SHAPE='$WSD_SHAPE' YAM_FT_EMA_INIT='$YAM_FT_EMA_INIT' YAM_FT_MIX_SWITCH='$YAM_FT_MIX_SWITCH' PORT=$P NNODES=$N NODE_RANK=$r MASTER=$MIP nohup bash $YF/run_ft.sh $NAME '$REPOS' --fsdp.hybrid_sharding_num_model_replicas=$N $ARGS > /dev/null 2>&1 & echo $h rank$r \$!"
done
