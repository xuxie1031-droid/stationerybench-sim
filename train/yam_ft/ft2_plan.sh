#!/bin/bash
# Round launcher used for every fine-tuning round from ft2 on (see docs/TRAINING.md for the exact per-round settings).
# Modes: dry | stage | check | ae1 | ae2 | lora2 | ft3_ae | ft3_lora | ae_wsd.  Rounds ft3..ft12 all ran `ft3_ae` (action
# expert only) or `ft3_lora` (vision LoRA + LLM LoRA + action expert) with FT3_* overrides, e.g. the final round:
#   train/rounds/ft12_lora.sh (FT3_STEPS=4000 FT3_NO_SWITCH=1 FT3_VIS=custom FT3_VIS_CUSTOM="n2=0.6,dr=0.4"
#   FT3_TASKS=... FT3_SRC_SHARE=... FT3_INIT=<previous round's EMA merged HF export> ... ft2_plan.sh ft3_lora)
#   train/yam_ft/ft2_plan.sh dry                       # print both runs' stage-1/2 configs + phase shares (ft2)
#   AE_HOSTS="local h2" AE_IP=<ip> LORA_HOSTS="h1 h2 h3 h4" LORA_IP=<ip of h1> train/yam_ft/ft2_plan.sh stage   # rsync data to local disks
#   ... train/yam_ft/ft2_plan.sh check                 # <=10-min 4-node LoRA check on LORA_HOSTS (step time + sha256)
#   ... train/yam_ft/ft2_plan.sh ae1 | lora1           # stage 1 (sim 1/2, first 80%)
#   ... INIT2=<stage-1 final HF export> train/yam_ft/ft2_plan.sh ae2 | lora2   # stage 2 (real 2/3, last 20%, lr x0.5)
# NOTE: as run, every round also co-trained a third StationeryBench task that is not part of this release; its data are not
# published, so TASKS defaults here are eraser_from_box + ruler_handover only (mkmix renormalises the missing share).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"; cd "$GYRE_TRAIN_WORK"
R2=$GYRE_TRAIN_WORK/runs
export TASKS=${TASKS:-"efb=1.25,rh=1.25"} VARIANTS=${VARIANTS:-"vnom=0.2,vdr=0.3,v3nom=0.2,v3dr=0.3"}
export EMA=${EMA-0.999} PHASE_W=${PHASE_W-"grasp:3,place:2,default:1"} PHASE_TRANS=${PHASE_TRANS-"15:2"}
export DATA_ROOT=${DATA_ROOT:-}  NW=${NW:-8}   # DATA_ROOT=$GYRE_SCRATCH/lerobot_data after `stage` (host-local copies)
AE_STEPS=${AE_STEPS:-7000}; LO_STEPS=${LO_STEPS:-5000}
AE_INIT=${AE_INIT:-$R2/ft2vis_ae/step1500-hf}; LO_INIT=${LO_INIT:-$R2/ft1_lora/step1500-merged-hf}
s1(){ echo $(($1*8/10)); }; s2(){ echo $(($1-$1*8/10)); }
# phase-1 final lr of a group = its base lr x scheduler alpha_f (stock cosine ends exactly there); read from config
p1lr(){ python - "$1" "$2" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(f"{float(c['optimizer'][sys.argv[2]]) * float(c['scheduler']['alpha_f']):.6g}")
PY
}
case ${1:-dry} in
  dry)
    echo "=== ft2_ae stage 1 (2 nodes) ==="; MODE=ae INIT=$AE_INIT STEPS=$(s1 $AE_STEPS) SAVE=1000 SIM_SHARE=0.5 AE_LR=2e-5 AE_WARMUP=500 HOSTS="${AE_HOSTS:-h0 h1}" MASTER_IP=x DRY=1 bash $YF/ft2.sh ft2_ae
    echo "=== ft2_ae stage 2: steps $(s2 $AE_STEPS), SIM_SHARE=0.3333, AE_LR=1e-5, init = stage-1 export ==="
    echo "=== ft2_lora stage 1 (4 nodes) ==="; MODE=lora INIT=$LO_INIT STEPS=$(s1 $LO_STEPS) SAVE=1000 SIM_SHARE=0.5 AE_LR=2e-5 AE_WARMUP=500 VLM_LR=5e-5 HOSTS="${LORA_HOSTS:-h0 h1 h2 h3}" MASTER_IP=x DRY=1 bash $YF/ft2.sh ft2_lora | head -4
    echo "=== ft2_lora stage 2: steps $(s2 $LO_STEPS), SIM_SHARE=0.3333, VLM_LR=2.5e-5 AE_LR=1e-5, init = stage-1 merged export ===" ;;
  stage)
    MIX=$(python $YF/mkmix.py --tasks "$TASKS" --variants "$VARIANTS" --sim-share 0.5)
    HOSTS="$AE_HOSTS $LORA_HOSTS" bash $YF/stage_data.sh "$MIX" ;;
  check)
    MIX=$(python $YF/mkmix.py --tasks "$TASKS" --variants "$VARIANTS" --sim-share 0.5)
    HOSTS="$LORA_HOSTS" MASTER_IP=$LORA_IP MIX="$MIX" STEPS=40 MODE=lora bash $YF/checkN.sh ;;
  ae1)   MODE=ae INIT=$AE_INIT STEPS=$(s1 $AE_STEPS) SAVE=1000 SIM_SHARE=0.5 AE_LR=2e-5 AE_WARMUP=500 HOSTS="$AE_HOSTS" MASTER_IP=$AE_IP PORT=29581 bash $YF/ft2.sh ft2_ae ;;
  ae2)   # phase 2 (decided 11-04): INIT = phase-1 EMA weights (runs/ft2_ae/ema/step<N1> exported here), not the
         # live export the chain passes in INIT2 (kept as control). EMA continues from the same EMA checkpoint, so the
         # shadow starts identical to the live weights. No warmup, start at phase-1 final lr, linear decay to 1%.
         C=runs/ft2_ae/config.yaml; AELR=$(p1lr $C action_expert_learning_rate); N1=$(s1 $AE_STEPS)
         EMA_CK=runs/ft2_ae/ema/step$N1; EMA_HF=$GYRE_TRAIN_WORK/$EMA_CK-hf; INIT_AE=${INIT2:?}
         if [ -z "$DRY" ]; then
           [ -f $EMA_HF/EXPORT_OK ] || NOCLIP=1 bash $YF/export_ckpt.sh $EMA_CK $EMA_HF
           if [ -f $EMA_HF/EXPORT_OK ] && python $YF/hf_trainable_hash.py $EMA_HF ${INIT2} ; then
             INIT_AE=$EMA_HF; echo "ae2: INIT = phase-1 EMA export $EMA_HF (live control: $INIT2)"
           else echo "ae2: WARNING EMA export missing or identical to live -> falling back to live INIT $INIT2"; fi
         else INIT_AE=$EMA_HF; fi
         YAM_FT_SCHED=cont CONT_END_FRAC=0.01 YAM_FT_EMA_INIT=$GYRE_TRAIN_WORK/$EMA_CK \
         MODE=ae INIT=$INIT_AE STEPS=$(s2 $AE_STEPS) SAVE=1000 SIM_SHARE=0.3333 AE_LR=$AELR AE_WARMUP=0 HOSTS="$AE_HOSTS" MASTER_IP=$AE_IP PORT=29583 bash $YF/ft2.sh ft2_ae_s2 ;;
  lora2) # phase 2: as ae2; per-group start = phase-1 finals (VLM groups, AE group).  EMA cannot continue across LoRA
         # runs (phase 2 merges the phase-1 adapters and injects fresh ones), so phase-2 EMA restarts at its INIT.
         C=runs/ft2_lora/config.yaml; VLR=$(p1lr $C llm_learning_rate); AELR=$(p1lr $C action_expert_learning_rate)
         YAM_FT_SCHED=cont CONT_END_FRAC=0.01 \
         MODE=lora INIT=${INIT2:?} STEPS=$(s2 $LO_STEPS) SAVE=1000 SIM_SHARE=0.3333 AE_LR=$AELR AE_WARMUP=0 VLM_LR=$VLR HOSTS="$LORA_HOSTS" MASTER_IP=$LORA_IP PORT=29587 bash $YF/ft2.sh ft2_lora_s2 ;;
  ft3_ae|ft3_lora)  # ft3: + additional synthetic demos.  WSD peak 1e-5 (LoRA VLM 2.5e-5), warmup 200, decay over the last 20%;
         # the real-2/3 data switch happens INSIDE the run at the decay start (YAM_FT_MIX_SWITCH), no re-warm.
         # per task: rec .4 / v2 .4 / v3 .2 (ruler: rec .4 / v2 .5 / v3 .1), visuals nom .4 / dr .6; PHASE_W off; EMA on.
         # visual split inside every source (v2, v3, rec): default nominal-heavy nom .6 / dr .4 (from 11-05); FT3_VIS=old -> .4/.6
         case ${FT3_VIS:-new} in old) FT3_VIS_SPEC="nom=0.4,dr=0.6";; nom70) FT3_VIS_SPEC="nom=0.7,dr=0.3";;
           new) FT3_VIS_SPEC="nom=0.6,dr=0.4";; custom) FT3_VIS_SPEC="${FT3_VIS_CUSTOM:?set FT3_VIS_CUSTOM}";; *) echo "FT3_VIS must be new|old|nom70|custom"; exit 2;; esac
         M=${1#ft3_}; [ $M = ae ] && ST=${FT3_STEPS:-3500} || ST=${FT3_STEPS:-3500}; SW=$((ST - ST/5))
         # INIT default: phase-2 final EMA export (release default when tied), else live, else fallbacks; FT3_INIT overrides
         if [ $M = ae ]; then IN=${FT3_INIT:-$(ls -d $R2/ft2_ae_s2/ema/step1400-hf $R2/ft2_ae_s2/step1400-hf 2>/dev/null | head -1)}; IN=${IN:-$R2/ft2vis_ae/step1500-hf}
         else IN=${FT3_INIT:-$(ls -d $R2/ft2_lora_s2/ema/step1000-merged-hf $R2/ft2_lora_s2/step1000-merged-hf 2>/dev/null | head -1)}; IN=${IN:-$R2/ft1_lora/step1500-merged-hf}; fi
         [ -z "$DRY" ] && [ ! -f "$IN/EXPORT_OK" ] && [ ! -f "$IN/config.json" ] && { echo "ft3: INIT $IN missing"; exit 1; }
         YAM_FT_SCHED=wsd WSD_DECAY_FRAC=0.2 WSD_END_FRAC=0 WSD_SHAPE=linear YAM_FT_MIX_SWITCH=$([ "${FT3_NO_SWITCH:-}" = 1 ] && echo "" || echo $SW:0.6667) PHASE_W= PHASE_TRANS= \
         TASKS="${FT3_TASKS:-efb=1.25,rh=1.25}" SRC_SHARE="${FT3_SRC_SHARE:-default:rec=0.4,v2=0.4,v3=0.2;rh:rec=0.4,v2=0.5,v3=0.1}" VIS="$FT3_VIS_SPEC" \
         MODE=$M INIT=$IN STEPS=$ST SAVE=${FT3_SAVE:-1000} SIM_SHARE=0.5 AE_LR=${FT3_AE_LR:-1e-5} AE_WARMUP=${FT3_AE_WARMUP:-200} VLM_LR=2.5e-5 \
         HOSTS="$([ $M = ae ] && echo "$AE_HOSTS" || echo "$LORA_HOSTS")" MASTER_IP=$([ $M = ae ] && echo "$AE_IP" || echo "$LORA_IP") \
         PORT=$([ $M = ae ] && echo 29591 || echo 29593) bash $YF/ft2.sh ${FT3_RUN:-ft3_$M} ;;
  ae_wsd) # possible AE restart (NOT launched automatically): WSD peak 1e-5, warmup 200, decay last 20% (linear to 0),
          # v3 share 20% (ruler 10%), PHASE_W off, EMA on, init ft2vis1500 (sharded ckpt -> needs an HF export first)
          YAM_FT_SCHED=wsd WSD_DECAY_FRAC=0.2 WSD_END_FRAC=0 WSD_SHAPE=linear PHASE_W= PHASE_TRANS= \
          TASKS="efb=1.25,rh=1.25" VARIANTS="vnom=0.4,vdr=0.6" V3_SHARE="efb=0.2,rh=0.1" \
          MODE=ae INIT=${WSD_INIT:-$R2/ft2vis_ae/step1500-hf} STEPS=${WSD_STEPS:-3000} SAVE=1000 \
          SIM_SHARE=0.5 AE_LR=1e-5 AE_WARMUP=200 HOSTS="$AE_HOSTS" MASTER_IP=$AE_IP PORT=29589 bash $YF/ft2.sh ft2_ae_wsd ;;
esac
