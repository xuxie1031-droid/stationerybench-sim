#!/bin/bash
# Export a training checkpoint to HF for evals; writes <HF_OUT>/EXPORT_OK only on success.
#   train/yam_ft/export_ckpt.sh runs/<NAME>/step<N>[-merged] <HF_OUT>   (env NOCLIP / EMA passed to eval/serve_ckpt.sh)
# The released weights are such an export of the final round's merged EMA checkpoint (runs/<final>/ema/step4000-merged).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"; cd "$GYRE_TRAIN_WORK"
CK=$1; OUT=$2; LOG=logs/export_$(echo $OUT | tr / _ | tail -c 80).log
rm -f $OUT/EXPORT_OK
HF_OUT=$OUT nice -n 10 bash $GYRE_ROOT/eval/serve_ckpt.sh $CK 0 --export-only > $LOG 2>&1; rc=$?
n=$(ls $OUT/model-*.safetensors 2>/dev/null | wc -l)
if [ $rc = 0 ] && grep -q "action_stats q01/q99 identical to released checkpoint: True" $LOG && [ "$n" = 5 ] && [ -f $OUT/norm_stats.json ]; then
  date -u +%FT%TZ > $OUT/EXPORT_OK; echo "EXPORT_OK $OUT"
else
  echo "EXPORT_FAILED rc=$rc shards=$n log=$LOG"; exit 1
fi
