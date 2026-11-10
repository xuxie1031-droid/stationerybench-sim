#!/usr/bin/env bash
# One-command closed-loop evaluation on one GPU.
#
#   scripts/eval.sh TASK VISUAL SEED0 N --model PATH|base [--arm NAME] [--server noclip|stock] [--port 8230]
#                   [--max-steps K] [--no-spring] [--confirm] [--ref ARM]
#
#   TASK    efb | rh | all            (efb = eraser_from_box, rh = ruler_handover)
#   VISUAL  nominal2 | nominal | dr_train | dr_heldout | off
#   SEED0 N episodes use env seeds SEED0 .. SEED0+N-1   (dev set: 1000 20; sealed set: 3000000 50, needs --confirm)
#   --model base        stock MolmoAct2-BimanualYAM ($MOLMOACT2_BASE_DIR) with the stock server
#   --model DIR         an HF-format checkpoint dir (config.json + norm_stats.json + model*.safetensors), e.g. the
#                       released fine-tuned weights; served with eval/host_server_yam_noclip.py (required for our
#                       fine-tuned weights: their sim targets were trained without the [-1,1] clip)
#   --model RUN/stepN   a raw training checkpoint: exported to HF first by eval/serve_ckpt.sh
#
# Starts the policy server on 127.0.0.1:PORT, waits until it is up, runs eval/driver.py (one sim process at a time),
# stops the server (by PID) and prints eval/report.py.  Episode lengths: efb 3600 steps (120 s), rh 2000 steps.
# Policy sampling is unseeded: per-episode outcomes vary between runs; compare counts with their Wilson intervals.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

TASK=${1:?TASK}; VISUAL=${2:?VISUAL}; SEED0=${3:?SEED0}; N=${4:?N}; shift 4
MODEL=""; ARM=""; SERVER=""; PORT=${PORT:-8230}; EXTRA=(); REF=""
while [ $# -gt 0 ]; do
  case $1 in
    --model) MODEL=$2; shift 2;;
    --arm) ARM=$2; shift 2;;
    --server) SERVER=$2; shift 2;;
    --port) PORT=$2; shift 2;;
    --max-steps) EXTRA+=(--max-steps "$2"); shift 2;;
    --no-spring) EXTRA+=(--no-spring); shift;;
    --confirm) EXTRA+=(--confirm); shift;;
    --ref) REF=$2; shift 2;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done
[ -n "$MODEL" ] || { echo "--model PATH|base is required"; exit 2; }
case $TASK in all) TASKS=efb,rh;; efb|rh) TASKS=$TASK;; *) echo "TASK must be efb|rh|all"; exit 2;; esac
[ -n "$ARM" ] || ARM="$( [ "$MODEL" = base ] && echo base || basename "$(readlink -f "$MODEL")" )"
mkdir -p "$GYRE_EVAL_DIR/logs"
SLOG="$GYRE_EVAL_DIR/logs/server_${ARM}_$(date -u +%Y%m%dT%H%M%S).log"

if [ "$MODEL" = base ]; then
  [ -f "$MOLMOACT2_BASE_DIR/norm_stats.json" ] || { echo "base model not found at MOLMOACT2_BASE_DIR=$MOLMOACT2_BASE_DIR (scripts/download.sh base)"; exit 1; }
  ( cd "$MOLMOACT2_DIR" && exec "$SIM_PY" examples/yam/host_server_yam.py --host 127.0.0.1 --port "$PORT" --repo-id "$MOLMOACT2_BASE_DIR" ) > "$SLOG" 2>&1 &
elif [ -f "$MODEL/config.json" ] && [ -f "$MODEL/norm_stats.json" ]; then
  if [ "${SERVER:-noclip}" = stock ]; then S=examples/yam/host_server_yam.py; else S="$GYRE_ROOT/eval/host_server_yam_noclip.py"; fi
  ( cd "$MOLMOACT2_DIR" && exec "$SIM_PY" "$S" --host 127.0.0.1 --port "$PORT" --repo-id "$(readlink -f "$MODEL")" ) > "$SLOG" 2>&1 &
else
  NOCLIP=$([ "${SERVER:-noclip}" = stock ] && echo 0 || echo 1) bash "$GYRE_ROOT/eval/serve_ckpt.sh" "$MODEL" "$PORT" > "$SLOG" 2>&1 &
fi
SPID=$!
cleanup() { kill "$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true; }
trap cleanup EXIT
echo "server pid $SPID, log $SLOG"
for _ in $(seq 360); do
  grep -q "Uvicorn running" "$SLOG" && break
  kill -0 "$SPID" 2>/dev/null || { echo "server exited; see $SLOG"; exit 1; }
  sleep 5
done
grep -q "Uvicorn running" "$SLOG" || { echo "server did not come up in 30 min; see $SLOG"; exit 1; }

"$SIM_PY" "$GYRE_ROOT/eval/driver.py" --arm "$ARM" --policy "$ARM" --ckpt "$MODEL" --url "http://127.0.0.1:$PORT/act" \
  --server-pid "$SPID" --visual "$VISUAL" --tasks "$TASKS" --seed0 "$SEED0" --n "$N" ${EXTRA[@]+"${EXTRA[@]}"}

if [ "$TASKS" != rh ]; then   # the eraser eval log must confirm the lid physics that was used
  n=$(grep -l "LID_SPRING=1" "$GYRE_EVAL_DIR"/runs/"$ARM"/"$VISUAL"/efb/*/eval.log 2>/dev/null | wc -l || true)
  echo "eraser chunks with LID_SPRING=1 in their log: $n"
fi
"$SIM_PY" "$GYRE_ROOT/eval/report.py" "$ARM" --visual "$VISUAL" --seeds "$SEED0-$((SEED0 + N - 1))" ${REF:+--ref "$REF"}
