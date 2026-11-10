#!/usr/bin/env bash
# Download weights / datasets from the Hugging Face Hub into $GYRE_WORK.
#   scripts/download.sh base finetuned eraser ruler real
#     base       allenai/MolmoAct2-BimanualYAM                    -> $MOLMOACT2_BASE_DIR
#     finetuned  the released fine-tuned weights (HF format)      -> $GYRE_WORK/models/MolmoAct2-YAM-StationeryBench-Sim
#     eraser     sim training data, eraser_from_box (LeRobot)     -> $LEROBOT_DATA_ROOT/sim/
#     ruler      sim training data, ruler_handover (LeRobot)      -> $LEROBOT_DATA_ROOT/sim/
#     real       the 16 Ai2 real YAM datasets used for co-training -> $LEROBOT_DATA_ROOT/real/   (see docs/DATA.md)
# Repository ids can be overridden: HF_FINETUNED, HF_DATA_ERASER, HF_DATA_RULER.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export HF_HUB_OFFLINE=0
hf() { "${HF_CLI:-$(dirname "$SIM_PY")/hf}" "$@"; }   # the hf CLI of the sim venv (huggingface_hub)
HF_FINETUNED=${HF_FINETUNED:-xuxie1031/MolmoAct2-YAM-StationeryBench-Sim}     # TODO: final repo name
HF_DATA_ERASER=${HF_DATA_ERASER:-xuxie1031/stationerybench-sim-eraser}        # TODO: final repo name
HF_DATA_RULER=${HF_DATA_RULER:-xuxie1031/stationerybench-sim-ruler}           # TODO: final repo name
REAL=(03012026-plate-cleaning-07 12122025-tool-07 13012026-scan-6 14122025-cut-12 15122025-cup-09 15122025-toy-07
      16122025-toilet-09 17122025-clo-14 17122025-med-06 19012026-charging-08 23012026-scoop-06 26122025-foldclo-04
      26122025-tablebuss-13 27012026-untangle-08 28112025-block-02 30112025-box-02)
for what in "$@"; do
  case $what in
    base)      hf download allenai/MolmoAct2-BimanualYAM --local-dir "$MOLMOACT2_BASE_DIR" ;;
    finetuned) hf download "$HF_FINETUNED" --local-dir "$GYRE_WORK/models/$(basename "$HF_FINETUNED")" ;;
    eraser)    hf download "$HF_DATA_ERASER" --repo-type dataset --local-dir "$LEROBOT_DATA_ROOT/sim" ;;
    ruler)     hf download "$HF_DATA_RULER" --repo-type dataset --local-dir "$LEROBOT_DATA_ROOT/sim" ;;
    real)      for r in "${REAL[@]}"; do hf download "allenai/$r" --repo-type dataset --local-dir "$LEROBOT_DATA_ROOT/real/$r"; done ;;
    *) echo "unknown item $what (base finetuned eraser ruler real)"; exit 2 ;;
  esac
done
