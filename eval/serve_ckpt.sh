#!/bin/bash
# Eval hook: export a fine-tuned MolmoAct2 training checkpoint to HF format and serve it with the stock YAM server.
#   eval/serve_ckpt.sh $GYRE_TRAIN_WORK/runs/<NAME>/step<N>[-merged] [PORT=8202] [--export-only]   (after source scripts/env.sh)
# (released weights are already HF exports: serve them directly, see scripts/eval.sh)
# - full/action-expert runs: pass runs/<NAME>/stepN ; LoRA runs: pass runs/<NAME>/stepN-merged
# - writes <ckpt>-hf (fp32 safetensors + norm_stats.json from the training config); the remote-code *.py files are
#   taken from the released MolmoAct2-BimanualYAM snapshot so the server's transformers 4.57 env loads it exactly
#   like the stock checkpoint (set KEEP_EXPORT_CODE=1 to keep the exporter's own copies instead)
# - runs trained with NOCLIP=1 are served with eval/host_server_yam_noclip.py automatically
# - then runs examples/yam/host_server_yam.py --repo-id <ckpt>-hf (bf16, same /act schema as the stock server)
set -e
: "${GYRE_ROOT:?source scripts/env.sh first}"
CKPT=$(readlink -f "$1"); PORT=${2:-8202}
# EMA=1: use the EMA weights saved by YAM_FT_EMA runs (runs/<NAME>/ema/step<N>[-merged])
if [ "${EMA:-0}" = 1 ]; then CKPT=$(dirname $CKPT)/ema/$(basename $CKPT); [ -d "$CKPT" ] || { echo "no EMA checkpoint $CKPT"; exit 1; }; fi
HF=${HF_OUT:-${CKPT}-hf}  # HF_OUT=/scratch/... keeps the export host-local
REL=${MOLMOACT2_BASE_DIR:?}   # local snapshot of allenai/MolmoAct2-BimanualYAM
if [ ! -f "$HF/norm_stats.json" ]; then
  ( source $GYRE_ROOT/train/env.sh; cd $E; CUDA_VISIBLE_DEVICES="" python -m olmo.hf_model.convert_molmoact2_to_hf "$CKPT" "$HF" )
  if [ -z "$KEEP_EXPORT_CODE" ]; then
    for f in configuration_molmoact2.py modeling_molmoact2.py processing_molmoact2.py image_processing_molmoact2.py video_processing_molmoact2.py inference.py chat_template.jinja; do
      [ -f $REL/$f ] && cp $REL/$f $HF/$f; done
  fi
  python3 - "$HF" "$REL" <<'PY'
import json, sys, shutil
hf, rel = sys.argv[1:3]
# The exporter writes action_mode=continuous, depth_mode=None, text max_position_embeddings=36864, which changes the
# released model's outputs (measured: up to 0.016 rad per chunk with identical weights).  With the released
# config.json the exported lr~0 checkpoint is bit-identical to the stock one, so use it when nothing else differs.
def flat(d, p=""):
    for k, v in d.items():
        if isinstance(v, dict): yield from flat(v, p + k + ".")
        else: yield p + k, v
ca, cb = dict(flat(json.load(open(f"{rel}/config.json")))), dict(flat(json.load(open(f"{hf}/config.json"))))
ok = {"action_mode", "depth_mode", "text_config.max_position_embeddings", "transformers_version", "_name_or_path"}
bad = [k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k) and k.split(".")[-1] not in ok and k not in ok
       and not (ca.get(k) is None and cb.get(k) in (False, True, 0, 1, "", None) or ".id2label." in k or ".label2id." in k)]
if bad:
    sys.exit(f"exported config differs from released in architecture keys {sorted(bad)}; not overriding")
shutil.copy(f"{rel}/config.json", f"{hf}/config.json")
print("config.json: architecture identical to released -> released config.json used")
a = json.load(open(f"{hf}/norm_stats.json"))["metadata_by_tag"]["yam_dual_molmoact2"]
b = json.load(open(f"{rel}/norm_stats.json"))["metadata_by_tag"]["yam_dual_molmoact2"]
for k in ("action_stats", "state_stats"):
    same = all(abs(x - y) < 1e-6 for q in ("q01", "q99") for x, y in zip(a[k][q], b[k][q]))
    print(f"norm_stats {k} q01/q99 identical to released checkpoint: {same}")
PY
fi
[ "$3" = "--export-only" ] && exit 0
cd $MOLMOACT2_DIR
SERVER=examples/yam/host_server_yam.py
# run trained with NOCLIP=1 (marker runs/<NAME>/NOCLIP) or NOCLIP=1 given here -> server variant without the clip
if [ -f "$(dirname $CKPT)/NOCLIP" ] || [ -f "$(dirname $(dirname $CKPT))/NOCLIP" ] || [ "${NOCLIP:-0}" = 1 ]; then SERVER=$GYRE_ROOT/eval/host_server_yam_noclip.py; fi
echo "serving $HF with $SERVER"
exec $SIM_PY $SERVER --host 127.0.0.1 --port $PORT --repo-id "$HF"
