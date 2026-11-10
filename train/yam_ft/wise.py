"""WiSE-FT weight interpolation between the pretrained MolmoAct2-BimanualYAM and a fine-tuned HF export:
   w = (1-alpha) * w_pre + alpha * w_ft   (alpha=1 -> fine-tuned, alpha=0 -> pretrained), per tensor in fp32.
  python wise.py --ft RUN/stepN-hf --alpha 0.5 0.65 0.8 --out-root $GYRE_SCRATCH/wise
-> <out-root>/<ftname>_a0.65/ = full HF dir (config/code/norm_stats copied from the fine-tuned export); serve it
   with the same server as the fine-tuned model (noclip variant for noclip-trained runs: NOCLIP=1 serve_ckpt.sh).
   --check: alpha 0 and 1 on the first shard must reproduce pre / ft bitwise (known-answer)."""
import os, sys  # noqa: E401  repo root on sys.path -> gyre_paths (scripts/env.sh does the same)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import gyre_paths as GP  # noqa: E402
import argparse, json, os, shutil, numpy as np, torch
from safetensors.torch import load_file, save_file
PRE = GP.BASE_MODEL_DIR
ap = argparse.ArgumentParser(); ap.add_argument("--ft", required=True); ap.add_argument("--alpha", type=float, nargs="+", default=[0.5, 0.65, 0.8])
ap.add_argument("--out-root", default=os.path.join(GP.SCRATCH, "wise")); ap.add_argument("--pre", default=PRE); ap.add_argument("--check", action="store_true")
a = ap.parse_args()
idx_ft = json.load(open(f"{a.ft}/model.safetensors.index.json"))["weight_map"]
idx_pre = json.load(open(f"{a.pre}/model.safetensors.index.json"))["weight_map"]
assert set(idx_ft) == set(idx_pre), "tensor sets differ"
shards = sorted(set(idx_ft.values()))
pre_cache = {}
def pre_tensor(k):
    f = idx_pre[k]
    if f not in pre_cache: pre_cache.clear(); pre_cache[f] = load_file(os.path.join(a.pre, f))
    return pre_cache[f][k]
if a.check:
    ft = load_file(os.path.join(a.ft, shards[0]))
    for al, ref in ((0.0, "pre"), (1.0, "ft")):
        ok = all(torch.equal(((1 - al) * pre_tensor(k).float() + al * v.float()).to(v.dtype), v if ref == "ft" else pre_tensor(k).to(v.dtype)) for k, v in ft.items())
        print(f"check alpha={al}: reproduces {ref} bitwise on {shards[0]} ({len(ft)} tensors): {ok}")
    raise SystemExit
name = os.path.basename(os.path.normpath(a.ft))
for al in a.alpha:
    out = os.path.join(a.out_root, f"{name}_a{al:g}"); os.makedirs(out, exist_ok=True)
    for f in os.listdir(a.ft):
        if not f.endswith(".safetensors"): shutil.copy(os.path.join(a.ft, f), out)
    for s in shards:
        ft = load_file(os.path.join(a.ft, s))
        save_file({k: ((1 - al) * pre_tensor(k).float() + al * v.float()).to(v.dtype).contiguous() for k, v in ft.items()}, os.path.join(out, s), metadata={"format": "pt"})
    print("wrote", out, flush=True)
