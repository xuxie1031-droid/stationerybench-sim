"""sha256 of the action-expert tensors of two HF exports; exit 0 iff both load and the hashes DIFFER.
  python hf_trainable_hash.py EMA_HF LIVE_HF"""
import sys, json, os, hashlib
from safetensors import safe_open
def h(d):
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    keys = sorted(k for k in idx if "action_expert" in k); m = hashlib.sha256(); hs = {}
    for k in keys:
        f = idx[k]
        if f not in hs: hs[f] = safe_open(os.path.join(d, f), "pt")
        t = hs[f].get_tensor(k).float().contiguous()
        m.update(k.encode()); m.update(t.numpy().tobytes())
    return m.hexdigest(), len(keys)
a, na = h(sys.argv[1]); b, nb = h(sys.argv[2])
print(f"action-expert tensors: {sys.argv[1]} {na} sha {a[:16]} | {sys.argv[2]} {nb} sha {b[:16]} | differ: {a != b}")
sys.exit(0 if (a != b and na == nb and na > 0) else 1)
