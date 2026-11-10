"""Report per-phase sample share (within sim) before/after phase weighting for a YAM_FT_REPOS mixture string, using
the same weights as training (env YAM_FT_PHASE_W / YAM_FT_PHASE_TRANS) and the mixture's per-dataset shares.
  YAM_FT_PHASE_W=grasp:3,place:2 python phase_share.py "sim/efb_v*:0.25,...,real/*:0.5" """
import os, sys  # noqa: E401  repo root on sys.path -> gyre_paths (scripts/env.sh does the same)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import gyre_paths as GP  # noqa: E402
import os, sys, glob, json, numpy as np, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_sampling as PS
root = os.environ.get("LEROBOT_DATA_ROOT", GP.LEROBOT_DATA)
w = PS.parse_weights(os.environ.get("YAM_FT_PHASE_W")); tr = os.environ.get("YAM_FT_PHASE_TRANS")
tr = (int(tr.split(":")[0]), float(tr.split(":")[1])) if tr else None
acc = collections.defaultdict(lambda: np.zeros(2)); missing = []
for item in sys.argv[1].split(","):
    pat, _, share = item.rpartition(":"); share = float(share)
    if not pat.startswith("sim/"): continue
    hits = sorted(os.path.relpath(os.path.dirname(os.path.dirname(p)), root) for p in glob.glob(os.path.join(root, pat, "meta", "info.json"))
                  if os.path.exists(os.path.join(os.path.dirname(p), "yam_ft_source.json")))
    n = [json.load(open(os.path.join(root, h, "meta", "info.json")))["total_frames"] for h in hits]
    for h, k in zip(hits, n):
        sc = os.path.join(root, h, "meta", "yam_ft_phase.npz")
        if not os.path.exists(sc): missing.append(h); continue
        fw, st, names = PS.frame_weights(sc, w, tr); s_h = share * k / sum(n)
        for k2, (a, b) in PS.phase_share(st, names, fw).items():
            acc[(h.split("/")[1].split("_")[0], k2)] += s_h * np.array([a, b])
tasks = sorted({t for t, _ in acc})
for t in tasks:
    tot = sum(v for (tt, _), v in acc.items() if tt == t)
    print(f"{t}: " + ", ".join(f"{p} {v[0]/tot[0]:.3f}->{v[1]/tot[1]:.3f}" for (tt, p), v in acc.items() if tt == t))
if missing: print("WARNING no phase sidecar (uniform):", missing)
