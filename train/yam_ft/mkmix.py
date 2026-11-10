"""Build a YAM_FT_REPOS mixture string from a compact spec (prints it).
  python mkmix.py --tasks efb=1,rh=1 --variants v2=1,vnom=1,vdr=1,v3=1 --sim-share 0.5 [--real 'real/*']
- task weights split the sim share; variant weights split each task's share; variant v2 = as-built demos_v2
  (sim/<t>_v2 or sim/<t>_v2_p*), vnom/vdr = visual nominal / DR re-renders (sim/<t>_vnom_p*, sim/<t>_vdr*_p*),
  v3 = DART demos (sim/<t>_v3*, as named when converted).  Missing variants are dropped and their weight
  re-distributed within the task.  --sim-share 1 = sim-only arm."""
import os, sys  # noqa: E401  repo root on sys.path -> gyre_paths (scripts/env.sh does the same)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import gyre_paths as GP  # noqa: E402
import argparse, glob, os
ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default="efb=1,rh=1"); ap.add_argument("--variants", default="v2=1,vnom=1,vdr=1,v3=1")
ap.add_argument("--sim-share", type=float, default=0.5); ap.add_argument("--real", default="real/*")
ap.add_argument("--src-share", default="", help="per-task data-source shares, ';'-separated task:spec, 'default' for "
                "all others, e.g. 'default:rec=0.4,v2=0.4,v3=0.2;rh:rec=0.4,v2=0.5,v3=0.1'; each source is split into "
                "visual variants by --vis.  Overrides --variants/--v3-share.  Sources: v2 (vnom/vdr), v3 (v3nom/v3dr), "
                "rec (recnom/recdr), raw (as-built v2).")
ap.add_argument("--vis", default="nom=0.4,dr=0.6", help="visual split inside each source for --src-share")
ap.add_argument("--v3-share", default="", help="per-task share of v3 DART data inside the task, e.g. efb=0.2,rh=0.1; "
                "v3 is then split v3nom/v3dr like vnom/vdr in --variants and the v2 variants share the rest")
a = ap.parse_args(); root = os.environ.get("LEROBOT_DATA_ROOT", GP.LEROBOT_DATA)
kv = lambda s: [(k, float(v)) for k, v in (x.split("=") for x in s.split(",") if x)]
PAT = {"v2": ["{t}_v2", "{t}_v2_p*"], "vnom": ["{t}_vnom*"], "vdr": ["{t}_vdr*"], "v3": ["{t}_v3*"]}
tasks = kv(a.tasks); tw = sum(w for _, w in tasks); out = []
v3s = dict(kv(a.v3_share)) if a.v3_share else None
if v3s:  # rebuild per-task variant weights: v2 part from --variants (vnom/vdr), v3 part mirrors its nom/dr split
    base = [(v, w) for v, w in kv(a.variants) if not v.startswith("v3")]
    bt = sum(w for _, w in base)
for t, w in tasks:
    vs = []
    vlist = kv(a.variants)
    if a.src_share:
        specs = dict(x.split(":", 1) for x in a.src_share.split(";") if x)
        ss = kv(specs.get(t, specs.get("default", "")))
        PFX = {"v2": "v", "v3": "v3", "rec": "rec", "v4": "v4", "rec3": "rec3", "rec4": "rec4", "rec5": "rec5", "rec6": "rec6", "rec7": "rec7"}  # v4 = eraser prop demos on the spring lid
        vlist = []
        for src, sw in ss:
            if src == "raw":
                vlist.append(("v2", sw)); continue
            for vis, vw in kv(a.vis):
                vlist.append((PFX[src] + vis, sw * vw))
    if v3s:
        f3 = v3s.get(t, 0.0)
        vlist = [(v, (1 - f3) * w / bt) for v, w in base] + [("v3" + v[1:], f3 * w / bt) for v, w in base]
    for v, vw in vlist:
        pats = [f"sim/{p.format(t=t)}" for p in PAT.get(v, [f"{{t}}_{v}*"])]
        # only finished conversions (converter writes meta/yam_ft_source.json last)
        hit = [p for p in pats if glob.glob(os.path.join(root, p, "meta", "yam_ft_source.json"))]
        if hit and vw > 0: vs.append((hit[0], vw))
    vt = sum(vw for _, vw in vs)
    out += [(p, a.sim_share * w / tw * vw / vt) for p, vw in vs]
if a.sim_share < 1:
    out.append((a.real, 1 - a.sim_share))
print(",".join(f"{p}:{w:.4f}" for p, w in out))
