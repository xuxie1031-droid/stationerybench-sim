"""Phase-weighted sampling inside sim datasets (opt-in; real datasets keep the stock uniform shuffle).

  YAM_FT_PHASE_W="grasp:3,place:2,default:1"   weights by phase name (approach/grasp/lift/handover/place/hold),
                                                 rubric stage ('s0'..'s3', 'hold') also accepted; 'default' = rest
  YAM_FT_PHASE_TRANS="15:2"   optional: frames within 15 of a stage change get an extra x2 (multiplies)
  YAM_FT_PHASE_REPOS="sim/*"  fnmatch patterns (comma) of repos the weights apply to (default sim/*)
Weights come from the sidecar meta/yam_ft_phase.npz (phase_labels.py).  Mechanism: IterableDatasetMixture._get_order
(per dataset, per epoch) returns len(dataset) indices drawn WITH replacement with p ~ frame weight from the same
seeded RNG stream; the mixture-level dataset choice (task / real shares) is untouched.  If all of a dataset's frame
weights are equal (or the repo does not match / has no sidecar), the stock shuffle is returned unchanged.
"""
import os, sys  # noqa: E401  repo root on sys.path -> gyre_paths (scripts/env.sh does the same)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import gyre_paths as GP  # noqa: E402
import fnmatch, json, logging, os
import numpy as np

log = logging.getLogger(__name__)
HOLD = 5


def parse_weights(spec):
    w = {}
    for item in (spec or "").split(","):
        if item.strip():
            k, v = item.split(":"); w[k.strip()] = float(v)
    return w


def frame_weights(sidecar, wspec, trans=None):
    z = np.load(sidecar, allow_pickle=True)
    stage = z["stage"].astype(np.int64); names = {int(k): v for k, v in json.loads(str(z["names"])).items()}
    default = wspec.get("default", 1.0)
    per_stage = {}
    for s, nm in names.items():
        key_s = "hold" if s == HOLD else f"s{s}"
        per_stage[s] = wspec.get(nm, wspec.get(key_s, default))
    w = np.array([per_stage.get(int(s), default) for s in range(HOLD + 1)])[stage]
    if trans:
        win, mult = trans
        w = w * np.where(z["dist_to_transition"] <= win, mult, 1.0)
    return w.astype(np.float64), stage, names


def phase_share(stage, names, w):
    tot_n, tot_w = len(stage), float(w.sum())
    return {names[s]: (float((stage == s).mean()), float(w[stage == s].sum() / tot_w) if tot_w > 0 else 0.0)
            for s in sorted(names)}


def install():
    from olmo.data.iterable_dataset_mixture import IterableDatasetMixture as M
    if getattr(M, "_yam_phase_installed", False):
        return
    wspec = parse_weights(os.environ.get("YAM_FT_PHASE_W"))
    trans = os.environ.get("YAM_FT_PHASE_TRANS")
    trans = (int(trans.split(":")[0]), float(trans.split(":")[1])) if trans else None
    pats = [p.strip() for p in os.environ.get("YAM_FT_PHASE_REPOS", "sim/*").split(",") if p.strip()]
    root = os.environ.get("LEROBOT_DATA_ROOT", GP.LEROBOT_DATA)
    orig = M._get_order
    cache = {}

    def weights_for(self, ix):
        if ix in cache:
            return cache[ix]
        ds = self.datasets[ix]; inner = getattr(ds, "dataset", ds)
        repo = inner._get_repo_id() if hasattr(inner, "_get_repo_id") else getattr(inner, "repo_id", None)
        p = None
        if repo and any(fnmatch.fnmatch(repo, pt) for pt in pats):
            sc = os.path.join(root, repo, "meta", "yam_ft_phase.npz")
            if os.path.exists(sc):
                w, stage, names = frame_weights(sc, wspec, trans)
                n = len(ds)
                if len(w) != len(inner):
                    raise RuntimeError(f"phase sidecar of {repo} has {len(w)} frames, dataset has {len(inner)}")
                if n > len(w):  # DeterministicDataset padding entries: mean weight
                    w = np.concatenate([w, np.full(n - len(w), w.mean())])
                if not np.all(w == w[0]):
                    p = w / w.sum()
                share = phase_share(stage, names, w[:len(stage)])
                print(f"PHASE_SAMPLING pid={os.getpid()} {repo}: " + str({k: f"{a:.3f}->{b:.3f}" for k, (a, b) in share.items()}), flush=True)
            else:
                log.warning("phase sampling: no sidecar for %s (uniform)", repo)
        cache[ix] = p
        return p

    def _get_order(self, dataset_ix, epoch):
        p = weights_for(self, dataset_ix)
        if p is None:
            return orig(self, dataset_ix, epoch)
        rs = np.random.RandomState(self.seed + epoch * 1771)
        return rs.choice(len(p), size=len(p), replace=True, p=p).astype(np.int32)

    M._get_order = _get_order
    M._yam_phase_installed = True
    print(f"PHASE_SAMPLING installed pid={os.getpid()}: weights {wspec} trans {trans} repos {pats}", flush=True)
