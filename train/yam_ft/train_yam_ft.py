"""Thin wrapper around experiments/launch_scripts/train_lerobot.py for MolmoAct2-BimanualYAM post-training.

Registers mixture 'yam_ft' = the repos in $YAM_FT_REPOS (comma separated, paths relative to $LEROBOT_DATA_ROOT,
optional ':weight' per repo = its share of sampled training examples, e.g.
YAM_FT_REPOS=sim/efb_v4n2_p0:0.25,sim/rh_vn2_p0:0.25,real/*:0.5 ; without weights the loader's
default applies: share proportional to sqrt(#frames)),
all under the SAME tag as the checkpoint ('yam_dual_molmoact2': same setup_type/control_mode prompt tokens, same
camera keys, action_horizon=n_action_steps=30, grippers not normalised).  Stats of all repos in the tag are merged,
so every repo must carry the pretrained stats (sim_to_lerobot.py --stats pretrained) to keep the checkpoint's
normalisation.  Everything after that is train_lerobot.py's CLI unchanged.

Extra mode: YAM_FT_DUMP=<out.npz> -> do not train; build the training mixture exactly as the trainer would and dump
raw wrapper examples + preprocessed tensors for YAM_FT_DUMP_INDICES (default 0,100,...) of every repo.
"""
import os, sys  # noqa: E401  repo root on sys.path -> gyre_paths (scripts/env.sh does the same)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import gyre_paths as GP  # noqa: E402
import os, sys

# torchcodec must load its FFmpeg libs BEFORE decord2 (imported by olmo.data.video_loader): decord's bundled FFmpeg
# otherwise interposes and every AV1 decode fails with "Could not push packet to decoder: Invalid data".
import torchcodec.decoders  # noqa: F401,E402

E = os.environ.get("E", os.path.join(GP.MOLMOACT2, "experiments"))
sys.path.insert(0, E)
os.chdir(E)

from launch_scripts import data_mixtures as dm  # noqa: E402

# Per-dataset clip mode of the normalised state/action (pretrained q01/q99 scale is kept in all modes):
#   YAM_FT_CLIP="real/*:clip,ref/*:clip,sim/*:noclip"   (fnmatch on repo id; first match wins; unmatched -> clip)
#   modes: clip (= pretraining, [-1,1]) | noclip | bound=X ([-X,X])      legacy YAM_FT_NOCLIP=1 == "*:noclip"
# Implementation: the normaliser's own [-1,1] clip is disabled and RobotProcessor.normalize_action/_state re-apply the
# per-repo clip on the normalised (mask=True) dims only -> 'clip' is bitwise identical to the original code path.
# Applied at import so spawned dataloader workers get it.  The discrete state tokenizer still saturates at [-1,1].
_CLIP_SPEC = os.environ.get("YAM_FT_CLIP") or ("*:noclip" if os.environ.get("YAM_FT_NOCLIP") == "1" else "")
if _CLIP_SPEC:
    import fnmatch as _fn
    import numpy as _np
    import torch as _torch
    import olmo.data.robot_processing as _rp  # noqa: E402

    _RULES = []
    for _item in _CLIP_SPEC.split(","):
        _pat, _, _mode = _item.strip().rpartition(":")
        if _mode.startswith("bound="):
            _lim = float(_mode.split("=", 1)[1])
        elif _mode == "clip":
            _lim = 1.0
        elif _mode == "noclip":
            _lim = None
        else:
            raise ValueError(f"YAM_FT_CLIP: bad mode {_mode!r} in {_item!r}")
        _RULES.append((_pat, _lim))

    def _limit_for(repo_id):
        for pat, lim in _RULES:
            if _fn.fnmatch(str(repo_id), pat):
                return lim
        return 1.0

    _rp._uses_bounded_normalized_range = lambda mode: False

    def _reclip(self, normalizers, out, repo_id):
        lim = _limit_for(repo_id)
        nz = self._get_normalizer(normalizers, repo_id)
        if out is None or lim is None or nz is None or nz.mode not in {"min_max", "q01_q99", "q10_q90"}:
            return out
        arr = out.detach().cpu().numpy() if _torch.is_tensor(out) else _np.asarray(out)
        c = _np.clip(arr, -lim, lim)
        if nz.mask is not None:
            c = _np.where(nz.mask, c, arr)
        if nz.zero_mask is not None:
            c = _np.where(nz.zero_mask, 0.0, c)
        if _torch.is_tensor(out):
            return _torch.as_tensor(c, device=out.device, dtype=out.dtype)
        return c.astype(arr.dtype, copy=False)

    _orig_na, _orig_ns = _rp.RobotProcessor.normalize_action, _rp.RobotProcessor.normalize_state
    _rp.RobotProcessor.normalize_action = lambda self, a, repo_id: _reclip(self, self.action_normalizers, _orig_na(self, a, repo_id), repo_id)
    _rp.RobotProcessor.normalize_state = lambda self, s, repo_id: _reclip(self, self.state_normalizers, _orig_ns(self, s, repo_id), repo_id)
    print("yam_ft clip rules:", _RULES, flush=True)


# Opt-in phase-weighted sampling inside sim datasets (YAM_FT_PHASE_W, see phase_sampling.py); installed at import so
# the spawned dataloader workers (where IterableDatasetMixture._get_order runs) get it too.
if os.environ.get("YAM_FT_PHASE_W") or os.environ.get("YAM_FT_PHASE_TRANS"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import phase_sampling as _ps  # noqa: E402
    _ps.install()


# Opt-in mid-run data-mix switch (YAM_FT_MIX_SWITCH="<step>:<real_share>", see mix_switch.py); at import so the
# dataloader workers (where IterableDatasetMixture runs) get it.
if os.environ.get("YAM_FT_MIX_SWITCH"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mix_switch as _mix_switch  # noqa: E402
    _mix_switch.install()


class _Weighted(tuple):
    """(dataset_name, root_size_factor) entry; str() gives the name so train_plan's family check accepts it."""
    def __str__(self):
        return self[0]


def parse_repos(spec):
    """'sim/a:0.25,sim/b:0.25,real/*:0.5' -> [(repo, weight)]; weight omitted -> loader default (sqrt(len)).
    A glob entry expands to all datasets under $LEROBOT_DATA_ROOT matching it; its weight is split between them
    proportionally to their frame counts (i.e. uniform over real frames)."""
    import glob, json
    root = os.environ["LEROBOT_DATA_ROOT"]
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        repo, _, w = item.partition(":")
        w = float(w) if w else None
        if any(ch in repo for ch in "*?["):
            hits = sorted(os.path.relpath(os.path.dirname(os.path.dirname(p)), root)
                          for p in glob.glob(os.path.join(root, repo, "meta", "info.json")))
            # converted sim datasets are complete only once the converter wrote meta/yam_ft_source.json
            hits = [h for h in hits if not h.startswith(("sim/", "hold/"))
                    or os.path.exists(os.path.join(root, h, "meta", "yam_ft_source.json"))]
            if not hits:
                raise ValueError(f"YAM_FT_REPOS: no dataset matches {repo} under {root}")
            n = [json.load(open(os.path.join(root, h, "meta", "info.json")))["total_frames"] for h in hits]
            out += [(h, None if w is None else w * k / sum(n)) for h, k in zip(hits, n)]
        else:
            out.append((repo, w))
    print("yam_ft mixture:", [(r, None if w is None else round(w, 4)) for r, w in out], flush=True)
    return out


def build_yam_ft():
    repos = parse_repos(os.environ["YAM_FT_REPOS"])
    mixture, meta = dm.build_single_lerobot_mixture(
        name="yam_ft", tag="yam_dual_molmoact2", repo_ids=[r for r, _ in repos], action_key="action",
        state_keys=["observation.state"],
        camera_keys=["observation.images.top", "observation.images.left", "observation.images.right"],
        normalize_gripper=False, setup_type="bimanual yam robotic arms in molmoact2",
        control_mode="absolute joint pose", action_horizon=30, n_action_steps=30)
    if any(w is not None for _, w in repos):
        if any(w is None for _, w in repos):
            raise ValueError("YAM_FT_REPOS: give a weight to every repo or to none")
        tot = sum(w for _, w in repos)
        # data_loader: per-dataset rate = sqrt(root_size_factor) (for factor >= 1) -> factor = (1e3 * share)^2
        # CAVEAT (kept as run; see docs/TRAINING.md "Configured vs. effective mixture"): for a share < 0.1 % the factor is
        # < 1 and MolmoAct2's data_loader then uses sqrt(len(dataset) * factor) instead, i.e. that dataset's rate is
        # multiplied by sqrt(#frames).  The effective mixture of every round is printed under "Sampling rates:" at startup.
        ents = [_Weighted((f"lerobot:{r}", (1e3 * w / tot) ** 2)) for r, w in repos]
        mixture = [(mixture[0][0], ents, mixture[0][2])]
    return mixture, meta


dm.MOLMOACT2_LEROBOT_MIXTURES["yam_ft"] = build_yam_ft


def dump(conf):
    import numpy as np
    import torch
    out = os.environ["YAM_FT_DUMP"]
    idx = [int(x) for x in os.environ.get("YAM_FT_DUMP_INDICES", "0,100,500,1000,2000").split(",")]
    preprocessor = conf.data._override_crop_mode(conf.model).build_preprocessor(
        for_inference=False, is_training=True, text_seq_len=conf.data.max_text_seq_len,
        max_seq_len=conf.data.sequence_length, include_image=False)
    datasets, rates = conf.data._build_mixture(preprocessor)
    res = {}
    if os.environ.get("YAM_FT_DEBUG_DECODE"):
        import lerobot.datasets.video_utils as vu, traceback
        for p in os.environ["YAM_FT_DEBUG_DECODE"].split(","):
            try:
                print("DEBUG decode", p, vu.decode_video_frames_torchcodec(p, [0.0], 1e-4).shape, flush=True)
            except Exception:
                traceback.print_exc()
    for dd in datasets:
        w = dd.dataset
        name = w._get_repo_id().replace("/", "__")
        for i in idx:
            if i >= len(w):
                continue
            ex = w.get(i, np.random.RandomState(1234), allow_random_retry=False)
            pre = preprocessor(dict(ex), np.random.RandomState(1234)) if callable(preprocessor) else None
            p = f"{name}/{i}/"
            res[p + "state"] = np.asarray(ex.get("state")) if ex.get("state") is not None else np.zeros(0)
            res[p + "action"] = np.asarray(ex["action"])
            for k in ("action_horizon_is_pad", "action_dim_is_pad"):
                if k in ex:
                    res[p + k] = np.asarray(ex[k])
            imgs = ex["image"] if isinstance(ex["image"], list) else [ex["image"]]
            for j, im in enumerate(imgs):
                res[p + f"image{j}"] = np.asarray(im)
            res[p + "prompt"] = np.array(repr(ex.get("messages", ex.get("question"))))
            res[p + "meta"] = np.array(repr({k: ex["metadata"].get(k) for k in ("episode_index", "frame_index", "task", "camera_keys_used")}))
            if isinstance(pre, dict):
                for k, v in pre.items():
                    if isinstance(v, (np.ndarray, torch.Tensor)):
                        res[p + "pre_" + k] = np.asarray(v)
    np.savez(out, **res)
    print("DUMPED", out, len(res), "arrays", flush=True)


def bench(conf):
    """YAM_FT_BENCH=N: time N fully preprocessed samples (DeterministicDataset.get = frame decode + robot/text/image
    preprocessing, i.e. one dataloader item) per dataset of the mixture, on CPU; no training."""
    import time
    import numpy as np
    n = int(os.environ["YAM_FT_BENCH"])
    preprocessor = conf.data._override_crop_mode(conf.model).build_preprocessor(
        for_inference=False, is_training=True, text_seq_len=conf.data.max_text_seq_len,
        max_seq_len=conf.data.sequence_length, include_image=False)
    datasets, rates = conf.data._build_mixture(preprocessor)
    rng = np.random.default_rng(0)
    for dd in datasets:
        name = dd.dataset._get_repo_id()
        idx = rng.integers(0, len(dd), n + 1)
        dd.get(int(idx[0]), 0)
        t0 = time.time(); tf = 0.0
        for i in idx[1:]:
            t1 = time.time(); dd.dataset.get(int(i), np.random.RandomState(0)); tf += time.time() - t1
            dd.get(int(i), 0)
        tot = (time.time() - t0) / n
        print(f"BENCH {name:34s} wrapper.get (decode+state/action) {tf/n*1000:6.1f} ms + full item {(tot - tf/n)*1000:6.1f} ms", flush=True)


def _install_param_hash():
    """YAM_FT_PARAM_HASH=1: after Trainer.fit, every rank prints a sha256 over its local copy of all trainable params
    (with replicate-only HSDP the local copy is the full parameter -> hashes must be identical across ranks)."""
    import hashlib
    import torch
    import torch.distributed as dist
    import olmo.train.trainer as T
    orig = T.Trainer.fit

    def fit(self):
        out = orig(self)
        h, n = hashlib.sha256(), 0
        for name, p in sorted(self.fsdp_model.named_parameters(), key=lambda x: x[0]):
            if not p.requires_grad:
                continue
            t = p.to_local() if hasattr(p, "to_local") else p
            h.update(name.encode()); h.update(t.detach().float().cpu().contiguous().numpy().tobytes()); n += t.numel()
        r = dist.get_rank() if dist.is_initialized() else 0
        print(f"PARAM_HASH rank={r} trainable_local_numel={n} sha256={h.hexdigest()}", flush=True)
        return out
    T.Trainer.fit = fit


def _install_accum_nosync():
    """YAM_FT_ACCUM_NOSYNC=1: with gradient accumulation, skip the FSDP2 gradient reduction on all but the last
    micro-batch of a step (FSDPModule.set_requires_gradient_sync), so multi-node replicate-only runs all-reduce the
    trainable grads once per optimizer step instead of once per micro-batch."""
    import olmo.train.trainer as T
    orig_split, orig_fwd = T.Trainer.split_batch, T.Trainer.model_forward

    def split_batch(self, batch):
        mbs = orig_split(self, batch)
        self._yam_mb_n, self._yam_mb_i = len(mbs), 0
        return mbs

    def model_forward(self, *a, **k):
        n = getattr(self, "_yam_mb_n", None)
        if n is not None and hasattr(self.fsdp_model, "set_requires_gradient_sync"):
            self._yam_mb_i = getattr(self, "_yam_mb_i", 0) + 1
            self.fsdp_model.set_requires_gradient_sync(self._yam_mb_i >= n)
        return orig_fwd(self, *a, **k)
    T.Trainer.split_batch, T.Trainer.model_forward = split_batch, model_forward


if __name__ == "__main__":  # guard: dataloader workers are spawned and re-import this module
    import launch_scripts.train_lerobot as tl
    if os.environ.get("YAM_FT_PARAM_HASH") == "1":
        _install_param_hash()
    if os.environ.get("YAM_FT_ACCUM_NOSYNC") == "1":
        _install_accum_nosync()
    if os.environ.get("YAM_FT_EMA"):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import ema as _ema
        _ema.install()
    if os.environ.get("YAM_FT_SCHED"):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import yam_sched as _sched
        _sched.install()
    if os.environ.get("YAM_FT_DUMP"):
        tl.run_trainer = dump
    if os.environ.get("YAM_FT_BENCH"):
        tl.run_trainer = bench
    tl.main()
