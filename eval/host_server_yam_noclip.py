"""Variant of examples/yam/host_server_yam.py whose action un-normalisation does NOT clip the normalised action to
[-1,1] (for checkpoints trained with YAM_FT_NOCLIP=1).  The original server file is untouched: this imports it and
rewrites only `_FeatureNormalizer.unnormalize` of the loaded remote-code module, with the single clip line removed.
State normalisation is left as is (the discrete state tokenizer clips to [-1,1] itself, so a state no-clip would be a
no-op).  Same CLI / wire format as host_server_yam.py.
  $SIM_PY eval/host_server_yam_noclip.py --repo-id <hf dir> --port 8231   (after source scripts/env.sh)
"""
import os, sys  # noqa: E401  repo root on sys.path -> gyre_paths (scripts/env.sh does the same)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gyre_paths as GP  # noqa: E402
import inspect, os, sys, textwrap

sys.path.insert(0, os.path.join(GP.MOLMOACT2, "examples", "yam"))
import host_server_yam as H  # noqa: E402

CLIP_LINE = "arr = np.clip(arr, -1.0, 1.0)"


def patch_noclip(model):
    mod = sys.modules[type(model).__module__]
    cls = mod._FeatureNormalizer
    if getattr(cls, "_yam_ft_noclip", False):
        return
    src = textwrap.dedent(inspect.getsource(cls.unnormalize))
    if src.count(CLIP_LINE) != 1:
        raise RuntimeError(f"noclip patch: expected exactly one '{CLIP_LINE}' in {mod.__name__}._FeatureNormalizer.unnormalize")
    src = src.replace(CLIP_LINE, "pass  # yam_ft noclip")
    ns = {}
    exec(compile(src, f"<noclip {mod.__name__}>", "exec"), mod.__dict__, ns)
    cls.unnormalize = ns["unnormalize"]
    cls._yam_ft_noclip = True
    H.log.info("noclip: patched %s._FeatureNormalizer.unnormalize (no [-1,1] clip)", mod.__name__)


_orig_init = H.Policy.__init__


def _init(self, *a, **k):
    _orig_init(self, *a, **k)
    patch_noclip(self.model)


H.Policy.__init__ = _init
H.Policy.noclip = True

if __name__ == "__main__":
    H.main()
