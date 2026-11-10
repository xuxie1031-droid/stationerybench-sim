"""Known-answer tests for mix_switch through the real IterableDatasetMixture.__iter__ (CPU)."""
import os, sys  # noqa: E401  repo root on sys.path -> gyre_paths (scripts/env.sh does the same)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import gyre_paths as GP  # noqa: E402
import os, sys, numpy as np
sys.path.insert(0, os.path.join(GP.MOLMOACT2, "experiments")); sys.path.insert(0, GP.YAM_FT)
from olmo.data.iterable_dataset_mixture import IterableDatasetMixture as M
class Inner:
    def __init__(s, r, n): s.r, s.n = r, n
    def _get_repo_id(s): return s.r
    def __len__(s): return s.n
class DS:
    def __init__(s, r, n): s.dataset = Inner(r, n)
    def __len__(s): return len(s.dataset)
    def get(s, i, epoch=0): return (s.dataset.r, i)
dss = [DS("sim/a", 5000), DS("sim/b", 3000), DS("real/x", 4000), DS("real/y", 2000)]
rates = [0.3, 0.2, 0.3, 0.2]; G = 16; STEP = 500; N = 20000
def draws():
    it = iter(M(dss, global_batch_size=G, mixture_rates=rates, seed=50189, world_size=1, rank=0)); return [next(it) for _ in range(N)]
base = draws()
os.environ["YAM_FT_MIX_SWITCH"] = f"{STEP}:0.6667"
import mix_switch; mix_switch.install()
sw = draws(); K = STEP * G; ok = True
a = sw[:K] == base[:K]; ok &= a
print(f"(a) first {K} examples (steps < {STEP}) bitwise identical to stock: {a}")
real = lambda seq: np.mean([r.startswith("real/") for r, _ in seq])
pre, post = real(sw[:K]), real(sw[K:])
b = abs(pre - 0.5) < 0.02 and abs(post - 0.6667) < 0.01; ok &= b
print(f"(b) real share before {pre:.4f} (0.5), after {post:.4f} (0.6667): {b}")
within = lambda seq, x, y: sum(r == x for r, _ in seq) / max(1, sum(r in (x, y) for r, _ in seq))
c = abs(within(sw[K:], "sim/a", "sim/b") - 0.6) < 0.02 and abs(within(sw[K:], "real/x", "real/y") - 0.6) < 0.02; ok &= c
print(f"(c) within-group proportions kept after switch (sim a/(a+b) {within(sw[K:],'sim/a','sim/b'):.3f}, real x/(x+y) {within(sw[K:],'real/x','real/y'):.3f}; expected 0.6): {c}")
d = sw[K:] != base[K:]; ok &= d
print(f"(d) stream after the switch differs from stock (the switch is active): {d}")
print("ALL PASS" if ok else "SOME FAILED")
