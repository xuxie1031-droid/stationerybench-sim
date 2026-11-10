"""CPU known-answer tests for phase-weighted sampling through the real IterableDatasetMixture.__iter__.
  python test_phase_sampling.py [N_DRAWS=300000]"""
import os, sys  # noqa: E401  repo root on sys.path -> gyre_paths (scripts/env.sh does the same)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import gyre_paths as GP  # noqa: E402
import os, sys, numpy as np
sys.path.insert(0, GP.YAM_FT)
E = os.path.join(GP.MOLMOACT2, "experiments"); sys.path.insert(0, E)
from olmo.data.iterable_dataset_mixture import IterableDatasetMixture as M
import phase_sampling as PS

ROOT = os.environ.setdefault("LEROBOT_DATA_ROOT", GP.LEROBOT_DATA)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300000


class Inner:
    def __init__(self, repo, n): self.repo, self.n = repo, n
    def _get_repo_id(self): return self.repo
    def __len__(self): return self.n


class DS:  # minimal DeterministicDataset stand-in: get() returns (repo, frame index)
    def __init__(self, repo, n): self.dataset = Inner(repo, n)
    def __len__(self): return len(self.dataset)
    def get(self, i, epoch=0): return (self.dataset.repo, i)


def nframes(repo):
    return len(np.load(f"{ROOT}/{repo}/meta/yam_ft_phase.npz")["stage"])


repos = ["sim/efb_v4n2_p0", "sim/rh_v2_p0"]
dss = [DS(r, nframes(r)) for r in repos] + [DS("real/fake", 50000)]
rates = [0.25, 0.25, 0.5]


def draws():
    it = iter(M(dss, global_batch_size=None, mixture_rates=rates, seed=50189, world_size=1, rank=0))
    return [next(it) for _ in range(N)]


orig = M._get_order


def reset():
    M._get_order = orig; M._yam_phase_installed = False


def run(spec, trans=None):
    reset(); os.environ["YAM_FT_PHASE_W"] = spec
    os.environ.pop("YAM_FT_PHASE_TRANS", None)
    if trans: os.environ["YAM_FT_PHASE_TRANS"] = trans
    PS.install(); return draws()


base = draws()  # stock sampler
ok = True
# (a) all weights 1 -> identical order
a = run("default:1")
ok_a = a == base; ok &= ok_a
print(f"(a) weights all 1: sample sequence identical to stock sampler over {N} draws: {ok_a}")
labels = {r: np.load(f"{ROOT}/{r}/meta/yam_ft_phase.npz", allow_pickle=True) for r in repos}
import json
def shares(seq, r):
    st = labels[r]["stage"]; names = {int(k): v for k, v in json.loads(str(labels[r]["names"])).items()}
    idx = np.array([i for rr, i in seq if rr == r]); s = st[idx]
    return {names[k]: float((s == k).mean()) for k in sorted(names)}, len(idx)
# (b) grasp weight 0 -> no grasp samples
b = run("grasp:0,default:1")
for r in repos:
    sh, n = shares(b, r); ok_b = sh["grasp"] == 0.0; ok &= ok_b
    print(f"(b) grasp:0 -> {r}: {n} draws, grasp share {sh['grasp']:.5f} (must be 0): {ok_b}")
# (c) empirical vs expected share
spec = "grasp:3,place:2,default:1"
c = run(spec)
for r in repos:
    w, st, names = PS.frame_weights(f"{ROOT}/{r}/meta/yam_ft_phase.npz", PS.parse_weights(spec))
    exp = PS.phase_share(st, names, w)
    sh, n = shares(c, r)
    worst = max(abs(sh[k] - exp[k][1]) for k in sh)
    ok_c = worst < 0.01; ok &= ok_c
    print(f"(c) {spec} {r}: {n} draws; per-phase share frames->expected->empirical: " +
          ", ".join(f"{k} {exp[k][0]:.3f}->{exp[k][1]:.3f}->{sh[k]:.3f}" for k in sh) + f" | max |emp-exp| {worst:.4f} <0.01: {ok_c}")
# real data untouched under weighting: same real draws as stock
real_same = [x for x in c if x[0] == "real/fake"] == [x for x in base if x[0] == "real/fake"]
ok &= real_same
print(f"real dataset: identical draws with and without phase weights: {real_same}")
# mixture-level shares unchanged
for nm, seq in (("stock", base), ("weighted", c)):
    print(f"dataset shares {nm}:", {r: round(sum(1 for x in seq if x[0] == r) / N, 4) for r in repos + ['real/fake']})
# transition window option
t = run("default:1", trans="15:3")
for r in repos:
    d = labels[r]["dist_to_transition"]; idx = np.array([i for rr, i in t if rr == r]); i0 = np.array([i for rr, i in base if rr == r])
    print(f"trans 15:3 {r}: share of draws within 15 frames of a stage change {np.mean(d[i0] <= 15):.3f} -> {np.mean(d[idx] <= 15):.3f}")
print("ALL PASS" if ok else "SOME FAILED")
