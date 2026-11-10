"""Known-answer tests for yam_sched (CPU, no trainer): builds the scheduler exactly as the trainer does
(SchedulerConfig(...).build() after install) and evaluates lr at the steps the trainer uses (1..S)."""
import os, sys  # noqa: E401  repo root on sys.path -> gyre_paths (scripts/env.sh does the same)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import gyre_paths as GP  # noqa: E402
import os, sys
sys.path.insert(0, os.path.join(GP.MOLMOACT2, "experiments")); sys.path.insert(0, GP.YAM_FT)
import olmo.train.optim as O
import yam_sched
def cfg(**w):
    base = dict(name=O.SchedulerType.multimodal, connector_t_warmup=200, vit_t_warmup=200, llm_t_warmup=200,
                action_expert_t_warmup=200, alpha_f=0.1, warmup_min_lr=0.0)
    base.update(w); return O.SchedulerConfig(**base)
ok = True
# stock cosine (phase 1) final lr = alpha_f * base at step S
stock = cfg(action_expert_t_warmup=500).build()
p1_final = stock.get_lr(2e-5, 5600, 5600, "action_expert")
print(f"phase-1 stock cosine: AE base 2e-5 -> lr at last step 5600 = {p1_final:.3e} (expected 2.000e-06)")
# (1) continuation
os.environ.update(YAM_FT_SCHED="cont", CONT_END_FRAC="0.01"); yam_sched.install()
sc = cfg(action_expert_t_warmup=100).build(); S = 1400
for g, base in (("action_expert", 2e-6), ("llm", 5e-6), ("vit", 5e-6), ("connector", 5e-6)):
    v = {s: sc.get_lr(base, s, S, g) for s in (0, 1, 2, 10, S // 2, S - 1, S)}
    start_ok = v[1] == base and v[0] == base; mono = v[2] < v[1]; end_ok = v[S] <= 0.01 * base * (1 + 1e-12)
    ok &= start_ok and mono and end_ok
    print(f"cont {g:13s} base {base:.1e}: " + " ".join(f"s{k}={x:.4e}" for k, x in v.items()) +
          f" | starts at base: {start_ok}, no warmup ramp (s2<s1): {mono}, end <=1%: {end_ok} ({v[S]/base:.4f})")
# (2) WSD 100-step dry schedule, warmup 10, decay last 20%, linear to 0
os.environ.update(YAM_FT_SCHED="wsd", WSD_DECAY_FRAC="0.2", WSD_END_FRAC="0", WSD_SHAPE="linear")
w = cfg(action_expert_t_warmup=10).build(); S = 100; base = 1e-5
curve = [w.get_lr(base, s, S, "action_expert") for s in range(0, S + 1)]
chk = {"warmup s5 (=0.5 base)": (curve[5], 0.5 * base), "warmup end s10": (curve[10], base), "mid-stable s50": (curve[50], base),
       "decay start s80": (curve[80], base), "decay mid s90": (curve[90], 0.5 * base), "end s100": (curve[100], 0.0)}
for k, (got, exp) in chk.items():
    good = abs(got - exp) <= 1e-15; ok &= good
    print(f"wsd {k:24s} lr={got:.4e} expected {exp:.4e}: {good}")
print("wsd curve every 5 steps:", " ".join(f"{curve[s]/base:.2f}" for s in range(0, S + 1, 5)))
os.environ["WSD_SHAPE"] = "sqrt"; ws = cfg(action_expert_t_warmup=10).build()
print("wsd 1-sqrt  every 5 :", " ".join(f"{ws.get_lr(base, s, S, 'action_expert')/base:.2f}" for s in range(0, S + 1, 5)))
os.environ.update(WSD_SHAPE="linear", WSD_DECAY_FRAC="0"); bad = cfg(action_expert_t_warmup=10).build()
badc = [bad.get_lr(base, s, S, "action_expert") for s in range(0, S + 1)]
differs = badc[100] == base and badc[90] == base; ok &= differs
print("wrong config decay_frac=0 every 5:", " ".join(f"{x/base:.2f}" for x in badc[::5]), f"| visibly different (no decay, ends at peak): {differs}")
print("ALL PASS" if ok else "SOME FAILED")
