"""CPU known-answer tests for yam_ft/ema.py's EMA class on a toy model trained with AdamW (20 steps).
(a) decay 0 == live weights bitwise; (b) decay 1 == initial weights bitwise; (c) decay 0.9 matches the float64
closed form; swap() twice restores the live weights bitwise; frozen params are not tracked; a deliberately wrong
update order (EMA before optimizer step) must FAIL check (a)."""
import os, sys  # noqa: E401  repo root on sys.path -> gyre_paths (scripts/env.sh does the same)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import gyre_paths as GP  # noqa: E402
import sys, torch
sys.path.insert(0, GP.YAM_FT)
from ema import EMA

def run(decay, steps=20, wrong_order=False):
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4))
    m[0].bias.requires_grad_(False)  # a frozen param
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    init = {n: p.detach().clone() for n, p in m.named_parameters()}
    ema = EMA(m.named_parameters(), decay, device="cpu", trace=8)
    for _ in range(steps):
        x = torch.randn(8, 16); loss = (m(x) ** 2).mean(); opt.zero_grad(); loss.backward()
        if wrong_order: ema.update(); opt.step()
        else: opt.step(); ema.update()
    return m, init, ema

ok = True
m, init, e = run(0.0)
a = all(torch.equal(s, p.detach()) for s, p in zip(e.shadow, e.params)); ok &= a
print(f"(a) decay=0: shadow == live weights bitwise ({len(e.shadow)} tensors, frozen bias not tracked: {'0.bias' not in e.names}): {a}")
m, init, e = run(1.0)
b = all(torch.equal(s, init[n]) for s, n in zip(e.shadow, e.names)) and not all(torch.equal(p.detach(), init[n]) for p, n in zip(e.params, e.names)); ok &= b
print(f"(b) decay=1: shadow == initial weights bitwise (and live weights did move): {b}")
m, init, e = run(0.9)
T, err = e.closed_form_check(); c = err < 1e-5; ok &= c
print(f"(c) decay=0.9, {T} steps: max rel err vs float64 closed form {err:.2e} (<1e-5): {c}")
live = [p.detach().clone() for p in e.params]; sh = [s.clone() for s in e.shadow]
e.swap(); sw1 = all(torch.equal(p.detach(), s) for p, s in zip(e.params, sh))
e.swap(); sw2 = all(torch.equal(p.detach(), l) for p, l in zip(e.params, live)) and all(torch.equal(s, x) for s, x in zip(e.shadow, sh))
ok &= sw1 and sw2
print(f"swap: model holds EMA after 1st swap: {sw1}; live weights + shadow restored bitwise after 2nd: {sw2}")
sd = e.state_dict(); e2 = EMA(m.named_parameters(), 0.9, device="cpu"); e2.load_state_dict(sd)
rt = e2.sha256() == e.sha256(); ok &= rt
print(f"state_dict round trip sha256 identical: {rt}")
m, init, e = run(0.0, wrong_order=True)
neg = not all(torch.equal(s, p.detach()) for s, p in zip(e.shadow, e.params))
print(f"negative control: EMA updated before optimizer.step -> check (a) fails as it should: {neg}")
ok &= neg
print("ALL PASS" if ok else "SOME FAILED")
