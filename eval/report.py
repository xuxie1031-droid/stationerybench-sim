#!/usr/bin/env python3
"""Success / stage report for one evaluated arm (M1b-style), with Wilson 95% CIs and an optional paired comparison.

  report.py ARM [--visual nominal2] [--seeds dev|sealed|A-B] [--ref REF_ARM] [--md OUT.md]

Reads the per-episode records that eval/driver.py wrote under $GYRE_EVAL_DIR/runs/ARM/<visual>/<task>/ (the same
records the milestone checks used).  Per task: successes / episodes with a Wilson 95% interval, the max-stage
histogram (rubric stages 0..4), mean max stage, and the M1b bar (dev: eraser >= 4/20, ruler >= 8/20; sealed: eraser
>= 10/50, ruler >= 20/50).  With --ref: same-seed pairs vs REF_ARM, success discordance (+b/-c) with an exact
two-sided McNemar (binomial) p, and a sign test on the max-stage difference.

Policy sampling is NOT seeded (flow-matching noise): the env layout of seed s is fixed, but two runs of the same
checkpoint on the same seed can differ.  Compare arms by counts and intervals, not per-episode outcomes.
"""
import argparse
import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gyre_paths as GP  # noqa: E402
sys.path.insert(0, GP.EVAL)
import driver as D  # noqa: E402

TASKS = [("efb", "eraser_from_box"), ("rh", "ruler_handover")]
BARS = {"dev": dict(efb=4, rh=8), "sealed": dict(efb=10, rh=20)}   # M1b bars (20 % eraser, 40 % ruler)


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def binom_two_sided(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def seed_range(spec):
    if spec == "dev":
        return D.DEV_SEEDS, "dev"
    if spec == "sealed":
        return D.SEALED_SEEDS, "sealed"
    a, b = (int(x) for x in spec.split("-"))
    return range(a, b + 1), None


def load(arm, visual, t):
    d = os.path.join(GP.AB, "runs", arm, visual, t)
    return {int(k): v for k, v in D.collect(t, d).items()} if os.path.isdir(d) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--visual", default="nominal2")
    ap.add_argument("--seeds", default="sealed", help="dev (1000-1019) | sealed (3000000-3000049) | A-B (inclusive)")
    ap.add_argument("--ref", default=None, help="paired comparison against this arm on the same seeds")
    ap.add_argument("--md", default=None, help="also write the report to this markdown file")
    a = ap.parse_args()
    seeds, kind = seed_range(a.seeds)
    out = [f"# {a.arm} ({a.visual}, seeds {seeds.start}-{seeds.stop - 1}, full length, unseeded policy sampling)\n",
           "| task | successes | rate (Wilson 95% CI) | max stage 0/1/2/3/4 | mean max stage |" + (" M1b bar |" if kind else ""),
           "|---|---|---|---|---|" + ("---|" if kind else "")]
    cmp_rows = []
    for t, name in TASKS:
        eps = load(a.arm, a.visual, t)
        have = [s for s in seeds if s in eps]
        k = sum(bool(eps[s]["success"]) for s in have)
        lo, hi = wilson(k, len(have))
        hist = "/".join(str(sum(eps[s]["max_stage"] == i for s in have)) for i in range(5))
        mms = sum(eps[s]["max_stage"] for s in have) / max(1, len(have))
        row = f"| {name} | {k}/{len(have)}{'' if len(have) == len(seeds) else f' (of {len(seeds)})'} | " \
              f"{100 * k / max(1, len(have)):.0f}% [{100 * lo:.0f}, {100 * hi:.0f}] | {hist} | {mms:.2f} |"
        if kind:
            need = BARS[kind][t]
            row += f" {'PASS' if len(have) == len(seeds) and k >= need else ('FAIL' if len(have) == len(seeds) else 'INCOMPLETE')} (>= {need}/{len(seeds)}) |"
        out.append(row)
        if a.ref:
            r = load(a.ref, a.visual, t)
            sh = [s for s in seeds if s in eps and s in r]
            ok = lambda e: bool(e["success"])  # noqa: E731
            b = sum(ok(eps[s]) and not ok(r[s]) for s in sh)
            c = sum(ok(r[s]) and not ok(eps[s]) for s in sh)
            dm = [eps[s]["max_stage"] - r[s]["max_stage"] for s in sh]
            up, dn = sum(x > 0 for x in dm), sum(x < 0 for x in dm)
            cmp_rows.append(f"| {name} | {len(sh)} | {sum(ok(r[s]) for s in sh)} -> {sum(ok(eps[s]) for s in sh)} | +{b}/-{c} | "
                            f"{binom_two_sided(b, c):.3g} | {sum(dm) / max(1, len(dm)):+.2f} (+{up}/-{dn}, p={binom_two_sided(up, dn):.3g}) |")
    if cmp_rows:
        out += ["", f"Paired vs {a.ref} (same seeds, same visual):", "",
                "| task | pairs | successes ref -> arm | discordant | McNemar exact p | mean max-stage diff (sign test) |",
                "|---|---|---|---|---|---|"] + cmp_rows
    out += ["", "Stages: StationeryBench 0-4 rubric evaluated from privileged sim state (envs/RUBRICS.md); 4 = success.",
            "Sampling is unseeded, so a re-run gives different per-episode outcomes; read the intervals."]
    txt = "\n".join(out)
    print(txt)
    if a.md:
        os.makedirs(os.path.dirname(os.path.abspath(a.md)), exist_ok=True)
        open(a.md, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
