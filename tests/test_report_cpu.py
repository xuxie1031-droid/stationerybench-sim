"""CPU known-answer test for the eval bookkeeping (no simulator, no GPU): eval/driver.py's per-episode parsers and
eval/report.py's Wilson intervals and paired McNemar counts on synthetic runner logs.

  python tests/test_report_cpu.py         (exit 0 = pass)

Checks: (1) EFB_EPISODE / [rubric] log lines are parsed into the right seeds, successes and max stages; (2) the Wilson
interval matches closed-form values; (3) the paired comparison counts discordant pairs correctly; (4) a deliberately
wrong seed offset in the parser input changes the parsed seeds (the test can fail)."""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))


def write_chunk(d, task, seed0, outcomes):
    os.makedirs(d, exist_ok=True)
    lines = []
    for i, (succ, ms) in enumerate(outcomes):
        if task == "efb":
            lines.append("EFB_EPISODE " + json.dumps(dict(episode=i, success=succ, steps=3600 if not succ else 1500,
                                                          max_stage=ms, final_stage=ms, max_lid_deg=50.0)))
        else:
            lines.append(f"[rubric] ep{i} {{'max_stage': {ms}, 'final_stage': {ms}, 'events': []}} steps=2000")
    open(os.path.join(d, "eval.log"), "w").write("\n".join(lines) + "\n")


def main():
    ok = True
    tmp = tempfile.mkdtemp()
    os.environ["GYRE_EVAL_DIR"] = tmp
    import importlib
    import driver as D
    import report as Rp
    importlib.reload(D)
    # arm A: efb 3/10 successes, rh 6/10 ; arm B (ref): efb 1/10, rh 6/10 (different seeds succeed)
    a_efb = [(i in (0, 1, 2), 4 if i in (0, 1, 2) else 2) for i in range(10)]
    b_efb = [(i == 5, 4 if i == 5 else 1) for i in range(10)]
    a_rh = [(i < 6, 4 if i < 6 else 3) for i in range(10)]
    b_rh = [(i >= 4, 4 if i >= 4 else 0) for i in range(10)]
    for arm, efb, rh in (("A", a_efb, a_rh), ("B", b_efb, b_rh)):
        write_chunk(f"{tmp}/runs/{arm}/nominal2/efb/s3000000_n10_t0", "efb", 3000000, efb)
        write_chunk(f"{tmp}/runs/{arm}/nominal2/rh/s3000000_n10_t0", "rh", 3000000, rh)
    got = D.collect("efb", f"{tmp}/runs/A/nominal2/efb")
    c1 = sorted(got) == list(range(3000000, 3000010)) and sum(e["success"] for e in got.values()) == 3 \
        and got[3000005]["max_stage"] == 2
    got_rh = D.collect("rh", f"{tmp}/runs/A/nominal2/rh")
    c1 &= sum(e["success"] for e in got_rh.values()) == 6 and got_rh[3000009]["max_stage"] == 3
    print("(1) log parsing:", "PASS" if c1 else "FAIL"); ok &= c1
    lo, hi = Rp.wilson(3, 10)
    c2 = abs(lo - 0.1078) < 1e-3 and abs(hi - 0.6032) < 1e-3 and Rp.wilson(0, 50)[1] < 0.072 and Rp.wilson(0, 50)[0] == 0.0
    print(f"(2) Wilson 3/10 = [{lo:.4f}, {hi:.4f}] (expect [0.1078, 0.6032]); 0/50 upper {Rp.wilson(0, 50)[1]:.4f}:", "PASS" if c2 else "FAIL")
    ok &= c2
    out = subprocess.run([sys.executable, os.path.join(ROOT, "eval", "report.py"), "A", "--visual", "nominal2",
                          "--seeds", "3000000-3000009", "--ref", "B"], capture_output=True, text=True,
                         env=dict(os.environ, GYRE_EVAL_DIR=tmp)).stdout
    # efb: A succeeds on 0,1,2 ; B on 5 -> +3/-1 ; rh: A on 0..5, B on 4..9 -> +4/-4
    c3 = "| eraser_from_box | 10 | 1 -> 3 | +3/-1 |" in out and "| ruler_handover | 10 | 6 -> 6 | +4/-4 |" in out
    print("(3) paired counts:", "PASS" if c3 else "FAIL"); ok &= c3
    if not c3:
        print(out)
    # (4) negative control: chunk dir named with a wrong seed0 -> parsed seeds shift, so (1) would fail
    write_chunk(f"{tmp}/runs/C/nominal2/efb/s3000100_n10_t0", "efb", 3000100, a_efb)
    got_c = D.collect("efb", f"{tmp}/runs/C/nominal2/efb")
    c4 = sorted(got_c) != list(range(3000000, 3000010))
    print("(4) negative control (wrong seed0 must not parse as the sealed seeds):", "PASS" if c4 else "FAIL"); ok &= c4
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
