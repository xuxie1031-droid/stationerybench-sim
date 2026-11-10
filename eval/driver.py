#!/usr/bin/env python3
"""Paired closed-loop A/B driver for the StationeryBench sim replicas (eraser_from_box, ruler_handover).

  driver.py --arm NAME --policy P --ckpt C --url http://127.0.0.1:8230/act --visual nominal2 \
            [--tasks efb,rh] [--seed0 1000 --n 20] [--chunk 10] [--server-pid PID] [--confirm]

Needs a running MolmoAct2 /act server at --url (scripts/eval.sh starts one).  Runs chunks of <=CHUNK
episodes (one sim process at a time on this host's GPU) under a stall watchdog, parses per-episode
stages from the task runner's own outputs (never trusts rc), re-runs the missing seeds (retry passes),
then appends one record per task to $GYRE_EVAL_DIR/results.json.
Outputs: $GYRE_EVAL_DIR/runs/<arm>/<visual>/<task>/s<seed>_n<k>_t<try>/{eval.log,episodes.json,...}
(full videos/frames under $GYRE_SCRATCH/eval/...).
Episode length per task = what every reported number used: eraser_from_box 3600 steps (120 s at 30 Hz,
StationeryBench's VLA horizon), ruler_handover 2000 steps (66.7 s).  --max-steps overrides both.
eraser_from_box runs on the spring-return lid (EFB_LID_SPRING=1, the release default); --no-spring
selects the original gravity-only lid.
"""
import os, sys  # noqa: E401  repo root on sys.path -> gyre_paths (scripts/env.sh does the same)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gyre_paths as GP  # noqa: E402
import argparse, ast, fcntl, glob, json, os, re, shutil, signal, socket, subprocess, sys, time

AB = GP.AB
SCR = os.path.join(GP.SCRATCH, "eval")  # host-local: full eval outputs (videos/frames/logs); only small files go to AB
REP_SEEDS = {1000, 1001, 1002, 3000000, 3000001, 3000002}  # representative videos copied next to the results
REPO = GP.MOLMOACT2
TASKS = {
    "efb": dict(name="eraser_from_box", env="StationeryEraserFromBox-v1",
                runner=os.path.join(GP.EVAL, "run_eval_efb.py"), steps=3600, stall=720),
    "rh": dict(name="ruler_handover", env="StationeryRulerHandover-v1",
               runner=os.path.join(GP.EVAL, "run_eval_rh.py"), steps=2000, stall=480),
}
DEV_SEEDS = range(1000, 1020)          # development set (20 per task)
SEALED_SEEDS = range(3000000, 3000050)  # sealed confirmation / release set (50 per task): never used for development or tuning
FIRST_EXTRA = 300  # env construction + first episode


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def parse_chunk(task, d, seed0):
    """-> {seed: dict(success,max_stage,final_stage,steps,video)} for finished episodes of one chunk dir."""
    out = {}
    lg = os.path.join(d, "eval.log")
    txt = open(lg, errors="ignore").read() if os.path.exists(lg) else ""
    if task == "efb":
        for m in re.finditer(r"EFB_EPISODE (\{.*\})", txt):
            e = json.loads(m.group(1))
            out[seed0 + int(e["episode"])] = dict(success=bool(e["success"]), max_stage=int(e["max_stage"]),
                                                 final_stage=int(e["final_stage"]), steps=int(e["steps"]),
                                                 ep=int(e["episode"]), max_lid_deg=round(e["max_lid_deg"], 1))
    else:
        for m in re.finditer(r"\[rubric\] ep(\d+) (\{.*\}) steps=(\d+)", txt):
            s = ast.literal_eval(m.group(2))
            ep = int(m.group(1))
            out[seed0 + ep] = dict(success=int(s.get("max_stage", 0)) >= 4, max_stage=int(s.get("max_stage", 0)),
                                   final_stage=int(s.get("final_stage", 0)), steps=int(m.group(3)), ep=ep,
                                   events=s.get("events", [])[:12], final_holder=s.get("final_holder"))
    for s, e in out.items():
        v = glob.glob(os.path.join(d, "*", "videos", "*", f"ep{e['ep']:03d}.mp4"))
        e["video"] = v[0] if v else None
    return out


def collect(task, tdir):
    eps = {}
    for d in sorted(glob.glob(os.path.join(tdir, "s*_n*_t*"))):
        m = re.match(r"s(\d+)_n(\d+)_t(\d+)", os.path.basename(d))
        ej = os.path.join(d, "episodes.json")
        got = {int(k): v for k, v in json.load(open(ej)).items()} if os.path.exists(ej) else parse_chunk(task, d, int(m.group(1)))
        for s, e in got.items():
            eps.setdefault(s, e)  # first complete record wins (deterministic env; retries only fill gaps)
    return eps


def gpu_pids():
    try:
        o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30).stdout
        return {int(x) for x in o.split() if x.strip().isdigit()}
    except Exception:
        return set()


def wait_gone(pids, limit=180):
    t = time.time()
    while time.time() - t < limit:
        if not (gpu_pids() & pids):
            return True
        time.sleep(5)
    return False


def server_ok(url, server_pid):
    if server_pid and not os.path.exists(f"/proc/{server_pid}"):
        return False
    return True


def run_chunk(args, task, cfg, tdir, seeds, attempt):
    s0, k = seeds[0], len(seeds)
    d = os.path.join(tdir, f"s{s0}_n{k}_t{attempt}")
    while os.path.exists(d):  # never overwrite an earlier attempt's log (its episodes are still parsed)
        attempt += 1
        d = os.path.join(tdir, f"s{s0}_n{k}_t{attempt}")
    meta = d
    d = SCR + meta[len(AB):]
    os.makedirs(meta, exist_ok=True); os.makedirs(d, exist_ok=True)
    open(os.path.join(meta, "HOST"), "w").write(f"{socket.gethostname()} {d}\n")
    cmd = [GP.SIM_PY, os.path.join(GP.EVAL, "run_eval_visual.py"), "--visual", args.visual,
           "--visual-seed", str(args.visual_seed), "--runner", cfg["runner"],
           "--policy-type", "remote-yam", "--remote-url", args.url, "-e", cfg["env"], "-n", str(k),
           "--seed", str(s0), "--max-episode-steps", str(args.max_steps or cfg["steps"]), "--max-videos", str(k),
           "--output-dir", d]
    env = dict(os.environ, MS_ASSET_DIR=GP.MS_ASSET_DIR,
               PYTHONPATH=os.pathsep.join([GP.ROOT, REPO] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])),
               HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    if task == "efb":   # spring-return lid unless --no-spring (the eval log must show "[eraser_task] LID_SPRING=1")
        env["EFB_LID_SPRING"] = "0" if args.no_spring else "1"
    open(os.path.join(d, "cmd.txt"), "w").write(" ".join(cmd) + "\n")
    lf = open(os.path.join(d, "eval.log"), "w")
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    open(os.path.join(d, "eval.pid"), "w").write(str(p.pid))
    log(f"[{args.arm}/{args.visual}/{task}] chunk seeds {s0}..{s0+k-1} try{attempt} pid {p.pid}")
    last_n, last_t, deadline_extra = 0, time.time(), FIRST_EXTRA
    status = "ok"
    while p.poll() is None:
        time.sleep(15)
        n = len(parse_chunk(task, d, s0))
        if n > last_n:
            last_n, last_t, deadline_extra = n, time.time(), 0
        if time.time() - last_t > cfg["stall"] + deadline_extra:
            status = "stall"
        if not server_ok(args.url, args.server_pid):
            status = "server_dead"
        if status != "ok":
            log(f"  {status} after {last_n}/{k} episodes -> killing pgid {p.pid}")
            try:
                os.killpg(p.pid, signal.SIGTERM); time.sleep(10); os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
            break
    lf.close()
    wait_gone({p.pid})
    got = parse_chunk(task, d, s0)
    for e in got.values():
        e["host"] = socket.gethostname()
    for sd, e in got.items():
        if sd in REP_SEEDS and e.get("video"):
            os.makedirs(os.path.join(meta, "videos"), exist_ok=True)
            dst = os.path.join(meta, "videos", f"seed{sd}.mp4"); shutil.copy(e["video"], dst); e["video_ab"] = dst
    json.dump({str(k): v for k, v in got.items()}, open(os.path.join(meta, "episodes.json"), "w"), indent=1)
    shutil.copy(os.path.join(d, "eval.log"), os.path.join(meta, "eval.log"))
    shutil.copy(os.path.join(d, "cmd.txt"), os.path.join(meta, "cmd.txt"))
    d = meta
    open(os.path.join(d, "STATUS"), "w").write(f"{status} rc={p.returncode} got={len(got)}/{k} {time.strftime('%FT%TZ', time.gmtime())}\n")
    log(f"  done {status} rc={p.returncode} got {len(got)}/{k}")
    return status


def append_results(rec):
    os.makedirs(AB, exist_ok=True)
    path = os.path.join(AB, "results.json")
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        data = json.load(open(path)) if os.path.exists(path) else []
        data = [r for r in data if not (r["arm"] == rec["arm"] and r["visual"] == rec["visual"] and r["task"] == rec["task"])]
        data.append(rec)
        tmp = path + ".tmp"
        json.dump(data, open(tmp, "w"), indent=1)
        os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8230/act")
    ap.add_argument("--server-pid", type=int, default=0)
    ap.add_argument("--visual", default="nominal")
    ap.add_argument("--visual-seed", type=int, default=0)
    ap.add_argument("--tasks", default="efb,rh")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--confirm", action="store_true", help="allow the reserved confirmation seeds 3000000-3000049")
    ap.add_argument("--max-steps", type=int, default=0, help="override the per-task episode length (default: efb 3600, rh 2000)")
    ap.add_argument("--no-spring", action="store_true", help="eraser_from_box with the original gravity-only lid")
    args = ap.parse_args()
    want = list(range(args.seed0, args.seed0 + args.n))
    if any(3000000 <= s <= 3000049 for s in want) and not args.confirm:
        sys.exit("REFUSED: seeds overlap the reserved confirmation set 3000000-3000049 (milestone claims only; pass --confirm)")
    for task in args.tasks.split(","):
        cfg = TASKS[task]
        tdir = os.path.join(AB, "runs", args.arm, args.visual, task)
        os.makedirs(tdir, exist_ok=True)
        for attempt in range(args.retries + 1):
            eps = collect(task, tdir)
            missing = [s for s in want if s not in eps]
            if not missing:
                break
            # contiguous runs of missing seeds, split to <= chunk
            runs, cur = [], [missing[0]]
            for s in missing[1:]:
                if s == cur[-1] + 1 and len(cur) < args.chunk:
                    cur.append(s)
                else:
                    runs.append(cur); cur = [s]
            runs.append(cur)
            for r in runs:
                st = run_chunk(args, task, cfg, tdir, r, attempt)
                if st == "server_dead":
                    log("server dead; aborting"); sys.exit(2)
        eps = collect(task, tdir)
        missing = [s for s in want if s not in eps]
        rec = dict(arm=args.arm, policy=args.policy, ckpt=args.ckpt, visual=args.visual,
                   visual_seed=args.visual_seed, task=cfg["name"], env_id=cfg["env"],
                   max_episode_steps=args.max_steps or cfg["steps"], lid_spring=(0 if args.no_spring else 1) if task == "efb" else None,
                   seeds=want, missing=missing,
                   n_done=len(want) - len(missing), success=sum(eps[s]["success"] for s in want if s in eps),
                   max_stage={str(s): eps[s]["max_stage"] for s in want if s in eps},
                   final_stage={str(s): eps[s]["final_stage"] for s in want if s in eps},
                   episodes={str(s): eps[s] for s in want if s in eps},
                   finished_utc=time.strftime("%FT%TZ", time.gmtime()))
        append_results(rec)
        ms = [eps[s]["max_stage"] for s in want if s in eps]
        log(f"[{args.arm}/{args.visual}/{task}] {rec['success']}/{rec['n_done']} success; max-stage hist "
            f"{[ms.count(i) for i in range(5)]}; missing {missing}")


if __name__ == "__main__":
    main()
