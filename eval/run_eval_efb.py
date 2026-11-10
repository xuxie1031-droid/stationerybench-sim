"""MolmoAct2 eval on StationeryEraserFromBox-v1 via sim_eval.run_eval.main (unmodified repo).

Registers the env (import), adds the verbatim spec instruction to run_eval's instruction table,
and wraps each episode's env.step to log the privileged rubric stage. Writes stages.json next to
run_eval's results.json.

usage (after `source scripts/env.sh`; normally called through eval/run_eval_visual.py --runner ...):
  cd $MOLMOACT2_DIR && $SIM_PY $GYRE_ROOT/eval/run_eval_efb.py \
      --policy-type remote-yam --remote-url http://127.0.0.1:8230/act \
      -e StationeryEraserFromBox-v1 -n 10 --seed 1000 --max-episode-steps 3600 --output-dir OUT
"""
import os, sys  # noqa: E401  repo root on sys.path -> gyre_paths (scripts/env.sh does the same)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gyre_paths as GP  # noqa: E402
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, GP.MOLMOACT2)
sys.path.insert(0, HERE)
import eraser_task as T  # noqa: E402  registers StationeryEraserFromBox-v1
import sim_eval.run_eval as rv  # noqa: E402

rv.DEFAULT_LANGUAGE_INSTRUCTIONS[T.ENV_ID] = T.INSTRUCTION
EPISODES = []
_orig_run_episode = rv._run_episode


def _run_episode(env, client, obs, instruction, max_steps):
    trace = []
    raw_step = env.step

    def step(a):
        out = raw_step(a)
        info = out[4]
        trace.append((int(info["stage"][0]), float(info["lid_angle_deg"][0]),
                      [round(float(x), 4) for x in info["eraser_box_frame"][0]]))
        return out
    env.step = step
    try:
        res = _orig_run_episode(env, client, obs, instruction, max_steps)
    finally:
        del env.step
    stages = [t[0] for t in trace]
    changes = [(i, s) for i, s in enumerate(stages) if i == 0 or s != stages[i - 1]]
    EPISODES.append(dict(
        episode=len(EPISODES), success=bool(res["success"]), steps=len(trace),
        max_stage=max(stages) if stages else 0, final_stage=stages[-1] if stages else 0,
        max_lid_deg=max(t[1] for t in trace) if trace else 0.0, final_lid_deg=trace[-1][1] if trace else 0.0,
        final_eraser_box_frame=trace[-1][2] if trace else None, stage_changes=changes[:200]))
    print("EFB_EPISODE", json.dumps({k: v for k, v in EPISODES[-1].items() if k != "stage_changes"}), flush=True)
    return res


rv._run_episode = _run_episode

if __name__ == "__main__":
    rv.main()
    out_root = None
    for i, a in enumerate(sys.argv):
        if a == "--output-dir":
            out_root = sys.argv[i + 1]
    if out_root:
        runs = sorted(glob.glob(os.path.join(out_root, "*", "results.json")))
        d = os.path.dirname(runs[-1]) if runs else out_root
        hist = {k: sum(e["max_stage"] == k for e in EPISODES) for k in range(5)}
        json.dump(dict(env_id=T.ENV_ID, instruction=T.INSTRUCTION, n=len(EPISODES),
                       success_rate=sum(e["success"] for e in EPISODES) / max(1, len(EPISODES)),
                       max_stage_hist=hist, episodes=EPISODES),
                  open(os.path.join(d, "stages.json"), "w"), indent=1)
        print("EFB_STAGES", d, json.dumps(hist))
