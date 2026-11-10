"""MolmoAct2 eval on StationeryRulerHandover-v1 via the repo's sim_eval.run_eval (unmodified).

Registers the env, sets the spec's verbatim instruction as the env default, and records the
privileged rubric (max / final stage, events) per episode into results.json.

  (after `source scripts/env.sh`; normally called through eval/run_eval_visual.py --runner ...)
  cd $MOLMOACT2_DIR && $SIM_PY $GYRE_ROOT/eval/run_eval_rh.py --policy-type remote-yam \
     --remote-url http://127.0.0.1:8210/act -e StationeryRulerHandover-v1 -n 10 --seed 1000 --output-dir OUT
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, ".."))
import gyre_paths  # noqa: E402,F401  (puts envs/ruler_handover and MolmoAct2's sim_eval on sys.path)

import ruler_handover_env as RH  # noqa: E402  (registers env)
import sim_eval.run_eval as RE  # noqa: E402

RE.DEFAULT_LANGUAGE_INSTRUCTIONS[RH.ENV_ID] = RH.INSTRUCTION

_rubrics = []
_orig_run_episode = RE._run_episode
_orig_evaluate_task = RE._evaluate_task


def _run_episode(env, *a, **k):
    res = _orig_run_episode(env, *a, **k)
    tr = getattr(env.unwrapped, "_tracker", None)
    summ = tr.summary() if tr is not None else {}
    _rubrics.append(summ)
    print(f"[rubric] ep{len(_rubrics) - 1} {summ} steps={res['steps']}", flush=True)
    return res


def _evaluate_task(env_id, client, config):
    _rubrics.clear()
    r = _orig_evaluate_task(env_id, client, config)
    for ep, summ in zip(r["episodes"], _rubrics):
        ep["rubric"] = summ
    ms = [s.get("max_stage", 0) for s in _rubrics]
    fs = [s.get("final_stage", 0) for s in _rubrics]
    r["max_stage_hist"] = {k: ms.count(k) for k in range(5)}
    r["final_stage_hist"] = {k: fs.count(k) for k in range(5)}
    r["mean_max_stage"] = sum(ms) / max(1, len(ms))
    return r


RE._run_episode = _run_episode
RE._evaluate_task = _evaluate_task

if __name__ == "__main__":
    RE.main()
