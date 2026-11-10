"""Render reset frames under off / nominal / dr variants.  look_test.py eraser|ruler OUTDIR SEED  (source scripts/env.sh)"""
import os, sys  # noqa: E401  repo root on sys.path -> gyre_paths (scripts/env.sh does the same)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import gyre_paths as GP  # noqa: E402
import sys, os, json, numpy as np, gymnasium as gym, imageio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visual
task, out, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(out, exist_ok=True)
if task == "eraser":
    import eraser_task as T; kw = {}
else:
    import ruler_handover_env as T; kw = dict(reward_mode="none")
env = gym.make(T.ENV_ID, obs_mode="rgb", control_mode="pd_joint_pos", render_mode="rgb_array",
               sensor_configs=dict(shader_pack="rt-fast"), human_render_camera_configs=dict(shader_pack="rt-fast"),
               sim_config=dict(sim_freq=150, control_freq=30), **kw)
variants = [("old", "off", 0), ("nominal", "nominal", 0)] + [(f"dr{k}", "dr_train", k) for k in range(3)] + [("ho0", "dr_heldout", 0)]
for tag, mode, k in variants:
    w = visual.VisualWrapper(env, mode=mode, variant=k)
    obs, info = w.reset(seed=seed)
    for c in ("top_cam", "left_cam", "right_cam"):
        imageio.imwrite(f"{out}/{tag}_{c}.png", obs["sensor_data"][c]["rgb"][0].cpu().numpy())
    imageio.imwrite(f"{out}/{tag}_operator.png", env.unwrapped.render()[0].cpu().numpy())
    json.dump(w.params, open(f"{out}/{tag}_params.json", "w"), indent=1, default=str)
    if mode == "off":
        # 'off' after a look was applied would keep it; only valid first
        pass
print("ok")
