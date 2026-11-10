# StationeryBench sim visuals: real-bench alignment + domain randomization

Render-only changes for the YAM StationeryBench replicas (`eraser_from_box`, `ruler_handover`). No task file is edited, and physics is untouched: no collision
geometry, mass, friction, pose or env RNG changes. Verified by bit-exact replay, see below.

## Files
| file | what |
|---|---|
| `visual.py` | `apply_nominal(env)`, `apply_nominal2(env)`, `apply_dr(env, rng, level, split)`, `sample_image_aug`, `augment_image`, `VisualWrapper`, `install(mode)`, `register_task_hook` |
| `texgen.py` | procedural texture pools: `textures/{nominal,train,heldout}/` + `pool.json` (shipped; regenerate with `python texgen.py OUTDIR`) |
| `real_ref/` | six real Ai2 YAM top/left/right frames (allenai/01122025-box-01, Apache-2.0) used for the look calibration |
| `tools/look_test.py`, `tools/contact_sheet.py` | reset frames per mode, comparison sheets |
| `NOMINAL2.md`, `nominal2_stats/` | the nominal2 look: calibration targets, stats, known-answer tests |
| `../../eval/run_eval_visual.py` | wraps any per-task eval runner with `--visual nominal\|nominal2\|dr_train\|dr_heldout\|off` |

## How it works
* Table: the ManiSkill table mesh is hidden (`visibility=0`). A textured render-only quad with explicit UVs replaces it at z=0, with a dark slab below. Texture columns (u) run along world x (1.209 m), rows (v) along world y (2.418 m), image 1536x3072 = 1.27 px/mm. `RenderShapeBox` is not used for textures because its per-face UVs are an atlas sub-rectangle (measured), which crops and stretches textures.
* Floor: the ManiSkill ground keeps its collision. Only its render material is recoloured near-black, without the grid texture.
* Backdrop: a render-only black panel with a textured front quad, just behind the table's far (arm-base) edge, top at z=0.35 m. Any taller and it would shadow the key light.
* Lights: ManiSkill's two directional lights are dimmed (nominal 0). A spot key light is added above and behind the arm bases, plus a weak directional fill and ambient light.
* Robot: two-tone recolour (dark links / light covers).
* Objects: per-task hooks.
* Cameras: render-camera local pose and focal length. The env's camera configs are not changed.
* **Material edits were being ignored.** With ray-tracing shaders (rt-fast), SAPIEN only re-uploads materials when the render scene's entity set changes. That is why "changing the table material at runtime did not take effect". `_apply` always removes and re-adds its own `vis__*` entities, which forces the refresh.
* Do not call `enable_shadow()` on the RT spot light: it then contributes nothing. RT shadows are traced anyway.
* Operations: after killing a SAPIEN process, wait until `nvidia-smi` shows no compute app before starting the next one. Overlapping processes on one GPU caused hangs in ~30% of runs.

## Nominal look (from `docs/assets/*-setup.jpg`)
* Table: dark, worn, low-saturation brown wood. It has grain along y, worn patches, dark knots, a few chips, one plank seam along x and one along y near the bases. Roughness 0.45, specular 0.45.
* Backdrop: glossy black (sRGB about 10,11,14) with faint smudges.
* Floor: near-black, 0.012.
* Key spot light: 9.0 x 5000 K blackbody. Azimuth 30 deg from -x toward +y, elevation 58 deg, distance 1.6 m, aimed at (-0.33, 0, 0); fov inner 0.9 / outer 1.7.
* Fill and ambient: fill 0.25 at 5500 K from the operator side; ambient 0.08.
* Calibration: the operator-view table median is (130,109,93). The photo mid-table median is (125,108,98).
* Ruler: as built (clear tint, alpha 0.3, ink ticks, white label band).
* Cups: off-white (0.90,0.89,0.86), roughness 0.5. The real cups are white ribbed cups.
* Robot: dark 0.10, covers 0.85.
* Image aug: none.

## DR ranges (level=1; continuous deviations from nominal scale with `level`)
| group | parameter | range |
|---|---|---|
| key light | azimuth | nominal +-75 deg |
| | elevation | 42-78 deg |
| | distance | 1.3-2.0 m |
| | intensity | x0.55-1.6 (log-uniform) |
| | colour temperature | 3000-7000 K |
| other lights | fill | 0-0.6 (4000-7000 K) |
| | ambient | 0.02-0.18 |
| | ManiSkill default lights | x0-0.25 |
| table | texture | 1 of 12 train woods (heldout: 4 other woods) |
| | tint | x0.75-1.25 |
| | roughness / specular | roughness 0.3-0.75, specular 0.3-0.6 |
| background | backdrop texture | 1 of 10 train (black, greys, browns, a light wall). Heldout: 4 others |
| | backdrop tint / roughness | tint x0.7-1.4, roughness 0.2-0.9 |
| | floor value | 0-0.12 |
| objects | generic / per-actor jitter | hue +-0.03 (about +-11 deg), sat x0.8-1.2, val x0.8-1.2 |
| | ruler | clear tint 0.8-1.0 per channel, alpha 0.15-0.5, ink 0.02-0.2 (blue-ish up to x2), label value x0.85-1.05 |
| | cups | palette of 10 train / 3 heldout colours. Both cups share a colour w.p. 0.7. Roughness 0.35-0.9 |
| robot | colour | dark 0.03-0.35, covers 0.6-0.95, roughness 0.3-0.8 |
| camera | translation | top_cam +-1.5 cm per axis; wrist cams +-0.5 cm (rigid mounts) |
| | rotation | +-2 deg per axis |
| | focal | x(1+-0.03), i.e. fov +-3% |
| image (per camera per episode) | exposure | 2^U(-0.4,0.4) |
| | gamma | 0.85-1.15 |
| | contrast (about mid-grey) | 0.85-1.15 |
| | saturation | 0.8-1.2 |
| | white balance | R,B gains 1+-0.08 |
| | gaussian noise | sigma 0-4 (8-bit), new noise every frame. **Not baked into the dataset videos** (incompressible: x7 file size, and H.264 would distort it). The sampled sigma is stored in `vis_params.image_aug.<cam>.noise_online`; add it in the dataloader. Applied live in eval. `--bake-noise` overrides |
| | blur | w.p. 0.5, sigma 0.3-1.0 px |
| | vignette | top 0-0.2, wrist 0-0.35 |

The DR rng is `visual.episode_rng(vis_seed, reset_seed, variant)`. It is independent of the env's
episode RNG, so the scene layout for a given seed is unchanged.

## Held-out split (`split="heldout"`, eval mode `dr_heldout`)
Training (`dr_train`) never samples any of the following:
* Wood textures `textures/heldout/wood_00..03`. Their seeds and palettes are disjoint from training: reddish mahogany, cold grey-brown, light honey, near-black ebony.
* Backdrops `heldout/backdrop_00..03`: dark red, blue-grey, off-white, dark green.
* Cup colours: yellow, blue, dark brown.
* Key-light colour temperature in [3600,4000) or [5600,6000) K.
* Key-light azimuth offset in [-45,-25) or [55,75) deg.
* Object hue shift |dh| in [0.03,0.05]. Training uses |dh| < 0.03.

Camera, robot and image-aug ranges are the same in both splits.

## Eval
```
source scripts/env.sh; cd $MOLMOACT2_DIR
$SIM_PY $GYRE_ROOT/eval/run_eval_visual.py --visual nominal2 [--visual-seed 0 --visual-level 1] \
   --runner $GYRE_ROOT/eval/run_eval_efb.py \
   --policy-type remote-yam --remote-url http://127.0.0.1:8230/act -e StationeryEraserFromBox-v1 -n 10 --seed 1000 \
   --max-episode-steps 3600 --output-dir OUT
```
The same works for `eval/run_eval_rh.py` (`-e StationeryRulerHandover-v1 --max-episode-steps 2000`). Any env id starting
with `Stationery` is wrapped. `scripts/eval.sh` does all of this (server + driver + report) in one command.
* `--visual off`: as built.
* `--visual nominal`: real-bench-aligned look (Robocurve setup photos), no image aug.
* `--visual nominal2`: brighter look calibrated to real Ai2 YAM camera frames (see NOMINAL2.md). **All released results use nominal2.**
* `dr_train` / `dr_heldout`: DR with image aug.
* In code: `env = visual.wrap(gym.make(...), "dr_heldout")`. Or call `visual.install_from_env()` and set `STATIONERY_VISUAL=<mode>`.

## nominal2
`--visual nominal2` (run_eval_visual) selects a brighter look calibrated to the frame-0 statistics of the real Ai2 YAM data. It has a soft area key and tinted sky, a white-cover / black-joint arm, and a light room wall seen only by the wrist cameras. Table, backdrop, objects, cameras and physics are as in nominal. The existing modes are unchanged (bit-identical frame-hash KAT). Parameters, stats and KATs are in `NOMINAL2.md`.
