# nominal2: brighter, soft-light, two-tone-arm look (opt-in)

A visual mode `nominal2` for the StationeryBench YAM replicas (ManiSkill 3, SAPIEN rt-fast). It is the look used for all released training data from round ft10 on and for all released evaluations. It sits next to `nominal`, `dr_train` and `dr_heldout`, and **does not change them**: the KATs below show bit-identical frames and scene edits. It is opt-in everywhere:

```
run_eval_visual.py --visual nominal2 ...       # eval
STATIONERY_VISUAL=nominal2 ...                 # visual.install_from_env()
visual.wrap(env, "nominal2") / visual.apply_nominal2(env)
```

## Why
With the nominal look, frame 0 of the sim demos was 25-40% darker than the real Ai2 YAM training data. It also had about half the contrast, large hard black arm shadows, and a uniformly dark-grey arm. The surfaces (dark wood table, glossy black backdrop) were matched to the Robocurve setup photos and are kept unchanged.

## What nominal2 changes (everything else = nominal)
| | nominal | **nominal2** |
|---|---|---|
| key light | spot 9.0 x 5000 K, az 30 deg / el 58 deg, 1.6 m from (-0.33, 0, 0), fov 0.9/1.7 rad: hard shadows | **area (parallelogram) light 0.40 x 0.30 m**, 54 x 4800 K, az 30 / el 62 deg, 0.9 m from (-0.33, 0, 0). Same direction as the spot, so shadows fall the same way, but with soft penumbrae and a photo-like falloff. No spot (`key_intensity` 0) |
| sky / ambient | 0.08, neutral | **0.20 x blackbody 4800 K** (`ambient_temp_k`). In rt-fast (path depth 1) this is the radiance the bounce ray sees, i.e. an AO-like soft fill |
| fill | directional 0.25 x 5500 K from the operator side | same |
| ManiSkill default lights | 0 | 0 |
| robot, dark parts (24 of 28 render parts: base, motors, joints, fingers) | albedo 0.10, roughness 0.5, metallic 0.5 (as built), specular 0 | **0.006**, roughness 0.4, **metallic 0**, specular 0.25 |
| robot, covers (4 parts: link_2 / link_3 covers, as built 0.9) | 0.85, roughness 0.5, metallic 0.5 | **0.72**, roughness 0.55, **metallic 0**, specular 0.25 |
| room wall | none: the top ~26% of the wrist frames is near-black floor | **matte wall (0.68, 0.67, 0.64)**, 0.9 m beyond the operator-side table edge (x = 1.37 m), up to z = 1.6 m. Only the wrist cameras see it (rows 0-92); the Ai2 wrist frames show a light wall there (mean luminance 0.52) |
| table / backdrop / floor textures + materials, object looks, cameras | | unchanged |

Why the arm albedos are so low: properly exposing the dark Robocurve wood (texture mean sRGB 0.30, i.e. about 0.07 linear) takes about 2x the nominal light. Under that light a "physical" black (0.03) renders mid-grey next to the table. 0.006 gives the real arm's black-vs-wood contrast. The as-built metallic 0.5 made the white covers render grey; metallic 0 makes them white plastic.

Code: `visual.NOMINAL2`, `visual.nominal2_params()`, `visual.apply_nominal2(env)`, and `VisualWrapper(mode="nominal2")`. `_apply` gained four optional, key-gated features: `light.area`, `light.ambient_temp_k`, `room_wall`, and `robot.{metallic,specular,dark_roughness,light_roughness}`. None of these keys exists in `NOMINAL` or in the DR params, so the nominal and DR code paths are untouched. One code-path change applies to every mode: robot materials are shared (28 parts, 7 material objects). Their as-built values are now recorded for all parts before any part is edited. Previously, a part visited after its shared material was edited recorded the edited value as "original". The 0.6 light/dark classification is unaffected (KAT: identical scene edits), and switching nominal2 -> nominal in one process now restores metallic 0.5 correctly.

Physics, cameras (pose / FOV / intrinsics), the table / backdrop / floor textures and materials, and object looks are identical to nominal. Image aug stays off, as for nominal.

## Calibration
Targets: frame-0 statistics of the 16 real Ai2 YAM datasets used for co-training (3 cameras). Luminance is 0.299R+0.587G+0.114B on sRGB [0,1]; saturation is (max-min)/max.

Procedure:
1. **CPU sweeps.** About 25 looks were swept with lavapipe RT, i.e. the same rt-fast shaders without the OptiX denoiser. Each look was rendered at frame 0 and a mid-episode frame of 2 demos per task on all 3 cameras. Statistics were computed after a linear-light blur, which undoes the Jensen bias of spp-2 noise; they then matched the GPU numbers to within about 0.01.
2. **GPU check.** One final GPU check (rt-fast + OptiX, exactly the dataset path) of intensity 48 / 54 / 62 kept 54.

Findings that shaped the look:
* **The table contrast is a lighting effect, not a texture problem.** The local (16 cm patch) contrast of `wood.jpg` equals the Robocurve photos' (cv 0.106 vs 0.07-0.106). In the sim top view it equals the real wood's (0.04 vs 0.04-0.046). The real/photo global table cv of about 0.27 comes from illumination falloff. So the texture is kept and the falloff is provided by a 40 x 30 cm area light 0.9 m away, rather than by a huge soft box: table cv nominal2 0.286 vs real 0.267.
* **Soft shadows.** A big soft box gives a flat, grey look: 1.2 x 0.8 m area light with a neutral sky of 0.9 gives top saturation 0.22; with sky 0.5 it gives table cv 0.13. Sky 0.2 tinted at 4800 K keeps the warm wood (top saturation 0.338 vs real 0.361) and leaves soft, visible shadows. In the nominal key shadow region, shadow/lit is 0.80 (nominal 0.41) and shadow p5 is 0.184 (nominal 0.093; the darkest real table pixels, p5 of wood-hue pixels, are 0.225).
* **The wrist cameras' dark band is the floor beyond the operator-side table edge, not the backdrop.** It covers 25.8% of each wrist frame (rows 0-92) at luminance 0.125. Real: a light wall, 0.52. Brightening the lights instead would over-expose the wrist-view table, which is already 0.42-0.44 vs real 0.39 in rows 93+. Hence the matte room wall. It has a price: it blocks the operator-side fill for parts of the table, which the calibration includes.

## Stats (GPU rt-fast + OptiX, like the datasets)
| camera | frame | source | n | mean lum | lum std | p5 lum | mean sat | table mean | table p5 | table cv | shadow p5 | shadow median | shadow/lit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top | frame 0 | real Ai2 (16 ds) | 16 | 0.437 | 0.190 | 0.135 | 0.361 | 0.395 | 0.225 | 0.267 | - | - | - |
| top | frame 0 | sim nominal | 6 | 0.330 | 0.113 | 0.129 | 0.293 | 0.323 | 0.138 | 0.259 | 0.093 | 0.135 | 0.412 |
| top | frame 0 | sim nominal2 | 6 | 0.434 | 0.159 | 0.191 | 0.338 | 0.439 | 0.286 | 0.286 | 0.184 | 0.338 | 0.804 |
| top | mid-episode | sim nominal | 6 | 0.357 | 0.183 | 0.119 | 0.259 | 0.310 | 0.137 | 0.316 | 0.114 | 0.147 | 0.442 |
| top | mid-episode | sim nominal2 | 6 | 0.451 | 0.236 | 0.172 | 0.293 | 0.415 | 0.250 | 0.306 | 0.198 | 0.284 | 0.683 |
| left | frame 0 | real Ai2 (16 ds) | 16 | 0.423 | 0.190 | 0.124 | 0.250 | 0.327 | 0.215 | 0.210 | - | - | - |
| left | frame 0 | sim nominal | 6 | 0.278 | 0.140 | 0.093 | 0.232 | 0.310 | 0.255 | 0.121 | 0.154 | 0.172 | 0.556 |
| left | frame 0 | sim nominal2 | 6 | 0.429 | 0.136 | 0.260 | 0.324 | 0.374 | 0.274 | 0.180 | 0.236 | 0.314 | 0.842 |
| left | mid-episode | sim nominal | 6 | 0.299 | 0.153 | 0.104 | 0.191 | 0.280 | 0.167 | 0.313 | 0.132 | 0.148 | 0.444 |
| left | mid-episode | sim nominal2 | 6 | 0.509 | 0.220 | 0.176 | 0.260 | 0.348 | 0.228 | 0.231 | 0.209 | 0.265 | 0.713 |
| right | frame 0 | real Ai2 (16 ds) | 16 | 0.425 | 0.189 | 0.136 | 0.254 | 0.323 | 0.207 | 0.233 | - | - | - |
| right | frame 0 | sim nominal | 6 | 0.265 | 0.129 | 0.093 | 0.227 | 0.301 | 0.242 | 0.140 | 0.134 | 0.155 | 0.509 |
| right | frame 0 | sim nominal2 | 6 | 0.412 | 0.129 | 0.229 | 0.322 | 0.359 | 0.260 | 0.178 | 0.166 | 0.238 | 0.662 |
| right | mid-episode | sim nominal | 6 | 0.266 | 0.155 | 0.081 | 0.228 | 0.269 | 0.126 | 0.383 | 0.123 | 0.145 | 0.431 |
| right | mid-episode | sim nominal2 | 6 | 0.359 | 0.164 | 0.126 | 0.300 | 0.345 | 0.202 | 0.268 | 0.180 | 0.262 | 0.679 |

Frame 0, nominal2 vs real: mean luminance top -0.7%, left +1.5%, right -3.0% (nominal: -24.5 / -34.4 / -37.7%). Luminance std top -16.6%, left -28.3%, right -32.0% (nominal: -40.6 / -26.4 / -31.9%). Saturation top -6.4%, wrists +27-30%.
n = 6 sim frames per row (2 demos x 3 StationeryBench replicas, incl. one task not in this release). Table / shadow columns: the sim uses the segmentation class map (table quad). Real uses a wood-hue mask (h 10-50 deg, s > 0.25). Shadow = table pixels that are black in a key-only render of nominal (where nominal's hard shadows fall).

**What is not matched, and why.** Mean luminance is within 10% on all 3 cameras. Contrast (global luminance std) is not: top is -17%, wrists about -30%. The like-for-like table contrast does match (cv 0.286 vs 0.267). The remaining gap is scene content, not rendering:
* Real frame 0 contains many high-contrast objects: bowls, cables, white power strips, crates.
* In the real wrist frames, two black finger pads fill the bottom corners. At sim frame 0 the grippers are closed (reset state), so the wrist view shows one small central fingertip; once the gripper opens, the finger positions match the real frames.

The same content gap explains why the sim p5 is higher than the real p5 (the real dark pixels are cables and fingers). The wrist saturation is +28%: the sim wrist view is mostly warm-lit wood, while the real ones contain grey/white walls and objects. Closing these gaps by lighting alone would need harsh vignetting or over-exposure; it was not done.

## KATs
1. **Existing modes unchanged: images.** A frame-hash known-answer test over the modes nominal, dr_train v0 and dr_heldout v0 (frames t=0 and t=60, 3 cameras, all replicas), run before and after adding nominal2: **54/54 frame hashes identical; the params JSON of every mode is identical**.
   Negative controls: the three modes give 3 distinct hashes, and t0 differs from t60 on 3/3 cameras. The calibration renderer's frames of nominal t0 are **bit-identical** to the dataset-path frames.
2. **Existing modes unchanged: scene edits, CPU.** A CPU check compared every render part's material on every entity, the lights (type, colour, pose, shape), the ambient light and the camera poses and intrinsics. It ran with old vs new `visual.py` for nominal / dr_train v0, v1 / dr_heldout v0 x seeds 0-2. **12/12 cases identical per task.** Mixed order in one process: nominal2 then nominal equals fresh nominal; nominal2 then dr_train equals fresh dr_train; nominal then nominal2 equals fresh nominal2. All identical. Negative controls: nominal vs nominal2 and nominal vs dr_train differ.
3. **Physics untouched under nominal2.** Re-rendering recorded episodes under nominal2 is bit-exact (identical stage sequence, max qpos/state difference 0.0), with a negative control that diverges.

## Using nominal2
Eval: `run_eval_visual.py --visual nominal2 ...` (`eval/driver.py` passes `--visual` through). The released
training datasets include nominal2 renders of every demonstration.

## Should the DR ranges be re-centred? (proposal, not applied)
**Yes, re-centre if nominal2 becomes the reference look.** As things stand, nominal2 lies outside, or at the edge of, the support of the current DR:
* The current DR has no area light and no tinted sky. Its ambient is 0.02-0.18 (nominal2: 0.20). Its robot dark albedo is 0.03-0.35 with metallic 0.5 (nominal2: 0.006, metallic 0). It has no room wall.
* Brightness: in the existing dr_0 / dr_1 renders (frame 0, 36 per variant per camera), the median mean luminance is top 0.31 / 0.35 and wrists 0.26-0.30. Only 14-19% (top) and 3-14% (wrists) of DR frames reach >= 0.40. Real: 0.42-0.44. Nominal2: about 0.43.

Proposed `dr2_train` / `dr2_heldout` (same machinery, levels and held-out logic as now, centred on NOMINAL2; continuous ranges scale with `level`):

| group | parameter | current DR (around nominal) | proposed (around nominal2) |
|---|---|---|---|
| key | type | spot always | area w.p. 0.8 (half-size 0.10-0.45 m: penumbra softness), hard spot w.p. 0.2 (Robocurve photo look) |
| | azimuth / elevation | 30 +-75 deg / 42-78 deg | same azimuth and held-out sectors / 45-80 deg |
| | distance | 1.3-2.0 m | area 0.7-1.3 m, spot 1.3-2.0 m |
| | intensity | x0.55-1.6 log-uniform | x0.6-1.6 of nominal2 (top-cam mean luminance about 0.33-0.53) |
| | colour temperature | 3000-7000 K (held-out bands) | same |
| sky | ambient | 0.02-0.18, neutral | 0.08-0.40 x blackbody 3500-7000 K |
| fill | intensity | 0-0.6 | 0-0.5 |
| room wall | albedo | none | grey 0.03-0.85 (incl. a dark room), hue +-0.05, absent w.p. 0.15 (nominal-like dark void); held-out: 2 tinted walls |
| robot | dark / covers / metallic | 0.03-0.35 / 0.6-0.95 / 0.5 fixed | 0.003-0.03 / 0.55-0.85 / 0-0.3, specular 0.1-0.5, roughness 0.3-0.7 |
| everything else | table / backdrop / floor / objects / camera / image aug | | unchanged (image exposure +-0.4 stops etc. now jitters around the brighter base) |

Implementation sketch: `apply_dr(env, rng, level, split, center="nominal")`. With `center="nominal2"` it takes `nominal2_params()` as the base and samples the rows above; `MODES += ("dr2_train", "dr2_heldout")`. Keep the old modes as they are, for comparability with existing datasets and evals.

## Outcome
nominal2 became the default look for all datasets from round ft10 on (training visual mix nominal2 0.6 / dr_train 0.4; the DR was not re-centred).
* It fixes what was measured: exposure is within 3% on all cameras (was -25 to -38%), shadows are soft (shadow/lit 0.41 -> 0.80), the top-view contrast is closer (-17% vs -41%), and the arm reads white-covers / black-joints.
* It keeps the Robocurve table and backdrop character.
* It is physics-neutral and leaves every existing mode bit-identical.

Open points:
* The room wall is an assumption about the unseen operator side of the Robocurve rig. A photo from the robot's side would settle it; until then, DR should randomise it, including a dark room.
* The remaining contrast gap is scene content (objects, cables, wrist finger pads). Lighting alone should not chase it.
