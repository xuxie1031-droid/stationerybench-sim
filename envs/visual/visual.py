"""Visual appearance alignment + domain randomization for the StationeryBench YAM sim replicas.

Render-only: nothing here touches collision shapes, masses, frictions, poses, the env's episode
RNG or the physx scene.  All scene changes are (a) material edits on existing render shapes,
(b) extra *render-only* sapien entities (no physx component) and (c) light / camera component
parameters.  Image-space augmentation is applied to the rendered uint8 frames.

Public API
----------
apply_nominal(env)                         -> dict   real-bench-aligned look (deterministic)
apply_nominal2(env)                        -> dict   opt-in variant (2026-11-07): same surfaces, brighter
                                                     exposure, soft (area + sky) light, white-cover /
                                                     black-joint arm.  See nominal2/README.md.
apply_dr(env, rng, level=1.0, split="train") -> dict  randomized look around nominal
sample_image_aug(rng, cams, level, split)  -> dict   per-camera image-aug params for one episode
augment_image(img, p, rng)                 -> uint8  apply one camera's aug to an HxWx3 frame
VisualWrapper(env, mode, level, vis_seed)  gym wrapper: applies the look at every reset (after
                                           the env's own reset, then re-renders the reset obs)
                                           and the image aug to every obs.
                                           mode in {"off","nominal","nominal2","dr_train","dr_heldout"}
register_task_hook(env_class_name, fn)     plug in per-task object looks (eraser_from_box etc.)

Why the old "table material change did not take effect": with the ray-tracing shaders (rt-fast)
SAPIEN only re-uploads materials when the render scene's structure changes.  Material edits
alone are ignored until an entity is added/removed.  apply_* always removes and re-adds its own
render-only entities, which forces the refresh (verified: probe1/probe2 in this directory).

Ranges (level=1; continuous jitters scale linearly with `level`, level 0 == nominal look):
see DR_RANGES below - every number used by apply_dr / sample_image_aug lives there.
"""
from __future__ import annotations

import colorsys
import json
import os

import numpy as np
import sapien

HERE = os.path.dirname(os.path.abspath(__file__))
TEX_DIR = os.path.join(HERE, "textures")
VIS = "vis__"                       # prefix of every entity this module adds
MODES = ("off", "nominal", "nominal2", "dr_train", "dr_heldout")

# Table-top aabb of ManiSkill's TableSceneBuilder (world, table rotated 90 deg)
TABLE_X = (-0.7402, 0.4689)
TABLE_Y = (-1.2149, 1.2030)
FLOOR_Z = -0.9196

# ------------------------------------------------------------------------- nominal look
NOMINAL = dict(
    table=dict(tex="nominal/wood.jpg", tint=1.0, roughness=0.45, specular=0.45),
    backdrop=dict(tex="nominal/backdrop.png", tint=1.0, roughness=0.25, specular=0.6),
    floor=dict(rgb=(0.012, 0.012, 0.014)),
    # key: spot light above the arm-base side, shadows fall toward the operator (+x) and to
    # operator-left (-y), as in the setup photos.  az measured from world -x toward +y.
    light=dict(key_az_deg=30.0, key_el_deg=58.0, key_dist=1.6, key_target=(-0.33, 0.0, 0.0),
               key_intensity=9.0, key_temp_k=5000.0, key_inner=0.9, key_outer=1.7,
               fill_intensity=0.25, fill_temp_k=5500.0, ambient=0.08, default_lights=0.0),
    robot=dict(dark=0.10, light=0.85, roughness=0.5),
    camera=dict(),                   # no jitter
)

# ------------------------------------------------------------------------- nominal2 look (opt-in)
# 2026-11-07.  Same table / backdrop / floor textures, object looks and cameras as NOMINAL; only the
# lights and the arm materials differ.  Goal: the exposure / contrast of the real Ai2 YAM training
# data (frame-0 stats of 16 real datasets) with soft shadows, while the surfaces keep the Robocurve
# setup-photo character (dark wood, black backdrop).  Calibration + numbers: nominal2/README.md.
#   light.key_intensity 0 -> no hard spot; the key is an area (parallelogram) light, `area`, 40 x 30 cm,
#   0.9 m from key_target in the nominal spot's direction -> soft penumbrae, photo-like falloff.
#   light.ambient (tinted by ambient_temp_k) is the sky radiance seen by the RT bounce ray (rt-fast:
#   path depth 1), i.e. an ambient-occlusion-like soft fill, not a flat add.
#   robot: covers white plastic, joints / motors / fingers black (as-built 0.9 / 0.25 parts),
#   metallic 0 (as-built 0.5 renders the covers grey).  dark 0.006: the dark wood needs ~2x the
#   nominal light, under which a "real" black albedo (0.03) renders mid-grey next to it.
#   room_wall: matte light wall 0.9 m beyond the operator-side table edge; only the wrist cameras see
#   it (top ~26% of their frame, nominal: near-black floor).  The Ai2 wrist frames show a light wall
#   there (rows 0-92 mean luminance 0.52).  Not in the Robocurve photos (taken from that side).
NOMINAL2 = dict(
    table=dict(NOMINAL["table"]),
    backdrop=dict(NOMINAL["backdrop"]),
    floor=dict(NOMINAL["floor"]),
    light=dict(key_az_deg=30.0, key_el_deg=58.0, key_dist=1.6, key_target=(-0.33, 0.0, 0.0),
               key_intensity=0.0, key_temp_k=5000.0, key_inner=0.9, key_outer=1.7,
               fill_intensity=0.25, fill_temp_k=5500.0, ambient=0.2, ambient_temp_k=4800.0, default_lights=0.0,
               area=[dict(az_deg=30.0, el_deg=62.0, dist=0.9, target=(-0.33, 0.0, 0.0),
                          half_w=0.2, half_h=0.15, intensity=54.0, temp_k=4800.0)]),
    robot=dict(dark=0.006, light=0.72, roughness=0.5, dark_roughness=0.4, light_roughness=0.55,
               metallic=0.0, specular=0.25),
    room_wall=dict(rgb=(0.68, 0.67, 0.64), dist=0.9, half_w=3.0, top_z=1.6),
    camera=dict(),
)

DR_RANGES = dict(
    # lights
    key_az_deg=75.0,                 # +- around nominal azimuth
    key_el_deg=(42.0, 78.0),         # absolute elevation range
    key_dist=(1.3, 2.0),             # m
    key_intensity_scale=(0.55, 1.6),  # x nominal (log-uniform)
    key_temp_k=(3000.0, 7000.0),
    fill_intensity=(0.0, 0.6),
    ambient=(0.02, 0.18),
    default_lights=(0.0, 0.25),      # ManiSkill's two directional lights, re-enabled at random
    # held-out light bins (never sampled by split="train")
    heldout_temp_bands=((3600.0, 4000.0), (5600.0, 6000.0)),
    heldout_az_sectors=((-45.0, -25.0), (55.0, 75.0)),   # relative to nominal az
    # surfaces
    table_tint=(0.75, 1.25), table_roughness=(0.3, 0.75),
    backdrop_tint=(0.7, 1.4), floor_v=(0.0, 0.12),
    # objects
    obj_hue=0.03,                    # +- hue shift (fraction of the colour wheel, 0.03 ~ 11 deg)
    obj_hue_heldout=(0.03, 0.05),    # held-out: |shift| in this band
    obj_sat=(0.8, 1.2), obj_val=(0.8, 1.2),
    robot_dark=(0.03, 0.35), robot_light=(0.6, 0.95), robot_roughness=(0.3, 0.8),
    # cameras (extrinsics relative to the mount, intrinsics)
    cam_trans_top=0.015,             # m, +- per axis (top_cam)
    cam_trans_wrist=0.005,           # m, +- per axis (wrist cams; rigid mounts)
    cam_rot_deg=2.0,                 # +- per axis (all cams)
    cam_focal=0.03,                  # focal length x (1 +- 0.03)  (== fov +-3%)
    # image space (per camera per episode; noise per frame)
    img_exposure_stops=0.4,          # gain 2**U(-0.4, 0.4)
    img_gamma=(0.85, 1.15), img_contrast=(0.85, 1.15), img_saturation=(0.8, 1.2),
    img_wb=0.08,                     # R and B gains 1 +- 0.08 (G fixed)
    img_noise_sigma=(0.0, 4.0),      # 8-bit units, gaussian per pixel per frame
    img_blur_p=0.5, img_blur_sigma=(0.3, 1.0),
    img_vignette_top=(0.0, 0.2), img_vignette_wrist=(0.0, 0.35),
)

CUP_PALETTE = {   # sRGB 0..1 (real: white ribbed ceramic-look cups)
    "nominal": (0.90, 0.89, 0.86),
    "train": [(0.95, 0.95, 0.93), (0.92, 0.89, 0.80), (0.80, 0.80, 0.80), (0.78, 0.66, 0.50),
              (0.75, 0.84, 0.92), (0.93, 0.80, 0.80), (0.80, 0.90, 0.78), (0.98, 0.98, 0.98),
              (0.62, 0.62, 0.64), (0.90, 0.85, 0.70)],
    "heldout": [(0.95, 0.88, 0.60), (0.70, 0.75, 0.95), (0.55, 0.45, 0.35)],
}
_TASK_HOOKS = {}
_TEXCACHE = {}


def register_task_hook(env_class_name, fn):
    """fn(env, ctx) where ctx = dict(mode='nominal'|'dr', rng, level, split, mats=_MatStore)."""
    _TASK_HOOKS[env_class_name] = fn


# ------------------------------------------------------------------------- helpers
def _u(env):
    return env.unwrapped


def _ss(env):
    return _u(env).scene.sub_scenes[0]


def _tex_file(rel):
    path = rel if os.path.isabs(rel) else os.path.join(TEX_DIR, rel)
    if path not in _TEXCACHE:
        try:
            _TEXCACHE[path] = sapien.render.RenderTexture2D(path, mipmap_levels=6)
        except Exception:  # noqa: BLE001
            _TEXCACHE[path] = sapien.render.RenderTexture2D(path)
    return _TEXCACHE[path]


def _tex_array(key, rgb_u8):
    if key not in _TEXCACHE:
        a = np.concatenate([rgb_u8, np.full(rgb_u8.shape[:2] + (1,), 255, np.uint8)], -1)
        try:   # mipmaps avoid moire on fine weave / print at grazing angles
            _TEXCACHE[key] = sapien.render.RenderTexture2D(np.ascontiguousarray(a), "R8G8B8A8Unorm", mipmap_levels=6, srgb=True)
        except Exception:  # noqa: BLE001
            _TEXCACHE[key] = sapien.render.RenderTexture2D(np.ascontiguousarray(a), "R8G8B8A8Unorm", srgb=True)
    return _TEXCACHE[key]


def _parts(rb):
    for si, sh in enumerate(rb.render_shapes):
        parts = sh.parts if hasattr(sh, "parts") else []
        for pi, p in enumerate(parts):
            yield si, pi, p.material


def kelvin_rgb(t):
    """Approximate blackbody colour (Tanner Helland fit), normalised to unit luminance."""
    t = t / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * np.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * np.log(t - 10) - 305.0447927307)
    c = np.clip(np.array([r, g, b]) / 255.0, 0, 1)
    c = c ** 2.2                                   # to linear
    return c / (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def _hsv_jitter(rgb, dh, ds, dv):
    h, s, v = colorsys.rgb_to_hsv(*[float(x) for x in rgb[:3]])
    r, g, b = colorsys.hsv_to_rgb((h + dh) % 1.0, float(np.clip(s * ds, 0, 1)), float(np.clip(v * dv, 0, 1)))
    return [r, g, b]


def _lu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


class _MatStore:
    """Remembers each render part's original material values so DR is always applied relative
    to the as-built look (needed for envs that do not rebuild their scene at every reset)."""

    def __init__(self):
        self.orig = {}

    def get(self, ent, si, pi, m):
        k = (ent.name, si, pi)
        if k not in self.orig:
            self.orig[k] = dict(base_color=list(m.base_color), roughness=m.roughness, specular=m.specular,
                                metallic=m.metallic, tex=m.base_color_texture)
        return self.orig[k]

    def restore(self, ent, si, pi, m):
        o = self.get(ent, si, pi, m)
        m.base_color = o["base_color"]; m.roughness = o["roughness"]; m.specular = o["specular"]
        m.metallic = o["metallic"]
        if o["tex"] is not None:
            m.base_color_texture = o["tex"]
        return o


def _state(env):
    """Per-scene state; reset when the env rebuilt its scene (marker entity gone)."""
    u, ss = _u(env), _ss(env)
    st = getattr(u, "_vis_state", None)
    has_marker = any(e.name == VIS + "marker" for e in ss.entities)
    if st is None or not has_marker or st.get("ss_id") != id(ss):
        st = dict(ss_id=id(ss), mats=_MatStore(), cams={}, lights=None)
        u._vis_state = st
    return st


def _robot_entities(u):
    return {l._objs[0].entity.name for l in u.agent.robot.get_links()}


def _named_entities(ss):
    return {e.name: e for e in ss.entities}


def _rb(e):
    return e.find_component_by_type(sapien.render.RenderBodyComponent)


def _add_render_box(ss, name, half, pose, material):
    e = sapien.Entity(); e.name = VIS + name
    rb = sapien.render.RenderBodyComponent()
    rb.attach(sapien.render.RenderShapeBox(list(half), material))
    e.add_component(rb); e.set_pose(pose); ss.add_entity(e)
    return e


def _quad_shape(center, ax_u, ax_v, half_u, half_v, material):
    """Render-only textured rectangle with explicit UVs (u along ax_u, v along ax_v, both 0..1;
    image column = u, image row = v).  RenderShapeBox uses an atlas-style per-face UV layout
    (measured: the top face covers only a sub-rectangle of the texture), so textured surfaces we
    own are built from explicit quads instead."""
    c, eu, ev = (np.asarray(x, np.float32) for x in (center, ax_u, ax_v))
    n = np.cross(eu, ev).astype(np.float32)
    V = np.stack([c - eu * half_u - ev * half_v, c + eu * half_u - ev * half_v,
                  c + eu * half_u + ev * half_v, c - eu * half_u + ev * half_v]).astype(np.float32)
    UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    F = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
    return sapien.render.RenderShapeTriangleMesh(V, F, np.repeat(n[None], 4, 0), UV, material)


def _add_render_shape(ss, name, shape, pose=None):
    e = sapien.Entity(); e.name = VIS + name
    rb = sapien.render.RenderBodyComponent(); rb.attach(shape)
    e.add_component(rb); e.set_pose(pose or sapien.Pose()); ss.add_entity(e)
    return e


def _add_light(ss, name, comp, pose):
    e = sapien.Entity(); e.name = VIS + name
    e.add_component(comp); e.set_pose(pose); ss.add_entity(e)
    return e


def _look_at_pose(eye, target):
    from mani_skill.utils import sapien_utils
    return sapien_utils.look_at(eye=list(eye), target=list(target)).sp


# ------------------------------------------------------------------------- core
def _apply(env, P, ctx):
    """Apply a fully-specified look P (see apply_nominal for the schema)."""
    u, ss = _u(env), _ss(env)
    st = _state(env)
    mats = st["mats"]
    # 1. remove our previous entities (also forces the RT material refresh on re-add)
    for e in list(ss.entities):
        if e.name.startswith(VIS):
            ss.remove_entity(e)
    ents = _named_entities(ss)
    robot = _robot_entities(u)
    # 2. table: hide the ManiSkill table mesh, add a textured render box whose top face is z=0
    table_e = u.table_scene.table._objs[0]
    trb = _rb(table_e)
    trb.visibility = 0.0
    T = P["table"]
    mt = sapien.render.RenderMaterial(base_color=[T["tint"]] * 3 + [1.0], roughness=T["roughness"],
                                      specular=T["specular"])
    mt.base_color_texture = _tex_file(T["tex"])
    hx, hy, hz = (TABLE_X[1] - TABLE_X[0]) / 2, (TABLE_Y[1] - TABLE_Y[0]) / 2, 0.02
    cx, cy = (TABLE_X[0] + TABLE_X[1]) / 2, (TABLE_Y[0] + TABLE_Y[1]) / 2
    # textured top quad at z=0: u (image columns) along +x, v (rows) along +y; image 1536x3072
    _add_render_shape(ss, "table", _quad_shape([cx, cy, 0.0], [1, 0, 0], [0, 1, 0], hx, hy, mt))
    medge = sapien.render.RenderMaterial(base_color=[0.06, 0.045, 0.035, 1.0], roughness=0.6)
    _add_render_box(ss, "table_slab", (hx, hy, hz), sapien.Pose([cx, cy, -hz - 0.0005]), medge)
    # table body below the top (dark, matte) so side views do not see through
    mtb = sapien.render.RenderMaterial(base_color=[0.03, 0.025, 0.02, 1.0], roughness=0.9)
    _add_render_box(ss, "table_body", (hx - 0.01, hy - 0.01, (FLOOR_Z * -1 - 0.05) / 2),
                    sapien.Pose([(TABLE_X[0] + TABLE_X[1]) / 2, (TABLE_Y[0] + TABLE_Y[1]) / 2,
                                 (FLOOR_Z + 0.0) / 2 - 0.02]), mtb)
    # 3. floor: recolour the ManiSkill ground plane (drop the grid texture)
    g = u.table_scene.ground._objs[0]
    for si, pi, m in _parts(_rb(g)):
        mats.get(g, si, pi, m)
        m.base_color_texture = None
        m.base_color = list(P["floor"]["rgb"]) + [1.0]
        m.roughness = 0.9; m.specular = 0.1
    # 4. backdrop: vertical panel just behind the table's far (arm-base) edge
    B = P["backdrop"]
    mb = sapien.render.RenderMaterial(base_color=[B["tint"]] * 3 + [1.0], roughness=B["roughness"],
                                      specular=B["specular"])
    mb.base_color_texture = _tex_file(B["tex"])
    # top at z=0.35 m: tall enough for every policy/operator view, low enough that the key light
    # (placed above and behind the arm bases) is not occluded by it.
    mbb = sapien.render.RenderMaterial(base_color=[0.01, 0.01, 0.012, 1.0], roughness=0.5)
    _add_render_box(ss, "backdrop", (0.01, 2.0, 0.635), sapien.Pose([TABLE_X[0] - 0.04, 0.0, -0.285]), mbb)
    # textured front face (facing +x): u along -y, v along -z (image top = top of the panel)
    _add_render_shape(ss, "backdrop_face", _quad_shape([TABLE_X[0] - 0.0295, 0.0, -0.285], [0, -1, 0], [0, 0, -1],
                                                         2.0, 0.635, mb))
    # 4b. optional matte room wall on the operator side (+x), seen only by the wrist cameras above the table's
    # near edge (nominal2; NOMINAL / DR have no such key -> nothing added)
    RW = P.get("room_wall")
    if RW:
        mw = sapien.render.RenderMaterial(base_color=list(RW["rgb"]) + [1.0], roughness=RW.get("roughness", 0.9),
                                          specular=RW.get("specular", 0.2))
        zc = (FLOOR_Z + RW["top_z"]) / 2
        _add_render_shape(ss, "room_wall", _quad_shape([TABLE_X[1] + RW["dist"], 0.0, zc], [0, 1, 0], [0, 0, -1],
                                                       RW["half_w"], (RW["top_z"] - FLOOR_Z) / 2, mw))
    # 5. lights
    L = P["light"]
    rs = ss.render_system
    if st["lights"] is None:
        st["lights"] = [(l, np.array(l.get_color())) for l in rs.get_lights() if not l.entity.name.startswith(VIS)]
    for l, c0 in st["lights"]:
        l.set_color((c0 * L["default_lights"]).tolist())
    if "ambient_temp_k" in L:        # nominal2: tinted sky (unit-luminance blackbody colour x ambient)
        rs.set_ambient_light((kelvin_rgb(L["ambient_temp_k"]) * L["ambient"]).tolist())
    else:
        rs.set_ambient_light([L["ambient"]] * 3)
    az, el = np.deg2rad(L["key_az_deg"]), np.deg2rad(L["key_el_deg"])
    tgt = np.array(L["key_target"])
    d = np.array([-np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    if L["key_intensity"] > 0:       # always true for nominal / DR; nominal2 uses an area key instead
        sp = sapien.render.RenderSpotLightComponent()
        sp.set_color((kelvin_rgb(L["key_temp_k"]) * L["key_intensity"]).tolist())
        sp.inner_fov = L["key_inner"]; sp.outer_fov = L["key_outer"]
        # NB: do not call sp.enable_shadow() - with default shadow near/far the RT spot light
        # contributes nothing; ray-traced shadows are produced regardless.
        _add_light(ss, "key", sp, _look_at_pose(tgt + L["key_dist"] * d, tgt))
    dl = sapien.render.RenderDirectionalLightComponent()
    dl.set_color((kelvin_rgb(L["fill_temp_k"]) * L["fill_intensity"]).tolist())
    _add_light(ss, "fill", dl, _look_at_pose([0.6, 0.0, 0.9], [-0.3, 0.0, 0.0]))
    # optional area (parallelogram) lights - nominal2 only (absent from NOMINAL / DR params).
    # Soft shadows: RT samples a random point on the parallelogram per shading point.
    for k, A in enumerate(L.get("area") or ()):
        a_az, a_el = np.deg2rad(A["az_deg"]), np.deg2rad(A["el_deg"])
        a_t = np.array(A["target"], np.float64)
        a_d = np.array([-np.cos(a_el) * np.cos(a_az), np.cos(a_el) * np.sin(a_az), np.sin(a_el)])
        pl = sapien.render.RenderParallelogramLightComponent()
        pl.set_shape(float(A["half_w"]), float(A["half_h"]))
        pl.set_color((kelvin_rgb(A["temp_k"]) * A["intensity"]).tolist())
        # one-sided emitter; with a look-at pose (x toward the target) it emits toward the target
        _add_light(ss, f"area{k}", pl, _look_at_pose(a_t + A["dist"] * a_d, a_t))
    # 6. robot colours (two tones: dark links / light covers, as built 0.25 / 0.9)
    R = P["robot"]
    # The robot's 28 render parts share 7 material objects: record every part's as-built values
    # before editing any of them (otherwise a part visited after its shared material was edited
    # would record the edited metallic as "original").  Classification below is unchanged.
    for name in robot:
        e = ents.get(name)
        if e is not None and _rb(e) is not None:
            for si, pi, m in _parts(_rb(e)):
                mats.get(e, si, pi, m)
    for name in robot:
        e = ents.get(name)
        if e is None or _rb(e) is None:
            continue
        for si, pi, m in _parts(_rb(e)):
            o = mats.get(e, si, pi, m)
            is_light = np.mean(o["base_color"][:3]) > 0.6
            v = R["light"] if is_light else R["dark"]
            m.base_color = [v, v, v, o["base_color"][3]]
            m.roughness = R.get("light_roughness" if is_light else "dark_roughness", R["roughness"])
            # metallic / specular: nominal2 sets them; nominal / DR keep the as-built values (the
            # restore is a no-op unless a nominal2 look was applied earlier in the same process)
            m.metallic = R.get("metallic", o["metallic"])
            m.specular = R.get("specular", o["specular"])
    # 7. task objects
    others = [e for n, e in ents.items() if n not in robot and not n.startswith(VIS)
              and e is not table_e and e is not g and _rb(e) is not None]
    hook = _TASK_HOOKS.get(type(u).__name__, _generic_objects)
    ctx = dict(ctx, mats=mats, objects=others, P=P)
    P["objects"] = hook(env, ctx)
    # 8. cameras
    _apply_cameras(env, st, P.get("camera", {}))
    # marker last (entity add also triggers the RT refresh)
    e = sapien.Entity(); e.name = VIS + "marker"; ss.add_entity(e)
    return P


def _apply_cameras(env, st, C):
    u = _u(env)
    for uid, sensor in u._sensors.items():
        cam = getattr(sensor, "camera", None)
        if cam is None or not hasattr(cam, "_render_cameras"):
            continue
        rc = cam._render_cameras[0]
        if uid not in st["cams"]:
            K = rc.get_intrinsic_matrix()
            st["cams"][uid] = (rc.get_local_pose(), float(K[0, 0]), float(K[1, 1]))
        p0, fx, fy = st["cams"][uid]
        j = C.get(uid)
        if j is None:
            rc.set_local_pose(p0); rc.set_focal_lengths(fx, fy)
        else:
            from transforms3d.euler import euler2quat
            from transforms3d.quaternions import qmult
            dq = euler2quat(*np.deg2rad(j["rot_deg"]))
            rc.set_local_pose(sapien.Pose(p=np.asarray(p0.p) + np.asarray(j["trans"]), q=qmult(p0.q, dq)))
            rc.set_focal_lengths(fx * j["focal"], fy * j["focal"])
        for a in ("_cached_local_pose", "_cached_intrinsic_matrix", "_cached_extrinsic_matrix", "_cached_model_matrix"):
            if hasattr(cam, a):
                setattr(cam, a, None)


# ------------------------------------------------------------------------- object hooks
def _jitter_entity(ctx, e, dh, ds, dv, rough=None):
    mats = ctx["mats"]
    for si, pi, m in _parts(_rb(e)):
        o = mats.restore(e, si, pi, m)
        if ctx["mode"] == "dr":
            m.base_color = _hsv_jitter(o["base_color"], dh, ds, dv) + [o["base_color"][3]]
            if rough is not None:
                m.roughness = rough


def _obj_jitter_params(ctx):
    rng, lvl = ctx["rng"], ctx["level"]
    R = DR_RANGES
    if ctx["split"] == "heldout":
        dh = rng.uniform(*R["obj_hue_heldout"]) * rng.choice([-1, 1]) * lvl
    else:
        dh = rng.uniform(-R["obj_hue"], R["obj_hue"]) * lvl
    ds = 1 + (rng.uniform(*R["obj_sat"]) - 1) * lvl
    dv = 1 + (rng.uniform(*R["obj_val"]) - 1) * lvl
    return float(dh), float(ds), float(dv)


def _generic_objects(env, ctx):
    """Unknown task: restore as-built colours (nominal) or per-actor HSV jitter (DR)."""
    out = {}
    for e in ctx["objects"]:
        if ctx["mode"] == "dr":
            dh, ds, dv = _obj_jitter_params(ctx)
            out[e.name] = dict(dh=dh, ds=ds, dv=dv)
        else:
            dh, ds, dv = 0.0, 1.0, 1.0
        _jitter_entity(ctx, e, dh, ds, dv)
    return out


def _ruler_hook(env, ctx):
    """Cups: white ribbed cups in the photo -> nominal (0.90,0.89,0.86), slightly glossy; DR from a
    palette (same colour for both cups w.p. 0.7).  Ruler: clear tint/alpha, ink, label band."""
    rng, mode, split = ctx["rng"], ctx["mode"], ctx["split"]
    mats = ctx["mats"]
    out = {}
    if mode == "dr":
        pal = CUP_PALETTE[split]
        c1 = pal[int(rng.integers(len(pal)))]
        c2 = c1 if rng.random() < 0.7 else pal[int(rng.integers(len(pal)))]
        cup_rough = float(rng.uniform(0.35, 0.9))
        tint = [float(rng.uniform(0.8, 1.0)) for _ in range(3)]
        alpha = float(rng.uniform(0.15, 0.5))
        ink = float(rng.uniform(0.02, 0.2)); ink_rgb = [ink, ink, min(1.0, ink * float(rng.uniform(1.0, 2.0)))]
        label = _hsv_jitter((0.95, 0.95, 0.95), 0.0, 1.0, float(rng.uniform(0.85, 1.05)))
    else:
        c1 = c2 = CUP_PALETTE["nominal"]; cup_rough = 0.5
        tint, alpha, ink_rgb, label = [0.90, 0.95, 1.0], 0.30, [0.08, 0.08, 0.08], [0.95, 0.95, 0.95]
    for e in ctx["objects"]:
        if e.name.endswith("cup_src") or e.name.endswith("cup_dst"):
            col = c1 if e.name.endswith("cup_src") else c2
            for si, pi, m in _parts(_rb(e)):
                o = mats.restore(e, si, pi, m)
                if o["base_color"][0] < 0.95:          # outer wall (inner/rim/floor are 0.97)
                    m.base_color = list(col) + [1.0]; m.roughness = cup_rough
                else:
                    m.base_color = [min(1.0, c * 1.05) for c in col] + [1.0]
            out[e.name] = list(col)
        elif e.name.endswith("ruler"):
            for si, pi, m in _parts(_rb(e)):
                o = mats.restore(e, si, pi, m)
                if o["base_color"][3] < 0.99:
                    m.base_color = tint + [alpha]
                elif o["base_color"][0] < 0.5:
                    m.base_color = ink_rgb + [1.0]
                else:
                    m.base_color = list(label) + [1.0]
            out["ruler"] = dict(tint=tint, alpha=alpha, ink=ink_rgb, label=list(label))
        else:
            _jitter_entity(ctx, e, *(_obj_jitter_params(ctx) if mode == "dr" else (0.0, 1.0, 1.0)))
    out["cup_roughness"] = cup_rough
    return out


def _cloth_tex(rgb, kind, seed=0, S=1024):
    """Woven cloth texture baked with colour `rgb` (linear base colour 0..1 -> sRGB in the image).
    kind 'outer': ~1 mm plain weave + slub noise (olive-grey book cloth); 'lining': finer, flatter
    black lining cloth."""
    key = ("cloth", kind, tuple(round(float(c), 3) for c in rgb), seed)
    if key in _TEXCACHE:
        return _TEXCACHE[key]
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    per = 4.0 if kind == "outer" else 3.0
    wu = 0.5 + 0.5 * np.cos(2 * np.pi * xx / per); wv = 0.5 + 0.5 * np.cos(2 * np.pi * yy / per)
    checker = ((xx // per + yy // per) % 2).astype(np.float32)
    weave = 0.6 * (checker * wu + (1 - checker) * wv) + 0.4
    from PIL import Image
    def noise(cell):
        g = r.standard_normal((S // cell + 2, S // cell + 2)).astype(np.float32)
        return np.asarray(Image.fromarray(g, mode="F").resize((S + 2 * cell, S + 2 * cell), Image.BICUBIC))[cell:cell + S, cell:cell + S]
    slub = 0.6 * noise(3) * (noise(40) > 0.3) + 0.4 * noise(60)
    amp = (0.18, 0.08) if kind == "outer" else (0.10, 0.05)
    mod = 1 + amp[0] * (weave - 0.7) / 0.3 + amp[1] * slub
    lin = np.clip(np.asarray(rgb[:3], np.float32)[None, None] * mod[..., None], 0, 1)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return _tex_array(key, (srgb * 255 + 0.5).astype(np.uint8))


# as-built colours (owner, by eye) used to classify box primitives
ERASER_BUILT = dict(cloth=(0.36, 0.35, 0.28), cloth_dark=(0.30, 0.29, 0.23), black=(0.025, 0.025, 0.03),
                    wood=(0.45, 0.30, 0.17))
# calibrated (linear albedo) against eraser_from_box-setup.jpg under the nominal lights: in the
# photo the cloth is about as bright as the table mid-tones (table albedo ~0.06-0.10 linear),
# the lining about half, the page-block walls a warm mid brown.
ERASER_NOMINAL = dict(cloth=(0.17, 0.16, 0.12), cloth_dark=(0.12, 0.112, 0.085), black=(0.022, 0.024, 0.028),
                      wood=(0.20, 0.115, 0.055))


def _eraser_hook(env, ctx):
    """eraser_from_box.  Book box (links box_base/box_lid): v2 render-only UV quads (*_uv, see
    eraser_task.UV_QUADS) get woven-cloth textures: lid_outer_uv olive-grey cloth, *_lining_uv
    black lining cloth.  Box primitives keep flat colours by class (cloth / dark cloth / black /
    wood, matched on the as-built colour).  DR: one shared HSV jitter for the cloth, value jitter
    for the lining, independent jitter for the wood and for the eraser."""
    mats, mode, rng = ctx["mats"], ctx["mode"], ctx["rng"]
    N = ERASER_NOMINAL
    if mode == "dr":
        jc, jw, je = _obj_jitter_params(ctx), _obj_jitter_params(ctx), _obj_jitter_params(ctx)
        bv = float(1 + rng.uniform(-0.5, 1.0) * ctx["level"])      # black lining value x0.5-2
        seed = int(rng.integers(4))
    else:
        jc = jw = je = (0.0, 1.0, 1.0); bv = 1.0; seed = 0
    cloth = _hsv_jitter(N["cloth"], *jc); cloth_d = _hsv_jitter(N["cloth_dark"], *jc)
    black = [min(1.0, c * bv) for c in N["black"]]; wood = _hsv_jitter(N["wood"], *jw)
    out = dict(cloth=cloth, black=black, wood=wood, eraser=je, textured=[])

    def cls(c):
        c = np.asarray(c[:3])
        for k, v in ERASER_BUILT.items():
            if np.abs(c - np.asarray(v)).max() < 0.02:
                return k
        return None

    col = dict(cloth=cloth, cloth_dark=cloth_d, black=black, wood=wood)
    for e in ctx["objects"]:
        if "box" not in e.name:
            _jitter_entity(ctx, e, *je)
            continue
        rb = _rb(e)
        for si, sh in enumerate(rb.render_shapes):
            name = sh.name if hasattr(sh, "name") else ""
            for pi, part in enumerate(sh.parts if hasattr(sh, "parts") else []):
                m = part.material
                o = mats.restore(e, si, pi, m)
                if name.endswith("_uv"):
                    kind = "outer" if name == "lid_outer_uv" else "lining"
                    m.base_color = [1.0, 1.0, 1.0, 1.0]
                    m.base_color_texture = _cloth_tex(cloth if kind == "outer" else black, kind, seed)
                    m.roughness = 0.95 if kind == "outer" else 0.9
                    out["textured"].append(name)
                else:
                    k = cls(o["base_color"])
                    if k is not None:
                        m.base_color = list(col[k]) + [1.0]
    out["textured"] = sorted(set(out["textured"]))
    return out


register_task_hook("StationeryEraserFromBoxEnv", _eraser_hook)
register_task_hook("StationeryRulerHandoverEnv", _ruler_hook)
# Any other env class falls back to _generic_objects (per-actor HSV jitter); owners can override
# with visual.register_task_hook("<EnvClassName>", fn).


# ------------------------------------------------------------------------- public API
def _pool():
    return json.load(open(os.path.join(TEX_DIR, "pool.json")))


def nominal_params():
    P = json.loads(json.dumps(NOMINAL))
    return P


def apply_nominal(env):
    """Real-bench-aligned look.  Deterministic; returns the parameter dict."""
    P = nominal_params()
    return _apply(env, P, dict(mode="nominal", rng=np.random.default_rng(0), level=0.0, split="nominal"))


def nominal2_params():
    P = json.loads(json.dumps(NOMINAL2))
    P["look"] = "nominal2"
    return P


def apply_nominal2(env):
    """Opt-in nominal2 look (brighter, soft light, white-cover arm).  Deterministic; object looks are
    the nominal ones (task hooks run with mode='nominal')."""
    P = nominal2_params()
    return _apply(env, P, dict(mode="nominal", rng=np.random.default_rng(0), level=0.0, split="nominal"))


def _sample_light(rng, level, split):
    R, N = DR_RANGES, NOMINAL["light"]
    lvl = level

    def in_bands(x, bands):
        return any(lo <= x < hi for lo, hi in bands)

    if split == "heldout":
        b = R["heldout_temp_bands"][int(rng.integers(len(R["heldout_temp_bands"])))]
        temp = rng.uniform(*b)
        s = R["heldout_az_sectors"][int(rng.integers(len(R["heldout_az_sectors"])))]
        daz = rng.uniform(*s)
    else:
        while True:
            temp = rng.uniform(*R["key_temp_k"])
            if not in_bands(temp, R["heldout_temp_bands"]):
                break
        while True:
            daz = rng.uniform(-R["key_az_deg"], R["key_az_deg"])
            if not in_bands(daz, R["heldout_az_sectors"]):
                break
    # level scales the deviation from nominal
    temp = N["key_temp_k"] + (temp - N["key_temp_k"]) * lvl
    daz = daz * lvl
    el = N["key_el_deg"] + (rng.uniform(*R["key_el_deg"]) - N["key_el_deg"]) * lvl
    dist = N["key_dist"] + (rng.uniform(*R["key_dist"]) - N["key_dist"]) * lvl
    inten = N["key_intensity"] * _lu(rng, *R["key_intensity_scale"]) ** lvl
    return dict(key_az_deg=float(N["key_az_deg"] + daz), key_el_deg=float(el), key_dist=float(dist),
                key_target=list(N["key_target"]), key_intensity=float(inten), key_temp_k=float(temp),
                key_inner=N["key_inner"], key_outer=N["key_outer"],
                fill_intensity=float(N["fill_intensity"] + (rng.uniform(*R["fill_intensity"]) - N["fill_intensity"]) * lvl),
                fill_temp_k=float(rng.uniform(4000, 7000)),
                ambient=float(N["ambient"] + (rng.uniform(*R["ambient"]) - N["ambient"]) * lvl),
                default_lights=float(rng.uniform(*R["default_lights"]) * lvl))


def _sample_cameras(env, rng, level):
    R = DR_RANGES
    out = {}
    for uid in _u(env)._sensors:
        t = R["cam_trans_top"] if uid == "top_cam" else R["cam_trans_wrist"]
        out[uid] = dict(trans=(rng.uniform(-t, t, 3) * level).tolist(),
                        rot_deg=(rng.uniform(-R["cam_rot_deg"], R["cam_rot_deg"], 3) * level).tolist(),
                        focal=float(1 + rng.uniform(-R["cam_focal"], R["cam_focal"]) * level))
    return out


def apply_dr(env, rng, level=1.0, split="train"):
    """Randomized look.  rng: np.random.Generator (use a per-episode generator that is independent
    of the env's episode RNG).  split: 'train' or 'heldout' (disjoint texture/palette pools and
    light bins, see DR_RANGES).  Returns the sampled parameter dict (JSON-serialisable)."""
    assert split in ("train", "heldout")
    R = DR_RANGES
    pool = _pool()[split]
    lvl = float(level)
    P = nominal_params()
    if lvl > 0:
        P["table"] = dict(tex=pool["wood"][int(rng.integers(len(pool["wood"])))],
                          tint=float(1 + (rng.uniform(*R["table_tint"]) - 1) * lvl),
                          roughness=float(0.45 + (rng.uniform(*R["table_roughness"]) - 0.45) * lvl),
                          specular=float(rng.uniform(0.3, 0.6)))
        P["backdrop"] = dict(tex=pool["backdrop"][int(rng.integers(len(pool["backdrop"])))],
                             tint=float(1 + (rng.uniform(*R["backdrop_tint"]) - 1) * lvl),
                             roughness=float(rng.uniform(0.2, 0.9)), specular=float(rng.uniform(0.2, 0.7)))
        fv = rng.uniform(*R["floor_v"]) * lvl
        P["floor"] = dict(rgb=[float(fv * rng.uniform(0.85, 1.15)) for _ in range(3)])
        P["light"] = _sample_light(rng, lvl, split)
        P["robot"] = dict(dark=float(0.10 + (rng.uniform(*R["robot_dark"]) - 0.10) * lvl),
                          light=float(0.85 + (rng.uniform(*R["robot_light"]) - 0.85) * lvl),
                          roughness=float(0.5 + (rng.uniform(*R["robot_roughness"]) - 0.5) * lvl))
        P["camera"] = _sample_cameras(env, rng, lvl)
    P["split"] = split; P["level"] = lvl
    return _apply(env, P, dict(mode="dr" if lvl > 0 else "nominal", rng=rng, level=lvl, split=split))


# ------------------------------------------------------------------------- image space
def sample_image_aug(rng, cams=("top_cam", "left_cam", "right_cam"), level=1.0, split="train"):
    """Per-camera, per-episode image-aug parameters (split does not change these ranges)."""
    R = DR_RANGES
    out = {}
    for c in cams:
        vig = R["img_vignette_top"] if c == "top_cam" else R["img_vignette_wrist"]
        out[c] = dict(
            gain=float(2 ** (rng.uniform(-R["img_exposure_stops"], R["img_exposure_stops"]) * level)),
            gamma=float(1 + (rng.uniform(*R["img_gamma"]) - 1) * level),
            contrast=float(1 + (rng.uniform(*R["img_contrast"]) - 1) * level),
            saturation=float(1 + (rng.uniform(*R["img_saturation"]) - 1) * level),
            wb=[float(1 + rng.uniform(-R["img_wb"], R["img_wb"]) * level), 1.0,
                float(1 + rng.uniform(-R["img_wb"], R["img_wb"]) * level)],
            noise=float(rng.uniform(*R["img_noise_sigma"]) * level),
            blur=float(rng.uniform(*R["img_blur_sigma"]) * level) if rng.random() < R["img_blur_p"] else 0.0,
            vignette=float(rng.uniform(*vig) * level),
        )
    return out


_VIG = {}
_NOISE = {}


def _lut(p):
    """Per-channel 8-bit LUT: WB*exposure gain -> contrast about mid-grey -> gamma."""
    x = np.arange(256, dtype=np.float32)[:, None] / 255.0
    x = x * (np.asarray(p["wb"], np.float32) * p["gain"])[None]
    x = (x - 0.5) * p["contrast"] + 0.5
    x = np.clip(x, 0, 1) ** p["gamma"]
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8).reshape(1, 256, 3)


def augment_image(img, p, rng):
    """img uint8 HxWx3 -> uint8.  Order: blur (optics) -> saturation -> LUT(WB, exposure,
    contrast about 0.5, gamma) -> vignette -> gaussian sensor noise (per frame) -> quantise.
    ~4 ms per 640x360 frame (cv2)."""
    import cv2
    x = np.ascontiguousarray(img)
    if p["blur"] > 0:
        x = cv2.GaussianBlur(x, (0, 0), p["blur"])
    if abs(p["saturation"] - 1) > 1e-3:
        g = cv2.cvtColor(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        x = cv2.addWeighted(x, p["saturation"], g, 1 - p["saturation"], 0)
    if "_lut" not in p:
        p["_lut"] = _lut(p)
    x = cv2.LUT(x, p["_lut"])
    H, W = x.shape[:2]
    if p["vignette"] > 0 or p["noise"] > 0:
        y = x.astype(np.float32)
        if p["vignette"] > 0:
            if (H, W) not in _VIG:
                yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
                _VIG[(H, W)] = (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (W / 2)) ** 2)[..., None]
            y *= (1 - p["vignette"] * _VIG[(H, W)])
        if p["noise"] > 0:
            if (H, W) not in _NOISE:   # bank of unit-gaussian frames; random frame + roll per call
                _NOISE[(H, W)] = np.random.default_rng(1234).standard_normal((8, H, W, 3)).astype(np.float32)
            bank = _NOISE[(H, W)]
            n = np.roll(bank[int(rng.integers(8))], (int(rng.integers(H)), int(rng.integers(W))), (0, 1))
            y += n * p["noise"]
        x = np.clip(y + 0.5, 0, 255).astype(np.uint8)
    return x


# ------------------------------------------------------------------------- wrapper
def episode_rng(vis_seed, ep_seed, variant=0):
    return np.random.default_rng([int(vis_seed), int(ep_seed), int(variant), 0x5157])


try:
    import gymnasium as gym
    _Wrapper = gym.Wrapper
except Exception:  # noqa: BLE001
    _Wrapper = object


class VisualWrapper(_Wrapper):
    """Applies the visual mode at every reset and the image aug to every observation.

    mode: off (as built) | nominal | nominal2 | dr_train | dr_heldout.  Per-episode DR rng =
    episode_rng(vis_seed, reset seed, variant) -> reproducible and independent of the env RNG.
    image_aug: None -> on for dr_* modes, off for nominal.
    image_noise: False -> per-frame gaussian sensor noise is sampled and stored as
    params['image_aug'][cam]['noise_online'] but not applied (used for baked mp4 training data:
    noise is incompressible, x7 file size, and H.264 would distort it; add it in the dataloader).
    """

    def __init__(self, env, mode="nominal", level=1.0, vis_seed=0, variant=0, image_aug=None, image_noise=True):
        super().__init__(env)
        self.image_noise = bool(image_noise)   # False: sensor noise sampled+recorded but not applied
        assert mode in MODES, mode
        self.mode, self.level, self.vis_seed, self.variant = mode, float(level), int(vis_seed), int(variant)
        self.image_aug = mode.startswith("dr") if image_aug is None else bool(image_aug)
        self.params, self.aug, self._n = None, None, 0
        self._noise_rng = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if self.mode == "off":
            return obs, info
        ep = seed if seed is not None else 10 ** 6 + self._n
        self._n += 1
        rng = episode_rng(self.vis_seed, ep, self.variant)
        if self.mode == "nominal":
            self.params = apply_nominal(self.env)
        elif self.mode == "nominal2":
            self.params = apply_nominal2(self.env)
        else:
            self.params = apply_dr(self.env, rng, self.level, "heldout" if self.mode == "dr_heldout" else "train")
        cams = list((obs.get("sensor_data") or {}).keys())
        self.aug = sample_image_aug(rng, cams, self.level) if self.image_aug else None
        if self.aug and not self.image_noise:
            for p in self.aug.values():
                p["noise_online"], p["noise"] = p["noise"], 0.0
        self.params["image_aug"] = None if self.aug is None else {c: dict(v) for c, v in self.aug.items()}
        self._noise_rng = np.random.default_rng([self.vis_seed, int(ep), self.variant, 77])
        obs = self.env.unwrapped.get_obs(info)      # re-render the reset frame with the new look
        return self._aug_obs(obs), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return self._aug_obs(obs), r, term, trunc, info

    def _aug_obs(self, obs):
        if not self.aug or "sensor_data" not in obs:
            return obs
        import torch
        for c, p in self.aug.items():
            d = obs["sensor_data"].get(c)
            if d is None or "rgb" not in d:
                continue
            t = d["rgb"]
            a = t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)
            outs = np.stack([augment_image(a[i], p, self._noise_rng) for i in range(a.shape[0])])
            d["rgb"] = torch.from_numpy(outs).to(t.device) if hasattr(t, "device") else outs
        return obs


def wrap(env, mode="nominal", **kw):
    return VisualWrapper(env, mode=mode, **kw) if mode != "off" else env


# ------------------------------------------------------------------------- eval hook
def install(mode="nominal", level=1.0, vis_seed=0, image_aug=None, prefix="Stationery"):
    """Patch gymnasium.make so every env whose id starts with `prefix` is wrapped in
    VisualWrapper(mode).  Used by run_eval_visual.py; safe to call before importing runners."""
    import gymnasium
    if mode == "off" or getattr(gymnasium.make, "_vis_patched", False):
        return
    orig = gymnasium.make

    def make(env_id, *args, **kw):
        env = orig(env_id, *args, **kw)
        eid = env_id if isinstance(env_id, str) else getattr(env_id, "id", "")
        if eid.startswith(prefix):
            env = VisualWrapper(env, mode=mode, level=level, vis_seed=vis_seed, image_aug=image_aug)
        return env

    make._vis_patched = True
    gymnasium.make = make
    try:
        import gymnasium.envs.registration as reg
        reg.make = make
    except Exception:  # noqa: BLE001
        pass


def install_from_env():
    """STATIONERY_VISUAL=<mode> [STATIONERY_VISUAL_LEVEL, STATIONERY_VISUAL_SEED]."""
    m = os.environ.get("STATIONERY_VISUAL", "off")
    install(m, level=float(os.environ.get("STATIONERY_VISUAL_LEVEL", 1.0)),
            vis_seed=int(os.environ.get("STATIONERY_VISUAL_SEED", 0)))
