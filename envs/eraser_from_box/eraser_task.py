"""
StationeryBench `eraser_from_box` replica for the bimanual YAM robot (ManiSkill 3 / SAPIEN 3).

Env id: StationeryEraserFromBox-v1

Spec (Robocurve StationeryBench, MIT): cloth-covered book-style box (~30 x 22 x 7 cm) with a
hinged lift lid, hinge on the far edge (toward the arm bases); one block eraser in its paper
sleeve resting on the box floor near the front-left corner; lid closed at start; box at bench
centre.

Frame mapping (checked against the template + top_cam render):
  * robot bases at x = -0.65, arms face +x; left arm at +y, right arm at -y.
  * operator stands at +x looking toward -x  ->  operator's LEFT = -y (robot right arm side),
    operator's FRONT (near edge) = +x, FAR edge (toward arms) = -x.
  * box frame: origin at centre of the box footprint on the table, +x = front (operator side),
    +y = operator's right. Hinge line is the top of the back (-x) edge.

Lid physics (design choice, documented).  The release default adds a preloaded torsion spring on top of this
(see LID_SPRING_* below: K=0.15 N m/rad toward -0.5 rad, D=0.2), so the lid returns from any angle, also past
vertical, unless it is held open.  Without the spring (lid_spring=0):
  * revolute joint, limits [0, 100 deg]; 0 = closed (lower limit acts as the closed stop, the
    lid does not collide with its own base link).
  * lid ~0.26 kg (board 30x22x0.8 cm), joint damping 0.05 N m s/rad, joint friction 0.02 N m.
    Below ~85 deg gravity torque (~0.28 N m * cos) beats friction and the lid falls shut
    (damped, no slam); between ~85 and 100 deg it stays open and rests on the 100 deg stop
    ("propped past vertical"). So it can be held open with a gripper, or flipped past
    vertical where it stays open by itself.
  * box base is a free (dynamic) body ~0.6 kg, friction 0.6 with the table; it can be pushed.

Rubric stage (privileged state only, see `evaluate`):
  0 no purposeful contact with the box or lid
  1 touched the lid: robot-lid contact force > 0.5 N at any step, or lid angle > 3 deg
  2 lid held open: lid angle >= 45 deg at any step
  3 eraser lifted out of the box: eraser centre >= 1.5 cm above the tray rim (box frame) or
    eraser centre outside the box outer footprint, at any step
  4 (instantaneous) eraser resting on the table outside the box AND lid closed:
      - eraser centre outside the box outer footprint expanded by 1 cm (box frame),
      - eraser centre height < 2.0 cm above the table (lying on the table),
      - no gripper-finger contact with the eraser (> 0.1 N), eraser speed < 5 cm/s,
      - lid angle < 5 deg.
  stage = 4 if the stage-4 condition holds now, else the max sticky stage reached so far.
  `success` = stage-4 condition holds at this step (ManiSkill terminates the episode on success,
  so success == stage 4 reached at any time). `max_stage` is the running maximum.
"""

import os, sys
from typing import Any, Dict

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.sensors.camera import CameraConfig

from sim_eval.robots.bimanual_yam import BimanualYAM  # noqa: F401 (registers agent)

ENV_ID = "StationeryEraserFromBox-v1"
INSTRUCTION = (
    "Open the lid of the box, take out one eraser from the box and place it on "
    "the table, then close the box. The lid is a lift-open lid, so it may be "
    "easier to just prop it open with the gripper."
)

# ---------------- geometry (metres, box frame) ----------------
BOX_DX = 0.22          # depth (front-back, along x)
BOX_DY = 0.30          # width (left-right, along y)
COVER_T = 0.008        # bottom cover / lid board thickness
SPINE_T = 0.012        # spine (hinge side) thickness
OVERHANG_FRONT = 0.015 # lid / bottom cover overhang past the tray at the front
OVERHANG_SIDE = 0.010
TRAY_WALL = 0.010
TRAY_FLOOR = 0.004
RIM_Z = 0.062          # top of tray walls (= lid underside when closed)
LID_TOP_Z = RIM_Z + COVER_T  # 0.070 total height
HINGE_X = -BOX_DX / 2  # back edge
HINGE_Z = LID_TOP_Z    # hinge at top-back edge of the cover
FLOOR_Z = COVER_T + TRAY_FLOOR  # 0.012 inner floor top
TRAY_X0 = HINGE_X + SPINE_T            # -0.098
TRAY_X1 = BOX_DX / 2 - OVERHANG_FRONT   # 0.095
TRAY_Y = BOX_DY / 2 - OVERHANG_SIDE     # 0.140
INNER_X0, INNER_X1 = TRAY_X0 + TRAY_WALL, TRAY_X1 - TRAY_WALL  # -0.088, 0.085
INNER_Y = TRAY_Y - TRAY_WALL                                   # 0.130

LID_LIMIT = np.deg2rad(100.0)
# spring-return lid (env kwarg lid_spring, or env var EFB_LID_SPRING).  DEFAULT ON in this release (=1): every
# released result, demonstration dataset of the final model line uses it.  lid_spring=0 / EFB_LID_SPRING=0 gives the
# original gravity-only lid described in the module docstring (used only by the earliest training rounds).
# joint drive with stiffness LID_SPRING_K toward a target LID_SPRING_TARGET below the closed stop, i.e. a
# preloaded torsion spring: closing torque K*(q - target) = 0.075 N m at 0 deg ... 0.34 N m at 100 deg,
# so the lid returns from any angle (also past vertical, where gravity alone would keep it open).
LID_SPRING_K = 0.15
LID_SPRING_TARGET = -0.5
LID_SPRING_D = float(os.environ.get("EFB_LID_SPRING_D", "0.2"))   # extra viscous damping so it closes in ~1 s, no slam
LID_DAMPING = 0.05
LID_FRICTION = 0.02
LID_DENSITY = 500.0
BASE_DENSITY = 600.0

ERASER_HALF = np.array([0.0325, 0.0115, 0.006])   # 6.5 x 2.3 x 1.2 cm block eraser
ERASER_DENSITY = 1500.0
ERASER_FRICTION = 0.8
# nominal eraser placement in box frame: front-left (operator) corner = +x, -y
ERASER_NOM = (0.040, -0.080)
ERASER_NOM_YAW = np.deg2rad(45.0)

# ---------------- reset randomisation ----------------
BOX_NOM_XY = (-0.30, 0.0)
BOX_POS_NOISE = 0.02
BOX_YAW_NOISE = np.deg2rad(5.0)
ERASER_POS_NOISE = 0.01
ERASER_YAW_NOISE = np.deg2rad(10.0)

# ---------------- evaluator thresholds ----------------
TOUCH_FORCE = 0.5
TOUCH_ANGLE = np.deg2rad(3.0)
OPEN_ANGLE = np.deg2rad(45.0)
LIFT_ABOVE_RIM = 0.015
OUTSIDE_MARGIN = 0.01
ON_TABLE_MAX_Z = 0.020
RELEASE_FORCE = 0.1
STILL_SPEED = 0.05
CLOSED_ANGLE = np.deg2rad(5.0)

# appearance (render-only; matched by eye to docs/assets/eraser_from_box-setup.jpg):
# olive-grey cloth outer cover, black cloth inner lining (lid underside, floor, inner walls),
# wood-grain "page block" tray walls.  Textures can be put on the explicitly UV-mapped quads
# named *_uv (see UV_QUADS) by the visual module's task hook.
C_CLOTH = [0.36, 0.35, 0.28, 1.0]        # olive-grey cloth (cover top, spine)
C_CLOTH_DARK = [0.30, 0.29, 0.23, 1.0]   # bottom cover
C_BLACK = [0.025, 0.025, 0.03, 1.0]      # black cloth lining
C_WOOD = [0.45, 0.30, 0.17, 1.0]
_QUAD_OBJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "quad_uv.obj")
# name -> (link, description of UV layout)
UV_QUADS = {
    "lid_outer_uv": ("box_lid", "lid top face (+z_lid); u along lid x (hinge->front, 0.22 m), v along +y (0.30 m)"),
    "lid_lining_uv": ("box_lid", "lid underside (faces -z_lid); u along lid x, v along -y"),
    "floor_lining_uv": ("box_base", "tray floor; u along box x (back->front), v along +y"),
    "wall_lining_front_uv": ("box_base", "inner face of front wall (faces -x); u along -y, v along +z"),
    "wall_lining_back_uv": ("box_base", "inner face of back wall (faces +x); u along +y, v along +z"),
    "wall_lining_left_uv": ("box_base", "inner face of -y wall (faces +y); u along +x, v along +z"),
    "wall_lining_right_uv": ("box_base", "inner face of +y wall (faces -y); u along -x, v along +z"),
}


def _ensure_quad_obj():
    """unit quad in the local XY plane, x,y in [-0.5,0.5], normal +z, UV (0,0)..(1,1) with u=+x, v=+y."""
    if os.path.exists(_QUAD_OBJ):
        return _QUAD_OBJ
    os.makedirs(os.path.dirname(_QUAD_OBJ), exist_ok=True)
    tmp = _QUAD_OBJ + f".{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write("v -0.5 -0.5 0\nv 0.5 -0.5 0\nv 0.5 0.5 0\nv -0.5 0.5 0\n"
                "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\n"
                "f 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n")
    os.replace(tmp, _QUAD_OBJ)
    return _QUAD_OBJ


def _quad(link, name, center, ax_u, ax_v, size_u, size_v, color, rough=0.9):
    """render-only explicitly UV-mapped rectangle: local x -> ax_u (u), local y -> ax_v (v), normal = u x v."""
    eu, ev = np.asarray(ax_u, float), np.asarray(ax_v, float)
    R = np.stack([eu, ev, np.cross(eu, ev)], 1)
    from mani_skill.utils.structs.pose import Pose as _P  # noqa
    import transforms3d.quaternions as tq
    link.add_visual_from_file(_ensure_quad_obj(), pose=sapien.Pose(p=list(center), q=tq.mat2quat(R)),
                              scale=(size_u, size_v, 1.0), material=_mat(color, rough), name=name)


def _mat(c, rough=0.8):
    return sapien.render.RenderMaterial(base_color=c, roughness=rough, metallic=0.0)


def _box(link, p, half, color, density, phys_mat, collide=True, name=""):
    pose = sapien.Pose(p=p)
    link.add_box_visual(pose=pose, half_size=half, material=_mat(color), name=name)
    if collide:
        link.add_box_collision(pose=pose, half_size=half, material=phys_mat, density=density)


def lid_q_from_axis():
    # joint frame x-axis = -y (box frame) so +angle lifts the front edge
    return [np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)]


@register_env(ENV_ID, max_episode_steps=3600)
class StationeryEraserFromBoxEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["bimanual_yam"]
    agent: BimanualYAM

    def __init__(self, *args, robot_uids="bimanual_yam", robot_init_qpos_noise=0.0,
                 randomize=True, lid_spring=None, **kwargs):
        if lid_spring is None:
            lid_spring = int(os.environ.get("EFB_LID_SPRING", "1"))
        self.lid_spring = int(lid_spring)
        if self.lid_spring and not globals().get("_SPRING_LOGGED"):
            print(f"[eraser_task] LID_SPRING=1 K={LID_SPRING_K} target={LID_SPRING_TARGET} D={LID_SPRING_D}", file=sys.stderr, flush=True); globals()["_SPRING_LOGGED"] = True
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.randomize = randomize
        # fresh PhysX scene every reset -> an episode depends only on its seed, not on what ran before
        kwargs.setdefault("reconfiguration_freq", 1)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ------------------------------------------------------------------ cameras
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.65, 0.0, 0.5], target=[-0.4, 0.0, 0.1])
        return [CameraConfig("base_camera", pose, 640, 480, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.20, -0.45, 0.50], target=[-0.38, 0.0, 0.08])
        return CameraConfig("render_camera", pose=pose, width=1280, height=720,
                            fov=1.1, near=0.01, far=100)

    # ------------------------------------------------------------------ scene
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        pm_box = physx.PhysxMaterial(static_friction=0.6, dynamic_friction=0.5, restitution=0.0)
        pm_lid = physx.PhysxMaterial(static_friction=0.8, dynamic_friction=0.7, restitution=0.0)

        ab = self.scene.create_articulation_builder()
        base = ab.create_link_builder()
        base.set_name("box_base")
        hx, hy = BOX_DX / 2, BOX_DY / 2
        # bottom cover (cloth)
        _box(base, [0, 0, COVER_T / 2], [hx, hy, COVER_T / 2], C_CLOTH_DARK, BASE_DENSITY, pm_box)
        # spine (cloth) at back, under the lid
        _box(base, [HINGE_X + SPINE_T / 2, 0, COVER_T + (RIM_Z - COVER_T) / 2],
             [SPINE_T / 2, hy, (RIM_Z - COVER_T) / 2], C_CLOTH, BASE_DENSITY, pm_box)
        # tray floor (black lining)
        tx_c = (TRAY_X0 + TRAY_X1) / 2
        tx_h = (TRAY_X1 - TRAY_X0) / 2
        _box(base, [tx_c, 0, COVER_T + TRAY_FLOOR / 2], [tx_h, TRAY_Y, TRAY_FLOOR / 2],
             C_BLACK, BASE_DENSITY, pm_box)
        wz_c = FLOOR_Z + (RIM_Z - FLOOR_Z) / 2
        wz_h = (RIM_Z - FLOOR_Z) / 2
        # front/back walls (wood)
        for x in (TRAY_X1 - TRAY_WALL / 2, TRAY_X0 + TRAY_WALL / 2):
            _box(base, [x, 0, wz_c], [TRAY_WALL / 2, TRAY_Y, wz_h], C_WOOD, BASE_DENSITY, pm_box)
        # side walls
        for s in (1, -1):
            _box(base, [tx_c, s * (TRAY_Y - TRAY_WALL / 2), wz_c],
                 [tx_h - TRAY_WALL, TRAY_WALL / 2, wz_h], C_WOOD, BASE_DENSITY, pm_box)
        # black cloth lining (render-only UV quads, 0.5 mm proud of the collision surfaces)
        e = 0.0005
        ix0, ix1, iy = INNER_X0, INNER_X1, INNER_Y
        _quad(base, "floor_lining_uv", [(ix0 + ix1) / 2, 0, FLOOR_Z + e], [1, 0, 0], [0, 1, 0], ix1 - ix0, 2 * iy, C_BLACK)
        _quad(base, "wall_lining_front_uv", [ix1 - e, 0, wz_c], [0, -1, 0], [0, 0, 1], 2 * iy, 2 * wz_h, C_BLACK)
        _quad(base, "wall_lining_back_uv", [ix0 + e, 0, wz_c], [0, 1, 0], [0, 0, 1], 2 * iy, 2 * wz_h, C_BLACK)
        _quad(base, "wall_lining_left_uv", [(ix0 + ix1) / 2, -iy + e, wz_c], [1, 0, 0], [0, 0, 1], ix1 - ix0, 2 * wz_h, C_BLACK)
        _quad(base, "wall_lining_right_uv", [(ix0 + ix1) / 2, iy - e, wz_c], [-1, 0, 0], [0, 0, 1], ix1 - ix0, 2 * wz_h, C_BLACK)

        lid = ab.create_link_builder(base)
        lid.set_name("box_lid")
        q = lid_q_from_axis()
        lid.set_joint_name("lid_hinge")
        lid.set_joint_properties(
            "revolute", limits=[[0.0, float(LID_LIMIT)]],
            pose_in_parent=sapien.Pose(p=[HINGE_X, 0, HINGE_Z], q=q),
            pose_in_child=sapien.Pose(p=[0, 0, 0], q=q),
            friction=LID_FRICTION, damping=LID_DAMPING,
        )
        # lid frame: origin at hinge, axes = box axes when closed; board spans x in [0, 0.22]
        _box(lid, [hx, 0, -COVER_T / 2 + 0.001], [hx, hy, COVER_T / 2 - 0.001], C_CLOTH,
             LID_DENSITY, pm_lid)
        _quad(lid, "lid_outer_uv", [hx, 0, 0.0005], [1, 0, 0], [0, 1, 0], 2 * hx, 2 * hy, C_CLOTH)
        _quad(lid, "lid_lining_uv", [hx + 0.006, 0, -COVER_T + 0.002 - 0.0005], [1, 0, 0], [0, -1, 0],
              2 * hx - 0.024, 2 * hy - 0.024, C_BLACK)
        ab.initial_pose = sapien.Pose(p=[BOX_NOM_XY[0], BOX_NOM_XY[1], 0.0])
        self.box = ab.build(name="book_box", fix_root_link=False)
        self.box_base_link = self.box.links_map["box_base"]
        self.lid_link = self.box.links_map["box_lid"]
        self.lid_joint = self.box.joints_map["lid_hinge"]
        if self.lid_spring:
            self.lid_joint.set_drive_properties(LID_SPRING_K, LID_SPRING_D)
            self.lid_joint.set_drive_target(torch.tensor([LID_SPRING_TARGET], dtype=torch.float32))

        # eraser: white rubber block + blue paper sleeve (visual)
        eb = self.scene.create_actor_builder()
        pm_er = physx.PhysxMaterial(static_friction=ERASER_FRICTION,
                                    dynamic_friction=ERASER_FRICTION * 0.9, restitution=0.0)
        eb.add_box_collision(half_size=ERASER_HALF.tolist(), material=pm_er, density=ERASER_DENSITY)
        eb.add_box_visual(half_size=ERASER_HALF.tolist(), material=_mat([0.95, 0.95, 0.93, 1.0], 0.9))
        sl = ERASER_HALF + np.array([-0.009, 0.0006, 0.0006])
        eb.add_box_visual(pose=sapien.Pose(p=[-0.004, 0, 0]), half_size=sl.tolist(),
                          material=_mat([0.10, 0.25, 0.70, 1.0], 0.7))
        eb.initial_pose = sapien.Pose(p=[BOX_NOM_XY[0], BOX_NOM_XY[1], 0.2])
        self.eraser = eb.build(name="eraser")

        self._max_stage = None
        self._touched = None
        self._opened = None
        self._lifted = None

    # ------------------------------------------------------------------ reset
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.agent.robot.set_pose(sapien.Pose(p=[-0.65, 0, 0.01]))
            self.agent.robot.set_qpos(self.agent.keyframes["home"].qpos)

            r = (lambda: (torch.rand((b,)) * 2 - 1)) if self.randomize else (lambda: torch.zeros((b,)))
            bx = BOX_NOM_XY[0] + r() * BOX_POS_NOISE
            by = BOX_NOM_XY[1] + r() * BOX_POS_NOISE
            byaw = r() * BOX_YAW_NOISE
            bp = torch.stack([bx, by, torch.zeros(b)], -1)
            bq = torch.stack([torch.cos(byaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(byaw / 2)], -1)
            self.box.set_pose(Pose.create_from_pq(bp, bq))
            self.box.set_qpos(torch.zeros((b, 1)))
            self.box.set_qvel(torch.zeros((b, 1)))
            self.box.set_root_linear_velocity(torch.zeros((b, 3)))
            self.box.set_root_angular_velocity(torch.zeros((b, 3)))

            ex = ERASER_NOM[0] + r() * ERASER_POS_NOISE
            ey = ERASER_NOM[1] + r() * ERASER_POS_NOISE
            eyaw = ERASER_NOM_YAW + r() * ERASER_YAW_NOISE
            c, s = torch.cos(byaw), torch.sin(byaw)
            wx = bx + c * ex - s * ey
            wy = by + s * ex + c * ey
            wz = torch.full((b,), FLOOR_Z + ERASER_HALF[2] + 0.0005)
            yaw = byaw + eyaw
            eq = torch.stack([torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(yaw / 2)], -1)
            self.eraser.set_pose(Pose.create_from_pq(torch.stack([wx, wy, wz], -1), eq))
            self.eraser.set_linear_velocity(torch.zeros((b, 3)))
            self.eraser.set_angular_velocity(torch.zeros((b, 3)))

            if self._max_stage is None or self._max_stage.shape[0] != self.num_envs:
                z = torch.zeros(self.num_envs, dtype=torch.long)
                self._max_stage, self._touched = z.clone(), torch.zeros(self.num_envs, dtype=torch.bool)
                self._opened, self._lifted = self._touched.clone(), self._touched.clone()
            self._max_stage[env_idx] = 0
            self._touched[env_idx] = False
            self._opened[env_idx] = False
            self._lifted[env_idx] = False

    # ------------------------------------------------------------------ privileged helpers
    def lid_angle(self):
        return self.box.get_qpos()[:, 0]

    def eraser_in_box_frame(self):
        bp = self.box_base_link.pose
        return (bp.inv() * self.eraser.pose).p

    def _robot_lid_force(self):
        tot = torch.zeros(self.num_envs, device=self.device)
        for link in self.agent.robot.get_links():
            if "link_" not in link.name:
                continue
            f = self.scene.get_pairwise_contact_forces(link, self.lid_link)
            tot = tot + torch.linalg.norm(f, dim=-1)
        return tot

    def _finger_eraser_force(self):
        a = self.agent
        tot = torch.zeros(self.num_envs, device=self.device)
        for link in [a.left_finger1_link, a.left_finger2_link, a.right_finger1_link, a.right_finger2_link]:
            f = self.scene.get_pairwise_contact_forces(link, self.eraser)
            tot = tot + torch.linalg.norm(f, dim=-1)
        return tot

    def evaluate(self) -> Dict[str, Any]:
        ang = self.lid_angle()
        ep = self.eraser_in_box_frame()
        ew = self.eraser.pose.p
        table_z = 0.0
        lid_force = self._robot_lid_force()
        finger_f = self._finger_eraser_force()
        speed = torch.linalg.norm(self.eraser.linear_velocity, dim=-1)

        outside = (ep[:, 0].abs() > BOX_DX / 2) | (ep[:, 1].abs() > BOX_DY / 2)
        outside_m = (ep[:, 0].abs() > BOX_DX / 2 + OUTSIDE_MARGIN) | (ep[:, 1].abs() > BOX_DY / 2 + OUTSIDE_MARGIN)
        above_rim = ep[:, 2] > RIM_Z + LIFT_ABOVE_RIM

        if self._max_stage is None:  # evaluate before first init (shouldn't happen)
            z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._touched, self._opened, self._lifted = z.clone(), z.clone(), z.clone()
            self._max_stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._touched |= (lid_force > TOUCH_FORCE) | (ang > TOUCH_ANGLE)
        self._opened |= ang >= OPEN_ANGLE
        self._lifted |= outside | above_rim

        on_table = outside_m & ((ew[:, 2] - table_z) < ON_TABLE_MAX_Z)
        released = (finger_f < RELEASE_FORCE) & (speed < STILL_SPEED)
        closed = ang < CLOSED_ANGLE
        cond4 = on_table & released & closed

        sticky = torch.where(self._lifted, 3, torch.where(self._opened, 2, torch.where(self._touched, 1, 0)))
        stage = torch.where(cond4, torch.full_like(sticky, 4), sticky)
        self._max_stage = torch.maximum(self._max_stage, stage)
        return {
            "success": cond4,
            "stage": stage,
            "max_stage": self._max_stage.clone(),
            "lid_angle_deg": torch.rad2deg(ang),
            "eraser_box_frame": ep,
            "eraser_on_table": on_table,
            "eraser_released": released,
            "lid_closed": closed,
            "lid_contact_force": lid_force,
        }

    # ------------------------------------------------------------------ snapshots
    def get_state_dict(self):
        """sim state (+controller) plus the evaluator's sticky rubric flags, so a restored snapshot
        continues the same stage bookkeeping. task = [max_stage, touched, opened, lifted] (float)."""
        sd = super().get_state_dict()
        if self._max_stage is not None:
            sd["task"] = torch.stack([self._max_stage.float(), self._touched.float(),
                                      self._opened.float(), self._lifted.float()], -1).clone()
        return sd

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        st = {k: v for k, v in state.items() if k != "task"}
        super().set_state_dict(st, env_idx)
        if "task" in state and self._max_stage is not None:
            t = torch.as_tensor(state["task"], device=self.device).reshape(-1, 4)
            self._max_stage[:] = t[:, 0].long(); self._touched[:] = t[:, 1] > 0.5
            self._opened[:] = t[:, 2] > 0.5; self._lifted[:] = t[:, 3] > 0.5

    def _get_obs_extra(self, info: dict) -> Dict[str, Any]:
        return dict(
            box_pose=self.box_base_link.pose.raw_pose,
            lid_qpos=self.box.get_qpos(),
            eraser_pose=self.eraser.pose.raw_pose,
        )

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        return info["max_stage"].float()

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        return info["max_stage"].float() / 4.0
