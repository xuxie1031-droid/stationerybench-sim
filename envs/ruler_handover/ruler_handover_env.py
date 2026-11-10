"""
StationeryBench `ruler_handover` replica for the bimanual YAM (ManiSkill 3.0.1 / SAPIEN 3.0.3).

Spec (Robocurve StationeryBench, MIT): "Transfer the ruler from the left cup to the right
cup, but you must hand it over to the other arm's gripper in mid-air."

Frame mapping
-------------
Sim world: robot base (bimanual_base) at (-0.65, 0, 0.01) on the far edge of the table, arms
reach toward +x.  Robot-left arm (`left_*` joints, MolmoAct2 state dims 0..6) is at y=+0.24,
robot-right arm (`right_*`, dims 7..13) at y=-0.24.

StationeryBench left/right are the OPERATOR's, who stands at the near edge (+x) facing the
bases (-x).  Operator-left = -y = robot-RIGHT arm side.  Therefore:
  * spec "left cup"  (ruler, source)      -> cup_src at y < 0, next to robot `right_*` arm
  * spec "right cup" (empty, destination) -> cup_dst at y > 0, next to robot `left_*` arm
  * "leaning toward the far edge"          -> ruler top tilts toward -x (toward the bases)
In the top_cam image (mounted behind the arms looking +x) image-left = +y, so the ruler cup
appears on the image RIGHT; in the operator-view render camera it appears on the left, as in
the real setup photo.

Rubric evaluator (privileged, per control step; see `RubricTracker` for exact thresholds)
-------------------------------------------------------------------------------------------
  1  any robot link touches the ruler (contact force >= 0.2 N)
  2  ruler held by exactly one gripper (`arm_grasping`: antipodal pad forces >= 0.5 N), clear of cup_src, not touching
     table / cups  -> that gripper is the "first holder" A
  3  after 2: a step where BOTH grippers grasp the ruler while it is mid-air (no contact with
     table/cups, clear of both cups), followed by a step where only gripper B != A holds it
     (A's links not touching it) while still mid-air
  4  after 3: ruler's bottom end inside cup_dst volume, ruler axis < 45 deg from vertical, no
     ruler-table contact, cup_dst upright (< 20 deg), no robot link touching the ruler,
     ruler speed < 0.05 m/s -- held for 10 consecutive steps (0.33 s).
Stages are strictly sequential, so sliding/dropping/carrying the ruler to cup_dst without the
mid-air two-gripper handover cannot reach 4.  `success` = max stage 4.
"""

from typing import Any, Dict

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from sim_eval.robots.bimanual_yam import BimanualYAM  # noqa: F401 (registers agent)

ENV_ID = "StationeryRulerHandover-v1"
INSTRUCTION = (
    "Transfer the ruler from the left cup to the right cup, but you must hand "
    "it over to the other arm's gripper in mid-air."
)

# ---------------------------------------------------------------- geometry constants
CUP_R_TOP = 0.045      # 12 oz paper cup: ~9 cm top diameter
CUP_R_BOT = 0.030      # ~6 cm bottom diameter
CUP_H = 0.115          # ~11.5 cm tall
CUP_WALL = 0.003
CUP_SEGMENTS = 28
CUP_MASS = 0.030       # real paper cup ~12 g; heavier here (see README) for stability with the leaning ruler
RULER_L, RULER_W, RULER_T = 0.30, 0.032, 0.002
RULER_DENSITY = 1190.0  # acrylic -> ~23 g

ROBOT_BASE = (-0.65, 0.0, 0.01)
ARM_BASE_Y = 0.24      # left_* arm at +0.24, right_* at -0.24 (MJCF)


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _qz(a):
    return np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)])


def _qy(a):
    return np.array([np.cos(a / 2), 0, np.sin(a / 2), 0])


def _mat2quat(R):
    import transforms3d
    return transforms3d.quaternions.mat2quat(R)


def build_paper_cup(scene, name, color=(0.93, 0.92, 0.88, 1.0)):
    """Thin-walled open frustum: CUP_SEGMENTS slanted wall boxes + a floor disc."""
    b = scene.create_actor_builder()
    mat_out = sapien.render.RenderMaterial(base_color=list(color), roughness=0.8)
    mat_in = sapien.render.RenderMaterial(base_color=[0.97, 0.96, 0.93, 1.0], roughness=0.9)
    phi = np.arctan2(CUP_R_TOP - CUP_R_BOT, CUP_H)
    slant = CUP_H / np.cos(phi)
    r_mid = 0.5 * (CUP_R_TOP + CUP_R_BOT) + CUP_WALL / 2
    half_tan = (r_mid * np.tan(np.pi / CUP_SEGMENTS)) * 1.15
    # wall area -> density giving CUP_MASS total (floor counted too)
    wall_vol = CUP_SEGMENTS * (2 * half_tan) * CUP_WALL * slant
    floor_t = 0.004
    floor_vol = np.pi * CUP_R_BOT ** 2 * floor_t
    density = CUP_MASS / (wall_vol + floor_vol)
    for k in range(CUP_SEGMENTS):
        a = 2 * np.pi * k / CUP_SEGMENTS
        q = _quat_mul(_qz(a), _qy(phi))
        p = [r_mid * np.cos(a), r_mid * np.sin(a), CUP_H / 2]
        pose = sapien.Pose(p=p, q=q)
        hs = [CUP_WALL / 2, half_tan, slant / 2]
        b.add_box_collision(pose=pose, half_size=hs, density=density)
        b.add_box_visual(pose=pose, half_size=hs, material=mat_out)
    # top rim lip (visual only) so the opening reads as a cup edge
    for k in range(CUP_SEGMENTS):
        a = 2 * np.pi * (k + 0.5) / CUP_SEGMENTS
        rr = CUP_R_TOP + CUP_WALL
        b.add_box_visual(pose=sapien.Pose(p=[rr * np.cos(a), rr * np.sin(a), CUP_H - 0.002], q=_qz(a)),
                         half_size=[0.0025, rr * np.tan(np.pi / CUP_SEGMENTS) * 1.1, 0.003], material=mat_in)
    # floor disc (cylinder axis is local x -> rotate to z)
    fpose = sapien.Pose(p=[0, 0, floor_t / 2], q=_qy(np.pi / 2))
    b.add_cylinder_collision(pose=fpose, radius=CUP_R_BOT + CUP_WALL, half_length=floor_t / 2, density=density)
    b.add_cylinder_visual(pose=fpose, radius=CUP_R_BOT + CUP_WALL, half_length=floor_t / 2, material=mat_in)
    b.initial_pose = sapien.Pose(p=[0, 0, 0.5])
    return b.build(name=name)


def build_ruler(scene, name="ruler"):
    """Clear plastic 30 cm ruler; local x = long axis, y = width, z = thickness.
    Rendered semi-transparent (alpha 0.3 light-blue tint; physically-based transmission renders
    near-black in rt-fast, so it is not used) plus opaque printed markings like the real one."""
    b = scene.create_actor_builder()
    hs = [RULER_L / 2, RULER_W / 2, RULER_T / 2]
    b.add_box_collision(half_size=hs, density=RULER_DENSITY)
    # rt-fast renders transmission>0 as near-black here; alpha-blended light tint reads as clear plastic
    clear = sapien.render.RenderMaterial(base_color=[0.90, 0.95, 1.0, 0.30], roughness=0.2, metallic=0.0)
    b.add_box_visual(half_size=hs, material=clear)
    ink = sapien.render.RenderMaterial(base_color=[0.08, 0.08, 0.08, 1.0], roughness=0.6)
    white = sapien.render.RenderMaterial(base_color=[0.95, 0.95, 0.95, 1.0], roughness=0.6)
    eps = RULER_T / 2 + 0.0002
    for i in range(31):  # cm ticks along one long edge, both faces
        x = -0.15 + 0.01 * i
        ln = 0.008 if i % 5 == 0 else 0.004
        for s in (+1, -1):
            b.add_box_visual(pose=sapien.Pose(p=[x, RULER_W / 2 - ln / 2, s * eps]),
                             half_size=[0.0004, ln / 2, 0.0001], material=ink)
    for s in (+1, -1):  # printed label band near the top end (as in the photo)
        b.add_box_visual(pose=sapien.Pose(p=[0.095, -0.004, s * eps]),
                         half_size=[0.035, 0.008, 0.0001], material=white)
    b.initial_pose = sapien.Pose(p=[0, 0, 0.6])
    return b.build(name=name)


def ruler_axes(q):
    """Return (u_long, v_width, n_thick) world axes for ruler quaternion q (wxyz)."""
    import transforms3d
    R = transforms3d.quaternions.quat2mat(q)
    return R[:, 0], R[:, 1], R[:, 2]


@register_env(ENV_ID, max_episode_steps=1500)
class StationeryRulerHandoverEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["bimanual_yam"]
    agent: BimanualYAM

    # Nominal cup centres (world).  cup_src = spec-left (operator's left) = sim -y,
    # cup_dst = spec-right = sim +y.  Symmetric about y=0 -> equal distance from each arm base.
    cup_x = -0.30                 # 0.35 m in front of the arm bases
    cup_y = 0.32                  # 8 cm outboard of each arm base (arms at +-0.24)
    cup_noise = 0.015
    lean_deg = (20.0, 28.0)       # ruler tilt from vertical, toward -x (far edge)
    yaw_noise_deg = 8.0
    settle_steps = 90             # physics substeps (0.6 s) run in reset() so obs starts at rest
    ruler_friction = 0.5
    cup_friction = 0.6

    def __init__(self, *args, robot_uids="bimanual_yam", robot_init_qpos_noise=0.02,
                 reconfiguration_freq=1, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, reconfiguration_freq=reconfiguration_freq, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.65, 0.0, 0.5], target=[-0.4, 0.0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # operator viewpoint: near (+x) edge looking at the bases, like the setup photo
        pose = sapien_utils.look_at(eye=[0.35, 0.0, 0.55], target=[-0.40, 0.0, 0.0])
        return CameraConfig("render_camera", pose=pose, width=960, height=540,
                            fov=1.0, near=0.01, far=100, shader_pack="rt-fast")

    # ------------------------------------------------------------------ scene
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self.cup_src = build_paper_cup(self.scene, "cup_src")
        self.cup_dst = build_paper_cup(self.scene, "cup_dst")
        self.ruler = build_ruler(self.scene)
        for act, fr in ((self.ruler, self.ruler_friction), (self.cup_src, self.cup_friction),
                        (self.cup_dst, self.cup_friction)):
            for body in act._bodies:
                for cs in body.get_collision_shapes():
                    cs.physical_material = physx.PhysxMaterial(static_friction=fr, dynamic_friction=fr,
                                                               restitution=0.0)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        assert self.num_envs == 1, "this task's evaluator is CPU / single-env"
        rng = self._episode_rng if hasattr(self, "_episode_rng") else np.random
        self.table_scene.initialize(env_idx)
        self.agent.robot.set_pose(sapien.Pose(p=list(ROBOT_BASE)))
        self.agent.robot.set_qpos(self.agent.keyframes["home"].qpos)

        def u(lo, hi):
            return float(rng.uniform(lo, hi))

        n = self.cup_noise
        src = np.array([self.cup_x + u(-n, n), -self.cup_y + u(-n, n)])
        dst = np.array([self.cup_x + u(-n, n), self.cup_y + u(-n, n)])
        self.cup_src.set_pose(sapien.Pose(p=[src[0], src[1], 0.0005], q=_qz(u(-np.pi, np.pi))))
        self.cup_dst.set_pose(sapien.Pose(p=[dst[0], dst[1], 0.0005], q=_qz(u(-np.pi, np.pi))))

        # ruler: bottom end inside cup_src, lower face resting on the far (-x) rim
        th = np.deg2rad(u(*self.lean_deg))
        yaw = np.deg2rad(u(-self.yaw_noise_deg, self.yaw_noise_deg))
        d = np.array([-np.sin(th), 0.0, np.cos(th)])          # bottom -> top (lean toward -x)
        w = np.array([0.0, 1.0, 0.0])
        nrm = np.cross(d, w)                                   # = (-cos, 0, -sin): lower face normal
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        d, w, nrm = Rz @ d, Rz @ w, Rz @ nrm
        rim = np.array([src[0], src[1], CUP_H]) + Rz @ np.array([-(CUP_R_TOP - 0.002), 0, 0])
        # line through rim contact, offset so the lower face touches the rim
        c_line = rim - nrm * (RULER_T / 2 + 0.0005)
        floor_z = 0.004 + 0.0015
        # point on the axis whose lowest corner sits at floor_z
        z_lowest_off = RULER_W / 2 * abs(w[2]) + RULER_T / 2 * abs(nrm[2])
        s_bot = (floor_z + z_lowest_off - c_line[2]) / d[2]
        bottom = c_line + s_bot * d
        center = bottom + d * RULER_L / 2
        R = np.stack([d, w, nrm], axis=1)
        self.ruler.set_pose(sapien.Pose(p=center, q=_mat2quat(R)))
        for a in (self.ruler, self.cup_src, self.cup_dst):
            a.set_linear_velocity(torch.zeros(1, 3))
            a.set_angular_velocity(torch.zeros(1, 3))
        self._tracker = None

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # let ruler/cups settle under gravity with the arms holding home (drive targets set)
        for _ in range(self.settle_steps):
            self.scene.step()
        self._tracker = RubricTracker(self)
        info = self.get_info()
        obs = self.get_obs(info)
        return obs, info

    # ------------------------------------------------------------------ snapshots
    def task_state(self):
        """JSON-able task state NOT covered by get_state_dict(): rubric history + elapsed steps.
        Save it next to get_state_dict() in every snapshot."""
        tr = self._tracker
        return dict(elapsed_steps=int(self._elapsed_steps[0]) if torch.is_tensor(self._elapsed_steps) else int(self._elapsed_steps),
                    rubric=None if tr is None else tr.export())

    def load_task_state(self, ts):
        if ts is None:
            return
        if ts.get("elapsed_steps") is not None:
            self._elapsed_steps[:] = int(ts["elapsed_steps"])
        if ts.get("rubric") is not None and self._tracker is not None:
            self._tracker.load(ts["rubric"])

    # ------------------------------------------------------------------ evaluation
    def evaluate(self) -> Dict[str, Any]:
        tr = getattr(self, "_tracker", None)
        if tr is None:
            z = torch.zeros(1, dtype=torch.bool, device=self.device)
            zi = torch.zeros(1, dtype=torch.int32, device=self.device)
            return dict(success=z, stage=zi, max_stage=zi)
        tr.update(int(self._elapsed_steps[0]) if torch.is_tensor(self._elapsed_steps) else int(self._elapsed_steps))
        dev = self.device
        return dict(
            success=torch.tensor([tr.max_stage >= 4], device=dev),
            stage=torch.tensor([tr.cur_stage], device=dev),
            max_stage=torch.tensor([tr.max_stage], device=dev),
        )

    def _get_obs_extra(self, info: Dict) -> Dict[str, Any]:
        return dict(ruler_pose=self.ruler.pose.raw_pose, cup_src_pose=self.cup_src.pose.raw_pose,
                    cup_dst_pose=self.cup_dst.pose.raw_pose)

    def compute_dense_reward(self, obs, action, info):
        return info["max_stage"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return info["max_stage"].float() / 4.0


def arm_grasping(env, arm, min_force=0.5, min_opp_deg=120.0):
    """Both finger pads of `arm` push on the ruler (>= min_force N each) from opposite sides.
    (BimanualYAM.is_grasping's finger-axis check does not match this MJCF's finger frames: it
    returns False even for a 20 N antipodal pinch, so an antipodal-force test is used instead.)"""
    f1 = getattr(env.agent, f"{arm}_finger1_link")
    f2 = getattr(env.agent, f"{arm}_finger2_link")
    a = env.scene.get_pairwise_contact_forces(f1, env.ruler)[0].cpu().numpy()
    b = env.scene.get_pairwise_contact_forces(f2, env.ruler)[0].cpu().numpy()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < min_force or nb < min_force:
        return False
    cosang = float(a @ b / (na * nb))
    return cosang <= np.cos(np.deg2rad(min_opp_deg))


# ====================================================================== evaluator
ARM_SRC, ARM_DST = "right", "left"   # robot arm next to cup_src / cup_dst (see frame mapping)


class RubricTracker:
    """Privileged 0-4 rubric state machine.  Thresholds (all documented here):"""
    TOUCH_N = 0.2            # stage 1: any robot link <-> ruler contact force
    CLEAR_MARGIN = 0.01      # ruler sample points must be outside cup cylinder (r_top+1cm) or >1cm above rim
    GRASP_MIN_N = 0.5        # arm_grasping: min force per finger pad, pad forces >=120 deg apart
    RELEASE_N = 0.05         # "released": no robot-ruler contact above this
    PLACE_TILT_DEG = 45.0    # ruler axis vs vertical when standing in the cup
    CUP_UPRIGHT_DEG = 20.0
    SETTLE_V = 0.05          # m/s ruler linear speed
    HOLD_STEPS = 10          # stage-4 predicate must persist (0.33 s @30 Hz)
    N_SAMPLES = 31

    def __init__(self, env: StationeryRulerHandoverEnv):
        self.env = env
        self.agent = env.agent
        # map physx entity names -> short labels
        self.name_map = {}
        for l in env.agent.robot.get_links():
            self.name_map[l._objs[0].entity.name] = l.name
        self.name_map[env.ruler._objs[0].name] = "ruler"
        self.name_map[env.cup_src._objs[0].name] = "cup_src"
        self.name_map[env.cup_dst._objs[0].name] = "cup_dst"
        self.name_map[env.table_scene.table._objs[0].name] = "table"
        self.name_map[env.table_scene.ground._objs[0].name] = "ground"
        self.robot_link_names = {l.name for l in env.agent.robot.get_links()}
        self.grip_links = {
            arm: {f"{arm}_link_6", f"{arm}_link_left_finger", f"{arm}_link_right_finger"}
            for arm in ("left", "right")
        }
        self.max_stage = 0
        self.cur_stage = 0
        self.first_holder = None
        self.both_midair_seen = False
        self.handover_done = False
        self.final_holder = None
        self.place_count = 0
        self.last_step = -1
        self.events = []
        self.init_ruler_p = env.ruler.pose.p[0].cpu().numpy().copy()
        self.log = []

    # ---------------- geometry helpers
    def _ruler_points(self):
        p = self.env.ruler.pose.p[0].cpu().numpy()
        q = self.env.ruler.pose.q[0].cpu().numpy()
        u, v, n = ruler_axes(q)
        s = np.linspace(-RULER_L / 2, RULER_L / 2, self.N_SAMPLES)
        pts = []
        for sw in (-RULER_W / 2, RULER_W / 2):
            pts.append(p[None] + s[:, None] * u[None] + sw * v[None])
        return p, u, np.concatenate(pts, 0)

    @staticmethod
    def _cup_frame(cup):
        p = cup.pose.p[0].cpu().numpy()
        q = cup.pose.q[0].cpu().numpy()
        import transforms3d
        R = transforms3d.quaternions.quat2mat(q)
        return p, R

    def _clear_of(self, cup, pts):
        c, R = self._cup_frame(cup)
        loc = (pts - c[None]) @ R  # cup local coords
        r = np.linalg.norm(loc[:, :2], axis=1)
        inside_cyl = (r < CUP_R_TOP + self.CLEAR_MARGIN) & (loc[:, 2] < CUP_H + self.CLEAR_MARGIN)
        return not inside_cyl.any()

    def _bottom_in_cup(self, cup, p, u):
        ends = [p - u * RULER_L / 2, p + u * RULER_L / 2]
        bottom = min(ends, key=lambda e: e[2])
        c, R = self._cup_frame(cup)
        loc = R.T @ (bottom - c)
        r_at = CUP_R_BOT + (CUP_R_TOP - CUP_R_BOT) * np.clip(loc[2] / CUP_H, 0, 1)
        return (np.linalg.norm(loc[:2]) < r_at + 0.003) and (-0.005 < loc[2] < CUP_H)

    def _contacts(self):
        """Aggregate |force| between the ruler and each other entity (last physx substep)."""
        dt = self.env.scene.px.timestep
        out = {}
        for c in self.env.scene.get_contacts():
            n0 = self.name_map.get(c.bodies[0].entity.name, c.bodies[0].entity.name)
            n1 = self.name_map.get(c.bodies[1].entity.name, c.bodies[1].entity.name)
            if n0 == "ruler":
                other = n1
            elif n1 == "ruler":
                other = n0
            else:
                continue
            imp = np.sum([pt.impulse for pt in c.points], axis=0) if len(c.points) else np.zeros(3)
            out[other] = out.get(other, 0.0) + float(np.linalg.norm(imp)) / dt
        return out

    def _cup_tilt_deg(self, cup):
        _, R = self._cup_frame(cup)
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))

    # ---------------- main update
    def update(self, step):
        if step == self.last_step:
            return
        self.last_step = step
        env = self.env
        con = self._contacts()
        robot_f = sum(f for k, f in con.items() if k in self.robot_link_names)
        arm_f = {arm: sum(f for k, f in con.items() if k in self.grip_links[arm] or
                          (k in self.robot_link_names and k.startswith(arm + "_")))
                 for arm in ("left", "right")}
        grasp = {arm: arm_grasping(env, arm, min_force=self.GRASP_MIN_N) for arm in ("left", "right")}
        env_f = sum(f for k, f in con.items() if k not in self.robot_link_names)
        touching_env = env_f > 0.05
        p, u, pts = self._ruler_points()
        clear_src = self._clear_of(env.cup_src, pts)
        clear_dst = self._clear_of(env.cup_dst, pts)
        midair = (not touching_env) and clear_src and clear_dst
        n_grasp = int(grasp["left"]) + int(grasp["right"])

        # stage 1
        if robot_f >= self.TOUCH_N and self.max_stage < 1:
            self._advance(1, step, "touch")
        # stage 2
        if self.max_stage >= 1 and self.first_holder is None and n_grasp == 1 and clear_src and not touching_env:
            self.first_holder = "left" if grasp["left"] else "right"
            self._advance(2, step, f"lifted by {self.first_holder}")
        # stage 3
        if self.first_holder is not None and not self.handover_done:
            if n_grasp == 2 and midair:
                self.both_midair_seen = True
            other = "left" if self.first_holder == "right" else "right"
            if (self.both_midair_seen and grasp[other] and not grasp[self.first_holder]
                    and arm_f[self.first_holder] < self.RELEASE_N and midair):
                self.handover_done = True
                self.final_holder = other
                self._advance(3, step, f"handover {self.first_holder}->{other}")
        # stage 4
        tilt = float(np.degrees(np.arccos(np.clip(abs(u[2]), 0, 1))))
        v = float(np.linalg.norm(env.ruler.linear_velocity[0].cpu().numpy()))
        place_ok = (self._bottom_in_cup(env.cup_dst, p, u) and tilt < self.PLACE_TILT_DEG
                    and con.get("table", 0.0) < 0.05 and con.get("ground", 0.0) < 0.05
                    and self._cup_tilt_deg(env.cup_dst) < self.CUP_UPRIGHT_DEG
                    and robot_f < self.RELEASE_N and v < self.SETTLE_V)
        self.place_count = self.place_count + 1 if place_ok else 0
        if self.handover_done and self.place_count >= self.HOLD_STEPS:
            self._advance(4, step, "placed in cup_dst")

        # instantaneous ("final") stage
        disp = np.linalg.norm(p - self.init_ruler_p)
        cur = 0
        if self.max_stage >= 1 and (robot_f >= self.TOUCH_N or disp > 0.01):
            cur = 1
        if self.max_stage >= 2 and n_grasp >= 1 and clear_src and not touching_env:
            cur = 2
        if self.max_stage >= 3 and grasp[self.final_holder] and not grasp[self.first_holder] and midair:
            cur = 3
        if self.max_stage >= 4 and place_ok:
            cur = 4
        self.cur_stage = cur
        self.log.append(dict(step=step, robot_f=round(robot_f, 3), gL=grasp["left"], gR=grasp["right"],
                             clear_src=clear_src, clear_dst=clear_dst, env_f=round(env_f, 3),
                             tilt=round(tilt, 1), place=place_ok, stage=cur))

    def _advance(self, k, step, why):
        if k > self.max_stage:
            self.max_stage = k
            self.events.append((k, step, why))

    _EXPORT = ("max_stage", "cur_stage", "first_holder", "both_midair_seen", "handover_done", "final_holder",
               "place_count", "last_step")

    def export(self):
        d = {k: getattr(self, k) for k in self._EXPORT}
        d["events"] = [list(e) for e in self.events]
        d["init_ruler_p"] = [float(x) for x in self.init_ruler_p]
        return d

    def load(self, d):
        for k in self._EXPORT:
            setattr(self, k, d[k])
        self.events = [tuple(e) for e in d["events"]]
        self.init_ruler_p = np.array(d["init_ruler_p"])

    def summary(self):
        return dict(max_stage=self.max_stage, final_stage=self.cur_stage, first_holder=self.first_holder,
                    final_holder=self.final_holder, events=self.events)
