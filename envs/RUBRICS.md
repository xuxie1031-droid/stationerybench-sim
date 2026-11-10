# Task specs and 0-4 stage rubrics

The setup, instruction and rubric text below is copied verbatim from StationeryBench
(https://github.com/robocurve/stationerybench, task reference https://robocurve.github.io/stationerybench),
Copyright (c) 2026 robocurve, MIT License (full notice in [NOTICE](../NOTICE)). Left/right are the operator's, who faces the
bench with the two arm bases at the far edge.

On the real robot the rubric is graded by an operator from video. In this replica every stage is computed from
privileged simulator state at every control step (30 Hz); the column "sim criterion" is our operationalisation. Stages
1-3 are latched (the reported stage is the running maximum), stage 4 = success. Each evaluator was validated with known-answer
tests that can fail (deliberately broken behaviours must not reach stage 4).

## eraser_from_box (`StationeryEraserFromBox-v1`, [envs/eraser_from_box/eraser_task.py](eraser_from_box/eraser_task.py))

Objects: cloth-covered book-style box with a hinged lift lid (about 30 x 22 x 7 cm); one block eraser in its paper sleeve.

Setup:
1. Place the box at the centre of the bench with the hinge at the far edge, toward the arms.
2. Put the eraser inside, resting on the floor of the box near the front-left corner.
3. Close the lid fully.

Instruction:
> Open the lid of the box, take out one eraser from the box and place it on the table, then close the box. The lid is a lift-open lid, so it may be easier to just prop it open with the gripper.

| Stage | StationeryBench rubric | sim criterion |
|---|---|---|
| 0 | No purposeful contact with the box or lid | none of the below |
| 1 | Touches the lid deliberately | robot-lid contact force > 0.5 N at any step, or lid angle > 3 deg |
| 2 | Lid held open far enough to reach inside | lid angle >= 45 deg at any step |
| 3 | Eraser lifted out of the box | eraser centre >= 1.5 cm above the tray rim (box frame) or outside the box footprint, at any step |
| 4 | Eraser released on the table outside the box and the lid closed | at one step: eraser centre outside the box footprint + 1 cm, < 2 cm above the table, no finger contact (> 0.1 N), eraser speed < 5 cm/s, lid angle < 5 deg |

Sim specifics: box base is a free body (~0.6 kg) that can be pushed; box pose (+-2 cm, +-5 deg) and eraser pose
(+-1 cm, +-10 deg) are randomised per seed. **Spring-return lid (release default):** a preloaded torsion spring
(K = 0.15 N m/rad toward -0.5 rad, extra damping D = 0.2) closes the lid from any angle unless it is held, so a policy
has to prop the lid while the other arm extracts the eraser. `EFB_LID_SPRING=0` restores the gravity-only lid (stays
open past ~85 deg), which only the earliest training rounds used. Episode length 3600 steps = 120 s at 30 Hz.

## ruler_handover (`StationeryRulerHandover-v1`, [envs/ruler_handover/ruler_handover_env.py](ruler_handover/ruler_handover_env.py))

Objects: two paper cups (about 12 oz); one clear plastic 12 inch (30 cm) ruler.

Setup:
1. Stand one cup near the left arm and one near the right arm, at the same distance from the arm bases, so the ruler cannot be moved without a handover.
2. Stand the ruler in the left cup, leaning toward the far edge of the bench.
3. Leave the right cup empty.

Instruction:
> Transfer the ruler from the left cup to the right cup, but you must hand it over to the other arm's gripper in mid-air.

| Stage | StationeryBench rubric | sim criterion |
|---|---|---|
| 0 | No purposeful contact with the ruler | none of the below |
| 1 | Touches or moves the ruler deliberately | any robot link touches the ruler (contact force >= 0.2 N) |
| 2 | Ruler lifted clear of the left cup in one gripper | ruler held by exactly one gripper (antipodal pad forces >= 0.5 N), clear of the source cup, not touching table/cups |
| 3 | Ruler held by the other gripper after a mid-air handover | after 2: a step where both grippers hold the ruler mid-air, then a step where only the other gripper holds it, still mid-air |
| 4 | Ruler standing in the right cup and released | after 3: ruler bottom inside the destination cup, axis < 45 deg from vertical, no table contact, cup upright (< 20 deg), no robot contact, ruler speed < 0.05 m/s, held for 10 consecutive steps |

Stages are strictly sequential, so carrying or sliding the ruler to the right cup without the mid-air two-gripper
handover cannot reach stage 4 (checked with a known-answer test). Operator-left = the
robot's right arm: the ruler starts next to the robot's `right_*` arm. Episode length 2000 steps = 66.7 s at 30 Hz
(shorter than StationeryBench's 120 s horizon).

## Differences from the real benchmark (read before comparing numbers)
* Success is decided by the privileged evaluator above, not by an operator.
* Object models are simplified (box primitives, a 28-segment paper-cup frustum, a rigid ruler); the cups are heavier
  than real paper cups (30 g) for stability with the leaning ruler.
* The robot is MolmoAct2's ManiSkill `bimanual_yam` agent (PD joint-position control at 30 Hz, sim at 150 Hz), not
  Robocurve's hardware; camera poses follow that agent's top and wrist cameras.
* The look (envs/visual) is aligned to Robocurve's setup photos (table, backdrop, objects) and, in `nominal2`, to the
  brightness and contrast of real Ai2 YAM camera frames. No Robocurve robot-camera frames were available.
