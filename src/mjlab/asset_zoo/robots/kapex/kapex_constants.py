"""KAPEX humanoid constants.

31 actuated DoF: 6 per leg (12), 3 waist, 2 head, 7 per arm (14). Toes are
fixed (welded), not actuated.
Actuators are position-controlled. Joint armatures are taken directly from the
MJCF default classes (they are already joint-space reflected inertias), and PD
gains are derived from them assuming a 10 Hz natural frequency, following the
same scheme as the Unitree G1 config.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

KAPEX_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "kapex" / "xmls" / "kapex.xml"
)
assert KAPEX_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(KAPEX_XML))


##
# Actuator config.
##


# PD gains (stiffness=kp, damping=kd) are the KAPEX RL gains provided by the
# hardware team. armature values are the joint-space reflected inertias from the
# MJCF <default class="actuator_*"> blocks; effort limits come from the per-joint
# actuatorfrcrange. Joints within a group share identical gains.
def _pos_actuator(
  target_names_expr: tuple[str, ...],
  stiffness: float,
  damping: float,
  effort_limit: float,
  armature: float,
) -> BuiltinPositionActuatorCfg:
  """Build a position actuator group with explicit PD gains."""
  return BuiltinPositionActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=stiffness,
    damping=damping,
    effort_limit=effort_limit,
    armature=armature,
  )


KAPEX_ACTUATOR_RO80 = _pos_actuator(  # hip yaw/roll, shoulder pitch/roll
  (
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
  ),
  stiffness=76.4411904,
  damping=5.09607936,
  effort_limit=100.0,
  armature=0.1556,
)
KAPEX_ACTUATOR_ANKLE_PITCH = _pos_actuator(
  (".*_ankle_pitch_joint",),
  stiffness=123.3,
  damping=8.22,
  effort_limit=100.0,
  armature=0.287359,
)
KAPEX_ACTUATOR_ANKLE_ROLL = _pos_actuator(
  (".*_ankle_roll_joint",),
  stiffness=59.4,
  damping=3.96,
  effort_limit=100.0,
  armature=0.117123,
)
KAPEX_ACTUATOR_RO100 = _pos_actuator(  # hip pitch, knee
  (".*_hip_pitch_joint", ".*_knee_joint"),
  stiffness=252.73368576,
  damping=16.848912384,
  effort_limit=300.0,
  armature=0.4512,
)
KAPEX_ACTUATOR_WAIST_ROLL = _pos_actuator(
  ("waist_roll_joint",),
  stiffness=208.09728,
  damping=13.873152,
  effort_limit=120.0,
  armature=0.2312,
)
KAPEX_ACTUATOR_WAIST_DIFF = _pos_actuator(  # waist yaw, pitch (differential drive)
  ("waist_yaw_joint", "waist_pitch_joint"),
  stiffness=416.19456,
  damping=27.746304,
  effort_limit=120.0,
  armature=0.4624,
)
KAPEX_ACTUATOR_HEAD_PITCH = _pos_actuator(
  ("head_pitch_joint",),
  stiffness=55.0,
  damping=0.65,
  effort_limit=8.3,
  armature=0.00414,
)
KAPEX_ACTUATOR_HEAD_YAW = _pos_actuator(
  ("head_yaw_joint",),
  stiffness=27.0,
  damping=0.2,
  effort_limit=3.0,
  armature=0.0008766,
)
KAPEX_ACTUATOR_SHOULDER_YAW_ELBOW = _pos_actuator(
  (".*_shoulder_yaw_joint", ".*_elbow_joint"),
  stiffness=8.01036832355424,
  damping=1.019911771773468,
  effort_limit=53.0,
  armature=0.0081162,
)
KAPEX_ACTUATOR_WRIST_YAW = _pos_actuator(
  (".*_wrist_yaw_joint",),
  stiffness=2.8424460673512497,
  damping=0.36191147368320004,
  effort_limit=30.0,
  armature=0.00288,
)
KAPEX_ACTUATOR_WRIST_RP = _pos_actuator(  # wrist roll, pitch (linear-driven)
  (".*_wrist_roll_joint", ".*_wrist_pitch_joint"),
  stiffness=5.0,
  damping=0.05,
  effort_limit=7.0,
  armature=0.00006464,
)

##
# Keyframe config.
##

# Knees-bent standing pose. Pelvis height chosen so the feet rest on the ground
# in this configuration (computed via forward kinematics). RSI overrides this
# during tracking, but it sets the action offset (default joint pose).
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.89),
  joint_pos={
    ".*_hip_pitch_joint": -0.3,
    ".*_knee_joint": 0.6,
    ".*_ankle_pitch_joint": -0.3,
    ".*_elbow_joint": -0.3,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# kapex names its collision geoms explicitly (Pelvis, Left_Foot, Chest, ...)
# rather than with a "*_collision" suffix. Feet get condim=3 with friction;
# every other body gets condim=1. Self-collisions stay enabled.
_COLLISION_GEOMS = (
  "Pelvis",
  "Chest",
  "Head",
  "Left_Upper_Leg",
  "Left_Lower_Leg",
  "Left_Foot",
  "Right_Upper_Leg",
  "Right_Lower_Leg",
  "Right_Foot",
  "Left_Upper_Arm",
  "Left_Forearm",
  "Right_Upper_Arm",
  "Right_Forearm",
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=_COLLISION_GEOMS,
  condim={r"(Left|Right)_Foot": 3, r".*": 1},
  priority={r"(Left|Right)_Foot": 1},
  friction={r"(Left|Right)_Foot": (0.6,)},
)

##
# Final config.
##

KAPEX_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    KAPEX_ACTUATOR_RO80,
    KAPEX_ACTUATOR_ANKLE_PITCH,
    KAPEX_ACTUATOR_ANKLE_ROLL,
    KAPEX_ACTUATOR_RO100,
    KAPEX_ACTUATOR_WAIST_ROLL,
    KAPEX_ACTUATOR_WAIST_DIFF,
    KAPEX_ACTUATOR_HEAD_PITCH,
    KAPEX_ACTUATOR_HEAD_YAW,
    KAPEX_ACTUATOR_SHOULDER_YAW_ELBOW,
    KAPEX_ACTUATOR_WRIST_YAW,
    KAPEX_ACTUATOR_WRIST_RP,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_kapex_robot_cfg() -> EntityCfg:
  """Get a fresh KAPEX robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=KAPEX_ARTICULATION,
  )


KAPEX_ACTION_SCALE: dict[str, float] = {}
for a in KAPEX_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  assert e is not None
  for n in a.target_names_expr:
    KAPEX_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_kapex_robot_cfg())

  viewer.launch(robot.spec.compile())
