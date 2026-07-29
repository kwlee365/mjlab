#!/usr/bin/env python3
"""Side-by-side DEPLOY comparison of retargeting methods for one motion.

Given several retargeting methods -- each with its own W&B registry of retargeted
motions AND its own trained cluster policies under
``logs/rsl_rl/<registry>_clusters/`` -- this deploys, for a single source motion,
every method's policy (each tracking THAT method's own retargeting, so every
policy runs in-distribution), records the resulting robot trajectories, and
renders them all in ONE MuJoCo scene, one robot per method laid out in a row:

  * each robot's mesh is tinted a distinct per-method color, and
  * the method name floats above each robot's head.

Because a method may store its motion left-right mirrored relative to the others
(e.g. OmniRetarget vs COLMO/GMR), the RECORDED trajectory of the ``--mirror``
methods is reflected before rendering -- the policy still deploys on its own
un-mirrored data (a mirror is an isometry, so tracking quality is preserved), we
only mirror the picture so every robot moves the same way.

Headless-friendly: with ``--record-video`` it renders offscreen to an mp4 (works
over SSH); otherwise it opens the native MuJoCo viewer (needs a display).

    uv run python compare_retargeting.py --robot g1 \
        --registries colmo_g1 gmr_g1 omniretarget_g1 --motion run1_subject2 \
        --record-video --video-path videos/run1.mp4

    # local display, live viewer
    uv run python compare_retargeting.py --robot g1 \
        --registries colmo_g1 gmr_g1 omniretarget_g1 --motion walk2_subject3
"""

from __future__ import annotations

import argparse
import os
import types
from dataclasses import asdict
from pathlib import Path
from typing import cast

import mujoco as mj
import numpy as np
import torch
from eval_clusters import (
  ROBOT_TASKS,
  find_cluster_checkpoint,
  read_clusters,
  resolve_org,
)

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.torch import configure_torch_backends

# Per-robot MJCF used to compose the render scene (same model the env deploys).
_ROBOTS_DIR = G1_XML.parent.parent.parent
ROBOT_XML = {
  "g1": G1_XML,
  "kapex": _ROBOTS_DIR / "kapex" / "xmls" / "kapex.xml",
}

# Distinct, high-contrast colors handed out to methods in order.
DEFAULT_COLORS = [
  (0.20, 0.80, 0.35, 1.0),  # green
  (0.25, 0.50, 1.00, 1.0),  # blue
  (1.00, 0.55, 0.10, 1.0),  # orange
  (0.75, 0.35, 0.90, 1.0),  # purple
  (0.95, 0.30, 0.35, 1.0),  # red
  (0.30, 0.80, 0.85, 1.0),  # cyan
]


# --------------------------------------------------------------------------- #
# Phase 1: deploy the policies in mjlab and record the robot trajectories.
# --------------------------------------------------------------------------- #
def build_compare_env(task_id, motion_files, device):
  """One env per method, each carrying that method's retargeted npz as its own
  clip of a multi-clip MotionCommand. The training failure terminations are kept
  (episode length is otherwise infinite) so record_deploy can detect when a
  policy fails and freeze that robot's playback there."""
  env_cfg = load_env_cfg(task_id, play=False)
  agent_cfg = load_rl_cfg(task_id)

  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise SystemExit(f"Task {task_id} is not a tracking task.")

  motion_cmd.motion_file = ""
  motion_cmd.motion_files = tuple(motion_files)
  motion_cmd.sampling_mode = "start"  # everyone starts at frame 0
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.joint_position_range = (0.0, 0.0)

  # Infinite episode length: the only resets are failure terminations (detected
  # in record_deploy) and motion wrap; no periodic time-out.
  env_cfg.episode_length_s = 1e9
  env_cfg.events.pop("push_robot", None)
  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.scene.num_envs = len(motion_files)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  command = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))
  return env, agent_cfg, command


def pin_clips_to_envs(command: MotionCommand) -> None:
  """Pin env i to clip i (method i), replacing the random clip choice."""

  def _pinned_resample(self: MotionCommand, env_ids: torch.Tensor) -> None:
    self.clip_ids[env_ids] = env_ids.to(self.clip_ids.dtype)
    self.time_steps[env_ids] = self.motion.clip_starts[self.clip_ids[env_ids]]
    root_pos = self.body_pos_w[env_ids, 0].clone()
    root_ori = self.body_quat_w[env_ids, 0].clone()
    root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()
    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]
    self._write_reference_state_to_sim(
      env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel
    )

  command._resample_command = types.MethodType(_pinned_resample, command)  # type: ignore[method-assign]


class MultiPolicy:
  """action[i] = policy_i(obs[i]): each env driven by its own method's policy."""

  def __init__(self, policies: list) -> None:
    self.policies = policies

  def __call__(self, obs):
    acts = [self.policies[i](obs[i : i + 1]) for i in range(len(self.policies))]
    return torch.cat(acts, dim=0)


def record_deploy(env, policy, n_methods, num_frames, device):
  """Deploy for ``num_frames`` control steps, recording each env's root pose
  (in its own env-local frame) and joint positions.

  When a policy triggers a failure termination, that robot is FROZEN: its
  recorded pose is held at the last frame before the failure for the rest of the
  clip (the env auto-resets on termination, so its post-step pose would be the
  teleported reset pose -- we ignore that and repeat the last good frame)."""
  robot = env.unwrapped.scene["robot"]
  origins = env.unwrapped.scene.env_origins  # [n, 3]
  recs = [{"root_pos": [], "root_rot": [], "dof_pos": []} for _ in range(n_methods)]
  frozen = [False] * n_methods
  frozen_at = [None] * n_methods

  def read():
    root_pos = (robot.data.root_link_pos_w - origins).cpu().numpy()  # env-local
    root_rot = robot.data.root_link_quat_w.cpu().numpy()  # wxyz
    dof_pos = robot.data.joint_pos.cpu().numpy()
    return root_pos, root_rot, dof_pos

  def append(i, root_pos, root_rot, dof_pos):
    recs[i]["root_pos"].append(root_pos)
    recs[i]["root_rot"].append(root_rot)
    recs[i]["dof_pos"].append(dof_pos)

  env.reset()
  obs = env.get_observations()
  # Frame 0: the reference start pose (all robots standing).
  root_pos, root_rot, dof_pos = read()
  for i in range(n_methods):
    append(i, root_pos[i], root_rot[i], dof_pos[i])

  for f in range(1, num_frames):
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env.step(actions)
    terminated = env.unwrapped.termination_manager.terminated.cpu().numpy()
    dones = dones.bool().cpu().numpy()
    root_pos, root_rot, dof_pos = read()
    for i in range(n_methods):
      if frozen[i]:
        append(
          i, recs[i]["root_pos"][-1], recs[i]["root_rot"][-1], recs[i]["dof_pos"][-1]
        )
      elif terminated[i] or dones[i]:
        # Failed this step -> freeze at the last good (pre-reset) frame.
        frozen[i] = True
        frozen_at[i] = f
        append(
          i, recs[i]["root_pos"][-1], recs[i]["root_rot"][-1], recs[i]["dof_pos"][-1]
        )
      else:
        append(i, root_pos[i], root_rot[i], dof_pos[i])

  for r in recs:
    for k in r:
      r[k] = np.asarray(r[k], dtype=np.float64)
  return recs, frozen_at


# --------------------------------------------------------------------------- #
# Phase 2: mirror + compose one MuJoCo scene of all robots and render it.
# --------------------------------------------------------------------------- #
def build_mirror_spec(xml_path):
  """(perm, sign) over the robot's actuated joints so a left-right mirror is
  ``mirrored_dof = dof[:, perm] * sign``: swap left/right partners and negate the
  roll/yaw joints (odd under a sagittal reflection); pitch keeps its sign."""
  m = mj.MjModel.from_xml_path(str(xml_path))
  names = [
    mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j)
    for j in range(m.njnt)
    if m.jnt_type[j] != mj.mjtJoint.mjJNT_FREE
  ]
  idx = {n: i for i, n in enumerate(names)}
  perm = np.arange(len(names))
  sign = np.ones(len(names))
  for i, n in enumerate(names):
    if n.startswith("left_"):
      partner = "right_" + n[len("left_") :]
    elif n.startswith("right_"):
      partner = "left_" + n[len("right_") :]
    else:
      partner = n
    perm[i] = idx.get(partner, i)
    sign[i] = -1.0 if ("roll" in n or "yaw" in n) else 1.0
  return perm, sign


def mirror_trajectory(rec, perm, sign):
  """Reflect a recorded trajectory across the world x-z plane (y -> -y)."""
  rec["root_pos"][:, 1] *= -1.0
  rec["root_rot"][:, 1] *= -1.0  # wxyz: negate x and z, keep w and y.
  rec["root_rot"][:, 3] *= -1.0
  rec["dof_pos"] = rec["dof_pos"][:, perm] * sign[None, :]


def build_scene_model(xml_path, n):
  """Compose one MuJoCo model with ``n`` name-prefixed copies of the robot, plus
  a ground plane (the robot XML has none; it carries its own trackcom light, so
  N attached robots already light the scene)."""
  parent = mj.MjSpec()
  parent.worldbody.add_geom(
    name="_floor",
    type=mj.mjtGeom.mjGEOM_PLANE,
    size=[0.0, 0.0, 0.05],
    pos=[0.0, 0.0, 0.0],
    rgba=[0.28, 0.30, 0.34, 1.0],
  )
  for i in range(n):
    child = mj.MjSpec.from_file(str(xml_path))
    frame = parent.worldbody.add_frame(pos=[0.0, 0.0, 0.0])
    parent.attach(child, prefix=f"r{i}_", frame=frame)
  return parent.compile()


def tint_robot(model, prefix, rgb):
  """Recolor a prefixed robot's visible geoms (keeps original alpha)."""
  rgb = np.asarray(rgb[:3], dtype=np.float32)
  for gid in range(model.ngeom):
    if model.geom_type[gid] == mj.mjtGeom.mjGEOM_PLANE:
      continue
    bid = model.geom_bodyid[gid]
    bname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or ""
    if bname.startswith(prefix):
      model.geom_rgba[gid, :3] = rgb


def null_geom_names(model):
  """Point every geom name at a null byte so the viewer draws no geom labels
  (only our explicit head labels on user_scn geoms should show)."""
  names_arr = np.frombuffer(model.names, dtype=np.uint8)
  null_pos = int(np.where(names_arr == 0)[0][0]) if (names_arr == 0).any() else 0
  for gid in range(model.ngeom):
    model.name_geomadr[gid] = null_pos


def add_label_sphere(scene, pos, rgba, label):
  """Add a small labeled marker sphere to a viewer/render scene."""
  if scene.ngeom >= scene.maxgeom:
    return
  geom = scene.geoms[scene.ngeom]
  mj.mjv_initGeom(
    geom,
    type=mj.mjtGeom.mjGEOM_SPHERE,
    size=np.array([0.05, 0.0, 0.0]),
    pos=np.asarray(pos, dtype=np.float64),
    mat=np.eye(3).flatten(),
    rgba=np.asarray(rgba, dtype=np.float32),
  )
  geom.label = label
  scene.ngeom += 1


def visualize(recs, labels, colors, robot, args):
  """Build the combined scene from the recorded trajectories and render it."""
  n = len(recs)
  xml_path = ROBOT_XML[robot]
  model = build_scene_model(xml_path, n)
  data = mj.MjData(model)

  axis_idx = 0 if args.axis == "x" else 1

  # Per-robot: qpos block, base/head body ids, lane offset, tint.
  slots = []
  for i, rec in enumerate(recs):
    prefix = f"r{i}_"
    free_jid = next(
      jid
      for jid in range(model.njnt)
      if model.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE
      and (mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid) or "").startswith(prefix)
    )
    qadr = int(model.jnt_qposadr[free_jid])
    ndof = rec["dof_pos"].shape[1]
    base_bid = int(model.jnt_bodyid[free_jid])
    head_bid = base_bid
    for cand in ("head_link", "torso_link", "pelvis"):
      try:
        head_bid = model.body(prefix + cand).id
        break
      except KeyError:
        continue
    offset = np.zeros(3)
    offset[axis_idx] = (i - (n - 1) / 2.0) * args.spacing
    if not args.no_tint:
      tint_robot(model, prefix, colors[i])
    slots.append(
      dict(
        qadr=qadr,
        ndof=ndof,
        base_bid=int(base_bid),
        head_bid=int(head_bid),
        offset=offset,
        root0_xy=rec["root_pos"][0, :2].copy(),
      )
    )
  null_geom_names(model)

  # Grow offscreen buffer + AA before building a renderer (attach drops these).
  if args.record_video:
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.video_width)
    model.vis.global_.offheight = max(
      int(model.vis.global_.offheight), args.video_height
    )
    model.vis.quality.offsamples = max(int(model.vis.quality.offsamples), args.msaa)

  total_frames = max(len(r["root_pos"]) for r in recs)

  def horizontal_shift(offset, root0_xy, cur_xy):
    if args.root_mode == "lock":  # pin root to the lane every frame (in place)
      return np.array([offset[0] - cur_xy[0], offset[1] - cur_xy[1], 0.0])
    if args.root_mode == "recenter":  # recenter on the lane at t=0, then free
      return np.array([offset[0] - root0_xy[0], offset[1] - root0_xy[1], 0.0])
    return np.array([offset[0], offset[1], 0.0])  # absolute

  def set_frame(cur_i):
    for rec, slot in zip(recs, slots, strict=True):
      t = min(cur_i, len(rec["root_pos"]) - 1)
      qadr, ndof = slot["qadr"], slot["ndof"]
      root = rec["root_pos"][t]
      shift = horizontal_shift(slot["offset"], slot["root0_xy"], root[:2])
      data.qpos[qadr : qadr + 3] = root + shift
      data.qpos[qadr + 3 : qadr + 7] = rec["root_rot"][t]
      data.qpos[qadr + 7 : qadr + 7 + ndof] = rec["dof_pos"][t]
    mj.mj_forward(model, data)

  def draw_labels(scene):
    if args.no_labels:
      return
    for slot, label, color in zip(slots, labels, colors, strict=True):
      head = data.xpos[slot["head_bid"]] + np.array([0.0, 0.0, args.label_height])
      add_label_sphere(scene, head, color, label)

  cam_azimuth = 180.0 if args.axis == "y" else 90.0

  if args.record_video:
    _render_video(model, data, set_frame, draw_labels, total_frames, cam_azimuth, args)
  else:
    _render_native(model, data, set_frame, draw_labels, total_frames, cam_azimuth, args)


def _render_native(
  model, data, set_frame, draw_labels, total_frames, cam_azimuth, args
):
  import time

  import mujoco.viewer as mjv

  paused = [False]

  def key_callback(keycode):
    if keycode == 32:  # SPACE
      paused[0] = not paused[0]

  set_frame(0)
  viewer = mjv.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False, key_callback=key_callback
  )
  viewer.opt.geomgroup[3] = (
    0  # hide collision geoms (mjlab: visual=grp2, collision=grp3)
  )
  viewer.opt.label = mj.mjtLabel.mjLABEL_NONE
  viewer.cam.azimuth = args.cam_azimuth if args.cam_azimuth is not None else cam_azimuth
  viewer.cam.elevation = args.cam_elevation
  viewer.cam.distance = args.cam_distance

  frame_dt = 1.0 / args.fps
  cur_i = 0
  print("[compare] native viewer (SPACE=pause, close window to quit)")
  while viewer.is_running():
    t0 = time.perf_counter()
    if not paused[0]:
      set_frame(cur_i)
    viewer.cam.lookat[:] = np.mean([data.xpos[s] for s in _base_bids(model)], axis=0)
    viewer.user_scn.ngeom = 0
    draw_labels(viewer.user_scn)
    viewer.sync()
    time.sleep(max(0.0, frame_dt - (time.perf_counter() - t0)))
    if paused[0]:
      continue
    cur_i += 1
    if cur_i >= total_frames:
      if args.loop:
        cur_i = 0
      else:
        break
  viewer.close()
  time.sleep(0.2)


def _base_bids(model):
  return [
    model.jnt_bodyid[jid]
    for jid in range(model.njnt)
    if model.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE
  ]


def _render_video(model, data, set_frame, draw_labels, total_frames, cam_azimuth, args):
  import imageio

  cam = mj.MjvCamera()
  mj.mjv_defaultCamera(cam)
  cam.azimuth = args.cam_azimuth if args.cam_azimuth is not None else cam_azimuth
  cam.elevation = args.cam_elevation
  cam.distance = args.cam_distance

  opt = mj.MjvOption()
  mj.mjv_defaultOption(opt)
  opt.geomgroup[3] = 0  # hide collision geoms (mjlab: visual=grp2, collision=grp3)
  opt.label = mj.mjtLabel.mjLABEL_NONE

  renderer = mj.Renderer(model, height=args.video_height, width=args.video_width)
  out = Path(args.video_path)
  out.parent.mkdir(parents=True, exist_ok=True)
  writer = imageio.get_writer(
    str(out), fps=args.fps, quality=args.video_quality, macro_block_size=None
  )
  base_bids = _base_bids(model)
  print(f"[compare] rendering {total_frames} frames -> {out}")
  for cur_i in range(total_frames):
    set_frame(cur_i)
    cam.lookat[:] = np.mean([data.xpos[b] for b in base_bids], axis=0)
    renderer.update_scene(data, camera=cam, scene_option=opt)
    draw_labels(renderer.scene)
    writer.append_data(renderer.render())
  writer.close()
  renderer.close()
  print(f"[compare] saved {out}")


# --------------------------------------------------------------------------- #
def main() -> None:
  import mjlab.tasks  # noqa: F401  (populate the task registry)

  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--robot", required=True, choices=sorted(ROBOT_TASKS))
  parser.add_argument(
    "--registries",
    nargs="+",
    required=True,
    help="One W&B registry per method (e.g. colmo_g1 gmr_g1 omniretarget_g1).",
  )
  parser.add_argument("--motion", required=True, help="Source motion name.")
  parser.add_argument("--labels", nargs="+", default=None, help="Per-method labels.")
  parser.add_argument(
    "--mirror",
    nargs="*",
    default=["omni"],
    help="Mirror (left-right) any method whose registry contains one of these "
    "substrings (default: 'omni'). Pass --mirror with no value to disable.",
  )
  parser.add_argument(
    "--clusters-csv",
    type=Path,
    default=Path(__file__).resolve().parent / "motion_clusters_assignments.csv",
  )
  parser.add_argument("--task", default=None)
  parser.add_argument(
    "--seconds",
    type=float,
    default=None,
    help="Deploy/playback duration (default: the whole motion).",
  )
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument("--org", default=None)
  parser.add_argument("--log-root", default="logs/rsl_rl")
  parser.add_argument("--device", default=None)
  # Layout / rendering.
  parser.add_argument("--spacing", type=float, default=1.3, help="Lane gap (m).")
  parser.add_argument("--axis", choices=["x", "y"], default="y")
  parser.add_argument(
    "--root-mode", choices=["lock", "recenter", "absolute"], default="recenter"
  )
  parser.add_argument("--no-tint", action="store_true")
  parser.add_argument("--no-labels", action="store_true")
  parser.add_argument("--label-height", type=float, default=0.55)
  parser.add_argument("--loop", action="store_true")
  parser.add_argument("--fps", type=int, default=50)
  parser.add_argument("--cam-distance", type=float, default=4.0)
  parser.add_argument("--cam-azimuth", type=float, default=None)
  parser.add_argument("--cam-elevation", type=float, default=-15.0)
  # Video.
  parser.add_argument("--record-video", action="store_true")
  parser.add_argument("--video-path", default=None)
  parser.add_argument("--video-width", type=int, default=1280)
  parser.add_argument("--video-height", type=int, default=720)
  parser.add_argument("--video-quality", type=int, default=8)
  parser.add_argument("--msaa", type=int, default=8)
  args = parser.parse_args()

  if args.robot not in ROBOT_XML:
    raise SystemExit(f"No render MJCF registered for robot '{args.robot}'.")
  if args.labels and len(args.labels) != len(args.registries):
    raise SystemExit("--labels must have one entry per --registries.")
  labels = args.labels or list(args.registries)
  colors = [
    DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(len(args.registries))
  ]
  if args.video_path is None:
    args.video_path = f"videos/compare_{args.motion}.mp4"

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  task_id = args.task or ROBOT_TASKS[args.robot]

  clusters = read_clusters(args.clusters_csv)
  if args.motion not in clusters:
    raise SystemExit(f"Motion '{args.motion}' not in {args.clusters_csv}.")
  cluster_id, action = clusters[args.motion]

  import wandb

  api = wandb.Api()

  motion_files: list[str] = []
  ckpts: list[Path] = []
  print(f"Motion : {args.motion} (cluster {cluster_id} {action})")
  for registry, label in zip(args.registries, labels, strict=True):
    exp_dir = (Path(args.log_root) / f"{registry}_clusters").resolve()
    ckpt = find_cluster_checkpoint(exp_dir, cluster_id, args.checkpoint)
    if ckpt is None:
      raise SystemExit(f"No cluster-{cluster_id} checkpoint under {exp_dir}")
    org = resolve_org(api, registry, args.org)
    art = api.artifact(f"{org}/wandb-registry-{registry}/{args.motion}:latest")
    motion_files.append(str(Path(art.download()) / "motion.npz"))
    ckpts.append(ckpt)
    print(f"  env {len(ckpts) - 1}: {label:<16s} npz={registry}  policy={ckpt.name}")

  # Phase 1: deploy + record.
  env, agent_cfg, command = build_compare_env(task_id, motion_files, device)
  pin_clips_to_envs(command)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  policies = []
  for ckpt in ckpts:
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(ckpt), map_location=device)
    policies.append(runner.get_inference_policy(device=device))
  step_dt = float(env.unwrapped.step_dt)
  lengths = command.motion.clip_lengths.tolist()
  if len(set(lengths)) > 1:
    print(
      f"[warn] clip lengths differ {lengths}; robots may drift out of sync and "
      f"playback is capped at the shortest ({min(lengths)} frames)."
    )
  # Stop one frame before the shortest clip's end: reaching clip_ends triggers a
  # motion wrap that teleports the robot back to frame 0, which would otherwise be
  # recorded as the final frame (a snap-back to the start pose).
  clip_len = max(1, int(min(lengths)) - 1)
  # Default: play the WHOLE motion; --seconds caps it if given.
  num_frames = (
    clip_len if args.seconds is None else min(int(args.seconds / step_dt), clip_len)
  )
  print(f"\n[deploy] recording {num_frames} frames ({num_frames * step_dt:.1f}s)...")
  recs, frozen_at = record_deploy(
    env, MultiPolicy(policies), len(motion_files), num_frames, device
  )
  env.close()
  for i, fa in enumerate(frozen_at):
    if fa is not None:
      print(f"[deploy] {labels[i]} terminated at frame {fa} -> frozen there")

  # Phase 2: mirror the requested methods, then compose + render.
  if args.mirror:
    perm, sign = build_mirror_spec(ROBOT_XML[args.robot])
    for i, registry in enumerate(args.registries):
      if any(s in registry for s in args.mirror):
        mirror_trajectory(recs[i], perm, sign)
        print(f"[mirror] {labels[i]} ({registry}) shown left-right mirrored")

  if args.record_video and not (
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
  ):
    os.environ.setdefault("MUJOCO_GL", "egl")  # headless offscreen render
  visualize(recs, labels, colors, args.robot, args)


if __name__ == "__main__":
  main()
