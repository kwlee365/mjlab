#!/usr/bin/env python3
"""Evaluate cluster tracking policies and compare retargeting methods.

For each motion in the cluster-assignment CSV, and for each retargeting method
given by ``--registries``, this loads the policy that method trained on the
motion's cluster (under ``logs/rsl_rl/<registry>_clusters/*_cluster<k>_*/``) and
runs it for ``--num-samples`` random-start rollouts of ``--episode-seconds``,
reporting the downstream-RL metrics from the retargeting-evaluation literature:

  - success_rate : fraction of rollouts that keep their balance -- the root never
                   drops more than ``--root-pos-fail`` m in HEIGHT nor tilts more
                   than ``--root-ori-fail-deg`` deg from the target (horizontal
                   drift is ignored; defaults 0.5 m / 45 deg).
  - Eg-mpbpe (mm): global mean per-body-part position error.
  - Empbpe (mm)  : root-relative mean per-body-part position error.
  - Empjpe (1e-3 rad): mean per-joint angular error.

Tracking errors are averaged over successful rollouts only. "Root" is the task's
anchor body (torso_link for G1). Each method's policy tracks THAT method's own
retargeting of the motion (in-distribution); results are aggregated per method so
you can compare retargeting algorithms on a common motion set.

The env is built once and reused; motions and checkpoints are swapped in, so the
whole sweep costs a single env build.

Run inside the mjlab env (needs W&B to fetch the motions):

    # compare three retargeting methods over every motion, 256 samples each
    uv run python eval_clusters.py --robot g1 \
        --registries colmo_g1 gmr_g1 omniretarget_g1 --output eval.csv

    # one method, a few clusters, fewer samples per motion
    uv run python eval_clusters.py --robot g1 --registries colmo_g1 \
        --clusters 2 8 --num-samples 64

    # cache motions locally: first run downloads + fills the dir, later runs read
    # from it and never touch W&B (much faster to re-run)
    uv run python eval_clusters.py --robot g1 \
        --registries colmo_g1 gmr_g1 omniretarget_g1 --motions-dir motion_cache
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionLoader
from mjlab.utils.lab_api.math import quat_error_magnitude
from mjlab.utils.torch import configure_torch_backends

# Tracking task id per robot; --robot selects one of these.
ROBOT_TASKS = {
  "g1": "Mjlab-Tracking-Flat-Unitree-G1",
  "kapex": "Mjlab-Tracking-Flat-Kapex",
}

# Paper metrics: success rate (%), global / root-relative body position error (mm),
# joint angular error (1e-3 rad).
METRIC_KEYS = ["success_rate", "eg_mpbpe_mm", "empbpe_mm", "empjpe_mrad"]


def read_clusters(csv_path: Path) -> "OrderedDict[str, tuple[int, str]]":
  """Return motion -> (cluster_id, action), preserving CSV order (grouped by
  cluster) so consecutive motions share a checkpoint."""
  motions: OrderedDict[str, tuple[int, str]] = OrderedDict()
  with open(csv_path, newline="") as f:
    for row in csv.DictReader(f):
      motions[row["motion"]] = (int(row["cluster"]), row.get("action", ""))
  if not motions:
    raise SystemExit(f"No rows in {csv_path}")
  return motions


def _checkpoint_step(path: Path) -> int:
  nums = re.findall(r"\d+", path.stem)
  return int(nums[-1]) if nums else -1


def _latest_checkpoint(run_dir: Path) -> Path | None:
  cands = [p for p in run_dir.glob("model_*.pt") if p.is_file()]
  if not cands:
    return None
  return max(cands, key=lambda p: (_checkpoint_step(p), p.stat().st_mtime))


def find_cluster_checkpoint(
  exp_dir: Path, cluster_id: int, checkpoint: str | None
) -> Path | None:
  """Latest checkpoint of the (most recently trained) run for a cluster."""
  run_dirs = [p for p in exp_dir.glob(f"*_cluster{cluster_id}_*") if p.is_dir()]
  best: tuple[float, Path] | None = None
  for run_dir in run_dirs:
    ckpt = _latest_checkpoint(run_dir)
    if ckpt is None:
      continue
    mtime = ckpt.stat().st_mtime
    if best is None or mtime > best[0]:
      best = (mtime, run_dir)
  if best is None:
    return None
  run_dir = best[1]
  if checkpoint is not None:
    ckpt = run_dir / checkpoint
    return ckpt if ckpt.is_file() else None
  return _latest_checkpoint(run_dir)


def resolve_org(api, registry: str, org: str | None) -> str:
  """Return the W&B org entity that hosts wandb-registry-<registry>."""
  default = api.default_entity
  candidates = [org] if org else [f"{default}-org", default]
  last_err: Exception | None = None
  for candidate in candidates:
    if candidate is None:
      continue
    try:
      list(api.artifact_types(project=f"{candidate}/wandb-registry-{registry}"))
      return candidate
    except Exception as exc:  # noqa: BLE001 - try the next candidate.
      last_err = exc
  raise SystemExit(f"Could not resolve org for registry '{registry}': {last_err}")


def build_eval_env(
  task_id: str, num_envs: int, device: str, corruption: bool, init_motion_file: str
):
  """Build the tracking env for evaluation: random start frame, no RSI
  perturbation, no push. The env failure terminations are removed and the episode
  length is effectively infinite -- each rollout runs a fixed number of steps and
  failure is judged in rollout() by the paper criterion (root drift), so a failed
  robot keeps stepping (its pose is genuine, never teleported by an auto-reset).
  ``init_motion_file`` seeds the MotionCommand so the env can build."""
  env_cfg = load_env_cfg(task_id, play=False)
  agent_cfg = load_rl_cfg(task_id)

  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise SystemExit(f"Task {task_id} is not a tracking task.")

  motion_cmd.motion_file = init_motion_file
  motion_cmd.motion_files = ()

  # Random start frame, exactly at the reference pose (RSI perturbation disabled).
  motion_cmd.sampling_mode = "uniform"
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.joint_position_range = (0.0, 0.0)

  env_cfg.episode_length_s = 1e9  # no time-out within the fixed-length rollout
  env_cfg.events.pop("push_robot", None)
  env_cfg.observations["actor"].enable_corruption = corruption
  for term in ("anchor_pos", "anchor_ori", "ee_body_pos"):
    env_cfg.terminations.pop(term, None)  # failure judged in rollout(), not by env
  env_cfg.scene.num_envs = num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  command = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))
  return env, agent_cfg, command


def swap_motion(command: MotionCommand, motion_file: str) -> None:
  """Point the running command at a single motion npz."""
  command.motion = MotionLoader(
    motion_file, command.body_indexes, device=command.device
  )
  command.time_steps.zero_()
  if hasattr(command, "clip_ids"):
    command.clip_ids.zero_()


def rollout(
  env,
  command: MotionCommand,
  policy,
  num_envs: int,
  num_steps: int,
  root_pos_fail: float,
  root_ori_fail: float,
  device: str,
) -> dict[str, float]:
  """Run ``num_envs`` random-start rollouts of ``policy`` on the current motion for
  ``num_steps`` steps and return the paper metrics:

    success_rate : fraction of rollouts whose root never drops past root_pos_fail
                   (metres, HEIGHT only) or tilts past root_ori_fail (radians).
    eg_mpbpe_mm  : global mean-per-body-position error (mm).
    empbpe_mm    : root-relative mean-per-body-position error (mm).
    empjpe_mrad  : mean per-joint angular error (1e-3 rad).

  Tracking metrics are averaged over the frames each rollout is still tracking
  (before any failure) and then over SUCCESSFUL rollouts only, so a rollout that
  drifts away does not pull the error down. "Root" is the anchor body (torso for
  G1). Failure is judged here (env terminations are off), so a drifting robot is
  never teleported by an auto-reset mid-rollout."""
  failed = torch.zeros(num_envs, dtype=torch.bool, device=device)
  sum_eg = torch.zeros(num_envs, device=device)
  sum_empbpe = torch.zeros(num_envs, device=device)
  sum_empjpe = torch.zeros(num_envs, device=device)
  frames = torch.zeros(num_envs, device=device)

  env.reset()
  obs = env.get_observations()

  for _ in range(num_steps):
    # Snapshot the reference this step is scored against BEFORE stepping: env.step
    # scores against the current frame then advances it, so the post-step robot
    # state pairs with the pre-step reference (matches the reward/metric pairing).
    ref_body_pos = command.body_pos_w.clone()  # global reference body positions
    ref_body_rel = command.body_pos_relative_w.clone()  # root-relative reference
    ref_joint = command.joint_pos.clone()
    ref_anchor_pos = command.anchor_pos_w.clone()
    ref_anchor_quat = command.anchor_quat_w.clone()

    with torch.no_grad():
      actions = policy(obs)
    obs, _, _, _ = env.step(actions)

    robot_body = command.robot_body_pos_w
    eg = (ref_body_pos - robot_body).norm(dim=-1).mean(dim=-1)  # global MPBPE (m)
    empbpe = (ref_body_rel - robot_body).norm(dim=-1).mean(dim=-1)  # root-rel (m)
    empjpe = (ref_joint - command.robot_joint_pos).abs().mean(dim=-1)  # joint (rad)

    # Height (z) only: horizontal global drift is a tracking error, not a loss of
    # balance, so it does not count as a failure here -- only a vertical drop
    # (falling / crouching) or excessive tilt does.
    root_pos_dev = (ref_anchor_pos[:, 2] - command.robot_anchor_pos_w[:, 2]).abs()
    root_ori_dev = quat_error_magnitude(ref_anchor_quat, command.robot_anchor_quat_w)
    failed = failed | (root_pos_dev > root_pos_fail) | (root_ori_dev > root_ori_fail)

    alive = ~failed  # skip the failing frame and everything after it
    sum_eg += torch.where(alive, eg, 0.0)
    sum_empbpe += torch.where(alive, empbpe, 0.0)
    sum_empjpe += torch.where(alive, empjpe, 0.0)
    frames += alive.float()

  success = ~failed
  denom = frames.clamp(min=1)
  eg_mm = (sum_eg / denom) * 1000.0
  empbpe_mm = (sum_empbpe / denom) * 1000.0
  empjpe_mrad = (sum_empjpe / denom) * 1000.0
  nan = float("nan")
  ok = success
  return {
    "success_rate": success.float().mean().item(),
    "eg_mpbpe_mm": eg_mm[ok].mean().item() if ok.any() else nan,
    "empbpe_mm": empbpe_mm[ok].mean().item() if ok.any() else nan,
    "empjpe_mrad": empjpe_mrad[ok].mean().item() if ok.any() else nan,
  }


def _mean(rows: list[dict], key: str) -> float:
  """Mean over rows, skipping NaN values (a motion whose samples all failed has
  NaN RMSE). Returns NaN if every value is NaN."""
  vals = [r[key] for r in rows if not math.isnan(r[key])]
  return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
  import mjlab.tasks  # noqa: F401  (populate the task registry)

  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
    "--robot", required=True, choices=sorted(ROBOT_TASKS), help="Selects the task."
  )
  parser.add_argument(
    "--registries",
    nargs="+",
    required=True,
    help="One W&B registry per retargeting method to evaluate and compare "
    "(e.g. colmo_g1 gmr_g1 omniretarget_g1). Each needs a matching "
    "logs/rsl_rl/<registry>_clusters/.",
  )
  parser.add_argument(
    "--num-samples",
    type=int,
    default=100,
    help="Random-start rollouts (samples) per motion (default: 100, per the paper).",
  )
  parser.add_argument(
    "--episode-seconds", type=float, default=5.0, help="Rollout length (default 5 s)."
  )
  parser.add_argument(
    "--root-pos-fail",
    type=float,
    default=0.25,
    help="A rollout fails if the root HEIGHT (z) drifts past this many metres "
    "(horizontal drift is ignored; default 0.25 -- a clear fall/crouch).",
  )
  parser.add_argument(
    "--root-ori-fail-deg",
    type=float,
    default=45.0,
    help="A rollout fails if the root orientation error exceeds this many degrees "
    "(default 45).",
  )
  parser.add_argument(
    "--clusters-csv",
    type=Path,
    default=Path(__file__).resolve().parent / "motion_clusters_assignments.csv",
  )
  parser.add_argument("--task", default=None, help="Override the task id.")
  parser.add_argument(
    "--motions", nargs="*", default=None, help="Only these motions (default: all)."
  )
  parser.add_argument(
    "--clusters", type=int, nargs="*", default=None, help="Only these clusters."
  )
  parser.add_argument(
    "--corruption",
    action="store_true",
    help="Enable observation noise during eval (default: off for clean tracking).",
  )
  parser.add_argument("--org", default=None, help="W&B org entity (auto-detected).")
  parser.add_argument("--checkpoint", default=None, help="Checkpoint file (latest).")
  parser.add_argument("--log-root", default="logs/rsl_rl")
  parser.add_argument(
    "--motions-dir",
    default=None,
    help="Local npz cache dir (layout <dir>/<registry>/<motion>.npz). When set, "
    "motions present there are read directly (no W&B), and any fetched from W&B "
    "are copied there -- so repeat runs need no download. Populates on first run.",
  )
  parser.add_argument("--output", default=None, help="Write per-motion CSV here.")
  parser.add_argument("--device", default=None)
  args = parser.parse_args()

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  task_id = args.task or ROBOT_TASKS[args.robot]

  motions = read_clusters(args.clusters_csv)
  names = list(motions)
  if args.motions:
    wanted = set(args.motions)
    names = [m for m in names if m in wanted]
  if args.clusters:
    keep = set(args.clusters)
    names = [m for m in names if motions[m][0] in keep]
  if not names:
    raise SystemExit("No motions selected after filtering.")

  motions_dir = Path(args.motions_dir).resolve() if args.motions_dir else None

  # Motion npzs live in W&B, but per-artifact API lookups are slow. With
  # --motions-dir we keep a plain <dir>/<registry>/<motion>.npz cache: a motion
  # already there is used directly (no W&B), and one fetched from W&B is copied
  # there so repeat runs need no network. W&B is initialised lazily -- only when a
  # motion must actually be downloaded (so an all-local run never touches W&B).
  _wb: dict = {"api": None, "org": {}}

  def get_npz(registry: str, motion: str) -> str:
    local = motions_dir / registry / f"{motion}.npz" if motions_dir else None
    if local is not None and local.exists():
      return str(local)
    if _wb["api"] is None:
      import wandb

      _wb["api"] = wandb.Api()
    api = _wb["api"]
    if registry not in _wb["org"]:
      _wb["org"][registry] = resolve_org(api, registry, args.org)
    art = api.artifact(
      f"{_wb['org'][registry]}/wandb-registry-{registry}/{motion}:latest"
    )
    path = Path(art.download()) / "motion.npz"
    if not path.exists():  # artifact downloaded but has no motion.npz
      raise FileNotFoundError(path)
    if local is not None:  # populate the local cache for next time
      local.parent.mkdir(parents=True, exist_ok=True)
      shutil.copyfile(path, local)
      return str(local)
    return str(path)

  # Per method: the cluster checkpoint for each motion (local only, no W&B).
  # find_cluster_checkpoint is per-cluster, so cache by cluster within a method.
  registries: list[str] = []
  ckpt_of: dict[tuple[str, str], Path] = {}
  for registry in args.registries:
    exp_dir = (Path(args.log_root) / f"{registry}_clusters").resolve()
    if not exp_dir.exists():
      print(f"[skip] {registry}: {exp_dir} not found")
      continue
    registries.append(registry)
    ckpt_by_cluster: dict[int, Path | None] = {}
    for motion in names:
      cluster_id = motions[motion][0]
      if cluster_id not in ckpt_by_cluster:
        ckpt_by_cluster[cluster_id] = find_cluster_checkpoint(
          exp_dir, cluster_id, args.checkpoint
        )
      ckpt = ckpt_by_cluster[cluster_id]
      if ckpt is not None:
        ckpt_of[(registry, motion)] = ckpt
  if not registries:
    raise SystemExit("No registry has an experiment dir under --log-root.")

  # Compare on a COMMON motion set: only motions every method can evaluate (all
  # have a checkpoint AND a downloadable npz), so per-method means are over the
  # same population rather than each method's own coverage.
  common = [m for m in names if all((r, m) in ckpt_of for r in registries)]
  no_ckpt = [m for m in names if m not in common]
  if no_ckpt:
    print(
      f"[warn] {len(no_ckpt)} motion(s) lack a checkpoint in some method, dropped "
      f"from the comparison: {no_ckpt}"
    )
  if not common:
    raise SystemExit("No motion has a trained checkpoint in every method.")

  # Pre-download every (method, common-motion) npz up front: this catches missing
  # npzs before any rollout (instead of crashing mid-sweep) and lets a motion that
  # is unavailable for one method be dropped for all, keeping the set common.
  npz_of: dict[tuple[str, str], str] = {}
  bad: set[str] = set()
  for registry in registries:
    for motion in common:
      try:
        npz_of[(registry, motion)] = get_npz(registry, motion)
      except Exception as exc:  # noqa: BLE001 - drop this motion from every method.
        print(f"[warn] {registry}/{motion}: npz unavailable ({type(exc).__name__})")
        bad.add(motion)
  if bad:
    common = [m for m in common if m not in bad]
    print(f"[warn] dropped {len(bad)} motion(s) with a missing npz: {sorted(bad)}")
  if not common:
    raise SystemExit("No motion has a usable npz in every method.")

  root_ori_fail = math.radians(args.root_ori_fail_deg)
  print(f"Task    : {task_id}")
  print(f"Methods : {registries}")
  print(f"Motions : {len(common)} common (each evaluated for every method)")
  print("\nEvaluation criteria")
  print("-" * 66)
  print(
    f"  rollouts : {args.num_samples} per motion, {args.episode_seconds}s each, "
    f"random start frame"
  )
  print(
    f"  success  : root (anchor/torso) never drops > {args.root_pos_fail} m in "
    f"HEIGHT\n             or tilts > {args.root_ori_fail_deg} deg from the target "
    f"(horizontal drift ignored)"
  )
  print("  Eg-mpbpe : global mean per-body-part position error (mm)")
  print("  Empbpe   : root-relative mean per-body-part position error (mm)")
  print("  Empjpe   : mean per-joint angular error (1e-3 rad)")
  print("  (tracking errors are averaged over successful rollouts only)")
  print("-" * 66 + "\n")

  env, agent_cfg, command = build_eval_env(
    task_id,
    args.num_samples,
    device,
    args.corruption,
    npz_of[(registries[0], common[0])],
  )
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  step_dt = float(env.unwrapped.step_dt)
  num_steps = int(args.episode_seconds / step_dt)

  # Method-by-method; motions stay in cluster-grouped order so the checkpoint
  # reloads at most once per cluster within a method.
  results: list[dict] = []
  loaded_ckpt: Path | None = None
  total = len(registries) * len(common)
  i = 0
  for registry in registries:
    for motion in common:
      i += 1
      cluster_id, action = motions[motion]
      ckpt = ckpt_of[(registry, motion)]
      if ckpt != loaded_ckpt:
        runner.load(str(ckpt), map_location=device)
        loaded_ckpt = ckpt
      policy = runner.get_inference_policy(device=device)
      swap_motion(command, npz_of[(registry, motion)])
      m = rollout(
        env,
        command,
        policy,
        args.num_samples,
        num_steps,
        args.root_pos_fail,
        root_ori_fail,
        device,
      )
      results.append(
        dict(method=registry, motion=motion, cluster=cluster_id, action=action, **m)
      )
      print(
        f"[{i}/{total}] {registry:<16s} {motion:<22s} (c{cluster_id}): "
        f"success={m['success_rate']:.2f} Eg-mpbpe={m['eg_mpbpe_mm']:.1f}mm "
        f"Empbpe={m['empbpe_mm']:.1f}mm Empjpe={m['empjpe_mrad']:.1f}e-3rad"
      )

  env.close()
  if not results:
    raise SystemExit("No motions were evaluated.")

  # Per-method comparison table (the headline result).
  by_method: dict[str, list[dict]] = {}
  for r in results:
    by_method.setdefault(r["method"], []).append(r)
  print("\n" + "=" * 78)
  print("RETARGETING METHOD COMPARISON  (mean over motions)")
  print("-" * 78)
  print(
    f"{'method':<18s} {'n':>4s} {'success%':>9s} {'Eg-mpbpe':>10s} "
    f"{'Empbpe':>10s} {'Empjpe':>10s}"
  )
  print(f"{'':<18s} {'':>4s} {'':>9s} {'(mm)':>10s} {'(mm)':>10s} {'(1e-3rad)':>10s}")
  for method in registries:
    rows = by_method.get(method)
    if not rows:
      continue
    print(
      f"{method:<18s} {len(rows):>4d} {_mean(rows, 'success_rate') * 100:>9.1f} "
      f"{_mean(rows, 'eg_mpbpe_mm'):>10.1f} {_mean(rows, 'empbpe_mm'):>10.1f} "
      f"{_mean(rows, 'empjpe_mrad'):>10.1f}"
    )
  print("=" * 78)

  if args.output:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "motion", "cluster", "action", *METRIC_KEYS]
    with open(out, "w", newline="") as f:
      w = csv.DictWriter(f, fieldnames=fields)
      w.writeheader()
      for r in results:
        w.writerow({k: r[k] for k in fields})
    print(f"\n[INFO] Per-motion metrics written to {out}")


if __name__ == "__main__":
  main()
