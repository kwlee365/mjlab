"""Play a tracking policy while it is still training, always following the
latest checkpoint.

It finds the currently-training run under ``logs/rsl_rl/<experiment>/`` (the one
with the most recently written ``model_*.pt``), loads its newest checkpoint, and
plays it in the viewer. Every ``--reload-interval-s`` seconds it polls the run
directory and hot-swaps to a newer checkpoint as soon as training saves one, so
you can watch the policy improve live.

The reference motion is taken from the run's ``params/env.yaml`` (the exact npz
the run trained on), so no W&B round-trip is needed. Override with
``--motion-file`` if desired.

Usage:
  uv run python play_latest.py Mjlab-Tracking-Flat-Unitree-G1
  uv run python play_latest.py Mjlab-Tracking-Flat-Unitree-G1 --run-dir logs/rsl_rl/g1_tracking/2026-07-09_11-35-54
  # follow a whole batch: auto-switch (and swap the motion) as it moves on
  uv run python play_latest.py Mjlab-Tracking-Flat-Unitree-G1 --experiment logs/rsl_rl/colmo --device cpu --viewer viser
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


def _checkpoint_step(path: Path) -> int:
  nums = re.findall(r"\d+", path.stem)
  return int(nums[-1]) if nums else -1


def _latest_checkpoint(run_dir: Path) -> Path | None:
  cands = [p for p in run_dir.glob("model_*.pt") if p.is_file()]
  if not cands:
    return None
  # Prefer the highest training step; break ties by modification time.
  return max(cands, key=lambda p: (_checkpoint_step(p), p.stat().st_mtime))


def _find_active_run_dir(experiment_dir: Path) -> Path:
  """Return the run dir whose newest checkpoint was modified most recently."""
  best: tuple[float, Path] | None = None
  for run_dir in experiment_dir.iterdir():
    if not run_dir.is_dir():
      continue
    ckpt = _latest_checkpoint(run_dir)
    if ckpt is None:
      continue
    mtime = ckpt.stat().st_mtime
    if best is None or mtime > best[0]:
      best = (mtime, run_dir)
  if best is None:
    raise FileNotFoundError(
      f"No checkpoints found under {experiment_dir}. Is training running?"
    )
  return best[1]


def _resolve_motion_file(run_dir: Path) -> str:
  """Read the reference motion path from the run's dumped env config."""
  env_yaml = run_dir / "params" / "env.yaml"
  if not env_yaml.exists():
    raise FileNotFoundError(f"{env_yaml} not found; pass --motion-file explicitly.")
  # Regex rather than a YAML parser: dump_yaml may emit Python-specific tags
  # that safe_load rejects, and we only need this one scalar.
  for line in env_yaml.read_text().splitlines():
    m = re.match(r"\s*motion_file:\s*(.+?)\s*$", line)
    if m:
      return m.group(1).strip().strip("'\"")
  raise ValueError(f"motion_file not found in {env_yaml}; pass --motion-file.")


class LatestCheckpointPolicy:
  """Callable policy that hot-reloads the newest checkpoint of the run it follows.

  The viewer calls ``policy(obs)`` every control step (see BaseViewer); we
  throttle the (cheap) directory poll to once per ``interval_s`` and only reload
  when a newer checkpoint appears. A checkpoint still being written is caught by
  the try/except and retried next poll.

  If ``experiment_dir`` is given, it also tracks whichever run under that
  experiment is currently active (newest checkpoint) and, when the active run
  changes (a batch moving to the next motion), swaps the reference motion in
  place -- no env/viewer rebuild. All runs must share the same task (obs/action
  layout), which holds for one robot's tracking runs.
  """

  def __init__(
    self,
    runner: MjlabOnPolicyRunner,
    env,
    run_dir: Path,
    device: str,
    interval_s: float = 2.0,
    experiment_dir: Path | None = None,
    is_tracking: bool = True,
  ) -> None:
    self.runner = runner
    self.env = env
    self.run_dir = run_dir
    self.device = device
    self.interval_s = interval_s
    self.experiment_dir = experiment_dir
    self.is_tracking = is_tracking
    self._current_motion: str | None = None
    self._policy = None
    self._loaded_path: Path | None = None
    self._loaded_mtime: float = -1.0
    self._last_check: float = 0.0
    self._reload(force=True)
    if self._policy is None:
      raise FileNotFoundError(f"No loadable checkpoint in {run_dir}")

  def _swap_motion(self, motion_file: str) -> None:
    from mjlab.tasks.tracking.mdp.commands import MotionLoader

    cmd = self.env.unwrapped.command_manager.get_term("motion")
    cmd.motion = MotionLoader(motion_file, cmd.body_indexes, device=cmd.device)
    cmd.time_steps.zero_()
    self.env.reset()
    self._current_motion = motion_file
    print(f"[play_latest] motion -> {Path(motion_file).parent.name}", flush=True)

  def _maybe_switch_run(self) -> None:
    if self.experiment_dir is None:
      return
    try:
      active = _find_active_run_dir(self.experiment_dir)
    except FileNotFoundError:
      return
    if active == self.run_dir:
      return
    print(f"[play_latest] active run -> {active.name}", flush=True)
    self.run_dir = active
    self._loaded_path = None  # force a checkpoint reload from the new run
    if self.is_tracking:
      try:
        new_motion = _resolve_motion_file(active)
      except (FileNotFoundError, ValueError):
        return  # params/env.yaml not written yet; retry next poll
      if new_motion != self._current_motion and Path(new_motion).exists():
        self._swap_motion(new_motion)

  def _reload(self, force: bool = False) -> None:
    now = time.perf_counter()
    if not force and now - self._last_check < self.interval_s:
      return
    self._last_check = now

    self._maybe_switch_run()

    latest = _latest_checkpoint(self.run_dir)
    if latest is None:
      return
    mtime = latest.stat().st_mtime
    if latest == self._loaded_path and mtime == self._loaded_mtime:
      return

    try:
      self.runner.load(
        str(latest), load_cfg={"actor": True}, strict=True, map_location=self.device
      )
      self._policy = self.runner.get_inference_policy(device=self.device)
    except Exception as exc:  # noqa: BLE001 - file may be mid-write; retry later.
      print(f"[play_latest] skip {latest.name} (still writing?): {type(exc).__name__}")
      return

    self._loaded_path = latest
    self._loaded_mtime = mtime
    print(f"[play_latest] now playing: {self.run_dir.name}/{latest.name}", flush=True)

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    self._reload()
    return self._policy(obs)  # type: ignore[misc]

  def reset(self) -> None:  # Called by the viewer on env reset.
    pass


@dataclass(frozen=True)
class Config:
  run_dir: str | None = None
  """A single run directory to follow. Defaults to the currently-training run."""
  experiment: str | None = None
  """Experiment dir (e.g. logs/rsl_rl/colmo) to follow: tracks whichever run is
  currently training and auto-switches (swapping the motion) when a batch moves
  on to the next run. Takes precedence over --run-dir."""
  motion_file: str | None = None
  """Reference motion npz. Defaults to the run's params/env.yaml value."""
  reload_interval_s: float = 2.0
  """How often (seconds) to poll for a newer checkpoint / run."""
  num_envs: int | None = None
  device: str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"


def run(task_id: str, cfg: Config) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  is_tracking = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  # Follow an experiment dir (auto-switching runs) or a single run.
  follow_experiment = cfg.experiment is not None
  experiment_dir: Path | None = None
  if follow_experiment:
    experiment_dir = Path(cfg.experiment).resolve()
    run_dir = _find_active_run_dir(experiment_dir)
    print(
      f"[play_latest] following experiment: {experiment_dir} (active: {run_dir.name})"
    )
  elif cfg.run_dir is not None:
    run_dir = Path(cfg.run_dir).resolve()
    print(f"[play_latest] following run: {run_dir}")
  else:
    run_dir = _find_active_run_dir(
      (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    )
    print(f"[play_latest] following run: {run_dir}")

  # Tracking tasks need a reference motion resolved from the run; velocity and
  # other command-driven tasks do not.
  if is_tracking:
    motion_file = cfg.motion_file or _resolve_motion_file(run_dir)
    if not Path(motion_file).exists():
      raise FileNotFoundError(f"Motion file does not exist: {motion_file}")
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = motion_file
    print(f"[play_latest] motion: {motion_file}")

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)

  policy = LatestCheckpointPolicy(
    runner,
    env,
    run_dir,
    device,
    interval_s=cfg.reload_interval_s,
    experiment_dir=experiment_dir if follow_experiment else None,
    is_tracking=is_tracking,
  )

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved = "native" if has_display else "viser"
  else:
    resolved = cfg.viewer

  print(f"[play_latest] viewer: {resolved} | polling every {cfg.reload_interval_s}s")
  if resolved == "native":
    NativeMujocoViewer(env, policy).run()
  else:
    ViserPlayViewer(env, policy).run()

  env.close()


def main() -> None:
  import mjlab.tasks  # noqa: F401  (populate the task registry)

  all_tasks = list_tasks()
  chosen_task, remaining = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  cfg = tyro.cli(
    Config,
    args=remaining,
    default=Config(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run(chosen_task, cfg)


if __name__ == "__main__":
  main()
