from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  kapex_flat_env_cfg,
  kapex_rough_env_cfg,
)
from .rl_cfg import kapex_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Kapex",
  env_cfg=kapex_rough_env_cfg(),
  play_env_cfg=kapex_rough_env_cfg(play=True),
  rl_cfg=kapex_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Kapex",
  env_cfg=kapex_flat_env_cfg(),
  play_env_cfg=kapex_flat_env_cfg(play=True),
  rl_cfg=kapex_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
