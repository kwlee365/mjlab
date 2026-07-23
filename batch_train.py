#!/usr/bin/env python3
"""Train one tracking policy per motion in a W&B registry.

Lists every collection in ``wandb-registry-<registry>`` and runs ``uv run train``
for each, one after another (a single GPU trains one policy at a time). Runs are
grouped on disk by registry:

    logs/rsl_rl/<registry>/<timestamp>_<motion>/model_*.pt

Run inside the mjlab env (it queries W&B):

    uv run python batch_train.py --registry colmo
    uv run python batch_train.py --registry colmo --filter 'walk.*' --num-envs 4096
    uv run python batch_train.py --registry colmo --max-iterations 10000 --dry-run

Heads up: a full tracking run is ~thousands of iterations (hours). Training every
motion in a 60-clip registry sequentially takes days -- use --filter to pick a
subset and/or --max-iterations to shorten, and --skip-existing to resume.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def resolve_org_and_motions(registry: str, org: str | None) -> tuple[str, list[str]]:
  """Return (org_entity, motion_names) for the given registry."""
  import wandb

  api = wandb.Api()
  default = api.default_entity
  # Registries live under the org entity, which is usually "<entity>-org".
  candidates = [org] if org else [f"{default}-org", default]
  last_err: Exception | None = None
  for candidate in candidates:
    if candidate is None:
      continue
    project = f"{candidate}/wandb-registry-{registry}"
    try:
      # The artifact type need not equal the registry name (e.g. after a
      # registry rename the type keeps its original value), so discover the
      # type(s) present and list collections for each.
      types = [t.name for t in api.artifact_types(project=project)]
    except Exception as exc:  # noqa: BLE001 - try the next candidate.
      last_err = exc
      continue
    names: set[str] = set()
    for type_name in types:
      try:
        for coll in api.artifact_collections(project_name=project, type_name=type_name):
          names.add(coll.name)
      except Exception:  # noqa: BLE001 - skip types that fail to enumerate.
        continue
    if names:
      return candidate, sorted(names)
    last_err = RuntimeError(f"no collections found in {project} (types={types})")
  raise SystemExit(
    f"Could not read registry '{registry}' under {candidates}: {last_err}"
  )


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
    "--registry", required=True, help="W&B registry name to train from."
  )
  parser.add_argument("--task", default="Mjlab-Tracking-Flat-Unitree-G1")
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument(
    "--max-iterations", type=int, default=None, help="Override training iterations."
  )
  parser.add_argument(
    "--filter", default=None, help="Only train motions whose name matches this regex."
  )
  parser.add_argument(
    "--org", default=None, help="Org entity (auto-detected if omitted)."
  )
  parser.add_argument(
    "--skip-existing",
    action="store_true",
    help="Skip motions that already have a run dir under logs/rsl_rl/<registry>/.",
  )
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()

  org, motions = resolve_org_and_motions(args.registry, args.org)
  if args.filter:
    pat = re.compile(args.filter)
    motions = [m for m in motions if pat.search(m)]
  if not motions:
    sys.exit(f"No motions to train in registry '{args.registry}' (after filter).")

  exp_dir = Path("logs/rsl_rl") / args.registry
  if args.skip_existing:
    kept = []
    for m in motions:
      if exp_dir.exists() and any(p.name.endswith(m) for p in exp_dir.glob(f"*_{m}")):
        print(f"[skip] {m} (run dir already exists)")
      else:
        kept.append(m)
    motions = kept

  print(f"\nRegistry : {org}/wandb-registry-{args.registry}")
  print(f"Task     : {args.task}")
  print(f"Motions  : {len(motions)} -> logs/rsl_rl/{args.registry}/<time>_<motion>/")
  print("A full tracking run takes hours; this is sequential.\n")

  ok, failed = 0, []
  for i, motion in enumerate(motions, 1):
    registry_name = f"{org}/wandb-registry-{args.registry}/{motion}"
    cmd = [
      "uv",
      "run",
      "train",
      args.task,
      "--registry-name",
      registry_name,
      "--env.scene.num-envs",
      str(args.num_envs),
      "--agent.experiment-name",
      args.registry,
      "--agent.run-name",
      motion,
    ]
    if args.max_iterations is not None:
      cmd += ["--agent.max-iterations", str(args.max_iterations)]

    print(f"\n===== [{i}/{len(motions)}] training '{motion}' =====")
    if args.dry_run:
      print("  (dry-run)", " ".join(cmd))
      continue

    result = subprocess.run(cmd)
    if result.returncode == 0:
      ok += 1
      print(f"  DONE '{motion}' ({ok} ok / {len(failed)} fail)")
    else:
      failed.append(motion)
      print(f"  FAILED '{motion}' ({ok} ok / {len(failed)} fail)")

  print(f"\nAll done: {ok} ok / {len(failed)} fail")
  if failed:
    print("Failed:", ", ".join(failed))
    sys.exit(1)


if __name__ == "__main__":
  main()
