#!/usr/bin/env python3
"""Train one multi-clip tracking policy per motion cluster.

Reads a cluster-assignment CSV (columns: cluster, motion, action), downloads
each cluster's motions from a W&B registry, and runs one training per cluster
with all of its clips passed via ``--env.commands.motion.motion-files``. Each
episode samples one clip and stays within it (see MotionCommand). Runs are
grouped as::

    logs/rsl_rl/<registry>_clusters/<time>_cluster<k>_<action>/

Run inside the mjlab env (it queries W&B):

    uv run python batch_train_clusters.py --registry colmo_g1
    uv run python batch_train_clusters.py --registry colmo_g1 --clusters 2 8
    uv run python batch_train_clusters.py --registry colmo_g1 --max-iterations 15000 --dry-run

A cluster policy is a harder problem than a single clip, so homogeneous clusters
(all run / all dance) converge best; very mixed clusters may need more iters.
Training is sequential (one GPU) -- use --clusters / --max-iterations to scope.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter, OrderedDict
from pathlib import Path

# Tracking task id per robot; --robot selects one of these.
ROBOT_TASKS = {
  "g1": "Mjlab-Tracking-Flat-Unitree-G1",
  "kapex": "Mjlab-Tracking-Flat-Kapex",
}


def read_clusters(csv_path: Path) -> "OrderedDict[int, list[tuple[str, str]]]":
  clusters: OrderedDict[int, list[tuple[str, str]]] = OrderedDict()
  with open(csv_path) as f:
    for row in csv.DictReader(f):
      clusters.setdefault(int(row["cluster"]), []).append(
        (row["motion"], row["action"])
      )
  return clusters


def resolve_org(api, registry: str, org: str | None) -> str:
  """Find the org entity that hosts wandb-registry-<registry>."""
  default = api.default_entity
  candidates = [org] if org else [f"{default}-org", default]
  for candidate in candidates:
    if candidate is None:
      continue
    try:
      list(api.artifact_types(project=f"{candidate}/wandb-registry-{registry}"))
      return candidate
    except Exception:  # noqa: BLE001 - try the next candidate.
      continue
  raise SystemExit(
    f"Registry '{registry}' not found under {candidates}. Check the name/org."
  )


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--registry", default="colmo_g1", help="W&B registry name.")
  parser.add_argument(
    "--clusters-csv",
    type=Path,
    default=Path(__file__).resolve().parent / "motion_clusters_assignments.csv",
    help="Cluster-assignment CSV (default: alongside this script, so it works "
    "regardless of the current working directory or where the repo is deployed).",
  )
  parser.add_argument(
    "--robot",
    required=True,
    choices=sorted(ROBOT_TASKS),
    help="Robot to train (required; no default). Selects the tracking task.",
  )
  parser.add_argument(
    "--task",
    default=None,
    help="Override the tracking task id (default: derived from --robot).",
  )
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--max-iterations", type=int, default=None)
  parser.add_argument(
    "--clusters",
    type=int,
    nargs="*",
    default=None,
    help="Only train these cluster ids (default: all).",
  )
  parser.add_argument("--org", default=None, help="Org entity (auto-detected).")
  parser.add_argument(
    "--skip-existing",
    action="store_true",
    help="Skip clusters that already have a run dir.",
  )
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()
  task = args.task or ROBOT_TASKS[args.robot]

  import wandb

  api = wandb.Api()
  org = resolve_org(api, args.registry, args.org)

  clusters = read_clusters(args.clusters_csv)
  ids = sorted(clusters)
  if args.clusters:
    wanted = set(args.clusters)
    ids = [k for k in ids if k in wanted]
  if not ids:
    sys.exit("No clusters selected.")

  experiment = f"{args.registry}_clusters"
  exp_dir = Path("logs/rsl_rl") / experiment
  total_clips = sum(len(clusters[k]) for k in ids)
  print(f"Registry : {org}/wandb-registry-{args.registry}")
  print(f"Clusters : {ids}  ({total_clips} clips total)")
  print(f"Output   : logs/rsl_rl/{experiment}/<time>_cluster<k>_<action>/")
  print("Sequential (one GPU); a cluster policy takes hours.\n")

  ok, failed = 0, []
  for k in ids:
    entries = clusters[k]
    motions = [m for m, _ in entries]
    label = Counter(a for _, a in entries).most_common(1)[0][0]
    run_name = f"cluster{k}_{label}"

    if args.skip_existing and exp_dir.exists() and any(exp_dir.glob(f"*_{run_name}")):
      print(f"[skip] {run_name} (run dir exists)")
      continue

    print(f"\n===== cluster {k} -> {run_name}: {len(motions)} clips =====")
    paths: list[str] = []
    for motion in motions:
      try:
        artifact = api.artifact(f"{org}/wandb-registry-{args.registry}/{motion}:latest")
        paths.append(str(Path(artifact.download()) / "motion.npz"))
      except Exception as exc:  # noqa: BLE001 - skip a missing motion.
        print(f"  [warn] skip '{motion}': {type(exc).__name__} {str(exc)[:60]}")
    if not paths:
      failed.append(run_name)
      print(f"  FAIL {run_name}: no motions downloaded")
      continue

    cmd = [
      "uv",
      "run",
      "train",
      task,
      "--env.commands.motion.motion-files",
      # mjlab's TYRO_FLAGS parse collections as Python literals
      # (`--flag (a, b, c)`), so pass the file list as one literal-tuple string
      # rather than space-separated values.
      repr(tuple(paths)),
      "--env.scene.num-envs",
      str(args.num_envs),
      "--agent.experiment-name",
      experiment,
      "--agent.run-name",
      run_name,
    ]
    if args.max_iterations is not None:
      cmd += ["--agent.max-iterations", str(args.max_iterations)]

    if args.dry_run:
      print(f"  (dry-run) {len(paths)} clips: {[Path(p).parent.name for p in paths]}")
      print(
        "   ", " ".join(cmd[:4] + ["--env.commands.motion.motion-files ..."] + cmd[-6:])
      )
      continue

    result = subprocess.run(cmd)
    if result.returncode == 0:
      ok += 1
      print(f"  DONE {run_name} ({ok} ok / {len(failed)} fail)")
    else:
      failed.append(run_name)
      print(f"  FAILED {run_name} ({ok} ok / {len(failed)} fail)")

  print(f"\nAll done: {ok} ok / {len(failed)} fail")
  if failed:
    print("Failed:", ", ".join(failed))
    sys.exit(1)


if __name__ == "__main__":
  main()
