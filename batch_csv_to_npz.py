#!/usr/bin/env python3
"""Batch-convert a folder of retargeted-motion CSVs to NPZ via mjlab.scripts.csv_to_npz.

Searches a directory (recursively) for ``*.csv`` and runs the csv_to_npz CLI once
per file, naming each output after the CSV. Failures are isolated (one bad clip
does not stop the batch) and summarized at the end. With ``--render`` it sets
MUJOCO_GL=egl and produces a video per motion.

Uses only the standard library; the mjlab environment is provided by the inner
``uv run`` call, so run it however you like:

    python3 batch_csv_to_npz.py humanoid_motion_datasets/g1/lafan1/colmo
    python3 batch_csv_to_npz.py humanoid_motion_datasets/g1/lafan1/colmo --render
    python3 batch_csv_to_npz.py <dir> --robot kapex --prefix kapex_
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
    "input_dir", type=Path, help="Folder with .csv motions (searched recursively)."
  )
  parser.add_argument("--input-fps", type=float, default=30.0)
  parser.add_argument("--output-fps", type=float, default=50.0)
  parser.add_argument("--robot", default="g1", choices=["g1", "kapex"])
  parser.add_argument(
    "--registry",
    default="motions",
    help="W&B registry to link motions into (target wandb-registry-<name>).",
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Render a video per motion (sets MUJOCO_GL=egl).",
  )
  parser.add_argument(
    "--prefix",
    default="",
    help="Optional prefix for each output-name (e.g. 'g1_').",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print the commands without running them.",
  )
  args = parser.parse_args()

  csvs = sorted(args.input_dir.rglob("*.csv"))
  if not csvs:
    sys.exit(f"No .csv files found under {args.input_dir}")
  print(f"Found {len(csvs)} CSV files under {args.input_dir}\n")

  env = dict(os.environ)
  if args.render:
    env["MUJOCO_GL"] = "egl"

  ok, failed = 0, []
  for i, csv in enumerate(csvs, 1):
    name = args.prefix + csv.stem
    cmd = [
      "uv",
      "run",
      "-m",
      "mjlab.scripts.csv_to_npz",
      "--input-file",
      str(csv),
      "--output-name",
      name,
      "--input-fps",
      str(args.input_fps),
      "--output-fps",
      str(args.output_fps),
      "--robot",
      args.robot,
      "--registry",
      args.registry,
    ]
    if args.render:
      cmd += ["--render", "True"]

    print(f"[{i}/{len(csvs)}] {csv.name} -> {name}")
    if args.dry_run:
      print("  (dry-run)", " ".join(cmd))
      continue

    result = subprocess.run(cmd, env=env)
    if result.returncode == 0:
      ok += 1
      print(f"  OK ({ok} ok / {len(failed)} fail)\n")
    else:
      failed.append(name)
      print(f"  FAIL ({ok} ok / {len(failed)} fail)\n")

  print(f"Done: {ok} ok / {len(failed)} fail")
  if failed:
    print("Failed:", ", ".join(failed))
    sys.exit(1)


if __name__ == "__main__":
  main()
