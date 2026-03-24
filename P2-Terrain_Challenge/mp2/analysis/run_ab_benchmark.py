#!/usr/bin/env python3
"""Single-command A/B runner for OLD vs NEW policies.

This script:
  1) Generates the 7 standard open-loop rollout CSVs for OLD and NEW policies
     by calling misc/test_policy_openloop.py with explicit --output paths.
  2) Runs analysis/ab_test_policies.py on the generated directories.

Example:
python3 P2-Terrain_Challenge/mp2/analysis/run_ab_benchmark.py \
  --old-policy /path/to/old.pt \
  --new-policy /path/to/new.pt \
  --out-root /tmp/mp2_ab
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

COMMANDS: Dict[str, Tuple[float, float, float]] = {
    "openloop_scheduled_forward_0.15.csv": (0.15, 0.0, 0.0),
    "openloop_scheduled_backward_0.15.csv": (-0.15, 0.0, 0.0),
    "openloop_scheduled_left_0.15.csv": (0.0, 0.15, 0.0),
    "openloop_scheduled_right_0.15.csv": (0.0, -0.15, 0.0),
    "openloop_scheduled_yaw_left_0.2.csv": (0.0, 0.0, 0.2),
    "openloop_scheduled_yaw_right_0.2.csv": (0.0, 0.0, -0.2),
    "openloop_scheduled_stop.csv": (0.0, 0.0, 0.0),
}


def run_cmd(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def generate_suite(
    python_exe: str,
    rollout_script: Path,
    policy_path: Path,
    out_dir: Path,
    duration: float,
    no_delay: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for file_name, (vx, vy, vyaw) in COMMANDS.items():
        out_csv = out_dir / file_name
        cmd = [
            python_exe,
            str(rollout_script),
            "--policy",
            str(policy_path),
            "--duration",
            str(duration),
            "--cmd_x",
            str(vx),
            "--cmd_y",
            str(vy),
            "--cmd_yaw",
            str(vyaw),
            "--output",
            str(out_csv),
        ]
        if no_delay:
            cmd.append("--no-delay")
        run_cmd(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-command OLD vs NEW benchmark runner")
    ap.add_argument("--old-policy", type=Path, required=True)
    ap.add_argument("--new-policy", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True, help="Output root dir for old/new CSV suites")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--python", type=str, default=sys.executable, help="Python executable")
    ap.add_argument("--rollout-script", type=Path, default=Path("P2-Terrain_Challenge/mp2/misc/test_policy_openloop.py"))
    ap.add_argument("--scorer-script", type=Path, default=Path("P2-Terrain_Challenge/mp2/analysis/ab_test_policies.py"))
    ap.add_argument("--no-delay", action="store_true", help="Disable delayed action buffer in rollout")
    args = ap.parse_args()

    old_dir = args.out_root / "old"
    new_dir = args.out_root / "new"

    print("\n=== Generating OLD suite ===")
    generate_suite(args.python, args.rollout_script, args.old_policy, old_dir, args.duration, args.no_delay)

    print("\n=== Generating NEW suite ===")
    generate_suite(args.python, args.rollout_script, args.new_policy, new_dir, args.duration, args.no_delay)

    print("\n=== Running scorer ===")
    run_cmd([
        args.python,
        str(args.scorer_script),
        "--old",
        str(old_dir),
        "--new",
        str(new_dir),
        "--settle",
        str(args.settle),
    ])

    print("\nDone. Suites: ")
    print(f"  OLD: {old_dir}")
    print(f"  NEW: {new_dir}")


if __name__ == "__main__":
    main()
