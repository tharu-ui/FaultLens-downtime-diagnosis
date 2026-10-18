#!/usr/bin/env python3
"""FaultLens Automation Runner.

Cross-platform runner providing parity with Makefile targets for environments
where `make` is unavailable (e.g., Windows PowerShell).

Usage:
    python scripts/run_all.py data [--count 300] [--out-dir data/sample_scenarios]
    python scripts/run_all.py test
    python scripts/run_all.py sanity
    python scripts/run_all.py tune
    python scripts/run_all.py all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add src to sys.path so modules can be imported directly
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Ensure subprocesses inherit PYTHONPATH
os.environ["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + os.environ.get("PYTHONPATH", "") if os.environ.get("PYTHONPATH") else "")


def run_data(count: int = 300, out_dir: str = "data/sample_scenarios") -> int:
    """Generate benchmark scenarios and train/val/test splits."""
    print(f"--> Generating {count} synthetic scenarios into '{out_dir}'...")
    from faultlens.generator import generate_benchmark_scenarios
    scenarios, splits = generate_benchmark_scenarios(
        num_scenarios=count,
        master_seed=42,
        out_dir=out_dir,
    )
    print(f"Generated {len(scenarios)} scenarios.")
    print(f"Splits: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    return 0


def run_test() -> int:
    """Run pytest suite with verbose output."""
    print("--> Running unit tests with pytest...")
    cmd = [sys.executable, "-m", "pytest", "-v", "tests/"]
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def run_sanity() -> int:
    """Run diagnoser sanity check on validation split."""
    print("--> Running baseline diagnoser sanity check...")
    from faultlens.diagnoser import run_sanity_check
    table = run_sanity_check()
    print("\n" + table.to_string(index=False) + "\n")
    return 0


def run_tune() -> int:
    """Tune policy thresholds on validation split."""
    print("--> Running policy threshold grid search on validation split...")
    from faultlens.policy import tune_policy_on_validation
    tuned = tune_policy_on_validation()
    print(f"Tuned Policy Parameters: tau_q={tuned['tau_q']:.3f}, tau_s={tuned['tau_s']:.3f}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="FaultLens Task Runner")
    parser.add_argument(
        "step",
        choices=["data", "test", "sanity", "tune", "all"],
        help="Step to execute",
    )
    parser.add_argument("--count", type=int, default=300, help="Scenario count for data step")
    parser.add_argument("--out-dir", type=str, default="data/sample_scenarios", help="Output directory")

    args = parser.parse_args()

    if args.step == "data":
        sys.exit(run_data(count=args.count, out_dir=args.out_dir))
    elif args.step == "test":
        sys.exit(run_test())
    elif args.step == "sanity":
        sys.exit(run_sanity())
    elif args.step == "tune":
        sys.exit(run_tune())
    elif args.step == "all":
        code = run_data(count=args.count, out_dir=args.out_dir)
        if code != 0:
            sys.exit(code)
        code = run_test()
        if code != 0:
            sys.exit(code)
        code = run_sanity()
        sys.exit(code)


if __name__ == "__main__":
    main()
