#!/usr/bin/env python3
"""FaultLens Automation Runner.

Cross-platform runner providing parity with Makefile targets for environments
where `make` is unavailable (e.g., Windows PowerShell).

Usage:
    python scripts/run_all.py data [--count 300] [--out-dir data/sample_scenarios]
    python scripts/run_all.py baseline
    python scripts/run_all.py test
    python scripts/run_all.py sanity
    python scripts/run_all.py tune
    python scripts/run_all.py experiments
    python scripts/run_all.py plots
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
os.environ["PYTHONPATH"] = str(SRC_DIR) + (
    os.pathsep + os.environ.get("PYTHONPATH", "") if os.environ.get("PYTHONPATH") else ""
)


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


def run_baseline(out_dir: str = "data/sample_scenarios") -> int:
    """Compute clean quality baseline from TRAIN split only."""
    print("--> Computing clean data quality baseline on TRAIN split...")
    from faultlens.quality import compute_quality_baseline

    payload = compute_quality_baseline(scenarios_dir=out_dir)
    print(f"Saved quality baseline to results/quality_baseline.json (N={payload['num_scenarios']} train scenarios)")
    return 0


def run_test() -> int:
    """Run pytest suite with verbose output."""
    print("--> Running unit tests with pytest...")
    cmd = [sys.executable, "-m", "pytest", "-v", "tests/"]
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def run_sanity() -> int:
    """Run diagnoser sanity check and Session 2.5 validation table on validation split."""
    print("--> Running baseline diagnoser sanity check...")
    from faultlens.diagnoser import run_sanity_check
    from faultlens.experiment import run_session2_5_validation_table

    table = run_sanity_check()
    print("\n" + table.to_string(index=False) + "\n")
    run_session2_5_validation_table()
    return 0


def run_tune() -> int:
    """Tune policy thresholds on pooled TRAIN+VALIDATION splits."""
    print("--> Running policy threshold grid search on TRAIN+VALIDATION splits...")
    from faultlens.policy import tune_policy_on_validation

    tuned = tune_policy_on_validation()
    print(
        f"Tuned Policy Parameters: gate_mode={tuned.get('gate_mode', 'weighted')}, "
        f"tau_q={tuned['tau_q']:.3f}, delta={tuned.get('delta', 0.15):.3f}, tau_s={tuned['tau_s']:.3f}"
    )
    return 0


def run_experiments() -> int:
    """Run full Session 3 benchmark experiment on held-out TEST split and generate figures."""
    print("--> Running full benchmark experiment on held-out TEST split (N=120)...")
    from faultlens.experiment import run_test_experiment
    from faultlens.plots import generate_all_figures

    tables = run_test_experiment()
    print("\n--- Headline Matched-Coverage Table (Test Split, N=120) ---")
    print(tables["matched_coverage"].to_string(index=False))

    created_figs = generate_all_figures()
    print(f"\nGenerated {len(created_figs)} figure files in results/figures/:")
    for fig_p in created_figs:
        print(f"  - {fig_p}")
    return 0


def run_plots() -> int:
    """Generate all benchmark figures from existing CSV tables."""
    print("--> Generating benchmark figures into results/figures/...")
    from faultlens.plots import generate_all_figures

    created_figs = generate_all_figures()
    for fig_p in created_figs:
        print(f"  - {fig_p}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="FaultLens Task Runner")
    parser.add_argument(
        "step",
        choices=["data", "baseline", "test", "sanity", "tune", "experiments", "plots", "all"],
        help="Step to execute",
    )
    parser.add_argument("--count", type=int, default=300, help="Scenario count for data step")
    parser.add_argument("--out-dir", type=str, default="data/sample_scenarios", help="Output directory")

    args = parser.parse_args()

    if args.step == "data":
        sys.exit(run_data(count=args.count, out_dir=args.out_dir))
    elif args.step == "baseline":
        sys.exit(run_baseline(out_dir=args.out_dir))
    elif args.step == "test":
        sys.exit(run_test())
    elif args.step == "sanity":
        sys.exit(run_sanity())
    elif args.step == "tune":
        sys.exit(run_tune())
    elif args.step == "experiments":
        sys.exit(run_experiments())
    elif args.step == "plots":
        sys.exit(run_plots())
    elif args.step == "all":
        for fn in (
            lambda: run_data(count=args.count, out_dir=args.out_dir),
            lambda: run_baseline(out_dir=args.out_dir),
            run_tune,
            run_sanity,
            run_experiments,
            run_test,
        ):
            code = fn()
            if code != 0:
                sys.exit(code)
        sys.exit(0)


if __name__ == "__main__":
    main()