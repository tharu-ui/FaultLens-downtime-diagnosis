#!/usr/bin/env python3
"""FaultLens Automation Runner.

Cross-platform runner providing parity with Makefile targets for environments
where `make` is unavailable (e.g., Windows PowerShell).

Usage:
    python scripts/run_all.py data [--count 300] [--out-dir data/sample_scenarios]
    python scripts/run_all.py extra_train [--count 120]
    python scripts/run_all.py fresh_test [--count 200]
    python scripts/run_all.py baseline
    python scripts/run_all.py test
    python scripts/run_all.py sanity
    python scripts/run_all.py tune
    python scripts/run_all.py train_reliability
    python scripts/run_all.py val_table_3_5
    python scripts/run_all.py experiments
    python scripts/run_all.py plots
    python scripts/run_all.py final
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


def run_extra_train(count: int = 120, out_dir: str = "data/sample_scenarios/extra_train") -> int:
    """Generate extra synthetic TRAIN scenarios with seed 1337, saved separately from splits.json."""
    print(f"--> Generating {count} extra synthetic TRAIN scenarios into '{out_dir}'...")
    from faultlens.reliability import generate_extra_train_scenarios

    scenarios, manifest = generate_extra_train_scenarios(
        num_scenarios=count,
        master_seed=1337,
        out_dir=out_dir,
    )
    print(f"Generated {len(scenarios)} extra train scenarios (seed={manifest['master_seed']}).")
    return 0


def run_fresh_test(count: int = 200, out_dir: str = "data/sample_scenarios/fresh_test") -> int:
    """Generate 200 fresh held-out synthetic scenarios with seed 4242 into data/sample_scenarios/fresh_test/."""
    print(f"--> Generating {count} fresh held-out synthetic scenarios (seed 4242) into '{out_dir}'...")
    from faultlens.reliability import generate_fresh_test_scenarios

    scenarios, manifest = generate_fresh_test_scenarios(
        num_scenarios=count,
        master_seed=4242,
        out_dir=out_dir,
    )
    print(
        f"Generated {len(scenarios)} fresh held-out scenarios with manifest at "
        f"'{Path(out_dir) / 'manifest.json'}' (seed={manifest['master_seed']})."
    )
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
    """Run diagnoser sanity check and Session 2.5 + 3.5 validation tables on validation split."""
    print("--> Running baseline diagnoser sanity check...")
    from faultlens.diagnoser import run_sanity_check
    from faultlens.experiment import run_session2_5_validation_table
    from faultlens.reliability import run_session3_5_validation_table

    table = run_sanity_check()
    print("\n" + table.to_string(index=False) + "\n")
    run_session2_5_validation_table()
    run_session3_5_validation_table()
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


def run_train_reliability() -> int:
    """Train learned reliability models on TRAIN, calibrate and choose ONE threshold on VALIDATION."""
    print("--> Training learned reliability models (TRAIN fit, VALIDATION calibration & threshold)...")
    from faultlens.reliability import train_and_save_reliability_models

    payload = train_and_save_reliability_models()
    full_val = payload["validation_summary"]["full"]
    conf_val = payload["validation_summary"]["confidence_only"]
    print(
        f"Saved models to results/reliability_models.json | "
        f"LearnedFaultLens val_thresh={full_val['threshold']:.4f}, cov={full_val['coverage']:.4f}, cw={full_val['confidently_wrong_rate']:.4f} | "
        f"ConfidenceOnlyLearned val_thresh={conf_val['threshold']:.4f}, cov={conf_val['coverage']:.4f}, cw={conf_val['confidently_wrong_rate']:.4f}"
    )
    return 0


def run_val_table_3_5() -> int:
    """Generate Session 3.5 validation sanity table on VALIDATION split only."""
    print("--> Running Session 3.5 validation sanity table (VALIDATION split only)...")
    from faultlens.reliability import run_session3_5_validation_table

    run_session3_5_validation_table()
    return 0


def run_experiments() -> int:
    """Run full Session 3 benchmark experiment on exploratory TEST split and generate figures."""
    print("--> Running benchmark experiment on exploratory TEST split (N=120)...")
    from faultlens.experiment import run_test_experiment
    from faultlens.plots import generate_all_figures

    tables = run_test_experiment()
    print("\n--- Headline Matched-Coverage Table (Exploratory Test Split, N=120) ---")
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


def run_final() -> int:
    """Run final evaluation ONCE on fresh_test (seed 4242) using frozen saved models and lock file."""
    print("--> Running one-time final evaluation on fresh held-out set (seed 4242)...")
    from faultlens.reliability import run_final_fresh_test_experiment

    df = run_final_fresh_test_experiment()
    print(f"Final evaluation complete ({len(df)} rows). Lock file created at results/final_run.lock.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="FaultLens Task Runner")
    parser.add_argument(
        "step",
        choices=[
            "data",
            "extra_train",
            "fresh_test",
            "baseline",
            "test",
            "sanity",
            "tune",
            "train_reliability",
            "val_table_3_5",
            "experiments",
            "plots",
            "final",
            "all",
        ],
        help="Step to execute",
    )
    parser.add_argument("--count", type=int, default=None, help="Scenario count for generation steps")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    if args.step == "data":
        cnt = args.count if args.count is not None else 300
        od = args.out_dir if args.out_dir is not None else "data/sample_scenarios"
        sys.exit(run_data(count=cnt, out_dir=od))
    elif args.step == "extra_train":
        cnt = args.count if args.count is not None else 120
        od = args.out_dir if args.out_dir is not None else "data/sample_scenarios/extra_train"
        sys.exit(run_extra_train(count=cnt, out_dir=od))
    elif args.step == "fresh_test":
        cnt = args.count if args.count is not None else 200
        od = args.out_dir if args.out_dir is not None else "data/sample_scenarios/fresh_test"
        sys.exit(run_fresh_test(count=cnt, out_dir=od))
    elif args.step == "baseline":
        od = args.out_dir if args.out_dir is not None else "data/sample_scenarios"
        sys.exit(run_baseline(out_dir=od))
    elif args.step == "test":
        sys.exit(run_test())
    elif args.step == "sanity":
        sys.exit(run_sanity())
    elif args.step == "tune":
        sys.exit(run_tune())
    elif args.step == "train_reliability":
        sys.exit(run_train_reliability())
    elif args.step == "val_table_3_5":
        sys.exit(run_val_table_3_5())
    elif args.step == "experiments":
        sys.exit(run_experiments())
    elif args.step == "plots":
        sys.exit(run_plots())
    elif args.step == "final":
        sys.exit(run_final())
    elif args.step == "all":
        # Note: `all` does NOT run `final` or touch the exploratory test split; runs dev/val pipeline + tests
        for fn in (
            lambda: run_data(count=args.count or 300, out_dir=args.out_dir or "data/sample_scenarios"),
            lambda: run_baseline(out_dir=args.out_dir or "data/sample_scenarios"),
            run_tune,
            run_train_reliability,
            run_val_table_3_5,
            run_test,
        ):
            code = fn()
            if code != 0:
                sys.exit(code)
        sys.exit(0)


if __name__ == "__main__":
    main()
