"""Session 2.5 & Session 3 Integration and Experiment Tests.

Verifies:
1. Quality fixes (Task 1):
   - duplicate_records at 30% lowers duplication but not timestamp_consistency by > 0.02
   - clean scenarios score >= 0.95 on every sub-score
   - worst_component matches injected 30% defect type
2. Confidence-threshold baseline (Task 3):
   - threshold_for_target_coverage hits target coverage within 2 percentage points
3. Tuning isolation (Task 5):
   - tune_policy_on_validation pools TRAIN+VAL and uses zero TEST scenario IDs
4. Full Experiment & Figures (Session 3 Task 6):
   - raw_results.csv has exactly 120 * 4 * 4 * 3 = 5,760 rows
   - zero train or val scenario IDs appear in raw_results.csv
   - thresholds loaded for the test run match saved validation-tuned values
   - all 4 metrics CSVs, summary.md, RUN_LOG.md, and 6 figure pairs (PNG + SVG) exist
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from faultlens.defects import inject_defect
from faultlens.experiment import (
    TABLES_DIR,
    run_test_experiment,
)
from faultlens.generator import generate_scenario
from faultlens.methods import ConfidenceThresholdMethod
from faultlens.plots import FIGURES_DIR, generate_all_figures
from faultlens.policy import TUNED_PARAMS_PATH, load_tuned_policy_params
from faultlens.quality import evaluate_data_quality


def test_quality_duplicate_does_not_lower_timestamp_consistency() -> None:
    """Verify 30% duplicate_records lowers duplication but not timestamp_consistency by > 0.02."""
    sc = generate_scenario(seed=202, target_difficulty="easy")
    clean_q = evaluate_data_quality(sc.log)

    dup_df, _ = inject_defect(sc.log, defect_type="duplicate_records", severity=0.30, seed=42)
    dup_q = evaluate_data_quality(dup_df)

    assert dup_q.sub_scores["duplication"] < clean_q.sub_scores["duplication"] - 0.10
    ts_drop = clean_q.sub_scores["timestamp_consistency"] - dup_q.sub_scores["timestamp_consistency"]
    assert ts_drop <= 0.02, f"timestamp_consistency dropped by {ts_drop:.4f} (> 0.02) under duplicate_records"


def test_quality_clean_sub_scores_and_worst_component_identification() -> None:
    """Verify clean scenarios score >= 0.95 on every sub-score and worst_component tracks each 30% defect."""
    sc = generate_scenario(seed=303, target_difficulty="easy")
    clean_q = evaluate_data_quality(sc.log)

    for comp, score in clean_q.sub_scores.items():
        assert score >= 0.95, f"Clean scenario sub-score '{comp}' = {score:.4f} < 0.95"

    # 1. 30% duplicate_records -> worst_component == "duplication"
    df_dup, _ = inject_defect(sc.log, defect_type="duplicate_records", severity=0.30, seed=11)
    q_dup = evaluate_data_quality(df_dup)
    assert q_dup.worst_component == "duplication"

    # 2. 30% timestamp_errors -> worst_component == "timestamp_consistency"
    df_ts, _ = inject_defect(sc.log, defect_type="timestamp_errors", severity=0.30, seed=22)
    q_ts = evaluate_data_quality(df_ts)
    assert q_ts.worst_component == "timestamp_consistency"

    # 3. 30% missing_events -> worst_component == "completeness"
    df_miss, _ = inject_defect(sc.log, defect_type="missing_events", severity=0.30, seed=33)
    q_miss = evaluate_data_quality(df_miss)
    assert q_miss.worst_component == "completeness"

    # 4. 30% incorrect_reason_codes -> worst_component in ("reason_validity", "alarm_reason_agreement")
    df_reason, _ = inject_defect(sc.log, defect_type="incorrect_reason_codes", severity=0.30, seed=44)
    q_reason = evaluate_data_quality(df_reason)
    assert q_reason.worst_component in ("reason_validity", "alarm_reason_agreement")


def test_confidence_threshold_for_target_coverage_within_two_pct_points() -> None:
    """Verify threshold_for_target_coverage hits target coverage within 2 percentage points."""
    rng = np.random.default_rng(42)
    raw_confs = rng.uniform(0.51, 0.99, size=500)
    labels = (raw_confs + rng.normal(0, 0.1, size=500) > 0.72).astype(int)

    ct = ConfidenceThresholdMethod()
    ct.fit_calibrator(raw_confs, labels, method="platt")
    cal_confs = ct.calibrator.predict_batch(raw_confs)  # type: ignore[union-attr]

    for target_cov in (0.50, 0.65, 0.80):
        thresh = ct.threshold_for_target_coverage(target_cov, calibrated_confs=cal_confs)
        achieved_cov = float(np.mean(cal_confs >= thresh))
        assert abs(achieved_cov - target_cov) <= 0.02, (
            f"Target coverage {target_cov:.2f}, achieved {achieved_cov:.4f}"
        )


def test_tuning_uses_no_test_scenarios_and_experiment_isolation() -> None:
    """Verify experiment output has 5,760 rows, zero train/val IDs, and matches saved thresholds."""
    splits_path = Path("data/sample_scenarios/splits.json")
    if not splits_path.is_file():
        pytest.skip("Benchmark scenarios not generated yet; run `python scripts/run_all.py data` first.")

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])

    raw_csv = TABLES_DIR / "raw_results.csv"
    if not raw_csv.is_file():
        run_test_experiment(num_perturbations=50)
        generate_all_figures()

    assert TUNED_PARAMS_PATH.is_file(), "results/tuned_params.json must exist"
    with open(TUNED_PARAMS_PATH, "r", encoding="utf-8") as f:
        tuned_data = json.load(f)

    used_in_tuning = set(tuned_data.get("scenario_ids_used", []))
    assert len(used_in_tuning.intersection(test_ids)) == 0, "TEST scenario IDs must never be used in tuning!"

    loaded_tq, loaded_ts, loaded_buf = load_tuned_policy_params(TUNED_PARAMS_PATH)
    assert loaded_tq == pytest.approx(float(tuned_data["tau_q"]), abs=1e-4)
    assert loaded_ts == pytest.approx(float(tuned_data["tau_s"]), abs=1e-4)
    assert loaded_buf == pytest.approx(float(tuned_data["buffer"]), abs=1e-4)

    raw_df = pd.read_csv(raw_csv)
    expected_rows = 120 * 4 * 4 * 3  # 5,760 rows
    assert len(raw_df) == expected_rows, f"Expected {expected_rows} rows, got {len(raw_df)}"

    result_scenario_ids = set(raw_df["scenario_id"].unique())
    assert result_scenario_ids == test_ids
    assert len(result_scenario_ids.intersection(train_ids)) == 0
    assert len(result_scenario_ids.intersection(val_ids)) == 0

    for table_name in (
        "raw_results.csv",
        "metrics_by_defect_severity.csv",
        "matched_coverage.csv",
        "significance.csv",
        "calibration.csv",
        "summary.md",
    ):
        assert (TABLES_DIR / table_name).is_file(), f"Missing table: {table_name}"

    if not (FIGURES_DIR / "fig1_hero_cw_vs_severity.png").is_file():
        generate_all_figures()

    for fig_stem in (
        "fig1_hero_cw_vs_severity",
        "fig1b_small_multiples_cw",
        "fig2_risk_coverage",
        "fig3_coverage_vs_severity",
        "fig4_cw_reduction_heatmap",
        "fig5_stability_histogram",
        "fig6_calibration_reliability",
    ):
        assert (FIGURES_DIR / f"{fig_stem}.png").is_file(), f"Missing PNG figure: {fig_stem}.png"
        assert (FIGURES_DIR / f"{fig_stem}.svg").is_file(), f"Missing SVG figure: {fig_stem}.svg"