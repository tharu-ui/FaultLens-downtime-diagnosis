"""Session 2.5, Session 3 & Session 3.5 Integration and Experiment Tests.

Verifies:
1. Quality fixes (Session 2.5 Task 1):
   - duplicate_records at 30% lowers duplication but not timestamp_consistency by > 0.02
   - clean scenarios score >= 0.95 on every sub-score
   - worst_component matches injected 30% defect type
2. Confidence-threshold baseline (Session 2.5 Task 3):
   - threshold_for_target_coverage hits target coverage within 2 percentage points
3. Tuning isolation & Session 3 artifacts (Session 3 Task 6):
   - tune_policy_on_validation pools TRAIN+VAL and uses zero TEST scenario IDs
   - raw_results.csv has 5,760 rows, zero train/val scenario IDs, and 6 figure pairs exist
4. Session 3.5 tests (Task 9a - 9h):
   a. repair_log does not alter true_cause or scenario.truth
   b. reliability features have no NaNs
   c. model fitting uses only TRAIN scenarios (zero val/test scenarios)
   d. calibration and the single threshold are tuned strictly on VALIDATION
   e. matched coverage is within 2 percentage points
   f. GroupKFold cross-validation folds never split a scenario across train and score
   g. final command refuses to rerun when results/final_run.lock exists
   h. final command fails when saved reliability model is missing
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
from faultlens.generator import Scenario, generate_scenario
from faultlens.methods import (
    ConfidenceOnlyLearnedMethod,
    ConfidenceThresholdMethod,
    LearnedFaultLensMethod,
    get_available_methods,
)
from faultlens.plots import FIGURES_DIR, generate_all_figures
from faultlens.policy import TUNED_PARAMS_PATH, load_tuned_policy_params
from faultlens.quality import evaluate_data_quality
from faultlens.reliability import (
    FULL_FEATURE_COLUMNS,
    LearnedReliabilityModel,
    build_feature_table,
    compute_attribution_accuracy,
    extract_reliability_features,
    is_defect_attributed,
    run_final_fresh_test_experiment,
    run_grouped_cross_validation,
    train_and_save_reliability_models,
)
from faultlens.repair import repair_log


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


# =====================================================================
# Session 3.5 Tests (Tasks 9a - 9h)
# =====================================================================


def test_repair_does_not_alter_true_cause_or_truth() -> None:
    """Task 9a: Verify repair_log never alters true_cause or scenario.truth across all defects."""
    sc = generate_scenario(seed=501, scenario_id="sc_repair_test", target_difficulty="close")
    orig_truth = sc.truth
    orig_true_causes = sc.log["true_cause"].tolist()

    for defect in ("duplicate_records", "missing_events", "incorrect_reason_codes", "timestamp_errors"):
        def_df, _ = inject_defect(sc.log, defect_type=defect, severity=0.30, seed=77)
        true_causes_before = def_df["true_cause"].tolist()

        rep = repair_log(def_df)
        assert sc.truth == orig_truth, "Scenario truth tuple was mutated!"
        assert def_df["true_cause"].tolist() == true_causes_before, "Input DataFrame true_cause was mutated!"
        assert set(rep.repaired_df["true_cause"].unique()).issubset(set(orig_true_causes))
        if defect != "duplicate_records":
            # Non-duplicate repairs preserve exact row-by-row true_cause sequence
            assert rep.repaired_df["true_cause"].tolist() == true_causes_before
        else:
            assert rep.merged_duplicates_count > 0


def test_reliability_features_have_no_nans() -> None:
    """Task 9b: Verify extracted reliability feature dicts and feature tables contain zero NaNs."""
    scs = [
        generate_scenario(seed=601, scenario_id="sc_feat_1"),
        generate_scenario(seed=602, scenario_id="sc_feat_2"),
    ]
    conditions = [
        ("clean", None, 0.0),
        ("missing_events", "missing_events", 0.30),
        ("incorrect_reason_codes", "incorrect_reason_codes", 0.30),
        ("timestamp_errors", "timestamp_errors", 0.30),
        ("duplicate_records", "duplicate_records", 0.30),
    ]
    feat_df = build_feature_table(scs, conditions=conditions, num_perturbations=25, seed=42)
    assert not feat_df[FULL_FEATURE_COLUMNS].isna().any().any(), "Feature table contains NaN values!"
    for col in FULL_FEATURE_COLUMNS:
        arr = feat_df[col].to_numpy(dtype=float)
        assert np.all(np.isfinite(arr)), f"Column {col} has non-finite values!"


def test_training_uses_only_train_scenarios_and_threshold_tuned_on_validation(tmp_path: Path) -> None:
    """Tasks 9c & 9d: Verify model fitting uses only TRAIN scenarios and threshold is tuned on VALIDATION."""
    scen_dir = tmp_path / "scenarios"
    train_scs = [generate_scenario(seed=700 + i, scenario_id=f"train_{i:02d}") for i in range(6)]
    val_scs = [generate_scenario(seed=750 + i, scenario_id=f"val_{i:02d}") for i in range(4)]
    test_scs = [generate_scenario(seed=800 + i, scenario_id=f"test_{i:02d}") for i in range(4)]

    for sc in train_scs + val_scs + test_scs:
        sc.save(scen_dir)

    splits = {
        "train": [s.scenario_id for s in train_scs],
        "val": [s.scenario_id for s in val_scs],
        "test": [s.scenario_id for s in test_scs],
    }
    with open(scen_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f)

    out_model_path = tmp_path / "reliability_models.json"
    payload = train_and_save_reliability_models(
        scenarios_dir=scen_dir,
        extra_train_dir=None,
        out_path=out_model_path,
        num_perturbations=20,
        target_coverage=0.65,
        seed=42,
    )

    assert payload["train_split_used"] == "train"
    assert payload["calibration_split_used"] == "val"
    assert payload["threshold_split_used"] == "val"

    train_used = set(payload["train_scenario_ids"])
    val_used = set(payload["val_scenario_ids"])
    test_set = set(splits["test"])

    assert train_used == set(splits["train"])
    assert val_used == set(splits["val"])
    assert train_used.isdisjoint(val_used)
    assert train_used.isdisjoint(test_set)
    assert val_used.isdisjoint(test_set)


def test_learned_matched_coverage_within_two_points() -> None:
    """Task 9e: Verify LearnedReliabilityModel threshold matching hits target coverage within 2 points."""
    rng = np.random.default_rng(123)
    n = 400
    df_synth = pd.DataFrame({col: rng.normal(0, 1, size=n) for col in FULL_FEATURE_COLUMNS})
    df_synth["raw_confidence"] = rng.uniform(0.51, 0.99, size=n)
    logits = 1.8 * df_synth["raw_confidence"] + 0.8 * df_synth["top_stability"] - 1.2
    probs_true = 1.0 / (1.0 + np.exp(-logits))
    df_synth["is_correct"] = (rng.uniform(0, 1, size=n) < probs_true).astype(int)

    train_part = df_synth.iloc[:200].reset_index(drop=True)
    val_part = df_synth.iloc[200:].reset_index(drop=True)

    for mode in ("full", "confidence_only"):
        model = LearnedReliabilityModel(feature_mode=mode)
        model.fit_on_train(train_part)
        for target_cov in (0.50, 0.65, 0.80):
            thresh = model.calibrate_and_select_threshold_on_validation(val_part, target_coverage=target_cov)
            cal_probs = model.predict_calibrated_prob_batch(val_part)
            achieved_cov = float(np.mean(cal_probs >= thresh))
            assert abs(achieved_cov - target_cov) <= 0.02, (
                f"Mode={mode}, target={target_cov:.2f}, achieved={achieved_cov:.4f}"
            )


def test_grouped_cv_never_splits_scenario_across_train_and_score() -> None:
    """Task 9f: Verify GroupKFold cross-validation never places the same scenario in both train and score folds."""
    scs = [generate_scenario(seed=900 + i, scenario_id=f"cv_sc_{i:02d}") for i in range(10)]
    conditions = [
        ("clean", None, 0.0),
        ("missing_events", "missing_events", 0.20),
        ("duplicate_records", "duplicate_records", 0.30),
    ]
    feat_df = build_feature_table(scs, conditions=conditions, num_perturbations=15, seed=42)
    cv_res = run_grouped_cross_validation(feat_df, feature_mode="full", n_splits=5, target_coverage=0.65)

    assert cv_res["n_splits"] == 5
    seen_score_scenarios = set()
    for fold_info in cv_res["folds"]:
        train_sc = set(fold_info["train_scenario_ids"])
        score_sc = set(fold_info["score_scenario_ids"])
        assert train_sc.isdisjoint(score_sc), f"Scenario leakage in fold {fold_info['fold']}"
        assert seen_score_scenarios.isdisjoint(score_sc), "Scenario scored in multiple CV folds!"
        seen_score_scenarios.update(score_sc)

    assert seen_score_scenarios == {s.scenario_id for s in scs}


def test_final_command_fails_when_model_missing_and_refuses_when_locked(tmp_path: Path) -> None:
    """Tasks 9g & 9h: Verify final command fails when saved model is missing and refuses when lock exists."""
    fresh_dir = tmp_path / "fresh_test"
    sc = generate_scenario(seed=4242, scenario_id="fresh_test_000")
    sc.save(fresh_dir)
    with open(fresh_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": "fresh_test",
                "master_seed": 4242,
                "num_scenarios": 1,
                "scenario_ids": ["fresh_test_000"],
            },
            f,
        )

    missing_model_path = tmp_path / "nonexistent_models.json"
    lock_file = tmp_path / "final_run.lock"
    out_csv = tmp_path / "final_results.csv"

    # Task 9h: Fails when saved model is missing
    with pytest.raises(FileNotFoundError):
        run_final_fresh_test_experiment(
            fresh_test_dir=fresh_dir,
            model_path=missing_model_path,
            lock_path=lock_file,
            out_csv_path=out_csv,
            num_perturbations=10,
        )

    # Save a valid trained model so we can test lock creation and refusal on second run
    feat_df = build_feature_table([sc], conditions=[("clean", None, 0.0), ("duplicate_records", "duplicate_records", 0.20)], num_perturbations=10)
    feat_df["is_correct"] = [1, 0]
    full_m = LearnedReliabilityModel("full").fit_on_train(feat_df)
    full_m.calibrate_and_select_threshold_on_validation(feat_df, target_coverage=0.50)
    conf_m = LearnedReliabilityModel("confidence_only").fit_on_train(feat_df)
    conf_m.calibrate_and_select_threshold_on_validation(feat_df, target_coverage=0.50)

    valid_model_path = tmp_path / "reliability_models.json"
    with open(valid_model_path, "w", encoding="utf-8") as f:
        json.dump({"models": {"full": full_m.to_dict(), "confidence_only": conf_m.to_dict()}}, f)

    # First run creates lock_file
    res_df = run_final_fresh_test_experiment(
        fresh_test_dir=fresh_dir,
        model_path=valid_model_path,
        lock_path=lock_file,
        out_csv_path=out_csv,
        num_perturbations=10,
    )
    assert lock_file.is_file(), "final_run.lock must be created after final run finishes"
    assert len(res_df) > 0

    # Task 9g: Refuses to rerun when lock file exists
    with pytest.raises(RuntimeError, match="already exists"):
        run_final_fresh_test_experiment(
            fresh_test_dir=fresh_dir,
            model_path=valid_model_path,
            lock_path=lock_file,
            out_csv_path=out_csv,
            num_perturbations=10,
        )


def test_available_methods_and_attribution_metric() -> None:
    """Verify get_available_methods() includes all 5 methods and attribution metric maps defects."""
    methods_map = get_available_methods()
    for expected_key in (
        "frequency_baseline",
        "confidence_threshold",
        "faultlens",
        "learned_faultlens",
        "confidence_only_learned",
    ):
        assert expected_key in methods_map

    assert is_defect_attributed("missing_events", "completeness") is True
    assert is_defect_attributed("incorrect_reason_codes", "reason_validity") is True
    assert is_defect_attributed("incorrect_reason_codes", "alarm_reason_agreement") is True
    assert is_defect_attributed("timestamp_errors", "timestamp_consistency") is True
    assert is_defect_attributed("duplicate_records", "duplication") is True
    assert is_defect_attributed("duplicate_records", "completeness") is False
    assert compute_attribution_accuracy("duplicate_records", ["duplication", "completeness"], has_attribution=False) == "N/A"
    assert compute_attribution_accuracy("duplicate_records", ["duplication", "completeness"], has_attribution=True) == 0.5
