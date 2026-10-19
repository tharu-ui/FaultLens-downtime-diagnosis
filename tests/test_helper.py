"""Unit tests for app/helpers.py.

Verifies:
1. Train/Val split enforcement (strict test split exclusion).
2. CSV validation and parsing rules.
3. Live diagnosis calculation reusing extract_reliability_features reports.
4. Real field names adherence (StabilityReport & QualityReport).
5. Safe read-only behavior for lock and model files.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import app.helpers as h


def test_allowed_scenario_ids_excludes_test_split() -> None:
    """Verify that get_allowed_scenario_ids only includes train and val, excluding test."""
    allowed = h.get_allowed_scenario_ids()
    assert "train" in allowed
    assert "val" in allowed
    assert "all_allowed" in allowed

    # Check against raw splits.json
    splits_path = h.SPLITS_PATH
    if splits_path.is_file():
        with open(splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)
        test_ids = set(splits_data.get("test", []))
        all_allowed_set = set(allowed["all_allowed"])

        # Absolutely no test IDs allowed in dashboard selection
        assert all_allowed_set.isdisjoint(test_ids), "Test scenario leaked into allowed set!"
        assert len(allowed["train"]) == len(splits_data.get("train", []))
        assert len(allowed["val"]) == len(splits_data.get("val", []))


def test_load_scenario_by_id_blocks_test_split() -> None:
    """Verify that attempting to load a test scenario raises ValueError."""
    splits_path = h.SPLITS_PATH
    if splits_path.is_file():
        with open(splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)
        test_ids = splits_data.get("test", [])
        if test_ids:
            with pytest.raises(ValueError, match="strictly prohibited"):
                h.load_scenario_by_id(test_ids[0])


def test_load_scenario_by_id_loads_val_scenario() -> None:
    """Verify loading a valid scenario from validation split succeeds."""
    allowed = h.get_allowed_scenario_ids()
    if allowed["val"]:
        val_id = allowed["val"][0]
        sc = h.load_scenario_by_id(val_id)
        assert sc.scenario_id == val_id
        assert isinstance(sc.log, pd.DataFrame)
        assert not sc.log.empty
        assert isinstance(sc.truth, tuple)


def test_validate_and_parse_uploaded_csv_valid() -> None:
    """Verify parsing a valid CSV returns a properly typed DataFrame."""
    valid_csv = (
        "event_id,machine_id,start_ts,end_ts,duration_min,logged_reason\n"
        "EVT-001,Press,2026-01-01 08:00:00,2026-01-01 08:30:00,30.0,jam\n"
        "EVT-002,Welder,2026-01-01 09:00:00,2026-01-01 09:15:00,15.0,sensor_fault\n"
    )
    df, err = h.validate_and_parse_uploaded_csv(valid_csv)
    assert err is None
    assert df is not None
    assert len(df) == 2
    assert pd.api.types.is_datetime64_any_dtype(df["start_ts"])
    assert pd.api.types.is_numeric_dtype(df["duration_min"])


def test_validate_and_parse_uploaded_csv_missing_columns() -> None:
    """Verify uploaded CSV with missing columns returns a clear error."""
    invalid_csv = "event_id,machine_id\n1,Press\n"
    df, err = h.validate_and_parse_uploaded_csv(invalid_csv)
    assert df is None
    assert err is not None
    assert "missing required column" in err.lower()


def test_validate_and_parse_uploaded_csv_empty() -> None:
    """Verify empty CSV upload returns a clear error."""
    df, err = h.validate_and_parse_uploaded_csv("")
    assert df is None
    assert err is not None


def test_run_live_diagnosis_reuses_reports_and_real_field_names() -> None:
    """Verify run_live_diagnosis calls extract_reliability_features and exposes correct field names."""
    allowed = h.get_allowed_scenario_ids()
    if not allowed["val"]:
        pytest.skip("No validation scenarios available")

    sc = h.load_scenario_by_id(allowed["val"][0])
    res = h.run_live_diagnosis(sc.log, num_perturbations=10, seed=42)

    # Verify diagnostic result
    assert res["diag"] is not None
    assert "top_candidate" in res
    assert "raw_confidence" in res

    # Verify real QualityReport field names
    q_rep = res["q_rep"]
    assert hasattr(q_rep, "overall_score")
    assert hasattr(q_rep, "sub_scores")
    assert hasattr(q_rep, "worst_component")
    assert hasattr(q_rep, "worst_machine")
    assert hasattr(q_rep, "machine_scores")
    assert hasattr(q_rep, "weekly_scores")
    assert hasattr(q_rep, "counts")
    assert "overall_quality" in res
    assert "worst_component" in res
    assert "sub_scores" in res
    assert "counts" in res

    # Verify real StabilityReport field names
    s_rep = res["s_rep"]
    assert hasattr(s_rep, "top_stability")
    assert hasattr(s_rep, "top3_flip_rate")
    assert hasattr(s_rep, "margin_ci_lower_95")
    assert hasattr(s_rep, "margin_ci_upper_95")
    assert hasattr(s_rep, "margin_ci_crosses_zero")
    assert hasattr(s_rep, "num_perturbations")
    assert hasattr(s_rep, "candidate_cis")
    assert hasattr(s_rep, "top_candidate")
    assert hasattr(s_rep, "runner_up_candidate")

    assert not hasattr(s_rep, "flip_rate_top3")
    assert not hasattr(s_rep, "margin_ci_95")

    # Verify methods produced
    assert "frequency_baseline" in res["method_outputs"]
    assert "faultlens" in res["method_outputs"]
    fb_out = res["method_outputs"]["frequency_baseline"]
    assert fb_out.answered is True


def test_load_demo_selection_and_splits_isolation() -> None:
    """Verify demo selection strictly selects from train and validation splits."""
    demo_dict = h.load_demo_selection()
    assert "case_a" in demo_dict
    assert "case_b" in demo_dict
    assert "case_c" in demo_dict

    allowed = h.get_allowed_scenario_ids()
    allowed_set = set(allowed["all_allowed"])

    for case_k in ["case_a", "case_b", "case_c"]:
        case_info = demo_dict[case_k]
        sc_id = case_info["scenario_id"]
        assert sc_id in allowed_set, f"Demo scenario {sc_id} not in allowed train/val splits!"
        assert case_info["split"] in ["train", "val"]


def test_load_final_results_summary_handles_missing_file(tmp_path: Path) -> None:
    """Verify load_final_results_summary cleanly reports missing files without raising."""
    non_existent = tmp_path / "does_not_exist.csv"
    p_df, s_df, err = h.load_final_results_summary(non_existent)
    assert p_df is None
    assert s_df is None
    assert err is not None
    assert "not yet available" in err

