"""Unit tests for the 4 controlled defect injectors."""

import numpy as np
import pandas as pd
import pytest
from faultlens.defects import (
    ALLOWED_DEFECTS,
    ALLOWED_SEVERITIES,
    inject_defect,
    inject_duplicate_records,
    inject_incorrect_reason_codes,
    inject_missing_events,
    inject_timestamp_errors,
)
from faultlens.generator import generate_scenario


@pytest.fixture
def clean_scenario():
    return generate_scenario(seed=123)


def test_zero_severity_is_noop(clean_scenario) -> None:
    """Verify that severity=0.0 returns identical dataframe without modification."""
    df_clean = clean_scenario.log
    for defect_type in ALLOWED_DEFECTS:
        corrupted, report = inject_defect(df_clean, defect_type=defect_type, severity=0.0, seed=999)
        assert report["affected_fraction"] == 0.0
        pd.testing.assert_frame_equal(corrupted, df_clean)


def test_defects_never_change_true_cause_or_scenario_truth(clean_scenario) -> None:
    """CRITICAL: Verify defect injectors never mutate true_cause or scenario.truth."""
    df_clean = clean_scenario.log
    original_truth = clean_scenario.truth

    for defect in ALLOWED_DEFECTS:
        for sev in [0.10, 0.20, 0.30]:
            corrupted, report = inject_defect(df_clean, defect_type=defect, severity=sev, seed=42)

            # Check that scenario ground truth is preserved
            assert clean_scenario.truth == original_truth

            # Check that remaining events keep their exact original true_cause
            if defect != "duplicate_records":
                merged = df_clean[["event_id", "true_cause"]].merge(
                    corrupted[["event_id", "true_cause"]],
                    on="event_id",
                    suffixes=("_orig", "_corrupt"),
                )
                assert (merged["true_cause_orig"] == merged["true_cause_corrupt"]).all()
            else:
                # In duplicate records, all rows originated from original events
                base_event_ids = df_clean["event_id"].unique()
                for _, row in corrupted.iterrows():
                    orig_id = row["event_id"].split("_")[0]
                    assert orig_id in base_event_ids
                    expected_cause = df_clean.loc[df_clean["event_id"] == orig_id, "true_cause"].iloc[0]
                    assert row["true_cause"] == expected_cause


@pytest.mark.parametrize("sev", [0.10, 0.20, 0.30])
def test_missing_events_severity_and_splits(clean_scenario, sev) -> None:
    """Verify missing_events severity counts within exact integer rounding."""
    df = clean_scenario.log
    n_orig = len(df)
    corrupted, report = inject_missing_events(df, severity=sev, seed=101)

    expected_affected = int(round(n_orig * sev))
    actual_affected = report["dropped_count"] + report["null_end_ts_count"]
    assert actual_affected == expected_affected
    assert len(corrupted) == n_orig - report["dropped_count"]
    assert corrupted["end_ts"].isna().sum() == report["null_end_ts_count"]


@pytest.mark.parametrize("sev", [0.10, 0.20, 0.30])
def test_incorrect_reason_codes_severity(clean_scenario, sev) -> None:
    """Verify incorrect_reason_codes changes exactly the requested proportion of records."""
    df = clean_scenario.log
    n_orig = len(df)
    corrupted, report = inject_incorrect_reason_codes(df, severity=sev, seed=202)

    expected_affected = int(round(n_orig * sev))
    assert report["altered_reasons_count"] == expected_affected
    assert len(corrupted) == n_orig


@pytest.mark.parametrize("sev", [0.10, 0.20, 0.30])
def test_timestamp_errors_applied(clean_scenario, sev) -> None:
    """Verify timestamp_errors applies machine offsets and event jitter."""
    df = clean_scenario.log
    corrupted, report = inject_timestamp_errors(df, severity=sev, seed=303)

    assert len(report["machine_offsets"]) == 4  # 4 machines
    assert report["jittered_events_count"] == int(round(len(df) * sev))
    assert len(corrupted) == len(df)
    assert (corrupted["duration_min"] > 0).all()


@pytest.mark.parametrize("sev", [0.10, 0.20, 0.30])
def test_duplicate_records_severity(clean_scenario, sev) -> None:
    """Verify duplicate_records creates duplicates and expands dataframe."""
    df = clean_scenario.log
    corrupted, report = inject_duplicate_records(df, severity=sev, seed=404)

    assert len(corrupted) > len(df)
    assert report["single_dupes_count"] > 0 or report["burst_dupes_count"] > 0
