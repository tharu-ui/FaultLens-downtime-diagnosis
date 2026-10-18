"""Unit tests for synthetic log generator."""

import pandas as pd
import pytest
from faultlens.generator import generate_scenario, load_generator_config


def test_generator_deterministic_by_seed() -> None:
    """Verify that identical seeds produce bitwise-identical scenario logs and metadata."""
    seed = 42
    sc1 = generate_scenario(seed=seed, scenario_id="sc_test_1")
    sc2 = generate_scenario(seed=seed, scenario_id="sc_test_2")

    assert sc1.truth == sc2.truth
    assert sc1.truth_downtime == pytest.approx(sc2.truth_downtime, rel=1e-5)
    assert sc1.margin == pytest.approx(sc2.margin, rel=1e-5)
    assert sc1.difficulty == sc2.difficulty

    # Verify identical event counts and column values
    assert len(sc1.log) == len(sc2.log)
    pd.testing.assert_frame_equal(sc1.log, sc2.log)


def test_clean_log_ground_truth_matches_recomputed() -> None:
    """Verify clean-log ground truth matches recomputed downtime leader."""
    for seed in [11, 22, 33, 44, 55]:
        sc = generate_scenario(seed=seed)
        df = sc.log

        # Recompute total true downtime independently
        totals = df.groupby(["machine_id", "true_cause"])["duration_min"].sum()
        expected_top_pair = tuple(totals.sort_values(ascending=False).index[0])
        expected_top_downtime = float(totals.sort_values(ascending=False).iloc[0])

        assert sc.truth == expected_top_pair
        assert sc.truth_downtime == pytest.approx(expected_top_downtime, rel=1e-4)


def test_scenario_columns_and_data_types() -> None:
    """Verify all specified columns are present with valid data types."""
    sc = generate_scenario(seed=77)
    df = sc.log

    expected_cols = [
        "event_id",
        "machine_id",
        "start_ts",
        "end_ts",
        "duration_min",
        "true_cause",
        "logged_reason",
        "alarm_code",
        "reason_source",
        "shift",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) > 100  # A few hundred events per 30-day simulation
    assert (df["duration_min"] > 0).all()
    assert set(df["reason_source"].unique()).issubset({"auto", "manual"})
    assert set(df["shift"].unique()).issubset({"Shift_1", "Shift_2", "Shift_3"})
    assert (df["duration_min"] < 1.0).sum() > 0  # Contains micro-stops
