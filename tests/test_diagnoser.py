"""Unit tests for shared diagnoser and calibration."""

import numpy as np
import pandas as pd
import pytest
from faultlens.defects import inject_missing_events
from faultlens.diagnoser import ConfidenceCalibrator, compute_ece, compute_raw_confidence, diagnose
from faultlens.generator import generate_scenario


def test_diagnoser_on_clean_easy_scenario() -> None:
    """Verify diagnoser correctly identifies ground truth on clean easy scenarios."""
    sc = generate_scenario(seed=777, target_difficulty="easy")
    result = diagnose(sc.log)

    assert result.top_candidate is not None
    assert result.top_candidate == sc.truth
    assert result.raw_confidence >= 0.50
    assert result.margin_ratio > 0.10
    assert result.events_skipped == 0
    assert result.events_analyzed == len(sc.log)


def test_diagnoser_skips_unclosed_stops() -> None:
    """Verify that events with missing end_ts are skipped during downtime summation."""
    sc = generate_scenario(seed=888)
    corrupted, report = inject_missing_events(sc.log, severity=0.20, seed=123)

    result = diagnose(corrupted)
    assert result.events_skipped == report["null_end_ts_count"]
    assert result.events_analyzed == len(corrupted) - report["null_end_ts_count"]


def test_raw_confidence_bounds() -> None:
    """Verify raw confidence formula bounds."""
    # Single candidate
    assert compute_raw_confidence(100.0, [100.0]) == 1.0

    # Equal top candidates
    assert compute_raw_confidence(100.0, [100.0, 100.0, 50.0]) == pytest.approx(0.50, abs=0.01)

    # Dominant candidate
    conf_dominant = compute_raw_confidence(1000.0, [1000.0, 10.0, 5.0])
    assert conf_dominant > 0.95


def test_isotonic_calibration_and_ece() -> None:
    """Verify isotonic calibration fitting and ECE calculation."""
    # Simulated validation results
    rng = np.random.default_rng(42)
    raw_confs = rng.uniform(0.55, 0.98, size=100)
    # Higher confidence slightly correlates with higher accuracy
    acc_probs = np.clip(raw_confs * 0.9, 0.1, 0.95)
    labels = (rng.uniform(0, 1, size=100) < acc_probs).astype(int)

    calibrator = ConfidenceCalibrator(method="isotonic")
    calibrator.fit(raw_confs, labels)

    test_confs = [0.60, 0.75, 0.90]
    preds = [calibrator.predict(c) for c in test_confs]

    for p in preds:
        assert 0.01 <= p <= 0.99

    ece_val, diagram = compute_ece(preds, [1, 1, 1], n_bins=5)
    assert 0.0 <= ece_val <= 1.0
    assert len(diagram["bin_accs"]) == 5
