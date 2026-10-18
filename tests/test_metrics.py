"""Unit tests for benchmark metrics with hand-computed toy examples."""

import numpy as np
import pytest
from faultlens.metrics import compute_aurc, compute_metrics, paired_bootstrap_test


def test_hand_computed_toy_metrics() -> None:
    """Verify metrics match manual hand calculations on a controlled 5-sample case."""
    ground_truths = [
        ("M1", "C1"),
        ("M1", "C2"),
        ("M2", "C1"),
        ("M2", "C2"),
        ("M3", "C1"),
    ]
    unvetted = [
        ("M1", "C1"),  # Correct
        ("M1", "C1"),  # Wrong
        ("M2", "C1"),  # Correct
        ("M2", "C1"),  # Wrong
        ("M3", "C2"),  # Wrong
    ]
    # Method answers 3 cases, abstains on 2
    answered_flags = [True, True, True, False, False]
    predictions = [
        ("M1", "C1"),  # Answered, correct
        ("M1", "C1"),  # Answered, wrong (confidently wrong)
        ("M2", "C1"),  # Answered, correct
        None,          # Abstained
        None,          # Abstained
    ]
    confidences = [0.90, 0.85, 0.75, 0.60, 0.55]

    m = compute_metrics(
        predictions=predictions,
        ground_truths=ground_truths,
        answered_flags=answered_flags,
        confidences=confidences,
        unvetted_predictions=unvetted,
        method_name="toy_test",
    )

    assert m.total_scenarios == 5
    assert m.answered_count == 3
    assert m.abstained_count == 2
    assert m.coverage == pytest.approx(3 / 5, abs=1e-4)
    assert m.abstention_rate == pytest.approx(2 / 5, abs=1e-4)
    assert m.accuracy == pytest.approx(2 / 5, abs=1e-4)
    assert m.selective_accuracy == pytest.approx(2 / 3, abs=1e-4)
    assert m.confidently_wrong_rate == pytest.approx(1 / 5, abs=1e-4)
    assert m.strict_confidently_wrong_rate == pytest.approx(1 / 5, abs=1e-4)

    # In abstained cases (indices 3 and 4), unvetted predictions were both wrong
    # So 2 out of 2 abstentions were good (prevented errors)
    assert m.good_abstention_share == pytest.approx(1.0, abs=1e-4)
    assert m.unnecessary_abstention_share == pytest.approx(0.0, abs=1e-4)


def test_paired_bootstrap_test() -> None:
    """Verify paired bootstrap comparison computes expected difference and p-value."""
    # Method A has 10% errors, Method B has 30% errors
    arr_a = np.array([0.0] * 90 + [1.0] * 10)
    arr_b = np.array([0.0] * 70 + [1.0] * 30)

    res = paired_bootstrap_test(arr_a, arr_b, n_bootstraps=500, seed=42)
    assert res["mean_diff"] == pytest.approx(-0.20, abs=0.01)
    assert res["ci_upper_95"] < 0.0  # Method A is statistically significantly lower
    assert res["p_value"] < 0.05
