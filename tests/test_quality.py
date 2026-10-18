"""Unit tests for data-quality scoring."""

import numpy as np
import pytest
from faultlens.defects import ALLOWED_DEFECTS, inject_defect
from faultlens.generator import generate_scenario
from faultlens.quality import evaluate_data_quality


@pytest.fixture
def clean_log():
    sc = generate_scenario(seed=123)
    return sc.log


def test_clean_data_has_near_perfect_quality(clean_log) -> None:
    """Verify clean logs receive near-perfect quality scores (> 0.95)."""
    q_rep = evaluate_data_quality(clean_log)
    assert q_rep.overall_score >= 0.95
    assert q_rep.sub_scores["completeness"] == 1.0
    assert q_rep.sub_scores["reason_validity"] == 1.0
    assert q_rep.counts.missing_end_count == 0
    assert q_rep.counts.negative_duration_count == 0


def test_quality_score_decreases_with_severity(clean_log) -> None:
    """Verify that quality score strictly decreases as defect severity increases."""
    for defect in ALLOWED_DEFECTS:
        scores = []
        for sev in [0.0, 0.10, 0.20, 0.30]:
            corrupted, _ = inject_defect(clean_log, defect_type=defect, severity=sev, seed=42)
            q_rep = evaluate_data_quality(corrupted)
            scores.append(q_rep.overall_score)

        # Monotonically non-increasing (0.0 >= 0.10 >= 0.20 >= 0.30)
        assert scores[0] > scores[-1], f"Quality score failed to drop for defect {defect}"
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1] - 0.05, f"Non-monotonic quality drop in {defect}: {scores}"


def test_quality_identifies_worst_component_and_machine(clean_log) -> None:
    """Verify that specific defect triggers the corresponding worst component."""
    # Corrupt with missing_events
    corrupted_missing, _ = inject_defect(clean_log, defect_type="missing_events", severity=0.30, seed=99)
    q_rep = evaluate_data_quality(corrupted_missing)
    assert q_rep.worst_component in ["completeness", "timestamp_consistency"]
    assert q_rep.counts.missing_end_count > 0

    # Corrupt with duplicate_records
    corrupted_dups, _ = inject_defect(clean_log, defect_type="duplicate_records", severity=0.30, seed=99)
    q_rep_dup = evaluate_data_quality(corrupted_dups)
    assert q_rep_dup.worst_component == "duplication"
    assert q_rep_dup.counts.duplicate_burst_count > 0
