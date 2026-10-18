"""Unit tests for the 3 comparative diagnostic methods."""

import pytest
from faultlens.defects import inject_timestamp_errors
from faultlens.diagnoser import diagnose
from faultlens.generator import generate_scenario
from faultlens.methods import ConfidenceThresholdMethod, FaultLensMethod, FrequencyBaselineMethod


def test_frequency_baseline_always_answers() -> None:
    """Verify frequency baseline never abstains regardless of noise."""
    sc = generate_scenario(seed=111)
    method = FrequencyBaselineMethod()

    output = method.predict(sc.log)
    assert output.answered is True
    assert output.top_candidate is not None
    assert output.flag in ["GREEN", "AMBER"]


def test_confidence_threshold_abstains_when_below_threshold() -> None:
    """Verify confidence threshold method abstains when confidence is low."""
    sc = generate_scenario(seed=222)
    # Set threshold very high
    method = ConfidenceThresholdMethod(threshold=0.99)
    output = method.predict(sc.log)

    if output.confidence < 0.99:
        assert output.answered is False
        assert output.flag == "RED"
        assert output.top_candidate is None


def test_faultlens_clean_data_no_worse_than_baselines() -> None:
    """Verify that on clean data, FaultLens does not perform noticeably worse than baselines."""
    sc = generate_scenario(seed=333, target_difficulty="easy")

    m_freq = FrequencyBaselineMethod()
    m_conf = ConfidenceThresholdMethod(threshold=0.60)
    m_faultlens = FaultLensMethod(tau_q=0.75, tau_s=0.75, num_perturbations=50)

    out_freq = m_freq.predict(sc.log)
    out_conf = m_conf.predict(sc.log)
    out_fl = m_faultlens.predict(sc.log)

    # All should identify ground truth on clean easy scenario
    assert out_freq.top_candidate == sc.truth
    assert out_conf.top_candidate == sc.truth
    assert out_fl.top_candidate == sc.truth
    assert out_fl.answered is True
    assert out_fl.flag in ["GREEN", "AMBER"]


def test_timestamp_errors_fix_reduces_accuracy_at_30_percent() -> None:
    """Verify timestamp_errors at 30% severity causes accuracy to drop relative to clean 0%."""
    scenarios = [generate_scenario(seed=500 + i, target_difficulty="close") for i in range(25)]

    clean_correct = 0
    corrupt_correct = 0

    for sc in scenarios:
        # Clean
        d_clean = diagnose(sc.log)
        if d_clean.top_candidate == sc.truth:
            clean_correct += 1

        # 30% timestamp errors
        corrupted, _ = inject_timestamp_errors(sc.log, severity=0.30, seed=sc.seed)
        d_corrupt = diagnose(corrupted)
        if d_corrupt.top_candidate == sc.truth:
            corrupt_correct += 1

    clean_acc = clean_correct / len(scenarios)
    corrupt_acc = corrupt_correct / len(scenarios)

    # Corrupt accuracy must be lower than clean accuracy
    assert corrupt_acc < clean_acc, f"Clean {clean_acc:.2f} vs Corrupt {corrupt_acc:.2f}"

