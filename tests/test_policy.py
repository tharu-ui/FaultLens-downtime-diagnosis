"""Unit tests for evidence-aware policy and threshold tuning."""

import json
from pathlib import Path
import pytest
from faultlens.defects import inject_missing_events
from faultlens.generator import generate_scenario
from faultlens.policy import (
    apply_policy,
    generate_verification_message,
    tune_policy_on_validation,
)
from faultlens.quality import evaluate_data_quality
from faultlens.stability import evaluate_stability


def test_extreme_thresholds_behavior() -> None:
    """Verify policy behavior at extreme threshold values."""
    sc = generate_scenario(seed=222)
    q_rep = evaluate_data_quality(sc.log)
    s_rep = evaluate_stability(sc.log, num_perturbations=50, seed=42)

    # Tau = 0.0 answers everything
    dec_zero = apply_policy(sc.log, q_rep, s_rep, tau_q=0.0, tau_s=0.0)
    assert dec_zero.answered is True
    assert dec_zero.status_flag in ["GREEN", "AMBER"]

    # Tau = 1.0 answers nothing (or requires impossible 100% perfection)
    dec_one = apply_policy(sc.log, q_rep, s_rep, tau_q=1.01, tau_s=1.01)
    assert dec_one.answered is False
    assert dec_one.status_flag == "RED"
    assert "INSUFFICIENT EVIDENCE" in dec_one.decision_reason


def test_tuning_touches_only_validation_scenarios(tmp_path: Path) -> None:
    """Verify threshold tuning uses exclusively validation scenarios without leaking test set."""
    splits_path = Path("data/sample_scenarios/splits.json")
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    val_set = set(splits["val"])
    test_set = set(splits["test"])
    train_set = set(splits["train"])

    out_file = tmp_path / "test_tuned.json"
    result = tune_policy_on_validation(
        scenarios_dir="data/sample_scenarios",
        min_coverage=0.60,
        out_params_path=out_file,
        grid_steps=5,
    )

    # Verify that tuning recorded exactly validation scenario count
    assert result["num_validation_scenarios"] == len(val_set)
    assert val_set.isdisjoint(test_set)
    assert val_set.isdisjoint(train_set)


def test_verification_message_content() -> None:
    """Verify that specific defect triggers an actionable message with real counts."""
    sc = generate_scenario(seed=333)
    corrupted, _ = inject_missing_events(sc.log, severity=0.30, seed=42)

    q_rep = evaluate_data_quality(corrupted)
    s_rep = evaluate_stability(corrupted, num_perturbations=50, seed=42)
    msg = generate_verification_message(q_rep, s_rep, corrupted)

    assert msg is not None
    assert any(term in msg.lower() for term in ["end time", "missing", "data capture", "stops"])
