"""FaultLens Evaluation Metrics & Statistical Analysis Module.

Implements all benchmark metrics:
- Overall accuracy & selective accuracy (accuracy on answered subset)
- Confidently-wrong rate (answered & incorrect over all scenarios)
- Strict confidently-wrong rate (answered, incorrect, & confidence >= 0.80)
- Coverage and abstention rate
- Abstention quality: good abstentions (prevented errors) vs unnecessary abstentions
- Risk-coverage curve and AURC (Area Under Risk-Coverage curve)
- Expected Calibration Error (ECE)
- Bootstrap 95% confidence intervals over scenarios
- Paired statistical comparison (paired bootstrap difference & p-values)
- Support for stratified slicing across defect types and severities
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from faultlens.diagnoser import compute_ece


@dataclass
class MetricSummary:
    """Consolidated metrics for an evaluated diagnostic method."""

    method_name: str
    total_scenarios: int
    answered_count: int
    abstained_count: int
    coverage: float
    abstention_rate: float
    accuracy: float
    selective_accuracy: float
    confidently_wrong_rate: float
    strict_confidently_wrong_rate: float
    mean_confidence: float
    ece: float
    good_abstention_share: float
    unnecessary_abstention_share: float
    aurc: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics summary."""
        return {
            "method_name": self.method_name,
            "total_scenarios": self.total_scenarios,
            "answered_count": self.answered_count,
            "abstained_count": self.abstained_count,
            "coverage": round(self.coverage, 4),
            "abstention_rate": round(self.abstention_rate, 4),
            "accuracy": round(self.accuracy, 4),
            "selective_accuracy": round(self.selective_accuracy, 4),
            "confidently_wrong_rate": round(self.confidently_wrong_rate, 4),
            "strict_confidently_wrong_rate": round(self.strict_confidently_wrong_rate, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "ece": round(self.ece, 4),
            "good_abstention_share": round(self.good_abstention_share, 4),
            "unnecessary_abstention_share": round(self.unnecessary_abstention_share, 4),
            "aurc": round(self.aurc, 4),
        }


def compute_metrics(
    predictions: List[Optional[Tuple[str, str]]],
    ground_truths: List[Tuple[str, str]],
    answered_flags: List[bool],
    confidences: List[float],
    unvetted_predictions: Optional[List[Tuple[str, str]]] = None,
    method_name: str = "method",
) -> MetricSummary:
    """Compute complete benchmark metric suite for a given set of evaluation points.

    Args:
        predictions: Top-1 predicted candidate, or None if abstained.
        ground_truths: True scenario ground truth tuples.
        answered_flags: Boolean flags indicating whether the method answered.
        confidences: Output confidence scores in [0, 1].
        unvetted_predictions: Optional raw predictions prior to abstention (for abstention quality).
        method_name: Name of the evaluated method.

    Returns:
        MetricSummary containing all standard benchmark metrics.
    """
    n = len(ground_truths)
    if n == 0:
        return MetricSummary(
            method_name=method_name,
            total_scenarios=0,
            answered_count=0,
            abstained_count=0,
            coverage=0.0,
            abstention_rate=0.0,
            accuracy=0.0,
            selective_accuracy=0.0,
            confidently_wrong_rate=0.0,
            strict_confidently_wrong_rate=0.0,
            mean_confidence=0.0,
            ece=0.0,
            good_abstention_share=0.0,
            unnecessary_abstention_share=0.0,
            aurc=0.0,
        )

    answered_arr = np.asarray(answered_flags, dtype=bool)
    conf_arr = np.asarray(confidences, dtype=float)
    n_answered = int(answered_arr.sum())
    n_abstained = n - n_answered

    # Correctness indicators
    is_correct = np.zeros(n, dtype=bool)
    for i in range(n):
        if answered_arr[i] and predictions[i] is not None:
            is_correct[i] = (predictions[i] == ground_truths[i])

    coverage = n_answered / n
    abstention_rate = 1.0 - coverage

    overall_acc = float(is_correct.sum() / n)
    selective_acc = float(is_correct[answered_arr].mean()) if n_answered > 0 else 0.0

    # Confidently-wrong rate: answered AND wrong over ALL scenarios
    wrong_and_answered = answered_arr & (~is_correct)
    cw_rate = float(wrong_and_answered.sum() / n)

    # Strict confidently-wrong: answered AND wrong AND conf >= 0.80
    strict_cw = answered_arr & (~is_correct) & (conf_arr >= 0.80)
    strict_cw_rate = float(strict_cw.sum() / n)

    mean_conf = float(conf_arr[answered_arr].mean()) if n_answered > 0 else 0.0

    # ECE on answered cases
    if n_answered > 0:
        ece_val, _ = compute_ece(conf_arr[answered_arr], is_correct[answered_arr].astype(int), n_bins=10)
    else:
        ece_val = 0.0

    # Abstention quality
    good_abstention_share = 0.0
    unnecessary_abstention_share = 0.0
    if n_abstained > 0 and unvetted_predictions is not None:
        abstained_mask = ~answered_arr
        raw_would_be_correct = np.array([
            unvetted_predictions[i] == ground_truths[i] for i in range(n)
        ], dtype=bool)

        good_abstentions = int((abstained_mask & (~raw_would_be_correct)).sum())
        unnecessary_abstentions = int((abstained_mask & raw_would_be_correct).sum())

        good_abstention_share = float(good_abstentions / n_abstained)
        unnecessary_abstention_share = float(unnecessary_abstentions / n_abstained)

    # Risk-coverage curve & AURC
    aurc = compute_aurc(confidences=conf_arr, correct_indicators=is_correct, answered_flags=answered_arr)

    return MetricSummary(
        method_name=method_name,
        total_scenarios=n,
        answered_count=n_answered,
        abstained_count=n_abstained,
        coverage=round(coverage, 4),
        abstention_rate=round(abstention_rate, 4),
        accuracy=round(overall_acc, 4),
        selective_accuracy=round(selective_acc, 4),
        confidently_wrong_rate=round(cw_rate, 4),
        strict_confidently_wrong_rate=round(strict_cw_rate, 4),
        mean_confidence=round(mean_conf, 4),
        ece=round(ece_val, 4),
        good_abstention_share=round(good_abstention_share, 4),
        unnecessary_abstention_share=round(unnecessary_abstention_share, 4),
        aurc=round(aurc, 4),
    )


def compute_aurc(
    confidences: np.ndarray,
    correct_indicators: np.ndarray,
    answered_flags: np.ndarray,
) -> float:
    """Compute Area Under the Risk-Coverage Curve (AURC).

    Sorts points by confidence descending. As coverage expands from 0 to max_coverage,
    risk is the cumulative fraction of errors among answered cases.
    """
    n = len(confidences)
    if n == 0:
        return 0.0

    # For points that were answered, rank by confidence descending
    answered_indices = np.where(answered_flags)[0]
    if len(answered_indices) == 0:
        return 0.0

    sorted_sub = answered_indices[np.argsort(-confidences[answered_indices])]

    coverages = [0.0]
    risks = [0.0]

    cum_errors = 0
    for k, idx in enumerate(sorted_sub, start=1):
        if not correct_indicators[idx]:
            cum_errors += 1
        cov = k / n
        risk = cum_errors / k
        coverages.append(cov)
        risks.append(risk)

    # Trapezoidal integration
    aurc = float(np.trapz(risks, coverages))
    return float(np.clip(aurc, 0.0, 1.0))


def bootstrap_metric_ci(
    metric_fn: Callable[[np.ndarray], float],
    indices: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute empirical bootstrap 95% confidence interval for any scalar metric.

    Returns:
        (mean_metric, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(indices)
    boot_vals = np.zeros(n_bootstraps, dtype=float)

    for b in range(n_bootstraps):
        b_idx = rng.choice(indices, size=n, replace=True)
        boot_vals[b] = metric_fn(b_idx)

    mean_val = float(np.mean(boot_vals))
    ci_low = float(np.percentile(boot_vals, 100 * (alpha / 2.0)))
    ci_high = float(np.percentile(boot_vals, 100 * (1.0 - alpha / 2.0)))
    return mean_val, ci_low, ci_high


def paired_bootstrap_test(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap comparison between two methods on matched scenarios.

    Tests null hypothesis that mean(metric_a - metric_b) == 0.

    Returns:
        mean_diff, ci_lower_95, ci_upper_95, p_value
    """
    rng = np.random.default_rng(seed)
    diffs = np.asarray(metric_a, dtype=float) - np.asarray(metric_b, dtype=float)
    n = len(diffs)

    boot_diffs = np.zeros(n_bootstraps, dtype=float)
    for b in range(n_bootstraps):
        b_idx = rng.choice(n, size=n, replace=True)
        boot_diffs[b] = np.mean(diffs[b_idx])

    mean_diff = float(np.mean(diffs))
    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))

    # Two-sided p-value: fraction of bootstraps crossing 0 in the opposite direction
    if mean_diff >= 0:
        p_val = 2.0 * float(np.mean(boot_diffs <= 0))
    else:
        p_val = 2.0 * float(np.mean(boot_diffs >= 0))
    p_val = float(np.clip(p_val, 0.0001, 1.0))

    return {
        "mean_diff": round(mean_diff, 4),
        "ci_lower_95": round(ci_low, 4),
        "ci_upper_95": round(ci_high, 4),
        "p_value": round(p_val, 4),
    }


def slice_metrics_by_defect(
    df_results: pd.DataFrame,
    method_name: str,
) -> pd.DataFrame:
    """Slice metrics by defect_type and severity.

    Expected columns in df_results:
    ['scenario_id', 'defect_type', 'severity', 'prediction', 'truth', 'answered', 'confidence', 'raw_prediction']
    """
    rows = []
    groups = df_results.groupby(["defect_type", "severity"])

    for (d_type, sev), grp in groups:
        m = compute_metrics(
            predictions=list(grp["prediction"]),
            ground_truths=list(grp["truth"]),
            answered_flags=list(grp["answered"]),
            confidences=list(grp["confidence"]),
            unvetted_predictions=list(grp["raw_prediction"]) if "raw_prediction" in grp.columns else None,
            method_name=method_name,
        )
        rows.append({
            "defect_type": d_type,
            "severity": float(sev),
            "scenarios": len(grp),
            "coverage": m.coverage,
            "accuracy": m.accuracy,
            "selective_accuracy": m.selective_accuracy,
            "confidently_wrong_rate": m.confidently_wrong_rate,
            "strict_cw_rate": m.strict_confidently_wrong_rate,
            "abstention_rate": m.abstention_rate,
            "good_abstention_share": m.good_abstention_share,
        })

    return pd.DataFrame(rows)
