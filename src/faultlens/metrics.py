"""FaultLens Evaluation Metrics & Statistical Analysis Module."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from faultlens.diagnoser import compute_ece

DEFECT_TO_COMPONENTS: Dict[str, Tuple[str, ...]] = {
    "missing_events": ("completeness",),
    "incorrect_reason_codes": ("reason_validity", "alarm_reason_agreement"),
    "timestamp_errors": ("timestamp_consistency",),
    "duplicate_records": ("duplication",),
}


def is_defect_attributed(
    defect_type: Optional[str],
    worst_component: Optional[str],
) -> Optional[bool]:
    """Check whether quality report's worst_component matches the injected defect type.

    Mapping (Task 4):
    - missing_events -> completeness
    - incorrect_reason_codes -> reason_validity or alarm_reason_agreement
    - timestamp_errors -> timestamp_consistency
    - duplicate_records -> duplication
    """
    if defect_type is None or defect_type in ("clean", "none", ""):
        return None
    expected = DEFECT_TO_COMPONENTS.get(str(defect_type))
    if expected is None or worst_component is None:
        return False
    return str(worst_component) in expected


def compute_attribution_accuracy(
    defect_type: Optional[str],
    worst_components: Sequence[Optional[str]],
    has_attribution: bool = True,
) -> Union[float, str]:
    """Compute attribution accuracy for a defect slice, or 'N/A' if not applicable."""
    if not has_attribution or defect_type is None or defect_type in ("clean", "none", ""):
        return "N/A"
    if len(worst_components) == 0:
        return 0.0
    matches = [bool(is_defect_attributed(defect_type, wc)) for wc in worst_components]
    return round(float(np.mean(matches)), 4)


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
    """Compute complete benchmark metric suite for a given set of evaluation points."""
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

    is_correct = np.zeros(n, dtype=bool)
    for i in range(n):
        if answered_arr[i] and predictions[i] is not None:
            is_correct[i] = (predictions[i] == ground_truths[i])

    coverage = n_answered / n
    abstention_rate = 1.0 - coverage

    overall_acc = float(is_correct.sum() / n)
    selective_acc = float(is_correct[answered_arr].mean()) if n_answered > 0 else 0.0

    wrong_and_answered = answered_arr & (~is_correct)
    cw_rate = float(wrong_and_answered.sum() / n)

    strict_cw = answered_arr & (~is_correct) & (conf_arr >= 0.80)
    strict_cw_rate = float(strict_cw.sum() / n)

    mean_conf = float(conf_arr[answered_arr].mean()) if n_answered > 0 else 0.0

    if n_answered > 0:
        ece_val, _ = compute_ece(conf_arr[answered_arr], is_correct[answered_arr].astype(int), n_bins=10)
    else:
        ece_val = 0.0

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
    """Compute Area Under the Risk-Coverage Curve (AURC)."""
    n = len(confidences)
    if n == 0:
        return 0.0

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

    if hasattr(np, "trapezoid"):
        aurc = float(np.trapezoid(risks, coverages))
    elif hasattr(np, "trapz"):
        aurc = float(np.trapz(risks, coverages))
    else:
        aurc = float(sum(0.5 * (risks[i] + risks[i + 1]) * (coverages[i + 1] - coverages[i]) for i in range(len(risks) - 1)))
    return float(np.clip(aurc, 0.0, 1.0))


def bootstrap_metric_ci(
    metric_fn: Callable[[np.ndarray], float],
    indices: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute empirical bootstrap 95% confidence interval for any scalar metric."""
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
    """Paired bootstrap comparison between two methods on matched scenarios."""
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
    """Slice metrics by defect_type and severity."""
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
