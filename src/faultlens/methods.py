"""FaultLens Evaluation Methods: The 3 Comparative Paradigms.

Provides unified interfaces for the three evaluated methods:
1. frequency_baseline: Naive downtime summation; always answers (never abstains).
2. confidence_threshold: Answers only if post-hoc calibrated probability exceeds threshold.
   Includes matched-coverage calibration tuning on validation split.
3. faultlens: Evidence-aware diagnosis combining data quality, perturbation stability,
   disciplined abstention, and actionable verification summaries.

All three methods rely on the identical underlying shared diagnoser.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from faultlens.diagnoser import ConfidenceCalibrator, DiagnosisResult, diagnose
from faultlens.policy import PolicyDecision, apply_policy, load_tuned_policy_params
from faultlens.quality import QualityReport, evaluate_data_quality
from faultlens.stability import StabilityReport, evaluate_stability


@dataclass
class MethodOutput:
    """Standardized output produced by all three evaluated diagnostic methods."""

    method_name: str
    top_candidate: Optional[Tuple[str, str]]
    answered: bool
    confidence: float
    flag: str  # "GREEN" | "AMBER" | "RED"
    evidence_summary: Optional[Dict[str, Any]]
    raw_confidence: float
    calibrated_confidence: Optional[float]
    top_downtime_min: float
    margin_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize method output to dictionary."""
        return {
            "method_name": self.method_name,
            "top_candidate": list(self.top_candidate) if self.top_candidate else None,
            "answered": self.answered,
            "confidence": round(self.confidence, 4),
            "flag": self.flag,
            "evidence_summary": self.evidence_summary,
            "raw_confidence": round(self.raw_confidence, 4),
            "calibrated_confidence": round(self.calibrated_confidence, 4) if self.calibrated_confidence is not None else None,
            "top_downtime_min": round(self.top_downtime_min, 2),
            "margin_ratio": round(self.margin_ratio, 4),
        }


class FrequencyBaselineMethod:
    """Method 1: Naive frequency baseline. Always answers, never abstains."""

    def __init__(self) -> None:
        self.method_name = "frequency_baseline"

    def predict(self, df: pd.DataFrame) -> MethodOutput:
        """Run shared diagnoser and produce unvetted diagnosis."""
        diag: DiagnosisResult = diagnose(df)
        raw_conf = diag.raw_confidence
        flag = "GREEN" if raw_conf >= 0.70 else "AMBER"

        return MethodOutput(
            method_name=self.method_name,
            top_candidate=diag.top_candidate,
            answered=True,  # Always answers
            confidence=raw_conf,
            flag=flag,
            evidence_summary={
                "strategy": "naive_summation",
                "message": "Always answers; no evidence verification performed.",
            },
            raw_confidence=raw_conf,
            calibrated_confidence=None,
            top_downtime_min=diag.top_downtime_min,
            margin_ratio=diag.margin_ratio,
        )


class ConfidenceThresholdMethod:
    """Method 2: Confidence-threshold baseline.

    Answers only if post-hoc calibrated probability >= threshold.
    """

    def __init__(
        self,
        calibrator: Optional[ConfidenceCalibrator] = None,
        threshold: float = 0.65,
        buffer: float = 0.05,
    ) -> None:
        self.method_name = "confidence_threshold"
        self.calibrator = calibrator
        self.threshold = threshold
        self.buffer = buffer

    def fit_calibrator(
        self,
        raw_confs: np.ndarray | List[float],
        labels: np.ndarray | List[int],
        method: str = "isotonic",
    ) -> "ConfidenceThresholdMethod":
        """Fit calibration model on validation set."""
        self.calibrator = ConfidenceCalibrator(method=method)
        self.calibrator.fit(raw_confs, labels)
        return self

    def find_threshold_for_coverage(
        self,
        calibrated_confs: np.ndarray | List[float],
        target_coverage: float,
    ) -> float:
        """Select confidence threshold yielding target coverage on validation set."""
        arr = np.asarray(calibrated_confs, dtype=float)
        if len(arr) == 0:
            return 0.50
        # Percentile corresponding to (1 - target_coverage)
        pct = max(0.0, min(100.0, (1.0 - target_coverage) * 100.0))
        tuned_t = float(np.percentile(arr, pct))
        self.threshold = tuned_t
        return tuned_t

    def predict(self, df: pd.DataFrame) -> MethodOutput:
        """Run shared diagnoser and abstain if calibrated confidence < threshold."""
        diag: DiagnosisResult = diagnose(df, calibrator=self.calibrator)
        raw_conf = diag.raw_confidence

        if self.calibrator is not None and self.calibrator.is_fitted:
            cal_conf = self.calibrator.predict(raw_conf)
        else:
            cal_conf = raw_conf

        answered = bool(cal_conf >= self.threshold)
        if answered:
            flag = "GREEN" if cal_conf >= (self.threshold + self.buffer) else "AMBER"
        else:
            flag = "RED"

        return MethodOutput(
            method_name=self.method_name,
            top_candidate=diag.top_candidate if answered else None,
            answered=answered,
            confidence=cal_conf,
            flag=flag,
            evidence_summary={
                "strategy": "calibrated_threshold",
                "threshold": round(self.threshold, 4),
                "calibrated_confidence": round(cal_conf, 4),
                "decision": "ANSWER" if answered else "ABSTAIN",
            },
            raw_confidence=raw_conf,
            calibrated_confidence=cal_conf,
            top_downtime_min=diag.top_downtime_min,
            margin_ratio=diag.margin_ratio,
        )


class FaultLensMethod:
    """Method 3: Evidence-Aware FaultLens method.

    Combines data quality scoring, perturbation stability testing,
    calibrated diagnosis, and disciplined abstention with what-to-verify messages.
    """

    def __init__(
        self,
        calibrator: Optional[ConfidenceCalibrator] = None,
        tau_q: Optional[float] = None,
        tau_s: Optional[float] = None,
        buffer: float = 0.05,
        num_perturbations: int = 100,
        require_margin_ci_positive: bool = False,
    ) -> None:
        self.method_name = "faultlens"
        self.calibrator = calibrator
        loaded_tq, loaded_ts, loaded_buf = load_tuned_policy_params()
        self.tau_q = tau_q if tau_q is not None else loaded_tq
        self.tau_s = tau_s if tau_s is not None else loaded_ts
        self.buffer = buffer if buffer is not None else loaded_buf
        self.num_perturbations = num_perturbations
        self.require_margin_ci_positive = require_margin_ci_positive

    def predict(self, df: pd.DataFrame, seed: int = 42) -> MethodOutput:
        """Run complete evidence-aware diagnosis pipeline."""
        diag: DiagnosisResult = diagnose(df, calibrator=self.calibrator)
        raw_conf = diag.raw_confidence
        cal_conf = self.calibrator.predict(raw_conf) if (self.calibrator and self.calibrator.is_fitted) else raw_conf

        q_rep: QualityReport = evaluate_data_quality(df)
        s_rep: StabilityReport = evaluate_stability(
            df,
            num_perturbations=self.num_perturbations,
            seed=seed,
        )

        decision: PolicyDecision = apply_policy(
            df=df,
            quality=q_rep,
            stability=s_rep,
            tau_q=self.tau_q,
            tau_s=self.tau_s,
            buffer=self.buffer,
            require_margin_ci_positive=self.require_margin_ci_positive,
        )

        return MethodOutput(
            method_name=self.method_name,
            top_candidate=diag.top_candidate if decision.answered else None,
            answered=decision.answered,
            confidence=cal_conf,
            flag=decision.status_flag,
            evidence_summary=decision.evidence_summary.to_dict() if decision.evidence_summary else None,
            raw_confidence=raw_conf,
            calibrated_confidence=cal_conf,
            top_downtime_min=diag.top_downtime_min,
            margin_ratio=diag.margin_ratio,
        )
