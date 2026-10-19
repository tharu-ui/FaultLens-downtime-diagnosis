"""FaultLens Evaluation Methods: Comparative & Learned Paradigms.

Provides unified interfaces for the evaluated diagnostic methods:
1. frequency_baseline: Naive downtime summation; always answers (never abstains).
2. confidence_threshold: Answers only if post-hoc calibrated probability exceeds threshold.
   Includes matched-coverage calibration tuning on validation split.
3. faultlens: Evidence-aware diagnosis combining data quality, perturbation stability,
   disciplined abstention, and actionable verification summaries.
4. learned_faultlens (Session 3.5): Logistic regression reliability model trained on TRAIN
   over quality sub-scores, stability, margin CI width, and log-repair sensitivity features,
   calibrated with a single threshold chosen on VALIDATION.
5. confidence_only_learned (Session 3.5): Ablation trained on raw_confidence only through
   the exact same reliability model code path.

All methods rely on the identical underlying shared diagnoser.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from faultlens.diagnoser import ConfidenceCalibrator, DiagnosisResult, diagnose
from faultlens.policy import (
    EvidenceSummary,
    PolicyDecision,
    apply_policy,
    generate_verification_message,
    load_tuned_gate_config,
    load_tuned_policy_params,
)
from faultlens.quality import QualityReport, evaluate_data_quality
from faultlens.reliability import (
    RELIABILITY_MODELS_PATH,
    LearnedReliabilityModel,
    extract_reliability_features,
    load_saved_reliability_model,
)
from faultlens.stability import StabilityReport, evaluate_stability


@dataclass
class MethodOutput:
    """Standardized output produced by all evaluated diagnostic methods."""

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
    Uses Platt scaling by default when fitting calibration so predicted probabilities
    are strictly monotonic in raw margin confidence (avoiding isotonic step plateaus).
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
        self._reference_confs: Optional[np.ndarray] = None

    def fit_calibrator(
        self,
        raw_confs: np.ndarray | List[float],
        labels: np.ndarray | List[int],
        method: str = "platt",
    ) -> "ConfidenceThresholdMethod":
        """Fit calibration model on validation set."""
        self.calibrator = ConfidenceCalibrator(method=method)
        self.calibrator.fit(raw_confs, labels)
        self._reference_confs = self.calibrator.predict_batch(raw_confs)
        return self

    def find_threshold_for_coverage(
        self,
        calibrated_confs: np.ndarray | List[float],
        target_coverage: float,
    ) -> float:
        """Select confidence threshold yielding target coverage on validation set."""
        arr = np.asarray(calibrated_confs, dtype=float)
        self._reference_confs = arr
        if len(arr) == 0:
            return 0.50
        if target_coverage <= 0.0:
            tuned_t = float(np.max(arr) + 1e-6)
            self.threshold = tuned_t
            return tuned_t
        if target_coverage >= 1.0:
            tuned_t = float(np.min(arr) - 1e-6)
            self.threshold = tuned_t
            return tuned_t
        # Percentile corresponding to (1 - target_coverage)
        pct = max(0.0, min(100.0, (1.0 - target_coverage) * 100.0))
        tuned_t = float(np.percentile(arr, pct))
        self.threshold = tuned_t
        return tuned_t

    def threshold_for_target_coverage(
        self,
        target: float,
        calibrated_confs: Optional[np.ndarray | List[float]] = None,
    ) -> float:
        """Thin wrapper around find_threshold_for_coverage for target coverage matching."""
        ref = calibrated_confs if calibrated_confs is not None else self._reference_confs
        if ref is None:
            raise ValueError("Provide calibrated_confs or call fit_calibrator/find_threshold_for_coverage first.")
        return self.find_threshold_for_coverage(ref, target_coverage=target)

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
        gate_mode: Optional[str] = None,
        delta: Optional[float] = None,
    ) -> None:
        self.method_name = "faultlens"
        self.calibrator = calibrator
        loaded_tq, loaded_ts, loaded_buf = load_tuned_policy_params()
        gate_cfg = load_tuned_gate_config()
        self.tau_q = tau_q if tau_q is not None else loaded_tq
        self.tau_s = tau_s if tau_s is not None else loaded_ts
        self.buffer = buffer if buffer is not None else loaded_buf
        self.gate_mode = gate_mode if gate_mode is not None else gate_cfg["gate_mode"]
        self.delta = delta if delta is not None else gate_cfg["delta"]
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
            gate_mode=self.gate_mode,
            delta=self.delta,
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


class LearnedFaultLensMethod:
    """Method 4 (Session 3.5): Learned Evidence-Aware FaultLens Method.

    Uses a logistic regression reliability classifier trained on TRAIN over the 16
    quality, stability, margin CI width, and log-repair features, Platt-calibrated on
    VALIDATION with a single threshold chosen on VALIDATION.
    """

    def __init__(
        self,
        model: Optional[LearnedReliabilityModel] = None,
        model_path: str | Path = RELIABILITY_MODELS_PATH,
        threshold: Optional[float] = None,
        buffer: float = 0.05,
        num_perturbations: int = 100,
    ) -> None:
        self.method_name = "learned_faultlens"
        self.model_path = Path(model_path)
        self.model = model
        if self.model is None and self.model_path.is_file():
            self.model = load_saved_reliability_model("full", self.model_path)
        self.threshold = (
            float(threshold)
            if threshold is not None
            else (self.model.threshold if self.model is not None else 0.65)
        )
        self.buffer = buffer
        self.num_perturbations = num_perturbations

    def _ensure_model(self) -> LearnedReliabilityModel:
        if self.model is None:
            self.model = load_saved_reliability_model("full", self.model_path)
            if self.threshold is None:
                self.threshold = self.model.threshold
        return self.model

    def threshold_for_target_coverage(
        self,
        target: float,
        calibrated_confs: Optional[np.ndarray | List[float]] = None,
    ) -> float:
        """Select threshold on calibrated reliability probabilities for target coverage."""
        mdl = self._ensure_model()
        ref = calibrated_confs if calibrated_confs is not None else mdl.validation_calibrated_probs
        if ref is None:
            raise ValueError("No reference calibrated probabilities available for coverage matching.")
        tuned_t = mdl.find_threshold_for_coverage(ref, target_coverage=target)
        self.threshold = tuned_t
        return tuned_t

    def predict(self, df: pd.DataFrame, seed: int = 42) -> MethodOutput:
        """Extract full reliability features, score with learned model, and attach evidence summary."""
        mdl = self._ensure_model()
        feats, diag, q_rep, s_rep, rep_rep = extract_reliability_features(
            df,
            num_perturbations=self.num_perturbations,
            seed=seed,
        )
        cal_prob = mdl.predict_calibrated_prob(feats)
        answered = bool(cal_prob >= self.threshold)

        if answered:
            flag = "GREEN" if cal_prob >= (self.threshold + self.buffer) else "AMBER"
        else:
            flag = "RED"

        v_msg = generate_verification_message(q_rep, s_rep, df)
        sorted_subs = dict(sorted(q_rep.sub_scores.items(), key=lambda kv: kv[1])[:2])

        ev_summary = {
            "strategy": "learned_reliability",
            "threshold": round(self.threshold, 4),
            "calibrated_reliability": round(cal_prob, 4),
            "worst_component": q_rep.worst_component,
            "quality_score": round(q_rep.overall_score, 4),
            "stability_pct": round(s_rep.top_stability * 100.0, 1),
            "weakest_sub_scores": {k: round(v, 4) for k, v in sorted_subs.items()},
            "repair_top_candidate_changed": rep_rep.top_candidate_changed,
            "repaired_rows_count": rep_rep.total_repaired_rows,
            "verification_message": v_msg,
        }

        return MethodOutput(
            method_name=self.method_name,
            top_candidate=diag.top_candidate if answered else None,
            answered=answered,
            confidence=cal_prob,
            flag=flag,
            evidence_summary=ev_summary,
            raw_confidence=diag.raw_confidence,
            calibrated_confidence=cal_prob,
            top_downtime_min=diag.top_downtime_min,
            margin_ratio=diag.margin_ratio,
        )


class ConfidenceOnlyLearnedMethod:
    """Method 5 (Session 3.5 Ablation): Learned model trained on raw_confidence only.

    Uses the exact same `LearnedReliabilityModel` code path as `LearnedFaultLensMethod`,
    restricted to `["raw_confidence"]` with no quality or repair attribution.
    """

    def __init__(
        self,
        model: Optional[LearnedReliabilityModel] = None,
        model_path: str | Path = RELIABILITY_MODELS_PATH,
        threshold: Optional[float] = None,
        buffer: float = 0.05,
    ) -> None:
        self.method_name = "confidence_only_learned"
        self.model_path = Path(model_path)
        self.model = model
        if self.model is None and self.model_path.is_file():
            self.model = load_saved_reliability_model("confidence_only", self.model_path)
        self.threshold = (
            float(threshold)
            if threshold is not None
            else (self.model.threshold if self.model is not None else 0.65)
        )
        self.buffer = buffer

    def _ensure_model(self) -> LearnedReliabilityModel:
        if self.model is None:
            self.model = load_saved_reliability_model("confidence_only", self.model_path)
            if self.threshold is None:
                self.threshold = self.model.threshold
        return self.model

    def threshold_for_target_coverage(
        self,
        target: float,
        calibrated_confs: Optional[np.ndarray | List[float]] = None,
    ) -> float:
        """Select threshold on calibrated probabilities for target coverage."""
        mdl = self._ensure_model()
        ref = calibrated_confs if calibrated_confs is not None else mdl.validation_calibrated_probs
        if ref is None:
            raise ValueError("No reference calibrated probabilities available for coverage matching.")
        tuned_t = mdl.find_threshold_for_coverage(ref, target_coverage=target)
        self.threshold = tuned_t
        return tuned_t

    def predict(self, df: pd.DataFrame) -> MethodOutput:
        """Run shared diagnoser and score using the confidence-only learned model."""
        mdl = self._ensure_model()
        diag: DiagnosisResult = diagnose(df)
        raw_conf = diag.raw_confidence
        cal_prob = mdl.predict_calibrated_prob({"raw_confidence": raw_conf})
        answered = bool(cal_prob >= self.threshold)

        if answered:
            flag = "GREEN" if cal_prob >= (self.threshold + self.buffer) else "AMBER"
        else:
            flag = "RED"

        return MethodOutput(
            method_name=self.method_name,
            top_candidate=diag.top_candidate if answered else None,
            answered=answered,
            confidence=cal_prob,
            flag=flag,
            evidence_summary={
                "strategy": "confidence_only_learned_ablation",
                "threshold": round(self.threshold, 4),
                "calibrated_confidence": round(cal_prob, 4),
                "attribution": "N/A",
            },
            raw_confidence=raw_conf,
            calibrated_confidence=cal_prob,
            top_downtime_min=diag.top_downtime_min,
            margin_ratio=diag.margin_ratio,
        )


def get_available_methods() -> Dict[str, Type[Any]]:
    """Return a dictionary mapping method name to method class for dynamic discovery."""
    return {
        "frequency_baseline": FrequencyBaselineMethod,
        "confidence_threshold": ConfidenceThresholdMethod,
        "faultlens": FaultLensMethod,
        "learned_faultlens": LearnedFaultLensMethod,
        "confidence_only_learned": ConfidenceOnlyLearnedMethod,
    }
