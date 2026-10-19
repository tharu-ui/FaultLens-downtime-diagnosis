"""FaultLens Shared Diagnoser and Confidence Calibration Module.

Performs downtime aggregation across (machine, logged_reason), computes
margin-based softmax raw confidence, and provides Isotonic / Platt
probability calibration with Expected Calibration Error (ECE) calculation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class Candidate:
    """Represents a diagnostic candidate machine-reason pair."""

    machine_id: str
    logged_reason: str
    downtime_min: float
    downtime_share: float


@dataclass
class DiagnosisResult:
    """Diagnostic output containing ranked candidates and confidence metrics."""

    top_candidate: Optional[Tuple[str, str]]  # (machine_id, logged_reason)
    top_downtime_min: float
    runner_up_candidate: Optional[Tuple[str, str]]
    runner_up_downtime_min: float
    margin_min: float
    margin_ratio: float
    raw_confidence: float
    calibrated_confidence: Optional[float]
    ranked_candidates: List[Candidate]
    total_valid_downtime_min: float
    events_analyzed: int
    events_skipped: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize diagnosis output to dictionary."""
        return {
            "top_candidate": list(self.top_candidate) if self.top_candidate else None,
            "top_downtime_min": round(self.top_downtime_min, 2),
            "runner_up_candidate": list(self.runner_up_candidate) if self.runner_up_candidate else None,
            "runner_up_downtime_min": round(self.runner_up_downtime_min, 2),
            "margin_min": round(self.margin_min, 2),
            "margin_ratio": round(self.margin_ratio, 4),
            "raw_confidence": round(self.raw_confidence, 4),
            "calibrated_confidence": round(self.calibrated_confidence, 4) if self.calibrated_confidence is not None else None,
            "total_valid_downtime_min": round(self.total_valid_downtime_min, 2),
            "events_analyzed": self.events_analyzed,
            "events_skipped": self.events_skipped,
            "candidates_count": len(self.ranked_candidates),
        }


def compute_raw_confidence(
    top_score: float,
    scores: List[float],
    temperature: float = 1.0,
    epsilon: float = 1e-6,
) -> float:
    """Compute margin-based softmax raw confidence."""
    if len(scores) <= 1 or top_score <= 0:
        return 1.0

    s1 = top_score
    s2 = scores[1]
    margin = s1 - s2

    rest_scores = scores[1:]
    mean_rest = float(np.mean(rest_scores)) if len(rest_scores) > 0 else 0.0
    scale = max(mean_rest, epsilon)
    z = margin / scale

    try:
        conf = 1.0 / (1.0 + math.exp(-z / max(0.01, temperature)))
    except OverflowError:
        conf = 1.0 if z > 0 else 0.5

    return float(np.clip(conf, 0.5, 1.0))


def diagnose(
    df: pd.DataFrame,
    temperature: float = 1.0,
    calibrator: Optional["ConfidenceCalibrator"] = None,
) -> DiagnosisResult:
    """Execute shared diagnosis on machine event logs."""
    if df.empty:
        return DiagnosisResult(
            top_candidate=None,
            top_downtime_min=0.0,
            runner_up_candidate=None,
            runner_up_downtime_min=0.0,
            margin_min=0.0,
            margin_ratio=0.0,
            raw_confidence=0.0,
            calibrated_confidence=None,
            ranked_candidates=[],
            total_valid_downtime_min=0.0,
            events_analyzed=0,
            events_skipped=0,
        )

    valid_mask = df["end_ts"].notna()
    if "duration_min" in df.columns:
        valid_mask = valid_mask & df["duration_min"].notna() & (df["duration_min"] > 0)

    events_analyzed = int(valid_mask.sum())
    events_skipped = int((~valid_mask).sum())

    valid_df = df[valid_mask].copy()

    if valid_df.empty:
        return DiagnosisResult(
            top_candidate=None,
            top_downtime_min=0.0,
            runner_up_candidate=None,
            runner_up_downtime_min=0.0,
            margin_min=0.0,
            margin_ratio=0.0,
            raw_confidence=0.0,
            calibrated_confidence=None,
            ranked_candidates=[],
            total_valid_downtime_min=0.0,
            events_analyzed=events_analyzed,
            events_skipped=events_skipped,
        )

    if "duration_min" not in valid_df.columns or valid_df["duration_min"].isna().any():
        valid_df["start_dt"] = pd.to_datetime(valid_df["start_ts"])
        valid_df["end_dt"] = pd.to_datetime(valid_df["end_ts"])
        valid_df["duration_min"] = (valid_df["end_dt"] - valid_df["start_dt"]).dt.total_seconds() / 60.0

    grouped = (
        valid_df.groupby(["machine_id", "logged_reason"])["duration_min"]
        .sum()
        .sort_values(ascending=False)
    )

    total_downtime = float(grouped.sum())
    candidates: List[Candidate] = []
    scores: List[float] = []

    for (m_id, reason), d_min in grouped.items():
        d_val = float(d_min)
        share = d_val / total_downtime if total_downtime > 0 else 0.0
        candidates.append(
            Candidate(
                machine_id=str(m_id),
                logged_reason=str(reason),
                downtime_min=d_val,
                downtime_share=share,
            )
        )
        scores.append(d_val)

    top_cand = (candidates[0].machine_id, candidates[0].logged_reason)
    top_score = scores[0]

    if len(candidates) > 1:
        runner_up = (candidates[1].machine_id, candidates[1].logged_reason)
        runner_up_score = scores[1]
        margin_min = top_score - runner_up_score
        margin_ratio = margin_min / top_score if top_score > 0 else 0.0
    else:
        runner_up = None
        runner_up_score = 0.0
        margin_min = top_score
        margin_ratio = 1.0

    raw_conf = compute_raw_confidence(top_score, scores, temperature=temperature)
    calibrated_conf = calibrator.predict(raw_conf) if calibrator is not None else None

    return DiagnosisResult(
        top_candidate=top_cand,
        top_downtime_min=top_score,
        runner_up_candidate=runner_up,
        runner_up_downtime_min=runner_up_score,
        margin_min=margin_min,
        margin_ratio=margin_ratio,
        raw_confidence=raw_conf,
        calibrated_confidence=calibrated_conf,
        ranked_candidates=candidates,
        total_valid_downtime_min=total_downtime,
        events_analyzed=events_analyzed,
        events_skipped=events_skipped,
    )


class ConfidenceCalibrator:
    """Fits post-hoc probability calibration (Isotonic Regression or Platt scaling)."""

    def __init__(self, method: str = "isotonic") -> None:
        self.method = method.lower()
        if self.method not in ("isotonic", "platt"):
            raise ValueError(f"Unsupported calibration method: {method}. Use 'isotonic' or 'platt'.")
        self.model: Any = None
        self.is_fitted: bool = False

    def fit(self, confidences: np.ndarray | List[float], labels: np.ndarray | List[int]) -> "ConfidenceCalibrator":
        """Fit calibration model on validation set."""
        x = np.asarray(confidences, dtype=float)
        y = np.asarray(labels, dtype=float)

        if len(x) == 0:
            raise ValueError("Cannot fit calibrator on empty dataset.")

        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
            self.model.fit(x, y)
        else:
            self.model = LogisticRegression(C=1.0, solver="lbfgs")
            self.model.fit(x.reshape(-1, 1), y.astype(int))

        self.is_fitted = True
        return self

    def predict(self, raw_confidence: float) -> float:
        """Transform a raw confidence score into a calibrated probability."""
        if not self.is_fitted:
            raise RuntimeError("Calibrator is not fitted yet. Call fit() first.")

        if self.method == "isotonic":
            val = float(self.model.predict(np.array([raw_confidence]))[0])
        else:
            val = float(self.model.predict_proba(np.array([[raw_confidence]]))[0, 1])

        return float(np.clip(val, 0.01, 0.99))

    def predict_batch(self, confidences: np.ndarray | List[float]) -> np.ndarray:
        """Transform an array of raw confidences into calibrated probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Calibrator is not fitted yet. Call fit() first.")
        x = np.asarray(confidences, dtype=float)
        if self.method == "isotonic":
            preds = self.model.predict(x)
        else:
            preds = self.model.predict_proba(x.reshape(-1, 1))[:, 1]
        return np.clip(preds, 0.01, 0.99)


def compute_ece(
    confidences: np.ndarray | List[float],
    labels: np.ndarray | List[int],
    n_bins: int = 10,
) -> Tuple[float, Dict[str, Any]]:
    """Compute Expected Calibration Error (ECE) and reliability diagram bin metrics."""
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(labels, dtype=float)
    n = len(conf)

    if n == 0:
        return 0.0, {"bin_edges": [], "bin_accs": [], "bin_confs": [], "bin_counts": []}

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    bin_accs: List[float] = []
    bin_confs: List[float] = []
    bin_counts: List[int] = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (conf >= low) & (conf <= high)
        else:
            in_bin = (conf >= low) & (conf < high)

        count = int(np.sum(in_bin))
        bin_counts.append(count)

        if count > 0:
            bin_acc = float(np.mean(acc[in_bin]))
            bin_conf = float(np.mean(conf[in_bin]))
            bin_accs.append(round(bin_acc, 4))
            bin_confs.append(round(bin_conf, 4))
            ece += (count / n) * abs(bin_acc - bin_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(round((low + high) / 2.0, 4))

    diagram_data = {
        "bin_edges": [round(float(b), 2) for b in bin_edges],
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "total_samples": n,
    }

    return float(round(ece, 4)), diagram_data


def run_sanity_check(
    scenarios_dir: str = "data/sample_scenarios",
    num_eval: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """Run a quick sanity check evaluating diagnoser accuracy on clean data and 30% defects."""
    import json
    from pathlib import Path
    from faultlens.defects import ALLOWED_DEFECTS, inject_defect
    from faultlens.generator import Scenario

    base_dir = Path(scenarios_dir)
    splits_file = base_dir / "splits.json"

    if splits_file.is_file():
        with open(splits_file, "r", encoding="utf-8") as f:
            splits = json.load(f)
        scenario_ids = splits.get("val", [])[:num_eval]
    else:
        scenario_ids = [d.name for d in base_dir.iterdir() if d.is_dir()][:num_eval] if base_dir.is_dir() else []

    if not scenario_ids:
        from faultlens.generator import generate_scenario
        scenarios = [generate_scenario(seed=1000 + i) for i in range(num_eval)]
    else:
        scenarios = [Scenario.load(base_dir / sc_id) for sc_id in scenario_ids]

    conditions = [("Clean (0% defect)", None, 0.0)]
    for d_name in ALLOWED_DEFECTS:
        conditions.append((f"{d_name} (30% severity)", d_name, 0.30))

    results_table: List[Dict[str, Any]] = []

    for cond_label, defect_type, sev in conditions:
        correct = 0
        total = 0
        raw_confs: List[float] = []

        for idx, sc in enumerate(scenarios):
            df_input = sc.log
            if defect_type is not None:
                df_input, _ = inject_defect(
                    df_input,
                    defect_type=defect_type,
                    severity=sev,
                    seed=seed + idx,
                )

            diag = diagnose(df_input)
            is_correct = (diag.top_candidate == sc.truth)
            if is_correct:
                correct += 1
            total += 1
            raw_confs.append(diag.raw_confidence)

        acc = (correct / total) * 100.0 if total > 0 else 0.0
        mean_conf = float(np.mean(raw_confs)) * 100.0 if raw_confs else 0.0
        results_table.append({
            "Condition": cond_label,
            "Scenarios": total,
            "Top-1 Accuracy (%)": round(acc, 1),
            "Mean Raw Conf (%)": round(mean_conf, 1),
            "Correct": correct,
            "Incorrect": total - correct,
        })

    summary_df = pd.DataFrame(results_table)
    return summary_df
