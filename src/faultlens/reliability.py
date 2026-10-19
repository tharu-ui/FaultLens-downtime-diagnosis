"""FaultLens Learned Reliability, Attribution & Evaluation Module (Session 3.5).

Builds a feature table per (scenario, defect, severity) combining:
- Shared diagnoser `raw_confidence`
- Perturbation stability (`top_stability`, `top3_flip_rate`, `margin_ci_width`)
- Six data-quality sub-scores (`completeness`, `reason_validity`, `duplication`,
  `timestamp_consistency`, `micro_stop_plausibility`, `alarm_reason_agreement`)
- Log repair outputs (`top_candidate_changed`, `margin_change`, and repaired row counts)

Provides:
1. 5-fold scenario-grouped cross-validation (`GroupKFold`) to report out-of-fold results
   without ever splitting a scenario across train and score folds.
2. Final model fitting strictly on `TRAIN` scenarios, post-hoc calibration on `VALIDATION`,
   and selection of ONE threshold on `VALIDATION` to hit the configured target coverage.
3. Single-feature ablation (`raw_confidence` only) through the exact same code path.
4. Defect attribution accuracy metric (`is_defect_attributed`, `compute_attribution_accuracy`).
5. Extra synthetic TRAIN scenario generator (`generate_extra_train_scenarios`) and fresh held-out
   scenario generator (`generate_fresh_test_scenarios`, 200 scenarios, seed 4242).
6. Validation sanity table (`run_session3_5_validation_table`) on `VALIDATION` only.
7. One-shot locked final evaluation (`run_final_fresh_test_experiment`) on `fresh_test`.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from faultlens.defects import ALLOWED_DEFECTS, inject_defect
from faultlens.diagnoser import ConfidenceCalibrator, DiagnosisResult, diagnose
from faultlens.generator import Scenario, generate_scenario, load_generator_config
from faultlens.policy import TUNED_PARAMS_PATH
from faultlens.quality import QualityReport, evaluate_data_quality, load_quality_baseline
from faultlens.repair import RepairReport, repair_log
from faultlens.stability import StabilityReport, evaluate_stability

RELIABILITY_MODELS_PATH = Path("results/reliability_models.json")
FINAL_LOCK_PATH = Path("results/final_run.lock")
DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiment.yaml")
DEFAULT_SCENARIOS_DIR = Path("data/sample_scenarios")
FRESH_TEST_SCENARIOS_DIR = Path("data/sample_scenarios/fresh_test")

QUALITY_SUB_SCORE_COLUMNS: List[str] = [
    "completeness",
    "reason_validity",
    "duplication",
    "timestamp_consistency",
    "micro_stop_plausibility",
    "alarm_reason_agreement",
]

FULL_FEATURE_COLUMNS: List[str] = [
    "raw_confidence",
    "top_stability",
    "top3_flip_rate",
    "completeness",
    "reason_validity",
    "duplication",
    "timestamp_consistency",
    "micro_stop_plausibility",
    "alarm_reason_agreement",
    "margin_ci_width",
    "top_candidate_changed",
    "margin_change",
    "merged_duplicates_count",
    "imputed_durations_count",
    "reinferred_reasons_count",
    "total_repaired_rows",
]

CONFIDENCE_ONLY_FEATURE_COLUMNS: List[str] = [
    "raw_confidence",
]

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
    """Check whether the quality report's worst_component matches the injected defect.

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


def get_feature_columns(feature_mode: str = "full") -> List[str]:
    """Return ordered feature column list for 'full' or 'confidence_only' mode."""
    mode = feature_mode.strip().lower()
    if mode == "full":
        return list(FULL_FEATURE_COLUMNS)
    if mode in ("confidence_only", "raw_confidence"):
        return list(CONFIDENCE_ONLY_FEATURE_COLUMNS)
    raise ValueError(f"Unsupported feature_mode: {feature_mode}. Use 'full' or 'confidence_only'.")


def get_target_coverage_from_config(config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG) -> float:
    """Read target coverage (`policy.target_coverage` or `policy.min_coverage`) from experiment.yaml."""
    path = Path(config_path)
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            if isinstance(cfg, dict):
                rel = cfg.get("reliability", {})
                if isinstance(rel, dict) and "target_coverage" in rel:
                    return float(rel["target_coverage"])
                pol = cfg.get("policy", {})
                if isinstance(pol, dict):
                    if "target_coverage" in pol:
                        return float(pol["target_coverage"])
                    if "min_coverage" in pol:
                        return float(pol["min_coverage"])
        except Exception:
            pass
    return 0.65


def extract_reliability_features(
    df: pd.DataFrame,
    diag: Optional[DiagnosisResult] = None,
    quality: Optional[QualityReport] = None,
    stability: Optional[StabilityReport] = None,
    repair_rep: Optional[RepairReport] = None,
    num_perturbations: int = 100,
    seed: int = 42,
    baseline_sub_scores: Optional[Dict[str, float]] = None,
    expected_micro_stop_rates: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], DiagnosisResult, QualityReport, StabilityReport, RepairReport]:
    """Extract the 16-dimensional NaN-free reliability feature dictionary for a single log."""
    if diag is None:
        diag = diagnose(df)
    if quality is None:
        quality = evaluate_data_quality(
            df,
            baseline_sub_scores=baseline_sub_scores,
            expected_micro_stop_rates=expected_micro_stop_rates,
        )
    if stability is None:
        stability = evaluate_stability(df, num_perturbations=num_perturbations, seed=seed)
    if repair_rep is None:
        repair_rep = repair_log(df)

    margin_ci_width = float(stability.margin_ci_upper_95 - stability.margin_ci_lower_95)

    features: Dict[str, float] = {
        "raw_confidence": float(diag.raw_confidence),
        "top_stability": float(stability.top_stability),
        "top3_flip_rate": float(stability.top3_flip_rate),
        "completeness": float(quality.sub_scores.get("completeness", 0.0)),
        "reason_validity": float(quality.sub_scores.get("reason_validity", 0.0)),
        "duplication": float(quality.sub_scores.get("duplication", 0.0)),
        "timestamp_consistency": float(quality.sub_scores.get("timestamp_consistency", 0.0)),
        "micro_stop_plausibility": float(quality.sub_scores.get("micro_stop_plausibility", 0.0)),
        "alarm_reason_agreement": float(quality.sub_scores.get("alarm_reason_agreement", 0.0)),
        "margin_ci_width": margin_ci_width,
        "top_candidate_changed": 1.0 if repair_rep.top_candidate_changed else 0.0,
        "margin_change": float(repair_rep.margin_change),
        "merged_duplicates_count": float(repair_rep.merged_duplicates_count),
        "imputed_durations_count": float(repair_rep.imputed_durations_count),
        "reinferred_reasons_count": float(repair_rep.reinferred_reasons_count),
        "total_repaired_rows": float(repair_rep.total_repaired_rows),
    }

    for k, v in list(features.items()):
        if not np.isfinite(v):
            features[k] = 0.0

    return features, diag, quality, stability, repair_rep


def build_feature_table(
    scenarios: Sequence[Scenario],
    conditions: Optional[Sequence[Tuple[str, Optional[str], float]]] = None,
    num_perturbations: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a NaN-free feature table across scenarios x (defect, severity) conditions."""
    if conditions is None:
        cond_list: List[Tuple[str, Optional[str], float]] = [("clean", None, 0.0)]
        for d_name in ALLOWED_DEFECTS:
            for sev in (0.10, 0.20, 0.30):
                cond_list.append((d_name, d_name, sev))
        conditions = cond_list

    baseline_subs, exp_micro = load_quality_baseline()
    rows: List[Dict[str, Any]] = []

    for sc_idx, sc in enumerate(scenarios):
        for c_idx, (cond_label, defect_type, sev) in enumerate(conditions):
            df_in = sc.log
            if defect_type is not None and sev > 0.0:
                inj_seed = seed + sc_idx * 1000 + c_idx * 17
                df_in, _ = inject_defect(df_in, defect_type=defect_type, severity=sev, seed=inj_seed)

            stab_seed = seed + 50000 + sc_idx * 1000 + c_idx * 17
            feat_dict, diag, q_rep, _, _ = extract_reliability_features(
                df_in,
                num_perturbations=num_perturbations,
                seed=stab_seed,
                baseline_sub_scores=baseline_subs,
                expected_micro_stop_rates=exp_micro,
            )
            is_correct = int(diag.top_candidate == sc.truth)

            row: Dict[str, Any] = {
                "scenario_id": sc.scenario_id,
                "defect": cond_label,
                "severity": float(sev),
                "is_correct": is_correct,
                "worst_component": q_rep.worst_component,
                "quality_score": q_rep.overall_score,
            }
            row.update(feat_dict)
            rows.append(row)

    df_out = pd.DataFrame(rows)
    for col in FULL_FEATURE_COLUMNS:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce").fillna(0.0)
    return df_out


class LearnedReliabilityModel:
    """Logistic regression reliability model with validation Platt calibration and single threshold.

    Both the full evidence model (`feature_mode="full"`) and the raw-confidence ablation
    (`feature_mode="confidence_only"`) execute through this identical class and code path.
    """

    def __init__(
        self,
        feature_mode: str = "full",
        threshold: float = 0.65,
        buffer: float = 0.05,
    ) -> None:
        self.feature_mode = feature_mode.strip().lower()
        self.feature_columns = get_feature_columns(self.feature_mode)
        self.threshold = float(threshold)
        self.buffer = float(buffer)

        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_scale: Optional[np.ndarray] = None
        self.coef: Optional[np.ndarray] = None
        self.intercept: float = 0.0

        self.cal_coef: float = 1.0
        self.cal_intercept: float = 0.0
        self.is_fitted: bool = False
        self.is_calibrated: bool = False
        self.validation_calibrated_probs: Optional[np.ndarray] = None

    def _extract_matrix(self, feature_df_or_dict: pd.DataFrame | Dict[str, float]) -> np.ndarray:
        if isinstance(feature_df_or_dict, dict):
            arr = np.array([[float(feature_df_or_dict.get(c, 0.0)) for c in self.feature_columns]], dtype=float)
        else:
            arr = feature_df_or_dict[self.feature_columns].to_numpy(dtype=float)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def fit_on_train(self, train_df: pd.DataFrame) -> "LearnedReliabilityModel":
        """Fit StandardScaler and LogisticRegression strictly on TRAIN feature rows."""
        x_raw = self._extract_matrix(train_df)
        y = train_df["is_correct"].to_numpy(dtype=int)

        if len(x_raw) == 0:
            raise ValueError("Cannot fit LearnedReliabilityModel on empty train_df.")

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_raw)
        self.scaler_mean = np.asarray(scaler.mean_, dtype=float)
        self.scaler_scale = np.where(
            np.asarray(scaler.scale_, dtype=float) == 0.0,
            1.0,
            np.asarray(scaler.scale_, dtype=float),
        )

        if len(np.unique(y)) < 2:
            self.coef = np.zeros(len(self.feature_columns), dtype=float)
            self.intercept = 2.0 if int(y[0]) == 1 else -2.0
        else:
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
            clf.fit(x_scaled, y)
            self.coef = np.asarray(clf.coef_[0], dtype=float)
            self.intercept = float(clf.intercept_[0])

        self.is_fitted = True
        return self

    def predict_raw_prob_batch(self, feature_df_or_dict: pd.DataFrame | Dict[str, float]) -> np.ndarray:
        """Predict uncalibrated logistic probabilities in [0.01, 0.99]."""
        if not self.is_fitted or self.scaler_mean is None or self.scaler_scale is None or self.coef is None:
            raise RuntimeError("LearnedReliabilityModel is not fitted yet.")
        x_raw = self._extract_matrix(feature_df_or_dict)
        x_scaled = (x_raw - self.scaler_mean) / self.scaler_scale
        logits = np.clip(x_scaled @ self.coef + self.intercept, -30.0, 30.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.clip(probs, 0.01, 0.99)

    def calibrate_and_select_threshold_on_validation(
        self,
        val_df: pd.DataFrame,
        target_coverage: float = 0.65,
    ) -> float:
        """Fit Platt scaling and select ONE operating threshold strictly on VALIDATION data."""
        raw_probs = self.predict_raw_prob_batch(val_df)
        y_val = val_df["is_correct"].to_numpy(dtype=int)

        if len(np.unique(y_val)) >= 2:
            cal_clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=42)
            cal_clf.fit(raw_probs.reshape(-1, 1), y_val)
            self.cal_coef = float(cal_clf.coef_[0, 0])
            self.cal_intercept = float(cal_clf.intercept_[0])
            if self.cal_coef < 0.0:
                self.cal_coef = 1.0
                self.cal_intercept = 0.0
        else:
            self.cal_coef = 1.0
            self.cal_intercept = 0.0

        self.is_calibrated = True
        cal_probs = self.predict_calibrated_prob_batch(val_df)
        self.validation_calibrated_probs = cal_probs
        self.threshold = self.find_threshold_for_coverage(cal_probs, target_coverage=target_coverage)
        return self.threshold

    def predict_calibrated_prob_batch(self, feature_df_or_dict: pd.DataFrame | Dict[str, float]) -> np.ndarray:
        """Predict validation-calibrated probabilities in [0.01, 0.99]."""
        raw_probs = self.predict_raw_prob_batch(feature_df_or_dict)
        if not self.is_calibrated:
            return raw_probs
        logits = np.clip(self.cal_coef * raw_probs + self.cal_intercept, -30.0, 30.0)
        cal_probs = 1.0 / (1.0 + np.exp(-logits))
        return np.clip(cal_probs, 0.01, 0.99)

    def predict_calibrated_prob(self, features: Dict[str, float]) -> float:
        """Predict calibrated probability for a single feature dictionary."""
        return float(self.predict_calibrated_prob_batch(features)[0])

    def find_threshold_for_coverage(
        self,
        calibrated_probs: np.ndarray | Sequence[float],
        target_coverage: float,
    ) -> float:
        """Choose threshold on calibrated probabilities to match target_coverage as closely as possible."""
        arr = np.asarray(calibrated_probs, dtype=float)
        if len(arr) == 0:
            return 0.50
        if target_coverage <= 0.0:
            return float(np.max(arr) + 1e-6)
        if target_coverage >= 1.0:
            return float(np.min(arr) - 1e-6)

        cand_thresholds = np.unique(
            np.concatenate([
                np.quantile(arr, np.linspace(0.0, 1.0, len(arr) + 1)),
                arr,
                arr + 1e-7,
                arr - 1e-7,
            ])
        )
        best_t = float(np.percentile(arr, max(0.0, min(100.0, (1.0 - target_coverage) * 100.0))))
        best_diff = abs(float(np.mean(arr >= best_t)) - target_coverage)
        for t_val in cand_thresholds:
            cov = float(np.mean(arr >= t_val))
            diff = abs(cov - target_coverage)
            if diff < best_diff:
                best_diff = diff
                best_t = float(t_val)
        return best_t

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model parameters, calibrator, and threshold to JSON-compatible dict."""
        return {
            "feature_mode": self.feature_mode,
            "feature_columns": self.feature_columns,
            "threshold": round(float(self.threshold), 6),
            "buffer": round(float(self.buffer), 4),
            "scaler_mean": [float(x) for x in (self.scaler_mean if self.scaler_mean is not None else [])],
            "scaler_scale": [float(x) for x in (self.scaler_scale if self.scaler_scale is not None else [])],
            "coef": [float(x) for x in (self.coef if self.coef is not None else [])],
            "intercept": float(self.intercept),
            "cal_coef": float(self.cal_coef),
            "cal_intercept": float(self.cal_intercept),
            "is_fitted": self.is_fitted,
            "is_calibrated": self.is_calibrated,
            "validation_calibrated_probs": [
                round(float(x), 6)
                for x in (self.validation_calibrated_probs if self.validation_calibrated_probs is not None else [])
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearnedReliabilityModel":
        """Reconstruct a fitted & calibrated LearnedReliabilityModel from dict without retraining."""
        obj = cls(
            feature_mode=str(data["feature_mode"]),
            threshold=float(data["threshold"]),
            buffer=float(data.get("buffer", 0.05)),
        )
        obj.feature_columns = [str(c) for c in data["feature_columns"]]
        obj.scaler_mean = np.asarray(data["scaler_mean"], dtype=float)
        obj.scaler_scale = np.asarray(data["scaler_scale"], dtype=float)
        obj.coef = np.asarray(data["coef"], dtype=float)
        obj.intercept = float(data["intercept"])
        obj.cal_coef = float(data.get("cal_coef", 1.0))
        obj.cal_intercept = float(data.get("cal_intercept", 0.0))
        obj.is_fitted = bool(data.get("is_fitted", True))
        obj.is_calibrated = bool(data.get("is_calibrated", True))
        val_probs = data.get("validation_calibrated_probs", [])
        obj.validation_calibrated_probs = np.asarray(val_probs, dtype=float) if val_probs else None
        return obj


def run_grouped_cross_validation(
    feature_df: pd.DataFrame,
    feature_mode: str = "full",
    n_splits: int = 5,
    target_coverage: float = 0.65,
) -> Dict[str, Any]:
    """Run 5-fold cross-validation grouped by `scenario_id` (GroupKFold) for out-of-fold reporting only.

    Guarantees that no scenario_id is ever split across train and score folds.
    """
    unique_scenarios = feature_df["scenario_id"].unique()
    actual_splits = min(n_splits, len(unique_scenarios))
    if actual_splits < 2:
        raise ValueError("Need at least 2 distinct scenarios for GroupKFold cross-validation.")

    gkf = GroupKFold(n_splits=actual_splits)
    groups = feature_df["scenario_id"].to_numpy(dtype=str)

    oof_probs = np.zeros(len(feature_df), dtype=float)
    oof_answered = np.zeros(len(feature_df), dtype=bool)
    fold_summaries: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, score_idx) in enumerate(gkf.split(feature_df, feature_df["is_correct"], groups=groups)):
        train_fold = feature_df.iloc[train_idx].reset_index(drop=True)
        score_fold = feature_df.iloc[score_idx].reset_index(drop=True)

        train_sc_set = set(train_fold["scenario_id"].unique())
        score_sc_set = set(score_fold["scenario_id"].unique())
        if not train_sc_set.isdisjoint(score_sc_set):
            raise RuntimeError(f"Scenario leakage across GroupKFold fold {fold_idx}!")

        model = LearnedReliabilityModel(feature_mode=feature_mode)
        model.fit_on_train(train_fold)

        # Calibrate and choose fold threshold on the training fold only (never touching score_fold)
        model.calibrate_and_select_threshold_on_validation(train_fold, target_coverage=target_coverage)

        fold_probs = model.predict_calibrated_prob_batch(score_fold)
        fold_ans = fold_probs >= model.threshold

        oof_probs[score_idx] = fold_probs
        oof_answered[score_idx] = fold_ans

        y_score = score_fold["is_correct"].to_numpy(dtype=bool)
        fold_cov = float(np.mean(fold_ans))
        fold_cw = float(np.mean(fold_ans & (~y_score)))
        fold_acc = float(np.mean(fold_ans & y_score))

        fold_summaries.append({
            "fold": fold_idx,
            "train_scenarios_count": len(train_sc_set),
            "score_scenarios_count": len(score_sc_set),
            "train_scenario_ids": sorted(list(train_sc_set)),
            "score_scenario_ids": sorted(list(score_sc_set)),
            "threshold": round(model.threshold, 4),
            "coverage": round(fold_cov, 4),
            "confidently_wrong_rate": round(fold_cw, 4),
            "accuracy": round(fold_acc, 4),
        })

    y_all = feature_df["is_correct"].to_numpy(dtype=bool)
    oof_cov = float(np.mean(oof_answered))
    oof_cw = float(np.mean(oof_answered & (~y_all)))
    oof_acc = float(np.mean(oof_answered & y_all))
    oof_sel_acc = float(np.mean(y_all[oof_answered])) if np.any(oof_answered) else 0.0

    return {
        "feature_mode": feature_mode,
        "n_splits": actual_splits,
        "oof_coverage": round(oof_cov, 4),
        "oof_confidently_wrong_rate": round(oof_cw, 4),
        "oof_accuracy": round(oof_acc, 4),
        "oof_selective_accuracy": round(oof_sel_acc, 4),
        "folds": fold_summaries,
    }


def train_and_save_reliability_models(
    scenarios_dir: str | Path = DEFAULT_SCENARIOS_DIR,
    extra_train_dir: Optional[str | Path] = "data/sample_scenarios/extra_train",
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG,
    out_path: str | Path = RELIABILITY_MODELS_PATH,
    num_perturbations: int = 100,
    target_coverage: Optional[float] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train final models on TRAIN, calibrate and choose ONE threshold on VALIDATION, and save to disk.

    Never reads or accesses `splits["test"]`.
    """
    base_dir = Path(scenarios_dir)
    splits_file = base_dir / "splits.json"
    if not splits_file.is_file():
        raise FileNotFoundError(f"Splits manifest not found at {splits_file}")

    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # Read strictly train and val splits; never read splits["test"]
    train_ids: List[str] = list(splits["train"])
    val_ids: List[str] = list(splits["val"])
    if not set(train_ids).isdisjoint(set(val_ids)):
        raise RuntimeError("Train and validation scenario IDs must be disjoint!")

    train_scenarios: List[Scenario] = [Scenario.load(base_dir / sc_id) for sc_id in train_ids]
    extra_train_ids: List[str] = []

    if extra_train_dir is not None:
        ext_path = Path(extra_train_dir)
        ext_manifest = ext_path / "extra_train_manifest.json"
        if ext_manifest.is_file():
            with open(ext_manifest, "r", encoding="utf-8") as f:
                ext_data = json.load(f)
            for ext_id in ext_data.get("scenario_ids", []):
                sc_p = ext_path / ext_id
                if (sc_p / "scenario.json").is_file():
                    train_scenarios.append(Scenario.load(sc_p))
                    extra_train_ids.append(str(ext_id))

    val_scenarios: List[Scenario] = [Scenario.load(base_dir / sc_id) for sc_id in val_ids]

    if target_coverage is None:
        target_coverage = get_target_coverage_from_config(config_path)

    train_feat_df = build_feature_table(
        train_scenarios,
        num_perturbations=num_perturbations,
        seed=seed,
    )
    val_feat_df = build_feature_table(
        val_scenarios,
        num_perturbations=num_perturbations,
        seed=seed + 10000,
    )

    # 1. GroupKFold out-of-fold CV on TRAIN scenarios for reporting only
    cv_full = run_grouped_cross_validation(
        train_feat_df,
        feature_mode="full",
        n_splits=5,
        target_coverage=target_coverage,
    )
    cv_conf_only = run_grouped_cross_validation(
        train_feat_df,
        feature_mode="confidence_only",
        n_splits=5,
        target_coverage=target_coverage,
    )

    # 2. Final saved models: fit on TRAIN, calibrate and choose ONE threshold on VALIDATION
    full_model = LearnedReliabilityModel(feature_mode="full")
    full_model.fit_on_train(train_feat_df)
    full_thresh = full_model.calibrate_and_select_threshold_on_validation(
        val_feat_df,
        target_coverage=target_coverage,
    )

    conf_only_model = LearnedReliabilityModel(feature_mode="confidence_only")
    conf_only_model.fit_on_train(train_feat_df)
    conf_only_thresh = conf_only_model.calibrate_and_select_threshold_on_validation(
        val_feat_df,
        target_coverage=target_coverage,
    )

    val_full_probs = full_model.predict_calibrated_prob_batch(val_feat_df)
    val_full_ans = val_full_probs >= full_thresh
    val_y = val_feat_df["is_correct"].to_numpy(dtype=bool)

    val_conf_probs = conf_only_model.predict_calibrated_prob_batch(val_feat_df)
    val_conf_ans = val_conf_probs >= conf_only_thresh

    payload: Dict[str, Any] = {
        "synthetic_disclosure": "Trained and calibrated on 100% synthetic scenarios (TRAIN for fitting, VALIDATION for calibration and threshold).",
        "train_split_used": "train",
        "calibration_split_used": "val",
        "threshold_split_used": "val",
        "target_coverage": round(float(target_coverage), 4),
        "train_scenario_ids": train_ids,
        "extra_train_scenario_ids": extra_train_ids,
        "val_scenario_ids": val_ids,
        "num_train_scenarios": len(train_scenarios),
        "num_val_scenarios": len(val_scenarios),
        "models": {
            "full": full_model.to_dict(),
            "confidence_only": conf_only_model.to_dict(),
        },
        "validation_summary": {
            "full": {
                "threshold": round(float(full_thresh), 4),
                "coverage": round(float(np.mean(val_full_ans)), 4),
                "confidently_wrong_rate": round(float(np.mean(val_full_ans & (~val_y))), 4),
                "accuracy": round(float(np.mean(val_full_ans & val_y)), 4),
            },
            "confidence_only": {
                "threshold": round(float(conf_only_thresh), 4),
                "coverage": round(float(np.mean(val_conf_ans)), 4),
                "confidently_wrong_rate": round(float(np.mean(val_conf_ans & (~val_y))), 4),
                "accuracy": round(float(np.mean(val_conf_ans & val_y)), 4),
            },
        },
        "grouped_cv_out_of_fold": {
            "full": cv_full,
            "confidence_only": cv_conf_only,
        },
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


def load_saved_reliability_model(
    feature_mode: str = "full",
    model_path: str | Path = RELIABILITY_MODELS_PATH,
) -> LearnedReliabilityModel:
    """Load saved LearnedReliabilityModel, calibrator, and validation threshold without retraining."""
    p = Path(model_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Saved reliability model file not found at '{p}'. Run `python scripts/run_all.py train_reliability` first."
        )
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    mode_key = "full" if feature_mode.strip().lower() == "full" else "confidence_only"
    models_dict = data.get("models", {})
    if mode_key not in models_dict:
        raise KeyError(f"Model mode '{mode_key}' missing from {p}")

    model_entry = models_dict[mode_key]
    if "threshold" not in model_entry or not model_entry.get("coef"):
        raise ValueError(f"Saved model '{mode_key}' in {p} is missing weights or threshold.")

    return LearnedReliabilityModel.from_dict(model_entry)


def generate_extra_train_scenarios(
    num_scenarios: int = 120,
    master_seed: int = 1337,
    out_dir: str | Path = "data/sample_scenarios/extra_train",
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Scenario], Dict[str, Any]]:
    """Task 6: Generate extra synthetic TRAIN scenarios with a new seed, saved separately from splits.json."""
    if config is None:
        config = load_generator_config()

    rng = np.random.default_rng(master_seed)
    scenario_seeds = rng.integers(low=1000000, high=1999999, size=num_scenarios)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    scenarios: List[Scenario] = []
    for idx, s_seed in enumerate(scenario_seeds):
        sc_id = f"extra_train_{idx:03d}"
        sc = generate_scenario(seed=int(s_seed), scenario_id=sc_id, config=config)
        sc.save(out_path)
        scenarios.append(sc)

    manifest = {
        "split": "extra_train",
        "synthetic_disclosure": "100% synthetic machine-event logs generated with master_seed=1337.",
        "master_seed": master_seed,
        "num_scenarios": len(scenarios),
        "scenario_ids": [s.scenario_id for s in scenarios],
    }
    manifest_file = out_path / "extra_train_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return scenarios, manifest


def generate_fresh_test_scenarios(
    num_scenarios: int = 200,
    master_seed: int = 4242,
    out_dir: str | Path = FRESH_TEST_SCENARIOS_DIR,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Scenario], Dict[str, Any]]:
    """Task 7a: Generate 200 fresh held-out synthetic scenarios with seed 4242 and its own manifest."""
    if config is None:
        config = load_generator_config()

    rng = np.random.default_rng(master_seed)
    scenario_seeds = rng.integers(low=2000000, high=2999999, size=num_scenarios)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    scenarios: List[Scenario] = []
    for idx, s_seed in enumerate(scenario_seeds):
        sc_id = f"fresh_test_{idx:03d}"
        sc = generate_scenario(seed=int(s_seed), scenario_id=sc_id, config=config)
        sc.save(out_path)
        scenarios.append(sc)

    manifest = {
        "split": "fresh_test",
        "synthetic_disclosure": "100% synthetic held-out scenarios generated with master_seed=4242.",
        "master_seed": master_seed,
        "num_scenarios": len(scenarios),
        "scenario_ids": [s.scenario_id for s in scenarios],
    }
    manifest_file = out_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return scenarios, manifest


def run_session3_5_validation_table(
    scenarios_dir: str | Path = DEFAULT_SCENARIOS_DIR,
    extra_train_dir: Optional[str | Path] = Path("data/sample_scenarios/extra_train"),
    model_path: str | Path = RELIABILITY_MODELS_PATH,
    out_csv_path: str | Path = Path("results/tables/session3_5_validation_table.csv"),
    num_perturbations: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Task 8: Generate Session 3.5 validation sanity table on VALIDATION split only.

    Never reads `splits.json["test"]`.
    For clean and each defect at 20% and 30%, compares Frequency, ConfThresh at matched
    coverage, FaultLensWeighted (existing), LearnedFaultLens, and ConfidenceOnlyLearned
    at the same coverage, reporting accuracy, confidently_wrong_rate, coverage, and
    attribution_accuracy, and honestly reporting where the new method does not win.
    """
    base_dir = Path(scenarios_dir)
    splits_file = base_dir / "splits.json"
    if not splits_file.is_file():
        raise FileNotFoundError(f"Splits file not found at {splits_file}")

    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    val_ids: List[str] = list(splits["val"])
    val_scenarios: List[Scenario] = [Scenario.load(base_dir / sc_id) for sc_id in val_ids]

    m_path = Path(model_path)
    if not m_path.is_file():
        train_and_save_reliability_models(
            scenarios_dir=base_dir,
            extra_train_dir=extra_train_dir,
            out_path=m_path,
            num_perturbations=num_perturbations,
            seed=seed,
        )

    full_model = load_saved_reliability_model("full", m_path)
    conf_only_model = load_saved_reliability_model("confidence_only", m_path)

    t_path = Path(TUNED_PARAMS_PATH)
    if t_path.is_file():
        with open(t_path, "r", encoding="utf-8") as f:
            tuned_meta = json.load(f)
        w_meta = tuned_meta.get("weighted_mode_summary", {})
        tau_q_w = float(w_meta.get("tau_q", tuned_meta.get("tau_q", 0.8321)))
        tau_s_w = float(w_meta.get("tau_s", tuned_meta.get("tau_s", 0.40)))
    else:
        tau_q_w = 0.8321
        tau_s_w = 0.40

    # Fit Platt calibrator for ConfidenceThreshold baseline on TRAIN scenarios only
    train_ids: List[str] = list(splits["train"])
    train_scenarios: List[Scenario] = [Scenario.load(base_dir / sc_id) for sc_id in train_ids]
    train_raw_confs: List[float] = []
    train_labels: List[int] = []
    for idx_t, sc_t in enumerate(train_scenarios):
        diag_t = diagnose(sc_t.log)
        train_raw_confs.append(diag_t.raw_confidence)
        train_labels.append(1 if diag_t.top_candidate == sc_t.truth else 0)
        for d_idx, d_name in enumerate(ALLOWED_DEFECTS):
            df_t_def, _ = inject_defect(sc_t.log, defect_type=d_name, severity=0.20, seed=seed + idx_t * 31 + d_idx)
            diag_td = diagnose(df_t_def)
            train_raw_confs.append(diag_td.raw_confidence)
            train_labels.append(1 if diag_td.top_candidate == sc_t.truth else 0)

    platt_cal = ConfidenceCalibrator(method="platt")
    platt_cal.fit(np.asarray(train_raw_confs, dtype=float), np.asarray(train_labels, dtype=int))

    conditions: List[Tuple[str, Optional[str], float]] = [("clean", None, 0.0)]
    for d_name in ALLOWED_DEFECTS:
        for sev in (0.20, 0.30):
            conditions.append((d_name, d_name, sev))

    val_feat_df = build_feature_table(
        val_scenarios,
        conditions=conditions,
        num_perturbations=num_perturbations,
        seed=seed + 10000,
    )

    rows: List[Dict[str, Any]] = []
    non_winning_slices: List[str] = []

    for cond_label, defect_type, sev in conditions:
        sub_df = val_feat_df[
            (val_feat_df["defect"] == cond_label) & (np.isclose(val_feat_df["severity"], sev))
        ].reset_index(drop=True)

        y_corr = sub_df["is_correct"].to_numpy(dtype=bool)
        raw_confs = sub_df["raw_confidence"].to_numpy(dtype=float)
        worst_comps = list(sub_df["worst_component"].astype(str))
        q_scores = sub_df["quality_score"].to_numpy(dtype=float)
        stabs = sub_df["top_stability"].to_numpy(dtype=float)

        # 1. LearnedFaultLens
        lfl_probs = full_model.predict_calibrated_prob_batch(sub_df)
        lfl_ans = lfl_probs >= full_model.threshold
        lfl_cov = float(np.mean(lfl_ans))
        lfl_acc = float(np.mean(lfl_ans & y_corr))
        lfl_cw = float(np.mean(lfl_ans & (~y_corr)))
        lfl_attr = compute_attribution_accuracy(defect_type, worst_comps, has_attribution=True)

        # 2. Frequency
        fr_cov = 1.0
        fr_acc = float(np.mean(y_corr))
        fr_cw = float(np.mean(~y_corr))
        fr_attr = "N/A"

        # 3. FaultLensWeighted (existing)
        flw_ans = (q_scores >= tau_q_w) & (stabs >= tau_s_w)
        flw_cov = float(np.mean(flw_ans))
        flw_acc = float(np.mean(flw_ans & y_corr))
        flw_cw = float(np.mean(flw_ans & (~y_corr)))
        flw_attr = compute_attribution_accuracy(defect_type, worst_comps, has_attribution=True)

        # 4. ConfThresh at matched coverage
        ct_cal_probs = platt_cal.predict_batch(raw_confs)
        ct_matched_t = full_model.find_threshold_for_coverage(ct_cal_probs, target_coverage=lfl_cov)
        ct_ans = ct_cal_probs >= ct_matched_t
        ct_cov = float(np.mean(ct_ans))
        ct_acc = float(np.mean(ct_ans & y_corr))
        ct_cw = float(np.mean(ct_ans & (~y_corr)))
        ct_attr = "N/A"

        # 5. ConfidenceOnlyLearned at the same coverage
        col_probs = conf_only_model.predict_calibrated_prob_batch(sub_df)
        col_matched_t = conf_only_model.find_threshold_for_coverage(col_probs, target_coverage=lfl_cov)
        col_ans = col_probs >= col_matched_t
        col_cov = float(np.mean(col_ans))
        col_acc = float(np.mean(col_ans & y_corr))
        col_cw = float(np.mean(col_ans & (~y_corr)))
        col_attr = "N/A"

        best_competitor_cw = min(ct_cw, flw_cw, col_cw)
        learned_wins_cw = bool(lfl_cw < best_competitor_cw)
        if not learned_wins_cw:
            non_winning_slices.append(
                f"{cond_label}@{int(round(sev * 100))}% (LearnedFaultLens CW={lfl_cw:.4f} vs "
                f"ConfThresh={ct_cw:.4f}, FaultLensWeighted={flw_cw:.4f}, ConfOnlyLearned={col_cw:.4f})"
            )

        rows.append({
            "defect": cond_label,
            "severity": float(sev),
            "freq_accuracy": round(fr_acc, 4),
            "freq_cw_rate": round(fr_cw, 4),
            "freq_coverage": round(fr_cov, 4),
            "freq_attribution_accuracy": fr_attr,
            "conf_thresh_matched_accuracy": round(ct_acc, 4),
            "conf_thresh_matched_cw_rate": round(ct_cw, 4),
            "conf_thresh_matched_coverage": round(ct_cov, 4),
            "conf_thresh_attribution_accuracy": ct_attr,
            "faultlens_weighted_accuracy": round(flw_acc, 4),
            "faultlens_weighted_cw_rate": round(flw_cw, 4),
            "faultlens_weighted_coverage": round(flw_cov, 4),
            "faultlens_weighted_attribution_accuracy": flw_attr,
            "learned_faultlens_accuracy": round(lfl_acc, 4),
            "learned_faultlens_cw_rate": round(lfl_cw, 4),
            "learned_faultlens_coverage": round(lfl_cov, 4),
            "learned_faultlens_attribution_accuracy": lfl_attr,
            "confidence_only_learned_accuracy": round(col_acc, 4),
            "confidence_only_learned_cw_rate": round(col_cw, 4),
            "confidence_only_learned_coverage": round(col_cov, 4),
            "confidence_only_learned_attribution_accuracy": col_attr,
            "learned_beats_all_competitors_cw": learned_wins_cw,
        })

    out_df = pd.DataFrame(rows)
    out_p = Path(out_csv_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_p, index=False)

    print("\n--- Session 3.5 Validation Sanity Table (VALIDATION Split Only, N=60) ---")
    print(out_df.to_string(index=False))
    if non_winning_slices:
        print("\nHonest Disclosure - Slices where LearnedFaultLens does NOT strictly beat all competitors on CW rate:")
        for s_msg in non_winning_slices:
            print(f"  - {s_msg}")
    else:
        print("\nLearnedFaultLens achieved the lowest CW rate across all evaluated validation slices.")

    return out_df


def run_final_fresh_test_experiment(
    fresh_test_dir: str | Path = FRESH_TEST_SCENARIOS_DIR,
    model_path: str | Path = RELIABILITY_MODELS_PATH,
    tuned_params_path: str | Path = TUNED_PARAMS_PATH,
    lock_path: str | Path = FINAL_LOCK_PATH,
    out_csv_path: str | Path = Path("results/tables/final_fresh_test_results.csv"),
    num_perturbations: int = 100,
    seed: int = 4242,
) -> pd.DataFrame:
    """Task 7b-d: Run the final evaluation ONCE on the fresh held-out set (seed 4242, 200 scenarios).

    Enforces:
    1. Refuses to run if `lock_path` (`results/final_run.lock`) already exists.
    2. Fails if the saved model and threshold (`results/reliability_models.json`) are missing.
    3. Evaluates strictly on `fresh_test_dir` (`manifest.json`), never reading `splits.json["test"]`.
    4. Creates `lock_path` upon completion.
    """
    l_path = Path(lock_path)
    if l_path.is_file():
        raise RuntimeError(
            f"Final evaluation lock file '{l_path}' already exists! Refusing to rerun final held-out evaluation."
        )

    m_path = Path(model_path)
    if not m_path.is_file():
        raise FileNotFoundError(
            f"Saved reliability model and threshold file '{m_path}' is missing! Cannot run final evaluation."
        )

    full_model = load_saved_reliability_model("full", m_path)
    conf_only_model = load_saved_reliability_model("confidence_only", m_path)
    if full_model.threshold is None or conf_only_model.threshold is None:
        raise ValueError("Saved reliability models are missing frozen thresholds.")

    f_dir = Path(fresh_test_dir)
    manifest_file = f_dir / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(
            f"Fresh held-out manifest not found at '{manifest_file}'. Run `python scripts/run_all.py fresh_test` first."
        )

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    scenario_ids: List[str] = list(manifest.get("scenario_ids", []))
    if not scenario_ids:
        raise ValueError(f"No scenario_ids listed in {manifest_file}")

    scenarios: List[Scenario] = [Scenario.load(f_dir / sc_id) for sc_id in scenario_ids]

    from faultlens.methods import (
        ConfidenceOnlyLearnedMethod,
        FaultLensMethod,
        FrequencyBaselineMethod,
        LearnedFaultLensMethod,
    )

    freq_m = FrequencyBaselineMethod()
    fl_m = FaultLensMethod(num_perturbations=num_perturbations)
    lfl_m = LearnedFaultLensMethod(model=full_model, model_path=m_path, num_perturbations=num_perturbations)
    col_m = ConfidenceOnlyLearnedMethod(model=conf_only_model, model_path=m_path)

    conditions: List[Tuple[str, Optional[str], float]] = [("clean", None, 0.0)]
    for d_name in ALLOWED_DEFECTS:
        for sev in (0.10, 0.20, 0.30):
            conditions.append((d_name, d_name, sev))

    rows: List[Dict[str, Any]] = []
    for sc_idx, sc in enumerate(scenarios):
        for c_idx, (cond_label, defect_type, sev) in enumerate(conditions):
            df_in = sc.log
            if defect_type is not None and sev > 0.0:
                df_in, _ = inject_defect(df_in, defect_type=defect_type, severity=sev, seed=seed + sc_idx * 100 + c_idx)

            eval_seed = seed + 90000 + sc_idx * 100 + c_idx
            for m_obj in (freq_m, fl_m, lfl_m, col_m):
                if isinstance(m_obj, (FaultLensMethod, LearnedFaultLensMethod)):
                    out = m_obj.predict(df_in, seed=eval_seed)
                else:
                    out = m_obj.predict(df_in)
                is_corr = bool(out.top_candidate == sc.truth) if out.answered and out.top_candidate else False
                rows.append({
                    "scenario_id": sc.scenario_id,
                    "defect": cond_label,
                    "severity": float(sev),
                    "method": out.method_name,
                    "answered": out.answered,
                    "correct": is_corr,
                    "confidence": out.confidence,
                    "raw_confidence": out.raw_confidence,
                })

    res_df = pd.DataFrame(rows)
    out_p = Path(out_csv_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_p, index=False)

    l_path.parent.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(l_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "locked_at": now_iso,
                "master_seed": seed,
                "num_scenarios": len(scenarios),
                "results_csv": str(out_p),
                "model_path": str(m_path),
            },
            f,
            indent=2,
        )

    return res_df
