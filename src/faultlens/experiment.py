"""FaultLens Full Benchmark Experiment Runner (Session 3)."""

from __future__ import annotations

import datetime
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from scipy.stats import binomtest

from faultlens.defects import ALLOWED_DEFECTS, ALLOWED_SEVERITIES, inject_defect
from faultlens.diagnoser import ConfidenceCalibrator, DiagnosisResult, compute_ece, diagnose
from faultlens.generator import Scenario
from faultlens.methods import (
    ConfidenceThresholdMethod,
    FaultLensMethod,
    FrequencyBaselineMethod,
)
from faultlens.metrics import (
    bootstrap_metric_ci,
    compute_metrics,
    paired_bootstrap_test,
)
from faultlens.policy import (
    TUNED_PARAMS_PATH,
    apply_policy,
    load_tuned_gate_config,
    load_tuned_policy_params,
    tune_policy_on_validation,
)
from faultlens.quality import (
    QualityReport,
    compute_quality_baseline,
    evaluate_data_quality,
    load_quality_baseline,
)
from faultlens.stability import StabilityReport, evaluate_stability

DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiment.yaml")
DEFAULT_SCENARIOS_DIR = Path("data/sample_scenarios")
TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")
RUN_LOG_PATH = Path("results/RUN_LOG.md")

FALLBACK_EXPERIMENT_CONFIG: Dict[str, Any] = {
    "experiment": {"name": "faultlens_eval_v1", "master_seed": 42, "num_scenarios": 300},
    "stability": {"num_perturbations": 200, "resample_fraction": 0.85, "jitter_std_min": 1.0},
    "quality_scoring": {
        "weights": {
            "completeness": 0.20,
            "reason_validity": 0.20,
            "duplication": 0.15,
            "timestamp_consistency": 0.20,
            "micro_stop_plausibility": 0.10,
            "alarm_reason_agreement": 0.15,
        },
        "near_duplicate_window_sec": 60.0,
    },
    "policy": {"gate_mode": "component", "delta": 0.15, "tau_q": 0.75, "tau_s": 0.80, "buffer": 0.05},
}


def load_experiment_config(config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG) -> Dict[str, Any]:
    """Load experiment configuration from YAML with resilient fallback if YAML is malformed."""
    path = Path(config_path)
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass
    return dict(FALLBACK_EXPERIMENT_CONFIG)


def _format_candidate(cand: Optional[Tuple[str, str]]) -> str:
    """Format a (machine_id, reason) tuple as 'machine_id/reason'."""
    if cand is None:
        return "ABSTAIN"
    return f"{cand[0]}/{cand[1]}"


def _parse_candidate(cand_str: str) -> Optional[Tuple[str, str]]:
    """Parse 'machine_id/reason' back into a 2-tuple, or None if abstained."""
    if not cand_str or cand_str == "ABSTAIN" or cand_str == "None":
        return None
    parts = str(cand_str).split("/", 1)
    if len(parts) == 2:
        return (parts[0], parts[1])
    return None


def fit_validation_calibrator_and_threshold(
    scenarios_dir: str | Path = DEFAULT_SCENARIOS_DIR,
    target_coverage: float = 0.6256,
    calibration_method: str = "platt",
    seed: int = 42,
) -> Tuple[ConfidenceCalibrator, float, np.ndarray, np.ndarray]:
    """Fit confidence calibrator and select threshold strictly on VALIDATION split."""
    base_dir = Path(scenarios_dir)
    splits_file = base_dir / "splits.json"
    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    val_ids = splits.get("val", [])
    test_ids = set(splits.get("test", []))
    if not val_ids:
        raise ValueError("No validation scenario IDs found in splits.json")

    assert len(set(val_ids).intersection(test_ids)) == 0, "Validation split must not overlap with test split!"

    val_scenarios = [Scenario.load(base_dir / sc_id) for sc_id in val_ids]

    raw_confs: List[float] = []
    labels: List[int] = []

    conditions: List[Tuple[Optional[str], float]] = [(None, 0.0)]
    for d_type in ALLOWED_DEFECTS:
        for sev in (0.10, 0.20, 0.30):
            conditions.append((d_type, sev))

    for sc_idx, sc in enumerate(val_scenarios):
        for d_idx, (defect_type, sev) in enumerate(conditions):
            df_in = sc.log
            if defect_type is not None and sev > 0.0:
                df_in, _ = inject_defect(
                    df_in,
                    defect_type=defect_type,
                    severity=sev,
                    seed=seed + sc_idx * 100 + d_idx,
                )
            diag = diagnose(df_in)
            raw_confs.append(diag.raw_confidence)
            labels.append(1 if diag.top_candidate == sc.truth else 0)

    raw_arr = np.asarray(raw_confs, dtype=float)
    label_arr = np.asarray(labels, dtype=int)

    calibrator = ConfidenceCalibrator(method=calibration_method)
    calibrator.fit(raw_arr, label_arr)
    cal_arr = calibrator.predict_batch(raw_arr)

    if len(np.unique(np.round(cal_arr, 4))) < 5 and calibration_method == "isotonic":
        calibrator = ConfidenceCalibrator(method="platt")
        calibrator.fit(raw_arr, label_arr)
        cal_arr = calibrator.predict_batch(raw_arr)

    conf_method = ConfidenceThresholdMethod(calibrator=calibrator)
    tuned_threshold = conf_method.find_threshold_for_coverage(cal_arr, target_coverage=target_coverage)

    return calibrator, tuned_threshold, cal_arr, raw_arr


def _mcnemar_exact_p_value(cw_a: np.ndarray, cw_b: np.ndarray) -> float:
    """Compute exact two-sided McNemar test p-value on binary confidently-wrong indicators."""
    a = np.asarray(cw_a, dtype=bool)
    b = np.asarray(cw_b, dtype=bool)
    n10 = int(np.sum(a & (~b)))
    n01 = int(np.sum((~a) & b))
    n_disc = n10 + n01
    if n_disc == 0:
        return 1.0
    res = binomtest(min(n10, n01), n=n_disc, p=0.5, alternative="two-sided")
    return float(round(res.pvalue, 4))


def run_session2_5_validation_table(
    scenarios_dir: str | Path = DEFAULT_SCENARIOS_DIR,
    out_csv_path: str | Path = Path("results/tables/session2_5_validation_table.csv"),
    seed: int = 42,
) -> pd.DataFrame:
    """Generate the Session 2.5 validation sanity table on VALIDATION split only (60 scenarios)."""
    base_dir = Path(scenarios_dir)
    splits_file = base_dir / "splits.json"
    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    val_ids = splits.get("val", [])
    val_scenarios = [Scenario.load(base_dir / sc_id) for sc_id in val_ids]

    baseline_subs, exp_micro = load_quality_baseline()

    t_path = Path(TUNED_PARAMS_PATH)
    if not t_path.is_file():
        tune_policy_on_validation(scenarios_dir=base_dir, out_params_path=t_path)

    with open(t_path, "r", encoding="utf-8") as f:
        tuned_meta = json.load(f)

    gate_cfg = load_tuned_gate_config(t_path)
    target_val_cov = float(tuned_meta.get("achieved_validation_coverage", 0.6256))

    iso_cal, _, iso_arr, raw_arr = fit_validation_calibrator_and_threshold(
        scenarios_dir=base_dir,
        target_coverage=target_val_cov,
        calibration_method="isotonic",
        seed=seed,
    )
    platt_cal, val_conf_threshold, platt_arr, _ = fit_validation_calibrator_and_threshold(
        scenarios_dir=base_dir,
        target_coverage=target_val_cov,
        calibration_method="platt",
        seed=seed,
    )

    print("\n--- Validation Calibrated Confidence Distribution (780 eval points) ---")
    print(
        f"Isotonic: min={np.min(iso_arr):.4f}, median={np.median(iso_arr):.4f}, "
        f"max={np.max(iso_arr):.4f}, unique_values={len(np.unique(np.round(iso_arr, 4)))}"
    )
    print(
        f"Platt:    min={np.min(platt_arr):.4f}, median={np.median(platt_arr):.4f}, "
        f"max={np.max(platt_arr):.4f}, unique_values={len(np.unique(np.round(platt_arr, 4)))}"
    )

    clean_records = []
    for idx, sc in enumerate(val_scenarios):
        diag = diagnose(sc.log, calibrator=platt_cal)
        q_rep = evaluate_data_quality(
            sc.log,
            baseline_sub_scores=baseline_subs,
            expected_micro_stop_rates=exp_micro,
        )
        s_rep = evaluate_stability(sc.log, num_perturbations=200, seed=seed + idx)
        dec = apply_policy(
            sc.log,
            q_rep,
            s_rep,
            tau_q=gate_cfg["tau_q"],
            tau_s=gate_cfg["tau_s"],
            buffer=gate_cfg["buffer"],
            gate_mode=gate_cfg["gate_mode"],
            delta=gate_cfg["delta"],
            baseline_sub_scores=baseline_subs,
        )
        m_val = diag.margin_ratio
        if m_val >= 0.18:
            m_cat = "easy (>=18%)"
        elif m_val <= 0.07:
            m_cat = "close (<=7%)"
        else:
            m_cat = "moderate (7-18%)"
        clean_records.append({
            "margin_category": m_cat,
            "top_stability": s_rep.top_stability,
            "answered": dec.answered,
        })

    clean_stab_df = pd.DataFrame(clean_records)
    print("\n--- Clean Validation Stability by Margin Category ---")
    for cat in ["easy (>=18%)", "moderate (7-18%)", "close (<=7%)"]:
        sub_c = clean_stab_df[clean_stab_df["margin_category"] == cat]
        if not sub_c.empty:
            print(
                f"{cat:18s}: n={len(sub_c):2d}, mean_top_stability={sub_c['top_stability'].mean():.4f}, "
                f"answered_share={sub_c['answered'].mean() * 100:.1f}%"
            )

    conditions: List[Tuple[str, Optional[str], float]] = [("clean", None, 0.0)]
    for d_name in ALLOWED_DEFECTS:
        for sev in (0.20, 0.30):
            conditions.append((d_name, d_name, sev))

    w_meta = tuned_meta.get("weighted_mode_summary", {"tau_q": gate_cfg["tau_q"], "tau_s": gate_cfg["tau_s"]})
    c_meta = tuned_meta.get("component_mode_summary", {"delta": gate_cfg["delta"], "tau_s": gate_cfg["tau_s"]})

    rows: List[Dict[str, Any]] = []
    for cond_name, defect_type, sev in conditions:
        fr_corr_list, ct_own_ans_list, ct_own_corr_list = [], [], []
        fl_ans_list, fl_corr_list = [], []
        fl_w_ans_list, fl_w_corr_list = [], []
        fl_c_ans_list, fl_c_corr_list = [], []
        cal_confs_cond: List[float] = []

        for idx, sc in enumerate(val_scenarios):
            df_in = sc.log
            if defect_type is not None and sev > 0.0:
                df_in, _ = inject_defect(df_in, defect_type=defect_type, severity=sev, seed=seed + 3000 + idx)

            diag = diagnose(df_in, calibrator=platt_cal)
            q_rep = evaluate_data_quality(
                df_in,
                baseline_sub_scores=baseline_subs,
                expected_micro_stop_rates=exp_micro,
            )
            s_rep = evaluate_stability(df_in, num_perturbations=200, seed=seed + 6000 + idx)

            is_corr = bool(diag.top_candidate == sc.truth)
            cal_c = platt_cal.predict(diag.raw_confidence)
            cal_confs_cond.append(cal_c)
            fr_corr_list.append(is_corr)

            ct_ans = bool(cal_c >= val_conf_threshold)
            ct_own_ans_list.append(ct_ans)
            ct_own_corr_list.append(ct_ans and is_corr)

            dec_main = apply_policy(
                df_in,
                q_rep,
                s_rep,
                tau_q=gate_cfg["tau_q"],
                tau_s=gate_cfg["tau_s"],
                buffer=gate_cfg["buffer"],
                gate_mode=gate_cfg["gate_mode"],
                delta=gate_cfg["delta"],
                baseline_sub_scores=baseline_subs,
            )
            fl_ans_list.append(dec_main.answered)
            fl_corr_list.append(dec_main.answered and is_corr)

            dec_w = apply_policy(
                df_in,
                q_rep,
                s_rep,
                tau_q=float(w_meta.get("tau_q", gate_cfg["tau_q"])),
                tau_s=float(w_meta.get("tau_s", gate_cfg["tau_s"])),
                gate_mode="weighted",
                baseline_sub_scores=baseline_subs,
            )
            fl_w_ans_list.append(dec_w.answered)
            fl_w_corr_list.append(dec_w.answered and is_corr)

            dec_c = apply_policy(
                df_in,
                q_rep,
                s_rep,
                tau_s=float(c_meta.get("tau_s", gate_cfg["tau_s"])),
                gate_mode="component",
                delta=float(c_meta.get("delta", gate_cfg["delta"])),
                baseline_sub_scores=baseline_subs,
            )
            fl_c_ans_list.append(dec_c.answered)
            fl_c_corr_list.append(dec_c.answered and is_corr)

        fr_corr_arr = np.asarray(fr_corr_list, dtype=bool)
        fl_ans_arr = np.asarray(fl_ans_list, dtype=bool)
        fl_corr_arr = np.asarray(fl_corr_list, dtype=bool)
        ct_own_ans_arr = np.asarray(ct_own_ans_list, dtype=bool)
        ct_own_corr_arr = np.asarray(ct_own_corr_list, dtype=bool)
        cal_arr_cond = np.asarray(cal_confs_cond, dtype=float)

        fl_cov = float(fl_ans_arr.mean())
        fr_acc = float(fr_corr_arr.mean())
        fr_cw = float((~fr_corr_arr).mean())

        ct_own_cov = float(ct_own_ans_arr.mean())
        ct_own_acc = float(ct_own_corr_arr.mean())
        ct_own_cw = float((ct_own_ans_arr & (~fr_corr_arr)).mean())

        ct_helper = ConfidenceThresholdMethod(calibrator=platt_cal)
        matched_t = ct_helper.threshold_for_target_coverage(fl_cov, calibrated_confs=cal_arr_cond)
        ct_mat_ans = cal_arr_cond >= matched_t
        ct_mat_cov = float(ct_mat_ans.mean())
        ct_mat_acc = float((ct_mat_ans & fr_corr_arr).mean())
        ct_mat_cw = float((ct_mat_ans & (~fr_corr_arr)).mean())

        fl_acc = float(fl_corr_arr.mean())
        fl_cw = float((fl_ans_arr & (~fr_corr_arr)).mean())
        rand_cw = fr_cw * fl_cov

        fl_w_ans_arr = np.asarray(fl_w_ans_list, dtype=bool)
        fl_c_ans_arr = np.asarray(fl_c_ans_list, dtype=bool)
        fl_w_cw = float((fl_w_ans_arr & (~fr_corr_arr)).mean())
        fl_c_cw = float((fl_c_ans_arr & (~fr_corr_arr)).mean())

        rows.append({
            "defect": cond_name,
            "severity": float(sev),
            "freq_acc": round(fr_acc, 4),
            "freq_cw": round(fr_cw, 4),
            "freq_cov": 1.0,
            "conf_own_acc": round(ct_own_acc, 4),
            "conf_own_cw": round(ct_own_cw, 4),
            "conf_own_cov": round(ct_own_cov, 4),
            "conf_matched_acc": round(ct_mat_acc, 4),
            "conf_matched_cw": round(ct_mat_cw, 4),
            "conf_matched_cov": round(ct_mat_cov, 4),
            "faultlens_acc": round(fl_acc, 4),
            "faultlens_cw": round(fl_cw, 4),
            "faultlens_cov": round(fl_cov, 4),
            "random_abstention_cw": round(rand_cw, 4),
            "faultlens_weighted_cw": round(fl_w_cw, 4),
            "faultlens_weighted_cov": round(float(fl_w_ans_arr.mean()), 4),
            "faultlens_component_cw": round(fl_c_cw, 4),
            "faultlens_component_cov": round(float(fl_c_ans_arr.mean()), 4),
        })

    val_table_df = pd.DataFrame(rows)
    out_p = Path(out_csv_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    val_table_df.to_csv(out_p, index=False)
    print(f"\nSaved Session 2.5 validation table to {out_p}:")
    print(val_table_df.to_string(index=False))
    return val_table_df


def run_test_experiment(
    scenarios_dir: str | Path = DEFAULT_SCENARIOS_DIR,
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG,
    tuned_params_path: str | Path = TUNED_PARAMS_PATH,
    tables_dir: str | Path = TABLES_DIR,
    num_perturbations: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Run the full Session 3 evaluation strictly on the 120 TEST scenarios."""
    cfg = load_experiment_config(config_path)
    master_seed = int(cfg.get("experiment", {}).get("master_seed", 42))
    if num_perturbations is None:
        num_perturbations = int(cfg.get("stability", {}).get("num_perturbations", 200))
    quality_weights = cfg.get("quality_scoring", {}).get("weights", None)
    near_dup_window = float(cfg.get("quality_scoring", {}).get("near_duplicate_window_sec", 60.0))

    baseline_subs, exp_micro = load_quality_baseline()

    t_path = Path(tuned_params_path)
    if not t_path.is_file():
        tune_policy_on_validation(scenarios_dir=scenarios_dir, out_params_path=t_path)

    with open(t_path, "r", encoding="utf-8") as f:
        tuned_meta = json.load(f)

    tau_q, tau_s, buffer = load_tuned_policy_params(t_path)
    gate_cfg = load_tuned_gate_config(t_path)
    gate_mode = gate_cfg["gate_mode"]
    delta = gate_cfg["delta"]
    target_val_cov = float(tuned_meta.get("achieved_validation_coverage", 0.6256))

    base_dir = Path(scenarios_dir)
    splits_file = base_dir / "splits.json"
    if not splits_file.is_file():
        raise FileNotFoundError(f"Splits manifest not found at {splits_file}")

    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_ids = set(splits.get("train", []))
    val_ids = set(splits.get("val", []))
    test_ids: List[str] = list(splits.get("test", []))

    if len(test_ids) != 120:
        raise ValueError(f"Expected 120 test scenarios, found {len(test_ids)}")
    if train_ids.intersection(test_ids) or val_ids.intersection(test_ids):
        raise RuntimeError("Data leakage detected: train/val scenario IDs overlap with test split!")

    calibrator, val_conf_threshold, val_cal_confs, _ = fit_validation_calibrator_and_threshold(
        scenarios_dir=scenarios_dir,
        target_coverage=target_val_cov,
        calibration_method="platt",
        seed=master_seed,
    )

    test_scenarios = [Scenario.load(base_dir / sc_id) for sc_id in test_ids]

    raw_rows: List[Dict[str, Any]] = []
    extra_rows: List[Dict[str, Any]] = []

    defect_list = list(ALLOWED_DEFECTS)
    severity_list = list(ALLOWED_SEVERITIES)

    for sc_idx, sc in enumerate(test_scenarios):
        truth_str = _format_candidate(sc.truth)

        clean_diag = diagnose(sc.log, calibrator=calibrator)
        clean_q = evaluate_data_quality(
            sc.log,
            weights=quality_weights,
            near_dup_window_sec=near_dup_window,
            baseline_sub_scores=baseline_subs,
            expected_micro_stop_rates=exp_micro,
        )
        clean_s = evaluate_stability(
            sc.log,
            num_perturbations=num_perturbations,
            seed=master_seed + 10000 + sc_idx,
        )

        for d_idx, defect_type in enumerate(defect_list):
            for s_idx, severity in enumerate(severity_list):
                if severity == 0.0:
                    df_eval = sc.log
                    diag = clean_diag
                    q_rep = clean_q
                    s_rep = clean_s
                else:
                    inj_seed = master_seed + sc_idx * 1000 + d_idx * 100 + s_idx * 10
                    df_eval, _ = inject_defect(
                        sc.log,
                        defect_type=defect_type,
                        severity=severity,
                        seed=inj_seed,
                    )
                    diag = diagnose(df_eval, calibrator=calibrator)
                    q_rep = evaluate_data_quality(
                        df_eval,
                        weights=quality_weights,
                        near_dup_window_sec=near_dup_window,
                        baseline_sub_scores=baseline_subs,
                        expected_micro_stop_rates=exp_micro,
                    )
                    stab_seed = master_seed + 20000 + sc_idx * 1000 + d_idx * 100 + s_idx * 10
                    s_rep = evaluate_stability(
                        df_eval,
                        num_perturbations=num_perturbations,
                        seed=stab_seed,
                    )

                raw_conf = diag.raw_confidence
                cal_conf = calibrator.predict(raw_conf)
                raw_top_cand = diag.top_candidate
                raw_top_str = _format_candidate(raw_top_cand)
                raw_is_correct = bool(raw_top_cand == sc.truth)

                freq_flag = "GREEN" if raw_conf >= 0.70 else "AMBER"
                raw_rows.append({
                    "scenario_id": sc.scenario_id,
                    "defect": defect_type,
                    "severity": float(severity),
                    "method": "frequency_baseline",
                    "top_candidate": raw_top_str,
                    "truth": truth_str,
                    "correct": raw_is_correct,
                    "answered": True,
                    "confidence": round(raw_conf, 4),
                    "quality_score": round(q_rep.overall_score, 4),
                    "top_stability": round(s_rep.top_stability, 4),
                    "flag": freq_flag,
                    "margin": round(diag.margin_ratio, 4),
                })
                extra_rows.append({
                    "top3_flip_rate": s_rep.top3_flip_rate,
                    "raw_top_candidate": raw_top_str,
                    "raw_confidence": raw_conf,
                    "calibrated_confidence": cal_conf,
                })

                ct_answered = bool(cal_conf >= val_conf_threshold)
                if ct_answered:
                    ct_flag = "GREEN" if cal_conf >= (val_conf_threshold + buffer) else "AMBER"
                    ct_cand_str = raw_top_str
                    ct_correct = raw_is_correct
                else:
                    ct_flag = "RED"
                    ct_cand_str = "ABSTAIN"
                    ct_correct = False

                raw_rows.append({
                    "scenario_id": sc.scenario_id,
                    "defect": defect_type,
                    "severity": float(severity),
                    "method": "confidence_threshold",
                    "top_candidate": ct_cand_str,
                    "truth": truth_str,
                    "correct": ct_correct,
                    "answered": ct_answered,
                    "confidence": round(cal_conf, 4),
                    "quality_score": round(q_rep.overall_score, 4),
                    "top_stability": round(s_rep.top_stability, 4),
                    "flag": ct_flag,
                    "margin": round(diag.margin_ratio, 4),
                })
                extra_rows.append({
                    "top3_flip_rate": s_rep.top3_flip_rate,
                    "raw_top_candidate": raw_top_str,
                    "raw_confidence": raw_conf,
                    "calibrated_confidence": cal_conf,
                })

                decision = apply_policy(
                    df=df_eval,
                    quality=q_rep,
                    stability=s_rep,
                    tau_q=tau_q,
                    tau_s=tau_s,
                    buffer=buffer,
                    gate_mode=gate_mode,
                    delta=delta,
                    baseline_sub_scores=baseline_subs,
                )
                fl_answered = decision.answered
                fl_cand_str = raw_top_str if fl_answered else "ABSTAIN"
                fl_correct = bool(fl_answered and raw_is_correct)

                raw_rows.append({
                    "scenario_id": sc.scenario_id,
                    "defect": defect_type,
                    "severity": float(severity),
                    "method": "faultlens",
                    "top_candidate": fl_cand_str,
                    "truth": truth_str,
                    "correct": fl_correct,
                    "answered": fl_answered,
                    "confidence": round(cal_conf, 4),
                    "quality_score": round(q_rep.overall_score, 4),
                    "top_stability": round(s_rep.top_stability, 4),
                    "flag": decision.status_flag,
                    "margin": round(diag.margin_ratio, 4),
                })
                extra_rows.append({
                    "top3_flip_rate": s_rep.top3_flip_rate,
                    "raw_top_candidate": raw_top_str,
                    "raw_confidence": raw_conf,
                    "calibrated_confidence": cal_conf,
                })

    out_tables_dir = Path(tables_dir)
    out_tables_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.DataFrame(raw_rows)
    raw_csv_path = out_tables_dir / "raw_results.csv"
    raw_df.to_csv(raw_csv_path, index=False)

    work_df = raw_df.copy()
    extra_df = pd.DataFrame(extra_rows)
    for col in extra_df.columns:
        work_df[col] = extra_df[col].to_numpy()

    metrics_df = build_metrics_by_defect_severity(work_df, seed=master_seed)
    metrics_df.to_csv(out_tables_dir / "metrics_by_defect_severity.csv", index=False)

    matched_df = build_matched_coverage_table(work_df, val_cal_confs=val_cal_confs)
    matched_df.to_csv(out_tables_dir / "matched_coverage.csv", index=False)

    sig_df = build_significance_table(work_df, matched_df=matched_df, seed=master_seed)
    sig_df.to_csv(out_tables_dir / "significance.csv", index=False)

    cal_df = build_calibration_table(work_df)
    cal_df.to_csv(out_tables_dir / "calibration.csv", index=False)

    write_summary_markdown(
        metrics_df=metrics_df,
        matched_df=matched_df,
        sig_df=sig_df,
        cal_df=cal_df,
        tau_q=tau_q,
        tau_s=tau_s,
        gate_mode=gate_mode,
        delta=delta,
        val_conf_threshold=val_conf_threshold,
        out_path=out_tables_dir / "summary.md",
    )

    write_run_log(
        num_test_scenarios=len(test_ids),
        master_seed=master_seed,
        num_perturbations=num_perturbations,
        tau_q=tau_q,
        tau_s=tau_s,
        gate_mode=gate_mode,
        delta=delta,
        val_conf_threshold=val_conf_threshold,
        out_path=RUN_LOG_PATH,
    )

    return {
        "raw_results": raw_df,
        "metrics_by_defect_severity": metrics_df,
        "matched_coverage": matched_df,
        "significance": sig_df,
        "calibration": cal_df,
    }


def build_metrics_by_defect_severity(
    work_df: pd.DataFrame,
    n_bootstraps: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Produce metrics_by_defect_severity.csv with bootstrap 95% CIs."""
    rows: List[Dict[str, Any]] = []
    methods_order = ["frequency_baseline", "confidence_threshold", "faultlens"]

    for method in methods_order:
        m_df = work_df[work_df["method"] == method]
        for defect in ALLOWED_DEFECTS:
            for sev in ALLOWED_SEVERITIES:
                grp = m_df[(m_df["defect"] == defect) & (np.isclose(m_df["severity"], sev))].reset_index(drop=True)
                if grp.empty:
                    continue

                preds = [_parse_candidate(x) for x in grp["top_candidate"]]
                truths = [_parse_candidate(x) for x in grp["truth"]]
                raw_preds = [_parse_candidate(x) for x in grp["raw_top_candidate"]]
                ans = list(grp["answered"].astype(bool))
                confs = list(grp["confidence"].astype(float))

                m_sum = compute_metrics(
                    predictions=preds,
                    ground_truths=truths,  # type: ignore[arg-type]
                    answered_flags=ans,
                    confidences=confs,
                    unvetted_predictions=raw_preds,  # type: ignore[arg-type]
                    method_name=method,
                )

                n = len(grp)
                idx_arr = np.arange(n)
                correct_arr = grp["correct"].to_numpy(dtype=float)
                ans_arr = grp["answered"].to_numpy(dtype=float)
                cw_arr = (grp["answered"].to_numpy(dtype=bool) & (~grp["correct"].to_numpy(dtype=bool))).astype(float)
                strict_cw_arr = (
                    grp["answered"].to_numpy(dtype=bool)
                    & (~grp["correct"].to_numpy(dtype=bool))
                    & (grp["confidence"].to_numpy(dtype=float) >= 0.80)
                ).astype(float)

                _, acc_low, acc_high = bootstrap_metric_ci(
                    lambda b_idx: float(np.mean(correct_arr[b_idx])),
                    idx_arr,
                    n_bootstraps=n_bootstraps,
                    seed=seed,
                )
                _, cw_low, cw_high = bootstrap_metric_ci(
                    lambda b_idx: float(np.mean(cw_arr[b_idx])),
                    idx_arr,
                    n_bootstraps=n_bootstraps,
                    seed=seed + 1,
                )
                _, scw_low, scw_high = bootstrap_metric_ci(
                    lambda b_idx: float(np.mean(strict_cw_arr[b_idx])),
                    idx_arr,
                    n_bootstraps=n_bootstraps,
                    seed=seed + 2,
                )
                _, cov_low, cov_high = bootstrap_metric_ci(
                    lambda b_idx: float(np.mean(ans_arr[b_idx])),
                    idx_arr,
                    n_bootstraps=n_bootstraps,
                    seed=seed + 3,
                )

                def _sel_acc_fn(b_idx: np.ndarray) -> float:
                    sub_ans = ans_arr[b_idx] > 0.5
                    if not np.any(sub_ans):
                        return 0.0
                    return float(np.mean(correct_arr[b_idx][sub_ans]))

                _, sel_low, sel_high = bootstrap_metric_ci(
                    _sel_acc_fn,
                    idx_arr,
                    n_bootstraps=n_bootstraps,
                    seed=seed + 4,
                )

                rows.append({
                    "method": method,
                    "defect": defect,
                    "severity": float(sev),
                    "scenarios": n,
                    "accuracy": m_sum.accuracy,
                    "accuracy_ci_lower": round(acc_low, 4),
                    "accuracy_ci_upper": round(acc_high, 4),
                    "confidently_wrong_rate": m_sum.confidently_wrong_rate,
                    "cw_ci_lower": round(cw_low, 4),
                    "cw_ci_upper": round(cw_high, 4),
                    "strict_confidently_wrong_rate": m_sum.strict_confidently_wrong_rate,
                    "strict_cw_ci_lower": round(scw_low, 4),
                    "strict_cw_ci_upper": round(scw_high, 4),
                    "coverage": m_sum.coverage,
                    "coverage_ci_lower": round(cov_low, 4),
                    "coverage_ci_upper": round(cov_high, 4),
                    "abstention_rate": m_sum.abstention_rate,
                    "selective_accuracy": m_sum.selective_accuracy,
                    "sel_acc_ci_lower": round(sel_low, 4),
                    "sel_acc_ci_upper": round(sel_high, 4),
                    "mean_top_stability": round(float(grp["top_stability"].mean()), 4),
                    "top3_flip_rate": round(float(grp["top3_flip_rate"].mean()), 4),
                    "good_abstention_share": m_sum.good_abstention_share,
                    "unnecessary_abstention_share": m_sum.unnecessary_abstention_share,
                    "aurc": m_sum.aurc,
                    "ece": m_sum.ece,
                })

    return pd.DataFrame(rows)


def build_matched_coverage_table(
    work_df: pd.DataFrame,
    val_cal_confs: np.ndarray,
) -> pd.DataFrame:
    """Produce matched_coverage.csv comparing FaultLens and ConfidenceThreshold at matched coverage."""
    rows: List[Dict[str, Any]] = []

    for defect in ALLOWED_DEFECTS:
        for sev in ALLOWED_SEVERITIES:
            sub = work_df[(work_df["defect"] == defect) & (np.isclose(work_df["severity"], sev))]
            if sub.empty:
                continue

            fl_sub = sub[sub["method"] == "faultlens"].sort_values("scenario_id").reset_index(drop=True)
            ct_sub = sub[sub["method"] == "confidence_threshold"].sort_values("scenario_id").reset_index(drop=True)
            fr_sub = sub[sub["method"] == "frequency_baseline"].sort_values("scenario_id").reset_index(drop=True)

            n = len(fl_sub)
            fl_cov = float(fl_sub["answered"].mean())
            fl_cw = float((fl_sub["answered"] & (~fl_sub["correct"])).mean())
            fl_acc = float(fl_sub["correct"].mean())
            fl_sel_acc = float(fl_sub.loc[fl_sub["answered"], "correct"].mean()) if fl_sub["answered"].any() else 0.0

            fr_cw = float((fr_sub["answered"] & (~fr_sub["correct"])).mean())
            ct_own_cov = float(ct_sub["answered"].mean())
            ct_own_cw = float((ct_sub["answered"] & (~ct_sub["correct"])).mean())

            ct_confs = ct_sub["confidence"].to_numpy(dtype=float)
            raw_correct = fr_sub["correct"].to_numpy(dtype=bool)

            cand_thresholds = np.unique(
                np.concatenate([
                    val_cal_confs,
                    np.quantile(ct_confs, np.linspace(0.0, 1.0, n + 1)),
                    ct_confs + 1e-7,
                ])
            )
            best_t = float(cand_thresholds[0])
            best_cov_diff = 2.0
            for t_cand in cand_thresholds:
                cov_cand = float(np.mean(ct_confs >= t_cand))
                diff = abs(cov_cand - fl_cov)
                if diff < best_cov_diff:
                    best_cov_diff = diff
                    best_t = float(t_cand)

            matched_ans = ct_confs >= best_t
            matched_cov = float(np.mean(matched_ans))
            matched_cw = float(np.mean(matched_ans & (~raw_correct)))
            matched_acc = float(np.mean(matched_ans & raw_correct))
            matched_sel_acc = float(np.mean(raw_correct[matched_ans])) if np.any(matched_ans) else 0.0

            random_abstention_cw = fr_cw * fl_cov

            rows.append({
                "defect": defect,
                "severity": float(sev),
                "scenarios": n,
                "faultlens_coverage": round(fl_cov, 4),
                "matched_threshold": round(best_t, 4),
                "conf_thresh_matched_coverage": round(matched_cov, 4),
                "conf_thresh_own_coverage": round(ct_own_cov, 4),
                "freq_cw_rate": round(fr_cw, 4),
                "conf_thresh_own_cw_rate": round(ct_own_cw, 4),
                "conf_thresh_matched_cw_rate": round(matched_cw, 4),
                "random_abstention_cw_rate": round(random_abstention_cw, 4),
                "faultlens_cw_rate": round(fl_cw, 4),
                "faultlens_accuracy": round(fl_acc, 4),
                "conf_thresh_matched_accuracy": round(matched_acc, 4),
                "faultlens_selective_accuracy": round(fl_sel_acc, 4),
                "conf_thresh_matched_selective_accuracy": round(matched_sel_acc, 4),
                "cw_reduction_vs_freq": round(fr_cw - fl_cw, 4),
                "cw_reduction_vs_matched_conf": round(matched_cw - fl_cw, 4),
            })

    return pd.DataFrame(rows)


def build_significance_table(
    work_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Produce significance.csv with paired bootstrap CIs and McNemar p-values on CW outcomes."""
    rows: List[Dict[str, Any]] = []

    def _compare_slice(
        slice_label: str,
        defect_label: str,
        sev_val: float,
        fl_cw: np.ndarray,
        base_cw: np.ndarray,
        baseline_name: str,
        seed_offset: int,
    ) -> Dict[str, Any]:
        res = paired_bootstrap_test(base_cw, fl_cw, n_bootstraps=1000, seed=seed + seed_offset)
        mcnemar_p = _mcnemar_exact_p_value(base_cw, fl_cw)
        return {
            "slice": slice_label,
            "defect": defect_label,
            "severity": sev_val,
            "comparison": f"faultlens_vs_{baseline_name}",
            "n_pairs": len(fl_cw),
            "baseline_cw_rate": round(float(np.mean(base_cw)), 4),
            "faultlens_cw_rate": round(float(np.mean(fl_cw)), 4),
            "cw_reduction_mean": res["mean_diff"],
            "ci_lower_95": res["ci_lower_95"],
            "ci_upper_95": res["ci_upper_95"],
            "bootstrap_p_value": res["p_value"],
            "mcnemar_p_value": mcnemar_p,
        }

    def_sub = work_df[work_df["severity"] > 0.0]
    fl_all = def_sub[def_sub["method"] == "faultlens"].sort_values(["scenario_id", "defect", "severity"])
    fr_all = def_sub[def_sub["method"] == "frequency_baseline"].sort_values(["scenario_id", "defect", "severity"])
    ct_all = def_sub[def_sub["method"] == "confidence_threshold"].sort_values(["scenario_id", "defect", "severity"])

    fl_cw_all = (fl_all["answered"].to_numpy(dtype=bool) & (~fl_all["correct"].to_numpy(dtype=bool))).astype(float)
    fr_cw_all = (fr_all["answered"].to_numpy(dtype=bool) & (~fr_all["correct"].to_numpy(dtype=bool))).astype(float)
    ct_cw_all = (ct_all["answered"].to_numpy(dtype=bool) & (~ct_all["correct"].to_numpy(dtype=bool))).astype(float)

    rows.append(_compare_slice("all_defective_pooled", "ALL", -1.0, fl_cw_all, fr_cw_all, "frequency_baseline", 1))
    rows.append(_compare_slice("all_defective_pooled", "ALL", -1.0, fl_cw_all, ct_cw_all, "confidence_threshold_own", 2))

    step = 10
    for _, m_row in matched_df.iterrows():
        d_name = str(m_row["defect"])
        sev = float(m_row["severity"])
        matched_t = float(m_row["matched_threshold"])

        sub = work_df[(work_df["defect"] == d_name) & (np.isclose(work_df["severity"], sev))]
        fl_s = sub[sub["method"] == "faultlens"].sort_values("scenario_id")
        fr_s = sub[sub["method"] == "frequency_baseline"].sort_values("scenario_id")
        ct_s = sub[sub["method"] == "confidence_threshold"].sort_values("scenario_id")

        fl_cw = (fl_s["answered"].to_numpy(dtype=bool) & (~fl_s["correct"].to_numpy(dtype=bool))).astype(float)
        fr_cw = (fr_s["answered"].to_numpy(dtype=bool) & (~fr_s["correct"].to_numpy(dtype=bool))).astype(float)
        ct_cw = (ct_s["answered"].to_numpy(dtype=bool) & (~ct_s["correct"].to_numpy(dtype=bool))).astype(float)

        ct_confs = ct_s["confidence"].to_numpy(dtype=float)
        raw_corr = fr_s["correct"].to_numpy(dtype=bool)
        ct_matched_cw = ((ct_confs >= matched_t) & (~raw_corr)).astype(float)

        s_label = f"{d_name}_{int(round(sev * 100))}pct"
        rows.append(_compare_slice(s_label, d_name, sev, fl_cw, fr_cw, "frequency_baseline", step))
        rows.append(_compare_slice(s_label, d_name, sev, fl_cw, ct_cw, "confidence_threshold_own", step + 1))
        rows.append(_compare_slice(s_label, d_name, sev, fl_cw, ct_matched_cw, "confidence_threshold_matched", step + 2))
        step += 3

    return pd.DataFrame(rows)


def build_calibration_table(work_df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Produce calibration.csv with reliability-diagram bin statistics and ECE."""
    rows: List[Dict[str, Any]] = []

    fr_df = work_df[work_df["method"] == "frequency_baseline"].copy()
    labels = fr_df["correct"].to_numpy(dtype=int)
    raw_confs = fr_df["raw_confidence"].to_numpy(dtype=float)
    cal_confs = fr_df["calibrated_confidence"].to_numpy(dtype=float)

    for cal_label, conf_arr in [("uncalibrated_raw", raw_confs), ("calibrated_platt", cal_confs)]:
        ece_val, diag_data = compute_ece(conf_arr, labels, n_bins=n_bins)
        edges = diag_data["bin_edges"]
        accs = diag_data["bin_accs"]
        confs = diag_data["bin_confs"]
        counts = diag_data["bin_counts"]

        for b_idx in range(n_bins):
            rows.append({
                "series": cal_label,
                "bin_index": b_idx,
                "bin_low": edges[b_idx],
                "bin_high": edges[b_idx + 1],
                "bin_mean_confidence": confs[b_idx],
                "bin_accuracy": accs[b_idx],
                "bin_count": counts[b_idx],
                "overall_ece": ece_val,
            })

    fl_df = work_df[(work_df["method"] == "faultlens") & (work_df["answered"])].copy()
    if not fl_df.empty:
        fl_labels = fl_df["correct"].to_numpy(dtype=int)
        fl_confs = fl_df["confidence"].to_numpy(dtype=float)
        fl_ece, fl_diag = compute_ece(fl_confs, fl_labels, n_bins=n_bins)
        edges = fl_diag["bin_edges"]
        for b_idx in range(n_bins):
            rows.append({
                "series": "faultlens_answered",
                "bin_index": b_idx,
                "bin_low": edges[b_idx],
                "bin_high": edges[b_idx + 1],
                "bin_mean_confidence": fl_diag["bin_confs"][b_idx],
                "bin_accuracy": fl_diag["bin_accs"][b_idx],
                "bin_count": fl_diag["bin_counts"][b_idx],
                "overall_ece": fl_ece,
            })

    return pd.DataFrame(rows)


def write_summary_markdown(
    metrics_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    tau_q: float,
    tau_s: float,
    gate_mode: str,
    delta: float,
    val_conf_threshold: float,
    out_path: Path,
) -> None:
    """Write results/tables/summary.md strictly from real computed table numbers."""
    lines: List[str] = [
        "# FaultLens Benchmark Summary (Test Split, N=120 Scenarios)",
        "",
        f"> **Disclosure**: All scenarios and event logs are 100% synthetic. Thresholds (`gate_mode={gate_mode}`, "
        f"`tau_q={tau_q:.4f}`, `delta={delta:.4f}`, `tau_s={tau_s:.4f}`, `conf_threshold={val_conf_threshold:.4f}`) "
        "were tuned on train and validation scenarios only. The test split (120 scenarios) was also used in an earlier exploratory run before the Session 2.5 fixes, so these results are reported as exploratory and not as a clean one shot held out result.",
        "",
        "## Headline Comparison at Matched Coverage (`matched_coverage.csv`)",
        "",
        "| Defect Type | Severity | FaultLens Cov (%) | Freq CW (%) | ConfThresh Own CW (%) | ConfThresh Matched CW (%) | Random Abstain CW (%) | FaultLens CW (%) | FaultLens Sel. Acc (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    clean_shown = False
    for _, row in matched_df.iterrows():
        sev = float(row["severity"])
        if sev == 0.0:
            if clean_shown:
                continue
            clean_shown = True
            d_label = "clean (0% defect)"
        else:
            d_label = str(row["defect"])

        lines.append(
            f"| {d_label} | {int(round(sev * 100))}% | "
            f"{row['faultlens_coverage'] * 100:.1f} | "
            f"{row['freq_cw_rate'] * 100:.1f} | "
            f"{row['conf_thresh_own_cw_rate'] * 100:.1f} | "
            f"{row['conf_thresh_matched_cw_rate'] * 100:.1f} | "
            f"{row['random_abstention_cw_rate'] * 100:.1f} | "
            f"{row['faultlens_cw_rate'] * 100:.1f} | "
            f"{row['faultlens_selective_accuracy'] * 100:.1f} |"
        )

    def_rows = matched_df[matched_df["severity"] > 0.0]
    mean_fr_cw = float(def_rows["freq_cw_rate"].mean()) * 100.0
    mean_ct_own_cw = float(def_rows["conf_thresh_own_cw_rate"].mean()) * 100.0
    mean_ct_mat_cw = float(def_rows["conf_thresh_matched_cw_rate"].mean()) * 100.0
    mean_fl_cw = float(def_rows["faultlens_cw_rate"].mean()) * 100.0
    mean_fl_cov = float(def_rows["faultlens_coverage"].mean()) * 100.0

    clean_row = matched_df[matched_df["severity"] == 0.0].iloc[0]
    clean_cov = float(clean_row["faultlens_coverage"]) * 100.0
    clean_fr_cw = float(clean_row["freq_cw_rate"]) * 100.0
    clean_fl_cw = float(clean_row["faultlens_cw_rate"]) * 100.0

    non_winning = def_rows[def_rows["faultlens_cw_rate"] >= def_rows["conf_thresh_matched_cw_rate"]]
    non_winning_strs: List[str] = []
    for _, nw in non_winning.iterrows():
        non_winning_strs.append(
            f"`{nw['defect']}` at {int(round(float(nw['severity']) * 100))}% "
            f"(FaultLens CW {nw['faultlens_cw_rate'] * 100:.1f}% vs Matched ConfThresh CW {nw['conf_thresh_matched_cw_rate'] * 100:.1f}%)"
        )

    raw_ece = float(cal_df[cal_df["series"] == "uncalibrated_raw"]["overall_ece"].iloc[0])
    cal_ece = float(cal_df[cal_df["series"] == "calibrated_platt"]["overall_ece"].iloc[0])

    pooled_sig = sig_df[sig_df["slice"] == "all_defective_pooled"]
    p_vs_freq = float(pooled_sig[pooled_sig["comparison"] == "faultlens_vs_frequency_baseline"]["bootstrap_p_value"].iloc[0])
    p_vs_ct = float(pooled_sig[pooled_sig["comparison"] == "faultlens_vs_confidence_threshold_own"]["bootstrap_p_value"].iloc[0])

    lines.extend([
        "",
        "## Key Findings (Traceable to CSV Outputs)",
        "",
        f"1. **Aggregate Confidently-Wrong Reduction**: Across all defective test conditions (10%, 20%, 30% severities), "
        f"the naive Frequency baseline produces a mean confidently-wrong (CW) rate of **{mean_fr_cw:.2f}%** (at 100% coverage), "
        f"the tuned Confidence-Threshold baseline achieves **{mean_ct_own_cw:.2f}%** CW, "
        f"the coverage-matched Confidence-Threshold baseline achieves **{mean_ct_mat_cw:.2f}%** CW, "
        f"and FaultLens achieves **{mean_fl_cw:.2f}%** CW at **{mean_fl_cov:.2f}%** mean coverage "
        f"(paired bootstrap $p = {p_vs_freq:.4f}$ vs Frequency, $p = {p_vs_ct:.4f}$ vs Own Confidence-Threshold).",
        f"2. **Clean Scenario Behavior & Unnecessary Abstentions**: On clean scenarios (0% defects), Frequency baseline CW is "
        f"**{clean_fr_cw:.2f}%** and FaultLens CW is **{clean_fl_cw:.2f}%**, while FaultLens answers **{clean_cov:.2f}%** of scenarios "
        f"(abstaining on **{100.0 - clean_cov:.2f}%** of clean scenarios when competing bottlenecks have narrow ground-truth margins "
        f"that flip under bootstrap perturbation).",
        "3. **Where FaultLens Does Not Beat the Matched Confidence-Threshold Baseline**: "
        + (
            "FaultLens has a higher or equal CW rate compared to the coverage-matched Confidence-Threshold baseline in: "
            + "; ".join(non_winning_strs)
            + ". Where single-event margin confidence already separates easy from close scenarios, thresholding on calibrated margin confidence alone is a strong competitor."
            if non_winning_strs
            else "FaultLens achieves lower CW rate than the coverage-matched Confidence-Threshold baseline across all evaluated defective cells."
        ),
        f"4. **Calibration (ECE)**: Post-hoc calibration reduces Expected Calibration Error (ECE) from **{raw_ece:.4f}** (raw softmax margin confidence) "
        f"to **{cal_ece:.4f}** (`calibration.csv`).",
        "",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_run_log(
    num_test_scenarios: int,
    master_seed: int,
    num_perturbations: int,
    tau_q: float,
    tau_s: float,
    gate_mode: str,
    delta: float,
    val_conf_threshold: float,
    out_path: Path,
) -> None:
    """Write results/RUN_LOG.md documenting execution metadata, seeds, versions, and commands."""
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    fence = "`" * 3
    lines = [
        "# FaultLens Experiment Run Log",
        "",
        f"- **Timestamp**: {now_iso}",
        f"- **Master Seed**: `{master_seed}`",
        f"- **Test Scenarios Evaluated**: `{num_test_scenarios}` (`splits.json[\"test\"]`, zero overlap with train/val)",
        f"- **Total Evaluation Rows**: `{num_test_scenarios * 4 * 4 * 3}` (`120 scenarios x 4 defects x 4 severities x 3 methods`)",
        f"- **Perturbations per Evaluation (`N`)**: `{num_perturbations}`",
        "- **Tuned Policy Parameters (`results/tuned_params.json`)**:",
        f"  - `gate_mode`: `{gate_mode}`",
        f"  - `delta`: `{delta:.4f}`",
        f"  - `tau_q`: `{tau_q:.4f}`",
        f"  - `tau_s`: `{tau_s:.4f}`",
        f"  - `val_conf_threshold`: `{val_conf_threshold:.4f}`",
        "",
        "## Environment & Library Versions",
        f"- **Python**: `{sys.version.split()[0]}` (`{platform.platform()}`)",
        f"- **NumPy**: `{np.__version__}`",
        f"- **pandas**: `{pd.__version__}`",
        f"- **scikit-learn**: `{sklearn.__version__}`",
        f"- **SciPy**: `{scipy.__version__}`",
        f"- **PyYAML**: `{yaml.__version__}`",
        "",
        "## Exact Reproduction Commands (Windows PowerShell)",
        f"{fence}powershell",
        "python scripts/run_all.py data",
        "python scripts/run_all.py baseline",
        "python scripts/run_all.py tune",
        "python scripts/run_all.py sanity",
        "python scripts/run_all.py experiments",
        "python scripts/run_all.py plots",
        "python scripts/run_all.py test",
        fence,
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    print("======================================================================")
    print("FaultLens Session 3: Running Held-Out Test Split Benchmark (N=120)")
    print("======================================================================")
    tables = run_test_experiment()
    print(f"Saved raw_results.csv: {len(tables['raw_results'])} rows")
    print(f"Saved metrics_by_defect_severity.csv: {len(tables['metrics_by_defect_severity'])} rows")
    print(f"Saved matched_coverage.csv: {len(tables['matched_coverage'])} rows")
    print(f"Saved significance.csv: {len(tables['significance'])} rows")
    print(f"Saved calibration.csv: {len(tables['calibration'])} rows")
    print("\n--- Matched Coverage Summary Table ---")
    print(tables["matched_coverage"].to_string(index=False))