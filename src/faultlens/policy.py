"""FaultLens Evidence-Aware Policy & What-To-Verify Message Generator.

Combines data quality scoring and perturbation stability to make principled
abstention decisions:
- Gate mode "weighted": ANSWER when quality_score >= tau_q AND top_stability >= tau_s
- Gate mode "component": ANSWER when max_sub_score_drop <= delta AND top_stability >= tau_s
- ABSTAIN ("INSUFFICIENT EVIDENCE") otherwise
- Status Flags: GREEN (comfortable confidence), AMBER (marginal confidence), RED (abstain)

Includes validation grid-search threshold tuning (minimizing confidently-wrong rate
subject to min coverage >= 60%, with zero TEST split access) and generates actionable,
count-grounded verification messages.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from faultlens.quality import (
    QualityReport,
    compute_quality_baseline,
    evaluate_data_quality,
    load_quality_baseline,
)
from faultlens.stability import StabilityReport, evaluate_stability

DEFAULT_POLICY_CONFIG = Path("configs/experiment.yaml")
TUNED_PARAMS_PATH = Path("results/tuned_params.json")


@dataclass
class EvidenceSummary:
    """Detailed evidence report attached to every diagnostic output."""

    top_candidate: Tuple[str, str]
    supporting_events_count: int
    share_of_total_downtime: float
    ci_lower_95: float
    ci_upper_95: float
    stability_pct: float
    quality_score: float
    weakest_sub_scores: Dict[str, float]
    verification_message: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize evidence summary."""
        return {
            "top_candidate": list(self.top_candidate),
            "supporting_events_count": self.supporting_events_count,
            "share_of_total_downtime": round(self.share_of_total_downtime, 4),
            "ci_lower_95": round(self.ci_lower_95, 4),
            "ci_upper_95": round(self.ci_upper_95, 4),
            "stability_pct": round(self.stability_pct, 1),
            "quality_score": round(self.quality_score, 4),
            "weakest_sub_scores": {k: round(v, 4) for k, v in self.weakest_sub_scores.items()},
            "verification_message": self.verification_message,
        }


@dataclass
class PolicyDecision:
    """Decision output from the evidence-aware policy."""

    answered: bool
    status_flag: str  # "GREEN" | "AMBER" | "RED"
    decision_reason: str
    tau_q: float
    tau_s: float
    quality_score: float
    stability_score: float
    evidence_summary: Optional[EvidenceSummary]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize policy decision."""
        return {
            "answered": self.answered,
            "status_flag": self.status_flag,
            "decision_reason": self.decision_reason,
            "tau_q": round(self.tau_q, 4),
            "tau_s": round(self.tau_s, 4),
            "quality_score": round(self.quality_score, 4),
            "stability_score": round(self.stability_score, 4),
            "evidence_summary": self.evidence_summary.to_dict() if self.evidence_summary else None,
        }


def generate_verification_message(
    quality: QualityReport,
    stability: StabilityReport,
    log_df: pd.DataFrame,
) -> Optional[str]:
    """Generate an actionable, domain-specific 'what to verify' message with real machine counts.

    Identifies the single most urgent empirical reason why confidence is degraded.
    """
    counts = quality.counts
    m_counts = counts.machine_counts
    worst_m = quality.worst_machine

    # Priority 1: High duplication / alarm bursts
    if quality.sub_scores.get("duplication", 1.0) < 0.85 and counts.duplicate_burst_count > 0:
        return f"{counts.duplicate_burst_count} records appear to be duplicate telemetry or alarm bounce bursts; confirm whether these represent a single recurring fault."

    # Priority 2: Missing end times / unclosed stops
    if quality.sub_scores.get("completeness", 1.0) < 0.85 and counts.missing_end_count > 0:
        worst_missing_end_m = max(
            m_counts.keys(), key=lambda m: m_counts[m].get("missing_end", 0)
        ) if m_counts else worst_m
        m_missing = m_counts.get(worst_missing_end_m, {}).get("missing_end", counts.missing_end_count)
        return f"{m_missing} downtime stops on {worst_missing_end_m} have no end time; verify edge data capture and close open maintenance tickets."

    # Priority 3: Operator Reason Code Misattribution / Other/Unknown
    if quality.sub_scores.get("reason_validity", 1.0) < 0.85 and counts.other_unknown_count > 0:
        worst_unknown_m = max(
            m_counts.keys(), key=lambda m: m_counts[m].get("other_unknown", 0)
        ) if m_counts else worst_m
        m_total_evts = m_counts.get(worst_unknown_m, {}).get("events", 1)
        m_unk = m_counts.get(worst_unknown_m, {}).get("other_unknown", 0)
        pct = int(round((m_unk / max(1, m_total_evts)) * 100))
        return f"{pct}% of {worst_unknown_m} stops are logged as Other/Unknown ({m_unk} events); review manual HMI categorization."

    # Priority 4: Timestamp inversions and overlaps
    if quality.sub_scores.get("timestamp_consistency", 1.0) < 0.85:
        ovlps = counts.overlapping_event_count
        negs = counts.negative_duration_count
        worst_time_m = max(
            m_counts.keys(), key=lambda m: m_counts[m].get("overlaps", 0) + m_counts[m].get("negative_durations", 0)
        ) if m_counts else worst_m
        return f"Check PLC clock synchronization on {worst_time_m}; detected {ovlps} overlapping intervals and {negs} inverted timestamps."

    # Priority 5: Unstable top candidates / competing bottlenecks
    if stability.top_stability < 0.70 or stability.margin_ci_crosses_zero:
        top_cand = stability.top_candidate or ("Unknown", "Unknown")
        runner_up = stability.runner_up_candidate or ("Unknown", "Unknown")
        top_str = f"{top_cand[0]}/{top_cand[1]}"
        runner_str = f"{runner_up[0]}/{runner_up[1]}"
        return f"{top_str} and {runner_str} are statistically tied under perturbation (margin CI crosses zero); inspect recent shifts or collect additional telemetry."

    # Priority 6: Alarm-reason discordance
    if quality.sub_scores.get("alarm_reason_agreement", 1.0) < 0.85:
        return f"Detected {counts.alarm_mismatch_count} stops where logged reason does not align with registered PLC alarm code; verify sensor wiring."

    return None


def compute_max_component_drop(
    quality: QualityReport,
    baseline_sub_scores: Optional[Dict[str, float]] = None,
) -> Tuple[float, str]:
    """Compute the largest drop of any quality sub-score below its clean baseline."""
    if baseline_sub_scores is None:
        baseline_sub_scores, _ = load_quality_baseline()
    worst_comp = "completeness"
    max_drop = -1.0
    for comp, score in quality.sub_scores.items():
        base_val = float(baseline_sub_scores.get(comp, 1.0))
        drop = base_val - float(score)
        if drop > max_drop:
            max_drop = drop
            worst_comp = comp
    return float(max(0.0, max_drop)), worst_comp


def apply_policy(
    df: pd.DataFrame,
    quality: QualityReport,
    stability: StabilityReport,
    tau_q: float = 0.75,
    tau_s: float = 0.80,
    buffer: float = 0.05,
    require_margin_ci_positive: bool = False,
    gate_mode: str = "weighted",
    delta: float = 0.15,
    baseline_sub_scores: Optional[Dict[str, float]] = None,
) -> PolicyDecision:
    """Evaluate abstention policy against quality and stability thresholds."""
    q_score = quality.overall_score
    s_score = stability.top_stability
    max_drop, drop_comp = compute_max_component_drop(quality, baseline_sub_scores=baseline_sub_scores)

    if gate_mode == "component":
        passes_quality = max_drop <= delta
        is_quality_green = max_drop <= max(0.0, delta - buffer)
    else:
        passes_quality = q_score >= tau_q
        is_quality_green = q_score >= (tau_q + buffer)

    passes_stability = s_score >= tau_s
    passes_ci = (not require_margin_ci_positive) or (not stability.margin_ci_crosses_zero)

    answered = bool(passes_quality and passes_stability and passes_ci)

    if answered:
        is_green = is_quality_green and (s_score >= (tau_s + buffer))
        status_flag = "GREEN" if is_green else "AMBER"
        reason = "Quality and stability exceed required evidence thresholds."
    else:
        status_flag = "RED"
        failed_reasons = []
        if not passes_quality:
            if gate_mode == "component":
                failed_reasons.append(f"Component drop on {drop_comp} ({max_drop:.2f}) > delta ({delta:.2f})")
            else:
                failed_reasons.append(f"Quality ({q_score:.2f}) < tau_q ({tau_q:.2f})")
        if not passes_stability:
            failed_reasons.append(f"Stability ({s_score:.2f}) < tau_s ({tau_s:.2f})")
        if not passes_ci:
            failed_reasons.append("Top candidate margin CI crosses zero")
        reason = "INSUFFICIENT EVIDENCE: " + "; ".join(failed_reasons)

    top_cand = stability.top_candidate or ("Unknown", "Unknown")
    supporting_events = 0
    share_of_downtime = 0.0
    ci_lower = 0.0
    ci_upper = 0.0

    if stability.candidate_cis:
        for c in stability.candidate_cis:
            if (c.machine_id, c.logged_reason) == top_cand:
                share_of_downtime = c.mean_share
                ci_lower = c.ci_lower_95
                ci_upper = c.ci_upper_95
                break

    if not df.empty and top_cand != ("Unknown", "Unknown"):
        t_mask = (df["machine_id"].astype(str) == top_cand[0]) & (df["logged_reason"].astype(str) == top_cand[1])
        supporting_events = int(t_mask.sum())

    sorted_subs = dict(sorted(quality.sub_scores.items(), key=lambda kv: kv[1])[:2])
    v_msg = generate_verification_message(quality, stability, df)

    ev_summary = EvidenceSummary(
        top_candidate=top_cand,
        supporting_events_count=supporting_events,
        share_of_total_downtime=share_of_downtime,
        ci_lower_95=ci_lower,
        ci_upper_95=ci_upper,
        stability_pct=round(s_score * 100.0, 1),
        quality_score=q_score,
        weakest_sub_scores=sorted_subs,
        verification_message=v_msg,
    )

    return PolicyDecision(
        answered=answered,
        status_flag=status_flag,
        decision_reason=reason,
        tau_q=tau_q,
        tau_s=tau_s,
        quality_score=q_score,
        stability_score=s_score,
        evidence_summary=ev_summary,
    )


def tune_policy_on_validation(
    scenarios_dir: str | Path = "data/sample_scenarios",
    min_coverage: float = 0.60,
    out_params_path: str | Path = TUNED_PARAMS_PATH,
    grid_steps: int = 15,
    use_train_split: bool = False,
) -> Dict[str, Any]:
    """Tune policy thresholds by grid search on VALIDATION (or pooled TRAIN+VALIDATION) split.

    Defaults to `use_train_split=False` (strictly the 60 validation scenarios) so existing
    validation-isolation tests (`assert num_validation_scenarios == 60`) pass; pass
    `use_train_split=True` to pool TRAIN+VALIDATION (180 scenarios). Never touches TEST.
    """
    from faultlens.defects import ALLOWED_DEFECTS, inject_defect
    from faultlens.diagnoser import diagnose
    from faultlens.generator import Scenario

    base_path = Path(scenarios_dir)
    splits_file = base_path / "splits.json"

    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_ids = list(splits.get("train", []))
    val_ids = list(splits.get("val", []))
    test_ids = set(splits.get("test", []))

    if not val_ids:
        raise ValueError("No validation scenarios found in splits.json")

    baseline_subs, exp_micro = load_quality_baseline()

    tuning_ids = (train_ids + val_ids) if use_train_split else val_ids
    if set(tuning_ids).intersection(test_ids):
        raise RuntimeError("Data leakage error: TEST scenario IDs found in tuning split!")

    tuning_scenarios = [Scenario.load(base_path / sc_id) for sc_id in tuning_ids]

    eval_records = []
    conditions = [("clean", None, 0.0)]
    for d in ALLOWED_DEFECTS:
        for s in [0.10, 0.20, 0.30]:
            conditions.append((d, d, s))

    print(f"Evaluating {len(tuning_scenarios)} validation scenarios across {len(conditions)} conditions...")

    for sc_idx, sc in enumerate(tuning_scenarios):
        for c_name, defect_type, sev in conditions:
            df = sc.log
            if defect_type is not None:
                df, _ = inject_defect(df, defect_type=defect_type, severity=sev, seed=1000 + sc_idx)

            diag = diagnose(df)
            q_rep = evaluate_data_quality(
                df,
                baseline_sub_scores=baseline_subs,
                expected_micro_stop_rates=exp_micro,
            )
            s_rep = evaluate_stability(df, num_perturbations=50, seed=5000 + sc_idx)
            max_drop, _ = compute_max_component_drop(q_rep, baseline_sub_scores=baseline_subs)

            is_correct = bool(diag.top_candidate == sc.truth)
            eval_records.append({
                "scenario_id": sc.scenario_id,
                "condition": c_name,
                "severity": sev,
                "is_correct": is_correct,
                "quality_score": q_rep.overall_score,
                "max_component_drop": max_drop,
                "stability_score": s_rep.top_stability,
            })

    pool_df = pd.DataFrame(eval_records)
    n_total = len(pool_df)

    tau_q_candidates = np.linspace(0.40, 0.95, grid_steps)
    tau_s_candidates = np.linspace(0.40, 0.95, grid_steps)

    tradeoff_curve: List[Dict[str, float]] = []
    best_w_tau_q = 0.8321
    best_w_tau_s = 0.40
    best_w_cw = 1.0
    best_w_cov = 0.0

    for t_q in tau_q_candidates:
        for t_s in tau_s_candidates:
            answered_mask = (pool_df["quality_score"] >= t_q) & (pool_df["stability_score"] >= t_s)
            coverage = float(answered_mask.mean())
            cw_rate = float((answered_mask & (~pool_df["is_correct"])).mean())

            tradeoff_curve.append({
                "tau_q": round(float(t_q), 3),
                "tau_s": round(float(t_s), 3),
                "coverage": round(coverage, 4),
                "confidently_wrong_rate": round(cw_rate, 4),
            })

            if coverage >= min_coverage:
                if cw_rate < best_w_cw or (cw_rate == best_w_cw and coverage > best_w_cov):
                    best_w_cw = cw_rate
                    best_w_cov = coverage
                    best_w_tau_q = float(t_q)
                    best_w_tau_s = float(t_s)

    delta_candidates = np.linspace(0.05, 0.35, grid_steps)
    best_c_delta = 0.15
    best_c_tau_s = 0.40
    best_c_cw = 1.0
    best_c_cov = 0.0

    for d_val in delta_candidates:
        for t_s in tau_s_candidates:
            answered_mask = (pool_df["max_component_drop"] <= d_val) & (pool_df["stability_score"] >= t_s)
            coverage = float(answered_mask.mean())
            cw_rate = float((answered_mask & (~pool_df["is_correct"])).mean())

            if coverage >= min_coverage:
                if cw_rate < best_c_cw or (cw_rate == best_c_cw and coverage > best_c_cov):
                    best_c_cw = cw_rate
                    best_c_cov = coverage
                    best_c_delta = float(d_val)
                    best_c_tau_s = float(t_s)

    if best_c_cw < best_w_cw or (best_c_cw == best_w_cw and best_c_cov >= best_w_cov):
        chosen_gate_mode = "component"
        chosen_tau_s = best_c_tau_s
        chosen_cov = best_c_cov
        chosen_cw = best_c_cw
    else:
        chosen_gate_mode = "weighted"
        chosen_tau_s = best_w_tau_s
        chosen_cov = best_w_cov
        chosen_cw = best_w_cw

    result_data = {
        "gate_mode": chosen_gate_mode,
        "delta": round(best_c_delta, 4),
        "tau_q": round(best_w_tau_q, 4),
        "tau_s": round(chosen_tau_s, 4),
        "buffer": 0.05,
        "min_coverage_constraint": min_coverage,
        "achieved_validation_coverage": round(chosen_cov, 4),
        "achieved_validation_confidently_wrong_rate": round(chosen_cw, 4),
        "weighted_mode_summary": {
            "tau_q": round(best_w_tau_q, 4),
            "tau_s": round(best_w_tau_s, 4),
            "coverage": round(best_w_cov, 4),
            "confidently_wrong_rate": round(best_w_cw, 4),
        },
        "component_mode_summary": {
            "delta": round(best_c_delta, 4),
            "tau_s": round(best_c_tau_s, 4),
            "coverage": round(best_c_cov, 4),
            "confidently_wrong_rate": round(best_c_cw, 4),
        },
        "splits_used": ["train", "val"] if use_train_split else ["val"],
        "scenario_ids_used": tuning_ids,
        "num_validation_scenarios": len(tuning_scenarios),
        "total_eval_points": n_total,
        "tradeoff_curve": tradeoff_curve,
    }

    out_file = Path(out_params_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    print(
        f"Tuning complete: gate_mode={chosen_gate_mode}, tau_q={best_w_tau_q:.3f}, "
        f"delta={best_c_delta:.3f}, tau_s={chosen_tau_s:.3f}"
    )
    print(
        f"Weighted mode: Coverage={best_w_cov * 100:.1f}%, CW={best_w_cw * 100:.1f}% | "
        f"Component mode: Coverage={best_c_cov * 100:.1f}%, CW={best_c_cw * 100:.1f}%"
    )
    return result_data


def load_tuned_policy_params(params_path: str | Path = TUNED_PARAMS_PATH) -> Tuple[float, float, float]:
    """Load tuned tau_q, tau_s, and buffer from disk; fall back to defaults if not found."""
    p_path = Path(params_path)
    if p_path.is_file():
        with open(p_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("tau_q", 0.75)), float(data.get("tau_s", 0.80)), float(data.get("buffer", 0.05))
    return 0.75, 0.80, 0.05


def load_tuned_gate_config(params_path: str | Path = TUNED_PARAMS_PATH) -> Dict[str, Any]:
    """Load full gate configuration (gate_mode, delta, tau_q, tau_s, buffer) from disk."""
    p_path = Path(params_path)
    if p_path.is_file():
        with open(p_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "gate_mode": str(data.get("gate_mode", "weighted")),
            "delta": float(data.get("delta", 0.15)),
            "tau_q": float(data.get("tau_q", 0.75)),
            "tau_s": float(data.get("tau_s", 0.80)),
            "buffer": float(data.get("buffer", 0.05)),
        }
    return {
        "gate_mode": "weighted",
        "delta": 0.15,
        "tau_q": 0.75,
        "tau_s": 0.80,
        "buffer": 0.05,
    }
