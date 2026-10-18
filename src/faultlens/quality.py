"""FaultLens Data Quality Scoring Module.

Evaluates industrial log quality per machine and per week across 6 core dimensions:
1. Completeness: valid start, end, and reason fields
2. Reason Validity: fraction of non-Other/Unknown reasons
3. Duplication: detection of near-duplicate or retry bursts
4. Timestamp Consistency: positive durations, ordered sequence, no same-machine overlaps
5. Micro-stop Plausibility: observed short-stop rate vs expected baseline
6. Alarm-Reason Agreement: consistency between symptom alarm codes and logged reasons

Provides a weighted composite quality score in [0, 1], identifies the worst component
and most degraded machine, and tracks exact countable defects for diagnostic messages.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiment.yaml")
DEFAULT_GENERATOR_CONFIG = Path("configs/generator.yaml")

DEFAULT_WEIGHTS = {
    "completeness": 0.20,
    "reason_validity": 0.20,
    "duplication": 0.15,
    "timestamp_consistency": 0.20,
    "micro_stop_plausibility": 0.10,
    "alarm_reason_agreement": 0.15,
}

EXPECTED_MICRO_STOP_RATES = {
    "Loader": 0.12,
    "Press": 0.08,
    "Welder": 0.10,
    "Packer": 0.15,
}


@dataclass
class QualityCounts:
    """Countable defect tallies used for generating actionable verification guidance."""

    total_events: int = 0
    missing_start_count: int = 0
    missing_end_count: int = 0
    missing_reason_count: int = 0
    other_unknown_count: int = 0
    duplicate_burst_count: int = 0
    negative_duration_count: int = 0
    overlapping_event_count: int = 0
    alarm_mismatch_count: int = 0
    micro_stop_count: int = 0
    machine_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Comprehensive data quality evaluation output."""

    overall_score: float
    sub_scores: Dict[str, float]
    worst_component: str
    worst_machine: str
    machine_scores: Dict[str, Dict[str, float]]
    weekly_scores: Dict[int, float]
    counts: QualityCounts

    def to_dict(self) -> Dict[str, Any]:
        """Serialize quality report to dictionary."""
        return {
            "overall_score": round(self.overall_score, 4),
            "sub_scores": {k: round(v, 4) for k, v in self.sub_scores.items()},
            "worst_component": self.worst_component,
            "worst_machine": self.worst_machine,
            "machine_scores": {
                m: {k: round(v, 4) for k, v in scores.items()}
                for m, scores in self.machine_scores.items()
            },
            "weekly_scores": {k: round(v, 4) for k, v in self.weekly_scores.items()},
            "counts": asdict(self.counts),
        }


def _load_alarm_mapping() -> Dict[str, List[str]]:
    """Load valid alarm codes associated with each cause from generator.yaml."""
    if DEFAULT_GENERATOR_CONFIG.is_file():
        with open(DEFAULT_GENERATOR_CONFIG, "r", encoding="utf-8") as f:
            gen_cfg = yaml.safe_load(f)
        alarm_cfg = gen_cfg.get("alarm_mapping", {})
        mapping: Dict[str, List[str]] = {}
        for cause, alarms in alarm_cfg.items():
            valid_alarms = [alarms.get("primary", "")] + alarms.get("secondary", [])
            mapping[cause] = [a for a in valid_alarms if a]
        return mapping
    return {
        "sensor_fault": ["SENSOR_ALARM", "CALIBRATION_ERR", "COMM_TIMEOUT"],
        "electrical_fault": ["SENSOR_ALARM", "POWER_SURGE", "VOLTAGE_DROP"],
        "overheating": ["THERMAL_HIGH", "TEMP_WARNING", "COOLANT_LOW"],
        "jam": ["FEED_BLOCK", "TRACK_OBSTRUCTION", "MOTOR_OVERLOAD"],
        "tool_wear": ["WEAR_LIMIT", "TORQUE_EXCEEDED", "VIBRATION_WARN"],
        "material_shortage": ["HOPPER_LOW", "FEED_EMPTY", "BUFFER_STARVE"],
    }


def evaluate_data_quality(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    near_dup_window_sec: float = 60.0,
    alarm_mapping: Optional[Dict[str, List[str]]] = None,
) -> QualityReport:
    """Compute 6-dimensional data quality scoring across machines and weeks.

    Args:
        df: Input machine-event DataFrame.
        weights: Sub-score weights summing to 1.0. Defaults to experiment.yaml weights.
        near_dup_window_sec: Time window (seconds) to cluster near-duplicate records.
        alarm_mapping: Allowed alarm codes per reason code.

    Returns:
        QualityReport with overall score, sub-scores, worst machine, and countable tallies.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if alarm_mapping is None:
        alarm_mapping = _load_alarm_mapping()

    total_events = len(df)
    if total_events == 0:
        empty_sub = {k: 0.0 for k in weights}
        return QualityReport(
            overall_score=0.0,
            sub_scores=empty_sub,
            worst_component="completeness",
            worst_machine="none",
            machine_scores={},
            weekly_scores={},
            counts=QualityCounts(),
        )

    # Standardize timestamps
    starts = pd.to_datetime(df["start_ts"])
    ends = pd.to_datetime(df["end_ts"])
    reasons = df["logged_reason"].astype(str)
    machines = df["machine_id"].astype(str)
    alarms = df["alarm_code"].astype(str) if "alarm_code" in df.columns else pd.Series(["UNKNOWN"] * total_events)

    # Calculate weekly segments
    min_start = starts.min()
    if pd.notna(min_start):
        week_indices = ((starts - min_start).dt.total_seconds() // (7 * 86400)).fillna(0).astype(int)
    else:
        week_indices = pd.Series([0] * total_events)

    unique_machines = list(machines.unique())
    machine_counts: Dict[str, Dict[str, int]] = {m: {} for m in unique_machines}

    # 1. Completeness: valid start, end, and reason
    missing_start = starts.isna()
    missing_end = ends.isna()
    missing_reason = reasons.isin(["nan", "None", "", "NaN"])
    valid_complete = (~missing_start) & (~missing_end) & (~missing_reason)
    completeness_score = float(valid_complete.mean())

    # 2. Reason Validity: fraction not in Other/Unknown
    is_other_unknown = reasons.str.lower().isin(["other", "unknown", "other/unknown", "nan", "none", ""])
    reason_validity_score = float((~is_other_unknown).mean())

    # 3. Timestamp Consistency & Negative Durations
    has_valid_times = (~missing_start) & (~missing_end)
    durations = pd.Series(index=df.index, dtype=float)
    durations[has_valid_times] = (ends[has_valid_times] - starts[has_valid_times]).dt.total_seconds() / 60.0

    negative_durations = has_valid_times & (durations <= 0)
    negative_count = int(negative_durations.sum())

    # Overlaps within the same machine
    overlapping_indices = set()
    for m in unique_machines:
        m_mask = (machines == m) & has_valid_times & (~negative_durations)
        m_df = pd.DataFrame({
            "idx": df.index[m_mask],
            "start": starts[m_mask],
            "end": ends[m_mask],
        }).sort_values("start")

        if len(m_df) > 1:
            prev_end = m_df["end"].shift(1)
            # An overlap occurs if start of event i < end of event i-1
            overlap_mask = m_df["start"] < prev_end
            for bad_idx in m_df.loc[overlap_mask, "idx"]:
                overlapping_indices.add(bad_idx)

    overlapping_count = len(overlapping_indices)
    timestamp_defects = negative_count + overlapping_count
    timestamp_consistency_score = float(max(0.0, 1.0 - (timestamp_defects / total_events)))

    # 4. Duplication: exact or near-duplicates (same machine, same reason, start within window)
    duplicate_indices = set()
    for m in unique_machines:
        m_mask = (machines == m) & (~missing_start)
        m_df = pd.DataFrame({
            "idx": df.index[m_mask],
            "start": starts[m_mask],
            "reason": reasons[m_mask],
        }).sort_values("start")

        if len(m_df) > 1:
            time_diff = (m_df["start"] - m_df["start"].shift(1)).dt.total_seconds().abs()
            same_reason = m_df["reason"] == m_df["reason"].shift(1)
            is_dup = same_reason & (time_diff <= near_dup_window_sec)
            for dup_idx in m_df.loc[is_dup, "idx"]:
                duplicate_indices.add(dup_idx)

    duplication_count = len(duplicate_indices)
    duplication_score = float(max(0.0, 1.0 - (duplication_count / total_events)))

    # 5. Micro-stop Plausibility: checks whether short-stop rate is within plausible industrial band [0.03, 0.25]
    is_micro = has_valid_times & (durations < 1.0) & (durations > 0)
    overall_observed = float(is_micro.sum() / max(1, total_events))
    if 0.03 <= overall_observed <= 0.25:
        micro_plausibility_score = 1.0
    else:
        diff = min(abs(overall_observed - 0.03), abs(overall_observed - 0.25))
        micro_plausibility_score = float(max(0.0, 1.0 - (diff / 0.15)))

    # 6. Alarm-Reason Agreement
    alarm_agreements: List[bool] = []
    for r_val, a_val in zip(reasons, alarms):
        if r_val in alarm_mapping:
            allowed = alarm_mapping[r_val]
            alarm_agreements.append(a_val in allowed)
        elif r_val.lower() in ["other", "unknown", "other/unknown"]:
            # Other/Unknown cannot verify alarm agreement
            alarm_agreements.append(False)
        else:
            alarm_agreements.append(True)

    alarm_mismatches = sum(not ag for ag in alarm_agreements)
    alarm_reason_agreement_score = float(np.mean(alarm_agreements)) if alarm_agreements else 1.0

    sub_scores = {
        "completeness": round(completeness_score, 4),
        "reason_validity": round(reason_validity_score, 4),
        "duplication": round(duplication_score, 4),
        "timestamp_consistency": round(timestamp_consistency_score, 4),
        "micro_stop_plausibility": round(micro_plausibility_score, 4),
        "alarm_reason_agreement": round(alarm_reason_agreement_score, 4),
    }

    # Weighted overall score
    w_sum = sum(weights.values())
    overall_score = float(sum(sub_scores[k] * weights.get(k, 0.0) for k in sub_scores) / w_sum)
    overall_score = float(np.clip(overall_score, 0.0, 1.0))

    # Evaluate per-machine scores
    machine_scores: Dict[str, Dict[str, float]] = {}
    machine_overall: Dict[str, float] = {}

    for m in unique_machines:
        m_mask = machines == m
        m_n = int(m_mask.sum())
        if m_n == 0:
            continue

        m_comp = float((valid_complete & m_mask).sum() / m_n)
        m_valid = float(((~is_other_unknown) & m_mask).sum() / m_n)
        m_dup = float(max(0.0, 1.0 - (len(duplicate_indices.intersection(df.index[m_mask])) / m_n)))
        m_neg = (negative_durations & m_mask).sum()
        m_ovlp = len(overlapping_indices.intersection(df.index[m_mask]))
        m_time = float(max(0.0, 1.0 - ((m_neg + m_ovlp) / m_n)))
        m_micro_rate = (is_micro & m_mask).sum() / m_n
        m_exp = EXPECTED_MICRO_STOP_RATES.get(m, 0.10)
        m_micro = float(max(0.0, 1.0 - (abs(m_micro_rate - m_exp) / max(m_exp, 0.12))))
        m_alarm = float(np.mean([ag for ag, mask in zip(alarm_agreements, m_mask) if mask]))

        m_subs = {
            "completeness": round(m_comp, 4),
            "reason_validity": round(m_valid, 4),
            "duplication": round(m_dup, 4),
            "timestamp_consistency": round(m_time, 4),
            "micro_stop_plausibility": round(m_micro, 4),
            "alarm_reason_agreement": round(m_alarm, 4),
        }
        m_score = float(sum(m_subs[k] * weights.get(k, 0.0) for k in m_subs) / w_sum)
        machine_scores[m] = m_subs
        machine_overall[m] = m_score

        machine_counts[m] = {
            "events": m_n,
            "missing_end": int((missing_end & m_mask).sum()),
            "other_unknown": int((is_other_unknown & m_mask).sum()),
            "duplicates": len(duplicate_indices.intersection(df.index[m_mask])),
            "negative_durations": int(m_neg),
            "overlaps": int(m_ovlp),
            "alarm_mismatches": int(sum(not ag for ag, mask in zip(alarm_agreements, m_mask) if mask)),
        }

    worst_component = min(sub_scores.keys(), key=lambda k: sub_scores[k])
    worst_machine = min(machine_overall.keys(), key=lambda m: machine_overall[m]) if machine_overall else "none"

    # Weekly score aggregation
    weekly_scores: Dict[int, float] = {}
    for w in sorted(week_indices.unique()):
        w_mask = week_indices == w
        if w_mask.sum() > 0:
            w_comp = float((valid_complete & w_mask).mean())
            w_valid = float(((~is_other_unknown) & w_mask).mean())
            w_score = 0.5 * w_comp + 0.5 * w_valid
            weekly_scores[int(w)] = round(w_score, 4)

    counts = QualityCounts(
        total_events=total_events,
        missing_start_count=int(missing_start.sum()),
        missing_end_count=int(missing_end.sum()),
        missing_reason_count=int(missing_reason.sum()),
        other_unknown_count=int(is_other_unknown.sum()),
        duplicate_burst_count=duplication_count,
        negative_duration_count=negative_count,
        overlapping_event_count=overlapping_count,
        alarm_mismatch_count=alarm_mismatches,
        micro_stop_count=int(is_micro.sum()),
        machine_counts=machine_counts,
    )

    return QualityReport(
        overall_score=round(overall_score, 4),
        sub_scores=sub_scores,
        worst_component=worst_component,
        worst_machine=worst_machine,
        machine_scores=machine_scores,
        weekly_scores=weekly_scores,
        counts=counts,
    )
