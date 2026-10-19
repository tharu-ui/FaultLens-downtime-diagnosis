"""FaultLens Data Quality Scoring Module.

Evaluates industrial log quality per machine and per week across 6 core dimensions:
1. Completeness: valid start, end, and reason fields
2. Reason Validity: fraction of non-Other/Unknown reasons
3. Duplication: detection of near-duplicate or retry bursts
4. Timestamp Consistency: positive durations, ordered sequence, no same-machine overlaps
5. Micro-stop Plausibility: observed short-stop rate vs expected baseline
6. Alarm-Reason Agreement: consistency between symptom alarm codes and logged reasons

Provides a weighted composite quality score in [0, 1], identifies the worst component
(by largest drop from the TRAIN clean baseline) and most degraded machine, and tracks
exact countable defects for diagnostic messages.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml

DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiment.yaml")
DEFAULT_GENERATOR_CONFIG = Path("configs/generator.yaml")
QUALITY_BASELINE_PATH = Path("results/quality_baseline.json")

DEFAULT_WEIGHTS = {
    "completeness": 0.20,
    "reason_validity": 0.20,
    "duplication": 0.15,
    "timestamp_consistency": 0.20,
    "micro_stop_plausibility": 0.10,
    "alarm_reason_agreement": 0.15,
}

DEFAULT_BASELINE_SUB_SCORES = {
    "completeness": 1.0,
    "reason_validity": 1.0,
    "duplication": 1.0,
    "timestamp_consistency": 1.0,
    "micro_stop_plausibility": 1.0,
    "alarm_reason_agreement": 1.0,
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
    fallback_mapping: Dict[str, List[str]] = {
        "sensor_fault": ["SENSOR_ALARM", "CALIBRATION_ERR", "COMM_TIMEOUT"],
        "electrical_fault": ["SENSOR_ALARM", "POWER_SURGE", "VOLTAGE_DROP"],
        "overheating": ["THERMAL_HIGH", "TEMP_WARNING", "COOLANT_LOW"],
        "jam": ["FEED_BLOCK", "TRACK_OBSTRUCTION", "MOTOR_OVERLOAD"],
        "tool_wear": ["WEAR_LIMIT", "TORQUE_EXCEEDED", "VIBRATION_WARN"],
        "material_shortage": ["HOPPER_LOW", "FEED_EMPTY", "BUFFER_STARVE"],
    }
    if DEFAULT_GENERATOR_CONFIG.is_file():
        try:
            with open(DEFAULT_GENERATOR_CONFIG, "r", encoding="utf-8") as f:
                gen_cfg = yaml.safe_load(f) or {}
            if isinstance(gen_cfg, dict):
                alarm_cfg = gen_cfg.get("alarm_mapping", {})
                if isinstance(alarm_cfg, dict) and alarm_cfg:
                    mapping: Dict[str, List[str]] = {}
                    for cause, alarms in alarm_cfg.items():
                        valid_alarms: List[str] = []
                        if isinstance(alarms, dict):
                            if "primary" in alarms or "secondary" in alarms:
                                primary = alarms.get("primary", "")
                                secondary = alarms.get("secondary", [])
                                if primary:
                                    valid_alarms.append(str(primary))
                                if isinstance(secondary, list):
                                    valid_alarms.extend(str(a) for a in secondary if a)
                            else:
                                valid_alarms.extend(str(k) for k in alarms.keys())
                        elif isinstance(alarms, list):
                            valid_alarms.extend(str(a) for a in alarms if a)
                        if valid_alarms:
                            mapping[str(cause)] = valid_alarms
                    if mapping:
                        for cause, default_list in fallback_mapping.items():
                            if cause not in mapping:
                                mapping[cause] = default_list
                        return mapping
        except Exception:
            pass
    return fallback_mapping


def load_quality_baseline(
    baseline_path: str | Path = QUALITY_BASELINE_PATH,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Load clean baseline sub-scores and expected micro-stop rates from JSON."""
    path = Path(baseline_path)
    if not path.is_file() and Path("data/sample_scenarios/splits.json").is_file():
        try:
            compute_quality_baseline(out_path=path)
        except Exception:
            pass

    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            base_subs = {
                k: float(v)
                for k, v in data.get("baseline_sub_scores", DEFAULT_BASELINE_SUB_SCORES).items()
            }
            exp_micro = {
                k: float(v)
                for k, v in data.get("expected_micro_stop_rates", EXPECTED_MICRO_STOP_RATES).items()
            }
            for k, v in DEFAULT_BASELINE_SUB_SCORES.items():
                base_subs.setdefault(k, v)
            for k, v in EXPECTED_MICRO_STOP_RATES.items():
                exp_micro.setdefault(k, v)
            return base_subs, exp_micro
        except Exception:
            pass
    return dict(DEFAULT_BASELINE_SUB_SCORES), dict(EXPECTED_MICRO_STOP_RATES)


def compute_quality_baseline(
    scenarios_dir: str | Path = "data/sample_scenarios",
    out_path: str | Path = QUALITY_BASELINE_PATH,
) -> Dict[str, Any]:
    """Derive expected micro-stop rates per machine and mean clean sub-scores from TRAIN split only."""
    from faultlens.generator import Scenario

    base_dir = Path(scenarios_dir)
    splits_file = base_dir / "splits.json"
    if not splits_file.is_file():
        raise FileNotFoundError(f"Splits manifest not found at {splits_file}")

    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_ids = splits.get("train", [])
    if not train_ids:
        raise ValueError("No train scenarios found in splits.json")

    train_scenarios = [Scenario.load(base_dir / sc_id) for sc_id in train_ids]

    machine_micro_counts: Dict[str, int] = {}
    machine_valid_counts: Dict[str, int] = {}

    for sc in train_scenarios:
        df = sc.log
        if df.empty:
            continue
        starts = pd.to_datetime(df["start_ts"])
        ends = pd.to_datetime(df["end_ts"])
        has_valid = starts.notna() & ends.notna()
        durs = (ends[has_valid] - starts[has_valid]).dt.total_seconds() / 60.0
        pos_mask = durs > 0
        valid_durs = durs[pos_mask]
        valid_machines = df.loc[has_valid, "machine_id"].astype(str)[pos_mask]

        for m in valid_machines.unique():
            m_durs = valid_durs[valid_machines == m]
            machine_valid_counts[m] = machine_valid_counts.get(m, 0) + len(m_durs)
            machine_micro_counts[m] = machine_micro_counts.get(m, 0) + int((m_durs < 1.0).sum())

    expected_micro_rates: Dict[str, float] = {}
    for m, default_rate in EXPECTED_MICRO_STOP_RATES.items():
        tot_m = machine_valid_counts.get(m, 0)
        if tot_m > 0:
            expected_micro_rates[m] = round(float(machine_micro_counts.get(m, 0) / tot_m), 4)
        else:
            expected_micro_rates[m] = default_rate

    for m, tot_m in machine_valid_counts.items():
        if m not in expected_micro_rates and tot_m > 0:
            expected_micro_rates[m] = round(float(machine_micro_counts[m] / tot_m), 4)

    sub_accum: Dict[str, List[float]] = {k: [] for k in DEFAULT_WEIGHTS}
    overall_accum: List[float] = []

    for sc in train_scenarios:
        rep = evaluate_data_quality(
            sc.log,
            baseline_sub_scores=DEFAULT_BASELINE_SUB_SCORES,
            expected_micro_stop_rates=expected_micro_rates,
        )
        overall_accum.append(rep.overall_score)
        for k in sub_accum:
            sub_accum[k].append(rep.sub_scores[k])

    baseline_sub_scores = {
        k: round(float(np.mean(vals)), 4) if vals else 1.0
        for k, vals in sub_accum.items()
    }
    baseline_overall = round(float(np.mean(overall_accum)), 4) if overall_accum else 1.0

    payload = {
        "split_used": "train",
        "num_scenarios": len(train_scenarios),
        "baseline_overall_score": baseline_overall,
        "baseline_sub_scores": baseline_sub_scores,
        "expected_micro_stop_rates": expected_micro_rates,
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


def evaluate_data_quality(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    near_dup_window_sec: float = 60.0,
    alarm_mapping: Optional[Dict[str, List[str]]] = None,
    baseline_sub_scores: Optional[Dict[str, float]] = None,
    expected_micro_stop_rates: Optional[Dict[str, float]] = None,
) -> QualityReport:
    """Compute 6-dimensional data quality scoring across machines and weeks."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if alarm_mapping is None:
        alarm_mapping = _load_alarm_mapping()
    if baseline_sub_scores is None or expected_micro_stop_rates is None:
        loaded_base, loaded_micro = load_quality_baseline()
        if baseline_sub_scores is None:
            baseline_sub_scores = loaded_base
        if expected_micro_stop_rates is None:
            expected_micro_stop_rates = loaded_micro

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

    df = df.reset_index(drop=True)

    starts = pd.to_datetime(df["start_ts"])
    ends = pd.to_datetime(df["end_ts"])
    reasons = df["logged_reason"].astype(str)
    machines = df["machine_id"].astype(str)
    alarms = (
        df["alarm_code"].astype(str)
        if "alarm_code" in df.columns
        else pd.Series(["UNKNOWN"] * total_events, index=df.index)
    )

    min_start = starts.min()
    if pd.notna(min_start):
        week_indices = ((starts - min_start).dt.total_seconds() // (7 * 86400)).fillna(0).astype(int)
    else:
        week_indices = pd.Series([0] * total_events, index=df.index)

    unique_machines = list(machines.unique())
    machine_counts: Dict[str, Dict[str, int]] = {m: {} for m in unique_machines}

    missing_start = starts.isna()
    missing_end = ends.isna()
    missing_reason = reasons.isin(["nan", "None", "", "NaN"])
    valid_complete = (~missing_start) & (~missing_end) & (~missing_reason)
    completeness_score = float(valid_complete.mean())

    has_valid_times = (~missing_start) & (~missing_end)
    durations = pd.Series(index=df.index, dtype=float)
    durations[has_valid_times] = (ends[has_valid_times] - starts[has_valid_times]).dt.total_seconds() / 60.0

    is_other_unknown = reasons.str.lower().isin(["other", "unknown", "other/unknown", "nan", "none", ""])
    is_brief_reset = has_valid_times & (durations > 0) & (durations < 1.5)
    unexplained_unknown = is_other_unknown & (~is_brief_reset)
    reason_validity_score = float((~unexplained_unknown).mean())

    duplicate_indices: Set[int] = set()
    for m in unique_machines:
        m_mask = (machines == m) & (~missing_start)
        m_df = pd.DataFrame({
            "idx": df.index[m_mask],
            "start": starts[m_mask],
            "reason": reasons[m_mask],
        }).sort_values(["start", "idx"])

        if len(m_df) > 1:
            for _, r_grp in m_df.groupby("reason", sort=False):
                if len(r_grp) <= 1:
                    continue
                r_starts = r_grp["start"].to_numpy()
                r_idxs = r_grp["idx"].to_numpy(dtype=int)
                diffs = (pd.Series(r_starts).diff().dt.total_seconds().abs()).to_numpy()

                cluster: List[int] = [int(r_idxs[0])]
                for pos in range(1, len(r_idxs)):
                    if diffs[pos] <= near_dup_window_sec:
                        cluster.append(int(r_idxs[pos]))
                    else:
                        if len(cluster) > 1:
                            keep_idx = min(cluster)
                            for c_idx in cluster:
                                if c_idx != keep_idx:
                                    duplicate_indices.add(c_idx)
                        cluster = [int(r_idxs[pos])]
                if len(cluster) > 1:
                    keep_idx = min(cluster)
                    for c_idx in cluster:
                        if c_idx != keep_idx:
                            duplicate_indices.add(c_idx)

    duplication_count = len(duplicate_indices)
    duplication_score = float(max(0.0, 1.0 - (duplication_count / total_events)))
    is_duplicate_series = df.index.isin(duplicate_indices)

    negative_durations = has_valid_times & (~is_duplicate_series) & (durations <= 0)
    negative_count = int(negative_durations.sum())

    overlapping_indices: Set[int] = set()
    for m in unique_machines:
        m_mask = (machines == m) & has_valid_times & (~negative_durations) & (~is_duplicate_series)
        m_df = pd.DataFrame({
            "idx": df.index[m_mask],
            "start": starts[m_mask],
            "end": ends[m_mask],
        }).sort_values(["start", "idx"])

        if len(m_df) > 1:
            prev_end = m_df["end"].shift(1)
            overlap_mask = m_df["start"] < prev_end
            for bad_idx in m_df.loc[overlap_mask, "idx"]:
                overlapping_indices.add(int(bad_idx))

    overlapping_count = len(overlapping_indices)
    timestamp_defects = negative_count + overlapping_count
    non_dup_total = max(1, total_events - duplication_count)
    timestamp_consistency_score = float(max(0.0, 1.0 - (timestamp_defects / non_dup_total)))

    valid_pos_non_dup = has_valid_times & (durations > 0) & (~is_duplicate_series)
    is_micro = valid_pos_non_dup & (durations < 1.0)
    valid_pos_count = int(valid_pos_non_dup.sum())

    if valid_pos_count > 0:
        overall_observed = float(is_micro.sum() / valid_pos_count)
        expected_overall = float(
            sum(
                int((valid_pos_non_dup & (machines == m)).sum()) * expected_micro_stop_rates.get(m, 0.10)
                for m in unique_machines
            )
            / valid_pos_count
        )
    else:
        overall_observed = 0.0
        expected_overall = float(np.mean(list(expected_micro_stop_rates.values()))) if expected_micro_stop_rates else 0.10

    micro_dev = max(0.0, abs(overall_observed - expected_overall) - 0.10)
    micro_plausibility_score = float(max(0.0, 1.0 - (micro_dev / 0.20)))

    alarm_agrees_list: List[bool] = []
    for r_val, a_val, is_unk in zip(reasons, alarms, is_other_unknown):
        if is_unk:
            alarm_agrees_list.append(True)
        elif r_val in alarm_mapping:
            allowed = alarm_mapping[r_val]
            alarm_agrees_list.append(a_val in allowed)
        else:
            alarm_agrees_list.append(True)

    alarm_agrees_series = pd.Series(alarm_agrees_list, index=df.index, dtype=bool)
    known_mask = ~is_other_unknown
    known_count = int(known_mask.sum())
    alarm_mismatches = int((known_mask & (~alarm_agrees_series)).sum())
    if known_count > 0:
        alarm_reason_agreement_score = float(max(0.0, 1.0 - (alarm_mismatches / known_count)))
    else:
        alarm_reason_agreement_score = 1.0

    sub_scores = {
        "completeness": round(completeness_score, 4),
        "reason_validity": round(reason_validity_score, 4),
        "duplication": round(duplication_score, 4),
        "timestamp_consistency": round(timestamp_consistency_score, 4),
        "micro_stop_plausibility": round(micro_plausibility_score, 4),
        "alarm_reason_agreement": round(alarm_reason_agreement_score, 4),
    }

    w_sum = sum(weights.values())
    overall_score = float(sum(sub_scores[k] * weights.get(k, 0.0) for k in sub_scores) / w_sum)
    overall_score = float(np.clip(overall_score, 0.0, 1.0))

    machine_scores: Dict[str, Dict[str, float]] = {}
    machine_overall: Dict[str, float] = {}

    for m in unique_machines:
        m_mask = machines == m
        m_n = int(m_mask.sum())
        if m_n == 0:
            continue

        m_comp = float((valid_complete & m_mask).sum() / m_n)
        m_valid = float(((~unexplained_unknown) & m_mask).sum() / m_n)
        m_dups = len(duplicate_indices.intersection(df.index[m_mask]))
        m_dup = float(max(0.0, 1.0 - (m_dups / m_n)))
        m_neg = int((negative_durations & m_mask).sum())
        m_ovlp = len(overlapping_indices.intersection(df.index[m_mask]))
        m_non_dup_n = max(1, m_n - m_dups)
        m_time = float(max(0.0, 1.0 - ((m_neg + m_ovlp) / m_non_dup_n)))

        m_valid_pos_mask = valid_pos_non_dup & m_mask
        m_valid_pos_n = int(m_valid_pos_mask.sum())
        m_exp = expected_micro_stop_rates.get(m, EXPECTED_MICRO_STOP_RATES.get(m, 0.10))
        if m_valid_pos_n > 0:
            m_micro_rate = float((is_micro & m_valid_pos_mask).sum() / m_valid_pos_n)
        else:
            m_micro_rate = m_exp
        m_micro_dev = max(0.0, abs(m_micro_rate - m_exp) - 0.10)
        m_micro = float(max(0.0, 1.0 - (m_micro_dev / 0.20)))

        m_known_mask = m_mask & known_mask
        m_known_n = int(m_known_mask.sum())
        m_alarm_mis = int((m_known_mask & (~alarm_agrees_series)).sum())
        m_alarm = float(max(0.0, 1.0 - (m_alarm_mis / m_known_n))) if m_known_n > 0 else 1.0

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
            "duplicates": m_dups,
            "negative_durations": m_neg,
            "overlaps": m_ovlp,
            "alarm_mismatches": m_alarm_mis,
        }

    worst_component = max(
        sub_scores.keys(),
        key=lambda k: (
            round(float(baseline_sub_scores.get(k, 1.0)) - sub_scores[k], 4),
            -sub_scores[k],
        ),
    )
    worst_machine = min(machine_overall.keys(), key=lambda m: machine_overall[m]) if machine_overall else "none"

    weekly_scores: Dict[int, float] = {}
    for w in sorted(week_indices.unique()):
        w_mask = week_indices == w
        if w_mask.sum() > 0:
            w_comp = float((valid_complete & w_mask).mean())
            w_valid = float(((~unexplained_unknown) & w_mask).mean())
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
