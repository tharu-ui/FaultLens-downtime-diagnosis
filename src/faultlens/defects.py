"""FaultLens Defect Injectors.

Implements exactly 4 industrial data defects:
1. missing_events (packet drops & unclosed stop intervals)
2. incorrect_reason_codes (confusion matrix & Other/Unknown)
3. timestamp_errors (per-machine clock drift & communication jitter)
4. duplicate_records (retry floods & sensor bounce bursts)

All injectors are pure, seeded, return (corrupted_df, defect_report),
and strictly preserve `true_cause` and scenario ground truth.
"""

from __future__ import annotations

import copy
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

DEFAULT_DEFECTS_CONFIG = Path("configs/defects.yaml")
ALLOWED_DEFECTS = (
    "missing_events",
    "incorrect_reason_codes",
    "timestamp_errors",
    "duplicate_records",
)
ALLOWED_SEVERITIES = (0.0, 0.10, 0.20, 0.30)


def load_defects_config(config_path: str | Path = DEFAULT_DEFECTS_CONFIG) -> Dict[str, Any]:
    """Load defects configuration from YAML."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Defects config file not found at: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_severity(severity: float) -> float:
    """Validate that severity is within expected bounds [0.0, 0.30]."""
    severity = float(round(severity, 4))
    if severity < 0.0 or severity > 0.35:
        raise ValueError(f"Severity {severity} out of supported range [0.0, 0.30]")
    return severity


def inject_missing_events(
    df: pd.DataFrame,
    severity: float,
    seed: int,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Inject missing events: completely drop a portion of rows and null end_ts for another portion.

    Industrial grounding: Wireless packet dropouts and unclosed/abandoned stop tickets.
    """
    severity = _validate_severity(severity)
    corrupted = df.copy(deep=True)
    n_total = len(df)

    if severity == 0.0 or n_total == 0:
        return corrupted, {
            "defect_type": "missing_events",
            "severity": severity,
            "seed": seed,
            "original_count": n_total,
            "corrupted_count": n_total,
            "dropped_count": 0,
            "null_end_ts_count": 0,
            "affected_fraction": 0.0,
        }

    rng = np.random.default_rng(seed)
    n_affected = int(round(n_total * severity))
    n_affected = max(1, min(n_affected, n_total))

    affected_indices = rng.choice(corrupted.index.to_numpy(), size=n_affected, replace=False)
    # 50% dropped, 50% unclosed (null end_ts)
    n_drop = int(round(n_affected * 0.50))
    drop_indices = affected_indices[:n_drop]
    null_indices = affected_indices[n_drop:]

    # Null end_ts and duration_min for unclosed records
    if len(null_indices) > 0:
        corrupted.loc[null_indices, "end_ts"] = pd.NaT
        corrupted.loc[null_indices, "duration_min"] = np.nan

    # Drop rows
    if len(drop_indices) > 0:
        corrupted = corrupted.drop(index=drop_indices).reset_index(drop=True)

    report = {
        "defect_type": "missing_events",
        "severity": severity,
        "seed": seed,
        "original_count": n_total,
        "corrupted_count": len(corrupted),
        "dropped_count": len(drop_indices),
        "null_end_ts_count": len(null_indices),
        "affected_fraction": round(n_affected / n_total, 4),
    }
    return corrupted, report


def inject_incorrect_reason_codes(
    df: pd.DataFrame,
    severity: float,
    seed: int,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Inject misclassified logged reason codes based on confusion map and catch-all Other/Unknown.

    Industrial grounding: Operator HMI misattribution and ambiguous alarm symptoms.
    CRITICAL: Does NOT touch true_cause.
    """
    severity = _validate_severity(severity)
    corrupted = df.copy(deep=True)
    n_total = len(df)

    if config is None:
        cfg_dict = load_defects_config()
    else:
        cfg_dict = config

    confusion_map = cfg_dict.get("defects", {}).get("incorrect_reason_codes", {}).get(
        "confusion_map",
        {
            "sensor_fault": {"electrical_fault": 0.60, "Other/Unknown": 0.40},
            "electrical_fault": {"sensor_fault": 0.65, "Other/Unknown": 0.35},
            "jam": {"material_shortage": 0.70, "Other/Unknown": 0.30},
            "material_shortage": {"jam": 0.60, "Other/Unknown": 0.40},
            "overheating": {"Other/Unknown": 0.70, "electrical_fault": 0.30},
            "tool_wear": {"jam": 0.45, "Other/Unknown": 0.55},
        },
    )

    if severity == 0.0 or n_total == 0:
        return corrupted, {
            "defect_type": "incorrect_reason_codes",
            "severity": severity,
            "seed": seed,
            "original_count": n_total,
            "corrupted_count": n_total,
            "altered_reasons_count": 0,
            "affected_fraction": 0.0,
        }

    rng = np.random.default_rng(seed)
    n_affected = int(round(n_total * severity))
    n_affected = max(1, min(n_affected, n_total))

    target_indices = rng.choice(corrupted.index.to_numpy(), size=n_affected, replace=False)
    altered_count = 0

    for idx in target_indices:
        current_reason = str(corrupted.at[idx, "logged_reason"])
        if current_reason in confusion_map:
            choices_dict = confusion_map[current_reason]
            c_keys = list(choices_dict.keys())
            c_probs = np.array(list(choices_dict.values()), dtype=float)
            c_probs = c_probs / c_probs.sum()
            new_reason = rng.choice(c_keys, p=c_probs)
        else:
            new_reason = "Other/Unknown"

        corrupted.at[idx, "logged_reason"] = new_reason
        altered_count += 1

    report = {
        "defect_type": "incorrect_reason_codes",
        "severity": severity,
        "seed": seed,
        "original_count": n_total,
        "corrupted_count": len(corrupted),
        "altered_reasons_count": altered_count,
        "affected_fraction": round(altered_count / n_total, 4),
    }
    return corrupted, report


def inject_timestamp_errors(
    df: pd.DataFrame,
    severity: float,
    seed: int,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Inject timestamp drift per machine and independent communication jitter.

    Applies per-machine clock offset plus independent Gaussian jitter on start_ts and end_ts,
    causing erroneous durations, negative durations, out-of-order records, and same-machine overlaps.
    The shared diagnoser skips non-positive durations, degrading effective candidate evidence.

    Industrial grounding: Unsynchronized PLC clocks, asynchronous telemetry buffering, and transmission latency.
    CRITICAL: Preserves true_cause and scenario.truth.
    """
    severity = _validate_severity(severity)
    corrupted = df.copy(deep=True)
    n_total = len(df)

    if severity == 0.0 or n_total == 0:
        return corrupted, {
            "defect_type": "timestamp_errors",
            "severity": severity,
            "seed": seed,
            "original_count": n_total,
            "corrupted_count": n_total,
            "machine_offsets": {},
            "jittered_events_count": 0,
            "affected_fraction": 0.0,
            "negative_durations_count": 0,
        }

    rng = np.random.default_rng(seed)

    # Convert to clean datetime64[ns] Series to prevent pandas 3 / 2.x dtype assignment warnings
    starts = pd.to_datetime(corrupted["start_ts"]).copy()
    ends = pd.to_datetime(corrupted["end_ts"]).copy()

    # 1. Machine-level clock drift
    machines = list(corrupted["machine_id"].unique())
    scale_factor = severity / 0.30  # At 30% severity, full 15 min drift
    machine_offsets: Dict[str, float] = {}

    for m in machines:
        offset_min = float(rng.uniform(-15.0 * scale_factor, 15.0 * scale_factor))
        machine_offsets[m] = round(offset_min, 2)
        m_mask = (corrupted["machine_id"] == m).to_numpy()
        offset_delta = pd.to_timedelta(offset_min, unit="m")
        starts.loc[m_mask] = starts.loc[m_mask] + offset_delta
        valid_end_mask = m_mask & ends.notna().to_numpy()
        ends.loc[valid_end_mask] = ends.loc[valid_end_mask] + offset_delta

    # 2. Independent event-level jitter on start_ts and end_ts for affected fraction
    n_jitter = int(round(n_total * severity))
    n_jitter = max(1, min(n_jitter, n_total))
    jitter_indices = rng.choice(corrupted.index.to_numpy(), size=n_jitter, replace=False)

    # Jitter standard deviation scales with severity (up to 14 minutes at 30% severity)
    jitter_std = max(2.5, 14.0 * scale_factor)
    start_jitters = rng.normal(loc=0.0, scale=jitter_std, size=len(jitter_indices))
    end_jitters = rng.normal(loc=0.0, scale=jitter_std, size=len(jitter_indices))

    for i, idx in enumerate(jitter_indices):
        starts.at[idx] = starts.at[idx] + pd.to_timedelta(start_jitters[i], unit="m")
        if pd.notna(ends.at[idx]):
            ends.at[idx] = ends.at[idx] + pd.to_timedelta(end_jitters[i], unit="m")

    # Assign entire columns whole to avoid partial dtype assignment
    corrupted["start_ts"] = starts
    corrupted["end_ts"] = ends

    # Recompute durations: independent jitter distorts durations and creates negative durations
    dur_series = (ends - starts).dt.total_seconds() / 60.0
    corrupted["duration_min"] = dur_series.round(2)

    neg_count = int((corrupted["duration_min"] <= 0).sum())

    report = {
        "defect_type": "timestamp_errors",
        "severity": severity,
        "seed": seed,
        "original_count": n_total,
        "corrupted_count": len(corrupted),
        "machine_offsets": machine_offsets,
        "jittered_events_count": len(jitter_indices),
        "affected_fraction": round(len(jitter_indices) / n_total, 4),
        "negative_durations_count": neg_count,
    }
    return corrupted, report


def inject_duplicate_records(
    df: pd.DataFrame,
    severity: float,
    seed: int,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Inject duplicate records: exact row clones and rapid micro-bursts from sensor bounce.

    Industrial grounding: MQTT at-least-once delivery retries and contact switch bounce.
    """
    severity = _validate_severity(severity)
    corrupted = df.copy(deep=True)
    n_total = len(df)

    if severity == 0.0 or n_total == 0:
        return corrupted, {
            "defect_type": "duplicate_records",
            "severity": severity,
            "seed": seed,
            "original_count": n_total,
            "corrupted_count": n_total,
            "single_dupes_count": 0,
            "burst_dupes_count": 0,
            "affected_fraction": 0.0,
        }

    rng = np.random.default_rng(seed)
    n_affected = int(round(n_total * severity))
    n_affected = max(1, min(n_affected, n_total))

    target_indices = rng.choice(corrupted.index.to_numpy(), size=n_affected, replace=False)
    # 70% single duplicates, 30% burst duplicates
    n_single = int(round(n_affected * 0.70))
    single_indices = target_indices[:n_single]
    burst_indices = target_indices[n_single:]

    new_rows: List[Dict[str, Any]] = []

    # Single duplicates
    for s_idx in single_indices:
        row = corrupted.loc[s_idx].to_dict()
        row["event_id"] = f"{row['event_id']}_dup"
        new_rows.append(row)

    # Burst duplicates (3 to 10 records clustered within 45s)
    burst_total_added = 0
    for b_idx in burst_indices:
        base_row = corrupted.loc[b_idx].to_dict()
        k_burst = int(rng.integers(low=3, high=11))
        for b_sub in range(1, k_burst + 1):
            b_row = dict(base_row)
            b_row["event_id"] = f"{base_row['event_id']}_burst{b_sub}"
            # Small jitter within 45 seconds
            sec_jitter = float(rng.uniform(-20.0, 45.0))
            if pd.notna(b_row["start_ts"]):
                b_row["start_ts"] = b_row["start_ts"] + timedelta(seconds=sec_jitter)
            if pd.notna(b_row["end_ts"]):
                b_row["end_ts"] = b_row["end_ts"] + timedelta(seconds=sec_jitter)
            new_rows.append(b_row)
            burst_total_added += 1

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        corrupted = pd.concat([corrupted, new_df], ignore_index=True)

    report = {
        "defect_type": "duplicate_records",
        "severity": severity,
        "seed": seed,
        "original_count": n_total,
        "corrupted_count": len(corrupted),
        "single_dupes_count": len(single_indices),
        "burst_dupes_count": burst_total_added,
        "affected_fraction": round(n_affected / n_total, 4),
    }
    return corrupted, report


def inject_defect(
    df: pd.DataFrame,
    defect_type: str,
    severity: float,
    seed: int,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Unified entry point for injecting one of the 4 controlled defects.

    Args:
        df: Clean input DataFrame.
        defect_type: One of 'missing_events', 'incorrect_reason_codes', 'timestamp_errors', 'duplicate_records'.
        severity: Float in [0.0, 0.30].
        seed: Random seed for repeatability.
        config: Optional configuration dictionary.

    Returns:
        (corrupted_df, defect_report).
    """
    if defect_type == "missing_events":
        return inject_missing_events(df, severity, seed, config)
    elif defect_type == "incorrect_reason_codes":
        return inject_incorrect_reason_codes(df, severity, seed, config)
    elif defect_type == "timestamp_errors":
        return inject_timestamp_errors(df, severity, seed, config)
    elif defect_type == "duplicate_records":
        return inject_duplicate_records(df, severity, seed, config)
    else:
        raise ValueError(
            f"Invalid defect_type '{defect_type}'. Allowed types are: {ALLOWED_DEFECTS}"
        )
