"""FaultLens Log Repair Module (Session 3.5).

Repairs a machine-event log by:
1. Merging duplicate bursts on the same machine and logged_reason within a short window.
2. Imputing missing or non-positive durations with the per-machine median positive duration.
3. Re-inferring logged_reason from alarm_code when logged_reason is Other or Unknown,
   using the configured alarm mapping.

Never alters `true_cause` or ground-truth `scenario.truth`. Uses the existing
shared `diagnose()` function unchanged to measure whether the top candidate
changes after repair, the change in margin, and counts of repaired rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from faultlens.diagnoser import DiagnosisResult, diagnose
from faultlens.quality import _load_alarm_mapping


@dataclass
class RepairReport:
    """Output of deterministic log repair and re-diagnosis comparison."""

    repaired_df: pd.DataFrame
    original_top_candidate: Optional[Tuple[str, str]]
    repaired_top_candidate: Optional[Tuple[str, str]]
    top_candidate_changed: bool
    original_margin_ratio: float
    repaired_margin_ratio: float
    margin_change: float
    abs_margin_change: float
    original_raw_confidence: float
    repaired_raw_confidence: float
    merged_duplicates_count: int
    imputed_durations_count: int
    reinferred_reasons_count: int
    total_repaired_rows: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize repair report metadata (excluding DataFrame)."""
        return {
            "original_top_candidate": list(self.original_top_candidate) if self.original_top_candidate else None,
            "repaired_top_candidate": list(self.repaired_top_candidate) if self.repaired_top_candidate else None,
            "top_candidate_changed": self.top_candidate_changed,
            "original_margin_ratio": round(self.original_margin_ratio, 4),
            "repaired_margin_ratio": round(self.repaired_margin_ratio, 4),
            "margin_change": round(self.margin_change, 4),
            "abs_margin_change": round(self.abs_margin_change, 4),
            "original_raw_confidence": round(self.original_raw_confidence, 4),
            "repaired_raw_confidence": round(self.repaired_raw_confidence, 4),
            "merged_duplicates_count": self.merged_duplicates_count,
            "imputed_durations_count": self.imputed_durations_count,
            "reinferred_reasons_count": self.reinferred_reasons_count,
            "total_repaired_rows": self.total_repaired_rows,
        }


def _build_alarm_to_cause_map(alarm_mapping: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Invert cause -> [alarms] mapping into alarm_code -> [candidate_causes]."""
    inv: Dict[str, List[str]] = {}
    for cause, alarms in alarm_mapping.items():
        for a_code in alarms:
            inv.setdefault(str(a_code), [])
            if str(cause) not in inv[str(a_code)]:
                inv[str(a_code)].append(str(cause))
    return inv


def repair_log(
    df: pd.DataFrame,
    near_dup_window_sec: float = 60.0,
    alarm_mapping: Optional[Dict[str, List[str]]] = None,
) -> RepairReport:
    """Repair a telemetry log and compare shared diagnoser output before vs after repair.

    Steps:
    1. Merge duplicate bursts on the same machine and logged_reason within `near_dup_window_sec`.
    2. Impute missing (`end_ts` null) or non-positive (`duration_min <= 0`) durations using
       the per-machine median positive duration.
    3. Re-infer `logged_reason` from `alarm_code` when `logged_reason` is Other/Unknown.
       Never modifies `true_cause`.
    """
    orig_diag: DiagnosisResult = diagnose(df)

    if df.empty:
        return RepairReport(
            repaired_df=df.copy(),
            original_top_candidate=orig_diag.top_candidate,
            repaired_top_candidate=orig_diag.top_candidate,
            top_candidate_changed=False,
            original_margin_ratio=orig_diag.margin_ratio,
            repaired_margin_ratio=orig_diag.margin_ratio,
            margin_change=0.0,
            abs_margin_change=0.0,
            original_raw_confidence=orig_diag.raw_confidence,
            repaired_raw_confidence=orig_diag.raw_confidence,
            merged_duplicates_count=0,
            imputed_durations_count=0,
            reinferred_reasons_count=0,
            total_repaired_rows=0,
        )

    if alarm_mapping is None:
        alarm_mapping = _load_alarm_mapping()
    alarm_to_causes = _build_alarm_to_cause_map(alarm_mapping)

    work = df.copy().reset_index(drop=True)
    starts = pd.to_datetime(work["start_ts"])
    ends = pd.to_datetime(work["end_ts"])
    machines = work["machine_id"].astype(str)
    reasons = work["logged_reason"].astype(str)

    # 1. Merge duplicate bursts on the same machine and logged_reason within near_dup_window_sec
    duplicate_indices: Set[int] = set()
    for m in machines.unique():
        m_mask = (machines == m) & starts.notna()
        m_sub = pd.DataFrame({
            "idx": work.index[m_mask],
            "start": starts[m_mask],
            "reason": reasons[m_mask],
        }).sort_values(["start", "idx"])

        if len(m_sub) > 1:
            for _, r_grp in m_sub.groupby("reason", sort=False):
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

    merged_duplicates_count = len(duplicate_indices)
    if merged_duplicates_count > 0:
        keep_mask = ~work.index.isin(duplicate_indices)
        work = work.loc[keep_mask].copy().reset_index(drop=True)

    # Re-extract series after dropping duplicate burst rows
    starts = pd.to_datetime(work["start_ts"])
    ends = pd.to_datetime(work["end_ts"])
    machines = work["machine_id"].astype(str)
    reasons = work["logged_reason"].astype(str)
    alarms = (
        work["alarm_code"].astype(str)
        if "alarm_code" in work.columns
        else pd.Series(["UNKNOWN"] * len(work), index=work.index)
    )

    # 2. Impute missing or non-positive durations with per-machine median positive duration
    calc_durs = pd.Series(np.nan, index=work.index, dtype=float)
    valid_ts_mask = starts.notna() & ends.notna()
    calc_durs[valid_ts_mask] = (ends[valid_ts_mask] - starts[valid_ts_mask]).dt.total_seconds() / 60.0

    if "duration_min" in work.columns:
        raw_durs = pd.to_numeric(work["duration_min"], errors="coerce")
        durs = raw_durs.where(raw_durs.notna() & valid_ts_mask, calc_durs)
    else:
        durs = calc_durs

    pos_mask = valid_ts_mask & durs.notna() & (durs > 0.0)
    global_median = float(durs[pos_mask].median()) if pos_mask.any() else 15.0
    if not np.isfinite(global_median) or global_median <= 0.0:
        global_median = 15.0

    machine_medians: Dict[str, float] = {}
    for m in machines.unique():
        m_pos = durs[pos_mask & (machines == m)]
        if len(m_pos) > 0:
            med_val = float(m_pos.median())
            machine_medians[m] = med_val if (np.isfinite(med_val) and med_val > 0.0) else global_median
        else:
            machine_medians[m] = global_median

    needs_imputation = (~valid_ts_mask) | durs.isna() | (durs <= 0.0)
    imputed_durations_count = int(needs_imputation.sum())

    repaired_durs = durs.copy()
    repaired_ends = ends.copy()
    if imputed_durations_count > 0:
        for idx in work.index[needs_imputation]:
            m_val = str(machines.loc[idx])
            imp_dur = round(machine_medians.get(m_val, global_median), 2)
            repaired_durs.loc[idx] = imp_dur
            if pd.notna(starts.loc[idx]):
                repaired_ends.loc[idx] = starts.loc[idx] + pd.Timedelta(float(imp_dur), unit="min")
    work["duration_min"] = repaired_durs
    work["end_ts"] = repaired_ends

    # 3. Re-infer logged_reason from alarm_code when logged_reason is Other or Unknown
    unknown_tokens = {"other", "unknown", "other/unknown", "nan", "none", ""}
    is_unk_mask = reasons.str.strip().str.lower().isin(unknown_tokens)

    # Precompute per-machine empirical cause counts among known reasons to disambiguate shared alarms
    machine_cause_counts: Dict[str, Dict[str, int]] = {}
    for m in machines.unique():
        m_known = reasons[(machines == m) & (~is_unk_mask)]
        counts_dict: Dict[str, int] = {}
        for r_val, cnt in m_known.value_counts().items():
            counts_dict[str(r_val)] = int(cnt)
        machine_cause_counts[m] = counts_dict

    reinferred_reasons_count = 0
    new_reasons = reasons.copy()
    for idx in work.index[is_unk_mask]:
        a_code = str(alarms.loc[idx])
        m_val = str(machines.loc[idx])
        cand_causes = alarm_to_causes.get(a_code, [])
        if len(cand_causes) == 1:
            new_reasons.loc[idx] = cand_causes[0]
            reinferred_reasons_count += 1
        elif len(cand_causes) > 1:
            m_freqs = machine_cause_counts.get(m_val, {})
            best_cause = max(cand_causes, key=lambda c: (m_freqs.get(c, 0), c == "sensor_fault"))
            new_reasons.loc[idx] = best_cause
            reinferred_reasons_count += 1

    work["logged_reason"] = new_reasons

    rep_diag: DiagnosisResult = diagnose(work)
    top_changed = bool(orig_diag.top_candidate != rep_diag.top_candidate)
    margin_diff = float(rep_diag.margin_ratio - orig_diag.margin_ratio)
    total_repaired = merged_duplicates_count + imputed_durations_count + reinferred_reasons_count

    return RepairReport(
        repaired_df=work,
        original_top_candidate=orig_diag.top_candidate,
        repaired_top_candidate=rep_diag.top_candidate,
        top_candidate_changed=top_changed,
        original_margin_ratio=orig_diag.margin_ratio,
        repaired_margin_ratio=rep_diag.margin_ratio,
        margin_change=round(margin_diff, 4),
        abs_margin_change=round(abs(margin_diff), 4),
        original_raw_confidence=orig_diag.raw_confidence,
        repaired_raw_confidence=rep_diag.raw_confidence,
        merged_duplicates_count=merged_duplicates_count,
        imputed_durations_count=imputed_durations_count,
        reinferred_reasons_count=reinferred_reasons_count,
        total_repaired_rows=total_repaired,
    )
