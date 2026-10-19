"""FaultLens Interactive Streamlit Dashboard.

An evidence-aware downtime diagnosis dashboard demonstrating how data quality scoring
and perturbation stability testing protect industrial maintenance teams from costly false alarms.

Features:
- Live exploration of benchmark scenarios (strictly from TRAIN and VAL splits).
- Interactive defect injection with reproducible seeds and 0.0 - 0.3 severity sliders.
- Comparison of 4 diagnostic methods: Frequency Baseline, FaultLens, Confidence-Only Learned, Learned FaultLens.
- Real-time evidence breakdown: Quality sub-scores, counts, bootstrap stability, and verification directives.
- Custom industrial log CSV upload with automatic validation.
- Final evaluation summary view loaded read-only from results.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 1. Ensure REPO_ROOT and src/ are on sys.path regardless of execution directory
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

import app.helpers as h
from faultlens.defects import inject_defect
from faultlens.generator import Scenario
from faultlens.quality import QualityReport
from faultlens.stability import StabilityReport


# Page configuration
st.set_page_config(
    page_title="FaultLens | Evidence-Aware Downtime Diagnosis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    .flag-green {
        color: #15803d;
        background-color: #dcfce7;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .flag-amber {
        color: #b45309;
        background-color: #fef3c7;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .flag-red {
        color: #b91c1c;
        background-color: #fee2e2;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .vmsg-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 4px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_flag_badge(flag: str) -> str:
    """Return colored HTML badge for diagnostic status flags."""
    f = str(flag).upper()
    if f == "GREEN":
        return '<span class="flag-green">● GREEN (Actionable)</span>'
    elif f == "AMBER":
        return '<span class="flag-amber">▲ AMBER (Review Needed)</span>'
    else:
        return '<span class="flag-red">■ RED (Abstained)</span>'


def format_candidate_str(cand: Optional[Tuple[str, str] | List[str]]) -> str:
    """Format (machine, cause) candidate tuple cleanly."""
    if not cand:
        return "None (Abstained)"
    return f"{cand[0]} - {cand[1]}"


def main() -> None:
    # Header Banner
    st.markdown('<div class="main-header">FaultLens: Evidence-Aware Downtime Diagnosis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Evaluating diagnostic reliability under controlled data defects using industrial machine-event logs.</div>',
        unsafe_allow_html=True,
    )

    # Sidebar: Data Source Selection & Controls
    st.sidebar.title("Configuration")
    
    data_source_mode = st.sidebar.radio(
        "Data Source",
        ["Curated Demo Cases", "Explore Train/Val Split", "Upload Custom Log CSV"],
        index=0,
    )

    current_log_df: Optional[pd.DataFrame] = None
    ground_truth: Optional[Tuple[str, str]] = None
    scenario_info_text: str = ""
    demo_context: Optional[Dict[str, Any]] = None

    # Load splits
    allowed_splits = h.get_allowed_scenario_ids()

    if data_source_mode == "Curated Demo Cases":
        demo_data = h.load_demo_selection()
        demo_case_key = st.sidebar.selectbox(
            "Select Curated Case",
            [
                "case_a: Confidently Wrong Baseline (Abstained by FaultLens)",
                "case_b: Clean Agreement (High Quality & Stability)",
                "case_c: Near-Tie Ambiguity (Overlapping CIs)",
            ],
            index=0,
        )
        case_id = demo_case_key.split(":")[0].strip()
        demo_case = demo_data.get(case_id, {})
        demo_context = demo_case

        target_sc_id = demo_case.get("scenario_id", "scenario_226")
        try:
            sc = h.load_scenario_by_id(target_sc_id)
            current_log_df = sc.log.copy()
            ground_truth = sc.truth
            scenario_info_text = f"**Scenario ID**: `{target_sc_id}` (Split: `{demo_case.get('split', 'val')}`). {demo_case.get('why_chosen', '')}"
        except Exception as exc:
            st.sidebar.error(f"Error loading demo scenario: {exc}")

    elif data_source_mode == "Explore Train/Val Split":
        split_choice = st.sidebar.radio("Split", ["Validation (Recommended)", "Train"], index=0)
        split_key = "val" if "Val" in split_choice else "train"
        available_ids = allowed_splits.get(split_key, [])
        if not available_ids:
            st.sidebar.warning("No scenarios available in chosen split.")
            selected_id = ""
        else:
            selected_id = st.sidebar.selectbox("Scenario ID", available_ids, index=0)

        if selected_id:
            try:
                sc = h.load_scenario_by_id(selected_id)
                current_log_df = sc.log.copy()
                ground_truth = sc.truth
                scenario_info_text = f"**Scenario ID**: `{selected_id}` ({split_choice}). Difficulty: `{sc.metadata.get('target_difficulty', 'standard')}`."
            except Exception as exc:
                st.sidebar.error(f"Error loading scenario: {exc}")

    else:
        # Custom CSV Upload
        uploaded_file = st.sidebar.file_uploader("Upload Downtime Log CSV", type=["csv"])
        if uploaded_file is not None:
            parsed_df, err = h.validate_and_parse_uploaded_csv(uploaded_file)
            if err:
                st.sidebar.error(err)
            else:
                current_log_df = parsed_df
                ground_truth = None  # Uploaded CSV has no ground truth
                scenario_info_text = f"**Custom Upload**: `{uploaded_file.name}` ({len(parsed_df)} events). Note: Ground truth is hidden for external uploads."
        else:
            st.sidebar.info("Upload a CSV containing `event_id, machine_id, start_ts, end_ts, duration_min, logged_reason`.")

    if current_log_df is None:
        st.info("👈 Please select or upload a dataset using the sidebar to begin analysis.")
        return

    # Defect Injection Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Defect Injection (Interactive)")
    
    # If case_a was selected, preset defect and severity for instant illustration
    default_defect_idx = 0
    default_sev_idx = 0
    if demo_context and "defect" in demo_context:
        d_name = demo_context["defect"]
        if d_name in h.ALLOWED_DEFECTS:
            default_defect_idx = h.ALLOWED_DEFECTS.index(d_name)
        s_val = float(demo_context.get("severity", 0.0))
        if s_val in h.ALLOWED_SEVERITIES:
            default_sev_idx = h.ALLOWED_SEVERITIES.index(s_val)

    selected_defect = st.sidebar.selectbox(
        "Defect Type",
        h.ALLOWED_DEFECTS,
        index=default_defect_idx,
        help="Simulate industrial data imperfections: missing records, reason code noise, clock drift, or retry bursts.",
    )

    selected_severity = st.sidebar.select_slider(
        "Severity",
        options=h.ALLOWED_SEVERITIES,
        value=h.ALLOWED_SEVERITIES[default_sev_idx],
        format_func=lambda x: f"{int(x*100)}%" if x > 0 else "0% (Clean)",
        help="Proportion of events corrupted by the chosen defect mechanism.",
    )

    # Fixed seed for inject_defect so a run is completely reproducible
    perturbation_seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, max_value=99999, step=1)
    
    # Apply defect
    working_df = current_log_df.copy()
    if selected_severity > 0.0 and data_source_mode != "Upload Custom Log CSV":
        working_df, defect_meta = inject_defect(
            working_df,
            defect_type=selected_defect,
            severity=selected_severity,
            seed=perturbation_seed,
        )

    # Top info banner
    st.markdown(scenario_info_text)
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Total Events", f"{len(working_df):,}")
    with col_stat2:
        tot_downtime = working_df["duration_min"].sum() / 60.0 if "duration_min" in working_df.columns else 0.0
        st.metric("Total Downtime", f"{tot_downtime:.1f} hrs")
    with col_stat3:
        n_machines = working_df["machine_id"].nunique() if "machine_id" in working_df.columns else 0
        st.metric("Monitored Machines", n_machines)
    with col_stat4:
        if ground_truth:
            st.metric("True Dominant Cause", f"{ground_truth[0]} : {ground_truth[1]}")
        else:
            st.metric("Ground Truth", "Unavailable (Uploaded Log)")

    # Run Live Diagnosis (reusing extract_reliability_features once)
    with st.spinner("Executing live diagnosis and evidence verification..."):
        diag_results = h.run_live_diagnosis(
            working_df,
            num_perturbations=h.NUM_PERTURBATIONS,
            seed=perturbation_seed,
        )

    # Method Comparison Section
    st.markdown("### Diagnostic Method Comparison")
    st.markdown(
        "Evaluation across the **4 comparative methods**. Note how naive methods remain confidently wrong under data corruption, while FaultLens abstains with an actionable verification recommendation."
    )

    cols = st.columns(4)
    method_keys = h.DISPLAYED_METHODS  # ["frequency_baseline", "faultlens", "confidence_only_learned", "learned_faultlens"]
    method_labels = {
        "frequency_baseline": "1. Frequency Baseline",
        "faultlens": "2. FaultLens (Weighted Gate)",
        "confidence_only_learned": "3. Confidence-Only Model",
        "learned_faultlens": "4. Learned FaultLens (Primary)",
    }

    outputs = diag_results["method_outputs"]

    for idx, m_key in enumerate(method_keys):
        mo = outputs.get(m_key)
        with cols[idx]:
            st.markdown(f"**{method_labels.get(m_key, m_key)}**")
            if mo is None:
                st.caption("Not available")
                continue

            # Status flag badge
            st.markdown(render_flag_badge(mo.flag), unsafe_allow_html=True)

            # Candidate & Confidence
            cand_str = format_candidate_str(mo.top_candidate)
            st.markdown(f"**Recommendation**: `{cand_str}`")
            st.markdown(f"**Confidence**: `{mo.confidence * 100.0:.1f}%`")
            st.markdown(f"**Answered**: `{'Yes' if mo.answered else 'No (Abstained)'}`")

            # Correctness against ground truth (strictly hidden if ground truth is unavailable)
            if ground_truth is not None:
                if not mo.answered:
                    st.markdown("🏷️ **Outcome**: `ABSTAINED` (Safe)")
                elif mo.top_candidate == ground_truth:
                    st.markdown("🏷️ **Outcome**: :green[**CORRECT**]")
                else:
                    st.markdown("🏷️ **Outcome**: :red[**CONFIDENTLY WRONG**]")

    # Actionable Verification Message (Prominently displayed when FaultLens abstains or flags review)
    v_msg = diag_results.get("verification_message")
    learned_fl_mo = outputs.get("learned_faultlens") or outputs.get("faultlens")
    if v_msg and learned_fl_mo and (not learned_fl_mo.answered or learned_fl_mo.flag != "GREEN"):
        st.markdown(
            f"""
            <div class="vmsg-box">
                <strong style="color: #92400e; font-size: 1.05rem;">⚠️ FaultLens Abstention & Actionable Verification Message:</strong><br>
                <span style="color: #78350f; font-size: 0.95rem;">{v_msg}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Detailed Evidence Panels: Quality & Stability
    st.markdown("---")
    tab_quality, tab_stability, tab_evaluation = st.tabs(
        ["📊 Data Quality Evidence", "🎯 Perturbation Stability Evidence", "📈 Final Evaluation Results"]
    )

    with tab_quality:
        st.subheader("Data Quality Audit")
        col_q1, col_q2 = st.columns([1, 1])
        
        q_rep: QualityReport = diag_results["q_rep"]

        with col_q1:
            st.metric("Overall Quality Score", f"{q_rep.overall_score * 100.0:.1f}%")
            st.markdown(f"**Worst Quality Dimension**: `{q_rep.worst_component}`")
            st.markdown(f"**Worst Machine Reliability**: `{q_rep.worst_machine}`")

            # Dimension sub-scores table
            sub_scores_data = []
            for comp, score in q_rep.sub_scores.items():
                sub_scores_data.append({"Quality Dimension": comp.replace("_", " ").title(), "Score (%)": round(score * 100.0, 1)})
            st.dataframe(pd.DataFrame(sub_scores_data), use_container_width=True, hide_index=True)

        with col_q2:
            st.markdown("**Exact Countable Defects Detected:**")
            counts_data = [
                {"Defect Type": "Missing End Timestamps", "Count": getattr(q_rep.counts, "missing_end_count", 0)},
                {"Defect Type": "Non Positive or Inverted Durations", "Count": getattr(q_rep.counts, "negative_duration_count", 0)},
                {"Defect Type": "Other / Unknown Reason Codes", "Count": getattr(q_rep.counts, "other_unknown_count", 0)},
                {"Defect Type": "Duplicate Records", "Count": getattr(q_rep.counts, "duplicate_burst_count", 0)},
                {"Defect Type": "Overlapping Events (same machine)", "Count": getattr(q_rep.counts, "overlapping_event_count", 0)},
                {"Defect Type": "Alarm Reason Mismatches", "Count": getattr(q_rep.counts, "alarm_mismatch_count", 0)},
                {"Defect Type": "Micro Stops Observed", "Count": getattr(q_rep.counts, "micro_stop_count", 0)},
            ]
            st.dataframe(pd.DataFrame(counts_data), use_container_width=True, hide_index=True)

    with tab_stability:
        st.subheader("Perturbation Stability & Margin Analysis")
        col_s1, col_s2 = st.columns([1, 1])
        
        s_rep: StabilityReport = diag_results["s_rep"]

        with col_s1:
            st.metric("Bootstrap Top Stability", f"{s_rep.top_stability * 100.0:.1f}%")
            st.markdown(f"**Top-3 Flip Rate**: `{s_rep.top3_flip_rate * 100.0:.1f}%`")
            st.markdown(f"**95% Margin CI**: `[{s_rep.margin_ci_lower_95:.3f}, {s_rep.margin_ci_upper_95:.3f}]`")
            if s_rep.margin_ci_crosses_zero:
                st.warning("⚠️ **Margin CI crosses zero**: Diagnostic ambiguity detected. Runner-up may be true cause.")
            else:
                st.success("✅ **Margin CI strictly positive**: Top candidate reliably dominates competitors.")

        with col_s2:
            st.markdown(f"**Candidate Bootstrap Intervals ({s_rep.num_perturbations} Perturbations):**")
            ci_rows = []
            for ci in s_rep.candidate_cis:
                ci_rows.append({
                    "Candidate": f"{ci.machine_id} / {ci.logged_reason}",
                    "Selection Rate (%)": round(ci.top_selection_rate * 100.0, 1),
                    "Mean Share of Downtime": round(ci.mean_share, 3),
                    "95% CI Range": f"[{ci.ci_lower_95:.3f}, {ci.ci_upper_95:.3f}]",
                })            
            st.dataframe(pd.DataFrame(ci_rows), use_container_width=True, hide_index=True)

    with tab_evaluation:
        st.subheader("Held-Out Test Set Evaluation Summary")
        st.markdown(
            "Benchmark metrics loaded strictly **read-only** from `results/tables/final_fresh_test_results.csv`."
        )

        pooled_df, by_defect_df, err = h.load_final_results_summary()
        if err or pooled_df is None or pooled_df.empty:
            st.info(
                "Final fresh test results will appear here once evaluation is generated. "
                "The dashboard maintains strict read-only guarantees over experimental locks."
            )
        else:
            st.markdown("#### Overall Pooled Summary (Coverage vs. Confidently-Wrong Rate)")
            st.dataframe(pooled_df, use_container_width=True, hide_index=True)

            if by_defect_df is not None and not by_defect_df.empty:
                st.markdown("#### Slice Breakdown by Defect and Severity")
                st.dataframe(by_defect_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
