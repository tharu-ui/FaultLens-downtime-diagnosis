"""FaultLens Benchmark Visualization Suite (Session 3).

Generates publication-grade figures (PNG at 200 DPI + SVG) in `results/figures/`
using a colorblind-safe Okabe-Ito palette with consistent method styling:
1. fig1_hero_cw_vs_severity.{png,svg} & fig1b_small_multiples_cw.{png,svg}:
   Confidently-wrong rate vs defect severity (overall mean + 2x2 per defect).
2. fig2_risk_coverage.{png,svg}:
   Risk-coverage curves for all 3 methods across defective test scenarios.
3. fig3_coverage_vs_severity.{png,svg}:
   Coverage vs defect severity showing FaultLens selective abstention as data degrades.
4. fig4_cw_reduction_heatmap.{png,svg}:
   Heatmap of defect type x severity vs percentage-point CW reduction (FaultLens vs Frequency).
5. fig5_stability_histogram.{png,svg}:
   Distribution of perturbation top_stability for correct vs incorrect diagnoses.
6. fig6_calibration_reliability.{png,svg}:
   Calibration reliability diagram comparing uncalibrated raw softmax vs Platt-calibrated confidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from faultlens.defects import ALLOWED_DEFECTS, ALLOWED_SEVERITIES

TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")

# Okabe-Ito colorblind-safe palette, consistent across all figures
METHOD_COLORS: Dict[str, str] = {
    "frequency_baseline": "#D55E00",     # Vermilion
    "confidence_threshold": "#0072B2",   # Blue
    "faultlens": "#009E73",              # Bluish green
}

METHOD_LABELS: Dict[str, str] = {
    "frequency_baseline": "Frequency Baseline (Always Answers)",
    "confidence_threshold": "Calibrated Confidence-Threshold",
    "faultlens": "FaultLens (Quality + Stability Gate)",
}

METHOD_MARKERS: Dict[str, str] = {
    "frequency_baseline": "s",
    "confidence_threshold": "^",
    "faultlens": "o",
}

DEFECT_TITLES: Dict[str, str] = {
    "missing_events": "Missing Events (Packet Drops & Null End)",
    "incorrect_reason_codes": "Incorrect Reason Codes (HMI Confusion)",
    "timestamp_errors": "Timestamp Errors (Clock Drift & Jitter)",
    "duplicate_records": "Duplicate Records (Retry & Bounce Bursts)",
}


def _save_both_formats(fig: plt.Figure, base_path: Path) -> List[Path]:
    """Save figure as both 200-DPI PNG and vector SVG."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = base_path.with_suffix(".png")
    svg_path = base_path.with_suffix(".svg")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return [png_path, svg_path]


def plot_hero_cw_vs_severity(metrics_df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> List[Path]:
    """Figure 1: Confidently-wrong rate vs defect severity (overall mean + 2x2 small multiples)."""
    saved: List[Path] = []

    # 1A: Hero aggregate line plot averaged across the 4 defects
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for method in ["frequency_baseline", "confidence_threshold", "faultlens"]:
        sub = metrics_df[metrics_df["method"] == method]
        grouped = sub.groupby("severity", as_index=False).agg(
            cw_mean=("confidently_wrong_rate", "mean"),
            cw_low=("cw_ci_lower", "mean"),
            cw_high=("cw_ci_upper", "mean"),
        ).sort_values("severity")

        x_pct = grouped["severity"].to_numpy(dtype=float) * 100.0
        y_pct = grouped["cw_mean"].to_numpy(dtype=float) * 100.0
        y_low = grouped["cw_low"].to_numpy(dtype=float) * 100.0
        y_high = grouped["cw_high"].to_numpy(dtype=float) * 100.0

        ax.plot(
            x_pct,
            y_pct,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=2.4,
            markersize=7,
        )
        ax.fill_between(x_pct, y_low, y_high, color=METHOD_COLORS[method], alpha=0.14)

    ax.set_title("Confidently-Wrong Rate vs. Injected Defect Severity (Exploratory Split)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Injected Defect Severity (%)", fontsize=11)
    ax.set_ylabel("Confidently-Wrong Rate (%) [Lower is Better]", fontsize=11)
    ax.set_xticks([0, 10, 20, 30])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True)
    saved.extend(_save_both_formats(fig, out_dir / "fig1_hero_cw_vs_severity"))

    # 1B: 2x2 Small Multiples per defect type
    fig2, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, defect in enumerate(ALLOWED_DEFECTS):
        ax_d = axes_flat[idx]
        d_sub = metrics_df[metrics_df["defect"] == defect]
        for method in ["frequency_baseline", "confidence_threshold", "faultlens"]:
            m_sub = d_sub[d_sub["method"] == method].sort_values("severity")
            x_pct = m_sub["severity"].to_numpy(dtype=float) * 100.0
            y_pct = m_sub["confidently_wrong_rate"].to_numpy(dtype=float) * 100.0
            y_low = m_sub["cw_ci_lower"].to_numpy(dtype=float) * 100.0
            y_high = m_sub["cw_ci_upper"].to_numpy(dtype=float) * 100.0

            ax_d.plot(
                x_pct,
                y_pct,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=2.0,
                markersize=6,
            )
            ax_d.fill_between(x_pct, y_low, y_high, color=METHOD_COLORS[method], alpha=0.12)

        ax_d.set_title(DEFECT_TITLES.get(defect, defect), fontsize=10.5, fontweight="bold")
        ax_d.set_xticks([0, 10, 20, 30])
        ax_d.grid(True, linestyle="--", alpha=0.4)
        if idx in (2, 3):
            ax_d.set_xlabel("Defect Severity (%)", fontsize=10)
        if idx in (0, 2):
            ax_d.set_ylabel("Confidently-Wrong Rate (%)", fontsize=10)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig2.suptitle(
        "Confidently-Wrong Rate by Defect Type and Severity (Exploratory Split, N=120 Scenarios)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig2.legend(handles, labels, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig2.tight_layout(rect=[0, 0.04, 1, 0.95])
    saved.extend(_save_both_formats(fig2, out_dir / "fig1b_small_multiples_cw"))
    return saved


def plot_risk_coverage_curves(raw_df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> List[Path]:
    """Figure 2: Risk-coverage curves for all 3 methods on defective exploratory evaluations."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    def_df = raw_df[raw_df["severity"] > 0.0].copy()

    for method in ["frequency_baseline", "confidence_threshold", "faultlens"]:
        m_df = def_df[def_df["method"] == method].copy()
        n_total = len(m_df)
        if n_total == 0:
            continue

        # Rank answered predictions first by composite ranking score, then confidence descending
        if method == "faultlens":
            score_col = 0.5 * m_df["quality_score"] + 0.5 * m_df["top_stability"]
        else:
            score_col = m_df["confidence"]

        m_df["rank_score"] = score_col
        ans_df = m_df[m_df["answered"]].sort_values("rank_score", ascending=False)

        if ans_df.empty:
            continue

        correct_flags = ans_df["correct"].to_numpy(dtype=bool)
        cum_errors = np.cumsum(~correct_flags)
        k_vals = np.arange(1, len(ans_df) + 1)
        coverages = (k_vals / n_total) * 100.0
        risks = (cum_errors / k_vals) * 100.0

        ax.plot(
            coverages,
            risks,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            linewidth=2.2,
        )
        # Mark operating point
        ax.scatter(
            [coverages[-1]],
            [risks[-1]],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            s=65,
            zorder=5,
        )

    ax.set_title("Selective Risk-Coverage Curves Across Defective Logs (Exploratory Split)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Coverage (% of Scenarios Answered)", fontsize=11)
    ax.set_ylabel("Selective Risk (% Error Among Answered) [Lower is Better]", fontsize=11)
    ax.set_xlim(0, 103)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True)
    return _save_both_formats(fig, out_dir / "fig2_risk_coverage")


def plot_coverage_vs_severity(metrics_df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> List[Path]:
    """Figure 3: Coverage vs severity across evaluated methods."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for method in ["frequency_baseline", "confidence_threshold", "faultlens"]:
        sub = metrics_df[metrics_df["method"] == method]
        grouped = sub.groupby("severity", as_index=False).agg(
            cov_mean=("coverage", "mean"),
            cov_low=("coverage_ci_lower", "mean"),
            cov_high=("coverage_ci_upper", "mean"),
        ).sort_values("severity")

        x_pct = grouped["severity"].to_numpy(dtype=float) * 100.0
        y_pct = grouped["cov_mean"].to_numpy(dtype=float) * 100.0
        y_low = grouped["cov_low"].to_numpy(dtype=float) * 100.0
        y_high = grouped["cov_high"].to_numpy(dtype=float) * 100.0

        ax.plot(
            x_pct,
            y_pct,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=2.4,
            markersize=7,
        )
        ax.fill_between(x_pct, y_low, y_high, color=METHOD_COLORS[method], alpha=0.14)

    ax.set_title("Diagnostic Coverage vs. Injected Defect Severity (Exploratory Split)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Injected Defect Severity (%)", fontsize=11)
    ax.set_ylabel("Diagnostic Coverage (% Answered)", fontsize=11)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", frameon=True)
    return _save_both_formats(fig, out_dir / "fig3_coverage_vs_severity")


def plot_cw_reduction_heatmap(matched_df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> List[Path]:
    """Figure 4: Heatmap of defect type x severity vs CW difference (FaultLens vs Frequency baseline)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    severities = [0.0, 0.10, 0.20, 0.30]
    matrix = np.zeros((len(ALLOWED_DEFECTS), len(severities)), dtype=float)

    for i, defect in enumerate(ALLOWED_DEFECTS):
        for j, sev in enumerate(severities):
            row = matched_df[(matched_df["defect"] == defect) & (np.isclose(matched_df["severity"], sev))]
            if not row.empty:
                matrix[i, j] = float(row["cw_reduction_vs_freq"].iloc[0]) * 100.0

    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("CW Difference vs Frequency Baseline (pp)", fontsize=10)

    ax.set_xticks(range(len(severities)))
    ax.set_xticklabels([f"{int(round(s * 100))}%" for s in severities], fontsize=10)
    ax.set_yticks(range(len(ALLOWED_DEFECTS)))
    ax.set_yticklabels(list(ALLOWED_DEFECTS), fontsize=10)

    max_val = float(np.max(matrix)) if matrix.size > 0 else 1.0
    for i in range(len(ALLOWED_DEFECTS)):
        for j in range(len(severities)):
            val = matrix[i, j]
            txt_color = "white" if val > 0.55 * max_val and max_val > 5.0 else "black"
            ax.text(j, i, f"{val:+.1f} pp", ha="center", va="center", color=txt_color, fontweight="bold", fontsize=10)

    ax.set_title("Confidently-Wrong Rate Difference (pp): FaultLens vs. Frequency (Exploratory Split)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Defect Severity", fontsize=11)
    ax.set_ylabel("Defect Type", fontsize=11)
    return _save_both_formats(fig, out_dir / "fig4_cw_reduction_heatmap")


def plot_stability_histogram(raw_df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> List[Path]:
    """Figure 5: Stability histogram showing top_stability for correct vs wrong unvetted diagnoses."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    fr_df = raw_df[raw_df["method"] == "frequency_baseline"]
    correct_stab = fr_df.loc[fr_df["correct"], "top_stability"].to_numpy(dtype=float)
    wrong_stab = fr_df.loc[~fr_df["correct"], "top_stability"].to_numpy(dtype=float)

    bins = np.linspace(0.0, 1.0, 16)
    ax.hist(
        correct_stab,
        bins=bins,
        alpha=0.65,
        color="#009E73",
        label=f"Correct Diagnoses (n={len(correct_stab)}, mean={np.mean(correct_stab):.2f})",
        density=True,
        edgecolor="black",
    )
    if len(wrong_stab) > 0:
        ax.hist(
            wrong_stab,
            bins=bins,
            alpha=0.65,
            color="#D55E00",
            label=f"Incorrect Diagnoses (n={len(wrong_stab)}, mean={np.mean(wrong_stab):.2f})",
            density=True,
            edgecolor="black",
        )

    ax.set_title("Distribution of Perturbation Top-1 Stability by Diagnosis Correctness (Exploratory Split)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Perturbation Top-1 Stability Score (top_stability)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True)
    return _save_both_formats(fig, out_dir / "fig5_stability_histogram")


def plot_calibration_reliability(cal_df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> List[Path]:
    """Figure 6: Calibration reliability diagram comparing raw vs calibrated confidence."""
    fig, ax = plt.subplots(figsize=(6.8, 5.2))

    ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", label="Perfect Calibration (Diagonal)")

    series_styles = [
        ("uncalibrated_raw", "Uncalibrated Raw Confidence", "#D55E00", "s"),
        ("calibrated_platt", "Calibrated Confidence (Platt)", "#0072B2", "o"),
        ("faultlens_answered", "FaultLens Answered Subset", "#009E73", "^"),
    ]

    for s_key, s_label, color, marker in series_styles:
        sub = cal_df[(cal_df["series"] == s_key) & (cal_df["bin_count"] > 0)].sort_values("bin_index")
        if sub.empty:
            continue
        ece_val = float(sub["overall_ece"].iloc[0])
        ax.plot(
            sub["bin_mean_confidence"].to_numpy(dtype=float),
            sub["bin_accuracy"].to_numpy(dtype=float),
            label=f"{s_label} (ECE={ece_val:.3f})",
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=6,
        )

    ax.set_title("Calibration Reliability Diagram Across Confidence Series (Exploratory Split)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Predicted Confidence", fontsize=11)
    ax.set_ylabel("Empirical Top-1 Accuracy", fontsize=11)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True)
    return _save_both_formats(fig, out_dir / "fig6_calibration_reliability")


def generate_all_figures(
    tables_dir: str | Path = TABLES_DIR,
    figures_dir: str | Path = FIGURES_DIR,
) -> List[Path]:
    """Load experiment CSVs from tables_dir and generate all 6 figure sets (PNG + SVG)."""
    t_dir = Path(tables_dir)
    f_dir = Path(figures_dir)

    raw_df = pd.read_csv(t_dir / "raw_results.csv")
    metrics_df = pd.read_csv(t_dir / "metrics_by_defect_severity.csv")
    matched_df = pd.read_csv(t_dir / "matched_coverage.csv")
    cal_df = pd.read_csv(t_dir / "calibration.csv")

    created_files: List[Path] = []
    created_files.extend(plot_hero_cw_vs_severity(metrics_df, out_dir=f_dir))
    created_files.extend(plot_risk_coverage_curves(raw_df, out_dir=f_dir))
    created_files.extend(plot_coverage_vs_severity(metrics_df, out_dir=f_dir))
    created_files.extend(plot_cw_reduction_heatmap(matched_df, out_dir=f_dir))
    created_files.extend(plot_stability_histogram(raw_df, out_dir=f_dir))
    created_files.extend(plot_calibration_reliability(cal_df, out_dir=f_dir))

    return created_files


if __name__ == "__main__":
    files = generate_all_figures()
    print(f"Generated {len(files)} figure files in '{FIGURES_DIR}':")
    for p in files:
        print(f"  - {p}")
