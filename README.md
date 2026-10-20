# FaultLens: Evidence-Aware Downtime Diagnosis

> **"Not just what the top cause is, but how much evidence supports it."**

*A hackathon project in the Open Innovation track (software-only).*

---

## The Problem
Downtime diagnoses on modern manufacturing lines depend on machine-event and SCADA logs that are frequently incomplete, misattributed, unclosed, or duplicated. When diagnostic algorithms evaluate noisy telemetry without checking evidence quality, they can produce confidently incorrect recommendations that send maintenance crews down the wrong path.

## The Contribution
1. **A reproducible evaluation framework** assessing diagnostic reliability under 4 controlled, industrial data defects (`missing_events`, `incorrect_reason_codes`, `timestamp_errors`, `duplicate_records`) across 4 severities (0%, 10%, 20%, 30%), using 300 synthetic scenarios with explicit ground truth.
2. **An evidence-aware method**: combines data-quality scoring, perturbation-based stability testing, and disciplined abstention with an actionable "what to verify" message when telemetry evidence is degraded.

## Hard Rules & Scope
- **100% Synthetic Data**: All events and scenarios are procedurally generated with fixed seeds. No real plant data and no runtime network calls.
- **Strictly Grounded Claims**: No unverified superlatives ("revolutionary", "first of its kind", "no tool does this").
- **Clear Evaluation Boundary**: Fixed 40/20/40 train/val/test splits. Models are trained on train and thresholds are chosen on validation only. The final numbers come from 200 fresh scenarios (seed 4242), run once with a frozen model. An earlier test split was used in exploratory runs and is not reported as a held out result.
- **Four Evaluated Methods**:
  1. `frequency_baseline`: naive downtime summation.
  2. `faultlens`: calibrated diagnoser, data quality score, stability testing and a weighted abstention gate.
  3. `confidence_only_learned`: learned reliability score that uses confidence alone.
  4. `learned_faultlens`: learned reliability score that combines confidence, quality, stability and a repair check.

---

## Directory Structure
```
faultlens/
  README.md CONTEXT.md Makefile requirements.txt
  configs/
    generator.yaml defects.yaml experiment.yaml
  src/faultlens/
    generator.py defects.py diagnoser.py quality.py stability.py
    policy.py methods.py metrics.py experiment.py plots.py
  app/
    dashboard.py
  scripts/
    demo_check.py
  data/sample_scenarios/
  results/tables/ results/figures/ results/RUN_LOG.md
  tests/
  docs/
    ASSUMPTIONS.md METHODOLOGY.md DEFECT_TAXONOMY.md LIMITATIONS.md
    ORIGINALITY.md CODE_WALKTHROUGH.md DEMO_RUNBOOK.md SLIDES.md
    PITCH_SCRIPT.md QA_PREP.md
```

## Quick Start
```bash
# Install dependencies
make install

# Generate 300 scenarios (40/20/40 split)
make data

# Run unit tests
make test
```

## Run the Dashboard
```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

The dashboard has three data sources: curated demo cases, the train and validation scenarios, and a CSV upload. Two sample files are in the sample_data folder: clean_log.csv and defective_log.csv. Uploaded files need the columns event_id, machine_id, start_ts, end_ts, duration_min, logged_reason.

Uploaded logs have no alarm codes, so the alarm agreement score is not informative for uploads. All data is synthetic. See docs/LIMITATIONS.md for the full list of limits.
