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
- **Clear Evaluation Boundary**: Fixed 40/20/40 train/val/test splits. All threshold tuning is performed on validation only; all benchmark numbers are evaluated on test.
- **Three Evaluated Methods ONLY**:
  1. `frequency_baseline`: Naive downtime summation.
  2. `confidence_threshold`: Margin-based softmax with post-hoc probability calibration.
  3. `faultlens`: Calibrated diagnoser + data quality score + stability testing + abstention policy.

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
