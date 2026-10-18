# FaultLens: Experiment Run Log

All experimental runs, command invocations, seeds, and execution logs are tracked here.

## Session 1: Baseline Diagnoser Sanity Check
- **Timestamp**: 2026-10-18
- **Evaluated Dataset**: 60 validation scenarios (`data/sample_scenarios/splits.json`)
- **Evaluation Command**: `PYTHONPATH=src python3 -m faultlens.diagnoser`
- **Method Evaluated**: Shared baseline diagnoser (naive downtime summation + margin-based softmax confidence)
- **Defects Tested**: All 4 defects at 30% severity vs. clean control

### Output Summary Table
| Condition | Scenarios | Top-1 Accuracy (%) | Mean Raw Conf (%) | Correct | Incorrect |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean (0% defect)** | 60 | 100.0% | 68.1% | 60 | 0 |
| **missing_events (30% severity)** | 60 | 73.3% | 70.4% | 44 | 16 |
| **incorrect_reason_codes (30% severity)** | 60 | 56.7% | 65.6% | 34 | 26 |
| **timestamp_errors (30% severity)** | 60 | 100.0% | 68.1% | 60 | 0 |
| **duplicate_records (30% severity)** | 60 | 50.0% | 70.6% | 30 | 30 |

### Key Observations
1. **Confidence Illusion**: On `duplicate_records`, accuracy drops from 100.0% to 50.0%, yet mean raw confidence rises from 68.1% to 70.6%. This illustrates how telemetry chatter inflates apparent confidence while misleading maintenance teams.
2. **Coding Noise Vulnerability**: On `incorrect_reason_codes` (30%), nearly half of all top-cause recommendations are wrong (56.7% accuracy), demonstrating that reason code confusion directly corrupts unweighted frequency rankings.
3. **Timestamp Invariance (Session 1 note)**: In Session 1, timestamp jitter shifted intervals together without affecting durations. This was resolved in Session 2 with independent start/end jitter and non-positive duration filtering.

---

## Session 2: 3-Method Validation Benchmark & Policy Tuning
- **Timestamp**: 2026-10-18
- **Evaluated Dataset**: 60 validation scenarios (`data/sample_scenarios/splits.json`) across 9 conditions (Clean + 4 defects at 20% and 30% severity)
- **Tuned Policy Parameters**: $\tau_q = 0.8321$, $\tau_s = 0.4000$, buffer $= 0.05$ (tuned on validation split via grid search under coverage constraint $\ge 60\%$)
- **Calibrator**: Isotonic Regression fitted strictly on validation clean scenarios

### 3-Method Validation Benchmark Table (N=60 per condition)
| Defect Condition | Freq Acc% | Freq CW% | Freq Cov% | ConfThresh Acc% | ConfThresh CW% | ConfThresh Cov% | FaultLens Acc% | FaultLens CW% | FaultLens Cov% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean (0%)** | 100.0% | 0.0% | 100.0% | 100.0% | 0.0% | 100.0% | 66.7% | **0.0%** | 66.7% |
| **missing_events (20%)** | 78.3% | 21.7% | 100.0% | 78.3% | 21.7% | 100.0% | 63.3% | **11.7%** | 75.0% |
| **missing_events (30%)** | 73.3% | 26.7% | 100.0% | 73.3% | 26.7% | 100.0% | 51.7% | **11.7%** | 63.3% |
| **incorrect_reason_codes (20%)** | 78.3% | 21.7% | 100.0% | 78.3% | 21.7% | 100.0% | 55.0% | **13.3%** | 68.3% |
| **incorrect_reason_codes (30%)** | 51.7% | 48.3% | 100.0% | 51.7% | 48.3% | 100.0% | 38.3% | **16.7%** | 55.0% |
| **timestamp_errors (20%)** | 83.3% | 16.7% | 100.0% | 83.3% | 16.7% | 100.0% | 61.7% | **5.0%** | 66.7% |
| **timestamp_errors (30%)** | 85.0% | 15.0% | 100.0% | 85.0% | 15.0% | 100.0% | 70.0% | **5.0%** | 75.0% |
| **duplicate_records (20%)** | 40.0% | 60.0% | 100.0% | 40.0% | 60.0% | 100.0% | 38.3% | **50.0%** | 88.3% |
| **duplicate_records (30%)** | 43.3% | 56.7% | 100.0% | 43.3% | 56.7% | 100.0% | 40.0% | **48.3%** | 88.3% |

### Core Findings
1. **Dramatic Error Reduction on Misattributions**: On `incorrect_reason_codes` at 30% severity, naive frequency and confidence threshold both suffer a devastating 48.3% confidently-wrong rate. FaultLens drops the confidently-wrong rate down to 16.7% (a ~3x reduction).
2. **Missing Event Robustness**: On `missing_events` at 30% severity, FaultLens reduces confidently-wrong rate from 26.7% to 11.7% by flagging unclosed stops.
3. **Timestamp Error Accuracy Drop**: With independent start/end jitter and non-positive duration filtering, accuracy under `timestamp_errors` drops to 83.3%-85.0% on baseline methods, and FaultLens suppresses confidently-wrong rate down to 5.0%.
4. **Honest Reporting on Duplication**: Under severe retry bursts (duplicate_records 20-30%), while FaultLens reduces confidently-wrong rate from 60.0% to 50.0% / 48.3%, duplication remains challenging because clustered micro-bursts mimic repetitive genuine short stops. This will be documented prominently in LIMITATIONS.md.

