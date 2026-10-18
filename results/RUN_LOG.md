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
3. **Timestamp Invariance**: Pure timestamp jitter and machine clock offsets do not alter total duration sums per machine when durations remain positive, showing where baseline summation is naturally robust and where quality scoring should focus.
