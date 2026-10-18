# FaultLens Benchmark Summary (Test Split, N=120 Scenarios)

*Disclosure*: All scenarios and event logs are 100% synthetic. Thresholds (gate_mode=weighted, tau_q=0.8321, delta=0.1500, tau_s=0.4000, conf_threshold=0.6970) were tuned on train and validation scenarios only. The test split (120 scenarios) was also used in an earlier exploratory run before the Session 2.5 fixes, so these results are reported as exploratory and not as a clean one-shot held-out result.

## Headline Comparison at Matched Coverage (`matched_coverage.csv`)

| Defect Type | Severity | FaultLens Cov (%) | Freq CW (%) | ConfThresh Own CW (%) | ConfThresh Matched CW (%) | Random Abstain CW (%) | FaultLens CW (%) | FaultLens Sel. Acc (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean (0% defect) | 0% | 66.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| missing_events | 10% | 69.2 | 20.8 | 2.5 | 5.8 | 14.4 | 7.5 | 89.2 |
| missing_events | 20% | 63.3 | 32.5 | 3.3 | 8.3 | 20.6 | 8.3 | 86.8 |
| missing_events | 30% | 57.5 | 30.8 | 8.3 | 10.8 | 17.7 | 10.8 | 81.2 |
| incorrect_reason_codes | 10% | 63.3 | 24.2 | 3.3 | 8.3 | 15.3 | 10.8 | 82.9 |
| incorrect_reason_codes | 20% | 65.0 | 26.7 | 10.0 | 11.7 | 17.3 | 15.0 | 76.9 |
| incorrect_reason_codes | 30% | 60.8 | 41.7 | 14.2 | 19.2 | 25.4 | 19.2 | 68.5 |
| timestamp_errors | 10% | 63.3 | 5.0 | 0.0 | 0.0 | 3.2 | 0.8 | 98.7 |
| timestamp_errors | 20% | 71.7 | 6.7 | 0.0 | 1.7 | 4.8 | 3.3 | 95.3 |
| timestamp_errors | 30% | 62.5 | 19.2 | 1.7 | 5.8 | 12.0 | 7.5 | 88.0 |
| duplicate_records | 10% | 73.3 | 43.3 | 25.8 | 30.0 | 31.8 | 30.0 | 59.1 |
| duplicate_records | 20% | 90.0 | 47.5 | 33.3 | 44.2 | 42.8 | 42.5 | 52.8 |
| duplicate_records | 30% | 91.7 | 55.8 | 34.2 | 50.0 | 51.2 | 48.3 | 47.3 |

## Key Findings (Traceable to CSV Outputs)

1. **Aggregate Confidently-Wrong Reduction**: Across all defective test conditions (10%, 20%, 30% severities), the naive Frequency baseline produces a mean confidently-wrong (CW) rate of **29.51%** (at 100% coverage), the tuned Confidence-Threshold baseline achieves **11.39%** CW, the coverage-matched Confidence-Threshold baseline achieves **16.32%** CW, and FaultLens achieves **17.01%** CW at **69.30%** mean coverage (paired bootstrap $p = 0.0001$ vs Frequency, $p = 0.0001$ vs Own Confidence-Threshold). The own-tuned Confidence-Threshold baseline has a lower mean CW (11.39%) than FaultLens (17.01%), each at its own tuned operating point, so the significant difference favors the baseline.
2. **Clean Scenario Behavior & Unnecessary Abstentions**: On clean scenarios (0% defects), Frequency baseline CW is **0.00%** and FaultLens CW is **0.00%**, while FaultLens answers **66.67%** of scenarios (abstaining on **33.33%** of clean scenarios when competing bottlenecks have narrow ground-truth margins that flip under bootstrap perturbation).
3. **Where FaultLens Does Not Beat the Matched Confidence-Threshold Baseline**: FaultLens has a higher or equal CW rate compared to the coverage-matched Confidence-Threshold baseline in: `missing_events` at 10% (FaultLens CW 7.5% vs Matched ConfThresh CW 5.8%); `missing_events` at 20% (FaultLens CW 8.3% vs Matched ConfThresh CW 8.3%); `missing_events` at 30% (FaultLens CW 10.8% vs Matched ConfThresh CW 10.8%); `incorrect_reason_codes` at 10% (FaultLens CW 10.8% vs Matched ConfThresh CW 8.3%); `incorrect_reason_codes` at 20% (FaultLens CW 15.0% vs Matched ConfThresh CW 11.7%); `incorrect_reason_codes` at 30% (FaultLens CW 19.2% vs Matched ConfThresh CW 19.2%); `timestamp_errors` at 10% (FaultLens CW 0.8% vs Matched ConfThresh CW 0.0%); `timestamp_errors` at 20% (FaultLens CW 3.3% vs Matched ConfThresh CW 1.7%); `timestamp_errors` at 30% (FaultLens CW 7.5% vs Matched ConfThresh CW 5.8%); `duplicate_records` at 10% (FaultLens CW 30.0% vs Matched ConfThresh CW 30.0%). Where single-event margin confidence already separates easy from close scenarios, thresholding on calibrated margin confidence alone is a strong competitor.
4. **Calibration (ECE)**: Post-hoc calibration reduces Expected Calibration Error (ECE) from **0.1121** (raw softmax margin confidence) to **0.0518** (`calibration.csv`).
