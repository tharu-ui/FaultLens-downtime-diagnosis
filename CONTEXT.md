# CONTEXT.md: FaultLens (read this first in every new session)

## What this project is
FaultLens: Evidence-Aware Downtime Diagnosis. Hackathon project, Open Innovation track, software only.
Tagline: "Not just what the top cause is, but how much evidence supports it."

Problem: Downtime diagnoses depend on machine-event logs that are often incomplete or inconsistent, so a confident top cause can be wrong without anyone noticing.

Contribution (do NOT claim more):
1. A reproducible way to evaluate diagnostic reliability under controlled data defects, using synthetic logs with known ground truth.
2. An evidence-aware method: data-quality scoring + stability testing + abstention with a "what to verify" message.

Research question: Does combining data-quality scoring with perturbation-based stability testing reduce confidently incorrect diagnoses, at an acceptable abstention rate, versus (a) a frequency baseline and (b) a calibrated confidence-threshold baseline?

## Hard rules
- All data is synthetic (disclose everywhere). No real data, no network calls at runtime, no API keys in the repo.
- Never write "first", "no tool does this", "current tools give no warning", "revolutionary".
- Do not call it an "AI project". The innovation is the evaluation method and abstention logic.
- Thresholds/weights tuned on VALIDATION only; final numbers reported on TEST only.
- Every number in docs/slides must come from real experiment output. Never invent results.
- Dashboard must do REAL live computation, not static screenshots.
- Report honestly, including where FaultLens does not help.

## Fixed design decisions
- Plain Python (NumPy/pandas), no SimPy. Python 3.11+.
- 4 machines: Loader, Press, Welder, Packer
- 6 causes: jam, sensor_fault, overheating, material_shortage, tool_wear, electrical_fault
- 4 defects ONLY: missing_events, incorrect_reason_codes, timestamp_errors, duplicate_records
- Severities: 0%, 10%, 20%, 30%
- 3 methods ONLY: frequency baseline, confidence-threshold baseline, FaultLens
- 300 scenarios, split 40/20/40 (train/val/test), fixed seeds
- Stability test: N=200 perturbations (configurable)
- Stack: NumPy, pandas, scikit-learn, SciPy, Matplotlib/Plotly, Streamlit, pytest, PyYAML, tqdm

## Repo structure
faultlens/
  README.md CONTEXT.md Makefile requirements.txt
  configs/ generator.yaml defects.yaml experiment.yaml
  src/faultlens/ generator.py defects.py diagnoser.py quality.py stability.py policy.py methods.py metrics.py experiment.py plots.py
  app/dashboard.py
  scripts/demo_check.py
  data/sample_scenarios/
  results/tables/ results/figures/ results/RUN_LOG.md
  tests/
  docs/ ASSUMPTIONS.md METHODOLOGY.md DEFECT_TAXONOMY.md LIMITATIONS.md ORIGINALITY.md CODE_WALKTHROUGH.md DEMO_RUNBOOK.md SLIDES.md PITCH_SCRIPT.md QA_PREP.md

## Session plan
1. Configs + generator + 4 defect injectors + shared diagnoser + calibration
2. Quality scoring, stability, policy, 3 methods, metrics
3. Full experiment run, tables, figures, tests
4. Dashboard, demo_check, DEMO_RUNBOOK
5. Docs, originality, code walkthrough, slides, pitch, Q&A

## Progress log (update at the end of every session)
- [x] Session 1 (Completed: Scaffolding, Generator, 4 Defect Injectors, Shared Diagnoser, Calibration, Tests, Sanity Run)
- [x] Session 2 (Completed: Timestamp Fix, Margin Distribution Check, Quality Scoring, Fast Vectorized Stability, Policy & Grid Tuning, 3 Methods, Metrics, 36 Passing Tests)
- [ ] Session 3
- [ ] Session 4
- [ ] Session 5

## Key decisions and assumptions so far
1. Synthetic generation: 30 days, 24/7 (three 8-hour shifts), 4 machines (Loader, Press, Welder, Packer), 6 causes, lognormal downtime distributions.
2. Ground truth: argmax total true downtime over (machine, true_cause). Margin distribution across 300 scenarios verified: min 0.0014, median 0.1663, max 0.6881, with 71/300 (23.7%) close calls (margin <= 0.07), satisfying the >=20% requirement.
3. 300 scenarios partitioned 40/20/40 into train (120), val (60), test (120) with fixed master seed 42. Partition manifest stored at `data/sample_scenarios/splits.json`.
4. Exactly 4 defect injectors: `missing_events`, `incorrect_reason_codes`, `timestamp_errors`, and `duplicate_records`.
5. Timestamp error resolution: `timestamp_errors` now applies per-machine clock offset plus independent Gaussian jitter to start_ts and end_ts ($\sigma$ up to 14 min at 30% severity), creating inverted timestamps and distorted intervals. The shared diagnoser strictly excludes non-positive durations ($\le 0$), causing accuracy to drop from 100% to 80.0%-85.0% under 30% severity.
6. Data quality scoring: 6 sub-scores in [0, 1] (completeness, reason_validity, duplication, timestamp_consistency, micro_stop_plausibility, alarm_reason_agreement), weighted overall score, worst component + worst machine identification, and exact countable tallies.
7. Perturbation stability testing: Vectorized NumPy `bincount` implementation executing N=200 bootstrap perturbations in under 0.01 seconds (< 10 ms), well below the 2.0s ceiling.
8. Tuned policy thresholds (grid search on validation split only, constrained to coverage $\ge 60\%$):
   - $\tau_q = 0.8321$
   - $\tau_s = 0.4000$
   - buffer $= 0.05$
   - Saved to `results/tuned_params.json` (achieved validation coverage: 62.6%, confidently-wrong rate: 10.9%).
9. 3 Comparative methods implemented in `src/faultlens/methods.py` (`FrequencyBaselineMethod`, `ConfidenceThresholdMethod`, `FaultLensMethod`).
10. Cross-platform execution: `scripts/run_all.py` created for platforms lacking `make` (Windows PowerShell), with subcommands `data`, `test`, `sanity`, `tune`, and `all`. `conftest.py` automatically injects `src/` to `sys.path`.

## Open issues
(none)
