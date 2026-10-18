# CONTEXT.md: FaultLens (read this first in every new session)

## What this project is
FaultLens: Evidence-Aware Downtime Diagnosis. Hackathon project, Open Innovation track, software only.
Tagline: "Not just what the top cause is, but how much evidence supports it."

Problem: Downtime diagnoses depend on machine-event logs that are often incomplete or inconsistent, so a confident top cause can be wrong without anyone noticing.

Contribution (do NOT claim more):
1. A reproducible way to evaluate diagnostic reliability under controlled data defects, using synthetic logs with known ground truth.
2. An evidence-aware method: data-quality scoring + stability testing + abstention with a "what to verify" message.

Research question: Does combining data-quality scoring with perturbation-based stability testing reduce confidently incorrect diagnoses, at an acceptable abstention rate, versus (a) a frequency baseline and (b) a calibrated confidence-threshold baseline?

Environment: Python 3.13, pandas<3, Windows, venv.

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
- [ ] Session 2
- [ ] Session 3
- [ ] Session 4
- [ ] Session 5

## Key decisions and assumptions so far
1. Synthetic generation: 30 days, 24/7 (three 8-hour shifts), 4 machines (Loader, Press, Welder, Packer), 6 causes, lognormal downtime distributions.
2. Ground truth: argmax total true downtime over (machine, true_cause). Margin metadata recorded and categorized as easy (>=18%), close (<=7%), or moderate.
3. 300 scenarios generated and partitioned 40/20/40 into train (120), val (60), test (120) with fixed master seed 42. Partition manifest stored at `data/sample_scenarios/splits.json`.
4. Exactly 4 defect injectors: `missing_events` (50% dropped, 50% null end_ts), `incorrect_reason_codes` (confusion matrix + Other/Unknown), `timestamp_errors` (per-machine clock drift + jitter), `duplicate_records` (70% single, 30% bursts of 3-10). Injectors strictly preserve `true_cause` and `scenario.truth`.
5. Raw confidence formula: Margin-based softmax over top-1 margin relative to runner-up mean scale: $z = (S_1 - S_2) / \max(\bar{S}_{\text{rest}}, \epsilon)$, $\text{raw\_conf} = 1 / (1 + \exp(-z / T))$, bounded in [0.5, 1.0].
6. Calibration: Isotonic regression and Platt scaling fit strictly on validation set; 10-bin Expected Calibration Error (ECE).
7. Baseline sanity check on 60 validation scenarios: Clean accuracy is 100.0% (mean raw conf 68.1%); missing_events (30%) drops to 73.3% accuracy; incorrect_reason_codes (30%) drops to 56.7% accuracy; timestamp_errors (30%) maintains 100.0%; duplicate_records (30%) crashes to 50.0% accuracy despite raw confidence paradoxically rising to 70.6%.

## Open issues
(none)
