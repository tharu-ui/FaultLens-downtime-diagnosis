# CONTEXT.md: FaultLens (read this first in every new session)

## What this project is
FaultLens: Evidence-Aware Downtime Diagnosis. Hackathon project, Open Innovation track, software only.
Tagline: "Not just what the top cause is, but how much evidence supports it."

Problem: Downtime diagnoses depend on machine-event logs that are often incomplete or inconsistent, so a confident top cause can be wrong without anyone noticing.

Contribution (do NOT claim more):
1. A reproducible way to evaluate diagnostic reliability under controlled data defects, using synthetic logs with known ground truth.
2. An evidence-aware method: data-quality scoring + stability testing + deterministic log-repair sensitivity + abstention with a "what to verify" message and defect attribution.

Research question: Does combining data-quality scoring, perturbation-based stability testing, and log-repair sensitivity reduce confidently incorrect diagnoses and identify the dominant data defect, at an acceptable abstention rate, versus (a) a frequency baseline and (b) a calibrated confidence-threshold baseline?

## Hard rules
- Environment: Python 3.13 or newer, pandas below 3, Windows PowerShell.
- All data is 100% synthetic (disclose everywhere, including extra train and fresh held-out sets). No real data, no network calls at runtime, no API keys in the repo.
- Never write "first", "no tool does this", "current tools give no warning", "revolutionary".
- Do not call it an "AI project". The innovation is the evaluation method, diagnostic repair sensitivity, defect attribution, and abstention logic.
- The original `splits.json["test"]` split was used in earlier exploratory runs and must NOT be used for any decision, tuning, or final claim.
- Final saved models use `TRAIN` for fitting and `VALIDATION` for calibration and choosing ONE threshold. Cross-validation (`GroupKFold` grouped by scenario) is used only to report out-of-fold results.
- Final numbers will come from the fresh held-out set (`data/sample_scenarios/fresh_test/`, 200 scenarios, `seed=4242`), run ONCE with `python scripts/run_all.py final`, which fails if saved models/thresholds are missing and creates `results/final_run.lock` so it cannot be rerun.
- Every number in docs/slides must come from real experiment output. Never invent results.
- Results so far are exploratory, and the own-tuned confidence threshold baseline has a lower confidently-wrong rate than rule-gated FaultLens on the exploratory split; report this honestly.

## Fixed design decisions
- Plain Python (NumPy/pandas), no SimPy. Python 3.13+, pandas < 3.
- 4 machines: Loader, Press, Welder, Packer
- 6 causes: jam, sensor_fault, overheating, material_shortage, tool_wear, electrical_fault
- 4 defects ONLY: missing_events, incorrect_reason_codes, timestamp_errors, duplicate_records
- Severities: 0%, 10%, 20%, 30%
- 5 methods (3 original unchanged + 2 Session 3.5 learned methods):
  1. `frequency_baseline` (`FrequencyBaselineMethod`)
  2. `confidence_threshold` (`ConfidenceThresholdMethod`)
  3. `faultlens` (`FaultLensMethod`)
  4. `learned_faultlens` (`LearnedFaultLensMethod`)
  5. `confidence_only_learned` (`ConfidenceOnlyLearnedMethod`)
- Dynamic method registry: `get_available_methods()` in `src/faultlens/methods.py`.
- 300 initial scenarios (`seed=42`, split 40/20/40 into train/val/exploratory-test), plus optional extra synthetic train scenarios (`seed=1337`), plus 200 fresh held-out scenarios (`seed=4242`, generated into `data/sample_scenarios/fresh_test/` with `manifest.json`).

## Repo structure
faultlens/
  README.md CONTEXT.md Makefile requirements.txt
  configs/ generator.yaml defects.yaml experiment.yaml
  src/faultlens/ generator.py defects.py diagnoser.py quality.py stability.py repair.py reliability.py policy.py methods.py metrics.py experiment.py plots.py
  app/dashboard.py
  scripts/run_all.py scripts/demo_check.py
  data/sample_scenarios/
  results/tables/ results/figures/ results/RUN_LOG.md
  tests/
  docs/ ASSUMPTIONS.md METHODOLOGY.md DEFECT_TAXONOMY.md LIMITATIONS.md ORIGINALITY.md CODE_WALKTHROUGH.md DEMO_RUNBOOK.md SLIDES.md PITCH_SCRIPT.md QA_PREP.md

## Session plan
1. Configs + generator + 4 defect injectors + shared diagnoser + calibration
2. Quality scoring, stability, policy, 3 methods, metrics
3. Exploratory full run, tables, figures, tests
3.5. Log repair (`repair.py`), learned reliability & confidence-only ablation (`reliability.py`), defect attribution metric (`metrics.py`), new method classes & `get_available_methods()` (`methods.py`), extra train & fresh held-out generators (`generator.py`), validation sanity table (`session3_5_validation_table.csv`), one-shot locked `final` command, neutral plot titles (`plots.py`), and Session 3.5 unit tests
4. Dashboard, demo_check, DEMO_RUNBOOK
5. Docs, originality, code walkthrough, slides, pitch, Q&A

## Progress log (update at the end of every session)
- [x] Session 1 (Completed: Scaffolding, Generator, 4 Defect Injectors, Shared Diagnoser, Calibration, Tests, Sanity Run)
- [x] Session 2 (Completed: Timestamp Fix, Margin Distribution Check, Quality Scoring, Fast Vectorized Stability, Policy & Grid Tuning, 3 Methods, Metrics, 36 Passing Tests)
- [x] Session 2.5 & Session 3 (Completed: Orthogonal Quality Sub-Scores, Clean Baseline from Train Split, Component-Aware Gate, Platt Monotonic Calibration, Exploratory Split Run, Figures, Integration Tests)
- [x] Session 3.5 (Completed: `repair.py`, `reliability.py` with `GroupKFold` & TRAIN-fit / VALIDATION-calibrated `LearnedReliabilityModel`, `LearnedFaultLensMethod` & `ConfidenceOnlyLearnedMethod`, `get_available_methods()`, attribution accuracy metric, `extra_train` & `fresh_test` generators, `session3_5_validation_table.csv`, locked `final` runner, neutral figure titles in `plots.py`, and Session 3.5 tests)
- [ ] Session 4
- [ ] Session 5

## Key decisions and assumptions so far
1. Exploratory honesty: On the exploratory split, rule-gated `FaultLensMethod` (`tau_q=0.8321, tau_s=0.40`) did not beat the coverage-matched calibrated confidence-threshold baseline, and the own-tuned confidence threshold baseline achieved a lower confidently-wrong rate than FaultLens. Furthermore, duplicate bursts bias downtime consistently across bootstrap resamples so perturbation stability alone does not catch them.
2. Deterministic log repair (`src/faultlens/repair.py`): Merges near-duplicate bursts on the same machine and reason within 60s, imputes missing/non-positive durations with the per-machine positive median, and re-infers `Other/Unknown` reasons from `alarm_code` via `configs/generator.yaml`. Never alters `true_cause` or `scenario.truth`.
3. Learned reliability & ablation (`src/faultlens/reliability.py`): Extracts 16 NaN-free features per `(scenario, defect, severity)`. Fits `LogisticRegression` on `TRAIN`, calibrates on `VALIDATION`, and picks ONE threshold on `VALIDATION` for target coverage (`results/reliability_models.json`). `GroupKFold` by `scenario_id` is used only for out-of-fold reporting.
4. Attribution accuracy (`src/faultlens/metrics.py`): Evaluates whether `quality_report.worst_component` matches the injected defect (`missing_events` -> `completeness`, `incorrect_reason_codes` -> `reason_validity` or `alarm_reason_agreement`, `timestamp_errors` -> `timestamp_consistency`, `duplicate_records` -> `duplication`). Confidence-only baselines report `"N/A"`.
5. Fresh held-out isolation (`fresh_test` & `final`): `python scripts/run_all.py fresh_test` generates 200 synthetic scenarios with `seed=4242` into `data/sample_scenarios/fresh_test/manifest.json` without evaluating them. `python scripts/run_all.py final` fails if `results/reliability_models.json` is missing and creates `results/final_run.lock` when finished, refusing to rerun if `results/final_run.lock` exists.
