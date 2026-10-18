# FaultLens: Assumptions Log

This document records all domain, mathematical, and implementation assumptions made in FaultLens (Session 1 onwards). Every assumption is explicit and verifiable.

## 1. Domain & Manufacturing Process Assumptions

1. **Synthetic Nature**: All machine-event logs and scenarios are 100% synthetic. No proprietary or real customer data is used. Ground truth downtime is explicitly recorded at the generator level.
2. **Topology**: The plant line consists of 4 discrete machines in sequence: `Loader`, `Press`, `Welder`, `Packer`.
3. **Failure Independency**: Machine failures occur as independent stochastic events driven by Poisson arrival processes (exponential Mean Time Between Failures [MTBF]), modulated by machine-specific propensities across 6 failure causes:
   - `jam`
   - `sensor_fault`
   - `overheating`
   - `material_shortage`
   - `tool_wear`
   - `electrical_fault`
4. **Repair Duration Distribution**: Downtime durations follow right-skewed lognormal distributions with parameters calibrated to realistic shop-floor intervals (median stops 10-25 minutes, with heavy tails representing stubborn breakdowns). A controlled fraction of micro-stops (<1.0 min) is included to represent quick resets.
5. **Shifts & Operating Time**: Standard 24/7 operations divided into three equal 8-hour shifts (`Shift_1`: 06:00-14:00, `Shift_2`: 14:00-22:00, `Shift_3`: 22:00-06:00) over 30 days.
6. **Reason Logging Mechanisms**:
   - `auto`: Direct PLC/SCADA automated alarm triggers for hardware faults (`sensor_fault`, `electrical_fault`, `overheating`, `tool_wear`, automated conveyor jams).
   - `manual`: Human operator entered reasons for operational issues (`material_shortage`) and micro-stops (<1.5 min) where operators clear jams without entering a formal ticket.
7. **Alarm Ambiguity**: Alarms are physical symptoms, not root causes. In particular, both `sensor_fault` (broken thermocouple/switch) and `electrical_fault` (voltage drop/short) raise `SENSOR_ALARM`.

## 2. Ground Truth & Margin Control Assumptions

1. **Ground Truth Definition**: The single true primary downtime driver (`scenario.truth`) is strictly defined as the `(machine_id, true_cause)` pair that accumulates the greatest total downtime minutes over the 30-day evaluation horizon:
   $$\text{truth} = \arg\max_{(m, c)} \sum_{i \in \text{events}(m, c)} \text{duration\_min}_i$$
2. **Margin Definition**: The ground truth margin is the normalized difference between the #1 candidate's downtime and the #2 candidate's downtime:
   $$\text{margin} = \frac{D_{(1)} - D_{(2)}}{D_{(1)}}$$
   - Easy scenarios exhibit a clear dominant bottleneck ($\text{margin} \ge 0.18$).
   - Close scenarios exhibit competing bottlenecks ($\text{margin} \le 0.07$).

## 3. Defect Injection Assumptions

1. **Orthogonality**: Exactly 4 defect types are evaluated independently at discrete severities (0%, 10%, 20%, 30%). No unmodeled compound defects are introduced in the benchmark.
2. **Invariance of Truth**: Defect injectors are pure transformations that alter logged columns (`logged_reason`, `start_ts`, `end_ts`, row inclusions) but **NEVER** modify `true_cause` or `scenario.truth`.
3. **Missing Events**: 50% of affected records are dropped completely (dropped telemetry packets); 50% have their `end_ts` set to null/NaN (unclosed stop intervals).
4. **Incorrect Reason Codes**: Follow an industry-realistic confusion matrix reflecting operator shortcuts and alarm overlaps (e.g., swapping `sensor_fault` and `electrical_fault`, or labelling ambiguous halts as `Other/Unknown`).
5. **Timestamp Errors**: Individual machine clocks experience systematic offsets ($\pm 1$ to $15$ min) and an affected fraction of events experiences independent Gaussian start/end communication jitter ($\sigma$ up to 14 min at 30% severity). This generates erroneous durations, timestamp inversions ($\text{end\_ts} \le \text{start\_ts}$), and same-machine interval overlaps.
6. **Non-Positive Duration Exclusion**: The shared diagnoser strictly filters out non-positive durations ($\text{duration\_min} \le 0$) and unclosed stops ($\text{end\_ts} \text{ is null}$), counting them as invalid events. Corrupted negative intervals are excluded from downtime accumulation, directly degrading candidate rankings under timestamp noise.
7. **Duplicate Records**: Ingestion pipeline retries and bounce contacts create duplicate single events (70%) and multi-record bursts of 3-10 repeated records within 45 seconds (30%).

## 4. Diagnoser & Confidence Metric Assumptions

1. **Ranking Criterion**: Downtime accumulation by `(machine, logged_reason)`:
   $$S_{(m, r)} = \sum_{j \in \text{valid\_stops}(m, r)} (\text{end\_ts}_j - \text{start\_ts}_j)$$
   Events with null or non-positive durations (e.g., unclosed stops) are skipped during accumulation.
2. **Raw Confidence Formula (Margin-based Softmax)**:
   Given candidate scores $S_1 \ge S_2 \ge \dots \ge S_K$ where $S_1$ is the top candidate score:
   - When only one candidate exists ($K = 1$), raw confidence is 1.0.
   - When multiple candidates exist ($K \ge 2$), we evaluate the top margin $\Delta = S_1 - S_2$ relative to the mean scale of the remaining candidates $\bar{S}_{\text{rest}} = \frac{1}{K-1} \sum_{j=2}^K S_j$:
     $$z = \frac{S_1 - S_2}{\max(\bar{S}_{\text{rest}}, \epsilon)}$$
     $$\text{raw\_confidence} = \frac{1}{1 + e^{-z / T}}$$
     where $T = 1.0$ is the softmax temperature.
     - When $S_1 = S_2$ (ambiguous dead heat), $z = 0 \implies \text{raw\_confidence} = 0.50$.
     - When $S_1$ decisively dominates all runner-up causes ($z \gg 1$), $\text{raw\_confidence} \to 1.0$.
3. **Calibration Assumption**: Raw confidence reflects heuristic margin separation, which tends to be overconfident under clean data and poorly calibrated under defects. Calibration maps raw confidence $c \in [0.5, 1.0] \to \hat{p} \in [0, 1]$ using Isotonic Regression (non-parametric monotonic binning) fit strictly on the validation set. ECE (Expected Calibration Error) is evaluated using 10 equal-width bins.
