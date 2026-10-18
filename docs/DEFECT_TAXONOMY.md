# FaultLens: Defect Taxonomy & Industrial Grounding

This document defines the 4 controlled data defects evaluated in FaultLens, their mathematical parameterization, and their real-world industrial basis.

## Benchmark Design Principles
1. **Purity**: Each injector operates independently on synthetic logs without mutating ground truth annotations (`true_cause`, `scenario.truth`).
2. **Discrete Severities**: Exactly 4 severities are tested: 0% (clean control), 10% (mild noise), 20% (moderate operational noise), 30% (severe degradation).
3. **Strict Scope**: Only 4 defects are permitted. No undocumented or mixed defect combinations are introduced.

---

## 1. Missing Events (`missing_events`)

### Industrial Basis
- **Wireless/Cellular Gateway Drops**: Industrial IoT edge gateways communicating over Wi-Fi or LoRaWAN frequently drop UDP packets or buffer frames during link disruptions.
- **Unclosed Stop Intervals**: A machine stops, but the operator or SCADA fails to register the restart or acknowledgment signal, leaving an open record without an `end_ts`.
- **Database Connection Timeouts**: Historian server write timeouts during network partition.

### Parameters & Mechanics
- **Severities**: $s \in \{0.0, 0.10, 0.20, 0.30\}$ of total log records randomly selected.
- **Split Mechanics**:
  - **Complete Dropping (50% of affected)**: The entire event row is pruned from the log.
  - **Unclosed Intervals (50% of affected)**: The row is retained, but `end_ts` is set to null (`NaN`/`None`), and calculated `duration_min` is invalidated.
- **Impact on Diagnosis**: Reduces sample size for low-frequency faults and undercounts long breakdowns if long intervals are preferentially dropped or unclosed.

---

## 2. Incorrect Reason Codes (`incorrect_reason_codes`)

### Industrial Basis
- **Operator Misattribution**: When lines halt, busy operators select the first plausible menu option on HMI touchscreens (e.g., logging a conveyor jam when the root cause was upstream material starvation).
- **Alarm Code Ambiguity**: PLC alarms report symptoms rather than causes (e.g., a tripping motor overcurrent alarm can be logged as "electrical_fault" when a mechanical jam physically seized the drive shaft).
- **Default/Catch-All Coding**: Operators selecting "Other", "Unknown", or default reason code when diagnosing root causes under shift time pressure.

### Parameters & Mechanics
- **Severities**: $s \in \{0.0, 0.10, 0.20, 0.30\}$ of total records subjected to reason replacement.
- **Confusion Map**:
  - `sensor_fault` $\to$ 60% `electrical_fault`, 40% `Other/Unknown`
  - `electrical_fault` $\to$ 65% `sensor_fault`, 35% `Other/Unknown`
  - `jam` $\to$ 70% `material_shortage`, 30% `Other/Unknown`
  - `material_shortage` $\to$ 60% `jam`, 40% `Other/Unknown`
  - `overheating` $\to$ 70% `Other/Unknown`, 30% `electrical_fault`
  - `tool_wear` $\to$ 45% `jam`, 55% `Other/Unknown`
- **Impact on Diagnosis**: Directly corrupts the logged downtime tally for candidates, causing the naive frequency diagnoser to misidentify top causes.

---

## 3. Timestamp Errors (`timestamp_errors`)

### Industrial Basis
- **PLC Clock Drift & Lack of NTP**: Legacy PLCs frequently drift by several minutes per month if not synchronized to an authoritative NTP time server.
- **Asynchronous Data Pipelines**: Batch data extractors push records with local PC clock timestamps rather than sensor generation timestamps.
- **Buffering & Jitter**: Multi-hop edge networks queue packets and release them in bursts with variable network transit delays.

### Parameters & Mechanics
- **Severities**: $s \in \{0.0, 0.10, 0.20, 0.30\}$.
  - When $s > 0$, machine clocks are assigned fixed systematic offsets:
    $$\Delta t_m \sim \text{Uniform}(-15.0 \times (s / 0.30), +15.0 \times (s / 0.30)) \text{ minutes}$$
  - In addition, an affected fraction $s$ of events experiences stochastic communication jitter:
    $$\delta t_{ij} \sim \mathcal{N}(0, \sigma^2), \quad \sigma = 2.5 \text{ min}$$
- **Impact on Diagnosis**: Shifts sequence ordering, causes physically impossible overlapping stops on the same machine, and distorts shift-boundary downtime allocation.

---

## 4. Duplicate Records (`duplicate_records`)

### Industrial Basis
- **Network Retry Floods**: MQTT QoS 1 ("at-least-once") or HTTP retry loops resend packets if an acknowledgment packet is dropped by an unreliable industrial switch.
- **Sensor Contact Bouncing / Chatter**: Proximity switches or photo-eyes repeatedly toggle on/off at boundary conditions, generating a rapid cascade of micro-events for a single mechanical stop.
- **Multi-Pipeline Ingestion**: Duplicate ingestion jobs (e.g., both an automated MES pipeline and a backup batch ETL) writing to the same operational store.

### Parameters & Mechanics
- **Severities**: $s \in \{0.0, 0.10, 0.20, 0.30\}$ of original events replicated.
- **Mechanics**:
  - **Single Duplicates (70% of affected)**: Exactly one identical row is appended into the log with the same timestamps and attributes.
  - **Burst Duplicates (30% of affected)**: A single event spawns a burst of $k \in [3, 10]$ records with timestamp jitter clustered within $\Delta t \le 45$ seconds.
- **Impact on Diagnosis**: Artificially inflates event frequency counts and downtime sums if deduplication is not performed, biasing frequency-driven algorithms toward chatter-prone sensors.
