"""FaultLens Synthetic Machine-Event Log Generator.

Generates reproducible industrial downtime logs across 4 machines and 6 root causes
over a 30-day operational period with realistic shift schedules, MTBF arrivals,
lognormal repair distributions, and symptom-based alarm mappings.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = Path("configs/generator.yaml")


@dataclass
class Scenario:
    """Represents a single synthetic plant simulation with known ground truth."""

    scenario_id: str
    seed: int
    log: pd.DataFrame
    truth: Tuple[str, str]  # (machine_id, true_cause)
    truth_downtime: float
    runner_up: Tuple[str, str]
    runner_up_downtime: float
    margin: float
    difficulty: str  # "easy" | "close" | "moderate"
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize scenario metadata (excluding full DataFrame for lightweight storage)."""
        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "truth": list(self.truth),
            "truth_downtime": float(self.truth_downtime),
            "runner_up": list(self.runner_up),
            "runner_up_downtime": float(self.runner_up_downtime),
            "margin": float(self.margin),
            "difficulty": self.difficulty,
            "metadata": self.metadata,
            "num_events": len(self.log),
        }

    def save(self, out_dir: str | Path) -> Path:
        """Save scenario log to CSV and metadata to JSON."""
        target_dir = Path(out_dir) / self.scenario_id
        target_dir.mkdir(parents=True, exist_ok=True)
        log_path = target_dir / "events.csv"
        meta_path = target_dir / "scenario.json"

        # Save dataframe
        self.log.to_csv(log_path, index=False)

        # Save metadata
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        return target_dir

    @classmethod
    def load(cls, scenario_dir: str | Path) -> "Scenario":
        """Load scenario from directory containing events.csv and scenario.json."""
        scenario_path = Path(scenario_dir)
        meta_file = scenario_path / "scenario.json"
        log_file = scenario_path / "events.csv"

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        log = pd.read_csv(log_file)
        # Parse timestamps
        log["start_ts"] = pd.to_datetime(log["start_ts"])
        log["end_ts"] = pd.to_datetime(log["end_ts"])

        return cls(
            scenario_id=meta["scenario_id"],
            seed=meta["seed"],
            log=log,
            truth=tuple(meta["truth"]),
            truth_downtime=meta["truth_downtime"],
            runner_up=tuple(meta["runner_up"]),
            runner_up_downtime=meta["runner_up_downtime"],
            margin=meta["margin"],
            difficulty=meta["difficulty"],
            metadata=meta.get("metadata", {}),
        )


def load_generator_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load the YAML configuration for synthetic log generation."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Generator config not found at: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_shift(dt: datetime) -> str:
    """Determine operational shift from timestamp (24h clock)."""
    hour = dt.hour
    if 6 <= hour < 14:
        return "Shift_1"
    elif 14 <= hour < 22:
        return "Shift_2"
    else:
        return "Shift_3"


def generate_scenario(
    seed: int,
    scenario_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    target_difficulty: Optional[str] = None,
) -> Scenario:
    """Generate a single 30-day synthetic plant operational scenario.

    Args:
        seed: Random seed guaranteeing deterministic generation.
        scenario_id: Optional unique identifier. Defaults to f"scenario_{seed:04d}".
        config: Optional loaded configuration dictionary.
        target_difficulty: Optional difficulty bias ("easy", "close", or None for natural).

    Returns:
        A fully initialized Scenario instance with ground truth downtime.
    """
    if config is None:
        config = load_generator_config()

    if scenario_id is None:
        scenario_id = f"scenario_{seed:04d}"

    rng = np.random.default_rng(seed)

    sim_cfg = config.get("simulation", {})
    duration_days = sim_cfg.get("duration_days", 30)
    start_base = datetime(2026, 1, 1, 0, 0, 0)
    horizon_seconds = duration_days * 24 * 3600
    end_horizon = start_base + timedelta(seconds=horizon_seconds)

    machines_cfg = config.get("machines", {})
    alarm_mapping = config.get("alarm_mapping", {})
    all_causes = config.get("causes", [
        "jam", "sensor_fault", "overheating", "material_shortage", "tool_wear", "electrical_fault"
    ])

    # Select difficulty profile if not explicitly specified
    if target_difficulty is None:
        # 35% easy, 35% close, 30% moderate
        diff_roll = rng.random()
        if diff_roll < 0.35:
            target_difficulty = "easy"
        elif diff_roll < 0.70:
            target_difficulty = "close"
        else:
            target_difficulty = "moderate"

    # Select dominant pair or close competitors to enforce margin control
    machine_names = list(machines_cfg.keys())
    primary_machine = rng.choice(machine_names)
    primary_cause = rng.choice(all_causes)

    competitor_machine = rng.choice([m for m in machine_names])
    competitor_cause = rng.choice([c for c in all_causes if (competitor_machine, c) != (primary_machine, primary_cause)])

    all_events: List[Dict[str, Any]] = []

    for machine_id, m_cfg in machines_cfg.items():
        base_mtbf_hours = m_cfg.get("base_mtbf_hours", 12.0)
        propensities: Dict[str, float] = dict(m_cfg.get("cause_propensities", {}))
        dur_cfg = m_cfg.get("duration_distribution", {})
        mean_dur_min = dur_cfg.get("mean_min", 15.0)
        sigma = dur_cfg.get("sigma", 0.85)
        micro_prob = dur_cfg.get("micro_stop_prob", 0.10)

        # Apply target difficulty adjustments
        causes_list = list(propensities.keys())
        probs = np.array([propensities[c] for c in causes_list], dtype=float)
        probs = probs / probs.sum()

        if target_difficulty == "easy" and machine_id == primary_machine:
            # Boost primary cause significantly
            idx = causes_list.index(primary_cause) if primary_cause in causes_list else 0
            probs[idx] *= 2.5
            probs = probs / probs.sum()
        elif target_difficulty == "close":
            if machine_id == primary_machine and primary_cause in causes_list:
                idx = causes_list.index(primary_cause)
                probs[idx] *= 1.4
                probs = probs / probs.sum()
            elif machine_id == competitor_machine and competitor_cause in causes_list:
                idx = causes_list.index(competitor_cause)
                probs[idx] *= 1.35
                probs = probs / probs.sum()

        curr_time = start_base
        while curr_time < end_horizon:
            # Poisson arrival: inter-arrival time exponential
            inter_arrival_hours = rng.exponential(scale=base_mtbf_hours)
            stop_start = curr_time + timedelta(hours=inter_arrival_hours)
            if stop_start >= end_horizon:
                break

            # Pick cause
            chosen_cause = rng.choice(causes_list, p=probs)

            # Sample duration
            is_micro = (rng.random() < micro_prob)
            if is_micro:
                duration_min = float(rng.uniform(0.15, 0.95))
            else:
                # Target lognormal parameters matching desired mean
                mu = math.log(mean_dur_min) - (0.5 * sigma * sigma)
                duration_min = float(rng.lognormal(mean=mu, sigma=sigma))
                duration_min = max(1.0, min(duration_min, 480.0))  # Capped at 8 hours max

                # Multiplier for dominant pair in easy mode
                if target_difficulty == "easy" and (machine_id, chosen_cause) == (primary_machine, primary_cause):
                    duration_min *= 1.35
                elif target_difficulty == "close":
                    if (machine_id, chosen_cause) == (primary_machine, primary_cause):
                        duration_min *= 1.10
                    elif (machine_id, chosen_cause) == (competitor_machine, competitor_cause):
                        duration_min *= 1.08

            duration_min = round(duration_min, 2)
            stop_end = stop_start + timedelta(minutes=duration_min)
            shift = _get_shift(stop_start)

            # Determine symptom alarm code
            cause_alarms = alarm_mapping.get(chosen_cause, {})
            primary_alarm = cause_alarms.get("primary", "GENERAL_ALARM")
            secondary_alarms = cause_alarms.get("secondary", ["WARN_GENERIC"])
            if rng.random() < 0.75:
                alarm_code = primary_alarm
            else:
                alarm_code = rng.choice(secondary_alarms)

            # Determine reason source
            if duration_min < 1.5 or chosen_cause == "material_shortage":
                reason_source = "manual"
            else:
                reason_source = "auto"

            logged_reason = chosen_cause  # Clean log has perfect fidelity

            all_events.append({
                "machine_id": machine_id,
                "start_ts": stop_start,
                "end_ts": stop_end,
                "duration_min": duration_min,
                "true_cause": chosen_cause,
                "logged_reason": logged_reason,
                "alarm_code": alarm_code,
                "reason_source": reason_source,
                "shift": shift,
            })

            # Machine is down during stop_end, next event occurs after recovery
            curr_time = stop_end

    # Sort all events chronologically
    all_events.sort(key=lambda ev: ev["start_ts"])
    for idx, ev in enumerate(all_events, start=1):
        ev["event_id"] = f"EVT-{idx:06d}"

    # Reorder columns according to specification:
    # event_id, machine_id, start_ts, end_ts, duration_min, true_cause, logged_reason, alarm_code, reason_source, shift
    cols = [
        "event_id",
        "machine_id",
        "start_ts",
        "end_ts",
        "duration_min",
        "true_cause",
        "logged_reason",
        "alarm_code",
        "reason_source",
        "shift",
    ]
    log_df = pd.DataFrame(all_events)[cols]

    # Compute Ground Truth: (machine, true_cause) pair with highest total true downtime
    downtime_summary = (
        log_df.groupby(["machine_id", "true_cause"])["duration_min"]
        .sum()
        .sort_values(ascending=False)
    )

    if len(downtime_summary) == 0:
        raise ValueError("Generated scenario produced zero stop events.")

    top_pair = tuple(downtime_summary.index[0])
    top_downtime = float(downtime_summary.iloc[0])

    if len(downtime_summary) > 1:
        runner_up_pair = tuple(downtime_summary.index[1])
        runner_up_downtime = float(downtime_summary.iloc[1])
        margin = (top_downtime - runner_up_downtime) / top_downtime
    else:
        runner_up_pair = top_pair
        runner_up_downtime = 0.0
        margin = 1.0

    # Categorize actual realized margin
    margin = round(margin, 4)
    if margin >= 0.18:
        realized_difficulty = "easy"
    elif margin <= 0.07:
        realized_difficulty = "close"
    else:
        realized_difficulty = "moderate"

    metadata = {
        "target_difficulty": target_difficulty,
        "primary_target": [primary_machine, primary_cause],
        "competitor_target": [competitor_machine, competitor_cause],
        "total_stops": len(log_df),
        "total_downtime_hours": round(float(log_df["duration_min"].sum()) / 60.0, 2),
        "micro_stops_count": int((log_df["duration_min"] < 1.0).sum()),
    }

    return Scenario(
        scenario_id=scenario_id,
        seed=seed,
        log=log_df,
        truth=top_pair,
        truth_downtime=top_downtime,
        runner_up=runner_up_pair,
        runner_up_downtime=runner_up_downtime,
        margin=margin,
        difficulty=realized_difficulty,
        metadata=metadata,
    )


def generate_benchmark_scenarios(
    num_scenarios: int = 300,
    master_seed: int = 42,
    out_dir: Optional[str | Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Scenario], Dict[str, List[str]]]:
    """Generate 300 synthetic scenarios and split 40/20/40 into train/val/test.

    Args:
        num_scenarios: Total number of scenarios (default 300).
        master_seed: Fixed master seed for scenario and split reproduction.
        out_dir: Destination directory for scenarios and splits.json.
        config: Loaded generator configuration.

    Returns:
        Tuple of (list_of_scenarios, split_dict).
    """
    if config is None:
        config = load_generator_config()

    rng = np.random.default_rng(master_seed)
    scenario_seeds = rng.integers(low=1000, high=999999, size=num_scenarios)

    scenarios: List[Scenario] = []
    for idx, s_seed in enumerate(scenario_seeds):
        sc_id = f"scenario_{idx:03d}"
        sc = generate_scenario(seed=int(s_seed), scenario_id=sc_id, config=config)
        scenarios.append(sc)

    # 40 / 20 / 40 split (exact counts: 120 train, 60 val, 120 test)
    n_train = int(round(num_scenarios * 0.40))
    n_val = int(round(num_scenarios * 0.20))
    n_test = num_scenarios - n_train - n_val

    all_indices = np.arange(num_scenarios)
    rng.shuffle(all_indices)

    train_indices = [int(i) for i in all_indices[:n_train]]
    val_indices = [int(i) for i in all_indices[n_train : n_train + n_val]]
    test_indices = [int(i) for i in all_indices[n_train + n_val :]]

    splits = {
        "train": [scenarios[i].scenario_id for i in train_indices],
        "val": [scenarios[i].scenario_id for i in val_indices],
        "test": [scenarios[i].scenario_id for i in test_indices],
    }

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for sc in scenarios:
            sc.save(out_path)

        splits_file = out_path / "splits.json"
        with open(splits_file, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)

    return scenarios, splits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FaultLens Synthetic Generator")
    parser.add_argument("--generate-split", action="store_true", help="Generate benchmark scenarios")
    parser.add_argument("--count", type=int, default=300, help="Number of scenarios")
    parser.add_argument("--out-dir", type=str, default="data/sample_scenarios", help="Output directory")
    args = parser.parse_args()

    if args.generate_split:
        print(f"Generating {args.count} scenarios to {args.out_dir}...")
        scs, sp = generate_benchmark_scenarios(num_scenarios=args.count, out_dir=args.out_dir)
        print(f"Generated {len(scs)} scenarios.")
        print(f"Splits: Train={len(sp['train'])}, Val={len(sp['val'])}, Test={len(sp['test'])}")
