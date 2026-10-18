"""Unit tests for train/val/test splits."""

import json
from pathlib import Path
from faultlens.generator import generate_benchmark_scenarios


def test_train_val_test_split_sizes_and_disjointness(tmp_path: Path) -> None:
    """Verify 40/20/40 split sizes (120/60/120 for N=300) and zero overlap."""
    n_scenarios = 300
    scenarios, splits = generate_benchmark_scenarios(
        num_scenarios=n_scenarios,
        master_seed=42,
        out_dir=tmp_path,
    )

    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])

    # Verify exact split counts
    assert len(train_ids) == 120  # 40%
    assert len(val_ids) == 60     # 20%
    assert len(test_ids) == 120   # 40%
    assert len(train_ids) + len(val_ids) + len(test_ids) == n_scenarios

    # Verify zero overlap (disjoint sets)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # Verify splits.json saved properly
    splits_file = tmp_path / "splits.json"
    assert splits_file.is_file()
    with open(splits_file, "r", encoding="utf-8") as f:
        loaded_splits = json.load(f)
    assert loaded_splits == splits
