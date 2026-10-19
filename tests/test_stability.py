import time
import pytest
from faultlens.generator import generate_scenario
from faultlens.stability import evaluate_stability


def test_stability_speed_under_two_seconds() -> None:
    """Verify stability testing executes in well under 2.0 seconds at N=200."""
    sc = generate_scenario(seed=101)
    t0 = time.perf_counter()
    rep = evaluate_stability(sc.log, num_perturbations=200, seed=42)
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.0, f"Stability test took {elapsed:.3f}s, exceeding 2.0s limit"
    assert rep.num_perturbations == 200
    assert len(rep.candidate_cis) > 0


def test_stability_high_on_dominant_low_on_close() -> None:
    """Verify stability is high on dominant scenarios and lower on close calls."""
    # Clearly dominant scenario (target_difficulty="easy")
    sc_easy = generate_scenario(seed=777, target_difficulty="easy")
    rep_easy = evaluate_stability(sc_easy.log, num_perturbations=100, seed=42)

    # Close scenario (target_difficulty="close")
    sc_close = generate_scenario(seed=555, target_difficulty="close")
    rep_close = evaluate_stability(sc_close.log, num_perturbations=100, seed=42)

    assert rep_easy.top_stability >= 0.85, f"Dominant scenario stability too low: {rep_easy.top_stability}"
    assert rep_close.top_stability < rep_easy.top_stability
    assert rep_close.top3_flip_rate > rep_easy.top3_flip_rate
