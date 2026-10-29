import numpy as np
import pytest

from recsys.stats import bootstrap_ci, paired_bootstrap_diff


def test_ci_brackets_the_mean_and_is_deterministic():
    values = np.random.default_rng(0).normal(0.5, 0.2, 300)
    low, high = bootstrap_ci(values)
    assert low < values.mean() < high
    assert (low, high) == bootstrap_ci(values)


def test_identical_models_have_zero_difference():
    x = np.random.default_rng(1).random(100)
    assert paired_bootstrap_diff(x, x) == (0.0, 0.0, 0.0)


def test_clearly_better_model_has_ci_above_zero():
    rng = np.random.default_rng(2)
    b = rng.random(200)
    mean, low, _ = paired_bootstrap_diff(b + 0.1 + rng.normal(0, 0.02, 200), b)
    assert mean == pytest.approx(0.1, abs=0.01)
    assert low > 0
