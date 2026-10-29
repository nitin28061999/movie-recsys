"""Small statistics helpers: bootstrap confidence intervals over users."""
import numpy as np

from recsys import config


def bootstrap_ci(values, n_boot: int = 1000, alpha: float = 0.05, seed: int = config.SEED):
    """Percentile-bootstrap CI for the mean of per-user values. Returns (low, high)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_diff(a, b, n_boot: int = 1000, alpha: float = 0.05, seed: int = config.SEED):
    """Mean of (a - b) over users, with a bootstrap CI. Returns (mean, low, high).

    Pairing matters: both models are scored on the same users, so resampling the
    per-user differences gives a much tighter, fairer interval than comparing two
    separate CIs.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    low, high = bootstrap_ci(diff, n_boot, alpha, seed)
    return float(diff.mean()), low, high
