import pandas as pd
import pytest


@pytest.fixture
def ratings():
    """Synthetic ratings: users 1-3 rate 20 distinct movies, user 4 only 3, user 5 rates 40."""
    rows = []
    for u in (1, 2, 3):
        for i in range(20):
            rows.append((u, 100 + (2 * i + u) % 40, 5.0 if i % 2 == 0 else 3.0, 1000 + i))
    for i in range(3):
        rows.append((4, 100 + i, 4.0, 2000 + i))
    for i in range(40):
        rows.append((5, 100 + i, 5.0, 3000 + i))
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
