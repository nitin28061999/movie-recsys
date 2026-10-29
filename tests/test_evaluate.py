import numpy as np
import pandas as pd
import pytest

from recsys.evaluate import evaluate_model, paired_vs_baseline, per_user_metrics
from recsys.models import Recommender


class FixedRecommender(Recommender):
    name = "fixed"

    def fit(self, train):
        return self

    def recommend(self, user_id, k=10):
        return [1, 2, 3][:k]


@pytest.fixture
def test_set():
    rows = [(1, 2, 5.0, 1), (1, 9, 2.0, 2), (2, 9, 5.0, 1)]  # user 1 likes item 2; user 2 item 9
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


def test_per_user_metrics_values(test_set):
    df = per_user_metrics(FixedRecommender(), test_set, k=3)
    assert df.loc[1, "precision"] == pytest.approx(1 / 3)
    assert df.loc[1, "recall"] == 1.0
    assert df.loc[2, "precision"] == 0.0
    assert df.loc[1, "recs"] == [1, 2, 3]


def test_evaluate_model_summary(test_set):
    out = evaluate_model(FixedRecommender(), test_set, n_items=10, k=3)
    assert out["precision@3"] == pytest.approx(1 / 6)
    assert out["coverage"] == pytest.approx(0.3)
    assert out["users_evaluated"] == 2


def test_paired_vs_baseline_flags_clear_differences():
    rng = np.random.default_rng(0)
    idx = pd.Index(range(1, 201), name="userId")
    base = pd.DataFrame(rng.random((200, 3)) * 0.1, columns=["precision", "recall", "ndcg"], index=idx)
    out = paired_vs_baseline({"base": base, "better": base + 0.2}, baseline="base")
    assert list(out["model"]) == ["better"]
    assert out.loc[0, "d_ndcg@10"].endswith("*")
    assert "d_recall@10" in out.columns
