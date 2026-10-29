import pandas as pd
import pytest

from recsys.models import ALSRecommender


@pytest.fixture
def train():
    """Two item clusters: users 1-3 watch {1,2,3}; users 4-6 watch {4,5,6}; user 7 only item 1."""
    rows = []
    for u in (1, 2, 3):
        rows += [(u, m, 5.0, t) for t, m in enumerate((1, 2, 3))]
    for u in (4, 5, 6):
        rows += [(u, m, 5.0, t) for t, m in enumerate((4, 5, 6))]
    rows.append((7, 1, 5.0, 0))
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


def make(**kwargs):
    return ALSRecommender(factors=4, iterations=20, **kwargs)


def test_recommends_from_same_cluster_and_excludes_seen(train):
    recs = make().fit(train).recommend(7, k=2)
    assert set(recs) == {2, 3}
    assert 1 not in recs


def test_scores_reflect_cluster_structure(train):
    model = make().fit(train)
    u = model.user_index_[1]
    score = lambda m: model.item_factors_[model.item_index_[m]] @ model.user_factors_[u]
    assert score(2) > score(5)


def test_same_seed_gives_same_factors(train):
    a, b = make().fit(train), make().fit(train)
    assert (a.item_factors_ == b.item_factors_).all()


def test_cold_user_falls_back_to_popularity(train):
    model = make().fit(train)
    assert model.recommend(99, k=3) == model.fallback_.recommend(99, k=3)


def test_liked_only_variant_ignores_low_ratings(train):
    train = train.copy()
    train.loc[(train.userId == 7) & (train.movieId == 1), "rating"] = 2.0
    model = make(min_rating=4.0).fit(train)
    assert model.name == "als-liked"
    assert 7 not in model.user_index_
