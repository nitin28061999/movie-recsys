import pandas as pd
import pytest

from recsys.models import ItemKNNRecommender


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


def test_recommends_from_same_cluster_and_excludes_seen(train):
    model = ItemKNNRecommender().fit(train)
    recs = model.recommend(7, k=2)
    assert set(recs) == {2, 3}
    assert 1 not in recs


def test_cold_user_falls_back_to_popularity(train):
    model = ItemKNNRecommender().fit(train)
    assert model.recommend(99, k=3) == model.fallback_.recommend(99, k=3)


def test_neighbour_pruning(train):
    model = ItemKNNRecommender(n_neighbors=1).fit(train)
    assert (model.sim_.getnnz(axis=1) <= 1).all()


def test_shrinkage_damps_similarity(train):
    loose = ItemKNNRecommender(shrinkage=0.0).fit(train)
    damped = ItemKNNRecommender(shrinkage=10.0).fit(train)
    i, j = loose.item_index_[1], loose.item_index_[2]
    assert damped.sim_[i, j] < loose.sim_[i, j]


def test_liked_only_variant_ignores_low_ratings(train):
    train = train.copy()
    train.loc[(train.userId == 7) & (train.movieId == 1), "rating"] = 2.0
    model = ItemKNNRecommender(min_rating=4.0).fit(train)
    assert model.name == "itemknn-liked"
    assert 7 not in model.user_index_  # no liked history -> popularity fallback
