import pandas as pd
import pytest

from recsys.models import ContentBasedRecommender


@pytest.fixture
def movies():
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5, 6, 7],
            "title": list("ABCDEFG"),
            "genres": [
                "Action", "Action", "Action", "Romance", "Romance", "Action|Romance", "Action",
            ],
        }
    )


@pytest.fixture
def train():
    rows = [(1, 1, 5.0, 1), (1, 4, 2.0, 2), (2, 1, 5.0, 1), (2, 2, 5.0, 2), (2, 3, 4.0, 3)]
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


def test_recommends_similar_genre_and_excludes_seen(movies, train):
    model = ContentBasedRecommender(movies).fit(train)
    recs = model.recommend(1, k=2)
    assert set(recs) == {2, 3}  # pure-Action items beat the Romance ones
    assert not set(recs) & {1, 4}  # already seen


def test_can_recommend_cold_item(movies, train):
    model = ContentBasedRecommender(movies).fit(train)
    assert 7 in model.recommend(1, k=5)  # movie 7 has zero ratings in train


def test_cold_user_falls_back_to_popularity(movies, train):
    model = ContentBasedRecommender(movies).fit(train)
    assert model.recommend(99, k=3)[0] == 1  # most-liked movie overall


def test_tags_variant_runs(movies, train):
    tags = pd.DataFrame({"userId": [1], "movieId": [5], "tag": ["Heist"], "timestamp": [1]})
    model = ContentBasedRecommender(movies, tags).fit(train)
    assert model.name == "content-genres+tags"
    assert len(model.recommend(1, k=3)) == 3
