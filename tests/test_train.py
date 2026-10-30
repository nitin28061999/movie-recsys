import pandas as pd
import pytest

from recsys import config, train


@pytest.fixture
def tiny_data(monkeypatch):
    movies = pd.DataFrame(
        {
            "movieId": range(1, 9),
            "title": [f"Film {i}" for i in range(1, 9)],
            "genres": ["Action"] * 4 + ["Romance"] * 4,
        }
    )
    rows = [(u, m, 5.0, m) for u in (1, 2, 3) for m in (1, 2, 3, 4)]
    rows += [(u, m, 5.0, m) for u in (4, 5, 6) for m in (5, 6, 7, 8)]
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    monkeypatch.setattr(train, "download_movielens", lambda: None)
    monkeypatch.setattr(train, "load_ratings", lambda: ratings)
    monkeypatch.setattr(train, "load_movies", lambda: movies)


def test_train_saves_a_loadable_bundle(tiny_data, tmp_path):
    path = train.train(tmp_path / "models" / "rec.joblib")
    assert path.exists()
    recommender, movies, ratings = train.load_or_train(path)
    assert recommender.name == "hybrid-als+knn"
    assert len(movies) == 8 and len(ratings) == 24
    assert len(recommender.recommend(1, k=2)) == 2


def test_missing_model_is_trained_on_demand(tiny_data, tmp_path):
    path = tmp_path / "rec.joblib"
    assert not path.exists()
    train.load_or_train(path)
    assert path.exists()


def test_changed_hyperparameters_trigger_retrain(tiny_data, tmp_path, monkeypatch):
    path = tmp_path / "rec.joblib"
    train.train(path)
    monkeypatch.setattr(config, "KNN_PARAMS", {"n_neighbors": 5, "shrinkage": 0.0})
    recommender, _, _ = train.load_or_train(path)
    knn = recommender.components[1][0]
    assert knn.n_neighbors == 5
