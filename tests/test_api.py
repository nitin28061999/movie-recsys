"""API tests against a tiny in-memory dataset (no downloads, no real model fitting cost)."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from recsys.models import ALSRecommender, HybridRecommender, ItemKNNRecommender

API_FILE = Path(__file__).resolve().parents[1] / "api" / "main.py"


def load_api_module():
    spec = importlib.util.spec_from_file_location("recsys_api", API_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def client(monkeypatch):
    api = load_api_module()

    movies = pd.DataFrame(
        {
            "movieId": range(1, 9),
            "title": [f"Film {i} (2000)" for i in range(1, 9)],
            "genres": ["Action"] * 4 + ["Romance"] * 4,
        }
    )
    rows = []
    for user in (1, 2, 3):  # action fans
        rows += [(user, m, 5.0, m) for m in (1, 2, 3, 4)]
    for user in (4, 5, 6):  # romance fans
        rows += [(user, m, 5.0, m) for m in (5, 6, 7, 8)]
    rows.append((7, 1, 5.0, 1))  # sees one action film, has plenty of room to be recommended more
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])

    def fake_build():
        als = ALSRecommender(factors=4, iterations=10, min_rating=4.0).fit(ratings)
        knn = ItemKNNRecommender(min_rating=4.0).fit(ratings)
        return HybridRecommender([(als, 1.0), (knn, 1.0)], prefit=True, name="hybrid"), movies, ratings

    monkeypatch.setattr(api, "build_recommender", fake_build)
    with TestClient(api.app) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_known_user_gets_personalised_unseen_recommendations(client):
    body = client.get("/recommendations/7?k=3").json()
    assert body["personalized"] is True
    assert body["strategy"] == "hybrid"
    ids = {r["movieId"] for r in body["recommendations"]}
    assert 1 not in ids  # already rated
    assert ids <= {2, 3, 4}  # action fans' items, not the romance cluster


def test_unknown_user_gets_flagged_popularity_fallback(client):
    a = client.get("/recommendations/9999?k=3").json()
    b = client.get("/recommendations/8888?k=3").json()
    assert a["personalized"] is False
    assert a["strategy"] == "popularity-fallback"
    assert a["recommendations"] == b["recommendations"]


def test_input_validation(client):
    assert client.get("/recommendations/0").status_code == 400
    assert client.get("/recommendations/1?k=0").status_code == 422
    assert client.get("/recommendations/1?k=51").status_code == 422
    assert client.get("/movies/search?q=").status_code == 422


def test_search_and_users(client):
    found = client.get("/movies/search?q=film 3").json()
    assert found["count"] == 1 and found["results"][0]["movieId"] == 3
    assert client.get("/users?limit=3").json() == {"count": 3, "userIds": [1, 2, 3]}
