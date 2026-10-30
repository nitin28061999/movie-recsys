"""Fit the production recommender on ALL ratings and save it for the API to load.

    python -m recsys.train

The saved bundle also records the hyperparameters it was trained with, so changing
ALS_PARAMS / KNN_PARAMS in config.py triggers a retrain instead of serving a stale model.
Only load artifacts you trained yourself: joblib files are pickles and can run code.
"""
import joblib

from recsys import config
from recsys.data import download_movielens, load_movies, load_ratings
from recsys.models import ALSRecommender, HybridRecommender, ItemKNNRecommender

ARTIFACT = config.MODEL_DIR / "recommender.joblib"


def fingerprint() -> dict:
    """Everything that, if changed, makes a saved model out of date."""
    return {
        "als": dict(config.ALS_PARAMS),
        "knn": dict(config.KNN_PARAMS),
        "threshold": config.RELEVANCE_THRESHOLD,
    }


def fit_production_model(ratings) -> HybridRecommender:
    als = ALSRecommender(min_rating=config.RELEVANCE_THRESHOLD, **config.ALS_PARAMS).fit(ratings)
    knn = ItemKNNRecommender(min_rating=config.RELEVANCE_THRESHOLD, **config.KNN_PARAMS).fit(ratings)
    return HybridRecommender([(als, 1.0), (knn, 1.0)], prefit=True, name="hybrid-als+knn")


def train(path=ARTIFACT):
    """Download data if needed, fit on every rating, and save the bundle. Returns the path."""
    download_movielens()
    ratings, movies = load_ratings(), load_movies()
    bundle = {
        "recommender": fit_production_model(ratings),
        "movies": movies,
        "ratings": ratings,
        "fingerprint": fingerprint(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_or_train(path=ARTIFACT):
    """Return (recommender, movies, ratings), training first if the saved model is missing/stale."""
    bundle = joblib.load(path) if path.exists() else None
    if bundle is None or bundle.get("fingerprint") != fingerprint():
        train(path)
        bundle = joblib.load(path)
    return bundle["recommender"], bundle["movies"], bundle["ratings"]


if __name__ == "__main__":
    print(f"saved {train()}")
