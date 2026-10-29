"""Run every model through the same split + metrics, print one results table."""
import pandas as pd

from recsys import config, metrics
from recsys.data import (
    download_movielens,
    load_movies,
    load_ratings,
    load_tags,
    temporal_split,
)
from recsys.models import (
    ContentBasedRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    Recommender,
)


def evaluate_model(
    model: Recommender,
    test: pd.DataFrame,
    n_items: int,
    k: int = config.TOP_K,
    threshold: float = config.RELEVANCE_THRESHOLD,
) -> dict:
    relevant_by_user = (
        test[test["rating"] >= threshold].groupby("userId")["movieId"].agg(set).to_dict()
    )
    p, r, n, all_recs = [], [], [], []
    for user_id, relevant in relevant_by_user.items():
        recs = model.recommend(user_id, k)
        all_recs.append(recs)
        p.append(metrics.precision_at_k(recs, relevant, k))
        r.append(metrics.recall_at_k(recs, relevant, k))
        n.append(metrics.ndcg_at_k(recs, relevant, k))
    count = max(len(p), 1)
    return {
        "model": model.name,
        f"precision@{k}": sum(p) / count,
        f"recall@{k}": sum(r) / count,
        f"ndcg@{k}": sum(n) / count,
        "coverage": metrics.catalog_coverage(all_recs, n_items),
        "users_evaluated": len(p),
    }


def main() -> None:
    download_movielens()
    ratings = load_ratings()
    train, val, _ = temporal_split(ratings)
    movies, tags = load_movies(), load_tags()
    n_items = len(movies)  # coverage is measured against the full catalog

    models: list[Recommender] = [  # add new models here
        PopularityRecommender(),
        ContentBasedRecommender(movies),
        ContentBasedRecommender(movies, tags),
        ItemKNNRecommender(),
        ItemKNNRecommender(min_rating=config.RELEVANCE_THRESHOLD),
    ]
    rows = [evaluate_model(m.fit(train), val, n_items) for m in models]
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
