"""Run every model through the same split + metrics, print one results table.

    python -m recsys.evaluate               # score on the validation split (use while developing)
    python -m recsys.evaluate --split test  # final numbers: train+val -> test (run this once)
"""
import argparse

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
    ALSRecommender,
    ContentBasedRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    Recommender,
)
from recsys.stats import paired_bootstrap_diff

COLUMNS = ["userId", "precision", "recall", "ndcg", "recs"]


def per_user_metrics(
    model: Recommender,
    test: pd.DataFrame,
    k: int = config.TOP_K,
    threshold: float = config.RELEVANCE_THRESHOLD,
) -> pd.DataFrame:
    """One row per evaluated user: precision, recall, ndcg and the recommended list."""
    relevant_by_user = (
        test[test["rating"] >= threshold].groupby("userId")["movieId"].agg(set).to_dict()
    )
    rows = []
    for user_id, relevant in relevant_by_user.items():
        recs = model.recommend(user_id, k)
        rows.append(
            {
                "userId": user_id,
                "precision": metrics.precision_at_k(recs, relevant, k),
                "recall": metrics.recall_at_k(recs, relevant, k),
                "ndcg": metrics.ndcg_at_k(recs, relevant, k),
                "recs": recs,
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS).set_index("userId")


def summarise(name: str, per_user: pd.DataFrame, n_items: int, k: int = config.TOP_K) -> dict:
    return {
        "model": name,
        f"precision@{k}": per_user["precision"].mean(),
        f"recall@{k}": per_user["recall"].mean(),
        f"ndcg@{k}": per_user["ndcg"].mean(),
        "coverage": metrics.catalog_coverage(per_user["recs"], n_items),
        "users_evaluated": len(per_user),
    }


def evaluate_model(
    model: Recommender,
    test: pd.DataFrame,
    n_items: int,
    k: int = config.TOP_K,
    threshold: float = config.RELEVANCE_THRESHOLD,
) -> dict:
    return summarise(model.name, per_user_metrics(model, test, k, threshold), n_items, k)


def build_models(movies: pd.DataFrame, tags: pd.DataFrame) -> list[Recommender]:
    return [  # add new models here
        PopularityRecommender(),
        ContentBasedRecommender(movies),
        ContentBasedRecommender(movies, tags),
        ItemKNNRecommender(),
        ItemKNNRecommender(min_rating=config.RELEVANCE_THRESHOLD),
        ALSRecommender(),
        ALSRecommender(min_rating=config.RELEVANCE_THRESHOLD),
    ]


def paired_vs_baseline(
    results: dict[str, pd.DataFrame], baseline: str = "popularity", k: int = config.TOP_K
) -> pd.DataFrame:
    """Per-model difference from the baseline with a paired bootstrap CI over users."""
    base = results[baseline]
    rows = []
    for name, df in results.items():
        if name == baseline:
            continue
        row = {"model": name}
        for metric in ("precision", "ndcg"):
            diff, low, high = paired_bootstrap_diff(df.loc[base.index, metric], base[metric])
            flag = "*" if low > 0 or high < 0 else ""
            row[f"d_{metric}@{k}"] = f"{diff:+.4f} [{low:+.4f}, {high:+.4f}]{flag}"
        rows.append(row)
    return pd.DataFrame(rows)


def main(split: str = "val") -> None:
    download_movielens()
    ratings = load_ratings()
    movies, tags = load_movies(), load_tags()
    train, val, test = temporal_split(ratings)
    if split == "test":
        train, eval_set = pd.concat([train, val]), test  # final: fit on train+val, score on test
    else:
        eval_set = val
    n_items = len(movies)  # coverage is measured against the full catalog

    results = {}
    for model in build_models(movies, tags):
        print(f"fitting {model.name} ...", flush=True)
        results[model.name] = per_user_metrics(model.fit(train), eval_set)

    table = pd.DataFrame([summarise(n, df, n_items) for n, df in results.items()])
    print(f"\nResults on the {split} split (k={config.TOP_K})")
    print(table.round(4).to_string(index=False))
    print("\nDifference vs popularity: mean over users [95% bootstrap CI]; * = CI excludes 0")
    print(paired_vs_baseline(results).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    main(parser.parse_args().split)
