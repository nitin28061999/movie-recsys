"""Small grid searches, scored on the VALIDATION split (never the test split).

    python -m recsys.tune als   # factors x regularization x alpha
    python -m recsys.tune knn   # n_neighbors x shrinkage
Results are printed and saved to reports/tuning_<name>.csv. Copy the winning values
into ALS_PARAMS / KNN_PARAMS in config.py.
"""
import argparse
import itertools
from functools import partial

import pandas as pd

from recsys import config
from recsys.data import download_movielens, load_movies, load_ratings, temporal_split
from recsys.evaluate import evaluate_model
from recsys.models import ALSRecommender, ItemKNNRecommender

LIKED = config.RELEVANCE_THRESHOLD
GRIDS = {
    "als": (
        partial(ALSRecommender, min_rating=LIKED),
        {"factors": [64, 128], "regularization": [1.0, 3.0], "alpha": [1.0, 3.0, 5.0, 10.0]},
    ),
    "knn": (
        partial(ItemKNNRecommender, min_rating=LIKED),
        {"n_neighbors": [100, 200, 400], "shrinkage": [0.0, 1.0, 3.0]},
    ),
}
SORT_BY = f"ndcg@{config.TOP_K}"


def tune(name: str) -> pd.DataFrame:
    factory, grid = GRIDS[name]
    download_movielens()
    train, val, _ = temporal_split(load_ratings())
    n_items = len(load_movies())

    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    rows = []
    for i, values in enumerate(combos, start=1):
        params = dict(zip(keys, values, strict=True))
        print(f"[{i}/{len(combos)}] {params}", flush=True)
        row = evaluate_model(factory(**params).fit(train), val, n_items)
        rows.append({**params, **{c: v for c, v in row.items() if c != "model"}})

    results = pd.DataFrame(rows).sort_values(SORT_BY, ascending=False).reset_index(drop=True)
    out = config.ROOT / "reports" / f"tuning_{name}.csv"
    out.parent.mkdir(exist_ok=True)
    results.to_csv(out, index=False)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("model", choices=list(GRIDS))
    table = tune(parser.parse_args().model)
    print(f"\nTop 5 by {SORT_BY} (validation split)")
    print(table.head(5).round(4).to_string(index=False))
