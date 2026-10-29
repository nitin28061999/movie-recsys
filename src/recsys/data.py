"""Data download, loading, and time-based splitting."""
import urllib.request
import zipfile

import pandas as pd

from recsys import config


def download_movielens(force: bool = False):
    """Download and unzip MovieLens into data/raw. Returns the dataset directory."""
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    target = config.DATA_RAW / config.MOVIELENS_DIRNAME
    if target.exists() and not force:
        return target
    zip_path = config.DATA_RAW / "movielens.zip"
    urllib.request.urlretrieve(config.MOVIELENS_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(config.DATA_RAW)
    zip_path.unlink()
    return target


def load_ratings(path=None) -> pd.DataFrame:
    path = path or (config.DATA_RAW / config.MOVIELENS_DIRNAME)
    return pd.read_csv(path / "ratings.csv")  # userId, movieId, rating, timestamp


def load_movies(path=None) -> pd.DataFrame:
    path = path or (config.DATA_RAW / config.MOVIELENS_DIRNAME)
    return pd.read_csv(path / "movies.csv")  # movieId, title, genres


def load_tags(path=None) -> pd.DataFrame:
    path = path or (config.DATA_RAW / config.MOVIELENS_DIRNAME)
    return pd.read_csv(path / "tags.csv")  # userId, movieId, tag, timestamp


def temporal_split(
    ratings: pd.DataFrame,
    val_frac: float = config.VAL_FRAC,
    test_frac: float = config.TEST_FRAC,
    min_interactions: int = config.MIN_USER_INTERACTIONS,
):
    """Per-user chronological split: oldest -> train, middle -> val, newest -> test.

    Users with fewer than `min_interactions` ratings go entirely to train.
    Returns (train, val, test) DataFrames.
    """
    df = ratings.sort_values(["userId", "timestamp"], kind="stable").copy()
    grp = df.groupby("userId")["userId"]
    n = grp.transform("size")
    pos = (grp.cumcount() + 1) / n  # relative position within the user's history

    eligible = n >= min_interactions
    is_test = eligible & (pos > 1 - test_frac)
    is_val = eligible & (pos > 1 - test_frac - val_frac) & ~is_test

    return df[~(is_test | is_val)], df[is_val], df[is_test]
