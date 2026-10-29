"""Common interface so evaluate.py can treat every model the same way."""
from abc import ABC, abstractmethod

import pandas as pd


class Recommender(ABC):
    name: str = "recommender"

    def __init__(self):
        self.seen_: dict[int, set[int]] = {}

    def _remember_seen(self, train: pd.DataFrame) -> None:
        self.seen_ = train.groupby("userId")["movieId"].agg(set).to_dict()

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> "Recommender":
        """Learn from train (columns: userId, movieId, rating, timestamp)."""

    @abstractmethod
    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Return up to k movie ids, best first, excluding items seen in train."""
