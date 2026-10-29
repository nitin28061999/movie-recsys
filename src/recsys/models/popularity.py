import pandas as pd

from recsys import config
from recsys.models.base import Recommender


class PopularityRecommender(Recommender):
    """Recommend the most-liked items (rating >= threshold) the user hasn't seen."""

    name = "popularity"

    def __init__(self, threshold: float = config.RELEVANCE_THRESHOLD):
        super().__init__()
        self.threshold = threshold
        self.ranked_: list[int] = []

    def fit(self, train: pd.DataFrame) -> "PopularityRecommender":
        self._remember_seen(train)
        liked = train[train["rating"] >= self.threshold]
        counts = liked["movieId"].value_counts(sort=True)
        # stable tie-break by item id so results are deterministic
        order = counts.reset_index()
        order.columns = ["movieId", "n"]
        order = order.sort_values(["n", "movieId"], ascending=[False, True])
        self.ranked_ = order["movieId"].tolist()
        return self

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        seen = self.seen_.get(user_id, set())
        out = []
        for item in self.ranked_:
            if item not in seen:
                out.append(item)
                if len(out) == k:
                    break
        return out
