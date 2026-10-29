"""Hybrid recommender: blend several models by reciprocal-rank fusion."""
import pandas as pd

from recsys.models.base import Recommender


class HybridRecommender(Recommender):
    """Combine ranked lists from several recommenders.

    Each component contributes weight / (rrf_k + rank) to every item in its top `depth`
    list; items are ordered by the summed score. Working on ranks instead of raw scores
    sidesteps the fact that kNN similarity sums and ALS dot products are on different
    scales. Items several models agree on rise to the top. Because every component already
    excludes seen items and falls back to popularity for cold users, so does the blend.

    prefit=True reuses components that were already fitted (saves refitting in evaluate).
    """

    def __init__(
        self,
        components: list[tuple[Recommender, float]],
        depth: int = 100,
        rrf_k: int = 60,
        prefit: bool = False,
        name: str = "hybrid",
    ):
        super().__init__()
        self.components = components
        self.depth = depth
        self.rrf_k = rrf_k
        self.prefit = prefit
        self.name = name

    def fit(self, train: pd.DataFrame) -> "HybridRecommender":
        if not self.prefit:
            for model, _ in self.components:
                model.fit(train)
        return self

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        fused: dict[int, float] = {}
        best_rank: dict[int, int] = {}
        for model, weight in self.components:
            for rank, item in enumerate(model.recommend(user_id, self.depth), start=1):
                fused[item] = fused.get(item, 0.0) + weight / (self.rrf_k + rank)
                best_rank[item] = min(best_rank.get(item, rank), rank)
        ordered = sorted(fused, key=lambda i: (-fused[i], best_rank[i], i))
        return ordered[:k]
