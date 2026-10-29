"""Item-based collaborative filtering (item-item kNN on implicit feedback)."""
import numpy as np
import pandas as pd
from scipy import sparse

from recsys.models.base import Recommender
from recsys.models.popularity import PopularityRecommender


class ItemKNNRecommender(Recommender):
    """Recommend items that are co-consumed with what the user already has.

    similarity(i, j) = cosine(i, j) on binary user-item data, damped by a shrinkage
    term co / (co + shrinkage) so pairs seen together only once or twice don't dominate.
    Each item keeps only its `n_neighbors` strongest neighbours. A candidate item's
    score is the sum of its similarities to the items in the user's history.

    min_rating=None  -> every interaction counts as a positive signal ("watched")
    min_rating=4.0   -> only liked items count
    Users with no usable history fall back to popularity. Items that never appear in
    train cannot be recommended (that's what the content model is for).
    """

    def __init__(
        self,
        n_neighbors: int = 50,
        shrinkage: float = 10.0,
        min_rating: float | None = None,
    ):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.shrinkage = shrinkage
        self.min_rating = min_rating
        self.name = "itemknn" if min_rating is None else "itemknn-liked"
        self.fallback_ = PopularityRecommender()

    def fit(self, train: pd.DataFrame) -> "ItemKNNRecommender":
        self._remember_seen(train)
        self.fallback_.fit(train)

        used = train if self.min_rating is None else train[train["rating"] >= self.min_rating]
        self.item_ids_ = np.sort(train["movieId"].unique())
        self.item_index_ = {m: i for i, m in enumerate(self.item_ids_)}
        self.pop_ = train["movieId"].value_counts().reindex(self.item_ids_).to_numpy()

        users = used["userId"].unique()
        self.user_index_ = {u: i for i, u in enumerate(users)}
        rows = used["userId"].map(self.user_index_).to_numpy()
        cols = used["movieId"].map(self.item_index_).to_numpy()
        shape = (len(users), len(self.item_ids_))
        history = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=shape)
        history.data[:] = 1.0  # collapse any duplicate (user, item) pairs to binary
        self.history_ = history
        self.sim_ = self._item_similarity(history)
        return self

    def _item_similarity(self, history: sparse.csr_matrix) -> sparse.csr_matrix:
        co = (history.T @ history).tocoo()  # co[i, j] = users who consumed both
        counts = np.asarray(history.sum(axis=0)).ravel()
        off_diag = co.row != co.col
        row, col, c = co.row[off_diag], co.col[off_diag], co.data[off_diag]
        cosine = c / np.sqrt(counts[row] * counts[col])
        sim = cosine * c / (c + self.shrinkage)
        n_items = history.shape[1]
        return self._keep_top_neighbors(sparse.csr_matrix((sim, (row, col)), shape=(n_items,) * 2))

    def _keep_top_neighbors(self, sim: sparse.csr_matrix) -> sparse.csr_matrix:
        n = self.n_neighbors
        keep = np.zeros(len(sim.data), dtype=bool)
        for i in range(sim.shape[0]):
            lo, hi = sim.indptr[i], sim.indptr[i + 1]
            if hi - lo <= n:
                keep[lo:hi] = True
            else:
                keep[lo + np.argpartition(sim.data[lo:hi], -n)[-n:]] = True
        rows = np.repeat(np.arange(sim.shape[0]), np.diff(sim.indptr))
        return sparse.csr_matrix(
            (sim.data[keep], (rows[keep], sim.indices[keep])), shape=sim.shape
        )

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        u = self.user_index_.get(user_id)
        if u is None:
            return self.fallback_.recommend(user_id, k)
        scores = (self.history_[u] @ self.sim_).toarray().ravel()
        seen = [self.item_index_[m] for m in self.seen_.get(user_id, ()) if m in self.item_index_]
        scores[seen] = -np.inf
        order = np.lexsort((-self.pop_, -scores))[:k]  # score desc, then popularity desc
        return [int(self.item_ids_[i]) for i in order if np.isfinite(scores[i])]
