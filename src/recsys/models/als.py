"""Matrix factorisation via implicit-feedback ALS (Hu, Koren & Volinsky, 2008)."""
import numpy as np
import pandas as pd
from scipy import sparse

from recsys import config
from recsys.models.base import Recommender
from recsys.models.popularity import PopularityRecommender


class ALSRecommender(Recommender):
    """Learn user and item vectors so that dot products predict who consumed what.

    Every observed (user, item) pair is a positive with confidence 1 + alpha; every
    unobserved pair is a weak negative with confidence 1. ALS alternates between solving
    all user vectors (items fixed) and all item vectors (users fixed); each solve is a
    small regularised least-squares problem with a closed-form answer.

    min_rating=None -> every interaction is a positive; min_rating=4.0 -> only liked items.
    Users with no usable history fall back to popularity.
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.1,
        alpha: float = 20.0,
        iterations: int = 15,
        min_rating: float | None = None,
        seed: int = config.SEED,
    ):
        super().__init__()
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.min_rating = min_rating
        self.seed = seed
        self.name = "als" if min_rating is None else "als-liked"
        self.fallback_ = PopularityRecommender()

    def fit(self, train: pd.DataFrame) -> "ALSRecommender":
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
        binary = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=shape)
        binary.data[:] = 1.0  # collapse duplicate (user, item) pairs
        by_user = (binary * self.alpha).tocsr()  # stores c - 1 for observed pairs
        by_item = by_user.T.tocsr()

        rng = np.random.default_rng(self.seed)
        self.item_factors_ = rng.normal(0.0, 0.01, (shape[1], self.factors))
        self.user_factors_ = np.zeros((shape[0], self.factors))
        for _ in range(self.iterations):
            self.user_factors_ = self._solve(by_user, self.item_factors_)
            self.item_factors_ = self._solve(by_item, self.user_factors_)
        return self

    def _solve(self, conf_minus_one: sparse.csr_matrix, fixed: np.ndarray) -> np.ndarray:
        """Closed-form update for every row, holding the other side's factors fixed."""
        gram = fixed.T @ fixed + self.regularization * np.eye(self.factors)
        out = np.zeros((conf_minus_one.shape[0], self.factors))
        indptr, indices, data = conf_minus_one.indptr, conf_minus_one.indices, conf_minus_one.data
        for row in range(conf_minus_one.shape[0]):
            lo, hi = indptr[row], indptr[row + 1]
            if lo == hi:
                continue  # nothing observed -> factors stay at zero
            y = fixed[indices[lo:hi]]
            c = data[lo:hi]
            a = gram + (y.T * c) @ y  # YtY + sum (c_j - 1) y_j y_j^T + reg * I
            b = (1.0 + c) @ y  # sum c_j y_j (preference is 1 for observed pairs)
            out[row] = np.linalg.solve(a, b)
        return out

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        u = self.user_index_.get(user_id)
        if u is None:
            return self.fallback_.recommend(user_id, k)
        scores = self.item_factors_ @ self.user_factors_[u]
        seen = [self.item_index_[m] for m in self.seen_.get(user_id, ()) if m in self.item_index_]
        scores[seen] = -np.inf
        order = np.lexsort((-self.pop_, -scores))[:k]  # score desc, then popularity desc
        return [int(self.item_ids_[i]) for i in order if np.isfinite(scores[i])]
