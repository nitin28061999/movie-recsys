"""Content-based recommender: TF-IDF item features + a per-user taste profile."""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from recsys import config
from recsys.models.base import Recommender
from recsys.models.popularity import PopularityRecommender


class ContentBasedRecommender(Recommender):
    """Score items by cosine similarity to the user's profile.

    Profile = rating-weighted average of the TF-IDF vectors of items the user liked.
    - Works for brand-new items (they only need metadata, not ratings).
    - Users with no liked items fall back to popularity.
    - Ties (common with genre-only features) are broken by item popularity.
    """

    def __init__(
        self,
        movies: pd.DataFrame,
        tags: pd.DataFrame | None = None,
        threshold: float = config.RELEVANCE_THRESHOLD,
    ):
        super().__init__()
        self.movies = movies.reset_index(drop=True)
        self.tags = tags
        self.threshold = threshold
        self.name = "content-genres+tags" if tags is not None else "content-genres"
        self.fallback_ = PopularityRecommender(threshold)

    def _documents(self) -> list[str]:
        docs = (
            self.movies["genres"]
            .fillna("")
            .str.replace("(no genres listed)", "", regex=False)
            .str.replace("|", " ", regex=False)
        )
        if self.tags is not None:
            tag_text = (
                self.tags.assign(tag=self.tags["tag"].astype(str).str.lower())
                .groupby("movieId")["tag"]
                .agg(" ".join)
            )
            docs = docs + " " + self.movies["movieId"].map(tag_text).fillna("")
        return docs.tolist()

    def fit(self, train: pd.DataFrame) -> "ContentBasedRecommender":
        self._remember_seen(train)
        self.fallback_.fit(train)

        self.item_ids_ = self.movies["movieId"].to_numpy()
        self.item_index_ = {m: i for i, m in enumerate(self.item_ids_)}

        # token_pattern keeps hyphenated genres like "sci-fi" as a single token
        vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+", sublinear_tf=True)
        self.X_ = vectorizer.fit_transform(self._documents())  # rows are L2-normalised

        pop = train["movieId"].value_counts().reindex(self.item_ids_, fill_value=0)
        self.pop_ = pop.to_numpy()

        liked = train[train["rating"] >= self.threshold]
        liked = liked[liked["movieId"].isin(self.item_index_)]
        users = liked["userId"].unique()
        self.user_index_ = {u: i for i, u in enumerate(users)}
        rows = np.array([self.user_index_[u] for u in liked["userId"]])
        cols = np.array([self.item_index_[m] for m in liked["movieId"]])
        weights = liked["rating"].to_numpy(dtype=float)
        user_item = sparse.csr_matrix(
            (weights, (rows, cols)), shape=(len(users), len(self.item_ids_))
        )
        self.profiles_ = normalize(user_item @ self.X_)
        return self

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        u = self.user_index_.get(user_id)
        if u is None:
            return self.fallback_.recommend(user_id, k)
        scores = (self.profiles_[u] @ self.X_.T).toarray().ravel()
        seen = [self.item_index_[m] for m in self.seen_.get(user_id, ()) if m in self.item_index_]
        scores[seen] = -np.inf
        order = np.lexsort((-self.pop_, -scores))[:k]  # score desc, then popularity desc
        return [int(self.item_ids_[i]) for i in order if np.isfinite(scores[i])]
