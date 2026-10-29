"""Ranking metrics. `recommended` is an ordered list of item ids; `relevant` is a set."""
import math
from collections.abc import Iterable


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    _check_k(k)
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    _check_k(k)
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Binary-relevance NDCG@k."""
    _check_k(k)
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def catalog_coverage(all_recommendations: Iterable[list], n_items: int) -> float:
    """Fraction of the catalog that appears in at least one user's recommendations."""
    seen = set()
    for recs in all_recommendations:
        seen.update(recs)
    return len(seen) / n_items if n_items else 0.0
