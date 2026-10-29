import pandas as pd
import pytest

from recsys.models import HybridRecommender, Recommender


class Stub(Recommender):
    def __init__(self, recs, name="stub"):
        super().__init__()
        self.recs, self.name, self.fit_calls = recs, name, 0

    def fit(self, train):
        self.fit_calls += 1
        return self

    def recommend(self, user_id, k=10):
        return self.recs[:k]


def test_items_both_models_agree_on_rise_to_the_top():
    a, b = Stub([1, 2, 3]), Stub([3, 4, 5])
    assert HybridRecommender([(a, 1.0), (b, 1.0)]).recommend(1, k=1) == [3]


def test_zero_weight_component_is_ignored_for_ordering():
    a, b = Stub([1, 2, 3]), Stub([3, 4, 5])
    assert HybridRecommender([(a, 1.0), (b, 0.0)]).recommend(1, k=3) == [1, 2, 3]


def test_weights_shift_the_blend():
    a, b = Stub([1, 2]), Stub([9, 8])
    assert HybridRecommender([(a, 1.0), (b, 3.0)]).recommend(1, k=1) == [9]
    assert HybridRecommender([(a, 3.0), (b, 1.0)]).recommend(1, k=1) == [1]


def test_prefit_skips_refitting_components():
    a, b = Stub([1]), Stub([2])
    HybridRecommender([(a, 1.0), (b, 1.0)], prefit=True).fit(pd.DataFrame())
    assert a.fit_calls == b.fit_calls == 0
    HybridRecommender([(a, 1.0), (b, 1.0)]).fit(pd.DataFrame())
    assert a.fit_calls == b.fit_calls == 1


def test_returns_at_most_k_unique_items():
    a, b = Stub([1, 2, 3, 4]), Stub([4, 3, 2, 1])
    recs = HybridRecommender([(a, 1.0), (b, 1.0)]).recommend(1, k=3)
    assert len(recs) == 3 and len(set(recs)) == 3


def test_empty_when_no_component_has_recommendations():
    assert HybridRecommender([(Stub([]), 1.0)]).recommend(1, k=5) == []


@pytest.mark.parametrize("depth", [1, 2])
def test_depth_limits_candidates(depth):
    a = Stub([1, 2, 3])
    assert HybridRecommender([(a, 1.0)], depth=depth).recommend(1, k=10) == [1, 2][:depth]
