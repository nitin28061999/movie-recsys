from recsys.models import PopularityRecommender


def test_excludes_seen_items(ratings):
    model = PopularityRecommender().fit(ratings)
    recs = model.recommend(1, k=5)
    assert len(recs) == 5
    assert not set(recs) & model.seen_[1]


def test_unknown_user_gets_global_top(ratings):
    model = PopularityRecommender().fit(ratings)
    assert model.recommend(9999, k=3) == model.ranked_[:3]
