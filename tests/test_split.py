from recsys.data import temporal_split


def test_no_overlap_and_nothing_lost(ratings):
    train, val, test = temporal_split(ratings)
    assert len(train) + len(val) + len(test) == len(ratings)
    keys = lambda d: set(zip(d.userId, d.movieId, d.timestamp))
    assert not (keys(train) & keys(val)) and not (keys(train) & keys(test))
    assert not (keys(val) & keys(test))


def test_no_temporal_leakage(ratings):
    """For every user, all train events happen no later than val, and val no later than test."""
    train, val, test = temporal_split(ratings)
    for user in test.userId.unique():
        tr = train[train.userId == user].timestamp.max()
        assert tr <= val[val.userId == user].timestamp.min()
        assert val[val.userId == user].timestamp.max() <= test[test.userId == user].timestamp.min()


def test_light_users_stay_in_train(ratings):
    train, val, test = temporal_split(ratings, min_interactions=5)
    assert (train.userId == 4).sum() == 3
    assert 4 not in set(val.userId) | set(test.userId)
