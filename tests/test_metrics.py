import math

import pytest

from recsys import metrics

REC = [1, 2, 3, 4]
REL = {2, 4}


def test_precision():
    assert metrics.precision_at_k(REC, REL, 4) == 0.5
    assert metrics.precision_at_k(REC, REL, 1) == 0.0


def test_recall():
    assert metrics.recall_at_k(REC, REL, 4) == 1.0
    assert metrics.recall_at_k(REC, REL, 2) == 0.5
    assert metrics.recall_at_k(REC, set(), 4) == 0.0


def test_ndcg_hand_computed():
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert metrics.ndcg_at_k(REC, REL, 4) == pytest.approx(dcg / ideal)
    assert metrics.ndcg_at_k([2, 4, 1, 3], REL, 4) == pytest.approx(1.0)


def test_coverage():
    assert metrics.catalog_coverage([[1, 2], [2, 3]], n_items=6) == 0.5


def test_invalid_k():
    with pytest.raises(ValueError):
        metrics.precision_at_k(REC, REL, 0)
