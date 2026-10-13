import pandas as pd
import pytest

from fraudguard.data import make_synthetic_transactions
from fraudguard.schema import MODEL_FEATURES, prepare_features


def test_generator_is_reproducible_and_imbalanced():
    left = make_synthetic_transactions(rows=500, seed=4)
    right = make_synthetic_transactions(rows=500, seed=4)
    pd.testing.assert_frame_equal(left, right)
    assert 0.001 < left["is_fraud"].mean() < 0.20


def test_feature_engineering_has_expected_contract():
    data = make_synthetic_transactions(rows=100, seed=8)
    features = prepare_features(data)
    assert list(features.columns) == MODEL_FEATURES
    assert (features["amount_to_average_ratio"] > 0).all()


def test_feature_engineering_rejects_missing_input():
    with pytest.raises(ValueError, match="Missing required"):
        prepare_features({"amount": 10})
