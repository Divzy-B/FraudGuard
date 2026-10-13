import pytest

from fraudguard.model import shap_explanation


def test_shap_explanation_returns_ranked_local_contributions(trained_artifact, sample_transaction):
    pytest.importorskip("shap")
    explanation = shap_explanation(trained_artifact, sample_transaction)
    assert isinstance(explanation["base_value"], float)
    assert 1 <= len(explanation["top_contributions"]) <= 5
    assert {"feature", "value", "shap_value"} <= explanation["top_contributions"][0].keys()
