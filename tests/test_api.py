import pytest
from pydantic import ValidationError

from fraudguard.api import Transaction, create_app
from fraudguard.model import save_artifact


def _endpoint(app, path):
    return next(route.endpoint for route in app.routes if route.path == path)


def test_health_predict_and_metadata(tmp_path, trained_artifact, sample_transaction):
    model_path = save_artifact(trained_artifact, tmp_path / "model.joblib")
    app = create_app(model_path)
    health = _endpoint(app, "/health")()
    response = _endpoint(app, "/predict")(Transaction(**sample_transaction))
    metadata = _endpoint(app, "/metadata")()
    assert health["status"] == "ok"
    assert response["transaction_id"] == "test-001"
    assert "average_precision" in metadata["metrics"]


def test_validation_rejects_invalid_hour(sample_transaction):
    sample_transaction["hour"] = 24
    with pytest.raises(ValidationError):
        Transaction(**sample_transaction)
