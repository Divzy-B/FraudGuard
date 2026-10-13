from __future__ import annotations

import pytest

from fraudguard.data import make_synthetic_transactions
from fraudguard.model import TrainingConfig, train_model


@pytest.fixture(scope="session")
def trained_artifact():
    data = make_synthetic_transactions(rows=1_500, seed=99)
    artifact, _ = train_model(data, TrainingConfig(n_estimators=25, review_capacity=0.08))
    return artifact


@pytest.fixture()
def sample_transaction():
    return {
        "transaction_id": "test-001",
        "amount": 820.0,
        "hour": 2,
        "merchant_risk_score": 0.87,
        "transactions_last_24h": 10,
        "failed_transactions_last_24h": 2,
        "account_age_days": 120.0,
        "distance_from_home_km": 250.0,
        "international": True,
        "card_present": False,
        "avg_customer_amount": 45.0,
        "device_trust_score": 0.18,
        "weekend": True,
    }
