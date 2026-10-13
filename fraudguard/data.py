"""Reproducible synthetic transaction data for demos and automated tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudguard.schema import prepare_features


def make_synthetic_transactions(rows: int = 12_000, seed: int = 2026) -> pd.DataFrame:
    """Generate intentionally fictional, reproducible transactions and fraud labels.

    Labels come from a stochastic risk process. No real customers, accounts, or
    transactions are used or represented by these records.
    """
    if rows < 100:
        raise ValueError("rows must be at least 100 to preserve a meaningful class mix")

    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, rows)
    average_amount = rng.lognormal(mean=3.8, sigma=0.55, size=rows).clip(8, 2_000)
    amount = (average_amount * rng.lognormal(mean=0.0, sigma=0.8, size=rows)).clip(1, 20_000)
    merchant_risk = rng.beta(2.0, 6.0, rows)
    tx_count = rng.poisson(2.6, rows)
    failed_count = rng.poisson(0.22, rows)
    account_age = rng.gamma(shape=2.3, scale=360.0, size=rows).clip(1, 8_000)
    distance = rng.gamma(shape=1.7, scale=18, size=rows).clip(0, 2_000)
    international = rng.binomial(1, 0.085, rows)
    card_present = rng.binomial(1, 0.58, rows)
    device_trust = rng.beta(5.0, 2.2, rows)
    weekend = rng.binomial(1, 2 / 7, rows)

    raw = pd.DataFrame(
        {
            "transaction_id": [f"syn-{seed}-{i:06d}" for i in range(rows)],
            "amount": amount,
            "hour": hour,
            "merchant_risk_score": merchant_risk,
            "transactions_last_24h": tx_count,
            "failed_transactions_last_24h": failed_count,
            "account_age_days": account_age,
            "distance_from_home_km": distance,
            "international": international,
            "card_present": card_present,
            "avg_customer_amount": average_amount,
            "device_trust_score": device_trust,
            "weekend": weekend,
        }
    )
    engineered = prepare_features(raw)
    # The intercept yields a rare positive class; the remaining terms model a
    # deliberately simplified fraud mechanism for training demonstrations.
    log_odds = (
        -5.1
        + 3.0 * merchant_risk
        + 1.0 * np.log1p(engineered["amount_to_average_ratio"])
        + 1.5 * engineered["is_unusual_hour"]
        + 0.28 * engineered["velocity_pressure"]
        + 1.5 * international
        - 1.2 * card_present
        - 1.6 * device_trust
        - 0.0010 * account_age
        + 0.009 * np.minimum(distance, 250)
    )
    probability = 1.0 / (1.0 + np.exp(-log_odds))
    raw["is_fraud"] = rng.binomial(1, probability)
    return raw
