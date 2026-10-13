"""Shared feature definitions and deterministic feature preparation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

RAW_FEATURES = [
    "amount",
    "hour",
    "merchant_risk_score",
    "transactions_last_24h",
    "failed_transactions_last_24h",
    "account_age_days",
    "distance_from_home_km",
    "international",
    "card_present",
    "avg_customer_amount",
    "device_trust_score",
    "weekend",
]

MODEL_FEATURES = RAW_FEATURES + [
    "amount_to_average_ratio",
    "is_unusual_hour",
    "velocity_pressure",
]


def prepare_features(records: pd.DataFrame | Mapping[str, object]) -> pd.DataFrame:
    """Create model features from raw transaction fields without label leakage."""
    frame = pd.DataFrame([records]) if isinstance(records, Mapping) else records.copy()
    missing = set(RAW_FEATURES).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required feature(s): {', '.join(sorted(missing))}")

    for name in ("international", "card_present", "weekend"):
        frame[name] = frame[name].astype(int)

    average = frame["avg_customer_amount"].clip(lower=1.0)
    frame["amount_to_average_ratio"] = frame["amount"] / average
    frame["is_unusual_hour"] = frame["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    frame["velocity_pressure"] = (
        frame["transactions_last_24h"] + 2 * frame["failed_transactions_last_24h"]
    )
    return frame.loc[:, MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
