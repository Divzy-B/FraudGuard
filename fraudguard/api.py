"""FastAPI application for fraud-risk inference and local explanations."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from fraudguard.data import make_synthetic_transactions
from fraudguard.model import load_artifact, save_artifact, score_record, shap_explanation, train_model


class Transaction(BaseModel):
    """A single synthetic-schema transaction; values must be pre-validated by callers."""

    model_config = ConfigDict(extra="forbid")
    transaction_id: str = Field(min_length=1, max_length=100, examples=["demo-001"])
    amount: Annotated[float, Field(gt=0, le=100_000)]
    hour: Annotated[int, Field(ge=0, le=23)]
    merchant_risk_score: Annotated[float, Field(ge=0, le=1)]
    transactions_last_24h: Annotated[int, Field(ge=0, le=500)]
    failed_transactions_last_24h: Annotated[int, Field(ge=0, le=100)]
    account_age_days: Annotated[float, Field(ge=0, le=50_000)]
    distance_from_home_km: Annotated[float, Field(ge=0, le=20_000)]
    international: bool
    card_present: bool
    avg_customer_amount: Annotated[float, Field(gt=0, le=100_000)]
    device_trust_score: Annotated[float, Field(ge=0, le=1)]
    weekend: bool


class Prediction(BaseModel):
    transaction_id: str | None
    fraud_probability: float
    fraud_decision: bool
    review_recommended: bool
    threshold: float
    model_version: str


class Contribution(BaseModel):
    feature: str
    value: float
    shap_value: float


class Explanation(Prediction):
    base_value: float
    top_contributions: list[Contribution]


def _default_artifact_path() -> Path:
    return Path(os.getenv("FRAUDGUARD_MODEL_PATH", "artifacts/fraudguard.joblib"))


def _ensure_artifact(path: Path) -> dict:
    if not path.exists():
        artifact, _ = train_model(make_synthetic_transactions(rows=6_000, seed=2026))
        save_artifact(artifact, path)
    return load_artifact(path)


def create_app(model_path: str | Path | None = None) -> FastAPI:
    path = Path(model_path) if model_path else _default_artifact_path()

    @lru_cache(maxsize=1)
    def artifact() -> dict:
        return _ensure_artifact(path)

    app = FastAPI(
        title="FraudGuard API",
        version="0.1.0",
        description="Synthetic-data fraud scoring demo. Not for production decisions.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        loaded = artifact()
        return {"status": "ok", "model_version": loaded["model_version"]}

    @app.get("/metadata")
    def metadata() -> dict:
        loaded = artifact()
        return {
            "model_version": loaded["model_version"],
            "trained_at": loaded["trained_at"],
            "threshold": loaded["threshold"],
            "metrics": loaded["metrics"],
            "feature_names": loaded["feature_names"],
        }

    @app.post("/predict", response_model=Prediction)
    def predict(transaction: Transaction) -> dict:
        return score_record(artifact(), transaction.model_dump())

    @app.post("/explain", response_model=Explanation)
    def explain(transaction: Transaction) -> dict:
        record = transaction.model_dump()
        result = score_record(artifact(), record)
        result.update(shap_explanation(artifact(), record))
        return result

    return app


app = create_app()
