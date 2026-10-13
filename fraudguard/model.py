"""Training, threshold selection, metrics, persistence, and SHAP explanations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from fraudguard.schema import MODEL_FEATURES, prepare_features


@dataclass(frozen=True)
class TrainingConfig:
    test_size: float = 0.25
    random_state: int = 2026
    review_capacity: float = 0.05
    n_estimators: int = 160
    max_depth: int = 4
    learning_rate: float = 0.06


def threshold_at_capacity(probability: np.ndarray, capacity: float) -> float:
    """Return an operating threshold that flags no more than the review capacity."""
    if not 0 < capacity <= 1:
        raise ValueError("capacity must be in (0, 1]")
    return float(np.quantile(np.asarray(probability), 1 - capacity, method="higher"))


def evaluate_predictions(y_true: pd.Series | np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    """Report rare-event ranking and operating-point metrics (not accuracy)."""
    truth = np.asarray(y_true, dtype=int)
    proba = np.asarray(probability, dtype=float)
    prediction = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(truth, proba)),
        "average_precision": float(average_precision_score(truth, proba)),
        "brier_score": float(brier_score_loss(truth, proba)),
        "precision_at_threshold": float(precision_score(truth, prediction, zero_division=0)),
        "recall_at_threshold": float(recall_score(truth, prediction, zero_division=0)),
        "f1_at_threshold": float(f1_score(truth, prediction, zero_division=0)),
        "review_rate": float(prediction.mean()),
        "fraud_prevalence": float(truth.mean()),
    }


def train_model(data: pd.DataFrame, config: TrainingConfig | None = None) -> tuple[dict[str, Any], dict[str, float]]:
    """Train an imbalance-weighted XGBoost model and choose an alert threshold."""
    config = config or TrainingConfig()
    if "is_fraud" not in data:
        raise ValueError("Training data must contain is_fraud")
    features = prepare_features(data)
    target = data["is_fraud"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=config.test_size, random_state=config.random_state, stratify=target
    )
    positives = int(y_train.sum())
    negatives = len(y_train) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Training split requires both fraud and non-fraud labels")
    model = XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_lambda=1.5,
        scale_pos_weight=negatives / positives,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=config.random_state,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    threshold = threshold_at_capacity(probability, config.review_capacity)
    metrics = evaluate_predictions(y_test, probability, threshold)
    artifact: dict[str, Any] = {
        "model": model,
        "feature_names": MODEL_FEATURES,
        "threshold": threshold,
        "config": asdict(config),
        "metrics": metrics,
        "trained_at": datetime.now(UTC).isoformat(),
        "model_version": "fraudguard-xgb-v1",
    }
    return artifact, metrics


def save_artifact(artifact: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination)
    return destination


def load_artifact(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def score_record(artifact: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    prepared = prepare_features(record)
    probability = float(artifact["model"].predict_proba(prepared)[0, 1])
    threshold = float(artifact["threshold"])
    return {
        "transaction_id": record.get("transaction_id"),
        "fraud_probability": probability,
        "fraud_decision": probability >= threshold,
        "review_recommended": probability >= threshold,
        "threshold": threshold,
        "model_version": artifact["model_version"],
    }


def shap_explanation(artifact: dict[str, Any], record: dict[str, Any], max_features: int = 5) -> dict[str, Any]:
    """Return local TreeSHAP feature effects for one prediction."""
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - dependency is declared, but message is useful
        raise RuntimeError("Install the optional SHAP dependency to use /explain") from exc

    prepared = prepare_features(record)
    explanation = shap.TreeExplainer(artifact["model"])(prepared)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, -1]
    contribution = values[0]
    order = np.argsort(np.abs(contribution))[::-1][:max_features]
    contributions = [
        {
            "feature": str(prepared.columns[index]),
            "value": float(prepared.iloc[0, index]),
            "shap_value": float(contribution[index]),
        }
        for index in order
    ]
    base_values = np.asarray(explanation.base_values).reshape(-1)
    return {"base_value": float(base_values[0]), "top_contributions": contributions}
