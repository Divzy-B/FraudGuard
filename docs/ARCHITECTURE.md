# Architecture

```text
Synthetic generator --> deterministic feature preparation --> XGBoost training
                                              |                    |
                                              |                    v
Client --> FastAPI validation --> /predict  joblib artifact + threshold
                            \-> /explain  --> TreeSHAP contributions
```

The generated data is intentionally fictional. `prepare_features` is the
single feature contract used by both training and serving, preventing training
and inference from silently diverging.

Training holds out a stratified test split. XGBoost receives `scale_pos_weight`
computed from its training fold. The alert threshold is selected from held-out
scores at a declared review capacity instead of using the default 0.5. The
persisted artifact contains the feature order, threshold, parameters, metrics,
timestamp, and model version alongside the estimator.

The API is deliberately stateless. It validates bounds, loads one local model
artifact, returns a score and an action recommendation, and can return local
TreeSHAP contributions. For a real deployment, add authenticated ingress,
feature-store parity checks, signed model registry artifacts, audit trails,
monitoring, drift analysis, and human decision review.
