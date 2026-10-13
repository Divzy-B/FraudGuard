# FraudGuard

FraudGuard is a GitHub-ready fraud-risk machine-learning portfolio project. It
creates reproducible **synthetic** transactions, trains an imbalance-aware
XGBoost classifier, chooses an alert threshold from review capacity, exposes
FastAPI scoring and local TreeSHAP explanations, and ships tests, Docker, and
GitHub Actions CI.

> **Safety and scope:** this is an educational demo, not a financial decision
> system. It uses no real customer data and must not be used to approve,
> decline, block, or investigate people or transactions.

## What is included

- Deterministic, seedable synthetic fraud data; no downloads or PII required.
- One shared feature transformation for training and serving.
- XGBoost with training-fold `scale_pos_weight` for class imbalance.
- Ranking and operating-point metrics: ROC-AUC, average precision, Brier score,
  precision, recall, F1, review rate, and prevalence.
- Review-capacity threshold selection rather than a fixed 0.5 cutoff.
- Persisted artifact with estimator, feature contract, parameters, threshold,
  metrics, timestamp, and model version.
- FastAPI `/health`, `/metadata`, `/predict`, and `/explain` endpoints.
- Local TreeSHAP explanations, Docker/Compose, pytest tests, and CI.

## Repository map

```text
fraudguard/       implementation package
tests/            unit and API tests
docs/             architecture, model card, operations notes
.github/          GitHub Actions workflow
Dockerfile        non-root FastAPI container
NOTICE.md         reference attribution and license note
```

## Quick start

Requires Python 3.10–3.12.

```bash
git clone <your-fork-url> fraudguard
cd fraudguard
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
python -m fraudguard.train --rows 12000 --seed 2026 --output artifacts/fraudguard.joblib
python -m uvicorn fraudguard.api:app --reload
```

Then open `http://127.0.0.1:8000/docs`. The model path defaults to
`artifacts/fraudguard.joblib`; set `FRAUDGUARD_MODEL_PATH` to use another
artifact.

## API example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "demo-001",
    "amount": 820.0,
    "hour": 2,
    "merchant_risk_score": 0.87,
    "transactions_last_24h": 10,
    "failed_transactions_last_24h": 2,
    "account_age_days": 120,
    "distance_from_home_km": 250,
    "international": true,
    "card_present": false,
    "avg_customer_amount": 45,
    "device_trust_score": 0.18,
    "weekend": true
  }'
```

`/predict` returns the probability, the artifact threshold, and a review
recommendation. POST the same body to `/explain` for local TreeSHAP values. A
positive SHAP value pushes the model's raw prediction higher; it does not prove
that a feature caused fraud.

## Evaluation approach

Fraud detection is a rare-event problem. Accuracy can appear strong while a
model misses nearly every fraud case, so FraudGuard emphasizes **average
precision** and recall/precision at the chosen review capacity. Training:

1. makes a stratified train/test split;
2. derives positive weighting only from the training fold;
3. trains XGBoost;
4. selects a threshold at the requested maximum review rate (default 5%);
5. reports both score quality and the selected operating point.

The data generator and trainer use fixed seeds by default. The CLI prints the
actual metrics and saves them in the artifact; do not present them as evidence
of real-world performance.

## Docker

```bash
docker compose up --build
curl http://localhost:8000/health
```

The image builds a synthetic demonstration artifact and runs the API as an
unprivileged user. `docker-compose.yml` adds a read-only filesystem and a tmpfs
mount. Production deployments need an external artifact registry, secret
management, authentication, observability, governance, and human review.

## CI/CD

GitHub Actions runs linting, tests, a fresh training smoke test on Python 3.10,
3.11, and 3.12, uploads the 3.12 CI model temporarily, and checks that the
Docker image builds. The workflow has read-only repository permissions and is
ready to be extended with registry publishing in a protected release workflow.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Model card](docs/MODEL_CARD.md)
- [Operations guide](docs/OPERATIONS.md)
- [Reference attribution](NOTICE.md)

## License and reference attribution

FraudGuard is MIT licensed; see [LICENSE](LICENSE). Its goals were informed by
the MIT-licensed [banking-fraud-risk-ml](https://github.com/Speaking-Data-NK/banking-fraud-risk-ml)
project by Speaking-Data-NK. FraudGuard is separately implemented and includes
no copied code, data, model artifacts, or documentation from that repository.
See [NOTICE.md](NOTICE.md) for the full attribution and combination guidance.
