"""Command-line entry point for model training."""

from __future__ import annotations

import argparse
import json

from fraudguard.data import make_synthetic_transactions
from fraudguard.model import TrainingConfig, save_artifact, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FraudGuard on synthetic transactions")
    parser.add_argument("--rows", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--review-capacity", type=float, default=0.05)
    parser.add_argument("--output", default="artifacts/fraudguard.joblib")
    args = parser.parse_args()
    data = make_synthetic_transactions(rows=args.rows, seed=args.seed)
    artifact, metrics = train_model(data, TrainingConfig(review_capacity=args.review_capacity))
    path = save_artifact(artifact, args.output)
    print(json.dumps({"artifact": str(path), "metrics": metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
