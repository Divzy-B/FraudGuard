# FraudGuard model card

## Intended use

This model is a learning and portfolio artifact. It demonstrates a reproducible
rare-event classification workflow with fully synthetic data. It must not be
used to approve, decline, block, or investigate real people or transactions.

## Inputs and output

Inputs describe a hypothetical transaction: amount, timing, merchant risk,
velocity, account age, distance, transaction modality, a synthetic device
trust score, and weekend indicator. Derived features are amount-to-average
ratio, unusual-hour status, and velocity pressure. The output is a model score
and a review recommendation based on a configurable capacity threshold.

## Evaluation

Because fraud is rare, accuracy is intentionally excluded from primary
evaluation. The training report records ROC-AUC, average precision (PR-AUC),
Brier score, precision, recall, F1, review rate, and prevalence. Run training
with a fixed seed to reproduce a result; report the exact artifact metrics,
not an assumed benchmark.

## Risks and safeguards

Synthetic results do not demonstrate real-world validity, calibration,
fairness, security, or regulatory compliance. Model explanations describe the
model, not causality. Keep people in the loop, test for disparate impact,
protect data, monitor drift, and use formal governance before any real use.
