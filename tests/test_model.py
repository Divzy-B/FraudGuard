from fraudguard.model import evaluate_predictions, score_record, threshold_at_capacity


def test_capacity_threshold_respects_review_limit():
    probability = [0.01, 0.02, 0.10, 0.40, 0.90]
    threshold = threshold_at_capacity(probability, capacity=0.40)
    assert sum(value >= threshold for value in probability) / len(probability) <= 0.40


def test_training_artifact_scores_a_transaction(trained_artifact, sample_transaction):
    result = score_record(trained_artifact, sample_transaction)
    assert 0 <= result["fraud_probability"] <= 1
    assert result["fraud_decision"] == (result["fraud_probability"] >= result["threshold"])


def test_imbalance_metrics_include_ranking_and_operating_values():
    metrics = evaluate_predictions([0, 0, 1, 1], [0.01, 0.2, 0.7, 0.8], 0.5)
    assert metrics["average_precision"] == 1.0
    assert metrics["review_rate"] == 0.5
