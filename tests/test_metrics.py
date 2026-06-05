from rag_system.metrics import average_precision, hit_rate_at_k, recall_at_k, reciprocal_rank


def test_retrieval_metrics_with_relevant_hits():
    retrieved = ["C1", "C2", "C3", "C4"]
    relevant = {"C2", "C4"}

    assert recall_at_k(retrieved, relevant, 3) == 0.5
    assert hit_rate_at_k(retrieved, relevant, 3) == 1.0
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert average_precision(retrieved, relevant) == ((1 / 2) + (2 / 4)) / 2


def test_retrieval_metrics_without_relevant_hits():
    retrieved = ["C1", "C2"]
    relevant = {"C9"}

    assert recall_at_k(retrieved, relevant, 2) == 0.0
    assert hit_rate_at_k(retrieved, relevant, 2) == 0.0
    assert reciprocal_rank(retrieved, relevant) == 0.0
    assert average_precision(retrieved, relevant) == 0.0

