"""
Unit tests for semantic entropy clustering, uncertainty metrics, and martingale drift.
"""

import pytest
from src.models.semantic_entropy import (
    cluster_equivalent_answers,
    compute_semantic_entropy,
    compute_martingale_drift
)


def test_cluster_equivalent_answers():
    # 3 answers that mean the same number, 2 that mean a different number
    cands = ["42", "42.0", "\\boxed{42}", "15", "15.0"]
    clusters = cluster_equivalent_answers(cands)
    assert len(clusters) == 2
    assert len(clusters[0]) == 3
    assert len(clusters[1]) == 2


def test_semantic_entropy_values():
    # All agree: entropy should be 0.0
    cands_unanimous = ["10", "10", "10", "10"]
    ent, maj_ans, maj_ratio = compute_semantic_entropy(cands_unanimous)
    assert ent == 0.0
    assert maj_ans == "10"
    assert maj_ratio == 1.0

    # 50/50 split across 2 clusters: entropy should be 1.0 bit
    cands_split = ["10", "10", "20", "20"]
    ent, maj_ans, maj_ratio = compute_semantic_entropy(cands_split)
    assert pytest.approx(ent, 0.01) == 1.0
    assert maj_ratio == 0.5


def test_martingale_drift():
    # Entropy decreasing over time (uncertainty resolving)
    entropies = [1.5, 0.8, 0.0]
    drifts = compute_martingale_drift(entropies)
    assert len(drifts) == 3
    assert drifts[0] == 0.0
    assert pytest.approx(drifts[1], 0.01) == -0.7
    assert pytest.approx(drifts[2], 0.01) == -0.8
