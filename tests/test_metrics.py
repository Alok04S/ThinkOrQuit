"""
Unit tests for evaluation metrics, RC-AUC calculation, and statistical hypothesis testing.
"""

import numpy as np
import pytest
from src.evaluation.metrics import compute_policy_metrics
from src.evaluation.risk_coverage import compute_risk_coverage_curve, compute_rc_auc
from src.evaluation.statistical_tests import paired_bootstrap_difference, holm_bonferroni_correction
from src.models.inference import ReasoningCheckpoint, ReasoningTrace
from src.policies.base import PolicyAction, PolicyDecision


def test_compute_policy_metrics():
    trace1 = ReasoningTrace(
        id="t1", question="q1", target="1", normalized_target="1", is_unanswerable=False,
        checkpoints=[ReasoningCheckpoint(1, 200, ["1"], "1", 1.0, 0.0, 0.0, 1.0)],
        total_tokens_full=200, final_ground_truth_correct=True
    )
    trace2 = ReasoningTrace(
        id="t2", question="q2", target="2", normalized_target="2", is_unanswerable=False,
        checkpoints=[ReasoningCheckpoint(1, 200, ["3"], "3", 0.6, 0.0, 0.0, 0.6)],
        total_tokens_full=200, final_ground_truth_correct=False
    )
    trace3 = ReasoningTrace(
        id="t3", question="q3", target="[ABSTAIN]", normalized_target="[ABSTAIN]", is_unanswerable=True,
        checkpoints=[ReasoningCheckpoint(1, 200, ["9"], "9", 0.4, 1.0, 0.0, 0.4)],
        total_tokens_full=200, final_ground_truth_correct=False
    )

    decisions = [
        PolicyDecision(PolicyAction.ANSWER, "1", 1, 200, 1.0),
        PolicyDecision(PolicyAction.ANSWER, "3", 1, 200, 0.6),  # Incorrect
        PolicyDecision(PolicyAction.ABSTAIN, "[ABSTAIN]", 1, 200, 0.4)  # Abstained
    ]

    res = compute_policy_metrics("TestPolicy", [trace1, trace2, trace3], decisions)
    assert res.total_examples == 3
    assert res.answered_count == 2
    assert res.abstained_count == 1
    assert res.correct_count == 1
    assert res.incorrect_count == 1
    assert pytest.approx(res.coverage, 0.01) == 2.0 / 3.0
    assert pytest.approx(res.selective_risk, 0.01) == 0.50
    assert pytest.approx(res.accuracy_on_answered, 0.01) == 0.50
    assert pytest.approx(res.effective_accuracy, 0.01) == 1.0 / 3.0


def test_rc_auc():
    cov = np.array([0.2, 0.5, 0.8, 1.0])
    risk = np.array([0.05, 0.10, 0.15, 0.20])
    auc = compute_rc_auc(cov, risk)
    assert auc > 0.0
    assert auc < 1.0


def test_paired_bootstrap():
    a = np.array([10.0, 12.0, 11.0, 9.0, 10.5])
    b = np.array([5.0, 6.0, 5.5, 4.5, 5.0])
    res = paired_bootstrap_difference(a, b, num_bootstrap=200, seed=42)
    assert res["observed_mean_diff"] > 0
    assert res["ci_lower"] > 0
    assert res["ci_upper"] > 0


def test_holm_bonferroni():
    p_vals = [0.001, 0.04, 0.02, 0.50]
    adj = holm_bonferroni_correction(p_vals, alpha=0.05)
    assert len(adj) == 4
    # First ranked (0.001) threshold is 0.05 / 4 = 0.0125 -> Significant
    assert adj[0][3] is True
