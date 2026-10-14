"""
Compute cost and Pareto efficiency trade-off analysis.
"""

from typing import Dict, List, Tuple
import numpy as np
from src.evaluation.metrics import PolicyEvaluationResult


def compute_token_savings_percentage(
    baseline_result: PolicyEvaluationResult,
    target_result: PolicyEvaluationResult
) -> float:
    """
    Calculate the percentage reduction in average reasoning tokens:
    Savings = (Tokens_baseline - Tokens_target) / Tokens_baseline * 100%
    """
    base_tok = baseline_result.avg_tokens_spent
    tgt_tok = target_result.avg_tokens_spent
    if base_tok <= 0:
        return 0.0
    return max(0.0, ((base_tok - tgt_tok) / base_tok) * 100.0)


def find_pareto_efficient_points(
    costs: np.ndarray,
    risks: np.ndarray
) -> np.ndarray:
    """
    Identify non-dominated Pareto frontier points minimizing both (cost, risk).
    Returns boolean mask of Pareto-optimal indices.
    """
    assert len(costs) == len(risks)
    n = len(costs)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j:
                # If j dominates i (cost_j <= cost_i and risk_j <= risk_i with at least one strict inequality)
                if (costs[j] <= costs[i] and risks[j] <= risks[i]) and (costs[j] < costs[i] or risks[j] < risks[i]):
                    is_pareto[i] = False
                    break

    return is_pareto
