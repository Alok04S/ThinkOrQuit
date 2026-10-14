"""
Evaluation and statistical benchmarking suite.
"""

from .metrics import PolicyEvaluationResult, compute_policy_metrics
from .risk_coverage import compute_risk_coverage_curve, compute_rc_auc
from .compute_cost import compute_token_savings_percentage, find_pareto_efficient_points
from .statistical_tests import paired_bootstrap_difference, holm_bonferroni_correction

__all__ = [
    "PolicyEvaluationResult",
    "compute_policy_metrics",
    "compute_risk_coverage_curve",
    "compute_rc_auc",
    "compute_token_savings_percentage",
    "find_pareto_efficient_points",
    "paired_bootstrap_difference",
    "holm_bonferroni_correction"
]
