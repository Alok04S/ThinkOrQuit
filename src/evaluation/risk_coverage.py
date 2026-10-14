"""
Risk-Coverage curve evaluation and trapezoidal RC-AUC computation.
"""

from typing import List, Tuple
import numpy as np


def compute_risk_coverage_curve(
    confidences: np.ndarray,
    is_correct: np.ndarray,
    num_thresholds: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the empirical Risk-Coverage curve by sweeping thresholds across confidence scores.
    
    Returns:
    - coverages: np.ndarray of shape (num_points,) in ascending order
    - selective_risks: np.ndarray of shape (num_points,)
    """
    assert len(confidences) == len(is_correct), "Length mismatch"
    n = len(confidences)
    if n == 0:
        return np.array([0.0]), np.array([0.0])

    thresholds = np.linspace(confidences.min(), confidences.max(), num_thresholds)
    coverages = []
    risks = []

    for th in thresholds:
        answered_mask = confidences >= th
        num_ans = np.sum(answered_mask)
        if num_ans == 0:
            cov = 0.0
            risk = 0.0
        else:
            cov = num_ans / float(n)
            num_err = np.sum(~is_correct[answered_mask])
            risk = num_err / float(num_ans)
            
        coverages.append(cov)
        risks.append(risk)

    # Sort points by coverage ascending
    cov_arr = np.array(coverages)
    risk_arr = np.array(risks)
    sort_order = np.argsort(cov_arr)

    return cov_arr[sort_order], risk_arr[sort_order]


def compute_rc_auc(coverages: np.ndarray, risks: np.ndarray) -> float:
    """
    Compute Area Under the Risk-Coverage Curve (RC-AUC) using trapezoidal numerical integration.
    A lower RC-AUC indicates superior selective prediction performance.
    """
    if len(coverages) < 2:
        return 0.0

    # Ensure coverage starts at 0 and ends at 1 if possible
    sorted_idx = np.argsort(coverages)
    x = coverages[sorted_idx]
    y = risks[sorted_idx]

    # Trapezoidal integration compatible with NumPy 2.0+ and 1.x
    trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    rc_auc = float(trap_fn(y, x))
    return max(0.0, rc_auc)
