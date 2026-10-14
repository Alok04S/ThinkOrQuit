"""
Rigorous statistical hypothesis testing: paired bootstrap resampling, confidence intervals,
and Holm-Bonferroni multiplicity correction.
"""

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


def paired_bootstrap_difference(
    metrics_a: np.ndarray,
    metrics_b: np.ndarray,
    num_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute paired bootstrap difference (Metric_A - Metric_B) with confidence intervals and p-value.
    
    Returns:
    - mean_diff: float
    - ci_lower: float
    - ci_upper: float
    - p_value_two_sided: float
    - p_value_one_sided: float (testing if A < B)
    """
    assert len(metrics_a) == len(metrics_b), "Paired arrays must have identical length"
    n = len(metrics_a)
    rng = np.random.RandomState(seed)

    diffs = metrics_a - metrics_b
    observed_mean_diff = float(np.mean(diffs))

    boot_diffs = np.zeros(num_bootstrap)
    for b in range(num_bootstrap):
        sample_indices = rng.choice(n, size=n, replace=True)
        boot_diffs[b] = np.mean(diffs[sample_indices])

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_diffs, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_diffs, 100.0 * (1.0 - alpha / 2.0)))

    # Empirical p-value under null hypothesis (mean diff = 0)
    centered_boot = boot_diffs - observed_mean_diff
    p_two_sided = float(np.mean(np.abs(centered_boot) >= np.abs(observed_mean_diff)))
    # One-sided test (H1: A < B, i.e., observed_mean_diff < 0)
    p_one_sided = float(np.mean(boot_diffs >= 0.0)) if observed_mean_diff < 0 else float(np.mean(boot_diffs <= 0.0))

    return {
        "observed_mean_diff": observed_mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value_two_sided": p_two_sided,
        "p_value_one_sided": p_one_sided,
        "is_significant": bool(ci_lower * ci_upper > 0 and p_two_sided < 0.05)
    }


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> List[Tuple[int, float, float, bool]]:
    """
    Apply Holm-Bonferroni step-down procedure to control Family-Wise Error Rate (FWER).
    
    Returns list of tuples: (original_index, raw_p_value, adjusted_threshold, is_significant)
    """
    m = len(p_values)
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])

    results = []
    has_failed = False
    
    for rank, (orig_idx, p_val) in enumerate(sorted_pairs):
        threshold = alpha / float(m - rank)
        if not has_failed and p_val <= threshold:
            sig = True
        else:
            sig = False
            has_failed = True
        results.append((orig_idx, p_val, threshold, sig))

    # Re-sort to original order
    results.sort(key=lambda x: x[0])
    return results
