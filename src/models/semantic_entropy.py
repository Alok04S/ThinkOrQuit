"""
Semantic entropy clustering, uncertainty quantification, and martingale drift calculation.
"""

from collections import Counter
import math
from typing import Any, Dict, List, Optional, Tuple
from src.data.loaders import normalize_answer, is_answer_correct


def cluster_equivalent_answers(candidate_answers: List[str]) -> List[List[str]]:
    """
    Cluster candidate answers into semantic equivalence classes based on normalized equality.
    """
    clusters: List[List[str]] = []
    
    for ans in candidate_answers:
        norm_ans = normalize_answer(ans)
        matched = False
        for cluster in clusters:
            rep = cluster[0]
            if is_answer_correct(norm_ans, rep):
                cluster.append(norm_ans)
                matched = True
                break
        if not matched:
            clusters.append([norm_ans])
            
    return clusters


def compute_semantic_entropy(candidate_answers: List[str]) -> Tuple[float, str, float]:
    """
    Compute Semantic Entropy over clustered candidate answers.
    
    Returns:
    - entropy: float (in bits, base 2)
    - majority_answer: str (the representative answer of the largest cluster)
    - majority_ratio: float (proportion of samples belonging to the majority cluster)
    """
    if not candidate_answers:
        return 0.0, "", 0.0

    clusters = cluster_equivalent_answers(candidate_answers)
    total_samples = len(candidate_answers)
    
    # Sort clusters by size descending
    clusters.sort(key=len, reverse=True)
    
    majority_cluster = clusters[0]
    majority_answer = majority_cluster[0]
    majority_ratio = len(majority_cluster) / total_samples

    entropy = 0.0
    for cluster in clusters:
        p = len(cluster) / total_samples
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy, majority_answer, majority_ratio


def compute_martingale_drift(
    intermediate_entropies: List[float]
) -> List[float]:
    """
    Compute information drift across sequential checkpoints:
    Delta_H_t = H_t - H_{t-1}
    Negative drift indicates uncertainty reduction; near-zero indicates stagnation.
    """
    if len(intermediate_entropies) <= 1:
        return [0.0]

    drifts = [0.0]  # t=0 drift defined as 0
    for t in range(1, len(intermediate_entropies)):
        drift = intermediate_entropies[t] - intermediate_entropies[t - 1]
        drifts.append(drift)

    return drifts
