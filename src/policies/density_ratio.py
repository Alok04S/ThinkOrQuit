"""
Covariate Shift density ratio estimation: w(x) = dP_target(x) / dP_source(x).
Used for Covariate-Shift-Weighted Conformal Risk Control (CSW-CS).
"""

from typing import List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.models.inference import ReasoningTrace


def extract_prompt_features(traces: List[ReasoningTrace]) -> np.ndarray:
    """
    Extract lightweight representation features from question prompts:
    - Character length
    - Word count
    - Digit count
    - Mathematical symbol count (+, -, *, /, =, ^, etc.)
    - Punctuation count
    """
    feats = []
    for tr in traces:
        q = tr.question
        char_len = len(q)
        words = q.split()
        word_len = len(words)
        digits = sum(c.isdigit() for c in q)
        math_ops = sum(c in "+-*/=^><_\\{}[]()$" for c in q)
        punct = sum(c in ".,;:?!" for c in q)
        feats.append([
            char_len,
            word_len,
            digits,
            math_ops,
            punct,
            math_ops / max(1, word_len)
        ])
    return np.array(feats, dtype=float)


class CovariateShiftWeightEstimator:
    """
    Estimates density ratios w(x) between source (e.g., GSM8K) and target (e.g., MATH).
    Fits a regularized logistic regression discriminator:
    w(x) = P(target | x) / P(source | x) * (N_source / N_target)
    """
    def __init__(self):
        self.clf = LogisticRegression(C=1.0, max_iter=200, random_state=42)
        self.is_fitted = False
        self.source_prior = 0.5

    def fit(self, source_traces: List[ReasoningTrace], target_traces: List[ReasoningTrace]) -> None:
        if not source_traces or not target_traces:
            self.is_fitted = False
            return

        X_src = extract_prompt_features(source_traces)
        X_tgt = extract_prompt_features(target_traces)

        y_src = np.zeros(len(X_src))
        y_tgt = np.ones(len(X_tgt))

        X = np.vstack([X_src, X_tgt])
        y = np.concatenate([y_src, y_tgt])

        # Normalize features
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-6
        X_norm = (X - self.mean) / self.std

        self.clf.fit(X_norm, y)
        self.source_prior = len(X_src) / float(len(X))
        self.is_fitted = True

    def predict_weights(self, traces: List[ReasoningTrace], clip_range: tuple = (0.1, 10.0)) -> np.ndarray:
        """Estimate density weights w(x) for given traces."""
        if not self.is_fitted:
            return np.ones(len(traces), dtype=float)

        X = extract_prompt_features(traces)
        X_norm = (X - self.mean) / self.std

        probs = self.clf.predict_proba(X_norm)
        p_tgt = probs[:, 1]
        p_src = np.clip(probs[:, 0], 1e-4, 1.0 - 1e-4)

        # Odds ratio adjusted for prior imbalance
        prior_ratio = self.source_prior / (1.0 - self.source_prior + 1e-6)
        weights = (p_tgt / p_src) * prior_ratio
        return np.clip(weights, clip_range[0], clip_range[1])
