"""
Lightweight Expected Utility and Continuation Value Estimator.
Predicts whether continuing reasoning yields sufficient marginal utility over immediate stopping or abstaining.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.data.loaders import is_answer_correct
from src.models.inference import ReasoningCheckpoint, ReasoningTrace


def extract_checkpoint_features(
    trace: ReasoningTrace,
    checkpoint: ReasoningCheckpoint
) -> np.ndarray:
    """
    Extract state features available at checkpoint t:
    - majority_confidence
    - semantic_entropy
    - martingale_drift
    - token_count
    - step_index
    - question_len
    """
    q_len = len(trace.question.split())
    return np.array([
        checkpoint.majority_confidence,
        checkpoint.semantic_entropy,
        checkpoint.martingale_drift,
        checkpoint.token_count / 1000.0,
        float(checkpoint.step_index),
        float(q_len) / 100.0
    ], dtype=float)


class LightweightValueEstimator:
    """
    Fits logistic probability models for:
    1. P(correct | stop now)
    2. P(correct | continue)
    """
    def __init__(self):
        self.stop_clf = LogisticRegression(C=1.0, max_iter=200, random_state=42)
        self.continue_clf = LogisticRegression(C=1.0, max_iter=200, random_state=42)
        self.is_fitted = False

    def fit(self, traces: List[ReasoningTrace]) -> None:
        X_stop, y_stop = [], []
        X_cont, y_cont = [], []

        for tr in traces:
            num_cp = len(tr.checkpoints)
            final_correct = tr.final_ground_truth_correct

            for idx, cp in enumerate(tr.checkpoints):
                feats = extract_checkpoint_features(tr, cp)
                is_curr_correct = is_answer_correct(cp.majority_answer, tr.normalized_target) if not tr.is_unanswerable else False
                
                X_stop.append(feats)
                y_stop.append(1 if is_curr_correct else 0)

                # For continue, only applicable if idx < num_cp - 1
                if idx < num_cp - 1:
                    X_cont.append(feats)
                    y_cont.append(1 if final_correct else 0)

        if len(X_stop) > 10 and len(np.unique(y_stop)) > 1:
            X_s = np.array(X_stop)
            self.mean_s = np.mean(X_s, axis=0)
            self.std_s = np.std(X_s, axis=0) + 1e-6
            self.stop_clf.fit((X_s - self.mean_s) / self.std_s, np.array(y_stop))
        else:
            self.mean_s = np.zeros(6)
            self.std_s = np.ones(6)

        if len(X_cont) > 10 and len(np.unique(y_cont)) > 1:
            X_c = np.array(X_cont)
            self.mean_c = np.mean(X_c, axis=0)
            self.std_c = np.std(X_c, axis=0) + 1e-6
            self.continue_clf.fit((X_c - self.mean_c) / self.std_c, np.array(y_cont))
            self.is_fitted = True
        else:
            self.mean_c = np.zeros(6)
            self.std_c = np.ones(6)
            self.is_fitted = False

    def predict_p_correct_stop(self, trace: ReasoningTrace, checkpoint: ReasoningCheckpoint) -> float:
        """Estimate P(correct | stop now)."""
        if not self.is_fitted:
            # Fallback to empirical confidence
            return float(np.clip(checkpoint.majority_confidence, 0.0, 1.0))
        feats = extract_checkpoint_features(trace, checkpoint)
        norm_feats = (feats - self.mean_s) / self.std_s
        prob = self.stop_clf.predict_proba(norm_feats.reshape(1, -1))[0, 1]
        return float(prob)

    def predict_p_correct_continue(self, trace: ReasoningTrace, checkpoint: ReasoningCheckpoint) -> float:
        """Estimate P(correct | continue to end)."""
        if not self.is_fitted:
            # Fallback based on confidence and entropy
            return float(np.clip(checkpoint.majority_confidence * (1.0 - checkpoint.semantic_entropy * 0.2), 0.0, 1.0))
        feats = extract_checkpoint_features(trace, checkpoint)
        norm_feats = (feats - self.mean_c) / self.std_c
        prob = self.continue_clf.predict_proba(norm_feats.reshape(1, -1))[0, 1]
        return float(prob)
