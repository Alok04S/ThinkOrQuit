"""
ThinkOrQuit: Sequential Conformal Optimal Stopping and Selective Abstention (SC-OSSA).
Unifies Value of Computation, Semantic Martingale Drift, and Covariate-Shift-Weighted Conformal Calibration.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from src.data.loaders import is_answer_correct
from src.models.inference import ReasoningCheckpoint, ReasoningTrace
from .base import BasePolicy, PolicyAction, PolicyDecision
from .density_ratio import CovariateShiftWeightEstimator
from .value_estimator import LightweightValueEstimator


class ThinkOrQuitPolicy(BasePolicy):
    """
    Proposed Method: Sequential Conformal Optimal Stopping and Selective Abstention.
    
    Tri-action decision space at each checkpoint t:
    1. ANSWER: if estimated risk <= calibrated conformal bound and continuation value <= stop value.
    2. CONTINUE: if estimated marginal gain from continuing exceeds token cost penalty and entropy has not stagnated.
    3. ABSTAIN: if problem is deemed unanswerable / epistemic risk exceeds acceptable threshold and continuing cannot salvage accuracy.
    """
    def __init__(
        self,
        nominal_alpha: float = 0.15,
        u_correct: float = 1.0,
        u_abstain: float = 0.0,
        u_error: float = -2.0,
        cost_penalty_per_100_tokens: float = 0.05,
        use_shift_weights: bool = True,
        use_martingale_drift: bool = True
    ):
        super().__init__(name="ThinkOrQuit_CSW_OSSA")
        self.nominal_alpha = nominal_alpha
        self.u_correct = u_correct
        self.u_abstain = u_abstain
        self.u_error = u_error
        self.cost_penalty = cost_penalty_per_100_tokens
        self.use_shift_weights = use_shift_weights
        self.use_martingale_drift = use_martingale_drift

        self.value_estimator = LightweightValueEstimator()
        self.shift_estimator = CovariateShiftWeightEstimator()
        self.calibrated_threshold = 0.70

    def calibrate(
        self,
        calibration_traces: List[ReasoningTrace],
        target_shift_traces: Optional[List[ReasoningTrace]] = None
    ) -> None:
        """
        Calibrate policy on calibration traces:
        1. Fit lightweight value estimator.
        2. Fit covariate-shift density ratio estimator (if target shift data provided).
        3. Determine covariate-shift-weighted conformal stopping threshold.
        """
        if not calibration_traces:
            self.is_calibrated = True
            return

        # 1. Fit value estimator
        self.value_estimator.fit(calibration_traces)

        # 2. Fit shift estimator if requested
        if self.use_shift_weights and target_shift_traces:
            self.shift_estimator.fit(calibration_traces, target_shift_traces)
            weights = self.shift_estimator.predict_weights(calibration_traces)
        else:
            weights = np.ones(len(calibration_traces))

        # 3. Covariate-Shift Weighted Conformal Calibration
        nonconf_scores = []
        sample_weights = []

        for idx, tr in enumerate(calibration_traces):
            w = weights[idx]
            for cp in tr.checkpoints:
                ans = cp.majority_answer
                correct = is_answer_correct(ans, tr.normalized_target) if not tr.is_unanswerable else False
                p_correct = self.value_estimator.predict_p_correct_stop(tr, cp)
                score = (1.0 - p_correct) if correct else 1.0
                nonconf_scores.append(score)
                sample_weights.append(w)

        if nonconf_scores:
            scores_arr = np.array(nonconf_scores)
            weights_arr = np.array(sample_weights) / np.sum(sample_weights)

            # Weighted quantile
            sort_idx = np.argsort(scores_arr)
            cum_w = np.cumsum(weights_arr[sort_idx])
            target_level = 1.0 - self.nominal_alpha
            cutoff_idx = np.searchsorted(cum_w, target_level)
            cutoff_idx = min(cutoff_idx, len(scores_arr) - 1)
            q_val = scores_arr[sort_idx[cutoff_idx]]

            self.calibrated_threshold = max(0.50, 1.0 - q_val)

        self.is_calibrated = True

    def evaluate_step(
        self,
        trace: ReasoningTrace,
        checkpoint: ReasoningCheckpoint,
        is_last_step: bool
    ) -> PolicyAction:
        p_stop = self.value_estimator.predict_p_correct_stop(trace, checkpoint)
        u_answer = p_stop * self.u_correct + (1.0 - p_stop) * self.u_error
        u_abstain = self.u_abstain

        # If at the last step, tri-action degenerates to Answer vs Abstain
        if is_last_step:
            if p_stop >= self.calibrated_threshold and u_answer > u_abstain:
                return PolicyAction.ANSWER
            return PolicyAction.ABSTAIN

        # Mid-generation: evaluate continuation value
        p_cont = self.value_estimator.predict_p_correct_continue(trace, checkpoint)
        marginal_tokens = 256.0
        cost_deduction = (marginal_tokens / 100.0) * self.cost_penalty
        
        # Penalize continuing if semantic martingale drift indicates information stagnation
        drift_penalty = 0.0
        if self.use_martingale_drift:
            # Stagnation penalty when Delta H >= -0.05 (entropy not decreasing)
            if checkpoint.martingale_drift >= -0.05:
                drift_penalty = 0.30

        u_continue = (p_cont * self.u_correct + (1.0 - p_cont) * self.u_error) - cost_deduction - drift_penalty

        # If confident enough to satisfy risk bound and Answer utility dominates Continue
        if p_stop >= self.calibrated_threshold and u_answer >= u_continue:
            return PolicyAction.ANSWER

        # If Continue utility dominates both Answer and Abstain, keep reasoning
        if u_continue > u_answer and u_continue > u_abstain:
            return PolicyAction.CONTINUE

        # If neither Answer nor Continue dominates Abstain, Quit now to save remaining tokens
        if u_abstain >= u_answer and u_abstain >= u_continue:
            return PolicyAction.ABSTAIN

        # Default fallback
        if u_answer > u_abstain:
            return PolicyAction.ANSWER
        return PolicyAction.ABSTAIN
