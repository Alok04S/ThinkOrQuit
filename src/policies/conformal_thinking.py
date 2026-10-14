"""
Conformal Thinking Policy (Wang et al., 2026 adaptation).
Implements dual-threshold risk control:
- Upper threshold (early stop when confident)
- Lower parametric threshold (preemptive abort for unsolvable instances)
"""

from typing import List, Optional
import numpy as np
from src.data.loaders import is_answer_correct
from src.models.inference import ReasoningCheckpoint, ReasoningTrace
from .base import BasePolicy, PolicyAction, PolicyDecision


class ConformalThinkingPolicy(BasePolicy):
    """
    Baseline 5: Distribution-free risk controlled stopping (Wang et al., 2026).
    Employs upper threshold tau_upper and lower threshold tau_lower.
    """
    def __init__(
        self,
        nominal_alpha: float = 0.15,
        tau_upper: float = 0.80,
        tau_lower: float = 0.25
    ):
        super().__init__(name="Conformal_Thinking")
        self.nominal_alpha = nominal_alpha
        self.tau_upper = tau_upper
        self.tau_lower = tau_lower

    def calibrate(self, calibration_traces: List[ReasoningTrace]) -> None:
        """
        Calibrate dual thresholds using conformal risk control on calibration traces.
        Non-conformity score: 1.0 - confidence.
        """
        if not calibration_traces:
            self.is_calibrated = True
            return

        scores = []
        for tr in calibration_traces:
            for cp in tr.checkpoints:
                ans = cp.majority_answer
                correct = is_answer_correct(ans, tr.normalized_target) if not tr.is_unanswerable else False
                score = 1.0 - cp.majority_confidence if correct else 1.0
                scores.append(score)

        n = len(scores)
        if n > 0:
            # Conformal quantile: ceil((n + 1) * (1 - alpha)) / n
            k = int(np.ceil((n + 1) * (1.0 - self.nominal_alpha)))
            k = min(max(1, k), n)
            q_val = float(np.partition(scores, k - 1)[k - 1])
            self.tau_upper = max(0.50, 1.0 - q_val)
            self.tau_lower = max(0.20, self.tau_upper * 0.35)

        self.is_calibrated = True

    def evaluate_step(
        self,
        trace: ReasoningTrace,
        checkpoint: ReasoningCheckpoint,
        is_last_step: bool
    ) -> PolicyAction:
        conf = checkpoint.majority_confidence

        # Upper threshold: confident enough to stop and answer
        if conf >= self.tau_upper:
            return PolicyAction.ANSWER

        # Lower threshold: unpromising / unsolvable -> preemptive quit
        if conf <= self.tau_lower and not is_last_step:
            return PolicyAction.ABSTAIN

        if is_last_step:
            # At terminal step, answer if above lower threshold, else abstain
            if conf > self.tau_lower:
                return PolicyAction.ANSWER
            return PolicyAction.ABSTAIN

        return PolicyAction.CONTINUE
