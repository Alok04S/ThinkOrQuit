"""
Dynamic Early Exit Policy (DEER adaptation, Yang et al., 2025).
"""

from typing import List, Optional
import numpy as np
from src.models.inference import ReasoningCheckpoint, ReasoningTrace
from .base import BasePolicy, PolicyAction, PolicyDecision


class DeerEarlyExitPolicy(BasePolicy):
    """
    Baseline 4: Dynamic early exit without abstention.
    Exits early at checkpoint t if majority confidence >= exit_threshold.
    Forces answer if maximum budget is reached. Never abstains.
    """
    def __init__(self, exit_threshold: float = 0.80):
        super().__init__(name="DEER_EarlyExit")
        self.exit_threshold = exit_threshold

    def calibrate(
        self,
        calibration_traces: List[ReasoningTrace],
        target_token_budget: Optional[float] = None
    ) -> None:
        """
        Calibrate early exit threshold on calibration traces to achieve target budget or accuracy.
        """
        if not calibration_traces:
            self.is_calibrated = True
            return

        # Default heuristic based on median confidence across early steps
        early_confs = [
            tr.checkpoints[0].majority_confidence
            for tr in calibration_traces
            if tr.checkpoints
        ]
        if early_confs:
            self.exit_threshold = float(np.quantile(early_confs, 0.70))
        self.is_calibrated = True

    def evaluate_step(
        self,
        trace: ReasoningTrace,
        checkpoint: ReasoningCheckpoint,
        is_last_step: bool
    ) -> PolicyAction:
        if checkpoint.majority_confidence >= self.exit_threshold:
            return PolicyAction.ANSWER
            
        if is_last_step:
            return PolicyAction.ANSWER
            
        return PolicyAction.CONTINUE
