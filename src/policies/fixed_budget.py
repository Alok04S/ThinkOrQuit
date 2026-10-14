"""
Fixed budget reasoning policies (Greedy Single-Step and Full Self-Consistency).
"""

from typing import List
from src.models.inference import ReasoningCheckpoint, ReasoningTrace
from .base import BasePolicy, PolicyAction, PolicyDecision


class FixedBudgetGreedyPolicy(BasePolicy):
    """
    Baseline 1: Deterministic early single-step budget.
    Stops immediately at checkpoint 1 and answers. Never abstains.
    """
    def __init__(self):
        super().__init__(name="FixedBudget_Greedy")
        self.is_calibrated = True

    def calibrate(self, calibration_traces: List[ReasoningTrace]) -> None:
        pass

    def evaluate_step(
        self,
        trace: ReasoningTrace,
        checkpoint: ReasoningCheckpoint,
        is_last_step: bool
    ) -> PolicyAction:
        return PolicyAction.ANSWER


class FixedBudgetSelfConsistencyPolicy(BasePolicy):
    """
    Baseline 2: Full-budget multi-sample self-consistency (Wang et al., 2023).
    Always reasons until the maximum token budget is reached. Never abstains.
    """
    def __init__(self):
        super().__init__(name="FixedBudget_SelfConsistency")
        self.is_calibrated = True

    def calibrate(self, calibration_traces: List[ReasoningTrace]) -> None:
        pass

    def evaluate_step(
        self,
        trace: ReasoningTrace,
        checkpoint: ReasoningCheckpoint,
        is_last_step: bool
    ) -> PolicyAction:
        if is_last_step:
            return PolicyAction.ANSWER
        return PolicyAction.CONTINUE
