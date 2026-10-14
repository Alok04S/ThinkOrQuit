"""
Abstract Base Class and data containers for reasoning and abstention policies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
from src.models.inference import ReasoningCheckpoint, ReasoningTrace


class PolicyAction(str, Enum):
    ANSWER = "ANSWER"
    CONTINUE = "CONTINUE"
    ABSTAIN = "ABSTAIN"


@dataclass
class PolicyDecision:
    action: PolicyAction
    predicted_answer: str
    stopped_at_step: int
    tokens_spent: int
    confidence_score: float

    @property
    def is_answered(self) -> bool:
        return self.action == PolicyAction.ANSWER

    @property
    def is_abstained(self) -> bool:
        return self.action == PolicyAction.ABSTAIN


class BasePolicy(ABC):
    """Abstract base class for all reasoning stopping and abstention policies."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_calibrated = False

    @abstractmethod
    def calibrate(self, calibration_traces: List[ReasoningTrace]) -> None:
        """Calibrate policy thresholds or estimators using calibration traces."""
        pass

    @abstractmethod
    def evaluate_step(
        self,
        trace: ReasoningTrace,
        checkpoint: ReasoningCheckpoint,
        is_last_step: bool
    ) -> PolicyAction:
        """Evaluate action at a specific decision checkpoint."""
        pass

    def run(self, trace: ReasoningTrace) -> PolicyDecision:
        """Execute policy over the sequential trace checkpoints."""
        num_checkpoints = len(trace.checkpoints)
        if num_checkpoints == 0:
            return PolicyDecision(
                action=PolicyAction.ABSTAIN,
                predicted_answer="[ABSTAIN]",
                stopped_at_step=0,
                tokens_spent=0,
                confidence_score=0.0
            )

        for idx, cp in enumerate(trace.checkpoints):
            is_last = (idx == num_checkpoints - 1)
            action = self.evaluate_step(trace, cp, is_last_step=is_last)

            if action == PolicyAction.ANSWER:
                return PolicyDecision(
                    action=PolicyAction.ANSWER,
                    predicted_answer=cp.majority_answer,
                    stopped_at_step=cp.step_index,
                    tokens_spent=cp.token_count,
                    confidence_score=cp.majority_confidence
                )
            elif action == PolicyAction.ABSTAIN:
                return PolicyDecision(
                    action=PolicyAction.ABSTAIN,
                    predicted_answer="[ABSTAIN]",
                    stopped_at_step=cp.step_index,
                    tokens_spent=cp.token_count,
                    confidence_score=cp.majority_confidence
                )
            # If CONTINUE, proceed to next checkpoint unless it was the last checkpoint
            if is_last:
                # Terminal fallback if policy continued until the end
                return self._terminal_fallback(trace, cp)

        # Fallback
        last_cp = trace.checkpoints[-1]
        return PolicyDecision(
            action=PolicyAction.ANSWER,
            predicted_answer=last_cp.majority_answer,
            stopped_at_step=last_cp.step_index,
            tokens_spent=last_cp.token_count,
            confidence_score=last_cp.majority_confidence
        )

    def _terminal_fallback(self, trace: ReasoningTrace, last_cp: ReasoningCheckpoint) -> PolicyDecision:
        """Default behavior when maximum budget is reached."""
        return PolicyDecision(
            action=PolicyAction.ANSWER,
            predicted_answer=last_cp.majority_answer,
            stopped_at_step=last_cp.step_index,
            tokens_spent=last_cp.token_count,
            confidence_score=last_cp.majority_confidence
        )
