"""
Evaluation metrics for selective prediction, risk control, and compute cost in reasoning models.
Follows Geifman & El-Yaniv (NeurIPS 2017) and Abbasi-Yadkori et al. (NeurIPS 2024).
"""

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
from src.data.loaders import is_answer_correct
from src.models.inference import ReasoningTrace
from src.policies.base import PolicyDecision


@dataclass
class PolicyEvaluationResult:
    policy_name: str
    total_examples: int
    answered_count: int
    abstained_count: int
    correct_count: int
    incorrect_count: int
    coverage: float
    selective_risk: float
    accuracy_on_answered: float
    effective_accuracy: float
    avg_tokens_spent: float
    cost_per_correct_answer: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "total_examples": self.total_examples,
            "answered_count": self.answered_count,
            "abstained_count": self.abstained_count,
            "correct_count": self.correct_count,
            "incorrect_count": self.incorrect_count,
            "coverage": round(self.coverage, 4),
            "selective_risk": round(self.selective_risk, 4),
            "accuracy_on_answered": round(self.accuracy_on_answered, 4),
            "effective_accuracy": round(self.effective_accuracy, 4),
            "avg_tokens_spent": round(self.avg_tokens_spent, 2),
            "cost_per_correct_answer": round(self.cost_per_correct_answer, 2)
        }


def compute_policy_metrics(
    policy_name: str,
    traces: List[ReasoningTrace],
    decisions: List[PolicyDecision]
) -> PolicyEvaluationResult:
    """
    Compute comprehensive selective prediction and compute metrics given traces and decisions.
    """
    assert len(traces) == len(decisions), "Traces and decisions must have matching length"
    total_examples = len(traces)
    if total_examples == 0:
        return PolicyEvaluationResult(
            policy_name=policy_name,
            total_examples=0,
            answered_count=0,
            abstained_count=0,
            correct_count=0,
            incorrect_count=0,
            coverage=0.0,
            selective_risk=0.0,
            accuracy_on_answered=0.0,
            effective_accuracy=0.0,
            avg_tokens_spent=0.0,
            cost_per_correct_answer=0.0
        )

    answered_count = 0
    abstained_count = 0
    correct_count = 0
    incorrect_count = 0
    total_tokens = 0

    for tr, dec in zip(traces, decisions):
        total_tokens += dec.tokens_spent

        if dec.is_abstained:
            abstained_count += 1
        else:
            answered_count += 1
            is_corr = is_answer_correct(dec.predicted_answer, tr.normalized_target) if not tr.is_unanswerable else False
            if is_corr:
                correct_count += 1
            else:
                incorrect_count += 1

    coverage = answered_count / float(total_examples)
    selective_risk = (incorrect_count / float(answered_count)) if answered_count > 0 else 0.0
    accuracy_on_answered = (correct_count / float(answered_count)) if answered_count > 0 else 0.0
    effective_accuracy = correct_count / float(total_examples)
    avg_tokens_spent = total_tokens / float(total_examples)
    cost_per_correct = (total_tokens / float(correct_count)) if correct_count > 0 else float("inf")

    return PolicyEvaluationResult(
        policy_name=policy_name,
        total_examples=total_examples,
        answered_count=answered_count,
        abstained_count=abstained_count,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        coverage=coverage,
        selective_risk=selective_risk,
        accuracy_on_answered=accuracy_on_answered,
        effective_accuracy=effective_accuracy,
        avg_tokens_spent=avg_tokens_spent,
        cost_per_correct_answer=cost_per_correct
    )
