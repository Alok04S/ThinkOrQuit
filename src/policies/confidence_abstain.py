"""
Confidence-Threshold Selective Abstention Policy (Abbasi-Yadkori et al., 2024 / Geifman & El-Yaniv, 2017).
"""

from typing import List, Optional
import numpy as np
from src.data.loaders import is_answer_correct
from src.models.inference import ReasoningCheckpoint, ReasoningTrace
from .base import BasePolicy, PolicyAction, PolicyDecision


class ConfidenceAbstainPolicy(BasePolicy):
    """
    Baseline 3: Post-hoc confidence abstention.
    Consumes full reasoning budget, then answers if final majority confidence >= threshold,
    otherwise abstains.
    """
    def __init__(self, threshold: float = 0.60):
        super().__init__(name="Confidence_Abstain")
        self.threshold = threshold

    def calibrate(
        self,
        calibration_traces: List[ReasoningTrace],
        target_risk: Optional[float] = None,
        target_coverage: Optional[float] = None
    ) -> None:
        """
        Calibrate confidence threshold on calibration traces.
        If target_risk is given, selects the minimum threshold achieving empirical risk <= target_risk.
        If target_coverage is given, selects the quantile achieving that coverage.
        """
        if not calibration_traces:
            self.is_calibrated = True
            return

        confidences = []
        is_corrects = []
        for tr in calibration_traces:
            last_cp = tr.checkpoints[-1]
            conf = last_cp.majority_confidence
            ans = last_cp.majority_answer
            correct = is_answer_correct(ans, tr.normalized_target) if not tr.is_unanswerable else False
            confidences.append(conf)
            is_corrects.append(correct)

        conf_arr = np.array(confidences)
        corr_arr = np.array(is_corrects)

        if target_coverage is not None:
            # Threshold is (1 - target_coverage) quantile of confidences
            self.threshold = float(np.quantile(conf_arr, max(0.0, 1.0 - target_coverage)))
        elif target_risk is not None:
            # Search candidate thresholds
            candidate_thresholds = np.unique(conf_arr)
            best_thresh = float(candidate_thresholds[0])
            for th in candidate_thresholds:
                answered_mask = conf_arr >= th
                if np.sum(answered_mask) > 0:
                    emp_risk = 1.0 - np.mean(corr_arr[answered_mask])
                    if emp_risk <= target_risk:
                        best_thresh = float(th)
                        break
            self.threshold = best_thresh

        self.is_calibrated = True

    def evaluate_step(
        self,
        trace: ReasoningTrace,
        checkpoint: ReasoningCheckpoint,
        is_last_step: bool
    ) -> PolicyAction:
        if not is_last_step:
            return PolicyAction.CONTINUE
            
        # At final step, compare confidence against threshold
        if checkpoint.majority_confidence >= self.threshold:
            return PolicyAction.ANSWER
        return PolicyAction.ABSTAIN
