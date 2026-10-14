"""
Unit tests for reasoning and abstention policies.
"""

from src.models.inference import ReasoningCheckpoint, ReasoningTrace
from src.policies import (
    FixedBudgetGreedyPolicy,
    FixedBudgetSelfConsistencyPolicy,
    ConfidenceAbstainPolicy,
    DeerEarlyExitPolicy,
    ConformalThinkingPolicy,
    ThinkOrQuitPolicy,
    PolicyAction
)


def create_dummy_trace(confidences=[0.4, 0.6, 0.8], tokens=[250, 500, 750], target="10", is_unans=False):
    checkpoints = [
        ReasoningCheckpoint(
            step_index=i+1,
            token_count=tokens[i],
            candidate_answers=[target]*int(confidences[i]*5) + ["0"]*int((1-confidences[i])*5),
            majority_answer=target if confidences[i] >= 0.5 else "0",
            majority_confidence=confidences[i],
            semantic_entropy=1.0 - confidences[i],
            martingale_drift=-0.1 if i > 0 else 0.0,
            verbalized_confidence=confidences[i]
        )
        for i in range(len(confidences))
    ]
    return ReasoningTrace(
        id="dummy_01",
        question="What is 5 + 5?",
        target=target,
        normalized_target=target,
        is_unanswerable=is_unans,
        checkpoints=checkpoints,
        total_tokens_full=tokens[-1],
        final_ground_truth_correct=True
    )


def test_fixed_budget_policies():
    trace = create_dummy_trace()
    
    greedy = FixedBudgetGreedyPolicy()
    dec_g = greedy.run(trace)
    assert dec_g.action == PolicyAction.ANSWER
    assert dec_g.stopped_at_step == 1
    assert dec_g.tokens_spent == 250

    sc = FixedBudgetSelfConsistencyPolicy()
    dec_sc = sc.run(trace)
    assert dec_sc.action == PolicyAction.ANSWER
    assert dec_sc.stopped_at_step == 3
    assert dec_sc.tokens_spent == 750


def test_confidence_abstain_policy():
    high_conf_trace = create_dummy_trace(confidences=[0.4, 0.6, 0.9])
    low_conf_trace = create_dummy_trace(confidences=[0.2, 0.3, 0.4])

    policy = ConfidenceAbstainPolicy(threshold=0.7)
    
    dec_high = policy.run(high_conf_trace)
    assert dec_high.action == PolicyAction.ANSWER

    dec_low = policy.run(low_conf_trace)
    assert dec_low.action == PolicyAction.ABSTAIN
    assert dec_low.predicted_answer == "[ABSTAIN]"


def test_deer_early_exit():
    # Confidence reaches 0.9 at step 2 -> should exit early at step 2
    trace = create_dummy_trace(confidences=[0.5, 0.9, 0.95], tokens=[250, 500, 750])
    deer = DeerEarlyExitPolicy(exit_threshold=0.85)
    dec = deer.run(trace)
    assert dec.action == PolicyAction.ANSWER
    assert dec.stopped_at_step == 2
    assert dec.tokens_spent == 500


def test_think_or_quit_policy():
    trace = create_dummy_trace(confidences=[0.85, 0.9, 0.95], tokens=[250, 500, 750])
    toq = ThinkOrQuitPolicy(nominal_alpha=0.15)
    toq.calibrate([trace])

    dec = toq.run(trace)
    assert dec.action in [PolicyAction.ANSWER, PolicyAction.ABSTAIN]
    assert dec.tokens_spent > 0
