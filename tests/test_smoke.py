"""
Smoke test: verifies end-to-end execution of data loading, trace replay,
policy calibration, inference, metric calculation, and bootstrap testing.
"""

import os
import pytest
from src.data.loaders import load_dataset
from src.data.splits import create_deterministic_splits
from src.data.validation import validate_dataset_splits
from src.models.inference import TraceReplayEngine
from src.policies import (
    FixedBudgetGreedyPolicy,
    FixedBudgetSelfConsistencyPolicy,
    ConfidenceAbstainPolicy,
    DeerEarlyExitPolicy,
    ConformalThinkingPolicy,
    ThinkOrQuitPolicy
)
from src.evaluation.metrics import compute_policy_metrics
from src.evaluation.statistical_tests import paired_bootstrap_difference
import numpy as np


def test_smoke_end_to_end():
    # 1. Load data
    proc_path = "data/processed/gsm8k_clean.jsonl"
    trace_path = "data/traces/gsm8k_traces.jsonl"
    assert os.path.exists(proc_path)
    assert os.path.exists(trace_path)

    records = load_dataset(proc_path, dataset_type="gsm8k")
    assert len(records) > 0

    # 2. Split data
    split = create_deterministic_splits(records, train_ratio=0.3, calib_ratio=0.2, val_ratio=0.2, test_ratio=0.3)
    val_report = validate_dataset_splits(split)
    assert val_report["audit_passed"] is True

    # 3. Load traces
    engine = TraceReplayEngine(trace_path)
    calib_ids = {r["id"] for r in split.calibration}
    test_ids = {r["id"] for r in split.test}

    calib_traces = [engine.get_trace(tid) for tid in calib_ids if engine.get_trace(tid)]
    test_traces = [engine.get_trace(tid) for tid in test_ids if engine.get_trace(tid)]
    assert len(calib_traces) > 0
    assert len(test_traces) > 0

    # 4. Initialize and calibrate all policies
    policies = [
        FixedBudgetGreedyPolicy(),
        FixedBudgetSelfConsistencyPolicy(),
        ConfidenceAbstainPolicy(threshold=0.7),
        DeerEarlyExitPolicy(exit_threshold=0.8),
        ConformalThinkingPolicy(nominal_alpha=0.15),
        ThinkOrQuitPolicy(nominal_alpha=0.15)
    ]

    results = {}
    for pol in policies:
        pol.calibrate(calib_traces)
        assert pol.is_calibrated is True

        decisions = [pol.run(tr) for tr in test_traces]
        assert len(decisions) == len(test_traces)
        
        # Verify decision contracts
        for d in decisions:
            assert d.action in ["ANSWER", "CONTINUE", "ABSTAIN"]
            assert d.tokens_spent >= 0
            assert 0.0 <= d.confidence_score <= 1.0

        res = compute_policy_metrics(pol.name, test_traces, decisions)
        results[pol.name] = res

        assert 0.0 <= res.coverage <= 1.0
        assert 0.0 <= res.selective_risk <= 1.0
        assert res.avg_tokens_spent > 0

    # 5. Bootstrap test
    dec_sc = [policies[1].run(tr) for tr in test_traces]
    dec_toq = [policies[5].run(tr) for tr in test_traces]
    tok_sc = np.array([d.tokens_spent for d in dec_sc])
    tok_toq = np.array([d.tokens_spent for d in dec_toq])

    boot = paired_bootstrap_difference(tok_toq, tok_sc, num_bootstrap=100)
    assert "observed_mean_diff" in boot
    assert "p_value_two_sided" in boot
