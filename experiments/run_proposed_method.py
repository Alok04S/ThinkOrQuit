"""
Experiment B: Evaluate ThinkOrQuit (Proposed CSW-OSSA) and perform statistical hypothesis testing against baselines.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.data.loaders import load_dataset
from src.data.splits import create_deterministic_splits
from src.models.inference import TraceReplayEngine
from src.policies import (
    FixedBudgetSelfConsistencyPolicy,
    ConfidenceAbstainPolicy,
    ConformalThinkingPolicy,
    ThinkOrQuitPolicy
)
from src.evaluation.metrics import compute_policy_metrics
from src.evaluation.compute_cost import compute_token_savings_percentage
from src.evaluation.statistical_tests import paired_bootstrap_difference
from src.utils.logging import log_experiment_to_registry, setup_logger


def run_proposed_method_experiment():
    logger = setup_logger("RunProposedMethod")
    logger.info("Starting Experiment B: ThinkOrQuit Primary Evaluation & Hypothesis Testing...")

    records = load_dataset("data/processed/gsm8k_clean.jsonl", dataset_type="gsm8k")
    split = create_deterministic_splits(records, seed=42)

    engine = TraceReplayEngine("data/traces/gsm8k_traces.jsonl")
    calib_traces = [engine.get_trace(r["id"]) for r in split.calibration if engine.get_trace(r["id"])]
    test_traces = [engine.get_trace(r["id"]) for r in split.test if engine.get_trace(r["id"])]

    # Reference policies
    sc_baseline = FixedBudgetSelfConsistencyPolicy()
    conf_baseline = ConfidenceAbstainPolicy(threshold=0.60)
    conformal_baseline = ConformalThinkingPolicy(nominal_alpha=0.15)
    toq_policy = ThinkOrQuitPolicy(nominal_alpha=0.15)

    policies = [sc_baseline, conf_baseline, conformal_baseline, toq_policy]
    decisions_map = {}
    metrics_map = {}

    for pol in policies:
        pol.calibrate(calib_traces)
        decs = [pol.run(tr) for tr in test_traces]
        decisions_map[pol.name] = decs
        res = compute_policy_metrics(pol.name, test_traces, decs)
        metrics_map[pol.name] = res.to_dict()
        logger.info(f"{pol.name:30s} | Cov: {res.coverage:.3f} | Risk: {res.selective_risk:.3f} | EffAcc: {res.effective_accuracy:.3f} | Tokens: {res.avg_tokens_spent:.1f}")

    # Compute Token Savings vs Full Self-Consistency (Hypothesis 2)
    toq_res = compute_policy_metrics("ThinkOrQuit", test_traces, decisions_map[toq_policy.name])
    sc_res = compute_policy_metrics("SelfConsistency", test_traces, decisions_map[sc_baseline.name])
    token_savings_pct = compute_token_savings_percentage(sc_res, toq_res)

    tok_sc = np.array([d.tokens_spent for d in decisions_map[sc_baseline.name]])
    tok_toq = np.array([d.tokens_spent for d in decisions_map[toq_policy.name]])
    h2_bootstrap = paired_bootstrap_difference(tok_toq, tok_sc, num_bootstrap=1000, seed=42)

    logger.info(f"\nHypothesis 2 Analysis (Token Cost Reduction):")
    logger.info(f"Token Savings: {token_savings_pct:.2f}% (Target: >= 25%)")
    logger.info(f"Observed Mean Diff: {h2_bootstrap['observed_mean_diff']:.1f} tokens")
    logger.info(f"95% CI: [{h2_bootstrap['ci_lower']:.1f}, {h2_bootstrap['ci_upper']:.1f}] | p-value: {h2_bootstrap['p_value_two_sided']:.4e}")
    h2_supported = bool(token_savings_pct >= 25.0 and h2_bootstrap["is_significant"] and h2_bootstrap["observed_mean_diff"] < 0)
    logger.info(f"H2 Status: {'SUPPORTED' if h2_supported else 'REJECTED'}")

    out_data = {
        "metrics": metrics_map,
        "token_savings_percentage": token_savings_pct,
        "h2_hypothesis_test": h2_bootstrap,
        "h2_status": "SUPPORTED" if h2_supported else "REJECTED"
    }

    out_path = "results/primary_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    log_experiment_to_registry(
        registry_path="results/registry.jsonl",
        experiment_id="EXP_B_PROPOSED_METHOD",
        hypothesis="H2: Adaptive tri-action policy reduces reasoning tokens >=25% at matched risk",
        config={"split_seed": 42, "num_test": len(test_traces), "nominal_alpha": 0.15},
        status="COMPLETED",
        metrics=out_data,
        output_artifacts={"primary_json": out_path},
        notes=f"H2 {'Supported' if h2_supported else 'Rejected'} with token reduction {token_savings_pct:.1f}%"
    )

    logger.info(f"Experiment B completed. Saved to {out_path}")


if __name__ == "__main__":
    run_proposed_method_experiment()
