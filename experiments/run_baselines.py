"""
Experiment A: Evaluate Baselines 1 to 5 on GSM8K in-distribution benchmark.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loaders import load_dataset
from src.data.splits import create_deterministic_splits
from src.models.inference import TraceReplayEngine
from src.policies import (
    FixedBudgetGreedyPolicy,
    FixedBudgetSelfConsistencyPolicy,
    ConfidenceAbstainPolicy,
    DeerEarlyExitPolicy,
    ConformalThinkingPolicy
)
from src.evaluation.metrics import compute_policy_metrics
from src.utils.logging import log_experiment_to_registry, setup_logger


def run_baselines_experiment():
    logger = setup_logger("RunBaselines")
    logger.info("Starting Experiment A: Baseline Benchmarking on GSM8K...")

    records = load_dataset("data/processed/gsm8k_clean.jsonl", dataset_type="gsm8k")
    split = create_deterministic_splits(records, seed=42)

    engine = TraceReplayEngine("data/traces/gsm8k_traces.jsonl")
    calib_traces = [engine.get_trace(r["id"]) for r in split.calibration if engine.get_trace(r["id"])]
    test_traces = [engine.get_trace(r["id"]) for r in split.test if engine.get_trace(r["id"])]

    logger.info(f"Calibration traces: {len(calib_traces)}, Test traces: {len(test_traces)}")

    baselines = [
        FixedBudgetGreedyPolicy(),
        FixedBudgetSelfConsistencyPolicy(),
        ConfidenceAbstainPolicy(threshold=0.60),
        DeerEarlyExitPolicy(exit_threshold=0.80),
        ConformalThinkingPolicy(nominal_alpha=0.15)
    ]

    results = {}
    for base in baselines:
        base.calibrate(calib_traces)
        decisions = [base.run(tr) for tr in test_traces]
        res = compute_policy_metrics(base.name, test_traces, decisions)
        results[base.name] = res.to_dict()
        logger.info(f"{base.name:30s} | Cov: {res.coverage:.3f} | Risk: {res.selective_risk:.3f} | EffAcc: {res.effective_accuracy:.3f} | Tokens: {res.avg_tokens_spent:.1f}")

    os.makedirs("results", exist_ok=True)
    out_path = "results/baselines_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log_experiment_to_registry(
        registry_path="results/registry.jsonl",
        experiment_id="EXP_A_BASELINES",
        hypothesis="Baseline characterization on in-distribution GSM8K",
        config={"split_seed": 42, "num_test": len(test_traces), "dataset": "gsm8k"},
        status="COMPLETED",
        metrics=results,
        output_artifacts={"summary_json": out_path},
        notes="Evaluated Baselines 1-5 under identical partition splits"
    )

    logger.info(f"Experiment A completed. Saved to {out_path}")


if __name__ == "__main__":
    run_baselines_experiment()
