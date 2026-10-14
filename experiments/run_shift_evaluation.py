"""
Experiment E: Distribution Shift Evaluation (Hypothesis 1).
Evaluates policy robustness under Difficulty Shift (MATH) and Epistemic Shift (AbstentionBench).
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
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
from src.evaluation.statistical_tests import paired_bootstrap_difference
from src.utils.logging import log_experiment_to_registry, setup_logger


def run_shift_evaluation_experiment():
    logger = setup_logger("RunShiftEvaluation")
    logger.info("Starting Experiment E: Distribution Shift Evaluation (H1)...")

    # 1. Source (GSM8K) calibration traces
    gsm_records = load_dataset("data/processed/gsm8k_clean.jsonl", dataset_type="gsm8k")
    gsm_split = create_deterministic_splits(gsm_records, seed=42)
    gsm_engine = TraceReplayEngine("data/traces/gsm8k_traces.jsonl")
    calib_traces = [gsm_engine.get_trace(r["id"]) for r in gsm_split.calibration if gsm_engine.get_trace(r["id"])]
    gsm_test_traces = [gsm_engine.get_trace(r["id"]) for r in gsm_split.test if gsm_engine.get_trace(r["id"])]

    # 2. Target 1: MATH (difficulty shift)
    math_records = load_dataset("data/processed/math_clean.jsonl", dataset_type="math")
    math_split = create_deterministic_splits(math_records, seed=42)
    math_engine = TraceReplayEngine("data/traces/math_traces.jsonl")
    math_test_traces = [math_engine.get_trace(r["id"]) for r in math_split.test if math_engine.get_trace(r["id"])]

    # 3. Target 2: AbstentionBench (unanswerability shift)
    abstain_records = load_dataset("data/processed/abstentionbench_clean.jsonl", dataset_type="abstentionbench")
    abstain_split = create_deterministic_splits(abstain_records, seed=42)
    abstain_engine = TraceReplayEngine("data/traces/abstentionbench_traces.jsonl")
    abstain_test_traces = [abstain_engine.get_trace(r["id"]) for r in abstain_split.test if abstain_engine.get_trace(r["id"])]

    # Setup policies calibrated ONLY on GSM8K source data
    sc_policy = FixedBudgetSelfConsistencyPolicy()
    conf_policy = ConfidenceAbstainPolicy(threshold=0.60)
    conformal_policy = ConformalThinkingPolicy(nominal_alpha=0.15)
    toq_policy = ThinkOrQuitPolicy(nominal_alpha=0.15, use_shift_weights=True)

    policies = [sc_policy, conf_policy, conformal_policy, toq_policy]
    for pol in policies:
        if pol == toq_policy:
            # Shift estimator uses math validation as representative unlabeled target
            math_val = [math_engine.get_trace(r["id"]) for r in math_split.validation if math_engine.get_trace(r["id"])]
            pol.calibrate(calib_traces, target_shift_traces=math_val)
        else:
            pol.calibrate(calib_traces)

    domains = [
        ("In-Distribution (GSM8K)", gsm_test_traces),
        ("Difficulty Shift (MATH)", math_test_traces),
        ("Unanswerability Shift (AbstentionBench)", abstain_test_traces)
    ]

    domain_results = {}
    for dname, test_set in domains:
        logger.info(f"\n--- Evaluating domain: {dname} (N={len(test_set)}) ---")
        d_res = {}
        for pol in policies:
            decs = [pol.run(tr) for tr in test_set]
            res = compute_policy_metrics(pol.name, test_set, decs)
            d_res[pol.name] = res.to_dict()
            logger.info(f"{pol.name:30s} | Cov: {res.coverage:.3f} | Risk: {res.selective_risk:.3f} | Tokens: {res.avg_tokens_spent:.1f}")
        domain_results[dname] = d_res

    # Hypothesis 1 Test: On Difficulty Shift (MATH), compare Selective Risk of ThinkOrQuit vs Confidence Abstain Baseline
    decs_toq_math = [toq_policy.run(tr) for tr in math_test_traces]
    decs_conf_math = [conf_policy.run(tr) for tr in math_test_traces]

    # Error indicators on answered subset
    loss_toq = np.array([1.0 if not tr.final_ground_truth_correct and d.is_answered else 0.0 for tr, d in zip(math_test_traces, decs_toq_math)])
    loss_conf = np.array([1.0 if not tr.final_ground_truth_correct and d.is_answered else 0.0 for tr, d in zip(math_test_traces, decs_conf_math)])

    h1_bootstrap = paired_bootstrap_difference(loss_toq, loss_conf, num_bootstrap=1000, seed=42)
    h1_supported = bool(h1_bootstrap["observed_mean_diff"] <= 0)

    logger.info(f"\nHypothesis 1 Analysis (Selective Risk under Shift):")
    logger.info(f"Observed Mean Diff (ToQ - Baseline): {h1_bootstrap['observed_mean_diff']:.4f}")
    logger.info(f"95% CI: [{h1_bootstrap['ci_lower']:.4f}, {h1_bootstrap['ci_upper']:.4f}] | p-value: {h1_bootstrap['p_value_two_sided']:.4f}")
    logger.info(f"H1 Status: {'SUPPORTED' if h1_supported else 'REJECTED'}")

    # Generate Figure: Selective Risk across domains
    os.makedirs("results/figures", exist_ok=True)
    fig_path = "results/figures/distribution_shift_degradation.png"

    domain_labels = ["GSM8K (ID)", "MATH (Difficulty)", "AbstentionBench (Unans)"]
    p_names = [sc_policy.name, conf_policy.name, conformal_policy.name, toq_policy.name]
    clean_names = ["Full Self-Consistency", "Confidence Abstain", "Conformal Thinking", "ThinkOrQuit (CSW-OSSA)"]
    colors = ["#7f7f7f", "#aec7e8", "#1f77b4", "#d62728"]

    x = np.arange(len(domain_labels))
    width = 0.20

    plt.figure(figsize=(9, 5.5), dpi=300)
    for idx, (p_key, c_name, col) in enumerate(zip(p_names, clean_names, colors)):
        risks = [
            domain_results["In-Distribution (GSM8K)"][p_key]["selective_risk"],
            domain_results["Difficulty Shift (MATH)"][p_key]["selective_risk"],
            domain_results["Unanswerability Shift (AbstentionBench)"][p_key]["selective_risk"]
        ]
        plt.bar(x + idx * width, risks, width, label=c_name, color=col)

    plt.axhline(y=0.15, color="black", linestyle="--", alpha=0.7, label="Nominal Risk Bound ($\\alpha=0.15$)")
    plt.xticks(x + width * 1.5, domain_labels, fontsize=10, fontweight="bold")
    plt.ylabel("Selective Risk $R(\\pi)$ (Error Rate on Answered)", fontsize=11)
    plt.title("Selective Risk under Distribution Shift across Benchmarks", fontsize=12, fontweight="bold")
    plt.legend(frameon=True, fontsize=9, loc="upper left")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    summary = {
        "domain_results": domain_results,
        "h1_hypothesis_test": h1_bootstrap,
        "h1_status": "SUPPORTED" if h1_supported else "REJECTED",
        "figure_path": fig_path
    }

    out_path = "results/shift_evaluation_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log_experiment_to_registry(
        registry_path="results/registry.jsonl",
        experiment_id="EXP_E_DISTRIBUTION_SHIFT",
        hypothesis="H1: At matched coverage, joint policy maintains lower selective risk under shift",
        config={"source": "gsm8k", "targets": ["math", "abstentionbench"]},
        status="COMPLETED",
        metrics=summary,
        output_artifacts={"figure": fig_path, "summary_json": out_path},
        notes=f"H1 {'Supported' if h1_supported else 'Rejected'} with MATH selective risk diff: {h1_bootstrap['observed_mean_diff']:.4f}"
    )

    logger.info(f"Experiment E completed. Figure saved to {fig_path}")


if __name__ == "__main__":
    run_shift_evaluation_experiment()
