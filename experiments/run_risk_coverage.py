"""
Experiment C: Generate empirical Risk-Coverage curves and compute exact RC-AUC across policies.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
from src.data.loaders import load_dataset, is_answer_correct
from src.data.splits import create_deterministic_splits
from src.models.inference import TraceReplayEngine
from src.policies import (
    ConfidenceAbstainPolicy,
    ConformalThinkingPolicy,
    ThinkOrQuitPolicy
)
from src.evaluation.risk_coverage import compute_risk_coverage_curve, compute_rc_auc
from src.utils.logging import log_experiment_to_registry, setup_logger


def run_risk_coverage_experiment():
    logger = setup_logger("RunRiskCoverage")
    logger.info("Starting Experiment C: Risk-Coverage Curve and RC-AUC Evaluation...")

    records = load_dataset("data/processed/gsm8k_clean.jsonl", dataset_type="gsm8k")
    split = create_deterministic_splits(records, seed=42)

    engine = TraceReplayEngine("data/traces/gsm8k_traces.jsonl")
    calib_traces = [engine.get_trace(r["id"]) for r in split.calibration if engine.get_trace(r["id"])]
    test_traces = [engine.get_trace(r["id"]) for r in split.test if engine.get_trace(r["id"])]

    # We evaluate selective prediction score distributions on test traces
    # 1. Raw majority confidence (Confidence Abstain baseline)
    conf_scores = np.array([tr.checkpoints[-1].majority_confidence for tr in test_traces])
    is_corr_base = np.array([is_answer_correct(tr.checkpoints[-1].majority_answer, tr.normalized_target) for tr in test_traces])

    # 2. Conformal Thinking scores (calibrated dual threshold)
    conformal_pol = ConformalThinkingPolicy(nominal_alpha=0.15)
    conformal_pol.calibrate(calib_traces)
    conf_thinking_scores = np.array([tr.checkpoints[-1].majority_confidence * (1.0 if tr.checkpoints[-1].majority_confidence > conformal_pol.tau_lower else 0.1) for tr in test_traces])

    # 3. ThinkOrQuit estimated value / calibrated stopping score
    toq_pol = ThinkOrQuitPolicy(nominal_alpha=0.15)
    toq_pol.calibrate(calib_traces)
    toq_scores = np.array([toq_pol.value_estimator.predict_p_correct_stop(tr, tr.checkpoints[-1]) for tr in test_traces])

    cov_base, risk_base = compute_risk_coverage_curve(conf_scores, is_corr_base)
    cov_conf, risk_conf = compute_risk_coverage_curve(conf_thinking_scores, is_corr_base)
    cov_toq, risk_toq = compute_risk_coverage_curve(toq_scores, is_corr_base)

    auc_base = compute_rc_auc(cov_base, risk_base)
    auc_conf = compute_rc_auc(cov_conf, risk_conf)
    auc_toq = compute_rc_auc(cov_toq, risk_toq)

    logger.info(f"RC-AUC (Lower is better):")
    logger.info(f"  Confidence Abstain Baseline: {auc_base:.4f}")
    logger.info(f"  Conformal Thinking Baseline: {auc_conf:.4f}")
    logger.info(f"  ThinkOrQuit (Proposed):      {auc_toq:.4f}")

    # Generate publication-grade figure
    os.makedirs("results/figures", exist_ok=True)
    fig_path = "results/figures/risk_coverage_curve.png"

    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(cov_base, risk_base, label=f"Confidence Abstain (AUC={auc_base:.3f})", linestyle="--", color="gray", linewidth=1.8)
    plt.plot(cov_conf, risk_conf, label=f"Conformal Thinking (AUC={auc_conf:.3f})", linestyle="-.", color="#1f77b4", linewidth=2.0)
    plt.plot(cov_toq, risk_toq, label=f"ThinkOrQuit (AUC={auc_toq:.3f})", linestyle="-", color="#d62728", linewidth=2.4)

    plt.title("Risk-Coverage Tradeoff on GSM8K Test Partition", fontsize=12, fontweight="bold")
    plt.xlabel("Answer Coverage $\\Phi(\\pi)$", fontsize=11)
    plt.ylabel("Selective Risk $R(\\pi)$ (Error on Answered)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    rc_summary = {
        "rc_auc_confidence_abstain": auc_base,
        "rc_auc_conformal_thinking": auc_conf,
        "rc_auc_think_or_quit": auc_toq,
        "figure_path": fig_path
    }

    out_path = "results/risk_coverage_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rc_summary, f, indent=2)

    log_experiment_to_registry(
        registry_path="results/registry.jsonl",
        experiment_id="EXP_C_RISK_COVERAGE",
        hypothesis="ThinkOrQuit achieves lower RC-AUC than confidence baseline",
        config={"dataset": "gsm8k", "num_points": 100},
        status="COMPLETED",
        metrics=rc_summary,
        output_artifacts={"figure": fig_path, "summary_json": out_path},
        notes=f"ThinkOrQuit RC-AUC: {auc_toq:.4f} vs Baseline: {auc_base:.4f}"
    )

    logger.info(f"Experiment C completed. Figure saved to {fig_path}")


if __name__ == "__main__":
    run_risk_coverage_experiment()
