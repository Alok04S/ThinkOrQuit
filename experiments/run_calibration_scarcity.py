"""
Experiment D: Calibration Data Scarcity (Hypothesis 3).
Evaluates stability and empirical risk violation |R_test - alpha| as calibration set size N varies in {50, 100, 250, 500}.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
from src.data.loaders import load_dataset
from src.data.splits import create_deterministic_splits, sample_calibration_subset
from src.models.inference import TraceReplayEngine
from src.policies import ConformalThinkingPolicy, ThinkOrQuitPolicy
from src.evaluation.metrics import compute_policy_metrics
from src.utils.logging import log_experiment_to_registry, setup_logger


def run_calibration_scarcity_experiment():
    logger = setup_logger("RunCalibrationScarcity")
    logger.info("Starting Experiment D: Calibration Data Scarcity Analysis (H3)...")

    records = load_dataset("data/processed/gsm8k_clean.jsonl", dataset_type="gsm8k")
    split = create_deterministic_splits(records, seed=42)

    engine = TraceReplayEngine("data/traces/gsm8k_traces.jsonl")
    full_calib_traces = [engine.get_trace(r["id"]) for r in split.calibration if engine.get_trace(r["id"])]
    test_traces = [engine.get_trace(r["id"]) for r in split.test if engine.get_trace(r["id"])]

    calib_sizes = [50, 100, 250, len(full_calib_traces)]
    nominal_alpha = 0.15
    num_subsample_trials = 20

    conformal_violations_by_size = {size: [] for size in calib_sizes}
    toq_violations_by_size = {size: [] for size in calib_sizes}

    for size in calib_sizes:
        logger.info(f"Evaluating calibration sample size N = {size}...")
        for trial in range(num_subsample_trials):
            sub_calib = sample_calibration_subset(full_calib_traces, subsample_size=size, seed=trial + 100)

            # Conformal Thinking
            cp = ConformalThinkingPolicy(nominal_alpha=nominal_alpha)
            cp.calibrate(sub_calib)
            dec_cp = [cp.run(tr) for tr in test_traces]
            res_cp = compute_policy_metrics("ConformalThinking", test_traces, dec_cp)
            conformal_violations_by_size[size].append(abs(res_cp.selective_risk - nominal_alpha))

            # ThinkOrQuit
            toq = ThinkOrQuitPolicy(nominal_alpha=nominal_alpha)
            toq.calibrate(sub_calib)
            dec_toq = [toq.run(tr) for tr in test_traces]
            res_toq = compute_policy_metrics("ThinkOrQuit", test_traces, dec_toq)
            toq_violations_by_size[size].append(abs(res_toq.selective_risk - nominal_alpha))

    mean_conf_violation = [np.mean(conformal_violations_by_size[s]) for s in calib_sizes]
    std_conf_violation = [np.std(conformal_violations_by_size[s]) for s in calib_sizes]
    mean_toq_violation = [np.mean(toq_violations_by_size[s]) for s in calib_sizes]
    std_toq_violation = [np.std(toq_violations_by_size[s]) for s in calib_sizes]

    # Check H3: Is ThinkOrQuit more stable at N=50?
    var_conf_50 = np.var(conformal_violations_by_size[50])
    var_toq_50 = np.var(toq_violations_by_size[50])
    h3_supported = bool(var_toq_50 <= var_conf_50)

    logger.info(f"\nHypothesis 3 Analysis (Sample Scarcity):")
    logger.info(f"N=50 Variance - Conformal Thinking: {var_conf_50:.6f} vs ThinkOrQuit: {var_toq_50:.6f}")
    logger.info(f"H3 Status: {'SUPPORTED' if h3_supported else 'REJECTED'}")

    os.makedirs("results/figures", exist_ok=True)
    fig_path = "results/figures/calibration_sample_efficiency.png"

    plt.figure(figsize=(7, 5), dpi=300)
    plt.errorbar(calib_sizes, mean_conf_violation, yerr=std_conf_violation, label="Conformal Thinking (Non-parametric)", fmt="-o", color="#1f77b4", capsize=4, linewidth=2)
    plt.errorbar(calib_sizes, mean_toq_violation, yerr=std_toq_violation, label="ThinkOrQuit (CSW-OSSA)", fmt="-s", color="#d62728", capsize=4, linewidth=2)

    plt.title("Risk Bound Violation vs Calibration Sample Size $N$", fontsize=12, fontweight="bold")
    plt.xlabel("Calibration Sample Size ($N$)", fontsize=11)
    plt.ylabel("Mean Risk Bound Violation $|\\hat{R}_{\\text{test}} - \\alpha|$", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    summary = {
        "calib_sizes": calib_sizes,
        "conformal_mean_violations": mean_conf_violation,
        "toq_mean_violations": mean_toq_violation,
        "variance_conf_n50": var_conf_50,
        "variance_toq_n50": var_toq_50,
        "h3_status": "SUPPORTED" if h3_supported else "REJECTED",
        "figure_path": fig_path
    }

    out_path = "results/calibration_scarcity_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log_experiment_to_registry(
        registry_path="results/registry.jsonl",
        experiment_id="EXP_D_CALIBRATION_SCARCITY",
        hypothesis="H3: Parametric regularized policy is more sample-efficient at small N than non-parametric conformal stopping",
        config={"sizes": calib_sizes, "trials": num_subsample_trials, "nominal_alpha": nominal_alpha},
        status="COMPLETED",
        metrics=summary,
        output_artifacts={"figure": fig_path, "summary_json": out_path},
        notes=f"H3 {'Supported' if h3_supported else 'Rejected'} with N=50 variance {var_toq_50:.5f} vs {var_conf_50:.5f}"
    )

    logger.info(f"Experiment D completed. Figure saved to {fig_path}")


if __name__ == "__main__":
    run_calibration_scarcity_experiment()
