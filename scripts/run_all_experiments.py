"""
Single-command orchestrator to execute the full experimental suite (Experiments A through F).
Guarantees 100% reproducible execution and automated audit trail generation.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_baselines import run_baselines_experiment
from experiments.run_proposed_method import run_proposed_method_experiment
from experiments.run_risk_coverage import run_risk_coverage_experiment
from experiments.run_calibration_scarcity import run_calibration_scarcity_experiment
from experiments.run_shift_evaluation import run_shift_evaluation_experiment
from experiments.run_ablations import run_ablations_experiment
from src.utils.logging import setup_logger


def main():
    logger = setup_logger("Orchestrator")
    logger.info("=================================================================")
    logger.info("      THINKORQUIT: FULL REPRODUCIBILITY EXPERIMENTAL PIPELINE     ")
    logger.info("=================================================================")
    start_time = time.time()

    logger.info("\n>>> Step 1/6: Running Experiment A (Baselines Benchmarking)...")
    run_baselines_experiment()

    logger.info("\n>>> Step 2/6: Running Experiment B (ThinkOrQuit & Hypothesis 2)...")
    run_proposed_method_experiment()

    logger.info("\n>>> Step 3/6: Running Experiment C (Risk-Coverage Curves & RC-AUC)...")
    run_risk_coverage_experiment()

    logger.info("\n>>> Step 4/6: Running Experiment D (Calibration Scarcity & Hypothesis 3)...")
    run_calibration_scarcity_experiment()

    logger.info("\n>>> Step 5/6: Running Experiment E (Distribution Shift & Hypothesis 1)...")
    run_shift_evaluation_experiment()

    logger.info("\n>>> Step 6/6: Running Experiment F (Ablations & Hypothesis 4)...")
    run_ablations_experiment()

    elapsed = time.time() - start_time
    logger.info("=================================================================")
    logger.info(f"   ALL EXPERIMENTS COMPLETED SUCCESSFULLY in {elapsed:.2f} seconds!")
    logger.info("   Outputs: results/registry.jsonl, results/figures/, results/*.json")
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
