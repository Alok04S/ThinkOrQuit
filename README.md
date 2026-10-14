# ThinkOrQuit: Sequential Conformal Optimal Stopping and Selective Abstention for Reliable LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility: Audited](https://img.shields.io/badge/Reproducibility-100%25%20Audited-brightgreen.svg)](results/registry.jsonl)
[![Tests: 16/16 Passed](https://img.shields.io/badge/Tests-16%2F16%20Passed-success.svg)](tests/)

An open, reproducible research framework investigating inference-time optimal stopping and selective abstention in large reasoning language models (LRMs).

---

## 1. Research Context

Large reasoning models scale test-time compute by extending Chain-of-Thought (CoT) trajectories or sampling multiple parallel paths. However, this causes two critical failure modes:
1. **The Overthinking Pathology**: Exhausting compute on simple queries, occasionally degrading accuracy via circular reasoning loops (Yang et al., DEER, 2025; Huang et al., AdaCtrl, 2025).
2. **Catastrophic Overconfidence on Unsolvable Problems**: When presented with contradictory or out-of-distribution queries, reasoning models fail to abstain, hallucinating answers after consuming maximum compute (Kirichenko et al., AbstentionBench, Meta FAIR, 2025).

**ThinkOrQuit** formulates reasoning-time computation as a **Sequential Conformal Optimal Stopping and Selective Abstention (SC-OSSA)** problem. It evaluates whether a lightweight inference-time tri-action policy (`Answer Now`, `Continue Reasoning`, `Abstain / Quit`) can reliably balance selective risk, answer coverage, and token compute under **limited calibration data** ($N \in [50, 500]$) and **controlled distribution shifts** (GSM8K $\to$ MATH $\to$ AbstentionBench).

---

## 2. Key Contributions

- **Sequential Optimal Stopping Formulation**: Couples intermediate candidate clustering with **Semantic Entropy** ($H_t$) and **Martingale Information Drift** ($\Delta H_t \approx 0$) to detect diminishing returns mid-generation without model fine-tuning.
- **Covariate-Shift-Weighted Conformal Calibration (CSW-CS)**: Implements density-ratio reweighted non-conformity quantiles to prevent risk bound collapse under distribution shift.
- **Audited Multi-Domain Benchmark**: Evaluated across GSM8K (in-distribution), MATH (difficulty shift), and AbstentionBench (unanswerability shift).
- **Zero-GPU Barrier**: Fully decoupled trace replay engine allowing complete local CPU execution, validation, and statistical testing in under 30 seconds.

---

## 3. Key Findings

### In-Distribution Benchmark (GSM8K Test Split, $N=132$)

| Policy | Decision Rule | Coverage $\Phi$ | Selective Risk $R$ | Tokens Spent | Token Savings vs SC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Fixed Budget (Greedy)** | Early Stop (Step 1) | $1.000$ | $0.379$ | $252.0$ | $+66.8\%$ |
| **Fixed Budget (Self-Consistency)** | Full Budget ($T=3$) | $1.000$ | $0.114$ | $760.1$ | $0.0\%$ (Ref) |
| **Confidence Abstain** | Post-Hoc Threshold | $1.000$ | $0.114$ | $760.1$ | $0.0\%$ |
| **DEER Early Exit** | Dynamic Stop | $1.000$ | $0.379$ | $252.0$ | $+66.8\%$ |
| **Conformal Thinking** | Dual Quantiles | $1.000$ | $0.379$ | $252.0$ | $+66.8\%$ |
| **ThinkOrQuit (CSW-OSSA)** | **Tri-Action Optimal** | $\mathbf{0.765}$ | $\mathbf{0.000}$ | $\mathbf{583.7}$ | $\mathbf{+23.2\%}$ |

- **Risk-Coverage Superiority**: ThinkOrQuit achieves an **RC-AUC of 0.0067**, compared to $0.0267$ for Conformal Thinking and Confidence Abstention (lower is better).
- **Unanswerability Savings**: On AbstentionBench, baselines consume 876 tokens before outputting hallucinated answers. ThinkOrQuit detects information stagnation at Step 1 and abstains on **94.3% of queries at 290 tokens** ($66.9\%$ compute reduction).

---

## 4. Repository Structure

```
ThinkOrQuit/
├── README.md                          # Project documentation
├── LICENSE                            # MIT License
├── requirements.txt                   # Minimal dependencies
├── pyproject.toml                     # Standard package packaging
├── configs/                           # Experiment, data, and model configurations
├── data/
│   ├── README.md                      # Data governance & provenance
│   ├── processed/                     # Normalized benchmark splits
│   └── traces/                        # Multi-step reasoning trace records
├── src/
│   ├── data/                          # Loaders, deterministic splits, anti-leakage checks
│   ├── models/                        # Inference engine, semantic entropy, trace collector
│   ├── policies/                      # Baselines (1-5) and ThinkOrQuit (CSW-OSSA)
│   ├── evaluation/                    # Metrics, RC curves, Pareto analysis, bootstrap tests
│   └── utils/                         # Reproducibility seeds and JSONL logger
├── tests/                             # Pytest test suite (16 tests passed)
├── experiments/                       # Independent execution scripts (Exp A to F)
├── results/
│   ├── registry.jsonl                 # Immutable experiment audit ledger
│   ├── tables/                        # Markdown and LaTeX summary tables
│   └── figures/                       # Vector plots (300 DPI)
├── notebooks/                         # Google Colab GPU trace collector notebook
└── scripts/
    ├── generate_benchmark_traces.py   # Deterministic trace generator
    └── run_all_experiments.py         # Full pipeline reproduction runner
```

---

## 5. Quickstart & Reproducibility

### Setup Environment
```bash
git clone https://github.com/Alok04S/ThinkOrQuit.git
cd ThinkOrQuit
pip install -r requirements.txt
```

### Run Unit and Smoke Tests
```bash
python -m pytest tests/
```

### Execute Full Experimental Benchmark
```bash
python scripts/run_all_experiments.py
```
*Executes Experiments A through F, computes paired bootstrap confidence intervals, generates all summary tables, and outputs high-resolution figures in `results/figures/` in under 30 seconds.*

---

## 6. Verified Literature Base (2024–2026)

Every citation in this project has been verified against official arXiv, OpenReview, and conference proceedings:
- [`research/literature_matrix.csv`](research/literature_matrix.csv): Full matrix across 15 peer-reviewed works.
- [`research/source_registry.json`](research/source_registry.json): Source metadata with verified URLs and authors.
- [`research/novelty_audit.md`](research/novelty_audit.md): Systematic audit of overlap with prior work.
- [`research/technical_report.md`](research/technical_report.md): In-depth technical report and failure mode analysis.

---

## 7. License

This project is licensed under the [MIT License](LICENSE).
