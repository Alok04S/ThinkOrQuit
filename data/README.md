# Data Governance & Dataset Documentation

## Overview

The ThinkOrQuit experimental framework evaluates adaptive reasoning and selective abstention across three distribution regimes:
1. **In-Distribution (ID) Arithmetic Reasoning**: GSM8K (Grade School Math 8K, Cobbe et al., 2021; MIT License).
2. **Difficulty & Domain Shift (OOD-Difficulty)**: MATH benchmark (Hendrycks et al., 2021; MIT License) - focused on Level 3–5 high school math and competition problems.
3. **Epistemic & Unanswerability Shift (OOD-Unanswerable)**: AbstentionBench (Kirichenko et al., Meta FAIR, 2025; CC-BY-4.0) - questions with false premises, contradictory constraints, or fundamentally unanswerable formulations where the correct behavior is selective abstention.

---

## Directory Structure

```
data/
├── README.md                      # Governance and provenance documentation
├── raw/                           # Immutable raw benchmark splits
│   ├── gsm8k/
│   ├── math/
│   └── abstentionbench/
├── processed/                     # Normalized instances with extracted ground-truth
│   ├── gsm8k_clean.jsonl
│   ├── math_clean.jsonl
│   └── abstentionbench_clean.jsonl
└── traces/                        # Reasoning trajectories at multiple compute checkpoints
    ├── gsm8k_traces.jsonl
    ├── math_traces.jsonl
    └── abstentionbench_traces.jsonl
```

---

## Partitioning & Data Split Governance

To guarantee statistical validity and prevent data leakage:
1. **Disjoint Hashes**: Splits are generated deterministically using SHA-256 hashes of normalized question strings.
2. **Four-Way Partition**:
   - **Train ($30\%$)**: Used solely for training lightweight parametric feature estimators (if applicable).
   - **Calibration ($20\%$)**: Reserved strictly for conformal risk control non-conformity quantile estimation and threshold setting. Never evaluated as test data.
   - **Validation ($20\%$)**: Hyperparameter verification and metric checking.
   - **Test ($30\%$)**: Final frozen evaluation partition. Never seen during calibration or hyperparameter tuning.
3. **Anti-Leakage Verification**: `src/data/validation.py` executes exact n-gram and SHA-256 checks to ensure $0\%$ overlap between partitions.
4. **Answer Normalization**:
   - Strips LaTeX formatting (`\boxed{}`, `\$`, `\text{}`).
   - Normalizes floating point and fraction expressions (e.g. `1/2` $\to$ `0.5`).
   - Normalizes whitespace, casing, and trailing units.
5. **No Label Fabrication**: Instances with malformed or ambiguous labels are flagged and removed with full logging.
