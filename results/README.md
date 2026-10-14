# ThinkOrQuit: Experimental Results and Audit Registry

## 1. Directory Structure

```
results/
├── README.md                          # Description of evaluation artifacts
├── registry.jsonl                     # Immutable ledger of all executed experiments
├── baselines_summary.json             # Experiment A: Baseline metrics on GSM8K
├── primary_comparison.json            # Experiment B: ThinkOrQuit vs baselines & H2 test
├── risk_coverage_summary.json         # Experiment C: Exact RC-AUC values
├── calibration_scarcity_summary.json  # Experiment D: Sample scarcity & H3 test
├── shift_evaluation_summary.json      # Experiment E: Distribution shift & H1 test
├── ablations_summary.json             # Experiment F: Ablations with Holm-Bonferroni & H4 test
├── figures/                           # Publication-grade vector plots (300 DPI)
│   ├── risk_coverage_curve.png
│   ├── calibration_sample_efficiency.png
│   ├── distribution_shift_degradation.png
│   └── ablation_study.png
└── tables/                            # Formatted result tables (LaTeX & Markdown)
```

## 2. Integrity and Provenance Policy
Every numerical metric and curve recorded here:
1. Was computed by executing the code in `experiments/` on actual parsed dataset splits;
2. Includes random seeds, sample counts, and execution timestamps logged to `registry.jsonl`;
3. Distinguishes statistically significant results ($p < 0.05$ with paired BCa bootstrap) from non-significant results;
4. Employs Holm-Bonferroni correction to strictly control Family-Wise Error Rate (FWER) across multiple hypotheses.
