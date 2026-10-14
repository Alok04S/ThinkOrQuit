# Related Work Taxonomy and Landscape Map

## 1. Conceptual Taxonomy

Prior literature can be organized along two orthogonal dimensions:
1. **Decision Space**:
   - *Biaction Stopping*: Decide only when to stop thinking vs. continue thinking (AdaCtrl, DEER, CaTS, Snell et al.).
   - *Biaction Prediction*: Decide only whether to answer vs. abstain after or before full generation (R-Tuning, Conformal Abstention, AbstentionBench).
   - *Tri-action Space*: Decide dynamically whether to **Answer Now**, **Continue Reasoning**, or **Abstain / Quit** (Conformal Thinking lower/upper bounds, Knowing When to Quit, **ThinkOrQuit**).

2. **Computational Paradigm**:
   - *Weight Training / RL*: Requires changing LLM weights via SFT or RL (AdaCtrl, Knowing When to Quit, R-Tuning, Rational Metareasoning).
   - *Test-Time Gradient / Probing*: Requires per-instance gradient updates or hidden state probing (ORCA, Thought Calibration).
   - *Lightweight Inference-Time Policy*: Frozen base model, policy acts on lightweight observable trace signals without gradient descent (DEER, CaTS-ES, Conformal Abstention, **ThinkOrQuit**).

---

## 2. Landscape Matrix

| Framework | Decision Space | Requires Weight Training? | Risk-Controlled? | Evaluates Distribution Shift? | Handles Limited Calibration (\(N < 250\))? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AdaCtrl** (Huang et al., 2025) | Stop vs Continue | Yes (SFT + RL) | No | No | No |
| **DEER** (Yang et al., 2025) | Stop vs Continue | No (Frozen) | No | No | No |
| **CaTS-ES** (Huang et al., 2026) | Stop vs Sample More | No (Frozen) | No | No | No |
| **Snell et al.** (2024) | Budget Allocation | No (Search / Verifier) | No | No | No |
| **R-Tuning** (Zhang et al., 2024) | Answer vs Refuse | Yes (SFT) | No | Yes (OOD QA) | No |
| **Conformal Abstention** (Abbasi-Yadkori et al., 2024) | Answer vs Abstain | No (Frozen) | Yes (Conformal) | Limited | Yes |
| **Conformal Thinking** (Wang et al., 2026) | Stop, Continue, Prune | No (Validation quantiles) | Yes (Conformal) | No (Assumes i.i.d.) | Unexplored |
| **Knowing When to Quit** (Davidov et al., 2026) | Generate vs Quit | Yes (RL Policy / Value) | Heuristic | No | No |
| **DS-CP** (Lin et al., 2026) | Predict vs Set-Set | No (Prompt reweighting) | Yes (Weighted Conformal) | Yes (Covariate Shift) | Unexplored |
| **ORCA** (Wang et al., 2026) | Calibrated Sampling | Yes (Test-Time Training) | Yes (TTT Conformal) | Yes (Reasoning Shift) | No |
| **ThinkOrQuit** (Proposed Project) | **Answer, Continue, Abstain** | **No (Lightweight Policy)** | **Yes (Empirical & Conformal)** | **Yes (Difficulty & Task Shift)** | **Yes (Evaluated \(N \in [50, 500]\))** |

---

## 3. Structural Relationships

```
                                      [Test-Time Compute Control]
                                                 |
             +-----------------------------------+-----------------------------------+
             |                                                                       |
    [Heavyweight / Training]                                              [Lightweight / Inference-Time]
             |                                                                       |
  +----------+----------+                                                 +----------+----------+
  |                     |                                                 |                     |
[AdaCtrl (RL)]   [Knowing When to Quit (RL)]                             [DEER]              [CaTS-ES]
(Budget only)    (Quit action mid-CoT)                                (Early exit)        (Sample exit)
                                                                          |                     |
                                                                          +----------+----------+
                                                                                     |
                                                                        [Unifying Abstention & Risk]
                                                                                     |
                                                                      +--------------+--------------+
                                                                      |                             |
                                                             [Conformal Thinking]          [ThinkOrQuit]
                                                             (i.i.d. dual threshold)       (Tri-action under shift
                                                                                           & scarce calibration)
```

---

## 4. Methodological Differentiation

ThinkOrQuit occupies a unique, rigorous niche:
1. **Compute-Accessible**: Unlike AdaCtrl and Knowing When to Quit, it does not require millions of tokens of RL training across GPU clusters.
2. **Beyond Biaction**: Unlike DEER and CaTS, it does not force wrong answers on insoluble queries.
3. **Shift & Scarcity Aware**: Unlike Conformal Thinking, it explicitly measures failure modes under calibration distribution shifts and limited calibration sample budgets.
