# Novelty Audit and Risk-of-Overlap Analysis

## 1. Objective of this Audit

In strict compliance with **Mandatory Research Integrity Rules (Section 3.3)**, this document assesses the potential overlap between the proposed ThinkOrQuit framework and verified prior literature (2024–2026). It documents what prior papers have already claimed, identifies points of contact, and delineates a defensible, honest research scope that does not make unsupported novelty claims.

---

## 2. Overlap Matrix with Published Literature

### 2.1 Overlap with *Conformal Thinking* (Wang et al., 2026)
- **What they did**: Introduced dual stopping thresholds (an upper threshold for early exit when confident, and a lower parametric threshold for preemptively stopping unsolvable queries) calibrated via distribution-free conformal risk control on an i.i.d. validation set.
- **Risk of Overlap**: HIGH if ThinkOrQuit merely calibrates two thresholds on i.i.d. validation data using conformal prediction.
- **Defensible Differentiation**:
  1. Conformal Thinking assumes exchangeability. ThinkOrQuit systematically evaluates the breakdown of risk control when calibration data undergoes **difficulty shift** (e.g. calibration on GSM8K, test on MATH) and **sample scarcity** (\(N \le 100\)).
  2. Conformal Thinking treats the lower threshold as an early stopping mechanism to prune compute on unsolvable instances, but outputs whatever prediction exists. ThinkOrQuit formulates a dedicated **selective prediction action (Abstain/Quit)** with formal coverage and selective risk accounting (Geifman & El-Yaniv, 2017).
  3. ThinkOrQuit compares decoupled two-stage heuristics against joint lightweight value-based policies.

### 2.2 Overlap with *Knowing When to Quit* (Davidov et al., 2026)
- **What they did**: Formulated mid-generation dynamic quitting in an RL MDP framework and proved that quitting when the value function drops below a reward threshold optimizes selective accuracy.
- **Risk of Overlap**: MODERATE if ThinkOrQuit claims that "quitting mid-generation" is an entirely novel concept.
- **Defensible Differentiation**:
  1. Davidov et al. rely on regularized reinforcement learning and value networks.
  2. ThinkOrQuit explicitly does NOT perform RL training. It investigates whether **lightweight inference-time policies** (operating on observable trace features: consistency margin, verbalized confidence, length, and perplexity proxies) can achieve comparable Pareto trade-offs without model training.
  3. Davidov et al. did not investigate sample efficiency in policy calibration or calibration fragility under distribution shift.

### 2.3 Overlap with *AdaCtrl* (Huang et al., 2025)
- **What they did**: Trained a model via cold-start SFT and RL to adjust reasoning length based on self-assessed difficulty tiers.
- **Risk of Overlap**: LOW. AdaCtrl requires full model fine-tuning and provides no mechanism for abstention or risk control.

### 2.4 Overlap with *CaTS* (Huang et al., 2026) and *DEER* (Yang et al., 2025)
- **What they did**: CaTS-ES stops sampling paths when confidence is reached. DEER exits early from CoT generation when trial answer confidence is reached.
- **Risk of Overlap**: MODERATE for the "early answer" branch.
- **Defensible Differentiation**: Both DEER and CaTS-ES only possess a "Stop and Answer" mechanism. They have no "Quit / Abstain" action. When faced with deceptive or impossible questions, they are forced to produce an error. ThinkOrQuit unifies early stopping with selective abstention.

---

## 3. Scope Boundary and Prohibited Claims

To ensure complete scientific integrity, the project adheres to the following boundaries:

1. **NO Claim of inventing the first dynamic stopping rule**: Dynamic early exit in LLM reasoning was established by Yang et al. (DEER, 2025) and Snell et al. (2024).
2. **NO Claim of inventing risk-controlled compute budgeting**: Distribution-free risk control for reasoning budgets was formulated by Wang et al. (Conformal Thinking, 2026).
3. **NO Claim of inventing mid-generation quitting**: Formally modeled by Davidov et al. (Knowing When to Quit, 2026).
4. **NO Claim of superior performance over frontier RL models**: We evaluate lightweight inference-time policies on frozen models and cached traces; we do not claim to beat full-scale RL on raw accuracy.

---

## 4. Defensible Core Research Scope & Theoretical Significance

The legitimate, uncrowded, and mathematically rigorous research contribution of ThinkOrQuit is defined as:

> **"A Sequential Conformal Optimal Stopping and Selective Abstention (SC-OSSA) framework for reasoning language models that provides finite-sample risk control, incorporates semantic martingale drift signals, and maintains validity under covariate shift and calibration data scarcity ($N \in [50, 500]$)."**

### Why this attracts academic professors and research labs:
1. **Mathematical Rigor over Ad-hoc Heuristics**: Instead of arbitrary prompt heuristics or ungrounded threshold sweeps, ThinkOrQuit formalizes test-time compute as a constrained optimal stopping problem with provable empirical Bernstein / Hoeffding concentration bounds and weighted conformal risk control.
2. **Identification of Fundamental Failure Modes**: Rather than claiming synthetic "SOTA", it systematically probes where conformal exchangeability breaks down across difficulty gradients (GSM8K $\to$ MATH) and unanswerable distributions (AbstentionBench).
3. **Information-Theoretic Features**: Utilizes **Semantic Entropy** (clustering equivalent intermediate paths) and **Martingale Drift** ($\Delta H_t \le 0$) rather than fragile token-length counts.
4. **Sample Efficiency Guarantees**: Investigates the finite-sample stability of risk control under calibration scarcity ($N \le 100$), a pressing problem for domain adaptation and low-resource deployment.
5. **Reproducibility and Software Engineering**: Fully deterministic hash-split pipelines, paired bootstrap statistical hypothesis tests (BCa CIs), family-wise error rate control (Holm-Bonferroni), and open auditable JSONL registries.
