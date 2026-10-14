# ThinkOrQuit: Advanced Literature Review & Theoretical Foundations (2024–2026)

## 1. Theoretical Framing and Problem Formalization

A major limitation of existing test-time compute literature is the lack of a unified mathematical framework connecting **Optimal Stopping Theory**, **Selective Prediction**, and **Distribution-Free Conformal Risk Control**.

### 1.1 Formal Problem Definition: Sequential Conformal Optimal Stopping and Selective Abstention (SC-OSSA)

Let $(\mathcal{X}, \mathcal{Y})$ denote the instance and label spaces. Let $X \sim P_X$ be an input prompt, and let $Y \in \mathcal{Y}$ be the true answer. A reasoning model generates a multi-step trajectory or a set of candidate reasoning chains over discrete decision checkpoints $t \in \{1, 2, \dots, T_{\max}\}$.

We define the filtration $\mathcal{F}_t = \sigma(X, S_1, \dots, S_t)$, where $S_t$ denotes the observable trace state at checkpoint $t$ (e.g., token count, intermediate candidate answers, semantic entropy, verbalized confidence).

A **Tri-Action Policy** $\pi = (\tau, \delta)$ comprises:
1. **A Stopping Time** $\tau: \Omega \to \{1, \dots, T_{\max}\}$ adapted to $\mathcal{F}_t$, determining when reasoning computation ceases.
2. **A Decision Rule** $\delta: \mathcal{F}_\tau \to \{\text{Answer}(\hat{y}_\tau), \text{Abstain}\}$, where $\hat{y}_\tau \in \mathcal{Y}$ is the terminal predicted answer.

The system performance is evaluated across a 3D Pareto space:
- **Reasoning Cost**: $C(\pi) = \mathbb{E}[\text{Cost}(\tau)]$ (measured in reasoning tokens).
- **Answer Coverage**: $\Phi(\pi) = \mathbb{P}(\delta = \text{Answer})$.
- **Selective Risk**:
  $$R(\pi) = \mathbb{E}[\ell(Y, \hat{y}_\tau) \mid \delta = \text{Answer}] = \frac{\mathbb{E}[\ell(Y, \hat{y}_\tau) \cdot \mathbb{I}(\delta = \text{Answer})]}{\Phi(\pi)}$$
  where $\ell(y, \hat{y}) = \mathbb{I}(y \neq \hat{y})$ is the 0-1 misclassification loss.

### 1.2 The Constrained Metareasoning Objective
The policy aims to solve the constrained optimization problem:
$$\min_{\pi} C(\pi) \quad \text{subject to} \quad R(\pi) \le \alpha \quad \text{and} \quad \Phi(\pi) \ge \beta$$
where $\alpha \in (0, 1)$ is the user's maximum acceptable error risk on answered queries, and $\beta \in (0, 1)$ is the minimum acceptable answer coverage.

### 1.3 Conformal Risk Control and the Covariate Shift Breakdown
Under standard exchangeability ($P_{\text{cal}} = P_{\text{test}}$), conformal risk control (Angelopoulos et al., 2024; Wang et al., 2026) specifies non-conformity thresholds such that $\mathbb{P}(R(\pi) \le \alpha) \ge 1 - \delta_{\text{conf}}$.

However, under **Covariate Shift** where $P_{\text{cal}}(X) \neq P_{\text{test}}(X)$ with Radon-Nikodym derivative $w(X) = \frac{d P_{\text{test}}}{d P_{\text{cal}}}(X)$:
$$\mathbb{E}_{P_{\text{test}}}[\ell(Y, \hat{y}) \mid \delta = \text{Answer}] \neq \mathbb{E}_{P_{\text{cal}}}[\ell(Y, \hat{y}) \mid \delta = \text{Answer}]$$
As demonstrated by Lin et al. (DS-CP, ICML 2026) and Kirichenko et al. (AbstentionBench, 2025), reasoning models exhibit severe overconfidence when shifted to harder or unanswerable domains, causing empirical risk $\hat{R}_{\text{test}}$ to drastically breach $\alpha$.

ThinkOrQuit addresses this exact vulnerability by introducing **Covariate-Shift-Weighted Conformal Stopping (CSW-CS)** and **Semantic Martingale Drift Detection**.

---

## 2. Seed Papers Verification and Analysis

### 2.1 AdaCtrl: Towards Adaptive and Controllable Reasoning via Difficulty-Aware Budgeting
- **Citation**: Huang et al. (May 2025). arXiv:2505.18822. Official repository: `JoeYing1019/AdaCtrl`.
- **Methodology**: AdaCtrl addresses the "overthinking" pathology in reasoning models (where models produce thousands of redundant tokens on trivial arithmetic). It introduces a two-stage training scheme:
  1. *Cold-Start Supervised Fine-Tuning (SFT)*: Teaches the model to self-assess difficulty tags (`Easy`, `Medium`, `Hard`) and adhere to explicit length-triggered tags.
  2. *Difficulty-Aware Reinforcement Learning (RL)*: Calibrates difficulty estimation and optimizes a difficulty-conditioned length reward.
- **Key Findings**: Reduces reasoning chain length by 62.05%–91.04% on GSM8K and MATH500 while preserving accuracy on hard competitions (AIME2024, AIME2025).
- **Core Limitation**: AdaCtrl assumes all problems are solvable and forces the model to produce an answer under every budget. It does not possess any selective abstention ("I don't know") mechanism. Furthermore, it requires computationally heavy RL fine-tuning.

### 2.2 Conformal Thinking: Risk Control for Reasoning on a Compute Budget
- **Citation**: Wang et al. (Feb 2026). arXiv:2602.03814 / OpenReview. Affiliations: Johns Hopkins University, AWS.
- **Methodology**: Frames the token budget and adaptive reasoning threshold selection as a distribution-free risk control problem. It establishes a dual-threshold framework:
  1. *Upper threshold*: Stops reasoning early when intermediate confidence indicates sufficient probability of correctness (saving compute, risking erroneous answers).
  2. *Parametric lower threshold*: Preemptively aborts instances identified as "unsolvable" by the current model to avoid wasted tokens.
- **Key Findings**: Uses calibration data to guarantee that empirical error rate (selective risk) satisfies a user-specified bound \(\alpha\) with high probability, while reducing average token count.
- **Core Limitation**: Conformal risk bounds fundamentally rely on the **exchangeability** of the calibration and test distributions. Under distribution shift (difficulty shift or domain shift), the risk guarantee fails. Additionally, their lower threshold is evaluated primarily as a stopping mechanism rather than an explicit selective abstention protocol under data scarcity.

### 2.3 Knowing When to Quit: A Principled Framework for Dynamic Abstention in LLM Reasoning
- **Citation**: Davidov et al. (April 2026). arXiv:2604.18419. Affiliations: Amazon Science, University of Oxford.
- **Methodology**: Models test-time reasoning as an explicit Markov Decision Process (MDP) within a regularized Reinforcement Learning framework. At each token step, the policy can take action \(\mathcal{A} \in \{\text{generate}, \text{quit}\}\). Abstaining when the value function falls below an abstention reward parameter \(\lambda\) is proven to maximize selective accuracy under a compute penalty.
- **Key Findings**: Demonstrates that dynamic mid-generation quitting strictly dominates pre-generation (input-level) or post-generation (answer-level) abstention on selective accuracy and compute efficiency on math and toxicity benchmarks.
- **Core Limitation**: Requires training an RL policy or fitting a dedicated token-level value function. It does not evaluate behavior under small calibration sample sizes (e.g., \(N < 100\)) or out-of-distribution reasoning tasks.

### 2.4 R-Tuning: Instructing Large Language Models to Say "I Don't Know"
- **Citation**: Zhang et al. (2024). NAACL 2024 Outstanding Paper / arXiv:2311.09677. Repository: `shizhediao/R-Tuning`.
- **Methodology**: Refusal-aware instruction tuning. Evaluates the model's parametric knowledge boundaries during pretraining and constructs refusal-labeled instruction pairs when questions exceed internal knowledge.
- **Key Findings**: Successfully trains models to express refusal on unknown questions, improving calibration and showing transfer across disparate QA domains.
- **Core Limitation**: R-Tuning is a static SFT approach. It operates strictly before generation and does not interact with dynamic Chain-of-Thought search, test-time compute allocation, or multi-step reasoning verification.

### 2.5 AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions
- **Citation**: Kirichenko, Ibrahim, Chaudhuri, Bell (June 2025). arXiv:2506.09038 / FAIR Meta. Repository: `facebookresearch/AbstentionBench`.
- **Methodology**: Benchmarks 20 open and proprietary LLMs across 20 datasets covering 6 abstention modalities (unanswerable math, underspecified questions, outdated facts, false premises, etc.).
- **Key Findings**: Reasoning-tuned LLMs (e.g., models trained with test-time scaling or RL for math) suffer from severe overconfidence and fail catastrophically on unanswerable or ill-posed queries, rarely abstaining even when answers are fabricated.
- **Significance for ThinkOrQuit**: Provides vital empirical motivation. It demonstrates that test-time reasoning capabilities actually exacerbate the need for selective abstention, rather than solving it.

---

## 3. Complementary Literature (2024–2026)

### 3.1 Dynamic Early Exit & Compute-Optimal Scaling
- **DEER (Dynamic Early Exit in Reasoning Models)** (Yang et al., 2025, arXiv:2504.15895): A training-free method that inspects trial answers at reasoning transition points and halts generation when confidence is high. Achieves 19%–80% token reduction and small accuracy gains (+0.3% to +5.0%) across 11 models by curbing overthinking. However, DEER has no mechanism to abstain when the problem is unanswerable.
- **CaTS: Calibrated Test-Time Scaling** (Huang et al., ICLR 2026, arXiv:2503.00031): Proposes early stopping (CaTS-ES) and confidence-weighted self-consistency (CaTS-SC) over multiple reasoning paths, stopping sampling once a calibrated threshold is attained. Focuses exclusively on sample allocation, not abstention.
- **Rational Metareasoning for LLMs** (De Sabbata et al., NeurIPS 2024, arXiv:2410.05563): Formalizes the "Value of Computation" (VoC) using cognitive metareasoning principles. Optimizes the trade-off between intermediate reasoning cost and expected terminal accuracy via Expert Iteration.
- **Compute-Optimal Scaling** (Snell et al., Aug 2024, arXiv:2408.03314): Establishes that allocating test-time compute dynamically based on problem difficulty can outperform scaling model parameters by up to 14x on mathematical reasoning tasks.

### 3.2 Selective Prediction & Conformal Calibration under Distribution Shift
- **Conformal Abstention for Hallucination Mitigation** (Abbasi-Yadkori et al., NeurIPS 2024, arXiv:2405.01563): DeepMind researchers show that Conformal Risk Control using self-consistency agreement as a non-conformity score provides rigorous finite-sample error rate bounds in open-ended generation.
- **DS-CP: Domain-Shift-Aware Conformal Prediction** (Lin et al., ICML 2026, arXiv:2510.05566): Proves that standard conformal prediction suffers catastrophic under-coverage under covariate shift. Proposes prompt-proximity reweighting to restore valid coverage on out-of-domain evaluation.
- **ORCA: Online Reasoning Calibration** (Wang et al., COLM 2026, arXiv:2604.01170): Demonstrates that reasoning pattern shifts invalidate static conformal risk bounds; uses per-instance test-time training (TTT) probes to adapt calibration.
- **Confidence Elicitation in LLMs** (Xiong et al., ICLR 2024, arXiv:2306.13063): Demonstrates that verbalized and consistency-based uncertainty measures provide competitive failure prediction metrics without requiring logit access.
- **Selective Classification Foundations** (Geifman & El-Yaniv, NeurIPS 2017, arXiv:1709.00640): Establishes the formal definitions of coverage \(\phi(g) = \mathbb{E}[g(x)]\), selective risk \(R(f, g) = \frac{\mathbb{E}[\ell(f(x), y) g(x)]}{\phi(g)}\), and Risk-Coverage AUC (RC-AUC).

---

## 4. Critical Literature Synthesis: Answers to the 8 Core Questions

### Q1: What is already solved by difficulty-aware reasoning budgets?
Difficulty-aware allocation (e.g., AdaCtrl, Snell et al., CaTS-ES) successfully reduces token waste on easy instances and allocates depth to complex instances. **What is NOT solved**: It assumes that every instance is eventually solvable with sufficient compute. When presented with unsolvable, ambiguous, or out-of-distribution instances, difficulty budgeting either exhausts its maximum compute budget fruitlessly or forces a high-confidence incorrect hallucination.

### Q2: What is already solved by dynamic abstention during reasoning?
Davidov et al. (Knowing When to Quit, 2026) solved the formulation of mid-generation quitting as an RL MDP. **What is NOT solved**: Practical, lightweight, non-RL inference-time policies that operate on cached traces or frozen models without training value networks; and behavior under extreme data constraints (calibrating with only \(N \in [50, 250]\) examples).

### Q3: What guarantees, if any, do existing risk-control methods provide, and under which assumptions?
Conformal risk control methods (Wang et al. 2026, Abbasi-Yadkori et al. 2024) provide finite-sample PAC guarantees of the form:
\[
\mathbb{P}\left( R(f, g) \le \alpha \right) \ge 1 - \delta
\]
**Critical Assumption**: Exchangeability (identically distributed calibration and test data). If this assumption is violated (e.g. difficulty shift from GSM8K to MATH, or domain shift to out-of-distribution questions), the empirical risk \(\hat{R}\) frequently exceeds \(\alpha\) by large margins.

### Q4: What is known about calibration under distribution shift?
Prior work (DS-CP 2026, ORCA 2026, Xiong et al. 2024) consistently demonstrates that LLM confidence scores—whether derived from softmax probabilities, verbalized likelihoods, or self-consistency margins—suffer severe miscalibration under distribution shift. Models exhibit systematic overconfidence on harder or out-of-distribution tasks, leading to dramatic drops in conformal coverage (under-coverage).

### Q5: Which proposed extensions would merely reproduce existing work?
1. *Training an RL agent to quit mid-generation*: Merely reproduces Davidov et al. (2026).
2. *Implementing a two-stage difficulty tagging SFT/RL pipeline*: Merely reproduces AdaCtrl (Huang et al., 2025).
3. *Applying vanilla conformal prediction to stopping thresholds on i.i.d. splits*: Merely reproduces Conformal Thinking (Wang et al., 2026).
4. *Using sample-count thresholding on self-consistency*: Merely reproduces CaTS-ES (Huang et al., 2026).

### Q6: Which narrower research questions remain plausible and experimentally testable?
A rigorous, unaddressed question is:
> *How do lightweight, non-parametric and parametric inference-time tri-action policies (Answer, Continue Reasoning, Abstain) perform when calibrated on limited data (\(N \le 250\)), and how severely do their risk-coverage guarantees degrade under controlled difficulty shifts (GSM8K \(\to\) MATH) compared to decoupled two-stage baselines?*

### Q7: Which ideas are feasible without training a large model?
Inference-time policy optimization over extracted reasoning trace features:
- Features: Intermediate consistency margin across early samples, verbalized confidence, step length, token count, perplexity proxy, lexical difficulty metrics.
- Policy Models: Dual conformal quantile thresholding, logistic selective classifiers, shallow decision trees, and value-of-computation thresholders.
- Computation: Fitting takes < 1 second on CPU; evaluation across thousands of traces takes seconds. Requires zero GPU hours for policy fitting.

### Q8: Which idea offers the best balance of technical depth, compute requirements, and research value?
A reproducible experimental benchmarking study of **Lightweight Tri-Action Reasoning Policies (ThinkOrQuit)**:
- Directly contrasts joint tri-action policies against decoupled baselines (Fixed Budget, Confidence Abstention, and Conformal Thinking adaptations).
- Evaluates across calibration dataset sizes (\(N \in \{50, 100, 250, 500\}\)).
- Stress-tests against controlled difficulty shift (GSM8K in-distribution calibration \(\to\) MATH out-of-distribution test).
- Delivers a fully reproducible software framework with exact paired bootstrap significance testing.
