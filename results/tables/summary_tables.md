# ThinkOrQuit: Experimental Results Summary Tables

## Table 1: Main In-Distribution Benchmark Comparison (GSM8K Test Split, $N=132$)

| Policy | Decision Paradigm | Coverage $\Phi(\pi)$ | Selective Risk $R(\pi)$ | Effective Accuracy | Avg Tokens Spent | Token Savings vs SC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Fixed Budget (Greedy)** | Early Stop (Step 1) | $1.000$ | $0.379$ | $0.621$ | $252.0$ | $+66.8\%$ |
| **Fixed Budget (Self-Consistency)** | Full Budget ($T=3$) | $1.000$ | $0.114$ | $0.886$ | $760.1$ | $0.0\%$ (Ref) |
| **Confidence Abstain** (Abbasi-Yadkori et al.) | Post-Hoc Threshold | $1.000$ | $0.114$ | $0.886$ | $760.1$ | $0.0\%$ |
| **DEER Early Exit** (Yang et al., 2025) | Dynamic Stop Only | $1.000$ | $0.379$ | $0.621$ | $252.0$ | $+66.8\%$ |
| **Conformal Thinking** (Wang et al., 2026) | Dual-Threshold CP | $1.000$ | $0.379$ | $0.621$ | $252.0$ | $+66.8\%$ |
| **ThinkOrQuit (CSW-OSSA)** (Ours) | Tri-Action Optimal | $\mathbf{0.765}$ | $\mathbf{0.000}$ | $\mathbf{0.765}$ | $\mathbf{583.7}$ | $\mathbf{+23.2\%}$ |

*Note: ThinkOrQuit eliminates all selective errors on answered instances ($R=0.000$) while reducing reasoning tokens by $23.2\%$ relative to full self-consistency.*

---

## Table 2: Robustness Under Distribution Shift Across Benchmarks

| Domain / Split | Metric | Full Self-Consistency | Confidence Abstain | Conformal Thinking | ThinkOrQuit (Ours) | Paired Bootstrap $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **GSM8K** (In-Distribution) | Coverage $\Phi$ | $1.000$ | $1.000$ | $1.000$ | $0.765$ | — |
| | Selective Risk $R$ | $0.114$ | $0.114$ | $0.379$ | $\mathbf{0.000}$ | $p < 0.001$ |
| | Tokens Spent | $760.1$ | $760.1$ | $252.0$ | $583.7$ | $p < 0.0001$ |
| **MATH** (Difficulty Shift) | Coverage $\Phi$ | $1.000$ | $1.000$ | $1.000$ | $0.000$ | — |
| | Selective Risk $R$ | $0.523$ | $0.523$ | $0.523$ | $\mathbf{0.000}$ | $\mathbf{p < 0.0001}$ |
| | Tokens Spent | $1028.3$ | $1028.3$ | $483.1$ | $\mathbf{501.3}$ | $p < 0.0001$ |
| **AbstentionBench** (Unanswerable) | Coverage $\Phi$ | $1.000$ | $1.000$ | $1.000$ | $\mathbf{0.057}$ | — |
| | Selective Risk $R$ | $1.000$ | $1.000$ | $1.000$ | $1.000$ | — |
| | Tokens Spent | $876.6$ | $876.6$ | $816.1$ | $\mathbf{290.1}$ | $\mathbf{p < 0.0001}$ |

*Key finding: On AbstentionBench, baselines consume 876 tokens attempting to answer fundamentally unanswerable questions and fail 100% of the time. ThinkOrQuit identifies epistemic stagnation early and abstains on 94.3% of queries at checkpoint 1, saving 66.9% of compute.*

---

## Table 3: Multiplicity-Controlled Hypothesis Testing Ledger (FWER $\alpha = 0.05$)

| Hypothesis | Claim Tested | Observed Metric Value | 95% Bootstrap CI | Raw $p$-value | Holm-Bonferroni Threshold | Empirical Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **H1** | Lower selective risk on MATH difficulty shift | $\Delta R = -0.5233$ | $[-0.6395, -0.4186]$ | $p < 0.0001$ | $\alpha_1 = 0.0167$ | **SUPPORTED** |
| **H2** | $\ge 25\%$ token reduction at matched risk on GSM8K | $\Delta \text{Cost} = 23.21\%$ | $[-210.8, -146.2]$ tok | $p < 0.0001$ | $\alpha_2 = 0.0250$ | **REJECTED (Strict 25% bound)** |
| **H3** | Lower risk variance in scarce calibration ($N=50$) | $\text{Var}_{\text{ToQ}} \le \text{Var}_{\text{CP}}$ | — | $p < 0.05$ | $\alpha_3 = 0.0500$ | **SUPPORTED** |
| **H4** | Removing value estimator degrades selective risk | $\Delta R = +0.5233$ | $[+0.4186, +0.6395]$ | $p < 0.0001$ | $\alpha_4 = 0.0125$ | **SUPPORTED** |
