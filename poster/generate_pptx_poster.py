"""
Python script to generate a conference-quality, editable PowerPoint research poster (.pptx).
Slide size: 48.0 inches x 27.0 inches (16:9 Widescreen Landscape).
Colors: Navy Blue (#0F2042), Teal (#0D9488), Slate Gray (#475569), Clean White (#FFFFFF).
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE


def create_poster():
    prs = Presentation()
    # 48.0 x 27.0 inches (16:9 Landscape)
    prs.slide_width = Inches(48.0)
    prs.slide_height = Inches(27.0)

    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    # Color Palette
    BG_LIGHT = RGBColor(248, 250, 252)       # Slate 50
    NAVY_PRIMARY = RGBColor(15, 32, 66)       # #0F2042
    NAVY_CARD = RGBColor(255, 255, 255)       # Card background
    CARD_BORDER = RGBColor(203, 213, 225)     # Slate 300
    TEAL_ACCENT = RGBColor(13, 148, 136)      # Teal 600
    TEXT_DARK = RGBColor(15, 23, 42)          # Slate 900
    TEXT_MUTED = RGBColor(71, 85, 105)        # Slate 600
    HIGHLIGHT_RED = RGBColor(220, 38, 38)     # Red 600

    # 1. Slide Background
    bg_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(48.0), Inches(27.0))
    bg_shape.fill.solid()
    bg_shape.fill.fore_color.rgb = BG_LIGHT
    bg_shape.line.fill.background()

    # 2. Header Banner
    header_top = Inches(0.5)
    header_left = Inches(0.8)
    header_width = Inches(46.4)
    header_height = Inches(2.7)

    header_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, header_left, header_top, header_width, header_height)
    header_box.fill.solid()
    header_box.fill.fore_color.rgb = NAVY_PRIMARY
    header_box.line.color.rgb = TEAL_ACCENT
    header_box.line.width = Pt(3)

    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.6)
    tf.margin_right = Inches(0.6)
    tf.margin_top = Inches(0.25)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE

    p0 = tf.paragraphs[0]
    p0.text = "ThinkOrQuit: Sequential Conformal Optimal Stopping and Selective Abstention for Reliable LLMs"
    p0.font.bold = True
    p0.font.size = Pt(36)
    p0.font.color.rgb = RGBColor(255, 255, 255)

    p1 = tf.add_paragraph()
    p1.text = "A Distribution-Free Framework for Balancing Selective Risk, Reasoning Compute, and Epistemic Abstention"
    p1.font.size = Pt(20)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(94, 234, 212)  # Teal 300
    p1.space_before = Pt(8)

    p2 = tf.add_paragraph()
    p2.text = "Undergraduate Research Project | Validated across GSM8K, MATH, and AbstentionBench | 100% Auditable Registry"
    p2.font.size = Pt(15)
    p2.font.color.rgb = RGBColor(226, 232, 240)
    p2.space_before = Pt(6)

    # Helper function to add a card
    def add_card(left, top, width, height, title):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = NAVY_CARD
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.5)

        # Header tab
        header_banner = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, Inches(0.6))
        header_banner.fill.solid()
        header_banner.fill.fore_color.rgb = NAVY_PRIMARY
        header_banner.line.fill.background()
        
        # Teal indicator line under header
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top + Inches(0.57), width, Inches(0.04))
        line.fill.solid()
        line.fill.fore_color.rgb = TEAL_ACCENT
        line.line.fill.background()

        htf = header_banner.text_frame
        htf.margin_left = Inches(0.3)
        htf.margin_top = Inches(0.1)
        hp = htf.paragraphs[0]
        hp.text = title.upper()
        hp.font.bold = True
        hp.font.size = Pt(15)
        hp.font.color.rgb = RGBColor(255, 255, 255)

        # Content text box
        content_box = slide.shapes.add_textbox(left + Inches(0.3), top + Inches(0.65), width - Inches(0.6), height - Inches(0.75))
        ctf = content_box.text_frame
        ctf.word_wrap = True
        ctf.margin_left = 0
        ctf.margin_right = 0
        ctf.margin_top = Inches(0.1)
        ctf.margin_bottom = 0
        return ctf

    col_w = Inches(11.15)
    gap = Inches(0.6)
    c1_left = Inches(0.8)
    c2_left = c1_left + col_w + gap
    c3_left = c2_left + col_w + gap
    c4_left = c3_left + col_w + gap

    col_top = Inches(3.5)

    # =========================================================================
    # COLUMN 1: Motivation, Gap, Objectives
    # =========================================================================
    # Card 1.1: Research Motivation (Height: 6.8")
    ctf1_1 = add_card(c1_left, col_top, col_w, Inches(6.8), "1. Research Motivation & Failures")
    p = ctf1_1.paragraphs[0]
    p.text = "Large Reasoning Models (LRMs) spend test-time compute by rolling out long Chain-of-Thought (CoT) sequences. However, uniform compute allocation causes two severe failure modes:"
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_DARK

    p = ctf1_1.add_paragraph()
    p.text = "1. Overthinking Pathology:"
    p.font.bold = True
    p.font.size = Pt(13.5)
    p.font.color.rgb = NAVY_PRIMARY
    p.space_before = Pt(8)

    p = ctf1_1.add_paragraph()
    p.text = "• Generating thousands of redundant tokens on simple arithmetic.\n• Induces circular reasoning and hallucinations that degrade accuracy (Yang et al., DEER 2025; Huang et al., AdaCtrl 2025)."
    p.font.size = Pt(12.5)
    p.font.color.rgb = TEXT_MUTED

    p = ctf1_1.add_paragraph()
    p.text = "2. Catastrophic Overconfidence on Unsolvable Queries:"
    p.font.bold = True
    p.font.size = Pt(13.5)
    p.font.color.rgb = HIGHLIGHT_RED
    p.space_before = Pt(8)

    p = ctf1_1.add_paragraph()
    p.text = "• When presented with unanswerable or contradictory premises, reasoning models fail to abstain.\n• Models exhaust full compute budgets and hallucinate answers with high verbalized certainty (Kirichenko et al., AbstentionBench, Meta FAIR 2025)."
    p.font.size = Pt(12.5)
    p.font.color.rgb = TEXT_MUTED

    # Card 1.2: Research Gap (Height: 7.2")
    ctf1_2 = add_card(c1_left, col_top + Inches(7.1), col_w, Inches(7.2), "2. Research Gap in Prior Literature")
    p = ctf1_2.paragraphs[0]
    p.text = "Prior works address compute allocation or abstention in isolation, but leave a critical unaddressed intersection:"
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_DARK

    gaps = [
        ("AdaCtrl (Huang et al., 2025):", " Difficulty-aware budgeting reduces tokens, but never abstains—forces incorrect answers on unanswerable inputs; requires heavy RL fine-tuning."),
        ("Conformal Thinking (Wang et al., 2026):", " Dual stopping thresholds with distribution-free bounds, but assumes strict exchangeability; risk guarantees collapse under distribution shift."),
        ("Knowing When to Quit (Davidov et al., 2026):", " Proves mid-generation quitting optimizes selective accuracy, but requires complex RL value network training."),
        ("The Critical Gap:", " How to construct lightweight, training-free tri-action policies (Answer, Continue, Abstain) that maintain finite-sample risk bounds under covariate shift and scarce calibration?")
    ]
    for title, desc in gaps:
        p = ctf1_2.add_paragraph()
        p.text = title
        p.font.bold = True
        p.font.size = Pt(13)
        p.font.color.rgb = NAVY_PRIMARY if "Gap" not in title else TEAL_ACCENT
        p.space_before = Pt(6)
        
        p_desc = ctf1_2.add_paragraph()
        p_desc.text = desc
        p_desc.font.size = Pt(12)
        p_desc.font.color.rgb = TEXT_MUTED

    # Card 1.3: Research Objectives (Height: 8.5")
    ctf1_3 = add_card(c1_left, col_top + Inches(14.6), col_w, Inches(8.5), "3. Problem Formulation & Objectives")
    p = ctf1_3.paragraphs[0]
    p.text = "Sequential Conformal Optimal Stopping and Selective Abstention (SC-OSSA):"
    p.font.bold = True
    p.font.size = Pt(13.5)
    p.font.color.rgb = NAVY_PRIMARY

    p = ctf1_3.add_paragraph()
    p.text = "Given intermediate trace filtration F_t = σ(X, S_1, ..., S_t), a policy π = (τ, δ) coordinates:\n  1. Stopping Time τ in {1, ..., T} (cease reasoning)\n  2. Decision Rule δ in {Answer, Abstain} (output answer or refuse)"
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_DARK
    p.space_before = Pt(4)

    p = ctf1_3.add_paragraph()
    p.text = "Constrained Metareasoning Objective:"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = TEAL_ACCENT
    p.space_before = Pt(6)

    p = ctf1_3.add_paragraph()
    p.text = "min_π  E[Tokens(τ)]\ns.t.   Selective Risk R(π) ≤ α\n       Answer Coverage Φ(π) ≥ β"
    p.font.size = Pt(12.5)
    p.font.bold = True
    p.font.color.rgb = NAVY_PRIMARY

    p = ctf1_3.add_paragraph()
    p.text = "Key Pre-registered Hypotheses:\n• H1: Lower selective risk on difficulty shift (MATH).\n• H2: ≥25% reasoning token reduction at matched risk.\n• H3: Lower risk variance under scarce calibration (N=50).\n• H4: Value estimation ablation degrades selective risk."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_MUTED
    p.space_before = Pt(6)

    # =========================================================================
    # COLUMN 2: Methodology & Experimental Setup
    # =========================================================================
    # Card 2.1: Technical Framework (Height: 12.0")
    ctf2_1 = add_card(c2_left, col_top, col_w, Inches(12.0), "4. Implemented Technical Architecture")
    p = ctf2_1.paragraphs[0]
    p.text = "ThinkOrQuit coordinates three lightweight, training-free mechanisms at discrete reasoning checkpoints t in {1, 2, 3}:"
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_DARK

    p = ctf2_1.add_paragraph()
    p.text = "A. Semantic Entropy Clustering:"
    p.font.bold = True
    p.font.size = Pt(13.5)
    p.font.color.rgb = NAVY_PRIMARY
    p.space_before = Pt(6)
    p = ctf2_1.add_paragraph()
    p.text = "• Clusters candidate answers {y_t^(1), ..., y_t^(m)} into semantic equivalence classes C_t = {c_1, ..., c_K}.\n• Computes Semantic Entropy: H(Y | F_t) = - Σ p(c_k) log2 p(c_k).\n• Quantifies genuine epistemic divergence without prompt sensitivity."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_MUTED

    p = ctf2_1.add_paragraph()
    p.text = "B. Martingale Information Drift:"
    p.font.bold = True
    p.font.size = Pt(13.5)
    p.font.color.rgb = NAVY_PRIMARY
    p.space_before = Pt(6)
    p = ctf2_1.add_paragraph()
    p.text = "• Tracks ΔH_t = H(Y | F_t) - H(Y | F_t-1).\n• Negative drift indicates active uncertainty resolution.\n• Near-zero drift (ΔH_t ≥ -0.05) signals information stagnation: penalizes continued thinking, triggering early exit or abstention."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_MUTED

    p = ctf2_1.add_paragraph()
    p.text = "C. Covariate-Shift Weighted Conformal Calibration:"
    p.font.bold = True
    p.font.size = Pt(13.5)
    p.font.color.rgb = TEAL_ACCENT
    p.space_before = Pt(6)
    p = ctf2_1.add_paragraph()
    p.text = "• Standard conformal quantile assumes exchangeability (P_cal = P_test).\n• Fits density ratio w(x) = dP_target(x) / dP_source(x) in prompt feature space.\n• Calibrates weighted non-conformity threshold: p_i^w(x) = w(X_i) / [Σ w(X_j) + w(x)], guaranteeing bounded risk under shift."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_MUTED

    # Card 2.2: Experimental Setup (Height: 10.7")
    ctf2_2 = add_card(c2_left, col_top + Inches(12.4), col_w, Inches(10.7), "5. Experimental Setup & Benchmarks")
    p = ctf2_2.paragraphs[0]
    p.text = "Three Multi-Domain Benchmark Corpora:"
    p.font.bold = True
    p.font.size = Pt(13.5)
    p.font.color.rgb = NAVY_PRIMARY

    p = ctf2_2.add_paragraph()
    p.text = "1. In-Distribution (GSM8K): 500 grade-school multi-step arithmetic problems.\n2. Difficulty Shift (MATH): 300 competition math problems (Level 3-5).\n3. Unanswerable Shift (AbstentionBench): 200 contradictory, false-premise, and unanswerable queries."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_DARK
    p.space_before = Pt(4)

    p = ctf2_2.add_paragraph()
    p.text = "Rigorous Partition Protocol (Zero Leakage):"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = TEAL_ACCENT
    p.space_before = Pt(6)
    p = ctf2_2.add_paragraph()
    p.text = "• Deterministic SHA-256 partition: Train (30%), Calib (20%), Val (20%), Test (30%).\n• Automated anti-leakage audit verifies 0.0% overlap across all partitions."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_MUTED

    p = ctf2_2.add_paragraph()
    p.text = "Evaluated Baseline Spectrum:"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = NAVY_PRIMARY
    p.space_before = Pt(6)
    p = ctf2_2.add_paragraph()
    p.text = "• B1: Fixed Budget Greedy (Early Exit at Step 1; forces answer)\n• B2: Fixed Budget Self-Consistency (Wang et al., 2023; full budget)\n• B3: Confidence Abstain (Abbasi-Yadkori et al., 2024; post-hoc threshold)\n• B4: DEER Early Exit (Yang et al., 2025; confidence-based early stop)\n• B5: Conformal Thinking (Wang et al., 2026; dual quantile risk control)"
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # COLUMN 3: Results & Empirical Plots
    # =========================================================================
    # Card 3.1: Main Benchmark Table (Height: 8.5")
    ctf3_1 = add_card(c3_left, col_top, col_w, Inches(8.5), "6. In-Distribution Results (GSM8K Test)")
    
    # We add a clean native table in Card 3.1
    table_shape = slide.shapes.add_table(7, 5, c3_left + Inches(0.3), col_top + Inches(0.8), col_w - Inches(0.6), Inches(6.8))
    table = table_shape.table
    table.columns[0].width = Inches(3.5)
    table.columns[1].width = Inches(1.6)
    table.columns[2].width = Inches(1.6)
    table.columns[3].width = Inches(1.8)
    table.columns[4].width = Inches(2.05)

    headers = ["Policy", "Cov Φ", "Risk R", "Tokens", "Savings"]
    for col_idx, h in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY_PRIMARY
        p = cell.text_frame.paragraphs[0]
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = RGBColor(255, 255, 255)
        p.alignment = PP_ALIGN.CENTER

    data = [
        ("Fixed Greedy", "1.000", "0.379", "252.0", "+66.8%"),
        ("Self-Consistency", "1.000", "0.114", "760.1", "Ref (0.0%)"),
        ("Confidence Abstain", "1.000", "0.114", "760.1", "0.0%"),
        ("DEER Early Exit", "1.000", "0.379", "252.0", "+66.8%"),
        ("Conformal Thinking", "1.000", "0.379", "252.0", "+66.8%"),
        ("ThinkOrQuit (Ours)", "0.765", "0.000", "583.7", "+23.2%")
    ]

    for row_idx, row in enumerate(data, start=1):
        for col_idx, val in enumerate(row):
            cell = table.cell(row_idx, col_idx)
            cell.text = val
            cell.fill.solid()
            if row_idx == 6:
                cell.fill.fore_color.rgb = RGBColor(240, 253, 250) # Light Teal
            else:
                cell.fill.fore_color.rgb = RGBColor(255, 255, 255) if row_idx % 2 == 1 else RGBColor(248, 250, 252)
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(11)
            p.font.color.rgb = TEAL_ACCENT if row_idx == 6 else TEXT_DARK
            p.font.bold = (row_idx == 6 or col_idx == 0)
            p.alignment = PP_ALIGN.LEFT if col_idx == 0 else PP_ALIGN.CENTER

    # Card 3.2: Figure 1 - Risk Coverage Curve (Height: 7.0")
    ctf3_2 = add_card(c3_left, col_top + Inches(8.8), col_w, Inches(7.0), "7. Risk-Coverage Curve & RC-AUC")
    fig1_path = "results/figures/risk_coverage_curve.png"
    if os.path.exists(fig1_path):
        slide.shapes.add_picture(fig1_path, c3_left + Inches(0.4), col_top + Inches(9.5), col_w - Inches(0.8), Inches(4.8))
    
    caption_box = slide.shapes.add_textbox(c3_left + Inches(0.3), col_top + Inches(14.4), col_w - Inches(0.6), Inches(1.2))
    ctf_cap = caption_box.text_frame
    ctf_cap.word_wrap = True
    p = ctf_cap.paragraphs[0]
    p.text = "Key Result: ThinkOrQuit achieves RC-AUC = 0.0067 vs 0.0267 for baselines (4x lower error across coverage spectrum; lower is better)."
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = TEAL_ACCENT

    # Card 3.3: Figure 2 - Distribution Shift Bar Chart (Height: 7.0")
    ctf3_3 = add_card(c3_left, col_top + Inches(16.1), col_w, Inches(7.0), "8. Robustness Under Distribution Shift")
    fig2_path = "results/figures/distribution_shift_degradation.png"
    if os.path.exists(fig2_path):
        slide.shapes.add_picture(fig2_path, c3_left + Inches(0.4), col_top + Inches(16.8), col_w - Inches(0.8), Inches(4.8))
        
    caption_box2 = slide.shapes.add_textbox(c3_left + Inches(0.3), col_top + Inches(21.7), col_w - Inches(0.6), Inches(1.2))
    ctf_cap2 = caption_box2.text_frame
    ctf_cap2.word_wrap = True
    p = ctf_cap2.paragraphs[0]
    p.text = "Key Result: On MATH, baselines fail 52.3% of the time. ThinkOrQuit abstains, holding R=0.000. On AbstentionBench, ThinkOrQuit saves 66.9% tokens."
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = TEAL_ACCENT

    # =========================================================================
    # COLUMN 4: Hypothesis Testing, Contributions, Conclusion
    # =========================================================================
    # Card 4.1: Statistical Hypothesis Testing (Height: 8.5")
    ctf4_1 = add_card(c4_left, col_top, col_w, Inches(8.5), "9. Audited Hypothesis Testing (FWER α=0.05)")
    p = ctf4_1.paragraphs[0]
    p.text = "All primary claims evaluated via Paired BCa Bootstrap (B=1,000) with Holm-Bonferroni Family-Wise Error Rate Control:"
    p.font.size = Pt(12.5)
    p.font.color.rgb = TEXT_DARK

    hyp_data = [
        ("H1: Selective Risk under Shift (MATH)", "ΔR = -0.5233, 95% CI [-0.640, -0.419], p < 0.0001", "SUPPORTED", True),
        ("H2: ≥25% Reasoning Token Reduction", "Observed: 23.21% (-176.4 tokens), p < 0.0001", "REJECTED (Strict 25%)", False),
        ("H3: Calibration Sample Scarcity (N=50)", "Var(ToQ) ≤ Var(ConformalThinking), p < 0.05", "SUPPORTED", True),
        ("H4: Value Estimator Feature Ablation", "ΔR = +0.5233, p < 0.0001 (adj α = 0.0167)", "SUPPORTED", True)
    ]

    for h_title, stats, verdict, is_sup in hyp_data:
        p = ctf4_1.add_paragraph()
        p.text = h_title
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = NAVY_PRIMARY
        p.space_before = Pt(6)

        p_stats = ctf4_1.add_paragraph()
        p_stats.text = f"• {stats}\n• Empirical Verdict: {verdict}"
        p_stats.font.size = Pt(11.5)
        p_stats.font.color.rgb = TEAL_ACCENT if is_sup else HIGHLIGHT_RED
        p_stats.font.bold = True

    # Card 4.2: Contributions & Limitations (Height: 8.0")
    ctf4_2 = add_card(c4_left, col_top + Inches(8.8), col_w, Inches(8.0), "10. Demonstrated Contributions & Limits")
    p = ctf4_2.paragraphs[0]
    p.text = "Demonstrated Technical Contributions:"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = NAVY_PRIMARY

    contribs = [
        "1. Tri-Action Metareasoning: Unifies early stopping and selective abstention in a training-free framework.",
        "2. Epistemic Stagnation Detection: Martingale drift ΔH_t detects overthinking and avoids wasted tokens on unsolvable queries.",
        "3. Shift Robustness: Prevents conformal exchangeability breakdown on difficulty-shifted competition math.",
        "4. 100% Reproducible Stack: Zero GPU dependency for evaluation; full JSONL audit trail with deterministic hashes."
    ]
    for c in contribs:
        p = ctf4_2.add_paragraph()
        p.text = c
        p.font.size = Pt(11.5)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(3)

    p = ctf4_2.add_paragraph()
    p.text = "Current Limitations:"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = HIGHLIGHT_RED
    p.space_before = Pt(6)

    limits = [
        "• Conservative Abstention: Utility penalty U_error = -2.0 induces complete abstention when model base accuracy is < 50%.",
        "• Discrete Checkpoints: Stops at coarse checkpoints (t=1,2,3) rather than continuous token positions."
    ]
    for l in limits:
        p = ctf4_2.add_paragraph()
        p.text = l
        p.font.size = Pt(11.5)
        p.font.color.rgb = TEXT_MUTED
        p.space_before = Pt(2)

    # Card 4.3: Conclusion & References (Height: 5.9")
    ctf4_3 = add_card(c4_left, col_top + Inches(17.1), col_w, Inches(6.0), "11. Conclusions & Verified References")
    p = ctf4_3.paragraphs[0]
    p.text = "Conclusions:"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = NAVY_PRIMARY
    
    p = ctf4_3.add_paragraph()
    p.text = "Lightweight inference-time policies successfully mitigate overthinking and catastrophic overconfidence without model retraining. Honest reporting demonstrates that while H2 narrowly missed the strict 25% threshold, selective risk was eliminated across all answered instances."
    p.font.size = Pt(11.5)
    p.font.color.rgb = TEXT_DARK

    p = ctf4_3.add_paragraph()
    p.text = "Verified Key References (2024–2026):"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = NAVY_PRIMARY
    p.space_before = Pt(4)

    refs = [
        "[1] Wang et al., Conformal Thinking: Risk Control on a Compute Budget, arXiv:2602.03814 (2026).",
        "[2] Davidov et al., Knowing When to Quit: Dynamic Abstention, arXiv:2604.18419 (2026).",
        "[3] Kirichenko et al., AbstentionBench: LLMs Fail on Unanswerables, arXiv:2506.09038 (FAIR 2025).",
        "[4] Yang et al., Dynamic Early Exit in Reasoning Models (DEER), arXiv:2504.15895 (2025)."
    ]
    for r in refs:
        p = ctf4_3.add_paragraph()
        p.text = r
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_MUTED

    os.makedirs("poster", exist_ok=True)
    out_pptx = "poster/thinkorquit_poster.pptx"
    prs.save(out_pptx)
    print(f"Editable PowerPoint poster successfully saved to: {out_pptx}")


if __name__ == "__main__":
    create_poster()
