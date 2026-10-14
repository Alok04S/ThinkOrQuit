"""
Trace collector: gathers multi-step reasoning traces with candidate samples, token expenditures,
and semantic entropy at discrete decision checkpoints.
"""

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional
from src.data.loaders import normalize_answer, is_answer_correct
from src.models.semantic_entropy import compute_semantic_entropy, compute_martingale_drift


def extract_answer_from_text(text: str) -> str:
    """Extract final candidate answer from reasoning text."""
    # Look for \boxed{...}
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    
    # Look for "#### <ans>"
    if "####" in text:
        return text.split("####")[-1].strip()
        
    # Look for "the answer is: <ans>"
    match = re.search(r"[Tt]he (?:final )?answer is:?\s*([^\n\.,]+)", text)
    if match:
        return match.group(1).strip()

    # Fallback to last token/number
    tokens = text.strip().split()
    return tokens[-1] if tokens else ""


def build_trace_record(
    item: Dict[str, Any],
    checkpoints_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Construct a unified reasoning trace dictionary from checkpoint observations.
    
    checkpoints_data contains list of:
    {
        "step_index": int,
        "token_count": int,
        "candidate_answers": List[str],
        "verbalized_confidence": float
    }
    """
    item_id = item["id"]
    question = item["question"]
    target = item["target"]
    norm_target = item["normalized_target"]
    is_unans = item["is_unanswerable"]

    entropies = []
    checkpoint_records = []

    for cp in checkpoints_data:
        candidates = cp["candidate_answers"]
        entropy, majority_ans, majority_ratio = compute_semantic_entropy(candidates)
        entropies.append(entropy)
        
        checkpoint_records.append({
            "step_index": cp["step_index"],
            "token_count": cp["token_count"],
            "candidate_answers": candidates,
            "majority_answer": majority_ans,
            "majority_confidence": majority_ratio,
            "semantic_entropy": entropy,
            "martingale_drift": 0.0,  # updated next
            "verbalized_confidence": cp.get("verbalized_confidence", majority_ratio)
        })

    drifts = compute_martingale_drift(entropies)
    for idx, d in enumerate(drifts):
        checkpoint_records[idx]["martingale_drift"] = d

    # Final correctness assessment
    final_cp = checkpoint_records[-1] if checkpoint_records else None
    final_ans = final_cp["majority_answer"] if final_cp else ""
    is_correct = is_answer_correct(final_ans, norm_target) if not is_unans else False
    total_tokens = final_cp["token_count"] if final_cp else 0

    return {
        "id": item_id,
        "question": question,
        "target": target,
        "normalized_target": norm_target,
        "is_unanswerable": is_unans,
        "checkpoints": checkpoint_records,
        "total_tokens_full": total_tokens,
        "final_ground_truth_correct": is_correct
    }
