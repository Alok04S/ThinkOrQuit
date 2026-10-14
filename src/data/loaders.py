"""
Dataset loaders and answer normalization logic for GSM8K, MATH, and AbstentionBench.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


def normalize_answer(ans: Optional[str]) -> str:
    """
    Normalize an extracted mathematical or textual answer string.
    
    Steps:
    1. Strip LaTeX delimiters ($...$, \\boxed{...}, \\text{...}).
    2. Normalize commas in numbers (e.g., '1,000' -> '1000').
    3. Normalize whitespace and lowercases text.
    4. Handle simple fractions into standard float representation if possible.
    """
    if ans is None:
        return ""
    
    s = str(ans).strip()
    
    # Check for explicit abstention markers
    if s.upper() in ["[ABSTAIN]", "ABSTAIN", "UNANSWERABLE", "I DON'T KNOW", "CANNOT BE DETERMINED"]:
        return "[ABSTAIN]"
    
    # Extract from LaTeX \boxed{...} if present
    boxed_match = re.search(r"\\boxed\{([^{}]+)\}", s)
    if boxed_match:
        s = boxed_match.group(1).strip()
        
    # Strip common LaTeX wrappers
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"[\$]", "", s)
    
    # Remove thousand-separator commas in numeric expressions: 1,234 -> 1234
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    
    # Remove trailing periods or punctuation
    s = s.rstrip(".").strip()
    
    # Check if numeric fraction like '1/2' or '3/4'
    frac_match = re.fullmatch(r"(-?\d+)\s*/\s*(\d+)", s)
    if frac_match:
        num, den = float(frac_match.group(1)), float(frac_match.group(2))
        if den != 0:
            val = num / den
            if val.is_integer():
                return str(int(val))
            return f"{val:.6g}"

    # Try numeric conversion
    try:
        val = float(s)
        if val.is_integer():
            return str(int(val))
        return f"{val:.6g}"
    except ValueError:
        pass

    return s.lower().strip()


def is_answer_correct(pred: Optional[str], target: Optional[str]) -> bool:
    """
    Evaluate equivalence between predicted answer and ground truth target.
    Handles exact string match, normalized float equivalence, and abstention matching.
    """
    norm_pred = normalize_answer(pred)
    norm_target = normalize_answer(target)

    if not norm_pred or not norm_target:
        return False
        
    if norm_pred == norm_target:
        return True

    # Numeric tolerance check
    try:
        val_pred = float(norm_pred)
        val_target = float(norm_target)
        return abs(val_pred - val_target) < 1e-4
    except ValueError:
        return False


def extract_gsm8k_target(solution_text: str) -> str:
    """Extract final answer from GSM8K solution (delimited by ####)."""
    if "####" in solution_text:
        return solution_text.split("####")[-1].strip()
    match = re.search(r"[Tt]he answer is:?\s*(\S+)", solution_text)
    if match:
        return match.group(1).strip()
    return solution_text.strip().split()[-1] if solution_text else ""


def extract_math_target(solution_text: str) -> str:
    """Extract boxed target from MATH benchmark solution."""
    match = re.findall(r"\\boxed\{([^{}]+)\}", solution_text)
    if match:
        return match[-1].strip()
    return extract_gsm8k_target(solution_text)


def load_dataset(file_path: str, dataset_type: str = "gsm8k") -> List[Dict[str, Any]]:
    """
    Load and parse a benchmark dataset from a JSONL file.
    
    Returns list of dicts with standardized keys:
    - id: str
    - question: str
    - target: str
    - normalized_target: str
    - is_unanswerable: bool
    - metadata: dict
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            
            item_id = str(item.get("id", f"{dataset_type}_{idx:05d}"))
            question = item.get("question", item.get("problem", "")).strip()
            
            # Ground truth resolution
            if dataset_type == "abstentionbench" or item.get("is_unanswerable", False):
                target = "[ABSTAIN]"
                is_unans = True
            elif "target" in item:
                target = str(item["target"])
                is_unans = target.upper() == "[ABSTAIN]"
            elif "answer" in item:
                target = extract_gsm8k_target(item["answer"])
                is_unans = False
            elif "solution" in item:
                target = extract_math_target(item["solution"])
                is_unans = False
            else:
                target = ""
                is_unans = False

            norm_target = normalize_answer(target)
            
            records.append({
                "id": item_id,
                "question": question,
                "target": target,
                "normalized_target": norm_target,
                "is_unanswerable": is_unans,
                "metadata": item.get("metadata", {})
            })

    return records
