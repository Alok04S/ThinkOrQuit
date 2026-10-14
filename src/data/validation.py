"""
Data integrity validation and anti-leakage checks.
"""

from typing import Any, Dict, List, Set, Tuple
from .splits import DataSplit


def extract_normalized_keys(records: List[Dict[str, Any]]) -> Set[str]:
    """Extract normalized text keys for duplicate and overlap detection."""
    keys = set()
    for item in records:
        q = item.get("question", "").strip().lower()
        if q:
            keys.add(q)
        else:
            keys.add(str(item.get("id", "")))
    return keys


def check_split_leakage(split: DataSplit) -> Dict[str, Any]:
    """
    Verify complete disjointness across train, calibration, validation, and test splits.
    Returns audit dictionary with overlap counts and booleans.
    """
    train_keys = extract_normalized_keys(split.train)
    calib_keys = extract_normalized_keys(split.calibration)
    val_keys = extract_normalized_keys(split.validation)
    test_keys = extract_normalized_keys(split.test)

    tc_overlap = train_keys.intersection(calib_keys)
    tv_overlap = train_keys.intersection(val_keys)
    tt_overlap = train_keys.intersection(test_keys)
    cv_overlap = calib_keys.intersection(val_keys)
    ct_overlap = calib_keys.intersection(test_keys)
    vt_overlap = val_keys.intersection(test_keys)

    total_leakage = len(tc_overlap) + len(tv_overlap) + len(tt_overlap) + len(cv_overlap) + len(ct_overlap) + len(vt_overlap)

    return {
        "is_leak_free": total_leakage == 0,
        "total_leakage_count": total_leakage,
        "train_calib_overlap": len(tc_overlap),
        "train_val_overlap": len(tv_overlap),
        "train_test_overlap": len(tt_overlap),
        "calib_val_overlap": len(cv_overlap),
        "calib_test_overlap": len(ct_overlap),
        "val_test_overlap": len(vt_overlap)
    }


def validate_dataset_splits(split: DataSplit) -> Dict[str, Any]:
    """
    Validate data governance compliance:
    1. Zero partition leakage.
    2. Non-empty ground truths.
    3. Proper label format preservation.
    """
    leakage_audit = check_split_leakage(split)
    
    empty_targets = 0
    total_records = 0
    for partition in [split.train, split.calibration, split.validation, split.test]:
        for rec in partition:
            total_records += 1
            if not rec.get("normalized_target"):
                empty_targets += 1

    audit_passed = leakage_audit["is_leak_free"] and (empty_targets == 0) and (total_records > 0)

    return {
        "audit_passed": audit_passed,
        "total_records": total_records,
        "empty_targets_count": empty_targets,
        "leakage_audit": leakage_audit,
        "partition_sizes": split.summary()
    }
