"""
Unit tests for data loaders, deterministic hash splitting, and anti-leakage validation.
"""

from src.data.loaders import normalize_answer, is_answer_correct
from src.data.splits import create_deterministic_splits, sample_calibration_subset
from src.data.validation import check_split_leakage, validate_dataset_splits


def test_normalize_answer():
    assert normalize_answer("\\boxed{42}") == "42"
    assert normalize_answer("1,000") == "1000"
    assert normalize_answer("3/6") == "0.5"
    assert normalize_answer("  $15.0$  ") == "15"
    assert normalize_answer("[ABSTAIN]") == "[ABSTAIN]"
    assert normalize_answer("I don't know") == "[ABSTAIN]"
    assert normalize_answer("cannot be determined") == "[ABSTAIN]"


def test_is_answer_correct():
    assert is_answer_correct("42", "42") is True
    assert is_answer_correct("0.5", "1/2") is True
    assert is_answer_correct("1000", "1,000") is True
    assert is_answer_correct("[ABSTAIN]", "[ABSTAIN]") is True
    assert is_answer_correct("42", "43") is False
    assert is_answer_correct("42", "[ABSTAIN]") is False


def test_deterministic_split_and_zero_leakage():
    fake_records = [
        {"id": f"q_{i}", "question": f"What is {i} plus {i*2}?", "target": str(i*3), "normalized_target": str(i*3), "is_unanswerable": False}
        for i in range(100)
    ]

    split1 = create_deterministic_splits(fake_records, train_ratio=0.3, calib_ratio=0.2, val_ratio=0.2, test_ratio=0.3, seed=42)
    split2 = create_deterministic_splits(fake_records, train_ratio=0.3, calib_ratio=0.2, val_ratio=0.2, test_ratio=0.3, seed=42)

    # Determinism
    assert len(split1.train) == len(split2.train)
    assert [r["id"] for r in split1.train] == [r["id"] for r in split2.train]

    # Zero leakage
    audit = check_split_leakage(split1)
    assert audit["is_leak_free"] is True
    assert audit["total_leakage_count"] == 0

    val_report = validate_dataset_splits(split1)
    assert val_report["audit_passed"] is True


def test_calibration_subsampling():
    fake_calib = [{"id": f"c_{i}", "question": f"q_{i}"} for i in range(100)]
    sub_50 = sample_calibration_subset(fake_calib, subsample_size=50, seed=42)
    assert len(sub_50) == 50
    assert len(set(r["id"] for r in sub_50)) == 50
