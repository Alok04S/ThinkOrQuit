"""
Deterministic dataset partitioning and calibration subset creation.
"""

from dataclasses import dataclass
import hashlib
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class DataSplit:
    train: List[Dict[str, Any]]
    calibration: List[Dict[str, Any]]
    validation: List[Dict[str, Any]]
    test: List[Dict[str, Any]]

    def summary(self) -> Dict[str, int]:
        return {
            "train_size": len(self.train),
            "calib_size": len(self.calibration),
            "val_size": len(self.validation),
            "test_size": len(self.test),
            "total_size": len(self.train) + len(self.calibration) + len(self.validation) + len(self.test)
        }


def hash_partition(key: str, seed: int = 42) -> float:
    """Map string key to a deterministic uniform float in [0, 1)."""
    salted = f"{seed}_{key.strip()}".encode("utf-8")
    hex_digest = hashlib.sha256(salted).hexdigest()
    int_val = int(hex_digest[:8], 16)
    return int_val / float(0xFFFFFFFF)


def create_deterministic_splits(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.30,
    calib_ratio: float = 0.20,
    val_ratio: float = 0.20,
    test_ratio: float = 0.30,
    seed: int = 42
) -> DataSplit:
    """
    Split records deterministically by hashing question text.
    Guarantees no data leakage across partitions.
    """
    assert abs((train_ratio + calib_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    t_bound = train_ratio
    c_bound = t_bound + calib_ratio
    v_bound = c_bound + val_ratio

    train_set, calib_set, val_set, test_set = [], [], [], []

    for item in records:
        key = item.get("question", item["id"])
        h = hash_partition(key, seed=seed)

        if h < t_bound:
            train_set.append(item)
        elif h < c_bound:
            calib_set.append(item)
        elif h < v_bound:
            val_set.append(item)
        else:
            test_set.append(item)

    return DataSplit(
        train=train_set,
        calibration=calib_set,
        validation=val_set,
        test=test_set
    )


def sample_calibration_subset(
    calibration_data: List[Dict[str, Any]],
    subsample_size: int,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Subsample calibration set deterministically to simulate data scarcity (N in {50, 100, 250, 500}).
    """
    if subsample_size >= len(calibration_data):
        return list(calibration_data)
        
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(calibration_data), size=subsample_size, replace=False)
    return [calibration_data[i] for i in sorted(indices)]
