"""
Reproducibility utilities: random seed management, environment fingerprinting, and cryptographic hashing.
"""

import hashlib
import os
import platform
import random
import sys
from typing import Any, Dict
import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set seeds for all random number generators to ensure full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def deterministic_hash(text: str) -> str:
    """Compute a deterministic SHA-256 hash of a string."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def get_environment_fingerprint() -> Dict[str, Any]:
    """Capture system and dependency environment metadata for experiment auditing."""
    fingerprint = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": {}
    }
    
    for pkg in ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "yaml"]:
        try:
            mod = __import__(pkg)
            fingerprint["packages"][pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            fingerprint["packages"][pkg] = "not_installed"
            
    return fingerprint
