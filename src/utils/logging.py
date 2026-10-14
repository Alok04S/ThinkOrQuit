"""
Experiment logging and registry management utilities.
"""

import datetime
import json
import logging
import os
from typing import Any, Dict, Optional


def setup_logger(name: str = "ThinkOrQuit", log_file: Optional[str] = None) -> logging.Logger:
    """Set up a standardized logger with console and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def log_experiment_to_registry(
    registry_path: str,
    experiment_id: str,
    hypothesis: str,
    config: Dict[str, Any],
    status: str,
    metrics: Dict[str, Any],
    output_artifacts: Dict[str, str],
    notes: Optional[str] = None
) -> None:
    """
    Append an immutable, auditable experiment record to the experiment registry JSONL.
    
    Fields comply with Phase Nine requirements:
    - experiment_id
    - timestamp
    - hypothesis
    - config
    - status (PLANNED, RUNNING, COMPLETED, FAILED, VERIFIED)
    - metrics
    - output_artifacts
    - notes
    """
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    
    record = {
        "experiment_id": experiment_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hypothesis": hypothesis,
        "status": status,
        "config": config,
        "metrics": metrics,
        "output_artifacts": output_artifacts,
        "notes": notes or ""
    }
    
    with open(registry_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
