"""
Model inference, semantic entropy quantification, and trace collection.
"""

from .inference import ReasoningCheckpoint, ReasoningTrace, TraceReplayEngine
from .semantic_entropy import (
    cluster_equivalent_answers,
    compute_semantic_entropy,
    compute_martingale_drift
)
from .trace_collector import extract_answer_from_text, build_trace_record

__all__ = [
    "ReasoningCheckpoint",
    "ReasoningTrace",
    "TraceReplayEngine",
    "cluster_equivalent_answers",
    "compute_semantic_entropy",
    "compute_martingale_drift",
    "extract_answer_from_text",
    "build_trace_record"
]
