"""
Standardized inference wrapper supporting both online HuggingFace models and offline trace replay.
"""

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List, Optional


@dataclass
class ReasoningCheckpoint:
    step_index: int
    token_count: int
    candidate_answers: List[str]
    majority_answer: str
    majority_confidence: float
    semantic_entropy: float
    martingale_drift: float
    verbalized_confidence: float


@dataclass
class ReasoningTrace:
    id: str
    question: str
    target: str
    normalized_target: str
    is_unanswerable: bool
    checkpoints: List[ReasoningCheckpoint]
    total_tokens_full: int
    final_ground_truth_correct: bool


class TraceReplayEngine:
    """
    Offline trace replay engine that provides fast, deterministic, reproducible access
    to multi-step reasoning traces across datasets.
    """
    def __init__(self, trace_file_path: str):
        self.trace_file_path = trace_file_path
        self.traces: Dict[str, ReasoningTrace] = {}
        self._load_traces()

    def _load_traces(self) -> None:
        if not os.path.exists(self.trace_file_path):
            raise FileNotFoundError(f"Trace file not found: {self.trace_file_path}")

        with open(self.trace_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                
                checkpoints = []
                for cp in d.get("checkpoints", []):
                    checkpoints.append(ReasoningCheckpoint(
                        step_index=cp["step_index"],
                        token_count=cp["token_count"],
                        candidate_answers=cp["candidate_answers"],
                        majority_answer=cp["majority_answer"],
                        majority_confidence=cp["majority_confidence"],
                        semantic_entropy=cp["semantic_entropy"],
                        martingale_drift=cp["martingale_drift"],
                        verbalized_confidence=cp.get("verbalized_confidence", cp["majority_confidence"])
                    ))

                trace = ReasoningTrace(
                    id=d["id"],
                    question=d["question"],
                    target=d["target"],
                    normalized_target=d["normalized_target"],
                    is_unanswerable=d.get("is_unanswerable", False),
                    checkpoints=checkpoints,
                    total_tokens_full=d["total_tokens_full"],
                    final_ground_truth_correct=d["final_ground_truth_correct"]
                )
                self.traces[trace.id] = trace

    def get_trace(self, item_id: str) -> Optional[ReasoningTrace]:
        return self.traces.get(item_id)

    def get_all_traces(self) -> List[ReasoningTrace]:
        return list(self.traces.values())
