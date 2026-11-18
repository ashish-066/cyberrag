"""
Agent State and Step Trace definitions for CyberRAG.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict


class AgentState(str, Enum):
    QUERY_ANALYSIS = "QUERY_ANALYSIS"
    RETRIEVAL = "RETRIEVAL"
    EVIDENCE_INSPECTION = "EVIDENCE_INSPECTION"
    RETRY_EXPANSION = "RETRY_EXPANSION"
    VERIFICATION = "VERIFICATION"
    CONFLICT_RESOLUTION = "CONFLICT_RESOLUTION"
    ANSWER_GENERATION = "ANSWER_GENERATION"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass
class AgentStepTrace:
    step_index: int
    state: AgentState
    thought: str
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output_summary: str
    action: str = ""
    decision_outcome: str = ""
    retry_or_termination_reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
