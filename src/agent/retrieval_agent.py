"""CyberRAG bounded verification workflow.

This is a fixed, deterministic pipeline (plan -> retrieve -> entity gate ->
NLI -> conflict resolution -> brief), NOT an autonomous ReAct agent: the LLM
may help plan and summarise but never decides control flow or verdicts.

All logic lives in ``src.core.service.VerificationService``; this class only
adapts its output to the report/trace shape the Streamlit UI and the FastAPI
workbench render. Every sub-claim is verified (H5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agent.agent_state import AgentState, AgentStepTrace
from src.core.service import VerificationService

_STEP_STATE = {
    "resolve_entities": AgentState.QUERY_ANALYSIS,
    "retrieve": AgentState.RETRIEVAL,
    "entity_gate": AgentState.EVIDENCE_INSPECTION,
    "dedupe": AgentState.EVIDENCE_INSPECTION,
    "date_filter": AgentState.EVIDENCE_INSPECTION,
    "nli": AgentState.VERIFICATION,
    "non_english_short_circuit": AgentState.INSUFFICIENT_EVIDENCE,
}


@dataclass
class GroundedReport:
    query: str
    status: str  # overall status
    support_tier: str  # NONE | SINGLE_SOURCE | CORROBORATED | CONFLICTING (strongest across claims)
    source_diversity: int  # distinct corpus publishers among supporting evidence
    supporting_evidence: List[Dict[str, Any]]
    contradicting_evidence: List[Dict[str, Any]]
    conflict_resolution: Optional[Dict[str, Any]]
    state_history: List[AgentState]
    execution_trace: List[AgentStepTrace]
    narrative: str
    claim_results: List[Dict[str, Any]] = field(default_factory=list)
    plan: Dict[str, Any] = field(default_factory=dict)
    related_entities: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


_TIER_ORDER = ["NONE", "SINGLE_SOURCE", "CORROBORATED", "CONFLICTING"]


class CyberRAGAgent:
    """Bounded verification workflow over the shared VerificationService."""

    def __init__(self, service: VerificationService):
        self.service = service

    def run(self, query: str, session_id: Optional[str] = None) -> GroundedReport:
        out = self.service.verify_question(query, session_id=session_id)
        results: List[Dict[str, Any]] = out["results"]
        trace: List[AgentStepTrace] = []
        states: List[AgentState] = []

        def add(state: AgentState, tool: str, tool_in: Dict[str, Any], summary: str, outcome: str = "") -> None:
            states.append(state)
            trace.append(
                AgentStepTrace(
                    step_index=len(trace) + 1, state=state, thought=summary, action=summary, tool_name=tool,
                    tool_input=tool_in, tool_output_summary=summary, decision_outcome=outcome or summary,
                )
            )

        plan = out["plan"]
        add(
            AgentState.QUERY_ANALYSIS, f"QuestionPlanner[{plan.get('planner')}]", {"query": query},
            f"intent={plan.get('intent')}; {len(plan.get('sub_claims', []))} sub-claim(s)",
        )
        for r in results:
            tr = self.service.get_trace(r["verification_id"], session_id)
            claim = f"{r['claim']['subject']} {r['claim']['relation']} {r['claim']['object']}"
            for step in tr["steps"]:
                if step["step"] == "resolve_entities":
                    continue
                add(_STEP_STATE.get(step["step"], AgentState.EVIDENCE_INSPECTION), step["step"],
                    {"claim": claim}, ", ".join(f"{k}={v}" for k, v in step.items() if k != "step"))
            if r["conflict"]["type"] != "NO_CONFLICT":
                add(AgentState.CONFLICT_RESOLUTION, "ContextResolver", {"claim": claim},
                    f"{r['conflict']['type']} ({r['conflict'].get('direction')}, {r['conflict'].get('delta_days')} days)")
        if not results:
            add(AgentState.INSUFFICIENT_EVIDENCE, "VerificationService", {"query": query},
                "No verifiable sub-claim could be planned; entity lookup only")
        add(AgentState.ANSWER_GENERATION, "VerificationService", {"claims": len(results)},
            f"overall_status={out['overall_status']}")

        supporting = [e for r in results for e in r["supporting"]]
        contradicting = [e for r in results for e in r["contradicting"]]
        tier = max((r["support_tier"] for r in results), key=_TIER_ORDER.index, default="NONE")
        conflicts = [r["conflict"] for r in results if r["conflict"]["type"] != "NO_CONFLICT"]
        diversity = len({
            (e["publisher"] or "").lower() for e in supporting
            if e["source_type"] == "corpus" and not e["synthetic"] and e["publisher"]
        })
        narrative = "\n\n".join(r["narrative"] or "" for r in results) or "No verifiable claim found in the question."
        return GroundedReport(
            query=query, status=out["overall_status"], support_tier=tier, source_diversity=diversity,
            supporting_evidence=supporting, contradicting_evidence=contradicting,
            conflict_resolution=conflicts[0] if conflicts else None, state_history=states,
            execution_trace=trace, narrative=narrative, claim_results=results, plan=plan,
            related_entities=out.get("related_entities", []), warnings=out["warnings"],
        )
